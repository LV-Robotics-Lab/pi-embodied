# Copyright 2026 The pi-embodied Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collision primitives shared by the ik service and the env servers' planning worlds.

An obstacle is a dict in some frame (metres): ``box`` (``position``, ``extent`` = full edge
lengths, optional ``quat_xyzw``), ``sphere`` (``center``, ``radius``), ``capsule``
(``position``, ``radius``, ``height`` = the length of its axis segment along local z,
optional ``quat_xyzw``) or ``halfspace`` (``point``, ``normal``; the obstacle is the side
the normal points away from). Every obstacle may carry a ``name``.

The ik service also accepts ``robot`` obstacles (another arm at its joints, see
``components/ik_server.py``); they are expanded there, not here.

Distances are signed: positive is a gap, negative a penetration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

KINDS = ("box", "sphere", "capsule", "halfspace")


def parse_obstacle(obstacle: Any) -> dict[str, Any]:
    """Validate one obstacle into numpy arrays (see the module docstring)."""
    if not isinstance(obstacle, dict) or "type" not in obstacle:
        raise ValueError("each obstacle is a dict with a 'type'")
    kind = str(obstacle["type"])
    out: dict[str, Any] = {"type": kind, "name": str(obstacle.get("name", ""))}

    def vec(key: str, n: int) -> np.ndarray:
        if key not in obstacle:
            raise ValueError(f"{kind} obstacle needs '{key}'")
        v = np.asarray(obstacle[key], dtype=np.float64).reshape(-1)
        if v.shape != (n,) or not np.isfinite(v).all():
            raise ValueError(f"{kind} obstacle '{key}' must be {n} finite numbers")
        return v

    def quat() -> np.ndarray:
        if "quat_xyzw" not in obstacle:
            return np.array([0.0, 0.0, 0.0, 1.0])
        q = vec("quat_xyzw", 4)
        if np.linalg.norm(q) < 1e-9:
            raise ValueError(f"{kind} obstacle quat_xyzw must be nonzero")
        return q / np.linalg.norm(q)

    def positive(key: str) -> float:
        value = float(obstacle.get(key, float("nan")))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{kind} obstacle '{key}' must be a positive number")
        return value

    if kind == "box":
        out["position"], out["extent"] = vec("position", 3), vec("extent", 3)
        if (out["extent"] <= 0).any():
            raise ValueError("box extent must be positive")
        out["quat_xyzw"] = quat()
    elif kind == "sphere":
        out["center"], out["radius"] = vec("center", 3), positive("radius")
    elif kind == "capsule":
        out["position"], out["radius"] = vec("position", 3), positive("radius")
        out["height"] = positive("height")
        out["quat_xyzw"] = quat()
    elif kind == "halfspace":
        out["point"], out["normal"] = vec("point", 3), vec("normal", 3)
        if np.linalg.norm(out["normal"]) < 1e-9:
            raise ValueError("halfspace normal must be nonzero")
        out["normal"] = out["normal"] / np.linalg.norm(out["normal"])
    else:
        raise ValueError(
            f"unknown obstacle type {kind!r}; box, sphere, capsule or halfspace"
        )
    return out


def to_wire(obstacle: dict[str, Any]) -> dict[str, Any]:
    """A parsed obstacle as JSON-ready lists (the ik service's input format)."""
    return {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in obstacle.items()
    }


def transform_obstacle(obstacle: dict[str, Any], matrix: np.ndarray) -> dict[str, Any]:
    """A parsed obstacle moved by the rigid transform ``matrix`` (4x4, target <- source)."""
    rot, trans = matrix[:3, :3], matrix[:3, 3]
    out = dict(obstacle)
    kind = obstacle["type"]
    if kind in ("box", "capsule"):
        out["position"] = rot @ obstacle["position"] + trans
        out["quat_xyzw"] = (
            Rotation.from_matrix(rot) * Rotation.from_quat(obstacle["quat_xyzw"])
        ).as_quat()
    elif kind == "sphere":
        out["center"] = rot @ obstacle["center"] + trans
    else:
        out["point"] = rot @ obstacle["point"] + trans
        out["normal"] = rot @ obstacle["normal"]
    return out


def point_distance(obstacle: dict[str, Any], points: Any) -> np.ndarray:
    """Signed distance from each point (N x 3) to a parsed obstacle's surface."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    kind = obstacle["type"]
    if kind == "sphere":
        return np.linalg.norm(p - obstacle["center"], axis=1) - obstacle["radius"]
    if kind == "halfspace":
        return (p - obstacle["point"]) @ obstacle["normal"]
    rot = Rotation.from_quat(obstacle["quat_xyzw"]).as_matrix()
    local = (p - obstacle["position"]) @ rot  # rows: R^T (p - c)
    if kind == "box":
        q = np.abs(local) - obstacle["extent"] / 2.0
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(q.max(axis=1), 0.0)
        return outside + inside
    half = obstacle["height"] / 2.0
    closest = np.zeros_like(local)
    closest[:, 2] = np.clip(local[:, 2], -half, half)
    return np.linalg.norm(local - closest, axis=1) - obstacle["radius"]


def sphere_clearance(
    obstacles: list[dict[str, Any]], spheres: Any
) -> tuple[float, int | None]:
    """Smallest gap between spheres (N x 4: xyz, radius) and parsed obstacles, and the index
    of the obstacle it is to (None without obstacles or spheres)."""
    s = np.asarray(spheres, dtype=np.float64).reshape(-1, 4)
    s = s[s[:, 3] > 0]
    worst, which = float("inf"), None
    if not len(s):
        return worst, which
    for i, obs in enumerate(obstacles):
        d = float((point_distance(obs, s[:, :3]) - s[:, 3]).min())
        if d < worst:
            worst, which = d, i
    return worst, which


# ---------------------------------------------------------------------------
# MuJoCo scenes (LIBERO, Robosuite): the planning world of a simulated scene
# ---------------------------------------------------------------------------

#: Body name prefixes of the robot itself in robosuite scenes (arm, gripper, pedestal).
ROBOT_BODY_PREFIXES = ("robot0_", "gripper0_", "mount0_")
#: Geoms farther than this from the robot base cannot touch a Panda (reach 0.855 m).
WORLD_RADIUS_M = 1.3
#: Geoms smaller than this (largest half extent) are ignored.
MIN_HALF_EXTENT_M = 0.002
#: The ik service's cuRobo planner holds this many boxes.
MAX_BOXES = 150


def mujoco_collision_world(
    sim: Any,
    base_pos: Any,
    *,
    robot_prefixes: tuple[str, ...] = ROBOT_BODY_PREFIXES,
    radius: float = WORLD_RADIUS_M,
    max_boxes: int = MAX_BOXES,
) -> list[dict[str, Any]]:
    """Every collidable geom of a MuJoCo scene that is not the robot, in world coordinates.

    A free-jointed object (a movable LIBERO object) becomes one axis-aligned box around all
    its geoms, named after its root body; a fixture geom (table, cabinet panel, wall) is an
    oriented box around the geom (``geom_aabb``), named ``<body>/<geom id>``; a plane is a
    halfspace. Geoms farther than ``radius`` from ``base_pos`` are left out, and the
    nearest ``max_boxes`` boxes are kept. The result is JSON-ready (lists)."""
    import mujoco

    m = getattr(sim.model, "_model", sim.model)
    d = getattr(sim.data, "_data", sim.data)
    base = np.asarray(base_pos, dtype=np.float64).reshape(3)

    def body_name(b: int) -> str:
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"

    def free_root(b: int) -> int | None:
        root = int(m.body_rootid[b])
        adr, num = int(m.body_jntadr[root]), int(m.body_jntnum[root])
        if num and int(m.jnt_type[adr]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return root
        return None

    boxes: list[tuple[float, dict[str, Any]]] = []
    planes: list[dict[str, Any]] = []
    movable: dict[int, list[np.ndarray]] = {}
    for g in range(int(m.ngeom)):
        if not (int(m.geom_contype[g]) or int(m.geom_conaffinity[g])):
            continue
        b = int(m.geom_bodyid[g])
        name = body_name(int(m.body_rootid[b]))
        if body_name(b).startswith(robot_prefixes) or name.startswith(robot_prefixes):
            continue
        pos = np.asarray(d.geom_xpos[g], dtype=np.float64)
        rot = np.asarray(d.geom_xmat[g], dtype=np.float64).reshape(3, 3)
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            planes.append(
                {
                    "type": "halfspace",
                    "name": f"{body_name(b)}/{g}",
                    "point": pos.tolist(),
                    "normal": rot[:, 2].tolist(),
                }
            )
            continue
        aabb = np.asarray(m.geom_aabb[g], dtype=np.float64)
        center, half = pos + rot @ aabb[:3], np.abs(aabb[3:])
        if half.max() < MIN_HALF_EXTENT_M:
            continue
        root = free_root(b)
        if root is not None:
            corners = np.array(
                [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            )
            movable.setdefault(root, []).append(center + (corners * half) @ rot.T)
            continue
        box = {
            "type": "box",
            "name": f"{body_name(b)}/{g}",
            "position": center.tolist(),
            "extent": (2 * half).tolist(),
            "quat_xyzw": Rotation.from_matrix(rot).as_quat().tolist(),
        }
        boxes.append((_box_gap(center, half, rot, base), box))
    for root, corner_sets in movable.items():
        pts = np.concatenate(corner_sets)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        center, half = (lo + hi) / 2, (hi - lo) / 2
        box = {
            "type": "box",
            "name": body_name(root),
            "position": center.tolist(),
            "extent": np.maximum(2 * half, 2 * MIN_HALF_EXTENT_M).tolist(),
        }
        boxes.append((_box_gap(center, half, np.eye(3), base), box))
    kept = sorted((b for b in boxes if b[0] <= radius), key=lambda b: b[0])
    return [b for _, b in kept[:max_boxes]] + planes


def _box_gap(center: np.ndarray, half: np.ndarray, rot: np.ndarray, p: np.ndarray):
    local = rot.T @ (p - center)
    return float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.0)))


__all__ = [
    "KINDS",
    "mujoco_collision_world",
    "parse_obstacle",
    "point_distance",
    "sphere_clearance",
    "to_wire",
    "transform_obstacle",
]
