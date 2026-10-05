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
#
# CaP-X's cuRobo planning server (capx/serving/launch_curobo_server.py) and OpenETA's
# configuration collision check (sim/mcp_server/collision.py), joined: the env servers plan
# a collision-free path before a move and check the robot against the world before each
# servo segment of it.

"""Collision-free moves for env servers: plan with ``ik.plan``, check with ``ik.check``.

An env server started with ``--ik <url>`` builds a :class:`MotionPlanner` next to its
:class:`~pi_embodied_services.utils.reach.ReachPreview`. Its motion primitives then

1. plan: :meth:`MotionPlanner.plan` asks the ik service for a joint path from the current
   joints to the target TCP pose clear of the env server's planning world, and returns it
   as TCP waypoints in the env server's frame (at most ``max_segments``, at least
   ``min_segment_m`` apart) with the planned joints at each; :func:`require_planned`
   refuses the move (``ValueError``) when no collision-free path exists;
2. execute the waypoints one servo segment at a time within the primitive's own limits
   (step budget, per-step clip), and before each segment :meth:`MotionPlanner.check` the
   current joints and the next planned configuration against the world: predicted contact
   stops the move (``stopped: "contact"``). Checks run at the waypoints only: between two of
   them the arm follows a Cartesian servo (the Franka servers' move_delta / rotate_delta,
   LIBERO's OSC), not the planned joint trajectory, and is not checked.

A planner that answers with an error (for instance PyRoKi given the dual rig's ``robot``
obstacle) or an invalid result refuses the move, and an unchecked segment stops it: ``--ik``
never falls back to unplanned motion by itself. Only while the ik service is unreachable,
and only with ``--ik-allow-unplanned``, is the plan ``unknown`` and the move runs unplanned
with a warning, as it did before ``--ik``; without the flag it is refused too.

The planning world
------------------
Each env server supplies its obstacles (``utils/collision.py``) in its own frame:

- LIBERO: every collidable geom of the MuJoCo scene that is not the robot (the table,
  fixtures, and one box per movable object), read from the sim at each plan and check;
- Franka: a static scene file (``PI_EMBODIED_IK_WORLD``, JSON list of obstacles in the
  robot base frame: table, walls, fixtures), empty without it;
- dual Franka: the same file (in the ``right_base`` frame) plus the other arm at its
  current joints (a ``robot`` obstacle, expanded into cuRobo's collision spheres).

Decision: planning geometry is infrastructure, not planner-visible truth. The sim's exact
geometry is used whether or not ``--privileged`` is on, as a real cell's collision scene
model would be, because it only ever refuses or stops motion; it never reaches the agent:
results and refusals carry counts, statuses and clearances, never obstacle poses or
extents (``nearest`` names the obstacle category, the text the agent already sees in the
scene). Obstacles within ``contact_radius`` of the start or goal TCP are left out of the
world (the object being grasped, held or placed on, and the surface under it): contact
there is the point of the move, and the primitive's own guards (z floor, grip) cover it.
That exception is for touching, not entering: a goal TCP more than
``FIXTURE_PENETRATION_M`` inside a ``fixed`` obstacle (the table, a panel, the floor;
``utils/collision.py``) is refused before any planning, since the left-out table would
otherwise let a target below its top be "planned" and executed to a stall (verify3 bug 45).
A movable object's bounding box is not a fixture: a grasp at a bowl's rim is inside it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from pi_embodied_services.utils import collision
from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.reach import pose_matrix
from pi_embodied_services.utils.rpc import RpcError, make_rpc_client
from pi_embodied_services.utils.transforms import invert_transform

logger = get_logger("motion")

DEFAULT_TIMEOUT_S = 60.0
#: Obstacles this close (m) to the start or goal TCP are the ones the move is about.
CONTACT_RADIUS_M = 0.04
#: A goal TCP deeper than this (m) inside a ``fixed`` obstacle is refused before planning.
FIXTURE_PENETRATION_M = 0.005
#: Servo segments per move, and their minimum length (m).
MAX_SEGMENTS = 10
MIN_SEGMENT_M = 0.02
#: A plan whose TCP path is longer than the straight line plus this (m) is refused.
MAX_DETOUR_M = 0.35
#: Clearance (m) below which a configuration counts as contact at execution time.
CHECK_MARGIN_M = 0.0
#: Env var naming a real robot's static scene file (JSON list of obstacles).
WORLD_ENV = "PI_EMBODIED_IK_WORLD"


def static_world() -> list[dict[str, Any]]:
    """The obstacles of ``$PI_EMBODIED_IK_WORLD`` (a JSON list, validated), or none. A
    static scene describes the cell's fixtures: its entries are ``fixed`` unless one says
    ``"fixed": false`` (an object that is moved)."""
    path = os.environ.get(WORLD_ENV, "").strip()
    if not path:
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("obstacles", [])
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of obstacles")
    return [
        collision.to_wire(collision.parse_obstacle({"fixed": True, **o}))
        if isinstance(o, dict)
        else collision.to_wire(collision.parse_obstacle(o))
        for o in data
    ]


def add_unplanned_argument(parser: Any) -> None:
    """``--ik-allow-unplanned``: with ``--ik``, let a move run unplanned while the ik service
    is unreachable (default: refuse it). A planner that answers with an error always refuses."""
    parser.add_argument(
        "--ik-allow-unplanned",
        action="store_true",
        help="with --ik, run moves unplanned while the ik service is unreachable "
        "(default: refuse them); planner errors always refuse",
    )


def planner_from_args(args: Any, robot: str) -> "MotionPlanner | None":
    """The env server's :class:`MotionPlanner` for ``--ik``, or None without it."""
    url = getattr(args, "ik", None)
    allow = bool(getattr(args, "ik_allow_unplanned", False))
    return MotionPlanner(url, robot, allow_unplanned=allow) if url else None


def ik_backend(url: str, *, client: Any = None, timeout_s: float = 30.0) -> str:
    """The backend the ik service at ``url`` runs (``ik.robots``); raises when it cannot say."""
    client = client or make_rpc_client(url)
    return str(client.call("ik.robots", (), {}, timeout_s=timeout_s)["backend"])


def _unreachable(exc: BaseException) -> bool:
    """Whether a call failed because the ik service could not be reached (connection refused,
    timeout), as opposed to the service answering with an error."""
    import urllib.error

    cause = exc.__cause__ if isinstance(exc, RpcError) else exc
    return isinstance(cause, OSError) and not isinstance(cause, urllib.error.HTTPError)


def _pose7(pos: Any, quat: Any) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    return np.concatenate(
        [np.asarray(pos, dtype=np.float64).reshape(3), q / np.linalg.norm(q)]
    )


def _apply(matrix: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rot = Rotation.from_matrix(matrix[:3, :3]) * Rotation.from_quat(pose[3:])
    return np.concatenate([matrix[:3, :3] @ pose[:3] + matrix[:3, 3], rot.as_quat()])


def select_waypoints(
    tcp_path: np.ndarray,
    *,
    max_segments: int = MAX_SEGMENTS,
    min_segment_m: float = MIN_SEGMENT_M,
) -> list[int]:
    """Indices of the TCP path to servo through (the start excluded, the goal included)."""
    n = len(tcp_path)
    if n < 2:
        return [n - 1] if n else []
    keep, last = [], 0
    for i in range(1, n - 1):
        moved = np.linalg.norm(tcp_path[i, :3] - tcp_path[last, :3])
        turned = (
            Rotation.from_quat(tcp_path[i, 3:])
            * Rotation.from_quat(tcp_path[last, 3:]).inv()
        ).magnitude()
        if moved >= min_segment_m or turned >= 0.2:
            keep.append(i)
            last = i
    keep.append(n - 1)
    if len(keep) > max_segments:
        pick = np.linspace(0, len(keep) - 1, max_segments).round().astype(int)
        keep = [keep[i] for i in sorted(set(pick.tolist()))]
    return keep


class MotionPlanner:
    """Plans collision-free moves and checks configurations through the ik service.

    ``robot`` is the ik service's model (``panda`` for the Franka TCP, ``panda_libero`` for
    robosuite's grip site). The optional ``client`` (anything with ``call(method, args,
    kwargs, timeout_s=)``) replaces the HTTP client in tests.
    """

    def __init__(
        self,
        url: str | None,
        robot: str,
        *,
        client: Any = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        contact_radius: float = CONTACT_RADIUS_M,
        max_segments: int = MAX_SEGMENTS,
        min_segment_m: float = MIN_SEGMENT_M,
        max_detour_m: float = MAX_DETOUR_M,
        margin: float = CHECK_MARGIN_M,
        allow_unplanned: bool = False,
    ) -> None:
        if client is None:
            if not url:
                raise ValueError("MotionPlanner needs the ik service URL")
            client = make_rpc_client(url)
        self.url = url
        self.robot = robot
        self.timeout_s = float(timeout_s)
        self.contact_radius = float(contact_radius)
        self.max_segments = int(max_segments)
        self.min_segment_m = float(min_segment_m)
        self.max_detour_m = float(max_detour_m)
        self.margin = float(margin)
        #: Only while the service is unreachable: run moves unplanned instead of refusing them.
        self.allow_unplanned = bool(allow_unplanned)
        self._client = client

    def _call(self, method: str, **kwargs: Any) -> Any:
        return self._client.call(method, (), kwargs, timeout_s=self.timeout_s)

    @staticmethod
    def _frames(base_pose: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
        """(base <- scene, scene <- base) transforms; identity without a base pose."""
        if base_pose is None:
            return np.eye(4), np.eye(4)
        scene_from_base = pose_matrix(base_pose["pos"], base_pose["quat_xyzw"])
        return invert_transform(scene_from_base), scene_from_base

    def world(
        self,
        obstacles: list[dict[str, Any]],
        *,
        base_pose: dict[str, Any] | None = None,
        near: list[Any] = (),
        left_out: Any = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """The obstacles in the robot base frame (wire format) and the names left out:
        ``left_out`` when given, else those within ``contact_radius`` of a ``near`` point
        (scene frame)."""
        base_from_scene, _ = self._frames(base_pose)
        kept, dropped = [], []
        for i, raw in enumerate(obstacles):
            obs = collision.parse_obstacle(raw)
            name = obs["name"] or f"{obs['type']}_{i}"
            obs["name"] = name
            if left_out is not None:
                drop = name in left_out
            else:
                drop = any(
                    float(collision.point_distance(obs, p)[0]) < self.contact_radius
                    for p in near
                )
            if drop:
                dropped.append(name)
                continue
            kept.append(
                collision.to_wire(collision.transform_obstacle(obs, base_from_scene))
            )
        return kept, dropped

    def inside_fixture(
        self, point: Any, obstacles: list[dict[str, Any]]
    ) -> tuple[str, float] | None:
        """The ``fixed`` obstacle (scene frame) ``point`` lies more than
        ``FIXTURE_PENETRATION_M`` inside, as (name, depth m: the distance to its nearest
        face), the deepest first; None when the point is clear of, on, or only grazing
        every fixture."""
        p = np.asarray(point, dtype=np.float64).reshape(3)
        deepest: tuple[str, float] | None = None
        for i, raw in enumerate(obstacles):
            obs = collision.parse_obstacle(raw)
            if not obs.get("fixed"):
                continue
            depth = -float(collision.point_distance(obs, p)[0])
            if depth > FIXTURE_PENETRATION_M and (
                deepest is None or depth > deepest[1]
            ):
                deepest = (obs["name"] or f"{obs['type']}_{i}", depth)
        return deepest

    def plan(
        self,
        q: Any,
        start_pos: Any,
        goal_pos: Any,
        goal_quat: Any,
        obstacles: list[dict[str, Any]],
        *,
        base_pose: dict[str, Any] | None = None,
        robots: list[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        """Plan the TCP from joints ``q`` (TCP at ``start_pos``) to ``goal_pos`` /
        ``goal_quat`` clear of ``obstacles``, all in the scene frame (the robot base sits at
        ``base_pose`` in it; None: the scene frame is the base frame). ``robots`` are
        ``robot`` obstacles already in the base frame (another arm).

        Returns ``{"status": "planned" | "blocked" | "unknown", "message", "waypoints"
        (scene-frame TCP poses xyz + xyzw, the goal last), "q_path" (the planned joints at
        each), "left_out" (obstacle names), "obstacles" (count kept), "path_m", "backend"}``.
        """
        goal = _pose7(goal_pos, goal_quat)
        start = np.asarray(start_pos, dtype=np.float64).reshape(3)
        out: dict[str, Any] = {
            "status": "unknown",
            "waypoints": [],
            "q_path": [],
            "left_out": [],
            "excluded_by_base": [],
            "obstacles": 0,
            "path_m": None,
            "backend": None,
        }
        # The hand can touch a fixture (the left-out world below) but never be inside one:
        # refused before the fixture is dropped from the world as "the surface under it".
        inside = self.inside_fixture(goal[:3], obstacles)
        if inside is not None:
            name, depth = inside
            out.update(
                status="blocked",
                message=(
                    f"the target lies inside {name} ({round(depth * 1000)} mm deep), "
                    "a fixture the hand cannot enter; aim above it"
                ),
            )
            return out
        base_from_scene, scene_from_base = self._frames(base_pose)
        kept, dropped = self.world(
            obstacles, base_pose=base_pose, near=[start, goal[:3]]
        )
        kept = kept + [dict(r) for r in robots]
        out.update(left_out=dropped, obstacles=len(kept))
        goal_base = _apply(base_from_scene, goal)
        try:
            result = self._call(
                "ik.plan",
                robot=self.robot,
                start_q=[float(v) for v in np.asarray(q, dtype=np.float64).reshape(-1)],
                goal_pose={
                    "pos": goal_base[:3].tolist(),
                    "quat_xyzw": goal_base[3:].tolist(),
                },
                obstacles=kept,
                waypoints=48,
            )
        except (RpcError, OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("motion plan failed: %s", exc)
            if _unreachable(exc) and self.allow_unplanned:
                out["message"] = f"IK service unreachable, path not planned: {exc}"
            else:
                out.update(status="blocked", message=f"path not planned: {exc}")
            return out
        if not isinstance(result, dict) or "ok" not in result:
            out.update(
                status="blocked",
                message=f"path not planned: invalid ik.plan result {result!r}",
            )
            return out
        out["backend"] = result.get("backend")
        # Obstacles the robot's own base already sits in (the table under a mounted arm).
        out["excluded_by_base"] = list(result.get("excluded_by_base") or [])
        if not result["ok"]:
            why = self._why_blocked(q, goal_base, kept, result.get("status"))
            out.update(
                status="blocked",
                message=f"no collision-free path: {result.get('error') or 'plan failed'}"
                + (f"; {why}" if why else ""),
            )
            return out
        path = np.asarray(result.get("path") or [], dtype=np.float64)
        tcp = np.asarray(result.get("tcp_path") or [], dtype=np.float64)
        if len(path) < 1 or tcp.shape != (len(path), 7):
            out.update(
                status="blocked",
                message="path not planned: ik.plan lacks path / tcp_path",
            )
            return out
        tcp_scene = np.stack([_apply(scene_from_base, p) for p in tcp])
        tcp_scene[-1, :3] = goal[:3]  # the servo aims at the requested target itself
        length = float(np.linalg.norm(np.diff(tcp_scene[:, :3], axis=0), axis=1).sum())
        straight = float(np.linalg.norm(goal[:3] - start))
        out["path_m"] = round(length, 4)
        if length > straight + self.max_detour_m:
            out.update(
                status="blocked",
                message=(
                    f"the only collision-free path found detours {length:.2f} m for a "
                    f"{straight:.2f} m move (limit +{self.max_detour_m:.2f} m)"
                ),
            )
            return out
        idx = select_waypoints(
            tcp_scene, max_segments=self.max_segments, min_segment_m=self.min_segment_m
        )
        out.update(
            status="planned",
            message=(
                f"collision-free path, {len(idx)} segment(s), {length:.2f} m, "
                f"{len(kept)} obstacle(s)"
            ),
            waypoints=[[round(float(v), 5) for v in tcp_scene[i]] for i in idx],
            q_path=[[float(v) for v in path[i]] for i in idx],
        )
        return out

    def _why_blocked(
        self,
        q: Any,
        goal_base: np.ndarray,
        kept: list[dict[str, Any]],
        status: Any = None,
    ) -> str | None:
        """Why a plan failed, by the planner's ``status`` first (cuRobo's MotionGenStatus):
        an invalid start (the arm already touches an obstacle where it is, or is in
        self-collision / past a joint limit: back off before aiming elsewhere), a path failure
        after cuRobo's own collision-aware IK found a clear goal (the goal is fine, the path
        is blocked), or an IK failure, which, like a planner without a status, is diagnosed
        from the ik service: the goal is unreachable (IK fails without obstacles), or the goal
        configuration found touches an obstacle (named; its box is a bound, so a round
        object's corners count). None when the service cannot tell."""
        code = str(status or "").upper().replace(" ", "_")
        if "INVALID_START_STATE" in code:
            return self._start_invalid(q, kept, str(status))
        if code and "IK_FAIL" not in code:
            return f"the goal is reachable and clear; the path between is blocked ({status})"
        try:
            sol = self._call(
                "ik.solve",
                robot=self.robot,
                target_pose={
                    "pos": goal_base[:3].tolist(),
                    "quat_xyzw": goal_base[3:].tolist(),
                },
                seed_q=[float(v) for v in np.asarray(q, dtype=np.float64).reshape(-1)],
            )
            if not sol.get("ok"):
                return f"the goal pose is out of reach ({sol.get('error')})"
            if not kept:
                return None
            chk = self._call(
                "ik.check",
                robot=self.robot,
                q=sol["q"],
                obstacles=kept,
                margin=self.margin,
            )
        except (RpcError, OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("blocked plan not diagnosed: %s", exc)
            return None
        if chk.get("collision_free"):
            if code:  # cuRobo's 32-seed collision-aware IK found no clear configuration
                return (
                    "the goal is reachable, but the planner found no collision-free "
                    f"configuration for it ({status}); aim a little further from the obstacles"
                )
            return "the goal is reachable and clear; the path between is blocked"
        name = str(chk.get("nearest") or "an obstacle").split("/")[0]
        gap = chk.get("min_clearance_m")
        mm = "" if gap is None else f", {round(float(gap) * 1000)} mm"
        return (
            f"the goal configuration found touches {name}'s bounding box{mm}; "
            "aim clear of it"
        )

    def _start_invalid(self, q: Any, kept: list[dict[str, Any]], status: str) -> str:
        """The planner refused the start configuration: name what the arm touches there."""
        if kept:
            try:
                chk = self._call(
                    "ik.check",
                    robot=self.robot,
                    q=[float(v) for v in np.asarray(q, dtype=np.float64).reshape(-1)],
                    obstacles=kept,
                    margin=self.margin,
                )
            except (RpcError, OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning("invalid start not diagnosed: %s", exc)
                chk = {}
            if chk and not chk.get("collision_free"):
                name = str(chk.get("nearest") or "an obstacle").split("/")[0]
                gap = chk.get("min_clearance_m")
                mm = "" if gap is None else f", {round(float(gap) * 1000)} mm"
                return (
                    f"the arm is already touching {name}'s bounding box where it is{mm}; "
                    "back off before planning"
                )
        if "WORLD_COLLISION" in status.upper():
            return (
                "the arm already stands within the planner's collision margin of an "
                f"obstacle ({status}); back off before planning"
            )
        return (
            f"the arm's current configuration is not a valid start ({status}: "
            "self-collision or a joint limit); back off before planning"
        )

    def check(
        self,
        qs: list[Any],
        obstacles: list[dict[str, Any]],
        *,
        base_pose: dict[str, Any] | None = None,
        left_out: Any = (),
        robots: list[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        """Whether the configurations ``qs`` are clear of the world (the plan's left-out
        obstacles stay out). ``{"status": "clear" | "contact" | "unknown",
        "min_clearance_m", "worst_index", "nearest", "message"}``."""
        kept, _ = self.world(obstacles, base_pose=base_pose, left_out=set(left_out))
        kept = kept + [dict(r) for r in robots]
        out: dict[str, Any] = {
            "status": "unknown",
            "min_clearance_m": None,
            "worst_index": None,
            "nearest": None,
        }
        try:
            result = self._call(
                "ik.check",
                robot=self.robot,
                path=[
                    [float(v) for v in np.asarray(q, dtype=np.float64).reshape(-1)]
                    for q in qs
                ],
                obstacles=kept,
                margin=self.margin,
            )
        except (RpcError, OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("collision check failed: %s", exc)
            if _unreachable(exc) and self.allow_unplanned:
                out["message"] = f"IK service unreachable, collision not checked: {exc}"
            else:
                # An unchecked segment is not a clear one: stop as for predicted contact.
                out.update(status="contact", message=f"collision not checked: {exc}")
            return out
        if not isinstance(result, dict) or "collision_free" not in result:
            out.update(
                status="contact",
                message=f"collision not checked: invalid ik.check result {result!r}",
            )
            return out
        clear = result.get("min_clearance_m")
        nearest = result.get("nearest")
        category = str(nearest).split("/")[0] if nearest else None
        out.update(
            status="clear" if result["collision_free"] else "contact",
            min_clearance_m=None if clear is None else round(float(clear), 4),
            worst_index=result.get("worst_index"),
            nearest=category,
            message=(
                "clear"
                if result["collision_free"]
                else f"predicted contact with {category} ({float(clear) * 1000:.0f} mm)"
            ),
        )
        return out


def require_planned(plan: dict[str, Any], action: str) -> dict[str, Any]:
    """Refuse ``action`` (raise ``ValueError``) unless a collision-free path was planned. Only
    ``unknown`` passes (with a warning; the move runs unplanned), and :meth:`MotionPlanner.plan`
    returns it only while the service is unreachable under ``--ik-allow-unplanned``: a planner
    error or an invalid answer is ``blocked``."""
    if plan.get("status") == "blocked":
        raise ValueError(f"{action} refused: {plan.get('message')}")
    if plan.get("status") != "planned":
        logger.warning("%s: path not planned (%s)", action, plan.get("message"))
    return plan


def follow_waypoints(
    waypoints: list[Any],
    *,
    tcp_pose: Callable[[], Any],
    move: Callable[[list[float]], dict[str, Any]],
    rotate: Callable[[list[float]], dict[str, Any]],
    check: Callable[[int], dict[str, Any]],
    pos_eps: float = 0.002,
    rot_eps: float = 0.02,
    stall_m: float = 0.01,
    stall_rad: float = 0.05,
) -> dict[str, Any]:
    """Servo through TCP waypoints (xyz + xyzw) with a translation primitive (``move``, a
    delta in the waypoints' frame) and a rotation primitive (``rotate``, extrinsic xyz
    Euler angles composed on the left, as the Franka servers' ``rotate_delta``), calling
    ``check(i)`` before segment ``i``: its ``contact`` stops the move.

    A segment that fails (its result says ``ok: false`` or carries an ``error``, or it
    ends more than ``stall_m`` / ``stall_rad`` short of its waypoint) stops the move: the
    next waypoints were planned from this one, not from wherever the arm stopped.

    Returns ``{"segments": done, "results": [...], "stopped": None | "contact" |
    "cancelled" | "stalled", "check": the stopping check, "stalled": {...} when stalled}``.
    """

    def failed(r: dict[str, Any]) -> bool:
        return r.get("ok") is False or bool(r.get("error"))

    results: list[dict[str, Any]] = []
    out: dict[str, Any] = {"segments": 0, "results": results, "stopped": None}
    for i, wp in enumerate(waypoints):
        verdict = check(i)
        if verdict.get("status") == "contact":
            out.update(stopped="contact", check=verdict)
            return out
        target = np.asarray(wp, dtype=np.float64).reshape(7)
        cur = np.asarray(tcp_pose(), dtype=np.float64).reshape(-1)
        delta = target[:3] - cur[:3]
        if np.linalg.norm(delta) > pos_eps:
            results.append(move([float(v) for v in delta]))
            if results[-1].get("cancelled"):
                out.update(stopped="cancelled", segments=i)
                return out
            cur = np.asarray(tcp_pose(), dtype=np.float64).reshape(-1)
            short = float(np.linalg.norm(target[:3] - cur[:3]))
            if failed(results[-1]) or short > stall_m:
                out.update(
                    stopped="stalled",
                    segments=i,
                    stalled={
                        "waypoint": i,
                        "move": "translate",
                        "short_m": round(short, 4),
                    },
                )
                return out
        turn = Rotation.from_quat(target[3:]) * Rotation.from_quat(cur[3:7]).inv()
        if turn.magnitude() > rot_eps:
            results.append(rotate([float(v) for v in turn.as_euler("xyz")]))
            if results[-1].get("cancelled"):
                out.update(stopped="cancelled", segments=i)
                return out
            cur = np.asarray(tcp_pose(), dtype=np.float64).reshape(-1)
            left = float(
                (
                    Rotation.from_quat(target[3:]) * Rotation.from_quat(cur[3:7]).inv()
                ).magnitude()
            )
            if failed(results[-1]) or left > stall_rad:
                out.update(
                    stopped="stalled",
                    segments=i,
                    stalled={
                        "waypoint": i,
                        "move": "rotate",
                        "short_rad": round(left, 4),
                    },
                )
                return out
        out["segments"] = i + 1
    return out


__all__ = [
    "CONTACT_RADIUS_M",
    "MotionPlanner",
    "WORLD_ENV",
    "follow_waypoints",
    "planner_from_args",
    "require_planned",
    "select_waypoints",
    "static_world",
]
