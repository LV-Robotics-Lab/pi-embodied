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

"""Grasp and placement planning an env server composes over its current observation.

The grasp servers (``components/contact_graspnet_server.py``, ``graspgenx_server.py``,
``anygrasp_server.py``, ``graspnet1b_server.py``) answer in the **GraspNet grasp frame** in the
camera's OpenCV frame: origin at the grasp center (between the finger pads), X = approach,
Y = closing (the fingers slide along it), Z = X x Y. Each server converts its model's native frame
itself (the ``*_candidates`` functions below); the env server only ever sees normalized candidates.

The env server knows what no model server knows: the current camera frames, their intrinsics
and extrinsics, and the robot's grasp-to-EEF calibration. :class:`GraspPlanner` therefore lives
in the env server. It

- takes an *observation snapshot* (one camera's rgb, depth, K, cam2world at the current step),
- segments the target (SAM3) or takes a mask id, asks a grasp server, transforms the candidates
  to the world frame and hands out short ids (``g3``) bound to that snapshot,
- composes AnyPlace's object placement transform with a chosen grasp into the place grasp pose
  (``p1``), refusing masks and grasps from different snapshots,
- resolves a grasp id to the robot's EEF pose (``GraspToEef``) for the motion primitives,
- claims one grasp or place id for execution (``claim_waypoints``): the whole pre-grasp ->
  grasp -> lift (or pre-place -> place -> retreat) path is resolved at once from that one
  candidate, so the motions that follow never re-read an id they have themselves expired,
- remembers the executed grasp (the robot now holds its object) so ``plan_place`` can be
  asked after the grasp, from the new observation and the gripper's actual pose,
- expires every id when the robot moves (``invalidate``: the facade's mutating RPCs are
  wrapped; with a state digest only when the robot state actually changed).

Nothing here imports torch; the model servers run in their own venvs.
"""

from __future__ import annotations

import base64
import io
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from pi_embodied_services.utils.detections import (
    MOTION_METHODS,
    DetectionBook,
    DetectionStale,
    Epoch,
    decode_mask_png,
    describe_mask,
    frame_signature,
    scene_changed,
)
from pi_embodied_services.utils.logging import get_logger

logger = get_logger("grasp")

GRASP_FRAME = "graspnet"
CAMERA_FRAME = "opencv"
BACKENDS = ("contact_graspnet", "graspgenx", "anygrasp", "graspnet1b")
#: The env-server grasp URLs (``--contact-graspnet`` ... ``--anyplace``) by key.
GRASP_URL_KEYS = (*BACKENDS, "anyplace")
#: Contact-GraspNet's Panda gripper: base frame origin to the finger pads along the approach.
CONTACT_GRASPNET_GRIPPER_DEPTH = 0.1034
#: Depth (m) beyond which a pixel is ignored when building the object's point cloud.
DEFAULT_DEPTH_TRUNCATION = 2.0
#: Candidates returned to the planner per inference.
DEFAULT_MAX_CANDIDATES = 10
#: How far above the grasp (against the approach) the pre-grasp pose sits.
DEFAULT_STANDOFF_M = 0.10
#: How far straight up the arm lifts after closing on a claimed grasp.
DEFAULT_LIFT_M = 0.10
#: A place pose must approach from above: its approach at most this far from straight down.
MAX_PLACE_TILT_RAD = math.radians(45)
#: Where the placed object may end up relative to the region's points: its centroid over the
#: region's xy extent (grown by the margin) and between the region's lowest point less the
#: margin and its top plus the clearance.
PLACE_XY_MARGIN_M = 0.03
PLACE_Z_MARGIN_M = 0.05
PLACE_MAX_CLEARANCE_M = 0.25
#: A held object is set down this far above the region's top surface, then released.
PLACE_SETTLE_CLEARANCE_M = 0.01
#: A placed object's footprint must lie over the region: at least this fraction of its points
#: within PLACE_FOOTPRINT_TOL_M (in xy) of a region point (a bowl on a plate's rim tips off).
PLACE_MIN_FOOTPRINT = 0.8
PLACE_FOOTPRINT_TOL_M = 0.015
#: The longest standoff or lift a claim accepts.
MAX_WAYPOINT_OFFSET_M = 0.30

#: Columns are the GraspNet basis vectors in a model-native basis whose Z is the approach and
#: X the closing direction (Contact-GraspNet, GraspGenX): GraspNet X = native Z, Y = native X,
#: Z = native Y. ``R_graspnet = R_native @ ZX_NATIVE_TO_GRASPNET``.
ZX_NATIVE_TO_GRASPNET = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64
)


class GraspError(ValueError):
    """A grasp or placement request the planner refuses (bad input, no backend, stale id)."""


# ---------------------------------------------------------------------------
# geometry


def is_rotation(R: np.ndarray, atol: float = 1e-5) -> bool:
    R = np.asarray(R, dtype=np.float64)
    return bool(
        R.shape == (3, 3)
        and np.isfinite(R).all()
        and np.allclose(R.T @ R, np.eye(3), atol=atol)
        and np.isclose(np.linalg.det(R), 1.0, atol=atol)
    )


def rigid(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def quat_xyzw(R: np.ndarray) -> list[float]:
    """The xyzw quaternion of a rotation matrix (scipy-free, Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = (
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        )
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        )
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
        )
    q = np.array([x, y, z, w])
    return [float(v) for v in q / np.linalg.norm(q)]


def quat_xyzw_matrix(q: Any) -> np.ndarray:
    """The rotation matrix of an xyzw quaternion (normalized first)."""
    x, y, z, w = np.asarray(q, dtype=np.float64).reshape(4)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not n > 0:
        raise GraspError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotvec_of(R: np.ndarray) -> np.ndarray:
    """The rotation vector (axis * angle, rad) of a rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    cos = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cos)
    if angle < 1e-9:
        return np.zeros(3)
    if angle > math.pi - 1e-6:
        # Near a half turn the skew part vanishes: the axis is R + I's dominant column.
        B = (R + np.eye(3)) / 2.0
        axis = B[:, int(np.argmax(np.diag(B)))]
        return axis / np.linalg.norm(axis) * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2.0 * math.sin(angle)) * angle


#: A parallel gripper's grasp turned half about its approach (the EEF's +z) is the same grasp.
FLIP_ABOUT_APPROACH = np.diag([-1.0, -1.0, 1.0])


def orientation_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """World-frame rotation vector from the current EEF orientation to the target, or to the
    target turned half about its approach when that is the shorter way (the fingers are
    symmetric)."""
    R_c = np.asarray(R_current, dtype=np.float64)
    options = (np.asarray(R_target, dtype=np.float64),)
    options += (options[0] @ FLIP_ABOUT_APPROACH,)
    errs = [rotvec_of(R @ R_c.T) for R in options]
    return min(errs, key=lambda e: float(np.linalg.norm(e)))


def yaw_of(R: np.ndarray) -> float:
    """World yaw: the right-hand angle of the EEF's x axis about world +z (pi's ``yawOf``)."""
    return float(math.atan2(R[1, 0], R[0, 0]))


def pitch_of(R: np.ndarray) -> float:
    """LIBERO's ``rotate_pitch`` angle: 0 = pointing down, +pi/2 = pointing +y (pi's ``pitchOf``)."""
    return float(math.atan2(R[1, 2], -R[2, 2]))


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Organized camera-frame points [H, W, 3] of a metric depth map (0 / non-finite = none)."""
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return np.stack([x, y, depth], axis=-1)


def valid_points(
    depth: np.ndarray,
    depth_min: float = 0.0,
    depth_max: float = DEFAULT_DEPTH_TRUNCATION,
):
    depth = np.asarray(depth, dtype=np.float64)
    return np.isfinite(depth) & (depth > depth_min) & (depth < depth_max)


def object_points(
    depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    *,
    depth_max: float = DEFAULT_DEPTH_TRUNCATION,
) -> tuple[np.ndarray, np.ndarray]:
    """(object points, scene points) float32 [N, 3] in the camera frame from a mask."""
    points = backproject(depth, K)
    valid = valid_points(depth, 0.0, depth_max)
    mask = np.asarray(mask).astype(bool)
    if mask.shape != valid.shape:
        raise GraspError(f"mask {mask.shape} does not match depth {valid.shape}")
    return (
        np.ascontiguousarray(points[valid & mask], dtype=np.float32),
        np.ascontiguousarray(points[valid & ~mask], dtype=np.float32),
    )


def make_candidate(
    *,
    score: float,
    rotation: np.ndarray,
    center: np.ndarray,
    width: float,
    depth: float,
    source_model: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One normalized candidate: the GraspNet grasp frame in the camera frame.

    ``rotation`` columns are approach, closing, normal; ``center`` is the grasp center; the two
    contact points sit ``width / 2`` along +-closing from it; ``depth`` is how far the fingers
    reach past the center along the approach (0 when unknown).
    """
    R = np.asarray(rotation, dtype=np.float64)
    if not is_rotation(R):
        raise GraspError("candidate rotation is not a proper rotation matrix")
    c = np.asarray(center, dtype=np.float64).reshape(3)
    if not np.isfinite(c).all():
        raise GraspError("candidate center is not finite")
    half = 0.5 * float(width) * R[:, 1]
    return {
        "score": float(score),
        "frame": CAMERA_FRAME,
        "grasp_frame": GRASP_FRAME,
        "source_model": source_model,
        "translation_xyz": [float(v) for v in c],
        "rotation_matrix": [[float(v) for v in row] for row in R],
        "width": float(width),
        "depth": float(depth),
        "contact_points_xyz": [
            [float(v) for v in c - half],
            [float(v) for v in c + half],
        ],
        **(extra or {}),
    }


def contact_graspnet_candidates(
    grasps: Any, scores: Any, contacts: Any, openings: Any
) -> list[dict[str, Any]]:
    """Contact-GraspNet (Panda base frame: Z approach, X closing, origin at the gripper base)
    -> GraspNet frame. The grasp center is the gripper base moved ``0.1034`` m along the
    approach, where the finger pads meet the object (t = c + w/2 b - d a in the paper)."""
    G = np.asarray(grasps, dtype=np.float64).reshape(-1, 4, 4)
    S = np.asarray(scores, dtype=np.float64).reshape(-1)
    W = np.asarray(openings, dtype=np.float64).reshape(-1)
    C = (
        np.asarray(contacts, dtype=np.float64).reshape(-1, 3)
        if np.size(contacts)
        else None
    )
    if not (len(G) == len(S) == len(W)):
        raise GraspError(
            "Contact-GraspNet returned mismatched grasp/score/opening counts"
        )
    out = []
    for i in range(len(G)):
        R = G[i, :3, :3] @ ZX_NATIVE_TO_GRASPNET
        center = G[i, :3, 3] + CONTACT_GRASPNET_GRIPPER_DEPTH * R[:, 0]
        extra: dict[str, Any] = {"gripper_base_xyz": [float(v) for v in G[i, :3, 3]]}
        if C is not None:
            extra["model_contact_xyz"] = [float(v) for v in C[i]]
        out.append(
            make_candidate(
                score=S[i],
                rotation=R,
                center=center,
                width=W[i],
                depth=0.0,
                source_model="contact_graspnet",
                extra=extra,
            )
        )
    return out


def graspgenx_candidates(
    grasps: Any, scores: Any, *, fingertip_xyz: Any, width: float, tags: Any = None
) -> list[dict[str, Any]]:
    """GraspGenX (configured gripper base frame: Z approach, X closing) -> GraspNet frame.
    The grasp center is the gripper's fingertip point (``fingertip`` of its config) in that
    frame; ``width`` is the gripper's open width (the model predicts poses, not openings)."""
    G = np.asarray(grasps, dtype=np.float64).reshape(-1, 4, 4)
    S = np.asarray(scores, dtype=np.float64).reshape(-1)
    tip = np.asarray(fingertip_xyz, dtype=np.float64).reshape(3)
    if len(G) != len(S):
        raise GraspError("GraspGenX returned mismatched grasp/score counts")
    out = []
    for i in range(len(G)):
        R = G[i, :3, :3] @ ZX_NATIVE_TO_GRASPNET
        center = G[i, :3, :3] @ tip + G[i, :3, 3]
        extra: dict[str, Any] = {"gripper_base_xyz": [float(v) for v in G[i, :3, 3]]}
        if tags is not None:
            extra["candidate_source"] = (
                "diffusion" if str(tags[i]) == "diff" else str(tags[i])
            )
        out.append(
            make_candidate(
                score=S[i],
                rotation=R,
                center=center,
                width=width,
                depth=0.0,
                source_model="graspgenx",
                extra=extra,
            )
        )
    return out


def anygrasp_candidates(
    grasps: Any, source_model: str = "anygrasp"
) -> list[dict[str, Any]]:
    """AnyGrasp or a GraspNet-1Billion model (graspnetAPI ``GraspGroup``: already the GraspNet
    frame) -> normalized dicts. graspnetAPI puts the fingertips ``depth`` along the approach
    past ``translation``; the candidate's center is there, where the finger pads close (the
    EEF sits at the center), so the grasp is not 1-4 cm shallow."""
    out = []
    for g in grasps:
        R = np.asarray(g.rotation_matrix, dtype=np.float64)
        t = np.asarray(g.translation, dtype=np.float64).reshape(3)
        out.append(
            make_candidate(
                score=float(g.score),
                rotation=R,
                center=t + float(g.depth) * R[:, 0],
                width=float(g.width),
                depth=0.0,
                source_model=source_model,
                extra={
                    "height": float(g.height),
                    "graspnet_translation": [float(v) for v in t],
                    "graspnet_depth": float(g.depth),
                },
            )
        )
    return out


def rank(candidates: list[dict[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    """Score-descending, stable, at most ``max_candidates``."""
    order = sorted(range(len(candidates)), key=lambda i: (-candidates[i]["score"], i))
    return [candidates[i] for i in order[:max_candidates]]


def transform_candidate(
    T: np.ndarray, cand: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) of a normalized candidate after the rigid transform ``T`` (4x4)."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3] @ np.asarray(cand["rotation_matrix"], dtype=np.float64)
    t = T[:3, :3] @ np.asarray(cand["translation_xyz"], dtype=np.float64) + T[:3, 3]
    return R, t


def upright_placement(
    T_place_camera: np.ndarray, cam2world: np.ndarray, object_camera: np.ndarray
) -> tuple[np.ndarray, float]:
    """AnyPlace's placement kept upright: the same landing point of the object's centroid,
    but only its turn about world +z (the yaw nearest the model's rotation). Returns the
    camera-frame transform and the tilt (rad) the model's rotation had.

    A gripper that approaches from above can only turn a held object about the vertical, and
    the model's tilt is not trustworthy outside its training objects (on the box, LIBERO's
    bowl came back tipped 50-60 deg even in the gravity-aligned frame). The object then rests
    as it was held, turned and moved onto the region."""
    T_cw = np.asarray(cam2world, dtype=np.float64)
    T_wc = np.linalg.inv(T_cw)
    T_w = T_cw @ np.asarray(T_place_camera, dtype=np.float64) @ T_wc
    R = T_w[:3, :3]
    tilt = math.acos(float(np.clip(R[2, 2], -1.0, 1.0)))
    yaw = math.atan2(R[1, 0] - R[0, 1], R[0, 0] + R[1, 1])
    Rz = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    if len(object_camera):
        c = T_cw[:3, :3] @ np.asarray(object_camera, dtype=np.float64).mean(axis=0)
        c = c + T_cw[:3, 3]
    else:
        c = np.zeros(3)
    landed = R @ c + T_w[:3, 3]
    upright = rigid(Rz, landed - Rz @ c)
    return T_wc @ upright @ T_cw, tilt


def support_height(
    object_camera: np.ndarray, scene_camera: np.ndarray, cam2world: np.ndarray
) -> float:
    """World z of the surface an object rests on: the scene's points around the object's
    footprint (within 3 cm of its xy extent, below its middle), where its bottom touches the
    support. The object's own lowest visible point is not it: its bottom is usually occluded,
    so that reads high and a place would press the object into its new support. Falls back to
    that point when too few scene points surround the object."""
    T = np.asarray(cam2world, dtype=np.float64)
    obj = np.asarray(object_camera, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]
    lowest = float(np.percentile(obj[:, 2], 5))
    if len(scene_camera) == 0:
        return lowest
    scene = np.asarray(scene_camera, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]
    lo, hi = obj[:, :2].min(axis=0) - 0.03, obj[:, :2].max(axis=0) + 0.03
    ring = scene[
        np.all(scene[:, :2] >= lo, axis=1)
        & np.all(scene[:, :2] <= hi, axis=1)
        & (scene[:, 2] < float(np.median(obj[:, 2])))
    ]
    if len(ring) < 20:
        return lowest
    return float(np.median(ring[:, 2]))


def compose_placement(T_place: np.ndarray, R_grasp: np.ndarray, t_grasp: np.ndarray):
    """Placement Grasp Composition: the pick grasp pose after the object placement transform
    (``p_placed = R @ p_current + t``, same frame): ``R_place = R_T R_g``, ``t_place = R_T t_g + t_T``."""
    T = np.asarray(T_place, dtype=np.float64)
    if (
        T.shape != (4, 4)
        or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
        or not is_rotation(T[:3, :3])
    ):
        raise GraspError("placement transform is not a rigid 4x4 transform")
    return T[:3, :3] @ np.asarray(R_grasp), T[:3, :3] @ np.asarray(t_grasp) + T[:3, 3]


# ---------------------------------------------------------------------------
# grasp-to-EEF calibration


@dataclass(frozen=True)
class GraspToEef:
    """The robot's rigid relation between the GraspNet grasp frame and its EEF frame.

    ``rotation`` columns are the EEF axes expressed in the grasp frame; ``translation`` is the
    EEF origin in the grasp frame (m). ``T_world_eef = T_world_grasp @ [rotation | translation]``.

    Measuring it for a new robot (once, then a constant in the env server config):

    1. Rotation. Point the gripper straight down and read the EEF orientation. The EEF axis
       that points along the fingers toward the object is the approach (grasp +X); the axis the
       fingers slide along is the closing direction (grasp +Y); the third follows from the
       right-hand rule. Write each EEF axis as a column in grasp coordinates.
    2. Translation. Close the gripper on a thin known object and read ``eef_pos``; the grasp
       center is the midpoint between the finger pads at contact. ``translation`` is
       (grasp center -> EEF origin) expressed in the grasp frame, typically a small offset along
       the approach only. Verify by commanding ``resolve_grasp(id).eef_position`` for a planned
       grasp and checking the pads close on the candidate's ``contact_points``.

    The defaults describe LIBERO's Panda: the EEF site sits between the finger pads, its +Z is
    the approach and its fingers slide along its +Y.
    """

    rotation: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
    )
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_config(cls, cfg: Any) -> GraspToEef:
        """From ``{"rotation": 3x3, "translation": [3]}`` (a YAML/JSON mapping); None = default."""
        if cfg is None:
            return cls()
        R = np.asarray(cfg.get("rotation", cls.rotation), dtype=np.float64)
        if not is_rotation(R):
            raise GraspError(
                "grasp_to_eef.rotation must be a proper 3x3 rotation matrix"
            )
        t = np.asarray(
            cfg.get("translation", cls.translation), dtype=np.float64
        ).reshape(3)
        return cls(
            tuple(tuple(float(v) for v in row) for row in R), tuple(float(v) for v in t)
        )

    def matrix(self) -> np.ndarray:
        return rigid(np.asarray(self.rotation), np.asarray(self.translation))

    def eef_pose(self, R_grasp: np.ndarray, t_grasp: np.ndarray) -> dict[str, Any]:
        """The EEF pose (world frame) of a world-frame grasp pose: position, xyzw, yaw, pitch."""
        T = rigid(R_grasp, t_grasp) @ self.matrix()
        R = T[:3, :3]
        return {
            "eef_position": [round(float(v), 5) for v in T[:3, 3]],
            "eef_quat_xyzw": [round(v, 6) for v in quat_xyzw(R)],
            "eef_yaw": round(yaw_of(R), 5),
            "eef_pitch": round(pitch_of(R), 5),
        }


# ---------------------------------------------------------------------------
# the planner


def _png_base64(rgb: np.ndarray) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), mode="RGB").save(
        buffer, format="PNG"
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


#: ``view(camera) -> {"rgb" uint8[H,W,3], "depth" float32[H,W] m (0 = none), "intrinsic_K" 3x3,
#: "extrinsic_cam2world" 4x4}``: one camera of the current observation, upright, OpenCV frame.
View = Callable[[str], dict[str, Any]]


@dataclass
class Snapshot:
    """The Shared Observation Snapshot: one camera at one observation."""

    observation: int
    camera: str
    view: dict[str, Any]
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    signature: dict[str, Any] | None = None


class GraspPlanner:
    """Grasp and placement primitives over an env server's current observation.

    ``view`` renders one camera now; ``backends`` maps a backend name (``BACKENDS``) to its
    server URL or a client with ``call(method, args, kwargs, timeout_s=)``; ``anyplace`` and
    ``sam3`` likewise (None = off); ``grasp_to_eef`` is the robot's calibration, or one per
    arm (``{"left": ..., "right": ...}``); ``eef_pose(arm) -> (xyz, quat_xyzw)`` gives the
    current EEF for the attachment crop; ``masks`` is an external DetectionBook whose ids
    ``plan_grasp`` also accepts (the stage-7 ``env.segment`` book), else only this planner's.
    With ``masks`` the planner shares that book's :class:`Epoch`: one id counter (an id names
    one mask, whichever book holds it) and one observation clock (a motion or a new
    observation expires both books). ``state_digest() -> comparable`` is the robot-state
    fingerprint the clock compares (:meth:`Epoch.set_digest`: an unmoved robot keeps its ids);
    ``holding(arm) -> bool | None`` says whether the gripper holds something (None: unknown),
    checked before ``plan_place`` trusts an executed grasp.
    """

    def __init__(
        self,
        view: View,
        *,
        cameras: list[str],
        backends: dict[str, Any] | None = None,
        anyplace: Any | None = None,
        sam3: Any | None = None,
        grasp_to_eef: GraspToEef | dict[str, GraspToEef] | None = None,
        eef_pose: Callable[[str | None], tuple[Any, Any] | None] | None = None,
        masks: DetectionBook | None = None,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        depth_truncation: float = DEFAULT_DEPTH_TRUNCATION,
        wrist_camera: str | None = None,
        state_digest: Callable[[], Any] | None = None,
        holding: Callable[[str | None], bool | None] | None = None,
        max_approach_tilt_deg: float | None = None,
    ) -> None:
        if not cameras:
            raise ValueError("GraspPlanner needs at least one camera")
        self._view = view
        self._cameras = list(cameras)
        self._backends = {k: self._client(v) for k, v in (backends or {}).items() if v}
        unknown = sorted(set(self._backends) - set(BACKENDS))
        if unknown:
            raise ValueError(
                f"unknown grasp backends {unknown}; one of {list(BACKENDS)}"
            )
        self._anyplace = self._client(anyplace) if anyplace else None
        self._sam3 = self._client(sam3) if sam3 else None
        self._calibration = grasp_to_eef if grasp_to_eef is not None else GraspToEef()
        self._eef_pose = eef_pose
        self._external = masks
        #: A hand that can only approach from above: grasp and place candidates approaching
        #: more than this from straight down are dropped before ranking (None keeps them all).
        self._max_tilt = (
            None
            if max_approach_tilt_deg is None
            else math.radians(float(max_approach_tilt_deg))
        )
        self._epoch = masks.epoch if masks is not None else Epoch()
        if state_digest is not None:
            self._epoch.set_digest(state_digest)
        self._holding = holding
        self._book = DetectionBook(self._epoch)
        self._book.bind(self._epoch.observation)
        self._epoch.on_tick(self._on_tick)
        self._snapshots: dict[str, Snapshot] = {}
        #: cameras captured during the current primitive call (one fresh capture per call)
        self._captured: set[str] = set()
        self._max = int(max_candidates)
        self._depth_max = float(depth_truncation)
        self._wrist = wrist_camera
        #: grasp ids in rank order per inference, for the greedy candidate policy
        self._rankings: dict[str, list[str]] = {}
        #: per arm (None for one arm): the grasp claimed for execution, which the gripper
        #: holds until a place is claimed, the env resets or ``holding`` says it does not
        self._held: dict[str | None, dict[str, Any]] = {}

    @staticmethod
    def _client(v: Any) -> Any:
        if isinstance(v, str):
            from pi_embodied_services.utils.rpc.http_rpc import HttpRpcClient

            return HttpRpcClient(v)
        return v

    # -- wiring ----------------------------------------------------------------

    @classmethod
    def from_args(
        cls,
        view: View,
        *,
        cameras: list[str],
        contact_graspnet: str | None = None,
        graspgenx: str | None = None,
        anygrasp: str | None = None,
        graspnet1b: str | None = None,
        anyplace: str | None = None,
        sam3: str | None = None,
        **kwargs: Any,
    ) -> GraspPlanner | None:
        """A planner for the ``--contact-graspnet/--graspgenx/--anygrasp/--graspnet1b/--anyplace``
        URLs, or None when none was given (the env server then changes nothing)."""
        if not (contact_graspnet or graspgenx or anygrasp or graspnet1b or anyplace):
            return None
        return cls(
            view,
            cameras=cameras,
            backends={
                "contact_graspnet": contact_graspnet,
                "graspgenx": graspgenx,
                "anygrasp": anygrasp,
                "graspnet1b": graspnet1b,
            },
            anyplace=anyplace,
            sam3=sam3,
            **kwargs,
        )

    def capabilities(self) -> dict[str, Any]:
        return {
            "grasp_backends": sorted(self._backends),
            "place": self._anyplace is not None,
            "segment": self._sam3 is not None,
            "cameras": list(self._cameras),
        }

    #: RPC methods that change the robot's state: every id expires once they ran.
    MUTATING = MOTION_METHODS

    def install(self, facade: Any, *, mutating: tuple[str, ...] = MUTATING) -> None:
        """Register the primitives on ``facade._rpc`` and wrap its mutating calls with
        :meth:`invalidate` (after they ran, whatever they returned)."""
        rpc: dict[str, Callable[..., Any]] = facade._rpc
        self._epoch.install(facade, mutating)
        reset = rpc.get("env.reset")
        if reset is not None:

            def reset_and_forget(*args: Any, **kwargs: Any) -> Any:
                try:
                    return reset(*args, **kwargs)
                finally:
                    self._held.clear()

            rpc["env.reset"] = reset_and_forget
        rpc["env.claim_waypoints"] = self.claim_waypoints
        rpc["env.release_held"] = self.release_held
        rpc["env.note_grasp_closed"] = self.note_grasp_closed
        rpc["env.plan_grasp"] = self.plan_grasp
        rpc["env.next_grasp"] = self.next_grasp
        rpc["env.resolve_grasp"] = self.resolve_grasp
        rpc["env.plan_place"] = self.plan_place
        rpc["env.segment_mask"] = self.segment_mask
        rpc["env.attachment_frames"] = self.attachment_frames
        rpc["env.grasp_capabilities"] = self.capabilities
        facade._readonly_methods.update(
            {"env.resolve_grasp", "env.attachment_frames", "env.grasp_capabilities"}
        )

    # -- snapshots -------------------------------------------------------------

    @property
    def epoch(self) -> Epoch:
        return self._epoch

    def attach_masks(self, book: DetectionBook) -> None:
        """Accept ``book``'s mask ids as ``mask_id`` (a perception installed after the planner,
        on the planner's epoch: one id counter, one observation clock)."""
        if book.epoch is not self._epoch:
            raise ValueError("the mask book must share the planner's epoch")
        self._external = book

    @property
    def observation(self) -> int:
        return self._epoch.observation

    def invalidate(self) -> list[str]:
        """The robot moved: a new observation; every id so far expires (in the external
        book too). Returns this planner's."""
        dropped = self._book.ids
        self._epoch.tick()
        return dropped

    def _on_tick(self, observation: int) -> None:
        self._snapshots = {}
        self._rankings = {}

    def _fresh_call(self) -> None:
        """A primitive call starts: its first use of each camera captures a new frame."""
        self._captured = set()

    def _snapshot(self, camera: str | None) -> Snapshot:
        """The camera's frame of the current observation, checked against a fresh capture.

        Every primitive call captures the camera anew (once per call). When the new frame
        shows the same scene as the cached one (:func:`scene_changed`: noise, not a moved
        object), the cached snapshot stays, and with it every id cut from it; otherwise the
        scene changed without the robot (someone moved an object, the table was restored), a
        new observation starts (every id expires) and the new frame is the snapshot."""
        camera = camera or self._cameras[0]
        if camera not in self._cameras:
            raise GraspError(f"unknown camera {camera!r}; one of {self._cameras}")
        snap = self._snapshots.get(camera)
        if snap is not None and camera in self._captured:
            return snap
        view = self._view(camera)
        for key in ("rgb", "depth", "intrinsic_K", "extrinsic_cam2world"):
            if key not in view:
                raise GraspError(f"the view of {camera!r} has no {key!r}")
        signature = frame_signature(view["rgb"], view["depth"])
        self._captured.add(camera)
        if snap is not None and not scene_changed(snap.signature, signature):
            return snap
        if snap is not None:
            self._epoch.tick()  # the scene changed under an unmoved robot
        snap = Snapshot(self._epoch.observation, camera, view, signature=signature)
        self._snapshots[camera] = snap
        return snap

    def _expired(self) -> list[str]:
        return self._book.drain_invalidated()

    # -- masks -----------------------------------------------------------------

    def _mask_item(self, mask_id: str) -> dict[str, Any]:
        """A current mask by id, from this planner's book or the external segment book."""
        mask_id = str(mask_id)
        book = self._book
        if self._external is not None and not book.known(mask_id):
            book = self._external
        try:
            item = book.get(mask_id)
        except DetectionStale as err:
            raise GraspError(str(err)) from None
        if "mask" not in item:
            raise GraspError(f"detection {mask_id} carries no mask")
        return item

    def segment_mask(
        self, object: str, camera: str | None = None, min_score: float = 0.2
    ) -> dict:
        """Segment ``object`` (a SAM3 text prompt) in the current image of ``camera`` and
        register the best mask under a short id.

        Returns:
            dict with ``found``; when found ``id`` (e.g. ``d4``, valid until the robot moves),
            ``camera``, ``observation``, ``score``, ``area_px``, ``centroid_rc`` and
            ``world_xyz`` (median of the mask's depth pixels in the world frame, or None).

        Example:
            >>> obj = segment_mask("black bowl"); region = segment_mask("plate")
        """
        self._fresh_call()
        out = self._segment(object, camera, min_score)
        out["expired_ids"] = self._expired()
        return out

    def _segment(
        self, object: str, camera: str | None = None, min_score: float = 0.2
    ) -> dict:
        """Segment ``object`` (a SAM3 text prompt) in the current image of ``camera`` and
        register the best mask under a short id.

        Returns:
            dict with ``found``; when found ``id`` (e.g. ``d4``, valid until the robot moves),
            ``camera``, ``observation``, ``score``, ``area_px``, ``centroid_rc`` and
            ``world_xyz`` (median of the mask's depth pixels in the world frame, or None).

        Example:
            >>> obj = segment_mask("black bowl"); region = segment_mask("plate")
        """
        if self._sam3 is None:
            raise GraspError(
                "segment_mask needs a SAM3 server (start the env server with --sam3)"
            )
        snap = self._snapshot(camera)
        text = str(object).strip()
        if not text:
            raise GraspError("object must be a non-empty text prompt")
        res = self._sam3.call(
            "sam3.segment",
            (),
            {
                "image_base64": _png_base64(snap.view["rgb"]),
                "text_prompt": text,
                "min_score": float(min_score),
            },
            timeout_s=120.0,
        )
        out: dict[str, Any] = {
            "found": False,
            "camera": snap.camera,
            "observation": self._epoch.observation,
        }
        if (
            not isinstance(res, dict)
            or not res.get("found")
            or not res.get("mask_png_base64")
        ):
            out["reason"] = (
                (res or {}).get("reason", "SAM3 found no mask")
                if isinstance(res, dict)
                else "bad SAM3 reply"
            )
            return out
        mask = decode_mask_png(res["mask_png_base64"])
        if mask.shape != snap.view["depth"].shape or not mask.any():
            out["reason"] = (
                f"SAM3 mask {mask.shape} does not match the {snap.view['depth'].shape} image"
            )
            return out
        id = self._register_mask(
            snap, mask, prompt=text, score=res.get("score"), box=res.get("box")
        )
        item = self._book.get(id)
        out.update(
            {
                "found": True,
                "id": id,
                "score": None
                if item["score"] is None
                else round(float(item["score"]), 3),
                "box": item.get("box"),
                "area_px": item["area_px"],
                "centroid_rc": item["centroid_rc"],
                "world_xyz": item["world_xyz"],
            }
        )
        return out

    def _segment_held(self, prompt: str, snap: Snapshot, eef_xyz: np.ndarray) -> str:
        """The held object's mask: of every SAM3 mask for ``prompt`` (LIBERO scenes often
        have two same-named objects), the one whose points lie nearest the gripper."""
        if self._sam3 is None:
            raise GraspError(
                "plan_place after a grasp needs a SAM3 server (or pass object_mask_id)"
            )
        res = self._sam3.call(
            "sam3.segment",
            (),
            {
                "image_base64": _png_base64(snap.view["rgb"]),
                "text_prompt": str(prompt),
                "min_score": 0.2,
                "all": True,
            },
            timeout_s=120.0,
        )
        found = res.get("detections") if isinstance(res, dict) else None
        if found is None and isinstance(res, dict) and res.get("found"):
            found = [res]  # a server without all=True answers with its best mask
        best: tuple[float, np.ndarray, dict] | None = None
        T = np.asarray(snap.view["extrinsic_cam2world"], dtype=np.float64)
        for det in found or []:
            if not det.get("mask_png_base64"):
                continue
            mask = decode_mask_png(det["mask_png_base64"])
            if mask.shape != snap.view["depth"].shape or not mask.any():
                continue
            pts, _scene = object_points(
                snap.view["depth"],
                snap.view["intrinsic_K"],
                mask,
                depth_max=self._depth_max,
            )
            if len(pts) < 10:
                continue
            w = pts.astype(np.float64) @ T[:3, :3].T + T[:3, 3]
            dist = float(np.min(np.linalg.norm(w - eef_xyz, axis=1)))
            if best is None or dist < best[0]:
                best = (dist, mask, det)
        if best is None:
            raise GraspError(f"could not segment the held {prompt!r} near the gripper")
        _dist, mask, det = best
        return self._register_mask(
            snap, mask, prompt=str(prompt), score=det.get("score"), box=det.get("box")
        )

    def _register_mask(self, snap: Snapshot, mask: np.ndarray, **meta: Any) -> str:
        view = snap.view
        desc = describe_mask(mask, view["depth"], view["intrinsic_K"])
        world = None
        pts, _scene = object_points(
            view["depth"], view["intrinsic_K"], mask, depth_max=self._depth_max
        )
        if len(pts) >= 10:
            T = np.asarray(view["extrinsic_cam2world"], dtype=np.float64)
            w = pts.astype(np.float64) @ T[:3, :3].T + T[:3, 3]
            world = [round(float(v), 4) for v in np.median(w, axis=0)]
        item = {
            "kind": "mask",
            "camera": snap.camera,
            **meta,
            **desc,
            "world_xyz": world,
            "mask": mask,
        }
        id = self._book.add(item, "d")
        snap.masks[id] = mask
        return id

    def _mask_for(
        self, snap: Snapshot, object: str | None, mask_id: str | None
    ) -> tuple[str, np.ndarray]:
        if bool(object) == (mask_id is not None):
            raise GraspError("give exactly one of object (text) or mask_id")
        if mask_id is not None:
            item = self._mask_item(mask_id)
            if item.get("camera") not in (None, snap.camera):
                raise GraspError(
                    f"mask {mask_id} was cut from camera {item.get('camera')!r}, not {snap.camera!r}"
                )
            mask = np.asarray(item["mask"]).astype(bool)
            if mask.shape != snap.view["depth"].shape:
                raise GraspError(
                    f"mask {mask_id} {mask.shape} does not match the {snap.view['depth'].shape} view"
                )
            return str(mask_id), mask
        seg = self._segment(str(object), snap.camera)
        if not seg["found"]:
            raise GraspError(f"could not segment {object!r}: {seg.get('reason')}")
        return seg["id"], snap.masks[seg["id"]]

    # -- grasps ----------------------------------------------------------------

    def _backend(self, backend: str | None) -> tuple[str, Any]:
        if not self._backends:
            raise GraspError(
                "no grasp backend: start the env server with --contact-graspnet, --graspgenx, --anygrasp or --graspnet1b"
            )
        if backend is None:
            name = next(n for n in BACKENDS if n in self._backends)
        else:
            name = str(backend)
            if name not in self._backends:
                raise GraspError(
                    f"backend {backend!r} is not configured; available: {sorted(self._backends)}"
                )
        return name, self._backends[name]

    def _calibration_for(self, arm: str | None) -> GraspToEef:
        if isinstance(self._calibration, dict):
            if arm is None:
                raise GraspError(f"arm is required; one of {sorted(self._calibration)}")
            try:
                return self._calibration[arm]
            except KeyError:
                raise GraspError(
                    f"unknown arm {arm!r}; one of {sorted(self._calibration)}"
                ) from None
        return self._calibration

    def _world_candidate(
        self, snap: Snapshot, cand: dict[str, Any], arm: str | None
    ) -> dict[str, Any]:
        T = np.asarray(snap.view["extrinsic_cam2world"], dtype=np.float64)
        R, t = transform_candidate(T, cand)
        cal = self._calibration_for(arm)
        contacts = [
            [round(float(v), 5) for v in T[:3, :3] @ np.asarray(c) + T[:3, 3]]
            for c in cand["contact_points_xyz"]
        ]
        return {
            "kind": "grasp",
            "camera": snap.camera,
            "frame": "world",
            "score": round(float(cand["score"]), 4),
            "backend": cand["source_model"],
            "position": [round(float(v), 5) for v in t],
            "approach": [round(float(v), 5) for v in R[:, 0]],
            "closing": [round(float(v), 5) for v in R[:, 1]],
            "rotation_matrix": [[round(float(v), 6) for v in row] for row in R],
            "width_m": round(float(cand["width"]), 4),
            "contact_points": contacts,
            **cal.eef_pose(R, t),
            "arm": arm,
            # camera-frame pose for the placement composition
            "_camera_R": np.asarray(cand["rotation_matrix"], dtype=np.float64),
            "_camera_t": np.asarray(cand["translation_xyz"], dtype=np.float64),
        }

    @staticmethod
    def _public(item: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in item.items() if not k.startswith("_") and k != "mask"}

    def plan_grasp(
        self,
        object: str | None = None,
        mask_id: str | None = None,
        camera: str | None = None,
        backend: str | None = None,
        arm: str | None = None,
        max_candidates: int | None = None,
        next_after: str | None = None,
        reason: str = "",
    ) -> dict:
        """Predict grasps for one object in the current observation, world frame, best first.
        With ``next_after`` (a grasp id that failed before the robot moved): reject it and return
        the next rank of its plan without planning again (``next_grasp``).

        Args:
            object: text prompt of the object to grasp (segmented with SAM3), or
            mask_id: a mask id from ``segment_mask`` / ``segment`` of the current observation.
            camera: the RGB-D camera to plan from (default the first configured).
            backend: ``contact_graspnet`` | ``graspgenx`` | ``anygrasp`` | ``graspnet1b`` (default: the
                first configured).
            arm: with two arms, which arm's calibration gives the EEF pose.
            max_candidates: at most this many (default 10).

        Returns:
            dict with ``observation``, ``mask_id``, ``backend``, ``candidates`` (each: ``id``
            such as ``g3``, ``rank``, ``score``, ``position`` (grasp center, m), ``approach``
            and ``closing`` unit vectors, ``width_m``, ``contact_points``, ``eef_position``,
            ``eef_quat_xyzw``, ``eef_yaw``, ``eef_pitch``), ``active`` (the id to try first)
            and ``expired_ids``. Ids are valid until the robot moves. When a candidate is
            refused (unreachable, collision, empty grasp), call ``next_grasp(id)`` for the next
            rank instead of planning again.

        Example:
            >>> g = plan_grasp("black bowl"); pose = resolve_grasp(g["active"], standoff=0.1)
        """
        if next_after:
            return self.next_grasp(str(next_after), reason)
        if not (object or "").strip() and not mask_id:
            raise ValueError("give object (text) or mask_id")
        name, client = self._backend(backend)
        self._fresh_call()
        snap = self._snapshot(camera)
        mask_id, mask = self._mask_for(snap, object, mask_id)
        view = snap.view
        obj_pts, _scene = object_points(
            view["depth"], view["intrinsic_K"], mask, depth_max=self._depth_max
        )
        if len(obj_pts) < 20:
            raise GraspError(
                f"mask {mask_id} has only {len(obj_pts)} pixels with depth; segment again"
            )
        n = int(max_candidates or self._max)
        # world up in the camera frame: R_cam_world @ z, with cam2world's rotation transposed
        up = np.asarray(view["extrinsic_cam2world"], dtype=np.float64)[
            :3, :3
        ].T @ np.array([0.0, 0.0, 1.0])
        started = time.perf_counter()
        res = client.call(
            f"{name}.plan",
            (),
            {
                "depth": np.ascontiguousarray(view["depth"], dtype=np.float32),
                "intrinsic_K": np.asarray(view["intrinsic_K"], dtype=np.float64),
                "mask": np.ascontiguousarray(mask, dtype=np.uint8),
                "rgb": np.ascontiguousarray(view["rgb"], dtype=np.uint8),
                # With the approach filter the ranking is cut after it: ask for more.
                "max_candidates": n if self._max_tilt is None else min(4 * n, 64),
                "up_direction_camera": [float(v) for v in up],
            },
            timeout_s=600.0,
        )
        latency = time.perf_counter() - started
        if not isinstance(res, dict) or "candidates" not in res:
            raise GraspError(f"{name} server returned no candidates field: {res!r}")
        if res.get("grasp_frame", GRASP_FRAME) != GRASP_FRAME:
            raise GraspError(
                f"{name} server answered in frame {res.get('grasp_frame')!r}, not {GRASP_FRAME!r}"
            )
        T_cw = np.asarray(view["extrinsic_cam2world"], dtype=np.float64)
        candidates, too_steep = self._from_above(list(res["candidates"]), T_cw)
        ranked = rank(candidates, n)
        # Where the object rests (the support under it), so a place can set it down at the
        # same height above its new support (claim_waypoints / plan_place).
        support_z = support_height(obj_pts, _scene, T_cw)
        ids: list[str] = []
        out_cands: list[dict[str, Any]] = []
        for i, cand in enumerate(ranked):
            item = self._world_candidate(snap, cand, arm)
            item.update(
                {
                    "rank": i,
                    "mask_id": mask_id,
                    "rejected": False,
                    "_support_z": support_z,
                }
            )
            id = self._book.add(item, "g")
            ids.append(id)
            out_cands.append(self._public(self._book.get(id)))
        for id in ids:
            self._rankings[id] = ids
        return {
            "observation": self._epoch.observation,
            "camera": snap.camera,
            "mask_id": mask_id,
            "backend": name,
            "frame": "world",
            "candidate_count": len(out_cands),
            "candidates": out_cands,
            "active": ids[0] if ids else None,
            **(
                {}
                if self._max_tilt is None
                else {
                    "max_approach_tilt_deg": round(math.degrees(self._max_tilt), 1),
                    "dropped_too_steep": too_steep,
                }
            ),
            "latency_s": round(latency, 3),
            "server": {k: v for k, v in res.items() if k not in ("candidates",)},
            "expired_ids": self._expired(),
        }

    def _from_above(
        self, candidates: list[dict[str, Any]], cam2world: np.ndarray
    ) -> tuple[list[dict[str, Any]], int]:
        """The camera-frame candidates whose approach is within the planner's tilt limit of
        straight down (all without one), and how many were dropped. Refuses, with the angles,
        when none remains: the hand could not execute any of them."""
        if self._max_tilt is None or not candidates:
            return candidates, 0
        R_cw = np.asarray(cam2world, dtype=np.float64)[:3, :3]
        tilts = []
        for c in candidates:
            a = R_cw @ np.asarray(c["rotation_matrix"], dtype=np.float64)[:, 0]
            tilts.append(
                math.acos(float(np.clip(-a[2] / max(np.linalg.norm(a), 1e-12), -1, 1)))
            )
        keep = [c for c, t in zip(candidates, tilts) if t <= self._max_tilt]
        if not keep:
            raise GraspError(
                f"all {len(candidates)} grasp candidates approach more than "
                f"{math.degrees(self._max_tilt):.0f} deg from straight down (the least "
                f"{math.degrees(min(tilts)):.0f} deg), which this hand cannot execute; "
                "plan from another camera or segment the object again"
            )
        return keep, len(candidates) - len(keep)

    def _grasp_item(self, grasp_id: str) -> dict[str, Any]:
        try:
            item = self._book.get(str(grasp_id))
        except DetectionStale as exc:
            raise GraspError(str(exc)) from None
        if item.get("kind") not in ("grasp", "placement"):
            raise GraspError(f"{grasp_id} is a {item.get('kind')} id, not a grasp")
        return item

    def next_grasp(self, grasp_id: str, reason: str = "") -> dict:
        """Greedy Grasp Candidate Policy: reject ``grasp_id`` (a structured, candidate-specific
        failure: unreachable, collision, the fingers closed on nothing) and activate the next
        rank of the same inference, without planning again.

        Returns:
            dict with ``rejected`` (the id and ``reason``), ``active`` (the next id or None when
            the ranking is exhausted: plan again from a new observation) and its ``candidate``.
        """
        item = self._grasp_item(grasp_id)
        item["rejected"] = True
        item["reject_reason"] = str(reason)
        self._book.reject(str(grasp_id))
        ranking = self._rankings.get(str(grasp_id), [])
        after = (
            ranking[ranking.index(str(grasp_id)) + 1 :]
            if str(grasp_id) in ranking
            else []
        )
        following = [i for i in after if not self._book.get(i)["rejected"]]
        active = following[0] if following else None
        return {
            "rejected": {"id": str(grasp_id), "reason": str(reason)},
            "active": active,
            "candidate": self._public(self._book.get(active)) if active else None,
            "remaining": len(following),
            "expired_ids": self._expired(),
        }

    def resolve_grasp(self, grasp_id: str, standoff: float = 0.0) -> dict:
        """The EEF pose to command for a grasp or place id of the current observation.

        Args:
            grasp_id: a ``g`` or ``p`` id from ``plan_grasp`` / ``plan_place``.
            standoff: metres to back off along the approach (0.1 = a pre-grasp pose above it).

        Returns:
            dict with ``eef_position`` (world, m), ``eef_quat_xyzw``, ``eef_yaw``, ``eef_pitch``,
            ``approach``, ``width_m`` and ``id``. Refused (``stale`` in the error) when the id is
            from an earlier observation: the robot moved since it was planned.

        Example:
            >>> pre = resolve_grasp("g1", standoff=0.1); move_to(pre["eef_position"])
        """
        item = self._grasp_item(grasp_id)
        approach = np.asarray(item["approach"], dtype=np.float64)
        pos = (
            np.asarray(item["eef_position"], dtype=np.float64)
            - float(standoff) * approach
        )
        return {
            "id": str(grasp_id),
            "kind": item["kind"],
            "observation": self._epoch.observation,
            "eef_position": [round(float(v), 5) for v in pos],
            "eef_quat_xyzw": item["eef_quat_xyzw"],
            "eef_yaw": item["eef_yaw"],
            "eef_pitch": item["eef_pitch"],
            "approach": item["approach"],
            "width_m": item["width_m"],
            "standoff_m": float(standoff),
            "arm": item.get("arm"),
        }

    # -- execution -------------------------------------------------------------

    def claim_waypoints(
        self,
        grasp_id: str,
        standoff: float = DEFAULT_STANDOFF_M,
        lift: float = DEFAULT_LIFT_M,
    ) -> dict:
        """Claim one grasp or place id of the current observation for execution.

        The whole path is resolved now, from this one candidate: a grasp gives ``pre_grasp``
        (``standoff`` back along the approach), ``grasp`` and ``lift`` (``lift`` straight up);
        a place gives ``pre_place``, ``place`` and ``retreat`` (back to ``pre_place``). The
        motions that follow expire the id, which is why they run on these coordinates and not
        on the id: the evidence chain ends at this claim, one observation, one candidate.

        A claimed grasp is remembered as the arm's held grasp (``plan_place`` accepts its id
        after the grasp, from the new observation); ``release_held`` forgets it once the
        place's executor opened the hand.

        Returns:
            dict with ``id``, ``kind``, ``observation``, ``waypoints`` (name -> world
            ``[x, y, z]`` of the EEF), ``steps`` (in order: ``{"to": name, "gripper": -1|+1}``
            or ``{"gripper": ...}`` to close / open in place), ``eef_quat_xyzw``,
            ``eef_yaw``, ``eef_pitch``, ``approach``, ``width_m``, ``arm``.

        Example:
            >>> c = claim_waypoints(plan_grasp("black bowl")["active"])
            >>> move_to(c["waypoints"]["pre_grasp"], gripper=-1)
        """
        # The scene must still be the one the id was planned on: a fresh capture of its
        # camera expires it when something moved since (the robot did not).
        self._fresh_call()
        self._snapshot(self._grasp_item(grasp_id)["camera"])
        item = self._grasp_item(grasp_id)
        grasp_id = str(grasp_id)
        if item.get("rejected"):
            raise GraspError(
                f"{grasp_id} was rejected ({item.get('reject_reason') or 'no reason'}); "
                "execute the plan's active candidate"
            )
        offsets = {"standoff": float(standoff), "lift": float(lift)}
        for name, value in offsets.items():
            if not (0.0 <= value <= MAX_WAYPOINT_OFFSET_M):
                raise GraspError(
                    f"{name} must be within [0, {MAX_WAYPOINT_OFFSET_M}] m, got {value}"
                )
        at = np.asarray(item["eef_position"], dtype=np.float64)
        approach = np.asarray(item["approach"], dtype=np.float64)
        pre = at - offsets["standoff"] * approach
        arm = item.get("arm")
        if item["kind"] == "grasp":
            waypoints = {
                "pre_grasp": pre,
                "grasp": at,
                "lift": at + np.array([0.0, 0.0, offsets["lift"]]),
            }
            steps = [
                {"to": "pre_grasp", "gripper": -1},
                {"to": "grasp", "gripper": -1},
                {"gripper": 1},
                {"to": "lift", "gripper": 1},
            ]
            self._held[arm] = {
                "grasp_id": grasp_id,
                "arm": arm,
                "camera": item["camera"],
                "observation": item["observation"],
                "mask_id": item.get("mask_id"),
                "prompt": self._prompt_of(item.get("mask_id")),
                "width_m": item["width_m"],
                # How high the EEF holds it above the surface it rested on.
                # (planned; note_grasp_closed replaces it with the TCP measured at the close)
                "support_z": item.get("_support_z"),
                "eef_above_support_m": (
                    None
                    if item.get("_support_z") is None
                    else float(at[2]) - float(item["_support_z"])
                ),
            }
        else:
            waypoints = {"pre_place": pre, "place": at, "retreat": pre}
            steps = [
                {"to": "pre_place", "gripper": 1},
                {"to": "place", "gripper": 1},
                {"gripper": -1},
                {"to": "retreat", "gripper": -1},
            ]
            # The held record stays until the executor has opened the hand (release_held): a
            # place that stalls on the way still holds the object and may plan again.
        return {
            "id": grasp_id,
            "kind": item["kind"],
            "observation": self._epoch.observation,
            "arm": arm,
            "waypoints": {
                k: [round(float(v), 5) for v in w] for k, w in waypoints.items()
            },
            "steps": steps,
            "eef_quat_xyzw": item["eef_quat_xyzw"],
            "eef_yaw": item["eef_yaw"],
            "eef_pitch": item["eef_pitch"],
            "approach": item["approach"],
            "width_m": item["width_m"],
            "standoff_m": offsets["standoff"],
            "lift_m": offsets["lift"] if item["kind"] == "grasp" else None,
            "expired_ids": self._expired(),
        }

    def note_grasp_closed(self, arm: str | None = None) -> dict:
        """The executor closed the fingers on ``arm``'s claimed grasp: the EEF height above the
        object's support is measured now (the TCP where it closed), not the planned grasp's."""
        record = self._held.get(arm)
        pose = self._eef_pose(arm) if self._eef_pose is not None else None
        if record is None or pose is None or record.get("support_z") is None:
            return {"noted": False}
        z = float(np.asarray(pose[0], dtype=np.float64)[2])
        record["eef_above_support_m"] = z - float(record["support_z"])
        return {
            "noted": True,
            "eef_above_support_m": round(record["eef_above_support_m"], 4),
        }

    def release_held(self, arm: str | None = None, opened: bool = False) -> dict:
        """Forget ``arm``'s held grasp once the hand let go: the executor's open step
        completed (``opened``) or ``holding(arm)`` confirms the fingers hold nothing. A hand
        that still holds (or cannot be checked and was not opened) keeps the record.

        Returns:
            dict with ``released`` (bool) and ``held`` (the grasp id kept, or None).
        """
        record = self._held.get(arm)
        if record is None:
            return {"released": False, "held": None}
        empty = self._holding(arm) is False if self._holding is not None else False
        if opened or empty:
            self._held.pop(arm, None)
            return {"released": True, "held": None}
        return {"released": False, "held": record["grasp_id"]}

    def held(self, arm: str | None = None) -> dict[str, Any] | None:
        """The grasp claimed for ``arm`` and not yet placed (None when there is none)."""
        record = self._held.get(arm)
        return None if record is None else dict(record)

    def _prompt_of(self, mask_id: str | None) -> str | None:
        if mask_id is None:
            return None
        for book in (self._book, self._external):
            if book is not None and book.known(str(mask_id)):
                try:
                    prompt = book.get(str(mask_id)).get("prompt")
                except DetectionStale:
                    return None
                return str(prompt) if prompt else None
        return None

    # -- placement -------------------------------------------------------------

    def plan_place(
        self,
        region_mask_id: str | None = None,
        grasp_id: str = "",
        object_mask_id: str | None = None,
        max_candidates: int | None = None,
        keep_tilt: bool = False,
        region: str | None = None,
        camera: str | None = None,
    ) -> dict:
        """Where to hold the grasped object so it comes to rest on the placement region.

        By default a placement keeps only AnyPlace's turn about the vertical (``upright_placement``),
        a deviation from upstream AnyPlace, whose full rotation may tilt the object:
        ``keep_tilt=True`` keeps the model's full rotation (a tilted insertion or placement)
        and drops the from-above approach check; only an executor that servos the full
        orientation (LIBERO's) can run such a place.

        AnyPlace predicts the object's placement transform from the object mask and the
        placement-region mask; the place grasp pose is that transform applied to the grasp
        pose (Placement Grasp Composition). Two ways to name the grasp:

        - a current ``g`` id, before executing it: the object mask, the region mask and the
          grasp must come from the same observation snapshot (same camera, no motion in
          between), or the call is refused;
        - the id of the grasp executed last (``claim_waypoints``), after the grasp: the object
          is in the gripper, so the grasp pose is the EEF's actual pose now, and the object
          mask (default: the grasp's text prompt segmented again) and the region mask come
          from the current observation. Refused when ``holding`` says the gripper is empty.

        Args:
            region_mask_id: mask id of the local surface it goes onto / into.
            grasp_id: the grasp (``g`` id) the object is or will be held with.
            object_mask_id: mask id of the object being placed; default the mask the grasp
                was planned on (before the grasp) or its prompt segmented now (after).

        Returns:
            dict with ``candidates`` (each: ``id`` such as ``p2``, ``eef_position``,
            ``eef_quat_xyzw``, ``eef_yaw``, ``eef_pitch``, ``object_position``: where the
            object's grasp center lands), ``active``, ``held`` and ``expired_ids``.

        Example:
            >>> p = plan_place(region["id"], "g1"); claim_waypoints(p["active"])
        """
        if not grasp_id:
            raise ValueError("plan_place needs grasp_id")
        if not region_mask_id:
            text = (region or "").strip()
            if not text:
                raise ValueError("give region (text) or region_mask_id")
            seg = self.segment_mask(text, camera=camera)
            if not seg.get("found"):
                raise ValueError(
                    f"could not segment region '{text}': {seg.get('reason', 'no mask')}"
                )
            region_mask_id = str(seg["id"])
        if self._anyplace is None:
            raise GraspError(
                "plan_place needs an AnyPlace server (start the env server with --anyplace)"
            )
        grasp_id = str(grasp_id)
        self._fresh_call()
        held = next((h for h in self._held.values() if h["grasp_id"] == grasp_id), None)
        live = self._book.known(grasp_id) and grasp_id in self._book.ids
        if held is not None and not live:
            return self._plan_place_held(
                held, region_mask_id, object_mask_id, max_candidates, keep_tilt
            )
        self._snapshot(
            self._grasp_item(grasp_id)["camera"]
        )  # fresh: expires a moved scene
        grasp = self._grasp_item(grasp_id)
        if grasp["kind"] != "grasp":
            raise GraspError(
                f"{grasp_id} is a placement id; plan_place needs the pick grasp"
            )
        if object_mask_id is None:
            object_mask_id = grasp["mask_id"]
        obj = self._mask_item(object_mask_id)
        region = self._mask_item(region_mask_id)
        observations = {
            "object_mask": obj["observation"],
            "region_mask": region["observation"],
            "grasp": grasp["observation"],
        }
        cameras = {obj.get("camera"), region.get("camera"), grasp["camera"]} - {None}
        if len(set(observations.values())) != 1 or len(cameras) != 1:
            raise GraspError(
                "plan_place needs the object mask, the region mask and the grasp from one observation "
                f"snapshot (same camera, no motion in between); got observations {observations}, cameras {sorted(cameras)}"
            )
        if grasp["mask_id"] != str(object_mask_id):
            raise GraspError(
                f"grasp {grasp_id} was planned on mask {grasp['mask_id']}, not {object_mask_id}"
            )
        return self._place(
            self._snapshot(grasp["camera"]),
            grasp_id=grasp_id,
            object_mask_id=str(object_mask_id),
            object_mask=np.asarray(obj["mask"]),
            region_mask_id=str(region_mask_id),
            region_mask=np.asarray(region["mask"]),
            grasp_R_camera=grasp["_camera_R"],
            grasp_t_camera=grasp["_camera_t"],
            width_m=grasp["width_m"],
            arm=grasp.get("arm"),
            max_candidates=max_candidates,
            held=False,
            keep_tilt=keep_tilt,
        )

    def _plan_place_held(
        self,
        held: dict[str, Any],
        region_mask_id: str,
        object_mask_id: str | None,
        max_candidates: int | None,
        keep_tilt: bool = False,
    ) -> dict:
        """``plan_place`` after the grasp: the grasp pose is the EEF's pose now."""
        arm = held["arm"]
        grasp_id = held["grasp_id"]
        if self._holding is not None and self._holding(arm) is False:
            self._held.pop(arm, None)
            raise GraspError(
                f"the gripper holds nothing: grasp {grasp_id} did not keep its object; "
                "plan and execute a new grasp"
            )
        pose = self._eef_pose(arm) if self._eef_pose is not None else None
        if pose is None:
            raise GraspError(
                "plan_place after a grasp needs the EEF pose, which this server does not give"
            )
        camera = self._mask_item(region_mask_id).get("camera") or held["camera"]
        snap = self._snapshot(camera)  # fresh: a changed scene expires the region mask
        region = self._mask_item(region_mask_id)
        if object_mask_id is None:
            if not held.get("prompt"):
                raise GraspError(
                    f"grasp {grasp_id} was planned on a mask without a text prompt; "
                    "segment the held object and pass object_mask_id"
                )
            object_mask_id = self._segment_held(
                held["prompt"], snap, np.asarray(pose[0], dtype=np.float64)
            )
        obj = self._mask_item(object_mask_id)
        observations = {
            "object_mask": obj["observation"],
            "region_mask": region["observation"],
        }
        cameras = {obj.get("camera"), region.get("camera"), snap.camera} - {None}
        if len(set(observations.values())) != 1 or len(cameras) != 1:
            raise GraspError(
                "plan_place needs the object and region masks from the current observation "
                f"(same camera); got observations {observations}, cameras {sorted(cameras)}"
            )
        cal = self._calibration_for(arm)
        T_world_eef = rigid(quat_xyzw_matrix(pose[1]), np.asarray(pose[0], dtype=float))
        T_world_grasp = T_world_eef @ np.linalg.inv(cal.matrix())
        T_camera_grasp = (
            np.linalg.inv(
                np.asarray(snap.view["extrinsic_cam2world"], dtype=np.float64)
            )
            @ T_world_grasp
        )
        return self._place(
            snap,
            grasp_id=grasp_id,
            object_mask_id=str(object_mask_id),
            object_mask=np.asarray(obj["mask"]),
            region_mask_id=str(region_mask_id),
            region_mask=np.asarray(region["mask"]),
            grasp_R_camera=T_camera_grasp[:3, :3],
            grasp_t_camera=T_camera_grasp[:3, 3],
            width_m=held["width_m"],
            arm=arm,
            max_candidates=max_candidates,
            held=True,
            keep_tilt=keep_tilt,
            eef_above_support=held.get("eef_above_support_m"),
        )

    def _place(
        self,
        snap: Snapshot,
        *,
        grasp_id: str,
        object_mask_id: str,
        object_mask: np.ndarray,
        region_mask_id: str,
        region_mask: np.ndarray,
        grasp_R_camera: np.ndarray,
        grasp_t_camera: np.ndarray,
        width_m: float,
        arm: str | None,
        max_candidates: int | None,
        held: bool,
        eef_above_support: float | None = None,
        keep_tilt: bool = False,
    ) -> dict:
        """AnyPlace's placements composed with the grasp, kept upright, refused when not
        executable; with ``eef_above_support`` (the executed grasp's EEF height above the
        surface the object rested on) each is set down onto the region's top surface under
        its landing point instead of the model's landing height."""
        view = snap.view
        n = int(max_candidates or self._max)
        started = time.perf_counter()
        res = self._anyplace.call(
            "anyplace.plan",
            (),
            {
                "rgb": np.ascontiguousarray(view["rgb"], dtype=np.uint8),
                "depth": np.ascontiguousarray(view["depth"], dtype=np.float32),
                "intrinsic_K": np.asarray(view["intrinsic_K"], dtype=np.float64),
                "object_mask": np.ascontiguousarray(object_mask, dtype=np.uint8),
                "region_mask": np.ascontiguousarray(region_mask, dtype=np.uint8),
                "max_candidates": n,
                # AnyPlace predicts in a gravity-aligned frame; it converts back to the camera's.
                "extrinsic_cam2world": np.asarray(
                    view["extrinsic_cam2world"], dtype=np.float64
                ),
            },
            timeout_s=600.0,
        )
        latency = time.perf_counter() - started
        if not isinstance(res, dict) or "placements" not in res:
            raise GraspError(f"anyplace server returned no placements field: {res!r}")
        T_cw = np.asarray(view["extrinsic_cam2world"], dtype=np.float64)
        cal = self._calibration_for(arm)
        obj_cam, _ = object_points(
            view["depth"], view["intrinsic_K"], object_mask, depth_max=self._depth_max
        )
        reg_cam, _ = object_points(
            view["depth"], view["intrinsic_K"], region_mask, depth_max=self._depth_max
        )
        region_w = reg_cam.astype(np.float64) @ T_cw[:3, :3].T + T_cw[:3, 3]
        region_xy = region_w[:: max(1, len(region_w) // 3000), :2]
        centre = np.median(region_w[:, :2], axis=0) if len(region_w) else None
        ids: list[str] = []
        cands: list[dict[str, Any]] = []
        refused: list[dict[str, Any]] = []
        accepted: list[tuple[float, dict[str, Any]]] = []
        for i, pl in enumerate(list(res["placements"])[:n]):
            T_model = np.asarray(pl["transform_matrix"], dtype=np.float64)
            if keep_tilt:
                T_place, tilt = T_model, 0.0
            else:
                T_place, tilt = upright_placement(T_model, T_cw, obj_cam)
            R_c, t_c = compose_placement(T_place, grasp_R_camera, grasp_t_camera)
            R = T_cw[:3, :3] @ R_c
            t = T_cw[:3, :3] @ t_c + T_cw[:3, 3]
            why = self._place_refusal(
                R,
                T_place,
                T_cw,
                obj_cam,
                region_w,
                check_approach=not keep_tilt,
                max_tilt=MAX_PLACE_TILT_RAD
                if self._max_tilt is None
                else min(self._max_tilt, MAX_PLACE_TILT_RAD),
            )
            if why:
                refused.append({"rank": i, "reason": why})
                continue
            footprint, offset, landed_at = self._landing(
                T_place, T_cw, obj_cam, region_xy, centre
            )
            if footprint is not None and footprint < PLACE_MIN_FOOTPRINT:
                refused.append(
                    {
                        "rank": i,
                        "reason": f"only {footprint:.0%} of the object's footprint would be "
                        "over the region (its edge)",
                    }
                )
                continue
            settled = None
            if eef_above_support is not None and len(obj_cam) and len(region_w):
                landed = T_place[:3, :3] @ obj_cam.astype(np.float64).mean(axis=0)
                landed = T_cw[:3, :3] @ (landed + T_place[:3, 3]) + T_cw[:3, 3]
                near = np.linalg.norm(region_w[:, :2] - landed[:2], axis=1) < 0.08
                top = float(
                    np.percentile(region_w[near if near.any() else ...][:, 2], 90)
                )
                eef_z = cal.eef_pose(R, t)["eef_position"][2]
                settled = top + eef_above_support + PLACE_SETTLE_CLEARANCE_M - eef_z
                t = t + np.array([0.0, 0.0, settled])
                t_c = t_c + T_cw[:3, :3].T @ np.array([0.0, 0.0, settled])
            item = {
                "kind": "placement",
                "camera": snap.camera,
                "frame": "world",
                "rank": i,
                "score": None
                if pl.get("score") is None
                else round(float(pl["score"]), 4),
                "backend": "anyplace",
                "model_tilt_deg": round(math.degrees(tilt), 1),
                "footprint_on_region": None
                if footprint is None
                else round(footprint, 3),
                "landing_offset_m": None if offset is None else round(offset, 4),
                "settled_m": None if settled is None else round(settled, 4),
                "source_grasp_id": grasp_id,
                "object_mask_id": object_mask_id,
                "region_mask_id": region_mask_id,
                "mask_id": object_mask_id,
                "position": [round(float(v), 5) for v in t],
                # Where the object's (visible points') centroid lands, not the grasp point: a
                # bowl grasped at its rim lands its centre a rim's width from the fingers.
                "object_position": [round(float(v), 5) for v in t]
                if landed_at is None
                else [
                    round(float(landed_at[0]), 5),
                    round(float(landed_at[1]), 5),
                    round(float(landed_at[2] + (settled or 0.0)), 5),
                ],
                "approach": [round(float(v), 5) for v in R[:, 0]],
                "closing": [round(float(v), 5) for v in R[:, 1]],
                "rotation_matrix": [[round(float(v), 6) for v in row] for row in R],
                "width_m": width_m,
                "contact_points": [],
                **cal.eef_pose(R, t),
                "arm": arm,
                "rejected": False,
                "_camera_R": R_c,
                "_camera_t": t_c,
            }
            accepted.append((offset if offset is not None else 0.0, item))
        # Nearest the region's centre first (stable: the model's order among equals).
        for _offset, item in sorted(accepted, key=lambda a: a[0]):
            id = self._book.add(item, "p")
            ids.append(id)
            cands.append(self._public(self._book.get(id)))
        if not ids and refused:
            raise GraspError(
                f"none of AnyPlace's {len(refused)} placements can be executed from above "
                f"onto the region: {refused}; segment the region again or place by hand"
            )
        for id in ids:
            self._rankings[id] = ids
        return {
            "observation": self._epoch.observation,
            "camera": snap.camera,
            "grasp_id": grasp_id,
            "held": held,
            "refused": refused,
            "object_mask_id": object_mask_id,
            "region_mask_id": region_mask_id,
            "candidate_count": len(cands),
            "candidates": cands,
            "active": ids[0] if ids else None,
            "latency_s": round(latency, 3),
            "server": {k: v for k, v in res.items() if k != "placements"},
            "expired_ids": self._expired(),
        }

    @staticmethod
    def _landing(
        T_place_camera: np.ndarray,
        cam2world: np.ndarray,
        object_camera: np.ndarray,
        region_xy: np.ndarray,
        centre: np.ndarray | None,
    ) -> tuple[float | None, float | None, np.ndarray | None]:
        """(fraction of the placed object's points over the region in xy, the xy distance of
        its centroid from the region's centre, that centroid in the world), or None for what
        cannot be measured."""
        if len(object_camera) == 0 or len(region_xy) == 0 or centre is None:
            return None, None, None
        T = np.asarray(T_place_camera, dtype=np.float64)
        obj = np.asarray(object_camera, dtype=np.float64)[
            :: max(1, len(object_camera) // 800)
        ]
        placed = (obj @ T[:3, :3].T + T[:3, 3]) @ cam2world[:3, :3].T + cam2world[:3, 3]
        d = np.linalg.norm(placed[:, None, :2] - region_xy[None, :, :], axis=2).min(
            axis=1
        )
        # A coarse depth image spaces the region's points further apart than the tolerance.
        probe = region_xy[:: max(1, len(region_xy) // 300)]
        gaps = np.linalg.norm(probe[:, None] - region_xy[None], axis=2)
        gaps[gaps == 0] = np.inf
        nearest = gaps.min(axis=1)
        nearest = nearest[np.isfinite(nearest)]
        spacing = float(np.median(nearest)) if len(nearest) else 0.0
        footprint = float(np.mean(d <= max(PLACE_FOOTPRINT_TOL_M, 1.5 * spacing)))
        offset = float(np.linalg.norm(placed[:, :2].mean(axis=0) - centre))
        return footprint, offset, placed.mean(axis=0)

    @staticmethod
    def _place_refusal(
        R_world: np.ndarray,
        T_place_camera: np.ndarray,
        cam2world: np.ndarray,
        object_camera: np.ndarray,
        region_world: np.ndarray,
        check_approach: bool = True,
        max_tilt: float = MAX_PLACE_TILT_RAD,
    ) -> str | None:
        """Why a composed place pose cannot be executed, or None: the gripper must approach
        from above (within ``MAX_PLACE_TILT_RAD`` of straight down), and the placed object's
        centroid must land over the region, not beside it or high above it."""
        approach = np.asarray(R_world, dtype=np.float64)[:, 0]
        tilt = math.acos(float(np.clip(-approach[2], -1.0, 1.0)))
        if check_approach and tilt > max_tilt:
            return (
                f"approach {[round(float(v), 2) for v in approach]} is "
                f"{math.degrees(tilt):.0f} deg from straight down"
            )
        if len(object_camera) == 0 or len(region_world) == 0:
            return None
        T = np.asarray(T_place_camera, dtype=np.float64)
        placed_cam = object_camera.astype(np.float64) @ T[:3, :3].T + T[:3, 3]
        c = cam2world[:3, :3] @ placed_cam.mean(axis=0) + cam2world[:3, 3]
        lo, hi = region_world.min(axis=0), region_world.max(axis=0)
        if np.any(c[:2] < lo[:2] - PLACE_XY_MARGIN_M) or np.any(
            c[:2] > hi[:2] + PLACE_XY_MARGIN_M
        ):
            return f"the object would land at {[round(float(v), 3) for v in c]}, off the region"
        if not (lo[2] - PLACE_Z_MARGIN_M <= c[2] <= hi[2] + PLACE_MAX_CLEARANCE_M):
            return (
                f"the object would land at z {c[2]:.3f}, not on the region "
                f"(z {lo[2]:.3f}-{hi[2]:.3f})"
            )
        return None

    # -- attachment probe ------------------------------------------------------

    def attachment_frames(
        self, arm: str | None = None, crop_fraction: float = 0.45
    ) -> dict:
        """The frames a VLM needs to judge whether the gripper holds the object: every
        configured camera, the wrist camera whole and the others cropped around the EEF's
        projection (when the env server supplies the EEF pose)."""
        frames = []
        eef = self._eef_pose(arm) if self._eef_pose is not None else None
        for camera in self._cameras:
            view = self._view(camera)
            rgb = np.asarray(view["rgb"], dtype=np.uint8)
            crop = None
            if camera != self._wrist and eef is not None:
                crop = self._crop_around(
                    view, np.asarray(eef[0], dtype=np.float64), crop_fraction
                )
                if crop is not None:
                    r0, c0, r1, c1 = crop
                    rgb = rgb[r0:r1, c0:c1]
            frames.append(
                {"camera": camera, "png_base64": _png_base64(rgb), "crop_rc": crop}
            )
        return {
            "observation": self._epoch.observation,
            "frames": frames,
            "eef_position": None if eef is None else [float(v) for v in eef[0]],
        }

    @staticmethod
    def _crop_around(
        view: dict[str, Any], point_world: np.ndarray, fraction: float
    ) -> list[int] | None:
        T = np.linalg.inv(np.asarray(view["extrinsic_cam2world"], dtype=np.float64))
        p = T[:3, :3] @ point_world + T[:3, 3]
        if p[2] <= 1e-6:
            return None
        K = np.asarray(view["intrinsic_K"], dtype=np.float64)
        col = K[0, 0] * p[0] / p[2] + K[0, 2]
        row = K[1, 1] * p[1] / p[2] + K[1, 2]
        h, w = np.asarray(view["depth"]).shape
        half_h, half_w = int(h * fraction / 2), int(w * fraction / 2)
        r0 = int(min(max(row - half_h, 0), h - 2 * half_h))
        c0 = int(min(max(col - half_w, 0), w - 2 * half_w))
        return [r0, c0, r0 + 2 * half_h, c0 + 2 * half_w]


__all__ = [
    "BACKENDS",
    "CAMERA_FRAME",
    "CONTACT_GRASPNET_GRIPPER_DEPTH",
    "DEFAULT_LIFT_M",
    "DEFAULT_STANDOFF_M",
    "GRASP_FRAME",
    "GRASP_URL_KEYS",
    "ZX_NATIVE_TO_GRASPNET",
    "GraspError",
    "GraspPlanner",
    "GraspToEef",
    "anygrasp_candidates",
    "backproject",
    "compose_placement",
    "contact_graspnet_candidates",
    "graspgenx_candidates",
    "make_candidate",
    "object_points",
    "pitch_of",
    "quat_xyzw",
    "orientation_error",
    "quat_xyzw_matrix",
    "rotvec_of",
    "rank",
    "transform_candidate",
    "upright_placement",
    "yaw_of",
]


# ---------------------------------------------------------------------------
# env server wiring


def add_grasp_arguments(parser: Any) -> None:
    """``--contact-graspnet/--graspgenx/--anygrasp/--graspnet1b/--anyplace <url>`` and
    ``--grasp-to-eef``."""
    for name, what in (
        ("contact-graspnet", "Contact-GraspNet server URL"),
        ("graspgenx", "GraspGenX server URL"),
        ("anygrasp", "AnyGrasp server URL"),
        ("graspnet1b", "GraspNet-1Billion server URL (graspnet-baseline or GSNet)"),
        ("anyplace", "AnyPlace server URL"),
    ):
        parser.add_argument(
            f"--{name}",
            default=None,
            help=f"{what}: adds env.plan_grasp (and env.plan_place)",
        )
    parser.add_argument(
        "--grasp-to-eef",
        default=None,
        help='grasp-to-EEF calibration JSON {"rotation": 3x3, "translation": [3]} (or one per arm '
        '{"left": {...}, "right": {...}}); default: the Panda hand (see utils/grasp.GraspToEef)',
    )
    parser.add_argument(
        "--max-approach-tilt-deg",
        type=float,
        default=None,
        help="drop grasp and place candidates approaching more than this from straight down "
        "(default: the robot's; LIBERO 30, others keep every candidate)",
    )


def install_grasp_planner(
    facade: Any,
    args: Any,
    *,
    view: View,
    cameras: list[str],
    eef_pose: Callable[[str | None], tuple[Any, Any] | None],
    perception: Any | None = None,
    wrist_camera: str | None = None,
    mutating: tuple[str, ...] = MOTION_METHODS,
) -> GraspPlanner | None:
    """Install a planner for parsed :func:`add_grasp_arguments` on a constructed facade
    (nothing without a grasp or place URL): the primitives, their ids sharing ``perception``'s
    book and epoch when there is one (``env.detect`` ids are accepted as ``mask_id``). Their
    ``code.api`` entries are the robot's manifest's (``common/grasp.json``)."""
    planner = GraspPlanner.from_args(
        view,
        cameras=cameras,
        masks=perception.book if perception is not None else None,
        sam3=getattr(args, "sam3", None)
        or (perception.sam3 if perception is not None else None),
        eef_pose=eef_pose,
        wrist_camera=wrist_camera,
        **urls_from_args(args),
    )
    if planner is None:
        return None
    planner.install(facade, mutating=mutating)
    return planner


def urls_from_args(args: Any) -> dict[str, Any]:
    """The ``GraspPlanner.from_args`` keyword arguments held by parsed ``add_grasp_arguments``."""
    import json

    cal: GraspToEef | dict[str, GraspToEef] | None = None
    if getattr(args, "grasp_to_eef", None):
        raw = json.loads(args.grasp_to_eef)
        if "rotation" not in raw and "translation" not in raw:
            cal = {arm: GraspToEef.from_config(cfg) for arm, cfg in raw.items()}
        else:
            cal = GraspToEef.from_config(raw)
    return {
        "contact_graspnet": getattr(args, "contact_graspnet", None),
        "graspgenx": getattr(args, "graspgenx", None),
        "anygrasp": getattr(args, "anygrasp", None),
        "graspnet1b": getattr(args, "graspnet1b", None),
        "anyplace": getattr(args, "anyplace", None),
        "grasp_to_eef": cal,
        **(
            {"max_approach_tilt_deg": float(args.max_approach_tilt_deg)}
            if getattr(args, "max_approach_tilt_deg", None) is not None
            else {}
        ),
    }
