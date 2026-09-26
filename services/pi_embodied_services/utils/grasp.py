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

The grasp servers (``components/graspnet_server.py``, ``graspgenx_server.py``,
``anygrasp_server.py``) answer in the **GraspNet grasp frame** in the camera's OpenCV frame:
origin at the grasp center (between the finger pads), X = approach, Y = closing (the fingers
slide along it), Z = X x Y. Each server converts its model's native frame itself (the
``*_candidates`` functions below); the env server only ever sees normalized candidates.

The env server knows what no model server knows: the current camera frames, their intrinsics
and extrinsics, and the robot's grasp-to-EEF calibration. :class:`GraspPlanner` therefore lives
in the env server. It

- takes an *observation snapshot* (one camera's rgb, depth, K, cam2world at the current step),
- segments the target (SAM3) or takes a mask id, asks a grasp server, transforms the candidates
  to the world frame and hands out short ids (``g3``) bound to that snapshot,
- composes AnyPlace's object placement transform with a chosen grasp into the place grasp pose
  (``p1``), refusing masks and grasps from different snapshots,
- resolves a grasp id to the robot's EEF pose (``GraspToEef``) for the motion primitives,
- expires every id when the robot moves (``invalidate``: the facade's mutating RPCs are wrapped).

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
    DetectionBook,
    DetectionStale,
    decode_mask_png,
    describe_mask,
)
from pi_embodied_services.utils.logging import get_logger

logger = get_logger("grasp")

GRASP_FRAME = "graspnet"
CAMERA_FRAME = "opencv"
BACKENDS = ("contact_graspnet", "graspgenx", "anygrasp")
#: Contact-GraspNet's Panda gripper: base frame origin to the finger pads along the approach.
CONTACT_GRASPNET_GRIPPER_DEPTH = 0.1034
#: Depth (m) beyond which a pixel is ignored when building the object's point cloud.
DEFAULT_DEPTH_TRUNCATION = 2.0
#: Candidates returned to the planner per inference.
DEFAULT_MAX_CANDIDATES = 10
#: How far above the grasp (against the approach) the pre-grasp pose sits.
DEFAULT_STANDOFF_M = 0.10

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


def anygrasp_candidates(grasps: Any) -> list[dict[str, Any]]:
    """AnyGrasp (graspnetAPI ``GraspGroup``: already the GraspNet frame) -> normalized dicts."""
    out = []
    for g in grasps:
        out.append(
            make_candidate(
                score=float(g.score),
                rotation=np.asarray(g.rotation_matrix, dtype=np.float64),
                center=np.asarray(g.translation, dtype=np.float64),
                width=float(g.width),
                depth=float(g.depth),
                source_model="anygrasp",
                extra={"height": float(g.height)},
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


class EvidenceBook(DetectionBook):
    """A DetectionBook whose ids carry their kind: ``d`` masks, ``g`` grasps, ``p`` placements."""

    def add(self, detection: dict[str, Any], prefix: str = "d") -> str:  # type: ignore[override]
        if self._observation is None:
            raise RuntimeError("no observation is bound; call bind() first")
        self._counter += 1
        id = f"{prefix}{self._counter}"
        self._items[id] = {**detection, "id": id, "observation": self._observation}
        self._history[id] = self._observation
        return id


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


class GraspPlanner:
    """Grasp and placement primitives over an env server's current observation.

    ``view`` renders one camera now; ``backends`` maps a backend name (``BACKENDS``) to its
    server URL or a client with ``call(method, args, kwargs, timeout_s=)``; ``anyplace`` and
    ``sam3`` likewise (None = off); ``grasp_to_eef`` is the robot's calibration, or one per
    arm (``{"left": ..., "right": ...}``); ``eef_pose(arm) -> (xyz, quat_xyzw)`` gives the
    current EEF for the attachment crop; ``masks`` is an external DetectionBook whose ids
    ``plan_grasp`` also accepts (the stage-7 ``env.segment`` book), else only this planner's.
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
        self._book = EvidenceBook()
        self._observation = 0
        self._book.bind(0)
        self._snapshots: dict[str, Snapshot] = {}
        self._max = int(max_candidates)
        self._depth_max = float(depth_truncation)
        self._wrist = wrist_camera
        #: grasp ids in rank order per inference, for the greedy candidate policy
        self._rankings: dict[str, list[str]] = {}

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
        graspnet: str | None = None,
        graspgenx: str | None = None,
        anygrasp: str | None = None,
        anyplace: str | None = None,
        sam3: str | None = None,
        **kwargs: Any,
    ) -> GraspPlanner | None:
        """A planner for the ``--graspnet/--graspgenx/--anygrasp/--anyplace`` URLs, or None when
        none was given (the env server then changes nothing)."""
        if not (graspnet or graspgenx or anygrasp or anyplace):
            return None
        return cls(
            view,
            cameras=cameras,
            backends={
                "contact_graspnet": graspnet,
                "graspgenx": graspgenx,
                "anygrasp": anygrasp,
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
    MUTATING = (
        "env.step",
        "env.chunk_step",
        "env.reset",
        "env.move_delta",
        "env.rotate_delta",
        "env.set_gripper",
        "env.recover_joint_posture",
        "code.run",
    )

    def install(self, facade: Any, *, mutating: tuple[str, ...] = MUTATING) -> None:
        """Register the primitives on ``facade._rpc`` and wrap its mutating calls with
        :meth:`invalidate` (after they ran, whatever they returned)."""
        rpc: dict[str, Callable[..., Any]] = facade._rpc
        for name in mutating:
            fn = rpc.get(name)
            if fn is None:
                continue

            def wrapped(*args: Any, _fn: Callable[..., Any] = fn, **kwargs: Any) -> Any:
                try:
                    return _fn(*args, **kwargs)
                finally:
                    self.invalidate()

            rpc[name] = wrapped
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

    def primitives(self) -> tuple[Any, ...]:
        """The planner's entries for the env server's primitive registry (``code.api``,
        ``components/code_api.py``), all in the high tier and none moving the robot."""
        from pi_embodied_services.components.code_api import Param, Primitive

        arm = {
            "arm": Param("string", "which arm's calibration (two-arm robots)", False)
        }
        return (
            Primitive(
                "plan_grasp",
                "env.plan_grasp",
                "Grasp candidates for one object of the current observation, world frame, best first; ids (g3) expire when the robot moves.",
                {
                    "object": Param(
                        "string", "SAM3 text prompt of the object (or mask_id)", False
                    ),
                    "mask_id": Param(
                        "string", "a mask id of this observation (or object)", False
                    ),
                    "camera": Param(
                        "string", "RGB-D camera (default the first configured)", False
                    ),
                    "backend": Param(
                        "string", "contact_graspnet | graspgenx | anygrasp", False
                    ),
                    **arm,
                    "max_candidates": Param("integer", "default 10", False),
                },
            ),
            Primitive(
                "next_grasp",
                "env.next_grasp",
                "Greedy candidate policy: reject a grasp id after a structured failure and activate the next rank of its plan.",
                {
                    "grasp_id": Param("string", "the failed id"),
                    "reason": Param("string", "why it failed", False),
                },
            ),
            Primitive(
                "resolve_grasp",
                "env.resolve_grasp",
                "The EEF pose to command for a grasp or place id; refused when the id is from an earlier observation.",
                {
                    "grasp_id": Param("string", "a g or p id"),
                    "standoff": Param(
                        "number", "m backed off along the approach", False
                    ),
                },
            ),
            Primitive(
                "plan_place",
                "env.plan_place",
                "AnyPlace: place poses (p ids) for a held object on a region; object mask, region mask and grasp must share one observation.",
                {
                    "object_mask_id": Param("string", "mask id of the object"),
                    "region_mask_id": Param(
                        "string", "mask id of the placement region"
                    ),
                    "grasp_id": Param("string", "the grasp the object is held with"),
                    "max_candidates": Param("integer", "default 5", False),
                },
            ),
            Primitive(
                "segment_mask",
                "env.segment_mask",
                "Segment an object (SAM3 text) in the current observation and register its mask under a short id (d2).",
                {
                    "object": Param("string", "SAM3 text prompt"),
                    "camera": Param(
                        "string", "camera (default the first configured)", False
                    ),
                    "min_score": Param("number", "default 0.2", False),
                },
            ),
        )

    # -- snapshots -------------------------------------------------------------

    @property
    def observation(self) -> int:
        return self._observation

    def invalidate(self) -> list[str]:
        """The robot moved: a new observation; every id so far expires. Returns them."""
        self._observation += 1
        self._snapshots = {}
        self._rankings = {}
        return self._book.bind(self._observation)

    def _snapshot(self, camera: str | None) -> Snapshot:
        camera = camera or self._cameras[0]
        if camera not in self._cameras:
            raise GraspError(f"unknown camera {camera!r}; one of {self._cameras}")
        snap = self._snapshots.get(camera)
        if snap is None:
            view = self._view(camera)
            for key in ("rgb", "depth", "intrinsic_K", "extrinsic_cam2world"):
                if key not in view:
                    raise GraspError(f"the view of {camera!r} has no {key!r}")
            snap = Snapshot(self._observation, camera, view)
            self._snapshots[camera] = snap
        return snap

    def _expired(self) -> list[str]:
        return self._book.drain_invalidated()

    # -- masks -----------------------------------------------------------------

    def _mask_item(self, mask_id: str) -> dict[str, Any]:
        """A current mask by id, from this planner's book or the external segment book."""
        try:
            return self._book.get(str(mask_id))
        except DetectionStale as own:
            if self._external is None:
                raise GraspError(str(own)) from None
            try:
                item = self._external.get(str(mask_id))
            except DetectionStale as ext:
                raise GraspError(str(ext)) from None
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
            "observation": self._observation,
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
                "no grasp backend: start the env server with --graspnet, --graspgenx or --anygrasp"
            )
        if backend is None:
            name = next(n for n in BACKENDS if n in self._backends)
        else:
            name = {"graspnet": "contact_graspnet"}.get(str(backend), str(backend))
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
    ) -> dict:
        """Predict grasps for one object in the current observation, world frame, best first.

        Args:
            object: text prompt of the object to grasp (segmented with SAM3), or
            mask_id: a mask id from ``segment_mask`` / ``segment`` of the current observation.
            camera: the RGB-D camera to plan from (default the first configured).
            backend: ``contact_graspnet`` | ``graspgenx`` | ``anygrasp`` (default: the first configured).
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
        name, client = self._backend(backend)
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
                "max_candidates": n,
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
        ranked = rank(list(res["candidates"]), n)
        ids: list[str] = []
        out_cands: list[dict[str, Any]] = []
        for i, cand in enumerate(ranked):
            item = self._world_candidate(snap, cand, arm)
            item.update({"rank": i, "mask_id": mask_id, "rejected": False})
            id = self._book.add(item, "g")
            ids.append(id)
            out_cands.append(self._public(self._book.get(id)))
        for id in ids:
            self._rankings[id] = ids
        return {
            "observation": self._observation,
            "camera": snap.camera,
            "mask_id": mask_id,
            "backend": name,
            "frame": "world",
            "candidate_count": len(out_cands),
            "candidates": out_cands,
            "active": ids[0] if ids else None,
            "latency_s": round(latency, 3),
            "server": {k: v for k, v in res.items() if k not in ("candidates",)},
            "expired_ids": self._expired(),
        }

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
            "observation": self._observation,
            "eef_position": [round(float(v), 5) for v in pos],
            "eef_quat_xyzw": item["eef_quat_xyzw"],
            "eef_yaw": item["eef_yaw"],
            "eef_pitch": item["eef_pitch"],
            "approach": item["approach"],
            "width_m": item["width_m"],
            "standoff_m": float(standoff),
            "arm": item.get("arm"),
        }

    # -- placement -------------------------------------------------------------

    def plan_place(
        self,
        object_mask_id: str,
        region_mask_id: str,
        grasp_id: str,
        max_candidates: int | None = None,
    ) -> dict:
        """Where to hold the grasped object so it comes to rest on the placement region.

        AnyPlace predicts the object's placement transform from the object mask and the
        placement-region mask; the place grasp pose is that transform applied to the chosen
        grasp (Placement Grasp Composition). The three ids must come from the same observation
        snapshot (same camera, no motion in between), or the call is refused.

        Args:
            object_mask_id: mask id of the object being placed.
            region_mask_id: mask id of the local surface it goes onto / into.
            grasp_id: the grasp (``g`` id) the object is or will be held with.

        Returns:
            dict with ``candidates`` (each: ``id`` such as ``p2``, ``eef_position``,
            ``eef_quat_xyzw``, ``eef_yaw``, ``eef_pitch``, ``object_position``: where the
            object's grasp center lands), ``active`` and ``expired_ids``.

        Example:
            >>> p = plan_place(obj["id"], region["id"], "g1"); resolve_grasp(p["active"], 0.05)
        """
        if self._anyplace is None:
            raise GraspError(
                "plan_place needs an AnyPlace server (start the env server with --anyplace)"
            )
        grasp = self._grasp_item(grasp_id)
        if grasp["kind"] != "grasp":
            raise GraspError(
                f"{grasp_id} is a placement id; plan_place needs the pick grasp"
            )
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
        snap = self._snapshot(grasp["camera"])
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
                "object_mask": np.ascontiguousarray(
                    np.asarray(obj["mask"]), dtype=np.uint8
                ),
                "region_mask": np.ascontiguousarray(
                    np.asarray(region["mask"]), dtype=np.uint8
                ),
                "max_candidates": n,
            },
            timeout_s=600.0,
        )
        latency = time.perf_counter() - started
        if not isinstance(res, dict) or "placements" not in res:
            raise GraspError(f"anyplace server returned no placements field: {res!r}")
        T_cw = np.asarray(view["extrinsic_cam2world"], dtype=np.float64)
        cal = self._calibration_for(grasp.get("arm"))
        ids: list[str] = []
        cands: list[dict[str, Any]] = []
        for i, pl in enumerate(list(res["placements"])[:n]):
            R_c, t_c = compose_placement(
                np.asarray(pl["transform_matrix"]),
                grasp["_camera_R"],
                grasp["_camera_t"],
            )
            R = T_cw[:3, :3] @ R_c
            t = T_cw[:3, :3] @ t_c + T_cw[:3, 3]
            item = {
                "kind": "placement",
                "camera": snap.camera,
                "frame": "world",
                "rank": i,
                "score": None
                if pl.get("score") is None
                else round(float(pl["score"]), 4),
                "backend": "anyplace",
                "source_grasp_id": str(grasp_id),
                "object_mask_id": str(object_mask_id),
                "region_mask_id": str(region_mask_id),
                "mask_id": str(object_mask_id),
                "position": [round(float(v), 5) for v in t],
                "object_position": [round(float(v), 5) for v in t],
                "approach": [round(float(v), 5) for v in R[:, 0]],
                "closing": [round(float(v), 5) for v in R[:, 1]],
                "rotation_matrix": [[round(float(v), 6) for v in row] for row in R],
                "width_m": grasp["width_m"],
                "contact_points": [],
                **cal.eef_pose(R, t),
                "arm": grasp.get("arm"),
                "rejected": False,
                "_camera_R": R_c,
                "_camera_t": t_c,
            }
            id = self._book.add(item, "p")
            ids.append(id)
            cands.append(self._public(self._book.get(id)))
        for id in ids:
            self._rankings[id] = ids
        return {
            "observation": self._observation,
            "camera": snap.camera,
            "grasp_id": str(grasp_id),
            "object_mask_id": str(object_mask_id),
            "region_mask_id": str(region_mask_id),
            "candidate_count": len(cands),
            "candidates": cands,
            "active": ids[0] if ids else None,
            "latency_s": round(latency, 3),
            "server": {k: v for k, v in res.items() if k != "placements"},
            "expired_ids": self._expired(),
        }

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
            "observation": self._observation,
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
    "DEFAULT_STANDOFF_M",
    "GRASP_FRAME",
    "ZX_NATIVE_TO_GRASPNET",
    "EvidenceBook",
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
    "rank",
    "transform_candidate",
    "yaw_of",
]


# ---------------------------------------------------------------------------
# env server wiring


def add_grasp_arguments(parser: Any) -> None:
    """``--graspnet/--graspgenx/--anygrasp/--anyplace <url>`` and ``--grasp-to-eef``."""
    for name, what in (
        ("graspnet", "Contact-GraspNet server URL"),
        ("graspgenx", "GraspGenX server URL"),
        ("anygrasp", "AnyGrasp server URL"),
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
        "graspnet": getattr(args, "graspnet", None),
        "graspgenx": getattr(args, "graspgenx", None),
        "anygrasp": getattr(args, "anygrasp", None),
        "anyplace": getattr(args, "anyplace", None),
        "grasp_to_eef": cal,
    }
