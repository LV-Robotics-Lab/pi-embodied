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

"""Wrist-view alignment (``--align-wrist``): OpenETA's ``compute_wrist_alignment``
(agent/tools/grasp_geometry.py:1200) as a server primitive, ``env.align_wrist``.

The gripper centre is projected into the wrist camera: the pixel the target should sit on. The
target pixel, lifted with its depth, gives how far the target sits beside that point in the
camera's image plane; that offset at the target's depth (``((u - u_d) z / fx, (v - v_d) z / fy,
0)``), rotated into the world and clamped to ``max_correction_m``, is the lateral correction.
``execute=False`` (the default) only reports it; ``execute=True`` then moves by it through the
robot's own motion method (LIBERO ``env.move_to`` to the aligned position, Franka the bounded
``env.move_delta``), so its limits, reach checks and stop handling apply. A robot supplies the
wrist view (``rgb``, ``depth`` m, ``intrinsic_K``, ``extrinsic_cam2world``), the gripper centre
in the world, and that move.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

DEFAULT_MAX_CORRECTION_M = 0.03


def _px(x: float) -> int:
    """Half-up rounding, as the pi client (Math.round) rounds pixels."""
    return int(np.floor(x + 0.5))


def alignment(
    K: np.ndarray,
    cam2world: np.ndarray,
    target: np.ndarray,
    gripper: np.ndarray,
    max_correction: float,
) -> dict[str, Any]:
    """The clamped lateral correction putting ``target`` (world) on the gripper centre's pixel."""
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(cam2world, dtype=np.float64)
    R, o = T[:3, :3], T[:3, 3]
    t = R.T @ (np.asarray(target, dtype=np.float64) - o)
    g = R.T @ (np.asarray(gripper, dtype=np.float64) - o)
    if not t[2] > 1e-4:
        raise ValueError("the target is not in front of the wrist camera")
    if not g[2] > 1e-4:
        raise ValueError("the gripper centre does not project into the wrist camera")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ud, vd = fx * g[0] / g[2] + cx, fy * g[1] / g[2] + cy
    u, v = fx * t[0] / t[2] + cx, fy * t[1] / t[2] + cy
    raw = R @ np.array([(u - ud) * t[2] / fx, (v - vd) * t[2] / fy, 0.0])
    norm = float(np.linalg.norm(raw))
    delta = raw * (max_correction / norm) if norm > max_correction else raw
    return {
        "desired_pixel": [_px(vd), _px(ud)],
        "target_pixel": [_px(v), _px(u)],
        "residual_px": round(float(np.hypot(u - ud, v - vd)), 1),
        "target_depth_m": round(float(t[2]), 4),
        "delta_world": [round(float(x), 5) for x in delta],
        "raw_correction_m": round(norm, 4),
        "clamped": bool(norm > max_correction),
        "aligned_xyz": [round(float(x), 5) for x in np.asarray(gripper) + delta],
    }


class WristAligner:
    """``env.align_wrist`` over one robot: ``view()`` the current wrist view, ``gripper()`` the
    gripper centre (world), ``move(aligned_xyz, delta_world)`` the correction through the
    robot's motion method (its result is returned as ``moved``)."""

    def __init__(
        self,
        view: Callable[[], dict[str, Any]],
        gripper: Callable[[], Any],
        move: Callable[[np.ndarray, np.ndarray], dict[str, Any]],
        *,
        move_with: str,
    ) -> None:
        self._view = view
        self._gripper = gripper
        self._move = move
        self._move_with = move_with

    def align_wrist(
        self,
        row: int,
        col: int,
        max_correction_m: float = DEFAULT_MAX_CORRECTION_M,
        execute: bool = False,
    ) -> dict:
        """Align the gripper with a target pixel of the current wrist image.

        Args:
            row, col: the target's pixel in the wrist image (row 0 = top).
            max_correction_m: the largest correction, m (0.005-0.05, default 0.03).
            execute: also move by the correction (default False: only report it).

        Returns:
            dict with ``desired_pixel`` (where the gripper centre projects), ``target_pixel``,
            ``residual_px``, ``target_depth_m``, ``delta_world`` (the lateral correction, m),
            ``clamped``, ``aligned_xyz`` (the EEF position after it) and, with ``execute``,
            ``moved`` (the motion's result). Raises when the pixel has no depth.

        Example:
            >>> a = align_wrist(250, 270)                   # look first
            >>> align_wrist(250, 270, execute=True)         # then move by it
        """
        view = self._view()
        depth = np.asarray(view["depth"], dtype=np.float64)
        row, col = int(row), int(col)
        h, w = depth.shape[:2]
        if not (0 <= row < h and 0 <= col < w):
            raise ValueError(
                f"pixel ({row}, {col}) out of bounds for the {h}x{w} wrist image"
            )
        z = float(depth[row, col])
        if not (np.isfinite(z) and z > 0):
            raise ValueError(
                f"no depth at wrist pixel ({row}, {col}); pick another pixel"
            )
        K = np.asarray(view["intrinsic_K"], dtype=np.float64)
        T = np.asarray(view["extrinsic_cam2world"], dtype=np.float64)
        p = np.array(
            [(col - K[0, 2]) * z / K[0, 0], (row - K[1, 2]) * z / K[1, 1], z, 1.0]
        )
        target = (T @ p)[:3]
        limit = float(np.clip(float(max_correction_m), 0.005, 0.05))
        gripper = np.asarray(self._gripper(), dtype=np.float64).reshape(3)
        out = alignment(K, T, target, gripper, limit)
        out.update(
            {
                "target_world": [round(float(x), 5) for x in target],
                "max_correction_m": limit,
                "executed": bool(execute),
                "move_with": self._move_with,
            }
        )
        if execute:
            out["moved"] = self._move(
                np.asarray(out["aligned_xyz"], dtype=np.float64),
                np.asarray(out["delta_world"], dtype=np.float64),
            )
        return out

    def install(self, facade: Any) -> None:
        """Register ``env.align_wrist``; its code.api entry is the robot manifest's ``align_wrist``
        (``requires: ["align_wrist"]``, ../../packages/embodied/src/primitives/manifests)."""
        facade._rpc["env.align_wrist"] = self.align_wrist


def add_align_wrist_argument(parser: Any) -> None:
    parser.add_argument(
        "--align-wrist",
        action="store_true",
        help="serve env.align_wrist (wrist-view alignment, utils/wrist_alignment.py)",
    )
