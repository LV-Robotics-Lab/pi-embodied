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

"""Code mode (``code.run``) on the real Franka servers: the RLinf and Polymetis single-arm
backends and the dual-arm rig (utils/code_real.py ``RealCodeMode``).

A program's motions run through the tools' facade methods, so the servers' own limits stay in
the path (Polymetis' per-call caps, workspace and floor; RLinf's pose clip and stop-polled servo
loops; ``--ik``'s reach check). pi's tool checks reach the server with ``code.set_limits``: the
per-call ``max_move_m`` (``--max-move``, tightened by the task's documented limit) and
``max_rotate_rad``, and the ``--workspace-xy`` / ``--z-floor`` box (``workspace_xy``,
``z_floor_m``), applied like pi's tools apply them: a move that ends outside the box is refused
unless it moves back toward it.

Refused in code mode: the dual rig's ``recover_joint_posture``, a joint-space reset of both arms
whose travel no per-call cap covers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pi_embodied_services.utils.code_real import RealCodeMode, vec3

#: Registry methods refused in code mode (see the module doc).
UNBOUNDED = frozenset({"env.recover_joint_posture"})


class FrankaCodeMode(RealCodeMode):
    """pi's Franka limits, each motion's translation and the video camera (the first external
    view, else the wrist: pi's episode video)."""

    _CODE_LIMITS = {
        "max_move_m": True,
        "max_rotate_rad": True,
        "z_floor_m": True,
        "workspace_xy": False,
    }

    def _video_frame(self, obs: dict) -> np.ndarray | None:
        extra = obs.get("extra_view_images")
        if isinstance(extra, np.ndarray) and extra.ndim == 4 and len(extra):
            return extra[0]
        main = obs.get("main_images")
        return main if isinstance(main, np.ndarray) else None

    def _code_tcp(self, arm: str | None) -> np.ndarray:
        """The TCP position the workspace box is checked in (single arm: the base frame)."""
        state = self._rpc["env.get_robot_state"]()
        return np.asarray(state["raw_base_state"]["tcp_pose"], dtype=np.float64)[:3]

    def _code_move_m(self, method: str, kwargs: dict) -> float:
        """move_delta's delta; a rotation turns the TCP in place, the gripper moves no TCP."""
        if method == "env.move_delta":
            return float(np.linalg.norm(vec3(kwargs["delta_xyz"], "delta_xyz")))
        return 0.0

    def _code_check(self, method: str, kwargs: dict) -> None:
        """Refuse a program's motion as pi's tools would, before it is commanded."""
        if method in UNBOUNDED:
            raise ValueError(
                f"{method[4:]} is not available in code mode: its motion is not bounded per "
                "call; ask the operator or use the tool"
            )
        if method == "env.rotate_delta":
            rpy, limit = (
                vec3(kwargs["delta_rpy"], "delta_rpy"),
                self._limit("max_rotate_rad"),
            )
            if not np.linalg.norm(rpy) <= limit:
                raise ValueError(
                    f"delta_rpy rotates {np.linalg.norm(rpy):.4f} rad; the limit is {limit} rad "
                    "per call. Split the rotation into smaller calls."
                )
        if method != "env.move_delta":
            return
        delta, limit = vec3(kwargs["delta_xyz"], "delta_xyz"), self._limit("max_move_m")
        if not np.linalg.norm(delta) <= limit:
            raise ValueError(
                f"delta_xyz moves {np.linalg.norm(delta):.4f} m; the limit is {limit} m per "
                "call. Split the motion into smaller calls."
            )
        box: Any = self._limit("workspace_xy")
        floor = self._limit("z_floor_m")
        tcp = self._code_tcp(kwargs.get("arm"))

        def outside(p: np.ndarray) -> float:
            d = max(0.0, floor - p[2])
            if box is not None:
                d += max(0.0, box[0] - p[0], p[0] - box[1])
                d += max(0.0, box[2] - p[1], p[1] - box[3])
            return d

        target = tcp + delta
        # As pi's tools: refused when it ends outside, unless it moves back toward the box.
        if outside(target) > 1e-6 and outside(target) >= outside(tcp) - 1e-6:
            where = f"x {box[0]}..{box[1]}, y {box[2]}..{box[3]}, " if box else ""
            raise ValueError(
                f"the move ends at {np.round(target, 3).tolist()}, outside the workspace "
                f"({where}z >= {floor} m; --workspace-xy / --z-floor)"
            )

    def set_code_limits(self, **limits: Any) -> dict[str, Any]:
        box = limits.get("workspace_xy")
        if box is not None and not (
            len(box) == 4 and box[0] < box[1] and box[2] < box[3]
        ):
            raise ValueError("workspace_xy must be [xmin, xmax, ymin, ymax]")
        return super().set_code_limits(**limits)


__all__ = ["UNBOUNDED", "FrankaCodeMode"]
