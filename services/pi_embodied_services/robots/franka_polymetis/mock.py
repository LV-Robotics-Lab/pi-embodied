# Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
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
# Adapted from Show-Harness core/franka/franka_interface.py (MockRobot) and
# core/franka/camera_utils.py (MockCamera).
# Modified by pi-embodied: test-only (constructing either class outside a test run
# raises); the robot tracks the commanded setpoint, models a surface the gripper
# cannot descend through, an object between the fingers (the measured width stops at
# it and the grasp flag is set), fingers that ignore commands (``jammed``) and a lost
# controller; the camera returns a fixed RGB-D frame with intrinsics.

"""Test doubles for the Polymetis NUC and the RealSense cameras. Not a simulator.

They exist so the env server's RPC shapes, limits, stop and reset flow can be tested
without hardware. Every constructor refuses to run unless ``PI_EMBODIED_MOCK_ROBOT``
is ``1`` (the tests set it; nothing else should).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

MOCK_ENV = "PI_EMBODIED_MOCK_ROBOT"


def require_test_env() -> None:
    if os.environ.get(MOCK_ENV) != "1":
        raise RuntimeError(
            f"the Franka Polymetis mocks are for tests only (set {MOCK_ENV}=1 in a "
            "test); they never drive or stand in for a robot"
        )


class MockPolymetisRobot:
    """The ``PolymetisRobot`` surface; the pose follows the last setpoint at once."""

    def __init__(
        self,
        pose: Any = (0.45, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0),
        *,
        width: float = 0.08,
        object_width: float | None = None,
        surface_z: float | None = None,
        home_pose: Any = (0.45, 0.0, 0.40, 1.0, 0.0, 0.0, 0.0),
    ) -> None:
        require_test_env()
        self.pose = np.asarray(pose, dtype=np.float64).copy()
        self.home_pose = np.asarray(home_pose, dtype=np.float64)
        self.q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.width = float(width)
        self.object_width = object_width
        self.surface_z = surface_z
        self.controller: str | None = None
        self.setpoints: list[np.ndarray] = []
        self.joint_setpoints = 0
        self.gripper_commands: list[bool] = []
        self.starts = 0
        self.closed = False
        self.grasped = False
        self.jammed = False  # the fingers do not follow commands (measured != command)

    def get_ee_pose(self) -> np.ndarray:
        return self.pose.copy()

    def get_joint_positions(self) -> np.ndarray:
        return self.q.copy()

    def start_cartesian_impedance(self, Kx: Any, Kxd: Any) -> None:
        self.controller = "cartesian"
        self.starts += 1

    def start_joint_impedance(self, Kq: Any = None, Kqd: Any = None) -> None:
        self.controller = "joint"

    def terminate_current_policy(self) -> None:
        if self.controller is None:
            raise RuntimeError("no controller running")
        self.controller = None

    def lose_controller(self) -> None:
        """A server-side reflex terminated the controller."""
        self.controller = None

    def update_desired_ee_pose(self, pose: Any) -> None:
        if self.controller != "cartesian":
            raise RuntimeError(
                "Tried to update policy with no controller running; call "
                "start_cartesian_impedance first"
            )
        p = np.asarray(pose, dtype=np.float64).copy()
        self.setpoints.append(p.copy())
        if self.surface_z is not None:
            p[2] = max(p[2], self.surface_z)
        self.pose = p

    def move_to_joint_positions(self, positions: Any, time_to_go: float) -> None:
        self.q = np.asarray(positions, dtype=np.float64).copy()
        self.pose = self.home_pose.copy()
        self.controller = None  # a joint move preempts Cartesian impedance

    def update_desired_joint_pos(self, pos: Any) -> None:
        if self.controller != "joint":
            raise RuntimeError("no controller running (joint impedance)")
        self.joint_setpoints += 1
        self.q = np.asarray(pos, dtype=np.float64).copy()
        self.pose = self.home_pose.copy()

    def control_gripper(self, close: bool) -> None:
        self.gripper_commands.append(bool(close))
        if self.jammed:
            return
        if close:
            self.width = 0.0002 if self.object_width is None else self.object_width
            self.grasped = self.object_width is not None
        else:
            self.width = 0.08
            self.grasped = False

    def get_gripper_position(self) -> np.ndarray:
        return np.array([self.width])

    def get_gripper_state(self) -> dict[str, Any]:
        return {"width": self.width, "is_grasped": self.grasped, "is_moving": False}

    def close(self) -> None:
        self.closed = True


class MockRGBD:
    """A fixed RGB-D frame: a colour gradient over a flat surface at ``depth_m``."""

    def __init__(
        self, serial: str = "mock", width: int = 640, height: int = 480, depth_m=0.5
    ) -> None:
        require_test_env()
        self.serial = serial
        self.width, self.height = int(width), int(height)
        self.depth_scale = 0.001
        self.depth_m = float(depth_m)
        self.closed = False

    def intrinsics(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": 600.0,
            "fy": 600.0,
            "ppx": self.width / 2,
            "ppy": self.height / 2,
            "distortion_model": "none",
            "coeffs": [0.0] * 5,
        }

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        rows = np.linspace(0, 255, self.height, dtype=np.float32)[:, None]
        cols = np.linspace(0, 255, self.width, dtype=np.float32)[None, :]
        rgb = np.stack(
            [
                np.broadcast_to(rows, (self.height, self.width)),
                np.broadcast_to(cols, (self.height, self.width)),
                np.full((self.height, self.width), 128.0),
            ],
            axis=-1,
        ).astype(np.uint8)
        depth = np.full((self.height, self.width), self.depth_m, dtype=np.float32)
        return rgb, depth

    def close(self) -> None:
        self.closed = True
