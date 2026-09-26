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

"""Test doubles for the ur_rtde arm and the Robotiq gripper. Not a simulator.

They exist so the env server's RPC shapes, limits, stop and reset flow can be tested
without hardware: the arm advances toward its commanded pose one ``step_m`` per
``busy`` poll, a wall stops it, and the gripper closes onto an optional object or
ignores commands (``jammed``). Every constructor refuses to run unless
``PI_EMBODIED_MOCK_ROBOT`` is ``1`` (the tests set it; nothing else should).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pi_embodied_services.components.cameras.mock import (
    MOCK_ENV,
    MockCamera,
    require_test_env,
)

__all__ = ["MOCK_ENV", "MockCamera", "MockRobotiq", "MockUrArm", "require_test_env"]

DOWN = (np.pi, 0.0, 0.0)  # tool z pointing at base -z


class MockUrArm:
    """The ``RtdeArm`` surface. ``step_m`` / ``step_rad`` per ``busy`` poll model the
    motion; ``wall_x`` stops the measured pose at that x; ``fail_after`` polls raises
    (a protective stop)."""

    def __init__(
        self,
        pose: Any = (0.45, 0.0, 0.30, *DOWN),
        *,
        joints: Any = (1.571, -1.571, 1.571, -1.571, -1.571, 0.0),
        step_m: float = 0.01,
        step_rad: float = 0.05,
        step_joint_rad: float = 0.2,
        wall_x: float | None = None,
        fail_after: int | None = None,
        serial: str | None = "2023300001",
    ) -> None:
        require_test_env()
        self.pose = np.asarray(pose, dtype=np.float64).copy()
        self.q = np.asarray(joints, dtype=np.float64).copy()
        self.step_m, self.step_rad, self.step_joint = step_m, step_rad, step_joint_rad
        self.wall_x, self.fail_after = wall_x, fail_after
        self.serial = serial
        self.goal: np.ndarray | None = None
        self.joint_goal: np.ndarray | None = None
        self.moves: list[np.ndarray] = []
        self.joint_moves: list[np.ndarray] = []
        self.stops: list[str] = []
        self.polls = 0
        self.closed = False

    def tcp_pose(self) -> np.ndarray:
        return self.pose.copy()

    def joints(self) -> np.ndarray:
        return self.q.copy()

    def joint_speeds(self) -> np.ndarray:
        return np.zeros(6)

    def status(self) -> dict[str, Any]:
        return {"robot_mode": 7, "safety_mode": 1, "protective_stopped": False}

    def identity(self) -> str | None:
        return self.serial

    def move_l(self, pose: Any, speed: float, accel: float) -> None:
        self.goal = np.asarray(pose, dtype=np.float64).copy()
        self.moves.append(self.goal.copy())
        self.speed, self.accel = speed, accel

    def move_j(self, q: Any, speed: float, accel: float) -> None:
        self.joint_goal = np.asarray(q, dtype=np.float64).copy()
        self.joint_moves.append(self.joint_goal.copy())

    def busy(self) -> bool:
        self.polls += 1
        if self.fail_after is not None and self.polls > self.fail_after:
            raise RuntimeError("RTDE: robot is protective stopped")
        if self.goal is not None:
            d = self.goal[:3] - self.pose[:3]
            n = float(np.linalg.norm(d))
            self.pose[:3] += d if n <= self.step_m else d / n * self.step_m
            r = self.goal[3:] - self.pose[3:]
            rn = float(np.linalg.norm(r))
            self.pose[3:] += r if rn <= self.step_rad else r / rn * self.step_rad
            if self.wall_x is not None and self.pose[0] > self.wall_x:
                self.pose[0] = self.wall_x
            arrived = np.allclose(self.pose, self.goal, atol=1e-9) or (
                self.wall_x is not None
                and self.goal[0] > self.wall_x
                and self.pose[0] >= self.wall_x - 1e-9
                and np.allclose(self.pose[1:], self.goal[1:], atol=1e-9)
            )
            if arrived:
                self.goal = None
            return not arrived
        if self.joint_goal is not None:
            d = self.joint_goal - self.q
            n = float(np.max(np.abs(d)))
            self.q += d if n <= self.step_joint else d / n * self.step_joint
            if np.allclose(self.q, self.joint_goal, atol=1e-9):
                self.joint_goal = None
                self.pose = np.array([0.45, 0.0, 0.40, *DOWN])
                return False
            return True
        return False

    def stop_l(self, decel: float) -> None:
        self.stops.append("stopL")
        self.goal = None

    def stop_j(self, decel: float) -> None:
        self.stops.append("stopJ")
        self.joint_goal = None

    def close(self) -> None:
        self.closed = True


class MockRobotiq:
    """The ``RobotiqGripper`` surface: 0 = open .. 255 = closed; the fingers stop at
    ``object_pos`` when closing onto an object; ``jammed`` fingers never move."""

    def __init__(
        self,
        position: int = 0,
        *,
        object_pos: int | None = None,
        jammed: bool = False,
        active: bool = True,
    ) -> None:
        require_test_env()
        self.pos = int(position)
        self.object_pos = object_pos
        self.jammed = jammed
        self.active = active
        self.obj = 3
        self.commands: list[int] = []
        self.activations = 0
        self.closed = False

    def activated(self) -> bool:
        return self.active

    def activate(self, timeout_s: float = 10.0) -> None:
        self.active = True
        self.activations += 1

    def position(self) -> int:
        return self.pos

    def object_status(self) -> int:
        return self.obj

    def fault(self) -> int:
        return 0

    def go_to(self, position: int, speed: int, force: int) -> None:
        self.commands.append(int(position))
        if self.jammed:
            self.obj = 3
            return
        if (
            position > self.pos
            and self.object_pos is not None
            and position > self.object_pos
        ):
            self.pos, self.obj = self.object_pos, 2
        else:
            self.pos, self.obj = int(position), 3

    def close(self) -> None:
        self.closed = True
