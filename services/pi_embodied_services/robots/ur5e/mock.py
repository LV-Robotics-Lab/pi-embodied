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
    (an RTDE error); ``protective_stop_after`` polls halts the motion and reports a
    protective stop in ``status``; ``accept_moves`` False makes moveL/moveJ return
    False (the controller rejected the command); ``home_pose`` is the TCP pose the
    arm reports once a joint move arrives."""

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
        protective_stop_after: int | None = None,
        accept_moves: bool = True,
        home_pose: Any = (0.45, 0.0, 0.40, *DOWN),
        serial: str | None = "2023300001",
    ) -> None:
        require_test_env()
        self.pose = np.asarray(pose, dtype=np.float64).copy()
        self.q = np.asarray(joints, dtype=np.float64).copy()
        self.step_m, self.step_rad, self.step_joint = step_m, step_rad, step_joint_rad
        self.wall_x, self.fail_after = wall_x, fail_after
        self.protective_stop_after = protective_stop_after
        self.protective_stopped = False
        self.accept_moves = accept_moves
        self.home_pose = np.asarray(home_pose, dtype=np.float64).copy()
        self.serial = serial
        self.goal: np.ndarray | None = None
        self.joint_goal: np.ndarray | None = None
        self.moves: list[np.ndarray] = []
        self.joint_moves: list[np.ndarray] = []
        self.stops: list[str] = []
        self.speeds = np.zeros(6)
        self.polls = 0
        self.closed = False

    def tcp_pose(self) -> np.ndarray:
        return self.pose.copy()

    def joints(self) -> np.ndarray:
        return self.q.copy()

    def joint_speeds(self) -> np.ndarray:
        return self.speeds.copy()

    def status(self) -> dict[str, Any]:
        return {
            "robot_mode": 7,
            "safety_mode": 3 if self.protective_stopped else 1,
            "protective_stopped": self.protective_stopped,
            "emergency_stopped": False,
            "program_running": True,
        }

    def identity(self) -> str | None:
        return self.serial

    def move_l(self, pose: Any, speed: float, accel: float) -> bool:
        if not self.accept_moves:
            return False
        self.goal = np.asarray(pose, dtype=np.float64).copy()
        self.moves.append(self.goal.copy())
        self.speed, self.accel = speed, accel
        return True

    def move_j(self, q: Any, speed: float, accel: float) -> bool:
        if not self.accept_moves:
            return False
        self.joint_goal = np.asarray(q, dtype=np.float64).copy()
        self.joint_moves.append(self.joint_goal.copy())
        return True

    def busy(self) -> bool:
        self.polls += 1
        if self.fail_after is not None and self.polls > self.fail_after:
            raise RuntimeError("RTDE: robot is protective stopped")
        if (
            self.protective_stop_after is not None
            and self.polls > self.protective_stop_after
        ):
            self.protective_stopped = True
            self.goal = self.joint_goal = None
            return False
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
                self.pose = self.home_pose.copy()
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
    ``object_pos`` when closing onto an object; ``jammed`` fingers never move.

    The status registers behave like the real ones: for ``stale_polls`` status reads
    after a command ``OBJ``/``POS`` still show the previous motion (the register lag
    the controller must not mistake for "settled"), then ``OBJ`` reads 0 (moving) for
    ``moving_polls`` reads before the fingers arrive. ``PRE`` echoes the request at once.
    """

    def __init__(
        self,
        position: int = 0,
        *,
        object_pos: int | None = None,
        jammed: bool = False,
        active: bool = True,
        stale_polls: int = 0,
        moving_polls: int = 0,
    ) -> None:
        require_test_env()
        self.pos = int(position)
        self.pre = int(position)
        self.object_pos = object_pos
        self.jammed = jammed
        self.active = active
        self.obj = 3
        self.stale_polls, self.moving_polls = int(stale_polls), int(moving_polls)
        self._pending: tuple[int, int] | None = None  # (final pos, final obj)
        self._stale_left = self._moving_left = 0
        self.commands: list[int] = []
        self.activations = 0
        self.stopped = 0
        self.closed = False

    def activated(self) -> bool:
        return self.active

    def activate(self, timeout_s: float = 10.0) -> None:
        self.active = True
        self.activations += 1

    def _advance(self) -> None:
        """One status read: the registers lag, then show motion, then the result."""
        if self._pending is None:
            return
        if self._stale_left > 0:
            self._stale_left -= 1
            return
        if self._moving_left > 0:
            self._moving_left -= 1
            self.obj = 0
            return
        self.pos, self.obj = self._pending
        self._pending = None

    def position(self) -> int:
        self._advance()
        return self.pos

    def requested_position(self) -> int:
        return self.pre

    def object_status(self) -> int:
        self._advance()
        return self.obj

    def fault(self) -> int:
        return 0

    def go_to(self, position: int, speed: int, force: int) -> None:
        self.commands.append(int(position))
        self.pre = int(position)
        if self.jammed:
            self._pending = None
            return
        if (
            position > self.pos
            and self.object_pos is not None
            and position > self.object_pos
        ):
            final = (self.object_pos, 2)
        else:
            final = (int(position), 3)
        self._pending = final
        self._stale_left, self._moving_left = self.stale_polls, self.moving_polls

    def stop(self) -> None:
        """``GTO 0``: the fingers stay where the status shows them."""
        self.stopped += 1
        self._pending = None

    def close(self) -> None:
        self.closed = True
