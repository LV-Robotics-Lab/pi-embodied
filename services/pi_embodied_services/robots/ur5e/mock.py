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
``async_status`` poll, a wall stops it, and the gripper closes onto an optional
object or ignores commands (``jammed``). Every constructor refuses to run unless
``PI_EMBODIED_MOCK_ROBOT`` is ``1`` (the tests set it; nothing else should).
"""

from __future__ import annotations

from collections.abc import Callable
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
    """The ``RtdeArm`` surface. ``step_m`` / ``step_rad`` per ``async_status`` poll
    model the motion; ``wall_x`` stops the measured pose at that x; ``fail_after``
    polls raises (an RTDE error); ``accept_moves`` False makes moveL/moveJ return False
    (the controller rejected the command); ``home_pose`` is the TCP pose the arm
    reports once a joint move arrives (and what ``forward_kinematics`` returns for any
    other joint vector), unless ``fk`` maps joints to a TCP pose.

    The async status register behaves like ur_rtde 1.6.5's
    (``getAsyncOperationProgressEx``: an operation id bumped when the control script's
    move thread starts, and a running bit): after moveL/moveJ returns, the register
    still shows the previous operation for ``stale_polls`` reads (the script has not
    run the new thread yet), while the arm already moves. ``protective_stop_after``
    polls freezes the arm in a protective stop with the register still reading
    "running" (the script is paused), and ``script_stop_after`` polls stops the control
    script the same way (an unreachable target: the controller's IK fails and the
    program halts); ``ensure_control`` re-uploads a stopped script unless
    ``reupload_fails``.
    """

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
        script_stop_after: int | None = None,
        stale_polls: int = 0,
        accept_moves: bool = True,
        reupload_fails: bool = False,
        home_pose: Any = (0.45, 0.0, 0.40, *DOWN),
        fk: Callable[[np.ndarray], Any] | None = None,
        joints_within_limits: bool = True,
        serial: str | None = "2023300001",
    ) -> None:
        require_test_env()
        self.pose = np.asarray(pose, dtype=np.float64).copy()
        self.q = np.asarray(joints, dtype=np.float64).copy()
        self.step_m, self.step_rad, self.step_joint = step_m, step_rad, step_joint_rad
        self.wall_x, self.fail_after = wall_x, fail_after
        self.protective_stop_after = protective_stop_after
        self.script_stop_after = script_stop_after
        self.protective_stopped = False
        self.program_running = True
        self.accept_moves = accept_moves
        self.reupload_fails = reupload_fails
        self.home_pose = np.asarray(home_pose, dtype=np.float64).copy()
        self.fk = fk
        self.joints_within_limits = joints_within_limits
        self.serial = serial
        self.goal: np.ndarray | None = None
        self.joint_goal: np.ndarray | None = None
        self.moves: list[np.ndarray] = []
        self.joint_moves: list[np.ndarray] = []
        self.fk_queries: list[np.ndarray] = []
        self.stops: list[str] = []
        self.speeds = np.zeros(6)
        self.polls = 0
        self.reuploads = 0
        self.closed = False
        self.stale_polls = int(stale_polls)
        self.op_id = 0
        self.running = False
        self._stale_left = 0
        self._stale_view = (0, False)

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
            "program_running": self.program_running,
        }

    def identity(self) -> str | None:
        return self.serial

    def ensure_control(self) -> str | None:
        if self.program_running:
            return None
        if self.reupload_fails:
            raise RuntimeError("reuploadScript failed: remote control is off")
        self.reuploads += 1
        self.program_running = True
        self.goal = self.joint_goal = None
        self.running = False
        return "the RTDE control script had stopped and was re-uploaded"

    def forward_kinematics(self, q: Any) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        self.fk_queries.append(q.copy())
        if self.fk is not None:
            return np.asarray(self.fk(q), dtype=np.float64)
        if np.allclose(q, self.q, atol=1e-9):
            return self.pose.copy()
        return self.home_pose.copy()

    def joints_within_safety_limits(self, q: Any) -> bool:
        return bool(self.joints_within_limits)

    def _start(self) -> None:
        """The script's move thread starts: a new operation id, running; the
        register keeps showing the previous state for ``stale_polls`` reads."""
        self._stale_view = (self.op_id, self.running)
        self._stale_left = self.stale_polls
        self.op_id = (self.op_id + 1) % 128
        self.running = True

    def move_l(self, pose: Any, speed: float, accel: float) -> bool:
        if not self.accept_moves or not self.program_running:
            return False
        self.goal = np.asarray(pose, dtype=np.float64).copy()
        self.joint_goal = None
        self.moves.append(self.goal.copy())
        self.speed, self.accel = speed, accel
        self._start()
        return True

    def move_j(self, q: Any, speed: float, accel: float) -> bool:
        if not self.accept_moves or not self.program_running:
            return False
        self.joint_goal = np.asarray(q, dtype=np.float64).copy()
        self.goal = None
        self.joint_moves.append(self.joint_goal.copy())
        self._start()
        return True

    def _advance(self) -> None:
        """One control period of motion."""
        if self.protective_stopped or not self.program_running:
            return
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
                self.running = False
        elif self.joint_goal is not None:
            d = self.joint_goal - self.q
            n = float(np.max(np.abs(d)))
            self.q += d if n <= self.step_joint else d / n * self.step_joint
            if self.fk is not None:
                self.pose = np.asarray(self.fk(self.q), dtype=np.float64).copy()
            if np.allclose(self.q, self.joint_goal, atol=1e-9):
                self.joint_goal = None
                self.running = False
                if self.fk is None:
                    self.pose = self.home_pose.copy()

    def async_status(self) -> tuple[int | None, bool]:
        """``(operation id, running)`` of the async register (one poll: the arm
        advances one step)."""
        self.polls += 1
        if self.fail_after is not None and self.polls > self.fail_after:
            raise RuntimeError("RTDE: robot is protective stopped")
        if (
            self.protective_stop_after is not None
            and self.polls > self.protective_stop_after
        ):
            self.protective_stopped = True
        if self.script_stop_after is not None and self.polls > self.script_stop_after:
            self.program_running = False
        self._advance()
        if self._stale_left > 0:
            self._stale_left -= 1
            return self._stale_view
        return self.op_id, self.running

    def busy(self) -> bool:
        """The pre-fix poll (``getAsyncOperationProgress() >= 0``), kept so tests
        can show what it concluded from a stale register."""
        return self.async_status()[1]

    def _halt(self, name: str) -> None:
        self.stops.append(name)
        if self.program_running and not self.protective_stopped:
            self.goal = self.joint_goal = None
            self.running = False

    def stop_l(self, decel: float) -> None:
        self._halt("stopL")

    def stop_j(self, decel: float) -> None:
        self._halt("stopJ")

    def close(self) -> None:
        self.closed = True


class MockRobotiq:
    """The ``RobotiqGripper`` surface: 0 = open .. 255 = closed; the fingers stop at
    ``object_pos`` when closing onto an object; ``jammed`` fingers never move; a close
    on nothing stops at ``closed_pos`` (a real 2F-85 reads ~227..230, not 255).

    The status registers behave like the real ones: for ``stale_polls`` status reads
    after a command ``OBJ``/``POS`` still show the previous motion (the register lag
    the controller must not mistake for "settled"), then for ``moving_polls`` reads
    ``POS`` walks toward the result while ``OBJ`` reads 0 (moving), or keeps its
    previous value when ``obj_lag`` (OBJ updating later than POS). ``PRE`` echoes the
    request at once.
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
        obj_lag: bool = False,
        closed_pos: int = 255,
    ) -> None:
        require_test_env()
        self.pos = int(position)
        self.pre = int(position)
        self.object_pos = object_pos
        self.jammed = jammed
        self.active = active
        self.obj = 3
        self.stale_polls, self.moving_polls = int(stale_polls), int(moving_polls)
        self.obj_lag = bool(obj_lag)
        self.closed_pos = int(closed_pos)
        self._pending: tuple[int, int] | None = None  # (final pos, final obj)
        self._from = self.pos
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
            done = self.moving_polls - self._moving_left + 1
            self._moving_left -= 1
            final = self._pending[0]
            frac = done / (self.moving_polls + 1)
            self.pos = int(round(self._from + (final - self._from) * frac))
            if not self.obj_lag:
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
            final = (min(int(position), self.closed_pos), 3)
        self._pending = final
        self._from = self.pos
        self._stale_left, self._moving_left = self.stale_polls, self.moving_polls

    def stop(self) -> None:
        """``GTO 0``: the fingers stay where the status shows them."""
        self.stopped += 1
        self._pending = None

    def close(self) -> None:
        self.closed = True
