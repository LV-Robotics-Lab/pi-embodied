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
# The driver calls follow OpenETA real/robots/ur5e.py (moveL/moveJ at speed 0.25,
# acceleration 0.5; Robotiq positions 0 = open .. 255 = closed). The safety layer is
# pi-embodied's (robots/franka_polymetis/control.py, robots/piper/controller.py):
# per-call translation/rotation limits that refuse instead of clamping, a workspace
# box and Z floor checked before anything is commanded, a tool tilt limit, a
# protective-stop check before and after every motion (moveL/moveJ return False
# when the controller rejects them; OpenETA ignored that), ``stop`` polled while a
# motion runs (stopL / stopJ bring the arm to rest, GTO 0 stops the gripper), a
# setpoint that is cleared after a stop or a failure, a reset that lifts clear
# before the joint move, and a gripper that reports jammed fingers and empty grasps.
# An async moveL/moveJ is awaited the way ur_rtde's examples do it (wait for the
# operation to start, then for it to end), a protective stop or a halted control
# script ends the wait at once, and the reset's joint move is checked with the
# controller's forward kinematics before it is commanded.
# OpenETA's move_to_pose passed ``rpy`` straight into moveL as a rotation vector when
# no ``rotvec`` was given (ur5e.py:177); ``pose_rotvec`` converts.

"""UR5e motion primitives and safety limits on top of the ur_rtde handles.

Poses are the UR convention ``[x, y, z, rx, ry, rz]`` (m, axis-angle rotation
vector) in the robot base frame; results also carry the pi-embodied 7-vector
``[x, y, z, qx, qy, qz, qw]``. The controller keeps a **setpoint**, the pose the
last successful motion reached, so consecutive small deltas accumulate without
integrating measurement noise; a stop, a timeout, a driver error or a target the
arm did not reach clears it, and the next command starts from the measured pose.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

#: Deceleration passed to stopL / stopJ (ur_rtde's defaults).
STOP_DECEL_L = 10.0
STOP_DECEL_J = 2.0
#: RTDE ``safety_mode`` values in which the arm may move (1 NORMAL, 2 REDUCED); the
#: others are stops (3 PROTECTIVE_STOP, 5 SAFEGUARD_STOP, 6/7 emergency stops, ...).
MOVABLE_SAFETY_MODES = (1, 2)
SAFETY_MODE_NAMES = {
    1: "normal",
    2: "reduced",
    3: "protective stop",
    4: "recovery",
    5: "safeguard stop",
    6: "system emergency stop",
    7: "robot emergency stop",
    8: "violation",
    9: "fault",
    10: "validate joint id",
    11: "undefined",
    12: "automatic mode safeguard stop",
    13: "three-position enabling stop",
}
#: RTDE ``robot_mode`` in which motion is possible (7 RUNNING: powered, brakes off).
ROBOT_MODE_RUNNING = 7
ROBOT_MODE_NAMES = {
    -1: "no controller",
    0: "disconnected",
    1: "confirm safety",
    2: "booting",
    3: "power off",
    4: "power on",
    5: "idle (brakes engaged)",
    6: "backdrive",
    7: "running",
    8: "updating firmware",
}
#: A Robotiq 2F-85 closed on nothing stops at POS ~227..230 of 255, which the linear
#: width map reads as 8.5..9.3 mm; 13 % of the stroke (11 mm on a 2F-85, 18 mm on a
#: 2F-140) sits above that with margin. The default ``gripper.empty_width_m`` is this
#: fraction of ``gripper.stroke_m``. A thin object still grasps: the gripper reports
#: contact (OBJ 2), and an empty grasp also requires that no contact was reported.
EMPTY_WIDTH_FRACTION = 0.13


def async_phase(
    before: tuple[int | None, bool], now: tuple[int | None, bool], seen_running: bool
) -> str:
    """Where an async moveL/moveJ issued after ``before`` is, from the async status
    register ``now`` (``(operation id, running)``, :meth:`RtdeArm.async_status`).

    ur_rtde 1.6.5: moveL/moveJ(async=True) returns once the control script has
    *spawned* its move thread; the thread writes "started" (a new operation id with
    the running bit) only when it first runs, so right after the call the register
    still shows the previous operation, which reads as "finished". The operation id
    is what tells them apart: ``pending`` while it is the old one, then ``running``
    or ``done``. Without an id (ur_rtde older than getAsyncOperationProgressEx: only
    ``progress >= 0``) the operation counts as done only after it was seen running.
    """
    id0, _ = before
    id1, run1 = now
    if id0 is not None and id1 is not None:
        if id1 == id0:
            return "pending"
        return "running" if run1 else "done"
    if run1:
        return "running"
    return "done" if seen_running else "pending"


def pose_rotvec(rotvec: Any = None, rpy: Any = None) -> np.ndarray:
    """The rotation vector of a pose given as ``rotvec`` or as extrinsic xyz ``rpy``
    (radians). Exactly one may be given; ``rpy`` is converted, never passed through."""
    if (rotvec is None) == (rpy is None):
        raise ValueError("give exactly one of rotvec or rpy")
    if rotvec is not None:
        v = np.asarray(rotvec, dtype=np.float64).reshape(-1)
        if v.shape != (3,) or not np.all(np.isfinite(v)):
            raise ValueError("rotvec must be 3 finite numbers")
        return v
    e = np.asarray(rpy, dtype=np.float64).reshape(-1)
    if e.shape != (3,) or not np.all(np.isfinite(e)):
        raise ValueError("rpy must be 3 finite numbers")
    return Rotation.from_euler("xyz", e).as_rotvec()


def pose7_of(pose6: np.ndarray) -> list[float]:
    """``[x, y, z, qx, qy, qz, qw]`` of a UR ``[x, y, z, rx, ry, rz]`` pose."""
    q = Rotation.from_rotvec(np.asarray(pose6[3:], dtype=np.float64)).as_quat()
    return [float(v) for v in (*pose6[:3], *q)]


def tool_tilt(rotvec: np.ndarray) -> float:
    """Angle (rad) between the tool z-axis and straight down (base -z)."""
    z = Rotation.from_rotvec(rotvec).apply([0.0, 0.0, 1.0])
    return float(math.acos(max(-1.0, min(1.0, -z[2]))))


def rotation_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (rad) between two rotation vectors."""
    return float((Rotation.from_rotvec(a) * Rotation.from_rotvec(b).inv()).magnitude())


@dataclass
class UR5eLimits:
    """Safety limits and motion settings (config/example.yaml documents each)."""

    z_floor_m: float | None = None
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None
    max_move_m: float = 0.08
    max_rotate_rad: float = 0.2
    max_tilt_rad: float | None = 0.5
    speed_mps: float = 0.25
    accel_mps2: float = 0.5
    joint_speed_radps: float = 0.25
    joint_accel_radps2: float = 0.5
    poll_s: float = 0.02
    start_timeout_s: float = 1.0
    move_timeout_s: float = 15.0
    reset_timeout_s: float = 30.0
    move_tolerance_m: float = 0.005
    rotate_tolerance_rad: float = 0.02
    joint_tolerance_rad: float = 0.02
    divergence_resync_m: float = 0.01
    divergence_resync_rad: float = 0.05
    reset_lift_m: float = 0.05
    reset_path_samples: int = 10
    gripper_stroke_m: float = 0.085
    gripper_speed: int = 255
    gripper_force: int = 100
    gripper_timeout_s: float = 5.0
    gripper_ack_timeout_s: float = 0.5
    gripper_poll_s: float = 0.05
    gripper_settle_polls: int = 2
    gripper_motion_eps_m: float = 0.003
    close_threshold_m: float = 0.075
    empty_width_m: float | None = round(EMPTY_WIDTH_FRACTION * 0.085, 4)
    begin_joints: tuple[float, ...] | None = None

    def validate(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            try:
                arr = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                raise ValueError(f"{f.name} must be numeric, got {value!r}") from None
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{f.name} must be finite, got {value!r}")
        if self.z_floor_m is None:
            raise ValueError(
                "limits.z_floor_m is not set: rest the closed gripper on the table, read "
                "the TCP z (python -m pi_embodied_services.robots.ur5e.env_server "
                "--robot-config <yaml> --read-pose) and write it to the config"
            )
        if self.workspace_min is None or self.workspace_max is None:
            raise ValueError(
                "limits.workspace_min and limits.workspace_max are required"
            )
        lo, hi = np.asarray(self.workspace_min), np.asarray(self.workspace_max)
        if lo.shape != (3,) or hi.shape != (3,) or not np.all(lo < hi):
            raise ValueError("workspace_min/max must be [x, y, z] with min < max")
        if self.z_floor_m < lo[2] - 1e-9 or self.z_floor_m >= hi[2]:
            raise ValueError("z_floor_m must lie inside the workspace z range")

        def bound(name: str, lo: float, hi: float) -> None:
            value = getattr(self, name)
            if not lo <= value <= hi:
                raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")

        bound("max_move_m", 1e-3, 0.2)
        bound("max_rotate_rad", 1e-2, 0.5)
        bound("speed_mps", 1e-3, 0.5)
        bound("accel_mps2", 1e-2, 2.0)
        bound("joint_speed_radps", 1e-3, 1.0)
        bound("joint_accel_radps2", 1e-2, 2.0)
        bound("poll_s", 0.0, 0.5)
        bound("start_timeout_s", 0.05, 10.0)
        bound("reset_path_samples", 1, 100)
        bound("gripper_settle_polls", 1, 20)
        bound("divergence_resync_rad", 1e-3, math.pi)
        bound("reset_lift_m", 0.0, 0.2)
        bound("gripper_stroke_m", 0.01, 0.3)
        bound("gripper_ack_timeout_s", 0.0, 5.0)
        if self.empty_width_m is not None and not (
            0.0 <= self.empty_width_m < self.close_threshold_m
        ):
            raise ValueError(
                "empty_width_m must be in [0, close_threshold_m) (null disables the "
                f"empty-grasp reopen), got {self.empty_width_m}"
            )
        if self.max_tilt_rad is not None:
            bound("max_tilt_rad", 0.0, math.pi)
        if self.begin_joints is not None and np.asarray(self.begin_joints).shape != (
            6,
        ):
            raise ValueError("begin_joints must be 6 joint angles (rad)")


class UR5eController:
    """The primitives on one arm and its gripper (``gripper`` may be None).

    ``stop_requested`` is polled while a motion runs (the facade's stop flag); the
    running moveL/moveJ is then stopped with stopL/stopJ and the result carries
    ``cancelled: true``. ``commands`` counts commands sent to the hardware, so a
    refusal (nothing sent) can be told from a fault.
    """

    def __init__(
        self,
        arm: Any,
        gripper: Any,
        limits: UR5eLimits,
        stop_requested: Callable[[], bool] = lambda: False,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        limits.validate()
        self.arm = arm
        self.gripper = gripper
        self.limits = limits
        self._stop = stop_requested
        self._sleep = sleep
        self._clock = clock
        self.target: np.ndarray | None = None
        self.commanded_open: bool | None = None
        self.commands = 0
        self._control_note: str | None = None

    # -- state -------------------------------------------------------------

    def measured(self) -> np.ndarray:
        pose = np.asarray(self.arm.tcp_pose(), dtype=np.float64)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise RuntimeError(f"the arm reported an invalid TCP pose {pose.tolist()}")
        return pose

    def setpoint(self) -> np.ndarray:
        """The pose the next command starts from: the setpoint while the measured
        pose is within ``divergence_resync_m`` / ``divergence_resync_rad`` of it,
        else the measured pose (the setpoint is re-anchored there)."""
        pose = self.measured()
        if (
            self.target is None
            or np.linalg.norm(self.target[:3] - pose[:3])
            > self.limits.divergence_resync_m
            or rotation_gap(self.target[3:], pose[3:])
            > self.limits.divergence_resync_rad
        ):
            self.target = pose.copy()
        return self.target.copy()

    def width(self) -> float | None:
        """Gripper opening in metres (stroke * (1 - pos/255)), None without a gripper."""
        if self.gripper is None:
            return None
        pos = float(self.gripper.position())
        return float(self.limits.gripper_stroke_m * (1.0 - pos / 255.0))

    def gripper_state(self) -> dict[str, Any]:
        if self.gripper is None:
            return {"present": False}
        width = self.width()
        obj = int(self.gripper.object_status())
        return {
            "present": True,
            "width_m": width,
            "position": int(self.gripper.position()),
            "object_status": obj,
            "grasped": obj in (1, 2),
            "closed": width is not None and width < self.limits.close_threshold_m,
        }

    def state(self) -> dict[str, Any]:
        pose = self.measured()
        grip = self.gripper_state()
        out: dict[str, Any] = {
            "tcp_pose": pose7_of(pose),
            "tcp_pose_rotvec": [float(v) for v in pose],
            "tool_tilt_rad": tool_tilt(pose[3:]),
            "joints": [float(v) for v in self.arm.joints()],
            "joint_speeds": [float(v) for v in self.arm.joint_speeds()],
            "setpoint_pose": None if self.target is None else pose7_of(self.target),
            "gripper_position": [grip["width_m"]] if grip["present"] else [],
            "gripper_open": (not grip["closed"]) if grip["present"] else None,
            "gripper_grasped": grip["grasped"] if grip["present"] else None,
            "gripper_commanded_open": self.commanded_open,
            "gripper": grip,
            "z_floor_m": self.limits.z_floor_m,
        }
        status = getattr(self.arm, "status", None)
        if status is not None:
            out["robot_status"] = status()
        return out

    # -- limits ------------------------------------------------------------

    def _violation(self, p: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(p)):
            return np.full(3, np.inf)
        lo = np.asarray(self.limits.workspace_min, dtype=np.float64).copy()
        hi = np.asarray(self.limits.workspace_max, dtype=np.float64)
        lo[2] = max(lo[2], float(self.limits.z_floor_m))
        return np.maximum(0.0, lo - p) + np.maximum(0.0, p - hi)

    def check_target(self, start: np.ndarray, target: np.ndarray, what: str) -> None:
        """Refuse a target outside the box / below the floor. From outside (hand-guided,
        pushed) a target is allowed only if no axis gets further out and the total
        distance outside shrinks."""
        out, was = self._violation(target[:3]), self._violation(start[:3])
        refused = not np.all(np.isfinite(out)) or bool(np.any(out > was + 1e-9))
        if out.sum() > 1e-6 and out.sum() >= was.sum() - 1e-6:
            refused = True
        if refused:
            lo, hi = self.limits.workspace_min, self.limits.workspace_max
            raise ValueError(
                f"{what} ends at {np.round(target[:3], 4).tolist()}, outside the workspace "
                f"(x {lo[0]}..{hi[0]}, y {lo[1]}..{hi[1]}, z {self.limits.z_floor_m}"
                f"..{hi[2]} m); nothing was commanded"
            )
        tilt_cap = self.limits.max_tilt_rad
        if tilt_cap is not None:
            tilt = tool_tilt(target[3:])
            if tilt > tilt_cap + 1e-9 and tilt > tool_tilt(start[3:]) + 1e-9:
                raise ValueError(
                    f"{what} tilts the tool {tilt:.3f} rad from straight down; the limit is "
                    f"{tilt_cap} rad; nothing was commanded"
                )

    @staticmethod
    def _vec3(value: Any, name: str) -> np.ndarray:
        v = np.asarray(value, dtype=np.float64).reshape(-1)
        if v.shape != (3,) or not np.all(np.isfinite(v)):
            raise ValueError(f"{name} must be 3 finite numbers")
        return v

    # -- motion ------------------------------------------------------------

    @staticmethod
    def _safety_reason(status: dict[str, Any]) -> str | None:
        for key in ("emergency_stopped", "protective_stopped"):
            if status.get(key):
                return (
                    f"{key.replace('_', ' ')} (safety mode {status.get('safety_mode')})"
                )
        mode = status.get("safety_mode")
        if mode is not None and int(mode) not in MOVABLE_SAFETY_MODES:
            name = SAFETY_MODE_NAMES.get(int(mode), "unknown")
            return f"in safety mode {int(mode)} ({name})"
        robot_mode = status.get("robot_mode")
        if robot_mode is not None and int(robot_mode) != ROBOT_MODE_RUNNING:
            name = ROBOT_MODE_NAMES.get(int(robot_mode), "unknown")
            return f"in robot mode {int(robot_mode)} ({name}), not running"
        return None

    def safety_stopped(self) -> str | None:
        """Why the robot cannot move right now (``protective_stopped``,
        ``emergency_stopped``, a safety mode other than normal/reduced, or a robot
        mode other than running), or None."""
        return self._safety_reason(self.arm.status())

    def _require_movable(self, what: str) -> None:
        """Refuse ``what`` before anything is commanded while the robot is stopped by
        its safety system; the setpoint is cleared (the arm may have been moved).
        A control script that stopped (an unreachable target, a program stopped on
        the pendant) is re-uploaded here, so no server restart is needed."""
        reason = self.safety_stopped()
        if reason is not None:
            self.target = None
            raise RuntimeError(
                f"{what} refused: the robot is {reason}. Clear the stop on the teach "
                "pendant (unlock protective stop / release the e-stop) and re-enable "
                "remote control; nothing was commanded"
            )
        ensure = getattr(self.arm, "ensure_control", None)
        if ensure is None:
            return
        try:
            note = ensure()
        except Exception as exc:
            self.target = None
            raise RuntimeError(
                f"{what} refused: the RTDE control script is not running and could not "
                f"be re-uploaded ({exc}). Check that the pendant is in Remote Control "
                "and no other program runs; nothing was commanded"
            ) from exc
        if note:
            self.target = None
            self._control_note = str(note)

    def _rejected(self, what: str) -> RuntimeError:
        """The controller returned False from moveL/moveJ: the command never ran."""
        self.target = None
        reason = self.safety_stopped()
        return RuntimeError(
            f"{what} was rejected by the controller"
            + (f": the robot is {reason}" if reason else "")
            + " (the target may be unreachable, remote control may be off, or the RTDE "
            "control script is not running); the setpoint was cleared"
        )

    def _halted(self) -> dict[str, Any] | None:
        """What ended a running motion from the robot side, or None: a safety stop
        (protective, safeguard, emergency; the arm is held by the safety system) or
        the control script no longer running (the controller rejected the target,
        e.g. no inverse kinematics solution, or the program was stopped)."""
        status = self.arm.status()
        reason = self._safety_reason(status)
        if reason is not None:
            return {
                "interrupted": "safety_stop",
                "reason": reason,
                "protective_stopped": bool(status.get("protective_stopped")),
            }
        if status.get("program_running") is False:
            return {
                "interrupted": "control_script_stopped",
                "reason": "the RTDE control script stopped",
                "protective_stopped": False,
            }
        return None

    def _await_motion(
        self,
        before: tuple[int | None, bool],
        stop: Callable[[], None],
        timeout_s: float,
    ) -> dict[str, Any]:
        """Wait for the async motion issued after the register read ``before``:
        first for it to start (``start_timeout_s``), then for it to end. ``stop``
        (stopL/stopJ) runs on a stop request, a timeout, an interruption, or a
        motion that never started. Returns the outcome flags and ``polls``."""
        lim = self.limits
        t0 = self._clock()
        out: dict[str, Any] = {"polls": 0, "started": False}
        while True:
            if self._stop():
                stop()
                out["cancelled"] = True
                return out
            phase = async_phase(before, self.arm.async_status(), out["started"])
            if phase == "done":
                out["started"] = True
                return out
            if phase == "running":
                out["started"] = True
            halted = self._halted()
            if halted is not None:
                try:
                    stop()
                except Exception:
                    pass  # a stopped script or a safety stop refuses stopL/stopJ
                out.update(halted)
                return out
            elapsed = self._clock() - t0
            if not out["started"] and elapsed > lim.start_timeout_s:
                stop()
                out["not_started"] = True
                return out
            if elapsed > timeout_s:
                stop()
                out["timed_out"] = True
                return out
            out["polls"] += 1
            self._sleep(lim.poll_s)

    def _report(self, result: dict[str, Any], run: dict[str, Any], what: str) -> None:
        """Outcome flags of ``_await_motion`` into a motion result (not ok unless the
        motion ran to its end)."""
        if run.get("cancelled"):
            result["cancelled"] = True
        if run.get("not_started"):
            result["ok"] = False
            result["not_started"] = True
            result["note"] = (
                f"{what} was accepted but the controller never reported it running "
                f"within {self.limits.start_timeout_s} s; it was stopped"
            )
        if run.get("timed_out"):
            result["timed_out"] = True
        if run.get("interrupted"):
            result["ok"] = False
            result["interrupted"] = run["interrupted"]
            if run["interrupted"] == "control_script_stopped":
                result["note"] = (
                    f"{what} ended because the RTDE control script stopped: the "
                    "controller could not execute the target (unreachable / no inverse "
                    "kinematics solution) or the program was stopped on the pendant. "
                    "The next command re-uploads the script; the setpoint was cleared"
                )
        if self._control_note:
            result["control_script_reuploaded"] = True
            result["control_note"] = self._control_note
            self._control_note = None

    def _after_motion(self, result: dict[str, Any], what: str) -> None:
        """A safety stop during the motion makes the result not ok and says so."""
        reason = self.safety_stopped()
        if reason is not None:
            self.target = None
            result["ok"] = False
            result["protective_stopped"] = "protective" in reason
            result["safety_stopped"] = True
            result.pop("timed_out", None)
            result["note"] = (
                f"{what} ended with the robot {reason}: it hit something or left the "
                "safety limits. Clear the stop on the teach pendant before the next "
                "command; the setpoint was cleared"
            )

    def _run_l(self, start: np.ndarray, target: np.ndarray, timeout_s: float):
        """Start a moveL to ``target`` and wait for it; stop it on ``stop`` or timeout."""
        lim = self.limits
        self._require_movable("the move")
        before = self.arm.async_status()
        if not self.arm.move_l(target, lim.speed_mps, lim.accel_mps2):
            raise self._rejected("moveL")
        self.commands += 1
        t0 = self._clock()
        try:
            run = self._await_motion(
                before, lambda: self.arm.stop_l(STOP_DECEL_L), timeout_s
            )
        except Exception as exc:
            self.target = None
            try:
                self.arm.stop_l(STOP_DECEL_L)
            except Exception:
                pass
            raise RuntimeError(
                f"motion aborted: {exc}; the setpoint was cleared"
            ) from exc
        final = self.measured()
        pos_err = float(np.linalg.norm(target[:3] - final[:3]))
        rot_err = rotation_gap(target[3:], final[3:])
        ok = (
            not run.get("cancelled")
            and not run.get("timed_out")
            and not run.get("not_started")
            and not run.get("interrupted")
            and pos_err <= lim.move_tolerance_m
            and rot_err <= lim.rotate_tolerance_rad
        )
        # The setpoint survives only a motion that arrived; otherwise the next command
        # starts from where the arm actually is.
        self.target = target.copy() if ok else None
        result: dict[str, Any] = {
            "ok": ok,
            "start_tcp_pose": pose7_of(start),
            "target_tcp_pose": pose7_of(target),
            "final_tcp_pose": pose7_of(final),
            "final_error_m": pos_err,
            "final_error_rad": rot_err,
            "steps_used": run["polls"],
            "elapsed_s": float(self._clock() - t0),
            "states": None,
        }
        self._report(result, run, "the move")
        if run.get("timed_out"):
            result["note"] = (
                f"moveL did not finish within {timeout_s} s and was stopped"
            )
        self._after_motion(result, "the move")
        return result

    def move_delta(self, delta_xyz: Any) -> dict[str, Any]:
        """Translate the TCP by a base-frame delta; the orientation is held."""
        lim = self.limits
        d = self._vec3(delta_xyz, "delta_xyz")
        norm = float(np.linalg.norm(d))
        if norm > lim.max_move_m + 1e-9:
            raise ValueError(
                f"delta_xyz moves {norm:.4f} m; the limit is {lim.max_move_m} m per call. "
                "Split the motion into smaller calls; nothing was commanded"
            )
        start = self.setpoint()
        target = start.copy()
        target[:3] += d
        self.check_target(self.measured(), target, "the move")
        result = self._run_l(start, target, lim.move_timeout_s)
        result["requested_delta_xyz_base"] = d.tolist()
        return result

    def rotate_delta(self, delta_rpy: Any) -> dict[str, Any]:
        """Turn the TCP by extrinsic xyz Euler angles about the base axes."""
        lim = self.limits
        e = self._vec3(delta_rpy, "delta_rpy")
        norm = float(np.linalg.norm(e))
        if norm > lim.max_rotate_rad + 1e-9:
            raise ValueError(
                f"delta_rpy rotates {norm:.4f} rad; the limit is {lim.max_rotate_rad} rad "
                "per call. Split the rotation into smaller calls; nothing was commanded"
            )
        start = self.setpoint()
        target = start.copy()
        target[3:] = (
            Rotation.from_euler("xyz", e) * Rotation.from_rotvec(start[3:])
        ).as_rotvec()
        self.check_target(self.measured(), target, "the rotation")
        result = self._run_l(start, target, lim.move_timeout_s)
        result["requested_delta_rpy_base"] = e.tolist()
        return result

    def move_pose(
        self, xyz: Any, rotvec: Any = None, rpy: Any = None
    ) -> dict[str, Any]:
        """Move to an absolute base-frame pose, within the per-call limits from the
        current setpoint (``rpy`` is converted to a rotation vector)."""
        lim = self.limits
        p = self._vec3(xyz, "xyz")
        start = self.setpoint()
        rv = (
            pose_rotvec(rotvec, rpy)
            if (rotvec is not None or rpy is not None)
            else start[3:].copy()
        )
        dist = float(np.linalg.norm(p - start[:3]))
        if dist > lim.max_move_m + 1e-9:
            raise ValueError(
                f"the pose is {dist:.4f} m from the current setpoint; the limit is "
                f"{lim.max_move_m} m per call; nothing was commanded"
            )
        angle = rotation_gap(rv, start[3:])
        if angle > lim.max_rotate_rad + 1e-9:
            raise ValueError(
                f"the pose turns the tool {angle:.4f} rad; the limit is {lim.max_rotate_rad} "
                "rad per call; nothing was commanded"
            )
        target = np.concatenate([p, rv])
        self.check_target(self.measured(), target, "the pose")
        result = self._run_l(start, target, lim.move_timeout_s)
        result["requested_pose_rotvec"] = target.tolist()
        return result

    # -- gripper -----------------------------------------------------------

    def _await_gripper(self, requested: int) -> tuple[bool, bool]:
        """Poll until the fingers settled after a ``go_to(requested)``; (settled,
        cancelled). A stop sends the gripper a stop.

        ``OBJ`` lags: right after the command it still reads the previous motion's
        value (3 = at position), and it can keep that value for a while after the
        fingers started moving (``POS`` already changes). The status counts as fresh
        once the gripper echoes the request (``PRE``) and either reports motion
        (``OBJ`` 0), the position changed, or the position equals the request. The
        fingers are settled only when, on top of that, ``POS`` read the same value
        ``gripper_settle_polls`` times in a row (they stopped) and ``OBJ`` agrees
        with the command's direction (contact while opening, 1, only on an open;
        contact while closing, 2, only on a close). Fingers that show no fresh
        status within ``gripper_ack_timeout_s`` are taken as settled where they are
        (so a real jam is still reported).
        """
        lim = self.limits
        t0 = self._clock()
        pos0 = int(self.gripper.position())
        fresh = False
        last_pos: int | None = None
        same = 0
        while True:
            if self._stop():
                try:
                    self.gripper.stop()
                except Exception:
                    pass
                return False, True
            elapsed = self._clock() - t0
            if int(self.gripper.requested_position()) == int(requested):
                obj = int(self.gripper.object_status())
                pos = int(self.gripper.position())
                same = same + 1 if pos == last_pos else 1
                last_pos = pos
                if obj == 0 or pos != pos0 or pos == int(requested):
                    fresh = True
                consistent = (
                    obj == 3
                    or (obj == 1 and requested < pos0)
                    or (obj == 2 and requested > pos0)
                )
                if (
                    fresh
                    and obj != 0
                    and consistent
                    and same >= lim.gripper_settle_polls
                ):
                    return True, False
                if not fresh and elapsed > lim.gripper_ack_timeout_s:
                    return True, False
            if elapsed > lim.gripper_timeout_s:
                return False, False
            self._sleep(lim.gripper_poll_s)

    def set_gripper(self, *, open: bool) -> dict[str, Any]:
        """Open (position 0) or close (255). A close that ends at or below
        ``empty_width_m`` (default :data:`EMPTY_WIDTH_FRACTION` of the stroke: a
        2F-85 closed on nothing reads ~9 mm) with no contact reported caught nothing
        and is reopened (``grasp_empty``); fingers that did not move toward the
        command and hold no object are ``gripper_jammed``."""
        lim = self.limits
        if self.gripper is None:
            raise ValueError("this UR5e has no gripper configured (robot.gripper.type)")
        if not self.gripper.activated():
            self.gripper.activate()
            self.commands += 1
        before = self.width()
        requested = 0 if open else 255
        self.gripper.go_to(requested, lim.gripper_speed, lim.gripper_force)
        self.commands += 1
        self.commanded_open = bool(open)
        steps = 1
        settled, cancelled = self._await_gripper(requested)
        grip = self.gripper_state()
        width = grip["width_m"]
        result: dict[str, Any] = {
            "target_gripper_open": bool(open),
            "object_detected": grip["grasped"],
        }
        if (
            not cancelled
            and abs(width - before) <= lim.gripper_motion_eps_m
            and not grip["grasped"]
            and grip["closed"] == bool(open)
        ):
            result["gripper_jammed"] = True
            result["note"] = (
                f"gripper jammed: the fingers stayed at {width:.4f} m after the "
                f"{'open' if open else 'close'} command and nothing is grasped"
            )
        if (
            not open
            and not cancelled
            and lim.empty_width_m is not None
            and width <= lim.empty_width_m
            and not grip["grasped"]
        ):
            self.gripper.go_to(0, lim.gripper_speed, lim.gripper_force)
            self.commands += 1
            self.commanded_open = True
            steps += 1
            _, cancelled = self._await_gripper(0)
            width = self.width()
            result["grasp_empty"] = True
            result["note"] = (
                f"empty grasp: the gripper closed to {width:.4f} m <= {lim.empty_width_m} m "
                "(nothing between the fingers) and was reopened"
            )
        result.update(
            ok=settled
            and not cancelled
            and not result.get("grasp_empty")
            and not result.get("gripper_jammed"),
            steps_used=steps,
            gripper_width_m=width,
        )
        if not settled and not cancelled:
            result["note"] = (
                f"the gripper did not settle within {lim.gripper_timeout_s} s"
            )
        if cancelled:
            result["cancelled"] = True
        return result

    # -- reset -------------------------------------------------------------

    def _fk(self, q: np.ndarray) -> np.ndarray:
        """The TCP pose at joints ``q`` from the controller's forward kinematics."""
        pose = np.asarray(self.arm.forward_kinematics(q), dtype=np.float64).reshape(-1)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise RuntimeError(
                f"forward kinematics returned an invalid pose {pose.tolist()}"
            )
        return pose

    def _box_text(self) -> str:
        lo, hi = self.limits.workspace_min, self.limits.workspace_max
        return (
            f"x {lo[0]}..{hi[0]}, y {lo[1]}..{hi[1]}, z {self.limits.z_floor_m}"
            f"..{hi[2]} m"
        )

    def check_joint_target(self, q: np.ndarray, what: str) -> np.ndarray:
        """Refuse joints ``q`` whose TCP (controller FK with the active TCP offset)
        is outside the workspace box / below the floor, or that the controller's
        safety configuration rejects. Returns the TCP pose at ``q``."""
        within = getattr(self.arm, "joints_within_safety_limits", None)
        if within is not None and not within(q):
            raise ValueError(
                f"{what}: the joints {np.round(q, 4).tolist()} are outside the "
                "controller's safety limits (joint limits in the safety configuration); "
                "nothing was commanded"
            )
        pose = self._fk(q)
        if np.any(self._violation(pose[:3]) > 1e-6):
            raise ValueError(
                f"{what}: the joints put the TCP at {np.round(pose[:3], 4).tolist()}, "
                f"outside the workspace ({self._box_text()}); fix calibration."
                "begin_joints or the box; nothing was commanded"
            )
        return pose

    def check_joint_path(self, q0: np.ndarray, q1: np.ndarray) -> str | None:
        """Why the moveJ from ``q0`` to ``q1`` would take the TCP further outside the
        workspace than it starts (or below the floor), or None.

        moveJ interpolates all joints linearly with one time scaling, so the path is
        the straight line in joint space; ``reset_path_samples`` points on it are
        checked with the controller's forward kinematics. Between samples the TCP
        can still dip (a swing through the table between two samples); keep the
        begin pose's path well clear of the floor."""
        n = int(self.limits.reset_path_samples)
        start_out = float(self._violation(self._fk(q0)[:3]).sum())
        for i in range(1, n + 1):
            s = i / n
            q = q0 + s * (q1 - q0)
            p = self._fk(q)
            out = float(self._violation(p[:3]).sum())
            if out > start_out + 1e-6:
                return (
                    f"the joint path to the begin pose leaves the workspace at "
                    f"{s:.0%} of the way (TCP {np.round(p[:3], 4).tolist()}; "
                    f"{self._box_text()})"
                )
        return None

    def move_joints(self, q: Any, *, check_path: bool = True) -> dict[str, Any]:
        """moveJ to ``q`` (6 joint angles), stoppable with stopJ; clears the setpoint.
        The target (and, with ``check_path``, sampled points of the joint path) are
        checked against the workspace with forward kinematics before anything
        moves: a target outside raises ValueError, a path that leaves the box
        returns a result with ``ok`` False and ``path_outside_workspace``."""
        lim = self.limits
        target = np.asarray(q, dtype=np.float64).reshape(-1)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise ValueError("joints must be 6 finite angles (rad)")
        self._require_movable("the joint move")
        self.check_joint_target(target, "the joint move")
        if check_path:
            blocked = self.check_joint_path(np.asarray(self.arm.joints()), target)
            if blocked is not None:
                return {
                    "ok": False,
                    "target_joints": target.tolist(),
                    "final_joints": [float(v) for v in self.arm.joints()],
                    "path_outside_workspace": True,
                    "note": f"{blocked}; nothing was commanded",
                }
        self.target = None
        before = self.arm.async_status()
        if not self.arm.move_j(target, lim.joint_speed_radps, lim.joint_accel_radps2):
            raise self._rejected("moveJ")
        self.commands += 1
        try:
            run = self._await_motion(
                before, lambda: self.arm.stop_j(STOP_DECEL_J), lim.reset_timeout_s
            )
        except Exception as exc:
            try:
                self.arm.stop_j(STOP_DECEL_J)
            except Exception:
                pass
            raise RuntimeError(f"joint motion aborted: {exc}") from exc
        err = float(np.max(np.abs(np.asarray(self.arm.joints()) - target)))
        ok = (
            not run.get("cancelled")
            and not run.get("timed_out")
            and not run.get("not_started")
            and not run.get("interrupted")
            and err <= lim.joint_tolerance_rad
        )
        if ok:
            self.target = self.measured()
        result: dict[str, Any] = {
            "ok": ok,
            "target_joints": target.tolist(),
            "final_joints": [float(v) for v in self.arm.joints()],
            "final_error_rad": err,
            "final_tcp_pose": pose7_of(self.measured()),
        }
        self._report(result, run, "the joint move")
        if run.get("timed_out"):
            result["note"] = (
                f"moveJ did not finish within {lim.reset_timeout_s} s and was stopped"
            )
        self._after_motion(result, "the joint move")
        return result

    def reset(self) -> dict[str, Any]:
        """Open the gripper, lift ``reset_lift_m`` clear of the workspace, then moveJ
        to ``begin_joints`` (refused when unset); the same sequence as the Franka
        servers.

        Before anything moves, the begin pose's TCP (controller forward kinematics)
        must lie inside the workspace box; after the lift, the joint path to it is
        sampled the same way (``move_joints``). A held object is released where the
        arm is, before the lift, and reported as ``released_object``: carrying it
        through an unchecked joint-space swing could fling it, and the reset between
        episodes must always run (as Franka's reset). The result is not ok when the
        lift or the joint move did not arrive.
        """
        lim = self.limits
        if lim.begin_joints is None:
            raise ValueError(
                "calibration.begin_joints is not set: jog the arm to its begin pose, run "
                "the env server with --read-pose and copy joints_rad; nothing was commanded"
            )
        self._require_movable("the reset")
        begin = np.asarray(lim.begin_joints, dtype=np.float64)
        begin_pose = self.check_joint_target(begin, "the reset")
        info: dict[str, Any] = {
            "begin_joints": list(lim.begin_joints),
            "begin_tcp_pose": pose7_of(begin_pose),
        }
        out: dict[str, Any] = {
            "ok": False,
            "gripper": None,
            "lift": None,
            "move": None,
            "info": info,
        }
        grip: dict[str, Any] = {"ok": True}
        if self.gripper is not None:
            held = self.gripper_state()
            if held["grasped"]:
                info["released_object"] = True
                info["note"] = (
                    f"the gripper held an object (width {held['width_m']:.4f} m); "
                    "it was released where the arm was"
                )
            grip = self.set_gripper(open=True)
            out["gripper"] = grip
            if grip.get("cancelled"):
                out["cancelled"] = True
                return out
        start = self.setpoint()
        room = float(lim.workspace_max[2]) - float(start[2])
        lift = max(0.0, min(lim.reset_lift_m, room))
        if lift > 1e-4:
            target = start.copy()
            target[2] += lift
            lifted = self._run_l(start, target, lim.move_timeout_s)
            info["lifted_m"] = lift
            out["lift"] = lifted
            if not lifted["ok"]:
                if lifted.get("cancelled"):
                    out["cancelled"] = True
                info["note"] = lifted.get(
                    "note", "the lift before the joint move did not arrive"
                )
                return out
        move = self.move_joints(begin)
        out["move"] = move
        if move.get("cancelled"):
            out["cancelled"] = True
        if move.get("path_outside_workspace"):
            info["begin_path_outside_workspace"] = True
            info["note"] = move["note"]
        ok = bool(move["ok"]) and bool(grip.get("ok", True))
        if move["ok"]:
            final = self.measured()
            if np.any(self._violation(final[:3]) > 1e-6):
                # Forward kinematics said inside; the measured TCP disagrees (a TCP
                # offset changed on the pendant?). Report it rather than trust it.
                info["begin_pose_outside_workspace"] = True
                info["note"] = (
                    f"the begin pose puts the TCP at {np.round(final[:3], 4).tolist()}, "
                    f"outside the workspace ({self._box_text()}) although forward "
                    "kinematics placed it inside; check the TCP offset on the pendant"
                )
                ok = False
        out["ok"] = ok
        return out

    def hold(self) -> None:
        """Stop whatever runs and clear the setpoint (parent death, shutdown)."""
        self.target = None
        for stop, decel in (
            (self.arm.stop_l, STOP_DECEL_L),
            (self.arm.stop_j, STOP_DECEL_J),
        ):
            try:
                stop(decel)
            except Exception:
                pass
