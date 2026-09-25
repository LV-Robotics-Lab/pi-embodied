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
# Adapted from Show-Harness:
# interpreters/real_atomic_controller.py, interpreters/franka_atomic_controller.py,
# core/franka/franka_session.py, core/launch.py (move_to_begin_franka) and
# core/runners/real.py (descend_travel).
# Modified by pi-embodied: token decoding removed (the pi units module grounds units);
# per-call translation/rotation limits refuse instead of clamping; the setpoint is
# ramped in bounded servo ticks and ``stop`` is polled between ticks; XY/Z workspace
# box; a blocked descent re-anchors the setpoint; a controller restart that would
# resume far from the measured pose aborts instead of retrying; configurable begin
# pose for reset (stoppable joint stream or Polymetis move_to_joint_positions);
# quaternion math in numpy (no scipy); results use the RLinf franka-env shapes.

"""Cartesian-impedance setpoint control of one Franka through a Polymetis NUC.

``robot`` is duck-typed on Show-Harness's ``FrankaInterface`` (the ZeroRPC client of
the NUC's ``franka_server``): ``get_ee_pose`` -> [x, y, z, qx, qy, qz, qw],
``get_joint_positions``, ``update_desired_ee_pose(pose7)``,
``start_cartesian_impedance(Kx, Kxd)``, ``terminate_current_policy``,
``control_gripper(close)``, ``get_gripper_position`` -> [width_m],
``move_to_joint_positions(q, time_to_go)``, ``start_joint_impedance`` and
``update_desired_joint_pos(q)``.

The controller keeps the commanded TCP setpoint (Show-Harness's target-pose
accumulation) so repeated small moves do not integrate sensor noise, and walks it to
each new target in servo ticks of at most ``servo_step_m`` / ``servo_step_rad``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

# Cartesian-impedance gains (xyz, rpy), Show-Harness core/franka/franka_session.py.
DEFAULT_KX = (750.0, 750.0, 750.0, 15.0, 15.0, 15.0)
DEFAULT_KXD = (37.0, 37.0, 37.0, 2.0, 2.0, 2.0)

# ---------------------------------------------------------------------------
# Quaternions, scipy order [qx, qy, qz, qw]
# ---------------------------------------------------------------------------


def quat_normalize(q: Any) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if not n > 0:
        raise ValueError("zero-norm quaternion")
    return q / n


def quat_mul(a: Any, b: Any) -> np.ndarray:
    """Hamilton product ``a * b`` (apply ``b`` first, then ``a``)."""
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def quat_conj(q: Any) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_from_rotvec(v: Any) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.array([0.5 * v[0], 0.5 * v[1], 0.5 * v[2], 1.0]) / math.sqrt(
            1.0 + 0.25 * angle * angle
        )
    axis = v / angle
    return np.concatenate([axis * math.sin(angle / 2), [math.cos(angle / 2)]])


def quat_to_rotvec(q: Any) -> np.ndarray:
    q = quat_normalize(q)
    if q[3] < 0:
        q = -q
    s = float(np.linalg.norm(q[:3]))
    if s < 1e-12:
        return 2.0 * q[:3]
    return q[:3] / s * (2.0 * math.atan2(s, q[3]))


def quat_from_euler_xyz(rpy: Any) -> np.ndarray:
    """scipy ``Rotation.from_euler("xyz", rpy)`` (extrinsic x, then y, then z)."""
    r, p, y = (float(v) for v in rpy)
    qx = quat_from_rotvec([r, 0.0, 0.0])
    qy = quat_from_rotvec([0.0, p, 0.0])
    qz = quat_from_rotvec([0.0, 0.0, y])
    return quat_mul(qz, quat_mul(qy, qx))


def quat_rotate(q: Any, v: Any) -> np.ndarray:
    qv = np.concatenate([np.asarray(v, dtype=np.float64), [0.0]])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[:3]


def quat_angle(a: Any, b: Any) -> float:
    """Rotation angle between two orientations (rad)."""
    return float(np.linalg.norm(quat_to_rotvec(quat_mul(a, quat_conj(b)))))


def flange_to_tcp(pose7: Any, offset: Any) -> tuple[np.ndarray, np.ndarray]:
    """(TCP position, quaternion) from the pose Polymetis reports and a TCP offset."""
    p = np.asarray(pose7, dtype=np.float64)
    if p.shape != (7,) or not np.all(np.isfinite(p)):
        raise RuntimeError(f"robot returned an invalid ee pose: {p.tolist()}")
    q = quat_normalize(p[3:])
    return p[:3] + quat_rotate(q, np.asarray(offset, dtype=np.float64)), q


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


@dataclass
class PolymetisLimits:
    """Safety limits and servo settings (config/example.yaml documents each)."""

    z_floor_m: float | None = None
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None
    max_move_m: float = 0.08
    max_rotate_rad: float = 0.2
    servo_step_m: float = 0.0025
    servo_step_rad: float = 0.0125
    tick_s: float = 0.05
    settle_steps: int = 4
    settle_dt_s: float = 0.05
    move_tolerance_m: float = 0.01
    rotate_tolerance_rad: float = 0.05
    descent_stall_ratio: float = 0.7
    divergence_resync_m: float = 0.03
    tcp_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    kx: tuple[float, ...] = DEFAULT_KX
    kxd: tuple[float, ...] = DEFAULT_KXD
    gripper_close_threshold_m: float = 0.07
    grasp_open_width_m: float = 0.06
    empty_width_m: float | None = 0.005
    gripper_settle_s: float = 2.5
    gripper_min_settle_s: float = 0.5
    gripper_poll_s: float = 0.05
    gripper_stable_eps_m: float = 0.002
    gripper_motion_eps_m: float = 0.005
    gripper_sustain_s: float = 0.3
    begin_joints: tuple[float, ...] | None = None
    begin_time_s: float = 4.0
    reset_method: str = "joint_stream"
    reset_lift_m: float = 0.05

    def validate(self) -> None:
        if self.z_floor_m is None:
            raise ValueError(
                "limits.z_floor_m is not set: rest the closed gripper on the table, "
                "read the TCP z (python -m pi_embodied_services.robots.franka_polymetis"
                ".env_server --robot-config <yaml> --read-pose) and write it to the config"
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
        for name in ("max_move_m", "max_rotate_rad", "servo_step_m", "servo_step_rad"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0")
        if self.servo_step_m > 0.01 or self.servo_step_rad > 0.05:
            raise ValueError(
                "servo_step_m > 0.01 or servo_step_rad > 0.05: the impedance setpoint "
                "would jump too far per tick"
            )
        if self.begin_joints is not None and len(self.begin_joints) != 7:
            raise ValueError("reset.begin_joints must list 7 joint angles (rad)")
        if self.reset_method not in ("joint_stream", "move_to_joint_positions"):
            raise ValueError(
                "reset.method must be joint_stream or move_to_joint_positions"
            )
        if len(self.kx) != 6 or len(self.kxd) != 6:
            raise ValueError("impedance.kx / kxd must have 6 values")


def is_controller_lost_error(exc: Exception) -> bool:
    """Polymetis "no controller running" errors (Show-Harness franka_atomic_controller)."""
    msg = str(exc).lower()
    return (
        "no controller running" in msg
        or "with no controller" in msg
        or "start_joint_impedance" in msg
        or "start_cartesian_impedance" in msg
    )


class PolymetisController:
    """Bounded Cartesian moves, rotations, gripper and reset for one Franka."""

    def __init__(
        self,
        robot: Any,
        limits: PolymetisLimits,
        stop_requested: Callable[[], bool] = lambda: False,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        limits.validate()
        self.robot = robot
        self.limits = limits
        self._stop = stop_requested
        self._sleep = sleep
        self._clock = clock
        self._offset = np.asarray(limits.tcp_offset_m, dtype=np.float64)
        self.target_pos: np.ndarray | None = None
        self.target_quat: np.ndarray | None = None
        self.gripper_open: bool | None = None
        self.restarts = 0

    # -- frames ------------------------------------------------------------

    def _tcp_to_flange(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        return np.concatenate([pos - quat_rotate(quat, self._offset), quat])

    def measured(self) -> tuple[np.ndarray, np.ndarray]:
        return flange_to_tcp(self.robot.get_ee_pose(), self._offset)

    def width(self) -> float:
        return float(np.asarray(self.robot.get_gripper_position()).reshape(-1)[0])

    # -- controller --------------------------------------------------------

    def start_impedance(self) -> None:
        """(Re)start Cartesian impedance at the current pose, then resync the setpoint."""
        try:
            self.robot.terminate_current_policy()
        except Exception:  # nothing running is fine
            pass
        self.robot.start_cartesian_impedance(
            np.asarray(self.limits.kx, dtype=float),
            np.asarray(self.limits.kxd, dtype=float),
        )
        self.sync()

    def sync(self) -> None:
        """Reset the setpoint to the measured pose (Show-Harness sync_from_robot)."""
        self.target_pos, self.target_quat = self.measured()
        if self.gripper_open is None:
            self.gripper_open = self.width() >= self.limits.gripper_close_threshold_m

    def _command(self, pos: np.ndarray, quat: np.ndarray) -> None:
        """Send one setpoint; restart a lost controller once (Show-Harness self-heal).

        A restart starts impedance at the measured pose. The same setpoint is re-sent
        only if it is within ``divergence_resync_m`` of that pose; otherwise (a reflex
        stopped the arm somewhere else) the setpoint stays at the measured pose and
        the call fails, so a restart never resumes a long move by itself.
        """
        pose = self._tcp_to_flange(pos, quat)
        try:
            self.robot.update_desired_ee_pose(pose)
            return
        except Exception as exc:
            if not is_controller_lost_error(exc):
                raise
        self.restarts += 1
        self.start_impedance()
        gap = float(np.linalg.norm(self.target_pos - pos))
        if gap > self.limits.divergence_resync_m:
            raise RuntimeError(
                f"the impedance controller was lost and restarted {gap:.3f} m from the "
                "commanded setpoint; the setpoint was reset to the measured pose and "
                "the motion aborted"
            )
        self.target_pos, self.target_quat = pos.copy(), quat.copy()
        self.robot.update_desired_ee_pose(pose)

    def _prepare(self) -> None:
        """Resync if the arm drifted far from the setpoint (pushed, reflex, restart)."""
        if self.target_pos is None:
            self.sync()
            return
        pos, _ = self.measured()
        if (
            float(np.linalg.norm(pos - self.target_pos))
            > self.limits.divergence_resync_m
        ):
            self.sync()

    def _wait(self, seconds: float) -> None:
        if seconds > 0:
            self._sleep(seconds)

    def _settle(self) -> bool:
        """Re-command the setpoint ``settle_steps`` times. False when stopped."""
        for _ in range(max(1, self.limits.settle_steps)):
            if self._stop():
                return False
            self._command(self.target_pos, self.target_quat)
            self._wait(self.limits.settle_dt_s)
        return True

    # -- limits ------------------------------------------------------------

    def _outside(self, p: np.ndarray) -> float:
        lo = np.asarray(self.limits.workspace_min, dtype=np.float64).copy()
        hi = np.asarray(self.limits.workspace_max, dtype=np.float64)
        lo[2] = max(lo[2], float(self.limits.z_floor_m))
        return float(np.sum(np.maximum(0.0, lo - p) + np.maximum(0.0, p - hi)))

    def check_target(self, start: np.ndarray, target: np.ndarray) -> None:
        """Refuse a target outside the box / below the floor unless it moves back in."""
        out = self._outside(target)
        if out > 1e-6 and out >= self._outside(start) - 1e-6:
            lo, hi = self.limits.workspace_min, self.limits.workspace_max
            raise ValueError(
                f"the move ends at {np.round(target, 4).tolist()}, outside the workspace "
                f"(x {lo[0]}..{hi[0]}, y {lo[1]}..{hi[1]}, z {self.limits.z_floor_m}"
                f"..{hi[2]} m); nothing was commanded"
            )

    # -- motion ------------------------------------------------------------

    def _ramp(
        self, pos: np.ndarray, quat: np.ndarray, dpos: np.ndarray, drot: np.ndarray
    ) -> tuple[int, bool]:
        """Walk the setpoint from (pos, quat) by (dpos, drot) in servo ticks."""
        lim = self.limits
        n = max(
            1,
            math.ceil(float(np.linalg.norm(dpos)) / lim.servo_step_m - 1e-9),
            math.ceil(float(np.linalg.norm(drot)) / lim.servo_step_rad - 1e-9),
        )
        for i in range(1, n + 1):
            if self._stop():
                return i - 1, True
            f = i / n
            p = pos + dpos * f
            q = quat_mul(quat_from_rotvec(drot * f), quat)
            self._command(p, q)
            self.target_pos, self.target_quat = p, q
            self._wait(lim.tick_s)
        return n, False

    def move_delta(self, delta_xyz: Any) -> dict[str, Any]:
        """Translate the TCP by a base-frame delta (m); refused beyond the limits."""
        delta = np.asarray(delta_xyz, dtype=np.float64).reshape(-1)
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            raise ValueError("delta_xyz must be 3 finite values")
        norm = float(np.linalg.norm(delta))
        if norm > self.limits.max_move_m + 1e-9:
            raise ValueError(
                f"delta_xyz moves {norm:.4f} m; the limit is {self.limits.max_move_m} m "
                "per call (limits.max_move_m). Split the move; nothing was commanded"
            )
        self._prepare()
        start_pos, start_quat = self.measured()
        origin = self.target_pos.copy()
        target = origin + delta
        self.check_target(origin, target)
        steps, cancelled = self._ramp(origin, self.target_quat, delta, np.zeros(3))
        if not cancelled:
            cancelled = not self._settle()
        final_pos, final_quat = self.measured()
        result: dict[str, Any] = {
            "ok": False,
            "requested_delta_xyz_base": delta.tolist(),
            "start_tcp_pose": np.concatenate([start_pos, start_quat]).tolist(),
            "final_tcp_pose": np.concatenate([final_pos, final_quat]).tolist(),
            "target_tcp_pose": self.target_pose().tolist(),
            "final_error_m": float(np.linalg.norm(target - final_pos)),
            "steps_used": steps,
            "states": None,
        }
        # Show-Harness descend_travel + proprioception DESCEND_STALL_RATIO.
        if delta[2] < -1e-3 and not cancelled:
            travelled = float(start_pos[2] - final_pos[2])
            if travelled < self.limits.descent_stall_ratio * -delta[2]:
                # Stop pressing: the setpoint stays at the height the arm reached.
                self.target_pos = self.target_pos.copy()
                self.target_pos[2] = final_pos[2]
                self._command(self.target_pos, self.target_quat)
                result["descent_blocked"] = True
                result["descent_travelled_m"] = travelled
                result["note"] = (
                    f"descent blocked: moved {travelled:.4f} of {-delta[2]:.4f} m down; "
                    "the gripper is resting on something. The setpoint was re-anchored "
                    "at the reached height"
                )
        result["ok"] = (
            not cancelled
            and not result.get("descent_blocked")
            and result["final_error_m"] <= self.limits.move_tolerance_m
        )
        if cancelled:
            result["cancelled"] = True
        return result

    def rotate_delta(self, delta_rpy: Any) -> dict[str, Any]:
        """Rotate the TCP in place: target = R(xyz euler delta) * current (base frame)."""
        rpy = np.asarray(delta_rpy, dtype=np.float64).reshape(-1)
        if rpy.shape != (3,) or not np.all(np.isfinite(rpy)):
            raise ValueError("delta_rpy must be 3 finite values")
        drot = quat_to_rotvec(quat_from_euler_xyz(rpy))
        angle = float(np.linalg.norm(drot))
        if angle > self.limits.max_rotate_rad + 1e-9:
            raise ValueError(
                f"delta_rpy rotates {angle:.4f} rad; the limit is "
                f"{self.limits.max_rotate_rad} rad per call (limits.max_rotate_rad). "
                "Split the rotation; nothing was commanded"
            )
        self._prepare()
        start_pos, start_quat = self.measured()
        origin_q = self.target_quat.copy()
        target_q = quat_mul(quat_from_rotvec(drot), origin_q)
        steps, cancelled = self._ramp(
            self.target_pos.copy(), origin_q, np.zeros(3), drot
        )
        if not cancelled:
            cancelled = not self._settle()
        final_pos, final_quat = self.measured()
        error = quat_angle(target_q, final_quat)
        result: dict[str, Any] = {
            "ok": not cancelled and error <= self.limits.rotate_tolerance_rad,
            "requested_delta_rpy_base": rpy.tolist(),
            "start_tcp_pose": np.concatenate([start_pos, start_quat]).tolist(),
            "final_tcp_pose": np.concatenate([final_pos, final_quat]).tolist(),
            "target_tcp_pose": self.target_pose().tolist(),
            "final_error_rad": error,
            "steps_used": steps,
            "states": None,
        }
        if cancelled:
            result["cancelled"] = True
        return result

    def target_pose(self) -> np.ndarray:
        return np.concatenate([self.target_pos, self.target_quat])

    # -- gripper -----------------------------------------------------------

    def _await_gripper(self, decisive_max_m: float | None) -> tuple[float, bool]:
        """Show-Harness _await_gripper_settled, polling ``stop``. -> (width, stopped)."""
        lim = self.limits
        start = self._clock()
        w0 = self.width()
        last, moved, stable_since = w0, False, None
        while True:
            if self._clock() - start >= lim.gripper_settle_s:
                return last, False
            if self._stop():
                return last, True
            self._wait(lim.gripper_poll_s)
            now = self._clock()
            cur = self.width()
            moved = moved or abs(cur - w0) > lim.gripper_motion_eps_m
            if abs(cur - last) <= lim.gripper_stable_eps_m:
                stable_since = now if stable_since is None else stable_since
            else:
                stable_since = None
            last = cur
            if (
                moved
                and stable_since is not None
                and now - stable_since >= lim.gripper_sustain_s
                and (decisive_max_m is None or cur <= decisive_max_m)
                and now - start >= lim.gripper_min_settle_s
            ):
                return cur, False

    def set_gripper(self, *, open: bool) -> dict[str, Any]:
        """Open or close; a close that ends at/below ``empty_width_m`` is reopened."""
        lim = self.limits
        self.robot.control_gripper(not open)  # Show-Harness: True = close
        self.gripper_open = bool(open)
        steps = 1
        width, cancelled = self._await_gripper(None if open else lim.grasp_open_width_m)
        result: dict[str, Any] = {"target_gripper_open": bool(open)}
        if (
            not open
            and not cancelled
            and lim.empty_width_m is not None
            and width <= lim.empty_width_m
        ):
            self.robot.control_gripper(False)
            self.gripper_open = True
            steps += 1
            width, cancelled = self._await_gripper(None)
            result["grasp_empty"] = True
            result["note"] = (
                f"empty grasp: the gripper closed to {width:.4f} m <= "
                f"{lim.empty_width_m} m (nothing between the fingers) and was reopened"
            )
        result.update(
            ok=not cancelled and not result.get("grasp_empty"),
            steps_used=steps,
            gripper_width_m=width,
        )
        if cancelled:
            result["cancelled"] = True
        return result

    # -- reset -------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        """Open, lift clear, move to ``begin_joints``, restart impedance at the begin pose."""
        lim = self.limits
        if lim.begin_joints is None:
            raise ValueError(
                "reset.begin_joints is not set: jog the arm to a safe begin pose, read "
                "the joints (--read-pose) and write them to the config"
            )
        info: dict[str, Any] = {"begin_joints": list(lim.begin_joints)}
        grip = self.set_gripper(open=True)
        info["gripper"] = grip
        if grip.get("cancelled"):
            return {"ok": False, "cancelled": True, "info": info}
        self._prepare()
        room = float(lim.workspace_max[2]) - float(self.target_pos[2])
        lift = max(0.0, min(lim.reset_lift_m, room))
        if lift > 1e-4:
            _, cancelled = self._ramp(
                self.target_pos.copy(),
                self.target_quat.copy(),
                np.array([0.0, 0.0, lift]),
                np.zeros(3),
            )
            info["lifted_m"] = lift
            if cancelled:
                return {"ok": False, "cancelled": True, "info": info}
        goal = np.asarray(lim.begin_joints, dtype=np.float64)
        info["method"] = lim.reset_method
        if lim.reset_method == "move_to_joint_positions":
            # Blocking on the NUC: not interruptible by ``stop``.
            self.robot.move_to_joint_positions(goal, float(lim.begin_time_s))
        else:
            q0 = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
            self.robot.start_joint_impedance(None, None)
            n = max(1, math.ceil(lim.begin_time_s / max(lim.tick_s, 1e-3)))
            for i in range(1, n + 1):
                if self._stop():
                    # Hold where the stream reached, then back to Cartesian control.
                    self.start_impedance()
                    return {"ok": False, "cancelled": True, "info": info}
                f = 0.5 - 0.5 * math.cos(math.pi * i / n)  # rest-to-rest
                self.robot.update_desired_joint_pos(q0 + (goal - q0) * f)
                self._wait(lim.tick_s)
            for _ in range(max(1, lim.settle_steps)):
                self.robot.update_desired_joint_pos(goal)
                self._wait(lim.settle_dt_s)
        self.start_impedance()
        joints = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
        info["joint_error_rad"] = float(np.max(np.abs(joints - goal)))
        return {"ok": info["joint_error_rad"] <= 0.05, "info": info}

    # -- state -------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """RLinf-shaped ``raw_base_state`` (tcp_pose xyzw, gripper_position, ...)."""
        pos, quat = self.measured()
        width = self.width()
        out: dict[str, Any] = {
            "tcp_pose": np.concatenate([pos, quat]).tolist(),
            "gripper_position": [width],
            "gripper_open": bool(self.gripper_open)
            if self.gripper_open is not None
            else width >= self.limits.gripper_close_threshold_m,
            "arm_joint_position": np.asarray(
                self.robot.get_joint_positions(), dtype=float
            ).tolist(),
        }
        if self.target_pos is not None:
            out["target_tcp_pose"] = self.target_pose().tolist()
        return out
