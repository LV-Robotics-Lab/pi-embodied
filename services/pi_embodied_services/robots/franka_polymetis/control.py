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
# core/franka/franka_session.py, core/launch.py (move_to_begin_franka),
# core/runners/real.py (descend_travel) and plugins/smooth/plugin.py (SmoothPlugin.plan,
# the min-jerk / cruise profile) with the chaining of RealAtomicController._drive_to_target.
# Modified by pi-embodied: token decoding removed (the pi units module grounds units);
# per-call translation/rotation limits refuse instead of clamping; the setpoint is
# ramped in bounded servo ticks and ``stop`` is polled between ticks; XY/Z workspace
# box checked per axis; a tool tilt limit; every tick compares the measured TCP with
# the setpoint and a blocked motion re-anchors the setpoint at the measured pose; at
# most one controller restart per call; validated (finite, bounded) limits, gains and
# begin joints; configurable begin pose for reset (stoppable joint stream or Polymetis
# move_to_joint_positions) at a bounded joint speed; quaternion math in numpy (no
# scipy); results use the RLinf franka-env shapes. The smooth profile is stretched in
# time (more waypoints at the same delay) until no waypoint advances more than the servo
# step cap, a chained move must arrive within ``smooth.chain_window_s`` of the previous
# one, and there is no min_waypoint_m thinning (impedance setpoints are not quantized).

"""Cartesian-impedance setpoint control of one Franka through a Polymetis NUC.

``robot`` is duck-typed on Show-Harness's ``FrankaInterface`` (the ZeroRPC client of
the NUC's ``franka_server``): ``get_ee_pose`` -> [x, y, z, qx, qy, qz, qw],
``get_joint_positions``, ``update_desired_ee_pose(pose7)``,
``start_cartesian_impedance(Kx, Kxd)``, ``terminate_current_policy``,
``control_gripper(close)``, ``get_gripper_position`` -> [width_m],
``get_gripper_state`` -> {width, is_grasped, is_moving} (Polymetis ``GripperState``),
``move_to_joint_positions(q, time_to_go)``, ``start_joint_impedance`` and
``update_desired_joint_pos(q)``.

The controller keeps the commanded TCP setpoint (Show-Harness's target-pose
accumulation) so repeated small moves do not integrate sensor noise, and walks it to
each new target in waypoints of at most ``servo_step_m`` / ``servo_step_rad`` per
``tick_s``: along a min-jerk profile when ``smooth`` is on (Show-Harness's smooth
plugin), else in equal steps.

Chaining (Show-Harness ``continuous`` + ``blend``): a translation sent with
``continuous=True`` (another move in the same direction follows at once) ends at cruise
speed without the settle re-commands; the next translation, if it arrives within
``smooth_chain_window_s`` and points within ~25 deg of it (cos >= 0.9), starts at that
speed, so the setpoint path is velocity-continuous and the arm does not stop between
the two. Anything else (another direction, a rotation, the gripper, a reset, a late
move) first brings the stream to rest.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

# Cartesian-impedance gains (xyz, rpy), Show-Harness core/franka/franka_session.py.
DEFAULT_KX = (750.0, 750.0, 750.0, 15.0, 15.0, 15.0)
DEFAULT_KXD = (37.0, 37.0, 37.0, 2.0, 2.0, 2.0)
# Largest accepted gains: 2x the Show-Harness defaults (nuc_server.py repeats these).
MAX_KX = (1500.0, 1500.0, 1500.0, 30.0, 30.0, 30.0)
MAX_KXD = (74.0, 74.0, 74.0, 4.0, 4.0, 4.0)
# Joint limits (rad): the intersection of the Franka Panda and FR3 datasheet ranges.
JOINT_MIN = (-2.7437, -1.7628, -2.8973, -3.0421, -2.8065, 0.5445, -2.8973)
JOINT_MAX = (2.7437, 1.7628, 2.8973, -0.1518, 2.8065, 3.7525, 2.8973)
# Reset joint speed cap. A rest-to-rest profile peaks at up to 1.875x (min-jerk) its
# mean speed, so a reset takes at least PEAK_TO_MEAN * distance / MAX_JOINT_SPEED.
MAX_JOINT_SPEED_RAD_S = 0.5
PEAK_TO_MEAN = 1.875
# Setpoint speed caps (servo step / tick).
MAX_SERVO_SPEED_M_S = 0.1
MAX_SERVO_SPEED_RAD_S = 0.5
# Show-Harness chains a move only onto a stream flowing in (nearly) the same direction.
CHAIN_MIN_COS = 0.9


def smooth_fractions(n: int, v0: float = 0.0, v1: float = 0.0) -> list[float]:
    """Show-Harness ``SmoothPlugin.plan``: ``n`` eased fractions in (0, 1].

    ``s(t) = v0 t + a3 t^3 + a4 t^4 + a5 t^5`` with ``s(0) = 0``, ``s(1) = 1``,
    ``s'(0) = v0``, ``s'(1) = v1`` and zero end accelerations; ``v0 = v1 = 0`` is the
    classic min-jerk ``10t^3 - 15t^4 + 6t^5``, ``v0 = v1 = 1`` a constant-rate line.
    Never steps backward, never passes the target, and lands exactly on it.
    """
    n = max(1, int(n))
    a_, b_ = 1.0 - float(v0), float(v1) - float(v0)
    a3, a4, a5 = 10.0 * a_ - 4.0 * b_, -15.0 * a_ + 7.0 * b_, 6.0 * a_ - 3.0 * b_
    out: list[float] = []
    prev = 0.0
    for i in range(1, n + 1):
        t = i / n
        s = float(v0) * t + a3 * t**3 + a4 * t**4 + a5 * t**5
        prev = min(1.0, max(s, prev))
        out.append(prev)
    out[-1] = 1.0
    return out


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


def tool_tilt(quat: Any) -> float:
    """Angle (rad) between the tool z-axis and straight down (base -z)."""
    down = -float(quat_rotate(quat, (0.0, 0.0, 1.0))[2])
    return math.acos(max(-1.0, min(1.0, down)))


# RLinf's Franka ``tcp_pose`` is libfranka's ``O_T_EE`` (RLinf/RLinf@807e5fd
# rlinf/robotics/parts/arms/franka.py ``tcp_pose  # FrankaState.O_T_EE``, read in
# franka_ros.py ``_on_arm_state_msg`` and franky.py ``get_state``). ``O_T_EE`` =
# ``O_T_F * F_T_EE`` with the Franka Hand's ``F_T_EE`` (libfranka/Desk default,
# franka_description franka_hand: hand joint rpy [0, 0, -pi/4], TCP z 0.1034 m):
# 0.1034 m along the flange z-axis and -45 deg (-pi/4) about it. Polymetis reports
# the flange (panda_link8), so matching RLinf needs both the offset and the yaw.
def tcp_yaw_quat(yaw_deg: float) -> np.ndarray:
    """Rotation of the TCP frame about the flange z-axis (``robot.tcp_yaw_deg``)."""
    return quat_from_rotvec([0.0, 0.0, math.radians(float(yaw_deg))])


def flange_to_tcp(
    pose7: Any, offset: Any, yaw_deg: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """(TCP position, quaternion) from the pose Polymetis reports and the TCP
    transform ``T_flange_tcp = Trans(offset) * Rz(yaw_deg)`` (offset in the flange
    frame). libfranka's Franka Hand TCP (``O_T_EE``, what RLinf reports as
    ``tcp_pose``) is ``offset = [0, 0, 0.1034]``, ``yaw_deg = -45``."""
    p = np.asarray(pose7, dtype=np.float64)
    if p.shape != (7,) or not np.all(np.isfinite(p)):
        raise RuntimeError(f"robot returned an invalid ee pose: {p.tolist()}")
    q = quat_normalize(p[3:])
    pos = p[:3] + quat_rotate(q, np.asarray(offset, dtype=np.float64))
    return pos, quat_normalize(quat_mul(q, tcp_yaw_quat(yaw_deg)))


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
    divergence_resync_m: float = 0.01
    max_tracking_error_m: float = 0.015
    max_tilt_rad: float = 0.5
    tcp_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tcp_yaw_deg: float = 0.0
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
    smooth: bool = True
    smooth_substeps: int = 20
    smooth_dt_s: float = 0.05
    smooth_blend: bool = False  # Show-Harness's Franka default: no chaining
    smooth_cruise: float = 1.0
    smooth_chain_window_s: float = 1.0

    def validate(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None or isinstance(value, str):
                continue
            try:
                arr = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                raise ValueError(f"{f.name} must be numeric, got {value!r}") from None
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{f.name} must be finite, got {value!r}")
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

        def bound(name: str, lo: float, hi: float) -> None:
            value = getattr(self, name)
            if not lo <= value <= hi:
                raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")

        bound("max_move_m", 1e-3, 0.2)
        bound("max_rotate_rad", 1e-2, 0.5)
        bound("servo_step_m", 1e-4, 0.01)
        bound("servo_step_rad", 1e-4, 0.05)
        bound("tick_s", 1e-3, 0.1)
        if (
            self.servo_step_m / self.tick_s > MAX_SERVO_SPEED_M_S + 1e-9
            or self.servo_step_rad / self.tick_s > MAX_SERVO_SPEED_RAD_S + 1e-9
        ):
            raise ValueError(
                f"servo_step_m / tick_s must be <= {MAX_SERVO_SPEED_M_S} m/s and "
                f"servo_step_rad / tick_s <= {MAX_SERVO_SPEED_RAD_S} rad/s"
            )
        if int(self.settle_steps) != self.settle_steps:
            raise ValueError("settle_steps must be an integer")
        bound("settle_steps", 1, 50)
        bound("settle_dt_s", 0.0, 1.0)
        bound("move_tolerance_m", 1e-3, 0.05)
        bound("rotate_tolerance_rad", 5e-3, 0.2)
        bound("descent_stall_ratio", 0.1, 1.0)
        bound("divergence_resync_m", 2e-3, 0.05)
        bound("max_tracking_error_m", 2e-3, 0.05)
        bound("max_tilt_rad", 1e-2, math.pi / 2)
        offset = np.asarray(self.tcp_offset_m)
        if offset.shape != (3,) or np.any(np.abs(offset) > 0.3):
            raise ValueError("robot.tcp_offset_m must be [x, y, z] within +-0.3 m")
        bound("tcp_yaw_deg", -180.0, 180.0)
        kx, kxd = np.asarray(self.kx), np.asarray(self.kxd)
        if kx.shape != (6,) or kxd.shape != (6,):
            raise ValueError("impedance.kx / kxd must have 6 values")
        if (
            np.any(kx <= 0)
            or np.any(kx > MAX_KX)
            or np.any(kxd < 0)
            or np.any(kxd > MAX_KXD)
        ):
            raise ValueError(
                f"impedance gains out of range: 0 < kx <= {list(MAX_KX)}, "
                f"0 <= kxd <= {list(MAX_KXD)} (2x the Show-Harness defaults)"
            )
        bound("gripper_close_threshold_m", 0.0, 0.1)
        bound("grasp_open_width_m", 0.0, 0.1)
        if self.empty_width_m is not None:
            bound("empty_width_m", 0.0, 0.1)
        bound("gripper_settle_s", 0.0, 10.0)
        bound("gripper_min_settle_s", 0.0, 10.0)
        bound("gripper_poll_s", 1e-3, 1.0)
        bound("begin_time_s", 0.5, 60.0)
        bound("reset_lift_m", 0.0, 0.2)
        if self.begin_joints is not None:
            q = np.asarray(self.begin_joints)
            if q.shape != (7,):
                raise ValueError("reset.begin_joints must list 7 joint angles (rad)")
            if np.any(q <= JOINT_MIN) or np.any(q >= JOINT_MAX):
                raise ValueError(
                    f"reset.begin_joints {q.tolist()} are outside the Franka joint "
                    f"limits {list(JOINT_MIN)} .. {list(JOINT_MAX)}"
                )
        if self.reset_method not in ("joint_stream", "move_to_joint_positions"):
            raise ValueError(
                "reset.method must be joint_stream or move_to_joint_positions"
            )
        for name in ("smooth", "smooth_blend"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false")
        if int(self.smooth_substeps) != self.smooth_substeps:
            raise ValueError("smooth.substeps must be an integer")
        bound("smooth_substeps", 1, 1000)
        bound("smooth_dt_s", 1e-3, 0.1)
        bound("smooth_cruise", 0.0, 1.5)
        bound("smooth_chain_window_s", 0.0, 10.0)


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
        self._yaw_inv = quat_conj(tcp_yaw_quat(limits.tcp_yaw_deg))
        self.target_pos: np.ndarray | None = None
        self.target_quat: np.ndarray | None = None
        # The last gripper command (True = open); None until one is sent. The state
        # reports the measured gripper, this only as ``gripper_commanded_open``.
        self.commanded_open: bool | None = None
        self.restarts = 0
        self._call_restarts = 0
        # Chained motion (smooth + blend): direction and deadline of a translation
        # that ended at cruise speed without settling (None = at rest).
        self._stream_dir: np.ndarray | None = None
        self._stream_until = 0.0

    # -- frames ------------------------------------------------------------

    def _tcp_to_flange(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Inverse of ``flange_to_tcp``: the flange pose7 Polymetis is commanded with."""
        q = quat_normalize(quat_mul(quat, self._yaw_inv))
        return np.concatenate([pos - quat_rotate(q, self._offset), q])

    def measured(self) -> tuple[np.ndarray, np.ndarray]:
        return flange_to_tcp(
            self.robot.get_ee_pose(), self._offset, self.limits.tcp_yaw_deg
        )

    def width(self) -> float:
        return float(np.asarray(self.robot.get_gripper_position()).reshape(-1)[0])

    def gripper_state(self) -> dict[str, Any]:
        """Measured gripper: width, grasp / moving flags (None when not reported) and
        ``closed`` = grasped, or narrower than ``gripper_close_threshold_m``."""
        raw = self.robot.get_gripper_state()
        width = float(raw["width"])

        def flag(key: str) -> bool | None:
            v = raw.get(key)
            return None if v is None else bool(v)

        grasped = flag("is_grasped")
        closed = bool(grasped) or width < self.limits.gripper_close_threshold_m
        return {
            "width": width,
            "grasped": grasped,
            "moving": flag("is_moving"),
            "closed": closed,
        }

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
        self._stream_dir = None
        self.target_pos, self.target_quat = self.measured()

    def _command(self, pos: np.ndarray, quat: np.ndarray) -> None:
        """Send one setpoint; restart a lost controller at most once per call.

        A restart starts impedance at the measured pose. The same setpoint is re-sent
        only on the first restart of the call and only if it is within
        ``divergence_resync_m`` of that pose. A far restart, or a second loss in the
        same call (a reflex that recurs), leaves impedance holding the measured pose
        and fails the call, so a restart never keeps pushing into an obstacle.
        """
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
            raise RuntimeError(f"refusing a non-finite setpoint {pos.tolist()}")
        pose = self._tcp_to_flange(pos, quat)
        try:
            self.robot.update_desired_ee_pose(pose)
            return
        except Exception as exc:
            if not is_controller_lost_error(exc):
                raise
        self.restarts += 1
        self._call_restarts += 1
        self.start_impedance()
        if self._call_restarts > 1:
            raise RuntimeError(
                "the impedance controller was lost again in the same call (a reflex "
                "or collision recurred); it holds the measured pose and the motion "
                "aborted"
            )
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

    def _displaced(self) -> bool:
        """Whether the setpoint is off the measured pose by more than the tolerances."""
        pos, quat = self.measured()
        return (
            float(np.linalg.norm(pos - self.target_pos)) > self.limits.move_tolerance_m
            or quat_angle(quat, self.target_quat) > self.limits.rotate_tolerance_rad
        )

    def _reanchor(self) -> None:
        """Setpoint := measured pose on all axes, so impedance stops pressing."""
        self._stream_dir = None
        self.target_pos, self.target_quat = self.measured()
        self._command(self.target_pos, self.target_quat)

    def _motion(self, body: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Run one motion call; it never ends with the setpoint displaced into contact.

        On an error the setpoint is re-anchored at the measured pose (impedance is
        restarted there if the controller is gone) and the error propagates.
        """
        self._call_restarts = 0
        try:
            result = body()
        except Exception:
            self._stream_dir = None
            try:
                if self.target_pos is None or self._displaced():
                    self.target_pos, self.target_quat = self.measured()
                    self.robot.update_desired_ee_pose(
                        self._tcp_to_flange(self.target_pos, self.target_quat)
                    )
            except Exception:
                try:
                    self.start_impedance()
                except Exception:  # the original error is the one to report
                    pass
            raise
        if self._displaced():
            self._reanchor()
            result["ok"] = False
            result["reanchored"] = True
            if "flowing" in result:
                result["flowing"] = False
        if self.target_pos is not None:
            result["target_tcp_pose"] = self.target_pose().tolist()
        result["controller_restarts"] = self._call_restarts
        return result

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

    def _stream_live(self) -> bool:
        """Whether a chained translation may still be flowing (within the window)."""
        return self._stream_dir is not None and self._clock() <= self._stream_until

    def end_stream(self) -> bool:
        """Show-Harness end_stream: bring a chained motion to rest (settle at the
        setpoint). A no-op at rest or once the chain window has passed. False when
        stopped."""
        live = self._stream_live()
        self._stream_dir = None
        return self._settle() if live else True

    # -- limits ------------------------------------------------------------

    def _violation(self, p: np.ndarray) -> np.ndarray:
        """Per-axis distance outside the box (the floor raises the z minimum)."""
        if not np.all(np.isfinite(p)):
            return np.full(3, np.inf)
        lo = np.asarray(self.limits.workspace_min, dtype=np.float64).copy()
        hi = np.asarray(self.limits.workspace_max, dtype=np.float64)
        lo[2] = max(lo[2], float(self.limits.z_floor_m))
        return np.maximum(0.0, lo - p) + np.maximum(0.0, p - hi)

    def check_target(
        self,
        start: np.ndarray,
        target: np.ndarray,
        *,
        strict: bool = True,
        what: str = "the move",
    ) -> None:
        """Refuse a target outside the box / below the floor.

        From outside the box (hand-guided, pushed) a target is allowed only if no axis
        gets further out and, when ``strict``, the total distance outside shrinks.
        """
        out, was = self._violation(target), self._violation(start)
        refused = not np.all(np.isfinite(out)) or bool(np.any(out > was + 1e-9))
        if strict and out.sum() > 1e-6 and out.sum() >= was.sum() - 1e-6:
            refused = True
        if refused:
            lo, hi = self.limits.workspace_min, self.limits.workspace_max
            raise ValueError(
                f"{what} ends at {np.round(target, 4).tolist()}, outside the workspace "
                f"(x {lo[0]}..{hi[0]}, y {lo[1]}..{hi[1]}, z {self.limits.z_floor_m}"
                f"..{hi[2]} m); nothing was commanded"
            )

    # -- motion ------------------------------------------------------------

    def _ticks(self, dpos: np.ndarray, drot: np.ndarray) -> int:
        lim = self.limits
        return max(
            1,
            math.ceil(float(np.linalg.norm(dpos)) / lim.servo_step_m - 1e-9),
            math.ceil(float(np.linalg.norm(drot)) / lim.servo_step_rad - 1e-9),
        )

    def plan(
        self, dist_m: float, angle_rad: float, v0: float = 0.0, v1: float = 0.0
    ) -> tuple[list[float], float]:
        """(fractions, delay) of one move's waypoints.

        ``smooth``: Show-Harness's min-jerk / cruise profile over ``smooth_substeps``
        waypoints ``smooth_dt_s`` apart, given more waypoints at the same delay (a longer
        move) until no waypoint advances more than the servo cap scaled to that delay.
        The count is sized for the rest-to-rest profile (the fastest peak, 1.875x the
        mean), so equal moves of a chain get equal durations and speeds meet at the
        joins. Otherwise: equal steps of at most the cap, ``tick_s`` apart.
        """
        lim = self.limits
        if not lim.smooth:
            n = self._ticks(
                np.array([dist_m, 0.0, 0.0]), np.array([angle_rad, 0.0, 0.0])
            )
            return [i / n for i in range(1, n + 1)], lim.tick_s
        scale = min(1.0, lim.smooth_dt_s / lim.tick_s)
        cap_m, cap_rad = lim.servo_step_m * scale, lim.servo_step_rad * scale
        n = int(lim.smooth_substeps)
        for _ in range(64):
            worst = max(
                float(np.max(np.diff([0.0, *smooth_fractions(n, a, b)])))
                for a, b in ((0.0, 0.0), (v0, v1))
            )
            over = max(worst * dist_m / cap_m, worst * angle_rad / cap_rad)
            if over <= 1.0 + 1e-9:
                break
            n = max(n + 1, math.ceil(n * over))
        return smooth_fractions(n, v0, v1), lim.smooth_dt_s

    def _ramp(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        dpos: np.ndarray,
        drot: np.ndarray,
        v0: float = 0.0,
        v1: float = 0.0,
    ) -> tuple[int, str | None]:
        """Walk the setpoint from (pos, quat) by (dpos, drot) along :meth:`plan`.

        Returns (waypoints, None | "cancelled" | "blocked"). ``stop`` is polled before
        every waypoint. After each the measured TCP is compared with the setpoint; a gap
        above ``max_tracking_error_m`` (contact) stops the ramp and re-anchors the
        setpoint at the measured pose.
        """
        lim = self.limits
        fractions, delay = self.plan(
            float(np.linalg.norm(dpos)), float(np.linalg.norm(drot)), v0, v1
        )
        for i, f in enumerate(fractions, start=1):
            if self._stop():
                return i - 1, "cancelled"
            p = pos + dpos * f
            q = quat_normalize(quat_mul(quat_from_rotvec(drot * f), quat))
            self.check_target(pos, p, strict=False)
            self._command(p, q)
            self.target_pos, self.target_quat = p, q
            self._wait(delay)
            measured, _ = self.measured()
            if float(np.linalg.norm(measured - p)) > lim.max_tracking_error_m:
                self._reanchor()
                return i, "blocked"
        return len(fractions), None

    def _blocked_note(self) -> str:
        return (
            f"blocked: the TCP lagged the setpoint by more than "
            f"{self.limits.max_tracking_error_m} m (contact or an obstacle); the "
            "motion stopped and the setpoint was re-anchored at the measured pose"
        )

    def move_delta(self, delta_xyz: Any, continuous: bool = False) -> dict[str, Any]:
        """Translate the TCP by a base-frame delta (m); refused beyond the limits.

        ``continuous``: another move in about the same direction follows at once (the
        rest of a repeated unit), so with ``smooth`` and ``smooth_blend`` this one ends
        at cruise speed without settling and the next starts at that speed.
        """
        delta = np.asarray(delta_xyz, dtype=np.float64).reshape(-1)
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            raise ValueError("delta_xyz must be 3 finite values")
        norm = float(np.linalg.norm(delta))
        if norm > self.limits.max_move_m + 1e-9:
            raise ValueError(
                f"delta_xyz moves {norm:.4f} m; the limit is {self.limits.max_move_m} m "
                "per call (limits.max_move_m). Split the move; nothing was commanded"
            )
        return self._motion(lambda: self._move(delta, bool(continuous)))

    def _move(self, delta: np.ndarray, continuous: bool = False) -> dict[str, Any]:
        lim = self.limits
        live, flowing = self._stream_live(), self._stream_dir
        self._prepare()
        start_pos, start_quat = self.measured()
        origin = self.target_pos.copy()
        target = origin + delta
        self.check_target(origin, target)
        norm = float(np.linalg.norm(delta))
        blend = lim.smooth and lim.smooth_blend and norm > 1e-9
        v0 = 0.0
        if (
            blend
            and live
            and self._stream_dir is not None  # _prepare did not resync
            and float(np.dot(delta / norm, flowing)) >= CHAIN_MIN_COS
        ):
            v0 = lim.smooth_cruise
        elif not self.end_stream():
            return self._move_result(
                delta, target, start_pos, start_quat, 0, "cancelled"
            )
        self._stream_dir = None
        v1 = lim.smooth_cruise if blend and continuous else 0.0
        steps, stopped = self._ramp(
            origin, self.target_quat, delta, np.zeros(3), v0, v1
        )
        if stopped is None and v1 > 0.0:
            # Flow into the next move: no settle, remember what the stream rides.
            self._stream_dir = delta / norm
            self._stream_until = self._clock() + lim.smooth_chain_window_s
        elif stopped is None and not self._settle():
            stopped = "cancelled"
        result = self._move_result(delta, target, start_pos, start_quat, steps, stopped)
        if lim.smooth:
            result["chained"] = v0 > 0.0
            result["flowing"] = self._stream_dir is not None
        return result

    def _move_result(
        self,
        delta: np.ndarray,
        target: np.ndarray,
        start_pos: np.ndarray,
        start_quat: np.ndarray,
        steps: int,
        stopped: str | None,
    ) -> dict[str, Any]:
        final_pos, final_quat = self.measured()
        result: dict[str, Any] = {
            "ok": False,
            "requested_delta_xyz_base": delta.tolist(),
            "start_tcp_pose": np.concatenate([start_pos, start_quat]).tolist(),
            "final_tcp_pose": np.concatenate([final_pos, final_quat]).tolist(),
            "final_error_m": float(np.linalg.norm(target - final_pos)),
            "steps_used": steps,
            "states": None,
        }
        if stopped == "blocked":
            result["blocked"] = True
            result["note"] = self._blocked_note()
        # Show-Harness descend_travel + proprioception DESCEND_STALL_RATIO.
        if delta[2] < -1e-3 and stopped != "cancelled":
            travelled = float(start_pos[2] - final_pos[2])
            if travelled < self.limits.descent_stall_ratio * -delta[2]:
                self._reanchor()  # stop pressing
                result["descent_blocked"] = True
                result["descent_travelled_m"] = travelled
                result["note"] = (
                    f"descent blocked: moved {travelled:.4f} of {-delta[2]:.4f} m down; "
                    "the gripper is resting on something. The setpoint was re-anchored "
                    "at the measured pose"
                )
        result["ok"] = (
            stopped is None
            and not result.get("descent_blocked")
            and result["final_error_m"] <= self.limits.move_tolerance_m
        )
        if stopped == "cancelled":
            result["cancelled"] = True
        return result

    def rotate_delta(self, delta_rpy: Any) -> dict[str, Any]:
        """Rotate the TCP in place: target = R(xyz euler delta) * current (base frame)."""
        lim = self.limits
        rpy = np.asarray(delta_rpy, dtype=np.float64).reshape(-1)
        if rpy.shape != (3,) or not np.all(np.isfinite(rpy)):
            raise ValueError("delta_rpy must be 3 finite values")
        drot = quat_to_rotvec(quat_from_euler_xyz(rpy))
        angle = max(float(np.linalg.norm(drot)), float(np.max(np.abs(rpy))))
        if angle > lim.max_rotate_rad + 1e-9:
            raise ValueError(
                f"delta_rpy rotates {angle:.4f} rad; the limit is "
                f"{lim.max_rotate_rad} rad per call and per component, without "
                "wrap-around (limits.max_rotate_rad). Split the rotation; nothing was "
                "commanded"
            )
        return self._motion(lambda: self._rotate(rpy, drot))

    def _check_rotation(self, pos: np.ndarray, quat: np.ndarray, drot: np.ndarray):
        """Refuse a rotation that tilts the tool past ``max_tilt_rad`` or swings the
        flange (the other end of ``tcp_offset_m``) further out of the box."""
        lim = self.limits
        tilt0 = tool_tilt(quat)
        flange0 = self._tcp_to_flange(pos, quat)[:3]
        n = self._ticks(np.zeros(3), drot)
        for i in range(1, n + 1):
            q = quat_normalize(quat_mul(quat_from_rotvec(drot * i / n), quat))
            tilt = tool_tilt(q)
            if tilt > lim.max_tilt_rad + 1e-9 and tilt > tilt0 + 1e-9:
                raise ValueError(
                    f"the rotation tilts the tool {tilt:.3f} rad from pointing straight "
                    f"down; the limit is {lim.max_tilt_rad} rad (limits.max_tilt_rad); "
                    "nothing was commanded"
                )
            self.check_target(
                flange0,
                self._tcp_to_flange(pos, q)[:3],
                strict=False,
                what="the rotation's flange",
            )

    def _rotate(self, rpy: np.ndarray, drot: np.ndarray) -> dict[str, Any]:
        self._prepare()
        start_pos, start_quat = self.measured()
        origin_p, origin_q = self.target_pos.copy(), self.target_quat.copy()
        self._check_rotation(origin_p, origin_q, drot)
        target_q = quat_normalize(quat_mul(quat_from_rotvec(drot), origin_q))
        steps, stopped = 0, None
        if not self.end_stream():  # a rotation never chains: start from rest
            stopped = "cancelled"
        if stopped is None:
            steps, stopped = self._ramp(origin_p, origin_q, np.zeros(3), drot)
        if stopped is None and not self._settle():
            stopped = "cancelled"
        final_pos, final_quat = self.measured()
        error = quat_angle(target_q, final_quat)
        result: dict[str, Any] = {
            "ok": stopped is None and error <= self.limits.rotate_tolerance_rad,
            "requested_delta_rpy_base": rpy.tolist(),
            "start_tcp_pose": np.concatenate([start_pos, start_quat]).tolist(),
            "final_tcp_pose": np.concatenate([final_pos, final_quat]).tolist(),
            "final_error_rad": error,
            "steps_used": steps,
            "states": None,
        }
        if stopped == "blocked":
            result["blocked"] = True
            result["note"] = self._blocked_note()
        if stopped == "cancelled":
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
        # Bring a chained motion to rest before the fingers act (Show-Harness).
        if not self.end_stream():
            return {
                "ok": False,
                "cancelled": True,
                "target_gripper_open": bool(open),
                "steps_used": 0,
                "gripper_width_m": self.width(),
            }
        self.robot.control_gripper(not open)  # Show-Harness: True = close
        self.commanded_open = bool(open)
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
            self.commanded_open = True
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
        if self.limits.begin_joints is None:
            raise ValueError(
                "reset.begin_joints is not set: jog the arm to a safe begin pose, read "
                "the joints (--read-pose) and write them to the config"
            )
        return self._motion(self._reset)

    def _reset(self) -> dict[str, Any]:
        lim = self.limits
        info: dict[str, Any] = {"begin_joints": list(lim.begin_joints)}
        grip = self.set_gripper(open=True)
        info["gripper"] = grip
        if grip.get("cancelled"):
            return {"ok": False, "cancelled": True, "info": info}
        self._prepare()
        room = float(lim.workspace_max[2]) - float(self.target_pos[2])
        lift = max(0.0, min(lim.reset_lift_m, room))
        if lift > 1e-4:
            _, stopped = self._ramp(
                self.target_pos.copy(),
                self.target_quat.copy(),
                np.array([0.0, 0.0, lift]),
                np.zeros(3),
            )
            info["lifted_m"] = lift
            if stopped == "blocked":
                info["note"] = self._blocked_note()
            if stopped:
                return {"ok": False, stopped: True, "info": info}
        goal = np.asarray(lim.begin_joints, dtype=np.float64)
        q0 = np.asarray(self.robot.get_joint_positions(), dtype=np.float64).reshape(-1)
        if q0.shape != (7,) or not np.all(np.isfinite(q0)):
            raise RuntimeError(
                f"robot returned invalid joint positions {q0.tolist()}; reset refused"
            )
        # Never faster than MAX_JOINT_SPEED_RAD_S on any joint.
        duration = max(
            float(lim.begin_time_s),
            PEAK_TO_MEAN * float(np.max(np.abs(goal - q0))) / MAX_JOINT_SPEED_RAD_S,
        )
        info["method"] = lim.reset_method
        info["duration_s"] = duration
        try:
            if lim.reset_method == "move_to_joint_positions":
                # Blocking on the NUC: not interruptible by ``stop``.
                self.robot.move_to_joint_positions(goal, duration)
            else:
                self.robot.start_joint_impedance(None, None)
                n = max(1, math.ceil(duration / lim.tick_s))
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
        except Exception:
            self.start_impedance()  # never leave joint impedance running
            raise
        self.start_impedance()
        joints = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
        info["joint_error_rad"] = float(np.max(np.abs(joints - goal)))
        ok = info["joint_error_rad"] <= 0.05
        if np.any(self._violation(self.target_pos) > 1e-6):
            info["begin_pose_outside_workspace"] = True
            info["note"] = (
                f"the begin pose puts the TCP at {np.round(self.target_pos, 4).tolist()}, "
                "outside the workspace box; fix reset.begin_joints or the box"
            )
            ok = False
        return {"ok": ok, "info": info}

    # -- state -------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """RLinf-shaped ``raw_base_state`` (tcp_pose xyzw, gripper_position, ...).

        The gripper keys are measured (``gripper_state``): ``gripper_position`` [width],
        ``gripper_open`` / ``gripper_closed``, ``gripper_grasped``, ``gripper_moving``;
        the last command is ``gripper_commanded_open`` (None before the first).
        """
        pos, quat = self.measured()
        grip = self.gripper_state()
        out: dict[str, Any] = {
            "tcp_pose": np.concatenate([pos, quat]).tolist(),
            "gripper_position": [grip["width"]],
            "gripper_open": not grip["closed"],
            "gripper_closed": grip["closed"],
            "gripper_grasped": grip["grasped"],
            "gripper_moving": grip["moving"],
            "gripper_commanded_open": self.commanded_open,
            "arm_joint_position": np.asarray(
                self.robot.get_joint_positions(), dtype=float
            ).tolist(),
        }
        if self.target_pos is not None:
            out["target_tcp_pose"] = self.target_pose().tolist()
        return out
