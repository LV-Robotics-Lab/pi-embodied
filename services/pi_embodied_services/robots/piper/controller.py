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
# Modified by pi-embodied: the Piper path of interpreters/real_atomic_controller.py and
# interpreters/piper_atomic_controller.py reduced to one guarded step
# (translation + yaw + gripper) per call: no token vocabulary (the units module maps
# units to deltas), no variable-step / rotation plugins, and every wait polls a stop
# callback so the RPC ``stop`` halts motion between waypoints. The smooth plugin
# (plugins/smooth/plugin.py SmoothPlugin.plan, on in configs/robot_piper.yaml) shapes the
# joint stream: its min-jerk / cruise profile evaluated at the stream rate, stretched
# when its peak would pass the speed caps, and chaining on ``continuous``. Added a
# per-call step/yaw refusal and an optional base-frame workspace box next to the
# Z floor.

"""One guarded Cartesian step on the AgileX Piper.

The controller keeps a Cartesian setpoint (position + extrinsic-xyz euler) and moves
it by one bounded delta per call:

1. Refuse a translation longer than ``max_step_m`` or a yaw beyond ``max_yaw_rad``.
2. Rotate the delta by the gripper heading when ``frame == "heading"`` (Show-Harness
   ``motion_frame: wrist``: +X along the gripper's horizontal heading, +Y its left).
3. Clamp the new setpoint into the workspace box and above the Z floor (a clamp is
   reported in ``notes``, not raised: the arm moves as far as it may).
4. Drive there. ``joint_stream`` (default): the straight line is solved to joint
   waypoints (bounded-orientation IK, ``kinematics.ik_bounded``) and streamed as MOVE J
   targets at ``joint_stream_hz``; with ``smooth`` (default) along Show-Harness's
   min-jerk profile over ``smooth_substeps * smooth_dt_s`` (1.0 s), longer when its peak
   would pass ``smooth_max_speed_mps`` / ``yaw_speed_radps``, else at the constant
   ``speed_mps``. Chaining (``smooth_blend``): a pure translation sent with
   ``continuous=True`` (another move in about the same direction follows at once) ends at
   cruise speed without the settle re-commands, and the next pure translation, within
   ``smooth_chain_window_s`` and ``cos >= 0.9`` of it, starts at that speed; anything else
   (another direction, a yaw, the gripper, a reset, a stop, a late move) first settles
   the stream at rest. The first unreachable waypoint
   stops the move there ("reach clamp"). ``endpose``: firmware MOVE P re-commanded
   ``settle_steps`` times. The joint backend is used only after the vendored FK
   reproduces the arm's own pose feedback within 1 cm.
   Every move not chained onto a flowing stream starts from the MEASURED pose and
   joints (IK targets shifted by the feedback-minus-FK offset), so a move that fell
   short never leaves a stale setpoint behind; a reach fallback ending more than
   ``FALLBACK_TOLERANCE_M`` short fails the step; a waypoint that would move the tool
   more than its planned share (plus ``JUMP_MARGIN_M``) in one tick is refused, and one
   that would turn the wrist more than its share (plus ``JUMP_MARGIN_RAD``) ends the
   move there.
5. Divergence guard: a measured pose far from what was commanded (the node drops
   commands silently when not enabled / not in mode 1) re-syncs the setpoint to the
   measured pose instead of letting it run away.
6. Gripper: command the width, wait until it settles, flag a command that did not
   move the fingers, and reopen after a close that caught nothing.

``robot`` is duck-typed (``ros_io.PiperRosArm``): ``get_ee_pose``,
``get_joint_positions``, ``get_gripper_width``, ``command_pose``, ``stream_joints``,
``set_gripper_width``, ``note_commanded_pose``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from pi_embodied_services.robots.franka_polymetis.control import smooth_fractions
from pi_embodied_services.robots.piper.kinematics import (
    JOINT_LIMITS_RAD,
    select_dh_variant,
)
from pi_embodied_services.utils.logging import get_logger

logger = get_logger("piper_controller")

#: Tool axis (EEF frame) whose horizontal projection is the gripper heading; +Z was
#: verified on the Piper by Show-Harness.
TOOL_AXIS = np.array([0.0, 0.0, 1.0])
#: A width change larger than this (m) means the fingers moved.
GRIPPER_MOTION_EPS_M = 0.005
GRIPPER_STABLE_EPS_M = 0.002
GRIPPER_SUSTAIN_S = 0.3
#: Show-Harness chains a move only onto a stream flowing in (nearly) the same direction.
CHAIN_MIN_COS = 0.9
#: A reach fallback (one MOVE P) that ends further than this (m) from its target fails
#: the step (the arm is then halted on two arms).
FALLBACK_TOLERANCE_M = 0.005
#: Per-waypoint jump caps: a streamed waypoint may move the tool at most its planned
#: share of the move plus this much (m, rad) past the previous one (the first: past
#: the measured joints). A position jump is a fault; a wrist turn ends the move there.
JUMP_MARGIN_M = 0.003
JUMP_MARGIN_RAD = math.radians(2.0)
#: A setpoint this far (m) from the measured pose at the start of a move is reported.
RESYNC_NOTE_M = 0.005


def _finite(v: Any) -> bool:
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(float(v))
    )


def profile_peak(v0: float = 0.0, v1: float = 0.0) -> float:
    """Peak speed of the smooth profile, in move lengths per move duration (1.875 at rest)."""
    fr = smooth_fractions(400, v0, v1)
    return float(np.max(np.diff([0.0, *fr]))) * 400


@dataclass
class PiperLimits:
    """Safety limits and motion tuning; every field is a key of the robot YAML."""

    #: Base-frame Z (m) the setpoint never goes below. Required when enable_z_floor.
    z_floor_m: float | None = None
    enable_z_floor: bool = True
    #: Optional base-frame box [x, y, z] (m); the setpoint is clamped into it.
    workspace_min: list[float] | None = None
    workspace_max: list[float] | None = None
    #: Largest translation (m) and |yaw| (rad) one call may command; larger is refused.
    max_step_m: float = 0.05
    max_yaw_rad: float = 0.2
    #: Largest |yaw| (rad) the gripper may be turned away from its heading at the last
    #: reset (Show-Harness rotation plugin: 150 deg); a turn beyond it is refused.
    max_total_yaw_rad: float = math.radians(150.0)
    #: Cartesian speed of the joint stream (m/s, rad/s).
    speed_mps: float = 0.05
    yaw_speed_radps: float = 0.5
    motion_backend: str = "joint_stream"
    joint_stream_hz: float = 50.0
    ori_flex_rad: float = math.radians(15.0)
    settle_steps: int = 6
    settle_dt_s: float = 0.1
    divergence_resync_m: float | None = 0.06
    #: None -> 1.5 x max_yaw_rad; <= 0 disables.
    divergence_resync_rad: float | None = None
    #: Gripper widths (m) and settle timing (s).
    open_width_m: float = 0.07
    empty_width_m: float | None = 0.005
    close_threshold_m: float = 0.06
    grasp_open_width_m: float = 0.055
    gripper_settle_s: float = 1.5
    gripper_min_settle_s: float = 0.5
    gripper_poll_dt_s: float = 0.05
    #: Joint-space reset: duration (s) and convergence tolerance (rad).
    reset_time_s: float = 4.0
    reset_tolerance_rad: float = 0.05
    #: Show-Harness smooth plugin (the ``smooth:`` section): min-jerk moves of
    #: substeps x dt_s, at most ``smooth_max_speed_mps`` at their peak; off = constant
    #: ``speed_mps``. Chaining: blend, cruise (boundary speed in move lengths per move
    #: duration) and the window in which the next move must start.
    smooth: bool = True
    smooth_substeps: int = 20
    smooth_dt_s: float = 0.05
    smooth_max_speed_mps: float = 0.1
    smooth_blend: bool = True
    smooth_cruise: float = 1.0
    smooth_chain_window_s: float = 1.0

    def validate(self) -> None:
        """Refuse any safety-relevant value that is missing, not finite or out of range.

        NaN compares false against every bound, so each check is written to fail on it.
        """
        if not isinstance(self.enable_z_floor, bool):
            raise ValueError("enable_z_floor must be true or false")
        if self.z_floor_m is not None and not (
            _finite(self.z_floor_m) and -1.0 <= self.z_floor_m <= 1.0
        ):
            raise ValueError(
                f"z_floor_m must be a finite base-frame z in [-1, 1] m, got {self.z_floor_m!r}"
            )
        if self.enable_z_floor and self.z_floor_m is None:
            raise ValueError(
                "z_floor_m is not calibrated: rest the gripper on the tabletop, read "
                "the EEF z from /puppet/end_pose_euler_<arm>, and set "
                "calibration.z_floor_m (or set limits.enable_z_floor: false)"
            )
        if (self.workspace_min is None) != (self.workspace_max is None):
            raise ValueError("set both workspace_min and workspace_max, or neither")
        if self.workspace_min is not None:
            box = [self.workspace_min, self.workspace_max]
            if not all(
                isinstance(b, (list, tuple)) and len(b) == 3 and all(map(_finite, b))
                for b in box
            ):
                raise ValueError("workspace_min/max must be [x,y,z] of finite numbers")
            lo = np.asarray(self.workspace_min, float)
            hi = np.asarray(self.workspace_max, float)
            if not np.all(lo < hi):
                raise ValueError("workspace_min/max must be [x,y,z] with min < max")
        if self.motion_backend not in ("joint_stream", "endpose"):
            raise ValueError("motion_backend must be 'joint_stream' or 'endpose'")
        for name in ("smooth", "smooth_blend"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false")
        # (lo, hi] ranges of every number; None where the key may be unset.
        bounds = {
            "max_step_m": (0.0, 0.1),
            "max_yaw_rad": (0.0, math.pi / 2),
            "max_total_yaw_rad": (0.0, math.pi - 1e-6),
            "speed_mps": (0.0, 0.3),
            "yaw_speed_radps": (0.0, 2.0),
            "joint_stream_hz": (1.0, 500.0),
            "ori_flex_rad": (-1e-12, math.radians(45.0)),
            "settle_steps": (0.0, 100.0),
            "settle_dt_s": (-1e-12, 1.0),
            "divergence_resync_m": (0.0, 0.2),
            "divergence_resync_rad": (-math.inf, math.pi),
            "open_width_m": (0.0, 0.1),
            "empty_width_m": (-1e-12, 0.1),
            "close_threshold_m": (0.0, 0.1),
            "grasp_open_width_m": (0.0, 0.1),
            "gripper_settle_s": (-1e-12, 10.0),
            "gripper_min_settle_s": (-1e-12, 10.0),
            "gripper_poll_dt_s": (0.0, 1.0),
            "reset_time_s": (0.05, 60.0),
            "reset_tolerance_rad": (0.0, 0.2),
            "smooth_substeps": (0.0, 1000.0),
            "smooth_dt_s": (1e-4, 1.0),
            "smooth_max_speed_mps": (1e-3, 0.3),
            "smooth_cruise": (-1e-12, 1.5),
            "smooth_chain_window_s": (-1e-12, 10.0),
        }
        optional = {"divergence_resync_m", "divergence_resync_rad", "empty_width_m"}
        for name, (lo, hi) in bounds.items():
            v = getattr(self, name)
            if v is None and name in optional:
                continue
            if not (_finite(v) and lo < v <= hi):
                rng = f"({lo:g}, {hi:g}]" if lo >= 0 else f"[0, {hi:g}]"
                if lo == -math.inf:
                    rng = f"<= {hi:g}"
                raise ValueError(f"{name} must be a finite number {rng}, got {v!r}")
        for name in ("settle_steps", "smooth_substeps"):
            if int(getattr(self, name)) != getattr(self, name):
                raise ValueError(f"{name} must be an integer")
        # The yaw budget is wrapped to (-pi, pi]: one call must not be able to jump
        # across the wrap (a 3 rad call would read as a small turn the other way).
        if self.max_yaw_rad > self.max_total_yaw_rad:
            raise ValueError("max_yaw_rad must not exceed max_total_yaw_rad")
        if self.max_yaw_rad + self.max_total_yaw_rad >= math.pi:
            raise ValueError(
                f"max_yaw_rad + max_total_yaw_rad must stay below pi (got "
                f"{self.max_yaw_rad:.3f} + {self.max_total_yaw_rad:.3f}): a larger "
                "per-call yaw could wrap past the accumulated-yaw cap"
            )


@dataclass
class StepReport:
    ok: bool = True
    cancelled: bool = False
    #: The move started at cruise speed, chained onto the previous one.
    chained: bool = False
    notes: list[str] = field(default_factory=list)


class Stopped(Exception):
    """Raised inside a drive when the stop callback turns true."""


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    return R.from_quat(np.asarray(quat, dtype=float)).as_euler("xyz")


def euler_to_quat(euler: np.ndarray) -> np.ndarray:
    return R.from_euler("xyz", np.asarray(euler, dtype=float)).as_quat()


class PiperController:
    """Guarded single steps on one Piper arm (see the module docstring)."""

    def __init__(
        self,
        robot: Any,
        limits: PiperLimits,
        stop_requested: Callable[[], bool] = lambda: False,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        limits.validate()
        self.robot = robot
        self.limits = limits
        self.stop_requested = stop_requested
        self.sleep = sleep
        self.clock = clock
        # Chained motion (smooth + blend): base-frame direction and deadline of a
        # translation that ended at cruise speed without settling (None = at rest).
        self._stream_dir: np.ndarray | None = None
        self._stream_until = 0.0
        self._target_pos: np.ndarray | None = None
        self._target_euler: np.ndarray | None = None
        self.gripper_closed: bool | None = None
        self._kin: Any = None
        #: Best-matching FK (even when too far off for the joint stream), for the
        #: reset-path floor/box check.
        self._fk: Any = None
        #: Base-z heading (extrinsic euler z) at the last reset; the yaw budget's origin.
        self._yaw_ref: float | None = None
        self._q_cmd: np.ndarray | None = None
        #: Feedback-minus-FK offset (position, rotation) measured when a move starts
        #: from rest: IK targets are shifted by it so the first waypoint is the arm's
        #: measured joints, not the setpoint's FK (which differs by up to 1 cm).
        self._off_pos = np.zeros(3)
        self._off_rot = R.identity()
        #: Robot commands sent (stream / MOVE P / gripper): a refusal that raises with
        #: this unchanged sent nothing.
        self.commands = 0
        self.backend = "endpose"
        rad = limits.divergence_resync_rad
        self._resync_rad = (
            1.5 * limits.max_yaw_rad if rad is None else (rad if rad > 0 else None)
        )

    # -- state ------------------------------------------------------------

    def sync(self) -> list[str]:
        """Reset the setpoint to the measured pose and (re)validate the joint backend.

        All-or-nothing: if any feedback read fails the setpoint stays invalid, so the
        next call re-syncs (and raises again while feedback is stale) instead of moving.
        """
        self._target_pos = self._target_euler = self._q_cmd = None
        self._stream_dir = None
        notes: list[str] = []
        pose = np.asarray(self.robot.get_ee_pose(), dtype=float)
        width = float(self.robot.get_gripper_width())
        backend, kin, q_cmd = "endpose", None, None
        q = np.asarray(self.robot.get_joint_positions(), dtype=float)[:6]
        fk, err = select_dh_variant(q, pose)
        if self.limits.motion_backend == "joint_stream":
            kin = fk
            if err > 0.01:
                kin = None
                notes.append(
                    f"joint_stream disabled: FK is {err * 1000:.1f} mm from the arm's "
                    "own pose feedback; using endpose (MOVE P)"
                )
                logger.warning(notes[-1])
            else:
                backend, q_cmd = "joint_stream", q
                margins = np.degrees(
                    np.minimum(q - JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1] - q)
                )
                tight = [
                    f"j{j + 1} {margins[j]:.1f} deg" for j in range(6) if margins[j] < 8
                ]
                if tight:
                    notes.append(f"start pose near a joint limit ({', '.join(tight)})")
        self.gripper_closed = width < self.limits.close_threshold_m
        self.backend, self._kin, self._q_cmd, self._fk = backend, kin, q_cmd, fk
        if kin is not None:
            self._measure_offset(pose, q)
        self._target_euler = quat_to_euler(pose[3:])
        self._target_pos = pose[:3].copy()
        if self._yaw_ref is None:
            self._yaw_ref = float(self._target_euler[2])
        return notes

    def _measure_offset(self, pose: np.ndarray, q: np.ndarray) -> None:
        fk_pos, fk_rot = self._kin.fk(q)
        self._off_pos = pose[:3] - np.asarray(fk_pos, dtype=float)
        self._off_rot = R.from_quat(pose[3:]) * R.from_matrix(fk_rot).inv()

    def _fk_pose(self, q: np.ndarray) -> tuple[np.ndarray, R]:
        """Where joints ``q`` put the tool, in the pose-feedback frame (FK + offset)."""
        pos, rot = self._kin.fk(q)
        return np.asarray(pos, float) + self._off_pos, self._off_rot * R.from_matrix(
            rot
        )

    def _restart(self, report: StepReport | None, note: bool = True) -> None:
        """Start the next move from the MEASURED pose (and joints), not the setpoint.

        A setpoint left behind by a move that fell short (a reach fallback, a clamp,
        dropped commands) would otherwise make the next move's first waypoint jump.
        """
        pose = np.asarray(self.robot.get_ee_pose(), dtype=float)
        if self.backend == "joint_stream":
            q = np.asarray(self.robot.get_joint_positions(), dtype=float)[:6]
            self._measure_offset(pose, q)
            self._q_cmd = q
        gap = float(np.linalg.norm(self._target_pos - pose[:3]))
        if note and report is not None and gap > RESYNC_NOTE_M:
            report.notes.append(
                f"setpoint re-synced to the measured pose ({gap * 1000:.1f} mm off)"
            )
        self._target_pos = pose[:3].copy()
        self._target_euler = quat_to_euler(pose[3:])
        # A later gripper command re-sends this pose (ros_io): never an unreached one.
        self.robot.note_commanded_pose(pose)

    # -- robot commands (counted) -------------------------------------------

    def _stream(self, q: np.ndarray) -> None:
        self.commands += 1
        self.robot.stream_joints(q)

    def _command_pose(self, pose: np.ndarray) -> None:
        self.commands += 1
        self.robot.command_pose(pose)

    def _set_width(self, width: float) -> None:
        self.commands += 1
        self.robot.set_gripper_width(width)

    def _invalidate(self) -> None:
        """Forget the setpoint after a motion failed or was aborted.

        The next call re-syncs from measured state, so nothing streams toward a target
        (or from joints) captured before the failure.
        """
        self._target_pos = self._target_euler = self._q_cmd = None
        self._stream_dir = None
        try:
            self.robot.note_commanded_pose(None)
        except Exception as exc:
            logger.warning("forgetting the commanded pose failed: %s", exc)

    def _ensure_synced(self) -> None:
        if self._target_pos is None:
            self.sync()

    @property
    def target_pose(self) -> np.ndarray:
        self._ensure_synced()
        return np.concatenate([self._target_pos, euler_to_quat(self._target_euler)])

    def yaw_from_reset(self, extra: float = 0.0) -> float:
        """How far (rad) the setpoint (turned ``extra`` more) is yawed from the reset heading."""
        self._ensure_synced()
        d = float(self._target_euler[2]) + extra - float(self._yaw_ref or 0.0)
        return math.remainder(d, 2 * math.pi)

    def heading_yaw(self) -> float | None:
        """Yaw of the gripper's horizontal heading (from the setpoint), or None."""
        self._ensure_synced()
        tool = R.from_euler("xyz", self._target_euler).apply(TOOL_AXIS)
        if math.hypot(tool[0], tool[1]) < 0.1:
            return None
        return float(math.atan2(tool[1], tool[0]))

    def state(self) -> dict[str, Any]:
        pose = np.asarray(self.robot.get_ee_pose(), dtype=float)
        width = float(self.robot.get_gripper_width())
        lim = self.limits
        heading = self.heading_yaw()
        return {
            "eef_pose": pose.tolist(),
            "eef_pos": pose[:3].tolist(),
            "eef_euler_xyz": quat_to_euler(pose[3:]).tolist(),
            "joints": np.asarray(self.robot.get_joint_positions(), float)[:6].tolist(),
            "gripper_width_m": width,
            "gripper_closed": width < lim.close_threshold_m,
            "target_pose": self.target_pose.tolist(),
            "heading_yaw_rad": heading,
            "yaw_from_reset_rad": self.yaw_from_reset(),
            "height_above_floor_m": (
                float(pose[2] - lim.z_floor_m) if lim.z_floor_m is not None else None
            ),
            "z_floor_m": lim.z_floor_m if lim.enable_z_floor else None,
            "workspace_min": lim.workspace_min,
            "workspace_max": lim.workspace_max,
            "motion_backend": self.backend,
            "force_torque": None,
        }

    # -- one step ---------------------------------------------------------

    def step(
        self,
        delta_xyz: Any = (0.0, 0.0, 0.0),
        yaw: float = 0.0,
        gripper: str | None = None,
        frame: str = "base",
        reopen_empty: bool = True,
        continuous: bool = False,
    ) -> dict[str, Any]:
        """Translate by ``delta_xyz`` (m), yaw by ``yaw`` (rad), then open/close.

        ``reopen_empty=False`` leaves a close that caught nothing closed (for a caller
        that runs its own recovery, e.g. the units layer's recovery plugin).
        ``continuous``: another translation in about the same direction follows at once
        (the rest of a repeated unit); with ``smooth`` and ``smooth_blend`` a pure
        translation then ends at cruise speed and the next one chains onto it.
        """
        delta = np.asarray(delta_xyz, dtype=float).reshape(-1)
        yaw = float(yaw)
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            raise ValueError("delta_xyz must be 3 finite numbers")
        if not math.isfinite(yaw):
            raise ValueError("yaw must be finite")
        if gripper not in (None, "open", "close"):
            raise ValueError("gripper must be 'open', 'close' or null")
        if frame not in ("base", "heading"):
            raise ValueError("frame must be 'base' or 'heading'")
        lim = self.limits
        norm = float(np.linalg.norm(delta))
        if norm > lim.max_step_m + 1e-9:
            raise ValueError(
                f"delta_xyz moves {norm:.4f} m; the limit is {lim.max_step_m} m per call"
            )
        if abs(yaw) > lim.max_yaw_rad + 1e-9:
            raise ValueError(
                f"yaw {yaw:.4f} rad exceeds the limit of {lim.max_yaw_rad} rad per call"
            )
        self._ensure_synced()
        report = StepReport()
        # Every move starts from the measured pose, except one chaining onto a flowing
        # stream (the arm lags the stream's setpoint by design there).
        chain = (
            yaw == 0
            and gripper is None
            and norm > 0
            and lim.smooth
            and lim.smooth_blend
            and self.backend == "joint_stream"
            and self._stream_live()
        )
        if chain:
            d = self._base_delta(delta, frame, None)
            chain = (
                float(np.dot(d / np.linalg.norm(d), self._stream_dir)) >= CHAIN_MIN_COS
            )
        live = self._stream_live()
        if not chain and not live:
            self._restart(report)
        if yaw != 0:
            turned = self.yaw_from_reset(yaw)
            if abs(turned) > lim.max_total_yaw_rad + 1e-9:
                raise ValueError(
                    f"yaw {yaw:.4f} rad would turn the gripper "
                    f"{math.degrees(turned):.0f} deg from its heading at the last reset; the "
                    f"limit is {math.degrees(lim.max_total_yaw_rad):.0f} deg "
                    "(limits.max_total_yaw_rad). Turn back first; nothing was commanded"
                )
        pre = np.asarray(self.robot.get_ee_pose(), dtype=float)
        base_delta = delta
        try:
            if not chain and live:
                # Settle the flowing stream at rest, then start from where it ended.
                self.end_stream()
                self._restart(report)
            elif not chain:
                self._stream_dir = None
            base_delta = self._base_delta(delta, frame, report) if norm > 0 else delta
            if norm > 0 or yaw != 0:
                pure = yaw == 0 and gripper is None
                self._move(base_delta, yaw, report, bool(continuous) and pure, chain)
            if gripper is not None:
                self.end_stream()
                self._gripper(gripper == "close", report, reopen_empty)
        except Stopped:
            report.cancelled = True
            report.ok = False
            report.notes.append("stopped")
            self._hold_after_stop()
        except BaseException:
            self._invalidate()
            raise
        post = np.asarray(self.robot.get_ee_pose(), dtype=float)
        out: dict[str, Any] = {
            "ok": report.ok,
            "requested_delta_xyz": delta.tolist(),
            "requested_yaw_rad": yaw,
            "frame": frame,
            "delta_xyz_base": base_delta.tolist(),
            "gripper_command": gripper,
            "pre_pose": pre.tolist(),
            "post_pose": post.tolist(),
            "target_pose": self.target_pose.tolist(),
            "moved_m": float(np.linalg.norm(post[:3] - pre[:3])),
            "gripper_width_m": float(self.robot.get_gripper_width()),
            "gripper_closed": self.gripper_closed,
            "notes": report.notes,
        }
        if self.limits.smooth and self.backend == "joint_stream":
            out["chained"] = report.chained
            out["flowing"] = self._stream_dir is not None
        if report.cancelled:
            out["cancelled"] = True
        return out

    def _base_delta(
        self, delta: np.ndarray, frame: str, report: StepReport | None
    ) -> np.ndarray:
        """``delta`` in the base frame (``heading``: rotated by the gripper heading)."""
        if frame != "heading":
            return delta
        heading = self.heading_yaw()
        if heading is None:
            if report is not None:
                report.notes.append("no gripper heading (tool vertical): base frame")
            return delta
        c, s = math.cos(heading), math.sin(heading)
        return np.array(
            [c * delta[0] - s * delta[1], s * delta[0] + c * delta[1], delta[2]]
        )

    # -- chained motion ---------------------------------------------------

    def _stream_live(self) -> bool:
        """Whether a chained translation may still be flowing (within the window)."""
        return self._stream_dir is not None and self.clock() <= self._stream_until

    def end_stream(self) -> None:
        """Show-Harness ``end_stream``: bring a chained motion to rest (settle at the last
        joint target). A no-op at rest or once the chain window has passed."""
        live = self._stream_live()
        self._stream_dir = None
        if live and self._q_cmd is not None:
            self._joint_settle()

    def plan(
        self, dist_m: float, angle_rad: float, v0: float = 0.0, v1: float = 0.0
    ) -> tuple[list[float], float]:
        """(fractions, delay) of one joint-stream move, at ``joint_stream_hz``.

        ``smooth``: Show-Harness's min-jerk / cruise profile (``SmoothPlugin.plan``) over
        ``smooth_substeps * smooth_dt_s``, evaluated at the stream rate (their joint
        stream resamples it there too), and stretched until its peak, sized on the
        rest-to-rest profile so equal moves of a chain get equal durations, stays within
        ``smooth_max_speed_mps`` and ``yaw_speed_radps``. Otherwise a constant-rate line
        at ``speed_mps`` (Show-Harness's smooth-off fallback).
        """
        lim = self.limits
        if lim.smooth:
            peak = max(profile_peak(), profile_peak(v0, v1))
            duration = max(
                lim.smooth_substeps * lim.smooth_dt_s,
                peak * dist_m / lim.smooth_max_speed_mps,
                peak * angle_rad / lim.yaw_speed_radps,
            )
        else:
            duration = max(
                dist_m / lim.speed_mps, angle_rad / lim.yaw_speed_radps, 0.15
            )
        n = max(2, int(round(lim.joint_stream_hz * duration)))
        if lim.smooth:
            return smooth_fractions(n, v0, v1), duration / n
        return [i / n for i in range(1, n + 1)], duration / n

    def _check_stop(self) -> None:
        if self.stop_requested():
            raise Stopped()

    def _wait(self, seconds: float) -> None:
        self._check_stop()
        if seconds > 0:
            self.sleep(seconds)

    def violation(self, pos: np.ndarray) -> np.ndarray:
        """Per-axis distance (m) of ``pos`` outside the workspace box / above-floor space."""
        lim = self.limits
        p = np.asarray(pos, dtype=float)
        out = np.zeros(3)
        if lim.workspace_min is not None:
            lo = np.asarray(lim.workspace_min, float)
            hi = np.asarray(lim.workspace_max, float)
            out = np.maximum(lo - p, 0.0) + np.maximum(p - hi, 0.0)
        if lim.enable_z_floor and lim.z_floor_m is not None:
            out[2] = max(out[2], lim.z_floor_m - p[2])
        return out

    def check_joint_path(self, start: np.ndarray, goal: np.ndarray, steps: int) -> None:
        """Refuse a joint-space move whose EEF path leaves the floor/box.

        The reset streams a straight joint-space line, which can dip below the Z floor
        or swing out of the box between two poses that are inside it. Every waypoint's
        FK position (shifted by the FK-vs-feedback offset measured at the start) must
        not be further outside than the start is (a reset may climb out of a violation,
        never deepen one), and the goal must be inside.
        """
        fk = self._fk
        if fk is None:
            raise ValueError(
                "no forward kinematics to check the joint path; sync first"
            )
        measured = np.asarray(self.robot.get_ee_pose(), dtype=float)[:3]
        offset = measured - np.asarray(fk.fk(start)[0], dtype=float)
        allowed = self.violation(measured) + 1e-3
        for i in range(1, steps + 1):
            q = start + (goal - start) * (i / steps)
            p = np.asarray(fk.fk(q)[0], dtype=float) + offset
            v = self.violation(p)
            bad = (v > allowed) | ((v > 1e-3) if i == steps else False)
            if np.any(bad):
                axes = "".join("xyz"[k] for k in range(3) if bad[k])
                where = "the goal" if i == steps else f"{i * 100 // steps}% of the way"
                raise ValueError(
                    f"joint move refused: at {where} the gripper would be at "
                    f"{np.round(p, 3).tolist()} m, {float(v.max()):.3f} m outside the Z floor / "
                    f"workspace box ({axes}); nothing was commanded"
                )

    def clamp_target(self, pos: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Clamp a setpoint into the workspace box and above the Z floor."""
        lim = self.limits
        out = np.asarray(pos, dtype=float).copy()
        notes: list[str] = []
        if lim.workspace_min is not None:
            lo = np.asarray(lim.workspace_min, float)
            hi = np.asarray(lim.workspace_max, float)
            clipped = np.clip(out, lo, hi)
            if not np.allclose(clipped, out):
                axes = "".join("xyz"[i] for i in range(3) if clipped[i] != out[i])
                notes.append(f"workspace: clamped {axes} to the box")
            out = clipped
        if lim.enable_z_floor and lim.z_floor_m is not None and out[2] < lim.z_floor_m:
            notes.append(
                f"z-floor: blocked {lim.z_floor_m - out[2]:.4f} m of descent "
                f"(floor {lim.z_floor_m:.4f} m)"
            )
            out[2] = lim.z_floor_m
        return out, notes

    def _move(
        self,
        delta: np.ndarray,
        yaw: float,
        report: StepReport,
        continuous: bool = False,
        chain: bool = False,
    ) -> None:
        start_pos = self._target_pos.copy()
        start_euler = self._target_euler.copy()
        target, notes = self.clamp_target(start_pos + delta)
        report.notes.extend(notes)
        self._target_pos = target
        self._target_euler = start_euler.copy()
        self._target_euler[2] += yaw
        if self.backend == "joint_stream":
            self._drive_joint_stream(start_pos, start_euler, report, continuous, chain)
        else:
            self._drive_endpose()
        self._guard_divergence(report)

    # -- backends ---------------------------------------------------------

    def _drive_endpose(self) -> None:
        pose = self.target_pose
        for _ in range(max(1, self.limits.settle_steps)):
            self._check_stop()
            self._command_pose(pose)
            self._wait(self.limits.settle_dt_s)

    def _joint_settle(self) -> None:
        for _ in range(max(1, self.limits.settle_steps)):
            self._check_stop()
            self._stream(self._q_cmd)
            self._wait(self.limits.settle_dt_s)

    def _drive_joint_stream(
        self,
        start_pos: np.ndarray,
        start_euler: np.ndarray,
        report: StepReport,
        continuous: bool = False,
        chain: bool = False,
    ) -> None:
        lim = self.limits
        dpos = self._target_pos - start_pos
        deuler = self._target_euler - start_euler
        dist = float(np.linalg.norm(dpos))
        angle = float(np.abs(deuler).max())
        # Chaining (Show-Harness piper_atomic_controller._drive_to_target): only a pure
        # translation continues a live stream in about its direction (decided in step).
        blend = lim.smooth and lim.smooth_blend
        pure = dist > 1e-9 and angle <= 1e-9
        unit = dpos / dist if dist > 1e-9 else None
        v0 = lim.smooth_cruise if chain else 0.0
        self._stream_dir = None
        report.chained = v0 > 0.0
        v1 = lim.smooth_cruise if blend and pure and continuous else 0.0
        fractions, dt = self.plan(dist, angle, v0, v1)
        n = len(fractions)
        # From rest q is the measured joints (step re-synced), so the first waypoint's
        # jump is measured against where the arm is.
        q = self._q_cmd
        prev_pos, prev_rot = self._kin.fk(q)
        off_inv = self._off_rot.inv()
        for i in range(1, n + 1):
            prev = fractions[i - 2] if i > 1 else 0.0
            try:
                self._check_stop()
            except Stopped:
                self._clamp_at(q, prev, "stopped", report)
                raise
            frac = fractions[i - 1]
            rot = off_inv * R.from_euler("xyz", start_euler + deuler * frac)
            sol, _dev = self._kin.ik_bounded(
                start_pos + dpos * frac - self._off_pos,
                rot.as_matrix(),
                q_seed=q,
                max_ori_dev_rad=lim.ori_flex_rad,
            )
            if sol is None:
                if i == 1:
                    self._reach_fallback(report)
                else:
                    self._clamp_at(q, prev, "reach clamp", report)
                return
            pos, rot_m = self._kin.fk(sol)
            share = frac - prev
            jump = float(np.linalg.norm(np.asarray(pos) - prev_pos))
            if jump > dist * share + JUMP_MARGIN_M:
                raise RuntimeError(
                    f"waypoint {i}/{n} would move the gripper {jump * 1000:.1f} mm in one "
                    f"tick (planned {dist * share * 1000:.1f} mm); refused, nothing past "
                    "the previous waypoint was commanded"
                )
            turn = float(R.from_matrix(rot_m @ prev_rot.T).magnitude())
            if turn > angle * share + JUMP_MARGIN_RAD:
                # The bounded IK bends the wrist at the reach boundary: never in one tick.
                report.notes.append(
                    f"wrist clamp: the next waypoint would turn the wrist "
                    f"{math.degrees(turn):.0f} deg in one tick"
                )
                self._clamp_at(q, prev, "reach clamp", report)
                return
            q, prev_pos, prev_rot = sol, np.asarray(pos), rot_m
            self._stream(q)
            if dt > 0:
                self.sleep(dt)
        self._q_cmd = q
        self.robot.note_commanded_pose(self.target_pose)
        if v1 > 0.0:
            # Flow into the next move: no settle; remember what the stream rides.
            self._stream_dir = unit
            self._stream_until = self.clock() + lim.smooth_chain_window_s
        else:
            self._joint_settle()

    def _reach_fallback(self, report: StepReport) -> None:
        """IK failed at the first waypoint: one MOVE P attempt, then truth from feedback.

        The setpoint is re-synced to the measured pose whatever MOVE P did, so the next
        move starts where the arm is; ending further than FALLBACK_TOLERANCE_M from the
        target fails the step (on two arms the arm is then halted).
        """
        report.notes.append(
            "reach fallback: IK failed at the first waypoint (arm at its reach limit "
            "or bad seed) -> one EndPoseCtrl attempt"
        )
        goal = self._target_pos.copy()
        # A stop mid-MOVE P holds the measured pose, not the pre-move joints.
        self._q_cmd = None
        self._drive_endpose()
        self._restart(report, note=False)
        short = float(np.linalg.norm(goal - self._target_pos))
        if short > FALLBACK_TOLERANCE_M:
            report.ok = False
            report.notes.append(
                f"reach fallback: MOVE P ended {short * 1000:.1f} mm short of its target "
                f"(tolerance {FALLBACK_TOLERANCE_M * 1000:.0f} mm) -> setpoint re-synced "
                "to the measured pose"
            )
            logger.warning(report.notes[-1])

    def _clamp_at(
        self, q_last: np.ndarray, frac: float, why: str, report: StepReport
    ) -> None:
        """End a joint-stream move at the last streamed waypoint; truthful setpoint."""
        self._q_cmd = q_last
        pos, rot = self._fk_pose(q_last)
        self._target_pos = pos
        self._target_euler = rot.as_euler("xyz")
        self.robot.note_commanded_pose(np.concatenate([pos, rot.as_quat()]))
        if why != "stopped":
            report.notes.append(
                f"{why}: move stopped at {frac * 100:.0f}% (workspace boundary at the "
                f"{math.degrees(self.limits.ori_flex_rad):.0f} deg orientation budget)"
            )
            self._joint_settle()
            # Position and yaw both restart from the arm, not the unreached target.
            self._restart(report)

    def _hold_after_stop(self) -> None:
        """After a stop: hold where the arm is and restart the setpoint from there."""
        try:
            if self.backend == "joint_stream" and self._q_cmd is not None:
                self._stream(self._q_cmd)
            else:
                self._command_pose(self.robot.get_ee_pose())
        except Exception as exc:
            logger.warning("hold after stop failed: %s", exc)
        self.sync()

    def _guard_divergence(self, report: StepReport) -> None:
        measured = np.asarray(self.robot.get_ee_pose(), dtype=float)
        if self.backend == "joint_stream" and self._q_cmd is not None:
            exp_pos, exp_rot = self._fk_pose(self._q_cmd)
            exp_quat = exp_rot.as_quat()
        else:
            exp_pos, exp_quat = self._target_pos, euler_to_quat(self._target_euler)
        pos_gap = float(np.linalg.norm(np.asarray(exp_pos) - measured[:3]))
        ori_gap = float(
            (R.from_quat(exp_quat) * R.from_quat(measured[3:]).inv()).magnitude()
        )
        lim = self.limits
        if (
            lim.divergence_resync_m is not None and pos_gap > lim.divergence_resync_m
        ) or (self._resync_rad is not None and ori_gap > self._resync_rad):
            self._target_pos = measured[:3].copy()
            self._target_euler = quat_to_euler(measured[3:])
            self.robot.note_commanded_pose(None)
            if self.backend == "joint_stream":
                self._q_cmd = np.asarray(self.robot.get_joint_positions(), float)[:6]
            report.ok = False
            report.notes.append(
                f"divergence: measured pose {pos_gap:.3f} m / "
                f"{math.degrees(ori_gap):.0f} deg from the commanded pose -> setpoint "
                "re-synced; commands may be dropped (arm not enabled / node not in "
                "mode 1) or the target is unreachable"
            )
            logger.warning(report.notes[-1])

    # -- gripper ----------------------------------------------------------

    def _gripper(self, close: bool, report: StepReport, reopen_empty: bool) -> None:
        lim = self.limits
        if self.gripper_closed is not None and self.gripper_closed == close:
            report.notes.append(f"gripper already {'closed' if close else 'open'}")
            return
        pre = float(self.robot.get_gripper_width())
        self._set_width(0.0 if close else lim.open_width_m)
        self.gripper_closed = close
        width = self._await_gripper(lim.grasp_open_width_m if close else None)
        if abs(width - pre) <= GRIPPER_MOTION_EPS_M:
            report.ok = False
            report.notes.append(
                f"gripper {'close' if close else 'open'} did not move the fingers "
                f"({pre:.4f} -> {width:.4f} m): the command was likely dropped (arm not "
                "enabled / node not in mode 1)"
            )
        empty = lim.empty_width_m is not None and width <= lim.empty_width_m
        if close and empty and not reopen_empty:
            report.notes.append(f"empty grasp: width {width:.4f} m (left closed)")
        elif close and empty:
            self._set_width(lim.open_width_m)
            self.gripper_closed = False
            self._await_gripper(None)
            report.notes.append(
                f"empty grasp: width {width:.4f} m <= {lim.empty_width_m:.4f} m -> "
                "reopened"
            )

    def _await_gripper(self, decisive_max_m: float | None) -> float:
        """Wait until the width moved, then held still, or the settle time ran out."""
        lim = self.limits
        clock = time.monotonic
        start = clock()
        w0 = last = float(self.robot.get_gripper_width())
        moved = False
        stable_since: float | None = None
        while clock() - start < lim.gripper_settle_s:
            self._wait(lim.gripper_poll_dt_s)
            now = clock()
            cur = float(self.robot.get_gripper_width())
            moved = moved or abs(cur - w0) > GRIPPER_MOTION_EPS_M
            if abs(cur - last) <= GRIPPER_STABLE_EPS_M:
                stable_since = now if stable_since is None else stable_since
            else:
                stable_since = None
            last = cur
            sustained = (
                stable_since is not None and now - stable_since >= GRIPPER_SUSTAIN_S
            )
            decisive = decisive_max_m is None or cur <= decisive_max_m
            if (
                moved
                and sustained
                and decisive
                and now - start >= lim.gripper_min_settle_s
            ):
                return cur
        return last

    # -- joint-space reset ------------------------------------------------

    def move_to_joints(self, target: Any) -> dict[str, Any]:
        """Stream a straight joint-space move to ``target`` (6 rad), then re-sync."""
        lim = self.limits
        goal = np.asarray(target, dtype=float).reshape(-1)
        if goal.shape != (6,) or not np.all(np.isfinite(goal)):
            raise ValueError("target joints must be 6 finite values (rad)")
        if np.any(goal < JOINT_LIMITS_RAD[:, 0]) or np.any(
            goal > JOINT_LIMITS_RAD[:, 1]
        ):
            raise ValueError(
                f"target joints {goal.tolist()} are outside the joint limits"
            )
        start = np.asarray(self.robot.get_joint_positions(), dtype=float)[:6]
        steps = max(2, int(round(lim.joint_stream_hz * max(0.1, lim.reset_time_s))))
        period = max(0.1, lim.reset_time_s) / steps
        # The Z floor and the box hold on the reset path too (refused before any motion).
        self._ensure_synced()
        self.check_joint_path(start, goal, steps)
        # The Cartesian setpoint is meaningless once the reset streams; a failure below
        # leaves it invalid so the next step re-syncs from the measured pose.
        self._invalidate()
        cancelled = False
        q = start
        try:
            for i in range(1, steps + 1):
                self._check_stop()
                q = start + (goal - start) * (i / steps)
                self._stream(q)
                self.sleep(period)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                err = np.abs(np.asarray(self.robot.get_joint_positions())[:6] - goal)
                if float(err.max()) < lim.reset_tolerance_rad:
                    break
                self._wait(0.05)
        except Stopped:
            cancelled = True
            self._stream(q)
        self.robot.note_commanded_pose(None)
        notes = self.sync()
        err = float(
            np.abs(np.asarray(self.robot.get_joint_positions())[:6] - goal).max()
        )
        if not cancelled and err < lim.reset_tolerance_rad:
            # At the begin/rest pose: the yaw budget restarts from this heading.
            self._yaw_ref = float(self._target_euler[2])
        out = {
            "ok": not cancelled and err < lim.reset_tolerance_rad,
            "target_joints": goal.tolist(),
            "max_joint_error_rad": err,
            "notes": notes,
        }
        if cancelled:
            out["cancelled"] = True
        return out
