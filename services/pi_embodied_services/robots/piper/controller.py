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
# units to deltas), no plugins (smooth / variable-step / rotation), and every wait
# polls a stop callback so the RPC ``stop`` halts motion between waypoints. Added a
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
   targets at ``joint_stream_hz``, at ``speed_mps``; the first unreachable waypoint
   stops the move there ("reach clamp"). ``endpose``: firmware MOVE P re-commanded
   ``settle_steps`` times. The joint backend is used only after the vendored FK
   reproduces the arm's own pose feedback within 1 cm.
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

    def validate(self) -> None:
        if self.enable_z_floor and self.z_floor_m is None:
            raise ValueError(
                "z_floor_m is not calibrated: rest the gripper on the tabletop, read "
                "the EEF z from /puppet/end_pose_euler_<arm>, and set "
                "calibration.z_floor_m (or set limits.enable_z_floor: false)"
            )
        if (self.workspace_min is None) != (self.workspace_max is None):
            raise ValueError("set both workspace_min and workspace_max, or neither")
        if self.workspace_min is not None:
            lo = np.asarray(self.workspace_min, float)
            hi = np.asarray(self.workspace_max, float)
            if lo.shape != (3,) or hi.shape != (3,) or not np.all(lo < hi):
                raise ValueError("workspace_min/max must be [x,y,z] with min < max")
        if self.motion_backend not in ("joint_stream", "endpose"):
            raise ValueError("motion_backend must be 'joint_stream' or 'endpose'")
        for name in ("max_step_m", "max_yaw_rad", "speed_mps", "yaw_speed_radps"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0")


@dataclass
class StepReport:
    ok: bool = True
    cancelled: bool = False
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
    ) -> None:
        limits.validate()
        self.robot = robot
        self.limits = limits
        self.stop_requested = stop_requested
        self.sleep = sleep
        self._target_pos: np.ndarray | None = None
        self._target_euler: np.ndarray | None = None
        self.gripper_closed: bool | None = None
        self._kin: Any = None
        self._q_cmd: np.ndarray | None = None
        self.backend = "endpose"
        rad = limits.divergence_resync_rad
        self._resync_rad = (
            1.5 * limits.max_yaw_rad if rad is None else (rad if rad > 0 else None)
        )

    # -- state ------------------------------------------------------------

    def sync(self) -> list[str]:
        """Reset the setpoint to the measured pose and (re)validate the joint backend."""
        notes: list[str] = []
        pose = np.asarray(self.robot.get_ee_pose(), dtype=float)
        self._target_pos = pose[:3].copy()
        self._target_euler = quat_to_euler(pose[3:])
        width = float(self.robot.get_gripper_width())
        self.gripper_closed = width < self.limits.close_threshold_m
        self.backend, self._kin, self._q_cmd = "endpose", None, None
        if self.limits.motion_backend != "joint_stream":
            return notes
        q = np.asarray(self.robot.get_joint_positions(), dtype=float)[:6]
        kin, err = select_dh_variant(q, pose)
        if err > 0.01:
            notes.append(
                f"joint_stream disabled: FK is {err * 1000:.1f} mm from the arm's own "
                "pose feedback; using endpose (MOVE P)"
            )
            logger.warning(notes[-1])
            return notes
        self.backend, self._kin, self._q_cmd = "joint_stream", kin, q
        margins = np.degrees(
            np.minimum(q - JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1] - q)
        )
        tight = [f"j{j + 1} {margins[j]:.1f} deg" for j in range(6) if margins[j] < 8]
        if tight:
            notes.append(f"start pose near a joint limit ({', '.join(tight)})")
        return notes

    def _ensure_synced(self) -> None:
        if self._target_pos is None:
            self.sync()

    @property
    def target_pose(self) -> np.ndarray:
        self._ensure_synced()
        return np.concatenate([self._target_pos, euler_to_quat(self._target_euler)])

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
    ) -> dict[str, Any]:
        """Translate by ``delta_xyz`` (m), yaw by ``yaw`` (rad), then open/close.

        ``reopen_empty=False`` leaves a close that caught nothing closed (for a caller
        that runs its own recovery, e.g. the units layer's recovery plugin).
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
        pre = np.asarray(self.robot.get_ee_pose(), dtype=float)
        base_delta = delta
        if frame == "heading" and norm > 0:
            heading = self.heading_yaw()
            if heading is None:
                report.notes.append("no gripper heading (tool vertical): base frame")
            else:
                c, s = math.cos(heading), math.sin(heading)
                base_delta = np.array(
                    [c * delta[0] - s * delta[1], s * delta[0] + c * delta[1], delta[2]]
                )
        try:
            if norm > 0 or yaw != 0:
                self._move(base_delta, yaw, report)
            if gripper is not None:
                self._gripper(gripper == "close", report, reopen_empty)
        except Stopped:
            report.cancelled = True
            report.ok = False
            report.notes.append("stopped")
            self._hold_after_stop()
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
        if report.cancelled:
            out["cancelled"] = True
        return out

    def _check_stop(self) -> None:
        if self.stop_requested():
            raise Stopped()

    def _wait(self, seconds: float) -> None:
        self._check_stop()
        if seconds > 0:
            self.sleep(seconds)

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

    def _move(self, delta: np.ndarray, yaw: float, report: StepReport) -> None:
        start_pos = self._target_pos.copy()
        start_euler = self._target_euler.copy()
        target, notes = self.clamp_target(start_pos + delta)
        report.notes.extend(notes)
        self._target_pos = target
        self._target_euler = start_euler.copy()
        self._target_euler[2] += yaw
        if self.backend == "joint_stream":
            self._drive_joint_stream(start_pos, start_euler, report)
        else:
            self._drive_endpose()
        self._guard_divergence(report)

    # -- backends ---------------------------------------------------------

    def _drive_endpose(self) -> None:
        pose = self.target_pose
        for _ in range(max(1, self.limits.settle_steps)):
            self._check_stop()
            self.robot.command_pose(pose)
            self._wait(self.limits.settle_dt_s)

    def _joint_settle(self) -> None:
        for _ in range(max(1, self.limits.settle_steps)):
            self._check_stop()
            self.robot.stream_joints(self._q_cmd)
            self._wait(self.limits.settle_dt_s)

    def _drive_joint_stream(
        self, start_pos: np.ndarray, start_euler: np.ndarray, report: StepReport
    ) -> None:
        lim = self.limits
        dpos = self._target_pos - start_pos
        deuler = self._target_euler - start_euler
        dist = float(np.linalg.norm(dpos))
        duration = max(
            dist / lim.speed_mps,
            float(np.abs(deuler).max()) / lim.yaw_speed_radps,
            0.15,
        )
        n = max(2, int(round(lim.joint_stream_hz * duration)))
        dt = duration / n
        q = self._q_cmd
        for i in range(1, n + 1):
            try:
                self._check_stop()
            except Stopped:
                self._clamp_at(q, (i - 1) / n, "stopped", report)
                raise
            frac = i / n
            rot = R.from_euler("xyz", start_euler + deuler * frac).as_matrix()
            sol, _dev = self._kin.ik_bounded(
                start_pos + dpos * frac, rot, q_seed=q, max_ori_dev_rad=lim.ori_flex_rad
            )
            if sol is None:
                if i == 1:
                    report.notes.append(
                        "reach fallback: IK failed at the first waypoint (arm at its "
                        "reach limit or bad seed) -> one EndPoseCtrl attempt"
                    )
                    self._drive_endpose()
                    self._q_cmd = np.asarray(self.robot.get_joint_positions(), float)[
                        :6
                    ]
                else:
                    self._clamp_at(q, (i - 1) / n, "reach clamp", report)
                return
            q = sol
            self.robot.stream_joints(q)
            if dt > 0:
                self.sleep(dt)
        self._q_cmd = q
        self.robot.note_commanded_pose(self.target_pose)
        self._joint_settle()

    def _clamp_at(
        self, q_last: np.ndarray, frac: float, why: str, report: StepReport
    ) -> None:
        """End a joint-stream move at the last streamed waypoint; truthful setpoint."""
        self._q_cmd = q_last
        pos, _ = self._kin.fk(q_last)
        self._target_pos = np.asarray(pos, dtype=float)
        self.robot.note_commanded_pose(self._kin.fk_pose7(q_last))
        if why != "stopped":
            report.notes.append(
                f"{why}: move stopped at {frac * 100:.0f}% (workspace boundary at the "
                f"{math.degrees(self.limits.ori_flex_rad):.0f} deg orientation budget)"
            )
            self._joint_settle()

    def _hold_after_stop(self) -> None:
        """After a stop: hold where the arm is and restart the setpoint from there."""
        try:
            if self.backend == "joint_stream" and self._q_cmd is not None:
                self.robot.stream_joints(self._q_cmd)
            else:
                self.robot.command_pose(self.robot.get_ee_pose())
        except Exception as exc:
            logger.warning("hold after stop failed: %s", exc)
        self.sync()

    def _guard_divergence(self, report: StepReport) -> None:
        measured = np.asarray(self.robot.get_ee_pose(), dtype=float)
        if self.backend == "joint_stream" and self._q_cmd is not None:
            exp_pos, exp_rot = self._kin.fk(self._q_cmd)
            exp_quat = R.from_matrix(exp_rot).as_quat()
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
        self.robot.set_gripper_width(0.0 if close else lim.open_width_m)
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
            self.robot.set_gripper_width(lim.open_width_m)
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
        cancelled = False
        q = start
        try:
            for i in range(1, steps + 1):
                self._check_stop()
                q = start + (goal - start) * (i / steps)
                self.robot.stream_joints(q)
                self.sleep(period)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                err = np.abs(np.asarray(self.robot.get_joint_positions())[:6] - goal)
                if float(err.max()) < lim.reset_tolerance_rad:
                    break
                self._wait(0.05)
        except Stopped:
            cancelled = True
            self.robot.stream_joints(q)
        self.robot.note_commanded_pose(None)
        notes = self.sync()
        err = float(
            np.abs(np.asarray(self.robot.get_joint_positions())[:6] - goal).max()
        )
        out = {
            "ok": not cancelled and err < lim.reset_tolerance_rad,
            "target_joints": goal.tolist(),
            "max_joint_error_rad": err,
            "notes": notes,
        }
        if cancelled:
            out["cancelled"] = True
        return out
