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
# The session wiring follows Show-Harness core/piper/piper_session.py,
# core/piper/dual_session.py and core/piper/config.py (Apache-2.0,
# github.com/showlab/Show-Harness @137d571).

"""RPC server owning one AgileX Piper arm, or both arms of the Cobot Magic rig.

Single arm (the default): the config names ``arm`` and the cameras ``front`` /
``wrist``. Dual arm (the Show-Harness rig, configs/robot_piper.yaml with the per-arm
``arms:`` block of configs/site/piper_arms.yaml): the config has an ``arms:`` mapping
with a ``left`` and a ``right`` block; everything outside it is shared, and each
block overrides the shared ``calibration`` / ``limits`` / ``gripper`` / ``motion`` /
``ros`` keys and names that arm's wrist camera, so each arm has its own Z floor,
workspace box (in its own base frame) and begin pose. The cameras are then ``front``
(shared), ``wrist_left`` and ``wrist_right``, and ``step`` / ``move_joints`` /
``halt_arm`` take ``arm``.

Everything goes over ROS topics (``ros_io``); start the CAN link(s), the camera
driver and the arm node(s) (mode 1) first. One call runs at a time, so on two arms
one arm moves per call; ``stop`` is lock-free and the running motion checks it
between joint waypoints / settle ticks / gripper polls and returns
``cancelled: true`` after holding the arm where it is.

Per-arm faults (two arms): a step that ends with a divergence, a reach fallback that
fell short or a dropped gripper command, or raises after commanding the arm (stale
feedback), halts that arm only: further motion of it is refused until a ``reset`` of
that arm succeeds, while the other arm keeps working. A refusal that sent nothing (a
too-long step, the yaw budget) does not halt. ``halt_arm`` stops an arm the same way on
request (e.g. its part of the task is done).

Reset: one arm (``arm``), or both only when asked (``both=True``). It opens the
gripper, so a gripper that holds something (closed wider than ``empty_width_m``) is
refused unless the caller passes ``release=True`` after the operator confirmed it.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pi_embodied_services.components.code_api import register_code_api
from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.robots.piper.controller import PiperController, PiperLimits
from pi_embodied_services.robots.piper.primitives import PIPER_PRIMITIVES
from pi_embodied_services.utils.logging import get_logger

logger = get_logger("piper_env_server")

DEFAULT_CONFIG = Path(__file__).with_name("config") / "example.yaml"
SIDES = ("left", "right")
#: Config sections a per-arm block may override key by key.
ARM_SECTIONS = (
    "calibration",
    "limits",
    "gripper",
    "motion",
    "smooth",
    "ros",
    "cameras",
)
#: The ``smooth:`` section (Show-Harness core/launch.py smooth_* keys) -> PiperLimits.
SMOOTH_KEYS = {
    "enabled": "smooth",
    "substeps": "smooth_substeps",
    "dt_s": "smooth_dt_s",
    "max_speed_mps": "smooth_max_speed_mps",
    "blend": "smooth_blend",
    "cruise": "smooth_cruise",
    "chain_window_s": "smooth_chain_window_s",
}


def is_dual(cfg: dict[str, Any]) -> bool:
    """Whether ``cfg`` describes both arms (an ``arms:`` block)."""
    return cfg.get("arms") is not None


def arm_config(cfg: dict[str, Any], side: str) -> dict[str, Any]:
    """The single-arm view of a dual config for ``side`` (Show-Harness core/piper/config.py
    ``arm_config``): the shared keys with that arm's block merged over each section."""
    if side not in SIDES:
        raise ValueError(f"arm must be 'left' or 'right', got {side!r}")
    arms = cfg.get("arms")
    if not isinstance(arms, dict) or any(
        not isinstance(arms.get(s), dict) for s in SIDES
    ):
        raise ValueError("arms: needs a `left` and a `right` block (the dual rig)")
    extra = set(arms) - set(SIDES)
    if extra:
        raise ValueError(f"arms: unknown arm(s) {sorted(extra)}; use left and right")
    block = arms[side]
    unknown = set(block) - set(ARM_SECTIONS)
    if unknown:
        raise ValueError(
            f"arms.{side}: unknown keys {sorted(unknown)} (allowed: {', '.join(ARM_SECTIONS)})"
        )
    out = copy.deepcopy({k: v for k, v in cfg.items() if k != "arms"})
    for section in ARM_SECTIONS:
        if block.get(section) is not None:
            out[section] = {**(out.get(section) or {}), **block[section]}
    out["arm"] = side
    return out


def arm_configs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{side: single-arm config}``: both arms of a dual config, else the one arm."""
    if is_dual(cfg):
        return {side: arm_config(cfg, side) for side in SIDES}
    return {cfg.get("arm", "left"): cfg}


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load the robot YAML (config/example.yaml, config/dual_example.yaml) and validate
    every arm's limits."""
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text()) or {}
    if not is_dual(cfg) and cfg.get("arm", "left") not in SIDES:
        raise ValueError("arm must be 'left' or 'right'")
    for side, acfg in arm_configs(cfg).items():
        try:
            limits_from_config(acfg).validate()
            check_binding(acfg)
        except ValueError as exc:
            raise ValueError(
                f"{side} arm: {exc}" if is_dual(cfg) else str(exc)
            ) from exc
    return cfg


def identity_source(cfg: dict[str, Any]) -> str:
    """``ros.identity`` of a single-arm config (default ``can_serial``)."""
    return str((cfg.get("ros") or {}).get("identity", "can_serial"))


def check_binding(cfg: dict[str, Any]) -> None:
    """A single-arm config's calibration must name the arm it was captured on
    (``calibration.arm_id``), unless ``ros.identity`` is ``none``."""
    if identity_source(cfg) == "none":
        return
    if (cfg.get("calibration") or {}).get("arm_id") in (None, ""):
        raise ValueError(
            "calibration.arm_id is not set: the calibration must name the arm it was "
            "captured on. On the rig run `python -m pi_embodied_services.robots.piper."
            "env_server --robot-config <yaml> --print-identity` and copy the value "
            "(or set ros.identity: none to skip the binding)"
        )


def limits_from_config(cfg: dict[str, Any]) -> PiperLimits:
    lim = dict(cfg.get("limits") or {})
    grip = dict(cfg.get("gripper") or {})
    motion = dict(cfg.get("motion") or {})
    cal = dict(cfg.get("calibration") or {})
    smooth = dict(cfg.get("smooth") or {})
    unknown_smooth = set(smooth) - set(SMOOTH_KEYS)
    if unknown_smooth:
        raise ValueError(f"unknown smooth keys: {sorted(unknown_smooth)}")
    kwargs: dict[str, Any] = {
        "z_floor_m": cal.get("z_floor_m"),
        **{k: lim[k] for k in lim if k in PiperLimits.__dataclass_fields__},
        **{f"gripper_{k}": grip[k] for k in ("settle_s", "min_settle_s") if k in grip},
        **{
            k: grip[k]
            for k in (
                "open_width_m",
                "empty_width_m",
                "close_threshold_m",
                "grasp_open_width_m",
            )
            if k in grip
        },
        **{k: motion[k] for k in motion if k in PiperLimits.__dataclass_fields__},
        **{SMOOTH_KEYS[k]: v for k, v in smooth.items()},
    }
    if "backend" in motion:
        kwargs["motion_backend"] = motion["backend"]
    if "ori_flex_deg" in motion:
        kwargs["ori_flex_rad"] = math.radians(float(motion["ori_flex_deg"]))
    unknown = set(lim) - set(PiperLimits.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown limits keys: {sorted(unknown)}")
    return PiperLimits(**kwargs)


def resize_with_pad(image: np.ndarray, size: int) -> np.ndarray:
    """Letterbox ``image`` into a ``size`` x ``size`` square (nearest neighbour)."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    rows = np.minimum((np.arange(nh) / scale).astype(int), h - 1)
    cols = np.minimum((np.arange(nw) / scale).astype(int), w - 1)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top : top + nh, left : left + nw] = image[rows][:, cols]
    return out


#: The limits every arm reports in get_env_meta.
META_LIMITS = (
    "z_floor_m",
    "enable_z_floor",
    "workspace_min",
    "workspace_max",
    "max_step_m",
    "max_yaw_rad",
    "max_total_yaw_rad",
    "speed_mps",
    "open_width_m",
    "empty_width_m",
)


class PiperEnvFacade(BaseEnvFacade):
    """The ``env.*`` protocol for one Piper arm or both (see the module docstring).

    ``robot`` is one arm (single-arm config) or ``{"left": arm, "right": arm}``.
    """

    SERVICE_NAME = "piper-env"

    def __init__(
        self,
        cfg: dict[str, Any],
        robot: Any,
        cameras: dict[str, Any],
        config_path: str = "",
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._config_path = config_path
        self._dual = is_dual(cfg)
        self._arm_cfgs = arm_configs(cfg)
        robots = robot if self._dual else {next(iter(self._arm_cfgs)): robot}
        if set(robots) != set(self._arm_cfgs):
            raise ValueError(f"robots {sorted(robots)} do not match the config's arms")
        self._robots = robots
        self._cameras = cameras
        self._controllers = {
            side: PiperController(
                robots[side],
                limits_from_config(acfg),
                stop_requested=self.stop_requested,
            )
            for side, acfg in self._arm_cfgs.items()
        }
        #: Two arms: why an arm is halted (refused until its reset).
        self._halted: dict[str, str] = {}
        # Each calibration (floor, box, poses) holds only on the arm it was captured on.
        self._arm_ids: dict[str, str | None] = {}
        for side, acfg in self._arm_cfgs.items():
            self._arm_ids[side] = self._bound_identity(side, acfg)
        for c in self._controllers.values():
            c.sync()

    def _bound_identity(self, side: str, acfg: dict[str, Any]) -> str | None:
        """Refuse an arm whose identity is not the one its calibration names."""
        if identity_source(acfg) == "none":
            return None
        check_binding(acfg)
        want = str(acfg["calibration"]["arm_id"])
        got = self._robots[side].identity()
        if got != want:
            where = f"arms.{side}.calibration" if self._dual else "calibration"
            raise ValueError(
                f"the {side} arm reports identity {got!r}, but {where}.arm_id is {want!r}: "
                "this calibration (Z floor, workspace, poses) belongs to another arm. Use "
                "that arm's config, or re-calibrate this arm and update arm_id"
            )
        return got

    def _register_rpc(self) -> None:
        for name in (
            "get_env_meta",
            "reset",
            "get_robot_state",
            "get_observation",
            "get_camera_meta",
            "step",
            "move_joints",
            "halt_arm",
        ):
            self._rpc[f"env.{name}"] = getattr(self, name)
        register_code_api(self, PIPER_PRIMITIVES)

    def close(self) -> None:
        for cam in self._cameras.values():
            cam.close()
        for robot in self._robots.values():
            robot.close()

    def _side(self, arm: str | None) -> str:
        """The arm a call addresses: required on two arms, optional (and checked) on one."""
        if arm is None:
            if self._dual:
                raise ValueError("two arms: pass arm='left' or arm='right'")
            return next(iter(self._controllers))
        if arm not in self._controllers:
            raise ValueError(
                f"arm {arm!r} is not driven here (have: {', '.join(self._controllers)})"
            )
        return arm

    # -- read -------------------------------------------------------------

    def _limits_meta(self, side: str) -> dict[str, Any]:
        lim = self._controllers[side].limits
        return {k: getattr(lim, k) for k in META_LIMITS}

    def get_env_meta(self) -> dict[str, Any]:
        sides = list(self._controllers)
        cals = {s: self._arm_cfgs[s].get("calibration") or {} for s in sides}
        meta: dict[str, Any] = {
            "ok": True,
            "robot": "piper",
            "arm": "dual" if self._dual else sides[0],
            "arms": sides if self._dual else [],
            "config_path": self._config_path,
            "cameras": sorted(self._cameras),
            "units_frame": (self._cfg.get("motion") or {}).get("units_frame", "base"),
            "limits": self._limits_meta(sides[0]),
            "has_begin_pose": all(
                c.get("begin_joints") is not None for c in cals.values()
            ),
            "has_rest_pose": all(
                c.get("rest_joints") is not None for c in cals.values()
            ),
            "arm_ids": dict(self._arm_ids),
            "motion_backend": self._controllers[sides[0]].backend,
            "smooth": {
                key: getattr(self._controllers[sides[0]].limits, field)
                for key, field in SMOOTH_KEYS.items()
            },
            "tasks": self._cfg.get("tasks") or {},
        }
        if self._dual:
            per = {s: self._limits_meta(s) for s in sides}
            # The shared per-call caps are the tighter arm's; floors and boxes stay per arm.
            meta["limits"] = {
                **per[sides[0]],
                "max_step_m": min(p["max_step_m"] for p in per.values()),
                "max_yaw_rad": min(p["max_yaw_rad"] for p in per.values()),
                "z_floor_m": None,
                "workspace_min": None,
                "workspace_max": None,
            }
            meta["arm_limits"] = per
            meta["motion_backend"] = {s: self._controllers[s].backend for s in sides}
            meta["halted"] = dict(self._halted)
        return meta

    def _arm_state(self, side: str) -> dict[str, Any]:
        state = self._controllers[side].state()
        # A reset opens the gripper: the client asks the operator first when this is true.
        state["holding_object"] = self._holding(side) is not None
        status = getattr(self._robots[side], "arm_status", lambda: None)()
        if status is not None:
            state["arm_status"] = status
        if self._dual:
            state["arm"] = side
            state["halted"] = self._halted.get(side)
        return state

    def get_robot_state(self, arm: str | None = None) -> dict[str, Any]:
        """One arm's state (flat); on two arms without ``arm``: ``{"arms": {side: state}}``."""
        if self._dual and arm is None:
            return {"arms": {s: self._arm_state(s) for s in self._controllers}}
        return self._arm_state(self._side(arm))

    def get_observation(self) -> dict[str, Any]:
        size = int((self._cfg.get("cameras") or {}).get("image_size") or 0)
        images = {}
        for name, cam in self._cameras.items():
            frame = cam.read()
            images[name] = resize_with_pad(frame, size) if size else frame
        return {"images": images, "robot_state": self.get_robot_state()}

    def get_camera_meta(self) -> dict[str, Any]:
        cams = self._cfg.get("cameras") or {}
        topics = {"front": cams.get("front")}
        for side, acfg in self._arm_cfgs.items():
            wrist = (acfg.get("cameras") or {}).get("wrist")
            topics["wrist_" + side if self._dual else "wrist"] = wrist
        return {
            "cameras": {
                name: {"topic": topics.get(name), "role": name}
                for name in self._cameras
            },
            "image_size": cams.get("image_size") or None,
        }

    # -- motion -----------------------------------------------------------

    def _guarded(self, side: str, run: Any) -> dict[str, Any]:
        """Run one motion of ``side``; on two arms a fault halts that arm only.

        A ``ValueError`` raised before any command reached the arm is a refusal of the
        arguments (step too long, yaw budget, joint path), not a fault: no halt."""
        if side in self._halted:
            raise RuntimeError(
                f"the {side} arm is halted ({self._halted[side]}); reset it before moving "
                "it again. The other arm is not affected."
            )
        sent = self._controllers[side].commands
        try:
            out = run()
        except Exception as exc:
            refused = (
                isinstance(exc, ValueError) and self._controllers[side].commands == sent
            )
            if self._dual and not refused:
                self._halted[side] = f"error: {exc}"
            raise
        if self._dual:
            out["arm"] = side
            if not out.get("ok", True) and not out.get("cancelled"):
                self._halted[side] = "; ".join(out.get("notes") or ["step failed"])
                out["halted"] = True
        return out

    def step(
        self,
        delta_xyz: Any = (0.0, 0.0, 0.0),
        yaw: float = 0.0,
        gripper: str | None = None,
        frame: str = "base",
        reopen_empty: bool = True,
        arm: str | None = None,
        continuous: bool = False,
    ) -> dict[str, Any]:
        """One guarded step of one arm: translate (m), yaw (rad), then open/close.

        ``continuous``: another translation in about the same direction follows at once
        (smooth chaining; see controller.py)."""
        side = self._side(arm)
        return self._guarded(
            side,
            lambda: self._controllers[side].step(
                delta_xyz,
                yaw=yaw,
                gripper=gripper,
                frame=frame,
                reopen_empty=reopen_empty,
                continuous=bool(continuous),
            ),
        )

    def move_joints(
        self, pose: str = "begin", arm: str | None = None
    ) -> dict[str, Any]:
        """Joint-space move of one arm to a calibrated pose: ``begin`` or ``rest``."""
        side = self._side(arm)
        return self._guarded(side, lambda: self._move_joints(side, pose))

    def _move_joints(self, side: str, pose: str) -> dict[str, Any]:
        if pose not in ("begin", "rest"):
            raise ValueError("pose must be 'begin' or 'rest'")
        joints = (self._arm_cfgs[side].get("calibration") or {}).get(f"{pose}_joints")
        if joints is None:
            where = f"arms.{side}.calibration" if self._dual else "calibration"
            raise ValueError(
                f"{where}.{pose}_joints is not set: jog the arm to the {pose} pose, "
                f"read /puppet/joint_{side} position[0:6] and write it to the config"
            )
        return self._controllers[side].move_to_joints(joints)

    def halt_arm(self, arm: str | None = None, reason: str = "") -> dict[str, Any]:
        """Stop one arm: hold it where it is and refuse its motion until its reset."""
        side = self._side(arm)
        # Latched first: a hold that fails (stale feedback) still leaves the arm refused.
        self._halted[side] = f"halted: {reason}" if reason else "halted on request"
        self._controllers[side]._hold_after_stop()
        return {"ok": True, "arm": side, "halted": self._halted[side]}

    def _holding(self, side: str) -> float | None:
        """The gripper width (m) when ``side``'s gripper holds something, else None."""
        c = self._controllers[side]
        width = float(self._robots[side].get_gripper_width())
        empty = c.limits.empty_width_m or 0.0
        return width if empty < width < c.limits.close_threshold_m else None

    def _reset_arm(self, side: str) -> dict[str, Any]:
        """Open the gripper, then move to the begin pose. The arm's halt is cleared only
        when both succeeded; a failed or cancelled reset leaves (or sets) it halted."""
        c = self._controllers[side]
        try:
            grip = c.step(gripper="open")
            move = (
                self._move_joints(side, "begin")
                if not grip.get("cancelled")
                else {"ok": False, "cancelled": True, "notes": ["not started"]}
            )
        except Exception as exc:
            if self._dual:
                self._halted[side] = f"reset failed: {exc}"
            raise
        cancelled = bool(grip.get("cancelled") or move.get("cancelled"))
        ok = bool(move.get("ok")) and bool(grip.get("ok")) and not cancelled
        if ok:
            self._halted.pop(side, None)
        elif self._dual:
            self._halted[side] = (
                "reset cancelled"
                if cancelled
                else "reset failed: "
                + "; ".join(
                    grip.get("notes", []) + move.get("notes", [])
                    or ["not at the begin pose"]
                )
            )
        return {
            "ok": ok,
            "gripper": grip,
            "move": move,
            **({"cancelled": True} if cancelled else {}),
        }

    def reset(
        self, arm: str | None = None, both: bool = False, release: bool = False
    ) -> dict[str, Any]:
        """Open the gripper, then move to the begin pose.

        Two arms: the named ``arm`` only, or both (one after the other) with
        ``both=True``; a halt is cleared only by that arm's successful reset.
        ``release``: the operator confirmed that a gripper holding an object may open
        (without it such a reset is refused before any motion)."""
        if self._dual and arm is None and not both:
            raise ValueError(
                "two arms: reset names its arm (arm='left' or arm='right'), or pass "
                "both=True to reset both; nothing was commanded"
            )
        if both and arm is not None:
            raise ValueError("pass arm or both=True, not both")
        sides = list(self._controllers) if both else [self._side(arm)]
        if not release:
            held = {s: w for s in sides if (w := self._holding(s)) is not None}
            if held:
                what = ", ".join(f"{s} ({w:.3f} m)" for s, w in held.items())
                raise ValueError(
                    f"the gripper of the {what} arm holds an object: reset opens it and "
                    "would drop it. Ask the operator, then pass release=True; nothing "
                    "was commanded"
                )
        if self._dual and both:
            arms = {}
            for side in sides:
                arms[side] = self._reset_arm(side)
                if not arms[side]["ok"]:
                    break
            ok = len(arms) == len(sides) and all(r["ok"] for r in arms.values())
            return {
                "ok": ok,
                "arms": arms,
                # Single-arm readers look at `move`: the first arm that did not arrive.
                "move": next((r["move"] for r in arms.values() if not r["ok"]), {}),
                "robot_state": self.get_robot_state(),
                **(
                    {"cancelled": True}
                    if any(r.get("cancelled") for r in arms.values())
                    else {}
                ),
            }
        side = sides[0]
        out = self._reset_arm(side)
        return {
            **out,
            **({"arm": side} if self._dual else {}),
            "robot_state": self.get_robot_state(side if self._dual else None),
        }


def build(cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Connect the ROS arm(s) and cameras named in ``cfg``."""
    from pi_embodied_services.robots.piper.ros_io import PiperRosArm, RosImageCamera

    acfgs = arm_configs(cfg)
    cams = cfg.get("cameras") or {}
    topics = {"front": cams.get("front")}
    for side, acfg in acfgs.items():
        topics["wrist_" + side if is_dual(cfg) else "wrist"] = (
            acfg.get("cameras") or {}
        ).get("wrist")
    robots: dict[str, Any] = {}
    cameras: dict[str, Any] = {}

    def release() -> None:
        for cam in cameras.values():
            cam.close()
        for robot in robots.values():
            robot.close()

    try:
        for side, acfg in acfgs.items():
            ros = acfg.get("ros") or {}
            robots[side] = PiperRosArm(
                arm=side,
                feedback_timeout_s=float(ros.get("feedback_timeout_s", 5.0)),
                max_feedback_age_s=float(ros.get("max_feedback_age_s", 0.5)),
                identity_source=identity_source(acfg),
            ).connect()
        for name, topic in topics.items():
            if topic:
                cameras[name] = RosImageCamera(
                    topic,
                    max_age_s=float(cams.get("max_age_s", 1.0)),
                    connect_timeout_s=float(cams.get("connect_timeout_s", 10.0)),
                )
    except Exception:
        release()
        raise
    if not cameras:
        release()
        raise ValueError("configure at least one camera topic (cameras.front / wrist)")
    return (robots if is_dual(cfg) else next(iter(robots.values()))), cameras


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--robot-config", default=None)
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Validate and print the resolved config, then exit without ROS.",
    )
    parser.add_argument(
        "--print-identity",
        action="store_true",
        help="Print each arm's identity (for calibration.arm_id), then exit; no motion.",
    )
    args = parser.parse_args()
    if args.print_identity:
        from pi_embodied_services.robots.piper.ros_io import arm_identity

        raw = (
            yaml.safe_load(Path(args.robot_config or DEFAULT_CONFIG).read_text()) or {}
        )
        for side, acfg in arm_configs(raw).items():
            print(f"{side}: arm_id: {arm_identity(side, identity_source(acfg))!r}")
        return 0
    cfg = load_config(args.robot_config)
    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return 0
    robot, cameras = build(cfg)
    facade = PiperEnvFacade(
        cfg, robot, cameras, config_path=str(args.robot_config or DEFAULT_CONFIG)
    )
    try:
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        facade.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
