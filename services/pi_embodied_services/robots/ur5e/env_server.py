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

"""RPC server owning one UR5e (ur_rtde), its Robotiq gripper and its cameras.

Methods and result shapes follow the Franka servers (``robots/franka_polymetis``):
``env.get_env_meta`` reports ``capabilities`` in the same layout, motions return the
franka-env keys, and ``env.set_gripper`` reports ``gripper_jammed`` / ``grasp_empty``.
Differences: ``env.move_pose`` (absolute pose within the per-call limits), no
``env.chunk_step`` (no VLA), and observations are per camera (``images`` /
``depths`` dicts) because the cameras may be RGB-only webcams or RTSP streams.

Safety, all on this server (pi's robot adds its own gate on top): a translation
beyond ``limits.max_move_m`` or a turn beyond ``max_rotate_rad`` is refused before
anything is commanded; so is a target outside the workspace box, below
``z_floor_m`` or tilting the tool past ``max_tilt_rad``. A running moveL/moveJ polls
``stop`` and is brought to rest with stopL/stopJ (``cancelled: true``); a stop,
timeout, driver error or missed target clears the setpoint, so the next command
starts from the measured pose. The config (limits, begin pose, every camera's
hand-eye calibration) is bound to one arm: ``calibration.arm_id`` must equal the
controller's serial number, and each calibration YAML must name the same arm.

Deploying: on the UR pendant enable Remote Control and the RTDE/URCap ports; on the
workstation install the services' ``ur5e`` extra (``ur-rtde``, cameras; add
``realsense-l515`` for an L515), copy ``config/example.yaml``, fill in the IP, the
cameras, then ``--print-identity`` (arm_id), ``--read-pose`` (floor, begin joints),
calibrate the cameras with ``robots/ur5e/calibrate.py`` and start pi with
``pi -e packages/embodied/src/ur5e --operator --arm-id <id> --robot-config <yaml>``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pi_embodied_services.components.cameras import (
    intrinsic_matrix,
    open_camera,
    parse_sources,
    validate_device,
)
from pi_embodied_services.components.code_api import register_code_api
from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.robots.ur5e.calibrate import load_calibration_yaml
from pi_embodied_services.robots.ur5e.control import (
    UR5eController,
    UR5eLimits,
    pose7_of,
)
from pi_embodied_services.robots.ur5e.primitives import UR5E_PRIMITIVES
from pi_embodied_services.utils.daemon import watch_parent_death
from pi_embodied_services.utils.logging import get_logger

logger = get_logger("ur5e_env_server")

DEFAULT_CONFIG = Path(__file__).with_name("config") / "example.yaml"

METHODS = (
    "get_env_meta",
    "reset",
    "get_robot_state",
    "get_observation",
    "get_camera_meta",
    "move_delta",
    "move_pose",
    "rotate_delta",
    "set_gripper",
)

#: ``gripper:`` section keys -> UR5eLimits fields.
_GRIPPER_KEYS = {
    "stroke_m": "gripper_stroke_m",
    "speed": "gripper_speed",
    "force": "gripper_force",
    "timeout_s": "gripper_timeout_s",
    "poll_s": "gripper_poll_s",
    "motion_eps_m": "gripper_motion_eps_m",
    "close_threshold_m": "close_threshold_m",
    "empty_width_m": "empty_width_m",
}
_LIMIT_KEYS = tuple(
    f
    for f in UR5eLimits.__dataclass_fields__
    if f not in set(_GRIPPER_KEYS.values()) and f != "begin_joints"
)


def _section(cfg: dict[str, Any], name: str, allowed) -> dict[str, Any]:
    section = cfg.get(name) or {}
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {name} keys {unknown}; valid: {sorted(allowed)}")
    return section


def _tuple(v: Any) -> Any:
    return tuple(float(x) for x in v) if isinstance(v, (list, tuple)) else v


def limits_from_config(cfg: dict[str, Any]) -> UR5eLimits:
    """Build and validate the limits from a loaded robot YAML."""
    kwargs: dict[str, Any] = {
        key: _tuple(value)
        for key, value in _section(cfg, "limits", _LIMIT_KEYS).items()
    }
    for key, value in _section(cfg, "gripper", _GRIPPER_KEYS).items():
        kwargs[_GRIPPER_KEYS[key]] = value
    joints = (cfg.get("calibration") or {}).get("begin_joints")
    if joints is not None:
        kwargs["begin_joints"] = _tuple(joints)
    limits = UR5eLimits(**kwargs)
    limits.validate()
    return limits


def identity_source(cfg: dict[str, Any]) -> str:
    """``robot.identity``: ``serial`` (default) or ``none``."""
    source = str((cfg.get("robot") or {}).get("identity", "serial"))
    if source not in ("serial", "none"):
        raise ValueError("robot.identity must be serial or none")
    return source


def check_binding(cfg: dict[str, Any]) -> None:
    """The config must name the arm it was captured on unless ``robot.identity: none``."""
    if identity_source(cfg) == "none":
        return
    if (cfg.get("calibration") or {}).get("arm_id") in (None, ""):
        raise ValueError(
            "calibration.arm_id is not set: the limits and camera calibrations must name "
            "the arm they were captured on. Run `python -m pi_embodied_services.robots."
            "ur5e.env_server --robot-config <yaml> --print-identity` and copy the value "
            "(or set robot.identity: none to skip the binding)"
        )


def camera_devices(
    cfg: dict[str, Any], override: str = ""
) -> tuple[str, list[str], dict[str, dict[str, Any]]]:
    """(main camera, extra cameras sorted, devices) from ``cameras.devices``, or from a
    ``--cameras name=type:source,...`` override (the config's devices supply
    ``mount`` / ``calibration`` / ``intrinsics`` for names they share)."""
    cams = cfg.get("cameras") or {}
    configured = cams.get("devices") or {}
    if not isinstance(configured, dict):
        raise ValueError("cameras.devices must be a mapping")
    if override.strip():
        devices = parse_sources(override)
        for name, dev in devices.items():
            for key in ("mount", "calibration", "intrinsics"):
                if key in (configured.get(name) or {}) and key not in dev:
                    dev[key] = configured[name][key]
    else:
        devices = {name: dict(dev or {}) for name, dev in configured.items()}
    if not devices:
        raise ValueError("configure at least one camera (cameras.devices or --cameras)")
    devices = {name: validate_device(name, dev) for name, dev in devices.items()}
    main = [name for name, dev in devices.items() if dev.get("main")]
    if len(main) != 1:
        raise ValueError("exactly one camera device must set main: true")
    return main[0], sorted(n for n in devices if n != main[0]), devices


def load_config(path: str | Path | None, cameras: str = "") -> dict[str, Any]:
    """Load and validate the robot YAML (see config/example.yaml)."""
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).expanduser().read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError("robot config must be a mapping")
    if not (cfg.get("robot") or {}).get("ip"):
        raise ValueError("robot.ip is required")
    gripper = (cfg.get("robot") or {}).get("gripper") or {}
    if str(gripper.get("type", "robotiq")) not in ("robotiq", "none"):
        raise ValueError("robot.gripper.type must be robotiq or none")
    limits_from_config(cfg)
    check_binding(cfg)
    camera_devices(cfg, cameras)
    return cfg


class UR5eEnvFacade(BaseEnvFacade):
    """The ``env.*`` protocol on one UR5e (see the module docstring)."""

    SERVICE_NAME = "ur5e-env"

    def __init__(
        self,
        cfg: dict[str, Any],
        arm: Any,
        gripper: Any,
        cameras: dict[str, Any],
        *,
        config_path: str = "",
        camera_flag: str = "",
        task_description: str = "",
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._config_path = config_path
        self._task = task_description
        self._arm = arm
        self._gripper = gripper
        self._cameras = cameras
        self._main, self._extras, self._devices = camera_devices(cfg, camera_flag)
        missing = sorted({self._main, *self._extras} - set(cameras))
        if missing:
            raise ValueError(f"cameras not connected: {missing}")
        self.arm_id = self._bound_identity()
        self.controller = UR5eController(
            arm, gripper, limits_from_config(cfg), self.stop_requested, sleep=sleep
        )
        self._calibrations: dict[str, dict[str, Any] | None] = {
            name: self._load_calibration(name) for name in cameras
        }
        self._meta = {name: self._camera_meta(name) for name in cameras}

    def _bound_identity(self) -> str | None:
        """Refuse an arm whose serial is not the one the config names."""
        if identity_source(self._cfg) == "none":
            return None
        check_binding(self._cfg)
        want = str(self._cfg["calibration"]["arm_id"])
        got = self._arm.identity()
        if got is None:
            raise ValueError(
                "the arm's serial number could not be read (dashboard port); set "
                "robot.identity: none only if this arm cannot be identified"
            )
        if str(got) != want:
            raise ValueError(
                f"the arm reports serial {got!r}, but calibration.arm_id is {want!r}: this "
                "config (Z floor, workspace, begin pose, camera calibrations) belongs to "
                "another arm. Use that arm's config, or re-calibrate this arm and update "
                "arm_id"
            )
        return str(got)

    def _load_calibration(self, name: str) -> dict[str, Any] | None:
        dev = self._devices[name]
        path = dev.get("calibration")
        if not path:
            return None
        cal = load_calibration_yaml(path)
        mount = dev.get("mount") or ("wrist" if cal["eye_on_hand"] else "fixed")
        if cal["eye_on_hand"] != (mount == "wrist"):
            raise ValueError(
                f"camera {name}: the calibration {path} is "
                f"{'eye-in-hand' if cal['eye_on_hand'] else 'eye-to-hand'} but the camera "
                f"is mounted {mount}"
            )
        if self.arm_id is not None and cal.get("arm_id") not in (None, self.arm_id):
            raise ValueError(
                f"camera {name}: the calibration {path} was made on arm "
                f"{cal['arm_id']!r}, not this arm ({self.arm_id!r}); re-calibrate it "
                "(robots/ur5e/calibrate.py) before using it"
            )
        if self.arm_id is not None and cal.get("arm_id") is None:
            raise ValueError(
                f"camera {name}: the calibration {path} names no arm_id; re-run "
                "calibrate.py solve with --arm-id"
            )
        serial = getattr(self._cameras[name], "serial", None)
        if (
            serial
            and cal.get("camera_serial")
            and str(cal["camera_serial"]) != str(serial)
        ):
            raise ValueError(
                f"camera {name}: the calibration {path} was made with camera "
                f"{cal['camera_serial']!r}, not {serial!r}"
            )
        return cal

    def _camera_meta(self, name: str) -> dict[str, Any]:
        cam = self._cameras[name]
        dev = self._devices[name]
        raw = cam.intrinsics()
        cal = self._calibrations[name]
        return {
            "name": name,
            "role": "main" if name == self._main else "extra",
            "mount": dev.get("mount"),
            "has_depth": bool(cam.has_depth),
            "raw_color_intrinsics": raw,
            "intrinsic_K": intrinsic_matrix(raw) if raw else None,
            "output_resolution": [raw["width"], raw["height"]] if raw else None,
            "depth_unit": "m",
            "extrinsic": None
            if cal is None
            else {
                "frame": "tcp" if cal["eye_on_hand"] else "base",
                "matrix": np.asarray(cal["matrix"]).tolist(),
                "path": cal["path"],
                "arm_id": cal.get("arm_id"),
            },
            **cam.describe(),
        }

    def serve(self, *, parent_watch: bool = False, **kwargs: Any) -> None:
        """Parent death stops the running motion (not only the loop)."""

        def on_death() -> None:
            self.request_stop()
            self.controller.hold()
            self._shutdown_event.set()

        if parent_watch:
            watch_parent_death(on_death)
        super().serve(parent_watch=False, **kwargs)

    def _builtin_dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if method == "shutdown":  # halt the running call before waiting for it
            self.request_stop()
        return super()._builtin_dispatch(method, args, kwargs)

    def _register_rpc(self) -> None:
        for name in METHODS:
            self._rpc[f"env.{name}"] = getattr(self, name)
        register_code_api(self, UR5E_PRIMITIVES)

    def close(self) -> None:
        for cam in self._cameras.values():
            try:
                cam.close()
            except Exception:
                pass
        try:
            self.controller.hold()
        except Exception:
            pass
        for handle in (self._gripper, self._arm):
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass

    # -- read -------------------------------------------------------------

    def capabilities(self) -> dict[str, Any]:
        lim = self.controller.limits
        return {
            "backend": "ur_rtde",
            "has_vla": False,
            "cameras": {
                "main": self._main,
                **{f"extra_{i}": n for i, n in enumerate(self._extras)},
            },
            "has_depth": bool(self._cameras[self._main].has_depth),
            "camera_depth": {n: bool(c.has_depth) for n, c in self._cameras.items()},
            "workspace": {
                "min": list(lim.workspace_min),
                "max": list(lim.workspace_max),
            },
            "z_floor_m": lim.z_floor_m,
            "table_z_m": lim.z_floor_m,
            "max_move_m": lim.max_move_m,
            "max_rotate_rad": lim.max_rotate_rad,
            "servo_step_m": None,
            "servo_step_rad": None,
            "empty_grasp_reopen_m": lim.empty_width_m,
            "arm_id": self.arm_id,
        }

    def get_env_meta(self) -> dict[str, Any]:
        lim = self.controller.limits
        return {
            "ok": True,
            "robot": "ur5e",
            "backend": "ur_rtde",
            "arm_id": self.arm_id,
            "config_path": self._config_path,
            "task_description": self._task,
            "cameras": sorted(self._cameras),
            "main_camera": self._main,
            "gripper": "robotiq" if self._gripper is not None else "none",
            "has_begin_pose": lim.begin_joints is not None,
            "limits": {
                "z_floor_m": lim.z_floor_m,
                "workspace_min": list(lim.workspace_min),
                "workspace_max": list(lim.workspace_max),
                "max_move_m": lim.max_move_m,
                "max_rotate_rad": lim.max_rotate_rad,
                "max_tilt_rad": lim.max_tilt_rad,
                "speed_mps": lim.speed_mps,
                "accel_mps2": lim.accel_mps2,
                "empty_width_m": lim.empty_width_m,
                "gripper_stroke_m": lim.gripper_stroke_m,
            },
            "capabilities": self.capabilities(),
            "tasks": self._cfg.get("tasks") or {},
        }

    def get_robot_state(self) -> dict[str, Any]:
        return {
            "raw_base_state": self.controller.state(),
            "backend": "ur_rtde",
            "arm_id": self.arm_id,
        }

    def get_observation(self) -> dict[str, Any]:
        """Live frames per camera: ``images[name]`` uint8 [H,W,3]; ``depths[name]``
        float32 [H,W] m for cameras with depth; ``timestamps[name]``."""
        images: dict[str, np.ndarray] = {}
        depths: dict[str, np.ndarray] = {}
        stamps: dict[str, float] = {}
        for name, cam in self._cameras.items():
            f = cam.read()
            images[name] = np.ascontiguousarray(f.rgb)
            if f.depth is not None:
                depths[name] = np.asarray(f.depth, dtype=np.float32)
            stamps[name] = float(f.timestamp_s)
        return {"images": images, "depths": depths, "timestamps": stamps}

    def get_camera_meta(self) -> dict[str, Any]:
        return {
            "source": "ur5e_cameras",
            "image_coordinate_convention": "pixel [u, v] maps to array [v, u]",
            "depth_unit": "m",
            "depth_aligned_to_color": True,
            "cameras": self._meta,
            "observation_camera_map": self.capabilities()["cameras"],
            "arm_id": self.arm_id,
        }

    # -- motion -----------------------------------------------------------

    def move_delta(self, delta_xyz: Any) -> dict[str, Any]:
        return self.controller.move_delta(delta_xyz)

    def move_pose(
        self, xyz: Any, rotvec: Any = None, rpy: Any = None
    ) -> dict[str, Any]:
        return self.controller.move_pose(xyz, rotvec=rotvec, rpy=rpy)

    def rotate_delta(self, delta_rpy: Any) -> dict[str, Any]:
        return self.controller.rotate_delta(delta_rpy)

    def set_gripper(self, *, open: bool) -> dict[str, Any]:
        result = self.controller.set_gripper(open=bool(open))
        result["robot_state"] = self.get_robot_state()
        result["states"] = None
        return result

    def reset(self) -> dict[str, Any]:
        """Open the gripper, moveJ to ``calibration.begin_joints`` (moves the arm)."""
        result = self.controller.reset()
        result["robot_state"] = self.get_robot_state()
        result["states"] = None
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arm(cfg: dict[str, Any]) -> tuple[Any, Any]:
    """Connect the arm and (when configured) the Robotiq gripper."""
    from pi_embodied_services.robots.ur5e.hardware import RobotiqGripper, RtdeArm

    rc = cfg.get("robot") or {}
    arm = RtdeArm(str(rc["ip"]), dashboard_port=int(rc.get("dashboard_port", 29999)))
    gc = rc.get("gripper") or {}
    gripper = None
    if str(gc.get("type", "robotiq")) == "robotiq":
        gripper = RobotiqGripper(str(rc["ip"]), int(gc.get("port", 63352)))
    return arm, gripper


def build(
    cfg: dict[str, Any], camera_flag: str = ""
) -> tuple[Any, Any, dict[str, Any]]:
    """Connect the arm, the gripper and every camera named in ``cfg`` / the flag."""
    arm, gripper = build_arm(cfg)
    cameras: dict[str, Any] = {}
    defaults = {k: v for k, v in (cfg.get("cameras") or {}).items() if k != "devices"}
    try:
        for name, dev in camera_devices(cfg, camera_flag)[2].items():
            cameras[name] = open_camera(name, dev, defaults)
    except Exception:
        for cam in cameras.values():
            cam.close()
        if gripper is not None:
            gripper.close()
        arm.close()
        raise
    return arm, gripper, cameras


def build_mock(
    cfg: dict[str, Any], camera_flag: str = ""
) -> tuple[Any, Any, dict[str, Any]]:
    """Test doubles (refuse to construct outside tests; see mock.py)."""
    from pi_embodied_services.robots.ur5e.mock import (
        DOWN,
        MockCamera,
        MockRobotiq,
        MockUrArm,
        require_test_env,
    )

    require_test_env()
    lim = limits_from_config(cfg)
    lo, hi = np.asarray(lim.workspace_min), np.asarray(lim.workspace_max)
    mid = (lo + hi) / 2
    arm = MockUrArm(
        (mid[0], mid[1], min(hi[2], lim.z_floor_m + 0.1), *DOWN),
        serial=str((cfg.get("calibration") or {}).get("arm_id") or "mock"),
    )
    gc = (cfg.get("robot") or {}).get("gripper") or {}
    gripper = MockRobotiq() if str(gc.get("type", "robotiq")) == "robotiq" else None
    cameras = {
        name: MockCamera(
            name,
            depth_m=0.5
            if dev["type"] == "realsense" and dev.get("depth", True)
            else None,
        )
        for name, dev in camera_devices(cfg, camera_flag)[2].items()
    }
    return arm, gripper, cameras


def read_pose(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read-only: TCP pose, joints and gripper width (nothing is commanded)."""
    arm, gripper = build_arm(cfg)
    try:
        pose = np.asarray(arm.tcp_pose(), dtype=np.float64)
        out = {
            "tcp_pose_xyzw": [round(v, 5) for v in pose7_of(pose)],
            "tcp_pose_rotvec": np.round(pose, 5).tolist(),
            "tcp_z_m": round(float(pose[2]), 5),
            "joints_rad": np.round(np.asarray(arm.joints()), 5).tolist(),
            "serial": arm.identity(),
        }
        if gripper is not None:
            try:
                stroke = float((cfg.get("gripper") or {}).get("stroke_m", 0.085))
                out["gripper_width_m"] = round(
                    stroke * (1 - gripper.position() / 255), 5
                )
            except Exception as exc:
                out["gripper_error"] = str(exc)
        return out
    finally:
        if gripper is not None:
            gripper.close()
        arm.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--transport", choices=["http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--robot-config", default=None)
    parser.add_argument(
        "--cameras",
        default="",
        help="Override cameras.devices: name=type:source,... (realsense:<serial>, "
        "webcam:<index|/dev/videoN>, rtsp://<url>); the first is the main camera",
    )
    parser.add_argument("--task-description", default="")
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Validate and print the config, then exit without touching hardware.",
    )
    parser.add_argument(
        "--print-identity",
        action="store_true",
        help="Print the arm's serial number (for calibration.arm_id), then exit; no motion.",
    )
    parser.add_argument(
        "--read-pose",
        action="store_true",
        help="Print the live TCP pose, joints and gripper width (the arm never moves).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Test doubles instead of hardware; refused unless PI_EMBODIED_MOCK_ROBOT=1.",
    )
    args = parser.parse_args(argv)
    if args.mock:
        from pi_embodied_services.robots.ur5e.mock import MOCK_ENV

        if os.environ.get(MOCK_ENV) != "1":
            parser.error(f"--mock is for tests only (needs {MOCK_ENV}=1)")
    if (
        args.read_pose or args.print_identity
    ):  # before validation: how the values are found
        path = Path(args.robot_config or DEFAULT_CONFIG).expanduser()
        raw = yaml.safe_load(path.read_text()) or {}
        if args.print_identity:
            arm, gripper = build_arm(raw)
            try:
                print(f"arm_id: {arm.identity()!r}")
            finally:
                arm.close()
            return 0
        print(json.dumps(read_pose(raw)))
        return 0
    cfg = load_config(args.robot_config, args.cameras)
    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return 0
    arm, gripper, cameras = (
        build_mock(cfg, args.cameras) if args.mock else build(cfg, args.cameras)
    )
    try:
        facade = UR5eEnvFacade(
            cfg,
            arm,
            gripper,
            cameras,
            config_path=str(args.robot_config or DEFAULT_CONFIG),
            camera_flag=args.cameras,
            task_description=args.task_description,
        )
    except Exception:
        for cam in cameras.values():
            cam.close()
        if gripper is not None:
            gripper.close()
        arm.close()
        raise
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
