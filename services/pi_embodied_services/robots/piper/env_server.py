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
# The session wiring follows Show-Harness core/piper/piper_session.py and
# core/piper/config.py (Apache-2.0, github.com/showlab/Show-Harness @137d571).

"""RPC server owning one AgileX Piper arm, its wrist camera and a front camera.

Everything goes over ROS topics (``ros_io``); start the CAN link, the camera driver
and the arm node (mode 1) first. One call runs at a time; ``stop`` is lock-free and
the running motion checks it between joint waypoints / settle ticks / gripper polls
and returns ``cancelled: true`` after holding the arm where it is.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.robots.piper.controller import PiperController, PiperLimits
from pi_embodied_services.utils.logging import get_logger

logger = get_logger("piper_env_server")

DEFAULT_CONFIG = Path(__file__).with_name("config") / "example.yaml"


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load the robot YAML (see config/example.yaml) and validate the limits."""
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text()) or {}
    if cfg.get("arm", "left") not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    limits_from_config(cfg).validate()
    return cfg


def limits_from_config(cfg: dict[str, Any]) -> PiperLimits:
    lim = dict(cfg.get("limits") or {})
    grip = dict(cfg.get("gripper") or {})
    motion = dict(cfg.get("motion") or {})
    cal = dict(cfg.get("calibration") or {})
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


class PiperEnvFacade(BaseEnvFacade):
    """The ``env.*`` protocol for one Piper arm."""

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
        self._robot = robot
        self._cameras = cameras
        self._controller = PiperController(
            robot, limits_from_config(cfg), stop_requested=self.stop_requested
        )
        self._controller.sync()

    def _register_rpc(self) -> None:
        for name in (
            "get_env_meta",
            "reset",
            "get_robot_state",
            "get_observation",
            "get_camera_meta",
            "step",
            "move_joints",
        ):
            self._rpc[f"env.{name}"] = getattr(self, name)

    def close(self) -> None:
        for cam in self._cameras.values():
            cam.close()
        self._robot.close()

    # -- read -------------------------------------------------------------

    def get_env_meta(self) -> dict[str, Any]:
        lim = self._controller.limits
        cal = self._cfg.get("calibration") or {}
        return {
            "ok": True,
            "robot": "piper",
            "arm": self._cfg.get("arm", "left"),
            "config_path": self._config_path,
            "cameras": sorted(self._cameras),
            "units_frame": (self._cfg.get("motion") or {}).get("units_frame", "base"),
            "limits": {
                k: getattr(lim, k)
                for k in (
                    "z_floor_m",
                    "enable_z_floor",
                    "workspace_min",
                    "workspace_max",
                    "max_step_m",
                    "max_yaw_rad",
                    "speed_mps",
                    "open_width_m",
                    "empty_width_m",
                )
            },
            "has_begin_pose": cal.get("begin_joints") is not None,
            "has_rest_pose": cal.get("rest_joints") is not None,
            "motion_backend": self._controller.backend,
            "tasks": self._cfg.get("tasks") or {},
        }

    def get_robot_state(self) -> dict[str, Any]:
        state = self._controller.state()
        status = getattr(self._robot, "arm_status", lambda: None)()
        if status is not None:
            state["arm_status"] = status
        return state

    def get_observation(self) -> dict[str, Any]:
        size = int((self._cfg.get("cameras") or {}).get("image_size") or 0)
        images = {}
        for name, cam in self._cameras.items():
            frame = cam.read()
            images[name] = resize_with_pad(frame, size) if size else frame
        return {"images": images, "robot_state": self.get_robot_state()}

    def get_camera_meta(self) -> dict[str, Any]:
        cams = self._cfg.get("cameras") or {}
        return {
            "cameras": {
                name: {"topic": cams.get(name), "role": name} for name in self._cameras
            },
            "image_size": cams.get("image_size") or None,
        }

    # -- motion -----------------------------------------------------------

    def step(
        self,
        delta_xyz: Any = (0.0, 0.0, 0.0),
        yaw: float = 0.0,
        gripper: str | None = None,
        frame: str = "base",
        reopen_empty: bool = True,
    ) -> dict[str, Any]:
        """One guarded step: translate (m), yaw (rad), then open/close the gripper."""
        return self._controller.step(
            delta_xyz,
            yaw=yaw,
            gripper=gripper,
            frame=frame,
            reopen_empty=reopen_empty,
        )

    def move_joints(self, pose: str = "begin") -> dict[str, Any]:
        """Joint-space move to a calibrated pose: ``begin`` or ``rest``."""
        if pose not in ("begin", "rest"):
            raise ValueError("pose must be 'begin' or 'rest'")
        joints = (self._cfg.get("calibration") or {}).get(f"{pose}_joints")
        if joints is None:
            raise ValueError(
                f"calibration.{pose}_joints is not set: jog the arm to the {pose} pose, "
                f"read /puppet/joint_<arm> position[0:6] and write it to the config"
            )
        return self._controller.move_to_joints(joints)

    def reset(self) -> dict[str, Any]:
        """Open the gripper, then move to the begin pose (both motions)."""
        grip = self._controller.step(gripper="open")
        move = self.move_joints("begin")
        return {
            "ok": bool(move.get("ok")) and not grip.get("cancelled"),
            "gripper": grip,
            "move": move,
            "robot_state": self.get_robot_state(),
            **({"cancelled": True} if move.get("cancelled") else {}),
        }


def build(cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Connect the ROS arm and cameras named in ``cfg``."""
    from pi_embodied_services.robots.piper.ros_io import PiperRosArm, RosImageCamera

    ros = cfg.get("ros") or {}
    robot = PiperRosArm(
        arm=cfg.get("arm", "left"),
        feedback_timeout_s=float(ros.get("feedback_timeout_s", 5.0)),
        max_feedback_age_s=float(ros.get("max_feedback_age_s", 0.5)),
    ).connect()
    cams = cfg.get("cameras") or {}
    cameras = {}
    try:
        for name in ("front", "wrist"):
            if cams.get(name):
                cameras[name] = RosImageCamera(
                    cams[name],
                    max_age_s=float(cams.get("max_age_s", 1.0)),
                    connect_timeout_s=float(cams.get("connect_timeout_s", 10.0)),
                )
    except Exception:
        for cam in cameras.values():
            cam.close()
        robot.close()
        raise
    if not cameras:
        robot.close()
        raise ValueError("configure at least one camera topic (cameras.front / wrist)")
    return robot, cameras


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
    args = parser.parse_args()
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
