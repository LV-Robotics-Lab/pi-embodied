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
# Adapted from Show-Harness core/franka/franka_session.py (FrankaSession: connect the
# NUC, start Cartesian impedance, external + wrist RealSense observations with the
# resize_with_pad letterbox) and core/franka/camera_utils.py (resize_with_pad).
# Modified by pi-embodied: served as the pi-embodied ``franka-env`` RPC protocol (the
# same methods and result shapes as robots/franka/env_server.py, the RLinf backend);
# depth and projection metadata for back_project; safety limits, reset to a begin
# pose and ``stop`` via control.py; test-only --mock; the ``smooth`` section holds
# Show-Harness's smooth plugin settings (core/launch.py smooth_* keys).

"""RPC server owning one Franka on a Polymetis NUC and its two RealSense cameras.

Same ``env.*`` methods and result shapes as the RLinf server
(``robots/franka/env_server.py``), so pi's franka robot drives either one; the
``capabilities`` in ``env.get_env_meta`` say what this backend lacks (no VLA
action space, so no ``env.chunk_step``).

Deploying (the NUC side is not part of Show-Harness either):

1. On the NUC (real-time kernel, libfranka matching the robot firmware), install
   Polymetis (``conda install -c pytorch -c fair-robotics -c aihabitat -c conda-forge
   polymetis``) and start its servers: ``launch_robot.py robot_client=franka_hardware
   robot_client.executable_cfg.robot_ip=<FCI IP>`` and ``launch_gripper.py
   gripper=franka_hand``.
2. Next to them run the ZeroRPC bridge: ``python nuc_server.py --port 4242`` (this
   package's ``nuc_server.py``; it needs ``zerorpc`` in the Polymetis env). Any
   server exposing the same methods works.
3. On the workstation (cameras plugged in here): install the services with the
   ``franka-polymetis`` dependencies (``zerorpc``, ``pyrealsense2``, ``omegaconf``;
   ``opencv-python-headless`` optional; the cameras are read through the shared
   ``components/cameras`` layer, ``realsense-l515`` for an L515), copy ``config/example.yaml``, fill in the
   NUC IP, camera serials (``rs-enumerate-devices | grep Serial``), the Z floor, the
   workspace box, the begin joints (``--read-pose`` prints the live values without
   moving the arm) and the easy_handeye calibration paths, then
   ``pi -e packages/embodied/src/franka --robot-backend polymetis --robot-config <yaml>``.

Only one backend may drive the arm at a time: stop the RLinf/franky stack (it talks
to the FCI directly) before starting Polymetis on the NUC, and vice versa.

Execution: one call at a time on the main thread (the ZeroRPC client is gevent-based
and must stay on the thread that made it). ``stop`` is lock-free; motions poll it
between servo ticks / gripper polls and return ``cancelled: true`` with the arm
holding the last commanded setpoint (at most one ``servo_step_m`` ahead). Parent
death (``--parent-watch``) and ``shutdown`` request a stop first.
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

from pi_embodied_services.components.code_api import register_code_api
from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.robots.franka.primitives import franka_primitives
from pi_embodied_services.robots.franka_polymetis.control import (
    PolymetisController,
    PolymetisLimits,
    flange_to_tcp,
)
from pi_embodied_services.utils import reach
from pi_embodied_services.utils.daemon import watch_parent_death
from pi_embodied_services.utils.detections import state_digest
from pi_embodied_services.utils.grasp import add_grasp_arguments, urls_from_args
from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.perception import (
    FRANKA_CAMERAS,
    Perception,
    franka_intrinsics,
)
from pi_embodied_services.utils.rpc.main_thread_serve import MainThreadServeMixin

logger = get_logger("franka_polymetis_env_server")

DEFAULT_CONFIG = Path(__file__).with_name("config") / "example.yaml"

#: Same ``env.*`` methods as ``robots/franka/env_server.FrankaEnvFacade._METHODS``.
METHODS = (
    "get_env_meta",
    "reset",
    "get_robot_state",
    "get_observation",
    "get_camera_meta",
    "move_delta",
    "rotate_delta",
    "set_gripper",
    "chunk_step",
)

_LIMIT_KEYS = (
    "z_floor_m",
    "workspace_min",
    "workspace_max",
    "max_move_m",
    "max_rotate_rad",
    "servo_step_m",
    "servo_step_rad",
    "tick_s",
    "settle_steps",
    "settle_dt_s",
    "move_tolerance_m",
    "rotate_tolerance_rad",
    "descent_stall_ratio",
    "divergence_resync_m",
    "max_tracking_error_m",
    "max_tilt_rad",
)
_GRIPPER_KEYS = {
    "close_threshold_m": "gripper_close_threshold_m",
    "grasp_open_width_m": "grasp_open_width_m",
    "empty_width_m": "empty_width_m",
    "settle_s": "gripper_settle_s",
    "min_settle_s": "gripper_min_settle_s",
}
_SMOOTH_KEYS = {
    "enabled": "smooth",
    "substeps": "smooth_substeps",
    "dt_s": "smooth_dt_s",
    "blend": "smooth_blend",
    "cruise": "smooth_cruise",
    "chain_window_s": "smooth_chain_window_s",
}
_RESET_KEYS = {
    "begin_joints": "begin_joints",
    "begin_time_s": "begin_time_s",
    "method": "reset_method",
    "lift_m": "reset_lift_m",
}


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


def limits_from_config(cfg: dict[str, Any]) -> PolymetisLimits:
    """Build and validate the limits from a loaded robot YAML."""
    kwargs: dict[str, Any] = {}
    for key, value in _section(cfg, "limits", _LIMIT_KEYS).items():
        kwargs[key] = _tuple(value)
    for key, value in _section(cfg, "gripper", _GRIPPER_KEYS).items():
        kwargs[_GRIPPER_KEYS[key]] = value
    for key, value in _section(cfg, "reset", _RESET_KEYS).items():
        kwargs[_RESET_KEYS[key]] = _tuple(value)
    for key, value in _section(cfg, "smooth", _SMOOTH_KEYS).items():
        kwargs[_SMOOTH_KEYS[key]] = value
    impedance = _section(cfg, "impedance", ("kx", "kxd"))
    for key in ("kx", "kxd"):
        if key in impedance:
            kwargs[key] = _tuple(impedance[key])
    robot = cfg.get("robot") or {}
    if robot.get("tcp_offset_m") is not None:
        kwargs["tcp_offset_m"] = _tuple(robot["tcp_offset_m"])
    if robot.get("tcp_yaw_deg") is not None:
        kwargs["tcp_yaw_deg"] = float(robot["tcp_yaw_deg"])
    limits = PolymetisLimits(**kwargs)
    limits.validate()
    return limits


def camera_roles(cfg: dict[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    """(main camera name, extra camera names sorted, devices) from ``cameras.devices``."""
    devices = (cfg.get("cameras") or {}).get("devices") or {}
    if not isinstance(devices, dict) or not devices:
        raise ValueError("cameras.devices must name the wrist and external RealSense")
    main = [name for name, dev in devices.items() if (dev or {}).get("main")]
    if len(main) != 1:
        raise ValueError("exactly one camera device must set main: true (the wrist)")
    return main[0], sorted(n for n in devices if n != main[0]), devices


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load and validate the robot YAML (see config/example.yaml)."""
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).expanduser().read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError("robot config must be a mapping")
    limits_from_config(cfg)
    camera_roles(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Letterbox (Show-Harness resize_with_pad) with matching depth and intrinsics
# ---------------------------------------------------------------------------


def letterbox_geometry(width: int, height: int, size: int) -> dict[str, Any]:
    """Show-Harness resize_with_pad: scale to fit ``size`` x ``size``, pad centred."""
    scale = min(size / width, size / height)
    new_w, new_h = int(width * scale), int(height * scale)
    return {
        "size": int(size),
        "resized": [new_w, new_h],
        "offset_xy": [(size - new_w) // 2, (size - new_h) // 2],
    }


def _nearest(image: np.ndarray, geo: dict[str, Any]) -> np.ndarray:
    h, w = image.shape[:2]
    new_w, new_h = geo["resized"]
    rows = np.minimum(((np.arange(new_h) + 0.5) * h / new_h).astype(int), h - 1)
    cols = np.minimum(((np.arange(new_w) + 0.5) * w / new_w).astype(int), w - 1)
    return image[rows][:, cols]


def letterbox(image: np.ndarray, geo: dict[str, Any], *, depth: bool) -> np.ndarray:
    """Letterbox RGB (bilinear, like cv2 INTER_LINEAR) or depth (nearest, 0 pad)."""
    new_w, new_h = geo["resized"]
    x0, y0 = geo["offset_xy"]
    size = geo["size"]
    resized = None
    if not depth:
        try:
            import cv2

            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            resized = None
    if resized is None:
        resized = _nearest(image, geo)
    out = np.zeros((size, size, *image.shape[2:]), dtype=image.dtype)
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


def letterbox_intrinsics(raw: dict[str, Any], geo: dict[str, Any] | None) -> list:
    """3x3 K of the emitted image (half-pixel-centre resize, then the pad offset)."""
    if geo is None:
        return [
            [raw["fx"], 0.0, raw["ppx"]],
            [0.0, raw["fy"], raw["ppy"]],
            [0.0, 0.0, 1.0],
        ]
    sx = geo["resized"][0] / raw["width"]
    sy = geo["resized"][1] / raw["height"]
    x0, y0 = geo["offset_xy"]
    return [
        [raw["fx"] * sx, 0.0, (raw["ppx"] + 0.5) * sx - 0.5 + x0],
        [0.0, raw["fy"] * sy, (raw["ppy"] + 0.5) * sy - 0.5 + y0],
        [0.0, 0.0, 1.0],
    ]


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class FrankaPolymetisFacade(MainThreadServeMixin, BaseEnvFacade):
    """The single-Franka ``env.*`` protocol on a Polymetis NUC."""

    SERVICE_NAME = "franka-polymetis-env"

    def __init__(
        self,
        cfg: dict[str, Any],
        robot: Any,
        cameras: dict[str, Any],
        *,
        config_path: str = "",
        task_description: str = "",
        sleep: Callable[[float], None] = time.sleep,
        perception: Perception | None = None,
        grasp: dict | None = None,
        ik_reach: reach.ReachPreview | None = None,
    ) -> None:
        self._perception = perception
        # --graspnet/--graspgenx/--anygrasp/--anyplace: env.plan_grasp, env.plan_place and the
        # grasp/placement ids over the wrist and external cameras, as on the RLinf backend.
        self._grasp_urls = grasp
        # --ik: env.preview_reach, and move_delta / rotate_delta refuse a target the ik
        # service cannot reach from the current joints (utils/reach.py).
        self._reach = ik_reach
        super().__init__()
        self._cfg = cfg
        self._config_path = config_path
        self._task = task_description
        self._robot = robot
        self._cameras = cameras
        self._main, self._extras, devices = camera_roles(cfg)
        missing = sorted({self._main, *self._extras} - set(cameras))
        if missing:
            raise ValueError(f"cameras not connected: {missing}")
        size = int((cfg.get("cameras") or {}).get("image_size") or 0)
        self._meta: dict[str, Any] = {}
        self._geo: dict[str, Any] = {}
        for name, cam in cameras.items():
            raw = cam.intrinsics()
            geo = (
                letterbox_geometry(raw["width"], raw["height"], size) if size else None
            )
            self._geo[name] = geo
            out = [geo["size"]] * 2 if geo else [raw["width"], raw["height"]]
            self._meta[name] = {
                "name": name,
                "serial_number": str((devices.get(name) or {}).get("serial", "")),
                "camera_type": "realsense",
                "raw_resolution": [raw["width"], raw["height"]],
                "output_resolution": out,
                "letterbox": geo,
                "crop_bounds_xyxy": None,
                "depth_scale": float(getattr(cam, "depth_scale", 0.0)),
                "depth_aligned_to_color": True,
                "extrinsic_cam2base": None,
                "extrinsic_cam2ee": None,
                "raw_color_intrinsics": raw,
                "intrinsic_K": letterbox_intrinsics(raw, geo),
            }
        self.controller = PolymetisController(
            robot, limits_from_config(cfg), self.stop_requested, sleep=sleep
        )
        self.controller.start_impedance()

    def serve(self, *, parent_watch: bool = False, **kwargs: Any) -> None:
        """Parent death also stops the running motion or reset (not only the loop)."""

        def on_death() -> None:
            self.request_stop()
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
        self._rpc["env.preview_reach"] = self.preview_reach
        # --sam3 / --unidepth: env.segment, env.select_detection, env.reject_detection,
        # env.enhance_depth over the latest env.get_observation (utils/perception.py). Ids
        # expire when the arm's pose or gripper changed, not on every observation.
        if self._perception is not None:
            self._perception.epoch.set_digest(self._state_digest)
            self._perception.install(self)
        primitives = franka_primitives(self._perception)
        grasp = self._grasp_planner()
        if grasp is not None:
            grasp.install(self)
            primitives = (*primitives, *grasp.primitives())
        register_code_api(self, primitives)

    def _state_digest(self) -> tuple:
        """The arm's TCP pose and gripper, rounded (``utils/detections.state_digest``)."""
        return state_digest(self.controller.state())

    def _grasp_planner(self):
        """The RLinf backend's planner (``franka/grasp_views.franka_grasp_planner``) over this
        server's own observation, camera meta and robot state."""
        from pi_embodied_services.robots.franka.grasp_views import (
            franka_grasp_planner,
        )

        return franka_grasp_planner(
            self,
            self._perception,
            self._grasp_urls,
            state_digest=self._state_digest,
        )

    def close(self) -> None:
        for cam in self._cameras.values():
            try:
                cam.close()
            except Exception:
                pass
        try:
            self._robot.terminate_current_policy()
        except Exception:
            pass
        self._robot.close()

    # -- read -------------------------------------------------------------

    def capabilities(self) -> dict[str, Any]:
        lim = self.controller.limits
        return {
            "backend": "polymetis",
            "has_vla": False,
            "cameras": {
                "main": self._main,
                **{f"extra_{i}": n for i, n in enumerate(self._extras)},
            },
            "has_depth": True,
            "workspace": {
                "min": list(lim.workspace_min),
                "max": list(lim.workspace_max),
            },
            "z_floor_m": lim.z_floor_m,
            # The floor is captured with the closed gripper resting on the table.
            "table_z_m": lim.z_floor_m,
            "max_move_m": lim.max_move_m,
            "max_rotate_rad": lim.max_rotate_rad,
            "servo_step_m": lim.servo_step_m,
            "servo_step_rad": lim.servo_step_rad,
            "empty_grasp_reopen_m": lim.empty_width_m,
        }

    def get_env_meta(self) -> dict[str, Any]:
        return {
            "ok": True,
            "action_dim": None,
            "action_scale": None,
            "use_relative_frame": False,
            "backend": "polymetis",
            "config_path": self._config_path,
            "task_description": self._task,
            "capabilities": self.capabilities(),
            "smooth": self.smooth_meta(),
        }

    def smooth_meta(self) -> dict[str, Any]:
        """The smooth-motion settings; ``chaining`` means move_delta takes
        ``continuous`` (the RLinf backend does not)."""
        lim = self.controller.limits
        return {
            "enabled": lim.smooth,
            "substeps": lim.smooth_substeps,
            "dt_s": lim.smooth_dt_s,
            "duration_s": lim.smooth_substeps * lim.smooth_dt_s,
            "blend": lim.smooth_blend,
            "cruise": lim.smooth_cruise,
            "chain_window_s": lim.smooth_chain_window_s,
            "chaining": lim.smooth and lim.smooth_blend,
        }

    def get_robot_state(self) -> dict[str, Any]:
        return {
            "raw_base_state": self.controller.state(),
            "action_dim": None,
            "action_scale": None,
            "use_relative_frame": False,
            "backend": "polymetis",
            "controller_restarts": self.controller.restarts,
        }

    def get_observation(self) -> dict[str, Any]:
        """Live frames under the RLinf keys: main = wrist, extra_view = the others."""
        try:
            frames = {name: cam.read() for name, cam in self._cameras.items()}
        except Exception as exc:
            logger.warning("live camera read failed: %s", exc)
            return {}

        def emit(name: str) -> tuple[np.ndarray, np.ndarray]:
            rgb, depth = frames[name].rgb, frames[name].depth
            geo = self._geo[name]
            if geo is None:
                return np.ascontiguousarray(rgb), depth.astype(np.float32)
            return (
                letterbox(rgb, geo, depth=False),
                letterbox(depth.astype(np.float32), geo, depth=True),
            )

        main_rgb, main_depth = emit(self._main)
        out: dict[str, Any] = {"main_images": main_rgb, "main_depths": main_depth}
        if self._extras:
            extras = [emit(name) for name in self._extras]
            out["extra_view_images"] = np.stack([e[0] for e in extras], axis=0)
            out["extra_view_depths"] = np.stack([e[1] for e in extras], axis=0)
        return out

    def get_camera_meta(self) -> dict[str, Any]:
        return {
            "source": "polymetis_realsense",
            "image_coordinate_convention": "pixel [u, v] maps to array [v, u]",
            "depth_unit": "m",
            "depth_aligned_to_color": True,
            "cameras": self._meta,
            "observation_camera_map": self.capabilities()["cameras"],
        }

    # -- reach preview (--ik) ---------------------------------------------

    def preview_reach(self, pos: Any, quat_xyzw: Any = None) -> dict[str, Any]:
        """Whether the TCP can reach a base-frame pose from the current joints (IK only, the
        arm does not move); ``quat_xyzw`` None keeps the current orientation."""
        if self._reach is None:
            return reach.no_service()
        state = self.controller.state()
        tcp = np.asarray(state["tcp_pose"], dtype=np.float64)
        return self._reach.preview(
            state["arm_joint_position"],
            pos,
            tcp[3:] if quat_xyzw is None else quat_xyzw,
        )

    def _check_reach(self, action: str, **delta: Any) -> None:
        """Refuse a motion whose end pose the ik service cannot reach (nothing is commanded)."""
        if self._reach is None:
            return
        state = self.controller.state()
        pos, quat = reach.delta_target(state["tcp_pose"], **delta)
        reach.require_reachable(
            self._reach.preview(state["arm_joint_position"], pos, quat), f"env.{action}"
        )

    # -- motion -----------------------------------------------------------

    def move_delta(self, delta_xyz: Any, continuous: bool = False) -> dict[str, Any]:
        self._check_reach("move_delta", delta_xyz=delta_xyz)
        return self.controller.move_delta(delta_xyz, continuous=bool(continuous))

    def rotate_delta(self, delta_rpy: Any) -> dict[str, Any]:
        self._check_reach("rotate_delta", delta_rpy=delta_rpy)
        return self.controller.rotate_delta(delta_rpy)

    def set_gripper(self, *, open: bool) -> dict[str, Any]:
        result = self.controller.set_gripper(open=bool(open))
        result["robot_state"] = self.get_robot_state()
        result["states"] = None
        return result

    def reset(self) -> dict[str, Any]:
        """Open the gripper, lift, move to ``reset.begin_joints`` (moves the arm)."""
        result = self.controller.reset()
        result["robot_state"] = self.get_robot_state()
        result["states"] = None
        return result

    def chunk_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ValueError(
            "env.chunk_step: the polymetis backend has no VLA action space "
            "(capabilities.has_vla is false); use the rlinf backend for vla_grasp"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build(cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Connect the NUC and the RealSense cameras named in ``cfg``."""
    from pi_embodied_services.components.cameras.realsense import RealSenseRGBD
    from pi_embodied_services.robots.franka_polymetis.hardware import PolymetisRobot

    rc = cfg.get("robot") or {}
    if not rc.get("nuc_ip"):
        raise ValueError("robot.nuc_ip is required")
    robot = PolymetisRobot(
        str(rc["nuc_ip"]),
        int(rc.get("nuc_port", 4242)),
        heartbeat_s=rc.get("heartbeat_s", 20.0),
        timeout_s=float(rc.get("rpc_timeout_s", 30.0)),
    )
    cams_cfg = cfg.get("cameras") or {}
    cameras: dict[str, Any] = {}
    try:
        for name, dev in camera_roles(cfg)[2].items():
            if not (dev or {}).get("serial"):
                raise ValueError(f"cameras.devices.{name}.serial is required")
            cameras[name] = RealSenseRGBD(
                str(dev["serial"]),
                width=int(dev.get("width", cams_cfg.get("width", 640))),
                height=int(dev.get("height", cams_cfg.get("height", 480))),
                fps=int(dev.get("fps", cams_cfg.get("fps", 30))),
            )
    except Exception:
        for cam in cameras.values():
            cam.close()
        robot.close()
        raise
    return robot, cameras


def build_mock(cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Test doubles (refuse to construct outside tests; see mock.py)."""
    from pi_embodied_services.robots.franka_polymetis.mock import (
        MockPolymetisRobot,
        MockRGBD,
        require_test_env,
    )

    require_test_env()
    lim = limits_from_config(cfg)
    lo, hi = np.asarray(lim.workspace_min), np.asarray(lim.workspace_max)
    mid = (lo + hi) / 2
    pose = [mid[0], mid[1], min(hi[2], lim.z_floor_m + 0.1), 1.0, 0.0, 0.0, 0.0]
    home = [mid[0], mid[1], min(hi[2], lim.z_floor_m + 0.15), 1.0, 0.0, 0.0, 0.0]
    robot = MockPolymetisRobot(pose, home_pose=home)
    return robot, {name: MockRGBD(name) for name in camera_roles(cfg)[2]}


def read_pose(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read-only: TCP pose, joints and gripper width (no controller is started)."""
    from pi_embodied_services.robots.franka_polymetis.hardware import PolymetisRobot

    rc = cfg.get("robot") or {}
    if not rc.get("nuc_ip"):
        raise ValueError("robot.nuc_ip is required")
    robot = PolymetisRobot(str(rc["nuc_ip"]), int(rc.get("nuc_port", 4242)))
    try:
        offset = rc.get("tcp_offset_m") or (0.0, 0.0, 0.0)
        yaw = float(rc.get("tcp_yaw_deg") or 0.0)
        pos, quat = flange_to_tcp(robot.get_ee_pose(), offset, yaw)
        return {
            "tcp_pose_xyzw": np.concatenate([pos, quat]).round(5).tolist(),
            "tcp_z_m": round(float(pos[2]), 5),
            "joints_rad": np.asarray(robot.get_joint_positions()).round(5).tolist(),
            "gripper_width_m": round(float(robot.get_gripper_position()[0]), 5),
        }
    finally:
        robot.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--transport", choices=["http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--robot-config", default=None)
    parser.add_argument("--task-description", default="")
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Validate and print the config, then exit without touching hardware.",
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
    parser.add_argument(
        "--sam3", default="", help="SAM3 server URL: adds env.segment and detection ids"
    )
    parser.add_argument(
        "--unidepth", default="", help="UniDepth server URL: adds env.enhance_depth"
    )
    add_grasp_arguments(parser)
    reach.add_ik_argument(parser)
    args = parser.parse_args(argv)
    if args.mock:
        from pi_embodied_services.robots.franka_polymetis.mock import MOCK_ENV

        if os.environ.get(MOCK_ENV) != "1":
            parser.error(f"--mock is for tests only (needs {MOCK_ENV}=1)")
    if args.read_pose:  # before validation: it is how the floor/begin pose are found
        path = Path(args.robot_config or DEFAULT_CONFIG).expanduser()
        print(json.dumps(read_pose(yaml.safe_load(path.read_text()) or {})))
        return 0
    cfg = load_config(args.robot_config)
    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return 0
    robot, cameras = build_mock(cfg) if args.mock else build(cfg)
    try:
        facade = FrankaPolymetisFacade(
            cfg,
            robot,
            cameras,
            config_path=str(args.robot_config or DEFAULT_CONFIG),
            task_description=args.task_description,
            perception=Perception.from_urls(
                sam3=args.sam3,
                unidepth=args.unidepth,
                cameras=FRANKA_CAMERAS,
                # Resolved per call, once the facade exists.
                intrinsics=lambda key: franka_intrinsics(facade.get_camera_meta(), key),
            ),
            grasp=urls_from_args(args),
            ik_reach=reach.reach_from_args(args, "panda"),
        )
    except Exception:
        for cam in cameras.values():
            cam.close()
        robot.close()
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
