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
# Adapted from Show-Harness core/franka/franka_interface.py (FrankaInterface) and
# core/franka/camera_utils.py (RealSenseCamera, hardware_reset_device).
# Modified by pi-embodied: the camera streams RGB plus depth aligned to color and
# reports its color intrinsics and depth scale (back_project needs both); frames are
# delivered in RGB order (rgb8 stream, no OpenCV); prints go to the logger; the
# ZeroRPC client takes an explicit call timeout.

"""Hardware handles: the NUC's Polymetis ``franka_server`` and RealSense RGB-D cameras.

Both import their drivers lazily (``zerorpc``, ``pyrealsense2``) so the env server and
its tests import without them.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from pi_embodied_services.utils.logging import get_logger

logger = get_logger("franka_polymetis_hw")


class PolymetisRobot:
    """ZeroRPC client of the Polymetis ``franka_server`` on the Franka NUC (port 4242).

    The method surface is Show-Harness's ``FrankaInterface``; ``nuc_server.py`` in this
    package is a reference implementation of the server side.
    """

    def __init__(
        self,
        ip: str,
        port: int = 4242,
        *,
        heartbeat_s: float | None = 20.0,
        timeout_s: float = 30.0,
    ) -> None:
        import zerorpc

        self.server = zerorpc.Client(heartbeat=heartbeat_s, timeout=timeout_s)
        self.server.connect(f"tcp://{ip}:{int(port)}")
        self.endpoint = f"tcp://{ip}:{int(port)}"

    def get_ee_pose(self) -> np.ndarray:
        return np.asarray(self.server.get_ee_pose(), dtype=np.float64)

    def get_joint_positions(self) -> np.ndarray:
        return np.asarray(self.server.get_joint_positions(), dtype=np.float64)

    def move_to_joint_positions(self, positions: Any, time_to_go: float) -> None:
        self.server.move_to_joint_positions(
            np.asarray(positions, dtype=float).tolist(), float(time_to_go)
        )

    def start_cartesian_impedance(self, Kx: Any, Kxd: Any) -> None:
        self.server.start_cartesian_impedance(
            np.asarray(Kx, dtype=float).tolist(), np.asarray(Kxd, dtype=float).tolist()
        )

    def start_joint_impedance(self, Kq: Any = None, Kqd: Any = None) -> None:
        self.server.start_joint_impedance(
            None if Kq is None else np.asarray(Kq, dtype=float).tolist(),
            None if Kqd is None else np.asarray(Kqd, dtype=float).tolist(),
        )

    def update_desired_ee_pose(self, pose: Any) -> None:
        self.server.update_desired_ee_pose(np.asarray(pose, dtype=float).tolist())

    def update_desired_joint_pos(self, pos: Any) -> None:
        self.server.update_desired_joint_pos(np.asarray(pos, dtype=float).tolist())

    def control_gripper(self, close: bool) -> None:
        """True closes (grasp), False opens."""
        self.server.control_gripper(bool(close))

    def get_gripper_position(self) -> np.ndarray:
        return np.asarray(self.server.get_gripper_position(), dtype=np.float64).reshape(
            1
        )

    def terminate_current_policy(self) -> None:
        self.server.terminate_current_policy()

    def close(self) -> None:
        try:
            self.server.close()
        except Exception:
            pass


def _hardware_reset(serial: str | None, settle_s: float = 5.0, timeout_s: float = 20.0):
    """Power-cycle a RealSense over USB and wait for it to re-enumerate (best effort)."""
    import pyrealsense2 as rs

    try:
        for dev in rs.context().query_devices():
            if serial is None or dev.get_info(rs.camera_info.serial_number) == serial:
                logger.warning("hardware_reset on RealSense %s", serial or "default")
                dev.hardware_reset()
                break
        else:
            return
    except Exception as exc:
        logger.warning("RealSense hardware_reset failed: %s", exc)
        return
    time.sleep(settle_s)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            serials = {
                d.get_info(rs.camera_info.serial_number)
                for d in rs.context().query_devices()
            }
        except Exception:
            serials = set()
        if serial is None or serial in serials:
            return
        time.sleep(1.0)


class RealSenseRGBD:
    """One RealSense (D435 external / D405 wrist): RGB plus depth aligned to color."""

    def __init__(
        self,
        serial: str,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        read_timeout_ms: int = 3000,
        read_retries: int = 2,
        start_retries: int = 3,
        warmup_frames: int = 3,
    ) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        self.serial = str(serial)
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.read_timeout_ms = int(read_timeout_ms)
        self.read_retries = max(1, int(read_retries))
        self.warmup_frames = int(warmup_frames)
        self.pipeline: Any = None
        self.profile: Any = None
        self.align = rs.align(rs.stream.color)
        for attempt in range(1, start_retries + 1):
            try:
                self._start()
                return
            except Exception as exc:
                logger.warning(
                    "RealSense %s start attempt %d/%d failed: %s",
                    self.serial,
                    attempt,
                    start_retries,
                    exc,
                )
                self._stop()
                if attempt == start_retries:
                    raise RuntimeError(
                        f"RealSense {self.serial} did not start; check the USB 3 "
                        "cable/port and `rs-enumerate-devices | grep Serial`"
                    ) from exc
                _hardware_reset(self.serial)

    def _start(self) -> None:
        rs = self._rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
        )
        config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        self.profile = self.pipeline.start(config)
        self.depth_scale = float(
            self.profile.get_device().first_depth_sensor().get_depth_scale()
        )
        for _ in range(self.warmup_frames):
            self.pipeline.wait_for_frames(timeout_ms=self.read_timeout_ms)

    def _stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = None

    def intrinsics(self) -> dict[str, Any]:
        """Color-stream intrinsics (the depth is aligned to it)."""
        rs = self._rs
        i = (
            self.profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        return {
            "width": int(i.width),
            "height": int(i.height),
            "fx": float(i.fx),
            "fy": float(i.fy),
            "ppx": float(i.ppx),
            "ppy": float(i.ppy),
            "distortion_model": str(i.model),
            "coeffs": [float(c) for c in i.coeffs],
        }

    def _read_once(self) -> tuple[np.ndarray, np.ndarray]:
        frames = self.pipeline.wait_for_frames(timeout_ms=self.read_timeout_ms)
        frames = self.align.process(frames)
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("incomplete RealSense frameset")
        rgb = np.asanyarray(color.get_data()).copy()
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        return rgb, depth_m

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """(rgb uint8 [H,W,3], depth float32 [H,W] in metres, 0 = no return)."""
        error: Exception | None = None
        for _ in range(self.read_retries):
            try:
                return self._read_once()
            except Exception as exc:
                error = exc
                time.sleep(0.05)
        logger.warning("RealSense %s read failed (%s); restarting", self.serial, error)
        self._stop()
        time.sleep(0.2)
        self._start()
        return self._read_once()

    def close(self) -> None:
        self._stop()
