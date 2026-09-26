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
# Adapted from Show-Harness core/franka/camera_utils.py (RealSenseCamera,
# hardware_reset_device). Modified by pi-embodied: RGB plus depth aligned to colour,
# colour intrinsics and depth scale reported (back_project needs both), rgb8 stream
# (no OpenCV), retries with a USB hardware reset, and (from OpenETA real/cameras/
# realsense.py) a separate depth stream format for the L515, whose depth and colour
# streams share no resolution. Moved here from robots/franka_polymetis/hardware.py so
# every real robot reads RealSense frames through one driver.

"""Intel RealSense D400 series and L515 as a :class:`Camera`.

``pyrealsense2`` is imported lazily. The L515 is supported only by pyrealsense2
2.54.1 (later releases dropped it): install the services' ``realsense-l515`` extra,
which pins that version, when an L515 is in the rig. The D400 family works with
any 2.5x release.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from pi_embodied_services.components.cameras.base import Camera, Frame
from pi_embodied_services.utils.logging import get_logger

logger = get_logger("cameras.realsense")

#: pyrealsense2 release that still supports the L515 (see the module docstring).
L515_PYREALSENSE = "2.54.1"


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


class RealSenseRGBD(Camera):
    """One RealSense (D435 external, D405 wrist, L515): RGB plus depth aligned to colour.

    ``serial`` None opens the first device. ``depth_width/height/fps`` request the depth
    stream at its own format (0 = the colour format); the L515 needs this (its depth
    runs at 640x480 or 1024x768 while colour runs at 1280x720 or 1920x1080).
    """

    kind = "realsense"

    def __init__(
        self,
        serial: str | None,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        depth: bool = True,
        depth_width: int = 0,
        depth_height: int = 0,
        depth_fps: int = 0,
        read_timeout_ms: int = 3000,
        read_retries: int = 2,
        start_retries: int = 3,
        warmup_frames: int = 3,
    ) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        self.serial = str(serial) if serial else None
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.has_depth = bool(depth)
        self.depth_format = (
            int(depth_width or width),
            int(depth_height or height),
            int(depth_fps or fps),
        )
        self.read_timeout_ms = int(read_timeout_ms)
        self.read_retries = max(1, int(read_retries))
        self.warmup_frames = int(warmup_frames)
        self.pipeline: Any = None
        self.profile: Any = None
        self.depth_scale = 0.0
        self.model = ""
        self.align = rs.align(rs.stream.color) if self.has_depth else None
        for attempt in range(1, start_retries + 1):
            try:
                self._start()
                return
            except Exception as exc:
                logger.warning(
                    "RealSense %s start attempt %d/%d failed: %s",
                    self.serial or "default",
                    attempt,
                    start_retries,
                    exc,
                )
                self._stop()
                if attempt == start_retries:
                    raise RuntimeError(
                        f"RealSense {self.serial or 'default'} did not start; check the "
                        "USB 3 cable/port and `rs-enumerate-devices | grep Serial`"
                        + (
                            f" (an L515 needs pyrealsense2 {L515_PYREALSENSE}, the "
                            "realsense-l515 extra)"
                            if "L515" in self.model
                            else ""
                        )
                    ) from exc
                _hardware_reset(self.serial)

    def _start(self) -> None:
        rs = self._rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
        )
        if self.has_depth:
            dw, dh, dfps = self.depth_format
            config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)
        self.profile = self.pipeline.start(config)
        device = self.profile.get_device()
        self.model = str(device.get_info(rs.camera_info.name))
        if not self.serial:
            self.serial = str(device.get_info(rs.camera_info.serial_number))
        if self.has_depth:
            self.depth_scale = float(device.first_depth_sensor().get_depth_scale())
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
        """Colour-stream intrinsics (the depth is aligned to it)."""
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
            # str(rs.distortion.x) is "distortion.x"; keep the librealsense name.
            "distortion_model": str(i.model).split(".")[-1],
            "coeffs": [float(c) for c in i.coeffs],
        }

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "serial_number": self.serial,
            "model": self.model,
            "depth_scale": self.depth_scale,
            "depth_aligned_to_color": self.has_depth,
        }

    def _read_once(self) -> Frame:
        frames = self.pipeline.wait_for_frames(timeout_ms=self.read_timeout_ms)
        captured = time.monotonic()
        if self.align is not None:
            frames = self.align.process(frames)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("incomplete RealSense frameset")
        rgb = np.asanyarray(color.get_data()).copy()
        depth_m = None
        if self.has_depth:
            depth = frames.get_depth_frame()
            if not depth:
                raise RuntimeError("incomplete RealSense frameset")
            depth_m = (
                np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
            )
        return Frame(
            rgb=rgb, depth=depth_m, timestamp_s=time.time(), monotonic_s=captured
        )

    def read(self) -> Frame:
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
