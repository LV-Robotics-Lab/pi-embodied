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
# After OpenETA real/cameras/webcam.py (V4L2 for /dev paths, MJPG request).
# Modified by pi-embodied: the Camera/Frame interface, config intrinsics, and an
# RTSP reader thread that keeps only the latest frame.

"""RGB-only sources through OpenCV: USB/UVC webcams and RTSP streams.

Neither has depth: ``Frame.depth`` is None, and depth comes from the UniDepth
component (``enhance_depth``) when a robot needs it. Intrinsics come from the config
(``intrinsics: {fx, fy, ppx, ppy}``, a one-off checkerboard calibration) or are None.
``opencv-python-headless`` is imported lazily (the services' ``cameras`` extra).
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from pi_embodied_services.components.cameras.base import Camera, Frame


def _rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame_bgr[:, :, ::-1])


class WebcamRGB(Camera):
    """A ``cv2.VideoCapture`` webcam: ``device`` is an index or a ``/dev/videoN`` path."""

    kind = "webcam"
    has_depth = False

    def __init__(
        self,
        device: int | str,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        intrinsics: dict[str, Any] | None = None,
        read_retries: int = 3,
    ) -> None:
        import cv2

        self.device = device
        self._intrinsics = intrinsics
        self.read_retries = max(1, int(read_retries))
        # V4L2 for /dev paths: OpenCV otherwise tries GStreamer first and logs errors.
        if isinstance(device, str) and device.startswith("/dev/"):
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise RuntimeError(f"webcam {device!r} did not open")
        # Many UVC webcams only stream fast as MJPG.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_FPS, int(fps))
        self.cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(width)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(height)

    def intrinsics(self) -> dict[str, Any] | None:
        return dict(self._intrinsics) if self._intrinsics else None

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "device": str(self.device)}

    def read(self) -> Frame:
        for _ in range(self.read_retries):
            ok, bgr = self.cap.read()
            if ok and bgr is not None:
                return Frame(rgb=_rgb(bgr), depth=None, timestamp_s=time.time())
            time.sleep(0.02)
        raise RuntimeError(f"webcam {self.device!r} returned no frame")

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            finally:
                self.cap = None


class RtspRGB(Camera):
    """An RTSP (or any ffmpeg URL) stream.

    A reader thread drains the stream continuously and keeps the latest decoded frame,
    so ``read`` returns the newest image instead of the one queued when the previous
    call ended (an RTSP source buffers several seconds otherwise). ``read`` waits for
    a frame newer than the last one returned, up to ``timeout_s``.
    """

    kind = "rtsp"
    has_depth = False

    def __init__(
        self,
        url: str,
        *,
        intrinsics: dict[str, Any] | None = None,
        timeout_s: float = 5.0,
        open_timeout_s: float = 10.0,
    ) -> None:
        import cv2

        self.url = url
        self._intrinsics = intrinsics
        self.timeout_s = float(timeout_s)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"RTSP stream {url!r} did not open")
        self.cap = cap
        self._lock = threading.Condition()
        self._latest: Frame | None = None
        self._seq = 0
        self._error: str | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._pump, daemon=True, name="rtsp")
        self._thread.start()
        with self._lock:
            if not self._lock.wait_for(
                lambda: self._latest is not None or self._error, timeout=open_timeout_s
            ):
                self.close()
                raise RuntimeError(
                    f"RTSP stream {url!r} sent no frame in {open_timeout_s}s"
                )
            if self._error:
                self.close()
                raise RuntimeError(f"RTSP stream {url!r}: {self._error}")

    def _pump(self) -> None:
        failures = 0
        while not self._closed:
            ok, bgr = self.cap.read()
            if not ok or bgr is None:
                failures += 1
                if failures >= 30:
                    with self._lock:
                        self._error = "stream ended"
                        self._lock.notify_all()
                    return
                time.sleep(0.05)
                continue
            failures = 0
            frame = Frame(rgb=_rgb(bgr), depth=None, timestamp_s=time.time())
            with self._lock:
                self._latest = frame
                self._seq += 1
                self._lock.notify_all()

    def intrinsics(self) -> dict[str, Any] | None:
        return dict(self._intrinsics) if self._intrinsics else None

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "url": self.url}

    def read(self) -> Frame:
        with self._lock:
            seq = self._seq
            if not self._lock.wait_for(
                lambda: self._seq > seq or self._error, timeout=self.timeout_s
            ):
                raise RuntimeError(
                    f"RTSP stream {self.url!r}: no new frame in {self.timeout_s}s"
                )
            if self._error:
                raise RuntimeError(f"RTSP stream {self.url!r}: {self._error}")
            assert self._latest is not None
            return self._latest

    def close(self) -> None:
        self._closed = True
        cap, self.cap = self.cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)
