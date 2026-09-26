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
# Modified by pi-embodied: the Camera/Frame interface, config intrinsics, the webcam
# drains its capture buffer before every read and reopens the device after a failed
# read, and an RTSP reader thread that keeps only the latest frame and reconnects.

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
    """A ``cv2.VideoCapture`` webcam: ``device`` is an index or a ``/dev/videoN`` path.

    V4L2 queues a few frames in the driver, so a plain ``cap.read()`` after the arm
    moved can return an image captured before the motion. ``read`` first drains the
    queue: ``grab`` calls that return at once are buffered frames and are discarded
    (up to ``drain_frames``); the first grab that has to wait a frame period is live
    and is the one decoded. A failed read reopens the device once before giving up.
    """

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
        drain_frames: int = 5,
    ) -> None:
        self.device = device
        self._intrinsics = intrinsics
        self.read_retries = max(1, int(read_retries))
        self.drain_frames = max(0, int(drain_frames))
        self._want = (int(width), int(height), int(fps))
        self.cap: Any = None
        self.reopens = 0
        self._open()

    def _open(self) -> None:
        import cv2

        width, height, fps = self._want
        # V4L2 for /dev paths: OpenCV otherwise tries GStreamer first and logs errors.
        if isinstance(self.device, str) and self.device.startswith("/dev/"):
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            raise RuntimeError(f"webcam {self.device!r} did not open")
        # Many UVC webcams only stream fast as MJPG.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        # The smallest queue the backend allows (best effort; V4L2 honours it).
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(fps)

    def _reopen(self) -> None:
        self.close()
        self.reopens += 1
        time.sleep(0.2)
        self._open()

    def intrinsics(self) -> dict[str, Any] | None:
        return dict(self._intrinsics) if self._intrinsics else None

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "device": str(self.device)}

    def _read_once(self) -> Frame | None:
        """Drain the queue, then decode the live frame; None when the device gave
        no frame."""
        period = 1.0 / max(1.0, self.fps)
        for _ in range(self.drain_frames):
            t0 = time.perf_counter()
            if not self.cap.grab():
                return None
            if time.perf_counter() - t0 >= 0.5 * period:
                break  # the grab waited for the sensor: this frame is live
        else:
            if not self.cap.grab():
                return None
        captured = time.monotonic()
        ok, bgr = self.cap.retrieve()
        if not ok or bgr is None:
            return None
        return Frame(
            rgb=_rgb(bgr), depth=None, timestamp_s=time.time(), monotonic_s=captured
        )

    def read(self) -> Frame:
        for attempt in range(2):
            if self.cap is None:
                self._open()
            for _ in range(self.read_retries):
                try:
                    frame = self._read_once()
                except Exception:
                    frame = None
                if frame is not None:
                    return frame
                time.sleep(0.02)
            if attempt == 0:
                try:
                    self._reopen()
                except Exception as exc:
                    raise RuntimeError(
                        f"webcam {self.device!r} returned no frame and did not reopen: {exc}"
                    ) from exc
        raise RuntimeError(f"webcam {self.device!r} returned no frame after reopening")

    def read_fresh(self) -> Frame:
        """``read`` already drains the queue."""
        return self.read()

    def close(self) -> None:
        cap, self.cap = self.cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


class RtspRGB(Camera):
    """An RTSP (or any ffmpeg URL) stream.

    A reader thread drains the stream continuously and keeps the latest decoded frame,
    so ``read`` returns the newest image instead of the one queued when the previous
    call ended (an RTSP source buffers several seconds otherwise). ``read`` waits for
    a frame that arrived after the call started, up to ``timeout_s``. When the stream
    ends the thread reopens it (``reconnect_s`` between attempts); reads fail while it
    is down and work again once frames flow.
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
        reconnect_s: float = 2.0,
    ) -> None:
        import cv2

        self.url = url
        self._intrinsics = intrinsics
        self.timeout_s = float(timeout_s)
        self.reconnect_s = float(reconnect_s)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"RTSP stream {url!r} did not open")
        self.cap = cap
        self.reconnects = 0
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

    def _reconnect(self) -> None:
        """Reopen the capture; ``_error`` stays set until a frame arrives."""
        import cv2

        cap, self.cap = self.cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        time.sleep(self.reconnect_s)
        if self._closed:
            return
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            self.cap = cap
            self.reconnects += 1

    def _pump(self) -> None:
        failures = 0
        while not self._closed:
            cap = self.cap
            ok, bgr = cap.read() if cap is not None else (False, None)
            if not ok or bgr is None:
                failures += 1
                if cap is None or failures >= 30:
                    with self._lock:
                        self._error = "stream ended; reconnecting"
                        self._lock.notify_all()
                    failures = 0
                    self._reconnect()
                    continue
                time.sleep(0.05)
                continue
            failures = 0
            frame = Frame(
                rgb=_rgb(bgr),
                depth=None,
                timestamp_s=time.time(),
                monotonic_s=time.monotonic(),
            )
            with self._lock:
                self._error = None
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
            if self._error:
                raise RuntimeError(f"RTSP stream {self.url!r}: {self._error}")
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

    def read_fresh(self) -> Frame:
        """``read`` already waits for a frame that arrived after the call."""
        return self.read()

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
