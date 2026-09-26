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
# read, and an RTSP reader thread that keeps only the latest frame and reconnects;
# reads are bounded by a timeout, close is idempotent and thread-safe, and frames
# carry the driver's (V4L2) or the stream's (RTSP PTS) capture time.

"""RGB-only sources through OpenCV: USB/UVC webcams and RTSP streams.

Neither has depth: ``Frame.depth`` is None, and depth comes from the UniDepth
component (``enhance_depth``) when a robot needs it. Intrinsics come from the config
(``intrinsics: {fx, fy, ppx, ppy}``, a one-off checkerboard calibration) or are None.
``opencv-python-headless`` is imported lazily (the services' ``cameras`` extra).

Capture times (``Frame.time_source``): a V4L2 webcam on Linux reports the kernel's
buffer timestamp (``CAP_PROP_POS_MSEC``, taken on ``CLOCK_MONOTONIC`` when the frame
started arriving, before it waited in the driver queue): ``backend``. Other webcam
backends report no usable capture time: ``host`` (the dequeue time). An RTSP frame's
presentation time is mapped to the host clock (:class:`PtsClock`): ``stream_pts``.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from typing import Any

import numpy as np

from pi_embodied_services.components.cameras.base import Camera, Frame, capture_times


def _rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame_bgr[:, :, ::-1])


def _ffmpeg_timeout_params(cv2: Any, open_ms: int, read_ms: int) -> list[int]:
    """``VideoCapture`` open/read timeouts for the FFmpeg backend (OpenCV >= 4.5.2;
    older builds lack the properties and get none)."""
    params: list[int] = []
    for name, ms in (
        ("CAP_PROP_OPEN_TIMEOUT_MSEC", open_ms),
        ("CAP_PROP_READ_TIMEOUT_MSEC", read_ms),
    ):
        prop = getattr(cv2, name, None)
        if prop is not None:
            params += [int(prop), int(ms)]
    return params


class PtsClock:
    """Maps an RTSP stream's presentation timestamps to ``time.monotonic``.

    OpenCV exposes the PTS (``CAP_PROP_POS_MSEC``, ms since the stream start) but not
    the RTCP sender reports that would tie it to wall time, so the mapping is the
    smallest ``arrival - pts`` seen so far: the least-delayed frame defines "live",
    and a frame that arrives later than that (it sat in a buffer) is dated back by
    the difference. The offset may creep up by ``drift_per_s`` per second so a camera
    clock running slow against the host does not make every frame look old; a
    buffer that builds up faster than that still shows. A PTS that is missing, does
    not advance or jumps back (a reconnect) re-anchors the mapping.
    """

    def __init__(self, drift_per_s: float = 1e-3) -> None:
        self.drift_per_s = float(drift_per_s)
        self.reset()

    def reset(self) -> None:
        self._offset: float | None = None
        self._last_pts: float | None = None
        self._last_arrival = 0.0

    def capture(self, pts_ms: float | None, arrival: float) -> float | None:
        """The capture time on ``time.monotonic`` of a frame with ``pts_ms`` that
        arrived at ``arrival``, or None when the PTS cannot be used."""
        if pts_ms is None or not math.isfinite(pts_ms) or pts_ms <= 0:
            self.reset()
            return None
        pts = float(pts_ms) / 1000.0
        if self._last_pts is not None and pts <= self._last_pts:
            self.reset()
        seen = arrival - pts
        if self._offset is None:
            self._offset = seen
        else:
            relaxed = self._offset + self.drift_per_s * max(
                0.0, arrival - self._last_arrival
            )
            self._offset = min(relaxed, seen)
        self._last_pts, self._last_arrival = pts, arrival
        return pts + self._offset


class WebcamRGB(Camera):
    """A ``cv2.VideoCapture`` webcam: ``device`` is an index or a ``/dev/videoN`` path.

    V4L2 queues a few frames in the driver, so a plain ``cap.read()`` after the arm
    moved can return an image captured before the motion. ``read`` first drains the
    queue: ``grab`` calls that return at once are buffered frames and are discarded
    (up to ``drain_frames``); the first grab that has to wait a frame period is live
    and is the one decoded. A failed read reopens the device once before giving up.

    A V4L2 ``grab`` on a stalled device blocks in the driver (10 s per call in
    OpenCV's backend, so a retry loop could hang for a minute). Every device call
    runs on a worker thread and ``read`` gives up after ``read_timeout_s``; while
    such a call is still blocked, reads fail at once instead of queueing behind it.
    ``close`` may be called from any thread, any number of times: a capture a
    blocked worker still uses is released by that worker when the call returns.
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
        read_timeout_s: float = 5.0,
    ) -> None:
        self.device = device
        self._intrinsics = intrinsics
        self.read_retries = max(1, int(read_retries))
        self.drain_frames = max(0, int(drain_frames))
        self.read_timeout_s = float(read_timeout_s)
        self._want = (int(width), int(height), int(fps))
        self.cap: Any = None
        self.reopens = 0
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._closed = False
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
        self._pos_msec = int(cv2.CAP_PROP_POS_MSEC)
        try:
            backend = str(cap.getBackendName())
        except Exception:
            backend = ""
        # Only V4L2 reports the kernel buffer timestamp (CLOCK_MONOTONIC, the clock
        # time.monotonic reads on Linux) as POS_MSEC.
        self._driver_stamps = sys.platform.startswith("linux") and backend == "V4L2"
        with self._lock:
            if self._closed:
                cap.release()
                raise RuntimeError(f"webcam {self.device!r} was closed")
            self.cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(fps)

    def _release(self) -> None:
        with self._lock:
            cap, self.cap = self.cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _reopen(self) -> None:
        self._release()
        self.reopens += 1
        time.sleep(0.2)
        self._open()

    def intrinsics(self) -> dict[str, Any] | None:
        return dict(self._intrinsics) if self._intrinsics else None

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "device": str(self.device)}

    def _read_once(self, cap: Any) -> Frame | None:
        """Drain the queue, then decode the live frame; None when the device gave
        no frame."""
        period = 1.0 / max(1.0, self.fps)
        for _ in range(self.drain_frames):
            t0 = time.perf_counter()
            if not cap.grab():
                return None
            if time.perf_counter() - t0 >= 0.5 * period:
                break  # the grab waited for the sensor: this frame is live
        else:
            if not cap.grab():
                return None
        stamp = None
        if self._driver_stamps:
            try:
                stamp = float(cap.get(self._pos_msec)) / 1000.0
            except Exception:
                stamp = None
        ok, bgr = cap.retrieve()
        if not ok or bgr is None:
            return None
        now_wall, now_mono = time.time(), time.monotonic()
        if stamp is not None and 0.0 < stamp and 0.0 <= now_mono - stamp <= 60.0:
            age = now_mono - stamp
            return Frame(
                rgb=_rgb(bgr),
                depth=None,
                timestamp_s=now_wall - age,
                monotonic_s=stamp,
                time_source="backend",
            )
        wall, mono, source = capture_times(None, "host")
        return Frame(
            rgb=_rgb(bgr),
            depth=None,
            timestamp_s=wall,
            monotonic_s=mono,
            time_source=source,
        )

    def _bounded_read(self, deadline: float) -> Frame | None:
        """``_read_once`` on a worker thread, abandoned at ``deadline``."""
        worker = self._worker
        if worker is not None and worker.is_alive():
            raise RuntimeError(
                f"webcam {self.device!r}: a previous read is still blocked in the driver"
            )
        with self._lock:
            cap = self.cap
        if cap is None:
            return None
        box: dict[str, Any] = {}

        def run() -> None:
            try:
                box["frame"] = self._read_once(cap)
            except Exception as exc:
                box["error"] = exc
            finally:
                with self._lock:
                    orphaned = self.cap is not cap
                if orphaned:  # closed or reopened while this call was blocked
                    try:
                        cap.release()
                    except Exception:
                        pass

        worker = threading.Thread(target=run, daemon=True, name="webcam-read")
        self._worker = worker
        worker.start()
        worker.join(max(0.0, deadline - time.monotonic()))
        if worker.is_alive():
            raise TimeoutError(
                f"webcam {self.device!r} gave no frame within {self.read_timeout_s}s "
                "(the device is stalled)"
            )
        if "error" in box:
            return None
        return box.get("frame")

    def read(self) -> Frame:
        if self._closed:
            raise RuntimeError(f"webcam {self.device!r} is closed")
        deadline = time.monotonic() + self.read_timeout_s
        for attempt in range(2):
            if self.cap is None:
                self._open()
            for _ in range(self.read_retries):
                frame = self._bounded_read(deadline)
                if frame is not None:
                    return frame
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"webcam {self.device!r} gave no frame within "
                        f"{self.read_timeout_s}s"
                    )
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
        """Release the device (idempotent; a blocked read releases it on return)."""
        with self._lock:
            self._closed = True
            cap, self.cap = self.cap, None
            worker = self._worker
        if cap is not None and (worker is None or not worker.is_alive()):
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

    The reader thread owns the capture: only it reads, reopens and releases it, so
    ``close`` (idempotent, any thread) only flags the stop and waits for the thread;
    a thread blocked in ffmpeg releases the capture when the call returns. Frames are
    dated by their presentation time (:class:`PtsClock`).
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
        self.url = url
        self._intrinsics = intrinsics
        self.timeout_s = float(timeout_s)
        self.open_timeout_s = float(open_timeout_s)
        self.reconnect_s = float(reconnect_s)
        self.reconnects = 0
        self._lock = threading.Condition()
        self._latest: Frame | None = None
        self._seq = 0
        self._error: str | None = None
        self._closed = False
        self._clock = PtsClock()
        self.cap: Any = self._open_capture()
        if self.cap is None:
            raise RuntimeError(f"RTSP stream {url!r} did not open")
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

    def _open_capture(self) -> Any:
        import cv2

        params = _ffmpeg_timeout_params(
            cv2, int(self.open_timeout_s * 1000), int(self.timeout_s * 1000)
        )
        cap = (
            cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
            if params
            else cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        )
        if not cap.isOpened():
            cap.release()
            return None
        self._pos_msec = int(cv2.CAP_PROP_POS_MSEC)
        return cap

    def _reconnect(self) -> None:
        """Reopen the capture (reader thread only); ``_error`` stays set until a
        frame arrives."""
        cap, self.cap = self.cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        self._clock.reset()
        deadline = time.monotonic() + self.reconnect_s
        while not self._closed and time.monotonic() < deadline:
            time.sleep(min(0.05, self.reconnect_s))
        if self._closed:
            return
        cap = self._open_capture()
        if cap is not None:
            self.cap = cap
            self.reconnects += 1

    def _pump(self) -> None:
        failures = 0
        try:
            while not self._closed:
                cap = self.cap
                ok, bgr = cap.read() if cap is not None else (False, None)
                arrival = time.monotonic()
                if self._closed:
                    break
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
                try:
                    pts_ms = float(cap.get(self._pos_msec))
                except Exception:
                    pts_ms = None
                captured = self._clock.capture(pts_ms, arrival)
                source = "stream_pts" if captured is not None else "host"
                mono = arrival if captured is None else min(captured, arrival)
                frame = Frame(
                    rgb=_rgb(bgr),
                    depth=None,
                    timestamp_s=time.time() - (arrival - mono),
                    monotonic_s=mono,
                    time_source=source,
                )
                with self._lock:
                    self._error = None
                    self._latest = frame
                    self._seq += 1
                    self._lock.notify_all()
        finally:
            cap, self.cap = self.cap, None
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            with self._lock:
                self._error = self._error or "stream closed"
                self._lock.notify_all()

    def intrinsics(self) -> dict[str, Any] | None:
        return dict(self._intrinsics) if self._intrinsics else None

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "url": self.url}

    def read(self) -> Frame:
        with self._lock:
            if self._closed:
                raise RuntimeError(f"RTSP stream {self.url!r} is closed")
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
        """Stop the reader thread; it releases the capture (idempotent)."""
        with self._lock:
            self._closed = True
            self._lock.notify_all()
        thread = getattr(self, "_thread", None)
        if thread is None:  # the constructor failed before the thread started
            cap, self.cap = getattr(self, "cap", None), None
            if cap is not None:
                cap.release()
            return
        if thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=2.0)
