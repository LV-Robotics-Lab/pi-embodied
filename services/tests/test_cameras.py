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

"""The shared camera layer's config parsing and interface (no devices)."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable

import numpy as np
import pytest

from pi_embodied_services.components.cameras import (
    CAMERA_TYPES,
    Frame,
    intrinsic_matrix,
    intrinsics_from_config,
    parse_source,
    parse_sources,
    undistort_normalized,
    undistort_pixels,
    validate_device,
)
from pi_embodied_services.components.cameras.base import _distort
from pi_embodied_services.components.cameras.mock import MOCK_ENV, MockCamera


def test_source_flags_parse_into_device_mappings():
    assert parse_source("realsense:123456") == {"type": "realsense", "serial": "123456"}
    assert parse_source("realsense:") == {"type": "realsense", "serial": None}
    assert parse_source("webcam:0") == {"type": "webcam", "device": 0}
    assert parse_source("webcam:/dev/video2") == {
        "type": "webcam",
        "device": "/dev/video2",
    }
    assert parse_source("rtsp://user:pw@cam/live") == {
        "type": "rtsp",
        "url": "rtsp://user:pw@cam/live",
    }
    assert parse_source("rtsp:rtsp://cam/live") == {
        "type": "rtsp",
        "url": "rtsp://cam/live",
    }
    for bad in ("kinect:0", "webcam", "webcam:", ""):
        with pytest.raises(ValueError):
            parse_source(bad)
    devices = parse_sources("wrist=realsense:1, front=webcam:0")
    assert list(devices) == ["wrist", "front"] and devices["wrist"]["main"] is True
    assert "main" not in devices["front"]
    with pytest.raises(ValueError, match="named twice"):
        parse_sources("a=webcam:0,a=webcam:1")
    assert parse_sources("") == {}
    assert CAMERA_TYPES == ("realsense", "webcam", "rtsp")


def test_device_validation():
    assert validate_device("w", {"serial": "1"})["type"] == "realsense", (
        "the default type"
    )
    with pytest.raises(ValueError, match="unknown keys"):
        validate_device("w", {"serial": "1", "exposure": 3})
    with pytest.raises(ValueError, match="webcam needs `device`"):
        validate_device("w", {"type": "webcam"})
    with pytest.raises(ValueError, match="rtsp needs `url`"):
        validate_device("w", {"type": "rtsp"})
    with pytest.raises(ValueError, match="mount must be"):
        validate_device("w", {"type": "webcam", "device": 0, "mount": "ceiling"})
    with pytest.raises(ValueError, match="must be a mapping"):
        validate_device("w", "webcam")


def test_config_intrinsics_take_the_realsense_layout():
    intr = intrinsics_from_config(
        {"fx": 500, "fy": 510, "cx": 320, "cy": 240}, 640, 480
    )
    assert intr["ppx"] == 320 and intr["ppy"] == 240 and intr["width"] == 640
    assert intrinsic_matrix(intr) == [
        [500.0, 0.0, 320.0],
        [0.0, 510.0, 240.0],
        [0.0, 0.0, 1.0],
    ]
    assert intrinsics_from_config(None, 640, 480) is None
    with pytest.raises(ValueError, match="missing \\['ppy'\\]"):
        intrinsics_from_config({"fx": 500, "fy": 510, "ppx": 320}, 640, 480)
    with pytest.raises(ValueError, match="finite"):
        intrinsics_from_config(
            {"fx": float("nan"), "fy": 1, "ppx": 1, "ppy": 1}, 640, 480
        )
    # Distortion: no coeffs = none; coeffs without a model = OpenCV's forward
    # Brown-Conrady, padded to [k1, k2, p1, p2, k3]; unknown models are refused.
    assert intr["distortion_model"] == "none" and intr["coeffs"] == [0.0] * 5
    base = {"fx": 500, "fy": 510, "cx": 320, "cy": 240}
    fwd = intrinsics_from_config({**base, "coeffs": [0.1, -0.2, 0.0, 0.0]}, 640, 480)
    assert fwd["distortion_model"] == "brown_conrady"
    assert fwd["coeffs"] == [0.1, -0.2, 0.0, 0.0, 0.0]
    inv = intrinsics_from_config(
        {**base, "distortion_model": "inverse_brown_conrady", "coeffs": [0.1] * 5},
        640,
        480,
    )
    assert inv["distortion_model"] == "inverse_brown_conrady"
    with pytest.raises(ValueError, match="distortion_model must be one of"):
        intrinsics_from_config({**base, "distortion_model": "kannala_brandt4"}, 1, 1)
    with pytest.raises(ValueError, match="at most 5"):
        intrinsics_from_config({**base, "coeffs": [0.0] * 6}, 640, 480)


def test_undistortion_follows_the_intrinsics_distortion_model():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-0.4, 0.4, (50, 2))
    coeffs = [0.08, -0.03, 0.002, -0.001, 0.01]
    # Forward models: distort, then undistort round-trips.
    for model, modified in (("brown_conrady", False), ("modified_brown_conrady", True)):
        distorted = _distort(xy, np.asarray(coeffs), modified)
        assert not np.allclose(distorted, xy)
        np.testing.assert_allclose(
            undistort_normalized(distorted, model, coeffs), xy, atol=1e-9
        )
    # The two forward models differ (tangential terms applied at different points).
    assert not np.allclose(
        _distort(xy, np.asarray(coeffs), False), _distort(xy, np.asarray(coeffs), True)
    )
    # The inverse model (RealSense colour) is undistorted the way librealsense
    # 2.54.1 deprojects it: iterating toward the inverse of the model its projection
    # (rs2_project_point_to_pixel) applies, the "modified" one. The single pass of
    # librealsense 2.20 (the polynomial applied once) is not that.
    inv = undistort_normalized(xy, "inverse_brown_conrady", coeffs)
    np.testing.assert_allclose(_distort(inv, np.asarray(coeffs), True), xy, atol=1e-6)
    assert not np.allclose(inv, _distort(xy, np.asarray(coeffs), False), atol=1e-4)
    assert not np.allclose(inv, undistort_normalized(xy, "brown_conrady", coeffs))
    # No coefficients or model none: identity; short coefficient lists are padded.
    np.testing.assert_allclose(undistort_normalized(xy, "none", coeffs), xy)
    np.testing.assert_allclose(undistort_normalized(xy, "brown_conrady", None), xy)
    np.testing.assert_allclose(
        undistort_normalized(xy, "brown_conrady", coeffs[:2]),
        undistort_normalized(xy, "brown_conrady", coeffs[:2] + [0, 0, 0]),
    )
    with pytest.raises(ValueError, match="not supported"):
        undistort_normalized(xy, "ftheta", coeffs)
    # Pixels go through K and back; the principal point is a fixed point.
    intr = {
        "fx": 600.0,
        "fy": 600.0,
        "ppx": 320.0,
        "ppy": 240.0,
        "distortion_model": "brown_conrady",
        "coeffs": coeffs,
    }
    uv = np.array([[320.0, 240.0], [100.0, 50.0]])
    out = undistort_pixels(uv, intr)
    np.testing.assert_allclose(out[0], [320.0, 240.0])
    assert np.linalg.norm(out[1] - uv[1]) > 1.0
    np.testing.assert_allclose(undistort_pixels(uv, {**intr, "coeffs": None}), uv)


def test_mock_camera_is_test_only_and_yields_frames(monkeypatch):
    monkeypatch.delenv(MOCK_ENV, raising=False)
    with pytest.raises(RuntimeError, match="tests only"):
        MockCamera()
    monkeypatch.setenv(MOCK_ENV, "1")
    cam = MockCamera("s", 320, 240)
    f = cam.read()
    assert isinstance(f, Frame) and f.has_depth
    assert f.rgb.shape == (240, 320, 3) and f.rgb.dtype == np.uint8
    assert f.depth.shape == (240, 320) and f.depth.dtype == np.float32
    assert cam.describe() == {
        "camera_type": "mock",
        "has_depth": True,
        "serial_number": "s",
    }
    rgb_only = MockCamera("w", depth_m=None)
    assert not rgb_only.has_depth and rgb_only.read().depth is None
    rgb_only.close()
    assert rgb_only.closed


def test_frames_carry_a_monotonic_capture_time_and_read_fresh_drains_a_queue(
    monkeypatch,
):
    monkeypatch.setenv(MOCK_ENV, "1")
    # A frame built without one gets the construction time (older drivers/mocks).
    before = time.monotonic()
    f = Frame(rgb=np.zeros((2, 2, 3), np.uint8), depth=None, timestamp_s=0.0)
    assert before <= f.monotonic_s <= time.monotonic() and f.age_s() < 1.0
    assert f.age_s(f.monotonic_s + 2.5) == pytest.approx(2.5)
    old = MockCamera("s", 4, 4, age_s=3.0).read()
    assert old.age_s() >= 3.0
    # A buffered source (a V4L2 queue) hands out the scene as of the previous read;
    # read_fresh discards that frame and returns the current scene.
    cam = MockCamera("q", 4, 4, buffered=True)
    cam.read()
    cam.scene = 5
    assert MockCamera.scene_of(cam.read()) == 0
    cam.scene = 6
    assert MockCamera.scene_of(cam.read_fresh()) == 6 and cam.reads == 4
    # A dead source raises for the configured reads, then recovers.
    dead = MockCamera("d", 4, 4, fail_reads=1)
    with pytest.raises(RuntimeError, match="no frame"):
        dead.read()
    assert dead.read().rgb.shape == (4, 4, 3)


#: Reference undistorted pixels from librealsense v2.54.1 src/rs.cpp
#: rs2_deproject_pixel_to_point (RS2_DISTORTION_INVERSE_BROWN_CONRADY branch copied
#: verbatim into C, float32, compiled and run; the point re-projected through K):
#: (camera, u, v) -> (u', v').
RS_2_54_1_INTRINSICS = (
    {"ppx": 321.7, "ppy": 243.4, "fx": 615.2, "fy": 615.0},
    {"ppx": 640.3, "ppy": 360.9, "fx": 910.1, "fy": 909.6},
)
RS_2_54_1_COEFFS = (
    [0.12, -0.25, 0.0021, -0.0015, 0.11],
    [-0.055, 0.065, -0.0007, 0.0006, -0.021],
)
RS_2_54_1_REFERENCE = [
    (0, 0.0, 0.0, 4.950963, 2.926066),
    (0, 639.0, 479.0, 634.649902, 474.989929),
    (0, 100.0, 400.0, 103.797043, 397.199615),
    (0, 320.0, 240.0, 320.000031, 239.999908),
    (0, 600.0, 20.0, 596.620178, 22.534443),
    (0, 10.0, 250.0, 15.627627, 249.571335),
    (1, 0.0, 0.0, -9.639566, -4.789729),
    (1, 1279.0, 719.0, 1287.515625, 724.410645),
    (1, 200.0, 650.0, 193.962830, 654.061096),
    (1, 640.0, 360.0, 640.000000, 360.000000),
    (1, 1200.0, 40.0, 1206.938721, 36.190563),
    (1, 20.0, 380.0, 10.557754, 380.592224),
]


def test_inverse_brown_conrady_matches_librealsense_2_54_1():
    worst_old = 0.0
    for cam, u, v, ur, vr in RS_2_54_1_REFERENCE:
        intr = {
            **RS_2_54_1_INTRINSICS[cam],
            "distortion_model": "inverse_brown_conrady",
            "coeffs": RS_2_54_1_COEFFS[cam],
        }
        got = undistort_pixels(np.array([[u, v]]), intr)[0]
        # float32 in the SDK, float64 here.
        np.testing.assert_allclose(got, [ur, vr], atol=1e-3)
        # The single-pass (librealsense 2.20) undistortion this replaced.
        K = RS_2_54_1_INTRINSICS[cam]
        f = np.array([K["fx"], K["fy"]])
        c = np.array([K["ppx"], K["ppy"]])
        xy = (np.array([[u, v]]) - c) / f
        once = _distort(xy, np.asarray(RS_2_54_1_COEFFS[cam]), False)[0] * f + c
        worst_old = max(worst_old, float(np.abs(once - [ur, vr]).max()))
    assert worst_old > 5.0, "the old semantics were several pixels off at the edges"


def test_legacy_distortion_model_spelling_is_accepted_and_normalized():
    # Calibration sample dirs written before the name was normalized store
    # str(rs.distortion.x) = "distortion.x" (as does the Franka driver).
    from pi_embodied_services.components.cameras import normalize_distortion_model

    assert (
        normalize_distortion_model("distortion.inverse_brown_conrady")
        == "inverse_brown_conrady"
    )
    assert normalize_distortion_model("brown_conrady") == "brown_conrady"
    assert normalize_distortion_model(None) == "none"
    base = {"fx": 600, "fy": 600, "ppx": 320, "ppy": 240, "coeffs": [0.1] * 5}
    legacy_name = "distortion.inverse_brown_conrady"
    legacy = intrinsics_from_config({**base, "distortion_model": legacy_name}, 640, 480)
    assert legacy["distortion_model"] == "inverse_brown_conrady"
    uv = np.array([[10.0, 20.0]])
    np.testing.assert_allclose(
        undistort_pixels(uv, {**base, "distortion_model": legacy_name}),
        undistort_pixels(uv, legacy),
    )
    with pytest.raises(ValueError, match="distortion_model must be one of"):
        intrinsics_from_config({**base, "distortion_model": "distortion.ftheta"}, 1, 1)


# -- capture times and driver threading (fake SDKs, no devices) ------------------------


class _FakeCap:
    """A ``cv2.VideoCapture`` double: ``gate`` cleared blocks grab/read (a stalled
    device); ``pos_ms`` is what ``CAP_PROP_POS_MSEC`` reads."""

    def __init__(self, delay_s: float = 0.0):
        self.gate = threading.Event()
        self.gate.set()
        self.delay_s = delay_s
        self.pos_ms: float | Callable[[], float] = 0.0
        self.released = 0
        self.busy = False
        self.released_while_busy = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        return True

    def getBackendName(self):
        return "V4L2"

    def get(self, prop):
        if prop == _FakeCv2.CAP_PROP_POS_MSEC:
            return self.pos_ms() if callable(self.pos_ms) else self.pos_ms
        return {_FakeCv2.CAP_PROP_FPS: 30.0}.get(prop, 0.0)

    def _io(self):
        self.busy = True
        try:
            self.gate.wait()
            time.sleep(self.delay_s)
        finally:
            self.busy = False
        return not self.released

    def grab(self):
        return self._io()

    def retrieve(self):
        return True, np.zeros((4, 4, 3), np.uint8)

    def read(self):
        ok = self._io()
        return ok, (np.zeros((4, 4, 3), np.uint8) if ok else None)

    def release(self):
        if self.busy:
            self.released_while_busy = True
        self.released += 1


class _FakeCv2:
    CAP_V4L2, CAP_FFMPEG = 200, 1900
    CAP_PROP_POS_MSEC, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT = 0, 3, 4
    CAP_PROP_FPS, CAP_PROP_FOURCC, CAP_PROP_BUFFERSIZE = 5, 6, 38
    CAP_PROP_OPEN_TIMEOUT_MSEC, CAP_PROP_READ_TIMEOUT_MSEC = 53, 54

    def __init__(self, cap: _FakeCap):
        self.cap = cap
        self.opened: list[tuple] = []

    def VideoCapture(self, *args):
        self.opened.append(args)
        return self.cap

    @staticmethod
    def VideoWriter_fourcc(*chars):
        return 0


def _fake_cv2(monkeypatch, cap: _FakeCap) -> _FakeCv2:
    fake = _FakeCv2(cap)
    monkeypatch.setitem(sys.modules, "cv2", fake)
    return fake


def test_a_stalled_webcam_read_is_bounded_and_close_is_safe(monkeypatch):
    from pi_embodied_services.components.cameras.webcam import WebcamRGB

    cap = _FakeCap()
    _fake_cv2(monkeypatch, cap)
    cam = WebcamRGB(0, drain_frames=0, read_timeout_s=0.3)
    assert cam.read().rgb.shape == (4, 4, 3)
    cap.gate.clear()  # the device stalls inside grab (V4L2 select: 10 s per call)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="within 0.3s"):
        cam.read()
    assert time.monotonic() - t0 < 1.0, "bounded by read_timeout_s, not ~60 s"
    with pytest.raises(RuntimeError, match="still blocked"):
        cam.read()  # fails at once instead of queueing behind the stuck call
    # close while the worker is inside grab: the capture is not pulled from under it;
    # the worker releases it when the call returns. close is idempotent.
    closers = [threading.Thread(target=cam.close) for _ in range(3)]
    for t in closers:
        t.start()
    for t in closers:
        t.join()
    assert cap.released == 0
    cap.gate.set()
    deadline = time.monotonic() + 2
    while cap.released == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert cap.released == 1 and not cap.released_while_busy
    cam.close()
    assert cap.released == 1
    with pytest.raises(RuntimeError, match="closed"):
        cam.read()


def test_v4l2_frames_are_dated_by_the_driver_timestamp(monkeypatch):
    from pi_embodied_services.components.cameras.webcam import WebcamRGB

    cap = _FakeCap()
    _fake_cv2(monkeypatch, cap)
    monkeypatch.setattr(sys, "platform", "linux")
    # The kernel stamped the buffer 0.8 s ago (it waited in the driver queue).
    cap.pos_ms = lambda: (time.monotonic() - 0.8) * 1000.0
    cam = WebcamRGB("/dev/video0", drain_frames=0)
    f = cam.read()
    assert f.time_source == "backend" and 0.75 < f.age_s() < 1.5
    assert abs((time.time() - f.timestamp_s) - f.age_s()) < 0.05
    # No usable stamp (0 before the first capture, or a non-V4L2 backend): dequeue time.
    cap.pos_ms = 0.0
    assert cam.read().time_source == "host"
    monkeypatch.setattr(sys, "platform", "darwin")
    cam2 = WebcamRGB(0, drain_frames=0)
    cap.pos_ms = lambda: (time.monotonic() - 0.8) * 1000.0
    assert cam2.read().time_source == "host"
    cam.close()
    cam2.close()


def test_rtsp_close_does_not_release_under_the_reader_and_is_idempotent(monkeypatch):
    from pi_embodied_services.components.cameras.webcam import RtspRGB

    cap = _FakeCap(delay_s=0.02)
    t0 = time.monotonic()
    cap.pos_ms = lambda: (time.monotonic() - t0) * 1000.0 + 1.0
    fake = _fake_cv2(monkeypatch, cap)
    cam = RtspRGB("rtsp://cam/live", timeout_s=1.0, open_timeout_s=2.0)
    # The FFmpeg open/read timeouts are passed where OpenCV supports them.
    assert fake.opened[0][2] == [53, 2000, 54, 1000]
    f = cam.read()
    assert f.time_source == "stream_pts" and f.age_s() < 0.5
    closers = [threading.Thread(target=cam.close) for _ in range(4)]
    for t in closers:
        t.start()
    for t in closers:
        t.join()
    assert cap.released == 1 and not cap.released_while_busy
    cam.close()
    assert cap.released == 1
    with pytest.raises(RuntimeError, match="closed"):
        cam.read()


def test_pts_clock_dates_buffered_rtsp_frames_back():
    from pi_embodied_services.components.cameras.webcam import PtsClock

    clock = PtsClock(drift_per_s=0.0)
    # Live frames: 30 fps, arriving 0.1 s after their PTS (network + decode).
    for i in range(10):
        pts = 1.0 + i / 30
        assert clock.capture(pts * 1000, 100.1 + pts) == pytest.approx(100.1 + pts)
    # The stream stalls and then delivers buffered frames: each is dated back to when
    # it was live, so its age shows.
    now = 104.0
    got = clock.capture((1.0 + 10 / 30) * 1000, now)
    assert now - got == pytest.approx(now - (100.1 + 1.0 + 10 / 30))
    assert now - got > 2.0
    # A PTS that jumps back (a reconnect) re-anchors; a missing PTS gives None.
    assert clock.capture(500.0, 200.0) == pytest.approx(200.0)
    assert clock.capture(None, 201.0) is None
    assert clock.capture(float("nan"), 201.0) is None
    # A camera clock running slow is absorbed at drift_per_s.
    slow = PtsClock(drift_per_s=1e-3)
    slow.capture(1000.0, 10.0)
    later = slow.capture(101000.0, 110.05)  # 50 ms of drift over 100 s
    assert 110.05 - later == pytest.approx(0.0, abs=1e-9)


def test_realsense_frames_use_the_device_timestamp_when_it_is_host_time():
    from types import SimpleNamespace

    from pi_embodied_services.components.cameras.realsense import RealSenseRGBD

    domain = SimpleNamespace(global_time="g", system_time="s", hardware_clock="h")
    cam = object.__new__(RealSenseRGBD)
    cam._rs = SimpleNamespace(timestamp_domain=domain)

    def frame(dom, age):
        stamp = (time.time() - age) * 1000.0
        return SimpleNamespace(
            get_frame_timestamp_domain=lambda: dom, get_timestamp=lambda: stamp
        )

    wall, mono, source = cam._capture_times(frame("g", 0.7))
    assert source == "device" and 0.65 < time.monotonic() - mono < 1.0
    assert abs(time.time() - wall - 0.7) < 0.05
    assert cam._capture_times(frame("s", 0.2))[2] == "backend"
    # The device's free-running clock is not host time: dequeue time.
    assert cam._capture_times(frame("h", 0.0))[2] == "host"
    # A stamp from the future or absurdly old is not trusted.
    assert cam._capture_times(frame("g", -5.0))[2] == "host"
    assert cam._capture_times(frame("g", 3600.0))[2] == "host"
    assert cam._capture_times(None)[2] == "host"
