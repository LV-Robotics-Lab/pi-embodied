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

import time

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
    # The inverse model (RealSense colour) stores undistorting coefficients: applied
    # once, no iteration, and not the same as inverting the forward model.
    inv = undistort_normalized(xy, "inverse_brown_conrady", coeffs)
    np.testing.assert_allclose(inv, _distort(xy, np.asarray(coeffs), False))
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
