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

import numpy as np
import pytest

from pi_embodied_services.components.cameras import (
    CAMERA_TYPES,
    Frame,
    intrinsic_matrix,
    intrinsics_from_config,
    parse_source,
    parse_sources,
    validate_device,
)
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
