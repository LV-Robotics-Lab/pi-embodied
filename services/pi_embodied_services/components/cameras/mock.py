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

"""A test-double camera: a fixed RGB gradient over a flat surface. Not a simulator.

It exists so env servers' observation shapes and metadata can be tested without
hardware. Constructing one outside a test run raises: ``PI_EMBODIED_MOCK_ROBOT``
must be ``1`` (the tests set it; nothing else should).
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from pi_embodied_services.components.cameras.base import Camera, Frame

MOCK_ENV = "PI_EMBODIED_MOCK_ROBOT"


def require_test_env() -> None:
    if os.environ.get(MOCK_ENV) != "1":
        raise RuntimeError(
            f"the camera mocks are for tests only (set {MOCK_ENV}=1 in a test); "
            "they never stand in for a sensor"
        )


class MockCamera(Camera):
    """``depth_m`` None makes it RGB-only (a webcam); otherwise a flat surface there."""

    kind = "mock"

    def __init__(
        self,
        serial: str = "mock",
        width: int = 640,
        height: int = 480,
        depth_m: float | None = 0.5,
    ) -> None:
        require_test_env()
        self.serial = serial
        self.width, self.height = int(width), int(height)
        self.depth_m = depth_m
        self.has_depth = depth_m is not None
        self.depth_scale = 0.001 if self.has_depth else 0.0
        self.closed = False
        self.reads = 0

    def intrinsics(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": 600.0,
            "fy": 600.0,
            "ppx": self.width / 2,
            "ppy": self.height / 2,
            "distortion_model": "none",
            "coeffs": [0.0] * 5,
        }

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "serial_number": self.serial}

    def read(self) -> Frame:
        self.reads += 1
        rows = np.linspace(0, 255, self.height, dtype=np.float32)[:, None]
        cols = np.linspace(0, 255, self.width, dtype=np.float32)[None, :]
        rgb = np.stack(
            [
                np.broadcast_to(rows, (self.height, self.width)),
                np.broadcast_to(cols, (self.height, self.width)),
                np.full((self.height, self.width), 128.0),
            ],
            axis=-1,
        ).astype(np.uint8)
        depth = (
            np.full((self.height, self.width), self.depth_m, dtype=np.float32)
            if self.depth_m is not None
            else None
        )
        return Frame(rgb=rgb, depth=depth, timestamp_s=time.time())

    def close(self) -> None:
        self.closed = True
