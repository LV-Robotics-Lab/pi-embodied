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

"""Depth fusion (utils/depth.py) and the unidepth server's request/response (no model)."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from pi_embodied_services.components.unidepth_server import UniDepthFacade, decode_rgb
from pi_embodied_services.utils.depth import DepthEstimator, fuse_depth


def _scene(h: int = 40, w: int = 40) -> np.ndarray:
    """A table-like plane: depth grows with the row."""
    return np.linspace(0.6, 1.2, h, dtype=np.float32)[:, None].repeat(w, axis=1)


def test_fuse_fills_only_sensor_holes_with_scaled_estimate() -> None:
    truth = _scene()
    sensor = truth.copy()
    sensor[10:20, 5:15] = 0.0  # a hole (RealSense returns 0 where it has no depth)
    sensor[30, 30] = np.nan
    mono = truth * 0.8  # the monocular estimate is off by a constant scale

    fused, report = fuse_depth(sensor, mono)

    assert report["mode"] == "filled"
    assert report["scale"] == pytest.approx(1.25, abs=1e-3)
    assert report["filled_pixels"] == 100 + 1
    # Sensor pixels are untouched; holes carry the scaled estimate, so they match truth.
    assert np.array_equal(fused[0], sensor[0])
    assert np.allclose(fused[10:20, 5:15], truth[10:20, 5:15], atol=1e-3)
    assert fused[30, 30] == pytest.approx(truth[30, 30], abs=1e-3)
    assert fused.dtype == np.float32
    assert report["median_residual_m"] < 1e-3


def test_fuse_refuses_when_sources_disagree_in_scale() -> None:
    sensor = _scene()
    sensor[0:5] = 0.0
    fused, report = fuse_depth(sensor, sensor * 0.1)  # 10x off: outside SCALE_BOUNDS
    assert report["mode"] == "sensor_only"
    assert report["reason"] == "scale_out_of_bounds"
    assert np.all(fused[0:5] == 0.0), "nothing was filled"
    assert np.array_equal(fused[5:], sensor[5:])


def test_fuse_needs_enough_overlap() -> None:
    sensor = np.zeros((40, 40), dtype=np.float32)
    sensor[0, :10] = 0.8  # 10 valid pixels only
    fused, report = fuse_depth(sensor, _scene())
    assert report["mode"] == "sensor_only"
    assert report["reason"] == "insufficient_overlap"
    assert report["overlap_pixels"] == 10
    assert np.count_nonzero(fused) == 10


def test_fuse_without_sensor_returns_estimate_as_is() -> None:
    mono = _scene()
    mono[0, 0] = -1.0  # invalid estimate pixel
    fused, report = fuse_depth(None, mono)
    assert report["mode"] == "mono_only"
    assert fused[0, 0] == 0.0
    assert np.array_equal(fused[1:], mono[1:])


def test_fuse_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        fuse_depth(np.zeros((4, 4), np.float32), np.zeros((4, 5), np.float32))


class FakeBackend:
    model_id = "fake/unidepth"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], np.ndarray | None]] = []

    def infer(self, rgb: np.ndarray, K: np.ndarray | None):
        self.calls.append((rgb.shape, K))
        depth = np.full(rgb.shape[:2], 0.7, dtype=np.float32)
        depth[0, 0] = np.nan  # the model may leave a non-finite pixel
        depth[0, 1] = -0.2
        return depth, np.ones(rgb.shape[:2], dtype=np.float32)


def _png_base64(rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_unidepth_estimate_request_and_response_shape() -> None:
    backend = FakeBackend()
    facade = UniDepthFacade(backend, resolution_level=4)
    rgb = np.full((6, 8, 3), 127, dtype=np.uint8)
    K = np.array([[100.0, 0, 4], [0, 100.0, 3], [0, 0, 1]])

    out = facade._dispatch("depth.estimate", (), {"rgb": rgb, "K": K})

    assert out["depth"].shape == (6, 8) and out["depth"].dtype == np.float32
    assert out["depth"][0, 0] == 0.0 and out["depth"][0, 1] == 0.0, "invalid -> 0"
    assert out["depth"][3, 3] == pytest.approx(0.7)
    assert out["confidence"].shape == (6, 8)
    assert out["model"] == "fake/unidepth"
    assert out["resolution_level"] == 4
    assert out["used_intrinsics"] is True
    assert out["valid_ratio"] == pytest.approx(46 / 48)
    assert out["depth_range_m"] == [pytest.approx(0.7), pytest.approx(0.7)]
    assert backend.calls[0][0] == (6, 8, 3)
    assert np.array_equal(backend.calls[0][1], K)

    # A PNG in place of the array, and no intrinsics.
    out = facade._dispatch("depth.estimate", (), {"rgb": _png_base64(rgb)})
    assert out["used_intrinsics"] is False and backend.calls[1][1] is None
    assert facade._dispatch("healthz", (), {})["service"] == "unidepth"


def test_unidepth_rejects_bad_inputs() -> None:
    facade = UniDepthFacade(FakeBackend(), resolution_level=4)
    with pytest.raises(ValueError):
        facade.estimate(np.zeros((6, 8), dtype=np.uint8))
    with pytest.raises(ValueError):
        facade.estimate(np.zeros((6, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        facade.estimate(np.zeros((6, 8, 3), dtype=np.uint8), K=np.eye(4))
    with pytest.raises(ValueError):
        facade.estimate(np.zeros((6, 8, 3), dtype=np.uint8), K=np.zeros((3, 3)))
    assert decode_rgb(np.zeros((2, 2, 3), np.uint8)).shape == (2, 2, 3)


class FakeRpc:
    """Stands in for HttpRpcClient: records the call and answers like the server."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append((method, dict(kwargs or {})))
        rgb = kwargs["rgb"]
        return {
            "depth": np.full(rgb.shape[:2], 0.9, np.float32),
            "model": "m",
            "inference_s": 0.1,
        }


def test_depth_estimator_client_contract() -> None:
    rpc = FakeRpc()
    depth, info = DepthEstimator(rpc).estimate(np.zeros((5, 7, 3), np.uint8), np.eye(3))
    assert depth.shape == (5, 7) and depth.dtype == np.float32
    assert info == {"model": "m", "inference_s": 0.1}
    method, kwargs = rpc.calls[0]
    assert method == "depth.estimate"
    assert kwargs["rgb"].dtype == np.uint8 and kwargs["K"].shape == (3, 3)
    with pytest.raises(ValueError):
        DepthEstimator(rpc).estimate(np.zeros((5, 7), np.uint8))
