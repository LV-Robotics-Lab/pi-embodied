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

"""Metric depth for env servers: fuse a sensor depth map with a monocular estimate.

OpenETA's ``enhance_depth`` (``agent/runtime/depth_enhancement.py``) is sensor-first:
valid sensor pixels are kept, and the monocular estimate only fills the pixels the
sensor missed, after its scale is aligned to the sensor over the pixels both see.
:func:`fuse_depth` keeps that policy (trimmed-median ratio, bounded scale) and adds the
case OpenETA leaves to a separate tool: a camera without depth (a webcam) takes the
monocular estimate as-is.

Invalid depth is ``0`` (or non-finite) on both sides; the fused map uses ``0`` for the
pixels neither source could fill, which is what ``back_project`` already treats as a
hole.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Sensor pixels closer than this or farther than this are holes.
MIN_DEPTH_M = 0.05
MAX_DEPTH_M = 10.0
#: The scale fit needs at least this many pixels seen by both sources.
MIN_OVERLAP_PIXELS = 200
#: Quantiles trimmed from both ends of the sensor/mono ratios before the median.
TRIM_FRACTION = 0.1
#: A fitted scale outside this range means the two sources disagree; nothing is filled.
SCALE_BOUNDS = (0.5, 2.0)


def valid_depth(depth: np.ndarray) -> np.ndarray:
    """Pixels with a usable metric depth."""
    return np.isfinite(depth) & (depth > MIN_DEPTH_M) & (depth < MAX_DEPTH_M)


def fuse_depth(
    sensor: np.ndarray | None,
    mono: np.ndarray,
    *,
    min_overlap: int = MIN_OVERLAP_PIXELS,
    scale_bounds: tuple[float, float] = SCALE_BOUNDS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fill the holes of ``sensor`` with ``mono`` scaled to their overlap.

    Returns the fused float32 depth (metres, ``0`` = no depth) and a report:
    ``mode`` is ``mono_only`` (no sensor depth: the estimate is returned as-is),
    ``filled`` (holes filled with ``scale * mono``) or ``sensor_only`` (nothing
    filled; ``reason`` says why: ``insufficient_overlap`` or ``scale_out_of_bounds``).
    """
    mono = np.asarray(mono, dtype=np.float32)
    if mono.ndim != 2:
        raise ValueError(f"mono depth must be [H, W], got shape {mono.shape}")
    mono_valid = valid_depth(mono)
    if sensor is None:
        fused = np.where(mono_valid, mono, 0.0).astype(np.float32)
        return fused, {
            "mode": "mono_only",
            "scale": 1.0,
            "overlap_pixels": 0,
            "filled_pixels": int(mono_valid.sum()),
            "filled_ratio": float(mono_valid.mean()),
            "sensor_valid_ratio": 0.0,
            "fused_valid_ratio": float(mono_valid.mean()),
        }
    sensor = np.asarray(sensor, dtype=np.float32)
    if sensor.shape != mono.shape:
        raise ValueError(
            f"sensor depth {sensor.shape} and mono depth {mono.shape} differ in shape"
        )
    sensor_valid = valid_depth(sensor)
    fused = np.where(sensor_valid, sensor, 0.0).astype(np.float32)
    report: dict[str, Any] = {
        "mode": "sensor_only",
        "scale": 1.0,
        "overlap_pixels": 0,
        "filled_pixels": 0,
        "filled_ratio": 0.0,
        "sensor_valid_ratio": float(sensor_valid.mean()),
        "fused_valid_ratio": float(sensor_valid.mean()),
    }
    overlap = sensor_valid & mono_valid
    report["overlap_pixels"] = int(overlap.sum())
    if report["overlap_pixels"] < min_overlap:
        report["reason"] = "insufficient_overlap"
        return fused, report
    ratios = sensor[overlap] / mono[overlap]
    lo, hi = np.quantile(ratios, [TRIM_FRACTION, 1.0 - TRIM_FRACTION])
    kept = ratios[(ratios >= lo) & (ratios <= hi)]
    scale = float(np.median(kept if kept.size else ratios))
    report["scale"] = scale
    if not (scale_bounds[0] <= scale <= scale_bounds[1]):
        report["reason"] = "scale_out_of_bounds"
        return fused, report
    aligned = mono * scale
    report["median_residual_m"] = float(
        np.median(np.abs(sensor[overlap] - aligned[overlap]))
    )
    fill = ~sensor_valid & mono_valid & valid_depth(aligned)
    fused[fill] = aligned[fill]
    report.update(
        {
            "mode": "filled",
            "filled_pixels": int(fill.sum()),
            "filled_ratio": float(fill.mean()),
            "fused_valid_ratio": float((sensor_valid | fill).mean()),
        }
    )
    return fused, report


class DepthEstimator:
    """Client of the ``unidepth`` service (``components/unidepth_server.py``)."""

    def __init__(self, client: Any) -> None:
        """``client`` has ``call(method, args, kwargs)``: an ``HttpRpcClient``."""
        self._client = client

    @classmethod
    def connect(cls, url: str) -> DepthEstimator:
        from pi_embodied_services.utils.rpc.http_rpc import HttpRpcClient

        return cls(HttpRpcClient(url))

    def estimate(
        self, rgb: np.ndarray, K: np.ndarray | None = None, *, timeout_s: float = 120.0
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Metric depth (float32 [H, W], metres) of ``rgb`` (uint8 [H, W, 3])."""
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be [H, W, 3], got shape {rgb.shape}")
        kwargs: dict[str, Any] = {"rgb": rgb}
        if K is not None:
            K = np.asarray(K, dtype=np.float64)
            if K.shape != (3, 3):
                raise ValueError(f"K must be 3x3, got shape {K.shape}")
            kwargs["K"] = K
        result = self._client.call("depth.estimate", (), kwargs, timeout_s=timeout_s)
        depth = np.asarray(result["depth"], dtype=np.float32)
        if depth.shape != rgb.shape[:2]:
            raise RuntimeError(
                f"unidepth returned depth {depth.shape} for image {rgb.shape[:2]}"
            )
        info = {k: v for k, v in result.items() if k not in ("depth", "confidence")}
        return depth, info


__all__ = [
    "MAX_DEPTH_M",
    "MIN_DEPTH_M",
    "DepthEstimator",
    "fuse_depth",
    "valid_depth",
]
