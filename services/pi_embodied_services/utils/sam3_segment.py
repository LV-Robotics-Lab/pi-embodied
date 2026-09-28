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

"""A simulator's ``env.segment``: the top SAM3 mask of a text prompt or a positive point on
the image the model sees, located in the world through the simulator's depth (the server
supplies ``locate``: pixels -> world xyz or None). The tool shows ``overlay_png_base64`` (the
mask tinted red on the image); a program's reply drops it and keeps ``mask``."""

from __future__ import annotations

import base64
import io
from collections.abc import Callable
from typing import Any

import numpy as np

#: Fewest mask pixels with depth for a world position.
MIN_POINTS = 10


class Sam3:
    """A lazily connected SAM3 client (``sam3.segment`` of services/.../sam3_server)."""

    def __init__(self, url: str | None):
        self.url = url or ""
        self._client: Any = None

    def __bool__(self) -> bool:
        return bool(self.url)

    def segment(self, rgb: np.ndarray, kwargs: dict) -> dict:
        from pi_embodied_services.utils.rpc.http_rpc import HttpRpcClient

        if not self.url:
            raise RuntimeError(
                "segment needs a SAM3 server (start the env server with --sam3)"
            )
        if self._client is None:
            self._client = HttpRpcClient(self.url)
        return self._client.call(
            "sam3.segment",
            kwargs={"image_base64": png_base64(rgb), **kwargs},
            timeout_s=120,
        )


def png_base64(rgb: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)).save(
        buf, format="PNG"
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_mask(png_b64: str) -> np.ndarray:
    from PIL import Image

    mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(png_b64))))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask >= 128


def sample(mask: np.ndarray, limit: int) -> list[tuple[int, int]]:
    """At most ``limit`` of the mask's (row, col) pixels, spread evenly (row-major order)."""
    rows, cols = np.nonzero(mask)
    stride = max(1, -(-len(rows) // max(1, limit)))
    return [(int(r), int(c)) for r, c in zip(rows[::stride], cols[::stride])]


def segment(
    sam3: Sam3,
    rgb: np.ndarray,
    locate: Callable[[list[tuple[int, int]]], list],
    *,
    prompt: str | None = None,
    point=None,
    min_score: float = 0.2,
    samples: int = 400,
    extra: dict | None = None,
) -> dict:
    """The top mask of ``prompt`` (or ``point`` [row, col]; exactly one) on ``rgb``: ``found``;
    when found ``score``, ``box``, ``mask`` bool[H, W], ``n_pixels``, ``n_valid``,
    ``centroid_pixel`` [row, col] (median), ``world_xyz`` (per-axis median of ``locate`` over at
    most ``samples`` mask pixels; None with ``world_error`` when fewer than MIN_POINTS have depth)
    and ``overlay_png_base64``; not found: ``reason``."""
    text = (prompt or "").strip()
    if bool(text) == (point is not None):
        raise ValueError("give exactly one of a text prompt or a point [row, col]")
    res = sam3.segment(
        rgb,
        {
            **({"text_prompt": text} if text else {"point": [int(v) for v in point]}),
            "min_score": float(min_score),
        },
    )
    if not res.get("found") or not res.get("mask_png_base64"):
        return {
            "found": False,
            "reason": res.get("reason", "no mask"),
            "fallback": "Pick a pixel in the image and use back_project.",
            **(extra or {}),
        }
    mask = decode_mask(res["mask_png_base64"])
    if mask.shape != rgb.shape[:2]:
        raise RuntimeError(
            f"SAM3 mask {mask.shape} does not match the {rgb.shape[:2]} image"
        )
    rows, cols = np.nonzero(mask)
    pts = [
        p
        for p in locate(sample(mask, samples))
        if p is not None and np.all(np.isfinite(p))
    ]
    out: dict[str, Any] = {
        "found": True,
        **(extra or {}),
        "score": None if res.get("score") is None else round(float(res["score"]), 3),
        "box": res.get("box"),
        "mask": mask,
        "n_pixels": int(mask.sum()),
        "n_valid": len(pts),
        "centroid_pixel": [int(round(np.median(rows))), int(round(np.median(cols)))],
        "world_xyz": None,
    }
    if len(pts) >= MIN_POINTS:
        out["world_xyz"] = [
            round(float(v), 4)
            for v in np.median(np.asarray(pts, dtype=np.float64), axis=0)
        ]
    else:
        out["world_error"] = f"too few pixels with depth ({len(pts)})"
    overlay = np.asarray(rgb[..., :3], dtype=np.float32).copy()
    overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([255.0, 0.0, 0.0])
    out["overlay_png_base64"] = png_base64(overlay.round().astype(np.uint8))
    return out


def for_program(out: Any) -> Any:
    """A segment result as a program receives it: no overlay picture."""
    if isinstance(out, dict):
        return {k: v for k, v in out.items() if k != "overlay_png_base64"}
    return out


def for_tool(out: dict) -> dict:
    """A segment result as the tool shows it: no mask array (the overlay shows it)."""
    return {k: v for k, v in out.items() if k != "mask"}


__all__ = ["MIN_POINTS", "Sam3", "for_program", "for_tool", "sample", "segment"]
