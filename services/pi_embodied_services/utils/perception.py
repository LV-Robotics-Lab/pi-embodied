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

"""Perception primitives an env server composes over its current observation.

The env server is the one place that has both the latest camera frames and their
intrinsics, so it (not the TS client) calls the model servers and keeps the results
bound to the observation they came from (OpenETA port spec, section 0.2):

- ``env.segment``: SAM3 on one camera of the current observation; ``all=True`` returns
  every mask with a short id (``d7``), otherwise the best one. Ids live in a
  :class:`DetectionBook` and die with the observation.
- ``env.select_detection`` / ``env.reject_detection``: pick or exclude one id (OpenETA's
  ``select_sam3_detection`` / ``reject_sam3_detection``); stale ids are refused with
  the observation they belonged to.
- ``env.enhance_depth``: UniDepth V2 on one camera; the estimate fills the holes of the
  sensor depth (or stands in for a camera without depth) and replaces that camera's
  depth in the current observation, so later ids project through it.

:class:`Perception` is installed on a facade after its own ``_register_rpc``: it wraps
``env.get_observation`` (every observation invalidates the ids) and the motion methods
(``MOTION_METHODS``: a move invalidates them too, observed or not), wraps
``env.get_env_meta`` (``capabilities.perception`` says what is on), and registers only the
primitives whose service URL was given. Without ``--sam3`` / ``--unidepth`` nothing changes.
The ids come from the facade's :class:`Epoch`, shared with the grasp planner when there is one.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Callable

import numpy as np

from pi_embodied_services.utils.depth import DepthEstimator, fuse_depth
from pi_embodied_services.utils.detections import (
    MOTION_METHODS,
    DetectionBook,
    DetectionStale,
    Epoch,
    decode_mask_png,
    describe_mask,
    overlay_masks,
    public,
)
from pi_embodied_services.utils.logging import get_logger

logger = get_logger("perception")

#: camera alias -> (image key, depth key, index into a stacked extra view or None)
Cameras = dict[str, tuple[str, str, int | None]]

#: The franka servers' observation layout: main = wrist, extra_view[0] = external.
FRANKA_CAMERAS: Cameras = {
    "wrist": ("main_images", "main_depths", None),
    "third_person": ("extra_view_images", "extra_view_depths", 0),
}


def _png_base64(rgb: np.ndarray) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), mode="RGB").save(
        buffer, format="PNG"
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _pick(obs: dict[str, Any], key: str, index: int | None) -> np.ndarray | None:
    value = obs.get(key)
    if value is None:
        return None
    array = np.asarray(value)
    if index is None:
        return array
    if array.ndim < 1 or index >= array.shape[0]:
        return None
    return array[index]


def franka_intrinsics(meta: Any, key: str) -> np.ndarray | None:
    """``intrinsic_K`` of observation key ``main`` / ``extra_N`` from franka camera meta."""
    if not isinstance(meta, dict) or "error" in meta:
        return None
    name = (meta.get("observation_camera_map") or {}).get(key)
    cam = (meta.get("cameras") or {}).get(name) if name else None
    K = (cam or {}).get("intrinsic_K")
    if K is None:
        return None
    K = np.asarray(K, dtype=np.float64)
    return K if K.shape == (3, 3) and np.all(np.isfinite(K)) else None


class Perception:
    """Segmentation ids and depth enhancement over an env server's current observation.

    ``sam3`` / ``unidepth`` are RPC clients (``call(method, args, kwargs)``) or None;
    ``cameras`` maps the aliases the tools take to observation keys; ``intrinsics``
    returns the 3x3 K of an observation image key (``main``, ``extra_0``) or None.
    """

    def __init__(
        self,
        *,
        sam3: Any | None = None,
        unidepth: Any | None = None,
        cameras: Cameras,
        intrinsics: Callable[[str], np.ndarray | None] = lambda key: None,
        epoch: Epoch | None = None,
    ) -> None:
        self._sam3 = sam3
        self._depth = DepthEstimator(unidepth) if unidepth is not None else None
        self._cameras = dict(cameras)
        self._intrinsics = intrinsics
        self._epoch = epoch if epoch is not None else Epoch()
        self._book = DetectionBook(self._epoch)
        self._frames: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
        self._enhanced: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_urls(
        cls,
        *,
        sam3: str | None,
        unidepth: str | None,
        cameras: Cameras,
        intrinsics: Callable[[str], np.ndarray | None] = lambda key: None,
    ) -> Perception | None:
        """A Perception for the given service URLs, or None when both are empty."""
        if not sam3 and not unidepth:
            return None
        from pi_embodied_services.utils.rpc.http_rpc import HttpRpcClient

        return cls(
            sam3=HttpRpcClient(sam3) if sam3 else None,
            unidepth=HttpRpcClient(unidepth) if unidepth else None,
            cameras=cameras,
            intrinsics=intrinsics,
        )

    # -- wiring ----------------------------------------------------------------

    def capabilities(self) -> dict[str, bool]:
        return {
            "segment": self._sam3 is not None,
            "enhance_depth": self._depth is not None,
        }

    def install(self, facade: Any) -> None:
        """Wrap ``env.get_observation`` / ``env.get_env_meta`` and add the primitives."""
        rpc: dict[str, Callable[..., Any]] = facade._rpc
        observe = rpc["env.get_observation"]
        meta = rpc["env.get_env_meta"]

        def get_observation(*args: Any, **kwargs: Any) -> Any:
            obs = observe(*args, **kwargs)
            self.observe(obs)
            return obs

        def get_env_meta(*args: Any, **kwargs: Any) -> Any:
            out = meta(*args, **kwargs)
            if isinstance(out, dict):
                caps = dict(out.get("capabilities") or {})
                caps["perception"] = self.capabilities()
                out = {**out, "capabilities": caps}
            return out

        rpc["env.get_observation"] = get_observation
        rpc["env.get_env_meta"] = get_env_meta
        self._epoch.install(facade, MOTION_METHODS)
        if self._sam3 is not None:
            rpc["env.segment"] = self.segment
            rpc["env.select_detection"] = self.select_detection
            rpc["env.reject_detection"] = self.reject_detection
            facade._readonly_methods.update(
                {"env.segment", "env.select_detection", "env.reject_detection"}
            )
        if self._depth is not None:
            rpc["env.enhance_depth"] = self.enhance_depth

    def observe(self, obs: Any) -> list[str]:
        """A new observation: cache its frames, invalidate every id. Returns the ids."""
        dropped = self._book.ids
        self._epoch.tick()
        self._frames = {}
        self._enhanced = {}
        if isinstance(obs, dict):
            for alias, (image_key, depth_key, index) in self._cameras.items():
                rgb = _pick(obs, image_key, index)
                if rgb is None or rgb.ndim != 3:
                    continue
                depth = _pick(obs, depth_key, index)
                if depth is not None and depth.shape != rgb.shape[:2]:
                    depth = None
                self._frames[alias] = (rgb, depth)
        return dropped

    # -- primitives ------------------------------------------------------------

    @property
    def book(self) -> DetectionBook:
        return self._book

    @property
    def epoch(self) -> Epoch:
        return self._epoch

    def frame(self, camera: str) -> tuple[np.ndarray, np.ndarray | None]:
        """The current observation's (rgb, depth) for a camera alias."""
        if camera not in self._cameras:
            raise ValueError(
                f"unknown camera {camera!r}; use one of {sorted(self._cameras)}"
            )
        if camera not in self._frames:
            raise ValueError(
                f"no current frame for camera {camera!r}: take an observation first"
                + (
                    ""
                    if self._frames
                    else " (the last env.get_observation returned no frames)"
                )
            )
        return self._frames[camera]

    def _K(self, camera: str) -> np.ndarray | None:
        image_key, _depth_key, index = self._cameras[camera]
        key = "main" if index is None else f"extra_{index}"
        try:
            return self._intrinsics(key)
        except Exception as exc:  # intrinsics are optional: no 3D, still masks
            logger.warning("intrinsics for %s unavailable: %s", camera, exc)
            return None

    def _sam3_call(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        result = self._sam3.call("sam3.segment", (), kwargs, timeout_s=120.0)
        if not isinstance(result, dict):
            raise RuntimeError(f"invalid SAM3 response: {result!r}")
        return result

    def segment(
        self,
        camera: str = "wrist",
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
        all: bool = False,
    ) -> dict[str, Any]:
        """SAM3 on the current frame of ``camera``; each mask gets a short id.

        ``all=False`` keeps the best mask only (one id). The result carries the
        overlay (uint8 [H, W, 3], every mask in its palette colour, the ids in rank
        order) and ``invalidated``: the ids dropped since the last perception call.
        """
        rgb, depth = self.frame(camera)
        invalidated = self._book.drain_invalidated()
        kwargs: dict[str, Any] = {
            "image_base64": _png_base64(rgb),
            "min_score": min_score,
        }
        if text_prompt is not None and text_prompt.strip():
            kwargs["text_prompt"] = text_prompt.strip()
        elif point is not None:
            kwargs["point"] = [int(point[0]), int(point[1])]
        else:
            raise ValueError("give a text_prompt or a point [row, col]")
        if all:
            raw = self._sam3_call({**kwargs, "all": True})
            candidates = list(raw.get("detections") or [])
        else:
            raw = self._sam3_call(kwargs)
            candidates = [raw] if raw.get("found") else []
        K = self._K(camera)
        detections: list[dict[str, Any]] = []
        masks: list[np.ndarray] = []
        for rank, cand in enumerate(candidates):
            png = cand.get("mask_png_base64")
            if not png:
                continue
            mask = decode_mask_png(png)
            if mask.shape != rgb.shape[:2] or not mask.any():
                continue
            item = {
                "camera": camera,
                "prompt": kwargs.get("text_prompt"),
                "point": kwargs.get("point"),
                "rank": rank,
                "score": cand.get("score"),
                "box": cand.get("box"),
                **describe_mask(mask, depth, K),
                "mask_png_base64": png,
                "mask": mask,
            }
            item["id"] = self._book.add(item)
            detections.append(public(item))
            masks.append(mask)
        out: dict[str, Any] = {
            "found": bool(detections),
            "observation": self._epoch.observation,
            "camera": camera,
            "count": len(detections),
            "detections": detections,
            "ids": [d["id"] for d in detections],
            "invalidated": invalidated,
        }
        if detections:
            out["overlay"] = overlay_masks(rgb, masks)
        elif raw.get("reason"):
            out["reason"] = raw["reason"]
        return out

    def _resolve(self, id: str, action: str) -> dict[str, Any]:
        invalidated = self._book.drain_invalidated()
        try:
            item = getattr(self._book, action)(str(id))
        except DetectionStale as exc:
            return {
                "ok": False,
                "id": id,
                "error": str(exc),
                "invalidated": invalidated,
                **self._book.summary(),
            }
        return {
            "ok": True,
            "detection": public(item),
            "invalidated": invalidated,
            **self._book.summary(),
        }

    def select_detection(self, id: str) -> dict[str, Any]:
        """Make ``id`` the selected mask of the current observation."""
        return self._resolve(id, "select")

    def reject_detection(self, id: str) -> dict[str, Any]:
        """Exclude ``id`` (it stays resolvable but is marked rejected)."""
        return self._resolve(id, "reject")

    def enhance_depth(self, camera: str = "wrist") -> dict[str, Any]:
        """Replace ``camera``'s depth in the current observation with the fused depth.

        Sensor holes are filled with the UniDepth estimate scaled to their overlap;
        a camera without depth takes the estimate as-is (``utils/depth.py``). Returns
        the fused depth (float32 [H, W], metres, 0 = none) and the fusion report.
        """
        rgb, sensor = self.frame(camera)
        mono, info = self._depth.estimate(rgb, self._K(camera))
        fused, report = fuse_depth(sensor, mono)
        self._frames[camera] = (rgb, fused)
        self._enhanced[camera] = report
        return {
            "ok": True,
            "observation": self._epoch.observation,
            "camera": camera,
            "depth": fused,
            "report": report,
            "estimate": info,
            "invalidated": self._book.drain_invalidated(),
        }


__all__ = ["FRANKA_CAMERAS", "Cameras", "Perception", "franka_intrinsics"]
