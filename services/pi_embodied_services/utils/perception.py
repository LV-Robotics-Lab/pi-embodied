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

A server without a Franka-style ``env.get_observation`` (the simulators, which render on
demand) gives :class:`Perception` a ``view(camera) -> {"rgb", "depth"?, "intrinsic_K"?}``
instead of the observation layout: frames are then rendered when a primitive first needs
them and dropped at every new observation of the :class:`Epoch` (a motion). Such servers
already serve their own ``env.segment`` code primitive, so they install the id primitives
under other names (``names``, :data:`SIM_NAMES`: ``env.detect`` for the segmentation with
ids); ``render_view`` builds the view from ``env.render_camera`` / ``env.get_camera_meta``.

:class:`Perception` is installed on a facade after its own ``_register_rpc``: it wraps
``env.get_observation`` (a new observation invalidates the ids when the robot state changed
since they were cut, or always when the facade gave the :class:`Epoch` no state digest, and
when its frames show another scene than the ones the ids were cut from: an object moved by
hand, a restored table; ``env.segment`` / ``env.enhance_depth`` take a fresh observation
first, so they never work on a frame older than the call) and
the motion methods (``MOTION_METHODS``: a move invalidates them too, observed or not), wraps
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
    frame_signature,
    overlay_masks,
    public,
    scene_changed,
)
from pi_embodied_services.utils.logging import get_logger

logger = get_logger("perception")

#: camera alias -> (image key, depth key, index into a stacked extra view, or the key of a
#: per-camera dict such as the dual Franka's ``raw_camera_frames``, or None)
Cameras = dict[str, tuple[str, str, int | str | None]]

#: ``view(camera)``: one camera now, ``{"rgb": uint8[H, W, 3], "depth": float32[H, W] (metres)
#: or None, "intrinsic_K": [3, 3] or None}`` (the grasp planner's ``utils/grasp.View`` shape).
View = Callable[[str], dict[str, Any]]

#: The primitives' RPC names on a Franka server (the defaults) and on the other servers,
#: whose own ``env.segment`` is the code-mode primitive.
NAMES = {
    "segment": "env.segment",
    "select_detection": "env.select_detection",
    "reject_detection": "env.reject_detection",
    "enhance_depth": "env.enhance_depth",
}
SIM_NAMES = {**NAMES, "segment": "env.detect"}

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


def _pick(obs: dict[str, Any], key: str, index: int | str | None) -> np.ndarray | None:
    value = obs.get(key)
    if value is None:
        return None
    if isinstance(index, str):
        entry = value.get(index) if isinstance(value, dict) else None
        return None if entry is None else np.asarray(entry)
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
    returns the 3x3 K of an observation image key (``main``, ``extra_0``, or a camera's
    dict key such as ``d455_rgb``) or None. With ``view`` the frames come from it instead
    (``cameras`` is then the list of aliases it renders) and are dropped at every tick of
    the epoch; ``names`` renames the primitives' RPC methods (:data:`SIM_NAMES`).
    """

    def __init__(
        self,
        *,
        sam3: Any | None = None,
        unidepth: Any | None = None,
        cameras: Cameras | list[str] | tuple[str, ...],
        intrinsics: Callable[[str], np.ndarray | None] = lambda key: None,
        epoch: Epoch | None = None,
        view: View | None = None,
        names: dict[str, str] | None = None,
    ) -> None:
        self._sam3 = sam3
        self._depth = DepthEstimator(unidepth) if unidepth is not None else None
        if view is None and not isinstance(cameras, dict):
            raise ValueError(
                "cameras must map aliases to observation keys without a view"
            )
        self._cameras = (
            dict(cameras)
            if isinstance(cameras, dict)
            else {c: ("", "", None) for c in cameras}
        )
        self._intrinsics = intrinsics
        self._view = view
        self._names = {**NAMES, **(names or {})}
        self._epoch = epoch if epoch is not None else Epoch()
        self._book = DetectionBook(self._epoch)
        self._book.bind(self._epoch.observation)
        self._frames: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
        self._signatures: dict[str, dict[str, Any]] = {}
        #: the facade's own env.get_observation (install), for a fresh capture per call
        self._capture: Callable[[], Any] | None = None
        self._intr: dict[str, np.ndarray | None] = {}
        self._enhanced: dict[str, dict[str, Any]] = {}
        if view is not None:
            # No observation call rebinds the book: the current one is, until the next motion.
            # A rendered frame belongs to the observation it was rendered in.
            self._epoch.on_tick(lambda _observation: self._forget())

    def _forget(self) -> None:
        self._frames = {}
        self._intr = {}
        self._enhanced = {}

    @classmethod
    def from_urls(
        cls,
        *,
        sam3: str | None,
        unidepth: str | None,
        cameras: Cameras | list[str] | tuple[str, ...],
        intrinsics: Callable[[str], np.ndarray | None] = lambda key: None,
        view: View | None = None,
        names: dict[str, str] | None = None,
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
            view=view,
            names=names,
        )

    # -- wiring ----------------------------------------------------------------

    def capabilities(self) -> dict[str, bool]:
        return {
            "segment": self._sam3 is not None,
            "enhance_depth": self._depth is not None,
        }

    def methods(self) -> dict[str, str]:
        """The RPC name of each primitive this Perception serves."""
        on = self.capabilities()
        return {
            k: v
            for k, v in self._names.items()
            if on["enhance_depth" if k == "enhance_depth" else "segment"]
        }

    def install(
        self, facade: Any, *, mutating: tuple[str, ...] = MOTION_METHODS
    ) -> None:
        """Wrap ``env.get_observation`` (without a view) / ``env.get_env_meta`` and the
        motion methods (``mutating``), and add the primitives under their ``names``."""
        rpc: dict[str, Callable[..., Any]] = facade._rpc
        meta = rpc["env.get_env_meta"]
        if self._view is None:
            observe = rpc["env.get_observation"]

            def get_observation(*args: Any, **kwargs: Any) -> Any:
                obs = observe(*args, **kwargs)
                self.observe(obs)
                return obs

            rpc["env.get_observation"] = get_observation
            self._capture = observe

        def get_env_meta(*args: Any, **kwargs: Any) -> Any:
            out = meta(*args, **kwargs)
            if isinstance(out, dict):
                caps = dict(out.get("capabilities") or {})
                caps["perception"] = self.capabilities()
                out = {**out, "capabilities": caps}
            return out

        rpc["env.get_env_meta"] = get_env_meta
        self._epoch.install(facade, mutating)
        n = self._names
        if self._sam3 is not None:
            rpc[n["segment"]] = self.segment
            rpc[n["select_detection"]] = self.select_detection
            rpc[n["reject_detection"]] = self.reject_detection
            facade._readonly_methods.update(
                {n["segment"], n["select_detection"], n["reject_detection"]}
            )
        if self._depth is not None:
            rpc[n["enhance_depth"]] = self.enhance_depth

    def observe(self, obs: Any) -> list[str]:
        """A new observation. The ids expire (and its frames replace the cached ones) when the
        robot state changed since they were cut (:meth:`Epoch.refresh`) or a camera shows
        another scene (:func:`scene_changed`: an object moved under an unmoved robot); else
        the cached frames, their ids and any enhanced depth stay. Returns the ids dropped."""
        frames: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
        if isinstance(obs, dict):
            for alias, (image_key, depth_key, index) in self._cameras.items():
                rgb = _pick(obs, image_key, index)
                if rgb is None or rgb.ndim != 3:
                    continue
                depth = _pick(obs, depth_key, index)
                if depth is not None and depth.shape != rgb.shape[:2]:
                    depth = None
                frames[alias] = (rgb[..., :3], depth)
        signatures = {a: frame_signature(rgb, d) for a, (rgb, d) in frames.items()}
        ids = self._book.ids
        # Without a state digest the wrapped motions already tick; an observation alone then
        # expires ids only through its frames.
        moved = self._epoch.refresh() if self._epoch.has_digest else False
        # A camera seen for the first time has no ids cut from it: nothing to compare.
        changed = any(a not in signatures for a in self._signatures) or any(
            scene_changed(self._signatures[a], s)
            for a, s in signatures.items()
            if a in self._signatures
        )
        if not (moved or changed):
            for alias in set(frames) - set(self._frames):
                self._frames[alias] = frames[alias]
                self._signatures[alias] = signatures[alias]
            return []
        if not moved:
            self._epoch.tick()
        self._frames = frames
        self._signatures = signatures
        self._enhanced = {}
        return ids

    # -- primitives ------------------------------------------------------------

    @property
    def sam3(self) -> Any | None:
        """The SAM3 client (the grasp planner segments its ``object`` text with it too)."""
        return self._sam3

    @property
    def book(self) -> DetectionBook:
        return self._book

    @property
    def epoch(self) -> Epoch:
        return self._epoch

    def frame(self, camera: str) -> tuple[np.ndarray, np.ndarray | None]:
        """The current observation's (rgb, depth) for a camera alias, after a fresh capture
        (``observe``: the cached frame stays while the scene is the same)."""
        if camera not in self._cameras:
            raise ValueError(
                f"unknown camera {camera!r}; use one of {sorted(self._cameras)}"
            )
        if self._capture is not None:
            self.observe(self._capture())
        elif camera not in self._frames and self._view is not None:
            v = self._view(camera)
            rgb = np.asarray(v["rgb"])
            depth = v.get("depth")
            depth = None if depth is None else np.asarray(depth, dtype=np.float32)
            if depth is not None:
                depth = depth.reshape(depth.shape[:2])
            if depth is not None and depth.shape != rgb.shape[:2]:
                depth = None
            K = v.get("intrinsic_K")
            K = None if K is None else np.asarray(K, dtype=np.float64)
            self._intr[camera] = (
                K
                if K is not None and K.shape == (3, 3) and np.all(np.isfinite(K))
                else None
            )
            self._frames[camera] = (rgb[..., :3], depth)
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
        if self._view is not None:
            return self._intr.get(camera)
        image_key, _depth_key, index = self._cameras[camera]
        key = (
            "main"
            if index is None
            else index
            if isinstance(index, str)
            else f"extra_{index}"
        )
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


def render_view(
    facade: Any,
    *,
    size: int | None = None,
    intrinsics: bool = True,
    flip: bool = False,
    cameras: dict[str, str] | None = None,
) -> View:
    """A :data:`View` over a simulator facade's ``env.render_camera`` (``depth=True``: rgb, or
    ``[rgb, depth]``) and ``env.get_camera_meta``: the frame the model sees. ``size`` passes
    ``height`` / ``width`` to both; ``flip`` turns a bottom-up render (robosuite's native
    orientation) upright; ``cameras`` maps an alias to the server's camera name. K is left out
    where it does not describe the image: ``intrinsics=False`` (a letterbox, an OpenGL
    convention), a flipped image, a camera without metadata, or metadata of another size."""
    rpc = facade._rpc

    def view(camera: str) -> dict[str, Any]:
        name = (cameras or {}).get(camera, camera)
        sized = {"height": size, "width": size} if size else {}
        out = rpc["env.render_camera"](camera_name=name, depth=True, **sized)
        rgb, depth = (
            (out[0], out[1])
            if isinstance(out, (list, tuple)) and len(out) == 2
            else (out, None)
        )
        rgb = np.asarray(rgb)
        if depth is not None:
            depth = np.asarray(depth, dtype=np.float32)
        if flip:
            rgb = np.ascontiguousarray(rgb[::-1])
            depth = None if depth is None else np.ascontiguousarray(depth[::-1])
        K = None
        if intrinsics and not flip:
            try:
                meta = rpc["env.get_camera_meta"](camera_name=name, **sized)
            except Exception:  # a camera without calibration: masks, no camera point
                meta = None
            if isinstance(meta, dict):
                h, w = meta.get("height"), meta.get("width")
                if (h is None or int(h) == rgb.shape[0]) and (
                    w is None or int(w) == rgb.shape[1]
                ):
                    K = meta.get("intrinsic_K")
        return {"rgb": rgb, "depth": depth, "intrinsic_K": K}

    return view


def install_perception(
    facade: Any,
    args: Any,
    *,
    cameras: Cameras | list[str] | tuple[str, ...],
    view: View | None = None,
    intrinsics: Callable[[str], np.ndarray | None] = lambda key: None,
    mutating: tuple[str, ...] = (),
    names: dict[str, str] | None = None,
) -> Perception | None:
    """Install a :class:`Perception` for ``args.sam3`` / ``args.unidepth`` on a constructed
    facade (nothing without either): the primitives under ``names`` (default
    :data:`SIM_NAMES`), ids expiring on :data:`MOTION_METHODS` plus the robot's own
    ``mutating`` motions."""
    perception = Perception.from_urls(
        sam3=getattr(args, "sam3", None) or None,
        unidepth=getattr(args, "unidepth", None) or None,
        cameras=cameras,
        intrinsics=intrinsics,
        view=view,
        names=SIM_NAMES if names is None else names,
    )
    if perception is not None:
        perception.install(facade, mutating=MOTION_METHODS + tuple(mutating))
    return perception


def add_perception_arguments(parser: Any, *, sam3: bool = False) -> None:
    """``--unidepth`` (and ``--sam3`` for a server without one): the perception services."""
    if sam3:
        parser.add_argument(
            "--sam3",
            default="",
            help="SAM3 server URL: adds env.detect / env.select_detection / env.reject_detection",
        )
    parser.add_argument(
        "--unidepth",
        default="",
        help="UniDepth server URL: adds env.enhance_depth",
    )


__all__ = [
    "FRANKA_CAMERAS",
    "NAMES",
    "SIM_NAMES",
    "Cameras",
    "Perception",
    "View",
    "add_perception_arguments",
    "franka_intrinsics",
    "install_perception",
    "render_view",
]
