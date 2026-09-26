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

"""Short ids for segmentation masks, bound to the observation they came from.

OpenETA's short-id evidence chain: the planner never handles pixels or poses, only ids
such as ``d3``; the host resolves an id to its mask, the frame it was cut from and that
frame's calibration. An id is valid only while the observation it belongs to is the
current one. Once the robot moves, every earlier id is invalid, so a stale mask can never
drive a motion (stage 8's grasp and place primitives accept only ids the book still holds).
With a state digest (:meth:`Epoch.set_digest`), "moved" means the robot state actually
changed: a motion call refused before it moved, or a new camera observation of an unmoved
robot, keeps the ids; without one every motion call and observation expires them.

Ids are never reused within a server process: a stale id names exactly one past mask,
and the error for it says which observation it belonged to.

One server may keep several books (the ``env.segment`` book of ``utils/perception.py``
and the grasp planner's of ``utils/grasp.py``). They share one :class:`Epoch`: a single
observation counter and a single id counter, so an id names one mask across all books
(no ``d1`` in two books for two objects) and a motion expires every book at once.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Callable
from typing import Any

import numpy as np

#: Overlay colours, one per detection rank (cycled), as RGB.
PALETTE = (
    (255, 64, 64),
    (64, 160, 255),
    (64, 220, 96),
    (255, 200, 32),
    (220, 96, 255),
    (32, 220, 220),
    (255, 128, 32),
    (160, 160, 160),
)


class DetectionStale(ValueError):
    """The id is unknown or belongs to an observation that is no longer current."""


def decode_mask_png(mask_png_base64: str) -> np.ndarray:
    """A SAM3 ``mask_png_base64`` (8-bit grey PNG, 255 = inside) as a bool [H, W]."""
    from PIL import Image

    with Image.open(io.BytesIO(base64.b64decode(mask_png_base64))) as image:
        return np.asarray(image.convert("L")) >= 128


def describe_mask(
    mask: np.ndarray, depth: np.ndarray | None, K: Any | None
) -> dict[str, Any]:
    """Pixel statistics of a mask and, with depth and intrinsics, its camera-frame point.

    ``centroid_rc`` is the median row/col of the mask pixels; ``depth_m`` the median
    valid depth over them; ``point_camera`` that depth at the centroid through ``K``
    (OpenCV camera frame, metres). Fields that cannot be computed are ``None``.
    """
    rows, cols = np.nonzero(mask)
    out: dict[str, Any] = {
        "area_px": int(rows.size),
        "centroid_rc": None,
        "depth_m": None,
        "point_camera": None,
    }
    if not rows.size:
        return out
    row, col = int(np.median(rows)), int(np.median(cols))
    out["centroid_rc"] = [row, col]
    if depth is None or depth.shape != mask.shape:
        return out
    from pi_embodied_services.utils.depth import valid_depth

    z = depth[mask]
    z = z[valid_depth(z)]
    out["depth_valid_px"] = int(z.size)
    if z.size < 10:
        return out
    z_med = float(np.median(z))
    out["depth_m"] = round(z_med, 5)
    if K is None:
        return out
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        return out
    ray = np.linalg.inv(K) @ np.array([col, row, 1.0])
    out["point_camera"] = [round(float(v), 5) for v in ray * z_med]
    return out


def overlay_masks(rgb: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """``rgb`` with each mask blended in its palette colour (uint8 [H, W, 3])."""
    out = np.asarray(rgb, dtype=np.float32).copy()
    for rank, mask in enumerate(masks):
        color = np.asarray(PALETTE[rank % len(PALETTE)], dtype=np.float32)
        out[mask] = 0.55 * out[mask] + 0.45 * color
    return out.astype(np.uint8)


#: RPC methods that may change the robot's state: every id expires once they ran and the
#: state changed (every time, without a digest).
MOTION_METHODS = (
    "env.step",
    "env.chunk_step",
    "env.reset",
    "env.move_to",
    "env.move_delta",
    "env.rotate_wrist",
    "env.rotate_delta",
    "env.set_gripper",
    "env.recover_joint_posture",
    "env.execute_grasp",
    "env.execute_place",
    "code.run",
)

#: The robot-state keys :func:`state_digest` reads, wherever they sit in the state tree.
DIGEST_KEYS = ("tcp_pose", "gripper_position", "gripper_open")


def state_digest(
    state: Any, keys: tuple[str, ...] = DIGEST_KEYS, decimals: int = 3
) -> tuple:
    """A comparable fingerprint of the robot's pose in a (nested) robot-state dict: every
    ``keys`` entry at any depth, rounded to ``decimals`` (1 mm / about 2 mrad of quaternion
    at 3), so sensor noise on an unmoved arm does not read as motion. Empty when none is
    present."""
    out: list[tuple[str, Any]] = []

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        for key in sorted(node, key=str):
            value = node[key]
            where = f"{path}.{key}" if path else str(key)
            if key in keys:
                arr = np.asarray(value)
                if arr.dtype.kind in "biuf":
                    arr = np.round(arr.astype(np.float64), decimals) + 0.0
                    out.append((where, tuple(arr.reshape(-1).tolist())))
                else:
                    out.append((where, repr(value)))
            else:
                walk(value, where)

    walk(state, "")
    return tuple(out)


class Epoch:
    """The observation clock and id counter shared by every book of one server.

    ``tick()`` starts a new observation and tells every listener (the books, the planner's
    snapshot cache); ``next_id`` hands out ids that are unique across all books.
    ``install`` wraps a facade's motion methods so that each of them :meth:`refresh`-es
    after it ran, whatever it returned; wrapping the same method twice is a no-op.

    ``set_digest(fn)`` gives the clock a robot-state fingerprint (``fn() -> comparable``):
    :meth:`refresh` then ticks only when it differs from the one the current observation
    started with (a refused, unmoved motion keeps the ids). Without a digest, or when it
    cannot be read, every refresh ticks.
    """

    def __init__(self, digest: Callable[[], Any] | None = None) -> None:
        self._observation = 0
        self._counter = 0
        self._listeners: list[Callable[[int], None]] = []
        self._wrapped: set[str] = set()
        self._digest = digest
        self._baseline: Any = None

    @property
    def observation(self) -> int:
        return self._observation

    def set_digest(self, digest: Callable[[], Any] | None) -> None:
        self._digest = digest
        self._baseline = None

    def _read_digest(self) -> Any:
        if self._digest is None:
            return None
        try:
            return self._digest()
        except Exception:  # an unreadable state counts as changed
            return None

    def on_tick(self, listener: Callable[[int], None]) -> None:
        self._listeners.append(listener)

    def tick(self) -> int:
        self._observation += 1
        self._baseline = self._read_digest()
        for listener in self._listeners:
            listener(self._observation)
        return self._observation

    def changed(self) -> bool:
        """Whether the robot state differs from the current observation's (True when unknown)."""
        if self._digest is None or self._baseline is None:
            return True
        now = self._read_digest()
        return now is None or now != self._baseline

    def refresh(self) -> bool:
        """Tick when the robot state changed (always without a digest); True when it ticked."""
        if not self.changed():
            return False
        self.tick()
        return True

    def next_id(self, prefix: str = "d") -> str:
        # The state the observation's ids are bound to, when no tick has recorded one yet.
        if self._baseline is None:
            self._baseline = self._read_digest()
        self._counter += 1
        return f"{prefix}{self._counter}"

    def install(self, facade: Any, methods: tuple[str, ...] = MOTION_METHODS) -> None:
        """Wrap ``facade._rpc[name]`` for each of ``methods`` (those present) with a refresh."""
        rpc: dict[str, Callable[..., Any]] = facade._rpc
        for name in methods:
            fn = rpc.get(name)
            if fn is None or name in self._wrapped:
                continue
            self._wrapped.add(name)

            def wrapped(*args: Any, _fn: Callable[..., Any] = fn, **kwargs: Any) -> Any:
                try:
                    return _fn(*args, **kwargs)
                finally:
                    self.refresh()

            rpc[name] = wrapped


class DetectionBook:
    """The masks of the current observation, by short id.

    ``bind(observation_id)`` makes an observation current and drops (invalidates) the
    ids of the previous one (the book follows its epoch: every tick binds it); ``add``
    registers masks and hands out ids (``prefix`` marks
    their kind: ``d`` masks, ``g`` grasps, ``p`` placements); ``get`` / ``select`` /
    ``reject`` accept only current ids. ``drain_invalidated`` returns the ids dropped
    since the last drain so the server can tell the client.
    """

    def __init__(self, epoch: Epoch | None = None) -> None:
        self._epoch = epoch if epoch is not None else Epoch()
        self._epoch.on_tick(self.bind)
        self._observation: int | None = None
        self._items: dict[str, dict[str, Any]] = {}
        self._history: dict[str, int] = {}
        self._selected: str | None = None
        self._rejected: list[str] = []
        self._invalidated: list[str] = []

    @property
    def observation(self) -> int | None:
        return self._observation

    @property
    def ids(self) -> list[str]:
        return list(self._items)

    @property
    def selected(self) -> str | None:
        return self._selected

    @property
    def rejected(self) -> list[str]:
        return list(self._rejected)

    def bind(self, observation_id: int) -> list[str]:
        """Make ``observation_id`` current; returns the ids this invalidates."""
        if observation_id == self._observation:
            return []
        dropped = list(self._items)
        self._observation = observation_id
        self._items.clear()
        self._selected = None
        self._rejected = []
        self._invalidated.extend(dropped)
        return dropped

    def drain_invalidated(self) -> list[str]:
        dropped, self._invalidated = self._invalidated, []
        return dropped

    @property
    def epoch(self) -> Epoch:
        return self._epoch

    def add(self, detection: dict[str, Any], prefix: str = "d") -> str:
        """Register one detection (its ``mask`` and metadata) and return its id."""
        if self._observation is None:
            raise RuntimeError("no observation is bound; call bind() first")
        id = self._epoch.next_id(prefix)
        self._items[id] = {**detection, "id": id, "observation": self._observation}
        self._history[id] = self._observation
        return id

    def known(self, id: str) -> bool:
        """Whether this book ever handed out ``id`` (current or stale)."""
        return id in self._history

    def get(self, id: str) -> dict[str, Any]:
        item = self._items.get(id)
        if item is not None:
            return item
        past = self._history.get(id)
        if past is None:
            raise DetectionStale(
                f"unknown detection id {id!r}; current ids: {list(self._items)}"
            )
        raise DetectionStale(
            f"detection {id} is stale: it belongs to observation {past}, the current "
            f"observation is {self._observation}; segment again and use a new id"
        )

    def select(self, id: str) -> dict[str, Any]:
        item = self.get(id)
        self._selected = id
        if id in self._rejected:
            self._rejected.remove(id)
        return item

    def reject(self, id: str) -> dict[str, Any]:
        item = self.get(id)
        if id not in self._rejected:
            self._rejected.append(id)
        if self._selected == id:
            self._selected = None
        return item

    def summary(self) -> dict[str, Any]:
        return {
            "observation": self._observation,
            "ids": list(self._items),
            "selected": self._selected,
            "rejected": list(self._rejected),
        }


def public(item: dict[str, Any]) -> dict[str, Any]:
    """The wire view of a book item: everything but the mask array."""
    return {k: v for k, v in item.items() if k != "mask"}


__all__ = [
    "DIGEST_KEYS",
    "MOTION_METHODS",
    "PALETTE",
    "DetectionBook",
    "DetectionStale",
    "Epoch",
    "decode_mask_png",
    "describe_mask",
    "overlay_masks",
    "public",
    "state_digest",
]
