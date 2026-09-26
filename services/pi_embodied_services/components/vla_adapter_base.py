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

"""Base for the third-party VLA adapters (OpenVLA, OpenVLA-OFT, GR00T).

Every adapter speaks the Pi0.5 wire format, so a robot mounts them the way it
mounts ``pi0_pick``: ``vla.predict(obs, options)`` takes the env-native LIBERO
observation (``main_images`` uint8 [1, H, W, 3], ``wrist_images`` or None,
``states`` float32 [1, 8], ``task_descriptions`` [str]) and returns a float32
action chunk [1, horizon, 7] in the env's own OSC space (xyz delta, axis-angle
delta, gripper -1 open / +1 close), ready for ``env.chunk_step``. The model's
preprocessing (flips, crops, prompt templates) and its action denormalisation
live in the adapter, not in the robot.

``options["seed"]`` seeds every RNG for that one call (``seeded``), so a
replayed episode gets bit-identical actions from a sampler; greedy models are
deterministic anyway.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from pi_embodied_services.components.vla_facade_base import (
    BaseVLAFacade,
    inference_seed,
    seeded,
)

ACTION_DIM = 7


@dataclass(frozen=True)
class Frame:
    """One env-native observation: images as the simulator renders them (LIBERO's are upside down)."""

    main: np.ndarray
    wrist: np.ndarray | None
    state: np.ndarray
    instruction: str


def _image(name: str, v: Any, required: bool) -> np.ndarray | None:
    if v is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    a = np.asarray(v)
    if a.ndim == 4:
        if a.shape[0] != 1:
            raise ValueError(f"{name}: batch must be 1, got {a.shape}")
        a = a[0]
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError(
            f"{name}: expected uint8 [1, H, W, 3], got {a.dtype} {a.shape}"
        )
    return a


def frame_of(obs: dict) -> Frame:
    """The Pi0.5 wire dict as one :class:`Frame`; shapes are checked, values are not copied."""
    main = _image("main_images", obs.get("main_images"), required=True)
    wrist = _image("wrist_images", obs.get("wrist_images"), required=False)
    state = np.asarray(obs["states"], dtype=np.float32).reshape(-1)
    tasks = obs.get("task_descriptions") or []
    if len(tasks) != 1 or not isinstance(tasks[0], str):
        raise ValueError(f"task_descriptions must be one string, got {tasks!r}")
    return Frame(main=main, wrist=wrist, state=state, instruction=tasks[0])


def flip180(img: np.ndarray) -> np.ndarray:
    """LIBERO renders upside down; the OpenVLA family trains on ``img[::-1, ::-1]``."""
    return np.ascontiguousarray(img[::-1, ::-1])


def resize_jpeg_lanczos(img: np.ndarray, size: int) -> np.ndarray:
    """OpenVLA's ``resize_image``: a JPEG round trip, then Lanczos-3 antialiased resizing to ``size``.

    The original uses tf.image (encode_jpeg / decode_jpeg / resize lanczos3 antialias); this is
    the PIL equivalent (quality 95, LANCZOS), so no TensorFlow in the adapter venvs.
    """
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=95)
    buf.seek(0)
    decoded = Image.open(buf).convert("RGB")
    return np.asarray(decoded.resize((size, size), Image.LANCZOS), dtype=np.uint8)


def center_crop_resize(img: np.ndarray, crop_scale: float) -> np.ndarray:
    """OpenVLA-OFT's ``crop_and_resize``: keep the centered ``sqrt(crop_scale)`` fraction of each side,
    then resize back to the input size with bilinear interpolation (its
    ``tf.image.crop_and_resize`` default)."""
    from PIL import Image

    h, w = img.shape[:2]
    frac = np.sqrt(crop_scale)
    ch, cw = h * frac, w * frac
    top, left = (h - ch) / 2, (w - cw) / 2
    box = (left, top, left + cw, top + ch)
    resized = Image.fromarray(img).resize((w, h), Image.BILINEAR, box=box)
    return np.asarray(resized, dtype=np.uint8)


def libero_gripper(actions: np.ndarray) -> np.ndarray:
    """OpenVLA's LIBERO action postprocessing: the gripper from [0, 1] to {-1, +1}
    (``normalize_gripper_action(binarize=True)``), then inverted
    (``invert_gripper_action``) because the dataset's +1 is "open" and LIBERO's is "close"."""
    out = np.array(actions, dtype=np.float32, copy=True)
    out[..., 6] = -np.sign(out[..., 6] * 2.0 - 1.0)
    return out


def snapshot(repo_or_path: str, revision: str | None) -> str:
    """A local checkpoint directory: ``repo_or_path`` itself when it exists, else the Hugging Face
    snapshot of that repo at ``revision`` (the pinned commit hash), fetched through the mirror in
    ``HF_ENDPOINT`` (the servers never go through a proxy of their own)."""
    if os.path.isdir(repo_or_path):
        return repo_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_or_path, revision=revision)


class ChunkVLAFacade(BaseVLAFacade):
    """A VLA that maps one :class:`Frame` to an action chunk [horizon, 7] in LIBERO's OSC space.

    Subclasses implement ``_act(frame) -> ndarray[horizon, 7]`` (already denormalised and
    gripper-converted), set ``SERVICE_NAME`` and ``horizon``, and load their model in
    ``__init__`` before calling ``super().__init__``. The base owns the RPC surface:

    - ``vla.predict(obs, options)`` -> float32 [1, horizon, 7]
    - ``vla.reset()`` -> ``{"ok": true}`` (``_reset`` for models with per-episode state)
    - ``vla.info()`` -> ``{"service", "model", "revision", "horizon", "action_dim", "wrist"}``
    """

    horizon: int = 1
    #: Whether the model consumes the wrist camera (info for the robot's prompt and result).
    uses_wrist: bool = False

    def __init__(self, *, model: str, revision: str | None):
        self._model_id = model
        self._revision = revision
        super().__init__()

    def _register_rpc(self):
        super()._register_rpc()
        self._rpc["vla.reset"] = self.reset
        self._rpc["vla.info"] = self.info

    def info(self) -> dict:
        return {
            "service": self.service_name,
            "model": self._model_id,
            "revision": self._revision,
            "horizon": self.horizon,
            "action_dim": ACTION_DIM,
            "wrist": self.uses_wrist,
        }

    def reset(self) -> dict:
        self._reset()
        return {"ok": True}

    def _reset(self) -> None:
        """Hook for per-episode model state; stateless models leave it."""

    def predict(self, obs: dict, options: dict | None = None) -> np.ndarray:
        frame = frame_of(obs)
        with seeded(inference_seed(options)):
            actions = np.asarray(self._act(frame), dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None]
        if (
            actions.shape != (self.horizon, ACTION_DIM)
            or not np.isfinite(actions).all()
        ):
            raise RuntimeError(
                f"{self.service_name}: expected finite actions [{self.horizon}, {ACTION_DIM}], "
                f"got {actions.shape}"
            )
        return actions[None]

    def _act(self, frame: Frame) -> np.ndarray:
        raise NotImplementedError


def add_server_args(p) -> None:
    """The CLI flags every model server takes (see PROTOCOL.md)."""
    p.add_argument("--transport", choices=["http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )


def apply_cuda_device(device: int | None) -> None:
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
