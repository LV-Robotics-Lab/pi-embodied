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

"""RPC server owning a UniDepth V2 metric monocular depth model.

Run manually with::

    HF_ENDPOINT=https://hf-mirror.com UNIDEPTH_MODEL=lpiccinelli/unidepth-v2-vitl14 \
        PYTHONPATH=/path/to/pi/services \
        python -m pi_embodied_services.components.unidepth_server --port 18500

Runs under the ``unidepth`` extra's own interpreter (UniDepth pins its own timm and
xformers), which need not have ``pi-embodied-services`` installed -- hence the
explicit ``PYTHONPATH``. The checkpoint is OpenETA's (``tools/unidepth_v2_core.py``):
``lpiccinelli/unidepth-v2-vitl14``, the ViT-L/14 UniDepth V2, fetched through
``huggingface_hub`` (so ``HF_ENDPOINT`` selects the mirror) or given as a local
directory; OpenETA's ``resolution_level`` 4 is the default.

The service exposes ``depth.estimate`` over HTTP: metric depth in metres for one RGB
frame, with the camera intrinsics when the caller knows them (the env servers do;
without them UniDepth predicts the camera too).
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.rpc import RpcFacade

logger = get_logger("unidepth_server")

DEFAULT_MODEL = "lpiccinelli/unidepth-v2-vitl14"
DEFAULT_RESOLUTION_LEVEL = 4


class UniDepthBackend:
    """The torch side: load UniDepth V2 once, ``infer(rgb, K)`` -> depth, confidence."""

    def __init__(self, model: str, resolution_level: int) -> None:
        try:
            import torch
            from unidepth.models import UniDepthV2
        except ImportError as exc:
            raise RuntimeError(
                "UniDepth dependencies are missing; install "
                '`pip install -e "services[unidepth]"` into its own environment'
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("unidepth_server requires a CUDA-capable GPU")
        source = model
        local = Path(model).expanduser()
        if local.exists():
            source = str(local.resolve())
        logger.info("loading UniDepth V2: %s", source)
        self.model_id = source
        self._model = UniDepthV2.from_pretrained(source).to("cuda").eval()
        self._model.resolution_level = int(resolution_level)
        self._torch = torch

    def infer(
        self, rgb: np.ndarray, K: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        torch = self._torch
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).cuda()
        camera = (
            torch.tensor(np.asarray(K, dtype=np.float32), device="cuda")
            if K is not None
            else None
        )
        with torch.inference_mode():
            pred = self._model.infer(image, camera)
        depth = pred["depth"].detach().float().cpu().numpy().squeeze()
        confidence = pred.get("confidence")
        if confidence is not None:
            confidence = confidence.detach().float().cpu().numpy().squeeze()
        # The GPU is shared with the VLA and SAM3: hand activations back between calls.
        torch.cuda.empty_cache()
        return depth, confidence


def decode_rgb(rgb: Any) -> np.ndarray:
    """A wire image (uint8 [H, W, 3] ndarray, or base64 PNG/JPEG) as uint8 [H, W, 3]."""
    if isinstance(rgb, str):
        from PIL import Image

        with Image.open(io.BytesIO(base64.b64decode(rgb, validate=True))) as image:
            rgb = np.asarray(image.convert("RGB"))
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"rgb must be [H, W, 3], got shape {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8, got {array.dtype}")
    return np.ascontiguousarray(array)


class UniDepthFacade(RpcFacade):
    """RPC server wrapping one UniDepth V2 backend."""

    SERVICE_NAME = "unidepth"

    def __init__(self, backend: Any, *, resolution_level: int) -> None:
        super().__init__()
        self._backend = backend
        self._resolution_level = int(resolution_level)
        self._lock = threading.Lock()
        self._rpc["depth.estimate"] = self.estimate
        self._readonly_methods.add("depth.estimate")

    def estimate(self, rgb: Any, K: Any | None = None) -> dict[str, Any]:
        """Metric depth of ``rgb``.

        ``K``: optional 3x3 pinhole intrinsics of the image as given (letterboxed or
        resized images need the matching K). Returns ``depth`` float32 [H, W] in
        metres (0 where the model gave no finite positive depth), ``confidence``
        float32 [H, W] (higher is better) when the model provides it, and metadata.
        """
        image = decode_rgb(rgb)
        camera = None
        if K is not None:
            camera = np.asarray(K, dtype=np.float64)
            if camera.shape != (3, 3) or not np.all(np.isfinite(camera)):
                raise ValueError("K must be a finite 3x3 matrix")
            if camera[0, 0] <= 0 or camera[1, 1] <= 0:
                raise ValueError("K must have positive focal lengths")
        started = time.perf_counter()
        with self._lock:
            depth, confidence = self._backend.infer(image, camera)
        depth = np.asarray(depth, dtype=np.float32)
        if depth.shape != image.shape[:2]:
            raise RuntimeError(
                f"UniDepth depth shape {depth.shape} does not match image {image.shape[:2]}"
            )
        valid = np.isfinite(depth) & (depth > 0)
        depth = np.where(valid, depth, 0.0).astype(np.float32)
        out: dict[str, Any] = {
            "depth": depth,
            "model": getattr(self._backend, "model_id", "unidepth-v2"),
            "resolution_level": self._resolution_level,
            "used_intrinsics": camera is not None,
            "valid_ratio": float(valid.mean()),
            "depth_range_m": [float(depth[valid].min()), float(depth[valid].max())]
            if valid.any()
            else None,
            "inference_s": round(time.perf_counter() - started, 4),
        }
        if confidence is not None:
            confidence = np.asarray(confidence, dtype=np.float32)
            if confidence.shape == depth.shape:
                out["confidence"] = confidence
        return out


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pi-embodied UniDepth V2 server")
    parser.add_argument("--transport", choices=["http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18500)
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--resolution-level",
        type=int,
        default=DEFAULT_RESOLUTION_LEVEL,
        help="UniDepth V2 resolution level 0-9 (OpenETA uses 4).",
    )
    parser.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    return parser


def main() -> None:
    """Load UniDepth V2 and serve until terminated."""
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    if not 0 <= args.resolution_level <= 9:
        raise SystemExit("--resolution-level must be between 0 and 9")
    model = os.environ.get("UNIDEPTH_MODEL") or DEFAULT_MODEL
    facade = UniDepthFacade(
        UniDepthBackend(model, args.resolution_level),
        resolution_level=args.resolution_level,
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
