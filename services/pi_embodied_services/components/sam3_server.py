# Copyright 2026 The RPent Authors.
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
#
# Modified by pi-embodied: import paths rewritten; healthz service name; HTTP is
# the only --transport; install hint points at services/.

"""RPC server owning the local SAM 3.0 image segmentation model.

Run manually with::

    SAM3_CHECKPOINT_PATH=/path/to/sam3.pt \
        python -m pi_embodied_services.components.sam3_server \
        --transport http --host 127.0.0.1 --port 8114

pi-embodied normally starts this process automatically. The service exposes a
``sam3.segment`` RPC method over HTTP.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import logging
import os
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.rpc import RpcFacade

logger = get_logger("sam3_server")


@dataclass
class Sam3Result:
    """Segmentation result with at most one compressed binary mask."""

    found: bool
    score: float | None = None
    box: list[float] | None = None
    mask_png_base64: str | None = None
    mask_shape: list[int] | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _encode_mask_png(mask: np.ndarray) -> str:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class Sam3Facade(RpcFacade):
    """RPC server wrapping the local SAM 3.0 image segmentation model."""

    SERVICE_NAME = "sam3"

    def __init__(self, checkpoint: str) -> None:
        super().__init__()
        self._load(checkpoint)
        self._register_rpc()

    def _load(self, checkpoint: str) -> None:
        """Load the official SAM 3.0 model and interactive point head."""
        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise RuntimeError(
                "local SAM3 dependencies are missing; install "
                '`pip install -e "services[sam3]"` or a LIBERO variant'
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("local SAM3 requires a CUDA-capable GPU")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(0)

        resolved = Path(checkpoint).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {resolved}")
        checkpoint_path = str(resolved)

        logger.info("loading SAM 3.0 checkpoint: %s", checkpoint_path)
        try:
            model = build_sam3_image_model(
                device="cuda",
                checkpoint_path=checkpoint_path,
                load_from_HF=False,
                enable_inst_interactivity=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to load SAM 3.0 checkpoint: {checkpoint_path}"
            ) from exc
        self._torch = torch
        self._model = model
        self._processor = Sam3Processor(model, device="cuda", confidence_threshold=0.0)
        self._device = "cuda"
        self._lock = threading.Lock()
        self._image_digest: str | None = None
        self._image_state: dict[str, Any] | None = None

    def _register_rpc(self) -> None:
        self._rpc["sam3.segment"] = self.segment
        self._readonly_methods.add("sam3.segment")

    def _segment_bytes(
        self,
        image_bytes: bytes,
        *,
        text_prompt: str | None,
        point: list[int] | None,
        min_score: float,
        all: bool = False,
    ) -> Sam3Result | dict[str, Any]:
        """Run one prompt against the cached latest-image features."""
        with self._lock:
            state = self._state_for_image(image_bytes)
            height = int(state["original_height"])
            width = int(state["original_width"])
            if text_prompt is not None:
                return self._segment_text(state, text_prompt, min_score, all)
            assert point is not None
            row, col = point
            if row < 0 or col < 0 or row >= height or col >= width:
                raise ValueError(
                    f"point [row, col] {point} is outside image shape "
                    f"[{height}, {width}]"
                )
            return self._segment_point(state, row, col, min_score, all)

    def _inference_context(self):
        if self._torch is None or not self._device.startswith("cuda"):
            return nullcontext()
        return self._torch.autocast("cuda", dtype=self._torch.bfloat16)

    def _state_for_image(self, image_bytes: bytes) -> dict[str, Any]:
        digest = hashlib.sha256(image_bytes).hexdigest()
        if digest == self._image_digest and self._image_state is not None:
            return self._image_state
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = source.convert("RGB")
        except Exception as exc:
            raise ValueError(f"invalid image data: {exc}") from exc
        with self._inference_context():
            state = self._processor.set_image(image)
        self._image_digest = digest
        self._image_state = state
        return state

    def _segment_text(
        self,
        state: dict[str, Any],
        prompt: str,
        min_score: float,
        all: bool = False,
    ) -> Sam3Result | dict[str, Any]:
        with self._inference_context():
            output = self._processor.set_text_prompt(prompt=prompt, state=state)
        select = self._select_all if all else self._select_top
        return select(
            masks=output.get("masks"),
            scores=output.get("scores"),
            boxes=output.get("boxes"),
            min_score=min_score,
        )

    def _segment_point(
        self,
        state: dict[str, Any],
        row: int,
        col: int,
        min_score: float,
        all: bool = False,
    ) -> Sam3Result | dict[str, Any]:
        if getattr(self._model, "inst_interactive_predictor", None) is None:
            raise RuntimeError("SAM3 instance interactivity is not enabled")
        point_coords = np.asarray([[col, row]], dtype=np.float32)
        point_labels = np.asarray([1], dtype=np.int64)
        with self._inference_context():
            masks, scores, _ = self._model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        select = self._select_all if all else self._select_top
        return select(
            masks=masks,
            scores=scores,
            boxes=None,
            min_score=min_score,
        )

    @staticmethod
    def _candidates(
        masks: Any, scores: Any, boxes: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        """SAM3's candidate tensors as (masks [N,H,W], scores [N], boxes or None)."""
        if masks is None or scores is None:
            return None
        masks_array = (
            masks
            if isinstance(masks, np.ndarray)
            else masks.detach().float().cpu().numpy()
        )
        scores_array = (
            scores
            if isinstance(scores, np.ndarray)
            else scores.detach().float().cpu().numpy()
        ).reshape(-1)
        if masks_array.size == 0 or scores_array.size == 0:
            return None
        if masks_array.ndim == 4 and masks_array.shape[1] == 1:
            masks_array = masks_array[:, 0]
        if masks_array.ndim == 2:
            masks_array = masks_array[None]
        if masks_array.ndim != 3 or masks_array.shape[0] != scores_array.shape[0]:
            raise RuntimeError(
                "unexpected SAM3 candidate shapes: "
                f"masks={masks_array.shape}, scores={scores_array.shape}"
            )
        boxes_array = None
        if boxes is not None:
            boxes_array = (
                boxes
                if isinstance(boxes, np.ndarray)
                else boxes.detach().float().cpu().numpy()
            )
        return masks_array, scores_array, boxes_array

    @staticmethod
    def _box_of(boxes_array: np.ndarray | None, index: int) -> list[float] | None:
        if boxes_array is None:
            return None
        if boxes_array.ndim >= 2 and index < boxes_array.shape[0]:
            return [float(value) for value in boxes_array[index].reshape(-1)[:4]]
        return None

    @classmethod
    def _select_all(
        cls,
        *,
        masks: Any,
        scores: Any,
        boxes: Any,
        min_score: float,
    ) -> dict[str, Any]:
        """Every non-empty candidate at or above ``min_score``, best first.

        ``detections[i]``: ``index`` (rank), ``score``, ``box`` (when SAM3 gave one),
        ``area_px``, ``mask_png_base64``, ``mask_shape``. Point prompts yield SAM3's
        three multimask candidates, text prompts one per instance found.
        """
        arrays = cls._candidates(masks, scores, boxes)
        if arrays is None:
            return {
                "found": False,
                "count": 0,
                "detections": [],
                "reason": "SAM3 returned no candidate",
            }
        masks_array, scores_array, boxes_array = arrays
        detections: list[dict[str, Any]] = []
        for source in np.argsort(-scores_array, kind="stable"):
            score = float(scores_array[source])
            if score < min_score:
                break
            mask = np.asarray(masks_array[source]) > 0
            if mask.ndim != 2 or not mask.any():
                continue
            detection: dict[str, Any] = {
                "index": len(detections),
                "score": score,
                "area_px": int(mask.sum()),
                "mask_png_base64": _encode_mask_png(mask),
                "mask_shape": [int(mask.shape[0]), int(mask.shape[1])],
            }
            box = cls._box_of(boxes_array, int(source))
            if box is not None:
                detection["box"] = box
            detections.append(detection)
        out: dict[str, Any] = {
            "found": bool(detections),
            "count": len(detections),
            "detections": detections,
        }
        if not detections:
            out["reason"] = (
                f"no candidate at or above min_score {min_score:.3f} "
                f"(top score {float(scores_array.max()):.3f})"
            )
        return out

    @classmethod
    def _select_top(
        cls,
        *,
        masks: Any,
        scores: Any,
        boxes: Any,
        min_score: float,
    ) -> Sam3Result:
        arrays = cls._candidates(masks, scores, boxes)
        if arrays is None:
            return Sam3Result(found=False, reason="SAM3 returned no candidate")
        masks_array, scores_array, boxes_array = arrays

        index = int(np.argmax(scores_array))
        score = float(scores_array[index])
        box = cls._box_of(boxes_array, index)
        if score < min_score:
            return Sam3Result(
                found=False,
                score=score,
                box=box,
                reason=f"top score {score:.3f} is below min_score {min_score:.3f}",
            )

        mask = np.asarray(masks_array[index]) > 0
        if mask.ndim != 2 or not mask.any():
            return Sam3Result(
                found=False,
                score=score,
                box=box,
                reason="SAM3 returned an empty mask",
            )
        return Sam3Result(
            found=True,
            score=score,
            box=box,
            mask_png_base64=_encode_mask_png(mask),
            mask_shape=[int(mask.shape[0]), int(mask.shape[1])],
        )

    def segment(
        self,
        image_base64: str,
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
        all: bool = False,
    ) -> dict[str, Any]:
        """One mask (the best) or, with ``all=True``, every mask for the prompt.

        ``all=False``: ``{"found", "score"?, "box"?, "mask_png_base64"?, "mask_shape"?,
        "reason"?}`` as before. ``all=True``: ``{"found", "count", "detections": [...],
        "reason"?}`` (see :meth:`_select_all`).
        """
        has_text = isinstance(text_prompt, str) and bool(text_prompt.strip())
        has_point = point is not None
        if has_text == has_point:
            raise ValueError("provide exactly one of text_prompt or point")
        if has_text:
            text_prompt = text_prompt.strip()
        if has_point and (not isinstance(point, list) or len(point) != 2):
            raise ValueError("point must be [row, col]")
        try:
            min_score = float(min_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"min_score must be a number, got {min_score!r}") from exc
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        if not isinstance(image_base64, str):
            raise ValueError("image_base64 must be a string")

        image_bytes = base64.b64decode(image_base64, validate=True)
        if not image_bytes:
            raise ValueError("image_base64 is empty")
        response = self._segment_bytes(
            image_bytes,
            text_prompt=text_prompt,
            point=point,
            min_score=min_score,
            all=bool(all),
        )
        return response if isinstance(response, dict) else response.to_dict()


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pi-embodied local SAM 3.0 server")
    parser.add_argument("--transport", choices=["http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8114)
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    return parser


def main() -> None:
    """Load SAM3 and serve until terminated."""
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    if args.cuda_device is not None:
        target = str(args.cuda_device)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None and prev != target:
            logging.warning(
                "CUDA_VISIBLE_DEVICES=%s is already set; overriding with --cuda-device=%s",
                prev,
                args.cuda_device,
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = target
    checkpoint = os.environ.get("SAM3_CHECKPOINT_PATH")
    if not checkpoint:
        raise RuntimeError(
            "SAM3_CHECKPOINT_PATH is not set; export the path to sam3.pt "
            "before starting pi-embodied"
        )
    facade = Sam3Facade(checkpoint)
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
