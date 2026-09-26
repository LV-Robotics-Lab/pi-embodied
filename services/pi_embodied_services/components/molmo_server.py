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
# the only --transport; install and PYTHONPATH hints point at services/; the point
# parser skips Molmo2's leading image index; activations are released after each call;
# --offload-blocks keeps decoder blocks in host memory for a shared GPU; --model
# molmopoint serves allenai/MolmoPoint-8B and its multi-image ``molmo.ground_set``.

"""RPC server owning the local Molmo visual-grounding model.

Run manually with::

    MOLMO_CHECKPOINT_PATH=/path/to/molmo PYTHONPATH=/path/to/pi/services \
        python -m pi_embodied_services.components.molmo_server \
        --transport http --host 127.0.0.1 --port 8115

Runs under the ``molmo`` extra's own interpreter, which need not have
``pi-embodied-services`` installed -- hence the explicit ``PYTHONPATH``.

Where SAM3 answers "which pixels are this phrase", Molmo answers "where would
you put the gripper" -- an open-vocabulary point on a named object, for phrases
no mask proposal names. The service exposes a ``molmo.ground`` RPC method over
HTTP.

Two backends, chosen with ``--model`` (the checkpoint path stays
``MOLMO_CHECKPOINT_PATH``):

- ``molmo2`` (default): ``allenai/Molmo2-8B``; ``molmo.ground`` as before.
- ``molmopoint``: ``allenai/MolmoPoint-8B`` (OpenETA's ``tools/molmopoint_core.py``,
  revision ``188130f``), a pointing model whose remote code returns pixel points with
  the index of the image they lie in. It adds ``molmo.ground_set(images_base64, query)``: OpenETA's
  Pointing Image Set -- one prompt over an ordered set of up to four images, the
  prompt preserved as authored (it may say "Image 1"), every point tagged with its
  0-based ``image_index``. Its ``molmo.ground`` is the one-image case (first point).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any

from PIL import Image

from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.rpc import RpcFacade

logger = get_logger("molmo_server")

#: Molmo2 writes ``<points coords="...">``: an image index, then ``point-id x y`` triples in
#: normalized thousandths (``1 1 424 446``); older outputs omit the image index (``1 424 446``).
_COORDS = re.compile(r"<(?:point|points)\b[^>]*\bcoords=[\"']([^\"']+)[\"']", re.I)


def _parse_point(answer: str) -> tuple[float, float] | None:
    """Return the first normalized Molmo2 point from generated markup."""
    coords = _COORDS.search(answer)
    if coords is None:
        return None
    tokens = re.split(r"[\s:;,]+", coords.group(1).split("\t")[0].strip())
    if not all(t.isdigit() for t in tokens) or len(tokens) < 3:
        return None
    if len(tokens) % 3 == 1:
        tokens = tokens[1:]
    x, y = float(tokens[1]), float(tokens[2])
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return x, y


@dataclass
class MolmoResult:
    """Wire response carrying at most one pixel."""

    point_xy: list[float] | None = None
    answer: str | None = None
    image_size: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


#: ``--model`` values; the HF ids are accepted as aliases.
BACKENDS = {
    "molmo2": "molmo2",
    "allenai/molmo2-8b": "molmo2",
    "molmopoint": "molmopoint",
    "allenai/molmopoint-8b": "molmopoint",
}
#: Pointing Image Set bounds (OpenETA's molmopoint_core).
MAX_IMAGES = 4
MAX_NEW_TOKENS_POINT_SET = 200


class MolmoFacade(RpcFacade):
    """RPC server wrapping the local Molmo visual-grounding model."""

    SERVICE_NAME = "molmo"

    def __init__(
        self, checkpoint: str, offload_blocks: int = 0, model: str = "molmo2"
    ) -> None:
        super().__init__()
        backend = BACKENDS.get(model.strip().lower())
        if backend is None:
            raise ValueError(
                f"unknown --model {model!r}; use one of {sorted(set(BACKENDS.values()))}"
            )
        self.backend = backend
        self._load(checkpoint, offload_blocks)
        self._register_rpc()

    def _load(self, checkpoint: str, offload_blocks: int = 0) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "local Molmo dependencies are missing; install "
                '`pip install -e "services[molmo]"`'
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("local Molmo requires a CUDA-capable GPU")
        if self.backend == "molmopoint":
            if offload_blocks > 0:
                raise ValueError(
                    "--offload-blocks is only supported with --model molmo2"
                )
            # MolmoPoint's remote code pads on the left and needs its slow processor
            # for the pointing metadata (OpenETA's molmopoint_core._load_processor).
            self._processor = AutoProcessor.from_pretrained(
                checkpoint,
                trust_remote_code=True,
                padding_side="left",
                local_files_only=True,
                use_fast=False,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                checkpoint,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map={"": 0},
                local_files_only=True,
            ).eval()
            self._torch = torch
            self._lock = threading.Lock()
            return
        processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True, local_files_only=True
        )
        kwargs: dict[str, Any] = {}
        if offload_blocks > 0:
            # Sharing a GPU: keep the last decoder blocks in host memory (needs
            # accelerate). Only whole decoder blocks move; the vision backbone
            # reads its weights outside accelerate's hooks and must stay on the GPU.
            with open(os.path.join(checkpoint, "model.safetensors.index.json")) as f:
                names = json.load(f)["weight_map"]
            blocks = 1 + max(
                int(m.group(1))
                for m in map(
                    re.compile(r"model\.transformer\.blocks\.(\d+)\.").match, names
                )
                if m
            )
            device_map: dict[str, int | str] = {
                "model.vision_backbone": 0,
                "model.transformer.wte": 0,
                "model.transformer.ln_f": 0,
                "lm_head": 0,
            }
            for i in range(blocks):
                device_map[f"model.transformer.blocks.{i}"] = (
                    "cpu" if i >= blocks - offload_blocks else 0
                )
            kwargs = {"device_map": device_map}
        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            **kwargs,
        )
        model = (model if offload_blocks > 0 else model.to("cuda")).eval()
        self._torch = torch
        self._model = model
        self._processor = processor
        self._lock = threading.Lock()

    def _register_rpc(self) -> None:
        self._rpc["molmo.ground"] = self.ground
        self._readonly_methods.add("molmo.ground")
        if self.backend == "molmopoint":
            self._rpc["molmo.ground_set"] = self.ground_set
            self._readonly_methods.add("molmo.ground_set")

    # -- MolmoPoint: Pointing Image Set ---------------------------------------

    def _point_set_bytes(self, images: list[bytes], query: str) -> dict[str, Any]:
        """One pointing prompt over an ordered image set (MolmoPoint remote code)."""
        decoded = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        sizes = [[int(im.size[0]), int(im.size[1])] for im in decoded]
        content: list[dict[str, Any]] = [{"type": "text", "text": query}]
        content.extend({"type": "image", "image": im} for im in decoded)
        inputs = self._processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            return_pointing_metadata=True,
        )
        metadata = inputs.pop("metadata")
        inputs = {
            key: value.to(self._model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self._lock, self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                logits_processor=self._model.build_logit_processor_from_inputs(inputs),
                max_new_tokens=MAX_NEW_TOKENS_POINT_SET,
                do_sample=False,
            )
            self._torch.cuda.empty_cache()
        answer = self._processor.post_process_image_text_to_text(
            generated[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        raw_points = self._model.extract_image_points(
            answer,
            metadata["token_pooling"],
            metadata["subpatch_mapping"],
            metadata["image_sizes"],
        )
        points = []
        for raw in raw_points.tolist() if hasattr(raw_points, "tolist") else raw_points:
            _object_id, image_index, x, y = raw
            image_index, x, y = int(image_index), float(x), float(y)
            if not 0 <= image_index < len(sizes):
                raise RuntimeError(f"MolmoPoint returned image index {image_index}")
            width, height = sizes[image_index]
            if not (0.0 <= x < width and 0.0 <= y < height):
                continue  # a point outside its image is noise, not evidence
            points.append(
                {
                    "id": f"point_{len(points):03d}",
                    "image_index": image_index,
                    "pixel_x": x,
                    "pixel_y": y,
                }
            )
        return {
            "points": points,
            "point_count": len(points),
            "answer": answer,
            "image_sizes": sizes,
            "coordinate_convention": {
                "origin": "top_left",
                "x_direction": "right",
                "y_direction": "down",
                "units": "pixels",
            },
        }

    def ground_set(self, images_base64: list[str], query: str) -> dict[str, Any]:
        """OpenETA's Pointing Image Set: ``images_base64`` (1-4 base64 PNG/JPEG, the order the
        prompt refers to as Image 1, 2, ...) and the prompt as authored. Every returned
        point carries the 0-based ``image_index`` of the image it lies in."""
        if self.backend != "molmopoint":
            raise ValueError("molmo.ground_set needs --model molmopoint")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("molmo.ground_set requires a non-empty query")
        if (
            not isinstance(images_base64, list)
            or not 1 <= len(images_base64) <= MAX_IMAGES
        ):
            raise ValueError(f"images_base64 must hold 1 to {MAX_IMAGES} base64 images")
        decoded: list[bytes] = []
        for index, image in enumerate(images_base64):
            if not isinstance(image, str):
                raise ValueError(f"images_base64[{index}] must be a base64 string")
            data = base64.b64decode(image, validate=True)
            if not data:
                raise ValueError(f"images_base64[{index}] is empty")
            decoded.append(data)
        return self._point_set_bytes(decoded, query.strip())

    def _ground_bytes(self, image_bytes: bytes, query: str) -> MolmoResult:
        """Run one grounding prompt against the loaded model."""
        if self.backend == "molmopoint":
            result = self._point_set_bytes([image_bytes], f"Point to {query}.")
            first = result["points"][0] if result["points"] else None
            return MolmoResult(
                point_xy=[first["pixel_x"], first["pixel_y"]] if first else None,
                answer=result["answer"],
                image_size=result["image_sizes"][0],
            )
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        prompt = (
            f"Point to {query} in this robot camera image. Choose the final "
            "safe manipulation point yourself, on visible object surface and "
            "away from edges. Return one point only."
        )
        inputs = self._processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image},
                    ],
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        with self._lock, self._torch.inference_mode():
            generated = self._model.generate(
                **inputs, max_new_tokens=48, do_sample=False
            )
            # The GPU is shared with the VLA and SAM3: hand activations back between calls.
            self._torch.cuda.empty_cache()
        answer = self._processor.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        normalized = _parse_point(answer)
        point = None
        if normalized is not None:
            x, y = normalized
            point = [
                x / 1000 * width,
                y / 1000 * height,
            ]
        return MolmoResult(point_xy=point, answer=answer, image_size=[width, height])

    def ground(self, image_base64: str, query: str) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("ground requires a non-empty query")
        query = query.strip()
        if not isinstance(image_base64, str):
            raise ValueError("image_base64 must be a string")

        image_bytes = base64.b64decode(image_base64, validate=True)
        if not image_bytes:
            raise ValueError("image_base64 is empty")
        return self._ground_bytes(image_bytes, query).to_dict()


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pi-embodied local Molmo server")
    parser.add_argument("--transport", choices=["http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8115)
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--model",
        default="molmo2",
        help="Backend: molmo2 (allenai/Molmo2-8B, default) or molmopoint "
        "(allenai/MolmoPoint-8B, adds molmo.ground_set); the HF ids are accepted too. "
        "MOLMO_CHECKPOINT_PATH points at that model's weights.",
    )
    parser.add_argument(
        "--offload-blocks",
        type=int,
        default=0,
        help="Keep this many decoder blocks (0.39 GB each for Molmo2-8B) in host "
        "memory, for a GPU shared with the VLA and SAM3 (molmo2 only).",
    )
    parser.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    return parser


def main() -> None:
    """Load Molmo and serve until terminated."""
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    checkpoint = os.environ.get("MOLMO_CHECKPOINT_PATH")
    if not checkpoint:
        raise RuntimeError(
            "MOLMO_CHECKPOINT_PATH is not set; export the path to the Molmo "
            "weights before starting pi-embodied"
        )
    facade = MolmoFacade(checkpoint, args.offload_blocks, args.model)
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
