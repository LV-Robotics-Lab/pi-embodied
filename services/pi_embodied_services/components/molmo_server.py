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
# --max-gpu-memory offloads part of the weights for a shared GPU.

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
"""

from __future__ import annotations

import argparse
import base64
import io
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


class MolmoFacade(RpcFacade):
    """RPC server wrapping the local Molmo visual-grounding model."""

    SERVICE_NAME = "molmo"

    def __init__(self, checkpoint: str, max_gpu_memory: str | None = None) -> None:
        super().__init__()
        self._load(checkpoint, max_gpu_memory)
        self._register_rpc()

    def _load(self, checkpoint: str, max_gpu_memory: str | None = None) -> None:
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
        processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True, local_files_only=True
        )
        kwargs: dict[str, Any] = {}
        if max_gpu_memory:
            # Sharing a GPU: keep this much of the weights on it, the rest in host
            # memory (needs accelerate).
            kwargs = {
                "device_map": "auto",
                "max_memory": {0: max_gpu_memory, "cpu": "512GiB"},
            }
        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            **kwargs,
        )
        model = (model if max_gpu_memory else model.to("cuda")).eval()
        self._torch = torch
        self._model = model
        self._processor = processor
        self._lock = threading.Lock()

    def _register_rpc(self) -> None:
        self._rpc["molmo.ground"] = self.ground
        self._readonly_methods.add("molmo.ground")

    def _ground_bytes(self, image_bytes: bytes, query: str) -> MolmoResult:
        """Run one grounding prompt against the loaded model."""
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
        "--max-gpu-memory",
        default=None,
        help="Cap the weights kept on the GPU (e.g. 13GiB) and offload the rest to "
        "host memory, for a GPU shared with the VLA and SAM3.",
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
    facade = MolmoFacade(checkpoint, args.max_gpu_memory)
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
