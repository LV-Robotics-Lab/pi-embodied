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

"""RPC server wrapping OpenVLA (openvla/openvla) for LIBERO.

One 7-D action per call from the agentview only (OpenVLA has no wrist camera
or proprioception): the frame (already rotated 180 degrees by RLinf's LiberoEnv,
like OpenVLA's LIBERO evaluation does) is JPEG round-tripped and Lanczos-resized to 224 like its
``resize_image``, center-cropped to 90% of its area and resized back like its ``crop_and_resize``
(the published fine-tunes were evaluated with ``center_crop=True``; ``--no-center-crop`` skips
it), prompted with ``In: What action should the robot take to {task}?\\nOut:`` and decoded
greedily; the gripper is binarised and inverted as in its LIBERO evaluation.

The checkpoint is a published LIBERO fine-tune, pinned by commit hash in
``OPENVLA_CHECKPOINTS`` and fetched through ``HF_ENDPOINT`` (hf-mirror) when
``--model-path`` is not a directory. Run it in the ``openvla`` venv
(``services[openvla]``, see README).
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable

import numpy as np

from pi_embodied_services.components.vla_adapter_base import (
    ChunkVLAFacade,
    Frame,
    add_server_args,
    apply_cuda_device,
    center_crop_resize,
    libero_gripper,
    resize_jpeg_lanczos,
    snapshot,
)
from pi_embodied_services.utils.logging import get_logger

logger = get_logger("openvla_server")

#: LIBERO fine-tunes published by the OpenVLA authors: suite -> (repo, commit hash).
OPENVLA_CHECKPOINTS: dict[str, tuple[str, str]] = {
    "libero_spatial": (
        "openvla/openvla-7b-finetuned-libero-spatial",
        "962318cec55ac10993ff0f5f43eda9a270b4c873",
    ),
    "libero_object": (
        "openvla/openvla-7b-finetuned-libero-object",
        "287d6cfdf12d07b1449505f66d9bf3550257e9b3",
    ),
    "libero_goal": (
        "openvla/openvla-7b-finetuned-libero-goal",
        "fa5ae1e7509348889295bba8e08621d8b55e9baf",
    ),
    "libero_10": (
        "openvla/openvla-7b-finetuned-libero-10",
        "80970322773f81baa2e22fe495d0487b93a05cfa",
    ),
}
IMAGE_SIZE = 224
#: OpenVLA's LIBERO evaluation crops the centered 90% of the image area (``center_crop=True``).
CROP_SCALE = 0.9

Policy = Callable[[np.ndarray, str], np.ndarray]


def prompt_for(instruction: str) -> str:
    """OpenVLA's LIBERO prompt (``get_vla_action``)."""
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


class OpenVLAFacade(ChunkVLAFacade):
    """``policy(image_224_rgb, instruction) -> raw 7-D action`` (denormalised, gripper in [0, 1])."""

    SERVICE_NAME = "openvla"
    horizon = 1
    uses_wrist = False

    def __init__(
        self,
        *,
        policy: Policy,
        model: str,
        revision: str | None,
        suite: str | None = None,
        center_crop: bool = True,
    ):
        self._policy = policy
        self._center_crop = center_crop
        super().__init__(model=model, revision=revision, suite=suite)

    def _act(self, frame: Frame) -> np.ndarray:
        image = resize_jpeg_lanczos(frame.main, IMAGE_SIZE)
        if self._center_crop:
            image = center_crop_resize(image, CROP_SCALE)
        return libero_gripper(
            np.asarray(self._policy(image, frame.instruction), np.float32)
        )


def load_policy(path: str, unnorm_key: str | None, attn: str) -> tuple[Policy, str]:
    """The checkpoint's processor + ``OpenVLAForActionPrediction`` on the GPU, as a :data:`Policy`."""
    import torch
    from PIL import Image
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import (
        PrismaticImageProcessor,
        PrismaticProcessor,
    )
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
    )

    # The LIBERO fine-tunes ship no remote code: register the repo's classes (its get_vla does).
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    keys = list(config.norm_stats)
    if unnorm_key is None:
        if len(keys) != 1:
            raise ValueError(f"--unnorm-key is required; the checkpoint has {keys}")
        unnorm_key = keys[0]
    elif unnorm_key not in keys:
        raise ValueError(
            f"unknown --unnorm-key {unnorm_key!r}; the checkpoint has {keys}"
        )
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        path,
        attn_implementation=attn,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).cuda()
    model.eval()

    def policy(image: np.ndarray, instruction: str) -> np.ndarray:
        inputs = processor(prompt_for(instruction), Image.fromarray(image)).to(
            "cuda", dtype=torch.bfloat16
        )
        with torch.inference_mode():
            action = model.predict_action(
                **inputs, unnorm_key=unnorm_key, do_sample=False
            )
        return np.asarray(action, dtype=np.float32)

    return policy, unnorm_key


def main() -> None:
    p = argparse.ArgumentParser()
    add_server_args(p)
    p.add_argument(
        "--model-path",
        default=os.environ.get("OPENVLA_CHECKPOINT_PATH"),
        help="checkpoint directory or Hugging Face repo id (default OPENVLA_CHECKPOINT_PATH env)",
    )
    p.add_argument(
        "--suite",
        choices=sorted(OPENVLA_CHECKPOINTS),
        default="libero_spatial",
        help="which published LIBERO fine-tune to fetch when --model-path is unset",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="commit hash to fetch (default: the pin in OPENVLA_CHECKPOINTS for --suite)",
    )
    p.add_argument(
        "--unnorm-key",
        default=None,
        help="norm_stats key for action denormalisation (default: the checkpoint's only key)",
    )
    p.add_argument(
        "--center-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="the 0.9 center crop of OpenVLA's LIBERO evaluation (the checkpoints were evaluated with it)",
    )
    p.add_argument(
        "--attn", default="sdpa", help="attention implementation (sdpa | eager)"
    )
    args = p.parse_args()
    apply_cuda_device(args.cuda_device)

    repo, pin = OPENVLA_CHECKPOINTS[args.suite]
    model = args.model_path or repo
    revision = args.revision or (pin if model == repo else None)
    # A custom checkpoint's suite is unknown: only the published fine-tunes report theirs.
    suite = args.suite if model == repo else None
    t0 = time.time()
    path = snapshot(model, revision)
    logger.info("loading OpenVLA from %s (revision=%s) ...", path, revision)
    policy, unnorm_key = load_policy(path, args.unnorm_key, args.attn)
    logger.info("model ready in %.1fs (unnorm_key=%s)", time.time() - t0, unnorm_key)
    OpenVLAFacade(
        policy=policy,
        model=model,
        revision=revision,
        suite=suite,
        center_crop=args.center_crop,
    ).serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
