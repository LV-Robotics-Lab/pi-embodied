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

"""RPC server wrapping OpenVLA-OFT (moojink/openvla-oft) for LIBERO.

An 8-step chunk of 7-D actions per call from the agentview and wrist images
(both already rotated 180 degrees by RLinf's LiberoEnv) and the 8-D proprio state, through
the repository's own ``get_vla_action`` (its 224 resize, 0.9 center crop,
prompt and L1 action head + proprio projector), then the gripper is binarised
and inverted as in its LIBERO evaluation.

The checkpoint is a published LIBERO fine-tune, pinned by commit hash in
``OPENVLA_OFT_CHECKPOINTS`` and fetched through ``HF_ENDPOINT`` (hf-mirror)
when ``--model-path`` is not a directory. Run it in the ``openvla-oft`` venv
(``services[openvla-oft]``, see README).
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np

from pi_embodied_services.components.vla_adapter_base import (
    ChunkVLAFacade,
    Frame,
    add_server_args,
    apply_cuda_device,
    libero_gripper,
    snapshot,
)
from pi_embodied_services.utils.logging import get_logger

logger = get_logger("openvla_oft_server")

#: LIBERO fine-tunes published by the OpenVLA-OFT authors: suite -> (repo, commit hash).
OPENVLA_OFT_CHECKPOINTS: dict[str, tuple[str, str]] = {
    "libero_spatial": (
        "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "6d0231af0e48c5985f1ff86908f4674b84bc049b",
    ),
    "libero_object": (
        "moojink/openvla-7b-oft-finetuned-libero-object",
        "4c89574e1c538b6c102f43f0526d60a9d3650148",
    ),
    "libero_goal": (
        "moojink/openvla-7b-oft-finetuned-libero-goal",
        "c2d0f9fbbd82674683b397ff923168a12f6a307b",
    ),
    "libero_10": (
        "moojink/openvla-7b-oft-finetuned-libero-10",
        "95220f9a3421a7ff12d4218e73d09ade830fa9a3",
    ),
    "libero_all": (
        "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
        "638918f3d1c2e43a39a8a20772bdb8b91835e4b7",
    ),
}
#: OpenVLA-OFT's LIBERO configuration (prismatic/vla/constants.py, run_libero_eval.py).
HORIZON = 8
PROPRIO_DIM = 8

#: ``policy(main, wrist, state, instruction) -> raw [8, 7] actions`` (denormalised, gripper in [0, 1]);
#: the images are the env frames at their native size.
Policy = Callable[[np.ndarray, np.ndarray, np.ndarray, str], np.ndarray]


class OpenVLAOFTFacade(ChunkVLAFacade):
    SERVICE_NAME = "openvla-oft"
    horizon = HORIZON
    uses_wrist = True

    def __init__(
        self,
        *,
        policy: Policy,
        model: str,
        revision: str | None,
        suite: str | None = None,
    ):
        self._policy = policy
        super().__init__(model=model, revision=revision, suite=suite)

    def _act(self, frame: Frame) -> np.ndarray:
        if frame.wrist is None:
            raise ValueError("OpenVLA-OFT needs the wrist image")
        if frame.state.shape != (PROPRIO_DIM,):
            raise ValueError(
                f"OpenVLA-OFT needs an {PROPRIO_DIM}-D state, got {frame.state.shape}"
            )
        raw = self._policy(frame.main, frame.wrist, frame.state, frame.instruction)
        return libero_gripper(np.asarray(raw, np.float32))


def load_policy(
    path: str, unnorm_key: str | None, center_crop: bool, repo: str | None
) -> tuple[Policy, str]:
    """The repository's model, processor, action head and proprio projector as a :data:`Policy`.

    ``experiments.robot`` is not part of the installed ``openvla-oft`` package: it is imported from
    ``repo`` (default: the clone the editable install points at). Its constants pick the LIBERO
    action chunk / proprio sizes from ``ROBOT_PLATFORM``.
    """
    import sys

    import prismatic
    import torch

    os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
    sys.path.insert(0, repo or os.path.dirname(os.path.dirname(prismatic.__file__)))
    from experiments.robot.openvla_utils import (
        get_action_head,
        get_processor,
        get_proprio_projector,
        get_vla,
        get_vla_action,
    )
    from transformers import AutoConfig

    keys = list(AutoConfig.from_pretrained(path, trust_remote_code=True).norm_stats)
    if unnorm_key is None:
        if len(keys) != 1:
            raise ValueError(f"--unnorm-key is required; the checkpoint has {keys}")
        unnorm_key = keys[0]
    elif unnorm_key not in keys:
        raise ValueError(
            f"unknown --unnorm-key {unnorm_key!r}; the checkpoint has {keys}"
        )
    # GenerateConfig of run_libero_eval.py, LIBERO defaults.
    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=path,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=center_crop,
        num_open_loop_steps=HORIZON,
        unnorm_key=unnorm_key,
        load_in_8bit=False,
        load_in_4bit=False,
        lora_rank=32,
    )
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    proprio_projector = get_proprio_projector(
        cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM
    )

    def policy(
        main: np.ndarray, wrist: np.ndarray, state: np.ndarray, instruction: str
    ) -> np.ndarray:
        obs = {"full_image": main, "wrist_image": wrist, "state": state}
        with torch.inference_mode():
            actions = get_vla_action(
                cfg,
                vla,
                processor,
                obs,
                instruction,
                action_head=action_head,
                proprio_projector=proprio_projector,
                use_film=False,
            )
        return np.stack([np.asarray(a, np.float32) for a in actions])

    return policy, unnorm_key


def main() -> None:
    p = argparse.ArgumentParser()
    add_server_args(p)
    p.add_argument(
        "--model-path",
        default=os.environ.get("OPENVLA_OFT_CHECKPOINT_PATH"),
        help="checkpoint directory or Hugging Face repo id (default OPENVLA_OFT_CHECKPOINT_PATH env)",
    )
    p.add_argument(
        "--suite",
        choices=sorted(OPENVLA_OFT_CHECKPOINTS),
        default="libero_spatial",
        help="which published LIBERO fine-tune to fetch when --model-path is unset",
    )
    p.add_argument(
        "--revision", default=None, help="commit hash (default: the pin for --suite)"
    )
    p.add_argument(
        "--unnorm-key",
        default=None,
        help="norm_stats key for action denormalisation (default: the checkpoint's only key)",
    )
    p.add_argument(
        "--no-center-crop",
        action="store_true",
        help="skip the 0.9 center crop (the checkpoints were trained with it)",
    )
    p.add_argument(
        "--repo",
        default=os.environ.get("OPENVLA_OFT_REPO"),
        help="openvla-oft clone providing experiments.robot (default: the editable install)",
    )
    args = p.parse_args()
    apply_cuda_device(args.cuda_device)

    repo, pin = OPENVLA_OFT_CHECKPOINTS[args.suite]
    model = args.model_path or repo
    revision = args.revision or (pin if model == repo else None)
    # A custom checkpoint's suite is unknown: only the published fine-tunes report theirs.
    suite = args.suite if model == repo else None
    t0 = time.time()
    path = snapshot(model, revision)
    logger.info("loading OpenVLA-OFT from %s (revision=%s) ...", path, revision)
    policy, unnorm_key = load_policy(
        path, args.unnorm_key, not args.no_center_crop, args.repo
    )
    logger.info("model ready in %.1fs (unnorm_key=%s)", time.time() - t0, unnorm_key)
    OpenVLAOFTFacade(policy=policy, model=model, revision=revision, suite=suite).serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
