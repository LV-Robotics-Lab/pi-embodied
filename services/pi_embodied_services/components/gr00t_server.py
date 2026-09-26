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

"""RPC server wrapping GR00T N1.6 / N1.7 (NVIDIA/Isaac-GR00T) for LIBERO.

Uses the ``libero_panda`` embodiment of Isaac-GR00T's ``embodiment_configs.py``:
videos ``image`` (agentview) and ``wrist_image``, the 8-D LIBERO state split into
``x, y, z, roll, pitch, yaw`` (1-D each) and ``gripper`` (2-D), 7-D actions with
the same keys over a 16-step horizon, and the instruction under
``annotation.human.action.task_description`` (RLinf's ``simulation_io.py`` does
the same split). A LIBERO fine-tune carries the embodiment's statistics; the
base models know the embodiment id but have no statistics, so a fine-tune is
required. Actions come back denormalised in LIBERO's OSC space except the
gripper, which the fine-tune emits in [0, 1] (dataset open = 1) and which is
binarised and inverted like OpenVLA's (RLinf's ``action_utils.py``). A flow-matching sampler: the per-call
``seed`` makes a replay bit-identical.

Checkpoints are pinned by commit hash in ``GR00T_CHECKPOINTS`` and fetched
through ``HF_ENDPOINT`` (hf-mirror) when ``--model-path`` is not a directory.
Run it in the ``gr00t`` venv (``services[gr00t]``, see README).
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
    libero_gripper,
    snapshot,
)
from pi_embodied_services.utils.logging import get_logger

logger = get_logger("gr00t_server")

#: name -> (repo, commit hash). ``n1.6-libero-spatial`` is RLinf's N1.6 LIBERO-Spatial SFT (it has the
#: libero_panda statistics); the bases are listed for fine-tunes that ship without a config.
GR00T_CHECKPOINTS: dict[str, tuple[str, str]] = {
    "n1.6-libero-spatial": (
        "RLinf/RLinf-Gr00t-N1.6-SFT-Spatial",
        "e39614af19b50feebee0310a31665995ad68f4fd",
    ),
    "n1.6-base": ("nvidia/GR00T-N1.6-3B", "d0814e7ecb19202e7c8468b46098b0b7ef3a6d61"),
    "n1.7-base": ("nvidia/GR00T-N1.7-3B", "2fc962b973bccdd5d8ce4f67cc63b264d6886495"),
}
EMBODIMENT = "libero_panda"
VIDEO_KEYS = ("image", "wrist_image")
STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
STATE_DIMS = (1, 1, 1, 1, 1, 1, 2)
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
LANGUAGE_KEY = "annotation.human.action.task_description"

#: ``policy(gr00t_obs) -> {action key: float[B=1, horizon, d]}`` in LIBERO's action space.
Policy = Callable[[dict], dict]


def libero_panda_obs(frame: Frame) -> dict:
    """The nested ``Gr00tPolicy`` observation: videos [B=1, T=1, H, W, 3] uint8, states
    [1, 1, d] float32 per key, and the instruction as [[str]]."""
    if frame.wrist is None:
        raise ValueError("GR00T libero_panda needs the wrist image")
    if frame.state.shape != (sum(STATE_DIMS),):
        raise ValueError(
            f"GR00T libero_panda needs a {sum(STATE_DIMS)}-D state, got {frame.state.shape}"
        )
    state: dict = {}
    i = 0
    for key, dim in zip(STATE_KEYS, STATE_DIMS):
        state[key] = frame.state[i : i + dim][None, None].astype(np.float32)
        i += dim
    return {
        "video": {
            VIDEO_KEYS[0]: frame.main[None, None],
            VIDEO_KEYS[1]: frame.wrist[None, None],
        },
        "state": state,
        "language": {LANGUAGE_KEY: [[frame.instruction]]},
    }


def actions_of(out: dict, horizon: int) -> np.ndarray:
    """The policy's per-key ``[1, horizon, d]`` arrays as one [horizon, 7] matrix."""
    cols = [
        np.asarray(out[key], np.float32).reshape(-1, horizon, 1)[0]
        for key in ACTION_KEYS
    ]
    return np.concatenate(cols, axis=1)


class Gr00tFacade(ChunkVLAFacade):
    SERVICE_NAME = "gr00t"
    uses_wrist = True

    def __init__(
        self, *, policy: Policy, model: str, revision: str | None, horizon: int
    ):
        self._policy = policy
        self.horizon = horizon
        super().__init__(model=model, revision=revision)

    def _act(self, frame: Frame) -> np.ndarray:
        # The libero_panda fine-tune emits the gripper in [0, 1] (dataset open = 1); RLinf maps
        # N1.6/N1.7 like OpenVLA: threshold at 0.5, then invert for LIBERO (action_utils.py).
        return libero_gripper(
            actions_of(self._policy(libero_panda_obs(frame)), self.horizon)
        )


def load_policy(path: str, embodiment: str) -> tuple[Policy, int]:
    """Isaac-GR00T's ``Gr00tPolicy`` on the GPU, as a :data:`Policy`, and its action horizon."""
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(EmbodimentTag(embodiment), path, device="cuda")
    horizon = len(policy.get_modality_config()["action"].delta_indices)

    def act(obs: dict) -> dict:
        out, _info = policy.get_action(obs)
        return out

    return act, horizon


def main() -> None:
    p = argparse.ArgumentParser()
    add_server_args(p)
    p.add_argument(
        "--model-path",
        default=os.environ.get("GR00T_CHECKPOINT_PATH"),
        help="checkpoint directory or Hugging Face repo id (default GR00T_CHECKPOINT_PATH env)",
    )
    p.add_argument(
        "--checkpoint",
        choices=sorted(GR00T_CHECKPOINTS),
        default="n1.6-libero-spatial",
        help="which pinned checkpoint to fetch when --model-path is unset",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="commit hash (default: the pin for --checkpoint)",
    )
    p.add_argument("--embodiment", default=EMBODIMENT, help="GR00T embodiment tag")
    args = p.parse_args()
    apply_cuda_device(args.cuda_device)

    repo, pin = GR00T_CHECKPOINTS[args.checkpoint]
    model = args.model_path or repo
    revision = args.revision or (pin if model == repo else None)
    t0 = time.time()
    path = snapshot(model, revision)
    logger.info(
        "loading GR00T from %s (revision=%s, embodiment=%s) ...",
        path,
        revision,
        args.embodiment,
    )
    policy, horizon = load_policy(path, args.embodiment)
    logger.info("model ready in %.1fs (horizon=%d)", time.time() - t0, horizon)
    Gr00tFacade(policy=policy, model=model, revision=revision, horizon=horizon).serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
