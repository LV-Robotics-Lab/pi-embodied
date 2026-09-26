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
# Modified by pi-embodied: import paths rewritten (runtime contracts now come from
# contract.py); healthz service name; HTTP is the only --transport; chunk_step
# polls the stop flag between native actions; deterministic torch and cuRobo
# L-BFGS, global RNGs reseeded on reset.

"""RPC server owning one RLinf RoboTwin environment."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Support direct execution from a services/ checkout before package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pi_embodied_services.components.code_api import register_code_api
from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.robots.robotwin.primitives import ROBOTWIN_PRIMITIVES
from pi_embodied_services.utils import ground_truth
from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.serialization import to_numpy_tree

logger = get_logger("robotwin_env_server")


@lru_cache(maxsize=None)
def _evaluation_language(task_config: str, task_name: str, seed: int) -> str | None:
    """Return the published language for an exact standard evaluation episode."""
    table_path = Path(__file__).resolve().parent / "eval" / f"{task_config}.json"
    if not table_path.is_file():
        return None
    table = json.loads(table_path.read_text())
    matches = [
        entry["task_language"]
        for entry in table.get("tasks", {}).get(task_name, [])
        if int(entry["seed"]) == int(seed)
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"duplicate RoboTwin language entries for {task_name} seed {seed}"
        )
    if not matches:
        return None
    language = matches[0]
    if not isinstance(language, str) or not language.strip():
        raise RuntimeError(
            f"invalid RoboTwin language entry for {task_name} seed {seed}"
        )
    return language.strip()


def _deterministic_cuda() -> None:
    """Make the worker's torch CUDA math, and so cuRobo's plans, repeat bitwise.

    Every ``ee`` action and ``env.plan_arm_path`` runs cuRobo, and cuRobo's
    trajectory optimization under torch's default nondeterministic CUDA kernels
    gives a different plan for the same start and target, both across processes
    and on repeated calls in one process (measured: 1e-4 rad and a few steps of
    trajectory length per plan, so the same ee actions end up in different
    states). torch's deterministic algorithms remove it at no measured cost
    (about 52 ms per plan either way). ``warn_only``: an op without a
    deterministic implementation warns instead of failing the episode. Must run
    before cuRobo builds and warms up its planners, i.e. before the first CUDA
    call; cuBLAS reads its workspace setting when its handle is created.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def _torch_lbfgs_step() -> None:
    """Make cuRobo compute its L-BFGS step with torch ops, not its fused kernel.

    The kernel (``lbfgs_step_kernel.cu``) runs one thread per variable and its
    block reduction sums shuffle lanes and shared-memory slots that no thread
    wrote whenever the variable count is not a multiple of 32 (a 28-step 6-DoF
    trajectory has 168), so each step adds whatever an earlier kernel left
    there. The same plan then differs by 1e-7 to 1e-3 rad depending on what ran
    on the GPU before, even with deterministic torch (measured: one of four
    stack_blocks_two attempts, and a whole pi record vs its replays). The torch
    path is deterministic and costs about 100 ms more per plan (52 to 153 ms).
    Must run before the planners are built: cuRobo captures the step into CUDA
    graphs during warmup.
    """
    if getattr(LBFGSOpt, "_pi_torch_step", False):
        return
    init = LBFGSOpt.__init__

    def torch_step_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.use_cuda_kernel = False

    LBFGSOpt.__init__ = torch_step_init
    LBFGSOpt._pi_torch_step = True


def _seed_globals(seed: int) -> None:
    """Seed the worker's global Python, numpy and torch RNGs for one reset.

    RoboTwin's scene setup seeds numpy and torch itself but not Python's
    ``random``; seeding all three here makes each reset start from the same RNG
    state whatever ran before it in this process.
    """
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def _teardown_env(env: Any) -> None:
    """Release an environment across RLinf teardown API versions."""
    offload = getattr(env, "offload", None)
    if callable(offload):
        offload(clear_cache=True)
        return

    close = getattr(env, "close", None)
    if callable(close):
        close(clear_cache=True)
        return

    raise RuntimeError(
        f"{type(env).__name__} provides neither callable offload() nor close() "
        "for teardown"
    )


from curobo.opt.newton.lbfgs import LBFGSOpt  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from robotwin.assets import validate_root  # noqa: E402
from robotwin.config import load_task_config  # noqa: E402

from pi_embodied_services.robots.robotwin.contract import (
    RoboTwinActionType,
)
from pi_embodied_services.robots.robotwin.reward_compat import (
    install_native_reward_compat,
)
from pi_embodied_services.robots.robotwin.rlinf_env import (
    RoboTwinAgentEnv,
)


class RoboTwinEnvFacade(BaseEnvFacade):
    """Expose common and typed RoboTwin environment RPC contracts."""

    SERVICE_NAME = "robotwin-env"

    def __init__(self, env: Any, *, metadata: dict[str, Any]):
        self._env = env
        self._metadata = dict(metadata)
        super().__init__()

    def _strip_single_env_value(self, value: Any, name: str) -> Any:
        """Remove one leading single-environment batch dimension."""
        if value is None:
            return None
        if hasattr(value, "shape"):
            shape = tuple(value.shape)
            if not shape or shape[0] != 1:
                raise RuntimeError(
                    f"{name} must have leading env dimension 1, got {shape}"
                )
            return value[0]
        if isinstance(value, list):
            if len(value) != 1:
                raise RuntimeError(
                    f"{name} must contain one environment, got {len(value)}"
                )
            return value[0]
        raise RuntimeError(f"{name} is missing a single-environment batch dimension")

    def _strip_single_env_observation(self, observation: Any) -> dict[str, Any]:
        if not isinstance(observation, dict):
            raise TypeError(
                f"RoboTwin observation must be a mapping, got {observation!r}"
            )
        return {
            key: self._strip_single_env_value(value, f"observation.{key}")
            for key, value in observation.items()
        }

    def _strip_single_signal(self, value: Any, name: str) -> Any:
        if (
            hasattr(value, "detach")
            and hasattr(value, "cpu")
            and hasattr(value, "numpy")
        ):
            value = value.detach().cpu().numpy()
        array = np.asarray(value).reshape(-1)
        if array.size != 1:
            raise RuntimeError(
                f"{name} must contain one environment, got {array.shape}"
            )
        return array[0].item()

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["env.plan_arm_path"] = self.plan_arm_path
        self._rpc["env.ground_truth_poses"] = self.ground_truth_poses
        register_code_api(self, ROBOTWIN_PRIMITIVES)

    def get_env_meta(self) -> dict[str, Any]:
        """Return immutable identity for endpoint compatibility checks."""
        return dict(self._metadata)

    def close(self):
        _teardown_env(self._env)

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        seed = int(self._metadata["seed"])
        _seed_globals(seed)
        observation, info = self._env.reset(env_idx=[0], env_seeds=[seed])
        episode_status = info["episode_status"]
        if episode_status["actual_seed"] != seed:
            raise RuntimeError(
                "RoboTwin exact seed mismatch: "
                f"requested {seed}, initialized {episode_status['actual_seed']}"
            )
        instruction_source = "native"
        task_name = self._metadata.get("task_name")
        task_config = self._metadata.get("task_config")
        if isinstance(task_name, str) and isinstance(task_config, str):
            published = _evaluation_language(task_config, task_name, seed)
            if published is not None:
                self._env.set_task_language(published, env_id=0)
                instruction_source = "evaluation_seed_table"
        instruction = self._env.get_task_language(0)
        info["requested_seed"] = seed
        info["instruction"] = instruction
        info["instruction_source"] = instruction_source
        return self._strip_single_env_observation(observation), info

    def step(
        self, action, *, action_type: RoboTwinActionType = "qpos"
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        expected_dim = 14 if action_type == "qpos" else 16
        array = np.asarray(action, dtype=np.float64)
        if array.shape != (expected_dim,):
            raise ValueError(
                f"RoboTwin common action must have shape ({expected_dim},)"
            )
        if not np.isfinite(array).all():
            raise ValueError("RoboTwin common action must contain only finite values")
        observation, reward, terminated, truncated, info = self._env.step(
            array, action_type=action_type
        )
        return (
            self._strip_single_env_observation(observation),
            self._strip_single_signal(reward, "reward"),
            bool(self._strip_single_signal(terminated, "terminated")),
            bool(self._strip_single_signal(truncated, "truncated")),
            info,
        )

    def chunk_step(
        self,
        actions,
        *,
        action_type: RoboTwinActionType = "qpos",
        return_all_frames: bool = False,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        expected_dim = 14 if action_type == "qpos" else 16
        array = np.asarray(actions, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != expected_dim:
            raise ValueError(
                f"RoboTwin common actions must have shape [N,{expected_dim}], N >= 1"
            )
        if not np.isfinite(array).all():
            raise ValueError("RoboTwin common actions must contain only finite values")
        observation_list, rewards, terminated, truncated, info_list = (
            self._env.chunk_step(
                array,
                action_type=action_type,
                return_all_frames=return_all_frames,
                should_stop=self.stop_requested,
            )
        )
        if len(observation_list) != 1 or len(info_list) != 1:
            raise RuntimeError(
                "RoboTwin chunk_step must return one environment, got "
                f"{len(observation_list)} obs / {len(info_list)} info"
            )
        observation = observation_list[0]
        if return_all_frames:
            observation = {
                "frames": observation["frames"],
                "final": self._strip_single_env_observation(observation["final"]),
            }
        else:
            observation = self._strip_single_env_observation(observation)
        return (
            observation,
            np.asarray(rewards)[0],
            np.asarray(terminated, dtype=bool)[0],
            np.asarray(truncated, dtype=bool)[0],
            info_list[0],
        )

    def render_camera(self, camera_name: str, depth: bool = False) -> Any:
        return self._env.render_camera(camera_name, depth=depth, env_id=0)

    def get_camera_meta(self, camera_name: str) -> dict[str, Any]:
        return self._env.get_camera_meta(camera_name, env_id=0)

    def get_task_language(self) -> str:
        return self._env.get_task_language(env_id=0)

    def plan_arm_path(self, arm: str, target_pose) -> dict[str, Any]:
        return self._env.plan_arm_path(0, arm, target_pose)

    def ground_truth_poses(self, names=None) -> dict:
        """World poses of ``names`` (default all) of the scene's actors (``--privileged``)."""
        return ground_truth.respond(self._env.object_poses(0), names)

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        return to_numpy_tree(super()._dispatch(method, args, kwargs))


def build_env_cfg(
    *,
    task_name: str,
    task_config: str,
    seed: int,
    assets_path: str,
    max_episode_steps: int = 10000,
) -> Any:
    """Build a single-env RLinf config from packaged RoboTwin resources."""
    native_task_config = OmegaConf.create(load_task_config(task_config))
    step_limit = int(max_episode_steps)
    native_task_config.task_name = task_name
    native_task_config.task_config = task_config
    native_task_config.step_lim = step_limit
    native_task_config.ckpt_setting = "hybrid_lingbot"
    native_task_config.policy_name = "hybrid_lingbot"
    native_task_config.planner_backend = "curobo"
    native_task_config.eval_video_log = False
    native_task_config.render_freq = 0

    return OmegaConf.create(
        {
            "env_type": "robotwin",
            "initial_env_seeds": [int(seed)],
            "auto_reset": False,
            "ignore_terminations": False,
            "reward_coef": 1.0,
            "use_custom_reward": True,
            "use_rel_reward": True,
            "center_crop": False,
            "seed": seed,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "max_steps_per_rollout_epoch": step_limit,
            "max_episode_steps": step_limit,
            "is_eval": True,
            "assets_path": assets_path,
            "seeds_path": None,
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": None,
            },
            "enable_offload": False,
            "task_config": native_task_config,
        }
    )


def make_env(
    task_name: str,
    task_config: str,
    seed: int,
    assets_path: str,
    max_episode_steps: int = 10000,
) -> RoboTwinAgentEnv:
    """Construct the only simulator owner used by a pi-embodied run."""
    # Temporary workaround for the pinned RoboTwin place_fan reward-construction
    # bug. Remove after the RoboTwin dependency includes the upstream fix.
    install_native_reward_compat(task_name)
    _deterministic_cuda()
    _torch_lbfgs_step()
    assets_identity = validate_root(assets_path)
    resolved_assets_path = Path(assets_identity["root"])
    os.environ["ROBOTWIN_ASSETS_PATH"] = str(resolved_assets_path)
    os.environ["ROBOTWIN_ASSETS_ROOT"] = str(resolved_assets_path)
    logger.info("RoboTwin assets: %s", assets_identity)
    print(
        f"robotwin_assets {json.dumps(assets_identity, sort_keys=True)}",
        flush=True,
    )
    cfg = build_env_cfg(
        task_name=task_name,
        task_config=task_config,
        seed=seed,
        assets_path=str(resolved_assets_path),
        max_episode_steps=max_episode_steps,
    )
    return RoboTwinAgentEnv(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=False,
    )


def main() -> None:
    from pi_embodied_services.robots.robotwin.contract import ROBOTWIN_TASK_CONFIGS

    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--task-name", required=True)
    parser.add_argument(
        "--task-config",
        choices=ROBOTWIN_TASK_CONFIGS,
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=10000,
        help="Episode action budget for the RoboTwin agent runtime.",
    )
    parser.add_argument("--assets-path", required=True)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()

    env = make_env(
        args.task_name,
        args.task_config,
        args.seed,
        args.assets_path,
        args.max_episode_steps,
    )
    from pi_embodied_services.robots.robotwin.contract import env_runtime_contract

    facade = RoboTwinEnvFacade(
        env,
        metadata=env_runtime_contract(
            task_name=args.task_name,
            task_config=args.task_config,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
        ),
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
