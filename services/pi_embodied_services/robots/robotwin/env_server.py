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

"""RPC server owning one RLinf RoboTwin environment.

Code mode (``code.run``, utils/code_exec.py ``CodeRunMixin``): a program calls the registry's
primitives in a sandboxed subprocess, so the server requires its RPC token and refuses other
business calls while a program runs. A raw step's reply to the program carries the robot state
and the episode status, not the camera observation (its head frames go to the run's video); the
run reports its native actions, the latest robot state and status (pi's ``Info``) and its video
(``_finish_run``).
"""

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
from pi_embodied_services.utils.code_exec import CodeRunMixin
from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.serialization import to_numpy_tree

logger = get_logger("robotwin_env_server")

#: Code mode: video frames one run hands back (halved, every other one kept, when full), and the
#: most native actions one ``chunk_step`` runs.
CODE_MAX_FRAMES = 128
CODE_MAX_CHUNK = 64
#: Code mode's translation estimate of a qpos action: metres the gripper may move per radian of
#: any one arm joint. An upper bound: the aloha-agilex arms (Piper) reach about 0.63 m, and no
#: joint is farther than that from the gripper.
JOINT_REACH_M = 0.7


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
from pi_embodied_services.utils.perception import (
    add_perception_arguments,
    install_perception,
    render_view,
)


class RoboTwinEnvFacade(CodeRunMixin, BaseEnvFacade):
    """Expose common and typed RoboTwin environment RPC contracts."""

    SERVICE_NAME = "robotwin-env"

    def __init__(self, env: Any, *, metadata: dict[str, Any]):
        self._env = env
        self._metadata = dict(metadata)
        # Code mode: the run's native actions, its latest robot state and status, the joint
        # target and eef16 state its next action starts from, and its video frames.
        self._run_steps = 0
        self._run_info: dict[str, Any] | None = None
        self._run_ref: tuple[np.ndarray, np.ndarray] | None = None
        self._run_frames: list[np.ndarray] = []
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
        self._rpc["env.policy_frame"] = self.policy_frame
        api = register_code_api(self, ROBOTWIN_PRIMITIVES)
        self._install_code_run(
            api,
            move_m=self._code_move_m,
            check=self._code_check,
            reply=self._code_reply,
            begin=self._begin_run,
            finish=self._finish_run,
        )

    # ---- code mode (run_code) ----

    def _begin_run(self) -> None:
        self._run_steps = 0
        self._run_info = None
        self._run_ref = None
        self._run_frames = []

    def _finish_run(self) -> dict[str, Any]:
        """The run's effect for pi: native actions executed, the latest robot state and episode
        status (pi's ``Info``, None when the program stepped nothing), success and whether the
        action budget is spent, and the run's video frames."""
        status = (self._run_info or {}).get("episode_status")
        limit = status and status.get("step_lim")
        return {
            "steps": self._run_steps,
            "info": self._run_info,
            "success": bool(status["eval_success"]) if status else None,
            "budget_exhausted": (
                bool(limit is not None and status["take_action_cnt"] >= limit)
                if status
                else None
            ),
            "frames": list(self._run_frames),
        }

    def _keep_frames(self, frames) -> None:
        for f in frames:
            if f is None:
                continue
            if len(self._run_frames) >= CODE_MAX_FRAMES:
                self._run_frames = self._run_frames[::2]
            self._run_frames.append(f)

    def _absorb_step_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """Keep a step's robot state and status for the run's finish and the next action's
        estimate; returns what the program receives of the info."""
        self._run_steps += int(info.get("executed_actions", 0))
        state = info.get("robot_state")
        if state is not None:
            self._run_info = {
                "robot_state": state,
                "episode_status": info.get("episode_status"),
            }
            self._run_ref = (
                np.asarray(state["qpos_target14"], dtype=np.float64).reshape(14),
                np.concatenate(
                    [
                        np.asarray(state["left_eef_pose"], dtype=np.float64),
                        [float(state["left_gripper"])],
                        np.asarray(state["right_eef_pose"], dtype=np.float64),
                        [float(state["right_gripper"])],
                    ]
                ),
            )
        return {
            k: info[k]
            for k in (
                "action_type",
                "requested_actions",
                "executed_actions",
                "robot_state",
                "episode_status",
                "cancelled",
            )
            if k in info
        }

    def _code_reply(self, method: str, out: Any) -> Any:
        """What a program receives: the policy frame's joint and eef16 state without its images
        (``get_state``; the program renders what it needs); a raw step's reward, flags, robot state
        and status, without the camera observation (its head frames go to the run's video)."""
        out = to_numpy_tree(out)
        if method == "env.policy_frame":
            return {k: out[k] for k in ("qpos", "qpos_target", "state")}
        if method not in ("env.step", "env.chunk_step"):
            return out
        obs, reward, terminated, truncated, info = out
        if isinstance(obs, dict) and "frames" in obs:
            self._keep_frames(obs["frames"])
        elif isinstance(obs, dict):
            self._keep_frames([obs.get("main_images")])
        return {
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": self._absorb_step_info(info),
        }

    def _code_ref(self) -> tuple[np.ndarray, np.ndarray]:
        """The joint target (qpos14) and eef16 state the program's next action starts from."""
        if self._run_ref is None:
            frame = to_numpy_tree(self._env.policy_frame(0))
            self._run_ref = (
                np.asarray(frame["qpos_target"], dtype=np.float64).reshape(14),
                np.asarray(frame["state"], dtype=np.float64).reshape(16),
            )
        return self._run_ref

    def _code_move_m(self, method: str, kwargs: dict) -> float:
        """How far a program's raw actions may move the grippers (the run's translation cap),
        from the state they start at: a qpos action's arm-joint changes times ``JOINT_REACH_M``
        (an upper bound), an ee action's change of the two eef positions."""
        if method not in ("env.step", "env.chunk_step"):
            return 0.0
        ee = kwargs.get("action_type") == "ee"
        key = "action" if method == "env.step" else "actions"
        rows = np.asarray(kwargs[key], dtype=np.float64).reshape(-1, 16 if ee else 14)
        qpos, eef = self._code_ref()
        prev, total = (eef if ee else qpos), 0.0
        for row in rows:
            d = row - prev
            if ee:
                total += float(np.linalg.norm(d[0:3]) + np.linalg.norm(d[8:11]))
            else:
                total += JOINT_REACH_M * float(np.abs(np.r_[d[0:6], d[7:13]]).sum())
            prev = row
        return total

    def _code_check(self, method: str, kwargs: dict) -> None:
        """Refuse a program's chunk longer than ``CODE_MAX_CHUNK``: a chunk polls the stop only
        between native actions (an ``ee`` one plans and runs a whole cuRobo path), and its
        frames and estimate must stay bounded."""
        if method == "env.chunk_step":
            n = len(np.asarray(kwargs["actions"], dtype=np.float64).reshape(-1))
            dim = 16 if kwargs.get("action_type") == "ee" else 14
            if n > CODE_MAX_CHUNK * dim:
                raise ValueError(
                    f"chunk_step runs at most {CODE_MAX_CHUNK} actions per call, got {n // dim}"
                )

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
        return_policy_frames: bool = False,
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
                return_policy_frames=return_policy_frames,
                should_stop=self.stop_requested,
            )
        )
        if len(observation_list) != 1 or len(info_list) != 1:
            raise RuntimeError(
                "RoboTwin chunk_step must return one environment, got "
                f"{len(observation_list)} obs / {len(info_list)} info"
            )
        observation = observation_list[0]
        if return_all_frames or return_policy_frames:
            observation = {
                **{
                    k: observation[k]
                    for k in ("frames", "policy_frames")
                    if k in observation
                },
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

    def policy_frame(self) -> dict[str, Any]:
        """What the VLA reads now: head and wrist RGB and the eef16 state, plus the joint state
        ``qpos`` (measured) and ``qpos_target`` (commanded), each 14 (the Flywheel's observation)."""
        return self._env.policy_frame(0)

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
    add_perception_arguments(parser, sam3=True)
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
    # --sam3 / --unidepth: env.detect, env.select_detection, env.reject_detection, env.enhance_depth.
    install_perception(
        facade,
        args,
        cameras=["head", "left_wrist", "right_wrist"],
        view=render_view(facade),
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
