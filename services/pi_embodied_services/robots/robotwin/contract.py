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
# Modified by pi-embodied: the runtime contracts and constants that the RoboTwin
# env/VLA servers need, extracted verbatim from robots/robotwin/robot_spec.py;
# the planner-side robot spec, toolkit wiring, dashboard spec and server
# spawning are omitted.

"""RoboTwin runtime contracts shared by the env server and the VLA server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Native RoboTwin task YAMLs exposed to the dashboard, the CLI, and the env
# server -- defined once so every consumer shows the same choices.
ROBOTWIN_TASK_CONFIGS = (
    "demo_clean",
    "demo_randomized",
)

#: Env-side camera names exposed by the RoboTwin EnvServer, in fixed order.
#: Shared across the env client, primitives, and toolkit.
ROBOTWIN_CAMERA_NAMES = (
    "head",
    "left_wrist",
    "right_wrist",
)

#: Supported native action representations for the RoboTwin agent runtime.
RoboTwinActionType = Literal["qpos", "ee"]

#: Episode-status keys the RoboTwin env client requires in every status mapping.
ROBOTWIN_STATUS_KEYS = (
    "eval_success",
    "take_action_cnt",
    "step_lim",
    "actual_seed",
)

#: Per-call RPC read timeout (seconds) for idempotent env queries.
ROBOTWIN_READ_TIMEOUT_S = 120.0

#: Per-call RPC timeout (seconds) for env calls that mutate episode state.
ROBOTWIN_STATE_CHANGE_TIMEOUT_S = 600.0


@dataclass(frozen=True)
class RoboTwinModelSpec:
    """Values required by the LingBot EEF runtime."""

    policy_name: str
    robot_config_relpath: str
    norm_stats: str
    qwen_base: str
    camera_order: tuple[str, ...]
    state_layout: str
    action_layout: str
    use_length: int


MODEL_SPEC = RoboTwinModelSpec(
    policy_name="robotwin_eef",
    robot_config_relpath="configs/robot_configs/robotwin_eef.yaml",
    norm_stats="norm_stats/robotwin_eef.json",
    qwen_base="qwen_base",
    camera_order=("cam_high", "cam_left_wrist", "cam_right_wrist"),
    state_layout="eef16",
    action_layout="eef16",
    use_length=50,
)


def env_runtime_contract(
    *,
    task_name: str,
    task_config: str,
    seed: int,
    max_episode_steps: int = 10000,
) -> dict[str, object]:
    """Return the identity required from a RoboTwin EnvServer."""
    return {
        "runtime": "rlinf_robotwin_env",
        "task_name": task_name,
        "task_config": task_config,
        "seed": int(seed),
        "seed_mode": "exact",
        "action_layouts": ["qpos14", MODEL_SPEC.action_layout],
        "execution": {
            "reset": True,
            "step": True,
            "chunk_step": True,
            "action_layouts": ["qpos14", MODEL_SPEC.action_layout],
            "chunk_step_all_frames": True,
            "step_limit": int(max_episode_steps),
        },
        "extensions": {
            "render_camera": {
                "camera_names": list(ROBOTWIN_CAMERA_NAMES),
                "metric_depth": True,
            },
            "get_camera_meta": True,
            "get_task_language": True,
            "plan_arm_path": True,
        },
    }


def vla_runtime_contract() -> dict[str, object]:
    """Return the identity required from a LingBot RoboTwin server.

    Only the hard contract fields that break inference input/output when
    mismatched are kept; descriptive behaviour fields live with the
    implementation (facade/transport), not in the runtime-equality check.
    """
    return {
        "runtime": "lingbotvla",
        "policy_name": MODEL_SPEC.policy_name,
        "camera_order": list(MODEL_SPEC.camera_order),
        "state_layout": MODEL_SPEC.state_layout,
        "action_layout": MODEL_SPEC.action_layout,
        "use_length": MODEL_SPEC.use_length,
    }
