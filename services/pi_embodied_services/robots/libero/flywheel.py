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
# Modified by pi-embodied: import paths rewritten.

"""LIBERO data rules and paths for the shared Flywheel implementation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pi_embodied_services.flywheel.episode import EpisodeWriter

_SUITE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

LIBERO_ARRAYS = {
    "main_images": {"shape": (256, 256, 3), "dtype": "uint8"},
    "wrist_images": {"shape": (256, 256, 3), "dtype": "uint8"},
    "states": {"shape": (8,), "dtype": "float32"},
    "actions": {"shape": (7,), "dtype": "float32"},
}


def success_mask(transitions: Any) -> Any:
    """LIBERO uses environment termination to signal task success."""
    return transitions["terminated"]


#: The Pi0.5 policy's input and output (the TS recorder's FLYWHEEL in packages/embodied/src/libero).
SPEC = {
    "robot": "libero",
    "robot_type": "panda",
    "fps": 20,
    "arrays": LIBERO_ARRAYS,
    "image_fields": ("main_images", "wrist_images"),
    "cameras": {"main_images": "agentview", "wrist_images": "wrist"},
    "state_names": [
        "eef_x",
        "eef_y",
        "eef_z",
        "eef_ax",
        "eef_ay",
        "eef_az",
        "gripper_l",
        "gripper_r",
    ],
    "action_names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
    "success_mask": success_mask,
    #: What one dataset shares: every episode of one LIBERO task.
    "group": ("suite", "task_id"),
}
LIBERO_SPEC = SPEC


def _task_root(root: Path | str, suite: str, task_id: int) -> Path:
    if not _SUITE.fullmatch(suite):
        raise ValueError(f"invalid LIBERO suite: {suite!r}")
    if type(task_id) is not int or task_id < 0:
        raise ValueError("task_id must be a non-negative integer")
    return (
        Path(root).expanduser().resolve()
        / "raw"
        / "libero"
        / suite
        / f"task_{task_id:02d}"
    )


def create_episode_writer(config: dict[str, Any], obs: dict[str, Any]) -> EpisodeWriter:
    """Create a writer using the current LIBERO task and policy observation."""
    parent = _task_root(config["root"], config["suite"], config["task_id"])
    seed = config["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    language = obs.get("task_descriptions")
    if not isinstance(language, str) or not language:
        raise ValueError("LIBERO observation has no task description")
    return EpisodeWriter(
        parent / f"seed_{seed:03d}",
        metadata={
            "suite": config["suite"],
            "task_id": config["task_id"],
            "seed": seed,
            "task_language": language,
        },
        initial_observation=obs,
        spec=LIBERO_SPEC,
    )
