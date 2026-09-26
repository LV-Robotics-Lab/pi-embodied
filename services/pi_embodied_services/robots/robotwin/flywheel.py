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

"""RoboTwin's Flywheel data rules: what the LingBot VLA reads and emits, the eef16 layout (the TS
recorder's FLYWHEEL in packages/embodied/src/robotwin). A scripted qpos step is recorded as the
eef16 pose it reached, so every action of an episode is in the one policy action space."""

from __future__ import annotations

from typing import Any

#: The camera sizes are the task config's: taken from the episodes.
_IMAGE = {"shape": None, "dtype": "uint8"}
_ARM = ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")


def success_mask(transitions: Any) -> Any:
    """RoboTwin's native `eval_success` after each native action."""
    return transitions["terminated"]


SPEC = {
    "robot": "robotwin",
    "robot_type": "aloha-agilex",
    # Timestamps only: native actions have no wall-clock rate.
    "fps": 25,
    "arrays": {
        "head_images": _IMAGE,
        "left_wrist_images": _IMAGE,
        "right_wrist_images": _IMAGE,
        "states": {"shape": (16,), "dtype": "float32"},
        "actions": {"shape": (16,), "dtype": "float32"},
    },
    "image_fields": ("head_images", "left_wrist_images", "right_wrist_images"),
    "cameras": {
        "head_images": "cam_high",
        "left_wrist_images": "cam_left_wrist",
        "right_wrist_images": "cam_right_wrist",
    },
    "state_names": [f"{arm}_{d}" for arm in ("left", "right") for d in _ARM],
    "action_names": [f"{arm}_{d}" for arm in ("left", "right") for d in _ARM],
    "success_mask": success_mask,
    #: What one dataset shares: every episode of one task in one task config.
    "group": ("task_config", "task_name"),
}
