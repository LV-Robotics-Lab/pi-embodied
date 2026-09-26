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

"""RoboCasa's Flywheel data rules: what the RLDX-1 VLA reads and what the env runs (the TS
recorder's FLYWHEEL in packages/embodied/src/robocasa)."""

from __future__ import annotations

from typing import Any

_IMAGE = {"shape": (256, 256, 3), "dtype": "uint8"}


def success_mask(transitions: Any) -> Any:
    """RoboCasa's success (`_check_success`) after each env step."""
    return transitions["terminated"]


SPEC = {
    "robot": "robocasa",
    "robot_type": "panda_omron",
    "fps": 20,
    "arrays": {
        "agentview_left_images": _IMAGE,
        "agentview_right_images": _IMAGE,
        "eye_in_hand_images": _IMAGE,
        # RLDX's state keys, concatenated in their order.
        "states": {"shape": (16,), "dtype": "float32"},
        # The PandaOmron composite env action.
        "actions": {"shape": (12,), "dtype": "float32"},
    },
    "image_fields": (
        "agentview_left_images",
        "agentview_right_images",
        "eye_in_hand_images",
    ),
    "cameras": {
        "agentview_left_images": "agentview_left",
        "agentview_right_images": "agentview_right",
        "eye_in_hand_images": "wrist",
    },
    "state_names": [
        *("gripper_qpos_l", "gripper_qpos_r"),
        *("base_x", "base_y", "base_z", "base_qx", "base_qy", "base_qz", "base_qw"),
        *("eef_rel_x", "eef_rel_y", "eef_rel_z"),
        *("eef_rel_qx", "eef_rel_qy", "eef_rel_qz", "eef_rel_qw"),
    ],
    "action_names": [
        *("eef_dx", "eef_dy", "eef_dz", "eef_drx", "eef_dry", "eef_drz", "gripper"),
        *("base_forward", "base_lateral", "base_yaw", "torso", "base_mode"),
    ],
    "success_mask": success_mask,
    #: What one dataset shares: every episode of one task in one split.
    "group": ("split", "task_name"),
}
