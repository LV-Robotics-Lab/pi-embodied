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

"""RoboDojo's Flywheel data rules (the TS recorder's FLYWHEEL in packages/embodied/src/robodojo):
every 25 Hz control step in RoboDojo's joint space, which is XPolicyLab's arx_x5 layout: state and
action ``[left joints6, left gripper, right joints6, right gripper]`` with the gripper normalized
(1 open .. 0 closed), and the head and two wrist cameras."""

from __future__ import annotations

from typing import Any

_IMAGE = {"shape": None, "dtype": "uint8"}
_MOTORS = [f"{arm}_joint_{i}" for arm in ("left", "right") for i in range(7)]


def success_mask(transitions: Any) -> Any:
    """RoboDojo's own success judgement (is_episode_end) after each control step."""
    return transitions["terminated"]


SPEC = {
    "robot": "robodojo",
    # XPolicyLab's converter names every dataset's robot so, since it merges robots.
    "robot_type": "unified_robot",
    "fps": 25,
    "arrays": {
        "head_images": _IMAGE,
        "left_wrist_images": _IMAGE,
        "right_wrist_images": _IMAGE,
        "states": {"shape": (14,), "dtype": "float32"},
        "actions": {"shape": (14,), "dtype": "float32"},
    },
    "image_fields": ("head_images", "left_wrist_images", "right_wrist_images"),
    "cameras": {
        "head_images": "cam_head",
        "left_wrist_images": "cam_left_wrist",
        "right_wrist_images": "cam_right_wrist",
    },
    "state_names": _MOTORS,
    "action_names": _MOTORS,
    "success_mask": success_mask,
    #: What one dataset shares: every episode of one task.
    "group": ("task",),
}
