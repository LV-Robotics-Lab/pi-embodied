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

"""Metaworld's Flywheel data rules (the TS recorder's FLYWHEEL in packages/embodied/src/robots/metaworld).

One transition per Metaworld control step of a motion (``env.move_delta``'s frames): the two
views the robot shows, the TCP and the finger opening, and the env's own ``[dx, dy, dz,
gripper]`` action in [-1, 1] as the servo applied it (1.0 = 1 cm of hand travel; gripper +1
closes, -1 opens)."""

from __future__ import annotations

from typing import Any

_IMAGE = {"shape": (256, 256, 3), "dtype": "uint8"}


def success_mask(transitions: Any) -> Any:
    """Metaworld's ``info["success"]`` after each control step."""
    return transitions["terminated"]


SPEC = {
    "robot": "metaworld",
    "robot_type": "sawyer",
    # Metaworld's control rate: frame_skip 5 of MuJoCo's 2.5 ms.
    "fps": 80,
    "arrays": {
        "agentview_images": _IMAGE,
        "wrist_images": _IMAGE,
        "states": {"shape": (4,), "dtype": "float32"},
        "actions": {"shape": (4,), "dtype": "float32"},
    },
    "image_fields": ("agentview_images", "wrist_images"),
    "cameras": {"agentview_images": "corner4", "wrist_images": "gripperPOV"},
    "state_names": ["tcp_x", "tcp_y", "tcp_z", "gripper_width"],
    "action_names": ["hand_dx", "hand_dy", "hand_dz", "gripper"],
    "success_mask": success_mask,
    #: What one dataset shares: every episode of one MT50 task.
    "group": ("task",),
}
