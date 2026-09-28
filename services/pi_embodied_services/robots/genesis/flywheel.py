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

"""Genesis's Flywheel data rules (the TS recorder's FLYWHEEL in packages/embodied/src/robots/genesis).

One transition per Genesis control step of a motion (``env.move_delta`` / ``env.set_gripper``
with ``record``): the front and wrist views, the TCP pose and the finger opening, and the step
as the env server's own ``env.step`` takes it: ``[dx, dy, dz]`` the TCP point the arm was
commanded to minus the TCP before the step (m, base frame; the IK holds the reset orientation),
and the gripper (+1 open, -1 close)."""

from __future__ import annotations

from typing import Any

#: The server's --view-size (256 by default): taken from the episodes.
_IMAGE = {"shape": None, "dtype": "uint8"}


def success_mask(transitions: Any) -> Any:
    """The task's success (cube_pick: lifted 8 cm for 5 steps, latched) after each step."""
    return transitions["terminated"]


SPEC = {
    "robot": "genesis",
    "robot_type": "panda",
    # One control step is the scene's dt (0.01 s).
    "fps": 100,
    "arrays": {
        "agentview_images": _IMAGE,
        "wrist_images": _IMAGE,
        "states": {"shape": (8,), "dtype": "float32"},
        "actions": {"shape": (4,), "dtype": "float32"},
    },
    "image_fields": ("agentview_images", "wrist_images"),
    "cameras": {"agentview_images": "front", "wrist_images": "wrist"},
    "state_names": [
        *("tcp_x", "tcp_y", "tcp_z", "tcp_qw", "tcp_qx", "tcp_qy", "tcp_qz"),
        "gripper_width",
    ],
    "action_names": ["tcp_dx", "tcp_dy", "tcp_dz", "gripper"],
    "success_mask": success_mask,
    #: What one dataset shares: every episode of one task.
    "group": ("task",),
}
