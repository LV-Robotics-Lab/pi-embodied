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

"""ManiSkill's Flywheel data rules (the TS recorder's FLYWHEEL in packages/embodied/src/maniskill).

One transition per ManiSkill control step of a motion (``env.servo``'s frames): the views the
robot shows (letterboxed to the server's view size), the TCP pose and the finger opening, and the
``pd_ee_delta_pos`` action the servo sent: ``[dx, dy, dz]`` in [-1, 1] (1.0 = 0.1 m, base frame)
and the arm's own gripper action.

Every ``--robot`` has the same action width, but not the same meaning (the Robotiq's gripper
action is the Panda's negated) nor the same cameras (the WidowX AI has no wrist camera), so
``SPACES`` has one spec per arm (``--space``, pi's /flywheel-export passes the episode's arm),
the raw path starts with the arm, and a dataset holds one arm, env id and scene (``group``)."""

from __future__ import annotations

from typing import Any

#: The server's view size (256 by default): taken from the episodes.
_IMAGE = {"shape": None, "dtype": "uint8"}


def success_mask(transitions: Any) -> Any:
    """ManiSkill's ``success`` after each control step."""
    return transitions["terminated"]


def _spec(robot_type: str, wrist: bool) -> dict[str, Any]:
    images = ("agentview_images", "wrist_images") if wrist else ("agentview_images",)
    return {
        "robot": "maniskill",
        "robot_type": robot_type,
        # The tabletop tasks' control_freq.
        "fps": 20,
        "arrays": {
            **{key: _IMAGE for key in images},
            "states": {"shape": (8,), "dtype": "float32"},
            "actions": {"shape": (4,), "dtype": "float32"},
        },
        "image_fields": images,
        "cameras": {key: key.removesuffix("_images") for key in images},
        "state_names": [
            *("tcp_x", "tcp_y", "tcp_z", "tcp_qw", "tcp_qx", "tcp_qy", "tcp_qz"),
            "gripper_width",
        ],
        "action_names": ["ee_dx", "ee_dy", "ee_dz", "gripper"],
        "success_mask": success_mask,
        #: What one dataset shares: one arm, env id and scene (a rig's cam_t moves its camera).
        "group": ("maniskill_robot", "env_id", "scene"),
    }


#: The Panda (the default ``--robot``, and the one the RLinf rigs run).
SPEC = _spec("panda", wrist=True)

#: One spec per ``--robot`` (``--space``).
SPACES = {
    "panda": SPEC,
    "xarm6_robotiq": _spec("xarm6_robotiq", wrist=True),
    "widowxai": _spec("widowxai", wrist=False),
}
