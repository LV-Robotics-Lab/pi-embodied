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

"""Robosuite's Flywheel data rules (the TS recorder's FLYWHEEL in packages/embodied/src/robots/robosuite).

One transition per robosuite control step of a motion (``env.move_to`` / ``env.move_delta`` /
``env.set_gripper`` with ``record``): the task camera and the wrist view at 256 px, robomimic's
low-dim robot state per arm (``eef_pos``, ``eef_quat`` xyzw, ``gripper_qpos``), and the composite
OSC_POSE action the servo sent, per arm ``[dx, dy, dz, ax, ay, az]`` in [-1, 1] (base frame)
then the held gripper command (+1 close, -1 open).

The tasks differ in their arms and grippers, so the width of the state and the action does:
``SPACES`` has one spec each (``--space``; pi's /flywheel-export passes the task's), and a
dataset holds one task (``group``)."""

from __future__ import annotations

from typing import Any

_IMAGE = {"shape": (256, 256, 3), "dtype": "uint8"}


def success_mask(transitions: Any) -> Any:
    """Robosuite's ``_check_success`` (Restack: CaP-X's rule), latched at its first step."""
    return transitions["terminated"]


def _spec(arms: tuple[str, ...], gripper: bool, robot_type: str) -> dict[str, Any]:
    fingers = ("gripper_qpos_l", "gripper_qpos_r") if gripper else ()
    state = [
        f"{arm}_{d}"
        for arm in arms
        for d in ("eef_x", "eef_y", "eef_z", "eef_qx", "eef_qy", "eef_qz", "eef_qw")
        + fingers
    ]
    action = [
        f"{arm}_{d}"
        for arm in arms
        for d in ("dx", "dy", "dz", "drx", "dry", "drz")
        + (("gripper",) if gripper else ())
    ]
    return {
        "robot": "robosuite",
        "robot_type": robot_type,
        # robosuite's control_freq.
        "fps": 20,
        "arrays": {
            "agentview_images": _IMAGE,
            "wrist_images": _IMAGE,
            "states": {"shape": (len(state),), "dtype": "float32"},
            "actions": {"shape": (len(action),), "dtype": "float32"},
        },
        "image_fields": ("agentview_images", "wrist_images"),
        "cameras": {
            "agentview_images": "agentview",
            "wrist_images": "robot0_eye_in_hand",
        },
        "state_names": state,
        "action_names": action,
        "success_mask": success_mask,
        #: What one dataset shares: every episode of one task.
        "group": ("task",),
    }


#: Lift, Stack, Restack, NutAssemblySquare: one Panda with its gripper.
SPEC = _spec(("robot0",), True, "panda")

#: The spaces by the tasks' arms (``--space``): TwoArmLift / TwoArmHandover, Wipe's fingerless
#: sponge; ``SPEC`` is the one-arm tasks'.
SPACES = {
    "one_arm": SPEC,
    "two_arm": _spec(("robot0", "robot1"), True, "panda_x2"),
    "wipe": _spec(("robot0",), False, "panda"),
}
