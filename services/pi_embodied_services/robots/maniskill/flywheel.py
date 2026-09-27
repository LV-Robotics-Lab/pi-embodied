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

The ``--robot`` arms differ in action meaning (the Robotiq's gripper action is the Panda's
negated), width (the stick has no gripper element; the bridge WidowX 250 S takes an unnormalised
``[dx, dy, dz]`` in m, three zero rotations and the gripper; the Panda pair one action per arm,
left then right) and cameras (the WidowX AI, the stick, the pair and the WidowX 250 S have no
wrist camera), so ``SPACES`` has one spec per arm (``--space``, pi's /flywheel-export passes the
episode's arm), the raw path starts with the arm, and a dataset holds one arm, env id and scene
(``group``). A two-arm robot's state is each arm's, in the same order as its actions."""

from __future__ import annotations

from typing import Any

#: The server's view size (256 by default): taken from the episodes.
_IMAGE = {"shape": None, "dtype": "uint8"}


def success_mask(transitions: Any) -> Any:
    """ManiSkill's ``success`` after each control step."""
    return transitions["terminated"]


#: One arm's state, and the delta-position action with the arm's own gripper action.
_ARM_STATE = (
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_qw",
    "tcp_qx",
    "tcp_qy",
    "tcp_qz",
    "gripper_width",
)
_DELTA_POS = ("ee_dx", "ee_dy", "ee_dz", "gripper")


def _spec(
    robot_type: str,
    wrist: bool,
    *,
    action: tuple[str, ...] = _DELTA_POS,
    arms: tuple[str, ...] = (),
) -> dict[str, Any]:
    images = ("agentview_images", "wrist_images") if wrist else ("agentview_images",)
    prefixes = [f"{a}_" for a in arms] or [""]
    state_names = [p + n for p in prefixes for n in _ARM_STATE]
    action_names = [p + n for p in prefixes for n in action]
    return {
        "robot": "maniskill",
        "robot_type": robot_type,
        # The tabletop tasks' control_freq.
        "fps": 20,
        "arrays": {
            **{key: _IMAGE for key in images},
            "states": {"shape": (len(state_names),), "dtype": "float32"},
            "actions": {"shape": (len(action_names),), "dtype": "float32"},
        },
        "image_fields": images,
        "cameras": {key: key.removesuffix("_images") for key in images},
        "state_names": state_names,
        "action_names": action_names,
        "success_mask": success_mask,
        #: What one dataset shares: one arm, env id and scene (a rig's cam_t moves its camera).
        "group": ("maniskill_robot", "env_id", "scene"),
        #: The arm this space describes: an export under it refuses another arm's episodes.
        "metadata": {"maniskill_robot": robot_type},
    }


#: The Panda (the default ``--robot``, and the one the RLinf rigs run).
SPEC = _spec("panda", wrist=True)

#: One spec per ``--robot`` (``--space``).
SPACES = {
    "panda": SPEC,
    "xarm6_robotiq": _spec("xarm6_robotiq", wrist=True),
    "widowxai": _spec("widowxai", wrist=False),
    "panda_stick": _spec("panda_stick", wrist=False, action=_DELTA_POS[:3]),
    "panda_pair": _spec("panda_pair", wrist=False, arms=("left", "right")),
    "widowx250s": _spec(
        "widowx250s",
        wrist=False,
        action=(
            *("ee_dx_m", "ee_dy_m", "ee_dz_m", "ee_droll", "ee_dpitch", "ee_dyaw"),
            "gripper",
        ),
    ),
}
