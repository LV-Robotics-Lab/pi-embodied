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

"""The RoboDojo primitive registry (``code.api``, ../../components/code_api.py)."""

from __future__ import annotations

from pi_embodied_services.components.code_api import (
    GROUND_TRUTH,
    TASK_LANGUAGE,
    Param,
    Primitive,
)

_ARM = Param("string", "'left' or 'right'")
_FRAMES = Param("boolean", "also return head frames along the motion", False)
_GRIPPER = Param("number", "gripper command first: 1 open .. 0 closed", False)

ROBODOJO_PRIMITIVES = (
    TASK_LANGUAGE,
    Primitive(
        "state",
        "env.state",
        "Both arms' end-effector poses (env frame), joints and grippers; success, score and step count.",
    ),
    Primitive(
        "move_to",
        "env.move_to",
        "Move one arm's end effector in a straight line to an env-frame xyz (m, at most 0.5 m away) and optional wxyz orientation; the other arm holds.",
        {
            "arm": _ARM,
            "xyz": Param("vec3", "env-frame [x, y, z] in m"),
            "quat_wxyz": Param("array", "[qw, qx, qy, qz] (default: keep)", False),
            "gripper": _GRIPPER,
            "return_frames": _FRAMES,
        },
        mutating=True,
    ),
    Primitive(
        "move_delta",
        "env.move_delta",
        "Translate one arm's end effector by an env-frame delta (m, at most 0.5 per call), orientation held.",
        {
            "arm": _ARM,
            "delta_xyz": Param("vec3", "env-frame [dx, dy, dz] in m"),
            "gripper": _GRIPPER,
            "return_frames": _FRAMES,
        },
        mutating=True,
    ),
    Primitive(
        "rotate_delta",
        "env.rotate_delta",
        "Turn one arm's gripper by a yaw (rad, + counter-clockwise seen from above; clipped to 0.8) about the vertical through it.",
        {
            "arm": _ARM,
            "yaw": Param("number", "rad about env +z"),
            "return_frames": _FRAMES,
        },
        mutating=True,
    ),
    Primitive(
        "set_gripper",
        "env.set_gripper",
        "Move one gripper to a value, 1 open .. 0 closed.",
        {
            "arm": _ARM,
            "value": Param("number", "1 open .. 0 closed"),
            "return_frames": _FRAMES,
        },
        mutating=True,
    ),
    Primitive(
        "go_home",
        "env.go_home",
        "Drive both arms back to their start joints (most tasks require it for success).",
        {"return_frames": _FRAMES},
        mutating=True,
    ),
    Primitive(
        "back_project",
        "env.back_project",
        "Env-frame xyz of [col, row] pixels of the latest head image, from its metric depth.",
        {
            "pixels": Param("array", "[[col, row], ...]"),
            "camera_name": Param("string", "'head'", False),
        },
    ),
    Primitive(
        "render_camera",
        "env.render_camera",
        "The latest frame of 'head', 'left_wrist' or 'right_wrist'.",
        {
            "camera_name": Param(
                "string", "'head', 'left_wrist' or 'right_wrist'", False
            )
        },
        tiers=("low",),
    ),
    Primitive(
        "get_camera_meta",
        "env.get_camera_meta",
        "The head camera's OpenCV intrinsics and camera-to-env extrinsic.",
        {"camera_name": Param("string", "'head'", False)},
        tiers=("low",),
    ),
    Primitive(
        "get_obs",
        "env.get_obs",
        "RoboDojo's native observation dict (vision, state, action, instruction).",
        {"depth": Param("boolean", "include metric depth", False)},
        tiers=("low",),
    ),
    Primitive(
        "step",
        "env.step",
        "One native RoboDojo action dict (left/right_arm_joint_state 6 + left/right_ee_joint_state 1, or left/right_ee_pose 7) = one 25 Hz control step.",
        {"action": Param("object", "RoboDojo action dict")},
        mutating=True,
        tiers=("low",),
    ),
    Primitive(
        "chunk_step",
        "env.chunk_step",
        "Native RoboDojo action dicts in order, in one call.",
        {
            "actions": Param("array", "list of action dicts"),
            "return_all_frames": Param("boolean", "a head frame per action", False),
        },
        mutating=True,
        tiers=("low",),
    ),
    GROUND_TRUTH,
)
