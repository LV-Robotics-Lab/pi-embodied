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

"""The RoboCasa primitive registry (``code.api``, ../../components/code_api.py).

The OSC servo, base navigation and the RLDX VLA run agent-side over ``step``; the server's high
tier is the task read-outs (success, progress, contact).
"""

from __future__ import annotations

from pi_embodied_services.components.code_api import (
    GROUND_TRUTH,
    TASK_LANGUAGE,
    Param,
    Primitive,
)

_CAMERA = {
    "camera_name": Param(
        "string",
        "'robot0_agentview_left', 'mobilebase0_navview', 'robot0_eye_in_hand', ...",
    ),
    "height": Param("integer", "pixels", False),
    "width": Param("integer", "pixels", False),
}

ROBOCASA_PRIMITIVES = (
    TASK_LANGUAGE,
    Primitive("check_success", "env.check_success", "Whether the task is solved."),
    Primitive(
        "get_task_progress",
        "env.get_task_progress",
        "The task's progress dict (which success conditions hold).",
    ),
    Primitive(
        "get_success_criteria_text",
        "env.get_success_criteria_text",
        "The task's success criteria as text.",
    ),
    Primitive(
        "grasp_contact",
        "env.grasp_contact",
        "Whether the gripper touches a task object.",
    ),
    Primitive(
        "render_camera",
        "env.render_camera",
        "One camera's RGB (and depth) frame, robosuite-native orientation.",
        {
            "camera_name": Param("string", "camera name"),
            "height": Param("integer", "pixels"),
            "width": Param("integer", "pixels"),
            "depth": Param("boolean", "also return the depth map"),
        },
        tiers=("low",),
    ),
    Primitive(
        "get_camera_meta",
        "env.get_camera_meta",
        "Camera intrinsics and extrinsics.",
        _CAMERA,
        tiers=("low",),
    ),
    Primitive(
        "get_camera_transform",
        "env.get_camera_transform",
        "The camera's world-to-pixel transform.",
        _CAMERA,
        tiers=("low",),
    ),
    Primitive(
        "step",
        "env.step",
        "One PandaOmron action [eef_pos 3, eef_rot 3, gripper, base 3, torso, base_mode].",
        {"flat_action": Param("array", "12 floats")},
        mutating=True,
        tiers=("low",),
    ),
    GROUND_TRUTH,
)
