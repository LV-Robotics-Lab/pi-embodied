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

"""The RoboTwin primitive registry (``code.api``, ../../components/code_api.py).

cuRobo's arm planning is the pose-level primitive the server holds; the LingBot VLA and the
scripted motions run agent-side over ``chunk_step``.
"""

from __future__ import annotations

from pi_embodied_services.components.code_api import (
    GROUND_TRUTH,
    TASK_LANGUAGE,
    Param,
    Primitive,
)

_ACTION_TYPE = Param(
    "string",
    "'qpos' (14 joints, default) or 'ee' (16: two 7-D poses and grippers)",
    False,
)

ROBOTWIN_PRIMITIVES = (
    TASK_LANGUAGE,
    Primitive(
        "plan_arm_path",
        "env.plan_arm_path",
        "Plan (not run) one arm's joint path to a world pose [x, y, z, qw, qx, qy, qz] with cuRobo.",
        {
            "arm": Param("string", "'left' or 'right'"),
            "target_pose": Param("array", "7 floats"),
        },
    ),
    Primitive(
        "render_camera",
        "env.render_camera",
        "One camera's RGB (and depth) frame.",
        {
            "camera_name": Param("string", "'head', 'left_wrist' or 'right_wrist'"),
            "depth": Param("boolean", "also return the depth map", False),
        },
        tiers=("low",),
    ),
    Primitive(
        "get_camera_meta",
        "env.get_camera_meta",
        "Camera intrinsics and extrinsics.",
        {"camera_name": Param("string", "'head', 'left_wrist' or 'right_wrist'")},
        tiers=("low",),
    ),
    Primitive(
        "step",
        "env.step",
        "One native action of both arms.",
        {"action": Param("array", "14 or 16 floats"), "action_type": _ACTION_TYPE},
        mutating=True,
        tiers=("low",),
    ),
    Primitive(
        "chunk_step",
        "env.chunk_step",
        "Run N native actions in one call.",
        {
            "actions": Param("array", "N x 14 or N x 16 floats"),
            "action_type": _ACTION_TYPE,
            "return_all_frames": Param("boolean", "one observation per action", False),
        },
        mutating=True,
        tiers=("low",),
    ),
    GROUND_TRUTH,
)
