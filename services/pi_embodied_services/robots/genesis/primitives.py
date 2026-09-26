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

"""The Genesis primitive registry (``code.api``, ../../components/code_api.py)."""

from __future__ import annotations

from pi_embodied_services.components.code_api import (
    GROUND_TRUTH,
    TASK_LANGUAGE,
    Param,
    Primitive,
)

_CAMERA = {"camera_name": Param("string", "'agentview' or 'wrist'", False)}

GENESIS_PRIMITIVES = (
    TASK_LANGUAGE,
    Primitive(
        "state",
        "env.state",
        "TCP pose (m, base frame), gripper opening and command, success, is_grasped, lift_m.",
    ),
    Primitive(
        "move_delta",
        "env.move_delta",
        "Translate the TCP by a base-frame delta (m; +x away from the base, +y to the robot's left,"
        " +z up) in ~2 cm IK decisions with the orientation held, after an optional gripper command."
        " Refused (nothing moves) beyond 0.2 m per call, below the Z floor or outside the workspace box.",
        {
            "delta_xyz": Param("vec3", "[dx, dy, dz] in m"),
            "gripper": Param("string", "'open' or 'close' first, holding still", False),
        },
        mutating=True,
    ),
    Primitive(
        "set_gripper",
        "env.set_gripper",
        "Open or close the gripper and hold; a close ending nearly shut reports grasp_empty.",
        {"open": Param("boolean", "true opens, false closes")},
        mutating=True,
    ),
    Primitive(
        "back_project",
        "env.back_project",
        "World xyz (m) of image pixels from the camera's depth; null where the depth is missing.",
        {**_CAMERA, "pixels": Param("array", "[[row, col], ...]")},
    ),
    Primitive(
        "render_camera",
        "env.render_camera",
        "The current frame of a camera (as the model sees it).",
        _CAMERA,
        tiers=("low",),
    ),
    Primitive(
        "get_camera_meta",
        "env.get_camera_meta",
        "Camera intrinsics (OpenCV K) and camera-to-world extrinsic.",
        _CAMERA,
        tiers=("low",),
    ),
    Primitive(
        "step",
        "env.step",
        "One control step of [dx, dy, dz, gripper] (m; +1 open / -1 close), under the same limits.",
        {"action": Param("array", "4 floats")},
        mutating=True,
        tiers=("low",),
    ),
    Primitive(
        "chunk_step",
        "env.chunk_step",
        "Run actions [N, 4] in one call; stops at success or stop.",
        {
            "actions": Param("array", "N x 4 floats"),
            "return_all_frames": Param("boolean", "one observation per action", False),
        },
        mutating=True,
        tiers=("low",),
    ),
    GROUND_TRUTH,
)
