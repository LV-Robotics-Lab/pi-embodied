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

"""The Metaworld primitive registry (``code.api``, ../../components/code_api.py)."""

from __future__ import annotations

from pi_embodied_services.components.code_api import (
    GROUND_TRUTH,
    TASK_LANGUAGE,
    Param,
    Primitive,
)

_CAMERA = {
    "camera_name": Param("string", "'agentview' or 'wrist'", False),
    "height": Param("integer", "pixels (default 256)", False),
    "width": Param("integer", "pixels (default 256)", False),
}

METAWORLD_PRIMITIVES = (
    TASK_LANGUAGE,
    Primitive(
        "state",
        "env.state",
        "TCP position (world: +y away from the robot, +x its right, +z up), gripper width and command, the task's success flags and metrics, the workspace box.",
    ),
    Primitive(
        "move_delta",
        "env.move_delta",
        "Translate the gripper by a world-frame delta (m), closed-loop; at most 0.2 m per call and inside the workspace box, else refused. Optionally open or close the gripper first.",
        {
            "delta_xyz": Param("vec3", "[dx, dy, dz] in m"),
            "gripper": Param(
                "string", "'open' or 'close' first (default: keep)", False
            ),
        },
        mutating=True,
    ),
    Primitive(
        "set_gripper",
        "env.set_gripper",
        "Open or close the gripper in place and hold until the fingers settle.",
        {"open": Param("boolean", "True opens, False closes")},
        mutating=True,
    ),
    Primitive(
        "render_camera",
        "env.render_camera",
        "One camera's RGB frame, and its metric depth map with depth=true.",
        {**_CAMERA, "depth": Param("boolean", "also return the depth map", False)},
        tiers=("low",),
    ),
    Primitive(
        "get_camera_meta",
        "env.get_camera_meta",
        "OpenCV intrinsics and camera-to-world extrinsic of a camera at a resolution.",
        _CAMERA,
        tiers=("low",),
    ),
    Primitive(
        "step",
        "env.step",
        "One action [dx, dy, dz, gripper] in [-1, 1]: 1 cm per unit, +1 closes the gripper.",
        {"action": Param("array", "4 floats")},
        mutating=True,
        tiers=("low",),
    ),
    Primitive(
        "chunk_step",
        "env.chunk_step",
        "Run actions [N, 4] in one call; stops early at success.",
        {
            "actions": Param("array", "N x 4 floats"),
            "return_all_frames": Param("boolean", "one observation per action", False),
        },
        mutating=True,
        tiers=("low",),
    ),
    GROUND_TRUTH,
)
