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

"""The BEHAVIOR / R1Pro primitive registry (``code.api``, ../../components/code_api.py): CaP-X's
R1ProControlApi motion set (navigate_to_pose, move_hand, grasp_object, open/close_gripper,
get_robot_position) in the high tier, the raw camera reads and control steps in the low tier, and
the simulator's ground truth in the privileged one."""

from __future__ import annotations

from pi_embodied_services.components.code_api import (
    GROUND_TRUTH,
    TASK_LANGUAGE,
    Param,
    Primitive,
)

_ARM = Param("string", "'left' or 'right'")
_POSE = {
    "arm": _ARM,
    "position": Param("vec3", "world [x, y, z] in m"),
    "quat_xyzw": Param(
        "array", "world xyzw orientation (default: the current one)", False
    ),
}
_CAMERA = {
    "camera_name": Param("string", "'head', 'left_wrist' or 'right_wrist'", False)
}

BEHAVIOR_PRIMITIVES = (
    TASK_LANGUAGE,
    Primitive(
        "get_robot_position",
        "env.get_robot_position",
        "World pose of the base (pos, quat_xyzw, yaw) and of both end effectors.",
    ),
    Primitive(
        "navigate_to_pose",
        "env.navigate_to_pose",
        "Drive the base to world (x, y) facing yaw (rad about +z), planning around obstacles; at most 5 m per call. Returns ok, the pose reached and the observation.",
        {
            "x": Param("number", "world x, m"),
            "y": Param("number", "world y, m"),
            "yaw": Param("number", "heading, rad"),
        },
        mutating=True,
        tiers=("high",),
    ),
    Primitive(
        "move_hand",
        "env.move_hand",
        "Plan and move an arm's end effector to a world pose, obstacles respected; reach about 1.5 m from the base. Returns ok, the eef pose reached, distance_left_m and the observation.",
        _POSE,
        mutating=True,
        tiers=("high",),
    ),
    Primitive(
        "grasp_object",
        "env.grasp_object",
        "Grasp at a world pose: open, hover pregrasp_offset_m above, descend, close, settle, lift. Judge the grasp from gripper_width, not ok.",
        {
            **_POSE,
            "pregrasp_offset_m": Param("number", "default 0.1 (0.02-0.5)", False),
        },
        mutating=True,
        tiers=("high",),
    ),
    Primitive(
        "open_gripper",
        "env.open_gripper",
        "Open an arm's gripper fully (releases what it holds).",
        {"arm": _ARM},
        mutating=True,
    ),
    Primitive(
        "close_gripper",
        "env.close_gripper",
        "Close an arm's gripper fully.",
        {"arm": _ARM},
        mutating=True,
    ),
    Primitive(
        "state",
        "env.state",
        "Base and end-effector poses, gripper widths, success, q_score and goal counts (no images).",
        tiers=("low",),
    ),
    Primitive(
        "render_camera",
        "env.render_camera",
        "The latest frame of a camera; with depth=true, [rgb, depth_m].",
        {**_CAMERA, "depth": Param("boolean", "also the metric depth", False)},
        tiers=("low",),
    ),
    Primitive(
        "get_camera_meta",
        "env.get_camera_meta",
        "Camera intrinsics and the OpenGL camera-to-world transform at the current pose.",
        _CAMERA,
        tiers=("low",),
    ),
    Primitive(
        "raw_obs",
        "env.raw_obs",
        "The robot's proprioception and joint positions.",
        tiers=("low",),
    ),
    Primitive(
        "step",
        "env.step",
        "One raw control step of the robot's full action vector.",
        {"action": Param("array", "the R1Pro action vector")},
        mutating=True,
        tiers=("low",),
    ),
    Primitive(
        "chunk_step",
        "env.chunk_step",
        "Run actions [N, action_dim] in one call; stops early on stop or the episode's end.",
        {
            "actions": Param("array", "N x action_dim floats"),
            "return_all_frames": Param("boolean", "one observation per action", False),
        },
        mutating=True,
        tiers=("low",),
    ),
    GROUND_TRUTH,
)
