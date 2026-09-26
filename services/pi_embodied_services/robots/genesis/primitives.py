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
        "TCP pose (m, base frame), gripper opening and command, success, is_grasped.",
        example='st = state()\nprint(st["tcp_pos"], st["gripper_width"], st["is_grasped"])',
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
        example='move_delta([0, 0, -0.1], gripper="open")  # open, then down 10 cm\nr = move_delta([0.05, 0, 0])\nprint(r["moved_m"], r["tcp_pos"])',
    ),
    Primitive(
        "set_gripper",
        "env.set_gripper",
        "Open or close the gripper and hold; a close ending nearly shut reports grasp_empty.",
        {"open": Param("boolean", "true opens, false closes")},
        mutating=True,
        example='r = set_gripper(False)\nprint(r["gripper_width"], r.get("grasp_empty", False))',
    ),
    Primitive(
        "back_project",
        "env.back_project",
        "World xyz (m) of image pixels from the camera's depth; null where the depth is missing.",
        {**_CAMERA, "pixels": Param("array", "[[row, col], ...]")},
        example='xyz = back_project("agentview", [[128, 128], [140, 120]])\nprint(xyz[0])  # None where there is no depth',
    ),
    Primitive(
        "render_camera",
        "env.render_camera",
        "The current frame of a camera (as the model sees it).",
        _CAMERA,
        tiers=("low",),
        example='rgb = render_camera("wrist")  # uint8 [256, 256, 3]',
    ),
    Primitive(
        "get_camera_meta",
        "env.get_camera_meta",
        "Camera intrinsics (OpenCV K) and camera-to-world extrinsic.",
        _CAMERA,
        tiers=("low",),
        example='meta = get_camera_meta("agentview")\nK, T = meta["intrinsic_K"], meta["extrinsic_cam2world"]',
    ),
    Primitive(
        "step",
        "env.step",
        "One control step of [dx, dy, dz, gripper] (m; +1 open / -1 close), under the same limits.",
        {"action": Param("array", "4 floats")},
        mutating=True,
        tiers=("low",),
        example='r = step([0, 0, 0.01, -1])  # up 1 cm, gripper closed\nprint(r["success"], r["state"]["tcp_pos"])',
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
        example='r = chunk_step([[0, 0, 0.01, -1]] * 10)  # 10 cm up in 1 cm steps\nprint(r["success"], r["state"]["tcp_pos"])',
    ),
    GROUND_TRUTH,
)
