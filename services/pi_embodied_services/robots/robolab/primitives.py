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

"""The RoboLab primitive registry (``code.api``, ../../components/code_api.py)."""

from __future__ import annotations

from pi_embodied_services.components.code_api import (
    GROUND_TRUTH,
    TASK_LANGUAGE,
    Param,
    Primitive,
)

_CAMERA = {"camera_name": Param("string", "'agentview' or 'wrist'", False)}

ROBOLAB_PRIMITIVES = (
    TASK_LANGUAGE,
    Primitive(
        "state",
        "env.state",
        "Hand pose, gripper width and command, success and step count.",
        example='st = state()\nprint(st["eef_pos"], st["gripper_width"], st["yaw_deg"], st["success"])',
    ),
    Primitive(
        "move_delta",
        "env.move_delta",
        "Translate the hand by a base-frame delta (m, at most 0.3 per call), optionally after a gripper command; the orientation is held.",
        {
            "delta_xyz": Param("vec3", "base-frame [dx, dy, dz] in m"),
            "gripper": Param("string", "'open' or 'close' first", False),
            "return_frames": Param(
                "boolean", "also return every control step's frames", False
            ),
        },
        mutating=True,
        example='move_delta([0, 0, 0.1], gripper="open")  # open, then lift 10 cm\nr = move_delta([0.05, 0, -0.1])\nprint(r["moved_m"], r["eef_pos"])',
    ),
    Primitive(
        "rotate_delta",
        "env.rotate_delta",
        "Turn the hand by a yaw (rad) about the base vertical axis through it (+ = counter-clockwise seen from above; clipped to 0.3 per call), holding its position and tilt.",
        {
            "yaw": Param("number", "rad about base +z"),
            "return_frames": Param(
                "boolean", "also return every decision's front frame", False
            ),
        },
        mutating=True,
        example='r = rotate_delta(0.3)  # counter-clockwise seen from above\nprint(r["yaw"], r["yaw_deg"])',
    ),
    Primitive(
        "render_camera",
        "env.render_camera",
        "The latest frame of a camera.",
        _CAMERA,
        tiers=("low",),
        example='rgb = render_camera("agentview")  # uint8 [H, W, 3]\nwrist = render_camera("wrist")      # fingertips at the top',
    ),
    Primitive(
        "get_camera_meta",
        "env.get_camera_meta",
        "The front camera's intrinsics and camera-to-base extrinsic.",
        _CAMERA,
        tiers=("low",),
        example='meta = get_camera_meta("agentview")\nK, T = meta["intrinsic_K"], meta["extrinsic_cam2world"]',
    ),
    Primitive(
        "step",
        "env.step",
        "One raw relative-IK control step [dx, dy, dz, drx, dry, drz, gripper] (no orientation hold).",
        {"action": Param("array", "7 floats")},
        mutating=True,
        tiers=("low",),
        example='r = step([0, 0, 0.02, 0, 0, 0, 0])  # up, gripper open (> 0.5 closes)\nprint(r["state"]["eef_pos"], r["terminated"])',
    ),
    Primitive(
        "chunk_step",
        "env.chunk_step",
        "Raw control steps [N, 7] in one call.",
        {
            "actions": Param("array", "N x 7 floats"),
            "return_all_frames": Param("boolean", "one observation per step", False),
        },
        mutating=True,
        tiers=("low",),
        example='r = chunk_step([[0, 0, -0.02, 0, 0, 0, 0]] * 8)  # 8 steps down\nprint(r["obs"]["eef_pos"], r["terminated"])',
    ),
    GROUND_TRUTH,
)
