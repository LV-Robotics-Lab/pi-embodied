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

"""The ManiSkill primitive registry (``code.api``, ../../components/code_api.py)."""

from __future__ import annotations

from pi_embodied_services.components.code_api import (
    GROUND_TRUTH,
    TASK_LANGUAGE,
    Param,
    Primitive,
)

_CAMERA = {"camera_name": Param("string", "'agentview' or 'wrist'", False)}

MANISKILL_PRIMITIVES = (
    TASK_LANGUAGE,
    Primitive(
        "state",
        "env.state",
        "TCP pose, gripper opening and the success flags.",
        example='st = state()\nprint(st["tcp_pos"], st["gripper_width"], st["info"].get("success"))',
    ),
    Primitive(
        "servo",
        "env.servo",
        "Drive the TCP to a world xyz (m) closed-loop with the gripper command held; returns [frames, info].",
        {
            "target_xyz": Param("vec3", "world [x, y, z] in m"),
            "gripper": Param("number", "> 0 open, <= 0 close"),
            "gain": Param("number", "error gain (default 1.3)", False),
            "tol_m": Param("number", "stop below this error (default 0.002)", False),
            "min_steps": Param("integer", "default 2", False),
            "max_steps": Param("integer", "default 8", False),
            "arm": Param("string", "two-arm robots: left or right", False),
        },
        mutating=True,
        example='x, y, z = state()["tcp_pos"]\nframes, info = servo([x, y, z - 0.02], 1)  # down 2 cm, gripper open\nprint(frames[-1]["tcp_pos"], info.get("is_grasped"))',
    ),
    Primitive(
        "render_camera",
        "env.render_camera",
        "The latest frame of a camera.",
        _CAMERA,
        tiers=("low",),
        example='rgb = render_camera("wrist")  # uint8 [256, 256, 3]',
    ),
    Primitive(
        "get_camera_meta",
        "env.get_camera_meta",
        "Camera intrinsics and camera-to-world extrinsic.",
        _CAMERA,
        tiers=("low",),
        example='meta = get_camera_meta("agentview")\nK, T = meta["intrinsic_K"], meta["extrinsic_cam2world"]',
    ),
    Primitive(
        "step",
        "env.step",
        "One pd_ee_delta_pos action [dx, dy, dz, gripper] (gripper > 0 open, < 0 close on every robot; [dx, dy, dz] without a gripper; two arms: both concatenated).",
        {"action": Param("array", "4 floats (3 without a gripper)")},
        mutating=True,
        tiers=("low",),
        example='r = step([0, 0, 0.1, 1])  # up about 1 cm; the Panda\'s gripper +1 opens\nprint(r["state"]["tcp_pos"], r["info"].get("success"))',
    ),
    Primitive(
        "chunk_step",
        "env.chunk_step",
        "Run actions [N, 4] in one call; stops early on termination or success.",
        {
            "actions": Param("array", "N x 4 floats (N x 3 without a gripper)"),
            "return_all_frames": Param("boolean", "one observation per action", False),
        },
        mutating=True,
        tiers=("low",),
        example='r = chunk_step([[0.1, 0, 0, 1]] * 5)  # about 5 cm along +x\nprint(r["steps"], r["state"]["tcp_pos"])',
    ),
    GROUND_TRUTH,
)
