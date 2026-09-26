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

"""The Robosuite primitive registry (``code.api``, ../../components/code_api.py).

CaP-X's tiers on this server: ``high`` (S2, FrankaControlApi) is perception (``segment`` with a
SAM3 server, ``back_project``) plus pose-level motion (``move_to`` with an optional orientation,
the gripper); ``low`` (S3, FrankaControlApiReduced) is the raw observation, relative moves and the
gripper; ``privileged`` (S1) adds the simulator's poses. Two-arm tasks take ``arm`` on every
motion primitive. CaP-X's joint-space IK (``solve_ik`` / ``move_to_joints``) is not offered: the
arms run OSC_POSE, so a Cartesian target is the primitive.
"""

from __future__ import annotations

from pi_embodied_services.components.code_api import (
    GROUND_TRUTH,
    TASK_LANGUAGE,
    Param,
    Primitive,
)

_ARM = {
    "arm": Param(
        "string", "'robot0' or 'robot1' on the two-arm tasks; omit on one arm", False
    )
}
_CAMERA = {"camera": Param("string", "'agentview' (default) or 'wrist'", False)}
_MOTION = {
    "quat_xyzw": Param(
        "array", "absolute world orientation to reach (default: keep)", False
    ),
    "rotvec": Param(
        "vec3",
        "instead: a world-frame turn (axis x angle, rad) from the current",
        False,
    ),
    "gripper": Param("string", "'open' / 'close' first (default: keep)", False),
    "tol_m": Param("number", "stop within this distance, m (default 0.005)", False),
    "tol_rad": Param("number", "orientation tolerance, rad (default 0.03)", False),
    "step_m": Param("number", "per-step travel cap, m (default 0.02)", False),
    "step_rad": Param("number", "per-step turn cap, rad (default 0.2)", False),
    "max_steps": Param("integer", "control-step budget (default 100)", False),
}

ROBOSUITE_PRIMITIVES = (
    TASK_LANGUAGE,
    Primitive(
        "get_state",
        "env.get_state",
        "Per arm: <arm>_eef_pos (m, world), <arm>_eef_quat (xyzw), <arm>_joint_pos, <arm>_gripper_width, <arm>_gripper_command; success, env_steps, table_z, home_eef_pos.",
        example='st = get_state()\nprint(st["robot0_eef_pos"], st["robot0_gripper_width"], st["success"])',
    ),
    Primitive(
        "get_observation",
        "env.get_observation",
        "agentview and wrist, each {rgb uint8[512,512,3], depth float32[512,512] m, intrinsic_K 3x3, extrinsic_cam2world 4x4}, plus get_state's fields.",
        example='obs = get_observation()\nrgb, depth = obs["agentview"]["rgb"], obs["agentview"]["depth"]\nK = obs["agentview"]["intrinsic_K"]; T = obs["agentview"]["extrinsic_cam2world"]',
    ),
    Primitive(
        "segment",
        "env.segment",
        "SAM3 mask of a text prompt in the current image: found, score, box, mask, centroid_rowcol, world_xyz (median through the depth).",
        {
            "prompt": Param("string", "what to segment, e.g. 'red cube'"),
            **_CAMERA,
            "min_score": Param("number", "SAM3 threshold (default 0.2)", False),
        },
        tiers=("high",),
        example='m = segment("red cube")\nif m["found"]:\n    print(m["world_xyz"], m["score"])',
    ),
    Primitive(
        "back_project",
        "env.back_project",
        "World xyz of pixel (row, col) of the current 512x512 image through its depth.",
        {
            "row": Param("integer", "pixel row (0 = top)"),
            "col": Param("integer", "pixel column"),
            **_CAMERA,
        },
        tiers=("high",),
        example='print(back_project(256, 300)["world_xyz"])',
    ),
    Primitive(
        "preview_reach",
        "env.preview_reach",
        "Whether the TCP can reach a world position (IK by the ik service; unknown without --ik).",
        {
            "pos": Param("vec3", "world [x, y, z] in m"),
            "quat_xyzw": Param("array", "orientation (default: current)", False),
            **_ARM,
        },
        tiers=("high",),
    ),
    Primitive(
        "move_to",
        "env.move_to",
        "Closed-loop OSC_POSE servo of the TCP to a world position, holding the orientation unless one is given; refused beyond the per-call cap, outside the workspace or below the table.",
        {"target_xyz": Param("vec3", "world [x, y, z] in m"), **_ARM, **_MOTION},
        mutating=True,
        tiers=("high",),
        example='move_to([0.05, 0.02, 0.95], gripper="open")\nr = move_to([0.05, 0.02, 0.83])\nprint(r["final_dist_m"])',
    ),
    Primitive(
        "move_delta",
        "env.move_delta",
        "Servo the TCP by a world-frame offset, holding the orientation; the same limits as move_to.",
        {"delta_xyz": Param("vec3", "world [dx, dy, dz] in m"), **_ARM, **_MOTION},
        mutating=True,
        tiers=("low",),
        example="move_delta([0, 0, 0.1])                    # lift 10 cm\nmove_delta([0, 0, 0], rotvec=[0, 0, 0.5])  # turn the wrist 0.5 rad",
    ),
    Primitive(
        "set_gripper",
        "env.set_gripper",
        "Open or close one gripper and hold; the command stays in force for later moves.",
        {
            "close": Param("boolean", "true closes, false opens"),
            **_ARM,
            "steps": Param(
                "integer", "control steps to drive the fingers (default 15)", False
            ),
        },
        mutating=True,
        example='r = set_gripper(True)\nprint(r["gripper_width"])  # near 0: closed on nothing',
    ),
    Primitive(
        "raw_obs",
        "env.raw_obs",
        "robosuite's robot*_ observations (joint state, eef pose, gripper); no object poses.",
        tiers=("low",),
        example='o = raw_obs()\nprint(o["robot0_eef_pos"], o["robot0_gripper_qpos"])',
    ),
    Primitive(
        "render_camera",
        "env.render_camera",
        "Upright RGB (and metric depth) of a camera.",
        {
            "camera_name": Param("string", "'agentview' or 'wrist'", False),
            "height": Param("integer", "pixels (default 512)", False),
            "width": Param("integer", "pixels (default 512)", False),
            "depth": Param("boolean", "also return the depth map", False),
        },
        tiers=("low",),
        example='rgb, depth = render_camera("wrist", 256, 256, depth=True)',
    ),
    Primitive(
        "get_camera_meta",
        "env.get_camera_meta",
        "OpenCV intrinsics and camera-to-world extrinsic for the upright image.",
        {
            "camera_name": Param("string", "'agentview' or 'wrist'", False),
            "height": Param("integer", "pixels (default 512)", False),
            "width": Param("integer", "pixels (default 512)", False),
        },
        tiers=("low",),
        example='meta = get_camera_meta("agentview")\nK, T = meta["intrinsic_K"], meta["extrinsic_cam2world"]',
    ),
    Primitive(
        "step",
        "env.step",
        "One raw composite action: per arm 6 OSC_POSE deltas in [-1, 1] then the gripper (+1 close, -1 open).",
        {"action": Param("array", "action_dim floats")},
        mutating=True,
        tiers=("low",),
        example="step([0, 0, 0.5, 0, 0, 0, -1])  # up 2.5 cm, gripper open (one arm)",
    ),
    GROUND_TRUTH,
)
