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
        "get_state",
        "env.policy_frame",
        "The joint state qpos (measured) and qpos_target (commanded), each 14 [left joints6, left gripper, right joints6, right gripper], and the eef16 state [left pose7 (xyz, wxyz), left gripper, right pose7, right gripper]; grippers 0 closed .. 1 open.",
        example='st = get_state()\nprint(st["qpos_target"], st["state"][:3])  # left eef xyz',
    ),
    Primitive(
        "plan_arm_path",
        "env.plan_arm_path",
        "Plan (not run) one arm's joint path to a world pose [x, y, z, qw, qx, qy, qz] with cuRobo.",
        {
            "arm": Param("string", "'left' or 'right'"),
            "target_pose": Param("array", "7 floats"),
        },
        example='plan = plan_arm_path("left", [-0.2, -0.05, 0.9, 0, 0.707, 0, 0.707])\nif plan["status"] == "Success":\n    print(plan["position"].shape)  # [waypoints, 6] joint path',
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
        example='rgb = render_camera("head")\nrgb, depth = render_camera("left_wrist", depth=True)  # depth in m, NaN = no hit',
    ),
    Primitive(
        "get_camera_meta",
        "env.get_camera_meta",
        "Camera intrinsics and extrinsics.",
        {"camera_name": Param("string", "'head', 'left_wrist' or 'right_wrist'")},
        tiers=("low",),
        example='meta = get_camera_meta("head")\nK, cam2world = meta["intrinsic_K"], meta["cam2world_gl"]  # OpenGL camera axes',
    ),
    Primitive(
        "step",
        "env.step",
        "One native action of both arms.",
        {"action": Param("array", "14 or 16 floats"), "action_type": _ACTION_TYPE},
        mutating=True,
        tiers=("low",),
        example='q = list(get_state()["qpos_target"])\nq[13] = 0.0  # close the right gripper (0 closed .. 1 open)\nr = step(q)\nprint(r["info"]["robot_state"]["right_gripper"], r["info"]["episode_status"]["eval_success"])',
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
        example='plan = plan_arm_path("right", [0.2, -0.05, 0.9, 0, 0.707, 0, 0.707])\nq = list(get_state()["qpos_target"])\npath = [q[:7] + list(j) + [q[13]] for j in plan["position"]]  # right arm joints\nr = chunk_step(path[::4])\nprint(r["info"]["executed_actions"], r["info"]["robot_state"]["right_eef_pose"])',
    ),
    GROUND_TRUTH,
)
