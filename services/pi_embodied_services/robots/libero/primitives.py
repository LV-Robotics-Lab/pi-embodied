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

"""The LIBERO primitive registry (``code.api``, ../../components/code_api.py).

``LIBERO_PRIMITIVES`` is the raw simulator surface; ``CODE_PRIMITIVES`` are the facade's
Cartesian and perception methods that code mode's programs call (``code.run``,
``utils/code_exec.py``), and ``libero_primitives(sam3)`` is what the server declares.
"""

from __future__ import annotations

from pi_embodied_services.components.code_api import (
    GROUND_TRUTH,
    TASK_LANGUAGE,
    Param,
    Primitive,
)

LIBERO_PRIMITIVES = (
    TASK_LANGUAGE,
    Primitive(
        "render_camera",
        "env.render_camera",
        "One camera's RGB (and depth) frame.",
        {
            "camera_name": Param(
                "string", "'agentview' or 'robot0_eye_in_hand'", False
            ),
            "height": Param("integer", "pixels (default 1024)", False),
            "width": Param("integer", "pixels (default 1024)", False),
            "depth": Param("boolean", "also return the depth map", False),
        },
    ),
    Primitive(
        "get_camera_meta",
        "env.get_camera_meta",
        "Camera intrinsics and extrinsics at a resolution.",
        {
            "camera_name": Param("string", "camera (default agentview)", False),
            "height": Param("integer", "pixels (default 256)", False),
            "width": Param("integer", "pixels (default 256)", False),
        },
    ),
    Primitive(
        "preview_reach",
        "env.preview_reach",
        "IK reach check from the current joints, nothing moves (--ik): {status: reachable | unreachable | unknown, q, position_err, orientation_err, message}.",
        {
            "pos": Param("vec3", "world-frame [x, y, z] in m"),
            "quat_xyzw": Param(
                "array", "orientation (default: the current one)", False
            ),
        },
    ),
    Primitive(
        "raw_obs",
        "env.code_raw_obs",
        "The simulator's current observation dict (the robot's proprioception and the images).",
        tiers=("low",),
    ),
    Primitive(
        "step",
        "env.step",
        "One OSC action [dx, dy, dz, drx, dry, drz, gripper] (gripper -1 open, +1 close).",
        {"action": Param("array", "7 floats")},
        mutating=True,
        tiers=("low",),
    ),
    Primitive(
        "chunk_step",
        "env.chunk_step",
        "Run a chunk of OSC actions [N, 7] in one call.",
        {
            "actions": Param("array", "N x 7 floats"),
            "return_all_frames": Param("boolean", "one observation per action", False),
        },
        mutating=True,
        tiers=("low",),
    ),
    GROUND_TRUTH,
)

# ---- code mode (run_code): the Cartesian and perception primitives the facade serves as
# ``env.*`` RPC methods (env_server.py). High = CaP-X's S2 (perception plus pose-level motion),
# low = S3 (relative moves only); the raw simulator surface above stays in its tiers.

_CAMERA = Param("string", "'agentview' (default) or 'wrist'", False)
_GRIPPER = Param(
    "number", "-1 opens, +1 closes and holds; omitted keeps the last command", False
)

CODE_PRIMITIVES = (
    Primitive(
        "get_state",
        "env.get_state",
        "Proprioception, no images: eef_pos [x, y, z] (m, world), eef_quat_xyzw, yaw (rad about "
        "world +z), gripper_width (about 0.08 open, below 0.01 closed on nothing), gripper_cmd "
        "(-1 open / +1 close), terminated (LIBERO judged the task done; it stays true), truncated.",
    ),
    Primitive(
        "get_observation",
        "env.get_observation",
        "The current 512x512 upright camera images with calibration: agentview and wrist, each "
        "{rgb uint8[512,512,3], depth float32[512,512] metres (0 = none), intrinsic_K 3x3, "
        "extrinsic_cam2world 4x4}, plus the get_state fields. Pixel (row, col) back-projects as "
        "x = (col - cx) z / fx, y = (row - cy) z / fy in the camera frame, then "
        "extrinsic_cam2world @ [x, y, z, 1]. The agentview faces the robot (its base at the image "
        "top); the wrist camera looks down between the fingers.",
    ),
    Primitive(
        "back_project",
        "env.back_project",
        "World xyz {world_xyz: [x, y, z]} of pixel (row, col) of the current 512x512 image of a "
        "camera (row 0 = top), through its depth; an error when the pixel has no depth.",
        {
            "row": Param("integer", "pixel row"),
            "col": Param("integer", "pixel column"),
            "camera": _CAMERA,
        },
        tiers=("high",),
    ),
    Primitive(
        "move_to",
        "env.move_to",
        "Servo the end effector to a world position, holding its orientation; returns eef_pos, "
        "final_dist_m (large: the reach stalled), steps_used, gripper_width, terminated. Keep one "
        "call under 0.30 m in xy; split longer moves into waypoints at carry height.",
        {
            "xyz": Param("vec3", "target [x, y, z] in metres (world frame)"),
            "gripper": _GRIPPER,
            "tol": Param(
                "number", "stop within this distance, m (default 0.012)", False
            ),
            "max_steps": Param(
                "integer", "env-step budget (default 80; about 2.5 cm per step)", False
            ),
        },
        mutating=True,
        tiers=("high",),
    ),
    Primitive(
        "rotate_wrist",
        "env.rotate_wrist",
        "Turn the gripper about world z, holding its position (about 0.1 rad per step); returns "
        "yaw, final_err, steps_used, eef_pos. Give target_yaw or delta_yaw (positive = "
        "counter-clockwise seen from above).",
        {
            "target_yaw": Param("number", "absolute yaw, rad (world frame)", False),
            "delta_yaw": Param("number", "relative turn, rad", False),
            "gripper": _GRIPPER,
            "max_steps": Param("integer", "env-step budget (default 40)", False),
        },
        mutating=True,
        tiers=("high",),
    ),
    Primitive(
        "move_delta",
        "env.move_delta",
        "Move the end effector by a world-frame offset (+x away from the robot toward the "
        "agentview camera, +y robot-left, +z up), holding its orientation; at most 0.10 m per "
        "call. Returns eef_pos, moved_m (well below the command: blocked by contact, the table or "
        "a limit), steps_used, gripper_width, terminated.",
        {
            "dxyz": Param("vec3", "[dx, dy, dz] in metres"),
            "gripper": _GRIPPER,
            "max_steps": Param("integer", "env-step budget (default 25)", False),
        },
        mutating=True,
        tiers=("low",),
    ),
    Primitive(
        "rotate_delta",
        "env.rotate_delta",
        "Turn the gripper about world z by delta_yaw radians (positive = counter-clockwise seen "
        "from above; at most pi/2 per call), holding its position; returns yaw, final_err, "
        "steps_used, eef_pos.",
        {
            "delta_yaw": Param("number", "relative turn, rad"),
            "gripper": _GRIPPER,
            "max_steps": Param("integer", "env-step budget (default 40)", False),
        },
        mutating=True,
        tiers=("low",),
    ),
    Primitive(
        "set_gripper",
        "env.set_gripper",
        "Close (True; the following moves hold +1) or open (False) the gripper in place; the "
        "fingers travel about 1 mm per step and stop on a grasped object. Returns gripper_width "
        "(about 0.08 open, 0.01-0.05 holding, near 0 closed on nothing), steps_used, terminated.",
        {
            "close": Param("boolean", "True closes, False opens"),
            "steps": Param(
                "integer", "env steps to drive the fingers (default 15)", False
            ),
        },
        mutating=True,
    ),
)

SEGMENT = Primitive(
    "segment",
    "env.segment",
    "SAM3 segmentation of the current 512x512 image of a camera by a text prompt: {found, "
    "score, box [x1, y1, x2, y2], mask bool[512,512], n_pixels, centroid_rowcol [row, col], "
    "world_xyz (median over the mask's pixels with depth, or None)}.",
    {
        "prompt": Param("string", "what to segment, e.g. 'black bowl'"),
        "camera": _CAMERA,
        "min_score": Param("number", "SAM3 score threshold (default 0.2)", False),
    },
    tiers=("high",),
)


# ---- planned grasps (with a grasp server): plan_grasp / plan_place hand out ids that the
# first motion expires, so a planned grasp or place runs as one primitive from one resolution
# of its id (GraspPlanner.claim_waypoints) instead of a move per waypoint.

GRASP_EXECUTION = (
    Primitive(
        "execute_grasp",
        "env.execute_grasp",
        "Execute one planned grasp (a g id of the current observation) from a single resolution "
        "of the id: open to the pre-grasp standoff back along its approach, descend to it with "
        "its pitch and yaw, close, lift straight up. Returns legs (each leg's final_dist_m; the "
        "gripper width after closing), gripper_width (0.01-0.05 holding, near 0 missed), "
        "eef_pos, terminated; error and stalled when a leg stopped short (the robot moved: plan "
        "again). Afterwards plan_place(region, that grasp id) plans from the held object.",
        {
            "grasp_id": Param("string", "a g id from plan_grasp / next_grasp"),
            "standoff": Param("number", "pre-grasp distance, m (default 0.10)", False),
            "lift": Param("number", "lift after closing, m (default 0.10)", False),
            "max_steps": Param(
                "integer", "env-step budget per leg (default 150)", False
            ),
        },
        mutating=True,
        tiers=("high",),
    ),
    Primitive(
        "execute_place",
        "env.execute_place",
        "Execute one planned place (a p id of the current observation) from a single resolution "
        "of the id: carry closed to the pre-place standoff, descend to the place pose, open, "
        "retreat. Returns legs, gripper_width, eef_pos, terminated (placing may finish the task).",
        {
            "place_id": Param("string", "a p id from plan_place"),
            "standoff": Param("number", "pre-place distance, m (default 0.10)", False),
            "max_steps": Param(
                "integer", "env-step budget per leg (default 150)", False
            ),
        },
        mutating=True,
        tiers=("high",),
    ),
)


def libero_primitives(sam3: bool = False, grasp: bool = False) -> tuple[Primitive, ...]:
    """The server's registry: the raw surface, the code-mode primitives, `segment` with a SAM3
    server, and the planned-grasp executors with a grasp server."""
    return (
        LIBERO_PRIMITIVES
        + CODE_PRIMITIVES
        + ((SEGMENT,) if sam3 else ())
        + (GRASP_EXECUTION if grasp else ())
    )
