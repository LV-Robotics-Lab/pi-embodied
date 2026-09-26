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

"""The Franka primitive registry (``code.api``), shared by both single-arm backends."""

from __future__ import annotations

from typing import Any

from pi_embodied_services.components.code_api import Param, Primitive

#: The single-arm primitives (``code.api``, ../../components/code_api.py), shared by the RLinf and
#: Polymetis backends, which serve the same env.* methods: the motion and state primitives the franka
#: tools use, each running through the facade's own limit checks.
FRANKA_PRIMITIVES = (
    Primitive(
        "get_robot_state",
        "env.get_robot_state",
        "The arm's state: TCP pose, joints and gripper width (m).",
        example='st = get_robot_state()["raw_base_state"]\nprint(st["tcp_pose"][:3], st["gripper_position"])',
    ),
    Primitive(
        "get_observation",
        "env.get_observation",
        "Live RGB-D frames: main_images/main_depths (wrist) and extra_view_* (the other cameras).",
        tiers=("low",),
        example='obs = get_observation()\nwrist, depth = obs["main_images"], obs["main_depths"]  # HxWx3 uint8, HxW m\nscene = obs["extra_view_images"][0]',
    ),
    Primitive(
        "get_camera_meta",
        "env.get_camera_meta",
        "Camera intrinsics, extrinsics and depth conventions.",
        tiers=("low",),
        example='meta = get_camera_meta()\nfor name, cam in meta["cameras"].items():\n    print(name, cam["intrinsic_K"])',
    ),
    Primitive(
        "preview_reach",
        "env.preview_reach",
        "IK reach check of a base-frame TCP pose from the current joints, nothing moves (--ik): {status: reachable | unreachable | unknown, q, position_err, orientation_err, message}.",
        {
            "pos": Param("vec3", "base-frame [x, y, z] in m"),
            "quat_xyzw": Param(
                "array", "orientation (default: the current one)", False
            ),
        },
    ),
    Primitive(
        "move_delta",
        "env.move_delta",
        "Translate the TCP by a base-frame delta in metres; refused beyond the per-call limit or outside the workspace.",
        {"delta_xyz": Param("vec3", "base-frame [dx, dy, dz] in m")},
        mutating=True,
        example='r = move_delta([0, 0, -0.03])  # 3 cm down; each call stays within the per-call limit\nprint(r["ok"], r["final_tcp_pose"][:3])',
    ),
    Primitive(
        "rotate_delta",
        "env.rotate_delta",
        "Rotate the TCP by a base-frame roll/pitch/yaw delta in radians; refused beyond the per-call limit.",
        {"delta_rpy": Param("vec3", "base-frame [droll, dpitch, dyaw] in rad")},
        mutating=True,
        example="rotate_delta([0, 0, 0.2])  # yaw 0.2 rad",
    ),
    Primitive(
        "set_gripper",
        "env.set_gripper",
        "Open (open=True) or close the gripper and wait for it to settle.",
        {"open": Param("boolean", "True opens, False closes")},
        mutating=True,
        example='r = set_gripper(False)  # close\nprint(r["ok"])',
    ),
)

_CAMERA = Param("string", "'wrist' (default) or 'third_person'", required=False)
_ID = Param(
    "string", "a detection id from segment (e.g. 'd3') of the current observation"
)

#: Served only when the env server was started with --sam3 (utils/perception.py).
SEGMENT_PRIMITIVES = (
    Primitive(
        "segment",
        "env.segment",
        "SAM3 masks on the current observation, each with a short id; all=True returns every "
        "candidate instead of the best. Ids die with the next observation.",
        {
            "camera": _CAMERA,
            "text_prompt": Param("string", "what to segment", required=False),
            "point": Param("array", "[row, col] positive point", required=False),
            "min_score": Param("number", "default 0.2", required=False),
            "all": Param("boolean", "every mask, not only the best", required=False),
        },
    ),
    Primitive(
        "select_detection",
        "env.select_detection",
        "Make one detection id the selected mask of the current observation.",
        {"id": _ID},
    ),
    Primitive(
        "reject_detection",
        "env.reject_detection",
        "Rule one detection id out; it stays listed as rejected.",
        {"id": _ID},
    ),
)

#: Served only when the env server was started with --unidepth.
ENHANCE_DEPTH_PRIMITIVES = (
    Primitive(
        "enhance_depth",
        "env.enhance_depth",
        "Fill the holes of a camera's depth with a UniDepth estimate scaled to the sensor "
        "(or supply depth where the camera has none); later segments use the filled depth.",
        {"camera": _CAMERA},
    ),
)


def franka_primitives(perception: Any | None) -> tuple[Primitive, ...]:
    """The single-arm registry plus the perception primitives the server actually serves."""
    caps = perception.capabilities() if perception is not None else {}
    return (
        FRANKA_PRIMITIVES
        + (SEGMENT_PRIMITIVES if caps.get("segment") else ())
        + (ENHANCE_DEPTH_PRIMITIVES if caps.get("enhance_depth") else ())
    )


_ARM = Param("string", "'left' or 'right'")

#: The dual-arm primitives (../dual_franka): the same methods with an ``arm``, plus the posture recovery.
DUAL_FRANKA_PRIMITIVES = (
    Primitive(
        "get_robot_state",
        "env.get_robot_state",
        "Both arms' state: TCP poses, joints and gripper widths.",
        example='st = get_robot_state()\nprint(st["left_arm"]["tcp_pose"][:3], st["right_arm"]["tcp_pose"][:3])',
    ),
    Primitive(
        "get_observation",
        "env.get_observation",
        "Live RGB-D frames of the rig's cameras.",
        tiers=("low",),
        example='obs = get_observation()\nprint(sorted(k for k in obs if k.endswith("_images")))',
    ),
    Primitive(
        "get_camera_meta",
        "env.get_camera_meta",
        "Camera intrinsics, serials and projection metadata.",
        tiers=("low",),
        example='meta = get_camera_meta()\nprint(meta["observation_camera_map"])',
    ),
    Primitive(
        "move_delta",
        "env.move_delta",
        "Translate one arm's TCP by a world-frame delta in metres; refused beyond the per-call limit.",
        {"arm": _ARM, "delta_xyz": Param("vec3", "world-frame [dx, dy, dz] in m")},
        mutating=True,
        example='move_delta("left", [0, 0, 0.03])  # the left TCP 3 cm up (right_base frame)',
    ),
    Primitive(
        "rotate_delta",
        "env.rotate_delta",
        "Rotate one arm's TCP by a world-frame roll/pitch/yaw delta in radians; refused beyond the per-call limit.",
        {
            "arm": _ARM,
            "delta_rpy": Param("vec3", "world-frame [droll, dpitch, dyaw] in rad"),
        },
        mutating=True,
        example='rotate_delta("right", [0, 0, -0.2])',
    ),
    Primitive(
        "set_gripper",
        "env.set_gripper",
        "Open (open=True) or close one arm's gripper and wait for it to settle.",
        {"arm": _ARM, "open": Param("boolean", "True opens, False closes")},
        mutating=True,
        example='set_gripper("right", True)  # open',
    ),
    Primitive(
        "recover_joint_posture",
        "env.recover_joint_posture",
        "Reset both arms to the configured joint posture, keeping each gripper open or closed.",
        {
            "reason": Param("string", "why", required=False),
            "return_to_start": Param(
                "boolean", "return the TCPs near their prior poses", required=False
            ),
        },
        mutating=True,
    ),
)
