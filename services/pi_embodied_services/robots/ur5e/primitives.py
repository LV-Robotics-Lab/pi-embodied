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

"""The UR5e primitive registry (``code.api``, ../../components/code_api.py): the guarded motion
and state methods the ur5e tools use (a real robot: no privileged tier)."""

from __future__ import annotations

from pi_embodied_services.components.code_api import Param, Primitive

UR5E_PRIMITIVES = (
    Primitive(
        "get_robot_state",
        "env.get_robot_state",
        "The arm's state: TCP pose (xyz + xyzw and the UR rotation vector), joints, setpoint, gripper width (m).",
    ),
    Primitive(
        "get_observation",
        "env.get_observation",
        "Live frames per camera: images[name] RGB, depths[name] metres for cameras with depth.",
        tiers=("low",),
    ),
    Primitive(
        "get_camera_meta",
        "env.get_camera_meta",
        "Per camera: intrinsics, depth availability, mount and hand-eye extrinsic (camera -> tcp or base).",
        tiers=("low",),
    ),
    Primitive(
        "move_delta",
        "env.move_delta",
        "Translate the TCP by a base-frame delta in metres, orientation held; refused beyond the per-call limit or outside the workspace.",
        {"delta_xyz": Param("vec3", "base-frame [dx, dy, dz] in m")},
        mutating=True,
    ),
    Primitive(
        "move_pose",
        "env.move_pose",
        "Move the TCP to an absolute base-frame pose within the per-call limits; rpy is converted to a rotation vector.",
        {
            "xyz": Param("vec3", "base-frame [x, y, z] in m"),
            "rotvec": Param("vec3", "axis-angle rotation vector (rad)", False),
            "rpy": Param("vec3", "extrinsic xyz Euler angles (rad)", False),
        },
        mutating=True,
    ),
    Primitive(
        "rotate_delta",
        "env.rotate_delta",
        "Rotate the TCP by a base-frame roll/pitch/yaw delta in radians; refused beyond the per-call limit or the tilt limit.",
        {"delta_rpy": Param("vec3", "base-frame [droll, dpitch, dyaw] in rad")},
        mutating=True,
    ),
    Primitive(
        "set_gripper",
        "env.set_gripper",
        "Open (open=True) or close the Robotiq gripper and wait for the fingers to settle; reports grasp_empty and gripper_jammed.",
        {"open": Param("boolean", "True opens, False closes")},
        mutating=True,
    ),
)
