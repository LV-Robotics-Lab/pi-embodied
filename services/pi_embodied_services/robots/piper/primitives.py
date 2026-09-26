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

"""The Piper primitive registry (``code.api``, ../../components/code_api.py): the guarded motion
and state methods the piper tools use (a real robot: no privileged tier)."""

from __future__ import annotations

from pi_embodied_services.components.code_api import Param, Primitive

_ARM = Param("string", "'left' or 'right' (two arms only)", False)

PIPER_PRIMITIVES = (
    Primitive(
        "get_robot_state",
        "env.get_robot_state",
        "One arm's state (TCP pose, gripper width); both arms' without `arm` on two.",
        {"arm": _ARM},
    ),
    Primitive(
        "get_observation",
        "env.get_observation",
        "The cameras' frames and the robot state.",
        tiers=("low",),
    ),
    Primitive(
        "get_camera_meta",
        "env.get_camera_meta",
        "Camera calibration.",
        tiers=("low",),
    ),
    Primitive(
        "step",
        "env.step",
        "One guarded step of one arm: translate (m), yaw (rad), then open/close; refused beyond the step limits or the workspace.",
        {
            "delta_xyz": Param("vec3", "[dx, dy, dz] in m", False),
            "yaw": Param("number", "rad", False),
            "gripper": Param("string", "'open' or 'close'", False),
            "frame": Param("string", "'base' (default) or 'heading'", False),
            "reopen_empty": Param(
                "boolean", "reopen a gripper that closed on nothing", False
            ),
            "arm": _ARM,
            "continuous": Param(
                "boolean", "chain with the next step (smooth stream)", False
            ),
        },
        mutating=True,
    ),
    Primitive(
        "move_joints",
        "env.move_joints",
        "Joint-space move of one arm to a calibrated pose.",
        {"pose": Param("string", "'begin' (default) or 'rest'", False), "arm": _ARM},
        mutating=True,
    ),
    Primitive(
        "halt_arm",
        "env.halt_arm",
        "Stop one arm and refuse its motion until its reset.",
        {"arm": _ARM, "reason": Param("string", "why", False)},
        mutating=True,
    ),
)
