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

"""UR5e data rules for its offline session data. Not a Flywheel robot: nothing records it for
the Flywheel, and the export CLI does not know it (``flywheel/specs.py`` ROBOTS).

What a UR5e session already persists (packages/embodied/src/ur5e, ``--out``): one state step
per mutating tool call, step 0 the operator-confirmed reset. ``states.jsonl`` holds a line per
step with ``state.raw_base_state`` (the controller's ``state()``: ``tcp_pose`` ``[x, y, z, qx,
qy, qz, qw]`` in the base frame, ``gripper_position`` ``[width_m]``, ``gripper_commanded_open``,
...), ``command`` (the tool and its parameters: ``move_delta``, ``move_pose``, ``rotate_delta``,
``gripper``, recorded as its ``open`` / ``close`` parameter, units mode's ``act``) and
``result``; ``step_NNNN/<camera>.rgb`` (+ ``.json`` with its
size) is each camera's frame after the call.

``SPEC`` describes that data as one Flywheel transition per tool call: the observation after
it (``state_of``) and the commanded motion (``action_of``). A converter from a ``--out``
directory to raw episodes is not written, because what it needs is not in ``--out``: the
episode's success (the operator's verdict, in the pi session's ``robot_result``), its task
language (the robot config's task instruction, in the session), and the camera names (the
robot config's ``cameras.devices``; ``SPEC`` names example.yaml's ``wrist`` and ``front``).
Tool calls are not control steps either: ``fps`` is nominal, one step per call."""

from __future__ import annotations

from typing import Any

import numpy as np

#: Camera sizes are the config's devices': taken from the episodes.
_IMAGE = {"shape": None, "dtype": "uint8"}
#: The motion tools as ``command.action`` names them; the gripper tool's own ``action``
#: parameter ("open" / "close") overwrites its name there.
_ARM = ("move_delta", "move_pose", "rotate_delta", "act", "gripper", "open", "close")


def success_mask(transitions: Any) -> Any:
    """The operator's verdict, set on the last step of an episode judged a success."""
    return transitions["terminated"]


SPEC = {
    "robot": "ur5e",
    "robot_type": "ur5e_robotiq",
    "fps": 1,
    "arrays": {
        "wrist_images": _IMAGE,
        "front_images": _IMAGE,
        "states": {"shape": (8,), "dtype": "float32"},
        "actions": {"shape": (7,), "dtype": "float32"},
    },
    "image_fields": ("wrist_images", "front_images"),
    "cameras": {"wrist_images": "wrist", "front_images": "front"},
    "state_names": [
        *("tcp_x", "tcp_y", "tcp_z", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"),
        "gripper_width",
    ],
    "action_names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
    "success_mask": success_mask,
    #: What one dataset would share: one arm and task.
    "group": ("arm_id", "task"),
}


def state_of(blob: dict[str, Any]) -> np.ndarray:
    """A states.jsonl step's state: the TCP pose (base frame, quaternion xyzw) and the gripper
    width (m)."""
    base = blob["state"]["raw_base_state"]
    width = base.get("gripper_position") or []
    if not width:
        raise ValueError(f"step {blob.get('step_idx')} has no gripper width")
    return np.asarray([*base["tcp_pose"], width[0]], dtype=np.float32)


def action_of(blob: dict[str, Any], before: dict[str, Any]) -> np.ndarray | None:
    """The motion step ``blob``'s tool call commanded from the step ``before`` it:
    ``[dx, dy, dz]`` (m) and ``[drx, dry, drz]`` (a rotation vector, rad) in the base frame, and
    the gripper command in force after it (+1 close, -1 open). None for a step that is no
    transition: the reset (an episode boundary), a refused or failed call, another tool."""
    from scipy.spatial.transform import Rotation

    command = blob.get("command") or {}
    tool = command.get("action")
    if tool not in _ARM or (blob.get("result") or {}).get("error"):
        return None
    xyz, rot = np.zeros(3), np.zeros(3)
    if tool == "move_delta":
        xyz = np.asarray(command["delta_xyz"], dtype=np.float64)
    elif tool == "move_pose":
        pose = before["state"]["raw_base_state"]["tcp_pose"]
        xyz = np.asarray(command["xyz"], dtype=np.float64) - pose[:3]
        target = None
        if command.get("rotvec") is not None:
            target = Rotation.from_rotvec(command["rotvec"])
        elif command.get("rpy") is not None:
            target = Rotation.from_euler("xyz", command["rpy"])
        if target is not None:
            rot = (target * Rotation.from_quat(pose[3:]).inv()).as_rotvec()
    elif tool == "rotate_delta":
        rot = Rotation.from_euler("xyz", command["delta_rpy"]).as_rotvec()
    elif tool == "act":
        move = command["move"]
        xyz = np.asarray(move["delta"], dtype=np.float64)
        rot = np.array([0.0, 0.0, float(move.get("yaw") or 0.0)])
    after = blob["state"]["raw_base_state"]
    # Before any gripper command, the fingers' own state.
    opened = after.get("gripper_commanded_open")
    if opened is None:
        opened = after.get("gripper_open")
    return np.asarray([*xyz, *rot, -1.0 if opened else 1.0], dtype=np.float32)
