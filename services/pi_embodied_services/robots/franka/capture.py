# Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
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
#
# Modified by pi-embodied: scripts/franka/capture_z_floor.py and capture_pose.py merged
# into one read-only client of a RUNNING Franka env server (env_server.py,
# ``env.get_robot_state``: RLinf's raw_base_state) instead of the Polymetis NUC
# interface; the floor goes to the robot's --z-floor flag (a named-floors YAML is
# optional), the pose to the robot config's workspace.reset_ee_pose.

"""Capture Franka calibration values from a running env server. Read-only: the arm never moves.

    # Z floor: rest the closed gripper ON the work surface, then
    python -m pi_embodied_services.robots.franka.capture --env http://127.0.0.1:18100 z-floor
        -> prints the TCP z (the value of pi's --z-floor for this robot) and the flag to pass
    python -m ... z-floor --name drawer --write floors.yaml    # also keep it as z_floors.drawer

    # Start pose: hand-guide / jog the arm to the pose, then
    python -m ... pose                                         # prints the TCP pose and joints
    python -m ... pose --write robots/franka/config/my_rig.yaml  # sets workspace.reset_ee_pose

The z is ``raw_base_state.tcp_pose[2]``, the value pi's franka robot compares --z-floor with
(src/franka/index.ts checkWorkspace). ``reset_ee_pose`` is RLinf's reset target: xyz plus
extrinsic xyz euler angles (rad) of the TCP quaternion.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from pi_embodied_services.utils.yaml_edit import write_value


def raw_state(client) -> dict[str, Any]:
    state = client.call("env.get_robot_state", timeout_s=30.0)
    raw = (state or {}).get("raw_base_state")
    if not isinstance(raw, dict):
        raise ValueError("get_robot_state has no raw_base_state")
    return raw


def tcp_pose(raw: dict[str, Any]) -> np.ndarray:
    pose = np.asarray(raw.get("tcp_pose", []), dtype=float).reshape(-1)
    if pose.size != 7:
        raise ValueError(
            f"raw_base_state.tcp_pose must be xyz + quaternion xyzw (7 values), got {pose.size}"
        )
    return pose


def reset_pose(pose: np.ndarray) -> list[float]:
    """xyz + extrinsic xyz euler (RLinf's reset_ee_pose) from xyz + quaternion xyzw."""
    euler = Rotation.from_quat(pose[3:7]).as_euler("xyz")
    return [round(float(v), 6) for v in (*pose[:3], *euler)]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--env",
        required=True,
        help="running Franka env server, e.g. http://127.0.0.1:18100",
    )
    sub = p.add_subparsers(dest="what", required=True)
    z = sub.add_parser(
        "z-floor", help="the TCP height with the gripper resting on the surface"
    )
    z.add_argument(
        "--name", default="default", help="the setting's name in --write's z_floors map"
    )
    z.add_argument(
        "--write", metavar="YAML", help="also store it as z_floors.<name> in this file"
    )
    pose = sub.add_parser(
        "pose", help="the current TCP pose (and joints) as the reset pose"
    )
    pose.add_argument(
        "--write",
        metavar="YAML",
        help="set workspace.reset_ee_pose in this robot config",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None, client=None) -> int:
    args = parse_args(argv)
    if client is None:
        from pi_embodied_services.utils.rpc.client_utils import make_rpc_client

        client = make_rpc_client(args.env)
    try:
        raw = raw_state(client)
        pose = tcp_pose(raw)
        if args.what == "z-floor":
            z = round(float(pose[2]), 5)
            out: dict[str, Any] = {"z_floor": z, "flag": f"--z-floor={z}"}
            if args.write:
                write_value(args.write, ["z_floors", args.name], z)
                out["written"] = f"{args.write}: z_floors.{args.name}"
            print(json.dumps(out))
            print(
                "[capture] downward moves below this are refused; the gripper must have rested on the surface.",
                file=sys.stderr,
            )
            return 0
        joints = np.asarray(raw.get("arm_joint_position", []), dtype=float).reshape(-1)
        out = {
            "tcp_pose_xyzw": [round(float(v), 6) for v in pose],
            "reset_ee_pose": reset_pose(pose),
            "joints": [round(float(v), 6) for v in joints],
        }
        if args.write:
            write_value(
                args.write, ["workspace", "reset_ee_pose"], out["reset_ee_pose"]
            )
            out["written"] = f"{args.write}: workspace.reset_ee_pose"
        print(json.dumps(out))
        return 0
    except Exception as exc:  # noqa: BLE001 - one clear line for the operator
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
