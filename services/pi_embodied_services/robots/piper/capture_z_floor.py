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
# Modified by pi-embodied: scripts/piper/capture_z_floor.py reads the arm through a
# RUNNING Piper env server (env_server.py, ``env.get_robot_state``) instead of its own
# ROS interface, and writes our config layout (calibration.z_floor_m, or on two arms
# arms.<side>.calibration.z_floor_m, as config/example.yaml and dual_example.yaml).

"""Capture one Piper arm's Z safety floor from a running env server. Read-only: nothing moves.

Rest that arm's closed gripper on the tabletop (hand-guide or jog it), then::

    python -m pi_embodied_services.robots.piper.capture_z_floor --env http://127.0.0.1:18120 [--arm left]
    python -m ... --env ... --arm left --write services/.../robots/piper/config/my_rig.yaml

The value is the arm's measured ``eef_pos[2]`` in its own base frame (the controller's
floor check compares the same). The env server refuses to start while z_floor_m is null, so a
first capture can run it with ``limits.enable_z_floor: false``; re-enable it after writing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from pi_embodied_services.utils.yaml_edit import write_value


def z_of(state: dict, arm: str | None) -> float:
    if "arms" in state:
        if arm is None:
            raise ValueError(f"two arms ({', '.join(state['arms'])}): pass --arm")
        state = state["arms"][arm]
    pos = state.get("eef_pos")
    if not isinstance(pos, (list, tuple)) or len(pos) != 3:
        raise ValueError("get_robot_state has no eef_pos")
    return round(float(pos[2]), 5)


def config_keys(path: str, arm: str | None) -> list[str]:
    """Where z_floor_m lives: arms.<side>.calibration on a two-arm config, else calibration."""
    cfg = (
        yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if Path(path).exists()
        else {}
    )
    if isinstance(cfg.get("arms"), dict):
        if arm not in cfg["arms"]:
            raise ValueError(f"{path} has arms {sorted(cfg['arms'])}; pass --arm")
        return ["arms", arm, "calibration", "z_floor_m"]
    return ["calibration", "z_floor_m"]


def main(argv: list[str] | None = None, client=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--env",
        required=True,
        help="running Piper env server, e.g. http://127.0.0.1:18120",
    )
    p.add_argument("--arm", choices=["left", "right"], help="which arm (two-arm rigs)")
    p.add_argument(
        "--write",
        metavar="YAML",
        help="store it as that arm's z_floor_m in this robot config",
    )
    args = p.parse_args(argv)
    if client is None:
        from pi_embodied_services.utils.rpc.client_utils import make_rpc_client

        client = make_rpc_client(args.env)
    try:
        state = client.call(
            "env.get_robot_state",
            kwargs={"arm": args.arm} if args.arm else {},
            timeout_s=30.0,
        )
        z = z_of(state, args.arm)
        out: dict = {"arm": args.arm, "z_floor_m": z}
        if args.write:
            keys = config_keys(args.write, args.arm)
            write_value(args.write, keys, z)
            out["written"] = f"{args.write}: {'.'.join(keys)}"
        print(json.dumps(out))
        return 0
    except Exception as exc:  # noqa: BLE001 - one clear line for the operator
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
