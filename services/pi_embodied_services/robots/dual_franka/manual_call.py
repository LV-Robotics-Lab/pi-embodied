# Copyright 2026 The RPent Authors.
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
# Modified by pi-embodied: rewritten from robots/dual_franka/dual_franka_manual_call.py
# (github.com/RLinf/RPent @eecf206) as a thin client of a RUNNING dual-Franka env
# server (env_server.py): it no longer builds the RPent toolkit, the planner-side
# primitives, the VLA or SAM3 clients; motion calls are dry-run by default and are
# checked with the limits pi-embodied's dual_franka tools apply (src/robot.ts
# checkMove / workspaceLimits, src/primitives/motion.ts checkRotate).

"""Call one dual-Franka env-server facade method by hand, for hardware bring-up.

Connects to a running dual-Franka env server (``env_server.py``, e.g. the one pi's
``--robot-env`` points at) and calls one of its facade methods, bypassing the planner::

    python -m pi_embodied_services.robots.dual_franka.manual_call --env http://127.0.0.1:18100 \\
        --z-floor 0.14 move_delta --arm right --delta 0 0 0.02            # dry run: prints the checked call
    python -m ... --env ... --z-floor 0.14 --execute move_delta --arm right --delta 0 0 0.02
    python -m ... --env ... get_robot_state                               # read-only calls always run

Motion calls (move_delta, rotate_delta, set_gripper, recover_joint_posture, reset) print the
validated request and send nothing unless ``--execute`` is given. The same limits as the dual_franka
pi tools apply before anything is sent: a move's norm is at most ``--max-move`` (m, default 0.1), a
rotation's at most ``--max-rotate`` (rad, default 0.5), and a move may not end outside the
``--workspace-xy`` box (right_base frame) or below ``--z-floor`` unless it moves back towards it
(the arm's current ``tcp_pose`` is read with ``get_robot_state``). ``--z-floor`` is required for
move_delta, as it is for the pi robot. ``reset`` is manual-only (no pi tool exposes a full reset).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

import numpy as np

READ_ONLY = ("get_env_meta", "get_robot_state", "get_camera_meta")
MOTION = ("move_delta", "rotate_delta", "set_gripper", "recover_joint_posture", "reset")
# Timeouts the pi tools use for the same calls (src/dual_franka/index.ts).
TIMEOUT_S = {"recover_joint_posture": 240.0, "reset": 180.0}
DEFAULT_WORKSPACE_XY = "0.1,1.15,-0.85,0.85"


def workspace_limits(xy: str, z_floor: str | None) -> tuple[list[float] | None, float]:
    """src/robot.ts workspaceLimits: the box (off when empty) and the required floor."""
    box = None
    if xy.strip():
        try:
            box = [float(v) for v in xy.split(",")]
        except ValueError:
            box = []
        if (
            len(box) != 4
            or not all(math.isfinite(v) for v in box)
            or not (box[0] < box[1] and box[2] < box[3])
        ):
            raise ValueError(
                f'--workspace-xy must be "xmin,xmax,ymin,ymax" (finite, min < max), got "{xy}"'
            )
    try:
        floor = float(z_floor) if z_floor not in (None, "") else math.nan
    except ValueError:
        floor = math.nan
    if not math.isfinite(floor):
        raise ValueError(
            f'--z-floor must be the lowest safe TCP z in m (e.g. 0.14), got "{z_floor or ""}"'
        )
    return box, floor


def check_move(delta: list[float], cap: float) -> None:
    norm = math.hypot(*delta)
    if not norm <= cap:
        raise ValueError(
            f"delta_xyz moves {norm:.4f} m; the limit is {cap} m per call. Split the motion."
        )


def check_rotate(rpy: list[float], cap: float) -> None:
    norm = math.hypot(*rpy)
    if not norm <= cap:
        raise ValueError(
            f"delta_rpy rotates {norm:.4f} rad; the limit is {cap} rad per call. Split the rotation."
        )


def check_workspace(
    tcp: list[float], delta: list[float], box: list[float] | None, floor: float
) -> None:
    """src/dual_franka/index.ts checkWorkspace: refuse a target outside, unless it moves back in."""

    def outside(p: list[float]) -> float:
        out = max(0.0, floor - p[2])
        if box:
            out += max(0.0, box[0] - p[0], p[0] - box[1]) + max(
                0.0, box[2] - p[1], p[1] - box[3]
            )
        return out

    target = [t + d for t, d in zip(tcp, delta)]
    if outside(target) > 1e-6 and outside(target) >= outside(tcp) - 1e-6:
        where = f"x {box[0]}..{box[1]}, y {box[2]}..{box[3]}, " if box else ""
        raise ValueError(
            f"the move ends at {[round(v, 3) for v in target]}, outside the right_base workspace ({where}z >= {floor} m)"
        )


def tcp_xyz(state: dict[str, Any], arm: str) -> list[float]:
    pose = (state.get(f"{arm}_arm") or {}).get("tcp_pose")
    xyz = np.asarray(pose if pose is not None else [], dtype=float).reshape(-1)[:3]
    if xyz.size != 3:
        raise ValueError(f"the {arm} arm's tcp_pose is missing from get_robot_state")
    return xyz.tolist()


def build_call(args: argparse.Namespace, read_state) -> tuple[str, dict[str, Any]]:
    """The checked (method, kwargs); raises ValueError with the reason. `read_state()` -> get_robot_state."""
    m = args.method
    if m in READ_ONLY or m == "reset":
        return m, {}
    if m == "move_delta":
        delta = [float(v) for v in args.delta]
        check_move(delta, args.max_move)
        box, floor = workspace_limits(args.workspace_xy, args.z_floor)
        check_workspace(tcp_xyz(read_state(), args.arm), delta, box, floor)
        return m, {"arm": args.arm, "delta_xyz": delta}
    if m == "rotate_delta":
        rpy = [float(v) for v in args.rpy]
        check_rotate(rpy, args.max_rotate)
        return m, {"arm": args.arm, "delta_rpy": rpy}
    if m == "set_gripper":
        return m, {"arm": args.arm, "open": args.state == "open"}
    if m == "recover_joint_posture":
        return m, {"reason": args.reason, "return_to_start": not args.stay}
    raise ValueError(f"unknown method {m}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--env",
        required=True,
        help="running dual-Franka env server, e.g. http://127.0.0.1:18100",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="really send a motion call (default: dry run)",
    )
    p.add_argument(
        "--max-move",
        type=float,
        default=0.1,
        help="largest move_delta norm per call, m",
    )
    p.add_argument(
        "--max-rotate",
        type=float,
        default=0.5,
        help="largest rotate_delta norm per call, rad",
    )
    p.add_argument(
        "--workspace-xy",
        default=DEFAULT_WORKSPACE_XY,
        help='right_base "xmin,xmax,ymin,ymax" ("" = off)',
    )
    p.add_argument(
        "--z-floor", default=None, help="lowest safe TCP z, m (required for move_delta)"
    )
    p.add_argument("--timeout", type=float, default=120.0, help="RPC timeout, s")
    sub = p.add_subparsers(dest="method", required=True)
    for name in READ_ONLY:
        sub.add_parser(name, help="read-only")
    arms = {"choices": ["left", "right"], "required": True}
    mv = sub.add_parser(
        "move_delta", help="translate one arm's TCP by DX DY DZ m (right_base)"
    )
    mv.add_argument("--arm", **arms)
    mv.add_argument(
        "--delta", nargs=3, type=float, required=True, metavar=("DX", "DY", "DZ")
    )
    rt = sub.add_parser(
        "rotate_delta", help="rotate one arm's TCP by roll pitch yaw rad"
    )
    rt.add_argument("--arm", **arms)
    rt.add_argument(
        "--rpy", nargs=3, type=float, required=True, metavar=("R", "P", "Y")
    )
    gr = sub.add_parser("set_gripper", help="open or close one gripper")
    gr.add_argument("--arm", **arms)
    gr.add_argument("state", choices=["open", "close"])
    rc = sub.add_parser(
        "recover_joint_posture", help="reset both arms' joints, keeping the grippers"
    )
    rc.add_argument("--reason", default="manual")
    rc.add_argument(
        "--stay",
        action="store_true",
        help="do not return the TCPs to their prior poses",
    )
    sub.add_parser(
        "reset", help="MANUAL ONLY: full env reset to the configured initial posture"
    )
    return p.parse_args(argv)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return (
            value.tolist()
            if value.size <= 64
            else f"<ndarray {value.dtype} {list(value.shape)}>"
        )
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    return value


def main(argv: list[str] | None = None, client=None) -> int:
    args = parse_args(argv)
    if client is None:
        from pi_embodied_services.utils.rpc.client_utils import make_rpc_client

        client = make_rpc_client(args.env)

    def call(method: str, kwargs: dict[str, Any]) -> Any:
        return client.call(
            f"env.{method}",
            kwargs=kwargs,
            timeout_s=TIMEOUT_S.get(method, args.timeout),
        )

    try:
        method, kwargs = build_call(args, lambda: call("get_robot_state", {}))
    except ValueError as exc:
        print(json.dumps({"ok": False, "method": args.method, "refused": str(exc)}))
        return 2
    if method in MOTION and not args.execute:
        print(
            json.dumps(
                {
                    "ok": True,
                    "dry_run": True,
                    "method": f"env.{method}",
                    "kwargs": kwargs,
                }
            )
        )
        return 0
    sent = {
        k: np.asarray(v, dtype=np.float32) if k.startswith("delta_") else v
        for k, v in kwargs.items()
    }
    result = call(method, sent)
    print(
        json.dumps(
            {
                "ok": True,
                "method": f"env.{method}",
                "kwargs": kwargs,
                "result": _plain(result),
            },
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
