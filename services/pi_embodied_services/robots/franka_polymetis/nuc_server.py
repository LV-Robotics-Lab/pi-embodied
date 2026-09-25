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

"""Reference ZeroRPC ``franka_server`` for the Franka NUC (runs ON the NUC).

Show-Harness talks to a Polymetis-based ``franka_server`` on port 4242 that its
repository does not contain; this is a minimal implementation of the method surface
its ``FrankaInterface`` (and this package's ``hardware.PolymetisRobot``) calls. It was
written against the Polymetis ``RobotInterface`` / ``GripperInterface`` API and has
not been run against hardware from this repository: bring it up with the arm in a
safe pose and the E-stop in hand, and check each method once (``--read-only`` serves
only the getters).

Run it in the Polymetis conda env on the NUC, after its robot and gripper servers::

    launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=<FCI IP>
    launch_gripper.py gripper=franka_hand
    pip install zerorpc
    python nuc_server.py --port 4242

Poses are [x, y, z, qx, qy, qz, qw] of Polymetis's end-effector link in the robot base
frame (metres); ``control_gripper(True)`` grasps, ``False`` opens. Gripper commands do
not block, so the client can poll the width (and ``stop`` between polls).

Every command is validated here too, independent of the workstation: finite values,
at most ``--max-step-m`` / ``--max-step-rad`` from the last setpoint or the measured
pose, gains at most 2x the Show-Harness defaults, joints inside the Franka limits,
``time_to_go`` long enough for 0.5 rad/s, and the optional ``--workspace-min/max`` /
``--z-floor`` (in the reported end-effector frame).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

# Keep in sync with control.py (this file runs on the NUC without the package).
MAX_KX = (1500.0, 1500.0, 1500.0, 30.0, 30.0, 30.0)
MAX_KXD = (74.0, 74.0, 74.0, 4.0, 4.0, 4.0)
JOINT_MIN = (-2.7437, -1.7628, -2.8973, -3.0421, -2.8065, 0.5445, -2.8973)
JOINT_MAX = (2.7437, 1.7628, 2.8973, -0.1518, 2.8065, 3.7525, 2.8973)
MAX_JOINT_SPEED_RAD_S = 0.5
PEAK_TO_MEAN = 1.875


@dataclass
class NucLimits:
    """Per-request guards, independent of the workstation's limits.

    Poses are in the frame this server reports (Polymetis's end-effector link), not
    the workstation's TCP. The box and floor are optional (``None`` skips them).
    """

    max_step_m: float = 0.01
    max_step_rad: float = 0.05
    max_joint_step_rad: float = 0.05
    min_time_to_go_s: float = 0.5
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None
    z_floor_m: float | None = None


def _finite(name: str, values, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape != (n,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be {n} finite values, got {values!r}")
    return arr


def _quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(a, b)))))


class FrankaServer:
    """The ``franka_server`` methods; every command is checked before it reaches
    Polymetis (finite values, bounded setpoint jump, gains, joint limits and speed,
    optional box / floor). ``tensor`` converts a list for Polymetis (torch)."""

    def __init__(self, robot, gripper, speed, force, limits: NucLimits, tensor):
        self.robot = robot
        self.gripper = gripper
        self.speed, self.force = float(speed), float(force)
        self.limits = limits
        self._t = tensor
        self.max_width = float(getattr(self.gripper.metadata, "max_width", 0.08))
        self._last_pose: np.ndarray | None = None  # last accepted ee setpoint
        self._last_joints: np.ndarray | None = None  # last accepted joint setpoint

    # -- guards --------------------------------------------------------------

    def _outside(self, p: np.ndarray) -> np.ndarray:
        lim = self.limits
        lo = np.full(3, -np.inf)
        hi = np.full(3, np.inf)
        if lim.workspace_min is not None:
            lo = np.asarray(lim.workspace_min, dtype=np.float64).copy()
        if lim.workspace_max is not None:
            hi = np.asarray(lim.workspace_max, dtype=np.float64)
        if lim.z_floor_m is not None:
            lo[2] = max(lo[2], float(lim.z_floor_m))
        return np.maximum(0.0, lo - p) + np.maximum(0.0, p - hi)

    def _check_pose(self, pose) -> np.ndarray:
        lim = self.limits
        p = _finite("pose", pose, 7)
        n = float(np.linalg.norm(p[3:]))
        if abs(n - 1.0) > 1e-3:
            raise ValueError(f"pose quaternion is not unit length ({n:.6f})")
        p[3:] /= n
        measured = np.asarray(self.get_ee_pose(), dtype=np.float64)

        def near(ref: np.ndarray | None) -> bool:
            return (
                ref is not None
                and float(np.linalg.norm(p[:3] - ref[:3])) <= lim.max_step_m
                and _quat_angle(p[3:], ref[3:]) <= lim.max_step_rad
            )

        # Either a small step from the last setpoint or a hold near the measured pose.
        if not (near(self._last_pose) or near(measured)):
            raise ValueError(
                f"update_desired_ee_pose jumps more than {lim.max_step_m} m / "
                f"{lim.max_step_rad} rad from both the last setpoint and the measured "
                "pose; refused"
            )
        if np.any(self._outside(p[:3]) > self._outside(measured[:3]) + 1e-9):
            raise ValueError(
                f"update_desired_ee_pose {np.round(p[:3], 4).tolist()} is outside the "
                "NUC workspace box / floor; refused"
            )
        return p

    def _check_joints(self, name: str, positions) -> np.ndarray:
        q = _finite(name, positions, 7)
        if np.any(q <= JOINT_MIN) or np.any(q >= JOINT_MAX):
            raise ValueError(f"{name} {q.tolist()} is outside the joint limits")
        return q

    # -- state ---------------------------------------------------------------

    def get_ee_pose(self):
        pos, quat = self.robot.get_ee_pose()
        return pos.tolist() + quat.tolist()

    def get_joint_positions(self):
        return self.robot.get_joint_positions().tolist()

    def get_joint_velocities(self):
        return self.robot.get_joint_velocities().tolist()

    def get_gripper_position(self):
        return float(self.gripper.get_state().width)

    # -- control -------------------------------------------------------------

    def move_to_joint_positions(self, positions, time_to_go):
        q = self._check_joints("positions", positions)
        t = float(time_to_go)
        q0 = _finite("joint positions", self.get_joint_positions(), 7)
        need = max(
            self.limits.min_time_to_go_s,
            PEAK_TO_MEAN * float(np.max(np.abs(q - q0))) / MAX_JOINT_SPEED_RAD_S,
        )
        if not (math.isfinite(t) and t >= need):
            raise ValueError(f"time_to_go {time_to_go!r} is below {need:.2f} s")
        self._last_pose = self._last_joints = None
        self.robot.move_to_joint_positions(self._t(q.tolist()), time_to_go=t)

    def start_cartesian_impedance(self, Kx, Kxd):
        kx, kxd = _finite("Kx", Kx, 6), _finite("Kxd", Kxd, 6)
        if (
            np.any(kx <= 0)
            or np.any(kx > MAX_KX)
            or np.any(kxd < 0)
            or np.any(kxd > MAX_KXD)
        ):
            raise ValueError(
                f"gains out of range: 0 < Kx <= {list(MAX_KX)}, "
                f"0 <= Kxd <= {list(MAX_KXD)}"
            )
        self._last_pose = self._last_joints = None
        self.robot.start_cartesian_impedance(
            Kx=self._t(kx.tolist()), Kxd=self._t(kxd.tolist())
        )

    def start_joint_impedance(self, Kq=None, Kqd=None):
        if Kq is not None or Kqd is not None:
            raise ValueError("custom joint impedance gains are not accepted")
        self._last_pose = self._last_joints = None
        self.robot.start_joint_impedance()

    def update_desired_ee_pose(self, pose):
        p = self._check_pose(pose)
        self.robot.update_desired_ee_pose(
            position=self._t(p[:3].tolist()), orientation=self._t(p[3:].tolist())
        )
        self._last_pose = p

    def update_desired_joint_pos(self, pos):
        q = self._check_joints("pos", pos)
        ref = self._last_joints
        if ref is None:
            ref = _finite("joint positions", self.get_joint_positions(), 7)
        if float(np.max(np.abs(q - ref))) > self.limits.max_joint_step_rad:
            raise ValueError(
                f"update_desired_joint_pos jumps more than "
                f"{self.limits.max_joint_step_rad} rad; refused"
            )
        self.robot.update_desired_joint_positions(self._t(q.tolist()))
        self._last_joints = q

    def terminate_current_policy(self):
        self._last_pose = self._last_joints = None
        self.robot.terminate_current_policy()

    def control_gripper(self, close):
        if close:
            self.gripper.grasp(speed=self.speed, force=self.force, blocking=False)
        else:
            self.gripper.goto(
                width=self.max_width, speed=self.speed, force=self.force, blocking=False
            )

    def set_gripper_position(self, width):
        w = float(width)
        if not (math.isfinite(w) and 0.0 <= w <= self.max_width):
            raise ValueError(f"width must be in [0, {self.max_width}] m, got {width!r}")
        self.gripper.goto(width=w, speed=self.speed, force=self.force, blocking=False)


READ_ONLY = (
    "get_ee_pose",
    "get_joint_positions",
    "get_joint_velocities",
    "get_gripper_position",
)


def main() -> int:
    import torch
    import zerorpc
    from polymetis import GripperInterface, RobotInterface

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4242)
    parser.add_argument(
        "--robot-ip", default="localhost", help="Polymetis robot server"
    )
    parser.add_argument(
        "--gripper-ip", default="localhost", help="Polymetis gripper server"
    )
    parser.add_argument("--gripper-speed", type=float, default=0.1)
    parser.add_argument("--gripper-force", type=float, default=20.0)
    parser.add_argument(
        "--read-only", action="store_true", help="Serve only the getters."
    )
    parser.add_argument(
        "--max-step-m",
        type=float,
        default=0.01,
        help="Largest ee setpoint jump per request from the last setpoint or the "
        "measured pose.",
    )
    parser.add_argument("--max-step-rad", type=float, default=0.05)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.05)
    parser.add_argument("--min-time-to-go-s", type=float, default=0.5)
    parser.add_argument(
        "--workspace-min",
        type=float,
        nargs=3,
        default=None,
        help="Optional box for the reported ee link (m), checked on every setpoint.",
    )
    parser.add_argument("--workspace-max", type=float, nargs=3, default=None)
    parser.add_argument("--z-floor", type=float, default=None)
    args = parser.parse_args()
    limits = NucLimits(
        max_step_m=args.max_step_m,
        max_step_rad=args.max_step_rad,
        max_joint_step_rad=args.max_joint_step_rad,
        min_time_to_go_s=args.min_time_to_go_s,
        workspace_min=tuple(args.workspace_min) if args.workspace_min else None,
        workspace_max=tuple(args.workspace_max) if args.workspace_max else None,
        z_floor_m=args.z_floor,
    )
    values = [v for v in vars(args).values() if isinstance(v, float)]
    values += [*(args.workspace_min or ()), *(args.workspace_max or ())]
    if not all(math.isfinite(v) for v in values):
        parser.error("numeric arguments must be finite")
    if not (
        0 < limits.max_step_m <= 0.02
        and 0 < limits.max_step_rad <= 0.1
        and 0 < limits.max_joint_step_rad <= 0.1
    ):
        parser.error("step limits must be in (0, 0.02] m, (0, 0.1] rad")
    server = FrankaServer(
        RobotInterface(ip_address=args.robot_ip),
        GripperInterface(ip_address=args.gripper_ip),
        args.gripper_speed,
        args.gripper_force,
        limits,
        lambda values: torch.tensor(values, dtype=torch.float32),
    )
    if args.read_only:
        server = type(
            "ReadOnlyFrankaServer", (), {m: getattr(server, m) for m in READ_ONLY}
        )()
    rpc = zerorpc.Server(server, heartbeat=20)
    rpc.bind(f"tcp://{args.host}:{args.port}")
    print(f"franka_server listening on tcp://{args.host}:{args.port}", flush=True)
    rpc.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
