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
"""

from __future__ import annotations

import argparse


class FrankaServer:
    def __init__(self, robot_ip: str, gripper_ip: str, speed: float, force: float):
        import torch
        from polymetis import GripperInterface, RobotInterface

        self._torch = torch
        self.robot = RobotInterface(ip_address=robot_ip)
        self.gripper = GripperInterface(ip_address=gripper_ip)
        self.speed, self.force = float(speed), float(force)
        self.max_width = float(getattr(self.gripper.metadata, "max_width", 0.08))

    def _t(self, values):
        return self._torch.tensor(values, dtype=self._torch.float32)

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
        self.robot.move_to_joint_positions(
            self._t(positions), time_to_go=float(time_to_go)
        )

    def start_cartesian_impedance(self, Kx, Kxd):
        self.robot.start_cartesian_impedance(Kx=self._t(Kx), Kxd=self._t(Kxd))

    def start_joint_impedance(self, Kq=None, Kqd=None):
        self.robot.start_joint_impedance(
            Kq=None if Kq is None else self._t(Kq),
            Kqd=None if Kqd is None else self._t(Kqd),
        )

    def update_desired_ee_pose(self, pose):
        self.robot.update_desired_ee_pose(
            position=self._t(pose[:3]), orientation=self._t(pose[3:])
        )

    def update_desired_joint_pos(self, pos):
        self.robot.update_desired_joint_positions(self._t(pos))

    def terminate_current_policy(self):
        self.robot.terminate_current_policy()

    def control_gripper(self, close):
        if close:
            self.gripper.grasp(speed=self.speed, force=self.force, blocking=False)
        else:
            self.gripper.goto(
                width=self.max_width, speed=self.speed, force=self.force, blocking=False
            )

    def set_gripper_position(self, width):
        self.gripper.goto(
            width=float(width), speed=self.speed, force=self.force, blocking=False
        )


READ_ONLY = (
    "get_ee_pose",
    "get_joint_positions",
    "get_joint_velocities",
    "get_gripper_position",
)


def main() -> int:
    import zerorpc

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
    args = parser.parse_args()
    server = FrankaServer(
        args.robot_ip, args.gripper_ip, args.gripper_speed, args.gripper_force
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
