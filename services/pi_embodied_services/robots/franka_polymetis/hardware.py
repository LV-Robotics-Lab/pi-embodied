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
# Adapted from Show-Harness core/franka/franka_interface.py (FrankaInterface).
# Modified by pi-embodied: prints go to the logger; the ZeroRPC client takes an
# explicit call timeout; the measured gripper state (width, grasp and moving flags) is
# read with get_gripper_state. The RealSense driver lives in the shared camera layer
# (components/cameras/realsense.py).

"""Hardware handle: the NUC's Polymetis ``franka_server``.

``zerorpc`` is imported lazily so the env server and its tests import without it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pi_embodied_services.utils.logging import get_logger

logger = get_logger("franka_polymetis_hw")


class PolymetisRobot:
    """ZeroRPC client of the Polymetis ``franka_server`` on the Franka NUC (port 4242).

    The method surface is Show-Harness's ``FrankaInterface``; ``nuc_server.py`` in this
    package is a reference implementation of the server side.
    """

    def __init__(
        self,
        ip: str,
        port: int = 4242,
        *,
        heartbeat_s: float | None = 20.0,
        timeout_s: float = 30.0,
    ) -> None:
        import zerorpc

        self.server = zerorpc.Client(heartbeat=heartbeat_s, timeout=timeout_s)
        self.server.connect(f"tcp://{ip}:{int(port)}")
        self.endpoint = f"tcp://{ip}:{int(port)}"
        self._gripper_state_rpc = True

    def get_ee_pose(self) -> np.ndarray:
        return np.asarray(self.server.get_ee_pose(), dtype=np.float64)

    def get_joint_positions(self) -> np.ndarray:
        return np.asarray(self.server.get_joint_positions(), dtype=np.float64)

    def move_to_joint_positions(self, positions: Any, time_to_go: float) -> None:
        self.server.move_to_joint_positions(
            np.asarray(positions, dtype=float).tolist(), float(time_to_go)
        )

    def start_cartesian_impedance(self, Kx: Any, Kxd: Any) -> None:
        self.server.start_cartesian_impedance(
            np.asarray(Kx, dtype=float).tolist(), np.asarray(Kxd, dtype=float).tolist()
        )

    def start_joint_impedance(self, Kq: Any = None, Kqd: Any = None) -> None:
        self.server.start_joint_impedance(
            None if Kq is None else np.asarray(Kq, dtype=float).tolist(),
            None if Kqd is None else np.asarray(Kqd, dtype=float).tolist(),
        )

    def update_desired_ee_pose(self, pose: Any) -> None:
        self.server.update_desired_ee_pose(np.asarray(pose, dtype=float).tolist())

    def update_desired_joint_pos(self, pos: Any) -> None:
        self.server.update_desired_joint_pos(np.asarray(pos, dtype=float).tolist())

    def control_gripper(self, close: bool) -> None:
        """True closes (grasp), False opens."""
        self.server.control_gripper(bool(close))

    def get_gripper_position(self) -> np.ndarray:
        return np.asarray(self.server.get_gripper_position(), dtype=np.float64).reshape(
            1
        )

    def get_gripper_state(self) -> dict[str, Any]:
        """Measured {width, is_grasped, is_moving}; a NUC server without
        ``get_gripper_state`` gives the width only (flags None)."""
        if self._gripper_state_rpc:
            try:
                raw = self.server.get_gripper_state()
                return {
                    "width": float(raw["width"]),
                    "is_grasped": raw.get("is_grasped"),
                    "is_moving": raw.get("is_moving"),
                }
            except Exception as exc:
                if getattr(exc, "name", "") != "NameError":  # zerorpc: no such method
                    raise
                logger.warning(
                    "the NUC server has no get_gripper_state (update nuc_server.py); "
                    "reporting the gripper width without the grasp flag"
                )
                self._gripper_state_rpc = False
        width = float(self.get_gripper_position()[0])
        return {"width": width, "is_grasped": None, "is_moving": None}

    def terminate_current_policy(self) -> None:
        self.server.terminate_current_policy()

    def close(self) -> None:
        try:
            self.server.close()
        except Exception:
            pass
