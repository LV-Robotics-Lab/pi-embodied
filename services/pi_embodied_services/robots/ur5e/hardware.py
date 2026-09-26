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
#
# After OpenETA real/robots/ur5e.py (ur_rtde receive/control split, lazy control
# connection) and real/robots/robotiq.py (the URCap socket register protocol).
# Modified by pi-embodied: motions are started asynchronously so the controller can
# poll ``stop`` and call stopL/stopJ; the arm's serial number is read for the
# calibration binding; the gripper client is read/write with non-blocking go_to.

"""Hardware handles: the UR5e over ur_rtde and a Robotiq gripper over the URCap socket.

Both import lazily (``ur_rtde``; the gripper needs only the standard library) so the
env server and its tests import without them. The controller (control.py) owns every
limit; these classes only move what they are told to.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

import numpy as np

from pi_embodied_services.utils.logging import get_logger

logger = get_logger("ur5e_hw")

#: Robotiq URCap socket on the UR controller.
ROBOTIQ_PORT = 63352
#: ``OBJ`` register (gOBJ in the 2F manual): 0 fingers moving toward the requested
#: position, 1 stopped by contact while opening, 2 stopped by contact while closing,
#: 3 at the requested position (nothing detected). The register keeps the previous
#: command's value until the fingers start moving, so right after ``GTO`` it can still
#: read 3 from the last motion; ``PRE`` (gPR, the echoed requested position) tells
#: whether a new command was taken.
OBJ_MOVING, OBJ_OPENING_CONTACT, OBJ_CLOSING_CONTACT, OBJ_AT_POSITION = 0, 1, 2, 3


class RtdeArm:
    """A UR arm over ur_rtde: state from ``RTDEReceiveInterface``, motion through
    ``RTDEControlInterface`` (connected on the first motion so read-only use works
    while the robot rejects external control), identity from the dashboard client."""

    def __init__(self, ip: str, *, dashboard_port: int = 29999) -> None:
        from rtde_receive import RTDEReceiveInterface

        self.ip = str(ip)
        self.dashboard_port = int(dashboard_port)
        self._recv = RTDEReceiveInterface(self.ip)
        self._ctrl: Any = None

    def _control(self) -> Any:
        if self._ctrl is None:
            from rtde_control import RTDEControlInterface

            self._ctrl = RTDEControlInterface(self.ip)
        return self._ctrl

    # -- state -------------------------------------------------------------

    def tcp_pose(self) -> np.ndarray:
        """``[x, y, z, rx, ry, rz]`` (m, axis-angle) in the base frame."""
        return np.asarray(self._recv.getActualTCPPose(), dtype=np.float64)

    def joints(self) -> np.ndarray:
        return np.asarray(self._recv.getActualQ(), dtype=np.float64)

    def joint_speeds(self) -> np.ndarray:
        return np.asarray(self._recv.getActualQd(), dtype=np.float64)

    def status(self) -> dict[str, Any]:
        """Safety state from the receive interface (works while control is refused)."""
        r = self._recv
        return {
            "robot_mode": int(r.getRobotMode()),
            "safety_mode": int(r.getSafetyMode()),
            "protective_stopped": bool(r.isProtectiveStopped()),
            "emergency_stopped": bool(r.isEmergencyStopped()),
            "program_running": bool(r.isProgramRunning()),
        }

    def identity(self) -> str | None:
        """The controller's serial number (dashboard ``get serial number``), or None."""
        try:
            from dashboard_client import DashboardClient
        except ImportError:
            return None
        try:
            client = DashboardClient(self.ip, self.dashboard_port)
            client.connect()
            try:
                return str(client.getSerialNumber()).strip() or None
            finally:
                client.disconnect()
        except Exception as exc:
            logger.warning("dashboard serial number unavailable: %s", exc)
            return None

    # -- motion (asynchronous: returns at once, poll ``busy``) ------------------

    def move_l(self, pose: Any, speed: float, accel: float) -> bool:
        """Start a moveL; False when the controller rejected it (unreachable pose,
        not in remote control, protective stop, RTDE script not running)."""
        return bool(
            self._control().moveL(
                [float(v) for v in pose], float(speed), float(accel), True
            )
        )

    def move_j(self, q: Any, speed: float, accel: float) -> bool:
        """Start a moveJ; False when the controller rejected it (see ``move_l``)."""
        return bool(
            self._control().moveJ(
                [float(v) for v in q], float(speed), float(accel), True
            )
        )

    def busy(self) -> bool:
        """Whether an asynchronous motion is still running."""
        return int(self._control().getAsyncOperationProgress()) >= 0

    def stop_l(self, decel: float) -> None:
        self._control().stopL(float(decel))

    def stop_j(self, decel: float) -> None:
        self._control().stopJ(float(decel))

    def close(self) -> None:
        for handle in (self._ctrl, self._recv):
            if handle is not None:
                try:
                    handle.disconnect()
                except Exception:
                    pass
        self._ctrl = self._recv = None


class RobotiqGripper:
    """A Robotiq 2F gripper through the URCap socket (``GET``/``SET <VAR>`` text protocol).

    Positions are 0 (open) .. 255 (closed). ``go_to`` returns at once; the controller
    polls :meth:`position` and :meth:`object_status` and decides when it settled.
    """

    def __init__(self, host: str, port: int = ROBOTIQ_PORT, *, timeout_s: float = 3.0):
        self._host, self._port, self._timeout = str(host), int(port), float(timeout_s)
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def _connect(self) -> socket.socket:
        if self._sock is None:
            sock = socket.create_connection(
                (self._host, self._port), timeout=self._timeout
            )
            sock.settimeout(self._timeout)
            self._sock = sock
        return self._sock

    def _get(self, var: str) -> int:
        with self._lock:
            sock = self._connect()
            try:
                sock.sendall(f"GET {var}\n".encode())
                reply = sock.recv(1024).decode(errors="replace").split()
            except OSError:
                self.close()
                raise
        if len(reply) < 2 or reply[0] != var:
            raise RuntimeError(f"Robotiq: bad reply to GET {var}: {' '.join(reply)!r}")
        return int(reply[-1])

    def _set(self, var: str, value: int) -> None:
        with self._lock:
            sock = self._connect()
            try:
                sock.sendall(f"SET {var} {int(value)}\n".encode())
                reply = sock.recv(1024).decode(errors="replace").strip()
            except OSError:
                self.close()
                raise
        if reply != "ack":
            raise RuntimeError(f"Robotiq: SET {var} {value} answered {reply!r}")

    def activated(self) -> bool:
        return self._get("STA") == 3 and self._get("ACT") == 1

    def activate(self, timeout_s: float = 10.0) -> None:
        """``SET ACT 0`` then ``1`` (the fingers home: a motion), wait for ``STA 3``."""
        if self.activated():
            return
        self._set("ACT", 0)
        self._set("ACT", 1)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._get("STA") == 3:
                return
            if self._get("FLT"):
                raise RuntimeError(
                    f"Robotiq fault {self._get('FLT')} during activation"
                )
            time.sleep(0.2)
        raise RuntimeError("Robotiq did not finish activating")

    def position(self) -> int:
        return self._get("POS")

    def requested_position(self) -> int:
        """``PRE``: the position request the gripper is acting on (echoed back once
        a ``GTO`` command was taken, so it tells a fresh status from a stale one)."""
        return self._get("PRE")

    def object_status(self) -> int:
        return self._get("OBJ")

    def fault(self) -> int:
        return self._get("FLT")

    def go_to(self, position: int, speed: int, force: int) -> None:
        self._set("SPE", max(0, min(255, int(speed))))
        self._set("FOR", max(0, min(255, int(force))))
        self._set("POS", max(0, min(255, int(position))))
        self._set("GTO", 1)

    def stop(self) -> None:
        """``SET GTO 0``: the fingers stop where they are (a cancelled command)."""
        self._set("GTO", 0)

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
