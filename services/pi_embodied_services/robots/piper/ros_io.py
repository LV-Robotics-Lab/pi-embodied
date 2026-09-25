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
# Modified by pi-embodied: PiperInterface (core/piper/piper_interface.py) and
# RosImageCamera (core/piper/ros_camera.py) merged into one module; the Franka-parity
# no-ops, enable/disable and the blocking joint move are dropped (the controller
# streams joint moves itself so ``stop`` can interrupt them); prints go to the logger.

"""ROS-topic transport for one AgileX Piper arm and its Orbbec cameras.

Talks to one ``piper_start_ms_node.py`` (AgileX ``cobot_magic`` Piper_ros stack)
launched in mode 1 (software control). The node owns the CAN bus and ``piper_sdk``;
this module only publishes and subscribes ROS topics, so it runs in the services
venv once ROS Noetic and the Piper workspace are sourced (``rospy`` is pure Python;
``rospkg`` / ``catkin_pkg`` must be installed in the venv):

    source /opt/ros/noetic/setup.bash
    source ~/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

Topics (per arm, ``<arm>`` in {left, right}):

  feedback (200 Hz in every mode)
    /puppet/joint_<arm>           sensor_msgs/JointState: position[0:6] joints (rad),
                                  position[6] gripper jaw opening (m)
    /puppet/end_pose_euler_<arm>  piper_msgs/PosCmd: x,y,z (m); roll,pitch,yaw (rad,
                                  extrinsic xyz)
    /puppet/arm_status_<arm>      piper_msgs/PiperStatusMsg: ctrl_mode, err_code, ...
  commands (consumed in mode 1 only, silently dropped otherwise)
    /puppet/pos_cmd_<arm>         piper_msgs/PosCmd: absolute end pose -> EndPoseCtrl
                                  (firmware MOVE P) and GripperCtrl(gripper m)
    /master/joint_<arm>           sensor_msgs/JointState: joints -> JointCtrl (MOVE J),
                                  position[6] -> GripperCtrl

Every command carries a gripper width, so the tracked target width rides along on
each pose/joint command. Poses at this boundary are [x,y,z,qx,qy,qz,qw] in the Piper
base frame (m), quaternions scipy xyzw from/to extrinsic-xyz euler.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from pi_embodied_services.utils.logging import get_logger

logger = get_logger("piper_ros")

PIPER_DOF = 6
#: Command-side clamp for the gripper width (m). The node's own upper clamp compares
#: metres against 80000, so the 80000 um CAN limit is enforced here.
PIPER_GRIPPER_MAX_CMD_M = 0.08


def _require_ros() -> tuple[Any, ...]:
    """Import rospy and the message types, with an actionable error if missing."""
    try:
        import rospy
        from sensor_msgs.msg import Image, JointState
    except ImportError as exc:
        raise ImportError(
            "rospy / sensor_msgs are not importable. Source ROS Noetic first "
            "(source /opt/ros/noetic/setup.bash) and install the services' [piper] "
            "extra (rospkg, catkin_pkg) into this Python."
        ) from exc
    try:
        from piper_msgs.msg import PiperStatusMsg, PosCmd
    except ImportError as exc:
        raise ImportError(
            "piper_msgs is not importable. Source the built Piper workspace: "
            "source ~/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash"
        ) from exc
    if not rospy.core.is_initialized():
        # One node per process; disable_signals keeps SIGINT/SIGTERM with the server.
        rospy.init_node("pi_embodied_piper", anonymous=True, disable_signals=True)
    return rospy, JointState, Image, PosCmd, PiperStatusMsg


class PiperRosArm:
    """ROS-topic client for one Piper arm (the controller's ``robot``)."""

    def __init__(
        self,
        arm: str = "left",
        feedback_timeout_s: float = 5.0,
        max_feedback_age_s: float = 0.5,
    ) -> None:
        if arm not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        self.arm = arm
        self.feedback_timeout_s = float(feedback_timeout_s)
        self.max_feedback_age_s = float(max_feedback_age_s)
        rospy, JointState, _image, PosCmd, PiperStatusMsg = _require_ros()
        self._rospy = rospy
        self._JointState = JointState
        self._PosCmd = PosCmd
        # Latest feedback as (msg, receipt monotonic time); swapped atomically.
        self._joint: tuple[Any, float] | None = None
        self._pose: tuple[Any, float] | None = None
        self._status: tuple[Any, float] | None = None
        self._last_err: int | None = None
        # Width carried by every command; adopted from the measured width on connect
        # so the first command cannot yank the fingers.
        self._gripper_target_m: float | None = None
        # Last commanded Cartesian pose; a gripper-only command re-sends it.
        self._last_pose: np.ndarray | None = None
        self._subs = [
            rospy.Subscriber(
                f"/puppet/joint_{arm}",
                JointState,
                self._on_joint,
                queue_size=1,
                tcp_nodelay=True,
            ),
            rospy.Subscriber(
                f"/puppet/end_pose_euler_{arm}",
                PosCmd,
                self._on_pose,
                queue_size=1,
                tcp_nodelay=True,
            ),
            rospy.Subscriber(
                f"/puppet/arm_status_{arm}",
                PiperStatusMsg,
                self._on_status,
                queue_size=1,
                tcp_nodelay=True,
            ),
        ]
        self._pub_pose = rospy.Publisher(
            f"/puppet/pos_cmd_{arm}", PosCmd, queue_size=1, tcp_nodelay=True
        )
        self._pub_joint = rospy.Publisher(
            f"/master/joint_{arm}", JointState, queue_size=1, tcp_nodelay=True
        )

    def _on_joint(self, msg: Any) -> None:
        self._joint = (msg, time.monotonic())

    def _on_pose(self, msg: Any) -> None:
        self._pose = (msg, time.monotonic())

    def _on_status(self, msg: Any) -> None:
        self._status = (msg, time.monotonic())
        err = int(getattr(msg, "err_code", 0))
        if err != self._last_err:
            if err:
                logger.warning("piper-%s arm_status err_code=%d", self.arm, err)
            self._last_err = err

    def connect(self) -> PiperRosArm:
        """Wait for feedback and for the node to subscribe to the command topics."""
        deadline = time.monotonic() + self.feedback_timeout_s
        while self._joint is None or self._pose is None:
            if self._rospy.is_shutdown():
                raise RuntimeError("ROS is shutting down")
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"no Piper feedback on /puppet/joint_{self.arm} or "
                    f"/puppet/end_pose_euler_{self.arm} after "
                    f"{self.feedback_timeout_s:.0f}s: is the arm node running "
                    "(mode:=1, CAN activated)?"
                )
            time.sleep(0.02)
        deadline = time.monotonic() + 3.0
        while (
            self._pub_pose.get_num_connections() < 1
            or self._pub_joint.get_num_connections() < 1
        ):
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"/puppet/pos_cmd_{self.arm} or /master/joint_{self.arm} has no "
                    "subscriber: the arm node is not in mode 1 (software control), "
                    "so every command would be dropped"
                )
            time.sleep(0.02)
        self._gripper_target_m = self.get_gripper_width()
        return self

    def close(self) -> None:
        for handle in [*self._subs, self._pub_pose, self._pub_joint]:
            try:
                handle.unregister()
            except Exception:
                pass

    # -- feedback ---------------------------------------------------------

    def _fresh(self, cached: tuple[Any, float] | None, topic: str) -> Any:
        if cached is None:
            raise RuntimeError(f"no message received yet on {topic}")
        msg, received = cached
        age = time.monotonic() - received
        if age > self.max_feedback_age_s:
            raise RuntimeError(
                f"feedback on {topic} is {age:.2f}s old (> {self.max_feedback_age_s:.2f}s):"
                " the arm node died or the CAN link dropped; not acting on frozen state"
            )
        return msg

    def get_ee_pose(self) -> np.ndarray:
        msg = self._fresh(self._pose, f"/puppet/end_pose_euler_{self.arm}")
        quat = R.from_euler("xyz", [msg.roll, msg.pitch, msg.yaw]).as_quat()
        return np.concatenate([[msg.x, msg.y, msg.z], quat])

    def get_joint_positions(self) -> np.ndarray:
        msg = self._fresh(self._joint, f"/puppet/joint_{self.arm}")
        return np.asarray(msg.position[:PIPER_DOF], dtype=float)

    def get_gripper_width(self) -> float:
        msg = self._fresh(self._joint, f"/puppet/joint_{self.arm}")
        return float(msg.position[PIPER_DOF])

    def arm_status(self) -> dict[str, Any] | None:
        if self._status is None:
            return None
        msg = self._status[0]
        return {
            k: getattr(msg, k)
            for k in ("ctrl_mode", "arm_status", "mode_feedback", "err_code")
            if hasattr(msg, k)
        }

    # -- commands ---------------------------------------------------------

    def _assert_commandable(self) -> None:
        self._fresh(self._joint, f"/puppet/joint_{self.arm}")
        if self._pub_pose.get_num_connections() < 1:
            raise RuntimeError(
                f"/puppet/pos_cmd_{self.arm} has no subscriber: the arm node is not in "
                "mode 1 (software control)"
            )

    def _width(self) -> float:
        width = (
            self._gripper_target_m
            if self._gripper_target_m is not None
            else self.get_gripper_width()
        )
        return float(np.clip(width, 0.0, PIPER_GRIPPER_MAX_CMD_M))

    def command_pose(self, pose7: np.ndarray) -> None:
        """Command an absolute end pose (MOVE P) plus the tracked gripper width."""
        self._assert_commandable()
        pose7 = np.asarray(pose7, dtype=float).reshape(-1)[:7]
        roll, pitch, yaw = R.from_quat(pose7[3:7]).as_euler("xyz")
        msg = self._PosCmd()
        msg.x, msg.y, msg.z = (float(v) for v in pose7[:3])
        msg.roll, msg.pitch, msg.yaw = float(roll), float(pitch), float(yaw)
        msg.gripper = self._width()
        msg.mode1 = 0
        msg.mode2 = 0
        self._pub_pose.publish(msg)
        self._last_pose = pose7.copy()

    def stream_joints(self, q: np.ndarray) -> None:
        """Stream one joint waypoint (MOVE J at speed 100) plus the gripper width."""
        self._assert_commandable()
        msg = self._JointState()
        msg.header.stamp = self._rospy.Time.now()
        msg.name = [f"joint{j}" for j in range(PIPER_DOF + 1)]
        msg.position = [
            float(v) for v in np.asarray(q, dtype=float).reshape(-1)[:PIPER_DOF]
        ] + [self._width()]
        msg.velocity = [0.0] * (PIPER_DOF + 1)
        msg.effort = [0.0] * (PIPER_DOF + 1)
        self._pub_joint.publish(msg)

    def set_gripper_width(self, width_m: float) -> None:
        """Command the jaw opening; re-sends the last pose so the arm does not move."""
        self._assert_commandable()
        self._gripper_target_m = float(np.clip(width_m, 0.0, PIPER_GRIPPER_MAX_CMD_M))
        self.command_pose(
            self._last_pose if self._last_pose is not None else self.get_ee_pose()
        )

    def note_commanded_pose(self, pose7: np.ndarray | None) -> None:
        """Record (or with None forget) the last commanded pose without publishing.

        After a joint-space move the next gripper-only command must re-send where the
        arm was driven, not a stale pre-move pose; after a divergence re-sync it must
        fall back to the measured pose.
        """
        self._last_pose = (
            None
            if pose7 is None
            else np.asarray(pose7, dtype=float).reshape(-1)[:7].copy()
        )


class RosImageCamera:
    """Latest frame of a ``sensor_msgs/Image`` color topic, as HxWx3 uint8 RGB."""

    def __init__(
        self, topic: str, max_age_s: float = 1.0, connect_timeout_s: float = 10.0
    ) -> None:
        self.topic = topic
        self.max_age_s = float(max_age_s)
        rospy, _joint, Image, _pos, _status = _require_ros()
        self._latest: tuple[np.ndarray, float] | None = None
        self._lock = threading.Lock()
        self._sub = rospy.Subscriber(
            topic, Image, self._on_image, queue_size=1, tcp_nodelay=True
        )
        deadline = time.monotonic() + float(connect_timeout_s)
        while self._latest is None:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"no image on {topic} after {connect_timeout_s:.0f}s: is "
                    "astra_camera multi_camera.launch running?"
                )
            time.sleep(0.02)

    def _on_image(self, msg: Any) -> None:
        try:
            frame = decode_image(msg)
        except Exception as exc:
            logger.warning("%s: decode error: %s", self.topic, exc)
            return
        with self._lock:
            self._latest = (frame, time.monotonic())

    def read(self) -> np.ndarray:
        with self._lock:
            cached = self._latest
        if cached is None or time.monotonic() - cached[1] > self.max_age_s:
            raise RuntimeError(
                f"no fresh frame on {self.topic} (camera driver stopped or stalled)"
            )
        return cached[0].copy()

    def close(self) -> None:
        try:
            self._sub.unregister()
        except Exception:
            pass


def decode_image(msg: Any) -> np.ndarray:
    """``sensor_msgs/Image`` -> HxWx3 uint8 RGB without cv_bridge."""
    enc = (msg.encoding or "").lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    if enc in ("rgb8", "bgr8"):
        arr = buf.reshape(h, step)[:, : w * 3].reshape(h, w, 3)
        return np.ascontiguousarray(arr[:, :, ::-1] if enc == "bgr8" else arr)
    if enc in ("rgba8", "bgra8"):
        arr = buf.reshape(h, step)[:, : w * 4].reshape(h, w, 4)[:, :, :3]
        return np.ascontiguousarray(arr[:, :, ::-1] if enc == "bgra8" else arr)
    if enc == "mono8":
        gray = buf.reshape(h, step)[:, :w].reshape(h, w, 1)
        return np.ascontiguousarray(np.repeat(gray, 3, axis=2))
    raise ValueError(f"unsupported image encoding {msg.encoding!r}")
