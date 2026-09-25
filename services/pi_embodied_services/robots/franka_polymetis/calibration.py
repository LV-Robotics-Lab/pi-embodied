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

"""Write a hand-eye result as the easy_handeye YAML pi's back_project reads.

Perception parity with the RLinf backend:

- Intrinsics need no conversion: ``env.get_camera_meta`` reports each RealSense's live
  color intrinsics (depth is aligned to color) as ``raw_color_intrinsics`` and the
  ``intrinsic_K`` of the emitted, letterboxed image, the same fields the RLinf server
  reports for its cropped/resized image.
- Extrinsics come from ``perception.calibration`` in the robot YAML, exactly as for the
  RLinf backend (``robots/franka/perception.load_calibration_bundle``): one
  easy_handeye YAML per camera, ``external`` (eye-on-base: camera pose in the robot
  base frame) and ``wrist`` (eye-on-hand: camera pose in the end-effector frame).
  Show-Harness itself has no extrinsic calibration (its agent works on images only),
  so there is nothing of theirs to convert; calibrate with easy_handeye, or convert
  another tool's result with this module.

The matrix is ``T_frame_camera``: it maps points in the camera's *optical* frame (x
right, y down, z forward; what K^-1 [u, v, 1] * depth gives) into the base frame
(external) or the end-effector frame (wrist). OpenCV ``cv2.calibrateHandEye`` returns
it directly: eye-in-hand (gripper poses in, ``R_cam2gripper``/``t_cam2gripper`` out)
gives the wrist matrix; eye-to-hand (pass the inverted base->gripper poses) gives the
external one. For the wrist, the end-effector frame must be the frame whose poses were
used to calibrate, and it must equal the TCP frame the server reports
(``raw_base_state.tcp_pose``: Polymetis's end-effector link moved by
``robot.tcp_offset_m`` and turned by ``robot.tcp_yaw_deg`` about its z-axis). A wrist
calibration made against the RLinf/libfranka TCP (``O_T_EE`` = flange * Franka Hand
``F_T_EE``: z 0.1034 m and -45 deg about z) needs both: ``tcp_offset_m: [0, 0,
0.1034]`` and ``tcp_yaw_deg: -45``. The offset alone leaves the camera frame turned
45 deg about the tool axis, i.e. several cm off for a camera ~10 cm from that axis.

    python -m pi_embodied_services.robots.franka_polymetis.calibration \\
        --matrix T_ee_wrist.json --eye-on-hand --out ~/.ros/easy_handeye/wrist.yaml

``--matrix`` takes a JSON 4x4 (or ``{"R": 3x3, "t": 3}``); ``--invert`` if the tool
gave the opposite direction (camera -> frame).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def matrix_to_quat_xyzw(r: Any) -> np.ndarray:
    """Rotation matrix -> unit quaternion [qx, qy, qz, qw] (qw >= 0)."""
    m = np.asarray(r, dtype=np.float64)
    if m.shape != (3, 3) or not np.allclose(m @ m.T, np.eye(3), atol=1e-4):
        raise ValueError("rotation must be an orthonormal 3x3 matrix")
    if np.linalg.det(m) < 0:
        raise ValueError("rotation has det < 0 (a reflection, not a rotation)")
    tr = float(np.trace(m))
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
            s / 4,
        ]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        q = [0.0, 0.0, 0.0, (m[k, j] - m[j, k]) / s]
        q[i] = s / 4
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
    q = np.asarray(q) / np.linalg.norm(q)
    return -q if q[3] < 0 else q


def easy_handeye_yaml(
    transform: Any,
    *,
    eye_on_hand: bool,
    robot_base_frame: str = "panda_link0",
    robot_effector_frame: str = "panda_EE",
    tracking_base_frame: str = "camera_color_optical_frame",
    tracking_marker_frame: str = "tag",
) -> dict[str, Any]:
    """The easy_handeye YAML mapping for a 4x4 ``T_frame_camera``."""
    t = np.asarray(transform, dtype=np.float64)
    if t.shape != (4, 4) or not np.allclose(t[3], [0, 0, 0, 1]):
        raise ValueError("transform must be a homogeneous 4x4 matrix")
    qx, qy, qz, qw = (float(v) for v in matrix_to_quat_xyzw(t[:3, :3]))
    return {
        "parameters": {
            "eye_on_hand": bool(eye_on_hand),
            "freehand_robot_movement": False,
            "robot_base_frame": robot_base_frame,
            "robot_effector_frame": robot_effector_frame,
            "tracking_base_frame": tracking_base_frame,
            "tracking_marker_frame": tracking_marker_frame,
        },
        "transformation": {
            "x": float(t[0, 3]),
            "y": float(t[1, 3]),
            "z": float(t[2, 3]),
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
        },
    }


def load_matrix(path: str | Path) -> np.ndarray:
    data = json.loads(Path(path).expanduser().read_text())
    if isinstance(data, dict):
        t = np.eye(4)
        t[:3, :3] = np.asarray(data["R"], dtype=np.float64)
        t[:3, 3] = np.asarray(data["t"], dtype=np.float64).reshape(3)
        return t
    return np.asarray(data, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--matrix", required=True, help="JSON 4x4, or {R, t}")
    parser.add_argument("--eye-on-hand", action="store_true", help="wrist camera")
    parser.add_argument(
        "--invert", action="store_true", help="input is camera -> frame"
    )
    parser.add_argument("--robot-base-frame", default="panda_link0")
    parser.add_argument("--robot-effector-frame", default="panda_EE")
    parser.add_argument("--tracking-base-frame", default="camera_color_optical_frame")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    t = load_matrix(args.matrix)
    if args.invert:
        t = np.linalg.inv(t)
    doc = easy_handeye_yaml(
        t,
        eye_on_hand=args.eye_on_hand,
        robot_base_frame=args.robot_base_frame,
        robot_effector_frame=args.robot_effector_frame,
        tracking_base_frame=args.tracking_base_frame,
    )
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
