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
# Modified by pi-embodied: reduced to the hand-eye calibration loaders used by
# pi-embodied's Franka robot (via ``python -c``); the planner-side RGB-D
# back-projection tools (which need the upstream session/toolkit) are omitted.
# Function bodies are unchanged.

"""Franka hand-eye calibration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pi_embodied_services.robots.franka.runtime_config import (
    get_perception_calibration_mapping,
    get_robot_config_path,
    load_easy_handeye_yaml,
)


def load_calibration_bundle() -> dict[str, Any]:
    """Load normalized hand-eye calibration for the Franka cameras.

    Reads the ``perception.calibration`` easy_handeye YAML mapping from the
    active robot config (``--robot-config``) and normalizes each camera entry.
    """
    sources = get_perception_calibration_mapping()
    if not sources:
        raise ValueError(
            "no hand-eye calibration configured: list easy_handeye YAMLs "
            "under perception.calibration in the robot config "
            f"({get_robot_config_path()})"
        )
    data: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for key, source in sources.items():
        source_path = Path(source).expanduser()
        try:
            data[key] = load_easy_handeye_yaml(source_path)
        except ValueError as exc:
            raise ValueError(
                f"cannot load easy_handeye calibration for {key!r}: {exc}"
            ) from exc
        paths[key] = source_path
    external = _calibration_entry(data, "external")
    wrist = _calibration_entry(data, "wrist")
    return {
        "external": _normalize_calibration(paths["external"], external),
        "wrist": _normalize_calibration(paths["wrist"], wrist),
        "convention": (
            "The YAML transformation is interpreted as the camera pose in the "
            "YAML base-frame coordinate system. The resulting matrix maps "
            "camera-frame homogeneous points into that base frame."
        ),
    }


def pose7_to_matrix(pose: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    """Convert [x, y, z, qx, qy, qz, qw] to a homogeneous transform."""
    arr = np.asarray(pose, dtype=np.float64)
    if arr.shape != (7,):
        raise ValueError(f"expected tcp_pose shape (7,), got {arr.shape}")
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = quat_xyzw_to_matrix(arr[3:])
    t[:3, 3] = arr[:3]
    return t


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion to a 3x3 rotation matrix."""
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_to_matrix(values: dict[str, Any]) -> np.ndarray:
    """Convert YAML qw/qx/qy/qz fields to a 3x3 rotation matrix."""
    return quat_xyzw_to_matrix(
        np.array(
            [
                float(values["qx"]),
                float(values["qy"]),
                float(values["qz"]),
                float(values["qw"]),
            ],
            dtype=np.float64,
        )
    )


def _calibration_entry(bundle: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in bundle:
        raise ValueError(
            f"Franka calibration missing the {name!r} camera entry (expected "
            "under perception.calibration in the robot config)"
        )
    return bundle[name]


def _normalize_calibration(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    params = data["parameters"]
    transform = data["transformation"]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_wxyz_to_matrix(transform)
    matrix[:3, 3] = [
        float(transform["x"]),
        float(transform["y"]),
        float(transform["z"]),
    ]
    return {
        "path": str(path),
        "source_name": data.get("source_name"),
        "parameters": params,
        "matrix": matrix,
        "base_frame": (
            params.get("robot_effector_frame")
            if params.get("eye_on_hand")
            else params.get("robot_base_frame")
        ),
        "tracking_base_frame": params.get("tracking_base_frame"),
        "eye_on_hand": bool(params.get("eye_on_hand")),
    }
