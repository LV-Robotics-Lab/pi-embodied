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
# Modified by pi-embodied: reduced to the calibration and base-frame helpers
# used by the dual-Franka env server and by pi-embodied's dual-Franka robot
# (via ``python -c``); the planner-side back-projection/segmentation tools
# (which need RPent's session/toolkit) are omitted. Function bodies are
# verbatim from RPent.

"""Dual-Franka calibration and base-frame helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pi_embodied_services.robots.franka.runtime_config import (
    _calibration_mapping_from,
    get_robot_config_path,
    load_easy_handeye_yaml,
    load_mapping,
)
from pi_embodied_services.utils.transforms import (
    invert_transform,
    transform_points,
    transform_pose,
)

ROBOT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "example.yaml"


def _load_perception_config() -> dict[str, Any]:
    """Load the perception section from the selected robot configuration.

    Holds the machine config ``easy_handeye`` does not produce: the tabletop
    ``localization_validity`` bounds and the inter-base ``base_frames``.
    """
    raw = load_mapping(get_robot_config_path(ROBOT_CONFIG_PATH))
    perception = raw.get("perception")
    if not isinstance(perception, dict):
        raise ValueError("Robot configuration missing the 'perception' section")
    return perception


def _projection_cameras() -> dict[str, dict[str, Any]]:
    perception = _load_perception_config()
    configured = perception.get("projection_views") or {}
    return _coerce_projection_views(configured)


def _coerce_projection_views(configured: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(configured, dict):
        raise ValueError("perception.projection_views must be a mapping")
    cameras: dict[str, dict[str, Any]] = {}
    for alias, raw_config in configured.items():
        if not isinstance(raw_config, dict):
            raise ValueError(f"perception.projection_views.{alias} must be a mapping")
        alias_str = str(alias)
        cameras[alias_str] = {
            "raw_key": str(raw_config.get("raw_key", f"{alias_str}_rgb")),
            "calibration_key": str(
                raw_config.get("calibration_key", f"{alias_str}_camera")
            ),
            "display_name": str(raw_config.get("display_name", alias_str)),
        }
    return cameras


def load_calibration_bundle() -> dict[str, Any]:
    """Load the dual-Franka perception calibration as one bundle.

    Combines the ``easy_handeye`` hand-eye transforms from the robot config's
    ``perception.calibration`` YAML mapping with the rest of that config's
    ``perception`` section (``base_frames``, ``localization_validity``).
    """
    perception = _load_perception_config()
    data = _load_calibration_from_config(perception)
    bundle = dict(data)
    bundle["base_frames"] = perception.get("base_frames") or {}
    for camera_key, validity in (perception.get("localization_validity") or {}).items():
        if camera_key in bundle:
            bundle[camera_key] = {
                **bundle[camera_key],
                "localization_validity": validity,
            }
    return bundle


def _load_calibration_from_config(perception: dict[str, Any]) -> dict[str, Any]:
    """Load hand-eye transforms from the config's easy_handeye YAML mapping."""
    sources = _calibration_mapping_from(perception)
    if not sources:
        raise ValueError(
            "no hand-eye calibration configured: list easy_handeye YAMLs "
            "under perception.calibration in the robot config "
            f"({get_robot_config_path()})"
        )
    data: dict[str, Any] = {}
    for camera_key, source in sources.items():
        try:
            data[camera_key] = load_easy_handeye_yaml(source)
        except ValueError as exc:
            raise ValueError(
                f"cannot load easy_handeye calibration for {camera_key!r}: {exc}"
            ) from exc
    return data


def transform_point_between_base_frames(
    point: Any,
    *,
    target: str,
    source: str,
    calibration: dict[str, Any] | None = None,
) -> np.ndarray:
    point_arr = np.asarray(point, dtype=np.float64).reshape(3)
    if target == source:
        return point_arr
    t_target_source = _base_frame_transform(
        calibration or load_calibration_bundle(),
        target=target,
        source=source,
    )
    return transform_points(t_target_source, point_arr)


def transform_pose_between_base_frames(
    pose: Any,
    *,
    target: str,
    source: str,
    calibration: dict[str, Any] | None = None,
) -> np.ndarray:
    """Transform an xyz+xyzw TCP pose between robot base frames."""
    pose_arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose_arr.size < 7:
        raise ValueError(
            f"expected xyz+quat pose with at least 7 values, got {pose_arr.shape}"
        )
    pose_arr = pose_arr[:7].copy()
    if target == source:
        return pose_arr
    t_target_source = _base_frame_transform(
        calibration or load_calibration_bundle(),
        target=target,
        source=source,
    )
    return transform_pose(t_target_source, pose_arr)


def _transform_to_matrix(transform: dict[str, Any]) -> np.ndarray:
    if "matrix" in transform:
        mat = np.asarray(transform["matrix"], dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError(f"expected 4x4 transform matrix, got {mat.shape}")
        return mat
    qw = float(transform["qw"])
    qx = float(transform["qx"])
    qy = float(transform["qy"])
    qz = float(transform["qz"])
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / np.linalg.norm(q)
    qw, qx, qy, qz = q
    rot = np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = [float(transform["x"]), float(transform["y"]), float(transform["z"])]
    return mat


def _base_frame_transform(
    calibration: dict[str, Any],
    *,
    target: str,
    source: str,
) -> np.ndarray:
    frames = calibration.get("base_frames") or {}
    key = f"T_{target}_{source}"
    if isinstance(frames.get(key), dict):
        return _transform_to_matrix(frames[key])
    inverse_key = f"T_{source}_{target}"
    if isinstance(frames.get(inverse_key), dict):
        return invert_transform(_transform_to_matrix(frames[inverse_key]))
    raise ValueError(f"missing base-frame transform {key}")
