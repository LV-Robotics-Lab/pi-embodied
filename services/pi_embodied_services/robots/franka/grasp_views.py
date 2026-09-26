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

"""The Franka servers' camera views for the grasp planner (``utils/grasp.GraspPlanner``).

A view is one camera of the current observation with its calibration: ``{"rgb", "depth"
(metres), "intrinsic_K", "extrinsic_cam2world"}``. The single arm's ``third_person`` camera is
fixed in the base frame (its easy_handeye ``external`` transform); the ``wrist`` camera rides
on the TCP (``T_base_tcp @ T_tcp_camera``, with the TCP read from the robot state at the same
time as the frames). The dual arm's registered projection views are fixed in ``right_base``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from pi_embodied_services.robots.franka.perception import pose7_to_matrix

#: camera alias -> (image key, depth key, index into a stacked extra view or None): the franka
#: servers' observation layout (main = wrist, extra_view[0] = external).
FRANKA_CAMERAS: dict[str, tuple[str, str, int | None]] = {
    "wrist": ("main_images", "main_depths", None),
    "third_person": ("extra_view_images", "extra_view_depths", 0),
}


def franka_intrinsics(meta: Any, key: str) -> np.ndarray | None:
    """``intrinsic_K`` of observation key ``main`` / ``extra_N`` from franka camera meta."""
    if not isinstance(meta, dict) or "error" in meta:
        return None
    name = (meta.get("observation_camera_map") or {}).get(key)
    cam = (meta.get("cameras") or {}).get(name) if name else None
    K = (cam or {}).get("intrinsic_K")
    if K is None:
        return None
    K = np.asarray(K, dtype=np.float64)
    return K if K.shape == (3, 3) and np.all(np.isfinite(K)) else None


def _pick(obs: dict[str, Any], key: str, index: int | None) -> np.ndarray:
    value = obs.get(key)
    if value is None:
        raise ValueError(f"the observation has no {key!r}")
    array = np.asarray(value)
    if index is not None:
        if array.ndim < 1 or index >= array.shape[0]:
            raise ValueError(f"the observation's {key!r} has no view {index}")
        array = array[index]
    return array


def franka_view(
    backend: Any, calibration: Callable[[], dict[str, Any]]
) -> Callable[[str], dict[str, Any]]:
    """``view(camera)`` for the single Franka (cameras ``wrist`` / ``third_person``)."""

    def view(camera: str) -> dict[str, Any]:
        if camera not in FRANKA_CAMERAS:
            raise ValueError(
                f"unknown camera {camera!r}; one of {sorted(FRANKA_CAMERAS)}"
            )
        image_key, depth_key, index = FRANKA_CAMERAS[camera]
        obs = backend.get_observation()
        rgb = _pick(obs, image_key, index)
        depth = np.asarray(_pick(obs, depth_key, index), dtype=np.float32)
        depth = depth.reshape(depth.shape[-2:]) if depth.ndim == 3 else depth
        K = franka_intrinsics(
            backend.get_camera_meta(), "main" if index is None else f"extra_{index}"
        )
        if K is None:
            raise ValueError(f"no intrinsics for camera {camera!r}")
        cal = calibration()
        if camera == "wrist":
            tcp = backend.get_robot_state()["raw_base_state"]["tcp_pose"]
            cam2world = pose7_to_matrix(tcp) @ np.asarray(
                cal["wrist"]["matrix"], dtype=np.float64
            )
        else:
            cam2world = np.asarray(cal["external"]["matrix"], dtype=np.float64)
        return {
            "rgb": np.ascontiguousarray(rgb[..., :3], dtype=np.uint8),
            "depth": np.ascontiguousarray(depth),
            "intrinsic_K": K,
            "extrinsic_cam2world": cam2world,
        }

    return view


def dual_franka_view(
    backend: Any,
    projection_views: dict[str, dict[str, Any]],
    camera_transform: Callable[[str], np.ndarray],
) -> Callable[[str], dict[str, Any]]:
    """``view(camera)`` for the dual Franka: a registered projection view (``d455``, ``base``),
    its RealSense colour intrinsics from the camera meta and ``T_right_base_camera`` from the
    calibration bundle (``camera_transform(calibration_key)``)."""

    def view(camera: str) -> dict[str, Any]:
        cfg = projection_views.get(camera)
        if cfg is None:
            raise ValueError(
                f"unknown camera {camera!r}; registered: {sorted(projection_views)}"
            )
        raw_key = cfg["raw_key"]
        obs = backend.get_observation()
        frames = obs.get("raw_camera_frames") or {}
        depths = obs.get("raw_camera_depths") or {}
        if raw_key not in frames or raw_key not in depths:
            raise ValueError(
                f"the observation has no RGB-D frame for {camera!r} ({raw_key})"
            )
        meta = backend.get_camera_meta() or {}
        intr = (meta.get(raw_key) or meta.get(camera) or {}).get("color_intrinsics")
        if not intr:
            raise ValueError(f"no colour intrinsics for camera {camera!r}")
        K = np.array(
            [
                [float(intr["fx"]), 0.0, float(intr.get("ppx", intr.get("cx")))],
                [0.0, float(intr["fy"]), float(intr.get("ppy", intr.get("cy")))],
                [0.0, 0.0, 1.0],
            ]
        )
        return {
            "rgb": np.ascontiguousarray(
                np.asarray(frames[raw_key])[..., :3], dtype=np.uint8
            ),
            "depth": np.ascontiguousarray(
                np.asarray(depths[raw_key], dtype=np.float32)
            ),
            "intrinsic_K": K,
            "extrinsic_cam2world": np.asarray(
                camera_transform(cfg["calibration_key"]), dtype=np.float64
            ),
        }

    return view


__all__ = ["dual_franka_view", "franka_view"]
