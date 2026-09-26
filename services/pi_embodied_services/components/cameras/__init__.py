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

"""Shared camera layer for the real-robot env servers (Franka, UR5e).

One interface (:class:`Camera` -> :class:`Frame`: RGB, aligned metric depth where the
sensor has it, timestamp; ``intrinsics()`` where known) over three sources:
RealSense D400/L515 (``realsense.py``), USB webcams and RTSP streams (``webcam.py``).
Drivers import their SDKs lazily; :func:`open_camera` builds one from a config
mapping and :func:`parse_sources` from a ``name=type:source,...`` flag.
"""

from pi_embodied_services.components.cameras.base import (
    CAMERA_TYPES,
    DISTORTION_MODELS,
    Camera,
    Frame,
    intrinsic_matrix,
    intrinsics_from_config,
    normalize_distortion_model,
    open_camera,
    parse_source,
    parse_sources,
    undistort_normalized,
    undistort_pixels,
    validate_device,
)

__all__ = [
    "CAMERA_TYPES",
    "DISTORTION_MODELS",
    "Camera",
    "Frame",
    "intrinsic_matrix",
    "intrinsics_from_config",
    "normalize_distortion_model",
    "open_camera",
    "parse_source",
    "parse_sources",
    "undistort_normalized",
    "undistort_pixels",
    "validate_device",
]
