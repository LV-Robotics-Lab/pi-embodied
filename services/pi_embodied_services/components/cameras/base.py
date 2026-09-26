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

"""The camera interface every real-robot env server reads through.

A camera yields :class:`Frame` objects: an RGB image, the metric depth aligned to it
when the sensor has depth (``None`` otherwise), and the wall-clock capture time. Its
``intrinsics()`` are the colour intrinsics in the RealSense layout (``fx, fy, ppx, ppy,
width, height``) when the device reports them or the config supplies them, else
``None``. Sources are described by a mapping (a robot YAML's ``cameras.devices.<name>``
or a ``name=type:source`` flag) and opened with :func:`open_camera`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

#: The camera types :func:`open_camera` knows.
CAMERA_TYPES = ("realsense", "webcam", "rtsp")
#: Config keys a camera device may carry.
DEVICE_KEYS = frozenset(
    {
        "type",
        "serial",
        "device",
        "url",
        "width",
        "height",
        "fps",
        "depth",
        "depth_width",
        "depth_height",
        "depth_fps",
        "intrinsics",
        "main",
        "mount",
        "calibration",
    }
)


@dataclass(frozen=True)
class Frame:
    """One capture: ``rgb`` uint8 [H, W, 3]; ``depth`` float32 [H, W] metres aligned to
    ``rgb`` (0 = no return) or None for an RGB-only source; ``timestamp_s`` wall clock."""

    rgb: np.ndarray
    depth: np.ndarray | None
    timestamp_s: float

    @property
    def has_depth(self) -> bool:
        return self.depth is not None


class Camera(ABC):
    """A started camera; ``read`` blocks for the next frame, ``close`` releases it."""

    #: The source type (``realsense`` / ``webcam`` / ``rtsp`` / ``mock``).
    kind: str = ""
    #: Whether ``read`` returns depth.
    has_depth: bool = False

    @abstractmethod
    def intrinsics(self) -> dict[str, Any] | None:
        """Colour intrinsics ``{fx, fy, ppx, ppy, width, height[, distortion_model,
        coeffs]}``, or None when neither the device nor the config supplies them."""

    @abstractmethod
    def read(self) -> Frame:
        """The next frame (blocks up to the source's timeout)."""

    @abstractmethod
    def close(self) -> None:
        """Release the device; idempotent."""

    def describe(self) -> dict[str, Any]:
        """Static metadata for ``env.get_camera_meta``."""
        return {"camera_type": self.kind, "has_depth": self.has_depth}


def intrinsics_from_config(
    value: Any, width: int, height: int
) -> dict[str, Any] | None:
    """A config ``intrinsics: {fx, fy, ppx|cx, ppy|cy}`` as the RealSense layout, or None."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("intrinsics must be a mapping with fx, fy, ppx (cx), ppy (cy)")
    ppx = value.get("ppx", value.get("cx"))
    ppy = value.get("ppy", value.get("cy"))
    missing = [
        k
        for k, v in (
            ("fx", value.get("fx")),
            ("fy", value.get("fy")),
            ("ppx", ppx),
            ("ppy", ppy),
        )
        if v is None
    ]
    if missing:
        raise ValueError(f"intrinsics is missing {missing}")
    out = {
        "width": int(value.get("width", width)),
        "height": int(value.get("height", height)),
        "fx": float(value["fx"]),
        "fy": float(value["fy"]),
        "ppx": float(ppx),
        "ppy": float(ppy),
        "distortion_model": str(value.get("distortion_model", "none")),
        "coeffs": [float(c) for c in value.get("coeffs", [0.0] * 5)],
    }
    if not all(np.isfinite([out["fx"], out["fy"], out["ppx"], out["ppy"]])):
        raise ValueError("intrinsics must be finite")
    return out


def intrinsic_matrix(intr: dict[str, Any]) -> list[list[float]]:
    """3x3 K of RealSense-layout intrinsics."""
    return [
        [float(intr["fx"]), 0.0, float(intr["ppx"])],
        [0.0, float(intr["fy"]), float(intr["ppy"])],
        [0.0, 0.0, 1.0],
    ]


def parse_source(spec: str) -> dict[str, Any]:
    """``type:source`` (a ``--cameras`` flag entry) as a device mapping.

    ``realsense:<serial>`` (empty serial = the first device), ``webcam:<index or
    /dev/videoN>``, ``rtsp://...`` or ``rtsp:<url>``.
    """
    s = spec.strip()
    if s.startswith("rtsp://") or s.startswith("rtsps://"):
        return {"type": "rtsp", "url": s}
    kind, sep, source = s.partition(":")
    kind = kind.strip().lower()
    if not sep or kind not in CAMERA_TYPES:
        raise ValueError(
            f"camera source {spec!r} must be realsense:<serial>, webcam:<index|/dev/videoN> "
            "or rtsp://<url>"
        )
    source = source.strip()
    if kind == "realsense":
        return {"type": "realsense", "serial": source or None}
    if kind == "webcam":
        if not source:
            raise ValueError("webcam needs a device index or /dev/videoN path")
        return {"type": "webcam", "device": int(source) if source.isdigit() else source}
    return {"type": "rtsp", "url": source}


def parse_sources(flag: str) -> dict[str, dict[str, Any]]:
    """``name=type:source,name=...`` as ``{name: device mapping}``; the first entry is
    marked ``main`` unless another says so."""
    devices: dict[str, dict[str, Any]] = {}
    for entry in filter(None, (e.strip() for e in flag.split(","))):
        name, sep, source = entry.partition("=")
        name = name.strip()
        if not sep or not name or not source.strip():
            raise ValueError(f"camera entry {entry!r} must be name=type:source")
        if name in devices:
            raise ValueError(f"camera {name!r} is named twice")
        devices[name] = parse_source(source)
    if devices and not any(d.get("main") for d in devices.values()):
        next(iter(devices.values()))["main"] = True
    return devices


def validate_device(name: str, dev: Any) -> dict[str, Any]:
    """Check one ``cameras.devices.<name>`` mapping and fill its type."""
    if not isinstance(dev, dict):
        raise ValueError(f"cameras.devices.{name} must be a mapping")
    unknown = sorted(set(dev) - DEVICE_KEYS)
    if unknown:
        raise ValueError(
            f"cameras.devices.{name}: unknown keys {unknown}; valid: {sorted(DEVICE_KEYS)}"
        )
    out = dict(dev)
    kind = str(out.get("type") or "realsense").lower()
    if kind not in CAMERA_TYPES:
        raise ValueError(f"cameras.devices.{name}.type must be one of {CAMERA_TYPES}")
    out["type"] = kind
    if kind == "webcam" and out.get("device") is None:
        raise ValueError(
            f"cameras.devices.{name}: webcam needs `device` (index or path)"
        )
    if kind == "rtsp" and not out.get("url"):
        raise ValueError(f"cameras.devices.{name}: rtsp needs `url`")
    mount = out.get("mount")
    if mount not in (None, "fixed", "wrist"):
        raise ValueError(f"cameras.devices.{name}.mount must be fixed or wrist")
    intrinsics_from_config(
        out.get("intrinsics"), int(out.get("width", 640)), int(out.get("height", 480))
    )
    return out


def open_camera(
    name: str, dev: dict[str, Any], defaults: dict[str, Any] | None = None
) -> Camera:
    """Open the camera a device mapping describes (imports its driver lazily)."""
    d = defaults or {}
    dev = validate_device(name, dev)
    width = int(dev.get("width", d.get("width", 640)))
    height = int(dev.get("height", d.get("height", 480)))
    fps = int(dev.get("fps", d.get("fps", 30)))
    kind = dev["type"]
    if kind == "realsense":
        from pi_embodied_services.components.cameras.realsense import RealSenseRGBD

        return RealSenseRGBD(
            dev.get("serial"),
            width=width,
            height=height,
            fps=fps,
            depth=bool(dev.get("depth", True)),
            depth_width=int(dev.get("depth_width", 0)),
            depth_height=int(dev.get("depth_height", 0)),
            depth_fps=int(dev.get("depth_fps", 0)),
        )
    from pi_embodied_services.components.cameras.webcam import RtspRGB, WebcamRGB

    intr = intrinsics_from_config(dev.get("intrinsics"), width, height)
    if kind == "webcam":
        return WebcamRGB(
            dev["device"], width=width, height=height, fps=fps, intrinsics=intr
        )
    return RtspRGB(
        str(dev["url"]), intrinsics=intr, timeout_s=float(d.get("read_timeout_s", 5.0))
    )
