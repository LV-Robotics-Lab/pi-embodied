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
when the sensor has depth (``None`` otherwise), the wall-clock capture time and a
monotonic capture time (``monotonic_s``) so a reader can tell how old a frame is, and
which clock those came from (``time_source``: the device's frame timestamp where the
source has one, else the time the frame was dequeued; see :data:`TIME_SOURCES`).
``read`` returns the next frame the driver hands out, which for a buffered source
may predate the call; ``read_fresh`` returns a frame captured after the call started.
Its ``intrinsics()`` are the colour intrinsics in the RealSense layout (``fx, fy, ppx,
ppy, width, height, distortion_model, coeffs``) when the device reports them or the
config supplies them, else ``None``. Sources are described by a mapping (a robot
YAML's ``cameras.devices.<name>`` or a ``name=type:source`` flag) and opened with
:func:`open_camera`.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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


#: Where a frame's capture time comes from (``Frame.time_source``):
#:
#: ``device``      the sensor's own frame timestamp mapped to the host clock (RealSense
#:                 ``global_time``: the device clock at the frame, translated by
#:                 librealsense to host time); closest to the exposure.
#: ``backend``     when the host driver received the frame, before any queue
#:                 (RealSense ``system_time``; a V4L2 buffer timestamp, which the
#:                 kernel takes at the start of the frame on ``CLOCK_MONOTONIC``).
#: ``stream_pts``  an RTSP frame's presentation time, mapped to the host clock by the
#:                 smallest arrival-minus-PTS offset seen since the stream opened (so
#:                 frames that sat in a buffer show their extra delay; the network and
#:                 decode latency of the least-delayed frame is not included).
#: ``host``        when the driver handed the frame to us (the dequeue time); a frame
#:                 that waited in a queue looks fresh. The fallback when the source
#:                 reports none of the above.
TIME_SOURCES = ("device", "backend", "stream_pts", "host")


@dataclass(frozen=True)
class Frame:
    """One capture: ``rgb`` uint8 [H, W, 3]; ``depth`` float32 [H, W] metres aligned to
    ``rgb`` (0 = no return) or None for an RGB-only source; ``timestamp_s`` wall clock
    and ``monotonic_s`` (on ``time.monotonic``) of the capture, both from
    ``time_source`` (see :data:`TIME_SOURCES`; ``monotonic_s`` defaults to
    construction)."""

    rgb: np.ndarray
    depth: np.ndarray | None
    timestamp_s: float
    monotonic_s: float = field(default_factory=time.monotonic)
    time_source: str = "host"

    @property
    def has_depth(self) -> bool:
        return self.depth is not None

    def age_s(self, now: float | None = None) -> float:
        """Seconds since the capture (``now`` on ``time.monotonic``)."""
        return float((time.monotonic() if now is None else now) - self.monotonic_s)


def capture_times(
    source_wall_s: float | None,
    source: str,
    *,
    now_wall: float | None = None,
    now_mono: float | None = None,
    max_age_s: float = 60.0,
) -> tuple[float, float, str]:
    """``(timestamp_s, monotonic_s, time_source)`` of a capture whose source reported
    the wall-clock time ``source_wall_s``; a missing, non-finite, future (beyond 50 ms
    of clock jitter) or implausibly old (> ``max_age_s``) value falls back to now and
    ``host``."""
    wall = time.time() if now_wall is None else now_wall
    mono = time.monotonic() if now_mono is None else now_mono
    if source_wall_s is not None and math.isfinite(source_wall_s):
        age = wall - float(source_wall_s)
        if -0.05 <= age <= max_age_s:
            return float(source_wall_s), mono - max(0.0, age), source
    return wall, mono, "host"


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

    def read_fresh(self) -> Frame:
        """A frame captured after this call started. Drivers whose SDK queues a frame
        between reads (a one-deep pipeline queue) discard one frame first; drivers
        that already return the newest frame override this with ``read``."""
        self.read()
        return self.read()

    @abstractmethod
    def close(self) -> None:
        """Release the device; idempotent."""

    def describe(self) -> dict[str, Any]:
        """Static metadata for ``env.get_camera_meta``."""
        return {"camera_type": self.kind, "has_depth": self.has_depth}


#: Distortion models :func:`undistort_pixels` handles (the librealsense names; an
#: OpenCV ``calibrateCamera`` result is ``brown_conrady`` with ``[k1, k2, p1, p2, k3]``).
DISTORTION_MODELS = (
    "none",
    "brown_conrady",
    "modified_brown_conrady",
    "inverse_brown_conrady",
)
#: Iterations of librealsense's inverse (``rs2_deproject_pixel_to_point``, v2.54.1
#: src/rs.cpp: "10 iterations determined empirically").
RS_DEPROJECT_ITERATIONS = 10


def normalize_distortion_model(model: Any) -> str:
    """The librealsense model name without the ``distortion.`` prefix.

    ``str(rs.distortion.inverse_brown_conrady)`` is ``"distortion.inverse_brown_conrady"``;
    calibration sample directories written before the name was normalized (and the
    Franka driver's intrinsics) store that spelling, so it is accepted here."""
    name = "none" if model is None else str(model).strip()
    if name.startswith("distortion."):
        name = name[len("distortion.") :]
    return name or "none"


def _distort(xy: np.ndarray, coeffs: np.ndarray, modified: bool) -> np.ndarray:
    """Brown-Conrady forward model on normalized points ``[N, 2]``: radial
    ``1 + k1 r^2 + k2 r^4 + k3 r^6`` and tangential ``p1, p2``. ``modified`` (the
    librealsense D4xx depth model) applies the tangential terms to the radially
    distorted point; plain Brown-Conrady (OpenCV) applies them to the undistorted one."""
    k1, k2, p1, p2, k3 = (float(c) for c in coeffs[:5])
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    f = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xr, yr = x * f, y * f
    tx, ty = (xr, yr) if modified else (x, y)
    xd = xr + 2.0 * p1 * tx * ty + p2 * (r2 + 2.0 * tx * tx)
    yd = yr + 2.0 * p2 * tx * ty + p1 * (r2 + 2.0 * ty * ty)
    return np.stack([xd, yd], axis=1)


def _rs_inverse_brown_conrady(xy: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """librealsense v2.54.1 ``rs2_deproject_pixel_to_point`` for
    ``RS2_DISTORTION_INVERSE_BROWN_CONRADY``, line for line (src/rs.cpp): a fixed
    point iteration that inverts the forward model ``rs2_project_point_to_pixel``
    applies to this model (radial, then tangential on the radially distorted point).

    librealsense 2.20 applied the polynomial once to the distorted point (the
    coefficients were taken to undistort directly); for a D4xx colour stream that is
    several pixels off near the image edge.
    """
    k1, k2, p1, p2, k3 = (float(c) for c in coeffs[:5])
    xo, yo = xy[:, 0].copy(), xy[:, 1].copy()
    x, y = xo.copy(), yo.copy()
    for _ in range(RS_DEPROJECT_ITERATIONS):
        r2 = x * x + y * y
        icdist = 1.0 / (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2)
        xq = x / icdist
        yq = y / icdist
        delta_x = 2.0 * p1 * xq * yq + p2 * (r2 + 2.0 * xq * xq)
        delta_y = 2.0 * p2 * xq * yq + p1 * (r2 + 2.0 * yq * yq)
        x = (xo - delta_x) * icdist
        y = (yo - delta_y) * icdist
    return np.stack([x, y], axis=1)


def undistort_normalized(
    xy: np.ndarray, model: str, coeffs: Any, iterations: int = 20
) -> np.ndarray:
    """Undistorted normalized image points of distorted ones ``[N, 2]`` under
    ``model``.

    ``inverse_brown_conrady`` (RealSense colour streams) is undistorted exactly as
    librealsense 2.54.1 deprojects it (:func:`_rs_inverse_brown_conrady`, ten
    fixed-point iterations), so a pinhole solve on the result agrees with the SDK's
    own ``rs2_deproject_pixel_to_point``. The forward models (``brown_conrady``,
    ``modified_brown_conrady``) are inverted by fixed-point iteration. Coefficients of
    the inverse model must never be handed to ``cv2.solvePnP`` as-is, which is why
    callers undistort the points here and solve with zero distortion.
    """
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    model = normalize_distortion_model(model)
    if model not in DISTORTION_MODELS:
        raise ValueError(
            f"distortion model {model!r} is not supported (one of {DISTORTION_MODELS}); "
            "kannala_brandt4 / ftheta lenses need their own undistortion"
        )
    c = np.zeros(5)
    given = np.asarray(coeffs if coeffs is not None else [], dtype=np.float64).reshape(
        -1
    )
    if given.size > 5:
        raise ValueError(
            f"expected at most 5 distortion coefficients, got {given.size}"
        )
    c[: given.size] = given
    if model == "none" or not np.any(c):
        return pts.copy()
    if model == "inverse_brown_conrady":
        return _rs_inverse_brown_conrady(pts, c)
    modified = model == "modified_brown_conrady"
    und = pts.copy()
    for _ in range(iterations):
        und = und + (pts - _distort(und, c, modified))
    return und


def undistort_pixels(uv: np.ndarray, intr: dict[str, Any]) -> np.ndarray:
    """Pixel coordinates ``[N, 2]`` corrected for the lens distortion in ``intr``
    (RealSense layout), so a pinhole solve with zero distortion applies to them."""
    K = np.asarray(intrinsic_matrix(intr), dtype=np.float64)
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    xy = (pts - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])
    und = undistort_normalized(
        xy, str(intr.get("distortion_model", "none")), intr.get("coeffs")
    )
    return und * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]


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
    coeffs = [float(c) for c in (value.get("coeffs") or [])]
    if len(coeffs) > 5:
        raise ValueError("intrinsics.coeffs is [k1, k2, p1, p2, k3] (at most 5 values)")
    coeffs += [0.0] * (5 - len(coeffs))
    # Coefficients without a model are an OpenCV calibration (forward Brown-Conrady).
    model = normalize_distortion_model(
        value.get("distortion_model") or ("brown_conrady" if any(coeffs) else "none")
    )
    if model not in DISTORTION_MODELS:
        raise ValueError(
            f"intrinsics.distortion_model must be one of {DISTORTION_MODELS}, got {model!r}"
        )
    out = {
        "width": int(value.get("width", width)),
        "height": int(value.get("height", height)),
        "fx": float(value["fx"]),
        "fy": float(value["fy"]),
        "ppx": float(ppx),
        "ppy": float(ppy),
        "distortion_model": model,
        "coeffs": coeffs,
    }
    if not all(np.isfinite([out["fx"], out["fy"], out["ppx"], out["ppy"], *coeffs])):
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
    timeout_s = float(d.get("read_timeout_s", 5.0))
    if kind == "webcam":
        return WebcamRGB(
            dev["device"],
            width=width,
            height=height,
            fps=fps,
            intrinsics=intr,
            read_timeout_s=timeout_s,
        )
    return RtspRGB(str(dev["url"]), intrinsics=intr, timeout_s=timeout_s)
