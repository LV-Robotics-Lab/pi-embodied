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

"""The geometric toolset an env server composes over its calibrated RGB-D cameras and its
gripper pose (OpenETA ``openeta-for-codex``: ``tools/embodied_mcp_server.py`` ``mark_point`` /
``move_to``, ``tools/pointcloud_pose_marking.py``).

- **Point-cloud views** (``env.point_views``): the cameras' depth, back-projected and fused into
  one world-frame cloud, drawn as three orthographic views (``pointcloud_top`` right +x / up +y
  seen from above, ``pointcloud_front`` right +x / up +z seen from -y, ``pointcloud_side``
  right +y / up +z seen from +x), each with a metric grid. Every pixel of a view is two world
  coordinates, whether a point was drawn there or not.
- **Marked points** (``env.mark_point``): a click on a camera image is the first visible
  surface at that pixel (its depth); a click on an orthographic view fixes that view's two axes
  and waits for a click on a complementary view (another pair of axes), which fixes the third:
  two views define any point, on a surface or in free space. The shared axis must agree within
  15 mm. Solved points keep their world coordinates for the episode; a pending half expires when
  the robot moves.
- **Grip-site targets** (``env.grip_target`` / ``env.move_grip``): an absolute position, a
  marked point or a delta (world frame, or the gripper's own ``[JAW, LAT, APP]`` axes), and an
  orientation given as the approach direction (the grip frame's +Z) and/or the jaw direction
  (its +X, the axis the fingers close along); omitted parts keep the current pose. A close is
  previewed first: the target and its finger-pad closing corridor drawn on zoomed orthographic
  views and the first camera, frozen under a ``preview_id`` that executes it unchanged, and only
  while the gripper has not moved since.
- **Residual and contact** (``env.grip_state``): after a motion, the grip site's actual position
  and orientation, the remaining delta to the target, and the robot's current contacts (a
  simulator's MuJoCo contacts; a real robot has none to report), drawn on the first camera.

:class:`GripGeometry` owns the state (marks, the rendered views' mapping, the pending close) and
takes the robot through callbacks, so the same toolset serves LIBERO, where the env server also
executes ``env.move_grip`` for code mode, and a real arm, whose client executes a planned target
through its own bounded motion primitives. The grip frame is the robot's tool frame turned by a
fixed rotation (``frame``): +X the jaw axis, +Z the approach. Nothing here imports torch.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

AXES = ("x", "y", "z")
#: Orthographic views: the image's horizontal and vertical world axes, the depth axis and which
#: end of it is seen (+1: its largest values), so no view is mirrored.
ORTHO = {
    "top": {"h": 0, "v": 1, "depth": 2, "near": 1, "seen_from": "+z (looking down)"},
    "front": {"h": 0, "v": 2, "depth": 1, "near": -1, "seen_from": "-y"},
    "side": {"h": 1, "v": 2, "depth": 0, "near": 1, "seen_from": "+x"},
}
SCENE_VIEWS = tuple(f"pointcloud_{k}" for k in ORTHO)
PREVIEW_VIEWS = tuple(f"preview_{k}" for k in ORTHO)
#: Two orthographic clicks must agree on their shared axis within this (OpenETA's 15 mm).
CONSISTENCY_TOLERANCE_M = 0.015
#: The scene views' longest side, and the zoomed preview views' size and half extent.
SCENE_PX = 512
PREVIEW_PX = 384
PREVIEW_HALF_M = 0.12
#: A pending close preview executes only while the grip site stays within these of its base.
PREVIEW_BASE_TOL_M = 0.002
PREVIEW_BASE_TOL_DEG = 1.0
#: Nominal finger-pad separation drawn for an open and a closed preview (Panda: 80 mm / 8 mm).
OPEN_APERTURE_M = 0.08
CLOSED_APERTURE_M = 0.008
#: Servo defaults of a grip-site move: position and orientation tolerance, step budget.
MOVE_TOL_M = 0.005
MOVE_TOL_RAD = 0.05
MOVE_MAX_STEPS = 200

MARK_COLORS = (
    (255, 80, 70),
    (80, 210, 255),
    (110, 255, 120),
    (255, 210, 70),
    (210, 120, 255),
    (255, 150, 80),
)
GRIP_COLOR = (255, 60, 220)
TARGET_COLOR = (255, 235, 60)
PAD_COLOR = (255, 150, 40)
CONTACT_COLOR = (60, 255, 255)
AXIS_COLORS = ((255, 80, 70), (80, 220, 100), (80, 140, 255))


class GeometryError(ValueError):
    """A request the toolset refuses (a bad view, a stale view, an inconsistent click...)."""


# ---- small math --------------------------------------------------------------------------


def _vec3(value: Any, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise GeometryError(f"{name} must be 3 finite numbers, got {value!r}")
    return a


def _unit(value: Any, name: str) -> np.ndarray:
    a = _vec3(value, name)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        raise GeometryError(f"{name} must not be the zero vector")
    return a / n


def quat_to_matrix(q_xyzw: Any) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(r: np.ndarray) -> list[float]:
    """xyzw of a rotation matrix (w >= 0)."""
    m = np.asarray(r, dtype=np.float64)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
            0.25 * s,
        ]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[2, 1] - m[1, 2]) / s,
        ]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
            (m[0, 2] - m[2, 0]) / s,
        ]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
            (m[1, 0] - m[0, 1]) / s,
        ]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return [float(v) for v in q]


def rotvec_of(r: np.ndarray) -> np.ndarray:
    """The rotation vector (axis x angle, angle in [0, pi]) of a rotation matrix."""
    m = np.asarray(r, dtype=np.float64)
    angle = math.acos(max(-1.0, min(1.0, (np.trace(m) - 1) / 2)))
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:
        cols = m + np.eye(3)
        axis = cols[:, int(np.argmax(np.linalg.norm(cols, axis=0)))]
        return axis / np.linalg.norm(axis) * angle
    w = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    return w * angle / (2 * math.sin(angle))


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """The angle (rad) between two rotation matrices."""
    return float(np.linalg.norm(rotvec_of(np.asarray(b) @ np.asarray(a).T)))


def resolve_rotation(
    current: np.ndarray, approach: Any = None, jaw: Any = None
) -> tuple[np.ndarray, str]:
    """The grip frame (columns: jaw +X, lateral +Y, approach +Z) for an approach and/or jaw
    direction; an omitted one follows the current frame. The jaw direction's sign is free (the
    gripper is symmetric): the one nearer the current frame is taken."""
    cur = np.asarray(current, dtype=np.float64)
    if approach is None and jaw is None:
        return cur.copy(), "current"
    z = _unit(approach, "approach") if approach is not None else cur[:, 2]

    def frame(x_hint: np.ndarray) -> np.ndarray | None:
        x = x_hint - z * float(np.dot(x_hint, z))
        if np.linalg.norm(x) < 1e-6:
            return None
        x = x / np.linalg.norm(x)
        return np.column_stack([x, np.cross(z, x), z])

    if jaw is not None:
        hint = _unit(jaw, "jaw")
        options = [f for f in (frame(hint), frame(-hint)) if f is not None]
        if not options:
            raise GeometryError("jaw must not be parallel to the approach direction")
        best = max(options, key=lambda f: float(np.trace(cur.T @ f)))
        return best, ("approach_and_jaw" if approach is not None else "jaw")
    r = frame(cur[:, 0])
    if r is None:
        # The current jaw is along the new approach: keep the current lateral axis instead.
        y = cur[:, 1] - z * float(np.dot(cur[:, 1], z))
        y = y / np.linalg.norm(y)
        r = np.column_stack([np.cross(y, z), y, z])
    return r, "approach"


# ---- images ------------------------------------------------------------------------------


def png_b64(rgb: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _font():
    from PIL import ImageFont

    return ImageFont.load_default()


def cloud_from_views(
    views: list[dict[str, Any]], max_depth: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """World points and colours of every valid depth pixel of the views (rgb, depth in metres,
    intrinsic_K, extrinsic_cam2world); plain back-projection, no object knowledge."""
    pts, cols = [], []
    for view in views:
        depth = np.asarray(view["depth"], dtype=np.float64)
        depth = depth.reshape(depth.shape[-2:])
        rgb = np.asarray(view["rgb"], dtype=np.uint8)[..., :3]
        K = np.asarray(view["intrinsic_K"], dtype=np.float64)
        T = np.asarray(view["extrinsic_cam2world"], dtype=np.float64)
        valid = np.isfinite(depth) & (depth > 0) & (depth < max_depth)
        rows, cs = np.nonzero(valid)
        if not len(rows):
            continue
        z = depth[rows, cs]
        cam = np.stack(
            [(cs - K[0, 2]) * z / K[0, 0], (rows - K[1, 2]) * z / K[1, 1], z], axis=1
        )
        pts.append(cam @ T[:3, :3].T + T[:3, 3])
        cols.append(rgb[rows, cs])
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(pts), np.concatenate(cols)


def scene_bounds(
    points: np.ndarray,
    envelope: np.ndarray | None,
    include: list[np.ndarray],
    *,
    quantile: float = 0.01,
    pad: float = 0.03,
    min_extent: float = 0.10,
) -> np.ndarray:
    """A 3x2 world box of the cloud inside ``envelope`` (quantile-trimmed, padded), grown to
    hold ``include`` (the grip site) and at least ``min_extent`` per axis."""
    p = np.asarray(points, dtype=np.float64)
    if envelope is not None and len(p):
        env = np.asarray(envelope, dtype=np.float64)
        p = p[np.all((p >= env[:, 0]) & (p <= env[:, 1]), axis=1)]
    if len(p) >= 20:
        lo = np.quantile(p, quantile, axis=0) - pad
        hi = np.quantile(p, 1 - quantile, axis=0) + pad
    elif include:
        lo = np.min(include, axis=0) - 0.3
        hi = np.max(include, axis=0) + 0.3
    else:
        raise GeometryError("no depth points to draw the point-cloud views from")
    for q in include:
        lo = np.minimum(lo, np.asarray(q) - 0.05)
        hi = np.maximum(hi, np.asarray(q) + 0.05)
    grow = np.maximum(0.0, min_extent - (hi - lo)) / 2
    return np.stack([lo - grow, hi + grow], axis=1)


@dataclass(frozen=True)
class OrthoSpec:
    """One orthographic view: pixel (x, y) is world ``h = h0 + x * m``, ``v = v1 - y * m``."""

    name: str
    kind: str
    h0: float
    v1: float
    m: float
    width: int
    height: int

    @property
    def axes(self) -> tuple[int, int]:
        o = ORTHO[self.kind]
        return o["h"], o["v"]

    def to_world(self, x: float, y: float) -> dict[int, float]:
        h, v = self.axes
        return {h: self.h0 + float(x) * self.m, v: self.v1 - float(y) * self.m}

    def to_pixel(self, p: np.ndarray) -> tuple[float, float]:
        h, v = self.axes
        return (float(p[h]) - self.h0) / self.m, (self.v1 - float(p[v])) / self.m

    def describe(self) -> dict[str, Any]:
        h, v = self.axes
        return {
            "view": self.name,
            "width": self.width,
            "height": self.height,
            "right": f"+{AXES[h]}",
            "up": f"+{AXES[v]}",
            "seen_from": ORTHO[self.kind]["seen_from"],
            f"{AXES[h]}_range_m": [
                round(self.h0, 4),
                round(self.h0 + (self.width - 1) * self.m, 4),
            ],
            f"{AXES[v]}_range_m": [
                round(self.v1 - (self.height - 1) * self.m, 4),
                round(self.v1, 4),
            ],
            "m_per_px": round(self.m, 6),
        }


def make_spec(name: str, kind: str, bounds: np.ndarray, px: int) -> OrthoSpec:
    o = ORTHO[kind]
    b = np.asarray(bounds, dtype=np.float64)
    eh, ev = b[o["h"], 1] - b[o["h"], 0], b[o["v"], 1] - b[o["v"], 0]
    m = max(eh, ev) / (px - 1)
    return OrthoSpec(
        name,
        kind,
        float(b[o["h"], 0]),
        float(b[o["v"], 1]),
        float(m),
        int(round(eh / m)) + 1,
        int(round(ev / m)) + 1,
    )


def render_ortho(
    points: np.ndarray, colors: np.ndarray, spec: OrthoSpec, splat: int = 1
) -> np.ndarray:
    """The cloud seen along the view's depth axis (nearest point per pixel), each point also
    filling the empty pixels within ``splat`` of it (nearer first), on a dark background with a
    5 cm grid labelled every 10 cm."""
    from PIL import Image, ImageDraw

    h, v = spec.axes
    o = ORTHO[spec.kind]
    W, H = spec.width, spec.height
    canvas = np.full((H, W, 3), 18, dtype=np.uint8)
    p = np.asarray(points, dtype=np.float64)
    if len(p):
        u = np.rint((p[:, h] - spec.h0) / spec.m).astype(np.int64)
        y = np.rint((spec.v1 - p[:, v]) / spec.m).astype(np.int64)
        inside = (u >= 0) & (u < W) & (y >= 0) & (y < H)
        u, y = u[inside], y[inside]
        key = p[inside, o["depth"]] * o["near"]
        col = np.clip(
            np.asarray(colors, dtype=np.float64)[inside] * 1.4 + 20, 0, 255
        ).astype(np.uint8)
        pix = y * W + u
        order = np.lexsort((-key, pix))
        first = np.r_[True, pix[order][1:] != pix[order][:-1]] if len(order) else []
        chosen = order[first]
        filled = np.zeros((H, W), dtype=bool)
        offsets = sorted(
            (
                (dy, dx)
                for dy in range(-splat, splat + 1)
                for dx in range(-splat, splat + 1)
            ),
            key=lambda o: o[0] * o[0] + o[1] * o[1],
        )
        for dy, dx in offsets:
            yy = np.clip(y[chosen] + dy, 0, H - 1)
            uu = np.clip(u[chosen] + dx, 0, W - 1)
            free = ~filled[yy, uu]
            canvas[yy[free], uu[free]] = col[chosen][free]
            filled[yy[free], uu[free]] = True
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = _font()
    for axis, horizontal in ((h, True), (v, False)):
        lo = spec.h0 if horizontal else spec.v1 - (H - 1) * spec.m
        hi = spec.h0 + (W - 1) * spec.m if horizontal else spec.v1
        k = math.ceil(lo / 0.05 - 1e-9)
        while k * 0.05 <= hi + 1e-9:
            val = k * 0.05
            major = k % 2 == 0
            shade = (70, 70, 70) if major else (38, 38, 38)
            if horizontal:
                px = (val - spec.h0) / spec.m
                draw.line([(px, 0), (px, H - 1)], fill=shade)
                if major:
                    draw.text(
                        (px + 2, H - 12), f"{val:.1f}", fill=(170, 170, 170), font=font
                    )
            else:
                py = (spec.v1 - val) / spec.m
                draw.line([(0, py), (W - 1, py)], fill=shade)
                if major:
                    draw.text(
                        (2, py + 1), f"{val:.1f}", fill=(170, 170, 170), font=font
                    )
            k += 1
    title = f"{spec.name}  right +{AXES[h].upper()}  up +{AXES[v].upper()}  (world m)"
    draw.rectangle((0, 0, min(W, 8 + 6 * len(title)), 13), fill=(0, 0, 0))
    draw.text((3, 1), title, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def project(view: dict[str, Any], p: np.ndarray) -> tuple[float, float] | None:
    """Pixel (x, y) of world point ``p`` in a camera view, or None behind the camera."""
    T = np.asarray(view["extrinsic_cam2world"], dtype=np.float64)
    K = np.asarray(view["intrinsic_K"], dtype=np.float64)
    c = np.linalg.inv(T) @ np.r_[np.asarray(p, dtype=np.float64), 1.0]
    if c[2] <= 1e-6:
        return None
    return float(K[0, 0] * c[0] / c[2] + K[0, 2]), float(
        K[1, 1] * c[1] / c[2] + K[1, 2]
    )


def surface_point(
    view: dict[str, Any], x: int, y: int, radius: int = 3
) -> np.ndarray | None:
    """World xyz of the first visible surface at pixel (x, y) of a camera view: its depth, or
    the median of the valid depths within ``radius`` px when the pixel itself has none."""
    depth = np.asarray(view["depth"], dtype=np.float64)
    depth = depth.reshape(depth.shape[-2:])
    H, W = depth.shape
    if not (0 <= x < W and 0 <= y < H):
        raise GeometryError(f"pixel ({x}, {y}) is outside the {W}x{H} image")
    z = depth[y, x]
    if not (np.isfinite(z) and z > 0):
        win = depth[
            max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1
        ]
        ok = win[np.isfinite(win) & (win > 0)]
        if not len(ok):
            return None
        z = float(np.median(ok))
    K = np.asarray(view["intrinsic_K"], dtype=np.float64)
    T = np.asarray(view["extrinsic_cam2world"], dtype=np.float64)
    cam = np.array([(x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z, 1.0])
    return (T @ cam)[:3]


# ---- the MuJoCo side (a simulator's worker process) --------------------------------------


def mujoco_grip_state(
    sim: Any,
    site: str = "gripper0_grip_site",
    pads: tuple[str, str] = (
        "gripper0_finger1_pad_collision",
        "gripper0_finger2_pad_collision",
    ),
    robot_prefixes: tuple[str, ...] = ("robot0_", "gripper0_"),
) -> dict[str, Any]:
    """The grip site's world pose, the finger pads' inner contact faces in the site frame, and
    the current contacts between the robot and anything else (at most 6, 4 mm apart), from a
    MuJoCo sim (robosuite's ``MjSim``: ``sim.model`` / ``sim.data``). A name the model lacks is
    looked up by its ends (robosuite 1.5 names them ``gripper0_right_grip_site`` and so on).
    Never raises: a failure is ``{"error": ...}`` (a worker's ``env_call`` has no try/except)."""

    def find(kind: str, name: str) -> int:
        try:
            return int(getattr(model, f"{kind}_name2id")(name))
        except Exception:  # noqa: BLE001
            prefix, _, rest = name.partition("_")
            id2name = getattr(model, f"{kind}_id2name")
            for i in range(int(getattr(model, f"n{kind}"))):
                n = id2name(i) or ""
                if n.startswith(prefix + "_") and n.endswith(rest):
                    return i
            raise ValueError(f"the model has no {kind} {prefix}_*{rest}") from None

    try:
        model, data = sim.model, sim.data
        sid = find("site", site)
        xyz = np.asarray(data.site_xpos[sid], dtype=np.float64).reshape(3)
        R = np.asarray(data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
        pad_local = []
        for name in pads:
            gid = find("geom", name)
            center = np.asarray(data.geom_xpos[gid], dtype=np.float64).reshape(3)
            gR = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
            size = np.asarray(model.geom_size[gid], dtype=np.float64)[:3]
            toward = xyz - center
            n = float(np.linalg.norm(toward))
            direction = toward / n if n > 1e-12 else np.zeros(3)
            # The pad is a box: its support distance toward the grip site.
            face = center + direction * float(np.sum(np.abs(gR.T @ direction) * size))
            pad_local.append((R.T @ (face - xyz)).tolist())

        def name_of(gid: int) -> str:
            try:
                n = model.geom_id2name(int(gid))
            except Exception:  # noqa: BLE001
                n = None
            return str(n) if n else f"geom:{gid}"

        found: list[tuple[float, dict[str, Any]]] = []
        for i in range(int(data.ncon)):
            c = data.contact[i]
            pos = np.asarray(c.pos, dtype=np.float64).reshape(3)
            names = [name_of(c.geom1), name_of(c.geom2)]
            robot = [n.startswith(robot_prefixes) for n in names]
            # The robot touching the world; not the robot touching itself.
            if sum(robot) != 1 or not np.all(np.isfinite(pos)):
                continue
            found.append(
                (
                    float(np.linalg.norm(pos - xyz)),
                    {
                        "xyz_m": [round(float(v), 4) for v in pos],
                        "robot_geom": names[0] if robot[0] else names[1],
                        "other_geom": names[1] if robot[0] else names[0],
                    },
                )
            )
        contacts: list[dict[str, Any]] = []
        for _, item in sorted(found, key=lambda t: t[0]):
            p = np.asarray(item["xyz_m"])
            if any(
                np.linalg.norm(p - np.asarray(c["xyz_m"])) < 0.004 for c in contacts
            ):
                continue
            contacts.append(item)
            if len(contacts) >= 6:
                break
        return {
            "site_xyz": xyz.tolist(),
            "site_xmat": R.tolist(),
            "pads_local": pad_local,
            "contacts": contacts,
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def jaw_frame(pads_local: Any) -> np.ndarray:
    """The fixed rotation from a site frame to the grip frame (jaw +X, approach +Z): identity
    when the pads lie along the site's X, a quarter turn about Z when they lie along its Y."""
    pads = np.asarray(pads_local, dtype=np.float64).reshape(2, 3)
    d = np.abs(pads[1] - pads[0])
    if d[1] > d[0]:
        return np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return np.eye(3)


# ---- the toolset -------------------------------------------------------------------------


def _splat(spec: OrthoSpec) -> int:
    """A zoomed view's splat radius: about 2 mm of cloud spacing per point."""
    return int(min(4, max(1, math.ceil(0.002 / spec.m))))


def _round(v: Any, d: int = 4) -> list[float]:
    return [round(float(x), d) for x in np.asarray(v, dtype=np.float64).reshape(-1)]


class GripGeometry:
    """Point-cloud views, marked points and grip-site targets over one robot.

    ``view(camera)`` renders one calibrated RGB-D camera now (rgb, depth m, intrinsic_K,
    extrinsic_cam2world), ``cameras`` the ones fused into the cloud (the first is the one
    previews and contacts are drawn on); ``tool_pose() -> (xyz, quat_xyzw)`` is the robot's
    controlled frame, whose position is the grip site; ``frame() -> 3x3`` the fixed rotation from
    it to the grip frame (default identity); ``pads() -> 2x3`` the finger pads' inner faces in
    the grip frame (default: along X at the current width); ``contacts() -> list`` the robot's
    current contacts (default none); ``width() -> m | None`` the finger opening; ``envelope() -> 3x2``
    the box the scene views are cut from (default 0.9 m around the grip site);
    ``state_digest()`` changes whenever the robot or scene moved. ``move(position, tool_R, tol_m,
    tol_rad, max_steps) -> dict`` servos the tool frame (``env.move_grip``;
    without it the robot's client executes ``env.grip_target``'s plans itself) and
    ``gripper(close) -> dict`` drives the fingers.
    """

    def __init__(
        self,
        view: Callable[[str], dict[str, Any]],
        *,
        cameras: list[str],
        tool_pose: Callable[[], tuple[Any, Any]],
        state_digest: Callable[[], Any],
        frame: Callable[[], Any] | None = None,
        pads: Callable[[], Any] | None = None,
        contacts: Callable[[], list[dict[str, Any]]] | None = None,
        width: Callable[[], float] | None = None,
        empty_width: float | None = None,
        envelope: Callable[[], Any] | None = None,
        move: Callable[..., dict[str, Any]] | None = None,
        gripper: Callable[[bool], dict[str, Any]] | None = None,
    ) -> None:
        if not cameras:
            raise ValueError("GripGeometry needs at least one camera")
        self._view = view
        self.cameras = list(cameras)
        self._tool_pose = tool_pose
        self._digest = state_digest
        self._frame_fn = frame
        self._frame: np.ndarray | None = None
        self._pads_fn = pads
        self._pads: np.ndarray | None = None
        self._contacts = contacts
        self._width = width
        self._empty = empty_width
        self._envelope = envelope
        self._move = move
        self._gripper = gripper
        self.reset()

    def reset(self) -> None:
        """A new episode: every mark, view and pending preview is dropped."""
        self._marks: dict[str, dict[str, Any]] = {}
        self._pending: dict[str, dict[str, Any]] = {}
        self._specs: dict[str, OrthoSpec] = {}
        self._specs_digest: Any = None
        self._scene: dict[str, Any] | None = None
        self._preview: dict[str, Any] | None = None

    # -- wiring ------------------------------------------------------------------------------

    def install(self, facade: Any) -> None:
        """Register ``env.point_views``, ``env.mark_point``, ``env.grip_target``,
        ``env.grip_state`` (and ``env.move_grip`` with a ``move``) on ``facade._rpc``; an
        ``env.reset`` there also forgets the episode's marks."""
        rpc = facade._rpc
        rpc["env.point_views"] = self.point_views
        rpc["env.mark_point"] = self.mark_point
        rpc["env.grip_target"] = self.grip_target
        rpc["env.grip_state"] = self.grip_state
        if self._move is not None:
            rpc["env.move_grip"] = self.move_grip
        reset = rpc.get("env.reset")
        if reset is not None:

            def reset_and_forget(*args: Any, **kwargs: Any) -> Any:
                try:
                    return reset(*args, **kwargs)
                finally:
                    self.reset()

            rpc["env.reset"] = reset_and_forget

    # -- robot state -------------------------------------------------------------------------

    def _grip_frame(self) -> np.ndarray:
        if self._frame is None:
            f = self._frame_fn() if self._frame_fn is not None else None
            self._frame = np.eye(3) if f is None else np.asarray(f, dtype=np.float64)
        return self._frame

    def grip(self) -> tuple[np.ndarray, np.ndarray]:
        """The grip site's world position and grip frame (jaw +X, approach +Z) now."""
        xyz, quat = self._tool_pose()
        R = quat_to_matrix(quat) @ self._grip_frame()
        return _vec3(xyz, "tool position"), R

    def tool_quat(self, R_grip: np.ndarray) -> list[float]:
        """The robot's tool-frame orientation that puts the grip frame at ``R_grip``."""
        return matrix_to_quat(np.asarray(R_grip) @ self._grip_frame().T)

    def _pad_local(self) -> np.ndarray:
        if self._pads is None and self._pads_fn is not None:
            p = self._pads_fn()
            if p is not None:
                self._pads = np.asarray(p, dtype=np.float64).reshape(2, 3)
        if self._pads is not None:
            return self._pads.copy()
        w = self._opening()
        w = OPEN_APERTURE_M if w is None else w
        return np.array([[-w / 2, 0.0, 0.0], [w / 2, 0.0, 0.0]])

    def _pads_at(self, p: np.ndarray, R: np.ndarray, aperture: float) -> np.ndarray:
        local = self._pad_local()
        mid = local.mean(axis=0)
        sign = np.sign(local[:, 0] - mid[0])
        if not np.all(np.abs(sign) > 0):
            sign = np.array([-1.0, 1.0])
        local[:, 0] = mid[0] + sign * aperture / 2
        return p + local @ R.T

    def _opening(self) -> float | None:
        """The finger opening (m), or None when the robot does not report it."""
        w = self._width() if self._width is not None else None
        return None if w is None else float(w)

    def _now(self) -> Any:
        try:
            return self._digest()
        except Exception:  # noqa: BLE001  an unreadable state counts as changed
            return object()

    # -- views -------------------------------------------------------------------------------

    def _scene_now(self) -> dict[str, Any]:
        """The fused cloud, its bounds and the camera views of the current state (cached)."""
        digest = self._now()
        if self._scene is not None and self._scene["digest"] == digest:
            return self._scene
        views = {c: self._view(c) for c in self.cameras}
        points, colors = cloud_from_views(list(views.values()))
        grip_xyz, _ = self.grip()
        envelope = None
        if self._envelope is not None:
            envelope = self._envelope()
        if envelope is None:
            envelope = np.stack(
                [grip_xyz - [0.9, 0.9, 0.7], grip_xyz + [0.9, 0.9, 0.7]], axis=1
            )
        bounds = scene_bounds(points, envelope, [grip_xyz])
        keep = np.all((points >= bounds[:, 0]) & (points <= bounds[:, 1]), axis=1)
        self._scene = {
            "digest": digest,
            "views": views,
            "points": points[keep],
            "colors": colors[keep],
            "bounds": bounds,
        }
        if self._specs_digest != digest:
            self._specs = {}
            self._specs_digest = digest
            # A pending half-click was made on the old scene.
            self._pending = {}
        return self._scene

    def _register(self, spec: OrthoSpec) -> OrthoSpec:
        self._specs[spec.name] = spec
        return spec

    def _draw(
        self,
        rgb: np.ndarray,
        to_px: Callable[[np.ndarray], tuple[float, float] | None],
        *,
        marks: list[str] | None = None,
        lines: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]] = (),
        extra: Callable[[Any, Callable], None] | None = None,
    ) -> np.ndarray:
        """Overlay the grip site (magenta ring) and the named solved marks on an image."""
        from PIL import Image, ImageDraw

        img = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8))
        draw = ImageDraw.Draw(img)
        font = _font()
        for a, b, color in lines:
            pa, pb = to_px(a), to_px(b)
            if pa is not None and pb is not None:
                draw.line([pa, pb], fill=color, width=2)
        if extra is not None:
            extra(draw, to_px)
        grip_xyz, _ = self.grip()
        g = to_px(grip_xyz)
        if g is not None:
            draw.ellipse(
                (g[0] - 6, g[1] - 6, g[0] + 6, g[1] + 6), outline=GRIP_COLOR, width=2
            )
        for i, pid in enumerate(marks if marks is not None else list(self._marks)):
            m = self._marks.get(pid)
            if m is None:
                continue
            q = to_px(np.asarray(m["xyz_m"]))
            if q is None:
                continue
            color = MARK_COLORS[i % len(MARK_COLORS)]
            draw.ellipse(
                (q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4), fill=color, outline=(0, 0, 0)
            )
            draw.text((q[0] + 6, q[1] - 6), pid, fill=color, font=font)
        return np.asarray(img)

    def _render(
        self, name: str, *, marks: list[str] | None = None, lines=(), extra=None
    ) -> str:
        """One view of the current scene as PNG base64, the grip site and marks drawn."""
        scene = self._scene_now()
        if name in self.cameras:
            view = scene["views"][name]
            rgb = np.asarray(view["rgb"], dtype=np.uint8)[..., :3]
            to_px = lambda p: project(view, p)  # noqa: E731
        else:
            kind = name.removeprefix("pointcloud_")
            if name not in SCENE_VIEWS:
                raise GeometryError(
                    f"unknown view {name!r}; one of {[*SCENE_VIEWS, *self.cameras]}"
                )
            spec = self._register(make_spec(name, kind, scene["bounds"], SCENE_PX))
            rgb = render_ortho(scene["points"], scene["colors"], spec)
            to_px = spec.to_pixel
        return png_b64(self._draw(rgb, to_px, marks=marks, lines=lines, extra=extra))

    def point_views(self, views: list[str] | None = None) -> dict[str, Any]:
        """The orthographic views of the fused cloud (default the three) and/or camera images,
        with the grip site and the solved marks drawn. Pixel (x, y) of any of them feeds
        ``mark_point``; an orthographic view's pixel is two world coordinates."""
        names = list(views) if views else list(SCENE_VIEWS)
        if len(names) > 5:
            raise GeometryError("at most 5 views per call")
        images = [self._render(n) for n in names]
        described = []
        for n in names:
            if n in self._specs:
                described.append(self._specs[n].describe())
            else:
                rgb = np.asarray(self._scene_now()["views"][n]["rgb"])
                described.append(
                    {
                        "view": n,
                        "width": int(rgb.shape[1]),
                        "height": int(rgb.shape[0]),
                        "camera": True,
                    }
                )
        grip_xyz, R = self.grip()
        return {
            "views": described,
            "images": images,
            "grip_xyz_m": _round(grip_xyz),
            "approach_world": _round(R[:, 2], 3),
            "jaw_world": _round(R[:, 0], 3),
            "marks": {k: m["xyz_m"] for k, m in self._marks.items()},
        }

    # -- marks -------------------------------------------------------------------------------

    def mark(self, point_id: str) -> np.ndarray:
        m = self._marks.get(str(point_id))
        if m is None:
            pending = (
                " (still pending: click a complementary view)"
                if point_id in self._pending
                else ""
            )
            raise GeometryError(
                f"no solved point {point_id!r}{pending}; have {sorted(self._marks)}"
            )
        return np.asarray(m["xyz_m"], dtype=np.float64)

    def mark_point(self, point_id: str, view: str, x: int, y: int) -> dict[str, Any]:
        """Mark world point ``point_id`` by pixel (x = column, y = row) of a view that
        ``point_views`` (or a preview) returned for the current state."""
        pid = str(point_id).strip()
        if not pid:
            raise GeometryError("point_id must be non-empty")
        x, y = int(x), int(y)
        scene = self._scene_now()
        if view in self.cameras:
            p = surface_point(scene["views"][view], x, y)
            if p is None:
                raise GeometryError(
                    f"no depth at or around ({x}, {y}) in {view}; click a surface, or two point-cloud views"
                )
            self._pending.pop(pid, None)
            self._marks[pid] = {"xyz_m": _round(p), "source": [view]}
            return {
                "point_id": pid,
                "status": "solved",
                "xyz_m": self._marks[pid]["xyz_m"],
                "source": [view],
                "views": [view, "pointcloud_top"],
                "images": [
                    self._render(view, marks=[pid]),
                    self._render("pointcloud_top", marks=[pid]),
                ],
            }
        spec = self._specs.get(view)
        if spec is None:
            raise GeometryError(
                f"view {view!r} was not rendered for the current state; call point_views (or preview) again"
            )
        if not (0 <= x < spec.width and 0 <= y < spec.height):
            raise GeometryError(
                f"pixel ({x}, {y}) is outside {view} ({spec.width}x{spec.height})"
            )
        fixed = spec.to_world(x, y)
        pending = self._pending.get(pid)
        if pending is not None and set(pending["fixed"]) != set(fixed):
            shared = (set(pending["fixed"]) & set(fixed)).pop()
            residual = abs(pending["fixed"][shared] - fixed[shared])
            if residual > CONSISTENCY_TOLERANCE_M:
                return {
                    "point_id": pid,
                    "status": "inconsistent",
                    "shared_axis": AXES[shared],
                    "residual_m": round(residual, 4),
                    "tolerance_m": CONSISTENCY_TOLERANCE_M,
                    "message": f"the two clicks disagree on {AXES[shared]} by {residual * 1000:.0f} mm; "
                    f"click {pending['view']} or {view} again",
                }
            xyz = np.zeros(3)
            for axis, val in {**pending["fixed"], **fixed}.items():
                xyz[axis] = val
            xyz[shared] = (pending["fixed"][shared] + fixed[shared]) / 2
            sources = [pending["view"], view]
            del self._pending[pid]
            self._marks[pid] = {"xyz_m": _round(xyz), "source": sources}
            return {
                "point_id": pid,
                "status": "solved",
                "xyz_m": self._marks[pid]["xyz_m"],
                "source": sources,
                "shared_axis_residual_m": round(residual, 4),
                "views": sources,
                "images": [self._render_spec(n, marks=[pid]) for n in sources],
            }
        self._pending[pid] = {"view": view, "fixed": fixed}
        missing = ({0, 1, 2} - set(fixed)).pop()
        needs = [
            s.name
            for s in self._specs.values()
            if missing in s.axes and s.name.split("_")[0] == view.split("_")[0]
        ]
        # The constraint drawn on the complementary views: a line along the missing axis.
        a, b = np.zeros(3), np.zeros(3)
        for axis, val in fixed.items():
            a[axis] = b[axis] = val
        lo, hi = scene["bounds"][missing]
        a[missing], b[missing] = lo - 1.0, hi + 1.0
        return {
            "point_id": pid,
            "status": "pending",
            "fixed_m": {AXES[k]: round(v, 4) for k, v in fixed.items()},
            "needs": needs,
            "message": f"{AXES[missing]} is still free: click the same feature in one of {needs} "
            "(the cyan line is where it can be)",
            "views": needs,
            "images": [
                self._render_spec(n, lines=[(a, b, (60, 255, 255))]) for n in needs
            ],
        }

    def _render_spec(self, name: str, *, marks=None, lines=(), extra=None) -> str:
        """Re-render a registered orthographic view (scene or preview) with overlays."""
        if name in SCENE_VIEWS:
            return self._render(name, marks=marks, lines=lines, extra=extra)
        spec = self._specs[name]
        scene = self._scene_now()
        rgb = render_ortho(scene["points"], scene["colors"], spec)
        return png_b64(
            self._draw(rgb, spec.to_pixel, marks=marks, lines=lines, extra=extra)
        )

    # -- targets -----------------------------------------------------------------------------

    def _resolve(self, args: dict[str, Any]) -> dict[str, Any]:
        """The target of a move request: position, grip frame, gripper; no side effects."""
        allowed = {
            "xyz",
            "point_id",
            "delta_mm",
            "delta_frame",
            "approach",
            "jaw",
            "gripper",
            "preview",
            "execute_preview_id",
            "max_steps",
        }
        unknown = sorted(k for k in args if k not in allowed)
        if unknown:
            raise GeometryError(f"unknown target field(s): {', '.join(unknown)}")
        given = {
            k: v
            for k, v in args.items()
            if v is not None and v is not False and v != ""
        }
        cur_p, cur_R = self.grip()
        if given.get("execute_preview_id"):
            extra = sorted(set(given) - {"execute_preview_id", "max_steps"})
            if extra:
                raise GeometryError(
                    f"execute_preview_id commits a frozen preview and takes no other target field ({', '.join(extra)})"
                )
            pending = self._preview
            pid = str(given["execute_preview_id"])
            if pending is None or pending["preview_id"] != pid:
                raise GeometryError(
                    f"preview {pid!r} is not pending; request the close again"
                )
            moved = float(np.linalg.norm(cur_p - pending["base_p"]))
            turned = math.degrees(rotation_angle(pending["base_R"], cur_R))
            if moved > PREVIEW_BASE_TOL_M or turned > PREVIEW_BASE_TOL_DEG:
                raise GeometryError(
                    f"preview {pid!r} is stale: the gripper moved {moved * 1000:.1f} mm / {turned:.1f} deg since; preview the close again"
                )
            return {**pending["target"], "preview_id": pid, "commit": True}
        sources = [k for k in ("xyz", "point_id", "delta_mm") if k in given]
        if len(sources) > 1:
            raise GeometryError(
                f"give at most one of xyz, point_id, delta_mm (got {', '.join(sources)})"
            )
        frame = str(given.get("delta_frame") or "world")
        if frame not in ("world", "grip_site"):
            raise GeometryError("delta_frame must be 'world' or 'grip_site'")
        if "xyz" in given:
            position = _vec3(given["xyz"], "xyz")
        elif "point_id" in given:
            position = self.mark(given["point_id"])
        elif "delta_mm" in given:
            d = _vec3(given["delta_mm"], "delta_mm") / 1000.0
            position = cur_p + (cur_R @ d if frame == "grip_site" else d)
        else:
            position = cur_p.copy()
        R, how = resolve_rotation(cur_R, given.get("approach"), given.get("jaw"))
        g = given.get("gripper")
        if g is not None and g not in ("open", "close"):
            raise GeometryError("gripper must be 'open' or 'close'")
        motion = bool(sources) or how != "current"
        if not motion and g is None:
            raise GeometryError("give a position, a direction or a gripper action")
        return {
            "position": position,
            "R": R,
            "orientation": how,
            "motion": motion,
            "gripper": g,
            "position_from": sources[0] if sources else "current",
        }

    def _describe(self, t: dict[str, Any]) -> dict[str, Any]:
        cur_p, cur_R = self.grip()
        return {
            "target": {
                "grip_xyz_m": _round(t["position"]),
                "approach_world": _round(t["R"][:, 2], 3),
                "jaw_world": _round(t["R"][:, 0], 3),
                "tool_quat_xyzw": _round(self.tool_quat(t["R"]), 5),
                "orientation": t["orientation"],
                "position_from": t["position_from"],
            },
            "current": {
                "grip_xyz_m": _round(cur_p),
                "approach_world": _round(cur_R[:, 2], 3),
                "jaw_world": _round(cur_R[:, 0], 3),
                "tool_quat_xyzw": _round(self.tool_quat(cur_R), 5),
            },
            "delta_mm": _round((t["position"] - cur_p) * 1000, 1),
            "rotation_deg": round(math.degrees(rotation_angle(cur_R, t["R"])), 1),
            "motion": t["motion"],
            "gripper": t["gripper"],
        }

    def planned_distance(self, args: dict[str, Any]) -> float:
        """How far a move request would take the grip site (for a run's translation cap)."""
        try:
            t = self._resolve(dict(args))
        except GeometryError:
            return 0.0
        return (
            float(np.linalg.norm(t["position"] - self.grip()[0]))
            if t["motion"]
            else 0.0
        )

    def grip_target(self, **args: Any) -> dict[str, Any]:
        """Resolve a move request without moving. ``status`` is ``execute`` (the caller moves
        to ``target`` and then runs ``gripper``) or ``preview`` (images of the target; a close
        freezes it under ``preview_id``, executed by ``execute_preview_id`` alone)."""
        return self._plan(args, images=True)[1]

    def _plan(
        self, args: dict[str, Any], images: bool
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        t = self._resolve(args)
        out = self._describe(t)
        if t.get("commit"):
            self._preview = None
            return t, {**out, "status": "execute", "preview_id": t["preview_id"]}
        close = t["gripper"] == "close"
        if not (args.get("preview") or close):
            if t["motion"]:
                self._preview = None
            return t, {**out, "status": "execute"}
        out["status"] = "preview"
        if close:
            cur_p, cur_R = self.grip()
            blob = json.dumps(
                [
                    _round(t["position"], 6),
                    _round(t["R"], 6),
                    _round(cur_p, 5),
                    _round(cur_R, 5),
                ]
            ).encode()
            out["preview_id"] = "pv" + hashlib.sha256(blob).hexdigest()[:8]
            self._preview = {
                "preview_id": out["preview_id"],
                "target": t,
                "base_p": cur_p,
                "base_R": cur_R,
            }
        if images:
            out["views"], out["images"] = self._preview_images(t)
        out["note"] = "geometry only: the preview does not predict contact or retention"
        return t, out

    def _preview_images(self, t: dict[str, Any]) -> tuple[list[str], list[str]]:
        """The target drawn on zoomed orthographic views around it (markable as preview_*)
        and on the first camera: the grip point, its approach (blue) and jaw (red) axes, the
        hand behind it, and the finger pads open (hollow) and, for a close, closed (filled)
        with the corridor they sweep."""
        scene = self._scene_now()
        p, R = t["position"], t["R"]
        close = t["gripper"] == "close"
        width = self._opening()
        width = OPEN_APERTURE_M if width is None else width
        start = self._pads_at(
            p, R, OPEN_APERTURE_M if t["gripper"] else max(width, CLOSED_APERTURE_M)
        )
        end = self._pads_at(p, R, CLOSED_APERTURE_M) if close else None
        hand = p - R[:, 2] * 0.10

        def extra(draw: Any, to_px: Callable) -> None:
            def seg(a, b, color, w=2):
                pa, pb = to_px(a), to_px(b)
                if pa is not None and pb is not None:
                    draw.line([pa, pb], fill=color, width=w)

            seg(hand, p, (200, 200, 200), 3)
            seg(p, p + R[:, 2] * 0.05, AXIS_COLORS[2])
            seg(p - R[:, 0] * 0.03, p + R[:, 0] * 0.03, AXIS_COLORS[0])
            for i in range(2):
                if end is not None:
                    seg(start[i], end[i], PAD_COLOR, 3)
                q = to_px(start[i])
                if q is not None:
                    draw.rectangle(
                        (q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4),
                        outline=PAD_COLOR,
                        width=2,
                    )
                if end is not None and (q := to_px(end[i])) is not None:
                    draw.rectangle(
                        (q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3), fill=PAD_COLOR
                    )
            q = to_px(p)
            if q is not None:
                draw.ellipse(
                    (q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4),
                    fill=TARGET_COLOR,
                    outline=(0, 0, 0),
                )

        box = np.stack([p - PREVIEW_HALF_M, p + PREVIEW_HALF_M], axis=1)
        names, images = [], []
        for name, kind in zip(PREVIEW_VIEWS, ORTHO):
            spec = self._register(make_spec(name, kind, box, PREVIEW_PX))
            rgb = render_ortho(
                scene["points"], scene["colors"], spec, splat=_splat(spec)
            )
            images.append(
                png_b64(self._draw(rgb, spec.to_pixel, marks=[], extra=extra))
            )
            names.append(name)
        cam = self.cameras[0]
        view = scene["views"][cam]
        rgb = np.asarray(view["rgb"], dtype=np.uint8)[..., :3]
        images.insert(
            0,
            png_b64(self._draw(rgb, lambda q: project(view, q), marks=[], extra=extra)),
        )
        names.insert(0, cam)
        return names, images

    # -- feedback ----------------------------------------------------------------------------

    def grip_state(
        self,
        target_xyz: Any = None,
        target_approach: Any = None,
        target_jaw: Any = None,
        tol_m: float = MOVE_TOL_M,
        tol_rad: float = MOVE_TOL_RAD,
    ) -> dict[str, Any]:
        """The grip site now; with a target, the remaining delta and rotation error and
        ``motion_status`` (reached within ``tol_m`` / ``tol_rad``, else not_reached); the
        robot's current contacts, drawn on the first camera when there are any."""
        p, R = self.grip()
        out: dict[str, Any] = {
            "grip_xyz_m": _round(p),
            "approach_world": _round(R[:, 2], 3),
            "jaw_world": _round(R[:, 0], 3),
        }
        w = self._opening()
        if w is not None:
            out["gripper_width"] = round(w, 4)
            if self._empty is not None:
                out["closed_on_nothing"] = w <= self._empty
        if (
            target_xyz is not None
            or target_approach is not None
            or target_jaw is not None
        ):
            tp = _vec3(target_xyz, "target_xyz") if target_xyz is not None else p
            TR, _ = resolve_rotation(R, target_approach, target_jaw)
            rem = tp - p
            err = rotation_angle(R, TR)
            out["remaining_delta_mm"] = _round(rem * 1000, 1)
            out["remaining_distance_mm"] = round(float(np.linalg.norm(rem)) * 1000, 1)
            out["rotation_error_deg"] = round(math.degrees(err), 1)
            reached = float(np.linalg.norm(rem)) <= float(tol_m) and err <= float(
                tol_rad
            )
            out["motion_status"] = "reached" if reached else "not_reached"
        contacts = self._contacts() if self._contacts is not None else None
        if contacts is not None:
            out["contacts"] = contacts
            if contacts:
                cam = self.cameras[0]
                view = self._view(cam)

                def extra(draw: Any, to_px: Callable) -> None:
                    for c in contacts:
                        q = to_px(np.asarray(c["xyz_m"]))
                        if q is not None:
                            draw.ellipse(
                                (q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3),
                                fill=CONTACT_COLOR,
                            )

                rgb = np.asarray(view["rgb"], dtype=np.uint8)[..., :3]
                out["views"] = [f"{cam}_contacts"]
                out["images"] = [
                    png_b64(
                        self._draw(
                            rgb, lambda q: project(view, q), marks=[], extra=extra
                        )
                    )
                ]
        return out

    # -- execution (a robot whose server moves: code mode) -----------------------------------

    def move_grip(self, max_steps: int = MOVE_MAX_STEPS, **args: Any) -> dict[str, Any]:
        """``grip_target``, then (status execute) the motion through ``move``, the gripper
        only when the motion reached, and ``grip_state``. A preview moves nothing."""
        t, plan = self._plan(args, images=False)
        if plan["status"] == "preview":
            return {**plan, "motion_status": "previewed"}
        out: dict[str, Any] = {"target": plan["target"]}
        status = "not_requested"
        target = (t["position"], t["R"][:, 2], t["R"][:, 0]) if t["motion"] else ()
        if t["motion"]:
            tool_R = t["R"] @ self._grip_frame().T
            out.update(
                self._move(
                    t["position"], tool_R, MOVE_TOL_M, MOVE_TOL_RAD, int(max_steps)
                )
            )
            status = self.grip_state(*target)["motion_status"]
        if t["gripper"] is not None:
            if status == "not_reached":
                out["gripper_skipped"] = "the motion did not reach its target"
            elif self._gripper is not None:
                g = self._gripper(t["gripper"] == "close")
                out["gripper"] = t["gripper"]
                out["gripper_steps"] = g.get("steps_used")
        final = self.grip_state(*target)
        for k in ("images", "views", "motion_status"):
            final.pop(k, None)
        return {**out, **final, "motion_status": status}
