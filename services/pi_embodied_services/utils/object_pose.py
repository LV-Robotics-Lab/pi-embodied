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

"""CaP-X's object poses from segmented depth points (capx/integrations/franka/libero.py
@53e9966), numpy only: the points of a mask, CaP-X's DBSCAN noise filter, an oriented box and
its quaternion, the multi-view merge rule, and the conversions between CaP-X's panda_hand
quaternions and the Panda grip site's yaw.

Differences from CaP-X: DBSCAN (eps 5 mm, 10 samples) is reimplemented (its non-noise set);
Open3D's oriented box after outlier removal is a PCA box of the filtered points.
"""

from __future__ import annotations

import math

import numpy as np

#: panda_hand -> the Panda grip site (the pose robosuite and LIBERO report): half a turn about
#: the hand's z (wxyz; measured on robosuite's Panda, whose gripper LIBERO uses).
HAND_TO_SITE_WXYZ = np.array([0.0, 0.0, 0.0, 1.0])
#: The grip site pointing down, as LIBERO's reset holds it (180 deg about x, wxyz).
SITE_DOWN_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])
#: The fallback grasp's TCP depth below the object's highest point.
GRASP_DEPTH_M = 0.03


def unit(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q / np.linalg.norm(q)


def mul(a, b) -> np.ndarray:
    """Hamilton product of wxyz quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def conj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def matrix(q_wxyz) -> np.ndarray:
    w, x, y, z = unit(q_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def wxyz_of_matrix(m) -> np.ndarray:
    """Rotation matrix -> wxyz (Shepperd)."""
    m = np.asarray(m, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        ]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        ]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        ]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        ]
    return unit(q)


def site_yaw(hand_wxyz) -> float:
    """The grip site's yaw (about world +z) for CaP-X's panda_hand orientation."""
    w, x, y, z = unit(mul(unit(hand_wxyz), HAND_TO_SITE_WXYZ))
    return math.atan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))


def hand_of_yaw(yaw: float) -> np.ndarray:
    """CaP-X's panda_hand wxyz of a grip site pointing down at ``yaw``."""
    rz = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    return unit(mul(mul(rz, SITE_DOWN_WXYZ), conj(HAND_TO_SITE_WXYZ)))


def mask_points(view: dict, mask) -> np.ndarray:
    """World points [N, 3] of a mask's pixels with depth (more than 1.5 cm) in an upright view
    ``{depth, intrinsic_K, extrinsic_cam2world}``."""
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    z = np.asarray(view["depth"], dtype=np.float64)[rows, cols]
    ok = np.isfinite(z) & (z > 0.015)
    rows, cols, z = rows[ok], cols[ok], z[ok]
    K = np.asarray(view["intrinsic_K"], dtype=np.float64)
    T = np.asarray(view["extrinsic_cam2world"], dtype=np.float64)
    cam = np.stack(
        [
            (cols - K[0, 2]) * z / K[0, 0],
            (rows - K[1, 2]) * z / K[1, 1],
            z,
            np.ones_like(z),
        ]
    )
    return (T @ cam)[:3].T


def _neighbour_counts(pts: np.ndarray, eps: float) -> np.ndarray:
    out = np.zeros(len(pts), dtype=np.int64)
    for i in range(0, len(pts), 256):
        d = np.linalg.norm(pts[i : i + 256, None, :] - pts[None, :, :], axis=2)
        out[i : i + 256] = (d <= eps).sum(axis=1)
    return out


def filter_noise(pts, eps: float = 0.005, min_samples: int = 10) -> np.ndarray:
    """CaP-X's DBSCAN(eps=0.005, min_samples=10) keeping every clustered point: the core points
    and the points within eps of one."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return pts
    core = _neighbour_counts(pts, eps) >= min_samples
    if not core.any():
        return pts[:0]
    keep = core.copy()
    cp = pts[core]
    for i in range(0, len(pts), 256):
        d = np.linalg.norm(pts[i : i + 256, None, :] - cp[None, :, :], axis=2)
        keep[i : i + 256] |= (d <= eps).any(axis=1)
    return pts[keep]


def min_dist(a: np.ndarray, b: np.ndarray) -> float:
    best = np.inf
    for i in range(0, len(a), 256):
        best = min(
            best,
            float(
                np.linalg.norm(a[i : i + 256, None, :] - b[None, :, :], axis=2).min()
            ),
        )
    return best


def merge_views(found: dict) -> np.ndarray:
    """CaP-X's multi-view rule over ``{camera: (points, score)}``: the agentview's points, or
    both views' when they come within 1 cm, else the higher-scoring view's."""
    agent, agent_score = found["agentview"]
    if "wrist" not in found:
        return agent
    wrist, wrist_score = found["wrist"]
    if len(wrist) and len(agent):
        if min_dist(agent, wrist) < 0.01:
            return np.concatenate([agent, wrist])
        return wrist if wrist_score > agent_score else agent
    return wrist if len(wrist) else agent


def obb(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A PCA box: centre, rotation (columns = its axes), full extents."""
    c = pts.mean(axis=0)
    _, R = np.linalg.eigh(np.cov((pts - c).T))
    R = R[:, ::-1]
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    local = (pts - c) @ R
    lo, hi = local.min(axis=0), local.max(axis=0)
    return c + R @ ((lo + hi) / 2), R, hi - lo


def pose_of_points(
    pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """CaP-X's get_object_pose from filtered points: the box centre and its wxyz, its z axis
    turned to point down; (None, None) with fewer than 3 points."""
    if len(pts) < 3:
        return None, None
    center, R, _ = obb(pts)
    if R[2, 2] > 0:
        R = R @ np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    return center, wxyz_of_matrix(R)


def topdown_grasp(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A grasp without a grasp server: top-down over the centre of the filtered points,
    ``GRASP_DEPTH_M`` below their highest point, the fingers closing across the box's shorter
    horizontal side; (position, panda_hand wxyz)."""
    if len(pts) < 3:
        raise ValueError("too few points for a grasp")
    center, R, extent = obb(pts)
    horiz = [i for i in range(3) if abs(R[2, i]) < 0.7] or [0, 1]
    short = min(horiz, key=lambda i: extent[i])
    yaw = math.atan2(R[1, short], R[0, short]) - math.pi / 2
    top = float(pts[:, 2].max())
    return np.array([center[0], center[1], top - GRASP_DEPTH_M]), hand_of_yaw(yaw)
