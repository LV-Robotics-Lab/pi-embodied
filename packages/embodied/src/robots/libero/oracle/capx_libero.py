# CaP-X's S2 LIBERO API (capx/integrations/franka/libero.py FrankaLiberoApi @53e9966) over this
# server's high tier (segment, get_observation, get_state, move_to, rotate_wrist, set_gripper;
# plan_grasp with a grasp server). --code-oracle prepends it to object_swap_7.py.
#
# Differences from CaP-X:
# - Segmentation: CaP-X points at the object with Molmo and segments from the point with SAM3,
#   falling back to SAM3 by text; there is no Molmo here, so SAM3 by text (`segment`, its top
#   mask) in each view. The views' points are combined by CaP-X's rule (both when they come
#   within 1 cm, else the higher score's).
# - get_object_pose: CaP-X filters the points with DBSCAN (eps 5 mm, 10 samples), which is
#   reimplemented in numpy, then takes Open3D's oriented box after outlier removal; here a PCA
#   box of the filtered points, its z flipped to point down as CaP-X does. The programs use only
#   the position.
# - sample_grasp_pose: CaP-X runs Contact-GraspNet on the scene and object point clouds. With a
#   grasp server the server's plan_grasp is asked (agentview) and its best candidate's EEF
#   position and yaw are returned; without one (plan_grasp is not declared) the grasp is
#   top-down over the centre of the object's filtered points, GRASP_DEPTH_M below their
#   highest point, the fingers closing across its shorter horizontal side.
# - Frames: CaP-X's LIBERO poses are in robot0_base's frame, which LIBERO mounts unrotated, and
#   the programs only use offsets and CaP-X's own perception or simulator poses, so the world
#   frame of this server's primitives stands in for it unchanged.
# - goto_pose: move_to servos the TCP (the grip site) holding its orientation, and rotate_wrist
#   turns it about the vertical; there is no pitch or roll primitive. So a pose's position is
#   reached exactly (CaP-X's positions are the TCP point: no TCP offset is applied) and of its
#   orientation only the yaw, with the gripper pointing down as LIBERO's reset holds it; the
#   yaw is CaP-X's panda_hand quaternion turned half a turn about the hand's z into the grip
#   site's frame, and the wrist turns to it or to the half turn away, whichever is nearer. Every
#   quaternion these programs pass is top-down, so nothing they ask for is lost. A move longer
#   than MAX_LEG_M is split into straight legs.
import math
import sys
import types

import numpy as np

MAX_LEG_M = 0.25
MOVE_STEPS = 120
#: panda_hand -> the grip site whose quaternion get_state reports: half a turn about the hand's
#: z (wxyz; measured on robosuite's Panda, whose gripper LIBERO uses).
HAND_TO_SITE_WXYZ = np.array([0.0, 0.0, 0.0, 1.0])
#: The grip site pointing down, as at reset (180 deg about x, wxyz).
SITE_DOWN_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

# The programs `import viser.transforms as vtf` and never use it: without viser in the sandbox
# an empty stand-in lets the import succeed.
try:
    import viser.transforms  # noqa: F401
except ImportError:
    _viser = types.ModuleType("viser")
    _viser.transforms = types.ModuleType("viser.transforms")
    sys.modules.setdefault("viser", _viser)
    sys.modules.setdefault("viser.transforms", _viser.transforms)


def _mul(a, b):
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


def _conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _unit(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q / np.linalg.norm(q)


def _matrix(q_wxyz):
    w, x, y, z = _unit(q_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _site_yaw(q_wxyz):
    """The yaw get_state reports for a grip site at CaP-X's panda_hand orientation q_wxyz."""
    w, x, y, z = _unit(_mul(_unit(q_wxyz), HAND_TO_SITE_WXYZ))
    return math.atan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))


def _hand_of_yaw(yaw):
    """CaP-X's panda_hand wxyz of a grip site pointing down at `yaw`."""
    rz = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    return _unit(_mul(_mul(rz, SITE_DOWN_WXYZ), _conj(HAND_TO_SITE_WXYZ)))


_EEF = {}


def _state():
    s = get_state()
    _EEF["pos"] = np.asarray(s["eef_pos"], dtype=np.float64)
    _EEF["yaw"] = float(s["yaw"])
    return s


def _eef():
    if "pos" not in _EEF:
        _state()
    return _EEF["pos"]


def _turn(q_wxyz):
    """Turn the wrist to q's yaw, or the half turn away from it, whichever is nearer (a
    parallel gripper grasps the same either way)."""
    if "yaw" not in _EEF:
        _state()
    goal = _site_yaw(q_wxyz)
    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi  # noqa: E731
    if abs(wrap(goal - _EEF["yaw"])) > math.pi / 2:
        goal = wrap(goal + math.pi)
    if abs(wrap(goal - _EEF["yaw"])) > 0.05:
        r = rotate_wrist(target_yaw=goal)
        _EEF["yaw"] = float(r.get("yaw", goal))
        if "eef_pos" in r:
            _EEF["pos"] = np.asarray(r["eef_pos"], dtype=np.float64)


def _move(target, q_wxyz):
    """Turn to q's yaw, then servo the TCP to a world point in legs of at most MAX_LEG_M."""
    _turn(q_wxyz)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    start = _eef()
    legs = max(1, math.ceil(np.linalg.norm(target - start) / MAX_LEG_M))
    for k in range(1, legs + 1):
        # From where the last leg really ended; a stalled leg leaves more than planned, and no
        # call may ask for more than MAX_LEG_M.
        here = _EEF["pos"]
        leg = here + (target - here) / max(legs - k + 1, math.ceil(np.linalg.norm(target - here) / MAX_LEG_M))
        r = move_to(leg.tolist(), max_steps=MOVE_STEPS)
        _EEF["pos"] = np.asarray(r.get("eef_pos", leg), dtype=np.float64)


def _goto_along_axis(position, quaternion_wxyz, z_approach=0.0):
    """CaP-X's goto_pose: first position + z_approach back along the approach axis, then the pose."""
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    if z_approach != 0.0:
        _move(pos + _matrix(quaternion_wxyz) @ np.array([0.0, 0.0, -z_approach]), quaternion_wxyz)
    _move(pos, quaternion_wxyz)


def _grip(close, steps):
    set_gripper(bool(close), steps=int(steps))


def _points(view, mask):
    rows, cols = np.nonzero(mask)
    z = np.asarray(view["depth"], dtype=np.float64)[rows, cols]
    ok = np.isfinite(z) & (z > 0.015)
    rows, cols, z = rows[ok], cols[ok], z[ok]
    K = np.asarray(view["intrinsic_K"], dtype=np.float64)
    T = np.asarray(view["extrinsic_cam2world"], dtype=np.float64)
    cam = np.stack([(cols - K[0, 2]) * z / K[0, 0], (rows - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)])
    return (T @ cam)[:3].T


def _neighbour_counts(pts, eps):
    """How many points (itself included) lie within eps of each point, in blocks."""
    out = np.zeros(len(pts), dtype=np.int64)
    for i in range(0, len(pts), 256):
        d = np.linalg.norm(pts[i : i + 256, None, :] - pts[None, :, :], axis=2)
        out[i : i + 256] = (d <= eps).sum(axis=1)
    return out


def _filter_noise(pts, eps=0.005, min_samples=10):
    """CaP-X's DBSCAN(eps=0.005, min_samples=10) keeping every clustered point: the core points
    and the points within eps of one (DBSCAN's non-noise set)."""
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


def _min_dist(a, b):
    best = np.inf
    for i in range(0, len(a), 256):
        best = min(best, float(np.linalg.norm(a[i : i + 256, None, :] - b[None, :, :], axis=2).min()))
    return best


def _object_points(text_prompt, use_multiview=True):
    """CaP-X's get_object_3d_points_and_masks_from_language: the prompt's mask in the agentview
    (and the wrist view), as world points; both views' points when they touch (1 cm), else the
    higher-scoring view's."""
    obs = get_observation()
    found = {}
    for cam in ("agentview", "wrist") if use_multiview else ("agentview",):
        seg = segment(text_prompt, camera=cam)
        if not seg.get("found"):
            raise ValueError(f"SAM3 segmentation failed for '{text_prompt}' on {cam}.")
        found[cam] = (_points(obs[cam], np.asarray(seg["mask"]).astype(bool)), float(seg.get("score") or 0.0))
    agent, agent_score = found["agentview"]
    if "wrist" not in found:
        return agent
    wrist, wrist_score = found["wrist"]
    if len(wrist) and len(agent):
        if _min_dist(agent, wrist) < 0.01:
            return np.concatenate([agent, wrist])
        return wrist if wrist_score > agent_score else agent
    return wrist if len(wrist) else agent


def _obb(pts):
    """A PCA box: centre, rotation (columns = its axes), full extents."""
    c = pts.mean(axis=0)
    _, R = np.linalg.eigh(np.cov((pts - c).T))
    R = R[:, ::-1]
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    local = (pts - c) @ R
    lo, hi = local.min(axis=0), local.max(axis=0)
    return c + R @ ((lo + hi) / 2), R, hi - lo


def _quat(m):
    """Rotation matrix -> wxyz (Shepperd)."""
    m = np.asarray(m, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return _unit(q)


#: The fallback grasp's TCP depth below the object's highest point.
GRASP_DEPTH_M = 0.03


def get_object_pose(object_name, use_multiview=True):
    """CaP-X: (position (3,), quaternion_wxyz (4,)) or (None, None)."""
    pts = _filter_noise(_object_points(object_name, use_multiview))
    if len(pts) < 3:
        return None, None
    center, R, _ = _obb(pts)
    if R[2, 2] > 0:
        R = R @ np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    return center, _quat(R)


def sample_grasp_pose(object_name, use_multiview=True):
    """CaP-X: a grasp (position (3,), quaternion_wxyz (4,)) for the object."""
    if callable(globals().get("plan_grasp")):
        plan = plan_grasp(object=object_name)
        best = next(c for c in plan["candidates"] if c["id"] == plan["active"])
        return np.asarray(best["eef_position"], dtype=np.float64), _hand_of_yaw(float(best["eef_yaw"]))
    pts = _filter_noise(_object_points(object_name, use_multiview))
    if len(pts) < 3:
        raise ValueError(f"No valid points after filtering for '{object_name}'")
    center, R, extent = _obb(pts)
    horiz = [i for i in range(3) if abs(R[2, i]) < 0.7] or [0, 1]
    short = min(horiz, key=lambda i: extent[i])
    yaw = math.atan2(R[1, short], R[0, short]) - math.pi / 2
    top = float(pts[:, 2].max())
    return np.array([center[0], center[1], top - GRASP_DEPTH_M]), _hand_of_yaw(yaw)


def goto_pose(position, quaternion_wxyz, z_approach=0.0):
    """CaP-X: first position + z_approach back along the gripper's approach axis, then the pose."""
    _goto_along_axis(position, quaternion_wxyz, z_approach)


def open_gripper():
    _grip(False, 40)


def close_gripper():
    _grip(True, 60)
