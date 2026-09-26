# CaP-X's S2 robosuite API for the cube tasks (capx/integrations/franka/control.py
# FrankaControlApi @53e9966) over this server's high tier (robots/robosuite/primitives.py:
# segment, get_observation, get_state, move_to, set_gripper; plan_grasp with a grasp server).
# --code-oracle prepends it to lift.py, stack.py and restack.py (their `# prelude:` line).
# No simulator state: poses come from SAM3 masks and the depth image, as in CaP-X.
#
# Differences from CaP-X:
# - Poses are in CaP-X's frame, robot0's mount base (fixed_mount0_base, robosuite 1.5.2: world
#   (-0.56, 0, 0.922), unrotated, the same in Lift, Stack and Restack); move_to takes the world.
# - goto_pose servos the TCP (the grip site) with move_to instead of solving IK for panda_hand:
#   CaP-X's positions are the fingertip point, so no TCP offset is applied. Quaternions are
#   CaP-X's panda_hand wxyz, turned half a turn about the hand's z into the grip site's frame
#   (HAND_TO_SITE_WXYZ, as capx_privileged.py). A move longer than the server's per-call cap is
#   split into straight legs of <= MAX_LEG_M.
# - get_object_pose: SAM3 by text is the server's `segment` (its top mask); the mask's points are
#   back-projected through get_observation's agentview depth (the robot0_robotview, CaP-X's
#   camera). Open3D's statistical outlier removal (20 neighbours, 2 std) is reimplemented in numpy;
#   its oriented box is a PCA box of the points (not of their convex hull) whose axes are ordered
#   and signed to match the base frame's x, y, z, so bbox extent[2] is the vertical one (CaP-X's
#   docstring: "extent ... x, y, z").
# - sample_grasp_pose: CaP-X runs Contact-GraspNet on the mask. With a grasp server the server's
#   plan_grasp (Contact-GraspNet & co.) is asked and its best candidate's EEF pose is returned.
#   Without one (plan_grasp is not declared) the grasp is top-down (CaP-X's "down", wxyz
#   0, 0, 1, 0) over the mask's box centre, halfway between the box's top and the table
#   (get_state's table_z), turned about the vertical so the fingers close across the box's
#   shorter horizontal side.
# - home_pose: CaP-X moves to a fixed joint configuration; here the TCP returns to its position
#   after reset (get_state's home_eef_pos) pointing down.
# - open_gripper / close_gripper drive the fingers for CaP-X's 30 control steps (set_gripper
#   stops early once the fingers stop moving).
import math

import numpy as np

MAX_LEG_M = 0.25
MOVE_STEPS = 200
GRIPPER_STEPS = (30, 30)  # open, close
TWO_ARM = False
#: CaP-X's frame (robot0's fixed_mount0_base) in the world: position, wxyz.
BASE0_POS = np.array([-0.56, 0.0, 0.922])
BASE0_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
#: panda_hand -> robosuite's grip site: half a turn about the hand's z (wxyz).
HAND_TO_SITE_WXYZ = np.array([0.0, 0.0, 0.0, 1.0])
#: CaP-X's gripper-down orientation (wxyz).
DOWN_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])


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


def _to_world(p):
    return BASE0_POS + _matrix(BASE0_WXYZ) @ np.asarray(p, dtype=np.float64).reshape(3)


def _from_world(p):
    return _matrix(BASE0_WXYZ).T @ (np.asarray(p, dtype=np.float64).reshape(3) - BASE0_POS)


def _points_from_world(pts):
    return (np.asarray(pts, dtype=np.float64).reshape(-1, 3) - BASE0_POS) @ _matrix(BASE0_WXYZ)


def _site_xyzw(q_wxyz):
    """CaP-X's panda_hand orientation (base frame) -> move_to's grip-site quaternion (world, xyzw)."""
    w, x, y, z = _unit(_mul(_mul(BASE0_WXYZ, _unit(q_wxyz)), HAND_TO_SITE_WXYZ))
    return [float(x), float(y), float(z), float(w)]


def _hand_wxyz(site_xyzw):
    """A grip-site quaternion (world, xyzw) -> CaP-X's panda_hand orientation (base frame, wxyz)."""
    x, y, z, w = site_xyzw
    return _unit(_mul(_conj(BASE0_WXYZ), _mul(np.array([w, x, y, z]), _conj(HAND_TO_SITE_WXYZ))))


_EEF = {}


def _kw(arm):
    return {"arm": arm} if TWO_ARM else {}


def _eef(arm):
    if arm not in _EEF:
        _EEF[arm] = np.asarray(get_state()[f"{arm}_eef_pos"], dtype=np.float64)
    return _EEF[arm]


def _move(arm, target_base, q_wxyz):
    """Servo `arm`'s TCP to a base-frame point, in legs of at most MAX_LEG_M."""
    target = _to_world(target_base)
    start = _eef(arm)
    legs = max(1, math.ceil(np.linalg.norm(target - start) / MAX_LEG_M))
    for k in range(1, legs + 1):
        leg = start + (target - start) * k / legs
        r = move_to(leg.tolist(), quat_xyzw=_site_xyzw(q_wxyz), max_steps=MOVE_STEPS, **_kw(arm))
        _EEF[arm] = np.asarray(r.get("final_eef_pos", leg), dtype=np.float64)


def _goto(arm, position, quaternion_wxyz, z_approach=0.0):
    """CaP-X's goto_pose: first position + z_approach back along the approach axis, then the pose."""
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    if z_approach != 0.0:
        _move(arm, pos + _matrix(quaternion_wxyz) @ np.array([0.0, 0.0, -z_approach]), quaternion_wxyz)
    _move(arm, pos, quaternion_wxyz)


def _grip(arm, close, steps):
    set_gripper(bool(close), steps=int(steps), **_kw(arm))


# ---- perception: SAM3 masks and the depth image ----


def _segment(prompt, camera="agentview"):
    seg = segment(prompt, camera=camera)
    if not seg.get("found"):
        raise ValueError(f"No sam3 detections for {prompt!r}")
    return np.asarray(seg["mask"]).astype(bool)


def _mask_points(mask, view):
    """World points of the mask's pixels that have depth."""
    rows, cols = np.nonzero(mask)
    depth = np.asarray(view["depth"], dtype=np.float64)
    z = depth[rows, cols]
    ok = np.isfinite(z) & (z > 0)
    rows, cols, z = rows[ok], cols[ok], z[ok]
    K = np.asarray(view["intrinsic_K"], dtype=np.float64)
    T = np.asarray(view["extrinsic_cam2world"], dtype=np.float64)
    cam = np.stack([(cols - K[0, 2]) * z / K[0, 0], (rows - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)])
    return (T @ cam)[:3].T


def _object_points(prompt):
    """The prompt's mask in the agentview, as base-frame points."""
    mask = _segment(prompt)
    pts = _points_from_world(_mask_points(mask, get_observation()["agentview"]))
    if len(pts) < 3:
        raise ValueError(f"too few depth points on {prompt!r}")
    return pts


def _knn_mean_dist(pts, k):
    """Each point's mean distance to its k nearest points (itself included), as Open3D."""
    k = min(k, len(pts))
    try:
        from scipy.spatial import cKDTree

        d, _ = cKDTree(pts).query(pts, k=k)
        return np.asarray(d, dtype=np.float64).reshape(len(pts), -1).mean(axis=1)
    except Exception:  # no scipy in the sandbox: brute force in blocks
        out = np.empty(len(pts))
        for i in range(0, len(pts), 512):
            d = np.linalg.norm(pts[i : i + 512, None, :] - pts[None, :, :], axis=2)
            out[i : i + 512] = np.sort(d, axis=1)[:, :k].mean(axis=1)
        return out


def _remove_statistical_outlier(pts, nb_neighbors=20, std_ratio=2.0):
    if len(pts) <= nb_neighbors:
        return pts
    d = _knn_mean_dist(pts, nb_neighbors)
    return pts[d <= d.mean() + std_ratio * d.std()]


def _obb(pts):
    """A PCA box: centre, rotation (columns ~ base x, y, z) and full extents."""
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((pts - c).T))
    R = np.zeros((3, 3))
    free_vec, free_axis = [0, 1, 2], [0, 1, 2]
    while free_vec:
        i, j = max(((i, j) for i in free_vec for j in free_axis), key=lambda ij: abs(vecs[ij[1], ij[0]]))
        R[:, j] = vecs[:, i] * (1.0 if vecs[j, i] >= 0 else -1.0)
        free_vec.remove(i)
        free_axis.remove(j)
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]
    local = (pts - c) @ R
    lo, hi = local.min(axis=0), local.max(axis=0)
    return c + R @ ((lo + hi) / 2), R, hi - lo


# ---- CaP-X's API ----


def get_object_pose(object_name, return_bbox_extent=False):
    """CaP-X: (position (3,), quaternion_wxyz (4,), bbox extent (3,) or None) from SAM3 + depth."""
    pts = _remove_statistical_outlier(_object_points(object_name), 20, 2.0)
    center, R, extent = _obb(pts)
    return center, _quat(R), (extent if return_bbox_extent else None)


def sample_grasp_pose(object_name):
    """CaP-X: a grasp (position (3,), quaternion_wxyz (4,)) for the object."""
    if callable(globals().get("plan_grasp")):
        plan = plan_grasp(object=object_name)
        best = next(c for c in plan["candidates"] if c["id"] == plan["active"])
        return _from_world(best["eef_position"]), _hand_wxyz(best["eef_quat_xyzw"])
    center, R, extent = _obb(_object_points(object_name))
    # The camera sees the top: grasp halfway between the top and the table.
    table = _from_world([0.0, 0.0, get_state()["table_z"]])[2]
    center = np.array([center[0], center[1], (center[2] + extent[2] / 2 + table) / 2])
    short = R[:, 0] if extent[0] <= extent[1] else R[:, 1]
    yaw = math.atan2(short[1], short[0]) - math.pi / 2
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    return center, _mul(np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]), DOWN_WXYZ)


def goto_pose(position, quaternion_wxyz, z_approach=0.0):
    """CaP-X: first position + z_approach back along the gripper's approach axis, then the pose."""
    _goto("robot0", position, quaternion_wxyz, z_approach)


def home_pose():
    home = np.asarray(get_state()["home_eef_pos"]["robot0"], dtype=np.float64)
    _move("robot0", _from_world(home), DOWN_WXYZ)


def open_gripper():
    _grip("robot0", False, GRIPPER_STEPS[0])


def close_gripper():
    _grip("robot0", True, GRIPPER_STEPS[1])
