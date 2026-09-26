# CaP-X's privileged spill-wipe API (capx/integrations/franka/spill_wipe_privileged.py
# FrankaControlSpillWipePrivilegedApi @53e9966) over this server's privileged tier
# (ground_truth_poses, get_state, move_to). --code-oracle prepends it to wipe_privileged.py.
# CaP-X's API has get_object_pose and goto_pose only (the wiping gripper has no fingers).
#
# Differences from CaP-X:
# - Frame: CaP-X's poses are in robot0's mount base, whose z = 0 its prompt calls "exactly" the
#   table surface, where the program wipes (z_wipe = 0.0). In robosuite 1.5.2 the mount base
#   (fixed_mount0_base) is at world (-0.41, 0, 0.922) and the table top at z 0.9, 2.2 cm lower;
#   the sponge's surface is at the grip site, so wiping at the mount's z = 0 would never touch
#   the table. The frame here is the mount's x, y (unrotated) with z = 0 on the table top
#   (world 0.9), the frame the prompt describes, so the program wipes on the table.
# - get_object_pose reads the dirt markers from ground_truth_poses (the `contact*` names, which
#   are robosuite's arena markers, the bodies CaP-X reads): their mean, and their extent plus
#   CaP-X's 5 cm margin in x and y. The object name is ignored, as in CaP-X.
# - goto_pose servos the TCP (the grip site) with move_to instead of solving IK for panda_hand:
#   CaP-X's positions are the fingertip point, so no TCP offset is applied. Quaternions are
#   CaP-X's panda_hand wxyz, turned half a turn about the hand's z into the grip site's frame
#   (HAND_TO_SITE_WXYZ, as capx_privileged.py). A move longer than the server's per-call cap is
#   split into straight legs of <= MAX_LEG_M. The path is a Cartesian line, not a joint-space one.
import math

import numpy as np

MAX_LEG_M = 0.25
MOVE_STEPS = 200
GRIPPER_STEPS = (40, 60)  # open, close
TWO_ARM = False
#: CaP-X's frame (robot0's fixed_mount0_base) in the world: position, wxyz.
BASE0_POS = np.array([-0.41, 0.0, 0.9])
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
        # From where the last leg really ended; a stalled leg leaves more than planned, and no
        # call may ask for more than MAX_LEG_M.
        here = _EEF[arm]
        leg = here + (target - here) / max(legs - k + 1, math.ceil(np.linalg.norm(target - here) / MAX_LEG_M))
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


def get_object_pose(object_name, return_bbox_extent=False):
    """CaP-X: the dirt markers' mean position, identity wxyz and their extent (+5 cm in x, y)."""
    poses = ground_truth_poses()["poses"]
    pts = np.array([_from_world(p["pos"]) for k, p in sorted(poses.items()) if k.startswith("contact")])
    if len(pts) == 0:
        raise ValueError("no dirt markers in ground_truth_poses")
    extent = pts.max(axis=0) - pts.min(axis=0)
    extent[0] += 0.05
    extent[1] += 0.05
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    return pts.mean(axis=0), quat, (extent if return_bbox_extent else None)


def goto_pose(position, quaternion_wxyz):
    """CaP-X: go to the pose (no approach offset in this API)."""
    _goto("robot0", position, quaternion_wxyz)
