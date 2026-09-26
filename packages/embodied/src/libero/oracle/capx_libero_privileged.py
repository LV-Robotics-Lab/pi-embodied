# CaP-X's privileged LIBERO API (capx/integrations/franka/libero_privileged.py
# FrankaLiberoPrivilegedApi @53e9966) over this server's privileged tier (ground_truth_poses,
# get_state, move_to, rotate_wrist, set_gripper). --code-oracle prepends it to
# object_swap_7_privileged.py.
#
# Differences from CaP-X:
# - get_object_pose matches the name as CaP-X's simulator does (_get_object_pose): "<name>_1"
#   exactly, else the one object whose name (without its _<n> suffix) contains the query or is
#   contained in it; ground_truth_poses lists LIBERO's objects (obj_body_id), so CaP-X's last
#   fallback to fixed bodies (stove, cabinet) has no counterpart and raises instead.
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
        leg = start + (target - start) * k / legs
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


def _object(object_name):
    poses = ground_truth_poses()["poses"]
    query = object_name.replace(" ", "_").lower()
    if f"{query}_1" in poses:
        return poses[f"{query}_1"]
    base = {}
    for k in poses:
        stem, _, n = k.rpartition("_")
        base.setdefault(stem if n.isdigit() and stem else k, []).append(k)
    matches = [b for b in sorted(base) if query in b or b in query]
    if len(matches) == 1 and len(base[matches[0]]) >= 1:
        return poses[sorted(base[matches[0]])[0]]
    raise KeyError(f"Object '{object_name}' not found. Available objects: {sorted(base)}")


def get_object_pose(object_name):
    """CaP-X: (position (3,), quaternion_wxyz (4,)) from the simulator."""
    p = _object(object_name)
    x, y, z, w = p["quat_xyzw"]
    return np.asarray(p["pos"], dtype=np.float64), np.array([w, x, y, z])


def sample_grasp_pose(object_name):
    """CaP-X: the object's position with the gripper down (wxyz 0, 1, 0, 0)."""
    pos, _ = get_object_pose(object_name)
    return pos, np.array([0, 1, 0, 0])


def goto_pose(position, quaternion_wxyz, z_approach=0.0):
    """CaP-X: first position + z_approach back along the gripper's approach axis, then the pose."""
    _goto_along_axis(position, quaternion_wxyz, z_approach)


def open_gripper():
    _grip(False, 40)


def close_gripper():
    _grip(True, 60)
