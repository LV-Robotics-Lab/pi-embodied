# CaP-X's privileged two-arm lift API (capx/integrations/franka/two_arm_lift_privileged.py
# FrankaTwoArmLiftPrivilegedApi @53e9966) over this server's privileged tier (ground_truth_poses,
# get_state, move_to, set_gripper). --code-oracle prepends it to two_arm_lift_privileged.py.
# CaP-X's functions() offers get_handle0_pos / get_handle1_pos (not get_handle0_pose), so the
# oracle takes its bounding-box branch there, and here: get_handle0_pose is not defined.
#
# Differences from CaP-X:
# - Poses are in CaP-X's frame, robot0's mount base (fixed_mount0_base, robosuite 1.5.2's
#   opposed arms: world (0, -0.56, 0.922), turned +90 deg about z), for both arms, as CaP-X's
#   goto_pose_arm1 takes robot0-frame poses.
# - get_handle<i>_pos: CaP-X's pot_poses handle<i> position (robosuite's handle<i>_xpos, a
#   site): ground_truth_poses' pot_handle0 / pot_handle1.
# - get_arm<i>_gripper_pose is CaP-X's robot<i>_cartesian_pos: the gripper body
#   (gripper<i>_right_eef, measured to sit at the TCP turned a quarter turn about z from the
#   eef frame) turned back a quarter turn and backed 0.107 m up the approach axis. That is the TCP's
#   own orientation (get_state's eef quat) and position minus 0.107 m along its z; fed back to
#   goto_pose it turns the wrist half a turn, as it did in CaP-X.
# - goto_pose_both: move_to moves one arm per call, so both arms advance in alternating legs of
#   at most BOTH_LEG_M (CaP-X interpolates both arms' joints together).
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
TWO_ARM = True
#: CaP-X's frame (robot0's fixed_mount0_base) in the world: position, wxyz.
BASE0_POS = np.array([0.0, -0.56, 0.922])
BASE0_WXYZ = np.array([0.7071067811865476, 0.0, 0.0, 0.7071067811865476])
#: panda_hand -> robosuite's grip site: half a turn about the hand's z (wxyz).
HAND_TO_SITE_WXYZ = np.array([0.0, 0.0, 0.0, 1.0])
#: CaP-X's gripper-down orientation (wxyz).
DOWN_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])

#: goto_pose_both: the arms move in turns of at most this much each.
BOTH_LEG_M = 0.04


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


def get_handle0_pos():
    """CaP-X: handle 0's position (3,), robot0's frame."""
    return _from_world(ground_truth_poses(["pot_handle0"])["poses"]["pot_handle0"]["pos"])


def get_handle1_pos():
    """CaP-X: handle 1's position (3,), robot0's frame."""
    return _from_world(ground_truth_poses(["pot_handle1"])["poses"]["pot_handle1"]["pos"])

def goto_pose_both(position0, quaternion_wxyz0, position1, quaternion_wxyz1, z_approach=0.0):
    """CaP-X: both arms at once (robot0's frame), each first to its z_approach point."""
    p0 = np.asarray(position0, dtype=np.float64).reshape(3)
    p1 = np.asarray(position1, dtype=np.float64).reshape(3)
    if z_approach != 0.0:
        back = np.array([0.0, 0.0, -z_approach])
        _move_both(
            [
                ("robot0", p0 + _matrix(quaternion_wxyz0) @ back, quaternion_wxyz0),
                ("robot1", p1 + _matrix(quaternion_wxyz1) @ back, quaternion_wxyz1),
            ]
        )
    _move_both([("robot0", p0, quaternion_wxyz0), ("robot1", p1, quaternion_wxyz1)])


def _gripper_pose(arm):
    """CaP-X's robot<i>_cartesian_pos: the gripper body turned a quarter turn about its z and
    backed 0.107 m up its approach axis, which is the TCP's orientation (get_state's eef quat)
    at the TCP minus 0.107 m along the hand's z, in robot0's frame."""
    s = get_state()
    x, y, z, w = s[f"{arm}_eef_quat"]
    q = _unit([w, x, y, z])
    pos = np.asarray(s[f"{arm}_eef_pos"], dtype=np.float64) - 0.107 * _matrix(q)[:, 2]
    return _from_world(pos), _unit(_mul(_conj(BASE0_WXYZ), q))


def get_arm0_gripper_pose():
    """CaP-X: (position (3,), quaternion_wxyz (4,)) of arm 0's gripper, robot0's frame."""
    return _gripper_pose("robot0")


def get_arm1_gripper_pose():
    """CaP-X: (position (3,), quaternion_wxyz (4,)) of arm 1's gripper, robot0's frame."""
    return _gripper_pose("robot1")

def _move_both(targets):
    """Both arms together: legs of at most BOTH_LEG_M taken in turn (move_to moves one arm)."""
    starts = {arm: _eef(arm) for arm, _, _ in targets}
    ends = {arm: _to_world(p) for arm, p, _ in targets}
    legs = max(1, max(math.ceil(np.linalg.norm(ends[a] - starts[a]) / BOTH_LEG_M) for a in ends))
    for k in range(1, legs + 1):
        for arm, _, q in targets:
            leg = starts[arm] + (ends[arm] - starts[arm]) * k / legs
            r = move_to(leg.tolist(), quat_xyzw=_site_xyzw(q), max_steps=MOVE_STEPS, **_kw(arm))
            _EEF[arm] = np.asarray(r.get("final_eef_pos", leg), dtype=np.float64)


def goto_pose_arm0(position, quaternion_wxyz, z_approach=0.0):
    """CaP-X: arm 0 to a pose (robot0's frame), approaching from z_approach back along its axis."""
    _goto("robot0", position, quaternion_wxyz, z_approach)


def goto_pose_arm1(position, quaternion_wxyz, z_approach=0.0):
    """CaP-X: arm 1 to a pose given in robot0's frame."""
    _goto("robot1", position, quaternion_wxyz, z_approach)


def open_gripper_arm0():
    _grip("robot0", False, GRIPPER_STEPS[0])


def close_gripper_arm0():
    _grip("robot0", True, GRIPPER_STEPS[1])


def open_gripper_arm1():
    _grip("robot1", False, GRIPPER_STEPS[0])


def close_gripper_arm1():
    _grip("robot1", True, GRIPPER_STEPS[1])
