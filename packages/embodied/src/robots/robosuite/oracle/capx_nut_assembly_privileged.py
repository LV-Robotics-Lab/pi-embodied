# CaP-X's privileged nut-assembly API (capx/integrations/franka/nut_assembly_privileged.py
# FrankaControlNutAssemblyPrivilegedApi @53e9966) over this server's privileged tier
# (ground_truth_poses, get_state, move_to, set_gripper). --code-oracle prepends it to
# nut_assembly_privileged.py.
#
# Differences from CaP-X:
# - Poses are in CaP-X's frame, robot0's mount base (fixed_mount0_base, robosuite 1.5.2:
#   world (-0.56, 0, 0.922), unrotated). The nut, handle and peg poses are CaP-X's
#   (robosuite_nut_assembly.py _get_nut_pose) from ground_truth_poses' SquareNut and peg1: the
#   nut's pose turned half a turn about its z and flipped (x, -y, -z) to a grasp frame, the handle
#   0.054 m along the nut's x, the peg 0.1 m up its z.
# - goto_pose servos the TCP (the grip site) with move_to instead of solving IK for panda_hand:
#   CaP-X's positions are the fingertip point, so no TCP offset is applied. Quaternions are
#   CaP-X's panda_hand wxyz, turned half a turn about the hand's z into the grip site's frame
#   (HAND_TO_SITE_WXYZ, as the server's goto_pose). A move longer than the server's per-call cap is
#   split into straight legs of <= MAX_LEG_M. The path is a Cartesian line, not a joint-space one.
# - goto_home_joint_position: CaP-X drives the joints back to their reset values (to re-seed its
#   IK); the arms here are Cartesian-servoed, so the TCP returns to its reset position
#   (get_state's home_eef_pos) with CaP-X's gripper-down orientation, the reset one.
import math

import numpy as np

MAX_LEG_M = 0.25
MOVE_STEPS = 200
GRIPPER_STEPS = (40, 60)  # open, close
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


#: CaP-X's grasp-frame flip (x, -y, -z) and half turn about z.
_FLIP = np.diag([1.0, -1.0, -1.0])
_RZ180 = np.diag([-1.0, -1.0, 1.0])
HANDLE_OFFSET = np.array([0.054, 0.0, 0.0])
PEG_HEIGHT = 0.1


def _nut_poses():
    poses = ground_truth_poses(["SquareNut", "peg1"])["poses"]
    B = _matrix(BASE0_WXYZ)

    def rot(p):
        x, y, z, w = p["quat_xyzw"]
        return _matrix([w, x, y, z])

    nut_p, nut_R = np.asarray(poses["SquareNut"]["pos"], dtype=np.float64), rot(poses["SquareNut"])
    peg_p, peg_R = np.asarray(poses["peg1"]["pos"], dtype=np.float64), rot(poses["peg1"])
    grasp_R = _quat(B.T @ nut_R @ _RZ180 @ _FLIP)
    return {
        "square_nut": (_from_world(nut_p), grasp_R),
        "square_nut_handle": (_from_world(nut_p + nut_R @ HANDLE_OFFSET), grasp_R),
        "square_peg": (
            _from_world(peg_p + peg_R @ np.array([0.0, 0.0, PEG_HEIGHT])),
            _quat(B.T @ peg_R @ _FLIP),
        ),
    }


def get_object_pose(object_name):
    """CaP-X: (position (3,), quaternion_wxyz (4,)) of the nut, its handle or the peg."""
    poses = _nut_poses()
    if all(i in object_name for i in ["square", "nut", "handle"]):
        return poses["square_nut_handle"]
    elif all(i in object_name for i in ["square", "nut"]):
        return poses["square_nut"]
    elif any(i in object_name for i in ["block", "peg"]):
        return poses["square_peg"]
    else:
        raise ValueError(f"Invalid object name: {object_name}")


def sample_grasp_pose(object_name):
    """CaP-X: the handle's (or the peg's) pose."""
    poses = _nut_poses()
    if all(i in object_name for i in ["square", "nut", "handle"]):
        return poses["square_nut_handle"]
    elif any(i in object_name for i in ["peg", "block"]):
        return poses["square_peg"]
    else:
        raise ValueError(f"Invalid object name: {object_name}")

def goto_pose(position, quaternion_wxyz, z_approach=0.0):
    """CaP-X: first position + z_approach back along the gripper's approach axis, then the pose."""
    _goto("robot0", position, quaternion_wxyz, z_approach)


def goto_home_joint_position():
    """CaP-X: back to the reset joint configuration; here the TCP's reset position, pointing down."""
    home = np.asarray(get_state()["home_eef_pos"]["robot0"], dtype=np.float64)
    _move("robot0", _from_world(home), DOWN_WXYZ)


def open_gripper():
    _grip("robot0", False, GRIPPER_STEPS[0])


def close_gripper():
    _grip("robot0", True, GRIPPER_STEPS[1])
