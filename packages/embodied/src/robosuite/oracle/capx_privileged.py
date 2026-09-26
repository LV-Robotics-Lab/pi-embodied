# CaP-X's privileged robosuite API (capx/integrations/franka/control_privileged.py @53e9966) over
# this server's registry (robots/robosuite/primitives.py: ground_truth_poses, get_state, move_to,
# set_gripper). --code-oracle prepends it to the *_privileged oracles (their `# prelude:` line).
# The oracle defines OBJECTS: CaP-X's object names -> {"name": the ground_truth_poses name,
# "extent": CaP-X's hard-coded bounding-box extent, m}.
#
# Differences from CaP-X, all in how a pose is reached, none in what the programs ask for:
# - goto_pose servos the TCP (robosuite's grip site) with move_to instead of solving IK for
#   panda_hand, so no TCP offset is applied: CaP-X's positions are already the fingertip point.
# - A move longer than the server's per-call cap is split into straight legs of <= MAX_LEG_M.
# - Quaternions are CaP-X's wxyz; move_to takes xyzw (converted here).
import math

import numpy as np

MAX_LEG_M = 0.25
ARM = None


def _xyzw(q_wxyz):
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def _matrix(q_wxyz):
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64).reshape(4) / np.linalg.norm(q_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _eef():
    return np.asarray(get_state()[f"{ARM or 'robot0'}_eef_pos"], dtype=np.float64)


def _move(target, q_wxyz):
    target = np.asarray(target, dtype=np.float64).reshape(3)
    start = _eef()
    legs = max(1, math.ceil(np.linalg.norm(target - start) / MAX_LEG_M))
    for k in range(1, legs + 1):
        kw = {"arm": ARM} if ARM else {}
        move_to(
            (start + (target - start) * k / legs).tolist(),
            quat_xyzw=_xyzw(q_wxyz),
            max_steps=200,
            **kw,
        )


def _object(object_name):
    for key, spec in OBJECTS.items():
        if key == object_name:
            return spec
    raise ValueError(f"Invalid object name: {object_name}")


def get_object_pose(object_name, return_bbox_extent=False):
    """CaP-X: (position (3,), quaternion_wxyz (4,), bbox extent (3,)) from the simulator."""
    spec = _object(object_name)
    pose = ground_truth_poses([spec["name"]])["poses"][spec["name"]]
    x, y, z, w = pose["quat_xyzw"]
    return (
        np.asarray(pose["pos"], dtype=np.float64),
        np.array([w, x, y, z]),
        np.asarray(spec["extent"], dtype=np.float64),
    )


def sample_grasp_pose(object_name):
    """CaP-X: the object's position with the gripper pointing down (wxyz 0, 0, 1, 0)."""
    pos, _, _ = get_object_pose(object_name)
    return pos, np.array([0.0, 0.0, 1.0, 0.0])


def goto_pose(position, quaternion_wxyz, z_approach=0.0):
    """CaP-X: first position + z_approach back along the gripper's approach axis, then the pose."""
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    if z_approach != 0.0:
        _move(pos + _matrix(quaternion_wxyz) @ np.array([0.0, 0.0, -z_approach]), quaternion_wxyz)
    _move(pos, quaternion_wxyz)


def open_gripper():
    set_gripper(False, steps=40, **({"arm": ARM} if ARM else {}))


def close_gripper():
    set_gripper(True, steps=60, **({"arm": ARM} if ARM else {}))
