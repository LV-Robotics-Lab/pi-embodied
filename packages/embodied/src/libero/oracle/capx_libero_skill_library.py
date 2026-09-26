# CaP-X's reduced LIBERO API with the skill library (capx/integrations/franka/libero_reduced.py
# FrankaLiberoApiReduced + libero_reduced_skill_library.py FrankaLiberoApiReducedSkillLibrary
# @53e9966), the functions the skill-library oracle calls, over this server's high tier
# (get_observation, segment, get_state, move_to, rotate_wrist, set_gripper). --code-oracle
# prepends it to object_swap_7_skill_library.py.
#
# Differences from CaP-X:
# - get_observation returns CaP-X's layout: obs["agentview"] and obs["robot0_eye_in_hand"], each
#   {"images": {"rgb", "depth" (H, W)}, "intrinsics", "pose_mat" (camera to world)}, from the
#   server's 512x512 upright views.
# - segment_sam3_text_prompt(rgb, prompt): the server segments its current view, not a given
#   image, so the rgb must be one of the latest get_observation's images (the program only passes
#   those); the answer is the top mask, [{"mask", "box", "score"}] or [].
# - point_prompt_molmo and segment_sam3_point_prompt: no Molmo and no SAM3 point prompt here; they
#   raise, which the program's own fallback code catches (it reaches them only when every text
#   prompt found nothing).
# - rotation_matrix_to_quaternion, mask_to_world_points and pixel_to_world_point are CaP-X's
#   skill-library code (numpy only; also what --code-helpers injects).
# - goto_pose is the reduced API's: z_approach > 0 first goes that far straight up (world z),
#   then to the pose; CaP-X's IK clamps the target into a box of its base frame, which is not
#   reproduced (the server bounds the motion itself). Gripper: CaP-X's 30 control steps.
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


_get_observation = get_observation
_LAST = {}
_CAMS = {"agentview": "agentview", "robot0_eye_in_hand": "wrist"}


def get_observation():
    """CaP-X's observation layout over the server's get_observation."""
    raw = _get_observation()
    obs = {}
    for name, cam in _CAMS.items():
        view = raw[cam]
        obs[name] = {
            "images": {"rgb": np.asarray(view["rgb"]), "depth": np.asarray(view["depth"], dtype=np.float32)},
            "intrinsics": np.asarray(view["intrinsic_K"], dtype=np.float64),
            "pose_mat": np.asarray(view["extrinsic_cam2world"], dtype=np.float64),
        }
        _LAST[cam] = obs[name]["images"]["rgb"]
    return obs


def segment_sam3_text_prompt(rgb, text_prompt):
    """CaP-X: SAM3 masks of the prompt in `rgb` ([] when none); rgb is a latest observation image."""
    rgb = np.asarray(rgb)
    cam = next((c for c, img in _LAST.items() if img.shape == rgb.shape and np.array_equal(img, rgb)), None)
    if cam is None:
        raise ValueError("segment_sam3_text_prompt: pass an image of the latest get_observation()")
    seg = segment(text_prompt, camera=cam)
    if not seg.get("found"):
        return []
    return [{"mask": np.asarray(seg["mask"]).astype(bool), "box": seg.get("box"), "score": float(seg.get("score") or 0.0)}]


def segment_sam3_point_prompt(rgb, point_coords):
    raise NotImplementedError("segment_sam3_point_prompt: no SAM3 point prompt on this server")


def point_prompt_molmo(image, text_prompt):
    raise NotImplementedError("point_prompt_molmo: no Molmo on this server")


def rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z] (Sheppard's method)."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])


def mask_to_world_points(mask, depth, intrinsics, extrinsics):
    """The mask's pixels with depth > 0 as (N, 3) world points."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return np.empty((0, 3))
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    z_vals = depth[ys, xs]
    valid = z_vals > 0
    ys, xs, z = ys[valid], xs[valid], z_vals[valid]
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    points_cam = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=-1)
    points_cam_hom = np.hstack([points_cam, np.ones((len(points_cam), 1))])
    return (extrinsics @ points_cam_hom.T).T[:, :3]


def pixel_to_world_point(u, v, z, intrinsics, extrinsics):
    """Pixel (u = col, v = row) at depth z -> [x, y, z] world."""
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    p_cam = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z, 1.0])
    return (extrinsics @ p_cam)[:3]


def goto_pose(position, quaternion_wxyz, z_approach=0.0):
    """CaP-X (reduced API): z_approach > 0 first goes that far above the pose (world z)."""
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    if z_approach > 0.0:
        _move(pos + np.array([0.0, 0.0, z_approach]), quaternion_wxyz)
    _move(pos, quaternion_wxyz)


def open_gripper():
    _grip(False, 30)


def close_gripper():
    _grip(True, 30)
