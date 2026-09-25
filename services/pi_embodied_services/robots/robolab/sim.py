# Copyright 2026 The Show-Harness Authors.
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
#
# Modified by pi-embodied: adapted from Show-Harness core/sim/robolab_task.py (@137d571);
# the Panda-hand relative-IK registration, the accessors, the orientation hold and the
# stepping helpers are kept, layout randomisation and the axis probe are dropped; the Kit app
# is pinned to one GPU (renderer included) and the short-finger Panda USD is resolved offline.

"""RoboLab (NVIDIA Isaac Lab) task glue: launch Kit, build one task in relative-IK mode.

Import order: Isaac Lab may only be imported after ``AppLauncher`` has started the Kit app,
and ``cv2`` before ``isaaclab``. Every isaaclab/robolab import therefore lives in a function
body; importing this module without Isaac Sim is safe.

The action is RoboLab's ``DroidRelIKActionCfg`` shape ``[dx, dy, dz, drx, dry, drz, gripper]``
on a Franka + Panda hand (``franka.py``): the first three are a base-frame displacement
divided by the arm action's ``scale`` (0.5), the rotation slots carry the orientation hold
(:func:`hold_orientation_rotvec`), and the gripper is binary (``> 0.5`` closes).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Show-Harness configs/robot_robolab.yaml camera presets (robolab/registrations/droid/camera_presets.py).
CAMERA_PRESETS = (
    "WRIST",
    "WRIST_LEFT",
    "WRIST_RIGHT",
    "WRIST_LEFT_RIGHT",
    "WRIST_LEFT_RIGHT_HEAD",
    "LEFT_RIGHT",
)
#: Isaac's public asset bucket the short-finger Panda USD references (Isaac Sim 5.0 layout).
S3_ISAAC_50 = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"
)
ASSETS = Path(__file__).resolve().parent / "assets"

# Native libraries Isaac Sim dlopen()s that are not pip dependencies. A missing one is reported
# as a mere [Error] line and the app segfaults a few plugins later (blaming rtx.mdltranslator).
_NATIVE_DEPS = {"libGLU.so.1": "libglu1-mesa"}


@dataclass
class TaskHandle:
    env: Any
    env_cfg: Any
    instruction: str
    env_name: str
    task: str
    ik_scale: float


# -- installation / app lifecycle -------------------------------------------


def robolab_root() -> Path:
    root = Path(os.environ.get("ROBOLAB_ROOT", Path.home() / "RoboLab"))
    if not (root / "robolab").is_dir():
        raise SystemExit(
            f"RoboLab checkout not found at {root} (looked for {root}/robolab). "
            "Clone https://github.com/NVLabs/RoboLab and set ROBOLAB_ROOT."
        )
    path = str(root)
    if path not in sys.path:
        sys.path.insert(0, path)
    return root


def check_native_deps() -> None:
    import ctypes

    missing = []
    for lib, package in _NATIVE_DEPS.items():
        try:
            ctypes.CDLL(lib)
        except OSError:
            missing.append((lib, package))
    if missing:
        raise SystemExit(
            f"Isaac Sim needs native libraries that are not installed: {[m[0] for m in missing]} "
            f"(apt install {' '.join(m[1] for m in missing)}). Without them it segfaults during "
            "stage creation with a misleading rtx.mdltranslator backtrace."
        )


def localize_panda_usd(out_dir: Path, isaac_assets: str | None) -> Path:
    """Write the short-finger Panda USD into ``out_dir`` with its references resolved.

    The shipped layer references Isaac's S3 bucket for the arm links and ``./Props`` for the
    fingers. With ``isaac_assets`` (a local mirror of the bucket's ``Assets/Isaac/5.0`` tree)
    the S3 prefix is rewritten to it, and every referenced file must exist, so a missing
    mirror file fails here instead of as an empty prim at stage load.
    """
    import re

    text = (ASSETS / "panda_short_finger.usda").read_text()
    text = text.replace("@./Props/", f"@{ASSETS / 'Props'}/")
    if isaac_assets:
        text = text.replace(S3_ISAAC_50, str(Path(isaac_assets).resolve()))
        refs = sorted(set(re.findall(r"references = @([^@]+)@", text)))
        missing = [r for r in refs if not Path(r).is_file()]
        if missing:
            raise SystemExit(f"--isaac-assets {isaac_assets} lacks {missing}")
    path = out_dir / "panda_short_finger.usda"
    path.write_text(text)
    return path


def launch_isaac(
    *, device: str, headless: bool = True, renderer_kwargs: dict | None = None
) -> Any:
    """Start the Kit app (MUST come before any isaaclab/robolab import) and return it.

    ``enable_cameras`` is forced on: without it Isaac Lab skips RTX sensor rendering and every
    camera observation comes back empty. The renderer is pinned to the simulation device and
    multi-GPU rendering is off, so Kit does not spread onto the other GPUs of the box (Vulkan
    ignores CUDA_VISIBLE_DEVICES).
    """
    robolab_root()
    check_native_deps()
    import argparse

    import cv2  # noqa: F401  -- must be imported before isaaclab. Do not remove.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    # An empty argv: the env server owns the command line.
    app_args = parser.parse_args([])
    app_args.headless = bool(headless)
    app_args.enable_cameras = True
    # AppLauncher sets the renderer's and physics' GPU to the device index; SimulationApp would
    # otherwise enable multi-GPU rendering on every GPU it finds.
    app_args.device = device
    app_args.multi_gpu = False
    for key, value in (renderer_kwargs or {}).items():
        setattr(app_args, key, value)
    return AppLauncher(app_args).app


# -- env construction --------------------------------------------------------


def make_task(
    task: str,
    *,
    device: str,
    seed: int = 0,
    instruction_type: str = "default",
    camera_preset: str = "WRIST_LEFT",
    robot: str = "franka",
    renderer: str = "realtime",
    rendering_type: str | None = None,
    output_dir: str | Path | None = None,
    enable_subtask: bool = False,
    episode_length_s: float | None = None,
) -> TaskHandle:
    """Register ``task`` against the relative-IK action space and construct its env.

    ``output_dir`` redirects RoboLab's own artefacts (``env_cfg.json``, the HDF5 recorder, which
    takes an exclusive lock: two servers must never share it). ``episode_length_s`` overrides the
    task's time limit; leave it None for evaluation (the benchmark's limit is part of the task).
    """
    robolab_root()
    import robolab.constants
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env
    from robolab.registrations.droid.auto_env_registrations_rel_ik import (
        auto_register_droid_rel_ik_envs,
    )
    from robolab.robots.droid import DroidRelIKActionCfg

    robolab.constants.VERBOSE = False
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = bool(enable_subtask)
    robolab.constants.RECORD_IMAGE_DATA = False
    if output_dir is not None:
        robolab.constants.set_output_dir(str(output_dir))

    if robot == "franka":
        ik_scale = _register_franka_rel_ik(task, camera_preset)
    elif robot == "droid":
        auto_register_droid_rel_ik_envs(
            task=task, cameras=_camera_preset(camera_preset)
        )
        ik_scale = float(getattr(DroidRelIKActionCfg().arm_action, "scale", 0.5))
    else:
        raise SystemExit(f"Unknown robot {robot!r}; choose 'franka' or 'droid'.")

    env_names = get_envs(task=task)
    if not env_names:
        raise SystemExit(
            f"No RoboLab environment registered for task {task!r}; task names are the class names "
            "in <ROBOLAB_ROOT>/robolab/tasks/benchmark/*.py, e.g. 'BananaInBowlTask'."
        )
    env, env_cfg = create_env(
        env_names[0],
        device=device,
        seed=seed,
        num_envs=1,
        use_fabric=True,
        instruction_type=instruction_type,
        policy="pi-embodied",
        renderer=renderer,
        rendering_mode=rendering_type,
    )
    if episode_length_s is not None:
        # max_episode_length is derived from cfg.episode_length_s on every read.
        env.cfg.episode_length_s = float(episode_length_s)
        env_cfg.episode_length_s = float(episode_length_s)
    # Do NOT clear env.recorder_manager: ManagerBasedEnv.reset() calls it unconditionally.
    return TaskHandle(
        env=env,
        env_cfg=env_cfg,
        instruction=rl_instruction(env_cfg),
        env_name=env_names[0],
        task=task,
        ik_scale=ik_scale,
    )


def _register_franka_rel_ik(task: str, camera_preset: str) -> float:
    """Register ``task`` against the Panda-hand embodiment (franka.py); returns the action scale.

    Mirrors ``robolab.registrations.droid.auto_env_registrations_rel_ik`` with our robot, action
    and camera configs. The physics timing (dt, decimation, render_interval) is RoboLab's,
    because the step-size calibration was measured against it.
    """
    from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import (
        generate_image_obs_from_cameras,
        generate_obs_cfg,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    from pi_embodied_services.robots.robolab.franka import (
        FrankaFrontCameraCfg,
        FrankaPandaCfg,
        FrankaProprioCfg,
        FrankaRelIKActionCfg,
        FrankaWristCameraCfg,
        contact_gripper,
    )

    # RoboLab's DROID cameras -> the ManiSkill-matched pair (wrist down the grasp axis, RLinf's
    # calibrated front view).
    swap = {
        "WristCameraCfg": FrankaWristCameraCfg,
        "OverShoulderLeftCameraCfg": FrankaFrontCameraCfg,
    }
    cameras = [swap.get(c.__name__, c) for c in _camera_preset(camera_preset)]
    image_obs = generate_image_obs_from_cameras(cameras)
    viewport = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    observations = generate_obs_cfg(
        {
            "image_obs": image_obs(),
            "proprio_obs": FrankaProprioCfg(),
            "viewport_cam": viewport(),
        }
    )
    # The wrist camera is robot-mounted (on FrankaPandaCfg); as a scene camera it would spawn
    # before its parent prim exists.
    scene_cameras = [c for c in cameras if c is not FrankaWristCameraCfg]
    actions = FrankaRelIKActionCfg()
    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=DEFAULT_TASK_SUBFOLDERS,
        tasks=task,
        pattern="*.py",
        env_prefix="",
        env_postfix="",
        observations_cfg=observations,
        actions_cfg=actions,
        robot_cfg=FrankaPandaCfg,
        camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        seed=1,
    )
    return float(getattr(actions.arm_action, "scale", 0.5))


def _camera_preset(name: str) -> list:
    from robolab.registrations.droid import camera_presets

    preset = getattr(camera_presets, str(name).upper(), None)
    if preset is None:
        raise SystemExit(
            f"Unknown camera preset {name!r}; choices: {list(CAMERA_PRESETS)}"
        )
    return preset


# -- tensor/obs accessors ----------------------------------------------------


def to_np(value: Any) -> np.ndarray:
    """Tensor (torch or warp) -> numpy, batch dim kept."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    try:
        import warp as wp

        if isinstance(value, wp.array):
            return wp.to_torch(value).detach().cpu().numpy()
    except Exception:  # noqa: BLE001
        pass
    return np.asarray(value)


def rl_rgb(obs: dict, camera_name: str) -> np.ndarray:
    """One camera's RGB (``obs["image_obs"][name]``, ``[num_envs, H, W, 3]``) as HWC uint8."""
    group = obs.get("image_obs")
    if group is None:
        raise KeyError(f"obs has no 'image_obs' group; keys: {sorted(obs)}")
    if camera_name not in group:
        raise KeyError(
            f"camera {camera_name!r} not in image_obs; available: {sorted(group)}"
        )
    arr = to_np(group[camera_name])
    if arr.ndim == 4:
        arr = arr[0]
    return np.ascontiguousarray(arr[..., :3].astype(np.uint8))


# The body the relative-IK action drives: the Panda hand first (franka.py), else the Robotiq
# flange of RoboLab's stock DroidCfg.
EE_BODY_CANDIDATES = ("panda_hand", "base_link")


def _ee_body_index(robot: Any) -> int:
    names = list(robot.data.body_names)
    for candidate in EE_BODY_CANDIDATES:
        if candidate in names:
            return names.index(candidate)
    raise SystemExit(
        f"No end-effector body among {list(EE_BODY_CANDIDATES)}; the robot has {names}."
    )


def rl_tcp(env: Any) -> np.ndarray:
    """End-effector (hand body) position (3,) in the env-local frame, noise-free."""
    robot = env.scene["robot"]
    pos = to_np(robot.data.body_pos_w)[:, _ee_body_index(robot), :]
    return (pos - to_np(env.scene.env_origins)[:, 0:3])[0].astype(float)


def rl_ee_quat(env: Any) -> np.ndarray:
    """The EE body's world orientation ``[qw, qx, qy, qz]``."""
    robot = env.scene["robot"]
    return to_np(robot.data.body_quat_w)[:, _ee_body_index(robot), :][0].astype(
        np.float64
    )


def ee_tilt_deg(quat_wxyz: np.ndarray) -> float:
    """Angle between the hand's approach axis (local +Z) and straight down."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return float(
        np.degrees(
            np.arccos(np.clip(float(axis @ np.array([0.0, 0.0, -1.0])), -1.0, 1.0))
        )
    )


def rl_gripper_width(env: Any) -> float:
    """Gripper opening in metres off the articulation (the proprio obs term carries noise).

    Panda hand: the sum of the two prismatic ``panda_finger_joint*``. Robotiq 2F-85: the
    ``finger_joint`` angle (0 open, pi/4 closed) mapped onto its 85 mm stroke.
    """
    robot = env.scene["robot"]
    names = list(robot.data.joint_names)
    qpos = to_np(robot.data.joint_pos)
    fingers = [i for i, n in enumerate(names) if n.startswith("panda_finger_joint")]
    if fingers:
        return float(qpos[0, fingers].sum())
    if "finger_joint" in names:
        frac = np.clip(
            float(qpos[0, names.index("finger_joint")]) / (np.pi / 4), 0.0, 1.0
        )
        return float((1.0 - frac) * 0.085)
    return 0.0


def rl_success(env: Any) -> bool:
    """The task's success predicate: RoboLab puts it in *terminated* (time_out is *truncated*);
    a terminated env is frozen and its result stays in ``_env_results``."""
    stored = getattr(env, "_env_results", {}).get(0)
    if stored is not None:
        return bool(stored)
    manager = getattr(env, "termination_manager", None)
    return bool(manager is not None and to_np(manager.terminated).reshape(-1)[0])


def rl_instruction(env_cfg: Any) -> str:
    instruction = getattr(env_cfg, "instruction", "")
    if isinstance(instruction, dict):
        return str(instruction.get("default", ""))
    return str(instruction)


# Proportional gain and per-step clamp of the orientation hold.
ORIENT_HOLD_GAIN = 1.0
ORIENT_HOLD_MAX_RAD = 0.15


def hold_orientation_rotvec(
    quat_ref: np.ndarray,
    quat_cur: np.ndarray,
    gain: float = ORIENT_HOLD_GAIN,
    max_rad: float = ORIENT_HOLD_MAX_RAD,
) -> np.ndarray:
    """World-frame axis-angle that pulls ``quat_cur`` back to ``quat_ref`` (both wxyz).

    Feeds the ``(drx, dry, drz)`` slots of the relative-IK action, which is what makes
    "rotation locked" true: zeros there mean "target the orientation you have now", so the
    saturated IK's orientation error would become the new setpoint (Show-Harness measured
    24.5 deg of drift over one RubiksCubeTask episode with zeros; this correction brings the
    final tilt to ~1.5 deg together with small per-step commands).
    """
    rw, rx, ry, rz = (float(v) for v in np.asarray(quat_ref, dtype=np.float64))
    iw, ix, iy, iz = (float(v) for v in np.asarray(quat_cur, dtype=np.float64))
    ix, iy, iz = -ix, -iy, -iz  # cur^-1
    q_err = np.array(
        [
            rw * iw - rx * ix - ry * iy - rz * iz,
            rw * ix + rx * iw + ry * iz - rz * iy,
            rw * iy - rx * iz + ry * iw + rz * ix,
            rw * iz + rx * iy - ry * ix + rz * iw,
        ]
    )
    if q_err[0] < 0.0:  # shortest arc
        q_err = -q_err
    vec = q_err[1:]
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.zeros(3)
    angle = 2.0 * float(np.arctan2(norm, float(np.clip(q_err[0], -1.0, 1.0))))
    rotvec = (vec / norm) * angle * float(gain)
    mag = float(np.linalg.norm(rotvec))
    return rotvec * (float(max_rad) / mag) if mag > float(max_rad) else rotvec


# -- stepping ----------------------------------------------------------------


def ensure_timeline_playing(max_updates: int = 2000) -> bool:
    """Pump the Kit app until the timeline plays: Isaac Lab advances physics only then, and
    after ``env.reset()`` in a headless app it can still be stopped (the arm silently never
    moves). RoboLab's own episode loop pumps at every step, so :func:`step` does too."""
    try:
        import omni.kit.app
        import omni.timeline
    except Exception:  # noqa: BLE001 -- not under a Kit app
        return False
    timeline = omni.timeline.get_timeline_interface()
    app = omni.kit.app.get_app()
    for _ in range(int(max_updates)):
        if timeline.is_playing():
            return True
        app.update()
    return timeline.is_playing()


def step(env: Any, action: np.ndarray) -> tuple[dict, bool, bool]:
    """One control step (``decimation`` 8 physics substeps at 1/120 s = 1/15 s); returns
    ``(obs, terminated, truncated)`` for env 0."""
    import torch

    ensure_timeline_playing()
    a = torch.as_tensor(
        np.asarray(action, dtype=np.float32)[None, :], device=env.device
    )
    obs, _reward, terminated, truncated, _info = env.step(a)
    return (
        obs,
        bool(to_np(terminated).reshape(-1)[0]),
        bool(to_np(truncated).reshape(-1)[0]),
    )


def reset(env: Any) -> dict:
    """Reset the scene for a new episode.

    ``reset_eval_state`` first: RobolabEnv freezes any env reset after it has stepped (in its
    eval loop a mid-run reset means "terminated, hold"), and a frozen env zeroes its actions.
    Reset twice, as RoboLab's own eval does: the first reset's RTX frame can be stale.
    """
    reset_state = getattr(env, "reset_eval_state", None)
    if callable(reset_state):
        reset_state()
    env.reset()
    obs, _ = env.reset()
    ensure_timeline_playing()
    return obs
