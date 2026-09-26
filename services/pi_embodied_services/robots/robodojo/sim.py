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
#
# The config assembly follows RoboDojo's src/eval_client/main.py (@726e9aa, RoboDojo
# Non-Commercial Research License); the task itself is RoboDojo's EvalEnv, built by its own
# create_eval_env, with the policy client stubbed out.

"""RoboDojo (Isaac Sim / Isaac Lab) task glue: launch Kit, build one task's ``EvalEnv``.

RoboDojo's ``create_eval_env`` derives an ``EvalEnv`` from the task class; its ``run_eval``
drives a whole episode against an XPolicyLab policy server. The env server instead drives the
same ``EvalEnv`` one action at a time: ``reset`` (one eval layout, picked by its id), then
``take_action`` / ``get_obs_batch`` / ``is_episode_end`` exactly as RoboDojo's own loop calls
them, so stepping, interpolation, gripper mapping and the success judgement are RoboDojo's.

Import order: Isaac Lab and RoboDojo's ``env`` package may only be imported after
``AppLauncher`` has started the Kit app. Every such import lives in a function body; importing
this module without Isaac Sim is safe (the tests use the pure helpers).
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import yaml

#: RoboDojo's capability dimensions (scripts/internal/task_inventory.py DIMENSION_TASKS); a
#: ``<task>_random`` variant inherits its base task's.
DIMENSIONS: dict[str, tuple[str, ...]] = {
    "generalization": (
        "stack_bowls",
        "push_T",
        "pack_objects_into_box",
        "fold_clothes",
        "hang_mugs",
        "sweep_blocks",
        "pour_liquid_into_cup",
        "make_toast",
        "arrange_largest_number",
        "sort_nesting_dolls_by_size",
        "store_laptop_and_headphones",
        "stack_blocks",
    ),
    "memory": (
        "cover_blocks",
        "match_and_pick_from_conveyor",
        "swap_blocks",
        "swap_T",
        "press_by_number",
        "imitate_sorting_sequence",
    ),
    "precision": (
        "fasten_screws",
        "plug_in_charger",
        "insert_tubes",
        "pour_balls_into_vase",
        "play_Xylophone",
        "deposit_coin",
        "insert_key",
        "build_tower",
    ),
    "long-horizon": (
        "fill_pen_holder",
        "classify_objects",
        "put_bottles_into_dustbin",
        "play_tic_tac_toe",
        "fill_egg_holder",
        "organize_table",
        "make_kong",
        "play_stacking_toy",
    ),
    "open": (
        "align_blocks",
        "general_pickup",
        "solve_equation",
        "stack_blocks_by_language",
        "classify_objects_by_language",
        "pick_from_conveyor_by_image",
        "store_tools_in_toolbox",
        "pour_by_language",
    ),
}
#: The eval config (env_cfg/<ENV_CFG_TYPE>.yml): the two ARX X5 arms, RoboDojo's arx_x5 profile.
ENV_CFG_TYPE = "arx_x5"
#: Every target camera of the dual X5 (env_cfg/camera/camera_config.yml plus the X5 wrist cameras).
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
ARMS = ("left", "right")
#: Kit settings that make each app update wait for its own frame (RoboDojo's render_sync).
ZERO_DELAY_KIT_ARGS = (
    "--/app/updateOrder/checkForHydraRenderComplete=1000",
    "--/app/renderer/waitIdle=true",
    "--/app/hydraEngine/waitIdle=true",
)
#: Isaac Sim 6.x moved the isaacsim.core.* / isaacsim.sensors.camera APIs RoboDojo is written
#: against to isaacsim/extsDeprecated; they are put back on the extension path and enabled.
EXTRA_EXTS = (
    "isaacsim.core.utils",
    "isaacsim.core.prims",
    "isaacsim.core.api",
    "isaacsim.sensors.camera",
)


def dimension(task: str) -> str | None:
    base = task.removesuffix("_random")
    return next((d for d, tasks in DIMENSIONS.items() if base in tasks), None)


# -- installation ------------------------------------------------------------


def robodojo_root() -> Path:
    """The RoboDojo checkout (``ROBODOJO_ROOT``), put on ``sys.path`` (RoboDojo imports ``env.*``,
    ``task.*``, ``utils.*`` as top-level packages)."""
    root = Path(os.environ.get("ROBODOJO_ROOT", Path.home() / "RoboDojo"))
    if not (root / "task" / "RoboDojo" / "task_registry.py").is_file():
        raise SystemExit(
            f"RoboDojo checkout not found at {root} (looked for task/RoboDojo/task_registry.py). "
            "Run services/pi_embodied_services/robots/robodojo/install_isaac61.sh and set ROBODOJO_ROOT."
        )
    path = str(root)
    if path not in sys.path:
        sys.path.insert(0, path)
    return root


def task_names(root: Path) -> list[str]:
    """The runnable tasks: a ``tasks/<name>.py`` with its ``config/<name>.yml`` (``_task.yml`` is
    the shared per-task settings file, not a task)."""
    tasks = root / "task" / "RoboDojo" / "tasks"
    configs = root / "task" / "RoboDojo" / "config"
    return sorted(
        p.stem
        for p in tasks.glob("*.py")
        if p.stem != "__init__" and (configs / f"{p.stem}.yml").is_file()
    )


def install_model_client_stub() -> None:
    """``src/eval_client/eval_env.py`` imports XPolicyLab's ``WsModelClient`` at module level and
    connects nothing until ``run_eval``; the env server never runs it, so a no-op client stands
    in (no XPolicyLab checkout needed). ``reset`` calls ``model_client.call("reset")``."""
    if "client_server.ws.model_client" in sys.modules:
        return

    class WsModelClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.calls: list[str] = []

        def call(self, func_name: str = "", **_: Any) -> None:
            self.calls.append(func_name)

        def close(self) -> None:
            pass

    for name in ("client_server", "client_server.ws"):
        mod = sys.modules.setdefault(name, types.ModuleType(name))
        mod.__path__ = []  # a namespace package for the submodule import
    stub = types.ModuleType("client_server.ws.model_client")
    stub.WsModelClient = WsModelClient
    sys.modules["client_server.ws.model_client"] = stub


def unstable_error() -> type[BaseException]:
    """RoboDojo's ``UnStableError``: the layout does not settle in simulation (it is skipped)."""
    from utils.cluttered_generator import UnStableError

    return UnStableError


# -- config assembly (src/eval_client/main.py) -------------------------------


def _yaml(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def task_settings(root: Path, task: str) -> dict[str, Any]:
    """``_task.yml``'s common settings overlaid with the task's (data_source, scene/robot/camera
    config, render_interval, robot_self_collision, eval_nums)."""
    info = _yaml(root / "task" / "RoboDojo" / "config" / "_task.yml")
    return {**(info.get("common") or {}), **((info.get("tasks") or {}).get(task) or {})}


def build_config(
    root: Path, task: str, *, eval_seed: int = 0, depth: bool = True
) -> tuple[dict[str, Any], int]:
    """The env config ``main.py`` hands ``create_eval_env`` for one env of ``task``, as a plain
    dict (the caller wraps it in OmegaConf), and the task's number of eval layouts.

    Deviations from main.py, all for one agent-driven env: ``num_envs`` 1, ``deploy_cfg`` points
    at no server (the client is stubbed), and with ``depth`` every camera also renders
    ``distance_to_image_plane`` (metric depth for back-projection; the RGB is unchanged).
    """
    cfg_root = root / "env_cfg"
    eval_cfg = _yaml(cfg_root / f"{ENV_CFG_TYPE}.yml")
    s = task_settings(root, task)
    names = eval_cfg["config"]
    sub = {
        "sim": _yaml(cfg_root / "sim" / f"{names['sim']}.yml"),
        "scene": _yaml(
            cfg_root / "scene" / f"{s.get('scene_config', names['scene'])}.yml"
        ),
        "camera": _yaml(
            cfg_root / "camera" / f"{s.get('camera_config', names['camera'])}.yml"
        ),
        "robot": _yaml(
            cfg_root / "robot" / f"{s.get('robot_config', names['robot'])}.yml"
        ),
    }
    # process_config: teleop tasks get PhysX stabilization, some tasks a finer render interval,
    # and some turn the arms' self-collision off.
    if s.get("data_source", "datagen") == "teleop":
        sub["sim"].setdefault("physx", {})["enable_stabilization"] = True
    if int(s.get("render_interval", 10)) != 10:
        sub["sim"]["render_interval"] = int(s["render_interval"])
    if not s.get("robot_self_collision", True):
        for r in sub["robot"]["robots"]:
            r["enabled_self_collisions"] = False
    # process_randomization: the eval profile randomizes nothing unless it says so.
    rand = eval_cfg.get("domain_randomization", {}) or {}
    sub["scene"].get("Table", {})["random"] = bool(rand.get("random_table", False))
    if "materials" in sub["scene"].get("Ground", {}):
        sub["scene"]["Ground"]["materials"]["random"] = bool(
            rand.get("random_ground", False)
        )
    sub["scene"].get("Background", {})["random"] = bool(
        rand.get("random_background", False)
    )
    sub["sim"].setdefault("scene", {})["num_envs"] = 1
    sub["sim"]["seed"] = [0]
    obs = eval_cfg.setdefault("observation", {})
    sub["camera"]["default_frequency"] = obs.get("collect_freq", 0)
    if depth:
        obs.setdefault("vision", {})["depth"] = True
        for name, ann in (sub["camera"].get("annotator") or {}).items():
            if isinstance(ann, dict) and ann.get("enabled", False):
                ann["distance_to_image_plane_capture"] = {
                    "type": "distance_to_image_plane",
                    "device": "cpu",
                }
    eval_num = int(s.get("eval_nums", 50))
    eval_cfg.update(
        task_name=task,
        num_envs=1,
        device_id=0,
        eval_batch=False,
        policy_name="pi-embodied",
        additional_info="pi",
        seed=int(eval_seed),
        physx_monitor_enabled=False,
        eval_num=eval_num,
    )
    task_env = _yaml(root / "task" / "RoboDojo" / "config" / f"{task}.yml")
    cfg = {
        **sub,
        "task_env": task_env,
        "eval_cfg": eval_cfg,
        "deploy_cfg": {
            "policy_name": "pi-embodied",
            "port": 0,
            "host": "127.0.0.1",
            "protocol": "ws",
            "policy_server_url": "ws://127.0.0.1:0",
            "evaluation_id": "pi-embodied",
            "trial_id": f"{task}-pi-embodied",
            "action_case_id": f"{task}_case",
            "repeat_index": None,
        },
    }
    return cfg, eval_num


def layout_count(root: Path, task: str, eval_seed: int = 0) -> int:
    """How many eval layouts ``Assets/Eval_Layout/RoboDojo/arx_x5/<eval_seed>`` holds for ``task``
    (SeedManager.init_eval: layout id i is the i-th ``<task>_<n>.json`` by n)."""
    import re

    d = root / "Assets" / "Eval_Layout" / "RoboDojo" / ENV_CFG_TYPE / str(eval_seed)
    pat = re.compile(rf"{re.escape(task)}_\d+\.json")
    return sum(1 for p in d.iterdir() if pat.fullmatch(p.name)) if d.is_dir() else 0


# -- app lifecycle ------------------------------------------------------------


def isaacsim_major() -> int:
    from importlib.metadata import version

    return int(version("isaacsim").split(".")[0])


def launch_isaac(*, headless: bool = True) -> Any:
    """Start the Kit app (MUST come before any isaaclab / RoboDojo ``env`` import) and return it.

    The GPU is chosen the way RoboDojo's eval_policy.sh does: the caller sets
    ``CUDA_VISIBLE_DEVICES`` to the one physical GPU before anything touches CUDA, and everything
    runs on ``cuda:0`` (RoboDojo hard-codes it: cuRobo's ``DeviceCfg``, its warp capture buffers).
    Kit then skips the hidden GPUs for rendering too. What RoboDojo's Isaac Lab fork
    (yuechen0614/IsaacLab @afca7b09) changes in ``AppLauncher`` is passed explicitly: cameras on
    (RTX sensors render headless), plus RoboDojo's render_sync zero-delay settings. On Isaac Sim 6
    the deprecated core extensions are added back (``EXTRA_EXTS``).
    """
    robodojo_root()
    import argparse

    import cv2  # noqa: F401  -- before isaaclab, as everywhere in Isaac Lab
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    args.headless = bool(headless)
    args.enable_cameras = True
    args.device = "cuda:0"
    args.multi_gpu = False
    kit = list(ZERO_DELAY_KIT_ARGS)
    if isaacsim_major() >= 6:
        import isaacsim

        deprecated = Path(isaacsim.__file__).resolve().parent / "extsDeprecated"
        kit.append(f"--ext-folder {deprecated}")
        kit += [f"--enable {e}" for e in EXTRA_EXTS]
    # Extra Kit settings for debugging, e.g. "--/crashreporter/enabled=false".
    kit += os.environ.get("ROBODOJO_KIT_ARGS", "").split()
    args.kit_args = " ".join(kit)
    return AppLauncher(args).app


def make_env(
    app: Any, root: Path, task: str, *, eval_seed: int = 0, depth: bool = True
) -> Any:
    """RoboDojo's ``EvalEnv`` for ``task`` (one env), from its own ``create_eval_env``.

    Its streaming video writers (``_stream_vision``, ffmpeg into ``eval_result/``) are turned
    off: the pi robot records the episode video itself. RoboDojo writes ``eval_result/`` and
    reads ``Assets/`` relative to the checkout, so the working directory becomes ``root``.
    """
    from omegaconf import OmegaConf

    install_model_client_stub()
    os.chdir(root)
    from src.eval_client.eval_env import create_eval_env

    cfg, _ = build_config(root, task, eval_seed=eval_seed, depth=depth)
    env = create_eval_env(OmegaConf.create(cfg), app)
    env._stream_vision = lambda *_a, **_k: None
    return env


# -- geometry -----------------------------------------------------------------


def quat_mul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def yaw_quat(q_wxyz, yaw: float) -> np.ndarray:
    """``q`` turned by ``yaw`` (rad) about world +z (counter-clockwise seen from above)."""
    turn = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    out = quat_mul(turn, np.asarray(q_wxyz, dtype=float))
    return out / np.linalg.norm(out)


def slerp(q0, q1, t: float) -> np.ndarray:
    """Spherical interpolation between wxyz quaternions (shortest arc)."""
    q0 = np.asarray(q0, dtype=float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=float) / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    a = np.arccos(d)
    return (np.sin((1 - t) * a) * q0 + np.sin(t * a) * q1) / np.sin(a)


def quat_angle(q0, q1) -> float:
    """The rotation angle (rad) between two wxyz quaternions."""
    d = abs(
        float(
            np.dot(
                np.asarray(q0) / np.linalg.norm(q0), np.asarray(q1) / np.linalg.norm(q1)
            )
        )
    )
    return float(2 * np.arccos(min(1.0, d)))


def waypoints(
    start, goal, q_start, q_goal, *, step_m: float, step_rad: float
) -> list[np.ndarray]:
    """Poses ``[x, y, z, qw, qx, qy, qz]`` from ``start`` to ``goal`` (excluded / included), one per
    native action: at most ``step_m`` of translation and ``step_rad`` of rotation apart."""
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    n = max(
        1,
        int(np.ceil(np.linalg.norm(goal - start) / step_m - 1e-9)),
        int(np.ceil(quat_angle(q_start, q_goal) / step_rad - 1e-9)),
    )
    return [
        np.concatenate(
            [start + (goal - start) * (i / n), slerp(q_start, q_goal, i / n)]
        )
        for i in range(1, n + 1)
    ]


def gl_to_cv(cam2world_gl: np.ndarray) -> np.ndarray:
    """A USD/OpenGL camera-to-world (x right, y up, looking down -z) as OpenCV's (y down, +z)."""
    return np.asarray(cam2world_gl, dtype=float) @ np.diag([1.0, -1.0, -1.0, 1.0])


def intrinsics(
    width: int, height: int, focal: float, h_aperture: float, v_aperture: float | None
) -> np.ndarray:
    """OpenCV K of a USD pinhole camera (focal length and apertures in the same unit); square
    pixels when the vertical aperture is unset."""
    fx = width * focal / h_aperture
    fy = height * focal / v_aperture if v_aperture else fx
    return np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])


def back_project(
    depth: np.ndarray, K: np.ndarray, cam2world_cv: np.ndarray, pixels, radius: int = 1
) -> list:
    """World xyz of each ``[col, row]`` pixel from metric depth along the optical axis (the
    ``distance_to_image_plane`` annotator), the median over a ``radius`` neighbourhood of valid
    depths; None where it has none."""
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    h, w = depth.shape
    K = np.asarray(K, dtype=float)
    out: list = []
    for col, row in pixels:
        c, r = int(round(col)), int(round(row))
        if not (0 <= c < w and 0 <= r < h):
            out.append(None)
            continue
        patch = depth[
            max(0, r - radius) : r + radius + 1, max(0, c - radius) : c + radius + 1
        ]
        valid = patch[np.isfinite(patch) & (patch > 0) & (patch < 1e4)]
        if valid.size == 0:
            out.append(None)
            continue
        z = float(np.median(valid))
        p = np.array(
            [(col - K[0, 2]) * z / K[0, 0], (row - K[1, 2]) * z / K[1, 1], z, 1.0]
        )
        out.append([round(float(v), 5) for v in (np.asarray(cam2world_cv) @ p)[:3]])
    return out


# -- readers over a live EvalEnv (need Isaac) ------------------------------------

#: Object types of RoboDojo's layout records (layout_manager.OBJECT_CONFIG_TYPES) that have a pose.
_POSED_TYPES = ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment")


def env_origin(env: Any) -> np.ndarray:
    """Env 0's origin in the stage (``is_relative`` poses subtract it)."""
    origins = env.scene_manager.env_origins if hasattr(env, "scene_manager") else None
    if origins is None:
        origins = env.env_origins
    return np.asarray(
        origins[0].cpu() if hasattr(origins[0], "cpu") else origins[0], dtype=float
    )[:3]


def camera_meta(env: Any, name: str, width: int, height: int) -> dict:
    """OpenCV K and camera-to-env (the frame of ``*_ee_pose``) of camera ``name``, read off its
    USD camera prim (focal length, apertures, world transform). RoboDojo's own
    ``get_camera_intrinsics`` assumes square pixels and ``get_camera_extrinsics`` returns the
    mount xform, not the optical frame, so neither is used."""
    from pxr import Usd, UsdGeom

    cm = env.camera_manager
    names = list(cm.camera_names[0])
    if name not in names:
        raise ValueError(f"unknown camera {name!r}; the scene has {names}")
    cam = cm.cameras[0][names.index(name)]
    prim = cam.prim if hasattr(cam, "prim") else None
    if prim is None:
        from isaacsim.core.utils.stage import get_current_stage

        prim = get_current_stage().GetPrimAtPath(cam.prim_path)
    ucam = UsdGeom.Camera(prim)
    focal = float(ucam.GetFocalLengthAttr().Get())
    hap = float(ucam.GetHorizontalApertureAttr().Get())
    vap = ucam.GetVerticalApertureAttr().Get()
    gl = np.array(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    ).T
    gl[:3, 3] -= env_origin(env)
    return {
        "intrinsic_K": intrinsics(
            width, height, focal, hap, float(vap) if vap else None
        ),
        "extrinsic_cam2world": gl_to_cv(gl),
        "width": int(width),
        "height": int(height),
        "frame": "env",
    }


def _np(v: Any) -> np.ndarray:
    return np.asarray(v.detach().cpu() if hasattr(v, "detach") else v, dtype=float)


def object_poses(env: Any) -> dict[str, dict]:
    """Every labelled object of the layout (RoboDojo's reward labels: ``bowl0``, ...) in the env
    frame, from the layout manager RoboDojo's success checks read."""
    from pi_embodied_services.utils import ground_truth

    lm = env.scene_manager.layout_manager
    out: dict[str, dict] = {}
    for t in _POSED_TYPES:
        records = lm.object_records_by_type.get(t)
        for rec in records.layout_records_by_env[0] if records is not None else []:
            label = rec.get("label")
            if not label or label in out:
                continue
            pos, rot = lm.get_instance_pose(env_idx=0, label=label, relative=True)
            if pos is None or rot is None:
                continue
            out[label] = ground_truth.pose(_np(pos)[:3], _np(rot)[:4])
    return out
