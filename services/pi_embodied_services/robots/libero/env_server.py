# Copyright 2026 The RPent Authors.
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
# Modified by pi-embodied: import paths rewritten; healthz service name; HTTP is
# the only --transport.

"""RPC server wrapping a single-env LIBERO environment."""

from __future__ import annotations

import argparse
import functools
import io
import math
import os
import random
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.utils import (
    collision,
    ground_truth,
    motion,
    object_pose,
    reach,
)
from pi_embodied_services.utils.code_exec import CodeRunMixin
from pi_embodied_services.utils.geometry import (
    GripGeometry,
    jaw_frame,
    mujoco_grip_state,
    quat_to_matrix,
    rotvec_of,
)
from pi_embodied_services.utils.grasp import (
    GraspPlanner,
    add_grasp_arguments,
    orientation_error,
    pitch_of,
    quat_xyzw_matrix,
    urls_from_args,
    yaw_of,
)
from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.perception import (
    add_perception_arguments,
    install_perception,
)
from pi_embodied_services.utils.serialization import to_numpy_tree

# MuJoCo env vars must be set BEFORE importing anything that touches MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
assert "mujoco" not in sys.modules, (
    "mujoco must not be imported before MUJOCO_GL/PYOPENGL_PLATFORM are set"
)

logger = get_logger("env_server")

os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

#: The grasp planner's cameras (alias -> LIBERO camera) and its render size.
GRASP_CAMERAS = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}
GRASP_RES = 512
#: Code mode reads the same cameras at the same size; one run returns at most this many
#: agentview frames (one per mutating primitive) for the episode video.
CODE_CAMERAS = GRASP_CAMERAS
CODE_RES = GRASP_RES
CODE_MAX_FRAMES = 32
#: The longest horizontal move one move_to (or an executor's first leg) makes: a longer one
#: sweeps low over the scene; split it into waypoints at carry height.
MAX_XY_MOVE_M = 0.30
#: A program's raw ``chunk_step`` runs at most this many actions in one call (a chunk is one
#: worker call that no stop interrupts; the run's wall clock must bound it).
CODE_MAX_CHUNK = 64
#: A program's ``render_camera`` renders at most this many pixels per side.
CODE_MAX_RENDER = 1024
#: goto_pose / home_pose split a move into straight move_to legs of at most this length.
MAX_LEG_M = 0.25
#: A goto_pose leg's move_to step budget.
LEG_STEPS = 120
#: The facade's motions beyond utils/detections.MOTION_METHODS: every planned id expires after
#: them (the grasp planner's and perception's id epochs).
LIBERO_MOTIONS = (
    "env.move_pose",
    "env.rotate_pitch",
    "env.release",
    "env.goto_pose",
    "env.home_pose",
    "env.open_gripper",
    "env.close_gripper",
)
#: The motions pi's tools run with ``tool_call=True`` (LiberoEnvFacade._for_tools).
TOOL_MOTIONS = (
    "env.move_to",
    "env.move_pose",
    "env.rotate_wrist",
    "env.rotate_pitch",
    "env.release",
    "env.set_gripper",
    "env.goto_pose",
    "env.home_pose",
    "env.open_gripper",
    "env.close_gripper",
)


class Refused(ValueError):
    """A motion refused before anything moved (too long in xy, unreachable, no collision-free
    path): a program gets the error, pi's tool shows it as the result's ``refused``."""


def _current_only(resolution, step) -> None:
    """The server holds the current state at 512x512: pi's tools look back at earlier states
    and read their 1024 images themselves."""
    if resolution not in (None, "low", "high") or step not in (None, -1):
        raise ValueError(
            "the env server reads the current 512x512 images only (no step / resolution)"
        )


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _finite(name: str, value) -> np.ndarray:
    """``value`` as float64, or a ValueError naming ``name`` when it holds NaN or infinity."""
    arr = np.asarray(value, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite, got {value!r}")
    return arr


# torch and LiberoEnv are only imported at call time (after --cuda-device
# sets CUDA_VISIBLE_DEVICES in main()); LiberoEnv transitively imports torch.
if TYPE_CHECKING:
    import torch  # noqa: F401  (transitive dep of LiberoEnv; type-check only)
    from rlinf.envs.libero.libero_env import LiberoEnv


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def build_env_cfg(
    *,
    task_suite_name: str = "libero_spatial",
    specific_reset_id: int = 0,
    seed: int = 0,
    max_episode_steps: int = 10000,
) -> Any:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "env_type": "libero",
            "task_suite_name": task_suite_name,
            "auto_reset": False,
            # Keep stepping after success (a units finish releases and lifts); success is
            # reported from RLinf's latched success_once instead (LiberoEnvFacade._succeeded).
            "ignore_terminations": True,
            "max_steps_per_rollout_epoch": max_episode_steps,
            "max_episode_steps": max_episode_steps,
            "use_rel_reward": False,
            "use_step_penalty": False,
            "reward_coef": 1.0,
            "reset_gripper_open": True,
            "is_eval": True,
            "seed": seed,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "use_ordered_reset_state_ids": True,
            "specific_reset_id": specific_reset_id,
            "video_cfg": {
                "save_video": True,
                "info_on_video": True,
                "video_base_dir": "/tmp/primitive_videos",
            },
            "init_params": {
                "camera_heights": 256,
                "camera_widths": 256,
                # Render depth too, so we can back-project pixels to world
                # from depth + camera calibration
                "camera_depths": True,
                "horizon": max_episode_steps,
                **(
                    {"robots": [os.environ["LIBERO_ROBOT_BASE"]]}
                    if os.environ.get("LIBERO_ROBOT_BASE")
                    else {}
                ),
            },
        }
    )
    return cfg


def _seeding_globals(env_fn):
    """Wrap a LIBERO worker ``env_fn`` so the env's ``seed()`` also seeds the
    worker process's global numpy and Python RNGs.

    LiberoEnv seeds its worker env right before every reset, but robosuite 1.5
    only stores that seed while its reset draws from the global RNGs, which
    each spawned worker leaves unseeded; the same actions then drift apart
    between processes (measured: up to 8.8 cm over one 835-step episode).
    With this every reset restores the same RNG state, so a replay is bitwise.
    """

    def fn():
        env = env_fn()
        seed_env = env.seed

        def seed(value):
            random.seed(value)
            np.random.seed(int(value) % 2**32)
            return seed_env(value)

        env.seed = seed
        return env

    return fn


def _exposing_poses(env_fn):
    """Wrap a LIBERO worker ``env_fn`` so the env answers ``ground_truth_poses()`` (through
    the worker's ``env_call``): the world poses of every body in LIBERO's own object list,
    ``obj_body_id`` (its movable objects and fixtures). The sim lives only in the worker
    process. It never raises: the worker loop (rlinf/envs/libero/venv.py ``_worker``) has no
    try/except around ``env_call``, so an exception there kills the worker and the env with
    it; a failure comes back as ``{"error": ...}`` and the facade raises it instead."""

    def fn():
        env = env_fn()

        def poses():
            try:
                rob = env
                while hasattr(rob, "env"):
                    rob = rob.env
                return ground_truth.mujoco_body_poses(rob.sim, rob.obj_body_id)
            except Exception as e:  # noqa: BLE001
                return {"error": f"{type(e).__name__}: {e}"}

        def robot_base_pose():
            # The arm's base body (where the IK model's link0 sits), for env.preview_reach.
            try:
                rob = env
                while hasattr(rob, "env"):
                    rob = rob.env
                body = "robot0_base"
                return ground_truth.mujoco_body_poses(
                    rob.sim, {body: rob.sim.model.body_name2id(body)}
                )[body]
            except Exception as e:  # noqa: BLE001
                return {"error": f"{type(e).__name__}: {e}"}

        def collision_world():
            # The planning world of env.plan_motion / move_to under --ik (utils/motion.py):
            # every collidable non-robot geom, world frame.
            try:
                rob = env
                while hasattr(rob, "env"):
                    rob = rob.env
                base = rob.sim.data.body_xpos[rob.sim.model.body_name2id("robot0_base")]
                return {"obstacles": collision.mujoco_collision_world(rob.sim, base)}
            except Exception as e:  # noqa: BLE001
                return {"error": f"{type(e).__name__}: {e}"}

        def grip_geometry():
            # The grip site, the finger pads and the robot's contacts (utils/geometry.py, --geometry).
            try:
                rob = env
                while hasattr(rob, "env"):
                    rob = rob.env
                return mujoco_grip_state(rob.sim)
            except Exception as e:  # noqa: BLE001
                return {"error": f"{type(e).__name__}: {e}"}

        env.ground_truth_poses = poses
        env.robot_base_pose = robot_base_pose
        env.collision_world = collision_world
        env.grip_geometry = grip_geometry
        return env

    return fn


def make_env(
    task_id: int,
    seed: int,
    suite_name: str = "libero_spatial",
    max_episode_steps: int = 10000,
) -> LiberoEnv:
    """Build a single-env LiberoEnv pinned to ``task_id`` / ``seed``."""
    from rlinf.envs.libero.libero_env import LiberoEnv
    from rlinf.envs.libero.utils import benchmark as _bench_mod

    suite = _bench_mod.get_benchmark(suite_name)()
    first_id = sum(len(suite.get_task_init_states(t)) for t in range(task_id))
    trials = len(suite.get_task_init_states(task_id))
    rid = first_id + (seed % trials)
    cfg = build_env_cfg(
        task_suite_name=suite_name,
        specific_reset_id=rid,
        seed=seed,
        max_episode_steps=max_episode_steps,
    )

    class SeededLiberoEnv(LiberoEnv):
        def get_env_fns(self):
            return [
                _exposing_poses(_seeding_globals(fn)) for fn in super().get_env_fns()
            ]

    return SeededLiberoEnv(
        cfg=cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None
    )


# ---------------------------------------------------------------------------
# Facade implementing the pi_embodied_services.robots.libero.env_client protocol
# ---------------------------------------------------------------------------


class LiberoEnvFacade(CodeRunMixin, BaseEnvFacade):
    """Implements :class:`pi_embodied_services.robots.libero.env_client.LiberoEnvClient`
    over :class:`rlinf.envs.libero.libero_env.LiberoEnv`.

    All return values are converted to CPU numpy so the agent process
    (which does not import torch) can consume them after the pickle round
    trip.
    """

    SERVICE_NAME = "libero-env"
    #: Set by __init__; class defaults so the registry can be built on a bare facade (tests).
    _sam3_url: str | None = None
    _grasp: "GraspPlanner | None" = None
    _motion: "motion.MotionPlanner | None" = None
    _plan: dict | None = None
    _geometry: GripGeometry | None = None
    _reach: "reach.ReachPreview | None" = None
    #: The env steps of a tool's motion (``tool_call``), else None.
    _record: list | None = None
    #: The gripper's reset pose (position, xyzw), home_pose's target.
    _home: tuple | None = None

    def __init__(
        self,
        env: LiberoEnv,
        *,
        meta: dict,
        sam3: str | None = None,
        grasp: dict | None = None,
        ik_reach: reach.ReachPreview | None = None,
        ik_motion: motion.MotionPlanner | None = None,
        geometry: bool = False,
    ):
        self._env = env
        self._env_idx = 0
        self._closed = False
        # --ik: env.preview_reach (utils/reach.py); the robot base pose it converts world
        # targets with is read from the worker once per reset.
        self._reach = ik_reach
        self._base_pose: dict | None = None
        # --ik: env.plan_motion / env.check_motion, and move_to plans a collision-free path
        # through the scene and checks the arm before each servo segment (utils/motion.py).
        self._motion = ik_motion
        self._plan: dict | None = None
        # Identifies what task/seed this server was launched with — the
        # client compares against its own expected values at construction
        # and refuses to talk to a stale or mis-configured server.
        self._meta = dict(meta)
        # Code mode: the episode state its primitives read (LIBERO's latched success and
        # truncation, the last commanded gripper, the latest obs) and the SAM3 server `segment`
        # asks; a run's step count, first success and frames are collected between begin/finish.
        self._sam3_url = sam3
        self._sam3 = None
        self._terminated = False
        self._truncated = False
        self._grip = -1.0
        self._last_obs: dict | None = None
        self._run_steps = 0
        self._run_success: int | None = None
        self._run_frames: list = []
        # --contact-graspnet/--graspgenx/--anygrasp/--graspnet1b/--anyplace: env.plan_grasp, env.plan_place and the
        # grasp/placement ids over the 512x512 upright agentview / wrist frames (utils/grasp.py);
        # None without them, and the server is unchanged.
        self._grasp = GraspPlanner.from_args(
            self._view,
            cameras=list(GRASP_CAMERAS),
            sam3=sam3,
            eef_pose=lambda arm: (self._eef(), self._quat_xyzw()),
            wrist_camera="wrist",
            state_digest=self._state_digest,
            holding=lambda arm: self._holding(),
            **(grasp or {}),
        )
        # --geometry: point-cloud views, marked points and grip-site targets (utils/geometry.py)
        # over the same 512x512 agentview / wrist frames; None without it.
        self._grip_mech_frame: np.ndarray | None = None
        if geometry:
            self._geometry = GripGeometry(
                self._view,
                cameras=list(GRASP_CAMERAS),
                tool_pose=lambda: (self._eef(), self._quat_xyzw()),
                state_digest=self._state_digest,
                frame=self._grip_frame,
                pads=self._grip_pads,
                contacts=lambda: self._grip_mech()["contacts"],
                width=self._gripper_width,
                empty_width=0.004,
                envelope=self._workspace_envelope,
                move=self._move_grip,
                gripper=lambda close: self._actuate(1.0 if close else -1.0),
            )
        super().__init__()

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc.update(
            {
                "env.raw_obs": self.raw_obs,
                "env.code_raw_obs": self.code_raw_obs,
                "env.render_camera": self.render_camera,
                "env.get_camera_meta": self.get_camera_meta,
                "env.get_task_language": self.get_task_language,
                "env.ground_truth_poses": self.ground_truth_poses,
                "env.preview_reach": self.preview_reach,
                **(
                    {
                        "env.plan_motion": self.plan_motion,
                        "env.check_motion": self.check_motion,
                    }
                    if self._motion is not None
                    else {}
                ),
            }
        )
        # The primitives of manifests/libero.json that the facade serves, registered before the
        # grasp planner wraps the mutating ones with its id invalidation. The motions pi's tools
        # run take `tool_call` (_for_tools): the result carries every env step for pi's video
        # and recorder, and a refusal is a result.
        self._rpc.update(
            {
                "env.get_state": self.get_state,
                "env.get_observation": self.get_observation,
                "env.back_project": self.back_project,
                "env.move_to": self.move_to,
                "env.move_pose": self.move_pose,
                "env.move_delta": self.move_delta,
                "env.rotate_wrist": self.rotate_wrist,
                "env.rotate_pitch": self.rotate_pitch,
                "env.rotate_delta": self.rotate_delta,
                "env.set_gripper": self.set_gripper,
                "env.release": self.release,
                **({"env.segment": self.segment} if self._sam3_url else {}),
                # CaP-X's high tier and its privileged variants.
                "env.get_object_pose": self.get_object_pose,
                "env.get_object_pose_privileged": self.get_object_pose_privileged,
                "env.sample_grasp_pose": self.sample_grasp_pose,
                "env.sample_grasp_pose_privileged": self.sample_grasp_pose_privileged,
                "env.goto_pose": self.goto_pose,
                "env.home_pose": self.home_pose,
                "env.open_gripper": self.open_gripper,
                "env.close_gripper": self.close_gripper,
            }
        )
        for name in TOOL_MOTIONS:
            self._rpc[name] = self._for_tools(
                name.removeprefix("env."), self._rpc[name]
            )
        self._readonly_methods.add("env.get_task_language")
        if self._geometry is not None:
            # Before the grasp planner, which wraps env.move_grip with its id invalidation.
            self._geometry.install(self)
        if self._grasp is not None:
            # Wrapped with the id invalidation like every motion (GraspPlanner.MUTATING).
            self._rpc["env.execute_grasp"] = self.execute_grasp
            self._rpc["env.execute_place"] = self.execute_place
            self._grasp.install(self, mutating=GraspPlanner.MUTATING + LIBERO_MOTIONS)
        # Code mode (run_code): code.api, the programs' whitelist and the startup self-check come
        # from packages/embodied/src/primitives/manifests/libero.json (with pi); the runner adds
        # the sandbox, the budgets and the stop handling.
        self._manifest_code_run(
            "libero",
            have=self._has,
            move_m=self._code_move_m,
            after=self._frame,
            check=self._code_check,
            begin=self._begin_run,
            finish=self._finish_run,
        )

    def _has(self, capability: str) -> bool:
        """What this server can serve of the manifest's ``requires``."""
        grasp = self._grasp
        return {
            "sam3": bool(self._sam3_url),
            "detections": "env.detect" in self._rpc,
            "ik": self._reach is not None,
            "motion": self._motion is not None,
            "grasp": grasp is not None,
            "place": grasp is not None and bool(grasp.capabilities().get("place")),
            "geometry": self._geometry is not None,
            "unidepth": "env.enhance_depth" in self._rpc,
            "privileged": True,
        }.get(capability, False)

    def _for_tools(self, name: str, fn):
        """``fn`` as pi's tools call it: with ``tool_call=True`` the result carries every env
        step it took (``transitions``: action, obs, reward, terminated, truncated; pi's episode
        video and Flywheel recorder) and a :class:`Refused` motion is a result with ``refused``.
        A program never passes ``tool_call`` (the manifest does not declare it)."""

        @functools.wraps(fn)
        def call(*args, tool_call: bool = False, **kwargs):
            if not tool_call:
                return fn(*args, **kwargs)
            self._record = []
            try:
                try:
                    out = fn(*args, **kwargs)
                except Refused as exc:
                    out = self._motion_result(name, 0, False, refused=str(exc))
                out["transitions"] = self._record
                return out
            finally:
                self._record = None

        return call

    # ---- grasp planning views ----

    def _eef(self) -> np.ndarray:
        return np.asarray(self.raw_obs()["robot0_eef_pos"], dtype=np.float64).reshape(3)

    def _quat_xyzw(self) -> np.ndarray:
        return np.asarray(self.raw_obs()["robot0_eef_quat"], dtype=np.float64).reshape(
            4
        )

    def _state_digest(self) -> tuple:
        """The sim's low-dim state (robot and object poses, finger joints), exactly: grasp and
        mask ids expire only when it changed, not on a call that did not step the sim."""
        raw = self.raw_obs()
        return tuple(
            (k, np.asarray(v).tobytes())
            for k, v in sorted(raw.items())
            if k.endswith(("_pos", "_quat", "_qpos"))
        )

    def _holding(self) -> bool:
        """Whether the fingers rest on something: neither open (about 0.08) nor closed on
        nothing (about 0)."""
        return 0.004 < self._gripper_width() < 0.075

    def _view(self, camera: str) -> dict:
        """One camera, upright: rgb uint8[S,S,3], depth float32[S,S] in metres (0 = none), K,
        cam2world. LIBERO renders upside down and its depth buffer is normalized (near/far);
        pi's tools flip and linearize the same way, so pixel (row, col) back-projects with K."""
        name = GRASP_CAMERAS[camera]
        rgb, depth = self.render_camera(name, GRASP_RES, GRASP_RES, depth=True)
        meta = self.get_camera_meta(name, GRASP_RES, GRASP_RES) or {}
        z = np.asarray(depth, dtype=np.float64).reshape(GRASP_RES, GRASP_RES)
        near, far = meta.get("depth_near"), meta.get("depth_far")
        if near is not None and far is not None:
            z = near / (1 - z * (1 - near / far))
        return {
            "rgb": np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)[::-1]),
            "depth": np.ascontiguousarray(z[::-1].astype(np.float32)),
            "intrinsic_K": np.asarray(meta["intrinsic_K"], dtype=np.float64),
            "extrinsic_cam2world": np.asarray(
                meta["extrinsic_cam2world"], dtype=np.float64
            ),
        }

    # ---- shape helpers ----

    def _strip(self, v):
        """Drop the leading env dim. ``v`` is either a batched numpy array
        (shape ``[B, ...]``), a length-B list (e.g. ``task_descriptions``),
        or ``None`` (optional images). LiberoEnv runs ``num_envs=1`` so
        index ``self._env_idx`` is always present."""
        if v is None:
            return None
        return v[self._env_idx]

    def _strip_obs(self, obs: dict) -> dict:
        """Strip the leading env dim from every value of a LIBERO obs dict."""
        return {k: self._strip(v) for k, v in obs.items()}

    def _expand_action(self, action) -> np.ndarray:
        """Inject the env dim onto a single-env action shaped ``[action_dim]``."""
        return np.asarray(action)[None]

    def _expand_chunk(self, actions) -> np.ndarray:
        """Inject the env dim onto a single-env chunk shaped
        ``[chunk_size, action_dim]``."""
        return np.asarray(actions)[None]

    # ---- gym-like surface ----

    def reset(self):
        obs, info = self._env.reset()
        obs = self._strip_obs(to_numpy_tree(obs))
        self._base_pose = None
        self._terminated = self._truncated = False
        self._grip = -1.0
        self._last_obs = obs
        self._home = (self._eef(), self._quat_xyzw())
        return obs, to_numpy_tree(info)

    def _succeeded(self, info) -> np.bool_:
        """LIBERO's success at or before this step. With ``ignore_terminations`` RLinf zeroes
        ``terminations`` and keeps stepping; its ``episode.success_once`` still latches success."""
        return np.bool_(self._strip(to_numpy_tree(info)["episode"]["success_once"]))

    def step(self, action):
        obs, rew, _term, trunc, info = self._env.step(self._expand_action(action))
        obs = self._strip_obs(to_numpy_tree(obs))
        term = self._succeeded(info)
        trunc = self._strip(to_numpy_tree(trunc))
        self._absorb(obs, bool(term), bool(np.any(trunc)))
        self._count_run_steps([bool(term)])
        if self._record is not None:
            self._record.append(
                {
                    "action": np.asarray(action, dtype=np.float32).reshape(-1),
                    "obs": obs,
                    "reward": self._strip(to_numpy_tree(rew)),
                    "terminated": bool(term),
                    "truncated": bool(np.any(trunc)),
                }
            )
        return (
            obs,
            self._strip(to_numpy_tree(rew)),
            term,
            trunc,
            to_numpy_tree(info),
        )

    def _absorb(self, obs: dict, term: bool, trunc: bool) -> None:
        """Episode state for the code primitives: success latches, truncation ends the episode."""
        self._terminated = self._terminated or term
        self._truncated = self._truncated or trunc
        self._last_obs = obs

    def chunk_step(self, actions, *, return_all_frames: bool = False):
        """Run a full action chunk in one RPC. ``actions`` shape
        ``[chunk_size, action_dim]`` (single env).

        Returns the 5-positional tuple
        ``(obs_or_list, reward, terminated, truncated, info)``. ``obs`` is
        ``list[Obs]`` when ``return_all_frames=True`` (full per-step
        trajectory), or just the final ``Obs`` dict when False (default).
        ``terminated`` / ``truncated`` carry shape ``[chunk_size]`` after
        the leading env dim is stripped — the agent reduces across the
        chunk itself.
        """
        obs_list, rew, _term, trunc, info = self._env.chunk_step(
            self._expand_chunk(actions)
        )
        obs_list = [self._strip_obs(to_numpy_tree(o)) for o in obs_list]
        term = np.array([self._succeeded(i) for i in info], dtype=bool)
        trunc = self._strip(to_numpy_tree(trunc))
        self._absorb(obs_list[-1], bool(term.any()), bool(np.any(trunc)))
        self._count_run_steps([bool(t) for t in term])
        obs_field = obs_list if return_all_frames else obs_list[-1]
        return (
            obs_field,
            self._strip(to_numpy_tree(rew)),
            term,
            trunc,
            to_numpy_tree(info),
        )

    def raw_obs(self) -> dict:
        return to_numpy_tree(self._env.current_raw_obs[self._env_idx])

    def code_raw_obs(self) -> dict:
        """The low tier's ``raw_obs``: the robot's own keys (``robot0_*``) and the images. The
        object poses in LIBERO's raw observation are privileged (``ground_truth_poses``)."""
        return {
            k: v
            for k, v in self.raw_obs().items()
            if k.startswith("robot0_") or k.endswith("_image") or k.endswith("_depth")
        }

    def get_env_meta(self) -> dict:
        """Return the meta info this server was launched with."""
        return dict(self._meta)

    def close(self) -> None:
        """Release the server-owned LIBERO environment once."""
        if self._closed:
            return
        self._env.env.close()
        self._closed = True

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ):
        return to_numpy_tree(
            self._env.render_camera(
                camera_name=camera_name,
                height=height,
                width=width,
                depth=depth,
            )
        )

    def get_camera_meta(
        self,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ) -> dict | None:
        return to_numpy_tree(
            self._env.get_camera_meta(
                camera_name=camera_name, height=height, width=width
            )
        )

    def get_task_language(self) -> str | None:
        return self._env.task_descriptions[self._env_idx]

    def ground_truth_poses(self, names=None) -> dict:
        """World poses of ``names`` (default all) from LIBERO's object list (``--privileged``).
        Unknown names are refused here, before the worker is asked."""
        worker = self._env.env.workers[self._env_idx]
        poses = worker.env_call("ground_truth_poses", target="self")
        if isinstance(poses.get("error"), str):
            raise RuntimeError(
                f"ground_truth_poses failed in the worker: {poses['error']}"
            )
        return ground_truth.respond(poses, names)

    # ---- code mode (run_code) --------------------------------------------------------------
    #
    # The facade methods behind manifests/libero.json's code primitives: programs reach them
    # only through CodeApi.resolve, and they step the env as pi's LIBERO tools do (`env.step`), with
    # LIBERO's success latched and `stop` honoured between env steps. Images come from `_view`
    # (512x512, upright), so pixel (row, col) of `get_observation` back-projects with its K.

    def _begin_run(self) -> None:
        self._run_steps = 0
        self._run_success = None
        self._run_frames = []

    def _finish_run(self) -> dict:
        """The run's effect for pi: env steps taken, the first success within them, the episode
        flags, the latest obs (pi's `Obs`), and one agentview frame per motion primitive."""
        return {
            "steps": self._run_steps,
            "success_step": self._run_success,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "obs": self._last_obs,
            "frames": list(self._run_frames),
        }

    def _code_move_m(self, method: str, kwargs: dict) -> float:
        """How far a program's call may move the arm (the run's translation cap): move_to's
        distance to its target, move_delta's norm, a raw OSC step's clipped translation."""
        if method == "env.goto_pose":
            target = np.asarray(kwargs["position"], dtype=np.float64).reshape(3)
            return float(
                np.linalg.norm(target - self._eef())
                + 2 * abs(float(kwargs.get("z_approach") or 0.0))
            )
        if method == "env.home_pose":
            home = self._home[0] if self._home is not None else self._eef()
            return float(np.linalg.norm(home - self._eef()))
        if method in ("env.move_to", "env.move_pose"):
            target = np.asarray(kwargs["xyz"], dtype=np.float64).reshape(3)
            return float(np.linalg.norm(target - self._eef()))
        if method == "env.move_delta":
            d = np.asarray(kwargs["dxyz"], dtype=np.float64).reshape(3)
            return float(np.linalg.norm(d))
        if method == "env.move_grip":
            return self._geometry.planned_distance(kwargs)
        if method in ("env.execute_grasp", "env.execute_place"):
            # The claimed path's legs: to the pre-pose, the standoff down, then the lift up
            # (a grasp) or the standoff back (a place).
            grasp = method == "env.execute_grasp"
            key = "grasp_id" if grasp else "place_id"
            standoff = float(_finite("standoff", kwargs.get("standoff", 0.10)))
            try:
                pre = self._grasp.resolve_grasp(kwargs[key], standoff=standoff)
            except Exception:
                return 0.0  # the call itself refuses the id
            last = (
                float(_finite("lift", kwargs.get("lift", 0.10))) if grasp else standoff
            )
            to_pre = np.asarray(pre["eef_position"], dtype=np.float64) - self._eef()
            return float(np.linalg.norm(to_pre) + standoff + last)
        if method in ("env.step", "env.chunk_step"):
            a = np.asarray(
                kwargs.get("action", kwargs.get("actions")), dtype=np.float64
            )
            a = a.reshape(-1, a.shape[-1])[:, :3]
            return float(np.linalg.norm(np.clip(a, -1, 1) * 0.05, axis=1).sum())
        return 0.0

    def _code_check(self, method: str, kwargs: dict) -> None:
        """Refuse a program's call that the run's wall clock could not bound."""
        if method == "env.chunk_step":
            n = len(np.asarray(kwargs.get("actions"), dtype=np.float64).reshape(-1, 7))
            if n > CODE_MAX_CHUNK:
                raise ValueError(
                    f"chunk_step runs at most {CODE_MAX_CHUNK} actions per call, got {n}"
                )
        if method == "env.render_camera":
            for k in ("height", "width"):
                if int(kwargs.get(k, 1024)) > CODE_MAX_RENDER:
                    raise ValueError(f"render_camera {k} is at most {CODE_MAX_RENDER}")

    def _yaw(self) -> float:
        x, y, z, w = self._quat_xyzw() / np.linalg.norm(self._quat_xyzw())
        return float(np.arctan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z)))

    def _gripper_width(self) -> float:
        q = np.asarray(self.raw_obs()["robot0_gripper_qpos"], dtype=np.float64).reshape(
            -1
        )
        return float(abs(q[0]) + abs(q[1]))

    def _live(self) -> bool:
        return not (self._terminated or self._truncated)

    def _count_run_steps(self, terms: list[bool]) -> None:
        """Every env step counts for the run, the low tier's raw ``step`` / ``chunk_step`` like
        the high tier's primitives; the first success within the run is its step."""
        for term in terms:
            self._run_steps += 1
            if term and self._run_success is None:
                self._run_success = self._run_steps

    def _act(self, action) -> None:
        """One env step of a primitive (pi's tools step the same way)."""
        self.step(np.asarray(action, dtype=np.float32))

    def _frame(self, _primitive=None) -> None:
        """After a mutating primitive (the runner's `after` hook): one agentview frame for the video."""
        if self._last_obs is not None and len(self._run_frames) < CODE_MAX_FRAMES:
            self._run_frames.append(self._last_obs["main_images"])

    def _motion_result(self, name: str, steps: int, cancelled: bool, **fields) -> dict:
        out = {
            "name": name,
            "steps_used": steps,
            "eef_pos": [round(float(v), 4) for v in self._eef()],
            "gripper_width": round(self._gripper_width(), 4),
            "terminated": self._terminated,
            "truncated": self._truncated,
            **fields,
        }
        if cancelled:
            out["cancelled"] = True
        return out

    @staticmethod
    def _servo_action(diff, step_clip: float, action_scale: float) -> list[float]:
        return [
            float(np.clip(np.clip(d, -step_clip, step_clip) / action_scale, -1, 1))
            for d in diff
        ]

    def _servo(
        self,
        target: np.ndarray,
        grip: float,
        tol: float,
        max_steps: int,
        step_clip: float,
        action_scale: float = 0.05,
        target_yaw: float | None = None,
        yaw_step_clip: float = 0.1,
    ):
        """Step toward `target` (world xyz) holding orientation, or turning toward `target_yaw`
        (rad) by at most `yaw_step_clip` per step; stops within `tol` (position only), at the
        step budget, at the episode's end, or at a stop."""
        steps = 0
        cancelled = False
        while steps < max_steps and self._live():
            if self.stop_requested():
                cancelled = True
                break
            diff = target - self._eef()
            if np.linalg.norm(diff) < tol:
                break
            a = [
                *self._servo_action(diff, step_clip, action_scale),
                0.0,
                0.0,
                0.0,
                grip,
            ]
            if target_yaw is not None:
                err = np.clip(
                    _wrap(target_yaw - self._yaw()), -yaw_step_clip, yaw_step_clip
                )
                a[5] = float(np.clip(err / 0.1, -1, 1))
            self._act(a)
            steps += 1
        return steps, cancelled

    def _grip_value(self, gripper) -> float:
        if gripper is None:
            return self._grip
        if isinstance(gripper, bool):
            return 1.0 if gripper else -1.0
        g = float(gripper)
        if g not in (-1.0, 1.0):
            raise ValueError("gripper must be -1 (open), +1 (close) or None (keep)")
        return g

    def get_state(self) -> dict:
        """Proprioception, no images.

        Returns:
            dict with ``eef_pos`` [x, y, z] (m, world frame), ``eef_quat_xyzw``, ``yaw`` (rad,
            about world +z), ``gripper_width`` (sum of the two finger joints: about 0.08 open,
            below 0.01 closed on nothing), ``gripper_cmd`` (-1 open / +1 close, the last command),
            ``terminated`` (LIBERO judged the task done; it stays true) and ``truncated`` (the
            episode's step limit ended it).
        """
        return {
            "eef_pos": [round(float(v), 4) for v in self._eef()],
            "eef_quat_xyzw": [round(float(v), 5) for v in self._quat_xyzw()],
            "yaw": round(self._yaw(), 4),
            "gripper_width": round(self._gripper_width(), 4),
            "gripper_cmd": int(self._grip),
            "terminated": self._terminated,
            "truncated": self._truncated,
        }

    def _camera(self, camera: str) -> str:
        if camera not in CODE_CAMERAS:
            raise ValueError(f"camera must be one of {list(CODE_CAMERAS)}")
        return CODE_CAMERAS[camera]

    def get_observation(self) -> dict:
        """The current camera images with calibration, plus the state of `get_state`.

        Returns:
            dict with ``agentview`` and ``wrist``, each ``{"rgb": uint8[512, 512, 3], "depth":
            float32[512, 512] (metres), "intrinsic_K": float64[3, 3], "extrinsic_cam2world":
            float64[4, 4]}``, and the `get_state` fields. Pixel (row, col) of an image
            back-projects as ``x = (col - cx) * z / fx``, ``y = (row - cy) * z / fy`` in the camera
            frame, then ``extrinsic_cam2world @ [x, y, z, 1]``. The agentview faces the robot
            (its base is at the image top); the wrist camera looks down between the fingers.

        Example:
            >>> obs = get_observation()
            >>> depth = obs["agentview"]["depth"]; K = obs["agentview"]["intrinsic_K"]
        """
        out = {camera: self._view(camera) for camera in CODE_CAMERAS}
        out.update(self.get_state())
        return out

    def _world_xyz(self, view: dict, row: int, col: int) -> np.ndarray | None:
        z = float(view["depth"][row, col])
        if not (z > 0) or not np.isfinite(z):
            return None
        K = view["intrinsic_K"]
        p = np.array(
            [(col - K[0, 2]) * z / K[0, 0], (row - K[1, 2]) * z / K[1, 1], z, 1.0]
        )
        return (view["extrinsic_cam2world"] @ p)[:3]

    def back_project(
        self,
        row: int | None = None,
        col: int | None = None,
        camera: str = "agentview",
        row_range=None,
        col_range=None,
        z_min: float | None = None,
        z_max: float | None = None,
        resolution: str | None = None,
        step: int | None = None,
    ) -> dict:
        """World xyz of pixel (row, col) of the current 512x512 image of `camera` (row 0 = top),
        or region mode: row_range + col_range (+ z_min / z_max) give ``center_xyz`` (midpoint of
        world x and y over the window's pixels with depth, median z) and ``median_xyz``.
        `resolution` and `step` are pi's tool's (its 1024 images and state history)."""
        _current_only(resolution, step)
        self._camera(camera)
        span = lambda r: r if r is not None and max(r) > min(r) else None  # noqa: E731
        rows, cols = span(row_range), span(col_range)
        if rows is not None or cols is not None:
            if rows is None or cols is None:
                raise ValueError("region mode needs both row_range and col_range")
            view = self._view(camera)
            r0, r1 = (int(np.clip(v, 0, CODE_RES)) for v in (min(rows), max(rows)))
            c0, c1 = (int(np.clip(v, 0, CODE_RES)) for v in (min(cols), max(cols)))
            pts = [
                p
                for p in (
                    self._world_xyz(view, r, c)
                    for r in range(r0, r1)
                    for c in range(c0, c1)
                )
                if p is not None
            ]
            if z_min is not None:
                pts = [p for p in pts if p[2] >= float(z_min)]
            if z_max is not None:
                pts = [p for p in pts if p[2] <= float(z_max)]
            if len(pts) < 8:
                raise ValueError(
                    f"too few valid pixels in region ({len(pts)}); widen the window or the z band"
                )
            arr = np.asarray(pts)
            lo, hi = arr.min(axis=0), arr.max(axis=0)
            return {
                "camera": camera,
                "mode": "region",
                "center_xyz": [
                    round(float((lo[0] + hi[0]) / 2), 4),
                    round(float((lo[1] + hi[1]) / 2), 4),
                    round(float(np.median(arr[:, 2])), 4),
                ],
                "median_xyz": [round(float(v), 4) for v in np.median(arr, axis=0)],
                "n_valid": len(pts),
            }
        if row is None or col is None:
            raise ValueError("give row and col, or row_range and col_range")
        row, col = int(row), int(col)
        if not (0 <= row < CODE_RES and 0 <= col < CODE_RES):
            raise ValueError(
                f"pixel ({row}, {col}) out of bounds for {CODE_RES}x{CODE_RES}"
            )
        p = self._world_xyz(self._view(camera), row, col)
        if p is None:
            raise ValueError(f"no depth at pixel ({row}, {col}); pick another pixel")
        return {
            "camera": camera,
            "pixel": [row, col],
            "world_xyz": [round(float(v), 4) for v in p],
        }

    def _sam3_mask(self, view: dict, prompt, point, min_score: float) -> dict:
        """SAM3's top mask of an upright view by a text prompt or a point [row, col]:
        ``{found, score, box, mask}`` or ``{found: False, reason}``."""
        import base64

        from PIL import Image

        from pi_embodied_services.utils.rpc.http_rpc import HttpRpcClient

        if not self._sam3_url:
            raise RuntimeError(
                "segment needs a SAM3 server (start the env server with --sam3)"
            )
        text = str(prompt or "").strip()
        if not text and point is None:
            raise ValueError("give a text prompt or a point [row, col]")
        if self._sam3 is None:
            self._sam3 = HttpRpcClient(self._sam3_url)
        buf = io.BytesIO()
        Image.fromarray(view["rgb"]).save(buf, format="PNG")
        query = {"text_prompt": text} if text else {"point": [int(v) for v in point]}
        res = self._sam3.call(
            "sam3.segment",
            kwargs={
                "image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                **query,
                "min_score": float(min_score),
            },
            timeout_s=120,
        )
        if not res.get("found") or not res.get("mask_png_base64"):
            return {"found": False, "reason": res.get("reason", "no mask")}
        mask = np.asarray(
            Image.open(io.BytesIO(base64.b64decode(res["mask_png_base64"])))
        )
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != (CODE_RES, CODE_RES):
            raise RuntimeError(
                f"SAM3 mask {mask.shape} does not match the {CODE_RES} image"
            )
        return {
            "found": True,
            "score": None
            if res.get("score") is None
            else round(float(res["score"]), 3),
            "box": res.get("box"),
            "mask": mask >= 128,
        }

    def segment(
        self,
        prompt: str | None = None,
        point=None,
        camera: str = "agentview",
        min_score: float = 0.2,
        step: int | None = None,
    ) -> dict:
        """SAM3 segmentation of the current 512x512 image of `camera` by a text prompt or a
        positive point [row, col], and the top mask's median world position through depth.
        `step` is pi's tool's (its state history)."""
        _current_only(None, step)
        self._camera(camera)
        view = self._view(camera)
        seg = self._sam3_mask(view, prompt, point, min_score)
        if not seg["found"]:
            return seg
        mask = seg["mask"]
        rows, cols = np.nonzero(mask)
        pts = [
            p
            for p in (self._world_xyz(view, int(r), int(c)) for r, c in zip(rows, cols))
            if p is not None
        ]
        out: dict[str, Any] = {
            **seg,
            "camera": camera,
            "n_pixels": int(mask.sum()),
            "centroid_rowcol": [int(np.median(rows)), int(np.median(cols))],
            "world_xyz": None,
        }
        if len(pts) >= 10:
            out["world_xyz"] = [
                round(float(v), 4) for v in np.median(np.asarray(pts), axis=0)
            ]
        return out

    def _require_planned(self, plan: dict, action: str) -> dict:
        try:
            return motion.require_planned(plan, action)
        except ValueError as exc:
            raise Refused(str(exc)) from exc

    def move_to(
        self,
        xyz,
        gripper=-1,
        tol: float = 0.012,
        step_clip: float = 0.025,
        max_steps: int = 80,
        action_scale: float = 0.05,
        target_yaw: float | None = None,
        yaw_step_clip: float = 0.1,
    ) -> dict:
        """Servo the end effector to a world position holding its orientation (or turning to
        `target_yaw`). Refused unmoved beyond 0.30 m in xy, and with --ik when unreachable or
        when no collision-free path exists; with --ik the arm follows the planned path and is
        checked against the scene before each segment (``stopped: contact``)."""
        target = _finite("xyz", xyz).reshape(3)
        tol = float(_finite("tol", tol))
        step_clip = float(_finite("step_clip", step_clip))
        action_scale = float(_finite("action_scale", action_scale))
        yaw_step_clip = float(_finite("yaw_step_clip", yaw_step_clip))
        if target_yaw is not None:
            target_yaw = float(_finite("target_yaw", target_yaw))
        self._check_xy(target, "move_to")
        grip = self._grip_value(gripper)
        self._require_reachable(target, "move_to")
        waypoints, planned = [target], None
        if self._motion is not None:
            plan = self._require_planned(
                self.plan_motion(target.tolist(), target_yaw=target_yaw), "move_to"
            )
            if plan["status"] == "planned":
                planned = plan
                waypoints = [
                    np.asarray(w[:3], dtype=np.float64) for w in plan["waypoints"]
                ]
        self._grip = grip
        max_steps = int(max_steps)
        steps, cancelled, stopped = 0, False, None
        for i, wp in enumerate(waypoints):
            if planned is not None:
                verdict = self.check_motion(segment=i)
                if verdict["status"] == "contact":
                    stopped = verdict
                    break
            last = i == len(waypoints) - 1
            n, cancelled = self._servo(
                wp,
                grip,
                tol if last else max(tol, 0.02),
                max_steps - steps,
                step_clip,
                action_scale,
                target_yaw,
                yaw_step_clip,
            )
            steps += n
            if cancelled or steps >= max_steps or not self._live():
                break
        extra: dict[str, Any] = {}
        if planned is not None:
            extra["planned"] = {
                "segments": len(waypoints),
                "path_m": planned["path_m"],
                "backend": planned["backend"],
            }
        if stopped is not None:
            extra["stopped"] = "contact"
            extra["contact"] = stopped["message"]
        return self._motion_result(
            "move_to",
            steps,
            cancelled,
            final_dist_m=round(float(np.linalg.norm(target - self._eef())), 4),
            **extra,
        )

    def _pitch(self) -> float:
        return pitch_of(quat_xyzw_matrix(self._quat_xyzw()))

    def move_pose(
        self,
        xyz,
        target_pitch: float | None = None,
        target_yaw: float | None = None,
        gripper=-1,
        step_clip: float = 0.02,
        pitch_step: float = 0.08,
        yaw_step: float = 0.08,
        tol: float = 0.012,
        ori_tol: float = 0.05,
        action_scale: float = 0.05,
        max_steps: int = 150,
    ) -> dict:
        """Servo xyz and the pitch / yaw targets together each step; stops within `tol` and
        `ori_tol`, at the budget, the episode's end or a stop."""
        target = _finite("xyz", xyz).reshape(3)
        pitch = (
            None
            if target_pitch is None
            else float(_finite("target_pitch", target_pitch))
        )
        yaw = None if target_yaw is None else float(_finite("target_yaw", target_yaw))
        step_clip, pitch_step, yaw_step, tol, ori_tol, action_scale = (
            float(_finite(k, v))
            for k, v in (
                ("step_clip", step_clip),
                ("pitch_step", pitch_step),
                ("yaw_step", yaw_step),
                ("tol", tol),
                ("ori_tol", ori_tol),
                ("action_scale", action_scale),
            )
        )
        grip = self._grip_value(gripper)
        self._grip = grip
        steps, cancelled = 0, False
        while steps < int(max_steps) and self._live():
            if self.stop_requested():
                cancelled = True
                break
            R = quat_xyzw_matrix(self._quat_xyzw())
            diff = target - self._eef()
            p_err = 0.0 if pitch is None else _wrap(pitch - pitch_of(R))
            y_err = 0.0 if yaw is None else _wrap(yaw - yaw_of(R))
            if (
                np.linalg.norm(diff) < tol
                and abs(p_err) < ori_tol
                and abs(y_err) < ori_tol
            ):
                break
            self._act(
                [
                    *self._servo_action(diff, step_clip, action_scale),
                    float(
                        np.clip(np.clip(p_err, -pitch_step, pitch_step) / 0.1, -1, 1)
                    ),
                    0.0,
                    float(np.clip(np.clip(y_err, -yaw_step, yaw_step) / 0.1, -1, 1)),
                    grip,
                ]
            )
            steps += 1
        return self._motion_result(
            "move_pose",
            steps,
            cancelled,
            final_dist_m=round(float(np.linalg.norm(target - self._eef())), 4),
            final_pitch=round(self._pitch(), 4),
        )

    def move_delta(self, dxyz, gripper=None, max_steps: int = 25) -> dict:
        """Move the end effector by a world-frame offset (at most 0.10 m), holding its
        orientation; ``moved_m`` well below the command means the move was blocked."""
        d = _finite("dxyz", dxyz).reshape(3)
        if np.linalg.norm(d) > 0.10 + 1e-9:
            raise ValueError(
                "move_delta moves at most 0.10 m per call; split the motion"
            )
        start = self._eef()
        grip = self._grip_value(gripper)
        self._require_reachable(start + d, "move_delta")
        self._grip = grip
        steps, cancelled = self._servo(start + d, grip, 0.004, int(max_steps), 0.025)
        return self._motion_result(
            "move_delta",
            steps,
            cancelled,
            moved_m=round(float(np.linalg.norm(self._eef() - start)), 4),
        )

    def _rotate(
        self,
        kind: str,
        target: float | None,
        delta: float | None,
        gripper,
        max_steps: int,
        tol: float,
        step_clip: float,
    ) -> dict:
        """Turn about world z (``yaw``, action[5]) or tilt (``pitch``, action[3]) to `target`
        or by `delta` (rad), holding the position; pi's rotate_wrist / rotate_pitch."""
        angle = self._yaw if kind == "yaw" else self._pitch
        if target is None and delta is None:
            raise ValueError(f"give target_{kind} or delta_{kind}")
        start = angle()
        goal = (
            float(_finite(f"target_{kind}", target))
            if target is not None
            else start + float(_finite(f"delta_{kind}", delta))
        )
        tol, step_clip = (
            float(_finite("tol", tol)),
            float(_finite("step_clip", step_clip)),
        )
        grip = self._grip_value(gripper)
        self._grip = grip
        steps, cancelled = 0, False
        while steps < int(max_steps) and self._live():
            if self.stop_requested():
                cancelled = True
                break
            err = _wrap(goal - angle())
            if abs(err) < tol:
                break
            a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, grip]
            a[5 if kind == "yaw" else 3] = float(
                np.clip(np.clip(err, -step_clip, step_clip) / 0.1, -1, 1)
            )
            self._act(a)
            steps += 1
        final = angle()
        return self._motion_result(
            "rotate_wrist" if kind == "yaw" else "rotate_pitch",
            steps,
            cancelled,
            **{
                f"start_{kind}": round(start, 4),
                f"target_{kind}": round(goal, 4),
                f"final_{kind}": round(final, 4),
                "final_err": round(_wrap(goal - final), 4),
            },
            **({"yaw": round(final, 4)} if kind == "yaw" else {}),
        )

    def rotate_wrist(
        self,
        target_yaw: float | None = None,
        delta_yaw: float | None = None,
        gripper=1,
        max_steps: int = 40,
        tol: float = 0.02,
        step_clip: float = 0.1,
    ) -> dict:
        """Turn the gripper about world z to `target_yaw` or by `delta_yaw` (rad, positive =
        counter-clockwise seen from above), holding its position."""
        return self._rotate(
            "yaw", target_yaw, delta_yaw, gripper, max_steps, tol, step_clip
        )

    def rotate_pitch(
        self,
        target_pitch: float | None = None,
        delta_pitch: float | None = None,
        gripper=1,
        max_steps: int = 40,
        tol: float = 0.02,
        step_clip: float = 0.1,
    ) -> dict:
        """Tilt the gripper about world x (pitch 0 = pointing down, +pi/2 = pointing +y) to
        `target_pitch` or by `delta_pitch` (rad), holding its position and yaw."""
        return self._rotate(
            "pitch", target_pitch, delta_pitch, gripper, max_steps, tol, step_clip
        )

    def rotate_delta(self, delta_yaw: float, gripper=None, max_steps: int = 40) -> dict:
        """Turn the gripper about world z by `delta_yaw` radians (at most pi/2 per call),
        holding its position; None keeps the gripper command."""
        if abs(float(_finite("delta_yaw", delta_yaw))) > np.pi / 2 + 1e-9:
            raise ValueError("rotate_delta turns at most pi/2 per call")
        return self._rotate("yaw", None, delta_yaw, gripper, max_steps, 0.02, 0.1)

    def _drive(self, grip: float, steps: int) -> tuple[int, bool]:
        """Hold the pose and command the gripper for `steps` env steps (fewer at the episode's
        end or a stop)."""
        self._grip = grip
        n, cancelled = 0, False
        while n < int(steps) and self._live():
            if self.stop_requested():
                cancelled = True
                break
            self._act([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, grip])
            n += 1
        return n, cancelled

    def set_gripper(self, gripper=-1, steps: int = 5) -> dict:
        """Hold the pose and drive the gripper (-1 open, +1 close) for `steps` env steps; the
        command holds for the following moves that keep it."""
        grip = self._grip_value(gripper)
        n, cancelled = self._drive(grip, steps)
        return self._motion_result("set_gripper", n, cancelled, gripper=int(grip))

    def release(self, max_steps: int = 20) -> dict:
        """Open the gripper in place for `max_steps` env steps (placing may end the task)."""
        start = self._gripper_width()
        n, cancelled = self._drive(-1.0, max_steps)
        return self._motion_result(
            "release",
            n,
            cancelled,
            start_gripper_opening=round(start, 4),
            final_gripper_opening=round(self._gripper_width(), 4),
        )

    def _actuate(self, grip: float, max_steps: int = 15) -> dict:
        """Drive the gripper until the fingers stop (on an object, or fully open / closed): at
        most `max_steps` steps, done once a step after the third moved them less than 0.5 mm
        (the planned executors' and --geometry's closing)."""
        self._grip = grip
        n, cancelled, prev = 0, False, float("nan")
        while n < max_steps and self._live():
            if self.stop_requested():
                cancelled = True
                break
            self._act([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, grip])
            n += 1
            width = self._gripper_width()
            if n > 3 and abs(width - prev) < 5e-4:
                break
            prev = width
        return self._motion_result("set_gripper", n, cancelled, gripper=int(grip))

    def _check_xy(self, target, what: str) -> None:
        """Refuse (unmoved) a move longer than ``MAX_XY_MOVE_M`` in xy."""
        xy = float(
            np.linalg.norm((np.asarray(target, dtype=np.float64) - self._eef())[:2])
        )
        if xy > MAX_XY_MOVE_M + 1e-9:
            raise Refused(
                f"{what} would move {xy:.3f} m in xy, more than {MAX_XY_MOVE_M} m: split it "
                "into waypoints at carry height"
            )

    # ---- planned grasps (--contact-graspnet/--graspgenx/--anygrasp/--graspnet1b/--anyplace) ----

    def _servo_pose(
        self,
        target: np.ndarray,
        quat_xyzw,
        grip: float,
        max_steps: int,
        tol: float = 0.012,
        ori_tol: float = 0.05,
    ):
        """Position and the full orientation (roll, pitch and yaw, any tilt direction) toward
        their targets each step: the OSC's rotation delta is the world-frame rotation vector
        to the target (``utils/grasp.orientation_error``), 0.1 rad per unit as pi's tools."""
        R_target = quat_xyzw_matrix(quat_xyzw)
        steps = 0
        cancelled = False
        while steps < max_steps and self._live():
            if self.stop_requested():
                cancelled = True
                break
            diff = target - self._eef()
            err = orientation_error(quat_xyzw_matrix(self._quat_xyzw()), R_target)
            if np.linalg.norm(diff) < tol and np.linalg.norm(err) < ori_tol:
                break
            rot = err * min(1.0, 0.08 / max(float(np.linalg.norm(err)), 1e-9))
            a = [
                *self._servo_action(diff, 0.02, 0.05),
                *[float(np.clip(v / 0.1, -1, 1)) for v in rot],
                grip,
            ]
            self._act(a)
            steps += 1
        return steps, cancelled

    def _execute_claim(self, claim: dict, max_steps: int, name: str) -> dict:
        """Run a claimed path (``GraspPlanner.claim_waypoints``) step by step; stops at the
        first leg that stalls (more than 3 cm short), a stop or the episode's end."""
        waypoints = {
            k: np.asarray(v, dtype=np.float64) for k, v in claim["waypoints"].items()
        }
        quat = claim["eef_quat_xyzw"]
        legs: list[dict] = []
        total = 0
        for leg in claim["steps"]:
            grip = float(leg["gripper"])
            self._grip = grip
            if "to" not in leg:
                out = self._actuate(grip)
                steps, cancelled = out["steps_used"], bool(out.get("cancelled"))
                legs.append(
                    {"gripper": int(grip), "gripper_width": out["gripper_width"]}
                )
                if grip > 0:
                    # Closed: the grasp's height above its support is measured here.
                    self._grasp.note_grasp_closed(claim.get("arm"))
                if grip < 0:
                    # The hand opened: the held grasp ends once the fingers hold nothing.
                    self._grasp.release_held(
                        claim.get("arm"), opened=not self._holding()
                    )
            else:
                target = waypoints[leg["to"]]
                steps, cancelled = self._servo_pose(target, quat, grip, int(max_steps))
                dist = float(np.linalg.norm(target - self._eef()))
                legs.append({"to": leg["to"], "final_dist_m": round(dist, 4)})
                # Short of the waypoint because the episode ended is not a stall.
                if dist > 0.03 and not cancelled and self._live():
                    total += steps
                    return self._motion_result(
                        name,
                        total,
                        False,
                        id=claim["id"],
                        legs=legs,
                        stalled=leg["to"],
                        error=f"stalled {dist:.3f} m short of {leg['to']}: unreachable or "
                        "blocked; plan again from the new observation",
                    )
            total += steps
            if cancelled or not self._live():
                return self._motion_result(
                    name, total, cancelled, id=claim["id"], legs=legs
                )
        return self._motion_result(name, total, False, id=claim["id"], legs=legs)

    def execute_grasp(
        self,
        grasp_id: str,
        standoff: float = 0.10,
        lift: float = 0.10,
        max_steps: int = 150,
    ) -> dict:
        """Execute one planned grasp from a single resolution of its id: open to the pre-grasp
        ``standoff`` back along its approach, descend to the grasp (pitch and yaw as planned),
        close, lift ``lift`` straight up.

        Args:
            grasp_id: a ``g`` id of the current observation (``plan_grasp``).
            standoff: pre-grasp distance, m (default 0.10).
            lift: lift after closing, m (default 0.10).
            max_steps: env-step budget per leg (default 150).

        Returns:
            dict with ``legs`` (each leg's final distance, the gripper width after closing),
            ``gripper_width`` (0.01-0.05 holding, near 0 missed), ``eef_pos``, ``terminated``;
            ``error`` and ``stalled`` when a leg stopped short. The id is spent either way;
            ``plan_place(region, grasp_id)`` afterwards plans from the held object.

        Example:
            >>> g = plan_grasp("black bowl"); r = execute_grasp(g["active"])
        """
        if self._grasp.resolve_grasp(str(grasp_id))["kind"] != "grasp":
            raise ValueError(f"{grasp_id} is a place id; use execute_place")
        pre = self._grasp.resolve_grasp(
            str(grasp_id), standoff=float(_finite("standoff", standoff))
        )
        self._check_xy(pre["eef_position"], "execute_grasp's pre-grasp leg")
        claim = self._grasp.claim_waypoints(
            str(grasp_id),
            float(_finite("standoff", standoff)),
            float(_finite("lift", lift)),
        )
        return self._execute_claim(claim, max_steps, "execute_grasp")

    def execute_place(
        self, place_id: str, standoff: float = 0.10, max_steps: int = 150
    ) -> dict:
        """Execute one planned place from a single resolution of its id: carry (closed) to the
        pre-place ``standoff`` back along its approach, descend to the place pose, open,
        retreat to the pre-place.

        Args:
            place_id: a ``p`` id of the current observation (``plan_place``).
            standoff: pre-place distance, m (default 0.10).
            max_steps: env-step budget per leg (default 150).

        Returns:
            dict with ``legs``, ``gripper_width``, ``eef_pos``, ``terminated`` (placing may
            finish the task); ``error`` and ``stalled`` when a leg stopped short.

        Example:
            >>> p = plan_place(segment_mask("plate")["id"], g["active"])
            >>> execute_place(p["active"])
        """
        if self._grasp.resolve_grasp(str(place_id))["kind"] != "placement":
            raise ValueError(f"{place_id} is a grasp id; use execute_grasp")
        pre = self._grasp.resolve_grasp(
            str(place_id), standoff=float(_finite("standoff", standoff))
        )
        self._check_xy(pre["eef_position"], "execute_place's pre-place leg")
        claim = self._grasp.claim_waypoints(
            str(place_id), float(_finite("standoff", standoff)), 0.0
        )
        return self._execute_claim(claim, max_steps, "execute_place")

    # ---- CaP-X's high tier (FrankaLiberoApi) and its privileged variant ------------------------
    #
    # Quaternions are CaP-X's: wxyz of panda_hand; the grip site LIBERO reports (and move_to /
    # rotate_wrist servo) is panda_hand turned half a turn about its z (utils/object_pose.py).
    # Positions are the grip site's (the TCP point; no offset). The motions go through the
    # registered env.move_to / env.rotate_wrist (their limits, stop handling and the planners' id
    # expiry), holding the gripper command.

    def _object_points(self, object_name: str, use_multiview: bool) -> np.ndarray:
        """CaP-X's get_object_3d_points_and_masks_from_language: the name's SAM3 mask in the
        agentview (and the wrist view) as world points, merged by CaP-X's rule, then filtered.
        No Molmo here: SAM3 by text in each view."""
        found = {}
        for camera in ("agentview", "wrist") if use_multiview else ("agentview",):
            view = self._view(camera)
            seg = self._sam3_mask(view, object_name, None, 0.2)
            if not seg["found"]:
                raise ValueError(
                    f"SAM3 segmentation failed for '{object_name}' on {camera}."
                )
            found[camera] = (
                object_pose.mask_points(view, seg["mask"]),
                float(seg.get("score") or 0.0),
            )
        return object_pose.filter_noise(object_pose.merge_views(found))

    def get_object_pose(self, object_name: str, use_multiview: bool = True) -> list:
        """CaP-X's get_object_pose: [position (3,), quaternion_wxyz (4,)] of an object named in
        words, or [None, None] when too few points survive the filter."""
        pos, q = object_pose.pose_of_points(
            self._object_points(str(object_name), bool(use_multiview))
        )
        if pos is None:
            return [None, None]
        return [[round(float(v), 4) for v in pos], [float(v) for v in q]]

    def sample_grasp_pose(self, object_name: str, use_multiview: bool = True) -> list:
        """CaP-X's sample_grasp_pose: [position (3,), quaternion_wxyz (4,)]. With a grasp server
        the active candidate of plan_grasp (agentview; its EEF position and yaw, gripper down);
        without one a top-down grasp over the filtered points' centre, GRASP_DEPTH_M below their
        top, closing across the shorter horizontal side."""
        if self._grasp is not None:
            plan = self._grasp.plan_grasp(object=str(object_name))
            best = next(c for c in plan["candidates"] if c["id"] == plan["active"])
            return [
                [float(v) for v in best["eef_position"]],
                [float(v) for v in object_pose.hand_of_yaw(float(best["eef_yaw"]))],
            ]
        pts = self._object_points(str(object_name), bool(use_multiview))
        if len(pts) < 3:
            raise ValueError(f"No valid points after filtering for '{object_name}'")
        pos, q = object_pose.topdown_grasp(pts)
        return [[round(float(v), 4) for v in pos], [float(v) for v in q]]

    def _ground_truth_object(self, object_name: str) -> dict:
        """CaP-X's simulator lookup (_get_object_pose): "<name>_1" exactly, else the one object
        whose name without its _<n> suffix contains the query or is contained in it."""
        poses = self.ground_truth_poses()["poses"]
        query = str(object_name).replace(" ", "_").lower()
        if f"{query}_1" in poses:
            return poses[f"{query}_1"]
        base: dict[str, list[str]] = {}
        for k in poses:
            stem, _, n = k.rpartition("_")
            base.setdefault(stem if n.isdigit() and stem else k, []).append(k)
        matches = [b for b in sorted(base) if query in b or b in query]
        if len(matches) == 1:
            return poses[sorted(base[matches[0]])[0]]
        raise KeyError(
            f"Object '{object_name}' not found. Available objects: {sorted(base)}"
        )

    def get_object_pose_privileged(self, object_name: str) -> list:
        """CaP-X's privileged get_object_pose: [position (3,), quaternion_wxyz (4,)] from the
        simulator."""
        p = self._ground_truth_object(object_name)
        x, y, z, w = p["quat_xyzw"]
        return [[float(v) for v in p["pos"]], [float(w), float(x), float(y), float(z)]]

    def sample_grasp_pose_privileged(self, object_name: str) -> list:
        """CaP-X's privileged sample_grasp_pose: the object's position, gripper down."""
        pos, _ = self.get_object_pose_privileged(object_name)
        return [pos, [0.0, 1.0, 0.0, 0.0]]

    def _turn_to(self, yaw: float, nearest: bool) -> dict | None:
        """rotate_wrist to `yaw` (or, `nearest`, the half turn away when that is closer: the
        fingers are symmetric), keeping the gripper command; None when already within 0.05."""
        if nearest and abs(_wrap(yaw - self._yaw())) > math.pi / 2:
            yaw = _wrap(yaw + math.pi)
        if abs(_wrap(yaw - self._yaw())) <= 0.05:
            return None
        return self._rpc["env.rotate_wrist"](target_yaw=yaw, gripper=None)

    def _legs_to(self, target: np.ndarray) -> tuple[int, bool]:
        """move_to `target` in straight legs of at most MAX_LEG_M, keeping the gripper command."""
        steps = 0
        start = self._eef()
        legs = max(1, math.ceil(np.linalg.norm(target - start) / MAX_LEG_M))
        for k in range(1, legs + 1):
            here = self._eef()
            left = max(
                legs - k + 1, math.ceil(np.linalg.norm(target - here) / MAX_LEG_M)
            )
            out = self._rpc["env.move_to"](
                (here + (target - here) / left).tolist(),
                gripper=None,
                max_steps=LEG_STEPS,
            )
            steps += int(out.get("steps_used", 0))
            if out.get("cancelled") or not self._live():
                return steps, bool(out.get("cancelled"))
        return steps, False

    def _go(self, name: str, stops: list, yaw: float, nearest: bool) -> dict:
        steps, cancelled = 0, False
        for stop in stops:
            turned = self._turn_to(yaw, nearest)
            if turned is not None:
                steps += int(turned.get("steps_used", 0))
                if turned.get("cancelled") or not self._live():
                    cancelled = bool(turned.get("cancelled"))
                    break
            n, cancelled = self._legs_to(stop)
            steps += n
            if cancelled or not self._live():
                break
        return self._motion_result(
            name,
            steps,
            cancelled,
            final_dist_m=round(float(np.linalg.norm(stops[-1] - self._eef())), 4),
            yaw=round(self._yaw(), 4),
        )

    def goto_pose(self, position, quaternion_wxyz, z_approach: float = 0.0) -> dict:
        """CaP-X's goto_pose: first `z_approach` metres back along the gripper's approach axis,
        then the pose. The grip site pointing down (LIBERO's reset orientation) turns to the
        quaternion's yaw, or the half turn away, whichever is nearer, then servos in straight
        legs; the quaternion's tilt is not servoed (``tilt_rad`` reports it)."""
        pos = _finite("position", position).reshape(3)
        q = object_pose.unit(_finite("quaternion_wxyz", quaternion_wxyz))
        z_approach = float(_finite("z_approach", z_approach))
        stops = []
        if z_approach != 0.0:
            stops.append(
                pos + object_pose.matrix(q) @ np.array([0.0, 0.0, -z_approach])
            )
        stops.append(pos)
        out = self._go("goto_pose", stops, object_pose.site_yaw(q), nearest=True)
        # The approach axis's angle from straight down (0: every orientation is reached).
        approach = object_pose.matrix(
            object_pose.mul(q, object_pose.HAND_TO_SITE_WXYZ)
        )[:, 2]
        out["tilt_rad"] = round(float(np.arccos(np.clip(-approach[2], -1, 1))), 4)
        return out

    def home_pose(self) -> dict:
        """CaP-X's home pose: the gripper's position and yaw at the episode's reset."""
        pos, quat = (
            self._home if self._home is not None else (self._eef(), self._quat_xyzw())
        )
        return self._go(
            "home_pose",
            [np.asarray(pos, dtype=np.float64)],
            yaw_of(quat_xyzw_matrix(quat)),
            nearest=False,
        )

    def open_gripper(self) -> dict:
        """CaP-X's open_gripper: 40 env steps open."""
        n, cancelled = self._drive(-1.0, 40)
        return self._motion_result("open_gripper", n, cancelled)

    def close_gripper(self) -> dict:
        """CaP-X's close_gripper: 60 env steps closed (the fingers stop on an object)."""
        n, cancelled = self._drive(1.0, 60)
        return self._motion_result("close_gripper", n, cancelled)

    # ---- grip-site geometry (--geometry, utils/geometry.py) ----

    def _grip_mech(self) -> dict:
        """The worker's grip site, finger pads and robot contacts (``mujoco_grip_state``)."""
        worker = self._env.env.workers[self._env_idx]
        out = worker.env_call("grip_geometry", target="self")
        if isinstance(out.get("error"), str):
            raise RuntimeError(f"grip geometry failed in the worker: {out['error']}")
        return out

    def _grip_frame(self) -> np.ndarray:
        """The fixed rotation from ``robot0_eef_quat``'s frame to the grip frame (jaw +X,
        approach +Z): LIBERO's eef quaternion is the hand body's, not the grip site's (OpenETA
        ``sim/unified_env.py``), so it is measured once against the site and its pads."""
        mech = self._grip_mech()
        hand = quat_to_matrix(self._quat_xyzw())
        site = np.asarray(mech["site_xmat"], dtype=np.float64)
        self._grip_mech_frame = jaw_frame(mech["pads_local"])
        return hand.T @ site @ self._grip_mech_frame

    def _grip_pads(self) -> np.ndarray:
        """The finger pads' inner faces in the grip frame."""
        pads = np.asarray(self._grip_mech()["pads_local"], dtype=np.float64)
        if self._grip_mech_frame is None:
            self._grip_mech_frame = jaw_frame(pads)
        return pads @ self._grip_mech_frame

    def _workspace_envelope(self) -> np.ndarray | None:
        """The box the point-cloud views are cut from: in front of and around the robot base
        (its reach), so walls and the floor far away do not set the views' scale."""
        try:
            worker = self._env.env.workers[self._env_idx]
            base = worker.env_call("robot_base_pose", target="self")
            pos = np.asarray(base["pos"], dtype=np.float64).reshape(3)
            R = quat_to_matrix(base["quat_xyzw"])
        except Exception:  # noqa: BLE001  the default envelope around the grip site
            return None
        lo, hi = np.array([-0.25, -0.8, -0.3]), np.array([1.1, 0.8, 1.1])
        corners = np.array(
            [
                [a, b, c]
                for a in (lo[0], hi[0])
                for b in (lo[1], hi[1])
                for c in (lo[2], hi[2])
            ]
        )
        world = corners @ R.T + pos
        return np.stack([world.min(axis=0), world.max(axis=0)], axis=1)

    def _move_grip(
        self, position, tool_R, tol_m: float, tol_rad: float, max_steps: int
    ) -> dict:
        """Servo the grip site to ``position`` and the eef frame to ``tool_R`` together (OSC
        deltas: world-frame translation and rotation vector), holding the gripper command."""
        target = np.asarray(position, dtype=np.float64).reshape(3)
        self._check_xy(target, "move_grip")
        goal = np.asarray(tool_R, dtype=np.float64)
        steps = 0
        cancelled = False
        while steps < max_steps and self._live():
            if self.stop_requested():
                cancelled = True
                break
            diff = target - self._eef()
            err = rotvec_of(goal @ quat_to_matrix(self._quat_xyzw()).T)
            angle = float(np.linalg.norm(err))
            if np.linalg.norm(diff) < tol_m and angle < tol_rad:
                break
            rot = err * (min(angle, 0.08) / angle) if angle > 1e-9 else np.zeros(3)
            self._act(
                [
                    *self._servo_action(diff, 0.025, 0.05),
                    *[float(np.clip(r / 0.1, -1, 1)) for r in rot],
                    self._grip,
                ]
            )
            steps += 1
        return {"steps_used": steps, **({"cancelled": True} if cancelled else {})}

    # ---- reach preview (--ik, utils/reach.py) ----

    def _require_reachable(self, target: np.ndarray, action: str) -> None:
        """With ``--ik``, refuse (ValueError, the robot unmoved) a target the ik service cannot
        reach from the current joints, as the Robosuite and Franka servers' motions do."""
        if self._reach is not None:
            try:
                reach.require_reachable(
                    self.preview_reach([float(v) for v in target]), f"env.{action}"
                )
            except ValueError as exc:
                raise Refused(str(exc)) from exc

    def preview_reach(self, xyz, quat_xyzw=None) -> dict:
        """Whether the gripper can reach a world position without moving: IK from the current
        joints, solved by the ik service (``--ik``); the sim is not touched. ``quat_xyzw`` None
        keeps the current orientation (as ``move_to`` does). ``status`` is ``reachable``,
        ``unreachable`` or ``unknown`` (no ik service answered; not approval)."""
        if self._reach is None:
            return reach.no_service()
        raw = to_numpy_tree(self._env.current_raw_obs[self._env_idx])
        return self._reach.preview(
            raw["robot0_joint_pos"],
            xyz,
            raw["robot0_eef_quat"] if quat_xyzw is None else quat_xyzw,
            base_pose=self._robot_base(),
        )

    def _robot_base(self) -> dict:
        if self._base_pose is None:
            worker = self._env.env.workers[self._env_idx]
            base = worker.env_call("robot_base_pose", target="self")
            if "error" in base:
                raise RuntimeError(f"robot base pose unavailable: {base['error']}")
            self._base_pose = base
        return self._base_pose

    def _scene(self) -> list:
        """The planning world (utils/motion.py): the scene's collidable geoms, world frame."""
        worker = self._env.env.workers[self._env_idx]
        out = worker.env_call("collision_world", target="self")
        if "error" in out:
            raise RuntimeError(f"collision world unavailable: {out['error']}")
        return out["obstacles"]

    # ---- collision-free motion (--ik, utils/motion.py) ----

    def plan_motion(self, pos, quat_xyzw=None, target_yaw=None) -> dict:
        """Plan a collision-free path of the gripper to a world position from the current
        joints (the ik service's ``ik.plan`` through the scene); nothing moves. The
        orientation is kept, or ``quat_xyzw``, or the current one turned to ``target_yaw``
        (rad, about world z). ``status`` is ``planned`` (``waypoints``: world TCP poses to
        servo through, the goal last), ``blocked`` (no collision-free path: move_to refuses)
        or ``unknown`` (no ik service answered). The plan is kept for ``check_motion``."""
        if self._motion is None:
            raise RuntimeError("plan_motion needs --ik")
        raw = self.raw_obs()
        quat = np.asarray(
            raw["robot0_eef_quat"] if quat_xyzw is None else quat_xyzw, dtype=np.float64
        )
        if target_yaw is not None:
            from scipy.spatial.transform import Rotation

            turn = Rotation.from_euler("z", float(target_yaw) - self._yaw())
            quat = (turn * Rotation.from_quat(quat)).as_quat()
        plan = self._motion.plan(
            raw["robot0_joint_pos"],
            raw["robot0_eef_pos"],
            np.asarray(pos, dtype=np.float64).reshape(3),
            quat,
            self._scene(),
            base_pose=self._robot_base(),
        )
        self._plan = plan
        return {
            k: plan[k]
            for k in (
                "status",
                "message",
                "waypoints",
                "path_m",
                "backend",
                "obstacles",
            )
        } | {
            "left_out": len(plan["left_out"]),
            "excluded_by_base": plan.get("excluded_by_base", []),
        }

    def check_motion(self, segment=None) -> dict:
        """Check the arm against the scene before a servo segment: the current joints, and
        with ``segment`` the last plan's configuration at that waypoint. ``status`` is
        ``clear``, ``contact`` (stop: predicted contact) or ``unknown``."""
        if self._motion is None:
            raise RuntimeError("check_motion needs --ik")
        qs = [self.raw_obs()["robot0_joint_pos"]]
        plan = self._plan or {}
        if segment is not None:
            q_path = plan.get("q_path") or []
            if not 0 <= int(segment) < len(q_path):
                raise ValueError(
                    f"segment {segment} is not a waypoint of the last plan_motion"
                )
            qs.append(q_path[int(segment)])
        return self._motion.check(
            qs,
            self._scene(),
            base_pose=self._robot_base(),
            left_out=plan.get("left_out") or (),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--suite", type=str, default="libero_spatial")
    p.add_argument("--task", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-episode-steps", type=int, default=10000)
    reach.add_ik_argument(p)
    motion.add_unplanned_argument(p)
    p.add_argument(
        "--sam3",
        type=str,
        default=None,
        help="SAM3 server URL: lets plan_grasp / segment_mask and code mode's `segment` segment objects by text",
    )
    add_grasp_arguments(p)
    p.add_argument(
        "--geometry",
        action="store_true",
        help="serve the geometric toolset: point-cloud views, marked points, grip-site targets",
    )
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device to pin MuJoCo EGL rendering and the torch "
        "default device to (physical CUDA ordinal).",
    )
    add_perception_arguments(p)
    args = p.parse_args()

    if args.cuda_device is not None:
        # Deliberately do NOT set CUDA_VISIBLE_DEVICES. robosuite (imported
        # transitively via libero) asserts at import time that
        # ``MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES`` (substring check),
        # which assumes the EGL index equals the CUDA ordinal and crashes on
        # multi-GPU boxes where the EGL order differs. That assertion is gated
        # on ``CUDA_VISIBLE_DEVICES != ""``, so leaving it unset skips it in
        # both this process and the multiprocessing-spawned render workers
        # (which inherit the env). Pin the two backends directly instead:
        #   - MuJoCo render device <- MUJOCO_EGL_DEVICE_ID (configure_egl_device)
        #   - torch default device  <- torch.cuda.set_device(N)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is set; clearing it and pinning via "
                "MUJOCO_EGL_DEVICE_ID + torch.cuda.set_device(--cuda-device=%s) "
                "instead (robosuite's CVD assertion is incompatible with EGL<->CUDA mapping)",
                prev,
                args.cuda_device,
            )
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        from pi_embodied_services.utils.egl import configure_egl_device

        configure_egl_device(args.cuda_device)
        import torch

        torch.cuda.set_device(args.cuda_device)

    raw_env = make_env(
        args.task,
        args.seed,
        suite_name=args.suite,
        max_episode_steps=args.max_episode_steps,
    )
    facade = LiberoEnvFacade(
        raw_env,
        meta={
            "suite": args.suite,
            "task": args.task,
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
        },
        ik_reach=reach.reach_from_args(args, "panda_libero"),
        ik_motion=motion.planner_from_args(args, "panda_libero"),
        sam3=args.sam3,
        grasp=urls_from_args(args),
        geometry=args.geometry,
    )
    # --sam3 / --unidepth: env.detect, env.select_detection, env.reject_detection, env.enhance_depth,
    # on the views the grasp planner and code mode read (its own env.segment stays).
    install_perception(
        facade,
        args,
        cameras=["agentview", "wrist"],
        view=facade._view,
        grasp=facade._grasp,
        mutating=LIBERO_MOTIONS,
    )
    try:
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        facade.close()


if __name__ == "__main__":
    main()
