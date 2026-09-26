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
import io
import os
import random
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from pi_embodied_services.components.code_api import register_code_api
from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.robots.libero.primitives import libero_primitives
from pi_embodied_services.utils import ground_truth, reach
from pi_embodied_services.utils.code_exec import (
    CodeRunner,
    describe_helpers,
    registry_primitives,
)
from pi_embodied_services.utils.grasp import (
    GraspPlanner,
    add_grasp_arguments,
    pitch_of,
    quat_xyzw_matrix,
    urls_from_args,
)
from pi_embodied_services.utils.logging import get_logger
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
#: A program's raw ``chunk_step`` runs at most this many actions in one call (a chunk is one
#: worker call that no stop interrupts; the run's wall clock must bound it).
CODE_MAX_CHUNK = 64
#: A program's ``render_camera`` renders at most this many pixels per side.
CODE_MAX_RENDER = 1024


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

        env.ground_truth_poses = poses
        env.robot_base_pose = robot_base_pose
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


class LiberoEnvFacade(BaseEnvFacade):
    """Implements :class:`pi_embodied_services.robots.libero.env_client.LiberoEnvClient`
    over :class:`rlinf.envs.libero.libero_env.LiberoEnv`.

    All return values are converted to CPU numpy so the agent process
    (which does not import torch) can consume them after the pickle round
    trip.
    """

    SERVICE_NAME = "libero-env"
    #: Code mode: a program could otherwise call this server's port itself (outside its
    #: budgets, and after its run); pi reads the token from the listening line.
    REQUIRE_TOKEN = True
    #: Set by __init__; class defaults so the registry can be built on a bare facade (tests).
    _sam3_url: str | None = None
    _grasp: "GraspPlanner | None" = None

    def __init__(
        self,
        env: LiberoEnv,
        *,
        meta: dict,
        sam3: str | None = None,
        grasp: dict | None = None,
        ik_reach: reach.ReachPreview | None = None,
    ):
        self._env = env
        self._env_idx = 0
        self._closed = False
        # --ik: env.preview_reach (utils/reach.py); the robot base pose it converts world
        # targets with is read from the worker once per reset.
        self._reach = ik_reach
        self._base_pose: dict | None = None
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
        # --graspnet/--graspgenx/--anygrasp/--anyplace: env.plan_grasp, env.plan_place and the
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
            }
        )
        # Code mode's primitives (primitives.py CODE_PRIMITIVES), registered before the grasp
        # planner wraps the mutating ones with its id invalidation.
        self._rpc.update(
            {
                "env.get_state": self.get_state,
                "env.get_observation": self.get_observation,
                "env.back_project": self.back_project,
                "env.move_to": self.move_to,
                "env.move_delta": self.move_delta,
                "env.rotate_wrist": self.rotate_wrist,
                "env.rotate_delta": self.rotate_delta,
                "env.set_gripper": self.set_gripper,
                **({"env.segment": self.segment} if self._sam3_url else {}),
            }
        )
        self._readonly_methods.add("env.get_task_language")
        primitives = libero_primitives(
            sam3=bool(self._sam3_url), grasp=self._grasp is not None
        )
        if self._grasp is not None:
            # Wrapped with the id invalidation like every motion (GraspPlanner.MUTATING).
            self._rpc["env.execute_grasp"] = self.execute_grasp
            self._rpc["env.execute_place"] = self.execute_place
            self._grasp.install(self)
            primitives = (*primitives, *self._grasp.primitives())
        api = register_code_api(self, primitives)
        # Code mode (run_code): a program's calls go through the registry's resolve to the
        # methods above; the runner adds the sandbox, the budgets and the stop handling.
        self._code = CodeRunner(
            registry_primitives(
                api,
                self._rpc,
                move_m=self._code_move_m,
                after=self._frame,
                check=self._code_check,
            ),
            stop_requested=self.stop_requested,
            # A timed-out program is killed; the robot gets a stop like an abort would send.
            on_timeout=lambda: self.request_stop(),
            begin=self._begin_run,
            finish=self._finish_run,
        )
        self._rpc["code.run"] = self._code.run
        self._rpc["code.helpers"] = describe_helpers
        self._readonly_methods.add("code.helpers")

    def _on_stop(self, generation: int) -> None:
        # A stop while a program runs kills its process; the primitive loops see stop_requested.
        self._code.abort()

    def _exclusive_call_active(self) -> bool:
        # While a program runs, only its own primitives (called in-process) touch the env.
        return self._code.active

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
    # The facade methods behind the registry's CODE_PRIMITIVES (primitives.py): programs reach them
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
        if method == "env.move_to":
            target = np.asarray(kwargs["xyz"], dtype=np.float64).reshape(3)
            return float(np.linalg.norm(target - self._eef()))
        if method == "env.move_delta":
            d = np.asarray(kwargs["dxyz"], dtype=np.float64).reshape(3)
            return float(np.linalg.norm(d))
        if method in ("env.execute_grasp", "env.execute_place"):
            # To the pre-pose, down the standoff and back up (or the lift), at most.
            key = "grasp_id" if method == "env.execute_grasp" else "place_id"
            try:
                pose = self._grasp.resolve_grasp(kwargs[key])
            except Exception:
                return 0.0  # the call itself refuses the id
            at = np.asarray(pose["eef_position"], dtype=np.float64)
            standoff = float(_finite("standoff", kwargs.get("standoff", 0.10)))
            return float(
                np.linalg.norm(at - self._eef())
                + 2 * standoff
                + float(kwargs.get("lift", 0.10))
            )
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
    ):
        """Step toward `target` (world xyz) holding orientation; stops within `tol`, at the
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
            self._act([*self._servo_action(diff, step_clip, 0.05), 0.0, 0.0, 0.0, grip])
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

    def back_project(self, row: int, col: int, camera: str = "agentview") -> dict:
        """World xyz of pixel (row, col) of the current 512x512 image of `camera` (row 0 = top).

        Args:
            row, col: pixel in the image `get_observation` returns for that camera.
            camera: "agentview" (default) or "wrist".

        Returns:
            dict with ``world_xyz`` [x, y, z] in metres. Raises when the pixel has no depth.

        Example:
            >>> p = back_project(300, 260)["world_xyz"]
        """
        row, col = int(row), int(col)
        if not (0 <= row < CODE_RES and 0 <= col < CODE_RES):
            raise ValueError(
                f"pixel ({row}, {col}) out of bounds for {CODE_RES}x{CODE_RES}"
            )
        self._camera(camera)
        p = self._world_xyz(self._view(camera), row, col)
        if p is None:
            raise ValueError(f"no depth at pixel ({row}, {col}); pick another pixel")
        return {
            "camera": camera,
            "pixel": [row, col],
            "world_xyz": [round(float(v), 4) for v in p],
        }

    def segment(
        self, prompt: str, camera: str = "agentview", min_score: float = 0.2
    ) -> dict:
        """SAM3 segmentation of the current 512x512 image of `camera` by a text prompt, and the
        top mask's median world position through the depth image.

        Args:
            prompt: what to segment, e.g. "black bowl".
            camera: "agentview" (default) or "wrist".
            min_score: SAM3 score threshold (default 0.2).

        Returns:
            dict with ``found`` (bool); when found: ``score``, ``box`` [x1, y1, x2, y2] (pixels),
            ``mask`` bool[512, 512], ``n_pixels``, ``centroid_rowcol`` [row, col] and
            ``world_xyz`` (median over the mask's pixels with depth, or None when too few).

        Example:
            >>> seg = segment("black bowl")
            >>> if seg["found"]: xyz = seg["world_xyz"]
        """
        import base64

        from PIL import Image

        from pi_embodied_services.utils.rpc.http_rpc import HttpRpcClient

        if not self._sam3_url:
            raise RuntimeError(
                "segment needs a SAM3 server (start the env server with --sam3)"
            )
        if self._sam3 is None:
            self._sam3 = HttpRpcClient(self._sam3_url)
        self._camera(camera)
        view = self._view(camera)
        buf = io.BytesIO()
        Image.fromarray(view["rgb"]).save(buf, format="PNG")
        res = self._sam3.call(
            "sam3.segment",
            kwargs={
                "image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                "text_prompt": str(prompt),
                "min_score": float(min_score),
            },
            timeout_s=120,
        )
        if not res.get("found") or not res.get("mask_png_base64"):
            return {"found": False, "reason": res.get("reason", "no mask")}
        mask_img = Image.open(io.BytesIO(base64.b64decode(res["mask_png_base64"])))
        mask = np.asarray(mask_img)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != (CODE_RES, CODE_RES):
            raise RuntimeError(
                f"SAM3 mask {mask.shape} does not match the {CODE_RES} image"
            )
        mask = mask >= 128
        rows, cols = np.nonzero(mask)
        pts = [
            p
            for p in (self._world_xyz(view, int(r), int(c)) for r, c in zip(rows, cols))
            if p is not None
        ]
        out: dict[str, Any] = {
            "found": True,
            "camera": camera,
            "score": None
            if res.get("score") is None
            else round(float(res["score"]), 3),
            "box": res.get("box"),
            "mask": mask,
            "n_pixels": int(mask.sum()),
            "centroid_rowcol": [int(np.median(rows)), int(np.median(cols))],
            "world_xyz": None,
        }
        if len(pts) >= 10:
            arr = np.asarray(pts)
            out["world_xyz"] = [round(float(v), 4) for v in np.median(arr, axis=0)]
        return out

    def move_to(
        self, xyz, gripper=None, tol: float = 0.012, max_steps: int = 80
    ) -> dict:
        """Servo the end effector to a world position, holding its orientation.

        Args:
            xyz: target [x, y, z] in metres (world frame). Keep one call under 0.30 m in xy;
                split longer moves into waypoints at carry height.
            gripper: -1 opens, +1 closes and holds (carry with +1); None (default) keeps the
                last command.
            tol: stop within this distance (default 0.012 m).
            max_steps: env-step budget (default 80; about 2.5 cm per step).

        Returns:
            dict with ``eef_pos``, ``final_dist_m``, ``steps_used``, ``gripper_width``,
            ``terminated``. A large ``final_dist_m`` means the reach stalled (contact, limits).

        Example:
            >>> move_to([0.05, 0.12, 0.25])          # above the target
            >>> move_to([0.05, 0.12, 0.06]); set_gripper(True); move_to([0.05, 0.12, 0.25])
        """
        target = _finite("xyz", xyz).reshape(3)
        tol = float(_finite("tol", tol))
        grip = self._grip_value(gripper)
        self._grip = grip
        steps, cancelled = self._servo(target, grip, tol, int(max_steps), 0.025)
        return self._motion_result(
            "move_to",
            steps,
            cancelled,
            final_dist_m=round(float(np.linalg.norm(target - self._eef())), 4),
        )

    def move_delta(self, dxyz, gripper=None, max_steps: int = 25) -> dict:
        """Move the end effector by a world-frame offset, holding its orientation.

        Args:
            dxyz: [dx, dy, dz] in metres (world frame: +x away from the robot toward the
                agentview camera, +y robot-left, +z up). Keep each call at or below 0.10 m.
            gripper: -1 opens, +1 closes and holds; None (default) keeps the last command.
            max_steps: env-step budget (default 25).

        Returns:
            dict with ``eef_pos``, ``moved_m`` (distance actually travelled), ``steps_used``,
            ``gripper_width``, ``terminated``. ``moved_m`` well below the command means the
            move was blocked (contact, table, workspace limit).

        Example:
            >>> move_delta([0, 0, -0.05])   # descend 5 cm
        """
        d = _finite("dxyz", dxyz).reshape(3)
        if np.linalg.norm(d) > 0.10 + 1e-9:
            raise ValueError(
                "move_delta moves at most 0.10 m per call; split the motion"
            )
        start = self._eef()
        grip = self._grip_value(gripper)
        self._grip = grip
        steps, cancelled = self._servo(start + d, grip, 0.004, int(max_steps), 0.025)
        return self._motion_result(
            "move_delta",
            steps,
            cancelled,
            moved_m=round(float(np.linalg.norm(self._eef() - start)), 4),
        )

    def _rotate(
        self, goal: float, grip: float, max_steps: int, tol: float, step_clip: float
    ):
        steps = 0
        cancelled = False
        wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
        while steps < max_steps and self._live():
            if self.stop_requested():
                cancelled = True
                break
            err = wrap(goal - self._yaw())
            if abs(err) < tol:
                break
            a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, grip]
            a[5] = float(np.clip(np.clip(err, -step_clip, step_clip) / 0.1, -1, 1))
            self._act(a)
            steps += 1
        return steps, cancelled, float(wrap(goal - self._yaw()))

    def rotate_wrist(
        self,
        target_yaw: float | None = None,
        delta_yaw: float | None = None,
        gripper=None,
        max_steps: int = 40,
    ) -> dict:
        """Turn the gripper about world z, holding its position.

        Args:
            target_yaw: absolute yaw in radians (world frame), or
            delta_yaw: relative turn in radians (positive = counter-clockwise seen from above).
            gripper: -1 / +1 / None (keep).
            max_steps: env-step budget (default 40; about 0.1 rad per step).

        Returns:
            dict with ``yaw`` (final), ``final_err`` (rad), ``steps_used``, ``eef_pos``.

        Example:
            >>> rotate_wrist(delta_yaw=1.5708)   # a quarter turn
        """
        if target_yaw is None and delta_yaw is None:
            raise ValueError("give target_yaw or delta_yaw")
        goal = (
            float(_finite("target_yaw", target_yaw))
            if target_yaw is not None
            else self._yaw() + float(_finite("delta_yaw", delta_yaw))
        )
        grip = self._grip_value(gripper)
        self._grip = grip
        steps, cancelled, err = self._rotate(goal, grip, int(max_steps), 0.02, 0.1)
        return self._motion_result(
            "rotate_wrist",
            steps,
            cancelled,
            yaw=round(self._yaw(), 4),
            final_err=round(err, 4),
        )

    def rotate_delta(self, delta_yaw: float, gripper=None, max_steps: int = 40) -> dict:
        """Turn the gripper about world z by `delta_yaw` radians (positive = counter-clockwise
        seen from above), holding its position. At most pi/2 per call.

        Returns:
            dict with ``yaw`` (final, rad), ``final_err``, ``steps_used``, ``eef_pos``.
        """
        if abs(float(_finite("delta_yaw", delta_yaw))) > np.pi / 2 + 1e-9:
            raise ValueError("rotate_delta turns at most pi/2 per call")
        return self.rotate_wrist(
            delta_yaw=float(delta_yaw), gripper=gripper, max_steps=max_steps
        )

    def set_gripper(self, close: bool, steps: int = 15) -> dict:
        """Close or open the gripper in place.

        Args:
            close: True closes (and the following moves hold +1), False opens.
            steps: env steps to drive the fingers (default 15; they stop early once they rest
                on a grasped object).

        Returns:
            dict with ``gripper_width`` (about 0.08 open; 0.01-0.05 holding an object; near 0
            closed on nothing), ``steps_used``, ``terminated`` (opening over the goal may end
            the task).

        Example:
            >>> set_gripper(True); state = get_state()   # then judge gripper_width
        """
        grip = 1.0 if close else -1.0
        self._grip = grip
        n = 0
        cancelled = False
        widths = [self._gripper_width()]
        while n < int(steps) and self._live():
            if self.stop_requested():
                cancelled = True
                break
            self._act([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, grip])
            n += 1
            widths.append(self._gripper_width())
            # LIBERO's fingers travel about 1 mm per step: they have stopped (on an object, or
            # fully open / closed) when three steps moved them less than 1 mm in total.
            if n > 3 and abs(widths[-1] - widths[-4]) < 1e-3:
                break
        return self._motion_result("set_gripper", n, cancelled, close=bool(close))

    # ---- planned grasps (--graspnet/--graspgenx/--anygrasp/--anyplace) ----

    def _pitch(self) -> float:
        return pitch_of(quat_xyzw_matrix(self._quat_xyzw()))

    def _servo_pose(
        self,
        target: np.ndarray,
        pitch: float,
        yaw: float,
        grip: float,
        max_steps: int,
        tol: float = 0.012,
        ori_tol: float = 0.05,
    ):
        """pi's ``move_pose`` rule: position, pitch and yaw toward their targets each step."""
        wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
        steps = 0
        cancelled = False
        while steps < max_steps and self._live():
            if self.stop_requested():
                cancelled = True
                break
            diff = target - self._eef()
            p_err = float(wrap(pitch - self._pitch()))
            y_err = float(wrap(yaw - self._yaw()))
            if np.linalg.norm(diff) < tol and max(abs(p_err), abs(y_err)) < ori_tol:
                break
            a = [*self._servo_action(diff, 0.02, 0.05), 0.0, 0.0, 0.0, grip]
            a[3] = float(np.clip(np.clip(p_err, -0.08, 0.08) / 0.1, -1, 1))
            a[5] = float(np.clip(np.clip(y_err, -0.08, 0.08) / 0.1, -1, 1))
            self._act(a)
            steps += 1
        return steps, cancelled

    def _execute_claim(self, claim: dict, max_steps: int, name: str) -> dict:
        """Run a claimed path (``GraspPlanner.claim_waypoints``) step by step; stops at the
        first leg that stalls (more than 3 cm short), a stop or the episode's end."""
        waypoints = {
            k: np.asarray(v, dtype=np.float64) for k, v in claim["waypoints"].items()
        }
        pitch, yaw = float(claim["eef_pitch"]), float(claim["eef_yaw"])
        legs: list[dict] = []
        total = 0
        for leg in claim["steps"]:
            grip = float(leg["gripper"])
            self._grip = grip
            if "to" not in leg:
                out = self.set_gripper(grip > 0)
                steps, cancelled = out["steps_used"], bool(out.get("cancelled"))
                legs.append(
                    {"gripper": int(grip), "gripper_width": out["gripper_width"]}
                )
            else:
                target = waypoints[leg["to"]]
                steps, cancelled = self._servo_pose(
                    target, pitch, yaw, grip, int(max_steps)
                )
                dist = float(np.linalg.norm(target - self._eef()))
                legs.append({"to": leg["to"], "final_dist_m": round(dist, 4)})
                if dist > 0.03 and not cancelled:
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
        claim = self._grasp.claim_waypoints(
            str(place_id), float(_finite("standoff", standoff)), 0.0
        )
        return self._execute_claim(claim, max_steps, "execute_place")

    # ---- reach preview (--ik, utils/reach.py) ----

    def preview_reach(self, pos, quat_xyzw=None) -> dict:
        """Whether the gripper can reach a world position without moving: IK from the current
        joints, solved by the ik service (``--ik``); the sim is not touched. ``quat_xyzw`` None
        keeps the current orientation (as ``move_to`` does). ``status`` is ``reachable``,
        ``unreachable`` or ``unknown`` (no ik service answered; not approval)."""
        if self._reach is None:
            return reach.no_service()
        raw = to_numpy_tree(self._env.current_raw_obs[self._env_idx])
        if self._base_pose is None:
            worker = self._env.env.workers[self._env_idx]
            base = worker.env_call("robot_base_pose", target="self")
            if "error" in base:
                raise RuntimeError(f"robot base pose unavailable: {base['error']}")
            self._base_pose = base
        return self._reach.preview(
            raw["robot0_joint_pos"],
            pos,
            raw["robot0_eef_quat"] if quat_xyzw is None else quat_xyzw,
            base_pose=self._base_pose,
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
    p.add_argument(
        "--sam3",
        type=str,
        default=None,
        help="SAM3 server URL: lets plan_grasp / segment_mask and code mode's `segment` segment objects by text",
    )
    add_grasp_arguments(p)
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
        sam3=args.sam3,
        grasp=urls_from_args(args),
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
