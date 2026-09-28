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

"""RoboCasa env server — hosts the raw robosuite env in a subprocess, exposes basic calls via RPC.

The motion primitives run here, for pi's tools and a program alike (``env.move_to``,
``env.move_delta``, ``env.rotate_pitch``, ``env.set_gripper``, ``env.release``,
``env.scripted_grasp``, ``env.navigate_to``, ``env.move_base`` and the high tier's
``env.goto_pose`` / ``env.home_pose`` / ``env.open_gripper`` / ``env.close_gripper``): the arm
servo inverts a measured action-to-world Jacobian, the base drives on a measured forward heading,
every step is polled for a stop and recorded (video frames, the Flywheel's records).

Code mode (``code.run``, utils/code_exec.py ``CodeRunMixin``): a program calls the manifest's
primitives (packages/embodied/src/primitives/manifests/robocasa.json) through its resolve, in a sandboxed subprocess, so the server requires
its RPC token and refuses other business calls while a program runs. A program's ``step`` receives
the robot's own observations only (the kitchen's object observations are privileged: they stay
out, as does the reward's info), and every step it takes adds an agentview frame to the run's
video; the run reports its env steps, the success and the new robot observation (``_finish_run``).
"""

import argparse
import base64
import inspect
import io
import math
import os
import re
import sys
from typing import Any

import numpy as np

from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.robots.robocasa import tasks
from pi_embodied_services.utils import ground_truth
from pi_embodied_services.utils.code_exec import CodeRunMixin
from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.perception import (
    add_perception_arguments,
    install_perception,
    render_view,
)
from pi_embodied_services.utils.rpc.main_thread_serve import MainThreadServeMixin

logger = get_logger("env_server")


DEFAULT_CAMS = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]

#: Code mode: the camera of the run's video (the agentview pi's tools show), its frame size, the
#: frames one run hands back (halved, every other one kept, when full) and the largest render a
#: program may ask for.
CODE_VIDEO_CAMERA = "robot0_agentview_left"
CODE_VIDEO_SIZE = 256
CODE_MAX_FRAMES = 128
CODE_MAX_RENDER = 1024
#: The PandaOmron controllers' travel per env step at a full (1.0) action
#: (robosuite default_pandaomron.json, control_freq 20): the arm's OSC_POSE moves at most 0.05 m,
#: the torso's JOINT_POSITION 0.05 m, the base's JOINT_VELOCITY 0.5 m/s for 1/20 s per axis.
ARM_M_PER_STEP = 0.05
TORSO_M_PER_STEP = 0.05
BASE_M_PER_STEP = 0.5 / 20
#: The motion methods (pi's tools and the programs' primitives run the same ones). The arm servo
#: measures the world motion of a unit arm action with three probe moves (PROBE_ACTION for
#: PROBE_STEPS steps per axis) and inverts it; the base's forward heading is measured by driving
#: forward CALIBRATE_BASE_STEPS steps. OSC_ROT_SCALE: rad per unit rotation action.
PROBE_ACTION = 0.4
PROBE_STEPS = 3
CALIBRATE_BASE_STEPS = 6
OSC_ROT_SCALE = 0.5
#: "hold": the finger servo gain that keeps the current width (carry without crushing).
HOLD_GAIN = 60.0
#: Steps the high tier's open_gripper / close_gripper drive the fingers.
GRIPPER_STEPS = 15
#: The cameras the motion methods record: the video's (top-down agentview, VIDEO_SIZE) and RLDX-1's
#: three (the Flywheel's observation while recording).
VIDEO_SIZE = 256
VLA_CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
#: The methods that step the env and answer a motion report (frames, the new robot obs).
MOTION_METHODS = (
    "env.move_to",
    "env.move_delta",
    "env.rotate_pitch",
    "env.set_gripper",
    "env.release",
    "env.scripted_grasp",
    "env.navigate_to",
    "env.move_base",
    "env.goto_pose",
    "env.home_pose",
    "env.open_gripper",
    "env.close_gripper",
)


def robot_obs(obs) -> dict:
    """The robot's own observations of a robosuite obs dict (``robot0_*``, no camera images):
    the kitchen's object observations (``obj_*``, ``<object>_pos`` ...) are privileged."""
    return {
        k: v
        for k, v in obs.items()
        if k.startswith("robot0_") and not k.endswith(("_image", "_depth"))
    }


def _pinv3(J: np.ndarray) -> np.ndarray:
    """Moore-Penrose inverse of a 3x3 matrix, as (JᵀJ + εI)⁻¹Jᵀ with a vanishing ε."""
    A = J.T @ J
    A = A + (1e-12 + 1e-10 * np.trace(A)) * np.eye(3)
    return np.linalg.inv(A) @ J.T


def _yaw(q_xyzw) -> float:
    """Yaw of an xyzw quaternion (scipy's ``as_euler("xyz")[2]``)."""
    x, y, z, w = (float(v) for v in q_xyzw)
    return math.atan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))


def _rot(q_xyzw) -> np.ndarray:
    """The rotation matrix of an xyzw quaternion."""
    x, y, z, w = (float(v) for v in q_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _vec(value, n: int, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be {n} finite numbers")
    return a


def _split_kwargs(split):
    """Replicate robocasa.utils.env_utils.create_env's split -> layout logic."""
    if split == "target":
        return {
            "obj_instance_split": "target",
            "layout_ids": None,
            "style_ids": None,
            "layout_and_style_ids": list(zip(range(1, 11), range(1, 11))),
        }
    if split == "pretrain":
        return {
            "obj_instance_split": "pretrain",
            "layout_ids": -2,
            "style_ids": -2,
            "layout_and_style_ids": None,
        }
    if split == "all":
        return {
            "obj_instance_split": None,
            "layout_ids": -3,
            "style_ids": -3,
            "layout_and_style_ids": None,
        }
    if split is None:
        return {
            "obj_instance_split": None,
            "layout_ids": None,
            "style_ids": None,
            "layout_and_style_ids": None,
        }
    raise ValueError('split must be {None,"all","pretrain","target"}')


class _Stopped(Exception):
    """A stop arrived during a motion (internal)."""


class RoboCasaEnvFacade(CodeRunMixin, MainThreadServeMixin, BaseEnvFacade):
    """Wraps the raw robosuite env and exposes ONLY basic calls via RPC.

    Mixes in :class:`MainThreadServeMixin` so every env op runs on a single
    thread (the MuJoCo EGL context must stay on one thread); the inherited
    ``serve`` handles this.
    """

    SERVICE_NAME = "robocasa-env"

    def __init__(
        self,
        task_name,
        split="target",
        seed=0,
        scene=None,
        camera_h=256,
        camera_w=256,
        cameras=None,
        use_camera_obs=False,
    ):
        """``scene`` (a manifest index, 0-49) picks the task's scene seed from the
        RoboCasa365 table and reseeds every reset with it; without one ``seed`` seeds
        the env once, as robosuite does."""
        super().__init__()
        self.table = tasks.load_table()
        self.cameras = list(cameras) if cameras else list(DEFAULT_CAMS)
        self.camera_h, self.camera_w = camera_h, camera_w
        self.use_camera_obs = use_camera_obs
        self.env = None
        # Code mode: the env steps (all of them, and before the run), the run's last robot
        # observation and its video frames.
        self._steps = 0
        self._run_start = 0
        self._run_obs: dict | None = None
        self._run_frames: list[np.ndarray] = []
        # The motion methods: the latest observation, the arm servo's calibration (world dpos per
        # unit arm action; dropped when the base turns), the base's measured forward heading
        # offset, the arm's reset pose in the base frame (home_pose), the frames and Flywheel
        # records of the motion in progress, and whether the Flywheel records (env.set_recording).
        self._obs: dict | None = None
        self._pos_jac: np.ndarray | None = None
        self._fwd_offset: float | None = None
        self._home_rel: np.ndarray | None = None
        self._motion_frames: list[np.ndarray] = []
        self._policy_frames: list[dict] = []
        self._recording = False
        self._motion_steps = 0
        # SAM3 (--sam3, with --detections): get_object_pose.
        self._sam3_url = ""
        self._sam3 = None
        self._make(task_name, split, seed, scene)

    def _make(self, task_name, split, seed, scene):
        """Build the robosuite env of ``task_name`` in ``split`` (closing the current one)."""
        import robocasa  # noqa: F401 — registers robocasa envs
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        if scene is not None:
            seed = tasks.scene_seed(self.table, task_name, split, scene)
        else:
            tasks.find_task(self.table, task_name)
        if self.env is not None:
            self.close()
        self.task_name, self.split, self.seed, self.scene = (
            task_name,
            split,
            seed,
            scene,
        )
        controller_config = load_composite_controller_config(
            controller=None, robot="PandaOmron"
        )
        env_kwargs = dict(
            env_name=task_name,
            robots="PandaOmron",
            controller_configs=controller_config,
            camera_names=self.cameras,
            camera_widths=self.camera_w,
            camera_heights=self.camera_h,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=self.use_camera_obs,  # off -> no per-step render (EGL-safe OSC loops)
            camera_depths=False,  # depth rendered on demand
            seed=seed,
            **_split_kwargs(split),
        )
        self.env = robosuite.make(**env_kwargs)
        self._meta = {
            "task_name": self.task_name,
            "split": self.split,
            "seed": self.seed,
            "scene": self.scene,
            "env_id": tasks.env_id(self.task_name, self.split)
            if self.split in self.table["splits"]
            else None,
            "camera_h": self.camera_h,
            "camera_w": self.camera_w,
        }

    def _register_rpc(self):
        """Register all RPC methods."""
        super()._register_rpc()
        self._rpc["env.list_tasks"] = self.list_tasks
        self._readonly_methods.add("env.list_tasks")
        self._rpc["env.check_success"] = self.check_success
        self._rpc["env.get_camera_transform"] = self.get_camera_transform
        self._rpc["env.grasp_contact"] = self.grasp_contact
        self._rpc["env.reassemble_env_action"] = self.reassemble_env_action
        self._rpc["env.get_success_criteria_text"] = self.get_success_criteria_text
        self._rpc["env.get_task_progress"] = self.get_task_progress
        self._rpc["env.ground_truth_poses"] = self.ground_truth_poses
        self._rpc["env.get_state"] = self.get_state
        self._rpc["env.set_recording"] = self.set_recording
        self._rpc["env.get_object_pose"] = self.get_object_pose
        for method in MOTION_METHODS:
            self._rpc[method] = getattr(self, method.removeprefix("env."))
        # Read-only methods
        self._readonly_methods.update(
            [
                "env.check_success",
                "env.get_camera_transform",
                "env.grasp_contact",
                "env.get_success_criteria_text",
                "env.get_task_progress",
                "env.get_state",
            ]
        )
        # The primitives are packages/embodied/src/primitives/manifests/robocasa.json (with pi);
        # code.api, the programs' whitelist and the startup self-check come from it.
        self._manifest_code_run(
            "robocasa",
            have=self._has,
            move_m=self._code_move_m,
            check=self._code_check,
            reply=self._code_reply,
            begin=self._begin_run,
            finish=self._finish_run,
        )

    def _has(self, capability: str) -> bool:
        """What this server can serve of the manifest's ``requires``."""
        return {
            "sam3": bool(self._sam3_url),
            "unidepth": "env.enhance_depth" in self._rpc,
            "privileged": True,
        }.get(capability, False)

    # ---- code mode (run_code) ----

    def _begin_run(self) -> None:
        self._run_start = self._steps
        self._run_obs = None
        self._run_frames = []

    def _finish_run(self) -> dict:
        """The run's effect for pi: env steps taken, the success, the robot's observation after
        its last step (the ``robot0_*`` arrays pi's ``obs`` reads; None when it took none) and
        the run's video frames (top-down agentview)."""
        return {
            "steps": self._steps - self._run_start,
            "success": self.check_success(),
            "obs": self._run_obs,
            "frames": list(self._run_frames),
        }

    def _keep_frame(self, frame) -> None:
        if len(self._run_frames) >= CODE_MAX_FRAMES:
            self._run_frames = self._run_frames[::2]
        self._run_frames.append(frame)

    def _code_reply(self, method: str, out):
        """What a program receives of a ``step``: the robot's observations, the reward and done
        (no object observations, no info); the step's agentview goes to the run's video. Of a
        motion: its report, its frames going to the run's video."""
        if method in MOTION_METHODS:
            out = dict(out)
            for f in out.pop("frames", []):
                self._keep_frame(f)
            out.pop("policy_frames", None)
            self._run_obs = out.pop("obs", None) or self._run_obs
            out.pop("env_steps", None)
            return out
        if method != "env.step":
            return out
        obs, reward, done, _info = out
        self._run_obs = robot_obs(obs)
        rgb = self.render_camera(
            CODE_VIDEO_CAMERA, CODE_VIDEO_SIZE, CODE_VIDEO_SIZE, False
        )
        # robosuite renders bottom-up; the video is top-down like pi's own frames.
        self._keep_frame(np.ascontiguousarray(np.asarray(rgb)[::-1]))
        return {"obs": self._run_obs, "reward": float(reward), "done": bool(done)}

    def _code_move_m(self, method: str, kwargs: dict) -> float:
        """How far one ``step`` may move the gripper (the run's translation cap): the arm's OSC
        travel, the torso's and the base's (drive and turn), each at the action's clipped size."""
        if method in MOTION_METHODS:
            return self._motion_move_m(method, kwargs)
        if method != "env.step":
            return 0.0
        a = np.clip(
            np.asarray(kwargs["flat_action"], dtype=np.float64).reshape(-1), -1, 1
        )
        if a.shape[0] < 11:
            return float(np.linalg.norm(a[:3])) * ARM_M_PER_STEP
        return float(
            np.linalg.norm(a[:3]) * ARM_M_PER_STEP
            + np.linalg.norm(a[7:9]) * BASE_M_PER_STEP
            # the base's yaw (0.5 rad/s) swings the gripper, within a metre of it, as far
            + abs(a[9]) * BASE_M_PER_STEP
            + abs(a[10]) * TORSO_M_PER_STEP
        )

    def _code_check(self, method: str, kwargs: dict) -> None:
        """Refuse a program's call that the run's wall clock could not bound."""
        if method in (
            "env.render_camera",
            "env.get_camera_meta",
            "env.get_camera_transform",
        ):
            for k in ("height", "width"):
                v = kwargs.get(k)
                if v is not None and int(v) > CODE_MAX_RENDER:
                    raise ValueError(f"{k} is at most {CODE_MAX_RENDER} in code mode")

    def get_env_meta(self):
        return self._meta

    def list_tasks(self, split):
        """The RoboCasa365 table's tasks of ``split`` (pretrain | target)."""
        return tasks.list_tasks(self.table, split)

    # ---- lifecycle ----
    def reset(self, task=None, split=None, scene=None):
        """Reset the env; ``task`` / ``split`` switch to another env (rebuilt when they
        differ from the current one), ``scene`` to another manifest scene of it."""
        if (task, split, scene) != (None, None, None):
            task = task if task is not None else self.task_name
            split = split if split is not None else self.split
            scene = scene if scene is not None else self.scene
            if (task, split) != (self.task_name, self.split):
                self._make(task, split, self.seed, scene)
            elif scene != self.scene:
                self.seed = tasks.scene_seed(self.table, task, split, scene)
                self.scene = scene
                self._meta.update(seed=self.seed, scene=self.scene)
        # RLDX_RESET_SEED=<episode_seed> -> reproduce the EXACT scene the fullshot eval
        # generated for that episode, seeded the SAME way as the eval's VideoRecordingWrapper
        # (random.seed + np.random.seed + robosuite env.rng/seed) BEFORE reset. Lets the
        # hybrid run on the IDENTICAL reset layouts fullshot was scored on (true paired
        # comparison). The eval formula: episode_seed = (run_seed + env_idx)*100000 + episode_id.
        # A manifest scene reseeds the same way with its scene seed, so every reset of it
        # samples the same kitchen and objects.
        rs_env = os.environ.get("RLDX_RESET_SEED")
        if rs_env or self.scene is not None:
            import random

            sd = int(rs_env) if rs_env else self.seed
            random.seed(sd)
            np.random.seed(sd)
            if hasattr(self.env, "seed"):
                self.env.seed = sd
            if hasattr(self.env, "rng"):
                self.env.rng = np.random.default_rng(sd)
        obs = self.env.reset()
        # A fresh scene: the servo and base calibrations start over; home is the arm's reset pose.
        self._obs = obs
        self._pos_jac = self._fwd_offset = None
        rel = obs.get("robot0_base_to_eef_pos") if isinstance(obs, dict) else None
        self._home_rel = None if rel is None else np.asarray(rel, dtype=np.float64)
        return obs

    def step(self, flat_action):
        """flat_action: np.ndarray[12] = [eef_pos(3), eef_rot(3), gripper(1),
        base_motion(4), control_mode(1)] in the PandaOmron composite layout."""
        a = np.asarray(flat_action, dtype=np.float64).reshape(-1)
        assert a.shape[0] == self.env.action_dim, (
            f"action dim {a.shape[0]} != env.action_dim {self.env.action_dim}"
        )
        obs, reward, done, info = self.env.step(a)
        self._steps += 1
        self._obs = obs
        if a.shape[0] >= 10 and a[9] != 0:
            # The base turned: the arm action's world directions changed.
            self._pos_jac = None
        return obs, reward, done, info

    # ---- motion (pi's tools and the programs' primitives run these same methods) ----
    #
    # The PandaOmron 12-D action is [eef_pos 3, eef_rot 3, gripper, base 3, torso, base_mode].
    # Every method polls the stop between env steps, records the top-down agentview after every
    # step (``frames``, the episode video) and, while the Flywheel records, what RLDX-1 reads
    # (``policy_frames``); its reply carries the robot's new observation (``obs``, robot0_* only)
    # and the env steps it took (``env_steps``).

    def _vec_obs(self, key: str) -> np.ndarray:
        if self._obs is None:
            raise RuntimeError("no observation yet: reset the env first")
        return np.asarray(self._obs[key], dtype=np.float64).reshape(-1)

    def _eef(self) -> np.ndarray:
        return self._vec_obs("robot0_eef_pos")

    def _finger(self) -> float:
        return float(self._vec_obs("robot0_gripper_qpos")[0])

    def _base_pos(self) -> np.ndarray:
        return self._vec_obs("robot0_base_pos")

    def _base_yaw(self) -> float:
        return _yaw(self._vec_obs("robot0_base_quat"))

    @staticmethod
    def _zero(base_mode: float = -1.0) -> np.ndarray:
        a = np.zeros(12)
        a[11] = base_mode
        return a

    def _grip(self, g: Any, q: float) -> float:
        """a[6] for a motion step: "close" +1, "open" -1, a number passes (clipped); "hold"
        (or None) servos the fingers back to width ``q``."""
        if g == "close":
            return 1.0
        if g == "open":
            return -1.0
        if isinstance(g, (int, float)) and not isinstance(g, bool):
            return float(np.clip(g, -1, 1))
        if g not in (None, "hold"):
            raise ValueError(
                f"gripper must be 'close', 'open', 'hold' or a number, got {g!r}"
            )
        return float(np.clip(HOLD_GAIN * (self._finger() - q), -1, 1))

    def _video_frame(self) -> np.ndarray:
        rgb = self.render_camera(CODE_VIDEO_CAMERA, VIDEO_SIZE, VIDEO_SIZE, False)
        return np.ascontiguousarray(np.asarray(rgb)[::-1])

    def _policy_frame(self) -> dict:
        """What RLDX-1 reads now (the Flywheel's observation): its three cameras, top-down, and
        its state keys."""
        return {
            "state": {
                "state.gripper_qpos": self._vec_obs("robot0_gripper_qpos"),
                "state.base_position": self._vec_obs("robot0_base_pos"),
                "state.base_rotation": self._vec_obs("robot0_base_quat"),
                "state.end_effector_position_relative": self._vec_obs(
                    "robot0_base_to_eef_pos"
                ),
                "state.end_effector_rotation_relative": self._vec_obs(
                    "robot0_base_to_eef_quat"
                ),
            },
            "video": {
                f"video.{c}": np.ascontiguousarray(
                    np.asarray(self.render_camera(c, VIDEO_SIZE, VIDEO_SIZE, False))[
                        ::-1
                    ]
                )
                for c in VLA_CAMERAS
            },
        }

    def _act(self, a: np.ndarray) -> bool:
        """One env step of a motion; False (nothing stepped) once a stop arrived."""
        if self.stop_requested():
            return False
        self.step(a)
        self._motion_steps += 1
        self._motion_frames.append(self._video_frame())
        if self._recording:
            self._policy_frames.append(
                {
                    "action": np.asarray(a, dtype=np.float64),
                    "success": self.check_success(),
                    **self._policy_frame(),
                }
            )
        return True

    def _motion(self, run) -> dict:
        """Run a motion body and answer its report with the frames, the Flywheel records, the
        new robot observation and the env steps taken."""
        self._motion_frames, self._policy_frames, self._motion_steps = [], [], 0
        try:
            report = run()
        except _Stopped:
            report = {"ok": False, "cancelled": True}
        out = {
            **report,
            "frames": self._motion_frames,
            "obs": robot_obs(self._obs) if self._obs is not None else None,
            "env_steps": self._motion_steps,
        }
        if self._recording:
            out["policy_frames"] = self._policy_frames
        self._motion_frames, self._policy_frames = [], []
        return out

    def _step_or_stop(self, a: np.ndarray) -> None:
        if not self._act(a):
            raise _Stopped

    def set_recording(self, on: bool = True) -> None:
        """Whether the motion methods return the Flywheel's per-step records (pi's
        --collect-flywheel-data)."""
        self._recording = bool(on)

    def _calibrate_arm(self, g: float) -> np.ndarray:
        """Probe the three arm action axes and measure the world motion: dpos ≈ J @ action."""
        cols = []
        for axis in range(3):
            p0 = self._eef()
            a = self._zero()
            a[axis] = PROBE_ACTION
            a[6] = g
            for _ in range(PROBE_STEPS):
                self._step_or_stop(a)
            cols.append((self._eef() - p0) / (PROBE_ACTION * PROBE_STEPS))
        self._pos_jac = np.stack(cols, axis=1)
        return self._pos_jac

    def _servo(
        self,
        target,
        gripper: Any = "hold",
        step_clip: float = 0.02,
        max_steps: int = 200,
        tol: float = 0.012,
    ) -> dict:
        """Closed-loop OSC servo of the eef to a world ``target``, orientation held."""
        target = _vec(target, 3, "xyz")
        q = self._finger()
        jinv = _pinv3(
            self._pos_jac
            if self._pos_jac is not None
            else self._calibrate_arm(self._grip(gripper, q))
        )

        def done(ok: bool, steps: int) -> dict:
            return {
                "ok": ok,
                "steps": steps,
                "final_dist": round(float(np.linalg.norm(target - self._eef())), 4),
                "eef": [round(float(v), 4) for v in self._eef()],
                "gripper_qpos": round(self._finger(), 4),
            }

        for i in range(int(max_steps)):
            err = target - self._eef()
            dist = float(np.linalg.norm(err))
            if dist < tol:
                return done(True, i)
            d = err if dist <= step_clip else err / dist * step_clip
            a = self._zero()
            a[:3] = np.clip(jinv @ d, -1, 1)
            a[6] = self._grip(gripper, q)
            self._step_or_stop(a)
        return done(False, int(max_steps))

    def _gripper_steps(self, g: float, steps: int) -> dict:
        a = self._zero()
        a[6] = float(np.clip(g, -1, 1))
        for _ in range(int(steps)):
            self._step_or_stop(a)
        return {
            "ok": True,
            "gripper_qpos": [
                round(float(v), 4) for v in self._vec_obs("robot0_gripper_qpos")
            ],
        }

    def move_to(
        self,
        xyz,
        gripper: Any = "hold",
        step_clip: float = 0.02,
        max_steps: int = 200,
        tol: float = 0.012,
    ) -> dict:
        """Servo the eef to a world xyz with the OSC controller, orientation held; ``gripper``
        "hold" keeps the finger width, "close" / "open" drive it."""
        return self._motion(
            lambda: self._servo(xyz, gripper, step_clip, max_steps, tol)
        )

    def move_delta(
        self,
        dxyz,
        gripper: Any = "hold",
        step_clip: float = 0.02,
        max_steps: int = 80,
    ) -> dict:
        """Servo the eef by a world-frame displacement (``move_to`` of current + dxyz)."""
        d = _vec(dxyz, 3, "dxyz")
        return self._motion(
            lambda: self._servo(self._eef() + d, gripper, step_clip, max_steps)
        )

    def rotate_pitch(
        self, target_pitch: float = 0.6, gripper: float = 1.0, n: int = 12
    ) -> dict:
        """Tilt the wrist about the control x axis by ``target_pitch`` rad (clamped to ±1.5)
        over ``n`` env steps, holding ``gripper``."""
        if int(n) < 1:
            raise ValueError("n must be at least 1")

        def run() -> dict:
            per = float(np.clip(target_pitch, -1.5, 1.5)) / int(n)
            a = self._zero()
            a[3] = float(np.clip(per / OSC_ROT_SCALE, -1, 1))
            a[6] = float(np.clip(gripper, -1, 1))
            for _ in range(int(n)):
                self._step_or_stop(a)
            return {"ok": True, "eef": [round(float(v), 4) for v in self._eef()]}

        return self._motion(run)

    def set_gripper(self, gripper: float = 1.0, steps: int = 10) -> dict:
        """Hold the eef and drive the gripper command (+1 close, -1 open) for ``steps`` steps."""
        return self._motion(lambda: self._gripper_steps(gripper, steps))

    def release(self, steps: int = 10) -> dict:
        """Open the gripper for ``steps`` steps, the eef held (``set_gripper(-1)``)."""
        return self._motion(lambda: self._gripper_steps(-1.0, steps))

    def scripted_grasp(
        self,
        xyz,
        approach_z: float = 0.1,
        grasp_z_offset: float = 0.0,
        step_clip: float = 0.02,
    ) -> dict:
        """Open, hover ``approach_z`` above ``xyz``, descend to ``grasp_z_offset``, close, lift."""
        t = _vec(xyz, 3, "xyz")

        def at(dz: float) -> np.ndarray:
            return t + np.array([0.0, 0.0, dz])

        def run() -> dict:
            self._gripper_steps(-1.0, 4)
            r = self._servo(at(approach_z), -1.0, step_clip)
            if not r["ok"]:
                return {**r, "stage": "approach"}
            r = self._servo(at(grasp_z_offset), -1.0, 0.012, 200, 0.01)
            if not r["ok"]:
                return {**r, "stage": "descent"}
            self._gripper_steps(1.0, 14)
            r = self._servo(at(approach_z + 0.05), "hold", 0.015)
            if not r["ok"]:
                return {**r, "stage": "lift"}
            return {
                "ok": True,
                "gripper_qpos": [
                    round(float(v), 4) for v in self._vec_obs("robot0_gripper_qpos")
                ],
                "eef": [round(float(v), 4) for v in self._eef()],
            }

        return self._motion(run)

    def _calibrate_forward(self, g: float) -> float:
        """Drive forward briefly and measure the world direction the base goes."""
        p0, y0 = self._base_pos(), self._base_yaw()
        a = self._zero(1.0)
        a[6] = float(np.clip(g, -1, 1))
        a[7] = 1.0
        for _ in range(CALIBRATE_BASE_STEPS):
            self._step_or_stop(a)
        dx, dy = (self._base_pos() - p0)[:2]
        self._fwd_offset = (
            math.atan2(dy, dx) - y0 if math.hypot(dx, dy) > 0.005 else 0.0
        )
        return self._fwd_offset

    def navigate_to(
        self, xy, tol: float = 0.2, max_steps: int = 300, gripper: Any = "hold"
    ) -> dict:
        """Drive the base toward a world (x, y): turn to face it, then drive forward
        closed-loop (the forward heading is measured once per scene); the arm is held."""
        goal = _vec(xy, 2, "xy")

        def run() -> dict:
            q = self._finger()
            if self._fwd_offset is None:
                self._calibrate_forward(self._grip(gripper, q))
            offset = self._fwd_offset or 0.0
            start = self._base_pos()[:2]

            def end(ok: bool, steps: int) -> dict:
                self._pos_jac = None  # the base moved: the arm servo calibrates again
                bp = self._base_pos()
                moved = float(np.linalg.norm(bp[:2] - start))
                return {
                    "ok": ok,
                    "steps": steps,
                    "final_dist": round(float(np.linalg.norm(goal - bp[:2])), 4),
                    "moved": round(moved, 4),
                    # barely moved: rammed a fixture (no path planning)
                    **({} if ok else {"stuck": moved < 0.12}),
                    "start_pos": [round(float(v), 4) for v in start],
                    "base_pos": [round(float(v), 4) for v in bp],
                }

            for i in range(int(max_steps)):
                bp = self._base_pos()
                to = goal - bp[:2]
                if float(np.linalg.norm(to)) < tol:
                    return end(True, i)
                e = math.atan2(to[1], to[0]) - (self._base_yaw() + offset) + math.pi
                dyaw = e - 2 * math.pi * math.floor(e / (2 * math.pi)) - math.pi
                a = self._zero(1.0)
                a[6] = self._grip(gripper, q)
                if abs(dyaw) > 0.3:
                    a[9] = math.copysign(1.0, dyaw)
                else:
                    a[7] = 1.0
                    a[9] = float(np.clip(dyaw * 1.5, -0.4, 0.4))
                self._step_or_stop(a)
            return end(False, int(max_steps))

        return self._motion(run)

    def move_base(
        self,
        forward: float = 0.0,
        lateral: float = 0.0,
        turn: float = 0.0,
        steps: int = 10,
        gripper: Any = "hold",
    ) -> dict:
        """Raw base velocities in the robot's frame (each clipped to [-1, 1]) for ``steps``
        env steps: +forward drives forward, +lateral strafes, +turn rotates counter-clockwise."""

        def run() -> dict:
            q = self._finger()
            a = self._zero(1.0)
            a[7] = float(np.clip(forward, -1, 1))
            a[8] = float(np.clip(lateral, -1, 1))
            a[9] = float(np.clip(turn, -1, 1))
            bp0 = self._base_pos()
            for _ in range(int(steps)):
                a[6] = self._grip(gripper, q)
                self._step_or_stop(a)
            bp1 = self._base_pos()
            return {
                "ok": True,
                "base_moved": [round(float(v), 4) for v in bp1 - bp0],
                "base_pos": [round(float(v), 4) for v in bp1],
            }

        return self._motion(run)

    # ---- the high tier (CaP-X's semantic functions) ----

    def goto_pose(self, position, z_approach: float = 0.0) -> dict:
        """CaP-X's goto_pose for the OSC servo, which holds the gripper's orientation: stop
        ``z_approach`` above ``position`` first, then go down to it."""
        p = _vec(position, 3, "position")

        def run() -> dict:
            if float(z_approach) > 0:
                r = self._servo(p + np.array([0.0, 0.0, float(z_approach)]))
                if not r["ok"]:
                    return {**r, "stage": "approach"}
            return self._servo(p)

        return self._motion(run)

    def _home(self) -> np.ndarray:
        if self._home_rel is None:
            raise RuntimeError("no home pose: reset the env first")
        return (
            self._base_pos() + _rot(self._vec_obs("robot0_base_quat")) @ self._home_rel
        )

    def home_pose(self) -> dict:
        """Move the eef back to its reset position relative to the base (where the base is now)."""
        return self._motion(lambda: self._servo(self._home()))

    def open_gripper(self) -> dict:
        """Open the gripper fully."""
        return self._motion(lambda: self._gripper_steps(-1.0, GRIPPER_STEPS))

    def close_gripper(self) -> dict:
        """Close the gripper fully (the fingers stop on an object)."""
        return self._motion(lambda: self._gripper_steps(1.0, GRIPPER_STEPS))

    def get_state(self) -> dict:
        """The robot's proprioception (robot0_* observations, no images) and the env steps."""
        if self._obs is None:
            raise RuntimeError("no observation yet: reset the env first")
        return {**robot_obs(self._obs), "env_steps": self._steps}

    def _world_map(
        self, camera: str, size: int = VIDEO_SIZE
    ) -> tuple[np.ndarray, np.ndarray]:
        """A camera's top-down RGB and per-pixel world xyz (pi's world map: T_p2w @ [col·z,
        row·z, z, 1] with the top-down row and the bottom-up depth of that row)."""
        rgb, depth = self.render_camera(camera, size, size, True)
        rgb = np.ascontiguousarray(np.asarray(rgb)[::-1])
        z = np.asarray(depth, dtype=np.float64)[::-1]
        T = np.asarray(self.get_camera_transform(camera, size, size), dtype=np.float64)
        rows, cols = np.mgrid[0:size, 0:size]
        pix = np.stack([cols * z, rows * z, z, np.ones_like(z)], axis=-1)
        return rgb, (pix @ T.T)[..., :3]

    def _segment(self, rgb: np.ndarray, prompt: str) -> np.ndarray | None:
        """SAM3's top mask of ``prompt`` in ``rgb`` (bool, top-down), or None."""
        from PIL import Image

        from pi_embodied_services.utils.rpc.http_rpc import HttpRpcClient

        if not self._sam3_url:
            raise RuntimeError("get_object_pose needs a SAM3 server (--sam3)")
        if self._sam3 is None:
            self._sam3 = HttpRpcClient(self._sam3_url)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        res = self._sam3.call(
            "sam3.segment",
            kwargs={
                "image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                "text_prompt": prompt,
                "min_score": 0.2,
            },
            timeout_s=120,
        )
        if not res.get("found") or not res.get("mask_png_base64"):
            return None
        mask = np.asarray(
            Image.open(io.BytesIO(base64.b64decode(res["mask_png_base64"])))
        )
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask >= 128 if mask.shape == rgb.shape[:2] else None

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> list:
        """CaP-X's get_object_pose from perception: [position (3,), quaternion_wxyz (4,),
        bbox_extent (3,) or None] of an object named in words, from the first camera (agentview,
        then wrist) whose SAM3 mask has depth. The position is the median of the mask's world
        points, the extent their 5th-95th percentile span per axis; the orientation is identity."""
        text = str(object_name).strip()
        if not text:
            raise ValueError("object_name is empty")
        for camera in ("robot0_agentview_left", "robot0_eye_in_hand"):
            rgb, xyz = self._world_map(camera)
            mask = self._segment(rgb, text)
            if mask is None:
                continue
            pts = xyz[mask]
            pts = pts[np.isfinite(pts).all(axis=1) & (np.abs(pts).sum(axis=1) > 1e-6)]
            if len(pts) < 10:
                continue
            pos = np.median(pts, axis=0)
            extent = np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0)
            return [
                [round(float(v), 4) for v in pos],
                [1.0, 0.0, 0.0, 0.0],
                [round(float(v), 4) for v in extent] if return_bbox_extent else None,
            ]
        raise ValueError(f"no SAM3 detection with depth for {text!r}")

    def _motion_move_m(self, method: str, kwargs: dict) -> float:
        """How far a program's motion call may move the gripper (the run's translation cap)."""
        eef = self._eef()
        if method == "env.move_to":
            return float(np.linalg.norm(_vec(kwargs["xyz"], 3, "xyz") - eef))
        if method == "env.move_delta":
            return float(np.linalg.norm(_vec(kwargs["dxyz"], 3, "dxyz")))
        if method == "env.scripted_grasp":
            t = _vec(kwargs["xyz"], 3, "xyz")
            up = float(kwargs.get("approach_z") or 0.1)
            down = float(kwargs.get("grasp_z_offset") or 0.0)
            return float(
                np.linalg.norm(t + [0, 0, up] - eef) + 2 * abs(up - down) + 0.05
            )
        if method == "env.goto_pose":
            p = _vec(kwargs["position"], 3, "position")
            up = float(kwargs.get("z_approach") or 0.0)
            return float(np.linalg.norm(p + [0, 0, up] - eef) + up)
        if method == "env.home_pose":
            return float(np.linalg.norm(self._home() - eef))
        if method == "env.navigate_to":
            goal = _vec(kwargs["xy"], 2, "xy")
            return float(np.linalg.norm(goal - self._base_pos()[:2]))
        if method == "env.move_base":
            n = int(kwargs.get("steps") or 10)
            v = [
                float(np.clip(kwargs.get(k) or 0.0, -1, 1))
                for k in ("forward", "lateral", "turn")
            ]
            return n * BASE_M_PER_STEP * (math.hypot(v[0], v[1]) + abs(v[2]))
        return 0.0

    def check_success(self):
        return bool(self.env._check_success())

    def render_camera(self, camera_name, height, width, depth):
        """sim.render in ROBOSUITE-NATIVE orientation (matches the camera
        transform matrices). rgb uint8 HxWx3, depth metric HxW."""
        import robosuite.utils.camera_utils as CU

        out = self.env.sim.render(
            width=width, height=height, camera_name=camera_name, depth=depth
        )
        if depth:
            rgb, d = out
            # Sanitize the raw OpenGL normalized depth into [0,1]: replace NaN/inf
            # (degenerate camera pose) then clip numerical overshoot. Otherwise an
            # assertion inside get_real_depth_map crashes the whole env server process.
            d = np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=0.0)
            d = np.clip(d, 0.0, 1.0)
            if d.ndim == 3:
                depth = CU.get_real_depth_map(self.env.sim, d)[..., 0]
            else:
                depth = CU.get_real_depth_map(self.env.sim, d[..., None])[..., 0]
            return rgb, depth
        return out

    def get_camera_meta(self, camera_name, height=None, width=None):
        import robosuite.utils.camera_utils as CU

        K = CU.get_camera_intrinsic_matrix(self.env.sim, camera_name, height, width)
        Ext = CU.get_camera_extrinsic_matrix(self.env.sim, camera_name)  # cam->world
        m = self.env.sim.model
        extent = m.stat.extent
        return {
            "camera_name": camera_name,
            "height": height,
            "width": width,
            "intrinsic": np.asarray(K, dtype=np.float64).tolist(),
            "extrinsic_cam2world": np.asarray(Ext, dtype=np.float64).tolist(),
            "depth_near": float(m.vis.map.znear * extent),
            "depth_far": float(m.vis.map.zfar * extent),
        }

    def get_camera_transform(self, camera_name, height=None, width=None):
        import robosuite.utils.camera_utils as CU

        T = CU.get_camera_transform_matrix(self.env.sim, camera_name, height, width)
        return np.linalg.inv(T)  # T_p2w

    def get_task_language(self) -> str | None:
        return self.env.get_ep_meta().get("lang")

    def grasp_contact(self):
        """Check if the gripper is currently contacting a task object."""
        try:
            robo = self.env  # robosuite Kitchen env
            grip = robo.robots[0].gripper  # {"right": GripperModel}
            for name, obj in robo.objects.items():
                try:
                    if robo._check_grasp(grip, obj):
                        return True, name
                except Exception:
                    continue
        except Exception:
            pass
        return False, None

    def reassemble_env_action(self, unmap_result):
        """Reassemble the unmap result into a flat action using the env's robots."""
        from robosuite.controllers.composite.composite_controller import (
            HybridMobileBase,
        )

        env_action = []
        for robot in self.env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            a = np.zeros(cc.action_limits[0].shape)
            for part_name in cc.part_controllers:
                s, e = cc._action_split_indexes[part_name]
                a[s:e] = unmap_result.pop(f"{pf}{part_name}")
            if isinstance(cc, HybridMobileBase):
                a[-1] = unmap_result.pop(f"{pf}base_mode")
            env_action.append(a)
        return np.concatenate(env_action)

    def get_success_criteria_text(self):
        """Return the success_criteria.md text for this task."""
        env = self.env
        out = []
        try:
            src = inspect.getsource(type(env)._check_success)
            out.append(
                "# SUCCESS CONDITION for this task (env._check_success)\n"
                "# You must make this return True. Object positions are NOT given —\n"
                "# localize every named object/fixture from the camera+world maps.\n\n"
                + src
            )
            try:
                import robocasa.utils.object_utils as OU

                for fn in sorted(set(re.findall(r"OU\.(\w+)\(", src))):
                    f = getattr(OU, fn, None)
                    if f is not None:
                        try:
                            out.append(
                                "## helper OU.%s\n%s" % (fn, inspect.getsource(f))
                            )
                        except Exception:
                            pass
            except Exception:
                pass
            for fix, meth in sorted(set(re.findall(r"self\.(\w+)\.(\w+)\(", src))):
                obj = getattr(env, fix, None)
                if obj is not None and hasattr(type(obj), meth):
                    try:
                        out.append(
                            "## %s.%s\n%s"
                            % (fix, meth, inspect.getsource(getattr(type(obj), meth)))
                        )
                    except Exception:
                        pass
        except Exception as ex:
            out.append("(_check_success extraction failed: %s)" % ex)
        return "\n\n".join(out)[:9000]

    def get_task_progress(self):
        """Return the progress dict for this task."""
        env = self.env
        prog = {}
        code = type(env)._check_success.__code__
        try:
            src = inspect.getsource(type(env)._check_success)
            # capture both `self.attr` AND dotted `self.fixture._attr` paths used in the
            # success check (e.g. self.coffee_machine._turned_on) — a bare-attr regex
            # would only grab "coffee_machine" (the fixture object) and miss the real
            # gating flag. Resolve each dotted path to its live scalar/bool value.
            for path in sorted(
                set(re.findall(r"self\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)", src))
            ):
                obj = env
                ok = True
                for part in path.split("."):
                    obj = getattr(obj, part, None)
                    if obj is None:
                        ok = False
                        break
                if not ok:
                    continue
                key = path.replace(".", "_")
                if isinstance(obj, (bool, np.bool_)):
                    prog[key] = bool(obj)
                elif isinstance(obj, (int, np.integer)):
                    prog[key] = int(obj)
                elif isinstance(obj, (float, np.floating)):
                    prog[key] = round(float(obj), 4)
        except Exception:
            pass
        # trace ONE read-only call of _check_success; grab its return-frame locals
        captured = {}

        def _tracer(frame, event, arg):
            if event == "call" and frame.f_code is code:

                def _local(f, e, a):
                    if e == "return":
                        captured.update(f.f_locals)
                    return _local

                return _local
            return None

        old = sys.gettrace()
        try:
            sys.settrace(_tracer)
            env._check_success()
        except Exception:
            pass
        finally:
            sys.settrace(old)
        for k, v in captured.items():
            if k == "self" or k in prog:
                continue
            if isinstance(v, (bool, np.bool_)):
                prog[k] = bool(v)
            elif isinstance(v, (int, np.integer)):
                prog[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                prog[k] = round(float(v), 4)
        return prog

    def ground_truth_poses(self, names=None):
        """World poses of ``names`` (default all) from the kitchen's own object list
        (``--privileged``): its objects (``obj_body_id``), then its fixtures' root bodies."""
        env = self.env
        ids = dict(env.obj_body_id)
        for name, fixture in env.fixtures.items():
            try:
                ids.setdefault(name, env.sim.model.body_name2id(fixture.root_body))
            except (KeyError, ValueError):
                pass  # a fixture merged into another body has none of its own
        return ground_truth.respond(ground_truth.mujoco_body_poses(env.sim, ids), names)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
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
    p.add_argument("--task-name", default="OpenDrawer")
    p.add_argument("--split", default="target")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--scene",
        type=int,
        default=None,
        help="RoboCasa365 manifest scene index (0-49); overrides --seed",
    )
    add_perception_arguments(p, sam3=True)
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

    facade = RoboCasaEnvFacade(
        args.task_name,
        split=args.split,
        seed=args.seed,
        scene=args.scene,
    )
    # --sam3 / --unidepth: env.detect, env.select_detection, env.reject_detection,
    # env.enhance_depth, on the upright 256 px views the model sees.
    facade._sam3_url = args.sam3 or ""
    install_perception(
        facade,
        args,
        cameras=["agentview", "navview", "wrist"],
        mutating=MOTION_METHODS,
        view=render_view(
            facade,
            size=256,
            flip=True,
            cameras={
                "agentview": "robot0_agentview_left",
                "navview": "mobilebase0_navview",
                "wrist": "robot0_eye_in_hand",
            },
        ),
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
