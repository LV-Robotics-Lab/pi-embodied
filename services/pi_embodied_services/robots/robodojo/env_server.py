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

"""RPC server wrapping one RoboDojo task (Isaac Sim / Isaac Lab) on its two ARX X5 arms.

One env per process: RoboDojo's heterogeneous parallel simulation (several envs, even several
tasks, in one Kit process) is not used; a parallel sweep runs one server per episode.

The native interface is RoboDojo's own ``EvalEnv`` (sim.make_env): ``env.reset(seed)`` loads eval
layout ``seed`` (layouts are pre-generated per task, ``Assets/Eval_Layout``; the seed is the layout
id), ``env.step(action)`` is one ``take_action`` with RoboDojo's action dict (``left_arm_joint_state``
6 + ``left_ee_joint_state`` 1, or ``left_ee_pose`` 7, and the same for ``right_``; validated by
``validate_action_dict``, interpolated over 10 physics steps = one 25 Hz control step), and
``env.get_obs`` is RoboDojo's observation dict (what XPolicyLab policies read). Each action counts
against the task's ``step_lim``; the episode ends on RoboDojo's ``is_episode_end`` (success when the
task's reward checks all pass, most of them including "both arms back at their start pose").

The agent's motion primitives act on one arm and hold the other at its last commanded joints:
``env.move_to`` / ``env.move_delta`` interpolate the end effector in the env frame (at most
``STEP_M`` and ``STEP_RAD`` per control step) through RoboDojo's cuRobo IK, ``env.rotate_delta``
turns the gripper about the vertical, ``env.set_gripper`` moves the normalized gripper
(1 open .. 0 closed, RoboDojo's ``ee_joint_state``), and ``env.go_home`` drives both arms back to
their reset joints. All of them go through ``take_action`` in joint mode, so RoboDojo counts and
judges them like any policy's actions.

Isaac Sim starts in ``main`` before the server binds; every call runs on the main thread (Kit is
not thread-safe).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from typing import Any

import numpy as np

from pi_embodied_services.components.code_api import register_code_api
from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.robots.robodojo import sim
from pi_embodied_services.robots.robodojo.primitives import ROBODOJO_PRIMITIVES
from pi_embodied_services.utils import ground_truth
from pi_embodied_services.utils.rpc.main_thread_serve import MainThreadServeMixin

#: End-effector travel per control step (25 Hz): 1 cm = 0.25 m/s.
STEP_M = 0.01
#: End-effector rotation per control step, rad (about 72 deg/s).
STEP_RAD = 0.05
#: Largest end-effector translation one call may command, m.
MAX_MOVE_M = 0.5
#: Largest yaw one rotate_delta call may command, rad (more is clipped and reported).
MAX_ROTATE_RAD = 0.8
#: An IK solution that moves any joint more than this in one control step is refused (a branch flip).
MAX_JOINT_STEP_RAD = 0.5
#: Control steps a gripper command is held so the fingers finish moving.
GRIPPER_STEPS = 8
#: go_home: largest joint change per control step, rad.
HOME_STEP_RAD = 0.05
#: Camera views the agent sees, RoboDojo's names -> the robot's.
VIEWS = {
    "cam_head": "head",
    "cam_left_wrist": "left_wrist",
    "cam_right_wrist": "right_wrist",
}
#: A video frame (head view) every this many control steps with return_frames (5 fps).
FRAME_EVERY = 5


def arm_key(arm: str, suffix: str) -> str:
    if arm not in sim.ARMS:
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
    return f"{arm}_{suffix}"


class RobodojoEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """One RoboDojo ``EvalEnv`` (``num_envs=1``) and the agent's motion primitives."""

    SERVICE_NAME = "robodojo-env"

    def __init__(self, *, app: Any, env: Any, meta: dict):
        super().__init__()
        self._app = app
        self._env = env
        self._meta = dict(meta)
        self._closed = False
        self._episodes = 0
        self._obs: dict = {}
        #: Last commanded joints (6) and normalized gripper per arm: what a one-arm motion holds.
        self._q: dict[str, np.ndarray] = {}
        self._grip: dict[str, float] = {}
        self._home: dict[str, np.ndarray] = {}
        self._seed = int(meta["seed"])
        self._layout_error: str | None = None
        #: Flywheel recording (env.set_recording): one policy frame per control step, handed back
        #: (and cleared) with the next observation.
        self._recording = False
        self._frames: list[dict] = []

    def _register_rpc(self) -> None:
        super()._register_rpc()
        for name in (
            "get_obs",
            "state",
            "move_to",
            "move_delta",
            "rotate_delta",
            "set_gripper",
            "go_home",
            "back_project",
            "ground_truth_poses",
            "set_recording",
        ):
            self._rpc[f"env.{name}"] = getattr(self, name)
        self._readonly_methods.add("env.state")
        register_code_api(self, ROBODOJO_PRIMITIVES)

    # ---- RoboDojo accessors ----

    def _robot(self, arm: str) -> Any:
        name = arm_key(arm, "arm")
        return next(r for r in self._env.robot_manager.robot_list if r.arm_name == name)

    def _joints(self, arm: str) -> np.ndarray:
        return np.asarray(
            self._env.robot_manager.get_joint(self._robot(arm), env_idx_list=[0])[0],
            dtype=float,
        )

    def _ee_pose(self, arm: str) -> np.ndarray:
        """``[x, y, z, qw, qx, qy, qz]`` of the arm's end-effector link in the env frame (RoboDojo's
        ``*_ee_pose`` observation and ``ee_pose`` action frame)."""
        rm = self._env.robot_manager
        return np.asarray(
            rm.get_real_endpose(self._robot(arm), env_idx_list=[0], is_relative=True)[
                0
            ],
            dtype=float,
        )

    def _ended(self) -> bool:
        return bool(self._env.end_flag[0])

    def _success(self) -> bool:
        return bool(self._env.end_flag[0] and self._env.success[0])

    def _truncated(self) -> bool:
        return (
            int(self._env.take_action_cnt[0]) >= int(self._env.step_lim)
            and not self._success()
        )

    def _score(self) -> float:
        """RoboDojo's episode score: 1 on success, else the task's partial-credit tiers / 100
        (run_eval's accounting), 0 for tasks without ``get_score``."""
        if self._success():
            return 1.0
        rm = self._env.reward_manager
        if not hasattr(self._env, "get_score"):
            return 0.0
        return float(rm.get_score()[0]) / 100.0

    # ---- observations ----

    def _observe(self) -> dict:
        """A fresh RoboDojo observation (renders the cameras)."""
        self._obs = self._env.get_obs()
        return self._obs

    def _image(self, cam: str) -> np.ndarray:
        vision = self._obs.get("vision", {})
        if cam not in vision:
            raise ValueError(
                f"no camera {cam!r} in the observation; have {sorted(vision)}"
            )
        return np.ascontiguousarray(
            np.asarray(vision[cam]["color"])[..., :3], dtype=np.uint8
        )

    def _images(self) -> dict:
        return {
            view: self._image(cam)
            for cam, view in VIEWS.items()
            if cam in self._obs.get("vision", {})
        }

    def _state(self) -> dict:
        arms = {}
        for arm in sim.ARMS:
            pose = self._ee_pose(arm)
            arms[arm] = {
                "eef_pos": pose[:3].astype(np.float32),
                "eef_quat_wxyz": pose[3:].astype(np.float32),
                "joints": self._joints(arm).astype(np.float32),
                "gripper": round(float(self._gripper_measured(arm)), 4),
                "gripper_command": round(float(self._grip.get(arm, 1.0)), 4),
                # What a one-arm motion holds this arm at (the last commanded joints).
                "joints_command": np.asarray(
                    self._q.get(arm, self._joints(arm)), dtype=np.float32
                ),
            }
        return {
            "arms": arms,
            "success": self._success(),
            "ended": self._ended(),
            "truncated": self._truncated(),
            "score": round(self._score(), 4),
            "env_steps": int(self._env.take_action_cnt[0]),
            "step_lim": int(self._env.step_lim),
            "seed": self._seed,
        }

    def _gripper_measured(self, arm: str) -> float:
        """The gripper opening normalized like the command (obs ``*_ee_joint_state``: 1 open, 0 closed)."""
        robot = self._robot(arm)
        val = float(
            np.asarray(
                self._env.robot_manager.get_end_effector_real_val(
                    robot, env_idx_list=[0]
                )[0]
            )[0]
        )
        lo, hi = robot.gripper_scale
        frac = (val - lo) / (hi - lo)
        return frac if robot.gripper_move["sign"] == 1 else 1.0 - frac

    def _pack(self) -> dict:
        self._observe()
        out = {**self._images(), **self._state()}
        if self._recording:
            out["policy_frames"], self._frames = self._frames, []
        return out

    def _vector(
        self, arm_values: dict[str, np.ndarray], grips: dict[str, float]
    ) -> np.ndarray:
        """``[left joints 6, left gripper, right joints 6, right gripper]`` (the Flywheel's layout)."""
        return np.concatenate(
            [np.r_[arm_values[a], grips[a]] for a in sim.ARMS]
        ).astype(np.float32)

    def _policy_frame(self, action: np.ndarray) -> dict:
        """The Flywheel's record of one control step: the three images after it, the measured
        joints and grippers, and the joint-space command it ran."""
        self._observe()
        state = self._vector(
            {a: self._joints(a) for a in sim.ARMS},
            {a: self._gripper_measured(a) for a in sim.ARMS},
        )
        return {**self._images(), "state": state, "action": action}

    def _recorded(self, action: np.ndarray) -> None:
        if self._recording:
            self._frames.append(self._policy_frame(action))

    def set_recording(self, on: bool = True) -> dict | None:
        """Turn per-step Flywheel recording on or off; on returns the current frame (the episode's first)."""
        self._recording = bool(on)
        self._frames = []
        if not self._recording:
            return None
        return self._policy_frame(self._vector(self._q, self._grip))

    # ---- stepping ----

    def _joint_action(self) -> dict:
        return {
            **{arm_key(a, "arm_joint_state"): self._q[a].tolist() for a in sim.ARMS},
            **{arm_key(a, "ee_joint_state"): [float(self._grip[a])] for a in sim.ARMS},
        }

    def _act(self, frames: list | None) -> None:
        """One control step with the held joint and gripper targets (RoboDojo's ``take_action``)."""
        self._env.take_action(self._joint_action())
        self._recorded(self._vector(self._q, self._grip))
        if frames is not None and int(self._env.take_action_cnt[0]) % FRAME_EVERY == 0:
            self._observe()
            frames.append(self._image("cam_head"))

    def _refuse(self) -> str | None:
        if self._layout_error:
            return self._layout_error
        if self._ended():
            return "the episode is over"
        return None

    # ---- gym-like surface ----

    def reset(self, seed: int | None = None):
        """Load eval layout ``seed`` (default: the server's ``--seed``) and start a new episode.

        A second reset closes the simulation first and relaunches it, as RoboDojo's main.py does
        between batches. An unstable layout (RoboDojo discards it from the benchmark) is reported
        in ``info.error`` and every motion is refused until the next reset.
        """
        seed = self._seed if seed is None else int(seed)
        n = int(self._meta["layouts"])
        if not 0 <= seed < n:
            raise ValueError(
                f"seed (layout id) must be in [0, {n}) for {self._meta['task']}, got {seed}"
            )
        env = self._env
        if self._episodes:
            env.close()
        self._episodes += 1
        self._seed = seed
        self._frames = []
        self._layout_error = None
        env.env_seeds = [seed]
        try:
            env.reset(seed=[seed])
        except sim.unstable_error():
            self._layout_error = f"layout {seed} is unstable in simulation (RoboDojo skips it); reset another seed"
        if self._layout_error is None:
            # run_eval's preamble: register the success checks and score tiers, start support arms.
            env.run_reward()
            if hasattr(env, "get_score"):
                env.get_score()
            if getattr(env, "interact", False) and hasattr(
                env, "query_support_arm_traj"
            ):
                env.query_support_arm_traj(env_idx=0)
        self._grip = {a: 1.0 for a in sim.ARMS}
        self._q = {a: self._joints(a) for a in sim.ARMS}
        self._home = {a: self._q[a].copy() for a in sim.ARMS}
        info = {"instruction": self.get_task_language(), "seed": seed}
        if self._layout_error:
            info["error"] = self._layout_error
        return self._pack(), info

    def step(self, action: dict):
        """One native RoboDojo action (``take_action``); returns (obs, score, terminated, truncated, info)."""
        why = self._refuse()
        if why:
            return (
                self._pack(),
                self._score(),
                self._success(),
                self._truncated(),
                {"error": why},
            )
        self._native(action)
        return (
            self._pack(),
            self._score(),
            self._success(),
            self._truncated(),
            {"success": self._success()},
        )

    def chunk_step(self, actions: list, *, return_all_frames: bool = False):
        """Native actions in order (an XPolicyLab chunk); stops at the episode's end or ``stop``."""
        frames, info = [], {}
        executed = 0
        for action in actions:
            if self.stop_requested():
                info["cancelled"] = True
                break
            why = self._refuse()
            if why:
                info["error"] = why
                break
            self._native(action)
            executed += 1
            if return_all_frames:
                self._observe()
                frames.append(self._image("cam_head"))
        info.update(executed=executed, success=self._success())
        obs = self._pack()
        if return_all_frames:
            obs["frames"] = frames
        return obs, self._success(), self._truncated(), info

    def _native(self, action: dict) -> None:
        """One RoboDojo action dict through ``validate_action_dict`` and ``take_action``; its
        gripper values become the held gripper commands."""
        self._env.validate_action_dict(action)
        self._env.take_action(action)
        for a in sim.ARMS:
            g = action.get(arm_key(a, "ee_joint_state"))
            if g is not None:
                self._grip[a] = float(
                    np.clip(np.asarray(g, dtype=float).reshape(-1)[0], 0.0, 1.0)
                )
            q = action.get(arm_key(a, "arm_joint_state"))
            # A joint action is its own command; an end-effector action's is where IK took the arm.
            self._q[a] = (
                np.asarray(q, dtype=float).reshape(-1)
                if q is not None
                else self._joints(a)
            )
        self._recorded(self._vector(self._q, self._grip))

    def get_obs(self, depth: bool = False) -> dict:
        """RoboDojo's observation dict for env 0 (``EvalEnv.get_obs``): ``vision`` per camera
        (``color`` HxWx3 uint8, ``depth`` metres with ``depth``), ``state`` / ``action`` (joint
        states, ``*_ee_pose``, normalized ``*_ee_joint_state``), ``instruction``."""
        obs = self._observe()
        if depth:
            return obs
        return {
            **obs,
            "vision": {
                k: {kk: vv for kk, vv in v.items() if kk != "depth"}
                for k, v in obs["vision"].items()
            },
        }

    # ---- motion primitives ----

    def _ik(self, arm: str, pose: np.ndarray) -> np.ndarray | None:
        res = self._env.robot_manager.solve_ik(
            target_pose=pose.tolist(), env_idx=0, robot=self._robot(arm)
        )
        if res.get("status") != "Success":
            return None
        return np.asarray(res["joint_value"], dtype=float).reshape(-1)

    def _gripper_to(self, arm: str, target: float, frames: list | None) -> int:
        """Command the gripper and hold ``GRIPPER_STEPS`` control steps; returns the steps run."""
        self._grip[arm] = float(np.clip(target, 0.0, 1.0))
        n = 0
        for _ in range(GRIPPER_STEPS):
            if self.stop_requested() or self._ended():
                break
            self._act(frames)
            n += 1
        return n

    def _travel(
        self, arm: str, goal_pos, goal_quat, gripper: float | None, return_frames: bool
    ) -> dict:
        """Shared body of move_to / move_delta / rotate_delta."""
        start = self._ee_pose(arm)
        report: dict[str, Any] = {"arm": arm}
        why = self._refuse()
        if why:
            return {**self._pack(), **report, "error": why, "moved_m": [0.0, 0.0, 0.0]}
        frames: list | None = [] if return_frames else None
        steps = 0
        if gripper is not None and abs(float(gripper) - self._grip[arm]) > 1e-6:
            steps += self._gripper_to(arm, float(gripper), frames)
        path = sim.waypoints(
            start[:3], goal_pos, start[3:], goal_quat, step_m=STEP_M, step_rad=STEP_RAD
        )
        done = 0
        stop = None
        for pose in path:
            if self.stop_requested():
                report["cancelled"] = True
                break
            if self._ended():
                break
            q = self._ik(arm, pose)
            if q is None:
                stop = "ik_failed"
                break
            jump = float(np.max(np.abs(q - self._q[arm])))
            if jump > MAX_JOINT_STEP_RAD:
                stop = f"ik_jump ({jump:.2f} rad in one step)"
                break
            self._q[arm] = q
            self._act(frames)
            steps += 1
            done += 1
        end = self._ee_pose(arm)
        report.update(
            moved_m=[round(float(v), 4) for v in end[:3] - start[:3]],
            final_error_m=round(
                float(np.linalg.norm(end[:3] - np.asarray(goal_pos, dtype=float))), 4
            ),
            waypoints=len(path),
            executed=done,
            control_steps=steps,
        )
        if stop:
            report["stopped"] = stop
            report["hint"] = (
                "the target may be out of reach or need another orientation; try a closer or higher waypoint"
            )
        if frames is not None:
            report["frames"] = frames
        return {**self._pack(), **report}

    def move_to(
        self,
        arm: str,
        xyz,
        quat_wxyz=None,
        gripper: float | None = None,
        return_frames: bool = False,
    ) -> dict:
        """Move one arm's end effector to ``xyz`` (env frame, m) and, optionally, orientation
        ``quat_wxyz`` (default: keep it), after an optional gripper command (1 open .. 0 closed).
        The other arm holds still. Straight-line interpolation, ``STEP_M`` per control step; stops
        at the first unreachable waypoint (``stopped``)."""
        xyz = np.asarray(xyz, dtype=float).reshape(3)
        start = self._ee_pose(arm)
        dist = float(np.linalg.norm(xyz - start[:3]))
        if not dist <= MAX_MOVE_M:
            raise ValueError(
                f"the target is {dist:.3f} m away; the limit is {MAX_MOVE_M} m per call"
            )
        quat = (
            start[3:]
            if quat_wxyz is None
            else np.asarray(quat_wxyz, dtype=float).reshape(4)
        )
        return self._travel(arm, xyz, quat, gripper, return_frames)

    def move_delta(
        self,
        arm: str,
        delta_xyz,
        gripper: float | None = None,
        return_frames: bool = False,
    ) -> dict:
        """Translate one arm's end effector by ``delta_xyz`` (env frame, m), orientation held."""
        delta = np.asarray(delta_xyz, dtype=float).reshape(3)
        norm = float(np.linalg.norm(delta))
        if not norm <= MAX_MOVE_M:
            raise ValueError(
                f"delta moves {norm:.3f} m; the limit is {MAX_MOVE_M} m per call"
            )
        start = self._ee_pose(arm)
        return {
            **self._travel(arm, start[:3] + delta, start[3:], gripper, return_frames),
            "commanded_m": [float(v) for v in delta],
        }

    def rotate_delta(self, arm: str, yaw: float, return_frames: bool = False) -> dict:
        """Turn one arm's gripper by ``yaw`` (rad, + counter-clockwise seen from above) about the
        vertical through its end effector; |yaw| beyond ``MAX_ROTATE_RAD`` is clipped."""
        requested = float(yaw)
        commanded = float(np.clip(requested, -MAX_ROTATE_RAD, MAX_ROTATE_RAD))
        start = self._ee_pose(arm)
        out = self._travel(
            arm, start[:3], sim.yaw_quat(start[3:], commanded), None, return_frames
        )
        end = self._ee_pose(arm)
        out.update(
            requested_yaw=requested,
            commanded_yaw=commanded,
            yaw=round(_yaw_between(start[3:], end[3:]), 4),
        )
        if commanded != requested:
            out["clipped"] = True
        return out

    def set_gripper(self, arm: str, value: float, return_frames: bool = False) -> dict:
        """Move one gripper to ``value`` (1 open .. 0 closed) and hold ``GRIPPER_STEPS`` control steps."""
        why = self._refuse()
        if why:
            return {**self._pack(), "arm": arm, "error": why}
        frames: list | None = [] if return_frames else None
        n = self._gripper_to(arm, float(value), frames)
        out = {**self._pack(), "arm": arm, "control_steps": n}
        if frames is not None:
            out["frames"] = frames
        return out

    def go_home(self, return_frames: bool = False) -> dict:
        """Drive both arms back to their joints at reset (joint-space, ``HOME_STEP_RAD`` per control
        step), grippers unchanged. Most tasks only succeed with both arms back at their start pose."""
        why = self._refuse()
        if why:
            return {**self._pack(), "error": why}
        frames: list | None = [] if return_frames else None
        start = {a: self._q[a].copy() for a in sim.ARMS}
        span = max(float(np.max(np.abs(self._home[a] - start[a]))) for a in sim.ARMS)
        n = max(1, math.ceil(span / HOME_STEP_RAD))
        steps = 0
        for i in range(1, n + 1):
            if self.stop_requested():
                break
            if self._ended():
                break
            for a in sim.ARMS:
                self._q[a] = start[a] + (self._home[a] - start[a]) * (i / n)
            self._act(frames)
            steps += 1
        # A few settling steps at the home joints (RoboDojo's origin check has a position threshold).
        for _ in range(3):
            if self._ended():
                break
            self._act(frames)
            steps += 1
        out = {**self._pack(), "control_steps": steps}
        if frames is not None:
            out["frames"] = frames
        return out

    # ---- perception / meta ----

    def state(self) -> dict:
        """The current state (no stepping, no rendering)."""
        return self._state()

    def render_camera(self, camera_name: str = "head", **_: Any):
        """Latest frame of ``head``, ``left_wrist`` or ``right_wrist``."""
        images = self._images() if self._obs else self._pack()
        if camera_name not in images:
            raise ValueError(f"camera_name must be one of {sorted(images)}")
        return images[camera_name]

    def get_camera_meta(self, camera_name: str = "head", **_: Any) -> dict:
        """OpenCV intrinsics and camera-to-env extrinsic (the frame of ``eef_pos``) of the head camera."""
        if camera_name != "head":
            raise ValueError("only the head camera (fixed, cam_head) is calibrated")
        h, w = self._image("cam_head").shape[:2]
        return sim.camera_meta(self._env, "cam_head", w, h)

    def back_project(self, pixels, camera_name: str = "head") -> dict:
        """Env-frame xyz of ``[col, row]`` pixels of the latest head image, from its same-step
        metric depth (None where there is no depth)."""
        if camera_name != "head":
            raise ValueError("back_project reads the head camera only")
        depth = self._obs.get("vision", {}).get("cam_head", {}).get("depth")
        if depth is None:
            raise ValueError(
                "no head depth in the latest observation (the server runs with --no-depth)"
            )
        meta = self.get_camera_meta("head")
        return {
            "frame": "env",
            "xyz": sim.back_project(
                depth, meta["intrinsic_K"], meta["extrinsic_cam2world"], pixels
            ),
        }

    def ground_truth_poses(self, names=None) -> dict:
        """Poses of ``names`` (default all) of the task's labelled objects (``--privileged``), in the
        env frame of ``eef_pos``."""
        return {
            **ground_truth.respond(sim.object_poses(self._env), names),
            "frame": "env",
        }

    def get_task_language(self) -> str:
        instr = getattr(self._env.obs_manager, "instruction", None)
        return str(instr[0]) if instr else ""

    def get_env_meta(self) -> dict:
        return {
            **self._meta,
            "seed": self._seed,
            "instruction": self.get_task_language(),
            "step_lim": int(self._env.step_lim),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._env.close()
        finally:
            print("[robodojo-env] closing Isaac Sim", flush=True)
            self._app.close()


def _yaw_between(q0, q1) -> float:
    """Signed heading change (rad) about world +z from q0 to q1 (wxyz), via the gripper x axis."""

    def heading(q):
        w, x, y, z = q
        return math.atan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))

    d = heading(q1) - heading(q0)
    return (d + math.pi) % (2 * math.pi) - math.pi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument(
        "--task",
        default="stack_bowls",
        help="RoboDojo task (task/RoboDojo/tasks/<task>.py)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="eval layout id (RoboDojo's seed; the first reset)",
    )
    p.add_argument(
        "--eval-seed",
        type=int,
        default=0,
        help="layout set (Assets/Eval_Layout/.../<eval-seed>)",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="physical GPU (CUDA_VISIBLE_DEVICES, as RoboDojo)",
    )
    p.add_argument(
        "--no-depth",
        action="store_true",
        help="skip the depth render (no back_project)",
    )
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="exit when stdin closes (the parent died)",
    )
    args = p.parse_args()

    # RoboDojo hard-codes cuda:0 (cuRobo, warp buffers): expose only the chosen GPU, before any CUDA init.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    # ROBODOJO_CACHE moves Kit's, the GL shader and warp's kernel caches (under ~ by default) to a
    # data disk; each is a few hundred MB and grows with new shaders and kernels.
    cache = os.environ.get("ROBODOJO_CACHE")
    if cache:
        os.environ.setdefault("XDG_CACHE_HOME", f"{cache}/xdg")
        os.environ.setdefault("__GL_SHADER_DISK_CACHE_PATH", f"{cache}/nv")
        os.environ.setdefault("WARP_CACHE_PATH", f"{cache}/warp")
    root = sim.robodojo_root()
    tasks = sim.task_names(root)
    if args.task not in tasks:
        raise SystemExit(f"unknown RoboDojo task {args.task!r}; have {tasks}")
    layouts = sim.layout_count(root, args.task, args.eval_seed)
    if not layouts:
        raise SystemExit(
            f"no eval layouts for {args.task} under {root}/Assets/Eval_Layout (download the assets)"
        )
    if not 0 <= args.seed < layouts:
        raise SystemExit(
            f"--seed must be a layout id in [0, {layouts}) for {args.task}"
        )

    app = sim.launch_isaac()
    # A failure after the app is up must not reach SimulationApp.close(), which can swallow the
    # traceback and exit 0: print it and leave hard.
    try:
        t0 = time.monotonic()
        env = sim.make_env(
            app, root, args.task, eval_seed=args.eval_seed, depth=not args.no_depth
        )
        _, eval_num = sim.build_config(root, args.task, eval_seed=args.eval_seed)
        facade = RobodojoEnvFacade(
            app=app,
            env=env,
            meta={
                "task": args.task,
                "seed": args.seed,
                "eval_seed": args.eval_seed,
                "dimension": sim.dimension(args.task),
                "layouts": layouts,
                "eval_num": eval_num,
                "robot": "dual_arx_x5",
                "env_cfg_type": sim.ENV_CFG_TYPE,
                "control_hz": 25,
                "step_m": STEP_M,
                "depth": not args.no_depth,
                "cuda_device": args.cuda_device,
                "root": str(root),
            },
        )
        _, info = facade.reset()
        print(
            f"[robodojo-env] {args.task} layout {args.seed} ready in {time.monotonic() - t0:.1f}s: "
            f"{info['instruction']!r}{' ' + info['error'] if 'error' in info else ''}",
            flush=True,
        )
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
