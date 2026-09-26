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

"""RPC server wrapping one robosuite 1.5 task (CaP-X's seven, ./tasks.py) with Pandas under
the OSC_POSE controller.

Motion is a closed-loop Cartesian servo (``env.move_to`` / ``env.move_delta``): every control
step commands the clipped position (and orientation) error through OSC_POSE, until the TCP is
within tolerance, the step budget runs out or ``stop`` arrives. The server refuses a target
beyond the per-call travel cap, outside the workspace box or below the z floor before anything
moves (``tasks.check_move``). The gripper command is held per arm across calls. Success is
robosuite's ``_check_success`` (Restack adds CaP-X's off-table rule), latched at its first step
(``success_once``); the episode keeps stepping after it.

Observation: the task camera (``robot0_robotview``, or CaP-X's overhead ``agentview`` on the
two-arm tasks) and the wrist camera, rendered on demand at 512 px, upright, as RGB and metric
depth with the calibration robosuite reports for the upright image, plus the arms' joint state
and TCP poses. Object poses leave the server only through ``env.ground_truth_poses`` (pi's
``--privileged``): ``env.raw_obs`` is the robots' state alone, CaP-X's ``cube_poses`` /
``nut_poses`` are not exposed, and the handover task renders no instance segmentation.

The primitive registry (``code.api``, ./primitives.py, components/code_api.py) declares the
same facade methods for a code-as-policy caller: ``move_to`` / ``move_delta`` / ``set_gripper``
step the same env under the same limits and stop generation as pi's tools; ``segment`` needs
``--sam3``, ``preview_reach`` and the reach check before a move need ``--ik``, ``plan_grasp`` and
friends a grasp server (``--graspnet`` ...; utils/grasp.py).
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import random
import sys
from typing import Any

import numpy as np

from pi_embodied_services.components.code_api import register_code_api
from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.robots.robosuite import tasks
from pi_embodied_services.robots.robosuite.primitives import ROBOSUITE_PRIMITIVES
from pi_embodied_services.utils import ground_truth, reach
from pi_embodied_services.utils.grasp import (
    GraspPlanner,
    add_grasp_arguments,
    urls_from_args,
)
from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.rpc.main_thread_serve import MainThreadServeMixin

# MuJoCo env vars must be set before anything imports mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
assert "mujoco" not in sys.modules, (
    "mujoco must not be imported before MUJOCO_GL/PYOPENGL_PLATFORM are set"
)

logger = get_logger("env_server")

WRIST_CAMERA = "robot0_eye_in_hand"
#: Per-arm OSC_POSE action: [dx, dy, dz, ax, ay, az] then the gripper (1 close, -1 open).
ARM_DIM = 6
OPEN, CLOSE = -1.0, 1.0
#: The episode video's frame: the task camera at this size, every ``video_every`` servo steps.
VIDEO_SIZE = 256
#: Image size of ``get_observation`` / ``segment`` / ``back_project`` (the code primitives).
CODE_RES = 512
#: The ik service's robot model for the Panda's grip site (components/ik_server.py ROBOTS).
IK_ROBOT = "panda_libero"


def rotvec_of(rot: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix (angle in [0, pi])."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(rot).as_rotvec()


def mat_of_rotvec(v) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(np.asarray(v, dtype=np.float64).reshape(3)).as_matrix()


def mat_of_quat_xyzw(q) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(np.asarray(q, dtype=np.float64).reshape(4)).as_matrix()


def make_restack_class():
    """CaP-X's Restack scene as a robosuite Stack subclass (robosuite_cubes_restack.py): both
    cubes 4 cm (RESTACK_CUBE_HALF), the green cube placed on the red one, 1 cm toward -x."""
    from robosuite.environments.manipulation.stack import Stack
    from robosuite.models.objects import BoxObject
    from robosuite.models.tasks import ManipulationTask
    from robosuite.utils.placement_samplers import UniformRandomSampler

    class StackedSampler(UniformRandomSampler):
        """The first object anywhere in the range, the second on top of it (CaP-X's
        StackedObjectRandomSampler, without the general-case tail it never reaches)."""

        def sample(self, fixtures=None, reference=None, on_top=True):
            placed = {} if fixtures is None else dict(fixtures)
            first, second = self.mujoco_objects
            base = np.asarray(self.reference_pos, dtype=np.float64)
            x = self._sample_x(first.horizontal_radius) + base[0]
            y = self._sample_y(first.horizontal_radius) + base[1]
            z = self.z_offset + base[2] - first.bottom_offset[-1]
            placed[first.name] = ((x, y, z), self._sample_quat(), first)
            gap = max(float(self.z_offset), 0.01)
            z2 = z + first.top_offset[-1] - second.bottom_offset[-1] + gap
            placed[second.name] = ((x - 0.01, y, z2), self._sample_quat(), second)
            return placed

    class RestackStack(Stack):
        def _load_model(self):
            super()._load_model()
            half = tasks.RESTACK_CUBE_HALF
            self.cubeA = BoxObject(
                name="cubeA",
                size=[half] * 3,
                rgba=[1, 0, 0, 1],
                material=self.cubeA.material,
            )
            self.cubeB = BoxObject(
                name="cubeB",
                size=[half] * 3,
                rgba=[0, 1, 0, 1],
                material=self.cubeB.material,
            )
            cubes = [self.cubeA, self.cubeB]
            self.placement_initializer = StackedSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.12, 0.16],
                y_range=[-0.12, 0.12],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
            self.model = ManipulationTask(
                mujoco_arena=self.model.mujoco_arena,
                mujoco_robots=[robot.robot_model for robot in self.robots],
                mujoco_objects=cubes,
            )

    return RestackStack


class RobosuiteEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """One robosuite env; every call runs on the main thread (MuJoCo EGL)."""

    SERVICE_NAME = "robosuite-env"

    def __init__(
        self,
        *,
        task: str,
        seed: int,
        camera_size: int = 512,
        max_episode_steps: int = 10000,
        max_move_m: float = 0.3,
        settle_steps: int = 20,
        video_every: int = 4,
        sam3: str | None = None,
        ik_reach: reach.ReachPreview | None = None,
        grasp: dict | None = None,
    ):
        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config

        if task not in tasks.TASKS:
            raise ValueError(f"unknown task {task!r}: {list(tasks.TASKS)}")
        self._task_name = task
        self._task = t = tasks.TASKS[task]
        self._seed = int(seed)
        self._size = int(camera_size)
        self._max_steps = int(max_episode_steps)
        self._max_move = float(max_move_m)
        self._settle = int(settle_steps)
        self._video_every = max(1, int(video_every))
        n = len(t.arms)
        cc = load_composite_controller_config(controller="BASIC", robot="Panda")
        common = dict(
            robots=["Panda"] * n,
            controller_configs=[cc] * n if n > 1 else cc,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,  # rendered on demand (render_camera, the video frames)
            use_object_obs=True,
            camera_names=[t.camera],
            camera_heights=self._size,
            camera_widths=self._size,
            horizon=self._max_steps,
            ignore_done=True,  # success does not end the episode; the client tracks steps
            renderer="mujoco",
            **t.kwargs,
        )
        if t.restack:
            self._env = make_restack_class()(**common)
        else:
            self._env = suite.make(t.env, **common)
        self._grip = {arm: OPEN for arm in t.arms}
        self._steps = 0
        self._success_step: int | None = None
        self._home: dict[str, np.ndarray] = {}
        self._closed = False
        self._cameras = {"agentview": t.camera, "wrist": WRIST_CAMERA}
        # --sam3: the `segment` primitive; --ik: env.preview_reach and the reach check before a
        # move (utils/reach.py); --graspnet & co: env.plan_grasp and friends (utils/grasp.py).
        self._sam3_url = sam3
        self._sam3 = None
        self._reach = ik_reach
        self._grasp = GraspPlanner.from_args(
            self._view,
            cameras=["agentview", "wrist"],
            sam3=sam3,
            eef_pose=lambda arm: self._eef(self._arm_index(arm)),
            wrist_camera="wrist",
            **(grasp or {}),
        )
        # The video frames of the motion call in progress.
        self._motion_frames: list[np.ndarray] = []
        self._meta: dict[str, Any] = {
            "task": task,
            "seed": self._seed,
            "env": t.env,
            "arms": list(t.arms),
            "gripper": t.gripper,
            "action_dim": int(self._env.action_dim),
            "controller": "OSC_POSE",
            "camera": t.camera,
            "wrist_camera": WRIST_CAMERA,
            "camera_size": self._size,
            "max_episode_steps": self._max_steps,
            "max_move_m": self._max_move,
            "language": t.language,
            "capabilities": {
                "segment": bool(sam3),
                "reach": ik_reach is not None,
                **(self._grasp.capabilities() if self._grasp else {}),
            },
        }
        super().__init__()

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc.update(
            {
                "env.raw_obs": self.raw_obs,
                "env.state": self.state,
                "env.move_to": self.move_to,
                "env.move_delta": self.move_delta,
                "env.set_gripper": self.set_gripper,
                "env.ground_truth_poses": self.ground_truth_poses,
                "env.preview_reach": self.preview_reach,
            }
        )
        self._readonly_methods.update(
            {
                "env.state",
                "env.raw_obs",
                "env.preview_reach",
                "env.get_state",
                "env.get_observation",
                "env.back_project",
                "env.segment",
            }
        )
        self._rpc.update(
            {
                "env.get_state": self.get_state,
                "env.get_observation": self.get_observation,
                "env.back_project": self.back_project,
                "env.segment": self.segment,
            }
        )
        grasp = getattr(self, "_grasp", None)
        if grasp is not None:
            grasp.install(self, mutating=GraspPlanner.MUTATING + ("env.move_to",))
        register_code_api(
            self, ROBOSUITE_PRIMITIVES + (grasp.primitives() if grasp else ())
        )

    # ---- frames and limits ----

    @property
    def _sim(self):
        return self._env.sim

    @property
    def _table_z(self) -> float:
        return float(self._env.table_offset[2])

    def _arm_index(self, arm: str | None) -> int:
        arms = self._task.arms
        if arm is None:
            if len(arms) > 1:
                raise ValueError(f"this task has two arms: pass arm={list(arms)}")
            return 0
        if arm not in arms:
            raise ValueError(f"unknown arm {arm!r}: {list(arms)}")
        return arms.index(arm)

    def _base(self, i: int) -> tuple[int, np.ndarray]:
        """Body id and world rotation of arm ``i``'s base: OSC_POSE reads its deltas in the
        base frame, and the ik service wants targets there."""
        robot = self._env.robots[i]
        body = self._sim.model.body_name2id(robot.robot_model.root_body)
        return body, np.asarray(self._sim.data.body_xmat[body]).reshape(3, 3)

    def _obs(self) -> dict:
        return self._env._get_observations()

    def _eef(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        obs = self._obs()
        return (
            np.asarray(obs[f"robot{i}_eef_pos"], dtype=np.float64).reshape(3),
            np.asarray(obs[f"robot{i}_eef_quat"], dtype=np.float64).reshape(4),
        )

    def _workspace(self) -> dict:
        """The world-frame box a move may end in: the table's footprint plus a margin, widened to
        0.15 m around each arm's start (the two-arm tables do not reach under the arms), from
        ``z_floor_m`` above the table top to 0.6 m above it."""
        size = np.asarray(self._env.table_full_size, dtype=np.float64)
        off = np.asarray(self._env.table_offset, dtype=np.float64)
        margin = 0.1
        lo = off[:2] - size[:2] / 2 - margin
        hi = off[:2] + size[:2] / 2 + margin
        for i in range(len(self._task.arms)):
            start = self._home.get(self._task.arms[i], self._eef(i)[0])[:2]
            lo = np.minimum(lo, start - 0.15)
            hi = np.maximum(hi, start + 0.15)
        return {
            "box": [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])],
            "z_floor": float(off[2] + self._task.z_floor_m),
            "z_ceiling": float(off[2] + 0.6),
            "table_z": float(off[2]),
        }

    def _action(self, parts: dict[int, np.ndarray]) -> np.ndarray:
        """The composite action: each arm's 6-D OSC delta (zeros = hold) and held gripper command."""
        out = []
        for i, arm in enumerate(self._task.arms):
            out.append(parts.get(i, np.zeros(ARM_DIM)))
            if self._task.gripper:
                out.append([self._grip[arm]])
        a = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in out])
        assert a.shape[0] == self._env.action_dim, (a.shape, self._env.action_dim)
        return a

    def _success(self) -> bool:
        ok = bool(self._env._check_success())
        if self._task.restack:
            e = self._env
            za = float(self._sim.data.body_xpos[e.cubeA_body_id][2]) - self._table_z
            zb = float(self._sim.data.body_xpos[e.cubeB_body_id][2]) - self._table_z
            ok = tasks.restack_success(ok, (za, zb))
        return ok

    def _step(self, action: np.ndarray) -> None:
        """One control step; success latched."""
        self._env.step(action)
        self._steps += 1
        self._success_step = tasks.latch(
            self._success_step, self._success(), self._steps
        )

    def _render(self, camera: str, size: int, depth: bool = False):
        """Upright rgb (and metric depth) of a camera: robosuite renders bottom-up, its
        calibration (``get_camera_meta``) is for the flipped image."""
        out = self._sim.render(camera_name=camera, width=size, height=size, depth=depth)
        if not depth:
            return np.ascontiguousarray(out[::-1])
        import robosuite.utils.camera_utils as CU

        rgb, d = out
        d = np.clip(np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        metric = CU.get_real_depth_map(self._sim, d[..., None] if d.ndim == 2 else d)
        return np.ascontiguousarray(rgb[::-1]), np.ascontiguousarray(
            metric[..., 0][::-1].astype(np.float32)
        )

    def _video_frame(self) -> None:
        """One task-camera frame for the episode video."""
        self._motion_frames.append(self._render(self._cameras["agentview"], VIDEO_SIZE))

    def _robot_state(self) -> dict:
        obs = self._obs()
        out: dict[str, Any] = {}
        for i, arm in enumerate(self._task.arms):
            for key in ("eef_pos", "eef_quat", "joint_pos"):
                out[f"{arm}_{key}"] = np.asarray(
                    obs[f"robot{i}_{key}"], dtype=np.float32
                )
            if self._task.gripper:
                q = np.asarray(obs[f"robot{i}_gripper_qpos"], dtype=np.float64)
                out[f"{arm}_gripper_qpos"] = q.astype(np.float32)
                out[f"{arm}_gripper_width"] = float(np.abs(q).sum())
                out[f"{arm}_gripper_command"] = (
                    "close" if self._grip[arm] > 0 else "open"
                )
        out["success"] = self._success_step is not None
        out["success_step"] = self._success_step
        out["truncated"] = self._steps >= self._max_steps
        out["env_steps"] = self._steps
        return out

    def _pack(self) -> dict:
        return {
            "agentview": self._render(self._cameras["agentview"], self._size),
            "wrist": self._render(self._cameras["wrist"], self._size),
            **self._robot_state(),
        }

    def _place_overhead_camera(self) -> None:
        """CaP-X's overhead view for the two-arm tasks (re-applied after every reset)."""
        if self._task.camera != "agentview":
            return
        cam = self._sim.model.camera_name2id("agentview")
        self._sim.model.cam_pos[cam] = tasks.OVERHEAD_CAMERA["pos"]
        self._sim.model.cam_quat[cam] = tasks.OVERHEAD_CAMERA["quat_wxyz"]
        self._sim.forward()

    # ---- gym-like surface ----

    def reset(self):
        """Restore the episode's initial scene: the same seed every time, so a reset is
        repeatable (robosuite 1.5's placement samplers draw from the global numpy RNG, which is
        reseeded here), then settle with the gripper open."""
        self._env.rng = np.random.default_rng(self._seed)
        np.random.seed(self._seed % 2**32)
        random.seed(self._seed)
        self._env.reset()
        self._place_overhead_camera()
        self._grip = {arm: OPEN for arm in self._task.arms}
        self._steps = 0
        self._success_step = None
        for _ in range(self._settle):
            self._env.step(self._action({}))
        self._home = {arm: self._eef(i)[0] for i, arm in enumerate(self._task.arms)}
        self._success_step = tasks.latch(None, self._success(), 0)
        return self._pack(), {"language": self._task.language}

    def step(self, action):
        """One raw composite action (per arm: 6 OSC_POSE deltas in [-1, 1] and the gripper)."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != self._env.action_dim:
            raise ValueError(
                f"action has {a.shape[0]} values; {self._env.action_dim} needed"
            )
        if self._task.gripper:
            for i, arm in enumerate(self._task.arms):
                self._grip[arm] = CLOSE if a[(i + 1) * (ARM_DIM + 1) - 1] > 0 else OPEN
        self._step(a)
        s = self._robot_state()
        return self._pack(), float(self._env.reward(a)), s["success"], s["truncated"], s

    def chunk_step(self, actions, *, return_all_frames: bool = False):
        """Raw actions [N, action_dim] in one call; stops at ``stop``. One entry per executed action."""
        frames, rews, terms, truncs = [], [], [], []
        info: dict = {}
        for a in np.asarray(actions, dtype=np.float64).reshape(
            -1, self._env.action_dim
        ):
            if self.stop_requested():
                info["cancelled"] = True
                break
            obs, rew, term, trunc, info = self.step(a)
            frames.append(obs)
            rews.append(rew)
            terms.append(term)
            truncs.append(trunc)
        if not frames:
            obs = self._pack()
            return ([obs] if return_all_frames else obs), [], [], [], info
        return (
            frames if return_all_frames else frames[-1],
            np.asarray(rews, dtype=np.float32),
            np.asarray(terms, dtype=bool),
            np.asarray(truncs, dtype=bool),
            info,
        )

    # ---- motion ----

    def _set_grip(self, arm: str, command: str) -> None:
        if not self._task.gripper:
            raise ValueError(f"{self._task_name}'s wiping gripper has no fingers")
        if command not in ("open", "close"):
            raise ValueError(f"gripper must be 'open' or 'close', got {command!r}")
        self._grip[arm] = CLOSE if command == "close" else OPEN

    def move_to(
        self,
        target_xyz,
        *,
        arm: str | None = None,
        quat_xyzw=None,
        rotvec=None,
        gripper: str | None = None,
        tol_m: float = 0.005,
        tol_rad: float = 0.03,
        step_m: float = 0.02,
        step_rad: float = 0.2,
        max_steps: int = 100,
    ) -> dict:
        """Servo one arm's TCP to a world position, holding its orientation.

        Args:
            target_xyz: [x, y, z] in metres, world frame (+x away from robot0, +y to its left,
                +z up; the table top is at ``get_state()["table_z"]``).
            arm: "robot0" or "robot1" on the two-arm tasks; omit on one arm.
            quat_xyzw: an absolute world orientation to reach as well, or None to keep the
                current one.
            rotvec: instead of quat_xyzw, a world-frame turn (axis x angle, rad) applied to the
                current orientation, e.g. [0, 0, 0.3] turns the wrist 0.3 rad about vertical.
            gripper: "open" / "close" sets the held gripper command first; None keeps it.
            tol_m, tol_rad: the servo stops within these of the target.
            max_steps: control-step budget (each moves at most ``step_m`` / ``step_rad``).

        Returns:
            dict with ``obs`` (the new observation) and ``info``: ``ok`` (target reached),
            ``final_eef_pos``, ``final_dist_m``, ``steps_used``, ``cancelled`` (a stop arrived).
            Raises, without moving, for a target more than the per-call cap away, outside the
            workspace box, below the z floor, or (with an ik service) unreachable.

        Example:
            >>> move_to([0.05, 0.02, 0.95], gripper="open")
            >>> move_to([0.05, 0.02, 0.83]); set_gripper(True)
        """
        i = self._arm_index(arm)
        name = self._task.arms[i]
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        pos, quat = self._eef(i)
        ws = self._workspace()
        tasks.check_move(
            tuple(pos),
            tuple(target),
            max_move_m=self._max_move,
            box=tuple(ws["box"]),
            z_floor=ws["z_floor"],
            z_ceiling=ws["z_ceiling"],
        )
        if quat_xyzw is not None and rotvec is not None:
            raise ValueError("give quat_xyzw or rotvec, not both")
        rot_target = None
        if quat_xyzw is not None:
            rot_target = mat_of_quat_xyzw(quat_xyzw)
        elif rotvec is not None:
            rot_target = mat_of_rotvec(rotvec) @ mat_of_quat_xyzw(quat)
        if self._reach is not None:
            reach.require_reachable(
                self.preview_reach(
                    target,
                    None if rot_target is None else self._quat_of(rot_target),
                    arm=name,
                ),
                "move_to",
            )
        if gripper is not None:
            self._set_grip(name, gripper)
        self._motion_frames = []
        _, base_rot = self._base(i)
        base_t = base_rot.T
        info: dict[str, Any] = {"ok": False, "cancelled": False}
        steps = 0
        for steps in range(int(max_steps) + 1):
            pos, quat = self._eef(i)
            err = target - pos
            rot_err = np.zeros(3)
            if rot_target is not None:
                rot_err = rotvec_of(rot_target @ mat_of_quat_xyzw(quat).T)
            if np.linalg.norm(err) < tol_m and np.linalg.norm(rot_err) < tol_rad:
                info["ok"] = True
                break
            if steps == int(max_steps):
                break
            if self.stop_requested():
                info["cancelled"] = True
                break
            d = err * min(1.0, step_m / max(np.linalg.norm(err), 1e-9))
            r = rot_err * min(1.0, step_rad / max(np.linalg.norm(rot_err), 1e-9))
            part = np.concatenate(
                [
                    np.clip(base_t @ d / tasks.OSC_POS_MAX_M, -1, 1),
                    np.clip(base_t @ r / tasks.OSC_ROT_MAX_RAD, -1, 1),
                ]
            )
            self._step(self._action({i: part}))
            if self._steps % self._video_every == 0:
                self._video_frame()
        pos, quat = self._eef(i)
        info.update(
            {
                "arm": name,
                "target_xyz": [round(float(v), 4) for v in target],
                "final_eef_pos": [round(float(v), 4) for v in pos],
                "final_dist_m": round(float(np.linalg.norm(target - pos)), 4),
                "steps_used": steps,
                "success": self._success_step is not None,
            }
        )
        if rot_target is not None:
            info["final_rot_err_rad"] = round(
                float(np.linalg.norm(rotvec_of(rot_target @ mat_of_quat_xyzw(quat).T))),
                4,
            )
        return self._motion_result(info)

    def move_delta(
        self,
        delta_xyz,
        *,
        arm: str | None = None,
        quat_xyzw=None,
        rotvec=None,
        gripper: str | None = None,
        tol_m: float = 0.005,
        tol_rad: float = 0.03,
        step_m: float = 0.02,
        step_rad: float = 0.2,
        max_steps: int = 100,
    ) -> dict:
        """Servo one arm's TCP by a world-frame offset, holding its orientation.

        Args:
            delta_xyz: [dx, dy, dz] in metres (world frame: +x away from robot0, +y to its
                left, +z up); at most the per-call cap.
            arm: "robot0" or "robot1" on the two-arm tasks; omit on one arm.
            The rest as `move_to`.

        Returns:
            `move_to`'s result.

        Example:
            >>> move_delta([0, 0, 0.1])            # lift 10 cm
            >>> move_delta([0, 0, 0], rotvec=[0, 0, 0.5])   # turn the wrist 0.5 rad
        """
        i = self._arm_index(arm)
        pos, _ = self._eef(i)
        return self.move_to(
            pos + np.asarray(delta_xyz, dtype=np.float64).reshape(3),
            arm=arm,
            quat_xyzw=quat_xyzw,
            rotvec=rotvec,
            gripper=gripper,
            tol_m=tol_m,
            tol_rad=tol_rad,
            step_m=step_m,
            step_rad=step_rad,
            max_steps=max_steps,
        )

    def set_gripper(
        self, close: bool | str, *, arm: str | None = None, steps: int = 15
    ) -> dict:
        """Hold the arm and open or close its gripper; the command stays in force for later moves.

        Args:
            close: True (or "close") closes and holds, False (or "open") opens.
            arm: "robot0" or "robot1" on the two-arm tasks; omit on one arm.
            steps: control steps to drive the fingers (they stop early once they no longer
                move, e.g. on a grasped object).

        Returns:
            dict with ``obs`` and ``info``: ``gripper`` ("open" / "close"), ``gripper_width`` (m,
            sum of the finger joints: about 0.08 open, near 0 closed on nothing), ``steps_used``.

        Example:
            >>> set_gripper(True); move_delta([0, 0, 0.1])
        """
        i = self._arm_index(arm)
        name = self._task.arms[i]
        command = close if isinstance(close, str) else ("close" if close else "open")
        self._set_grip(name, command)
        self._motion_frames = []
        prev = None
        used = 0
        cancelled = False
        for used in range(1, int(steps) + 1):
            if self.stop_requested():
                cancelled = True
                break
            self._step(self._action({}))
            if self._steps % self._video_every == 0:
                self._video_frame()
            width = self._robot_state()[f"{name}_gripper_width"]
            if used > 3 and prev is not None and abs(width - prev) < 5e-4:
                break
            prev = width
        state = self._robot_state()
        return self._motion_result(
            {
                "ok": not cancelled,
                "arm": name,
                "gripper": command,
                "gripper_width": round(state[f"{name}_gripper_width"], 4),
                "steps_used": used,
                "cancelled": cancelled,
                "success": state["success"],
            }
        )

    def _motion_result(self, info: dict) -> dict:
        """A motion call's answer: the new observation and the video frames it produced."""
        return {
            "obs": self._pack(),
            "info": {**info, "frames": list(self._motion_frames)},
        }

    # ---- read-only ----

    def raw_obs(self) -> dict:
        """The robots' state alone (robosuite's ``robot*_`` observations): object poses are
        privileged (``ground_truth_poses``)."""
        return {
            k: np.asarray(v) for k, v in self._obs().items() if k.startswith("robot")
        }

    def state(self) -> dict:
        """TCP poses, gripper widths, joint positions and the success flags (no stepping)."""
        return {
            **self._robot_state(),
            "home_eef_pos": dict(self._home),
            "table_z": self._table_z,
        }

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 512,
        width: int = 512,
        depth=False,
    ):
        """Upright rgb uint8[H,W,3] of ``agentview`` (the task camera) or ``wrist``, or
        ``[rgb, depth]`` with metric depth float32[H,W]."""
        if height != width:
            raise ValueError("square renders only")
        return self._render(self._cameras[camera_name], int(height), bool(depth))

    def get_camera_meta(
        self, camera_name: str = "agentview", height: int = 512, width: int = 512
    ) -> dict:
        """OpenCV intrinsics and camera-to-world extrinsic (robosuite's, for the upright
        image at this resolution); depth from ``render_camera`` is already metric."""
        import robosuite.utils.camera_utils as CU

        cam = self._cameras[camera_name]
        return {
            "camera_name": cam,
            "height": int(height),
            "width": int(width),
            "intrinsic_K": np.asarray(
                CU.get_camera_intrinsic_matrix(self._sim, cam, int(height), int(width)),
                dtype=np.float64,
            ),
            "extrinsic_cam2world": np.asarray(
                CU.get_camera_extrinsic_matrix(self._sim, cam), dtype=np.float64
            ),
            "depth_metric": True,
        }

    def get_task_language(self) -> str:
        return self._task.language

    def get_env_meta(self) -> dict:
        return {**self._meta, **self._workspace()}

    def _objects(self) -> dict[str, int]:
        """The scene's objects by MuJoCo body id: robosuite's task objects (cubes, nuts, pot,
        hammer), the nut assembly's pegs and Wipe's dirt markers."""
        m = self._sim.model
        out: dict[str, int] = {}
        for obj in self._env.model.mujoco_objects:
            out[obj.name] = m.body_name2id(obj.root_body)
        for attr in ("peg1_body_id", "peg2_body_id"):
            if hasattr(self._env, attr):
                out[attr[:-8]] = int(getattr(self._env, attr))
        for marker in getattr(self._env.model.mujoco_arena, "markers", []):
            out[marker.name] = m.body_name2id(marker.root_body)
        return out

    def ground_truth_poses(self, names=None) -> dict:
        """World poses of the scene's objects (the simulator's ground truth; privileged).

        Args:
            names: object names to report, or None for all. Unknown names raise, listing them.

        Returns:
            ``{"frame": "world", "poses": {name: {"pos": [x, y, z], "quat_xyzw": [...]}}}`` in
            metres; the two-arm tasks add the grasp sites ``pot_handle0`` / ``pot_handle1`` and
            ``hammer_handle``.

        Example:
            >>> cube = ground_truth_poses(["cube"])["poses"]["cube"]["pos"]
        """
        objects = self._objects()
        poses = ground_truth.mujoco_body_poses(self._sim, objects)
        e = self._env
        # The two-arm tasks' grasp points are sites, not bodies.
        for site, key, body in (
            ("handle0_site_id", "pot_handle0", "pot"),
            ("handle1_site_id", "pot_handle1", "pot"),
            ("hammer_handle_site_id", "hammer_handle", "hammer"),
        ):
            if hasattr(e, site) and body in objects:
                poses[key] = ground_truth.pose(
                    self._sim.data.site_xpos[getattr(e, site)],
                    self._sim.data.body_xquat[objects[body]],
                )
        return ground_truth.respond(poses, names)

    @staticmethod
    def _quat_of(rot: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        return Rotation.from_matrix(rot).as_quat()

    def preview_reach(self, pos, quat_xyzw=None, *, arm: str | None = None) -> dict:
        """Whether an arm can reach a world position without moving: IK from its current joints
        by the ik service (``--ik``); the sim is not touched.

        Args:
            pos: target [x, y, z] in metres (world frame), the point `move_to` would take.
            quat_xyzw: target orientation; None (default) keeps the current one, as `move_to`.
            arm: "robot0" or "robot1" on the two-arm tasks; omit on one arm.

        Returns:
            dict with ``status`` ("reachable", "unreachable" or "unknown" when no ik service
            answers), ``reachable``, ``q`` (the IK solution), ``position_err``,
            ``orientation_err`` and ``message``. `move_to` refuses an "unreachable" target.

        Example:
            >>> if preview_reach([0.1, 0.0, 0.85])["status"] == "unreachable": ...
        """
        if self._reach is None:
            return reach.no_service()
        i = self._arm_index(arm)
        body, _ = self._base(i)
        base = ground_truth.mujoco_body_poses(self._sim, {"base": body})["base"]
        obs = self._obs()
        return self._reach.preview(
            obs[f"robot{i}_joint_pos"],
            pos,
            obs[f"robot{i}_eef_quat"] if quat_xyzw is None else quat_xyzw,
            base_pose=base,
        )

    # ---- the code primitives (./primitives.py) ----------------------------------------------

    def get_state(self) -> dict:
        """Proprioception, no images.

        Returns:
            dict with, per arm ``<arm>_eef_pos`` [x, y, z] (m, world), ``<arm>_eef_quat``
            (xyzw), ``<arm>_joint_pos`` (7), ``<arm>_gripper_width`` (m; about 0.08 open, near 0
            closed on nothing) and ``<arm>_gripper_command``; ``success`` (robosuite's check,
            latched), ``env_steps``, ``table_z`` (the table top, m) and ``home_eef_pos``.

        Example:
            >>> z = get_state()["robot0_eef_pos"][2]
        """
        s = self.state()
        return {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()
        }

    def _view(self, camera: str) -> dict:
        """One camera, upright: rgb uint8[S,S,3], depth float32[S,S] in metres, K, cam2world."""
        rgb, depth = self._render(self._cameras[camera], CODE_RES, depth=True)
        meta = self.get_camera_meta(camera, CODE_RES, CODE_RES)
        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsic_K": meta["intrinsic_K"],
            "extrinsic_cam2world": meta["extrinsic_cam2world"],
        }

    def get_observation(self) -> dict:
        """The current camera images with calibration, plus the state of `get_state`.

        Returns:
            dict with ``agentview`` (the task camera) and ``wrist``, each ``{"rgb": uint8[512,
            512, 3], "depth": float32[512, 512] (metres), "intrinsic_K": float64[3, 3],
            "extrinsic_cam2world": float64[4, 4]}``, and the `get_state` fields. Pixel (row,
            col) back-projects as ``x = (col - cx) * z / fx``, ``y = (row - cy) * z / fy`` in
            the camera frame, then ``extrinsic_cam2world @ [x, y, z, 1]``.

        Example:
            >>> obs = get_observation()
            >>> depth = obs["agentview"]["depth"]; K = obs["agentview"]["intrinsic_K"]
        """
        out = {camera: self._view(camera) for camera in ("agentview", "wrist")}
        out.update(self.get_state())
        return out

    @staticmethod
    def _world_xyz(view: dict, row: int, col: int) -> np.ndarray | None:
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
            prompt: what to segment, e.g. "red cube".
            camera: "agentview" (default) or "wrist".
            min_score: SAM3 score threshold (default 0.2).

        Returns:
            dict with ``found`` (bool); when found: ``score``, ``box`` [x1, y1, x2, y2] (pixels),
            ``mask`` bool[512, 512], ``n_pixels``, ``centroid_rowcol`` [row, col] and
            ``world_xyz`` (median over the mask's pixels with depth, or None when too few).

        Example:
            >>> seg = segment("red cube")
            >>> if seg["found"]: xyz = seg["world_xyz"]
        """
        from PIL import Image

        from pi_embodied_services.utils.rpc.http_rpc import HttpRpcClient

        if not self._sam3_url:
            raise RuntimeError(
                "segment needs a SAM3 server (start the env server with --sam3)"
            )
        if self._sam3 is None:
            self._sam3 = HttpRpcClient(self._sam3_url)
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
        mask = np.asarray(
            Image.open(io.BytesIO(base64.b64decode(res["mask_png_base64"])))
        )
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
            "score": round(float(res.get("score", 0.0)), 3),
            "box": res.get("box"),
            "mask": mask,
            "n_pixels": int(mask.sum()),
            "centroid_rowcol": [int(np.median(rows)), int(np.median(cols))],
            "world_xyz": None,
        }
        if len(pts) >= 10:
            out["world_xyz"] = [round(float(v), 4) for v in np.median(pts, axis=0)]
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--task", choices=tasks.TASK_NAMES, default="Lift")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--camera-size", type=int, default=512)
    p.add_argument("--max-episode-steps", type=int, default=10000)
    p.add_argument("--max-move", type=float, default=0.3, help="per-call travel cap, m")
    p.add_argument(
        "--sam3",
        type=str,
        default=None,
        help="SAM3 server URL; adds the `segment` primitive to code mode's high tier",
    )
    reach.add_ik_argument(p)
    add_grasp_arguments(p)
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU for MuJoCo EGL rendering (physical CUDA ordinal)",
    )
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    args = p.parse_args()

    if args.cuda_device is not None:
        # robosuite asserts MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES when the latter is
        # set, assuming the EGL order is the CUDA order; pin the EGL device directly instead.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        from pi_embodied_services.utils.egl import configure_egl_device

        configure_egl_device(args.cuda_device)

    facade = RobosuiteEnvFacade(
        task=args.task,
        seed=args.seed,
        camera_size=args.camera_size,
        max_episode_steps=args.max_episode_steps,
        max_move_m=args.max_move,
        sam3=args.sam3,
        ik_reach=reach.reach_from_args(args, IK_ROBOT),
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
