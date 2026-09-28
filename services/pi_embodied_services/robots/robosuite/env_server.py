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

The primitive manifest (packages/embodied/src/primitives/manifests/robosuite.json, read by
components/manifest.py for ``code.api``) declares the
same facade methods for a code-as-policy caller: ``move_to`` / ``move_delta`` / ``set_gripper``
step the same env under the same limits and stop generation as pi's tools; ``segment`` needs
``--sam3``, ``preview_reach`` and the reach check before a move need ``--ik``, ``plan_grasp`` and
friends a grasp server (``--contact-graspnet`` ...; utils/grasp.py).

Code mode (``code.run``, utils/code_exec.py ``CodeRunMixin``): a program calls those primitives
through the registry's resolve, in a sandboxed subprocess, so the server requires its RPC token
and refuses other business calls while a program runs. A primitive's reply to the program drops
the bulk a program does not need (a motion's observation images and video frames, which go to the
run's video instead); the run reports its control steps, the latched success and the new
observation (``_finish_run``).
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

from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.robots.robosuite import tasks
from pi_embodied_services.utils import ground_truth, reach
from pi_embodied_services.utils.code_exec import CodeRunMixin
from pi_embodied_services.utils.geometry import (
    GripGeometry,
    jaw_frame,
    mujoco_grip_state,
    quat_to_matrix,
)
from pi_embodied_services.utils.grasp import (
    GraspPlanner,
    add_grasp_arguments,
    urls_from_args,
)
from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.perception import (
    add_perception_arguments,
    install_perception,
)
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
#: The camera size of a recorded control step (a motion call's ``record``).
RECORD_SIZE = 256
#: Image size of ``get_observation`` / ``segment`` / ``back_project`` (the code primitives).
CODE_RES = 512
#: The ik service's robot model for the Panda's grip site (components/ik_server.py ROBOTS).
IK_ROBOT = "panda_libero"
#: Code mode: video frames one run hands back (halved, every other one kept, when full) and the
#: largest render a program may ask for.
CODE_MAX_FRAMES = 128
CODE_MAX_RENDER = 1024


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


#: Motion methods whose reply to a program is their info without frames and images.
MOTION_REPLIES = (
    "env.move_to",
    "env.move_delta",
    "env.set_gripper",
    "env.goto_pose",
    "env.home_pose",
    "env.open_gripper",
    "env.close_gripper",
    "env.move_to_joints",
    "env.move_along_trajectory",
)

#: panda_hand -> robosuite's grip site: half a turn about the hand's z (wxyz).
HAND_TO_SITE_WXYZ = np.array([0.0, 0.0, 0.0, 1.0])


def _unit_quat(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q / np.linalg.norm(q)


def _quat_mul(a, b) -> np.ndarray:
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


def _quat_matrix(q_wxyz) -> np.ndarray:
    w, x, y, z = _unit_quat(q_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _slerp(a, b, t: float) -> np.ndarray:
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        return _unit_quat(a + (b - a) * t)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * a + np.sin(t * th) * b) / np.sin(th)


class RobosuiteEnvFacade(CodeRunMixin, MainThreadServeMixin, BaseEnvFacade):
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
        geometry: bool = False,
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
        # move (utils/reach.py); --contact-graspnet & co: env.plan_grasp and friends (utils/grasp.py).
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
        # --geometry (one-arm tasks): point-cloud views, marked points and grip-site targets
        # (utils/geometry.py); pi executes a target through env.move_to with its quat_xyzw.
        if geometry and len(t.arms) != 1:
            raise ValueError(
                f"--geometry needs a one-arm task; {task} has {len(t.arms)}"
            )
        self._geometry = self._geometry_kit() if geometry else None
        # The video frames of the motion call in progress.
        self._motion_frames: list[np.ndarray] = []
        # Its recorded control steps (``record=True``), else None.
        self._record: list[dict] | None = None
        # Code mode: the control steps before the run, and the run's video frames.
        self._run_start = 0
        self._run_frames: list[np.ndarray] = []
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
                "geometry": geometry,
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
                # CaP-X's high tier (and its privileged variant) and joint-space parts.
                "env.get_object_pose": self.get_object_pose,
                "env.get_object_pose_privileged": self.get_object_pose_privileged,
                "env.sample_grasp_pose": self.sample_grasp_pose,
                "env.sample_grasp_pose_privileged": self.sample_grasp_pose_privileged,
                "env.goto_pose": self.goto_pose,
                "env.home_pose": self.home_pose,
                "env.open_gripper": self.open_gripper,
                "env.close_gripper": self.close_gripper,
                "env.solve_ik": self.solve_ik,
                "env.move_to_joints": self.move_to_joints,
                "env.traj_plan": self.traj_plan,
                "env.move_along_trajectory": self.move_along_trajectory,
            }
        )
        grasp = getattr(self, "_grasp", None)
        geometry = getattr(self, "_geometry", None)
        if geometry is not None:
            geometry.install(self)
        if grasp is not None:
            grasp.install(self, mutating=GraspPlanner.MUTATING + ("env.move_to",))
        # The primitives are packages/embodied/src/primitives/manifests/robosuite.json (with
        # pi); code.api, the programs' whitelist and the startup self-check come from it.
        self._manifest_code_run(
            "robosuite",
            have=self._has,
            move_m=self._code_move_m,
            check=self._code_check,
            reply=self._code_reply,
            begin=self._begin_run,
            finish=self._finish_run,
        )

    def _has(self, capability: str) -> bool:
        """What this server can serve of the manifest's ``requires``."""
        grasp = getattr(self, "_grasp", None)
        return {
            "sam3": bool(getattr(self, "_sam3_url", None)),
            "ik": getattr(self, "_reach", None) is not None,
            "grasp": grasp is not None,
            "place": grasp is not None and bool(grasp.capabilities().get("place")),
            "geometry": getattr(self, "_geometry", None) is not None,
            "unidepth": "env.enhance_depth" in self._rpc,
            "fingers": self._task.gripper,
            "privileged": True,
        }.get(capability, False)

    # ---- code mode (run_code) ----

    def _begin_run(self) -> None:
        self._run_start = self._steps
        self._run_frames = []

    def _finish_run(self) -> dict:
        """The run's effect for pi: control steps taken, the latched success, the new
        observation (the tools' ``obs``) and the run's video frames."""
        return {
            "steps": self._steps - self._run_start,
            "success": self._success_step is not None,
            "success_step": self._success_step,
            "obs": self._pack(),
            "frames": list(self._run_frames),
        }

    def _keep_frames(self, frames) -> None:
        for f in frames:
            if len(self._run_frames) >= CODE_MAX_FRAMES:
                self._run_frames = self._run_frames[::2]
            self._run_frames.append(f)

    def _code_reply(self, method: str, out: Any) -> Any:
        """What a program receives: a motion's info without its frames (they go to the run's
        video) and without the observation images; a raw step's robot state, not its images."""
        if method in MOTION_REPLIES and isinstance(out, dict) and "info" in out:
            info = dict(out["info"])
            self._keep_frames(info.pop("frames", []))
            return info
        if method == "env.step":
            self._keep_frames([self._render(self._cameras["agentview"], VIDEO_SIZE)])
            _obs, rew, success, truncated, state = out
            return {
                "reward": rew,
                "success": success,
                "truncated": truncated,
                "state": state,
            }
        if method == "env.segment" and isinstance(out, dict):
            # The overlay is the tool's picture; the program has the mask.
            return {k: v for k, v in out.items() if k != "overlay_png_base64"}
        return out

    def _code_move_m(self, method: str, kwargs: dict) -> float:
        """How far a program's call may move an arm (the run's translation cap)."""
        joint_or_pose = (
            "env.goto_pose",
            "env.home_pose",
            "env.move_to_joints",
            "env.move_along_trajectory",
        )
        i = self._arm_index(kwargs.get("arm")) if method in joint_or_pose else 0
        if method == "env.goto_pose":
            target = np.asarray(kwargs["position"], dtype=np.float64).reshape(3)
            return float(
                np.linalg.norm(target - self._eef(i)[0])
                + 2 * abs(float(kwargs.get("z_approach", 0.0) or 0.0))
            )
        if method == "env.home_pose":
            home = self._home.get(self._task.arms[i], self._eef(i)[0])
            return float(np.linalg.norm(home - self._eef(i)[0]))
        if method == "env.move_to_joints":
            return float(
                np.linalg.norm(self._fk(kwargs["joints"], i)[0] - self._eef(i)[0])
            )
        if method == "env.move_along_trajectory":
            here, total = self._eef(i)[0], 0.0
            for q in np.asarray(kwargs["trajectory"], dtype=np.float64)[:100]:
                p = self._fk(q, i)[0]
                total += float(np.linalg.norm(p - here))
                here = p
            return total
        if method == "env.move_to":
            target = np.asarray(kwargs["xyz"], dtype=np.float64).reshape(3)
            pos, _ = self._eef(self._arm_index(kwargs.get("arm")))
            return float(np.linalg.norm(target - pos))
        if method == "env.move_delta":
            d = np.asarray(kwargs["delta_xyz"], dtype=np.float64).reshape(3)
            return float(np.linalg.norm(d))
        if method == "env.step":
            a = np.asarray(kwargs["action"], dtype=np.float64).reshape(-1)
            per = ARM_DIM + (1 if self._task.gripper else 0)
            return float(
                sum(
                    np.linalg.norm(np.clip(a[k * per : k * per + 3], -1, 1))
                    * tasks.OSC_POS_MAX_M
                    for k in range(len(self._task.arms))
                )
            )
        return 0.0

    def _code_check(self, method: str, kwargs: dict) -> None:
        """Refuse a program's call that the run's wall clock could not bound."""
        if method == "env.render_camera":
            for k in ("height", "width"):
                if int(kwargs.get(k, 512)) > CODE_MAX_RENDER:
                    raise ValueError(f"render_camera {k} is at most {CODE_MAX_RENDER}")
        if method in ("env.move_to", "env.move_delta"):
            if int(kwargs.get("max_steps", 100)) > 400:
                raise ValueError("max_steps is at most 400 per call in code mode")
        if method == "env.set_gripper" and int(kwargs.get("steps", 15)) > 200:
            raise ValueError("steps is at most 200 per call in code mode")

    # ---- grip-site geometry (--geometry, utils/geometry.py) ----

    def _grip_mech(self) -> dict:
        out = mujoco_grip_state(self._sim)
        if isinstance(out.get("error"), str):
            raise RuntimeError(f"grip geometry: {out['error']}")
        return out

    def _geometry_kit(self) -> GripGeometry:
        """robot0's grip site: the tool frame is its ``eef_quat`` frame, turned once onto the
        MuJoCo grip site and its finger pads (jaw +X, approach +Z)."""
        jaw: dict[str, np.ndarray] = {}

        def frame() -> np.ndarray:
            mech = self._grip_mech()
            jaw["J"] = jaw_frame(mech["pads_local"])
            hand = quat_to_matrix(self._eef(0)[1])
            return hand.T @ np.asarray(mech["site_xmat"], dtype=np.float64) @ jaw["J"]

        def pads() -> np.ndarray:
            p = np.asarray(self._grip_mech()["pads_local"], dtype=np.float64)
            return p @ jaw.get("J", jaw_frame(p))

        def envelope() -> np.ndarray:
            ws = self._workspace()
            b = ws["box"]
            return np.array(
                [
                    [b[0], b[1]],
                    [b[2], b[3]],
                    [ws["table_z"] - 0.05, ws["z_ceiling"]],
                ]
            )

        gripper = self._task.gripper
        return GripGeometry(
            self._view,
            cameras=["agentview", "wrist"],
            tool_pose=lambda: self._eef(0),
            # Every control step (and a reset, which forgets the marks) changes the scene.
            state_digest=lambda: self._steps,
            frame=frame,
            pads=pads if gripper else None,
            contacts=lambda: self._grip_mech()["contacts"],
            width=(lambda: self._robot_state()["robot0_gripper_width"])
            if gripper
            else None,
            empty_width=0.004 if gripper else None,
            envelope=envelope,
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
        """One control step; success latched. While a motion records, the step's cameras
        (RECORD_SIZE), robot state, the composite ``action`` and the latched ``success``."""
        self._env.step(action)
        self._steps += 1
        self._success_step = tasks.latch(
            self._success_step, self._success(), self._steps
        )
        if self._record is not None:
            self._record.append(
                {
                    "agentview": self._render(self._cameras["agentview"], RECORD_SIZE),
                    "wrist": self._render(self._cameras["wrist"], RECORD_SIZE),
                    **self._robot_state(),
                    "action": np.asarray(action, dtype=np.float32),
                }
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
        xyz,
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
        record: bool = False,
    ) -> dict:
        """Servo one arm's TCP to a world position, holding its orientation.

        Args:
            xyz: [x, y, z] in metres, world frame (+z up; on the one-arm tasks +x away
                from robot0 and +y to its left; on the two-arm tasks the robots face each other
                along y, robot0 at -y facing +y; the table top is at ``get_state()["table_z"]``).
            arm: "robot0" or "robot1" on the two-arm tasks; omit on one arm.
            quat_xyzw: an absolute world orientation to reach as well, or None to keep the
                current one.
            rotvec: instead of quat_xyzw, a world-frame turn (axis x angle, rad) applied to the
                current orientation, e.g. [0, 0, 0.3] turns the wrist 0.3 rad about vertical.
            gripper: "open" / "close" sets the held gripper command first; None keeps it.
            tol_m, tol_rad: the servo stops within these of the target.
            max_steps: control-step budget (each moves at most ``step_m`` / ``step_rad``).
            record: also return every control step (``info["steps"]``: the cameras at
                RECORD_SIZE, the robot state, the composite action, ``success``).

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
        target = np.asarray(xyz, dtype=np.float64).reshape(3)
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
        self._record = [] if record else None
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
        record: bool = False,
    ) -> dict:
        """Servo one arm's TCP by a world-frame offset, holding its orientation.

        Args:
            delta_xyz: [dx, dy, dz] in metres (world frame, as `move_to`); at most the
                per-call cap.
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
            record=record,
        )

    def set_gripper(
        self,
        close: bool | str,
        *,
        arm: str | None = None,
        steps: int = 15,
        record: bool = False,
    ) -> dict:
        """Hold the arm and open or close its gripper; the command stays in force for later moves.

        Args:
            close: True (or "close") closes and holds, False (or "open") opens.
            arm: "robot0" or "robot1" on the two-arm tasks; omit on one arm.
            steps: control steps to drive the fingers (they stop early once they no longer
                move, e.g. on a grasped object).
            record: as `move_to`.

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
        self._record = [] if record else None
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
        """A motion call's answer: the new observation, the video frames it produced and,
        when it recorded, its control steps."""
        steps, self._record = self._record, None
        return {
            "obs": self._pack(),
            "info": {
                **info,
                "frames": list(self._motion_frames),
                **({"steps": steps} if steps is not None else {}),
            },
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
        data = self._sim.data
        # The two-arm tasks' grasp points are not bodies: TwoArmLift's handles are sites
        # (handle0_site_id / handle1_site_id), TwoArmHandover's handle a geom
        # (hammer_handle_geom_id); each takes its object's orientation.
        for attr, key, body, positions in (
            ("handle0_site_id", "pot_handle0", "pot", data.site_xpos),
            ("handle1_site_id", "pot_handle1", "pot", data.site_xpos),
            ("hammer_handle_geom_id", "hammer_handle", "hammer", data.geom_xpos),
        ):
            if hasattr(e, attr) and body in objects:
                poses[key] = ground_truth.pose(
                    positions[getattr(e, attr)], data.body_xquat[objects[body]]
                )
        return ground_truth.respond(poses, names)

    @staticmethod
    def _quat_of(rot: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        return Rotation.from_matrix(rot).as_quat()

    def preview_reach(self, xyz, quat_xyzw=None, *, arm: str | None = None) -> dict:
        """Whether an arm can reach a world position without moving: IK from its current joints
        by the ik service (``--ik``); the sim is not touched.

        Args:
            xyz: target [x, y, z] in metres (world frame), the point `move_to` would take.
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
            xyz,
            obs[f"robot{i}_eef_quat"] if quat_xyzw is None else quat_xyzw,
            base_pose=base,
        )

    # ---- the code primitives (manifests/robosuite.json) ------------------------------------------

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

    def back_project(
        self,
        row: int | None = None,
        col: int | None = None,
        camera: str = "agentview",
        row_range=None,
        col_range=None,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> dict:
        """World xyz of pixel (row, col) of the current 512x512 image of `camera` (row 0 = top),
        or, in region mode (row_range and col_range, optionally a z band), the midpoint of the
        window's world x and y and its median z.

        Args:
            row, col: pixel in the image `get_observation` returns for that camera.
            camera: "agentview" (default) or "wrist".
            row_range, col_range: [first, last) pixel rows and columns of a window.
            z_min, z_max: region mode: keep pixels with world z in this band.

        Returns:
            dict with ``world_xyz`` [x, y, z] in metres (pixel mode), or ``center_xyz``,
            ``median_xyz`` and ``n_valid`` (region mode). Raises when there is no depth.

        Example:
            >>> p = back_project(300, 260)["world_xyz"]
            >>> top = back_project(row_range=[200, 260], col_range=[240, 300])["center_xyz"]
        """
        view = self._view(camera)

        def span(r):
            return r if r is not None and len(r) == 2 and max(r) > min(r) else None

        rows, cols = span(row_range), span(col_range)
        if rows is not None or cols is not None:
            if rows is None or cols is None:
                raise ValueError("region mode needs both row_range and col_range")
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
                and (z_min is None or p[2] >= z_min)
                and (z_max is None or p[2] <= z_max)
            ]
            if len(pts) < 8:
                raise ValueError(
                    f"too few valid pixels in the region ({len(pts)}); widen the window or the z band"
                )
            a = np.asarray(pts)
            return {
                "camera": camera,
                "mode": "region",
                "center_xyz": [
                    round(float((a[:, 0].min() + a[:, 0].max()) / 2), 4),
                    round(float((a[:, 1].min() + a[:, 1].max()) / 2), 4),
                    round(float(np.median(a[:, 2])), 4),
                ],
                "median_xyz": [round(float(v), 4) for v in np.median(a, axis=0)],
                "n_valid": len(pts),
            }
        if row is None or col is None:
            raise ValueError("give row and col, or row_range and col_range")
        row, col = int(row), int(col)
        if not (0 <= row < CODE_RES and 0 <= col < CODE_RES):
            raise ValueError(
                f"pixel ({row}, {col}) out of bounds for {CODE_RES}x{CODE_RES}"
            )
        p = self._world_xyz(view, row, col)
        if p is None:
            raise ValueError(f"no depth at pixel ({row}, {col}); pick another pixel")
        return {
            "camera": camera,
            "pixel": [row, col],
            "world_xyz": [round(float(v), 4) for v in p],
        }

    def segment(
        self,
        prompt: str | None = None,
        point=None,
        camera: str = "agentview",
        min_score: float = 0.2,
    ) -> dict:
        """SAM3 segmentation of the current 512x512 image of `camera` by a text prompt or a
        positive point, and the top mask's median world position through the depth image.

        Args:
            prompt: what to segment, e.g. "red cube" (or give point).
            point: a positive point [row, col] instead of a prompt.
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
        text = (prompt or "").strip()
        if bool(text) == (point is not None):
            raise ValueError("give exactly one of a text prompt or a point [row, col]")
        view = self._view(camera)
        buf = io.BytesIO()
        Image.fromarray(view["rgb"]).save(buf, format="PNG")
        res = self._sam3.call(
            "sam3.segment",
            kwargs={
                "image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                **(
                    {"text_prompt": text}
                    if text
                    else {"point": [int(v) for v in point]}
                ),
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
        else:
            out["world_error"] = f"too few valid depth pixels ({len(pts)})"
        # The tool shows the mask over the image (a program's reply drops it).
        overlay = view["rgb"].astype(np.float32)
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([255.0, 0.0, 0.0])
        png = io.BytesIO()
        Image.fromarray(overlay.astype(np.uint8)).save(png, format="PNG")
        out["overlay_png_base64"] = base64.b64encode(png.getvalue()).decode("ascii")
        return out

    # ---- CaP-X's high tier (FrankaControlApi) and its joint-space parts (FrankaControlApiReduced) --
    #
    # Quaternions are CaP-X's: wxyz of panda_hand. Robosuite's grip site (what move_to servos) is
    # panda_hand turned half a turn about its own z (measured on bjb2, 2026-09-27), so an
    # orientation is converted with HAND_TO_SITE_WXYZ on the way in and back on the way out.
    # Positions are the grip site's (CaP-X's fingertip point): no TCP offset. The motions go through
    # the registered env.move_to / env.set_gripper (their limits, stop handling and the planners' id
    # expiry), as the tools do.

    def _site_xyzw(self, q_wxyz) -> list[float]:
        w, x, y, z = _quat_mul(_unit_quat(q_wxyz), HAND_TO_SITE_WXYZ)
        return [float(x), float(y), float(z), float(w)]

    @staticmethod
    def _hand_wxyz(q_xyzw) -> list[float]:
        x, y, z, w = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
        return [
            float(v) for v in _quat_mul(_unit_quat([w, x, y, z]), HAND_TO_SITE_WXYZ)
        ]

    def _object_points(self, object_name: str) -> tuple[np.ndarray, dict]:
        """The object's world points through the depth of the first camera whose SAM3 mask finds it."""
        for camera in ("agentview", "wrist"):
            seg = self.segment(prompt=str(object_name), camera=camera)
            if not seg.get("found"):
                continue
            view = self._view(camera)
            rows, cols = np.nonzero(seg["mask"])
            pts = [
                q
                for q in (
                    self._world_xyz(view, int(r), int(c)) for r, c in zip(rows, cols)
                )
                if q is not None
            ]
            if len(pts) >= 10:
                return np.asarray(pts), seg
        raise ValueError(f"no SAM3 detection with depth for {object_name!r}")

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> list:
        """CaP-X's get_object_pose from perception: [position (3,), quaternion_wxyz (4,),
        bbox_extent (3,) or None] of an object named in words. The position is the median of the
        mask's depth points; the extent the 5th-95th percentile span per world axis; the
        orientation is identity (CaP-X: disregard it, use the grasp quaternion or [0, 0, 1, 0]).

        Example:
            pos, quat, ext = get_object_pose("green cube", return_bbox_extent=True)
        """
        pts, _ = self._object_points(object_name)
        pos = np.median(pts, axis=0)
        extent = np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0)
        return [
            [round(float(v), 4) for v in pos],
            [1.0, 0.0, 0.0, 0.0],
            [round(float(v), 4) for v in extent] if return_bbox_extent else None,
        ]

    def get_object_pose_privileged(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> list:
        """CaP-X's privileged get_object_pose: the simulator's pose of an object named in words
        (tasks.OBJECT_NAMES, or a ground-truth name) and CaP-X's hard-coded extent.

        Example:
            pos, quat, ext = get_object_pose("red cube", return_bbox_extent=True)
        """
        names = tasks.OBJECT_NAMES.get(self._task_name, {})
        sim_name, extent = names.get(str(object_name), (str(object_name), None))
        poses = self.ground_truth_poses([sim_name])["poses"]
        pose = poses[sim_name]
        x, y, z, w = pose["quat_xyzw"]
        return [
            [float(v) for v in pose["pos"]],
            [float(w), float(x), float(y), float(z)],
            list(extent) if (return_bbox_extent and extent is not None) else None,
        ]

    def sample_grasp_pose(self, object_name: str, arm: str | None = None) -> list:
        """CaP-X's sample_grasp_pose: [position (3,), quaternion_wxyz (4,)] of a grasp of the
        object. With a grasp server, the best candidate of plan_grasp; without one, the mask's
        centre with the gripper pointing straight down (wxyz [0, 0, 1, 0]).

        Example:
            pos, quat = sample_grasp_pose("red cube")
            goto_pose(pos, quat, z_approach=0.1)
        """
        if self._grasp is not None:
            plan = self._grasp.plan_grasp(object=str(object_name), arm=arm)
            best = plan["candidates"][0]
            return [
                [float(v) for v in best["eef_position"]],
                self._hand_wxyz(best["eef_quat_xyzw"]),
            ]
        pts, _ = self._object_points(object_name)
        return [
            [round(float(v), 4) for v in np.median(pts, axis=0)],
            [0.0, 0.0, 1.0, 0.0],
        ]

    def sample_grasp_pose_privileged(
        self, object_name: str, arm: str | None = None
    ) -> list:
        """CaP-X's privileged sample_grasp_pose: the object's simulator position with the gripper
        pointing down (wxyz [0, 0, 1, 0]).

        Example:
            pos, quat = sample_grasp_pose("red cube")
        """
        pos, _, _ = self.get_object_pose_privileged(object_name)
        return [pos, [0.0, 0.0, 1.0, 0.0]]

    def goto_pose(
        self,
        position,
        quaternion_wxyz,
        z_approach: float = 0.0,
        arm: str | None = None,
    ) -> dict:
        """CaP-X's goto_pose: servo the grip site to a pose, first z_approach metres back along
        the gripper's approach axis when given; a move longer than the per-call cap runs as
        straight legs of move_to. Returns the last leg's report.

        Example:
            goto_pose([0.05, 0.0, 0.83], [0, 0, 1, 0], z_approach=0.1)
        """
        target = np.asarray(position, dtype=np.float64).reshape(3)
        q = _unit_quat(quaternion_wxyz)
        stops = []
        if float(z_approach):
            stops.append(
                target + _quat_matrix(q) @ np.array([0.0, 0.0, -float(z_approach)])
            )
        stops.append(target)
        out: dict = {}
        leg_m = 0.9 * self._max_move
        i = self._arm_index(arm)
        for stop in stops:
            start = self._eef(i)[0]
            legs = max(1, int(np.ceil(np.linalg.norm(stop - start) / leg_m)))
            for k in range(legs):
                here = self._eef(i)[0]
                left = legs - k
                leg = here + (stop - here) / max(
                    left, int(np.ceil(np.linalg.norm(stop - here) / leg_m))
                )
                out = self._rpc["env.move_to"](
                    leg, arm=arm, quat_xyzw=self._site_xyzw(q), max_steps=200
                )
                if self.stop_requested():
                    return out
        return out

    def home_pose(self, arm: str | None = None) -> dict:
        """CaP-X's home_pose: back to the arm's reset position, gripper pointing down.

        Example:
            home_pose()
        """
        i = self._arm_index(arm)
        home = self._home.get(self._task.arms[i], self._eef(i)[0])
        return self.goto_pose(home, [0.0, 0.0, 1.0, 0.0], arm=arm)

    def open_gripper(self, arm: str | None = None) -> dict:
        """CaP-X's open_gripper (40 control steps at most).

        Example:
            open_gripper()
        """
        return self._rpc["env.set_gripper"](False, arm=arm, steps=40)

    def close_gripper(self, arm: str | None = None) -> dict:
        """CaP-X's close_gripper (60 control steps at most; the fingers stop on an object).

        Example:
            close_gripper()
        """
        return self._rpc["env.set_gripper"](True, arm=arm, steps=60)

    def solve_ik(
        self, position, quaternion_wxyz, arm: str | None = None
    ) -> list[float]:
        """CaP-X's solve_ik: the arm's 7 joint angles for a grip-site pose, by the ik service
        (--ik) from the current joints. Raises when it has no solution.

        Example:
            q = solve_ik([0.05, 0.0, 0.95], [0, 0, 1, 0])
            move_to_joints(q)
        """
        if self._reach is None:
            raise RuntimeError("solve_ik needs the ik service (--ik)")
        r = self.preview_reach(position, self._site_xyzw(quaternion_wxyz), arm=arm)
        if r.get("status") != "reachable" or r.get("q") is None:
            raise ValueError(f"no IK solution: {r.get('message') or r.get('status')}")
        return [float(v) for v in np.asarray(r["q"]).reshape(-1)[:7]]

    def _fk(self, joints, i: int) -> tuple[np.ndarray, np.ndarray]:
        """The grip site's world position and xyzw orientation at ``joints`` (a scratch copy of
        the MuJoCo state; the sim is not touched)."""
        import mujoco

        robot = self._env.robots[i]
        model = self._sim.model._model
        data = mujoco.MjData(model)
        data.qpos[:] = self._sim.data.qpos
        q = np.asarray(joints, dtype=np.float64).reshape(-1)
        idx = list(robot._ref_joint_pos_indexes)[: len(q)]
        if len(q) != len(idx):
            raise ValueError(f"joints has {len(q)} values; the arm has {len(idx)}")
        data.qpos[idx] = q
        mujoco.mj_kinematics(model, data)
        site = robot.eef_site_id[robot.arms[0]]
        pos = np.array(data.site_xpos[site])
        return pos, self._quat_of(np.array(data.site_xmat[site]).reshape(3, 3))

    def move_to_joints(self, joints, arm: str | None = None) -> dict:
        """CaP-X's move_to_joints: drive the arm to a joint configuration. The arms run OSC_POSE,
        so the grip site servos (move_to, its per-call cap and workspace) to the pose those joints
        put it at (forward kinematics); the joints follow the controller, not the target exactly.

        Example:
            move_to_joints(solve_ik([0.05, 0.0, 0.95], [0, 0, 1, 0]))
        """
        i = self._arm_index(arm)
        pos, quat = self._fk(joints, i)
        return self._rpc["env.move_to"](
            pos, arm=arm, quat_xyzw=list(quat), max_steps=200
        )

    def traj_plan(
        self, start_pose_wxyz_xyz, end_pose_wxyz_xyz, arm: str | None = None
    ) -> list:
        """CaP-X's traj_plan: joint waypoints [N, 7] from one pose (wxyz then xyz) to another.
        Here a straight Cartesian line in 2 cm steps with the orientation slerped, solved by IK
        from each waypoint's predecessor (CaP-X runs PyRoKi's trajectory optimisation).

        Example:
            traj = traj_plan([0, 0, 1, 0, 0.0, 0.0, 1.0], [0, 0, 1, 0, 0.05, 0.0, 0.9])
            move_along_trajectory(traj)
        """
        a = np.asarray(start_pose_wxyz_xyz, dtype=np.float64).reshape(7)
        b = np.asarray(end_pose_wxyz_xyz, dtype=np.float64).reshape(7)
        n = max(1, min(100, int(np.ceil(np.linalg.norm(b[4:] - a[4:]) / 0.02))))
        out = []
        for k in range(1, n + 1):
            t = k / n
            q = _slerp(_unit_quat(a[:4]), _unit_quat(b[:4]), t)
            out.append(self.solve_ik(a[4:] + (b[4:] - a[4:]) * t, q, arm=arm))
        return out

    def move_along_trajectory(self, trajectory, arm: str | None = None) -> dict:
        """CaP-X's move_along_trajectory: move_to_joints through every waypoint (at most 100).

        Example:
            move_along_trajectory(traj_plan(start, end))
        """
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 2 or len(traj) > 100:
            raise ValueError("trajectory must be [N, 7] with N <= 100")
        out: dict = {}
        for q in traj:
            out = self.move_to_joints(q, arm=arm)
            if self.stop_requested():
                break
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
        "--geometry",
        action="store_true",
        help="serve the geometric toolset (one-arm tasks): point-cloud views, marked points, grip-site targets",
    )
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
    add_perception_arguments(p)
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
        mutating=("env.move_to",),
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
