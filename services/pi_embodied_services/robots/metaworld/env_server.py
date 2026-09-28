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
# Task table and camera after OpenETA sim/envs/metaworld (main 7d4a0a1, Apache-2.0):
# its metaworld_config.json instructions were the starting point of INSTRUCTIONS, and
# its agentview (camera_id 2 moved to [0.75, 0.075, 0.7]) is Metaworld's `corner4`.

"""RPC server wrapping one Metaworld Sawyer task (``metaworld==3.1.1``, MuJoCo 3.3.0).

Action ``[dx, dy, dz, gripper]`` in [-1, 1]: a world-frame position delta of the mocap-driven
hand, 1.0 = ``ACTION_SCALE_M`` (1 cm) per control step, and the gripper effort (+1 close,
-1 open). Observations carry the ``agentview`` (``corner4``) and ``wrist`` (``gripperPOV``)
RGB at ``view_size`` px, the TCP position (the point between the finger pads) and the gripper
opening; ``info`` is the task's own metrics as plain scalars (``success``, ``grasp_success``,
``near_object``, ``obj_to_target``, ...). Success is the env's ``info["success"]``.

The env class is used directly (no ``metaworld.MT1`` task list): ``env.reset`` reseeds the
env's RNG and the process's global numpy and Python RNGs with the episode seed, so a seed
draws the same object layout in any process and every reset restores the same state.

Perception runs here for the tools and programs alike: ``env.back_project`` (a pixel, a pixel
list or a region of the current image through the rendered depth) and ``env.segment`` (SAM3 with
``--sam3``, located through the same depth). ``env.execute_grasp`` / ``env.execute_place`` run a
planned grasp id as bounded ``move_delta`` legs (utils/grasp_chain.py). The hand is a mocap
target welded to the Sawyer: there is no joint-space control to offer (CaP-X's ``solve_ik`` /
``move_to_joints`` do not apply).

Code mode (``code.run``, utils/code_exec.py ``CodeRunMixin``): a program calls the primitives of
the robot's manifest (packages/embodied/src/primitives/manifests/metaworld.json, read by
components/manifest.py) in a sandboxed subprocess, so the server requires its RPC token and
refuses other business calls while a program runs. What a program receives of a primitive's
result carries no object state: Metaworld's 39-D observation holds the object and goal poses and
its info metrics (``obj_to_target``, ``near_object``, the shaped reward, ...) are computed from
them, so only the robot's own state and the success flags (``PROGRAM_INFO``) reach it; a motion's
images go to the run's video instead (``_code_reply``).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from typing import Any

import numpy as np

from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.utils import grasp_chain as chain
from pi_embodied_services.utils import ground_truth, sam3_segment
from pi_embodied_services.utils.code_exec import CodeRunMixin
from pi_embodied_services.utils.gpu import add_cuda_argument, pin_egl
from pi_embodied_services.utils.grasp import (
    GraspPlanner,
    add_grasp_arguments,
    urls_from_args,
)
from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.perception import (
    add_perception_arguments,
    install_perception,
    render_view,
)
from pi_embodied_services.utils.rpc.main_thread_serve import MainThreadServeMixin

# MuJoCo env vars must be set BEFORE importing anything that touches MuJoCo. EGL by default;
# MUJOCO_GL=osmesa renders on the CPU (no GPU needed at 256 px).
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
assert "mujoco" not in sys.modules, (
    "mujoco must not be imported before MUJOCO_GL/PYOPENGL_PLATFORM are set"
)

logger = get_logger("env_server")

#: The metaworld release this server is written against (its MT50 task names are ``*-v3``).
METAWORLD_VERSION = "3.1.1"

#: MT50: every Metaworld v3 task and its instruction. The names are Metaworld's
#: ``ALL_V3_ENVIRONMENTS`` keys; the texts describe what the task's ``success`` checks, and a
#: target's colour is the colour its ``goal`` site renders in.
INSTRUCTIONS: dict[str, str] = {
    "assembly-v3": "Pick up the green ring nut and lower it onto the red peg so the peg passes through the ring.",
    "basketball-v3": "Pick up the basketball and drop it through the hoop.",
    "bin-picking-v3": "Pick up the green block from the left bin and place it in the right bin.",
    "box-close-v3": "Pick up the box lid and place it on top of the open box to close it.",
    "button-press-topdown-v3": "Press the red button on top of the box by pushing straight down on it.",
    "button-press-topdown-wall-v3": "Move around the wall, then press the red button on top of the box by pushing straight down on it.",
    "button-press-v3": "Press the red button on the front of the box by pushing it horizontally, away from the robot.",
    "button-press-wall-v3": "Move around the wall, then press the red button on the front of the box by pushing it horizontally.",
    "coffee-button-v3": "Press the button on the front of the coffee machine.",
    "coffee-pull-v3": "Grasp the mug under the coffee machine and pull it to the green target.",
    "coffee-push-v3": "Grasp the mug and push it under the coffee machine, onto the green target.",
    "dial-turn-v3": "Turn the dial half a turn (180 degrees) to the red target.",
    "disassemble-v3": "Grasp the green ring nut sitting on the red peg and lift it off the peg.",
    "door-close-v3": "Push the open door closed.",
    "door-lock-v3": "Rotate the lock knob on the door clockwise until it is locked.",
    "door-open-v3": "Pull the door handle to open the door.",
    "door-unlock-v3": "Rotate the lock knob on the door counter-clockwise until it is unlocked.",
    "hand-insert-v3": "Push the block into the hole in the table, following it in with the gripper.",
    "drawer-close-v3": "Push the open drawer closed.",
    "drawer-open-v3": "Grasp the drawer handle and pull the drawer open.",
    "faucet-open-v3": "Turn the faucet handle counter-clockwise to open the faucet.",
    "faucet-close-v3": "Turn the faucet handle clockwise to close the faucet.",
    "hammer-v3": "Pick up the hammer and hit the nail into the wall with it.",
    "handle-press-side-v3": "Press the handle down, approaching it from the side.",
    "handle-press-v3": "Press the handle down.",
    "handle-pull-side-v3": "Grasp the handle from the side and pull it up.",
    "handle-pull-v3": "Grasp the handle and pull it up.",
    "lever-pull-v3": "Pull the lever up to the red target.",
    "pick-place-wall-v3": "Pick up the red puck, carry it around the wall and place it on the blue target.",
    "pick-out-of-hole-v3": "Pick the red puck out of the hole and lift it to the blue target ball.",
    "pick-place-v3": "Pick up the red puck and hold it at the blue target ball floating above the table.",
    "plate-slide-v3": "Slide the plate along the table into the cabinet goal.",
    "plate-slide-side-v3": "Slide the plate sideways into the cabinet goal.",
    "plate-slide-back-v3": "Slide the plate out of the cabinet back to the red target.",
    "plate-slide-back-side-v3": "Slide the plate sideways out of the cabinet to the red target.",
    "peg-insert-side-v3": "Pick up the peg and insert it sideways into the hole in the box.",
    "peg-unplug-side-v3": "Grasp the peg and unplug it sideways from the box.",
    "soccer-v3": "Push the soccer ball into the goal.",
    "stick-push-v3": "Pick up the stick and use it to push the thermos to the green target.",
    "stick-pull-v3": "Pick up the stick and use it to pull the thermos to the green target.",
    "push-v3": "Push the red puck to the green target.",
    "push-wall-v3": "Push the red puck around the wall to the green target.",
    "push-back-v3": "Push the red puck back toward the robot, onto the green target.",
    "reach-v3": "Move the gripper to the red target ball floating above the table (not the red puck on the table).",
    "reach-wall-v3": "Move the gripper around the wall to the red target ball (not the red puck on the table).",
    "shelf-place-v3": "Pick up the blue block and place it on the shelf at the green target.",
    "sweep-into-v3": "Sweep the block into the hole in the table.",
    "sweep-v3": "Sweep the block off the table to the blue target.",
    "window-open-v3": "Push the window handle to slide the window open.",
    "window-close-v3": "Push the window handle to slide the window closed.",
}
TASKS = list(INSTRUCTIONS)
#: Metaworld's ML45 split (``env_dict.ML45_V3``): 45 train tasks, 5 held-out test tasks.
ML45_TEST = [
    "bin-picking-v3",
    "box-close-v3",
    "hand-insert-v3",
    "door-lock-v3",
    "door-unlock-v3",
]
ML45_TRAIN = [t for t in TASKS if t not in ML45_TEST]
#: MuJoCo cameras of Metaworld's xyz_base.xml behind each view. ``corner4`` is OpenETA's
#: agentview (a fixed camera at the robot's front right, looking down at the table);
#: ``gripperPOV`` tracks the hand and looks past the fingers.
CAMERAS = {"agentview": "corner4", "wrist": "gripperPOV"}
#: Views rendered upside down by their camera and turned 180 degrees here (OpenETA's
#: ``[::-1, ::-1]``): with it +z is up in the agentview, the robot at the bottom left.
ROTATED_VIEWS = {"agentview"}
#: Square view size (px), as ManiSkill's.
VIEW_SIZE = 256
#: Metres the hand moves per control step at action 1.0 (Metaworld ``action_scale``).
ACTION_SCALE_M = 0.01
OPEN = -1.0
CLOSE = 1.0
#: Control steps a gripper command is held: closing on nothing settles after ~18
#: (measured: the pad distance stops changing), opening after ~10.
GRIPPER_STEPS = 20
#: Control steps a move may take after its mocap target is reached, for the hand to settle.
SETTLE_STEPS = 12
#: TCP movement per control step below which the hand has settled, m.
SETTLED_M = 0.0005
#: Largest translation ``env.move_delta`` accepts per call, m.
MAX_MOVE_M = 0.2
#: The Sawyer stand and the table, fixed at the origin: not objects for ``ground_truth_poses``.
STAND_BODIES = {"base", "controller_box", "pedestal", "pedestal_feet", "torso"}
#: A target may leave the workspace box by this much (m) before it is refused.
BOX_TOL_M = 0.005
#: Code mode: the info keys a program may see (success flags; the other metrics are object state),
#: the video frames one run hands back (halved, every other one kept, when full), the largest
#: render and the most actions one ``chunk_step`` may ask for.
PROGRAM_INFO = ("success", "success_once", "grasp_success")
CODE_MAX_FRAMES = 128
CODE_MAX_RENDER = 1024
CODE_MAX_CHUNK = 200
#: back_project / segment resolutions: the images shown, or the 1024 px render.
RESOLUTIONS = {"low": VIEW_SIZE, "high": 1024}
#: The facade's motions: grasp and detection ids expire after any of them.
MOTIONS = (
    "env.move_delta",
    "env.set_gripper",
    "env.execute_grasp",
    "env.execute_place",
    "env.step",
    "env.chunk_step",
)


def instruction(task: str) -> str:
    """The instruction of ``task`` (an unknown task lists the table)."""
    if task not in INSTRUCTIONS:
        raise ValueError(
            f"unknown Metaworld task {task!r}; the {len(TASKS)} tasks are {TASKS}"
        )
    return INSTRUCTIONS[task]


def seed_all(env, seed: int) -> None:
    """Seed the env's RNG (its object layout) and the process's global numpy and Python RNGs.

    Metaworld draws the layout from ``env.np_random`` (``seeded_rand_vec``), but MuJoCo-side
    helpers and any library code drawing from the global RNGs would otherwise drift between
    processes; seeding all three makes a reset bitwise repeatable, as on LIBERO."""
    random.seed(seed)
    np.random.seed(int(seed) % 2**32)
    env.seed(int(seed))


def make_env(task: str, *, max_path_length: int = 10**9):
    """One Metaworld task env, layout drawn from its seeded RNG at every reset.

    The class is used directly instead of ``metaworld.MT1``: MT1 bakes one ``rand_vec`` per
    task object, so a seed would index a task list instead of seeding the layout. The env's
    own step limit (``max_path_length`` 500, after which ``step`` raises) is lifted: the
    planner's budget ends an episode."""
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS

    instruction(task)
    env = ALL_V3_ENVIRONMENTS[task](render_mode=None)
    env._set_task_called = True
    env._freeze_rand_vec = False
    env.seeded_rand_vec = True
    env._partially_observable = False
    env.max_path_length = int(max_path_length)
    return env


def camera_meta(
    model, data, camera: str, height: int, width: int, rotated: bool = False
) -> dict:
    """OpenCV intrinsics and camera-to-world extrinsic of a MuJoCo camera.

    MuJoCo cameras look along -z with y up and set the vertical field of view; OpenCV looks
    along +z with y down, so the rotation's y and z columns flip. A view turned 180 degrees
    (``rotated``) is a camera rolled half a turn about its optical axis."""
    cam = model.camera(camera)
    fovy = math.radians(float(model.cam_fovy[cam.id]))
    f = (height / 2) / math.tan(fovy / 2)
    k = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)
    r = np.asarray(data.cam_xmat[cam.id], dtype=np.float64).reshape(3, 3)
    r = r @ np.diag([1.0, -1.0, -1.0])
    if rotated:
        r = r @ np.diag([-1.0, -1.0, 1.0])
    c2w = np.eye(4)
    c2w[:3, :3] = r
    c2w[:3, 3] = np.asarray(data.cam_xpos[cam.id], dtype=np.float64)
    return {
        "camera_name": camera,
        "height": int(height),
        "width": int(width),
        "intrinsic_K": k,
        "extrinsic_cam2world": c2w,
    }


def object_bodies(model, robot_root: str = "right_arm_base_link") -> list[str]:
    """Named MuJoCo bodies of the scene that are not the robot (the subtree under
    ``robot_root`` and its stand), the mocap target or the world: the
    ``env.ground_truth_poses`` list."""
    root = model.body(robot_root).id
    out = []
    for i in range(1, model.nbody):
        name = model.body(i).name
        if not name or name == "mocap" or name in STAND_BODIES:
            continue
        j = i
        while j > 0 and j != root:
            j = int(model.body_parentid[j])
        if j != root:
            out.append(name)
    return out


def program_info(info: dict) -> dict:
    """The success flags of a step's info: what a program may see of it."""
    return {k: info[k] for k in PROGRAM_INFO if k in info}


class MetaworldEnvFacade(CodeRunMixin, MainThreadServeMixin, BaseEnvFacade):
    """One Metaworld task env; every call runs on the main thread (MuJoCo EGL)."""

    SERVICE_NAME = "metaworld-env"
    #: Control steps since the launch, and code mode's run: the count before it and its video
    #: frames (class defaults so the registry can be built on a bare facade, as in the tests).
    _steps = 0
    _run_start = 0
    _run_frames: list | tuple = ()
    #: --sam3 (env.segment) and the grasp planner (env.execute_*): set in main().
    _sam3 = sam3_segment.Sam3(None)
    _grasp: GraspPlanner | None = None

    def __init__(self, *, task: str, seed: int, view_size: int = VIEW_SIZE):
        super().__init__()
        self._env = make_env(task)
        self._seed = int(seed)
        self._view_size = int(view_size)
        self._renderers: dict[tuple[int, int, bool], Any] = {}
        self._obs = np.zeros(39)
        self._info: dict = {}
        self._success_once = False
        #: The gripper effort held between calls (+1 close, -1 open); a reset opens.
        self._gripper_effort = OPEN
        self._closed = False
        #: TCP offset from the mocap body: the workspace box in TCP coordinates.
        self._box = None
        self._meta = {
            "task": task,
            "seed": self._seed,
            "metaworld": METAWORLD_VERSION,
            "agentview": CAMERAS["agentview"],
            "wrist": CAMERAS["wrist"],
            "view_size": self._view_size,
            "action_scale_m": ACTION_SCALE_M,
            "max_move_m": MAX_MOVE_M,
            "gripper_steps": GRIPPER_STEPS,
            "action_space": list(self._env.action_space.shape),
        }

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc.update(
            {
                "env.state": self.state,
                "env.raw_obs": self.raw_obs,
                "env.move_delta": self.move_delta,
                "env.set_gripper": self.set_gripper,
                "env.ground_truth_poses": self.ground_truth_poses,
                "env.back_project": self.back_project,
                "env.segment": self.segment,
                # Served always (the id wrappers expire ids after them); they need the planner.
                "env.execute_grasp": self.execute_grasp,
                "env.execute_place": self.execute_place,
            }
        )
        self._readonly_methods.update({"env.back_project", "env.segment"})
        # The primitives are packages/embodied/src/primitives/manifests/metaworld.json (with
        # pi); code.api, the programs' whitelist and the startup self-check come from it.
        self._manifest_code_run(
            "metaworld",
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
            "sam3": bool(self._sam3),
            "grasp": self._grasp is not None,
            "place": self._grasp is not None
            and bool(self._grasp.capabilities().get("place")),
            "unidepth": "env.enhance_depth" in self._rpc,
            "privileged": True,
        }.get(capability, False)

    def install_grasp(self, planner: GraspPlanner | None) -> None:
        """The grasp planner's primitives (their ids expire after every motion)."""
        self._grasp = planner
        if planner is not None:
            planner.install(self, mutating=GraspPlanner.MUTATING + MOTIONS)

    # ---- helpers ----

    def _tcp(self) -> np.ndarray:
        return np.asarray(self._env.tcp_center, dtype=np.float64).reshape(3)

    def _gripper_width(self) -> float:
        """Distance between the finger pads, m (0.094 open, 0.03 closed on nothing)."""
        env = self._env
        return float(
            np.linalg.norm(env.get_body_com("leftpad") - env.get_body_com("rightpad"))
        )

    def _mocap(self) -> np.ndarray:
        return np.asarray(self._env.data.mocap_pos, dtype=np.float64).reshape(-1)[:3]

    def _renderer(self, height: int, width: int, depth: bool):
        import mujoco

        key = (int(height), int(width), depth)
        r = self._renderers.get(key)
        if r is None:
            r = mujoco.Renderer(self._env.model, height=int(height), width=int(width))
            if depth:
                r.enable_depth_rendering()
            self._renderers[key] = r
        return r

    def _render(self, camera: str, height: int, width: int, depth: bool = False):
        r = self._renderer(height, width, depth)
        r.update_scene(self._env.data, camera=CAMERAS.get(camera, camera))
        out = r.render()
        if camera in ROTATED_VIEWS:
            out = out[::-1, ::-1]
        return np.ascontiguousarray(
            out.astype(np.float32) if depth else out.astype(np.uint8)
        )

    def _pack(self) -> dict:
        return {
            "agentview": self._render("agentview", self._view_size, self._view_size),
            "wrist": self._render("wrist", self._view_size, self._view_size),
            "tcp_pos": self._tcp().astype(np.float32),
            "gripper_width": self._gripper_width(),
            "obs": np.asarray(self._obs, dtype=np.float32),
        }

    @staticmethod
    def _flat(info: dict) -> dict:
        out = {}
        for key, value in info.items():
            arr = np.asarray(value).reshape(-1)
            if arr.size == 1:
                v = arr[0].item()
                out[key] = bool(v) if key in ("success", "grasp_success") else v
        return out

    def _step(self, action) -> tuple:
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(4), -1, 1)
        obs, rew, term, trunc, info = self._env.step(a)
        self._steps += 1
        if a[3] != 0:
            # The effort held between calls follows a raw step's too (a later move keeps it).
            self._gripper_effort = CLOSE if a[3] > 0 else OPEN
        self._obs = np.asarray(obs, dtype=np.float64)
        self._info = self._flat(info)
        self._success_once |= bool(self._info.get("success"))
        self._info["success_once"] = self._success_once
        return float(rew), bool(self._info.get("success")), bool(trunc), self._info

    def _workspace(self) -> dict:
        """The mocap box shifted to the TCP: where ``move_delta`` may put the fingertips."""
        env = self._env
        offset = self._tcp() - self._mocap()
        return {
            "min": (np.asarray(env.mocap_low, dtype=np.float64) + offset)
            .round(4)
            .tolist(),
            "max": (np.asarray(env.mocap_high, dtype=np.float64) + offset)
            .round(4)
            .tolist(),
        }

    # ---- code mode (run_code) ----

    def _begin_run(self) -> None:
        self._run_start = self._steps
        self._run_frames = []

    def _finish_run(self) -> dict:
        """The run's effect for pi: control steps taken, the latched success, the new
        observation (the tools' ``obs``), its success flags, the gripper command and the run's
        video frames (agentview and wrist side by side, as pi's video records them)."""
        return {
            "steps": self._steps - self._run_start,
            "success": self._success_once,
            "obs": self._pack(),
            "info": program_info(self._info),
            "gripper": "close" if self._gripper_effort > 0 else "open",
            "frames": list(self._run_frames),
        }

    def _keep_frames(self, packs) -> None:
        for p in packs:
            if len(self._run_frames) >= CODE_MAX_FRAMES:
                self._run_frames = self._run_frames[::2]
            self._run_frames.append(
                np.concatenate([p["agentview"], p["wrist"]], axis=1)
            )

    @staticmethod
    def _robot_state(pack: dict) -> dict:
        """An observation without its images and without Metaworld's 39-D ``obs``."""
        return {
            "tcp_pos": np.asarray(pack["tcp_pos"]).round(5).tolist(),
            "gripper_width": round(float(pack["gripper_width"]), 5),
        }

    def _code_reply(self, method: str, out: Any) -> Any:
        """What a program receives: no object state (the 39-D obs, the info metrics, the
        shaped reward) and no images of a motion (they go to the run's video)."""
        if method == "env.segment":
            return sam3_segment.for_program(out)
        if method in (
            "env.move_delta",
            "env.set_gripper",
            "env.execute_grasp",
            "env.execute_place",
        ):
            out = dict(out)
            self._keep_frames(out.pop("frames", []))
            out["info"] = program_info(out.get("info", {}))
            return out
        if method == "env.state":
            return {**out, "info": program_info(out["info"])}
        if method == "env.step":
            pack, _rew, success, truncated, info = out
            self._keep_frames([pack])
            return {
                "success": success,
                "truncated": truncated,
                "info": program_info(info),
                "state": self._robot_state(pack),
            }
        if method == "env.chunk_step":
            packs, _rews, terms, truncs, info = out
            packs = packs if isinstance(packs, list) else [packs]
            self._keep_frames(packs)
            states = [self._robot_state(p) for p in packs]
            return {
                "steps": int(len(terms)),
                "success": bool(np.any(terms)),
                "truncated": bool(np.any(truncs)),
                "info": program_info(info),
                "state": states[-1],
                **({"states": states} if len(states) > 1 else {}),
            }
        return out

    def _code_move_m(self, method: str, kwargs: dict) -> float:
        """How far a program's call may move the hand (the run's translation cap)."""
        if method == "env.move_delta":
            d = np.asarray(kwargs["delta_xyz"], dtype=np.float64).reshape(3)
            return float(np.linalg.norm(d))
        if method in ("env.step", "env.chunk_step"):
            a = kwargs["action"] if method == "env.step" else kwargs["actions"]
            a = np.clip(np.asarray(a, dtype=np.float64).reshape(-1, 4)[:, :3], -1, 1)
            return float(np.linalg.norm(a, axis=1).sum() * ACTION_SCALE_M)
        return 0.0

    def _code_check(self, method: str, kwargs: dict) -> None:
        """Refuse a program's call that the run's wall clock could not bound."""
        if method in ("env.render_camera", "env.get_camera_meta"):
            for k in ("height", "width"):
                if int(kwargs.get(k, VIEW_SIZE)) > CODE_MAX_RENDER:
                    raise ValueError(f"{method[4:]} {k} is at most {CODE_MAX_RENDER}")
        if method == "env.chunk_step":
            n = np.asarray(kwargs["actions"], dtype=np.float64).size // 4
            if n > CODE_MAX_CHUNK:
                raise ValueError(
                    f"chunk_step takes at most {CODE_MAX_CHUNK} actions per call in code mode"
                )

    # ---- gym-like surface ----

    def reset(self, seed: int | None = None):
        """Reset to ``seed`` (default: the launch seed): the layout is drawn from the seeded
        RNG, the hand goes to the task's start pose with the gripper open."""
        s = self._seed if seed is None else int(seed)
        seed_all(self._env, s)
        obs, info = self._env.reset()
        self._obs = np.asarray(obs, dtype=np.float64)
        self._info = self._flat(info)
        self._success_once = False
        self._gripper_effort = OPEN
        self._box = self._workspace()
        self._meta["workspace"] = self._box
        return self._pack(), self._info

    def step(self, action):
        rew, term, trunc, info = self._step(action)
        return self._pack(), rew, term, trunc, info

    def chunk_step(self, actions, *, return_all_frames: bool = False):
        """Run ``actions`` [N, 4] in one call; stops early on ``stop`` or success. Arrays hold
        one entry per executed action."""
        frames, rews, terms, truncs = [], [], [], []
        info = dict(self._info)
        for action in np.asarray(actions, dtype=np.float32).reshape(-1, 4):
            if self.stop_requested():
                info["cancelled"] = True
                break
            rew, term, trunc, info = self._step(action)
            if return_all_frames:
                frames.append(self._pack())
            rews.append(rew)
            terms.append(term)
            truncs.append(trunc)
            if term:
                break
        last = frames[-1] if frames else self._pack()
        return (
            (frames or [last]) if return_all_frames else last,
            np.asarray(rews, dtype=np.float32),
            np.asarray(terms, dtype=bool),
            np.asarray(truncs, dtype=bool),
            info,
        )

    def _servo(self, delta: np.ndarray, gripper: float, hold: int = 0):
        """Move the mocap target by ``delta`` at up to 1 cm per control step with the gripper
        effort held, then let the hand settle: the hand follows the mocap through a weld with
        lag, so commanding the hand's own error overshoots (measured 7 cm for a 5 cm command).
        Stops when the mocap has arrived and the TCP moved less than SETTLED_M in a step (or
        SETTLE_STEPS after arrival), at success or at ``stop``. ``hold`` steps run first with
        the target fixed (a gripper change). One agentview+wrist frame per step, with the
        ``[dx, dy, dz, gripper]`` action it applied and its ``success``."""
        target = self._mocap() + delta
        frames: list = []
        info = dict(self._info)
        cancelled = False
        settle = 0
        last = self._tcp()
        for k in range(
            int(hold)
            + int(math.ceil(np.linalg.norm(delta) / ACTION_SCALE_M - 1e-9))
            + SETTLE_STEPS
        ):
            if self.stop_requested():
                cancelled = True
                break
            err = target - self._mocap() if k >= hold else np.zeros(3)
            arrived = k >= hold and np.linalg.norm(err) < 1e-6
            if arrived:
                settle += 1
                if (
                    np.linalg.norm(self._tcp() - last) < SETTLED_M
                    or settle > SETTLE_STEPS
                ):
                    break
            last = self._tcp()
            a = np.append(np.clip(err / ACTION_SCALE_M, -1, 1), gripper)
            _rew, term, _trunc, info = self._step(a)
            # The env action of the step and its success, for the Flywheel recorder.
            frames.append(
                {**self._pack(), "action": a.astype(np.float32), "success": term}
            )
            if term:
                break
        return frames, info, cancelled

    def move_delta(
        self,
        delta_xyz,
        gripper: str | None = None,
        *,
        tol_m: float = 0.006,
    ) -> dict:
        """Translate the TCP by a world-frame ``delta_xyz`` (m), first opening or closing
        the gripper (``gripper`` "open" / "close", held for GRIPPER_STEPS) when given.
        Refuses (nothing commanded) a delta above ``MAX_MOVE_M`` or a target outside the
        workspace box. Closed-loop on the mocap target with a settle (``_servo``): at most
        ceil(|delta| / 1 cm) + SETTLE_STEPS control steps. Returns the frames of every step,
        so the caller can record the video."""
        delta = np.asarray(delta_xyz, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(delta))
        if not norm <= MAX_MOVE_M:
            raise ValueError(
                f"delta moves {norm:.4f} m; the limit is {MAX_MOVE_M} m per call. Split the motion."
            )
        if gripper not in (None, "open", "close"):
            raise ValueError(f"gripper must be 'open' or 'close', got {gripper!r}")
        start = self._tcp()
        target = start + delta
        box = self._box or self._workspace()
        lo, hi = np.asarray(box["min"]) - BOX_TOL_M, np.asarray(box["max"]) + BOX_TOL_M
        if np.any(target < lo) or np.any(target > hi):
            raise ValueError(
                f"target {target.round(4).tolist()} is outside the workspace box "
                f"min {box['min']} max {box['max']}; nothing commanded"
            )
        effort = (
            self._gripper_effort
            if gripper is None
            else (OPEN if gripper == "open" else CLOSE)
        )
        self._gripper_effort = effort
        # A gripper change holds the arm still until the fingers settle, then the move runs.
        frames, info, cancelled = self._servo(
            delta, effort, hold=GRIPPER_STEPS if gripper is not None else 0
        )
        if not frames:
            frames = [self._pack()]
        end = self._tcp()
        return {
            "ok": bool(np.linalg.norm(target - end) < tol_m)
            or bool(info.get("success")),
            "requested_delta_xyz": delta.round(5).tolist(),
            "start_tcp_pos": start.round(5).tolist(),
            "final_tcp_pos": end.round(5).tolist(),
            "final_error_m": round(float(np.linalg.norm(target - end)), 5),
            "moved_m": (end - start).round(5).tolist(),
            "gripper": "close" if effort > 0 else "open",
            "gripper_width": round(self._gripper_width(), 5),
            "steps_used": len(frames),
            "frames": frames,
            "info": info,
            **({"cancelled": True} if cancelled else {}),
        }

    def set_gripper(self, close: bool) -> dict:
        """Close (``close=True``) or open the gripper in place for GRIPPER_STEPS control
        steps."""
        return self.move_delta([0, 0, 0], "close" if close else "open")

    # ---- perception through the rendered depth ----

    def _world_map(self, camera: str, size: int) -> tuple[np.ndarray, np.ndarray]:
        """The current image of ``camera`` at ``size`` px and the world xyz of every pixel
        (NaN where the depth is invalid)."""
        if camera not in CAMERAS:
            raise ValueError(f"unknown camera {camera!r}; one of {sorted(CAMERAS)}")
        rgb, depth = self.render_camera(camera, size, size, depth=True)
        meta = self.get_camera_meta(camera, size, size)
        K = np.asarray(meta["intrinsic_K"], dtype=np.float64)
        T = np.asarray(meta["extrinsic_cam2world"], dtype=np.float64)
        rows, cols = np.mgrid[0:size, 0:size]
        z = np.asarray(depth, dtype=np.float64)
        x = (cols - K[0, 2]) * z / K[0, 0]
        y = (rows - K[1, 2]) * z / K[1, 1]
        cam = np.stack([x, y, z], axis=-1)
        xyz = cam @ T[:3, :3].T + T[:3, 3]
        valid = np.isfinite(xyz).all(-1) & (np.abs(xyz).sum(-1) > 1e-6) & (z > 0)
        xyz[~valid] = np.nan
        return rgb, xyz

    @staticmethod
    def _size(resolution: str) -> int:
        if resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {sorted(RESOLUTIONS)}")
        return RESOLUTIONS[resolution]

    def back_project(
        self,
        row: int | None = None,
        col: int | None = None,
        camera: str = "agentview",
        resolution: str = "low",
        row_range=None,
        col_range=None,
        z_min: float | None = None,
        z_max: float | None = None,
        pixels=None,
    ):
        """World xyz of pixel (row, col) of the current image (``resolution`` low = the
        images shown, high = 1024 px) through the rendered depth. Region mode (row_range +
        col_range, optional z band): ``center_xyz`` is the midpoint of world x and y over the
        window and the median z. ``pixels`` [[row, col], ...] returns a list (None where the
        depth is invalid). Errors come back as ``error``, as the tool shows them."""
        size = self._size(resolution)
        _rgb, xyz = self._world_map(camera, size)

        def at(r: int, c: int):
            if not (0 <= r < size and 0 <= c < size):
                return None
            p = xyz[r, c]
            return None if np.isnan(p).any() else [round(float(v), 4) for v in p]

        if pixels is not None:
            return [at(int(r), int(c)) for r, c in pixels]
        base = {"camera": camera, "resolution": resolution}

        def span(v):
            return v if v is not None and max(v) > min(v) else None

        rows, cols = span(row_range), span(col_range)
        if rows or cols:
            if not (rows and cols):
                return {"error": "region mode needs both row_range and col_range"}
            r0, r1 = (int(np.clip(v, 0, size)) for v in (min(rows), max(rows)))
            c0, c1 = (int(np.clip(v, 0, size)) for v in (min(cols), max(cols)))
            pts = xyz[r0:r1, c0:c1].reshape(-1, 3)
            pts = pts[~np.isnan(pts).any(1)]
            if z_min is not None:
                pts = pts[pts[:, 2] >= z_min]
            if z_max is not None:
                pts = pts[pts[:, 2] <= z_max]
            if len(pts) < 8:
                return {
                    "error": f"too few valid pixels in region ({len(pts)}); widen the window or the z band"
                }
            lo, hi = pts.min(0), pts.max(0)
            return {
                **base,
                "mode": "region",
                "center_xyz": [
                    round(float((lo[0] + hi[0]) / 2), 4),
                    round(float((lo[1] + hi[1]) / 2), 4),
                    round(float(np.median(pts[:, 2])), 4),
                ],
                "median_xyz": [round(float(v), 4) for v in np.median(pts, 0)],
                "n_valid": int(len(pts)),
            }
        if row is None or col is None:
            return {"error": "give row and col, or row_range and col_range"}
        if not (0 <= row < size and 0 <= col < size):
            return {"error": f"pixel ({row},{col}) out of bounds for {size}x{size}"}
        p = at(int(row), int(col))
        if p is None:
            return {"error": f"invalid world xyz at ({row},{col}); pick another pixel"}
        return {**base, "pixel": [int(row), int(col)], "world_xyz": p}

    def segment(
        self,
        prompt: str | None = None,
        point=None,
        camera: str = "agentview",
        resolution: str = "low",
        min_score: float = 0.2,
    ) -> dict:
        """SAM3 mask of a text prompt (or a positive point [row, col]) on the current image
        of ``camera``, located through the rendered depth: ``found``, ``score``, ``box``,
        ``mask``, ``n_pixels``, ``centroid_pixel``, ``world_xyz`` (median over the mask) and
        the tool's ``overlay_png_base64``."""
        rgb, xyz = self._world_map(camera, self._size(resolution))

        def locate(px):
            out = []
            for r, c in px:
                p = xyz[r, c]
                out.append(None if np.isnan(p).any() else p.tolist())
            return out

        return sam3_segment.segment(
            self._sam3,
            rgb,
            locate,
            prompt=prompt,
            point=point,
            min_score=min_score,
            samples=10**9,
            extra={"camera": camera, "resolution": resolution},
        )

    # ---- planned grasps (utils/grasp_chain.py) ----

    def _chain(self, kind: str, grasp_id: str, standoff: float | None) -> dict:
        if self._grasp is None:
            raise RuntimeError(
                f"execute_{kind} needs the grasp planner (start with --contact-graspnet & co)"
            )
        out = chain.run_chain(
            kind,
            grasp_id,
            rpc=self._rpc,
            current=self._tcp,
            max_step=MAX_MOVE_M,
            move=lambda d, g: self.move_delta(list(d), g),
            gripper=lambda g: self.set_gripper(g == "close"),
            stop=self.stop_requested,
            solved=lambda: self._success_once,
            standoff=standoff,
            # The Sawyer hand points down at yaw 0 (the planner's eef_pose).
            yaw=lambda: 0.0,
        )
        frames = out.pop("frames", []) or [self._pack()]
        steps = out.pop("control_steps", 0)
        return {
            **out,
            "gripper": "close" if self._gripper_effort > 0 else "open",
            "gripper_width": round(self._gripper_width(), 5),
            "steps_used": steps,
            "frames": frames,
            "info": dict(self._info),
        }

    def execute_grasp(self, grasp_id: str, standoff: float | None = None) -> dict:
        """Run one planned grasp id: open at the standoff back along its approach, descend,
        close, lift, each leg as bounded move_delta calls; refused (nothing moves) unless it
        approaches from nearly straight above. ``frames`` as move_delta's."""
        return self._chain("grasp", grasp_id, standoff)

    def execute_place(self, place_id: str, standoff: float | None = None) -> dict:
        """Run one planned place id: to the pre-place above it, descend, open, retreat."""
        return self._chain("place", place_id, standoff)

    def state(self) -> dict:
        """TCP position, gripper opening and the task metrics (no stepping): the
        ``get_state`` primitive."""
        return {
            "tcp_pos": self._tcp().round(5).tolist(),
            "gripper_width": round(self._gripper_width(), 5),
            "gripper_command": "close" if self._gripper_effort > 0 else "open",
            "success": bool(self._info.get("success", False)),
            "success_once": self._success_once,
            "info": dict(self._info),
            "workspace": self._box or self._workspace(),
        }

    def raw_obs(self) -> dict:
        """Metaworld's 39-D observation and the flattened step info."""
        return {
            "obs": np.asarray(self._obs, dtype=np.float64),
            "info": dict(self._info),
        }

    def ground_truth_poses(self, names=None) -> dict:
        """World poses of ``names`` (default all) of the scene's non-robot bodies and the
        task's ``goal`` (``--privileged``)."""
        env = self._env
        poses = {
            name: ground_truth.pose(env.data.body(name).xpos, env.data.body(name).xquat)
            for name in object_bodies(env.model)
        }
        if env._target_pos is not None:
            poses["goal"] = ground_truth.pose(env._target_pos, [1, 0, 0, 0])
        return ground_truth.respond(poses, names)

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = VIEW_SIZE,
        width: int = VIEW_SIZE,
        depth: bool = False,
    ):
        """RGB uint8[H,W,3] of ``agentview`` / ``wrist`` (or a MuJoCo camera name) at the
        current state, or ``[rgb, depth_m float32[H,W]]``; rows top-first."""
        rgb = self._render(camera_name, height, width)
        if not depth:
            return rgb
        return [rgb, self._render(camera_name, height, width, depth=True)]

    def get_camera_meta(
        self,
        camera_name: str = "agentview",
        height: int = VIEW_SIZE,
        width: int = VIEW_SIZE,
    ) -> dict:
        return camera_meta(
            self._env.model,
            self._env.data,
            CAMERAS.get(camera_name, camera_name),
            height,
            width,
            rotated=camera_name in ROTATED_VIEWS,
        )

    def get_task_language(self) -> str:
        return instruction(self._meta["task"])

    def get_env_meta(self) -> dict:
        return dict(self._meta)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
        self._env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--task", default="reach-v3", help=f"one of {TASKS}")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--view-size", type=int, default=VIEW_SIZE, help="square view, px")
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    add_perception_arguments(p, sam3=True)
    add_grasp_arguments(p)
    add_cuda_argument(p)
    args = p.parse_args()
    # CUDA and the EGL renderer on one GPU (utils/gpu.py): --cuda-device, the deployment's, or
    # the first CUDA_VISIBLE_DEVICES entry.
    pin_egl(args.cuda_device)

    facade = MetaworldEnvFacade(
        task=args.task, seed=args.seed, view_size=args.view_size
    )
    # --sam3: env.segment; --sam3 / --unidepth: env.detect, env.select_detection,
    # env.reject_detection, env.enhance_depth.
    facade._sam3 = sam3_segment.Sam3(args.sam3)
    view = render_view(facade)
    perception = install_perception(
        facade, args, cameras=["agentview", "wrist"], view=view, mutating=MOTIONS
    )
    # --contact-graspnet & co: env.plan_grasp, env.claim_waypoints and friends over the same views
    # (the gripper points straight down (xyzw, 180 deg about x)); env.execute_grasp /
    # env.execute_place run the claimed path as move_delta legs.
    facade.install_grasp(
        GraspPlanner.from_args(
            view,
            cameras=["agentview", "wrist"],
            masks=perception.book if perception is not None else None,
            sam3=args.sam3 or (perception.sam3 if perception is not None else None),
            eef_pose=lambda arm: (facade._tcp(), np.array([1.0, 0.0, 0.0, 0.0])),
            wrist_camera="wrist",
            # Ids expire only when the sim stepped: a refused, unmoved execute_grasp keeps them,
            # so its "ask plan_grasp for the next candidate (next_after)" can be followed.
            state_digest=lambda: facade._steps,
            **urls_from_args(args),
        )
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
