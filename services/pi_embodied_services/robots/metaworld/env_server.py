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
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from typing import Any

import numpy as np

from pi_embodied_services.components.code_api import register_code_api
from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.robots.metaworld.primitives import METAWORLD_PRIMITIVES
from pi_embodied_services.utils import ground_truth
from pi_embodied_services.utils.logging import get_logger
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
#: ``ALL_V3_ENVIRONMENTS`` keys; the texts describe what the task's ``success`` checks.
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
    "dial-turn-v3": "Turn the dial half a turn (180 degrees) to the green target.",
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
    "lever-pull-v3": "Pull the lever up to the green target.",
    "peg-insert-side-v3": "Pick up the peg and insert it sideways into the hole in the box.",
    "pick-place-wall-v3": "Pick up the red puck, carry it around the wall and place it on the green target.",
    "pick-out-of-hole-v3": "Pick the red puck out of the hole and lift it to the green target.",
    "reach-v3": "Move the gripper to the green target.",
    "push-back-v3": "Push the red puck back toward the robot, onto the green target.",
    "push-v3": "Push the red puck to the green target.",
    "pick-place-v3": "Pick up the red puck and hold it at the green target.",
    "plate-slide-v3": "Slide the plate along the table into the cabinet goal.",
    "plate-slide-side-v3": "Slide the plate sideways into the cabinet goal.",
    "plate-slide-back-v3": "Slide the plate out of the cabinet back to the green target.",
    "plate-slide-back-side-v3": "Slide the plate sideways out of the cabinet to the green target.",
    "peg-unplug-side-v3": "Grasp the peg and unplug it sideways from the box.",
    "soccer-v3": "Push the soccer ball into the goal.",
    "stick-push-v3": "Pick up the stick and use it to push the thermos to the green target.",
    "stick-pull-v3": "Pick up the stick and use it to pull the thermos to the green target.",
    "push-wall-v3": "Push the red puck around the wall to the green target.",
    "reach-wall-v3": "Move the gripper around the wall to the green target.",
    "shelf-place-v3": "Pick up the blue block and place it on the shelf at the green target.",
    "sweep-into-v3": "Sweep the block into the hole in the table.",
    "sweep-v3": "Sweep the block off the table to the green target.",
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


def _delta_m(args, kwargs) -> float:
    """Translation of a `move_delta` primitive call (the run's accumulated-move cap)."""
    d = kwargs.get("delta_xyz", args[0] if args else None)
    return float(np.linalg.norm(np.asarray(d, dtype=np.float64).reshape(3)))


class MetaworldEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """One Metaworld task env; every call runs on the main thread (MuJoCo EGL)."""

    SERVICE_NAME = "metaworld-env"

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
            }
        )
        register_code_api(self, METAWORLD_PRIMITIVES)

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
        the target fixed (a gripper change). One agentview+wrist frame per step."""
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
            frames.append(self._pack())
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

    def set_gripper(self, open: bool) -> dict:
        """Open or close the gripper in place for GRIPPER_STEPS control steps."""
        return self.move_delta([0, 0, 0], "open" if open else "close") | {
            "target_gripper_open": bool(open)
        }

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
    args = p.parse_args()

    facade = MetaworldEnvFacade(
        task=args.task, seed=args.seed, view_size=args.view_size
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
