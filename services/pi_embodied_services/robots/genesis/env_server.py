# Copyright 2025 The RLinf Authors.
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
# Modified by pi-embodied: OpenETA sim/envs/genesis (genesis_env.py, tasks/cube_pick.py at
# 7d4a0a1: the Franka MJCF, home pose, gains, cube size and sampling box) rebuilt as one
# single-env RPC server on Genesis 1.4 with a translation-only IK controller, a wrist
# camera, a lift-based success and the pi-embodied motion limits.

"""RPC server wrapping one Genesis scene: a Franka Panda, a table plane and the task's
objects (``cube_pick``: one 4 cm cube).

The robot is driven in the base frame by ``env.move_delta`` (an IK servo that holds the
reset orientation, in ~2 cm decisions) and ``env.set_gripper``; ``env.step`` /
``env.chunk_step`` take the raw ``[dx, dy, dz, gripper]`` action (metres, +1 open / -1
close) for VLA-style clients. Every observation carries the front (``agentview``) and
wrist RGB, the TCP pose, the gripper opening and the task flags. Limits (a workspace box, a
Z floor, a per-call cap) are checked here, before anything moves.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from pi_embodied_services.components.code_api import register_code_api
from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.robots.genesis.primitives import GENESIS_PRIMITIVES
from pi_embodied_services.utils import ground_truth
from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.rpc.main_thread_serve import MainThreadServeMixin

logger = get_logger("env_server")

#: OpenETA sim/envs/genesis/tasks: the tasks and their text (cube_pick is the only one).
TASKS = {"cube_pick": "Pick up the red cube from the table and lift it."}
CAMERAS = ("agentview", "wrist")
#: OpenETA tasks/cube_pick.py: the Genesis Franka MJCF, its 7 + 2 dofs and home pose.
FRANKA_MJCF = "xml/franka_emika_panda/panda.xml"
MOTOR_DOFS = list(range(7))
FINGER_DOFS = [7, 8]
HOME_QPOS = [0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8, 0.04, 0.04]
#: Panda hand -> TCP (the point between the fingertip pads), m along the hand's +z.
TCP_OFFSET_M = 0.1034
#: One finger's slide when open, m (the width is twice it).
FINGER_OPEN_M = 0.04
#: A closed gripper at or below this width holds nothing, m.
EMPTY_WIDTH_M = 0.005
#: OpenETA's cube: 4 cm, sampled over the reachable table in front of the robot.
CUBE_SIZE_M = 0.04
CUBE_X = (0.45, 0.75)
CUBE_Y = (-0.25, 0.25)
#: Success: the cube's bottom face this far above the table for SUCCESS_HOLD_STEPS control
#: steps (a lift, not a graze; OpenETA's grasp+distance hold counter replaced by a height).
LIFT_M = 0.08
SUCCESS_HOLD_STEPS = 5
#: Motion limits, checked before anything moves (the TCP must stay inside).
WORKSPACE = {"min": [0.25, -0.40, 0.012], "max": [0.85, 0.40, 0.60]}
Z_FLOOR_M = WORKSPACE["min"][2]
MAX_MOVE_M = 0.2
#: Metres per decision (Show-Harness's 2 cm convention) and the servo per decision.
STEP_M = 0.02
SERVO = {"tol_m": 0.002, "min_steps": 3, "max_steps": 20}
#: Extra control steps at the end of a move to close the PD lag on the final target.
FINAL_STEPS = 40
#: The servo's integral correction: the IK target is offset by this share of the remaining
#: error each step (the PD position controller sags under gravity), within OFFSET_MAX_M.
OFFSET_GAIN = 0.5
OFFSET_MAX_M = 0.03
GRIPPER_STEPS = 25
SETTLE_STEPS = 20
#: Fewest front-camera pixels the task object may show at reset (a 4 cm cube at the far
#: edge of the box covers ~120 of 256x256).
MIN_VISIBLE_PX = 20
#: Front camera: in front of the table facing the robot (base at the image top, image
#: right = +y), close enough for a 4 cm cube to cover ~15 px; both views 256x256.
AGENTVIEW = {"pos": [1.05, 0.25, 0.65], "lookat": [0.50, 0.0, 0.10], "fov_deg": 45.0}
VIEW_SIZE = 256
#: Wrist camera: on the hand, 5 cm toward the +y finger, looking along the hand's +z (the
#: approach direction); image up = the hand's -y, so the fingertips sit at the top edge.
WRIST_OFFSET = [0.0, 0.05, 0.0]
WRIST_FOV_DEG = 90.0
_WRIST_R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def _np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def check_target(start, delta) -> np.ndarray:
    """The TCP target of a base-frame move, or a ValueError naming the limit it breaks:
    the per-call cap, the Z floor, the workspace box. Nothing moves on a refusal."""
    start = np.asarray(start, dtype=np.float64).reshape(3)
    delta = np.asarray(delta, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(delta))
    if not norm <= MAX_MOVE_M:
        raise ValueError(
            f"delta moves {norm:.3f} m; the limit is {MAX_MOVE_M} m per call. Split the motion."
        )
    target = start + delta
    if target[2] < Z_FLOOR_M:
        raise ValueError(
            f"target z {target[2]:.3f} is below the floor {Z_FLOOR_M} m (the TCP would hit the table)"
        )
    lo, hi = np.asarray(WORKSPACE["min"]), np.asarray(WORKSPACE["max"])
    if np.any(target < lo) or np.any(target > hi):
        raise ValueError(
            f"target {np.round(target, 3).tolist()} leaves the workspace box "
            f"{WORKSPACE['min']}..{WORKSPACE['max']} m"
        )
    return target


def waypoints(start, target, step_m: float = STEP_M) -> list[np.ndarray]:
    """Evenly spaced ~step_m waypoints from start to target (at least one)."""
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = max(1, int(np.ceil(np.linalg.norm(target - start) / step_m - 1e-9)))
    return [start + (target - start) * (i + 1) / n for i in range(n)]


def grasped(contacts: dict, finger_links: tuple[int, int], width: float) -> bool:
    """Both fingers touch the object (its contact pairs name each finger link) and the
    gripper is not fully open."""
    links = np.concatenate(
        [
            _np(contacts.get("link_a", [])).reshape(-1),
            _np(contacts.get("link_b", [])).reshape(-1),
        ]
    )
    return (
        all(int(f) in set(links.astype(int).tolist()) for f in finger_links)
        and width < 2 * FINGER_OPEN_M - 1e-3
    )


def lifted(
    cube_z: float, half: float = CUBE_SIZE_M / 2, lift_m: float = LIFT_M
) -> bool:
    """The cube's bottom face is at least lift_m above the table (z = 0)."""
    return bool(cube_z - half >= lift_m)


def segmentation_index(seg_idx_dict: dict, entity_idx: int) -> int | None:
    """The value Genesis's segmentation image gives an entity: not ``entity.idx`` but the
    index the renderer assigned when it registered the entity's geoms (``seg_idxc``: 0 is the
    background, then 1, 2, ... in registration order). ``scene.segmentation_idx_dict`` maps
    that index to the entity idx at ``segmentation_level="entity"`` (a tuple at the link and
    geom levels); None when the entity was never rendered."""
    for idxc, key in seg_idx_dict.items():
        if isinstance(key, tuple):
            key = key[0]
        if key == entity_idx:
            return int(idxc)
    return None


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """Equal-ratio resize into a ``size`` square with centred black bars."""
    from PIL import Image

    h, w = image.shape[:2]
    if h == size and w == size:
        return np.ascontiguousarray(image)
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = np.asarray(Image.fromarray(image).resize((nw, nh), Image.BILINEAR))
    out = np.zeros((size, size, 3), dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def cam2world_cv(transform_gl: np.ndarray) -> np.ndarray:
    """Genesis camera transform (OpenGL: looks along -z, +y up) -> OpenCV camera-to-world
    (+z forward, +y down), the inverse of ``Camera.extrinsics``."""
    t = np.asarray(transform_gl, dtype=np.float64).copy()
    t[:3, 1:3] *= -1
    return t


def back_project(
    depth: np.ndarray, K: np.ndarray, cam2world: np.ndarray, pixels
) -> list:
    """World xyz of each (row, col) pixel from a metric depth image (null where the depth
    is missing or the pixel is out of the image)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    out: list = []
    h, w = depth.shape[:2]
    for row, col in pixels:
        r, c = int(row), int(col)
        if not (0 <= r < h and 0 <= c < w):
            out.append(None)
            continue
        z = float(depth[r, c])
        if not np.isfinite(z) or z <= 0:
            out.append(None)
            continue
        p = cam2world @ np.array([(c - cx) * z / fx, (r - cy) * z / fy, z, 1.0])
        out.append([round(float(v), 5) for v in p[:3]])
    return out


class GenesisEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """One Genesis scene (no batch dimension); every call runs on the main thread."""

    SERVICE_NAME = "genesis-env"

    def __init__(
        self,
        *,
        task: str = "cube_pick",
        seed: int = 0,
        backend: str = "gpu",
        dt: float = 0.01,
        substeps: int = 2,
        view_size: int = VIEW_SIZE,
    ):
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; one of {sorted(TASKS)}")
        import genesis as gs
        import torch

        self._gs, self._torch = gs, torch
        if not getattr(gs, "_initialized", False):
            gs.init(
                backend={"gpu": gs.gpu, "cuda": gs.cuda, "cpu": gs.cpu}[backend],
                precision="32",
                logging_level="warning",
            )
        self._task = task
        self._seed = int(seed)
        self._view_size = int(view_size)
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=float(dt), substeps=int(substeps)),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            vis_options=gs.options.VisOptions(segmentation_level="entity"),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=False,
        )
        self._scene.add_entity(gs.morphs.Plane())
        self._robot = self._scene.add_entity(gs.morphs.MJCF(file=FRANKA_MJCF))
        self._cube = self._scene.add_entity(
            gs.morphs.Box(size=(CUBE_SIZE_M,) * 3, pos=(0.65, 0.0, CUBE_SIZE_M / 2)),
            surface=gs.surfaces.Default(color=(0.85, 0.1, 0.1)),
        )
        self._cams = {
            "agentview": self._scene.add_camera(
                res=(self._view_size, self._view_size),
                pos=AGENTVIEW["pos"],
                lookat=AGENTVIEW["lookat"],
                fov=AGENTVIEW["fov_deg"],
                GUI=False,
            ),
            "wrist": self._scene.add_camera(
                res=(self._view_size, self._view_size),
                pos=(0.0, 0.0, 1.0),
                lookat=(0.0, 0.0, 0.0),
                fov=WRIST_FOV_DEG,
                GUI=False,
            ),
        }
        self._scene.build()
        self._hand = self._robot.get_link("hand")
        self._fingers = (
            int(self._robot.get_link("left_finger").idx),
            int(self._robot.get_link("right_finger").idx),
        )
        offset = np.eye(4)
        offset[:3, :3] = _WRIST_R
        offset[:3, 3] = WRIST_OFFSET
        self._cams["wrist"].attach(self._hand, offset)
        # OpenETA tasks/cube_pick.py post_build gains and force limits.
        self._robot.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
        )
        self._robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self._robot.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )
        self._hold_quat: np.ndarray | None = None
        self._offset = np.zeros(3)
        self._gripper_open = True
        self._success = False
        self._hold = 0
        self._steps = 0
        self._closed = False
        self._meta = {
            "task": task,
            "seed": self._seed,
            "instruction": TASKS[task],
            "backend": backend,
            "dt": float(dt),
            "substeps": int(substeps),
            "view_size": self._view_size,
            "agentview": AGENTVIEW,
            "wrist_offset": WRIST_OFFSET,
            "wrist_fov_deg": WRIST_FOV_DEG,
            "step_m": STEP_M,
            "workspace": WORKSPACE,
            "z_floor_m": Z_FLOOR_M,
            "max_move_m": MAX_MOVE_M,
            "lift_m": LIFT_M,
            "empty_width_m": EMPTY_WIDTH_M,
            "objects": ["cube"],
        }

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["env.move_delta"] = self.move_delta
        self._rpc["env.set_gripper"] = self.set_gripper
        self._rpc["env.state"] = self.state
        self._rpc["env.back_project"] = self.back_project
        self._rpc["env.ground_truth_poses"] = self.ground_truth_poses
        register_code_api(self, GENESIS_PRIMITIVES)

    # ---- kinematics ----

    def _hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            _np(self._hand.get_pos()).reshape(3).astype(np.float64),
            _np(self._hand.get_quat()).reshape(4).astype(np.float64),
        )

    @staticmethod
    def _rot(q_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = q_wxyz
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    def _tcp(self) -> np.ndarray:
        p, q = self._hand_pose()
        return p + self._rot(q) @ np.array([0.0, 0.0, TCP_OFFSET_M])

    def _qpos(self) -> np.ndarray:
        return _np(self._robot.get_dofs_position()).reshape(-1).astype(np.float64)

    def _width(self) -> float:
        q = self._qpos()
        return float(q[FINGER_DOFS[0]] + q[FINGER_DOFS[1]])

    def _command_arm(self, target_tcp: np.ndarray) -> None:
        """IK to put the TCP at ``target_tcp`` with the reset orientation held; command it."""
        q = self._robot.inverse_kinematics(
            link=self._hand,
            pos=self._torch.as_tensor(target_tcp, dtype=self._torch.float32),
            quat=self._torch.as_tensor(self._hold_quat, dtype=self._torch.float32),
            local_point=[0.0, 0.0, TCP_OFFSET_M],
            dofs_idx_local=MOTOR_DOFS,
        )
        self._robot.control_dofs_position(q[: len(MOTOR_DOFS)], MOTOR_DOFS)

    def _command_gripper(self) -> None:
        w = FINGER_OPEN_M if self._gripper_open else 0.0
        self._robot.control_dofs_position(
            self._torch.tensor([w, w], dtype=self._torch.float32), FINGER_DOFS
        )

    def _step(self) -> None:
        self._scene.step()
        self._steps += 1
        cube_z = float(_np(self._cube.get_pos()).reshape(3)[2])
        self._hold = self._hold + 1 if lifted(cube_z) else 0
        self._success = self._success or self._hold >= SUCCESS_HOLD_STEPS

    # ---- observation ----

    def _render(self, name: str, depth: bool = False):
        cam = self._cams[name]
        if name == "wrist":
            cam.move_to_attach()
        rgb, d, _seg, _n = cam.render(rgb=True, depth=depth)
        rgb = _np(rgb)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        rgb = rgb.astype(np.uint8)
        return (rgb, _np(d).astype(np.float32)) if depth else rgb

    def _grasped(self) -> bool:
        return grasped(
            self._robot.get_contacts(with_entity=self._cube),
            self._fingers,
            self._width(),
        )

    def _state(self) -> dict:
        p, q = self._hand_pose()
        cube_z = float(_np(self._cube.get_pos()).reshape(3)[2])
        return {
            "tcp_pos": self._tcp().astype(np.float32),
            "tcp_quat_wxyz": q.astype(np.float32),
            "gripper_width": round(self._width(), 5),
            "gripper_command": "open" if self._gripper_open else "close",
            "qpos": self._qpos().astype(np.float32),
            "success": bool(self._success),
            "is_grasped": self._grasped(),
            "lift_m": round(max(0.0, cube_z - CUBE_SIZE_M / 2), 4),
            "env_steps": self._steps,
        }

    def _obs(self) -> dict:
        return {
            "agentview": letterbox(self._render("agentview"), self._view_size),
            "wrist": letterbox(self._render("wrist"), self._view_size),
            **self._state(),
        }

    def _frame(self) -> np.ndarray:
        """The episode video's frame: the two views side by side."""
        return np.concatenate(
            [
                letterbox(self._render("agentview"), self._view_size),
                letterbox(self._render("wrist"), self._view_size),
            ],
            axis=1,
        )

    def visible_pixels(self) -> dict:
        """Front-camera pixels of each task object (Genesis's entity segmentation)."""
        _rgb, _d, seg, _n = self._cams["agentview"].render(rgb=False, segmentation=True)
        seg = _np(seg)
        idx = segmentation_index(self._scene.segmentation_idx_dict, int(self._cube.idx))
        if idx is None:
            raise RuntimeError("the cube has no segmentation index (not rendered)")
        return {"cube": int((seg == idx).sum())}

    def check_visible(self) -> None:
        """Refuse an episode whose front view does not show the task object."""
        px = self.visible_pixels()
        self._meta["visible_px"] = px
        hidden = {k: v for k, v in px.items() if v < MIN_VISIBLE_PX}
        if hidden:
            raise RuntimeError(
                f"task objects not visible in the agentview after reset: {hidden} px "
                f"(need >= {MIN_VISIBLE_PX}); refusing the episode"
            )

    # ---- gym-like surface ----

    def reset(self, seed: int | None = None):
        """Reset to ``seed`` (default: the launch seed): the robot at the home pose with the
        gripper open, the cube at the seed's place on the table, SETTLE_STEPS of physics."""
        seed = self._seed if seed is None else int(seed)
        rng = np.random.default_rng(seed)
        self._scene.reset()
        self._success, self._hold, self._steps = False, 0, 0
        self._offset = np.zeros(3)
        self._gripper_open = True
        t = self._torch
        home = t.tensor(HOME_QPOS, dtype=t.float32)
        self._robot.set_qpos(home, zero_velocity=True)
        self._robot.control_dofs_position(home[: len(MOTOR_DOFS)], MOTOR_DOFS)
        self._command_gripper()
        xy = [rng.uniform(*CUBE_X), rng.uniform(*CUBE_Y)]
        self._cube.set_pos(t.tensor([*xy, CUBE_SIZE_M / 2], dtype=t.float32))
        self._cube.set_quat(t.tensor([1.0, 0.0, 0.0, 0.0], dtype=t.float32))
        for _ in range(SETTLE_STEPS):
            self._scene.step()
        self._hold_quat = self._hand_pose()[1]
        self._meta["layout"] = {"cube": [round(float(v), 4) for v in xy]}
        self.check_visible()
        return self._obs(), {"instruction": TASKS[self._task], "seed": seed}

    def _apply(self, action) -> None:
        a = np.asarray(action, dtype=np.float64).reshape(4)
        target = check_target(self._tcp(), a[:3])
        self._gripper_open = a[3] >= 0
        self._command_gripper()
        self._command_arm(target)
        self._step()

    def step(self, action):
        """One control step of ``[dx, dy, dz, gripper]`` (m, +1 open / -1 close)."""
        self._apply(action)
        return self._obs(), 0.0, bool(self._success), False, {"success": self._success}

    def chunk_step(self, actions, *, return_all_frames: bool = False):
        frames: list = []
        info: dict = {}
        for action in np.asarray(actions, dtype=np.float64).reshape(-1, 4):
            if self.stop_requested():
                info["cancelled"] = True
                break
            self._apply(action)
            frames.append(self._obs() if return_all_frames else None)
            if self._success:
                break
        n = len(frames)
        last = self._obs()
        return (
            frames if return_all_frames else last,
            np.zeros(n, dtype=np.float32),
            np.array([self._success] * n, dtype=bool),
            np.zeros(n, dtype=bool),
            {"success": self._success, **info},
        )

    def _servo(
        self, target: np.ndarray, max_steps: int = SERVO["max_steps"]
    ) -> tuple[int, bool]:
        """Drive the TCP to ``target`` in SERVO['min_steps']..``max_steps`` control steps
        (IK re-solved each step); returns (steps, cancelled)."""
        for k in range(max_steps):
            if self.stop_requested():
                return k, True
            err = target - self._tcp()
            if k >= SERVO["min_steps"] and np.linalg.norm(err) < SERVO["tol_m"]:
                return k, False
            self._offset = np.clip(
                self._offset + OFFSET_GAIN * err, -OFFSET_MAX_M, OFFSET_MAX_M
            )
            self._command_arm(target + self._offset)
            self._command_gripper()
            self._step()
        return max_steps, False

    def _grip(self) -> tuple[int, bool]:
        self._command_arm(self._tcp() + self._offset)
        self._command_gripper()
        for k in range(GRIPPER_STEPS):
            if self.stop_requested():
                return k, True
            self._step()
        return GRIPPER_STEPS, False

    def move_delta(
        self, delta_xyz, *, gripper: str | None = None, return_frames: bool = False
    ):
        """Translate the TCP by a base-frame delta (m), after an optional gripper command
        (the arm holds still while the fingers settle). Refused (nothing moves) beyond
        MAX_MOVE_M, below Z_FLOOR_M or outside WORKSPACE. Returns the observation plus
        commanded_m, moved_m, decisions, control_steps[, frames, cancelled]."""
        start = self._tcp()
        target = check_target(start, delta_xyz)
        if gripper not in (None, "open", "close"):
            raise ValueError(
                f"gripper must be 'open', 'close' or null, not {gripper!r}"
            )
        frames: list = []
        steps = 0
        cancelled = False
        decisions = 0
        if gripper is not None and (gripper == "open") != self._gripper_open:
            self._gripper_open = gripper == "open"
            n, cancelled = self._grip()
            steps += n
            decisions += 1
            if return_frames:
                frames.append(self._frame())
        if not cancelled and np.linalg.norm(target - start) > 0:
            for wp in waypoints(start, target):
                n, cancelled = self._servo(wp)
                steps += n
                decisions += 1
                if return_frames:
                    frames.append(self._frame())
                if cancelled or self._success:
                    break
            if not cancelled and not self._success:
                # The PD lags the 2 cm waypoints; settle on the final target.
                n, cancelled = self._servo(target, FINAL_STEPS)
                steps += n
        end = self._tcp()
        out = {
            **self._obs(),
            "commanded_m": [
                round(float(v), 4) for v in np.asarray(delta_xyz, dtype=np.float64)
            ],
            "moved_m": [round(float(v), 4) for v in end - start],
            "decisions": decisions,
            "control_steps": steps,
        }
        if return_frames:
            out["frames"] = frames
        if cancelled:
            out["cancelled"] = True
        return out

    def set_gripper(self, *, open: bool, return_frames: bool = False):
        """Open or close the gripper and hold GRIPPER_STEPS; a close that ends at or below
        EMPTY_WIDTH_M reports ``grasp_empty``."""
        self._gripper_open = bool(open)
        n, cancelled = self._grip()
        out = {**self._obs(), "control_steps": n}
        if return_frames:
            out["frames"] = [self._frame()]
        if not open and self._width() <= EMPTY_WIDTH_M:
            out["grasp_empty"] = True
        if cancelled:
            out["cancelled"] = True
        return out

    def state(self) -> dict:
        """The observation without images (no stepping)."""
        return self._state()

    def render_camera(
        self, camera_name: str = "agentview", depth: bool = False, **_: Any
    ):
        """The current frame of ``agentview`` or ``wrist`` (as the model sees it), or
        ``[rgb, depth_m]`` with the metric depth."""
        if camera_name not in CAMERAS:
            raise ValueError(f"unknown camera {camera_name!r}; one of {CAMERAS}")
        if depth:
            rgb, d = self._render(camera_name, depth=True)
            return [letterbox(rgb, self._view_size), d]
        return letterbox(self._render(camera_name), self._view_size)

    def get_camera_meta(self, camera_name: str = "agentview", **_: Any) -> dict:
        """OpenCV intrinsics and camera-to-world extrinsic of a camera at its own
        resolution (the images are square, so none of the letterbox applies)."""
        if camera_name not in CAMERAS:
            raise ValueError(f"unknown camera {camera_name!r}; one of {CAMERAS}")
        cam = self._cams[camera_name]
        if camera_name == "wrist":
            cam.move_to_attach()
        return {
            "intrinsic_K": np.asarray(cam.intrinsics, dtype=np.float64),
            "extrinsic_cam2world": cam2world_cv(cam.transform),
            "width": int(cam.res[0]),
            "height": int(cam.res[1]),
        }

    def back_project(self, camera_name: str = "agentview", pixels=None) -> list:
        """World xyz of ``pixels`` [[row, col], ...] of the current camera image (null where
        the depth is missing)."""
        if camera_name not in CAMERAS:
            raise ValueError(f"unknown camera {camera_name!r}; one of {CAMERAS}")
        _rgb, depth = self._render(camera_name, depth=True)
        meta = self.get_camera_meta(camera_name)
        return back_project(
            depth, meta["intrinsic_K"], meta["extrinsic_cam2world"], pixels or []
        )

    def ground_truth_poses(self, names=None) -> dict:
        """World poses of the task objects (``--privileged``)."""
        return ground_truth.respond(
            {
                "cube": ground_truth.pose(
                    _np(self._cube.get_pos()).reshape(3),
                    _np(self._cube.get_quat()).reshape(4),
                )
            },
            names,
        )

    def get_task_language(self) -> str:
        return TASKS[self._task]

    def get_env_meta(self) -> dict:
        return dict(self._meta)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._scene.destroy()
        finally:
            self._gs.destroy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--task", choices=sorted(TASKS), default="cube_pick")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["gpu", "cuda", "cpu"], default="gpu")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--substeps", type=int, default=2)
    p.add_argument("--view-size", type=int, default=VIEW_SIZE)
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    args = p.parse_args()

    facade = GenesisEnvFacade(
        task=args.task,
        seed=args.seed,
        backend=args.backend,
        dt=args.dt,
        substeps=args.substeps,
        view_size=args.view_size,
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
