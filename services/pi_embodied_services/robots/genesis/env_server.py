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

Success (``--success-rule``, recorded in the env meta and pi's result): ``grasp`` (default) is
OpenETA's own cube_pick rule: both fingers in contact, neither finger fully open, and the EEF
point (0.11 m along the hand's z) within 0.08 m of the cube's centre, for 3 consecutive control
steps. ``lift`` is the stricter rule this port used before: the cube's bottom face 8 cm above the
table for 5 control steps.

Joint space (CaP-X's reduced API): ``env.solve_ik`` (Genesis's IK for a TCP pose, nothing moves),
``env.move_to_joints`` (a PD servo to a 7-joint target, stopped when the TCP would leave the
workspace box or go below the Z floor), ``env.traj_plan`` (IK waypoints along a straight Cartesian
path) and ``env.move_along_trajectory``.

Code mode (``code.run``, utils/code_exec.py ``CodeRunMixin``): a program calls the primitives of
the robot's manifest (packages/embodied/src/primitives/manifests/genesis.json, read by
components/manifest.py) in a sandboxed subprocess; they reach the same facade methods on the
same (main) thread as the tools' RPCs. What a primitive hands the program drops the camera images
(they go to the run's video) and the cube's height (``lift_m``: object state no camera measures);
the run reports its control steps, the latched success and the new observation (``_finish_run``).
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np

from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.utils import grasp_chain as chain
from pi_embodied_services.utils import ground_truth, reach, sam3_segment
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

logger = get_logger("env_server")

#: OpenETA sim/envs/genesis/tasks: the tasks and their text (cube_pick is the only one).
#: The text states the success rule's condition (``--success-rule``).
TASKS = {"cube_pick": "Pick up the red cube from the table."}
TASK_TEXT = {
    ("cube_pick", "grasp"): "Pick up the red cube from the table.",
    ("cube_pick", "lift"): "Pick up the red cube from the table and lift it.",
}
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
#: Success rules (``--success-rule``). ``grasp``: OpenETA tasks/cube_pick.py at 7d4a0a1 (its
#: compute_step_outcomes): both finger links in contact, both finger joints below fully open,
#: the EEF point (hand + 0.11 m along its z) within GRASP_DIST_M of the cube centre, held for
#: GRASP_HOLD_STEPS consecutive control steps. ``lift``: the cube's bottom face LIFT_M above the
#: table for SUCCESS_HOLD_STEPS control steps (this port's earlier, stricter rule).
SUCCESS_RULES = ("grasp", "lift")
UPSTREAM_EEF_OFFSET_M = 0.11
GRASP_DIST_M = 0.08
GRASP_HOLD_STEPS = 3
LIFT_M = 0.08
SUCCESS_HOLD_STEPS = 5
#: Panda joint limits (rad) and the joint servo's defaults (move_to_joints).
JOINT_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
JOINT_TOL_RAD = 0.01
JOINT_MAX_STEPS = 200
#: traj_plan: Cartesian spacing of its IK waypoints, m, and the most waypoints one plan returns.
TRAJ_STEP_M = 0.02
TRAJ_MAX_POINTS = 50
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
#: Code mode: video frames one run hands back (halved, every other one kept, when full), the most
#: actions one chunk_step call may take, and the state a program never receives (the cube's height).
CODE_MAX_FRAMES = 128
CODE_MAX_CHUNK = 200
CODE_HIDDEN = ("lift_m",)
#: The facade's motions: grasp and detection ids expire after any of them.
MOTIONS = (
    "env.move_delta",
    "env.set_gripper",
    "env.move_to_joints",
    "env.move_along_trajectory",
    "env.execute_grasp",
    "env.execute_place",
    "env.step",
    "env.chunk_step",
)


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


def upstream_grasped(
    finger_contact: tuple[bool, bool], finger_q, eef_cube_dist: float
) -> bool:
    """OpenETA cube_pick's ``success_instant``: both fingers touch something, neither finger
    joint is fully open (< FINGER_OPEN_M), and the EEF point is within GRASP_DIST_M of the cube."""
    q = np.asarray(finger_q, dtype=np.float64).reshape(2)
    return bool(
        finger_contact[0]
        and finger_contact[1]
        and q[0] < FINGER_OPEN_M
        and q[1] < FINGER_OPEN_M
        and eef_cube_dist < GRASP_DIST_M
    )


def finger_contacts(contacts: dict, finger_links: tuple[int, int]) -> tuple[bool, bool]:
    """Whether each finger link is in any contact (OpenETA reads a Contact sensor per finger)."""
    links = set(
        np.concatenate(
            [
                _np(contacts.get("link_a", [])).reshape(-1),
                _np(contacts.get("link_b", [])).reshape(-1),
            ]
        )
        .astype(int)
        .tolist()
    )
    return (int(finger_links[0]) in links, int(finger_links[1]) in links)


def slerp(q0, q1, t: float) -> np.ndarray:
    """Spherical interpolation of two unit quaternions (any component order)."""
    a = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
    b = np.asarray(q1, dtype=np.float64) / np.linalg.norm(q1)
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        out = a + t * (b - a)
        return out / np.linalg.norm(out)
    theta = np.arccos(dot)
    return (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)


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


class GenesisEnvFacade(CodeRunMixin, MainThreadServeMixin, BaseEnvFacade):
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
        success_rule: str = "grasp",
    ):
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; one of {sorted(TASKS)}")
        if success_rule not in SUCCESS_RULES:
            raise ValueError(
                f"unknown success rule {success_rule!r}; one of {SUCCESS_RULES}"
            )
        self._rule = success_rule
        #: --sam3 (env.segment), --ik (env.preview_reach), the grasp planner: set in main().
        self._sam3 = sam3_segment.Sam3(None)
        self._reach = None
        self._grasp: GraspPlanner | None = None
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
        #: The TCP point the arm was last commanded to (``_command_arm``).
        self._cmd_tcp: np.ndarray | None = None
        #: The Flywheel records of the motion call in progress (``record=True``), else None.
        self._record: list | None = None
        self._offset = np.zeros(3)
        self._gripper_open = True
        self._success = False
        self._hold = 0
        self._steps = 0
        self._closed = False
        # Code mode: the control steps before the run, and the run's video frames.
        self._run_start = 0
        self._run_frames: list[np.ndarray] = []
        self._meta = {
            "task": task,
            "seed": self._seed,
            "instruction": TASK_TEXT[(task, success_rule)],
            "success_rule": success_rule,
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
        self._rpc["env.segment"] = self.segment
        self._rpc["env.ground_truth_poses"] = self.ground_truth_poses
        self._rpc["env.solve_ik"] = self.solve_ik
        self._rpc["env.move_to_joints"] = self.move_to_joints
        self._rpc["env.traj_plan"] = self.traj_plan
        self._rpc["env.move_along_trajectory"] = self.move_along_trajectory
        # Served always (so the perception and planner wrappers expire ids after them); they
        # need the grasp planner (--contact-graspnet & co), which code.api requires too.
        self._rpc["env.execute_grasp"] = self.execute_grasp
        self._rpc["env.execute_place"] = self.execute_place
        self._readonly_methods.update({"env.solve_ik", "env.traj_plan"})
        # The primitives are packages/embodied/src/primitives/manifests/genesis.json (with pi);
        # code.api, the programs' whitelist and the startup self-check come from it.
        self._manifest_code_run(
            "genesis",
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
            "ik": self._reach is not None,
            "grasp": self._grasp is not None,
            "place": self._grasp is not None
            and bool(self._grasp.capabilities().get("place")),
            "unidepth": "env.enhance_depth" in self._rpc,
            "privileged": True,
        }.get(capability, False)

    def install_grasp(self, planner: GraspPlanner | None) -> None:
        """The grasp planner's primitives, and the chains that run its ids here."""
        self._grasp = planner
        if planner is None:
            return
        planner.install(self, mutating=GraspPlanner.MUTATING + MOTIONS)

    # ---- code mode (run_code) ----

    def _begin_run(self) -> None:
        self._run_start = self._steps
        self._run_frames = []

    def _finish_run(self) -> dict:
        """The run's effect for pi: control steps taken, the latched success, the new
        observation (the tools' ``obs``) and the run's video frames."""
        return {
            "steps": self._steps - self._run_start,
            "success": bool(self._success),
            "obs": self._obs(),
            "frames": list(self._run_frames),
        }

    def _keep_frame(self, obs: dict) -> None:
        """The run video's frame of an observation: the two views side by side (as ``_frame``)."""
        if len(self._run_frames) >= CODE_MAX_FRAMES:
            self._run_frames = self._run_frames[::2]
        self._run_frames.append(
            np.concatenate([obs["agentview"], obs["wrist"]], axis=1)
        )

    def _program_state(self, obs: dict) -> dict:
        """An observation as a program receives it: no images (they go to the run's video)
        and no object state."""
        self._keep_frame(obs)
        return {
            k: v
            for k, v in obs.items()
            if k not in CAMERAS and k not in CODE_HIDDEN and k != "frames"
        }

    def _code_reply(self, method: str, out: Any) -> Any:
        """What a program receives of a primitive's result: the same facade method as the tools
        call, without the images and the cube's height."""
        if method in (
            "env.move_delta",
            "env.set_gripper",
            "env.move_to_joints",
            "env.move_along_trajectory",
            "env.execute_grasp",
            "env.execute_place",
        ):
            return self._program_state(out)
        if method == "env.segment":
            return sam3_segment.for_program(out)
        if method == "env.state":
            return {k: v for k, v in out.items() if k not in CODE_HIDDEN}
        if method == "env.step":
            obs, rew, success, truncated, _info = out
            return {
                "reward": rew,
                "success": success,
                "truncated": truncated,
                "state": self._program_state(obs),
            }
        if method == "env.chunk_step":
            obs, _rew, _success, _trunc, info = out
            states = (
                [self._program_state(o) for o in obs]
                if isinstance(obs, list)
                else self._program_state(obs)
            )
            key = "states" if isinstance(obs, list) else "state"
            return {"success": bool(info.get("success")), **info, key: states}
        return out

    def _code_move_m(self, method: str, kwargs: dict) -> float:
        """How far a program's call may move the TCP (the run's translation cap)."""
        if method == "env.move_delta":
            d = np.asarray(kwargs["delta_xyz"], dtype=np.float64).reshape(3)
            return float(np.linalg.norm(d))
        if method == "env.step":
            a = np.asarray(kwargs["action"], dtype=np.float64).reshape(4)
            return float(np.linalg.norm(a[:3]))
        if method == "env.chunk_step":
            a = np.asarray(kwargs["actions"], dtype=np.float64).reshape(-1, 4)
            return float(np.linalg.norm(a[:, :3], axis=1).sum())
        return 0.0

    def _code_check(self, method: str, kwargs: dict) -> None:
        """Refuse a program's call that the run's wall clock could not bound."""
        if method == "env.chunk_step":
            n = np.asarray(kwargs["actions"], dtype=np.float64).reshape(-1, 4).shape[0]
            if n > CODE_MAX_CHUNK:
                raise ValueError(
                    f"chunk_step takes at most {CODE_MAX_CHUNK} actions per call in code mode"
                )

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
        self._cmd_tcp = np.asarray(target_tcp, dtype=np.float64).reshape(3)

    def _command_gripper(self) -> None:
        w = FINGER_OPEN_M if self._gripper_open else 0.0
        self._robot.control_dofs_position(
            self._torch.tensor([w, w], dtype=self._torch.float32), FINGER_DOFS
        )

    def _step(self) -> None:
        # While recording: the step as ``env.step`` takes it, the commanded TCP point minus the
        # TCP before the step, and the gripper (+1 open / -1 close).
        action = None
        if self._record is not None:
            cmd = self._tcp() if self._cmd_tcp is None else self._cmd_tcp
            action = np.append(cmd - self._tcp(), 1.0 if self._gripper_open else -1.0)
        self._scene.step()
        self._steps += 1
        self._hold = self._hold + 1 if self._success_instant() else 0
        need = GRASP_HOLD_STEPS if self._rule == "grasp" else SUCCESS_HOLD_STEPS
        self._success = self._success or self._hold >= need
        if action is not None:
            self._record.append({**self._obs(), "action": action.astype(np.float32)})

    def _success_instant(self) -> bool:
        """This control step's success signal under the episode's rule (``--success-rule``)."""
        cube = _np(self._cube.get_pos()).reshape(3).astype(np.float64)
        if self._rule == "lift":
            return lifted(float(cube[2]))
        p, q = self._hand_pose()
        eef = p + self._rot(q) @ np.array([0.0, 0.0, UPSTREAM_EEF_OFFSET_M])
        return upstream_grasped(
            finger_contacts(self._robot.get_contacts(), self._fingers),
            self._qpos()[FINGER_DOFS],
            float(np.linalg.norm(eef - cube)),
        )

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

    def _hand_yaw(self) -> float:
        """The hand's yaw (the planner's eef_yaw: the EEF x axis about world +z)."""
        _, q = self._hand_pose()
        w, x, y, z = (float(v) for v in np.asarray(q).reshape(4))
        return math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))

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
        self._cmd_tcp = None
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
        return self._obs(), {"instruction": self._meta["instruction"], "seed": seed}

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
        self,
        delta_xyz,
        *,
        gripper: str | None = None,
        return_frames: bool = False,
        record: bool = False,
    ):
        """Translate the TCP by a base-frame delta (m), after an optional gripper command
        (the arm holds still while the fingers settle). Refused (nothing moves) beyond
        MAX_MOVE_M, below Z_FLOOR_M or outside WORKSPACE. Returns the observation plus
        commanded_m, moved_m, decisions, control_steps[, frames, steps, cancelled];
        ``record`` adds ``steps``: every control step's observation with its ``action``."""
        start = self._tcp()
        target = check_target(start, delta_xyz)
        if (
            getattr(self, "_reach", None) is not None
            and np.linalg.norm(target - start) > 0
        ):
            # --ik: a target the ik service cannot reach is refused before anything moves.
            reach.require_reachable(
                self._rpc["env.preview_reach"](target.tolist()), "move_delta"
            )
        if gripper not in (None, "open", "close"):
            raise ValueError(
                f"gripper must be 'open', 'close' or null, not {gripper!r}"
            )
        self._record = [] if record else None
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
        if record:
            out["steps"] = self._record
        self._record = None
        if cancelled:
            out["cancelled"] = True
        return out

    def set_gripper(
        self, close: bool, *, return_frames: bool = False, record: bool = False
    ):
        """Close (``close=True``) or open the gripper and hold GRIPPER_STEPS; a close that ends
        at or below EMPTY_WIDTH_M reports ``grasp_empty``. ``record`` as ``move_delta``."""
        open = not bool(close)
        self._gripper_open = open
        self._record = [] if record else None
        n, cancelled = self._grip()
        out = {**self._obs(), "control_steps": n}
        if record:
            out["steps"] = self._record
        self._record = None
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

    def _locate(self, camera: str, pixels) -> list:
        if camera not in CAMERAS:
            raise ValueError(f"unknown camera {camera!r}; one of {CAMERAS}")
        _rgb, depth = self._render(camera, depth=True)
        meta = self.get_camera_meta(camera)
        return back_project(
            depth, meta["intrinsic_K"], meta["extrinsic_cam2world"], pixels
        )

    def back_project(
        self,
        row: int | None = None,
        col: int | None = None,
        camera: str = "agentview",
        pixels=None,
    ):
        """World xyz (m, base frame) of pixel (row, col) of the current camera image through
        the simulator's depth: ``{camera, pixel, world_xyz}``, or ``error`` where there is no
        depth. ``pixels`` [[row, col], ...] instead returns a list (null where no depth)."""
        if pixels is not None:
            return self._locate(camera, pixels)
        if row is None or col is None:
            raise ValueError("give row and col (or pixels)")
        [p] = self._locate(camera, [[row, col]])
        out: dict[str, Any] = {"camera": camera, "pixel": [int(row), int(col)]}
        if p is None:
            out["error"] = "no depth at that pixel (background or out of the image)"
        else:
            out["world_xyz"] = p
        return out

    def segment(
        self,
        prompt: str | None = None,
        point=None,
        camera: str = "agentview",
        min_score: float = 0.2,
    ) -> dict:
        """SAM3 mask of a text prompt (or a positive point [row, col]) on the current image of
        ``camera``, its pixels back-projected through the depth (at most 400, evenly spread):
        ``found``, ``score``, ``box``, ``mask``, ``n_pixels``, ``centroid_pixel``, ``world_xyz``
        (their median) and the tool's ``overlay_png_base64``."""
        if camera not in CAMERAS:
            raise ValueError(f"unknown camera {camera!r}; one of {CAMERAS}")
        rgb = letterbox(self._render(camera), self._view_size)
        return sam3_segment.segment(
            self._sam3,
            rgb,
            lambda px: self._locate(camera, px),
            prompt=prompt,
            point=point,
            min_score=min_score,
            extra={"camera": camera},
        )

    # ---- planned grasps (utils/grasp_chain.py) ----

    def _chain(
        self, kind: str, grasp_id: str, standoff: float | None, record: bool
    ) -> dict:
        if self._grasp is None:
            raise RuntimeError(
                f"execute_{kind} needs the grasp planner (start with --contact-graspnet & co)"
            )

        recorded: list = []

        def move(delta, g):
            r = self.move_delta(
                list(delta), gripper=g, return_frames=True, record=record
            )
            recorded.extend(r.pop("steps", None) or [])
            return r

        def grip(g):
            r = self.set_gripper(g == "close", return_frames=True, record=record)
            recorded.extend(r.pop("steps", None) or [])
            return r

        out = chain.run_chain(
            kind,
            grasp_id,
            rpc=self._rpc,
            current=self._tcp,
            max_step=MAX_MOVE_M,
            move=move,
            gripper=grip,
            stop=self.stop_requested,
            solved=lambda: self._success,
            standoff=standoff,
            yaw=self._hand_yaw,
        )
        frames, steps = out.pop("frames", []), out.pop("control_steps", 0)
        out = {**self._obs(), **out, "control_steps": steps, "frames": frames}
        if record:
            out["steps"] = recorded
        return out

    def execute_grasp(
        self, grasp_id: str, standoff: float | None = None, *, record: bool = False
    ) -> dict:
        """Run one planned grasp id: open at the standoff back along its approach, descend,
        close, lift, each leg as bounded move_delta calls; refused (nothing moves) unless it
        approaches from nearly straight above."""
        return self._chain("grasp", grasp_id, standoff, record)

    def execute_place(
        self, place_id: str, standoff: float | None = None, *, record: bool = False
    ) -> dict:
        """Run one planned place id: to the pre-place above it, descend, open, retreat."""
        return self._chain("place", place_id, standoff, record)

    # ---- joint space (CaP-X's solve_ik / move_to_joints / traj_plan) ----

    def _ik(self, position, quat_wxyz, init=None) -> np.ndarray:
        t = self._torch
        kwargs: dict[str, Any] = dict(
            link=self._hand,
            pos=t.as_tensor(np.asarray(position, dtype=np.float64), dtype=t.float32),
            quat=t.as_tensor(np.asarray(quat_wxyz, dtype=np.float64), dtype=t.float32),
            local_point=[0.0, 0.0, TCP_OFFSET_M],
            dofs_idx_local=MOTOR_DOFS,
        )
        if init is not None:
            full = self._qpos().copy()
            full[: len(MOTOR_DOFS)] = init
            try:
                q = self._robot.inverse_kinematics(
                    **kwargs, init_qpos=t.as_tensor(full, dtype=t.float32)
                )
            except (
                TypeError
            ):  # a Genesis without init_qpos: it starts from the current joints
                q = self._robot.inverse_kinematics(**kwargs)
        else:
            q = self._robot.inverse_kinematics(**kwargs)
        return _np(q).reshape(-1)[: len(MOTOR_DOFS)].astype(np.float64)

    def solve_ik(self, position, quaternion_wxyz=None) -> list:
        """Joint angles (7, rad) that put the TCP at ``position`` (m, base frame) with the hand
        at ``quaternion_wxyz`` (default: the orientation held since reset); nothing moves."""
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat = (
            self._hold_quat
            if quaternion_wxyz is None
            else np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        )
        return [round(float(v), 5) for v in self._ik(pos, quat)]

    def _check_joints(self, joints) -> np.ndarray:
        q = np.asarray(joints, dtype=np.float64).reshape(-1)
        if q.shape != (len(MOTOR_DOFS),) or not np.all(np.isfinite(q)):
            raise ValueError(f"joints must be {len(MOTOR_DOFS)} finite angles (rad)")
        bad = [i for i in range(len(q)) if not JOINT_LOW[i] <= q[i] <= JOINT_HIGH[i]]
        if bad:
            raise ValueError(f"joint(s) {bad} outside the Panda's limits")
        return q

    def _inside(self, p: np.ndarray) -> bool:
        lo, hi = np.asarray(WORKSPACE["min"]), np.asarray(WORKSPACE["max"])
        return bool(np.all(p >= lo - 1e-6) and np.all(p <= hi + 1e-6))

    def _joint_servo(self, q: np.ndarray, tol_rad: float, max_steps: int) -> dict:
        t = self._torch
        steps, cancelled, error = 0, False, None
        for steps in range(1, int(max_steps) + 1):
            if self.stop_requested():
                cancelled = True
                break
            self._robot.control_dofs_position(
                t.as_tensor(q, dtype=t.float32), MOTOR_DOFS
            )
            self._command_gripper()
            self._step()
            if not self._inside(self._tcp()):
                error = "stopped: the TCP left the workspace box (or went below the Z floor)"
                break
            if np.max(np.abs(self._qpos()[: len(MOTOR_DOFS)] - q)) < tol_rad:
                break
            if self._success:
                break
        # Cartesian motions servo afresh from here (the integral term and command were theirs).
        self._offset = np.zeros(3)
        self._cmd_tcp = None
        if error:
            self._command_arm(self._tcp())
        err = float(np.max(np.abs(self._qpos()[: len(MOTOR_DOFS)] - q)))
        return {
            "control_steps": steps,
            "joint_error_rad": round(err, 4),
            **({"cancelled": True} if cancelled else {}),
            **({"error": error} if error else {}),
        }

    def move_to_joints(
        self,
        joints,
        tol_rad: float = JOINT_TOL_RAD,
        max_steps: int = JOINT_MAX_STEPS,
        return_frames: bool = False,
    ) -> dict:
        """PD servo of the 7 arm joints to ``joints`` (rad) with the gripper held, until every
        joint is within ``tol_rad`` or ``max_steps`` control steps; refused outside the joint
        limits, stopped when the TCP leaves the workspace box. Returns the observation plus
        control_steps and joint_error_rad."""
        if not 1 <= int(max_steps) <= 400:
            raise ValueError("max_steps must be within [1, 400]")
        q = self._check_joints(joints)
        r = self._joint_servo(q, float(tol_rad), int(max_steps))
        return {
            **self._obs(),
            **r,
            **({"frames": [self._frame()]} if return_frames else {}),
        }

    def traj_plan(self, start_pose_wxyz_xyz, end_pose_wxyz_xyz) -> list:
        """Joint waypoints [N, 7] from one TCP pose to another (each ``[qw, qx, qy, qz, x, y,
        z]``): a straight Cartesian line in TRAJ_STEP_M steps (orientation slerped), IK at each
        waypoint seeded with the previous one; nothing moves."""
        a = np.asarray(start_pose_wxyz_xyz, dtype=np.float64).reshape(7)
        b = np.asarray(end_pose_wxyz_xyz, dtype=np.float64).reshape(7)
        n = int(np.ceil(np.linalg.norm(b[4:] - a[4:]) / TRAJ_STEP_M)) + 1
        n = max(2, min(TRAJ_MAX_POINTS, n))
        out: list = []
        q = None
        for i in range(n):
            s = i / (n - 1)
            q = self._ik(a[4:] + s * (b[4:] - a[4:]), slerp(a[:4], b[:4], s), init=q)
            out.append([round(float(v), 5) for v in q])
        return out

    def move_along_trajectory(self, trajectory, return_frames: bool = False) -> dict:
        """move_to_joints through each waypoint of ``trajectory`` [N, 7] (CaP-X: tolerance
        0.025 rad, at most 15 control steps each); stops at an error, a stop or success."""
        traj = [self._check_joints(q) for q in trajectory]
        if len(traj) > 100:
            raise ValueError("at most 100 waypoints per call")
        steps, frames, last = 0, [], {}
        for q in traj:
            last = self._joint_servo(q, 0.025, 15)
            steps += last["control_steps"]
            if return_frames:
                frames.append(self._frame())
            if last.get("error") or last.get("cancelled") or self._success:
                break
        out = {**self._obs(), **last, "control_steps": steps, "waypoints": len(traj)}
        if return_frames:
            out["frames"] = frames
        return out

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


def _xyzw(wxyz) -> np.ndarray:
    q = np.asarray(wxyz, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]])


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
        "--success-rule",
        choices=SUCCESS_RULES,
        default="grasp",
        help="grasp: OpenETA cube_pick's grasp+distance rule (default); lift: the cube 8 cm up",
    )
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    add_perception_arguments(p, sam3=True)
    reach.add_ik_argument(p)
    add_grasp_arguments(p)
    add_cuda_argument(p)
    args = p.parse_args()
    # CUDA and the EGL renderer on one GPU (utils/gpu.py): --cuda-device, the deployment's, or
    # the first CUDA_VISIBLE_DEVICES entry.
    pin_egl(args.cuda_device)

    facade = GenesisEnvFacade(
        task=args.task,
        seed=args.seed,
        backend=args.backend,
        dt=args.dt,
        substeps=args.substeps,
        view_size=args.view_size,
        success_rule=args.success_rule,
    )
    # --sam3: env.segment (and the perception primitives below).
    facade._sam3 = sam3_segment.Sam3(args.sam3)
    # --sam3 / --unidepth: env.detect, env.select_detection, env.reject_detection, env.enhance_depth.
    # --ik: env.preview_reach (the Panda stands at the world origin, so targets are base-frame) and
    # move_delta refuses a target the ik service cannot reach.
    facade._reach = reach.reach_from_args(args, "panda")
    reach.install_preview_reach(
        facade,
        facade._reach,
        joints=lambda: facade._qpos()[: len(MOTOR_DOFS)],
        eef_quat_xyzw=lambda: _xyzw(facade._hand_pose()[1]),
    )
    view = render_view(facade)
    perception = install_perception(
        facade, args, cameras=["agentview", "wrist"], view=view, mutating=MOTIONS
    )
    # --contact-graspnet & co: env.plan_grasp, env.claim_waypoints and friends over the same views
    # (the hand's quaternion as xyzw); env.execute_grasp / env.execute_place run the claimed path
    # as move_delta legs (utils/grasp_chain.py).
    facade.install_grasp(
        GraspPlanner.from_args(
            view,
            cameras=["agentview", "wrist"],
            masks=perception.book if perception is not None else None,
            sam3=args.sam3 or (perception.sam3 if perception is not None else None),
            eef_pose=lambda arm: (facade._tcp(), _xyzw(facade._hand_pose()[1])),
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
