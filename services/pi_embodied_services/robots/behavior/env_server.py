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
# The primitive set follows CaP-X's R1ProControlApi (capx/integrations/r1pro/control.py @53e9966):
# navigate_to_pose, move_hand, grasp_object, open/close_gripper, get_robot_position. Not
# CaP-X's move_hand(ignore_all_obstacles=True): the planner keeps its obstacles. Not its
# pick_up_radio_reward: success is BDDL's, the "picked" judgement is a reference field only.

"""RPC server wrapping one BEHAVIOR-1K challenge task on the R1Pro (OmniGibson / Isaac Sim).

The motion primitives run OmniGibson's StarterSemanticActionPrimitives (cuRobo plans for the
holonomic base and the arms) one control step at a time, so ``stop`` interrupts them between
steps. Success is the BDDL task's ``success`` termination, latched at the first control step
it holds (a goal reached mid-primitive and undone by its end still counts, as on the other
simulators), and ``q_score`` BEHAVIOR's partial-success metric (1 once success is latched);
both go into every observation. What the simulator knows and a camera cannot see (the object OmniGibson's
grasping holds, the reference "picked" judgement) is reported apart, in ``privileged``, and
the pi robot shows it to the planner only under ``--privileged``.

Isaac Sim starts in ``main`` (minutes: the scene is a whole house), before the server binds, so
healthz answers only once the task is loaded. Every call runs on the main thread (Kit is not
thread-safe). ``--gpu-id`` picks the simulator's physical GPU (``pin_isaac``); the perception
servers should sit on another one.

Code mode (``code.run``, utils/code_exec.py ``CodeRunMixin``): a program calls the primitives of
the robot's manifest (packages/embodied/src/primitives/manifests/behavior.json, read by
components/manifest.py; pi's tools take their schemas from the same file) from a sandboxed
subprocess; ``code.run`` is itself a business call, so
it runs on the main thread like every tool's RPC, and the primitive calls it answers go to the same
facade methods on that same thread (never from another one: Kit is not thread-safe). What a
primitive hands the program drops the camera images (the head frame goes to the run's video) and
the simulator-only ``privileged`` block (the object in hand, the reference "picked"); the run
reports its control steps, the latched success and the new observation (``_finish_run``).

Perception runs here for the tools and the programs alike: ``segment`` (SAM3, ``--sam3``),
``point`` (Molmo, ``--molmo``) and ``back_project`` read the latest camera frames through their
metric depth. ``move_to_joints`` / ``move_along_trajectory`` (CaP-X's joint-space primitives)
drive an arm's absolute position JointController straight to joint targets, clipped to the
joint limits; there is no ``solve_ik`` / ``traj_plan``: CaP-X solves the R1Pro's IK with PyRoKi
on its own URDF, which this server does not have.
"""

from __future__ import annotations

import argparse
import base64
import io
import math
import os
import sys
import time
import traceback
from typing import Any

import numpy as np

from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.robots.behavior import sim
from pi_embodied_services.robots.behavior.tasks import LANGUAGE, TASK_INDEX, TASK_NAMES
from pi_embodied_services.utils import ground_truth
from pi_embodied_services.utils.code_exec import CodeRunMixin
from pi_embodied_services.utils.gpu import pin_isaac
from pi_embodied_services.utils.perception import (
    add_perception_arguments,
    install_perception,
    render_view,
)
from pi_embodied_services.utils.rpc.main_thread_serve import MainThreadServeMixin

#: Control steps one primitive may take before it is declared stuck.
MAX_PRIMITIVE_STEPS = 3000
#: Above the grasp pose the pre-grasp and the lift go, m.
PREGRASP_OFFSET_M = 0.10
#: CaP-X's "picked" threshold: an object in hand risen this much above its reset height, m.
PICKED_RISE_M = 0.005
#: Farthest a single navigate_to_pose may go from the base, m (the planner refuses more).
MAX_NAVIGATE_M = 5.0
#: Farthest move_hand target from the arm's shoulder-height base point, m (beyond reach).
MAX_HAND_REACH_M = 1.5
#: move_hand_delta: the largest relative step (m) and turn (rad) one call commands.
MAX_HAND_STEP_M = 0.1
MAX_HAND_YAW_RAD = 0.3
#: Code mode: video frames one run hands back (halved, every other one kept, when full), the most
#: raw actions one chunk_step call may take, and the motion primitives (their results are an
#: observation plus a report).
CODE_MAX_FRAMES = 128
CODE_MAX_CHUNK = 300
#: Depth beyond this is no hit (OmniGibson's depth_linear on the sky), m.
MAX_DEPTH_M = 20.0
#: move_to_joints: the default joint tolerance (rad) and control-step budget; one waypoint of
#: move_along_trajectory gets WAYPOINT_STEPS; a trajectory has at most MAX_WAYPOINTS.
JOINT_TOL_RAD = 0.01
JOINT_MAX_STEPS = 300
WAYPOINT_STEPS = 60
MAX_WAYPOINTS = 100
MOTIONS = (
    "env.navigate_to_pose",
    "env.move_hand",
    "env.move_hand_delta",
    "env.grasp_object",
    "env.open_gripper",
    "env.close_gripper",
    "env.move_to_joints",
    "env.move_along_trajectory",
)
#: The perception reads whose tool picture a program does not receive.
PICTURES = ("env.segment", "env.point")


def check_arm(arm: str) -> str:
    if arm not in sim.ARMS:
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
    return arm


def as_pose(xyz, quat_xyzw, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A world pose from a position and an optional xyzw orientation (default ``current``)."""
    pos = np.asarray(xyz, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(pos)):
        raise ValueError(f"xyz must be finite, got {xyz!r}")
    quat = (
        current
        if quat_xyzw is None
        else np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    )
    norm = float(np.linalg.norm(quat))
    if not norm > 0:
        raise ValueError("quat_xyzw must not be zero")
    return pos, quat / norm


def project(depth: np.ndarray, meta: dict) -> np.ndarray:
    """World xyz [H, W, 3] of every pixel of an OmniGibson camera: OpenGL convention (looks along
    -Z, +Y up), so ``x = (c - cx) d / fx, y = -(r - cy) d / fy, z = -d`` before the cam-to-world
    transform. NaN where there is no hit."""
    d = np.asarray(depth, dtype=np.float64).reshape(np.shape(depth)[:2])
    h, w = d.shape
    K = np.asarray(meta["intrinsic_K"], dtype=np.float64)
    T = np.asarray(meta["extrinsic_cam2world"], dtype=np.float64)
    rows, cols = np.mgrid[0:h, 0:w]
    cam = np.stack(
        [(cols - K[0, 2]) * d / K[0, 0], -(rows - K[1, 2]) * d / K[1, 1], -d], axis=-1
    )
    xyz = cam @ T[:3, :3].T + T[:3, 3]
    xyz[~((d > 0) & (d < MAX_DEPTH_M))] = np.nan
    return xyz


def _png_base64(rgb: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _median_xyz(pts: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in np.median(pts, axis=0)]


class BehaviorEnvFacade(CodeRunMixin, MainThreadServeMixin, BaseEnvFacade):
    """One BEHAVIOR task env plus OmniGibson's semantic primitives."""

    SERVICE_NAME = "behavior-env"

    def __init__(
        self,
        *,
        handle: sim.Handle,
        meta: dict,
        max_primitive_steps: int = MAX_PRIMITIVE_STEPS,
        sam3: str | None = None,
        molmo: str | None = None,
    ):
        super().__init__()
        self._h = handle
        self._env = handle.env
        self._robot = handle.robot
        self._task = handle.task
        self._ctrl = handle.controller
        self._meta = dict(meta)
        self._max_steps = max_primitive_steps
        self._closed = False
        self._obs: dict = {}
        self._info: dict | None = None
        self._steps = 0
        self._terminated = self._truncated = False
        #: BDDL success, latched at the first control step it holds (the other simulators'
        #: success_once): a goal reached mid-primitive and undone by its end still counts.
        self._solved = False
        self._initial_goals: list[list[bool]] = []
        self._initial_heights: dict[str, float] = {}
        # Code mode: the control steps before the run, and the run's video frames.
        self._run_start = 0
        self._run_frames: list[np.ndarray] = []
        # --sam3 / --molmo: the segment and point primitives (clients made at the first call).
        self._sam3_url = sam3 or None
        self._molmo_url = molmo or None
        self._clients: dict[str, Any] = {}
        #: World xyz per pixel of each camera's latest frame, for the env step it was made at.
        self._world_maps: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["env.navigate_to_pose"] = self.navigate_to_pose
        self._rpc["env.move_hand"] = self.move_hand
        self._rpc["env.move_hand_delta"] = self.move_hand_delta
        self._rpc["env.grasp_object"] = self.grasp_object
        self._rpc["env.open_gripper"] = self.open_gripper
        self._rpc["env.close_gripper"] = self.close_gripper
        self._rpc["env.get_robot_position"] = self.get_robot_position
        self._rpc["env.raw_obs"] = self.raw_obs
        self._rpc["env.state"] = self.state
        self._rpc["env.ground_truth_poses"] = self.ground_truth_poses
        self._rpc["env.segment"] = self.segment
        self._rpc["env.point"] = self.point
        self._rpc["env.back_project"] = self.back_project
        self._rpc["env.move_to_joints"] = self.move_to_joints
        self._rpc["env.move_along_trajectory"] = self.move_along_trajectory
        # The primitives are packages/embodied/src/primitives/manifests/behavior.json (with
        # pi); code.api, the programs' whitelist and the startup self-check come from it.
        self._manifest_code_run(
            "behavior",
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
            "molmo": bool(self._molmo_url),
            "unidepth": "env.enhance_depth" in self._rpc,
            "privileged": True,
        }.get(capability, False)

    # ---- code mode (run_code) ----

    def _begin_run(self) -> None:
        self._run_start = self._steps
        self._run_frames = []

    def _finish_run(self) -> dict:
        """The run's effect for pi: control steps taken, the latched success, the episode's end
        flags, the new observation (the tools' ``obs``) and the run's video frames."""
        return {
            "steps": self._steps - self._run_start,
            "success": bool(self._solved),
            "terminated": bool(self._terminated),
            "truncated": bool(self._truncated),
            "obs": self._pack(),
            "frames": list(self._run_frames),
        }

    def _program_obs(self, obs: dict) -> dict:
        """An observation as a program receives it: the head frame goes to the run's video; no
        images, no depth and no simulator-only ``privileged`` block."""
        if "head" in obs:
            if len(self._run_frames) >= CODE_MAX_FRAMES:
                self._run_frames = self._run_frames[::2]
            self._run_frames.append(obs["head"])
        hidden = {*sim.CAMERAS, *(f"{c}_depth" for c in sim.CAMERAS), "privileged"}
        return {k: v for k, v in obs.items() if k not in hidden}

    def _code_reply(self, method: str, out: Any) -> Any:
        """What a program receives of a primitive's result: the same facade method as the tools
        call, without the images and the simulator-only state."""
        if method in MOTIONS or method == "env.state":
            return self._program_obs(out)
        if method in PICTURES and isinstance(out, dict):
            # The picture is the tool's; the program has the numbers (and segment's mask).
            return {k: v for k, v in out.items() if k != "overlay_png_base64"}
        if method == "env.step":
            obs, rew, terminated, truncated, info = out
            return {
                "reward": rew,
                "terminated": terminated,
                "truncated": truncated,
                "success": bool(info.get("success")),
                "state": self._program_obs(obs),
            }
        if method == "env.chunk_step":
            obs, terminated, truncated, info = out
            many = isinstance(obs, list)
            return {
                "terminated": terminated,
                "truncated": truncated,
                **info,
                "states" if many else "state": (
                    [self._program_obs(o) for o in obs]
                    if many
                    else self._program_obs(obs)
                ),
            }
        return out

    def _code_move_m(self, method: str, kwargs: dict) -> float:
        """How far a program's call may move the robot (the run's translation cap): the base's
        drive, a hand's travel, or a bound of what raw actions command."""
        if method == "env.navigate_to_pose":
            pos, _q, _yaw = sim.base_pose(self._robot)
            goal = np.asarray([kwargs["x"], kwargs["y"]], dtype=np.float64)
            return float(np.linalg.norm(goal - pos[:2]))
        if method in ("env.move_hand", "env.grasp_object"):
            p, _q = sim.eef_pose(self._robot, check_arm(kwargs["arm"]))
            target = np.asarray(kwargs["xyz"], dtype=np.float64).reshape(3)
            if method == "env.move_hand":
                return float(np.linalg.norm(target - p))
            # To the pre-grasp above the target, down to it and back up.
            offset = float(kwargs.get("pregrasp_offset_m") or PREGRASP_OFFSET_M)
            above = target + np.array([0.0, 0.0, offset])
            return float(np.linalg.norm(above - p)) + 2 * offset
        if method == "env.move_hand_delta":
            d = np.asarray(kwargs.get("delta_xyz", (0.0, 0.0, 0.0)), dtype=np.float64)
            return float(np.linalg.norm(d))
        if method in ("env.move_to_joints", "env.move_along_trajectory"):
            arm = check_arm(kwargs["arm"])
            prev = sim.arm_joints(self._robot, arm)
            total = 0.0
            waypoints = (
                [kwargs["joints"]]
                if method == "env.move_to_joints"
                else kwargs["trajectory"]
            )
            for q in waypoints:
                q = np.asarray(q, dtype=np.float64).reshape(-1)
                if q.shape != prev.shape:
                    raise ValueError(
                        f"a joint target has {q.size} values; the {arm} arm has {prev.size}"
                    )
                total += (
                    float(np.max(np.abs(q - prev), initial=0.0)) * sim.REACH_PER_RAD_M
                )
                prev = q
            return total
        if method == "env.step":
            return sim.action_move_m(self._robot, [kwargs["action"]])
        if method == "env.chunk_step":
            return sim.action_move_m(self._robot, kwargs["actions"])
        return 0.0

    def _code_check(self, method: str, kwargs: dict) -> None:
        """Refuse a program's call that the run's wall clock could not bound."""
        if method == "env.chunk_step" and len(kwargs["actions"]) > CODE_MAX_CHUNK:
            raise ValueError(
                f"chunk_step takes at most {CODE_MAX_CHUNK} actions per call in code mode"
            )

    # ---- stepping ----

    def _absorb(self, result: tuple) -> None:
        """Record one ``env.step`` result."""
        obs, _reward, term, trunc, info = result
        self._obs = obs
        self._info = info
        self._steps += 1
        self._terminated |= bool(term)
        self._truncated |= bool(trunc)
        self._solved = self._solved or sim.success(info, self._task)

    def _run(self, actions) -> tuple[int, bool]:
        """Step a primitive's actions until it finishes, a stop arrives or the episode ends."""
        return sim.run(
            self._env,
            actions,
            self._absorb,
            lambda: self.stop_requested() or self._truncated,
            self._max_steps,
        )

    def _primitive(self, name: str, *phases: tuple[str, Any]) -> dict:
        """Run the phases of one primitive (``(label, generator)`` in order); the report has
        ``ok``, ``phase`` (the last one started), ``steps`` and, on a refusal or failure,
        ``error`` (OmniGibson's ActionPrimitiveError or a stuck primitive), or ``cancelled``."""
        report: dict[str, Any] = {"primitive": name, "steps": 0, "ok": False}
        if self._truncated:
            return {**report, "error": "the episode is over (max steps)"}
        for label, gen in phases:
            report["phase"] = label
            try:
                n, cancelled = self._run(gen)
            except self._h.error as e:
                report["error"] = f"{label}: {e}"
                return report
            except RuntimeError as e:
                report["error"] = f"{label}: {e}"
                return report
            report["steps"] += n
            if cancelled:
                report["cancelled"] = True
                return report
            if self._truncated:
                report["error"] = "the episode is over (max steps)"
                return report
        report["ok"] = True
        return report

    # ---- observations ----

    def _goals(self) -> dict:
        now = sim.goal_satisfaction(self._task)
        flat = [v for opt in now for v in opt]
        return {
            "satisfied": int(sum(flat)) if flat else 0,
            "total": len(now[0]) if now else 0,
            "now": now,
        }

    def _privileged(self) -> dict:
        """What only the simulator knows: the object in each hand and CaP-X's "picked"
        judgement (an object in hand risen above its reset height)."""
        held = {arm: sim.in_hand(self._robot, arm) for arm in sim.ARMS}
        heights = sim.object_heights(self._task)
        picked = False
        for arm in sim.ARMS:
            name = held[arm]
            if name is None:
                continue
            # object_scope keys are BDDL instances, the held object is a scene object: match by name.
            for inst, z0 in self._initial_heights.items():
                entity = self._task.object_scope[inst]
                if (
                    str(entity.name) == name
                    and heights.get(inst, z0) > z0 + PICKED_RISE_M
                ):
                    picked = True
        return {"in_hand": held, "picked": picked}

    def _state(self) -> dict:
        solved = self._solved
        goals = self._goals()
        pos, quat, yaw = sim.base_pose(self._robot)
        eef = {}
        for arm in sim.ARMS:
            p, q = sim.eef_pose(self._robot, arm)
            eef[arm] = {
                "pos": p.astype(np.float32),
                "quat_xyzw": q.astype(np.float32),
                "gripper_width": round(sim.gripper_width(self._robot, arm), 5),
            }
        return {
            "base_pos": pos.astype(np.float32),
            "base_quat_xyzw": quat.astype(np.float32),
            "base_yaw": round(yaw, 4),
            "eef": eef,
            "success": solved,
            "q_score": round(sim.q_score(solved, goals["now"], self._initial_goals), 4),
            "goals": {"satisfied": goals["satisfied"], "total": goals["total"]},
            "terminated": self._terminated,
            "truncated": self._truncated,
            "env_steps": self._steps,
            "privileged": self._privileged(),
        }

    def _pack(self) -> dict:
        return {**sim.images(self._obs, self._robot), **self._state()}

    # ---- gym-like surface ----

    def reset(self):
        """Load the task instance again; latch the goal predicates and object heights it starts with."""
        self._steps = 0
        self._terminated = self._truncated = False
        self._solved = False
        self._info = None
        self._obs, info = self._env.reset()
        self._run(self._ctrl._settle_robot())
        self._initial_goals = sim.goal_satisfaction(self._task)
        self._initial_heights = sim.object_heights(self._task)
        return self._pack(), {"instruction": self._meta["instruction"], **(info or {})}

    def step(self, action):
        """One raw control step of the robot's full action vector."""
        self._absorb(self._env.step(np.asarray(action, dtype=np.float32).reshape(-1)))
        return (
            self._pack(),
            0.0,
            self._terminated,
            self._truncated,
            {"success": self._solved},
        )

    def chunk_step(self, actions, *, return_all_frames: bool = False):
        """Raw control steps ``[N, action_dim]``; stops early on ``stop`` or the episode's end."""
        frames, info = [], {}
        for action in np.asarray(actions, dtype=np.float32):
            if self.stop_requested():
                info["cancelled"] = True
                break
            obs, _r, term, trunc, info = self.step(action)
            frames.append(obs)
            if term or trunc:
                break
        last = frames[-1] if frames else self._pack()
        return (
            (frames if return_all_frames else last),
            self._terminated,
            self._truncated,
            info,
        )

    # ---- primitives ----

    def navigate_to_pose(self, x: float, y: float, yaw: float) -> dict:
        """Drive the base to world ``(x, y, yaw)`` (cuRobo base plan). Refused beyond
        ``MAX_NAVIGATE_M`` from the current base position."""
        goal = np.asarray([x, y, yaw], dtype=np.float64)
        if not np.all(np.isfinite(goal)):
            raise ValueError(f"pose must be finite, got {[x, y, yaw]!r}")
        pos, _q, _yaw = sim.base_pose(self._robot)
        dist = float(np.linalg.norm(goal[:2] - pos[:2]))
        if not dist <= MAX_NAVIGATE_M:
            raise ValueError(
                f"goal is {dist:.2f} m away; the limit is {MAX_NAVIGATE_M} m per call"
            )
        report = self._primitive(
            "navigate_to_pose",
            (
                "navigate",
                self._ctrl._navigate_to_pose((float(x), float(y), float(yaw))),
            ),
        )
        end, _q, end_yaw = sim.base_pose(self._robot)
        report.update(
            goal=[float(v) for v in goal],
            reached_pos=[round(float(v), 4) for v in end],
            reached_yaw=round(end_yaw, 4),
            distance_left_m=round(float(np.linalg.norm(goal[:2] - end[:2])), 4),
            yaw_left_rad=round(
                float(math.remainder(goal[2] - end_yaw, 2 * math.pi)), 4
            ),
        )
        return {**self._pack(), **report}

    def _hand(self, arm: str, pos: np.ndarray, quat: np.ndarray, **kw):
        """A ``_move_hand`` generator for ``arm`` (the primitives act on ``controller.arm``)."""
        self._ctrl.arm = arm
        return self._ctrl._move_hand((sim.tensor(pos), sim.tensor(quat)), **kw)

    def _hand_report(self, arm: str, target: np.ndarray, report: dict) -> dict:
        p, q = sim.eef_pose(self._robot, arm)
        report.update(
            arm=arm,
            eef_pos=[round(float(v), 4) for v in p],
            eef_quat_xyzw=[round(float(v), 4) for v in q],
            distance_left_m=round(float(np.linalg.norm(target - p)), 4),
            gripper_width=round(sim.gripper_width(self._robot, arm), 5),
        )
        return report

    def move_hand(self, arm: str, xyz, quat_xyzw=None) -> dict:
        """Plan and move ``arm``'s end effector to a world pose, obstacles respected (cuRobo).
        The orientation defaults to the current one. Refused beyond ``MAX_HAND_REACH_M`` of the base."""
        arm = check_arm(arm)
        _p, cur = sim.eef_pose(self._robot, arm)
        pos, quat = as_pose(xyz, quat_xyzw, cur)
        base, _q, _yaw = sim.base_pose(self._robot)
        reach = float(np.linalg.norm(pos[:2] - base[:2]))
        if not reach <= MAX_HAND_REACH_M:
            raise ValueError(
                f"target is {reach:.2f} m from the base in xy; the arm reaches {MAX_HAND_REACH_M} m: navigate first"
            )
        report = self._primitive("move_hand", ("move", self._hand(arm, pos, quat)))
        return {**self._pack(), **self._hand_report(arm, pos, report)}

    def hand_delta_target(
        self, arm: str, delta_xyz, yaw: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """The world pose ``move_hand_delta`` servoes to: the EEF moved by a base-frame delta
        (+x ahead of the base, +y to its left, +z up) and turned by ``yaw`` about world +z."""
        from scipy.spatial.transform import Rotation

        arm = check_arm(arm)
        d = np.asarray(delta_xyz, dtype=np.float64).reshape(-1)
        if d.shape != (3,) or not np.all(np.isfinite(d)) or not math.isfinite(yaw):
            raise ValueError("delta_xyz must be 3 finite numbers and yaw finite")
        if not float(np.linalg.norm(d)) <= MAX_HAND_STEP_M:
            raise ValueError(
                f"delta_xyz moves {np.linalg.norm(d):.3f} m; the limit is {MAX_HAND_STEP_M} m per call"
            )
        if not abs(float(yaw)) <= MAX_HAND_YAW_RAD:
            raise ValueError(
                f"yaw {yaw:.3f} rad; the limit is {MAX_HAND_YAW_RAD} rad per call"
            )
        _base, _q, base_yaw = sim.base_pose(self._robot)
        c, s_ = math.cos(base_yaw), math.sin(base_yaw)
        world = np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1], d[2]])
        p, q = sim.eef_pose(self._robot, arm)
        turned = Rotation.from_rotvec([0.0, 0.0, float(yaw)]) * Rotation.from_quat(q)
        return p + world, turned.as_quat()

    def move_hand_delta(
        self, arm: str, delta_xyz, yaw: float = 0.0, gripper: str | None = None
    ) -> dict:
        """A small relative step of ``arm``'s end effector: an optional gripper command
        ("open" / "close") first, then a move by a base-frame ``delta_xyz`` (m; +x ahead of the
        base, +y to its left, +z up; at most ``MAX_HAND_STEP_M``) turned by ``yaw`` (rad about
        world +z; at most ``MAX_HAND_YAW_RAD``), planned like ``move_hand``. A zero step with
        no gripper command holds still. The units (pi's ``act``) run on it."""
        arm = check_arm(arm)
        if gripper not in (None, "open", "close"):
            raise ValueError(
                f"gripper must be 'open', 'close' or null, got {gripper!r}"
            )
        pos, quat = self.hand_delta_target(arm, delta_xyz, yaw)
        phases = []
        if gripper is not None:
            self._ctrl.arm = arm
            gen = (
                self._ctrl._execute_release()
                if gripper == "open"
                else self._ctrl._execute_grasp()
            )
            phases.append((gripper, gen))
        if float(np.linalg.norm(np.asarray(delta_xyz, dtype=np.float64))) > 0 or yaw:
            phases.append(("move", self._hand(arm, pos, quat)))
        if not phases:
            phases.append(("hold", self._ctrl._settle_robot()))
        report = self._primitive("move_hand_delta", *phases)
        return {**self._pack(), **self._hand_report(arm, pos, report)}

    def grasp_object(
        self,
        arm: str,
        xyz,
        quat_xyzw=None,
        pregrasp_offset_m: float = PREGRASP_OFFSET_M,
    ) -> dict:
        """Grasp at a world pose: open, move to the pre-grasp above it, approach (sticky grasping
        closes first and stops on contact; assisted closes after), settle, lift back to the
        pre-grasp. ``ok`` says the motions ran; whether something is held shows in
        ``gripper_width`` (and, privileged, ``in_hand``)."""
        arm = check_arm(arm)
        offset = float(pregrasp_offset_m)
        if not 0.02 <= offset <= 0.5:
            raise ValueError(
                f"pregrasp_offset_m must be within [0.02, 0.5], got {offset}"
            )
        _p, cur = sim.eef_pose(self._robot, arm)
        pos, quat = as_pose(xyz, quat_xyzw, cur)
        above = pos + np.array([0.0, 0.0, offset])
        self._ctrl.arm = arm
        sticky = getattr(self._robot, "grasping_mode", "sticky") == "sticky"
        approach = (
            [
                ("close", self._ctrl._execute_grasp()),
                ("approach", self._hand(arm, pos, quat, stop_on_ag=True)),
            ]
            if sticky
            else [
                ("approach", self._hand(arm, pos, quat)),
                ("close", self._ctrl._execute_grasp()),
            ]
        )
        report = self._primitive(
            "grasp_object",
            ("open", self._ctrl._execute_release()),
            ("pregrasp", self._hand(arm, above, quat)),
            *approach,
            ("settle", self._ctrl._settle_robot()),
            ("lift", self._hand(arm, above, quat)),
        )
        report["grasping_mode"] = "sticky" if sticky else "assisted"
        return {**self._pack(), **self._hand_report(arm, above, report)}

    def _fingers(self, arm: str, open_: bool) -> dict:
        arm = check_arm(arm)
        self._ctrl.arm = arm
        gen = self._ctrl._execute_release() if open_ else self._ctrl._execute_grasp()
        report = self._primitive(
            "open_gripper" if open_ else "close_gripper", ("fingers", gen)
        )
        report.update(
            arm=arm, gripper_width=round(sim.gripper_width(self._robot, arm), 5)
        )
        return {**self._pack(), **report}

    def open_gripper(self, arm: str) -> dict:
        """Open ``arm``'s gripper fully (OmniGibson's release; an object still held is an error)."""
        return self._fingers(arm, True)

    def close_gripper(self, arm: str) -> dict:
        """Close ``arm``'s gripper fully."""
        return self._fingers(arm, False)

    def get_robot_position(self) -> dict:
        """World pose of the base (position, xyzw, yaw) and of both end effectors; no stepping."""
        pos, quat, yaw = sim.base_pose(self._robot)
        eef = {}
        for arm in sim.ARMS:
            p, q = sim.eef_pose(self._robot, arm)
            eef[arm] = {
                "pos": [round(float(v), 4) for v in p],
                "quat_xyzw": [round(float(v), 4) for v in q],
            }
        return {
            "pos": [round(float(v), 4) for v in pos],
            "quat_xyzw": [round(float(v), 4) for v in quat],
            "yaw": round(yaw, 4),
            "eef": eef,
        }

    # ---- joint space (CaP-X's move_to_joints / move_along_trajectory) ----

    def _joint_target(self, arm: str, joints) -> np.ndarray:
        """The full-body position target: the current joints with ``arm``'s replaced by
        ``joints`` (clipped to the joint limits)."""
        idx = sim.to_np(self._robot.arm_control_idx[arm]).astype(int).reshape(-1)
        target = np.asarray(joints, dtype=np.float64).reshape(-1)
        if target.shape != idx.shape or not np.all(np.isfinite(target)):
            raise ValueError(
                f"joints must be {idx.size} finite angles (rad) for the {arm} arm, got {joints!r}"
            )
        lo, hi = sim.joint_limits(self._robot)
        q = sim.to_np(self._robot.get_joint_positions()).astype(np.float64).copy()
        q[idx] = np.clip(target, lo[idx], hi[idx])
        return q

    def _servo_joints(self, arm: str, q: np.ndarray, tol: float, steps: int):
        """Actions holding the full-body target ``q`` until ``arm``'s joints are within ``tol``
        of it or ``steps`` control steps ran (a generator for ``_primitive``)."""
        idx = sim.to_np(self._robot.arm_control_idx[arm]).astype(int).reshape(-1)
        action = sim.joint_target_action(self._robot, q)
        for _ in range(steps):
            if np.max(np.abs(sim.arm_joints(self._robot, arm) - q[idx])) <= tol:
                return
            yield action

    def _joint_report(self, arm: str, q: np.ndarray, tol: float, report: dict) -> dict:
        idx = sim.to_np(self._robot.arm_control_idx[arm]).astype(int).reshape(-1)
        now = sim.arm_joints(self._robot, arm)
        left = float(np.max(np.abs(now - q[idx]), initial=0.0))
        p, quat = sim.eef_pose(self._robot, arm)
        report.update(
            arm=arm,
            joints=[round(float(v), 4) for v in now],
            joints_left_rad=round(left, 4),
            reached=left <= tol,
            eef_pos=[round(float(v), 4) for v in p],
            eef_quat_xyzw=[round(float(v), 4) for v in quat],
        )
        return report

    def move_to_joints(
        self,
        joints,
        arm: str,
        tol_rad: float = JOINT_TOL_RAD,
        max_steps: int = JOINT_MAX_STEPS,
    ) -> dict:
        """Drive ``arm`` to a joint configuration (rad, the arm's control order; clipped to the
        joint limits): its position controller holds the target until every joint is within
        ``tol_rad`` or ``max_steps`` control steps ran. No collision checking: move_hand plans
        around obstacles, this does not. Returns ok, reached, joints, joints_left_rad and the
        observation."""
        arm = check_arm(arm)
        if not 0 < float(tol_rad) <= 0.5 or not 1 <= int(max_steps) <= self._max_steps:
            raise ValueError(
                f"tol_rad must be in (0, 0.5] and max_steps in [1, {self._max_steps}]"
            )
        q = self._joint_target(arm, joints)
        report = self._primitive(
            "move_to_joints",
            ("move", self._servo_joints(arm, q, float(tol_rad), int(max_steps))),
        )
        return {**self._pack(), **self._joint_report(arm, q, float(tol_rad), report)}

    def move_along_trajectory(self, trajectory, arm: str) -> dict:
        """Drive ``arm`` through joint waypoints ``[N, dof]`` (N <= MAX_WAYPOINTS), each held
        until within JOINT_TOL_RAD or WAYPOINT_STEPS control steps; stops early on a stop or the
        episode's end. Returns the last waypoint's report with ``waypoints`` run."""
        arm = check_arm(arm)
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 2 or not 1 <= len(traj) <= MAX_WAYPOINTS:
            raise ValueError(
                f"trajectory must be [N, dof] joint waypoints with 1 <= N <= {MAX_WAYPOINTS}"
            )
        targets = [self._joint_target(arm, q) for q in traj]
        report = self._primitive(
            "move_along_trajectory",
            *(
                (
                    f"waypoint {i}",
                    self._servo_joints(arm, q, JOINT_TOL_RAD, WAYPOINT_STEPS),
                )
                for i, q in enumerate(targets)
            ),
        )
        # Waypoints finished: all, or those before the one in progress when it ended.
        report["waypoints"] = (
            len(targets)
            if report["ok"]
            else int(report.get("phase", "waypoint 0").split()[-1])
        )
        return {
            **self._pack(),
            **self._joint_report(arm, targets[-1], JOINT_TOL_RAD, report),
        }

    # ---- perception (the tools' and the programs') ----

    def _client(self, name: str, url: str | None, flag: str):
        if not url:
            raise RuntimeError(
                f"{name} needs a server (start the env server with {flag})"
            )
        if name not in self._clients:
            from pi_embodied_services.utils.rpc.http_rpc import HttpRpcClient

            self._clients[name] = HttpRpcClient(url)
        return self._clients[name]

    def _world_map(self, camera: str) -> tuple[np.ndarray, np.ndarray]:
        """The latest rgb of ``camera`` and the world xyz of its pixels (NaN: no depth), cached
        per env step (the cameras move with the robot)."""
        if camera not in sim.CAMERAS:
            raise ValueError(
                f"camera must be one of {sorted(sim.CAMERAS)}, got {camera!r}"
            )
        hit = self._world_maps.get(camera)
        if hit is not None and hit[0] == self._steps:
            return hit[1], hit[2]
        imgs = sim.images(self._obs, self._robot)
        meta = sim.camera_meta(self._robot, camera)
        if meta["convention"] != "opengl":
            raise RuntimeError(
                f"camera {camera}: unexpected convention {meta['convention']}"
            )
        rgb, xyz = imgs[camera], project(imgs[f"{camera}_depth"], meta)
        self._world_maps[camera] = (self._steps, rgb, xyz)
        return rgb, xyz

    def segment(
        self,
        prompt: str | None = None,
        point=None,
        camera: str = "head",
        min_score: float = 0.2,
    ) -> dict:
        """SAM3 segmentation of the latest image of ``camera`` by a text prompt or a positive
        point [row, col]; the top mask projected through that camera's metric depth.

        Returns found, camera, score, box, mask (bool [H, W]), n_pixels, n_valid,
        centroid_pixel [row, col], world_xyz (median over the mask) and top_xyz (median of its
        highest tenth: a grasp point), or world_error when too few pixels have depth; the tool
        also gets ``overlay_png_base64``."""
        from PIL import Image

        text = (prompt or "").strip()
        if bool(text) == (point is not None):
            raise ValueError("give exactly one of a text prompt or a point [row, col]")
        sam3 = self._client("segment", self._sam3_url, "--sam3")
        rgb, xyz = self._world_map(camera)
        res = sam3.call(
            "sam3.segment",
            kwargs={
                "image_base64": _png_base64(rgb),
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
            return {
                "found": False,
                "camera": camera,
                "error": res.get("reason") or "no mask",
                "fallback": "Pick pixels in the image and use back_project.",
            }
        mask = np.asarray(
            Image.open(io.BytesIO(base64.b64decode(res["mask_png_base64"])))
        )
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != rgb.shape[:2]:
            raise RuntimeError(
                f"SAM3 mask {mask.shape} does not match the {rgb.shape[:2]} image"
            )
        mask = mask >= 128
        rows, cols = np.nonzero(mask)
        pts = xyz[mask]
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        out: dict[str, Any] = {
            "found": True,
            "camera": camera,
            "score": None
            if res.get("score") is None
            else round(float(res["score"]), 3),
            "box": res.get("box"),
            "mask": mask,
            "n_pixels": int(mask.sum()),
            "n_valid": len(pts),
            "centroid_pixel": [int(np.median(rows)), int(np.median(cols))]
            if len(rows)
            else None,
        }
        if len(pts) < 10:
            out["world_xyz"] = None
            out["world_error"] = f"too few valid depth pixels ({len(pts)})"
        else:
            out["world_xyz"] = _median_xyz(pts)
            top = pts[np.argsort(-pts[:, 2])][: max(10, len(pts) // 10)]
            out["top_xyz"] = _median_xyz(top)
        overlay = rgb.astype(np.float32)
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([255.0, 0.0, 0.0])
        out["overlay_png_base64"] = _png_base64(overlay.round().astype(np.uint8))
        return out

    def _around(
        self, xyz: np.ndarray, row: int, col: int, k: int
    ) -> list[float] | None:
        """Median world xyz of the valid pixels in a (2k+1)^2 window, or None (< 3 valid)."""
        h, w = xyz.shape[:2]
        win = xyz[
            max(0, row - k) : min(h, row + k + 1), max(0, col - k) : min(w, col + k + 1)
        ]
        pts = win.reshape(-1, 3)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        return None if len(pts) < 3 else _median_xyz(pts)

    def point(self, query: str, camera: str = "head") -> dict:
        """Molmo points at what a short noun phrase names in the latest image of ``camera``:
        found, pixel [row, col], world_xyz (median of a 7x7 window through the depth), Molmo's
        answer; the tool also gets the image with the point marked."""
        molmo = self._client("point", self._molmo_url, "--molmo")
        rgb, xyz = self._world_map(camera)
        res = molmo.call(
            "molmo.ground",
            kwargs={"image_base64": _png_base64(rgb), "query": str(query)},
            timeout_s=120,
        )
        xy = res.get("point_xy")
        if not xy:
            return {
                "found": False,
                "camera": camera,
                "answer": res.get("answer"),
                "fallback": "Use segment or back_project.",
            }
        h, w = rgb.shape[:2]
        col = int(np.clip(round(float(xy[0])), 0, w - 1))
        row = int(np.clip(round(float(xy[1])), 0, h - 1))
        marked = rgb.copy()
        r = max(2, min(h, w) // 80)
        marked[max(0, row - r) : row + r + 1, max(0, col - r) : col + r + 1] = (
            255,
            32,
            32,
        )
        return {
            "found": True,
            "camera": camera,
            "pixel": [row, col],
            "world_xyz": self._around(xyz, row, col, 3),
            "answer": res.get("answer"),
            "overlay_png_base64": _png_base64(marked),
        }

    def back_project(
        self,
        row: int | None = None,
        col: int | None = None,
        camera: str = "head",
        row_range=None,
        col_range=None,
    ) -> dict:
        """World xyz of pixel (row, col) (row 0 = top) of the latest image of ``camera``, the
        median of a 3x3 window through its metric depth; region mode (row_range and col_range,
        [first, last)) returns the midpoint of the window's world x and y with its median z
        (center_xyz), the median (median_xyz) and n_valid."""
        _rgb, xyz = self._world_map(camera)
        h, w = xyz.shape[:2]

        def span(r):
            return r if r is not None and len(r) == 2 and max(r) > min(r) else None

        rows, cols = span(row_range), span(col_range)
        if rows is not None or cols is not None:
            if rows is None or cols is None:
                raise ValueError("region mode needs both row_range and col_range")
            r0, r1 = (int(np.clip(v, 0, h)) for v in (min(rows), max(rows)))
            c0, c1 = (int(np.clip(v, 0, w)) for v in (min(cols), max(cols)))
            pts = xyz[r0:r1, c0:c1].reshape(-1, 3)
            pts = pts[np.all(np.isfinite(pts), axis=1)]
            if len(pts) < 8:
                raise ValueError(
                    f"too few valid pixels in the region ({len(pts)}); widen the window"
                )
            return {
                "camera": camera,
                "mode": "region",
                "center_xyz": [
                    round(float((pts[:, 0].min() + pts[:, 0].max()) / 2), 4),
                    round(float((pts[:, 1].min() + pts[:, 1].max()) / 2), 4),
                    round(float(np.median(pts[:, 2])), 4),
                ],
                "median_xyz": _median_xyz(pts),
                "n_valid": len(pts),
            }
        if row is None or col is None:
            raise ValueError("give row and col, or row_range and col_range")
        row, col = int(row), int(col)
        if not (0 <= row < h and 0 <= col < w):
            raise ValueError(f"pixel ({row}, {col}) out of bounds for {w}x{h}")
        p = self._around(xyz, row, col, 1)
        if p is None:
            raise ValueError(f"no depth at ({row}, {col}); pick another pixel")
        return {"camera": camera, "pixel": [row, col], "world_xyz": p}

    # ---- reads ----

    def raw_obs(self) -> dict:
        """The robot's proprioception dict as OmniGibson produces it (no task state: the BDDL
        low-dim observation carries object poses and is privileged)."""
        frames = self._obs.get(self._robot.name, {})
        proprio = frames.get("proprio")
        return {
            "proprio": None
            if proprio is None
            else sim.to_np(proprio).astype(np.float32),
            "joint_positions": sim.to_np(self._robot.get_joint_positions()).astype(
                np.float32
            ),
            "joint_names": [str(n) for n in self._robot.joints],
        }

    def state(self) -> dict:
        """The observation without images (no stepping)."""
        return self._state()

    def render_camera(self, camera_name: str = "head", depth: bool = False, **_: Any):
        """The latest frame of ``head``, ``left_wrist`` or ``right_wrist``; with ``depth``,
        ``[rgb, depth_m]``."""
        if camera_name not in sim.CAMERAS:
            raise ValueError(
                f"camera_name must be one of {sorted(sim.CAMERAS)}, got {camera_name!r}"
            )
        imgs = sim.images(self._obs, self._robot)
        return (
            [imgs[camera_name], imgs[f"{camera_name}_depth"]]
            if depth
            else imgs[camera_name]
        )

    def get_camera_meta(self, camera_name: str = "head", **_: Any) -> dict:
        """Intrinsics and the OpenGL cam-to-world transform of a camera, at the current pose."""
        return sim.camera_meta(self._robot, camera_name)

    def get_task_language(self) -> str:
        return self._meta["instruction"]

    def get_env_meta(self) -> dict:
        return dict(self._meta)

    def ground_truth_poses(self, names=None) -> dict:
        """World poses of the task's BDDL objects (pi's --privileged)."""
        return ground_truth.respond(sim.object_poses(self._task), names)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._env.close()
        finally:
            print("[behavior-env] closing OmniGibson", flush=True)
            og = self._h.og
            if og is not None and getattr(og, "sim", None) is not None:
                og.shutdown()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument(
        "--task", default=TASK_NAMES[0], choices=TASK_NAMES, metavar="ACTIVITY"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the task's pre-sampled instance id (the challenge's instances)",
    )
    p.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="physical GPU for Isaac Sim (default: PI_EMBODIED_CUDA_DEVICE, else the first "
        "CUDA_VISIBLE_DEVICES entry, else 0)",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=480,
        help="camera frames are square, this many px",
    )
    p.add_argument("--grasping-mode", default="sticky", choices=["sticky", "assisted"])
    p.add_argument(
        "--max-steps",
        type=int,
        default=20000,
        help="the task's truncation, control steps",
    )
    p.add_argument(
        "--max-primitive-steps",
        type=int,
        default=MAX_PRIMITIVE_STEPS,
        help="control steps one primitive may take before it is declared stuck",
    )
    p.add_argument(
        "--curobo-batch-size",
        type=int,
        default=3,
        help="the primitives' cuRobo batch (OmniGibson's default 3): parallel rollouts per plan; "
        "its GPU memory grows with it, 1 fits a 32 GB GPU shared with other servers",
    )
    p.add_argument(
        "--data-path",
        default=os.environ.get("OMNIGIBSON_DATA_PATH"),
        help="OmniGibson's data dir (og_dataset, assets); default OMNIGIBSON_DATA_PATH or the install's",
    )
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="exit when stdin closes (the parent died)",
    )
    add_perception_arguments(p, sam3=True)
    p.add_argument(
        "--molmo", default="", help="Molmo server URL: the `point` primitive"
    )
    args = p.parse_args()

    # OmniGibson's macros read these at import: set them before anything imports it.
    # Only the chosen GPU is visible (utils/gpu.py pin_isaac), so OmniGibson runs on its index 0;
    # sim.set_render_gpu says so the way the installed Isaac Sim needs it.
    args.gpu_id = pin_isaac(args.gpu_id)
    if args.gpu_id is None:
        args.gpu_id = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    sim.set_render_gpu()
    os.environ["OMNIGIBSON_HEADLESS"] = "1"
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    if args.data_path:
        os.environ["OMNIGIBSON_DATA_PATH"] = args.data_path
    if args.data_path is None:
        from omnigibson.macros import gm

        args.data_path = gm.DATA_PATH
    scene, instances = sim.find_task_scene(args.data_path, args.task)
    if args.seed not in instances:
        raise SystemExit(
            f"{args.task} has instances {instances}; --seed {args.seed} is not one"
        )
    # OmniGibson 3.9's instances are overlays on instance 0's template (sim.instance_split);
    # CaP-X's 3.7 dataset has a full template per instance.
    split = sim.instance_split(args.data_path, scene, args.task, args.seed)
    config = sim.task_config(
        activity=args.task,
        scene_model=scene,
        instance_id=args.seed if split is None else 0,
        image_size=args.image_size,
        grasping_mode=args.grasping_mode,
        max_steps=args.max_steps,
    )
    t0 = time.monotonic()
    handle = sim.launch(config, curobo_batch_size=args.curobo_batch_size)
    # A failure after Kit is up must not reach its shutdown, which can swallow the traceback and
    # exit 0: print it and leave hard.
    try:
        if split is not None:
            handle.env.reset()
            sim.load_task_instance(handle, split, args.seed)
        facade = BehaviorEnvFacade(
            handle=handle,
            meta={
                "task": args.task,
                "task_index": TASK_INDEX[args.task],
                "seed": args.seed,
                "instruction": LANGUAGE[args.task],
                "scene_model": scene,
                "instances": instances,
                "instance_split": split,
                "robot": "R1Pro",
                "cameras": sorted(sim.CAMERAS),
                "image_size": args.image_size,
                "grasping_mode": args.grasping_mode,
                "max_steps": args.max_steps,
                "curobo_batch_size": args.curobo_batch_size,
                "gpu_id": args.gpu_id,
            },
            max_primitive_steps=args.max_primitive_steps,
            sam3=args.sam3,
            molmo=args.molmo,
        )
        facade.reset()
        # --sam3 / --unidepth: env.detect, env.select_detection, env.reject_detection,
        # env.enhance_depth. The cameras follow OpenGL: no pinhole K for the camera point.
        install_perception(
            facade,
            args,
            cameras=sorted(sim.CAMERAS),
            view=render_view(facade, intrinsics=False),
            mutating=(
                "env.navigate_to_pose",
                "env.move_hand",
                "env.move_hand_delta",
                "env.grasp_object",
                "env.open_gripper",
                "env.close_gripper",
                "env.move_to_joints",
                "env.move_along_trajectory",
            ),
        )
        print(
            f"[behavior-env] {args.task} instance {args.seed} in {scene} ready in "
            f"{time.monotonic() - t0:.1f}s: {LANGUAGE[args.task]!r}",
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
