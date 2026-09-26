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
steps. Success is the BDDL task's ``success`` termination and ``q_score`` BEHAVIOR's
partial-success metric; both are read from the simulator after every call and go into every
observation. What the simulator knows and a camera cannot see (the object OmniGibson's
grasping holds, the reference "picked" judgement) is reported apart, in ``privileged``, and
the pi robot shows it to the planner only under ``--privileged``.

Isaac Sim starts in ``main`` (minutes: the scene is a whole house), before the server binds, so
healthz answers only once the task is loaded. Every call runs on the main thread (Kit is not
thread-safe). ``OMNIGIBSON_GPU_ID`` (``--gpu-id``) picks the simulator's GPU; the perception
servers should sit on another one.
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
from pi_embodied_services.robots.behavior import sim
from pi_embodied_services.robots.behavior.primitives import BEHAVIOR_PRIMITIVES
from pi_embodied_services.robots.behavior.tasks import LANGUAGE, TASK_INDEX, TASK_NAMES
from pi_embodied_services.utils import ground_truth
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


def check_arm(arm: str) -> str:
    if arm not in sim.ARMS:
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
    return arm


def as_pose(position, quat_xyzw, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A world pose from a position and an optional xyzw orientation (default ``current``)."""
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(pos)):
        raise ValueError(f"position must be finite, got {position!r}")
    quat = (
        current
        if quat_xyzw is None
        else np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    )
    norm = float(np.linalg.norm(quat))
    if not norm > 0:
        raise ValueError("quat_xyzw must not be zero")
    return pos, quat / norm


class BehaviorEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """One BEHAVIOR task env plus OmniGibson's semantic primitives."""

    SERVICE_NAME = "behavior-env"

    def __init__(
        self,
        *,
        handle: sim.Handle,
        meta: dict,
        max_primitive_steps: int = MAX_PRIMITIVE_STEPS,
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
        self._initial_goals: list[list[bool]] = []
        self._initial_heights: dict[str, float] = {}

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["env.navigate_to_pose"] = self.navigate_to_pose
        self._rpc["env.move_hand"] = self.move_hand
        self._rpc["env.grasp_object"] = self.grasp_object
        self._rpc["env.open_gripper"] = self.open_gripper
        self._rpc["env.close_gripper"] = self.close_gripper
        self._rpc["env.get_robot_position"] = self.get_robot_position
        self._rpc["env.raw_obs"] = self.raw_obs
        self._rpc["env.state"] = self.state
        self._rpc["env.ground_truth_poses"] = self.ground_truth_poses
        register_code_api(self, BEHAVIOR_PRIMITIVES)

    # ---- stepping ----

    def _absorb(self, result: tuple) -> None:
        """Record one ``env.step`` result."""
        obs, _reward, term, trunc, info = result
        self._obs = obs
        self._info = info
        self._steps += 1
        self._terminated |= bool(term)
        self._truncated |= bool(trunc)

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
        solved = sim.success(self._info, self._task)
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
            {"success": sim.success(self._info, self._task)},
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

    def move_hand(self, arm: str, position, quat_xyzw=None) -> dict:
        """Plan and move ``arm``'s end effector to a world pose, obstacles respected (cuRobo).
        The orientation defaults to the current one. Refused beyond ``MAX_HAND_REACH_M`` of the base."""
        arm = check_arm(arm)
        _p, cur = sim.eef_pose(self._robot, arm)
        pos, quat = as_pose(position, quat_xyzw, cur)
        base, _q, _yaw = sim.base_pose(self._robot)
        reach = float(np.linalg.norm(pos[:2] - base[:2]))
        if not reach <= MAX_HAND_REACH_M:
            raise ValueError(
                f"target is {reach:.2f} m from the base in xy; the arm reaches {MAX_HAND_REACH_M} m: navigate first"
            )
        report = self._primitive("move_hand", ("move", self._hand(arm, pos, quat)))
        return {**self._pack(), **self._hand_report(arm, pos, report)}

    def grasp_object(
        self,
        arm: str,
        position,
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
        pos, quat = as_pose(position, quat_xyzw, cur)
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
        "--gpu-id", type=int, default=0, help="GPU for Isaac Sim (OMNIGIBSON_GPU_ID)"
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
        "--data-path",
        default=os.environ.get("OMNIGIBSON_DATA_PATH"),
        help="OmniGibson's data dir (og_dataset, assets); default OMNIGIBSON_DATA_PATH or the install's",
    )
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="exit when stdin closes (the parent died)",
    )
    args = p.parse_args()

    # OmniGibson's macros read these at import: set them before anything imports it.
    os.environ["OMNIGIBSON_GPU_ID"] = str(args.gpu_id)
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
    config = sim.task_config(
        activity=args.task,
        scene_model=scene,
        instance_id=args.seed,
        image_size=args.image_size,
        grasping_mode=args.grasping_mode,
        max_steps=args.max_steps,
    )
    t0 = time.monotonic()
    handle = sim.launch(config)
    # A failure after Kit is up must not reach its shutdown, which can swallow the traceback and
    # exit 0: print it and leave hard.
    try:
        facade = BehaviorEnvFacade(
            handle=handle,
            meta={
                "task": args.task,
                "task_index": TASK_INDEX[args.task],
                "seed": args.seed,
                "instruction": LANGUAGE[args.task],
                "scene_model": scene,
                "instances": instances,
                "robot": "R1Pro",
                "cameras": sorted(sim.CAMERAS),
                "image_size": args.image_size,
                "grasping_mode": args.grasping_mode,
                "max_steps": args.max_steps,
                "gpu_id": args.gpu_id,
            },
            max_primitive_steps=args.max_primitive_steps,
        )
        facade.reset()
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
