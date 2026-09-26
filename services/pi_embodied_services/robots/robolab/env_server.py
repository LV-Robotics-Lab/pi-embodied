# Copyright 2026 The Show-Harness Authors.
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
# Modified by pi-embodied: interpreters/robolab_atomic_controller.py and the execution loop of
# core/sim/mvtoken_robolab_runner.py (@137d571) turned into an RPC env server; the calibration
# of configs/robot_robolab.yaml is kept, the VLM loop lives in the pi robot.

"""RPC server wrapping one RoboLab (Isaac Lab) task on a Franka + Panda hand.

``env.move_delta`` is the motion primitive: a base-frame translation in metres runs as
ceil(|delta| / 2 cm) decisions (Show-Harness's action lattice); each decision commands
``command_gain`` x its share over ``steps_per_decision`` relative-IK control steps, because
RoboLab's relative IK achieves a constant ~28% of what it is asked for (0.072 m commanded over
8 steps measured 20.1 mm). The orientation is held at the reset pose by a per-step correction
in the rotation slots. A gripper command holds still for ``gripper_hold_steps`` so the fingers
finish moving. Success is the task's own termination predicate.

Isaac Sim starts in ``__init__`` (tens of seconds, minutes on a cold shader cache), before the
server binds, so healthz answers only once the task is loaded. Every call runs on the main
thread (the Kit app is not thread-safe).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from pi_embodied_services.components.code_api import register_code_api
from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.robots.robolab import sim
from pi_embodied_services.robots.robolab.primitives import ROBOLAB_PRIMITIVES
from pi_embodied_services.utils import ground_truth
from pi_embodied_services.utils.rpc.main_thread_serve import MainThreadServeMixin

#: configs/robot_robolab.yaml: physical metres per decision (the MVTOKEN 2 cm convention).
STEP_M = 0.02
#: Commanded per physical metre: step_m 0.072 commanded over 8 control steps measured 20.12 mm
#: (sd 0.13 mm, max off-axis 0.32 mm) on BananaInBowlTask.
COMMAND_GAIN = 0.072 / 0.02
STEPS_PER_DECISION = 8
GRIPPER_HOLD_STEPS = 10
#: Per-control-step safety cap on the IK target step, m.
MAX_DELTA_M = 0.05
#: Largest translation one call may command, m.
MAX_MOVE_M = 0.3
#: BinaryJointPositionZeroToOneAction: action > 0.5 closes.
OPEN, CLOSE = 0.0, 1.0
#: The wrist camera's raw frame has the fingertips entering from the left; 270 deg CCW puts them
#: at the top (configs/robot_robolab.yaml wrist_rotation_degrees, measured for the Panda hand).
WRIST_ROT90 = 3


class RobolabEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """One RoboLab task env (``num_envs=1``) plus the Show-Harness relative-IK controller."""

    SERVICE_NAME = "robolab-env"

    def __init__(self, *, app: Any, handle: sim.TaskHandle, meta: dict):
        super().__init__()
        self._app = app
        self._h = handle
        self._env = handle.env
        self._meta = dict(
            meta, ik_scale=handle.ik_scale, instruction=handle.instruction
        )
        self._obs: dict = {}
        self._closed = False
        self._gripper = OPEN
        self._quat_ref: np.ndarray | None = None
        self._steps = 0
        self._terminated = self._truncated = False

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["env.move_delta"] = self.move_delta
        self._rpc["env.state"] = self.state
        self._rpc["env.ground_truth_poses"] = self.ground_truth_poses
        register_code_api(self, ROBOLAB_PRIMITIVES)

    # ---- helpers ----

    def _images(self) -> dict:
        agentview = sim.rl_rgb(self._obs, self._meta["agentview_camera"])
        wrist = np.ascontiguousarray(
            np.rot90(sim.rl_rgb(self._obs, self._meta["wrist_camera"]), k=WRIST_ROT90)
        )
        return {"agentview": agentview, "wrist": wrist}

    def _state(self) -> dict:
        quat = sim.rl_ee_quat(self._env)
        subtask = sim.rl_subtask(self._env) if self._meta.get("subtask") else None
        return {
            **({"subtask": subtask} if subtask is not None else {}),
            "eef_pos": sim.rl_tcp(self._env).astype(np.float32),
            "eef_quat_wxyz": quat.astype(np.float32),
            "tilt_deg": round(sim.ee_tilt_deg(quat), 2),
            "gripper_width": round(sim.rl_gripper_width(self._env), 5),
            "gripper_command": "close" if self._gripper == CLOSE else "open",
            "success": sim.rl_success(self._env) or self._terminated,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "env_steps": self._steps,
        }

    def _pack(self) -> dict:
        return {**self._images(), **self._state()}

    def _action(self, delta_m: np.ndarray) -> np.ndarray:
        """``[dx, dy, dz, rot hold, gripper]`` for one control step (the hold uses the current pose)."""
        delta = (
            np.clip(np.asarray(delta_m, dtype=float), -MAX_DELTA_M, MAX_DELTA_M)
            / self._h.ik_scale
        )
        rot = np.zeros(3)
        if self._quat_ref is not None:
            rot = (
                sim.hold_orientation_rotvec(self._quat_ref, sim.rl_ee_quat(self._env))
                / self._h.ik_scale
            )
        return np.asarray([*delta, *rot, self._gripper], dtype=np.float32)

    def _control(self, delta_m: np.ndarray, n: int) -> tuple[int, bool]:
        """Send ``n`` control steps of ``delta_m`` each; returns (steps run, cancelled)."""
        for i in range(n):
            if self.stop_requested():
                return i, True
            self._obs, term, trunc = sim.step(self._env, self._action(delta_m))
            self._steps += 1
            self._terminated |= term
            self._truncated |= trunc
            if term or trunc:
                # RoboLab freezes a terminated env; a truncated one was reset by Isaac Lab.
                return i + 1, False
        return n, False

    # ---- gym-like surface ----

    def reset(self):
        """Reset the scene, open the gripper, hold still ``settle_steps``, latch the orientation."""
        self._quat_ref = None
        self._gripper = OPEN
        self._steps = 0
        self._terminated = self._truncated = False
        self._obs = sim.reset(self._env)
        self._control(np.zeros(3), int(self._meta["settle_steps"]))
        self._quat_ref = sim.rl_ee_quat(self._env)
        return self._pack(), {"instruction": self._h.instruction}

    def step(self, action):
        """One raw relative-IK control step ``[dx, dy, dz, drx, dry, drz, gripper]`` (no hold)."""
        self._obs, term, trunc = sim.step(
            self._env, np.asarray(action, dtype=np.float32).reshape(-1)
        )
        self._steps += 1
        self._terminated |= term
        self._truncated |= trunc
        return self._pack(), 0.0, term, trunc, {"success": sim.rl_success(self._env)}

    def chunk_step(self, actions, *, return_all_frames: bool = False):
        """Raw control steps ``[N, 7]``; stops early on ``stop``, termination or truncation."""
        frames, info = [], {}
        for action in np.asarray(actions, dtype=np.float32).reshape(-1, 7):
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

    def move_delta(
        self, delta_xyz, gripper: str | None = None, return_frames: bool = False
    ) -> dict:
        """Translate the hand by a base-frame ``delta_xyz`` (m), optionally after a gripper command.

        Returns the new observation plus ``moved_m`` and step counts. With ``return_frames``,
        ``frames`` holds the agentview after the gripper hold and after every decision.
        Refused once the episode has terminated or timed out.
        """
        start = sim.rl_tcp(self._env)
        report: dict[str, Any] = {"commanded_m": [float(v) for v in delta_xyz]}
        frames: list[np.ndarray] = []
        if self._terminated or self._truncated:
            return {
                **self._pack(),
                **report,
                "error": "the episode is over",
                "moved_m": [0.0, 0.0, 0.0],
            }
        delta = np.asarray(delta_xyz, dtype=float).reshape(3)
        norm = float(np.linalg.norm(delta))
        if not norm <= MAX_MOVE_M:
            raise ValueError(
                f"delta moves {norm:.3f} m; the limit is {MAX_MOVE_M} m per call"
            )
        steps = 0
        cancelled = False
        if gripper is not None:
            if gripper not in ("open", "close"):
                raise ValueError(
                    f"gripper must be 'open', 'close' or null, got {gripper!r}"
                )
            target = CLOSE if gripper == "close" else OPEN
            if target != self._gripper:
                self._gripper = target
                steps, cancelled = self._control(np.zeros(3), GRIPPER_HOLD_STEPS)
                if return_frames:
                    frames.append(self._images()["agentview"])
        decisions = math.ceil(norm / STEP_M - 1e-9) if norm > 1e-9 else 0
        per_step = delta / max(decisions, 1) * COMMAND_GAIN / STEPS_PER_DECISION
        done = 0
        for _ in range(decisions):
            if cancelled or self._terminated or self._truncated:
                break
            n, cancelled = self._control(per_step, STEPS_PER_DECISION)
            steps += n
            done += 1
            if return_frames:
                frames.append(self._images()["agentview"])
        end = sim.rl_tcp(self._env)
        report.update(
            moved_m=[round(float(v), 4) for v in end - start],
            decisions=done,
            control_steps=steps,
        )
        if cancelled:
            report["cancelled"] = True
        if return_frames:
            report["frames"] = frames
        return {**self._pack(), **report}

    def state(self) -> dict:
        """The current state (no stepping, no images)."""
        return self._state()

    def ground_truth_poses(self, names=None) -> dict:
        """Poses of ``names`` (default all) of the task's objects (``--privileged``), in the
        frame of ``eef_pos``."""
        return ground_truth.respond(sim.rl_object_poses(self._env), names)

    def render_camera(self, camera_name: str = "agentview", **_: Any):
        """Latest frame of ``agentview`` (front camera) or ``wrist`` (rotated, fingertips at the top)."""
        return self._images()[camera_name]

    def get_camera_meta(self, camera_name: str = "agentview", **_: Any) -> dict:
        """OpenCV intrinsics and camera-to-base extrinsic of the front camera."""
        if camera_name != "agentview":
            raise ValueError(
                "only the agentview (front camera) has a fixed calibration"
            )
        from pi_embodied_services.robots.robolab import franka

        cam2base = np.eye(4)
        cam2base[:3, :3] = np.asarray(franka.FRONT_CAM_R)
        cam2base[:3, 3] = franka.FRONT_CAM_POS
        return {
            "intrinsic_K": np.asarray(franka.FRONT_CAM_K, dtype=np.float64).reshape(
                3, 3
            ),
            "extrinsic_cam2world": cam2base,
            "width": franka.FRONT_CAM_W,
            "height": franka.FRONT_CAM_H,
        }

    def get_task_language(self) -> str:
        return self._h.instruction

    def get_env_meta(self) -> dict:
        return dict(self._meta)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._env.close()
        finally:
            print("[robolab-env] closing Isaac Sim", flush=True)
            self._app.close()


def fresh_dir(path: Path) -> Path:
    """Create ``path``; refuse a non-empty one (RoboLab's HDF5 recorder locks its output dir)."""
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"output dir {path} is not empty; refusing to reuse it")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--task", default="BananaInBowlTask", help="RoboLab task class name")
    p.add_argument(
        "--seed", type=int, default=0, help="env seed (fixed at construction)"
    )
    p.add_argument(
        "--instruction-type",
        default="default",
        choices=["default", "vague", "specific"],
    )
    p.add_argument(
        "--enable-subtask",
        action="store_true",
        help="track RoboLab's subtask progress (partial-credit score; extra physics queries per step)",
    )
    p.add_argument("--camera-preset", default="WRIST_LEFT", choices=sim.CAMERA_PRESETS)
    p.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="physical GPU for physics, rendering and torch",
    )
    p.add_argument(
        "--renderer", default="realtime", choices=["realtime", "pathtracing"]
    )
    p.add_argument(
        "--rendering-type",
        default=None,
        choices=[None, "performance", "balanced", "quality"],
    )
    p.add_argument(
        "--episode-length-s",
        type=float,
        default=None,
        help="override the task's time limit",
    )
    p.add_argument(
        "--settle-steps",
        type=int,
        default=8,
        help="hold steps after reset (num_steps_wait)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="RoboLab's artefacts (must be new or empty); default "
        "$PI_EMBODIED_LOGS/robolab/<task>-s<seed>-<time>-<pid>",
    )
    p.add_argument(
        "--isaac-assets",
        default=os.environ.get("ROBOLAB_ISAAC_ASSETS"),
        help="local mirror of Isaac's Assets/Isaac/5.0 tree (default: fetch from Isaac's S3)",
    )
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="exit when stdin closes (the parent died)",
    )
    args = p.parse_args()

    # Vulkan ignores CUDA_VISIBLE_DEVICES, so the renderer is pinned by index (sim.launch_isaac);
    # CUDA must see the same numbering.
    if os.environ.pop("CUDA_VISIBLE_DEVICES", None) is not None:
        print(
            "[robolab-env] ignoring CUDA_VISIBLE_DEVICES; using --cuda-device",
            flush=True,
        )
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    device = f"cuda:{args.cuda_device}"
    logs = Path(
        os.environ.get("PI_EMBODIED_LOGS", Path.home() / ".cache" / "pi-embodied")
    )
    out = Path(
        args.output_dir
        or logs
        / "robolab"
        / f"{args.task}-s{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    )
    fresh_dir(out)
    os.environ["ROBOLAB_PANDA_USD"] = str(
        sim.localize_panda_usd(out, args.isaac_assets)
    )

    app = sim.launch_isaac(device=device)
    # A failure after the app is up must not reach SimulationApp.close(), which can swallow the
    # traceback and exit 0: print it and leave hard.
    try:
        t0 = time.monotonic()
        handle = sim.make_task(
            args.task,
            device=device,
            seed=args.seed,
            instruction_type=args.instruction_type,
            camera_preset=args.camera_preset,
            renderer=args.renderer,
            rendering_type=args.rendering_type,
            output_dir=out,
            episode_length_s=args.episode_length_s,
            enable_subtask=args.enable_subtask,
        )
        facade = RobolabEnvFacade(
            app=app,
            handle=handle,
            meta={
                "task": args.task,
                "seed": args.seed,
                "instruction_type": args.instruction_type,
                "subtask": bool(args.enable_subtask),
                "robot": "franka",
                "agentview_camera": "front_cam",
                "wrist_camera": "wrist_cam",
                "step_m": STEP_M,
                "command_gain": COMMAND_GAIN,
                "steps_per_decision": STEPS_PER_DECISION,
                "gripper_hold_steps": GRIPPER_HOLD_STEPS,
                "settle_steps": args.settle_steps,
                "episode_length_s": float(handle.env.cfg.episode_length_s),
                "control_hz": 15,
                "output_dir": str(out),
            },
        )
        facade.reset()
        print(
            f"[robolab-env] {args.task} ready in {time.monotonic() - t0:.1f}s: {handle.instruction!r}",
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
