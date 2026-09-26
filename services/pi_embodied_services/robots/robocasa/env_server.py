# Copyright 2026 The RPent Authors.
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
# Modified by pi-embodied: import paths rewritten; healthz service name; HTTP is
# the only --transport.

"""RoboCasa env server — hosts the raw robosuite env in a subprocess, exposes basic calls via RPC.

Code mode (``code.run``, utils/code_exec.py ``CodeRunMixin``): a program calls the registry's
primitives (primitives.py) through its resolve, in a sandboxed subprocess, so the server requires
its RPC token and refuses other business calls while a program runs. A program's ``step`` receives
the robot's own observations only (the kitchen's object observations are privileged: they stay
out, as does the reward's info), and every step it takes adds an agentview frame to the run's
video; the run reports its env steps, the success and the new robot observation (``_finish_run``).
"""

import argparse
import inspect
import os
import re
import sys

import numpy as np

from pi_embodied_services.components.code_api import register_code_api
from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.robots.robocasa import tasks
from pi_embodied_services.robots.robocasa.primitives import ROBOCASA_PRIMITIVES
from pi_embodied_services.utils import ground_truth
from pi_embodied_services.utils.code_exec import CodeRunMixin
from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.perception import (
    add_perception_arguments,
    install_perception,
    render_view,
)
from pi_embodied_services.utils.rpc.main_thread_serve import MainThreadServeMixin

logger = get_logger("env_server")


DEFAULT_CAMS = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]

#: Code mode: the camera of the run's video (the agentview pi's tools show), its frame size, the
#: frames one run hands back (halved, every other one kept, when full) and the largest render a
#: program may ask for.
CODE_VIDEO_CAMERA = "robot0_agentview_left"
CODE_VIDEO_SIZE = 256
CODE_MAX_FRAMES = 128
CODE_MAX_RENDER = 1024
#: The PandaOmron controllers' travel per env step at a full (1.0) action
#: (robosuite default_pandaomron.json, control_freq 20): the arm's OSC_POSE moves at most 0.05 m,
#: the torso's JOINT_POSITION 0.05 m, the base's JOINT_VELOCITY 0.5 m/s for 1/20 s per axis.
ARM_M_PER_STEP = 0.05
TORSO_M_PER_STEP = 0.05
BASE_M_PER_STEP = 0.5 / 20


def robot_obs(obs) -> dict:
    """The robot's own observations of a robosuite obs dict (``robot0_*``, no camera images):
    the kitchen's object observations (``obj_*``, ``<object>_pos`` ...) are privileged."""
    return {
        k: v
        for k, v in obs.items()
        if k.startswith("robot0_") and not k.endswith(("_image", "_depth"))
    }


def _split_kwargs(split):
    """Replicate robocasa.utils.env_utils.create_env's split -> layout logic."""
    if split == "target":
        return {
            "obj_instance_split": "target",
            "layout_ids": None,
            "style_ids": None,
            "layout_and_style_ids": list(zip(range(1, 11), range(1, 11))),
        }
    if split == "pretrain":
        return {
            "obj_instance_split": "pretrain",
            "layout_ids": -2,
            "style_ids": -2,
            "layout_and_style_ids": None,
        }
    if split == "all":
        return {
            "obj_instance_split": None,
            "layout_ids": -3,
            "style_ids": -3,
            "layout_and_style_ids": None,
        }
    if split is None:
        return {
            "obj_instance_split": None,
            "layout_ids": None,
            "style_ids": None,
            "layout_and_style_ids": None,
        }
    raise ValueError('split must be {None,"all","pretrain","target"}')


class RoboCasaEnvFacade(CodeRunMixin, MainThreadServeMixin, BaseEnvFacade):
    """Wraps the raw robosuite env and exposes ONLY basic calls via RPC.

    Mixes in :class:`MainThreadServeMixin` so every env op runs on a single
    thread (the MuJoCo EGL context must stay on one thread); the inherited
    ``serve`` handles this.
    """

    SERVICE_NAME = "robocasa-env"

    def __init__(
        self,
        task_name,
        split="target",
        seed=0,
        scene=None,
        camera_h=256,
        camera_w=256,
        cameras=None,
        use_camera_obs=False,
    ):
        """``scene`` (a manifest index, 0-49) picks the task's scene seed from the
        RoboCasa365 table and reseeds every reset with it; without one ``seed`` seeds
        the env once, as robosuite does."""
        super().__init__()
        self.table = tasks.load_table()
        self.cameras = list(cameras) if cameras else list(DEFAULT_CAMS)
        self.camera_h, self.camera_w = camera_h, camera_w
        self.use_camera_obs = use_camera_obs
        self.env = None
        # Code mode: the env steps (all of them, and before the run), the run's last robot
        # observation and its video frames.
        self._steps = 0
        self._run_start = 0
        self._run_obs: dict | None = None
        self._run_frames: list[np.ndarray] = []
        self._make(task_name, split, seed, scene)

    def _make(self, task_name, split, seed, scene):
        """Build the robosuite env of ``task_name`` in ``split`` (closing the current one)."""
        import robocasa  # noqa: F401 — registers robocasa envs
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        if scene is not None:
            seed = tasks.scene_seed(self.table, task_name, split, scene)
        else:
            tasks.find_task(self.table, task_name)
        if self.env is not None:
            self.close()
        self.task_name, self.split, self.seed, self.scene = (
            task_name,
            split,
            seed,
            scene,
        )
        controller_config = load_composite_controller_config(
            controller=None, robot="PandaOmron"
        )
        env_kwargs = dict(
            env_name=task_name,
            robots="PandaOmron",
            controller_configs=controller_config,
            camera_names=self.cameras,
            camera_widths=self.camera_w,
            camera_heights=self.camera_h,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=self.use_camera_obs,  # off -> no per-step render (EGL-safe OSC loops)
            camera_depths=False,  # depth rendered on demand
            seed=seed,
            **_split_kwargs(split),
        )
        self.env = robosuite.make(**env_kwargs)
        self._meta = {
            "task_name": self.task_name,
            "split": self.split,
            "seed": self.seed,
            "scene": self.scene,
            "env_id": tasks.env_id(self.task_name, self.split)
            if self.split in self.table["splits"]
            else None,
            "camera_h": self.camera_h,
            "camera_w": self.camera_w,
        }

    def _register_rpc(self):
        """Register all RPC methods."""
        super()._register_rpc()
        self._rpc["env.list_tasks"] = self.list_tasks
        self._readonly_methods.add("env.list_tasks")
        self._rpc["env.check_success"] = self.check_success
        self._rpc["env.get_camera_transform"] = self.get_camera_transform
        self._rpc["env.grasp_contact"] = self.grasp_contact
        self._rpc["env.reassemble_env_action"] = self.reassemble_env_action
        self._rpc["env.get_success_criteria_text"] = self.get_success_criteria_text
        self._rpc["env.get_task_progress"] = self.get_task_progress
        self._rpc["env.ground_truth_poses"] = self.ground_truth_poses
        # Read-only methods
        self._readonly_methods.update(
            [
                "env.check_success",
                "env.get_camera_transform",
                "env.grasp_contact",
                "env.get_success_criteria_text",
                "env.get_task_progress",
            ]
        )
        api = register_code_api(self, ROBOCASA_PRIMITIVES)
        self._install_code_run(
            api,
            move_m=self._code_move_m,
            check=self._code_check,
            reply=self._code_reply,
            begin=self._begin_run,
            finish=self._finish_run,
        )

    # ---- code mode (run_code) ----

    def _begin_run(self) -> None:
        self._run_start = self._steps
        self._run_obs = None
        self._run_frames = []

    def _finish_run(self) -> dict:
        """The run's effect for pi: env steps taken, the success, the robot's observation after
        its last step (the ``robot0_*`` arrays pi's ``obs`` reads; None when it took none) and
        the run's video frames (top-down agentview)."""
        return {
            "steps": self._steps - self._run_start,
            "success": self.check_success(),
            "obs": self._run_obs,
            "frames": list(self._run_frames),
        }

    def _keep_frame(self, frame) -> None:
        if len(self._run_frames) >= CODE_MAX_FRAMES:
            self._run_frames = self._run_frames[::2]
        self._run_frames.append(frame)

    def _code_reply(self, method: str, out):
        """What a program receives of a ``step``: the robot's observations, the reward and done
        (no object observations, no info); the step's agentview goes to the run's video."""
        if method != "env.step":
            return out
        obs, reward, done, _info = out
        self._run_obs = robot_obs(obs)
        rgb = self.render_camera(
            CODE_VIDEO_CAMERA, CODE_VIDEO_SIZE, CODE_VIDEO_SIZE, False
        )
        # robosuite renders bottom-up; the video is top-down like pi's own frames.
        self._keep_frame(np.ascontiguousarray(np.asarray(rgb)[::-1]))
        return {"obs": self._run_obs, "reward": float(reward), "done": bool(done)}

    def _code_move_m(self, method: str, kwargs: dict) -> float:
        """How far one ``step`` may move the gripper (the run's translation cap): the arm's OSC
        travel, the torso's and the base's (drive and turn), each at the action's clipped size."""
        if method != "env.step":
            return 0.0
        a = np.clip(
            np.asarray(kwargs["flat_action"], dtype=np.float64).reshape(-1), -1, 1
        )
        if a.shape[0] < 11:
            return float(np.linalg.norm(a[:3])) * ARM_M_PER_STEP
        return float(
            np.linalg.norm(a[:3]) * ARM_M_PER_STEP
            + np.linalg.norm(a[7:9]) * BASE_M_PER_STEP
            # the base's yaw (0.5 rad/s) swings the gripper, within a metre of it, as far
            + abs(a[9]) * BASE_M_PER_STEP
            + abs(a[10]) * TORSO_M_PER_STEP
        )

    def _code_check(self, method: str, kwargs: dict) -> None:
        """Refuse a program's call that the run's wall clock could not bound."""
        if method in (
            "env.render_camera",
            "env.get_camera_meta",
            "env.get_camera_transform",
        ):
            for k in ("height", "width"):
                v = kwargs.get(k)
                if v is not None and int(v) > CODE_MAX_RENDER:
                    raise ValueError(f"{k} is at most {CODE_MAX_RENDER} in code mode")

    def get_env_meta(self):
        return self._meta

    def list_tasks(self, split):
        """The RoboCasa365 table's tasks of ``split`` (pretrain | target)."""
        return tasks.list_tasks(self.table, split)

    # ---- lifecycle ----
    def reset(self, task=None, split=None, scene=None):
        """Reset the env; ``task`` / ``split`` switch to another env (rebuilt when they
        differ from the current one), ``scene`` to another manifest scene of it."""
        if (task, split, scene) != (None, None, None):
            task = task if task is not None else self.task_name
            split = split if split is not None else self.split
            scene = scene if scene is not None else self.scene
            if (task, split) != (self.task_name, self.split):
                self._make(task, split, self.seed, scene)
            elif scene != self.scene:
                self.seed = tasks.scene_seed(self.table, task, split, scene)
                self.scene = scene
                self._meta.update(seed=self.seed, scene=self.scene)
        # RLDX_RESET_SEED=<episode_seed> -> reproduce the EXACT scene the fullshot eval
        # generated for that episode, seeded the SAME way as the eval's VideoRecordingWrapper
        # (random.seed + np.random.seed + robosuite env.rng/seed) BEFORE reset. Lets the
        # hybrid run on the IDENTICAL reset layouts fullshot was scored on (true paired
        # comparison). The eval formula: episode_seed = (run_seed + env_idx)*100000 + episode_id.
        # A manifest scene reseeds the same way with its scene seed, so every reset of it
        # samples the same kitchen and objects.
        rs_env = os.environ.get("RLDX_RESET_SEED")
        if rs_env or self.scene is not None:
            import random

            sd = int(rs_env) if rs_env else self.seed
            random.seed(sd)
            np.random.seed(sd)
            if hasattr(self.env, "seed"):
                self.env.seed = sd
            if hasattr(self.env, "rng"):
                self.env.rng = np.random.default_rng(sd)
        return self.env.reset()

    def step(self, flat_action):
        """flat_action: np.ndarray[12] = [eef_pos(3), eef_rot(3), gripper(1),
        base_motion(4), control_mode(1)] in the PandaOmron composite layout."""
        a = np.asarray(flat_action, dtype=np.float64).reshape(-1)
        assert a.shape[0] == self.env.action_dim, (
            f"action dim {a.shape[0]} != env.action_dim {self.env.action_dim}"
        )
        obs, reward, done, info = self.env.step(a)
        self._steps += 1
        return obs, reward, done, info

    def check_success(self):
        return bool(self.env._check_success())

    def render_camera(self, camera_name, height, width, depth):
        """sim.render in ROBOSUITE-NATIVE orientation (matches the camera
        transform matrices). rgb uint8 HxWx3, depth metric HxW."""
        import robosuite.utils.camera_utils as CU

        out = self.env.sim.render(
            width=width, height=height, camera_name=camera_name, depth=depth
        )
        if depth:
            rgb, d = out
            # Sanitize the raw OpenGL normalized depth into [0,1]: replace NaN/inf
            # (degenerate camera pose) then clip numerical overshoot. Otherwise an
            # assertion inside get_real_depth_map crashes the whole env server process.
            d = np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=0.0)
            d = np.clip(d, 0.0, 1.0)
            if d.ndim == 3:
                depth = CU.get_real_depth_map(self.env.sim, d)[..., 0]
            else:
                depth = CU.get_real_depth_map(self.env.sim, d[..., None])[..., 0]
            return rgb, depth
        return out

    def get_camera_meta(self, camera_name, height=None, width=None):
        import robosuite.utils.camera_utils as CU

        K = CU.get_camera_intrinsic_matrix(self.env.sim, camera_name, height, width)
        Ext = CU.get_camera_extrinsic_matrix(self.env.sim, camera_name)  # cam->world
        m = self.env.sim.model
        extent = m.stat.extent
        return {
            "camera_name": camera_name,
            "height": height,
            "width": width,
            "intrinsic": np.asarray(K, dtype=np.float64).tolist(),
            "extrinsic_cam2world": np.asarray(Ext, dtype=np.float64).tolist(),
            "depth_near": float(m.vis.map.znear * extent),
            "depth_far": float(m.vis.map.zfar * extent),
        }

    def get_camera_transform(self, camera_name, height=None, width=None):
        import robosuite.utils.camera_utils as CU

        T = CU.get_camera_transform_matrix(self.env.sim, camera_name, height, width)
        return np.linalg.inv(T)  # T_p2w

    def get_task_language(self) -> str | None:
        return self.env.get_ep_meta().get("lang")

    def grasp_contact(self):
        """Check if the gripper is currently contacting a task object."""
        try:
            robo = self.env  # robosuite Kitchen env
            grip = robo.robots[0].gripper  # {"right": GripperModel}
            for name, obj in robo.objects.items():
                try:
                    if robo._check_grasp(grip, obj):
                        return True, name
                except Exception:
                    continue
        except Exception:
            pass
        return False, None

    def reassemble_env_action(self, unmap_result):
        """Reassemble the unmap result into a flat action using the env's robots."""
        from robosuite.controllers.composite.composite_controller import (
            HybridMobileBase,
        )

        env_action = []
        for robot in self.env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            a = np.zeros(cc.action_limits[0].shape)
            for part_name in cc.part_controllers:
                s, e = cc._action_split_indexes[part_name]
                a[s:e] = unmap_result.pop(f"{pf}{part_name}")
            if isinstance(cc, HybridMobileBase):
                a[-1] = unmap_result.pop(f"{pf}base_mode")
            env_action.append(a)
        return np.concatenate(env_action)

    def get_success_criteria_text(self):
        """Return the success_criteria.md text for this task."""
        env = self.env
        out = []
        try:
            src = inspect.getsource(type(env)._check_success)
            out.append(
                "# SUCCESS CONDITION for this task (env._check_success)\n"
                "# You must make this return True. Object positions are NOT given —\n"
                "# localize every named object/fixture from the camera+world maps.\n\n"
                + src
            )
            try:
                import robocasa.utils.object_utils as OU

                for fn in sorted(set(re.findall(r"OU\.(\w+)\(", src))):
                    f = getattr(OU, fn, None)
                    if f is not None:
                        try:
                            out.append(
                                "## helper OU.%s\n%s" % (fn, inspect.getsource(f))
                            )
                        except Exception:
                            pass
            except Exception:
                pass
            for fix, meth in sorted(set(re.findall(r"self\.(\w+)\.(\w+)\(", src))):
                obj = getattr(env, fix, None)
                if obj is not None and hasattr(type(obj), meth):
                    try:
                        out.append(
                            "## %s.%s\n%s"
                            % (fix, meth, inspect.getsource(getattr(type(obj), meth)))
                        )
                    except Exception:
                        pass
        except Exception as ex:
            out.append("(_check_success extraction failed: %s)" % ex)
        return "\n\n".join(out)[:9000]

    def get_task_progress(self):
        """Return the progress dict for this task."""
        env = self.env
        prog = {}
        code = type(env)._check_success.__code__
        try:
            src = inspect.getsource(type(env)._check_success)
            # capture both `self.attr` AND dotted `self.fixture._attr` paths used in the
            # success check (e.g. self.coffee_machine._turned_on) — a bare-attr regex
            # would only grab "coffee_machine" (the fixture object) and miss the real
            # gating flag. Resolve each dotted path to its live scalar/bool value.
            for path in sorted(
                set(re.findall(r"self\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)", src))
            ):
                obj = env
                ok = True
                for part in path.split("."):
                    obj = getattr(obj, part, None)
                    if obj is None:
                        ok = False
                        break
                if not ok:
                    continue
                key = path.replace(".", "_")
                if isinstance(obj, (bool, np.bool_)):
                    prog[key] = bool(obj)
                elif isinstance(obj, (int, np.integer)):
                    prog[key] = int(obj)
                elif isinstance(obj, (float, np.floating)):
                    prog[key] = round(float(obj), 4)
        except Exception:
            pass
        # trace ONE read-only call of _check_success; grab its return-frame locals
        captured = {}

        def _tracer(frame, event, arg):
            if event == "call" and frame.f_code is code:

                def _local(f, e, a):
                    if e == "return":
                        captured.update(f.f_locals)
                    return _local

                return _local
            return None

        old = sys.gettrace()
        try:
            sys.settrace(_tracer)
            env._check_success()
        except Exception:
            pass
        finally:
            sys.settrace(old)
        for k, v in captured.items():
            if k == "self" or k in prog:
                continue
            if isinstance(v, (bool, np.bool_)):
                prog[k] = bool(v)
            elif isinstance(v, (int, np.integer)):
                prog[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                prog[k] = round(float(v), 4)
        return prog

    def ground_truth_poses(self, names=None):
        """World poses of ``names`` (default all) from the kitchen's own object list
        (``--privileged``): its objects (``obj_body_id``), then its fixtures' root bodies."""
        env = self.env
        ids = dict(env.obj_body_id)
        for name, fixture in env.fixtures.items():
            try:
                ids.setdefault(name, env.sim.model.body_name2id(fixture.root_body))
            except (KeyError, ValueError):
                pass  # a fixture merged into another body has none of its own
        return ground_truth.respond(ground_truth.mujoco_body_poses(env.sim, ids), names)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device to pin MuJoCo EGL rendering and the torch "
        "default device to (physical CUDA ordinal).",
    )
    p.add_argument("--task-name", default="OpenDrawer")
    p.add_argument("--split", default="target")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--scene",
        type=int,
        default=None,
        help="RoboCasa365 manifest scene index (0-49); overrides --seed",
    )
    add_perception_arguments(p, sam3=True)
    args = p.parse_args()

    if args.cuda_device is not None:
        # Deliberately do NOT set CUDA_VISIBLE_DEVICES. robosuite (imported
        # transitively via libero) asserts at import time that
        # ``MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES`` (substring check),
        # which assumes the EGL index equals the CUDA ordinal and crashes on
        # multi-GPU boxes where the EGL order differs. That assertion is gated
        # on ``CUDA_VISIBLE_DEVICES != ""``, so leaving it unset skips it in
        # both this process and the multiprocessing-spawned render workers
        # (which inherit the env). Pin the two backends directly instead:
        #   - MuJoCo render device <- MUJOCO_EGL_DEVICE_ID (configure_egl_device)
        #   - torch default device  <- torch.cuda.set_device(N)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is set; clearing it and pinning via "
                "MUJOCO_EGL_DEVICE_ID + torch.cuda.set_device(--cuda-device=%s) "
                "instead (robosuite's CVD assertion is incompatible with EGL<->CUDA mapping)",
                prev,
                args.cuda_device,
            )
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        from pi_embodied_services.utils.egl import configure_egl_device

        configure_egl_device(args.cuda_device)
        import torch

        torch.cuda.set_device(args.cuda_device)

    facade = RoboCasaEnvFacade(
        args.task_name,
        split=args.split,
        seed=args.seed,
        scene=args.scene,
    )
    # --sam3 / --unidepth: env.detect, env.select_detection, env.reject_detection,
    # env.enhance_depth, on the upright 256 px views the model sees.
    install_perception(
        facade,
        args,
        cameras=["agentview", "navview", "wrist"],
        view=render_view(
            facade,
            size=256,
            flip=True,
            cameras={
                "agentview": "robot0_agentview_left",
                "navview": "mobilebase0_navview",
                "wrist": "robot0_eye_in_hand",
            },
        ),
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
