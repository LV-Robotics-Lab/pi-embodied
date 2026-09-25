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
# Modified by pi-embodied: adapted from Show-Harness core/sim/maniskill_task.py and
# core/sim/maniskill_scenes.py (the stock scenes here; the RLinf real2sim rigs in
# ./scenes.py) into an RPC env server; the atomic-token controller lives in the pi robot.

"""RPC server wrapping one ManiSkill 3 env in ``pd_ee_delta_pos``.

Action ``[dx, dy, dz, gripper]`` in [-1, 1]: a base-frame position delta normalised by
the arm's 0.1 m bound, and the Panda mimic gripper (+1 open, -1 close). Observations
carry the agentview (``base_camera``; ``external_cam`` on the RLinf rigs) and wrist
(``hand_camera``) RGB, the TCP pose and the gripper opening; ``info`` is flattened to
plain scalars (``success``, ``is_grasped``, ...). An env id of ./scenes.py (BlockPAP-v1,
BlockStack-v1) runs that rig with Show-Harness's calibrated cameras, reset and views.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.rpc.main_thread_serve import MainThreadServeMixin

logger = get_logger("env_server")

#: Show-Harness core/sim/maniskill_scenes.py SCENES (stock ManiSkill rows): the task text.
INSTRUCTIONS = {
    # PickCube's success is the cube inside a goal sphere (goal_thresh 2.5 cm, up to 0.3 m above
    # the table), not the lift alone: the text names it, and SHOW_GOALS renders the sphere.
    "PickCube-v1": "pick up the red cube and move it into the green goal sphere",
    "StackCube-v1": "stack the red cube on top of the green cube",
    "PushCube-v1": "push the cube to the goal marker",
    "PullCube-v1": "pull the cube to the goal marker",
    "PokeCube-v1": "poke the cube to the goal marker",
    "LiftPegUpright-v1": "lift the peg upright",
}
CAMERAS = {"agentview": "base_camera", "wrist": "hand_camera"}
#: The actors each task needs the model to see, by env attribute: checked in the agentview
#: at every reset (``check_visible``).
TASK_ACTORS = {
    "PickCube-v1": ["cube", "goal_site"],
    "StackCube-v1": ["cubeA", "cubeB"],
    "PushCube-v1": ["obj", "goal_region"],
    "PullCube-v1": ["obj", "goal_region"],
    "PokeCube-v1": ["cube", "peg", "goal_region"],
    "LiftPegUpright-v1": ["peg"],
}
#: Success markers a task keeps in ``_hidden_objects`` (drawn for the human viewer only, never
#: in the sensor cameras) although its success depends on them: shown to the cameras at every
#: reset, so the model can see the goal (and ``check_visible`` can require it).
SHOW_GOALS = {"PickCube-v1": ["goal_site"]}
#: Fewest agentview pixels (640x480 sensor) a task actor may show; a 4 cm cube at the far
#: edge of the workspace covers ~60.
MIN_VISIBLE_PX = 20
OPEN = 1.0
#: Stock Panda arm_pd_ee_delta_pos position bound: action 1.0 = 0.1 m.
DELTA_BOUND_M = 0.1


def _np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def show_goals(env, names: list[str]) -> None:
    """Take the actors ``names`` of ``env`` (unwrapped) out of ``_hidden_objects`` and show them.

    ManiSkill hides every hidden object before each sensor capture; ``_load_scene`` re-adds
    them on a reconfiguring reset, so this runs after every reset."""
    goals = [getattr(env, n) for n in names]
    env._hidden_objects = [
        o for o in env._hidden_objects if not any(o is g for g in goals)
    ]
    for g in goals:
        g.show_visual()


def _letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """Equal-ratio resize into a ``size`` square with centred black bars (Show-Harness
    ``prepare_view(square_size=...)``, the real rigs' ``resize_with_pad``)."""
    from PIL import Image

    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = np.asarray(Image.fromarray(image).resize((nw, nh), Image.BILINEAR))
    out = np.zeros((size, size, 3), dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _orient(image: np.ndarray, degrees: int, flip: str) -> np.ndarray:
    """Show-Harness core/record/images.rotate_and_flip: rotate CCW, then flip."""
    k = (int(degrees) % 360) // 90
    if k:
        image = np.rot90(image, k=k)
    if flip in ("vertical", "both"):
        image = image[::-1]
    if flip in ("horizontal", "both"):
        image = image[:, ::-1]
    return np.ascontiguousarray(image)


#: Show-Harness core/sim/maniskill_scenes.py WRIST_MOUNTS["centered"]: the D415 orientation
#: on ``panda_hand`` without the stock rig's 2 cm lateral ``camera_link`` hop, so the finger
#: pair is centred (on their rig ``wrist_flip: both`` then puts the fingertips at the top;
#: the stock scenes' start pose needs a 270 deg rotation instead, see ``main``).
_Q_D415 = [0.0, 0.7071068, 0.0, 0.7071068]  # wxyz


def center_wrist_camera() -> None:
    """Move the stock ``panda_wristcam`` hand camera to the centred mount.

    Patched in place rather than registered under a new uid: the stock tabletop scenes
    (TableSceneBuilder) only place and pose robots whose uid they know."""
    import sapien
    from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
    from mani_skill.sensors.camera import CameraConfig

    PandaWristCam._sensor_configs = property(
        lambda self: [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0.035, 0.0, 0.036], q=_Q_D415),
                width=256,
                height=256,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["panda_hand"],
            )
        ]
    )


#: Agentview for the stock scenes. Show-Harness's calibrated ``external_cam`` exists only on
#: the RLinf real2sim rigs (BlockPAP-v1 / BlockStack-v1, ./scenes.py). The stock
#: ``base_camera`` (eye [0.3, 0, 0.6], 128 px) faces the robot head-on, so the arm and
#: gripper hide a cube under the TCP. This pose (compared against the stock camera and
#: 7 others on PickCube/StackCube/PushCube seed 0, at reset and with the fingertips at the
#: cube) sits low in front of the robot, 15 deg toward its left: the cube stays visible at
#: reset and beside the fingers during the descent, and image left/right and bottom/top
#: stay close to the robot's -y/+y and +x/-x. 640x480 like their external_cam.
AGENTVIEW = {
    "eye": [0.6, 0.16, 0.45],
    "target": [-0.08, 0.0, 0.05],
    "fov_deg": 60.0,
    "width": 640,
    "height": 480,
}


class ManiskillEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """One ManiSkill env (``num_envs=1``); every call runs on the main thread."""

    SERVICE_NAME = "maniskill-env"

    def __init__(
        self,
        *,
        env_id: str,
        seed: int,
        robot_uids: str = "panda_wristcam",
        wrist_mount: str = "centered",
        control_mode: str = "pd_ee_delta_pos",
        sim_backend: str = "physx_cpu",
        agentview: str = "oblique",
        view_size: int = 256,
        max_episode_steps: int = 100_000,
        settle_steps: int = 8,
        wrist_rotation: int = 270,
        wrist_flip: str = "none",
        scene: dict | None = None,
    ):
        super().__init__()
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 -- registers the stock env ids

        from pi_embodied_services.robots.maniskill import scenes

        #: An RLinf real2sim rig (./scenes.py) and its options, or None for a stock scene.
        self._rig = scenes.SCENES.get(env_id)
        self._scene = None
        self._cameras = CAMERAS
        table_z = 0.0
        if self._rig:
            # The rig fixes the robot, both cameras and the view transform (its training
            # contract); the stock-scene camera arguments do not apply.
            self._scene = scenes.scene_options(env_id, **(scene or {}))
            self._env = scenes.make_env(
                env_id,
                self._scene,
                obs_mode="rgb+segmentation",
                control_mode=control_mode,
                sim_backend=sim_backend,
                max_episode_steps=int(max_episode_steps),
            )
            self._cameras = scenes.CAMERAS
            table_z = float(self._env.unwrapped.TABLE_Z)
            robot_uids, agentview = self._rig.robot_uids, scenes.CAMERAS["agentview"]
            wrist_mount = self._scene["wrist_mount"]
            wrist_rotation = scenes.VIEWS["wrist"]["rotation"]
            wrist_flip = scenes.VIEWS["wrist"]["flip"]
        else:
            if scene:
                raise ValueError(f"--scene options apply to {list(scenes.SCENES)} only")
            if wrist_mount == "centered":
                center_wrist_camera()
            self._env = gym.make(
                env_id,
                num_envs=1,
                obs_mode="rgb+segmentation",
                control_mode=control_mode,
                robot_uids=robot_uids,
                sim_backend=sim_backend,
                max_episode_steps=int(max_episode_steps),
                sensor_configs=self._sensor_configs(agentview),
            )
        self._seed = int(seed)
        self._settle_steps = int(settle_steps)
        self._wrist_rotation = int(wrist_rotation)
        self._wrist_flip = wrist_flip
        self._view_size = int(view_size)
        self._obs: dict = {}
        self._closed = False
        self._meta = {
            "env_id": env_id,
            "seed": self._seed,
            "robot_uids": robot_uids,
            "control_mode": control_mode,
            "sim_backend": sim_backend,
            "agentview": agentview,
            "view_size": self._view_size,
            "settle_steps": self._settle_steps,
            "wrist_mount": wrist_mount,
            "wrist_rotation": self._wrist_rotation,
            "wrist_flip": wrist_flip,
            "scene": self._scene,
            "table_z": table_z,
            "action_space": list(self._env.action_space.shape),
        }

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["env.state"] = self.state
        self._rpc["env.servo"] = self.servo

    # ---- helpers ----

    @staticmethod
    def _sensor_configs(agentview: str) -> dict:
        """The agentview pose (``oblique`` = AGENTVIEW, or the scene's ``stock`` camera)
        and a 256 px wrist render."""
        cfg: dict = {"hand_camera": {"width": 256, "height": 256}}
        if agentview == "oblique":
            from mani_skill.utils import sapien_utils

            cfg["base_camera"] = {
                "pose": sapien_utils.look_at(AGENTVIEW["eye"], AGENTVIEW["target"]),
                "fov": np.deg2rad(AGENTVIEW["fov_deg"]),
                "width": AGENTVIEW["width"],
                "height": AGENTVIEW["height"],
            }
        return cfg

    @property
    def _agent(self):
        return self._env.unwrapped.agent

    def _rgb(self, obs: dict, name: str) -> np.ndarray:
        """One view through Show-Harness's transform: orient (wrist only) -> letterbox.

        Their 4:3 wrist crop is not applied: after the 270 deg rotation the fingertips sit
        at the top and bottom of the left edge, and the crop would cut them off."""
        rgb = _np(obs["sensor_data"][self._cameras[name]]["rgb"])[0].astype(np.uint8)
        if self._rig:
            from pi_embodied_services.robots.maniskill import scenes

            v = scenes.VIEWS[name]
            return scenes.prepare_view(
                rgb, v["rotation"], v["flip"], v["crop"], self._view_size
            )
        if name == "wrist":
            rgb = _orient(rgb, self._wrist_rotation, self._wrist_flip)
        return _letterbox(rgb, self._view_size) if self._view_size else rgb

    def _state(self) -> dict:
        tcp = self._agent.tcp.pose
        qpos = _np(self._agent.robot.get_qpos()).reshape(-1)
        return {
            "tcp_pos": _np(tcp.p).reshape(-1).astype(np.float32),
            "tcp_quat_wxyz": _np(tcp.q).reshape(-1).astype(np.float32),
            "gripper_width": float(qpos[-1] + qpos[-2]),
            "qpos": qpos.astype(np.float32),
        }

    def _pack(self, obs: dict) -> dict:
        self._obs = obs
        return {
            "agentview": self._rgb(obs, "agentview"),
            "wrist": self._rgb(obs, "wrist"),
            **self._state(),
        }

    @staticmethod
    def _info(info: dict) -> dict:
        out = {}
        for key, value in info.items():
            if isinstance(value, dict):
                continue
            arr = _np(value).reshape(-1)
            if arr.size == 1:
                out[key] = arr[0].item()
            elif arr.size:
                out[key] = arr
        return out

    def _step(self, action) -> tuple:
        a = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, rew, term, trunc, info = self._env.step(a)
        info = self._info(info)
        if self._rig:
            # BlockStack reports no grasp flag at all (BlockPAP only its lift-based
            # is_cube_grasped): add ManiSkill's contact-based one for the carried object.
            env = self._env.unwrapped
            held = env.agent.is_grasping(getattr(env, self._rig.carried))
            info["is_grasped"] = bool(_np(held).reshape(-1)[0])
        return (
            obs,
            float(_np(rew).reshape(-1)[0]),
            bool(_np(term).reshape(-1)[0]),
            bool(_np(trunc).reshape(-1)[0]),
            info,
        )

    # ---- gym-like surface ----

    def reset(self, seed: int | None = None):
        """Reset to ``seed`` (default: the launch seed), then hold still with the gripper
        open for ``settle_steps`` (Show-Harness ``reset_maniskill``); an RLinf rig resets
        like its training episodes (``scenes.reset``)."""
        hold = np.array([0.0, 0.0, 0.0, OPEN], dtype=np.float32)
        if self._rig:
            from pi_embodied_services.robots.maniskill import scenes

            last: list = []
            self._meta["layout"] = scenes.reset(
                self._env,
                self._meta["env_id"],
                self._scene,
                self._seed if seed is None else int(seed),
                lambda: last.append(self._step(hold)),
            )
            obs, info = last[-1][0], last[-1][4]
            self.check_visible(obs)
            return self._pack(obs), info
        obs, info = self._env.reset(seed=self._seed if seed is None else int(seed))
        info = self._info(info)
        goals = SHOW_GOALS.get(self._meta["env_id"], [])
        if goals:
            show_goals(self._env.unwrapped, goals)
            obs = self._env.unwrapped.get_obs()
        for _ in range(self._settle_steps):
            obs, _r, _te, _tr, info = self._step(hold)
        self.check_visible(obs)
        return self._pack(obs), info

    def visible_pixels(self, obs: dict) -> dict:
        """Agentview pixels of each task actor (per-actor segmentation of base_camera)."""
        seg = _np(obs["sensor_data"][self._cameras["agentview"]]["segmentation"])
        seg = seg[0, ..., 0]
        env = self._env.unwrapped
        out = {}
        actors = [self._rig.carried, self._rig.target] if self._rig else None
        for name in actors or TASK_ACTORS.get(self._meta["env_id"], []):
            ids = _np(getattr(env, name).per_scene_id).reshape(-1)
            out[name] = int(np.isin(seg, ids).sum())
        return out

    def check_visible(self, obs: dict) -> None:
        """Refuse an episode whose agentview does not show every task actor: a hidden
        object makes the planner's failure meaningless (the stock PickCube camera hid the
        cube behind the gripper). Raising fails the reset, so the robot never starts."""
        px = self.visible_pixels(obs)
        self._meta["visible_px"] = px
        hidden = {k: v for k, v in px.items() if v < MIN_VISIBLE_PX}
        if hidden:
            raise RuntimeError(
                f"task objects not visible in the agentview after reset: {hidden} px "
                f"(need >= {MIN_VISIBLE_PX}); refusing the episode"
            )

    def step(self, action):
        obs, rew, term, trunc, info = self._step(action)
        return self._pack(obs), rew, term, trunc, info

    def chunk_step(self, actions, *, return_all_frames: bool = False):
        """Run ``actions`` [N, 4] in one call; stops early on ``stop``, termination,
        truncation or success. Arrays hold one entry per executed action."""
        frames, rews, terms, truncs = [], [], [], []
        info: dict = {}
        for action in np.asarray(actions, dtype=np.float32).reshape(-1, 4):
            if self.stop_requested():
                info["cancelled"] = True
                break
            obs, rew, term, trunc, info = self._step(action)
            frames.append(obs)
            rews.append(rew)
            terms.append(term)
            truncs.append(trunc)
            if term or trunc or info.get("success"):
                break
        if not frames:
            obs = self._pack(self._obs)
            return ([obs] if return_all_frames else obs), [], [], [], info
        packed = [self._pack(o) for o in frames] if return_all_frames else None
        last = packed[-1] if packed else self._pack(frames[-1])
        return (
            packed if return_all_frames else last,
            np.asarray(rews, dtype=np.float32),
            np.asarray(terms, dtype=bool),
            np.asarray(truncs, dtype=bool),
            info,
        )

    def servo(
        self,
        target_xyz,
        gripper: float,
        *,
        gain: float = 1.3,
        tol_m: float = 0.002,
        min_steps: int = 2,
        max_steps: int = 8,
    ):
        """Drive the TCP to ``target_xyz`` (world, m) with the gripper command held: each
        control step commands ``clip(error * gain / 0.1)``, until the error is below
        ``tol_m`` (after ``min_steps``), ``max_steps``, success or ``stop``. The closed-loop
        2 cm execution of Show-Harness's real2sim tokenizer; open-loop steps fall short when
        the arm reverses (PD lag). Returns ``[frames, info]`` with one frame per step."""
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        frames: list = []
        info: dict = {}
        for k in range(int(max_steps)):
            if self.stop_requested():
                info["cancelled"] = True
                break
            err = target - self._state()["tcp_pos"]
            if k >= int(min_steps) and np.linalg.norm(err) < tol_m:
                break
            a = np.append(np.clip(err * gain / DELTA_BOUND_M, -1, 1), gripper)
            obs, _rew, term, trunc, info = self._step(a)
            frames.append(self._pack(obs))
            if term or trunc or info.get("success"):
                break
        if not frames:
            frames.append(self._pack(self._obs))
        return frames, info

    def state(self) -> dict:
        """TCP pose, gripper opening and the current success flags (no stepping)."""
        info = self._info(self._env.unwrapped.evaluate())
        return {**self._state(), "info": info}

    def render_camera(self, camera_name: str = "agentview", **_: Any):
        """Latest frame of ``agentview`` or ``wrist``, as the model sees it."""
        return self._rgb(self._obs, camera_name)

    def get_camera_meta(self, camera_name: str = "agentview", **_: Any) -> dict:
        """OpenCV intrinsics and camera-to-world extrinsic of a sensor camera (raw sensor
        pixels, before the orientation and letterbox of ``render_camera``)."""
        param = self._obs["sensor_param"][self._cameras[camera_name]]
        w2c = np.eye(4)
        w2c[:3] = _np(param["extrinsic_cv"])[0]
        return {
            "intrinsic_K": _np(param["intrinsic_cv"])[0],
            "extrinsic_cam2world": np.linalg.inv(w2c),
        }

    def get_task_language(self) -> str:
        if self._rig:
            return self._rig.instruction
        return INSTRUCTIONS.get(self._meta["env_id"], self._meta["env_id"])

    def get_env_meta(self) -> dict:
        return dict(self._meta)

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
    p.add_argument("--env-id", default="BlockPAP-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--robot-uids", default="panda_wristcam")
    p.add_argument(
        "--wrist-mount", choices=["centered", "camera_link"], default="centered"
    )
    p.add_argument("--sim-backend", default="physx_cpu")
    p.add_argument("--agentview", choices=["oblique", "stock"], default="oblique")
    p.add_argument(
        "--view-size", type=int, default=256, help="letterbox square, px (0 = raw)"
    )
    p.add_argument("--settle-steps", type=int, default=8)
    # Measured on the stock scenes (PickCube seed 0, cube projected through the wrist
    # calibration while stepping MV_LEFT / MV_FWD): the centred camera renders image right =
    # -x, image down = +y, i.e. 90 deg off Show-Harness's RLinf rig, whose calibrated start
    # pose rolls the hand; their `flip: both` would put MV_FWD at the image left here.
    # Rotating 270 (CCW, no flip) gives the agentview's convention: image right = +y
    # (MV_RIGHT), image bottom = +x (MV_FWD); the fingertips sit at the left edge and the
    # point under the TCP at mid-height, 31-41 % of the width from the left.
    p.add_argument("--wrist-rotation", type=int, choices=[0, 90, 180, 270], default=270)
    p.add_argument(
        "--wrist-flip",
        choices=["none", "vertical", "horizontal", "both"],
        default="none",
    )
    p.add_argument(
        "--scene",
        default="",
        help="RLinf rig options as key=value,... (scenes.py: table_tex, cam_t, traj_id, "
        "layout, cam_jitter, wrist_mount, wrist_resolution); the rig ignores the camera flags",
    )
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    args = p.parse_args()

    facade = ManiskillEnvFacade(
        env_id=args.env_id,
        seed=args.seed,
        robot_uids=args.robot_uids,
        sim_backend=args.sim_backend,
        agentview=args.agentview,
        view_size=args.view_size,
        settle_steps=args.settle_steps,
        wrist_mount=args.wrist_mount,
        wrist_rotation=args.wrist_rotation,
        wrist_flip=args.wrist_flip,
        scene=dict(kv.split("=", 1) for kv in args.scene.split(",") if kv) or None,
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
