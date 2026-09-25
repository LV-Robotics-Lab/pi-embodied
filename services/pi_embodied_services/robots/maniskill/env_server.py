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
# core/sim/maniskill_scenes.py (stock scenes only; the RLinf real2sim rigs are not
# public) into an RPC env server; the atomic-token controller lives in the pi robot.

"""RPC server wrapping one ManiSkill 3 env in ``pd_ee_delta_pos``.

Action ``[dx, dy, dz, gripper]`` in [-1, 1]: a base-frame position delta normalised by
the arm's 0.1 m bound, and the Panda mimic gripper (+1 open, -1 close). Observations
carry the agentview (``base_camera``) and wrist (``hand_camera``) RGB, the TCP pose and
the gripper opening; ``info`` is flattened to plain scalars (``success``,
``is_grasped``, ...).
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
    "PickCube-v1": "pick up the red cube",
    "StackCube-v1": "stack the red cube on top of the green cube",
    "PushCube-v1": "push the cube to the goal marker",
    "PullCube-v1": "pull the cube to the goal marker",
    "PokeCube-v1": "poke the cube to the goal marker",
    "LiftPegUpright-v1": "lift the peg upright",
}
CAMERAS = {"agentview": "base_camera", "wrist": "hand_camera"}
OPEN = 1.0
#: Stock Panda arm_pd_ee_delta_pos position bound: action 1.0 = 0.1 m.
DELTA_BOUND_M = 0.1


def _np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


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
#: pair is centred; with ``wrist_flip: both`` the fingertips are at the top and wrist left ==
#: agentview left (their training contract).
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
        camera_resolution: int = 512,
        max_episode_steps: int = 100_000,
        settle_steps: int = 8,
        wrist_rotation: int = 0,
        wrist_flip: str = "both",
    ):
        super().__init__()
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 -- registers the stock env ids

        if wrist_mount == "centered":
            center_wrist_camera()

        self._env = gym.make(
            env_id,
            num_envs=1,
            obs_mode="rgb",
            control_mode=control_mode,
            robot_uids=robot_uids,
            sim_backend=sim_backend,
            max_episode_steps=int(max_episode_steps),
            sensor_configs=dict(
                width=int(camera_resolution), height=int(camera_resolution)
            ),
        )
        self._seed = int(seed)
        self._settle_steps = int(settle_steps)
        self._wrist_rotation = int(wrist_rotation)
        self._wrist_flip = wrist_flip
        self._obs: dict = {}
        self._closed = False
        self._meta = {
            "env_id": env_id,
            "seed": self._seed,
            "robot_uids": robot_uids,
            "control_mode": control_mode,
            "sim_backend": sim_backend,
            "camera_resolution": int(camera_resolution),
            "settle_steps": self._settle_steps,
            "wrist_mount": wrist_mount,
            "wrist_rotation": self._wrist_rotation,
            "wrist_flip": wrist_flip,
            "action_space": list(self._env.action_space.shape),
        }

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["env.state"] = self.state
        self._rpc["env.servo"] = self.servo

    # ---- helpers ----

    @property
    def _agent(self):
        return self._env.unwrapped.agent

    def _rgb(self, obs: dict, name: str) -> np.ndarray:
        rgb = _np(obs["sensor_data"][CAMERAS[name]]["rgb"])[0].astype(np.uint8)
        return (
            _orient(rgb, self._wrist_rotation, self._wrist_flip)
            if name == "wrist"
            else np.ascontiguousarray(rgb)
        )

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
        return (
            obs,
            float(_np(rew).reshape(-1)[0]),
            bool(_np(term).reshape(-1)[0]),
            bool(_np(trunc).reshape(-1)[0]),
            self._info(info),
        )

    # ---- gym-like surface ----

    def reset(self, seed: int | None = None):
        """Reset to ``seed`` (default: the launch seed), then hold still with the gripper
        open for ``settle_steps`` (Show-Harness ``reset_maniskill``)."""
        obs, info = self._env.reset(seed=self._seed if seed is None else int(seed))
        info = self._info(info)
        hold = np.array([0.0, 0.0, 0.0, OPEN], dtype=np.float32)
        for _ in range(self._settle_steps):
            obs, _r, _te, _tr, info = self._step(hold)
        return self._pack(obs), info

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
        """Latest frame of ``agentview`` or ``wrist`` (sensor resolution)."""
        return self._rgb(self._obs, camera_name)

    def get_camera_meta(self, camera_name: str = "agentview", **_: Any) -> dict:
        """OpenCV intrinsics and camera-to-world extrinsic of a sensor camera."""
        param = self._obs["sensor_param"][CAMERAS[camera_name]]
        w2c = np.eye(4)
        w2c[:3] = _np(param["extrinsic_cv"])[0]
        return {
            "intrinsic_K": _np(param["intrinsic_cv"])[0],
            "extrinsic_cam2world": np.linalg.inv(w2c),
        }

    def get_task_language(self) -> str:
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
    p.add_argument("--env-id", default="PickCube-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--robot-uids", default="panda_wristcam")
    p.add_argument(
        "--wrist-mount", choices=["centered", "camera_link"], default="centered"
    )
    p.add_argument("--sim-backend", default="physx_cpu")
    p.add_argument("--camera-resolution", type=int, default=512)
    p.add_argument("--settle-steps", type=int, default=8)
    # Show-Harness configs/robot_maniskill.yaml: wrist_rotation_degrees 0, wrist_flip both
    # (for the centred mount). The stock panda_wristcam (camera_link) needs rotation 270,
    # flip none for the same image directions, with the fingers at the left edge instead.
    p.add_argument("--wrist-rotation", type=int, choices=[0, 90, 180, 270], default=0)
    p.add_argument(
        "--wrist-flip",
        choices=["none", "vertical", "horizontal", "both"],
        default="both",
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
        camera_resolution=args.camera_resolution,
        settle_steps=args.settle_steps,
        wrist_mount=args.wrist_mount,
        wrist_rotation=args.wrist_rotation,
        wrist_flip=args.wrist_flip,
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
