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

"""OmniGibson glue for the BEHAVIOR env server: the task config, the launch, and reads of the
R1Pro, its cameras and the BDDL task that need no simulator import (duck-typed on OmniGibson's
objects, so the env server is testable against a mock).

Conventions: OmniGibson stores poses as torch tensors with xyzw quaternions in the world frame;
its ``VisionSensor`` pose follows the OpenGL camera (looks along -Z, +Y up), so a depth pixel
back-projects as ``x = (u - cx) d / fx, y = -(v - cy) d / fy, z = -d`` before the cam-to-world
transform. ``depth_linear`` is metric depth.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

#: The R1Pro's camera links: ZED on the head, a RealSense on each wrist (OmniGibson sensor
#: names are ``<robot>:<link>:Camera:0``).
CAMERAS: dict[str, str] = {
    "head": "zed_link",
    "left_wrist": "left_realsense_link",
    "right_wrist": "right_realsense_link",
}
ARMS = ("left", "right")
#: Metric depth from OmniGibson's VisionSensor.
DEPTH_MODALITY = "depth_linear"
#: The challenge's demonstrations metadata, under ``gm.DATA_PATH``.
CHALLENGE_DIR = "2025-challenge-task-instances"


def to_np(x: Any) -> np.ndarray:
    """A torch tensor (any device) or array-like as a numpy array."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    return np.asarray(x)


def tensor(a) -> Any:
    """A float32 torch tensor of ``a`` for OmniGibson's primitives (a numpy array when torch is
    not installed, as in the tests)."""
    try:
        import torch
    except ImportError:
        return np.asarray(a, dtype=np.float32)
    return torch.as_tensor(np.asarray(a, dtype=np.float32))


def quat_yaw(q_xyzw) -> float:
    """Yaw (rad) of an xyzw quaternion about +z."""
    x, y, z, w = (float(v) for v in np.asarray(q_xyzw, dtype=np.float64).reshape(4))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_mat(q_xyzw) -> np.ndarray:
    """Rotation matrix of an xyzw quaternion."""
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# ---------------------------------------------------------------------------
# the task and its scene


def find_task_scene(
    data_path: str | os.PathLike, activity: str, definition_id: int = 0
) -> tuple[str, list[int]]:
    """The scene model that holds ``activity``'s pre-sampled instances in OmniGibson's dataset,
    and the instance ids it has, found by the instance files' names
    (``og_dataset/scenes/<scene>/json/<scene>_task_<activity>_instances/
    <scene>_task_<activity>_<definition>_<instance>_template.json``). The challenge samples every
    task in one scene; several scenes are an error, none means the dataset is missing."""
    root = Path(data_path) / "og_dataset" / "scenes"
    pattern = re.compile(
        rf"^(?P<scene>.+)_task_{re.escape(activity)}_{definition_id}_(?P<inst>\d+)_template\.json$"
    )
    found: dict[str, set[int]] = {}
    for scene_dir in sorted(root.glob("*")):
        inst_dir = scene_dir / "json" / f"{scene_dir.name}_task_{activity}_instances"
        if not inst_dir.is_dir():
            continue
        for f in inst_dir.iterdir():
            m = pattern.match(f.name)
            if m and m.group("scene") == scene_dir.name:
                found.setdefault(scene_dir.name, set()).add(int(m.group("inst")))
    if not found:
        raise FileNotFoundError(
            f"no pre-sampled instances of {activity!r} (definition {definition_id}) under {root}; "
            "download the BEHAVIOR-1K dataset (install.sh) or set OMNIGIBSON_DATA_PATH"
        )
    if len(found) > 1:
        raise ValueError(f"{activity!r} is sampled in several scenes: {sorted(found)}")
    ((scene, ids),) = found.items()
    return scene, sorted(ids)


def task_config(
    *,
    activity: str,
    scene_model: str,
    instance_id: int,
    image_size: int,
    grasping_mode: str,
    max_steps: int,
    definition_id: int = 0,
) -> dict:
    """OmniGibson's env config for one challenge task on the R1Pro: the ``r1pro_primitives.yaml``
    robot (position-controlled joints, the holonomic base as joints; what
    StarterSemanticActionPrimitives drives) with RGB and metric depth on the three cameras, in a
    ``BehaviorTask`` on its pre-sampled instance with the presampled robot pose. ``max_steps`` is
    the task's truncation (primitives spend hundreds of steps each)."""
    if grasping_mode not in ("sticky", "assisted"):
        raise ValueError(
            f"grasping_mode must be sticky or assisted, got {grasping_mode!r}"
        )
    joints = {
        "name": "JointController",
        "motor_type": "position",
        "command_input_limits": None,
        "use_delta_commands": False,
        "use_impedances": False,
    }
    return {
        "env": {
            "action_frequency": 30,
            "physics_frequency": 120,
            "device": None,
            "automatic_reset": False,
            "flatten_action_space": False,
            "flatten_obs_space": False,
            "use_external_obs": False,
            "external_sensors": None,
        },
        "render": {"viewer_width": 1280, "viewer_height": 720},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
            "trav_map_resolution": 0.1,
            "default_erosion_radius": 0.0,
            "trav_map_with_objects": True,
            "num_waypoints": 1,
            "waypoint_resolution": 0.2,
            "load_object_categories": None,
            "not_load_object_categories": None,
            "load_room_types": None,
            "load_room_instances": None,
            "load_task_relevant_only": False,
            "seg_map_resolution": 1.0,
            "scene_source": "OG",
            "include_robots": False,
        },
        "robots": [
            {
                "type": "R1Pro",
                "obs_modalities": ["rgb", DEPTH_MODALITY, "proprio"],
                "include_sensor_names": None,
                "exclude_sensor_names": None,
                "scale": 1.0,
                "self_collisions": True,
                "action_normalize": False,
                "action_type": "continuous",
                "grasping_mode": grasping_mode,
                "sensor_config": {
                    "VisionSensor": {
                        "sensor_kwargs": {
                            "image_height": image_size,
                            "image_width": image_size,
                        }
                    }
                },
                "controller_config": {
                    "base": {
                        "name": "HolonomicBaseJointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "use_impedances": False,
                    },
                    "trunk": dict(joints),
                    "arm_left": dict(joints),
                    "arm_right": dict(joints),
                    "gripper_left": dict(joints),
                    "gripper_right": dict(joints),
                },
            }
        ],
        "objects": [],
        "task": {
            "type": "BehaviorTask",
            "activity_name": activity,
            "activity_definition_id": definition_id,
            "activity_instance_id": instance_id,
            "predefined_problem": None,
            "online_object_sampling": False,
            "debug_object_sampling": None,
            "highlight_task_relevant_objects": False,
            "use_presampled_robot_pose": True,
            "termination_config": {"max_steps": max_steps},
            "reward_config": {"r_potential": 1.0},
        },
    }


class Handle:
    """A launched task: OmniGibson's env, its robot, the semantic primitives and the task."""

    def __init__(self, *, og: Any, env: Any, controller: Any, error: type):
        self.og = og
        self.env = env
        self.robot = env.robots[0]
        self.task = env.task
        self.controller = controller
        #: OmniGibson's ActionPrimitiveError: a primitive that refuses or fails (not a crash).
        self.error = error


def launch(config: dict) -> Handle:
    """Start OmniGibson (Isaac Sim) headless on ``OMNIGIBSON_GPU_ID`` and load the task.

    Object states and transition rules stay on (BDDL predicates need them); GPU dynamics off
    (the primitives' cuRobo planner owns the GPU). Imports OmniGibson here: its macros read the
    environment at import, so the caller sets ``OMNIGIBSON_*`` first.
    """
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    import omnigibson as og
    from omnigibson.action_primitives.action_primitive_set_base import (
        ActionPrimitiveError,
    )
    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitives,
    )
    from omnigibson.macros import gm

    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = True
    gm.USE_GPU_DYNAMICS = False
    env = og.Environment(configs=config)
    controller = StarterSemanticActionPrimitives(
        env, env.robots[0], enable_head_tracking=False
    )
    return Handle(og=og, env=env, controller=controller, error=ActionPrimitiveError)


# ---------------------------------------------------------------------------
# reads of the robot, the cameras and the task


def sensor(robot: Any, camera: str) -> Any:
    """The robot's VisionSensor behind a CAMERAS key."""
    if camera not in CAMERAS:
        raise ValueError(f"camera must be one of {sorted(CAMERAS)}, got {camera!r}")
    name = f"{robot.name}:{CAMERAS[camera]}:Camera:0"
    try:
        return robot.sensors[name]
    except KeyError:
        raise KeyError(f"{name} not among {sorted(robot.sensors)}") from None


def camera_meta(robot: Any, camera: str) -> dict:
    """OpenCV-style intrinsics and the OpenGL camera-to-world transform of a camera."""
    s = sensor(robot, camera)
    pos, quat = s.get_position_orientation()
    cam2world = np.eye(4)
    cam2world[:3, :3] = quat_mat(to_np(quat))
    cam2world[:3, 3] = to_np(pos).astype(np.float64)
    return {
        "camera": camera,
        "intrinsic_K": to_np(s.intrinsic_matrix).astype(np.float64).reshape(3, 3),
        "extrinsic_cam2world": cam2world,
        "convention": "opengl",
        "width": int(s.image_width),
        "height": int(s.image_height),
    }


def images(obs: dict, robot: Any) -> dict[str, np.ndarray]:
    """``<camera>`` uint8[H,W,3] and ``<camera>_depth`` float32[H,W] (m) of the three cameras."""
    out: dict[str, np.ndarray] = {}
    frames = obs[robot.name]
    for camera, link in CAMERAS.items():
        cam = frames[f"{robot.name}:{link}:Camera:0"]
        out[camera] = np.ascontiguousarray(to_np(cam["rgb"])[..., :3]).astype(
            np.uint8, copy=False
        )
        out[f"{camera}_depth"] = np.ascontiguousarray(
            to_np(cam[DEPTH_MODALITY]).astype(np.float32)
        )
    return out


def base_pose(robot: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """World position, xyzw orientation and yaw of the robot base."""
    pos, quat = robot.get_position_orientation()
    pos, quat = to_np(pos).astype(np.float64), to_np(quat).astype(np.float64)
    return pos, quat, quat_yaw(quat)


def eef_pose(robot: Any, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """World position and xyzw orientation of an arm's end effector."""
    pos, quat = robot.get_eef_pose(arm=arm)
    return to_np(pos).astype(np.float64), to_np(quat).astype(np.float64)


def gripper_width(robot: Any, arm: str) -> float:
    """Finger opening of an arm's gripper, m (the sum of its finger joint positions)."""
    q = to_np(robot.get_joint_positions())
    idx = to_np(robot.gripper_control_idx[arm]).astype(int)
    return float(np.sum(q[idx]))


def in_hand(robot: Any, arm: str) -> str | None:
    """Name of the object OmniGibson's grasping holds in ``arm``, or None (simulator state)."""
    obj = robot._ag_obj_in_hand[arm]
    return None if obj is None else str(obj.name)


def success(info: dict | None, task: Any) -> bool:
    """BDDL success: the task's ``success`` termination in the last step's info, else the task's
    latched ``success`` (a per-env tensor on OmniGibson 3.9, a bool on 3.7)."""
    done = (info or {}).get("done")
    if isinstance(done, dict) and "success" in done:
        return bool(done["success"])
    s = getattr(task, "success", None)
    if s is None:
        return False
    s = to_np(s).reshape(-1)
    return bool(s[0]) if s.size else False


def goal_satisfaction(task: Any) -> list[list[bool]]:
    """Per goal option, per predicate: satisfied now. OmniGibson 3.9's
    ``get_goal_option_satisfaction``; on older versions the predicate termination's
    ``goal_status`` (one option)."""
    fn = getattr(task, "get_goal_option_satisfaction", None)
    if fn is not None:
        return [[bool(v) for v in option] for option in fn(0)]
    status = task._termination_conditions["predicate"].goal_status
    if isinstance(status, (list, tuple)):
        status = status[0]
    satisfied, unsatisfied = set(status["satisfied"]), set(status["unsatisfied"])
    return [[i in satisfied for i in sorted(satisfied | unsatisfied)]]


def q_score(solved: bool, now: list[list[bool]], initial: list[list[bool]]) -> float:
    """BEHAVIOR's partial-success Q-score (``omnigibson.metrics.task_metric.compute_q_score``):
    1 on success, else the fraction of goal predicates unsatisfied at reset and satisfied now,
    maximized over the goal options."""
    if solved:
        return 1.0
    scores = []
    for now_opt, init_opt in zip(now, initial):
        if not now_opt:
            scores.append(0.0)
            continue
        newly = sum(int(n and not i) for n, i in zip(now_opt, init_opt))
        scores.append(newly / len(now_opt))
    return max(scores) if scores else 0.0


def object_poses(task: Any) -> dict[str, dict]:
    """World poses of the task's BDDL objects (``env.ground_truth_poses``): the instances of
    ``task.object_scope`` that exist in the scene (systems and unsampled ones have no pose)."""
    from pi_embodied_services.utils import ground_truth

    poses: dict[str, dict] = {}
    for inst, entity in task.object_scope.items():
        if getattr(entity, "is_system", False) or not getattr(entity, "exists", True):
            continue
        pos, quat = entity.get_position_orientation()
        x, y, z, w = to_np(quat).reshape(4)
        poses[str(inst)] = ground_truth.pose(to_np(pos), [w, x, y, z])
    return poses


def object_heights(task: Any) -> dict[str, float]:
    """World z of every task object that exists (the reference ``picked`` judgement compares
    against these, taken at reset)."""
    return {name: p["pos"][2] for name, p in object_poses(task).items()}


def run(
    env: Any,
    actions: Iterable[Any],
    on_step: Callable[[Any], None],
    stop: Callable[[], bool],
    max_steps: int,
) -> tuple[int, bool]:
    """Step ``env`` through a primitive's action generator: returns (steps, cancelled).
    ``on_step`` gets each ``env.step`` result. Stops early on ``stop()`` (cancelled) or after
    ``max_steps`` (raises RuntimeError: a primitive that never converges)."""
    steps = 0
    for action in actions:
        if stop():
            return steps, True
        if steps >= max_steps:
            raise RuntimeError(f"primitive exceeded {max_steps} control steps")
        on_step(env.step(action))
        steps += 1
    return steps, False
