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
# Modified by pi-embodied: adapted from Show-Harness core/sim/robolab_franka.py (@137d571);
# the robot USD path comes from ROBOLAB_PANDA_USD, which the env server sets to a copy of
# assets/panda_short_finger.usda with its references resolved (sim.localize_panda_usd).

"""Franka Panda (panda_hand) embodiment for RoboLab, matching the ManiSkill/real rigs.

RoboLab ships a Franka + Robotiq 2F-85 (``DroidCfg``) with DROID's off-centre wrist camera.
This module declares, in RoboLab's own config vocabulary (the "bring your own robot" path),
a Franka + Panda hand wearing the real rig's short yellow fingertips, a wrist camera centred
between the fingers looking down the grasp axis, and RLinf's calibrated front camera (the
ManiSkill agentview). It imports isaaclab at module scope, so import it only after
``sim.launch_isaac`` has started the Kit app.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass, noise
from robolab.robots.droid import BinaryJointPositionZeroToOneActionCfg, _to_torch

from pi_embodied_services.robots.robolab.sim import lab_quat

# -- wrist camera: ManiSkill's "centered" mount (3.5 cm along the hand's +X, 3.6 cm along +Z,
# looking down the grasp axis), 256x256, 90 deg FOV. Raw frames show the fingertips entering
# from the LEFT edge; the env server rotates them 270 deg CCW so they sit at the top.
WRIST_CAM_POS = (0.035, 0.0, 0.036)
WRIST_CAM_RESOLUTION = 256
WRIST_CAM_APERTURE = 20.955
WRIST_CAM_FOCAL = WRIST_CAM_APERTURE / 2.0

_WRIST_CAM = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/panda_hand/wrist_cam",
    height=WRIST_CAM_RESOLUTION,
    width=WRIST_CAM_RESOLUTION,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=WRIST_CAM_FOCAL,
        focus_distance=0.4,
        horizontal_aperture=WRIST_CAM_APERTURE,
        vertical_aperture=WRIST_CAM_APERTURE,
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=WRIST_CAM_POS, rot=lab_quat((1.0, 0.0, 0.0, 0.0)), convention="ros"
    ),
)

# -- front camera: RLinf's calibrated RealSense D435 (the ManiSkill BlockPAP agentview), in the
# robot base frame, OpenCV convention; 640x480.
FRONT_CAM_R = (
    (0.02816316, 0.21788680, -0.97556762),
    (0.99959024, -0.00114196, 0.02860160),
    (0.00511786, -0.97597338, -0.21782968),
)
FRONT_CAM_POS = (1.1002696, -0.00701879, 0.2589829)
FRONT_CAM_K = (607.875, 0.0, 348.961, 0.0, 607.719, 270.486, 0.0, 0.0, 1.0)
FRONT_CAM_W, FRONT_CAM_H = 640, 480


def _quat_wxyz_from_matrix(matrix) -> tuple[float, float, float, float]:
    m = np.asarray(matrix, dtype=float)
    trace = m.trace()
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = (
            0.25 / s,
            (m[2, 1] - m[1, 2]) * s,
            (m[0, 2] - m[2, 0]) * s,
            (m[1, 0] - m[0, 1]) * s,
        )
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + m[i, i] - m[j, j] - m[k, k]))
        out = [0.0, 0.0, 0.0, 0.0]
        out[0] = (m[k, j] - m[j, k]) / s
        out[i + 1] = 0.25 * s
        out[j + 1] = (m[j, i] + m[i, j]) / s
        out[k + 1] = (m[k, i] + m[i, k]) / s
        q = tuple(out)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


FRONT_CAM_QUAT = lab_quat(_quat_wxyz_from_matrix(FRONT_CAM_R))

_FRONT_CAM = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/front_cam",
    height=FRONT_CAM_H,
    width=FRONT_CAM_W,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=list(FRONT_CAM_K),
        width=FRONT_CAM_W,
        height=FRONT_CAM_H,
        clipping_range=(0.01, 10.0),
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=FRONT_CAM_POS, rot=FRONT_CAM_QUAT, convention="ros"
    ),
)


@configclass
class FrankaFrontCameraCfg:
    """ManiSkill-identical front view. A scene camera (world-fixed), not robot-mounted."""

    front_cam = _FRONT_CAM


PANDA_USD = os.environ["ROBOLAB_PANDA_USD"]
PANDA_FINGER_OPEN_M = 0.04
PANDA_FINGER_CLOSED_M = 0.0
PANDA_MAX_WIDTH_M = 2 * PANDA_FINGER_OPEN_M

# RoboLab DroidCfg's arm pose (the same TCP reset height), wrist roll at +45 deg so the
# fingers line up with the base axes.
FRANKA_HOME_QPOS: dict[str, float] = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.79613,
    "panda_joint3": 0.0,
    "panda_joint4": -2.75597,
    "panda_joint5": 0.0,
    "panda_joint6": 1.95980,
    "panda_joint7": 0.785,
    "panda_finger_joint.*": PANDA_FINGER_OPEN_M,
}

# RoboLab's grasp predicates (require_gripper_detached, ...) read contacts off this body.
contact_gripper = {"gripper": "{ENV_REGEX_NS}/robot/panda_leftfinger"}


@configclass
class FrankaPandaCfg:
    """Franka + Panda hand, with the centre-mounted wrist camera attached."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PANDA_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=5.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=lab_quat((1.0, 0.0, 0.0, 0.0)),
            joint_pos=dict(FRANKA_HOME_QPOS),
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    wrist_cam = _WRIST_CAM


# RoboLab >= 0.3.1 requires the EE body its recorder logs.
FrankaPandaCfg.ee_recorder_bodies = {"ee_pose": "panda_hand"}


@configclass
class FrankaWristCameraCfg:
    """Exposes the robot-mounted wrist camera's name to the image-obs generator."""

    wrist_cam = _WRIST_CAM


@configclass
class FrankaRelIKActionCfg:
    """Relative EE-pose control, 7 dims ``(dx, dy, dz, drx, dry, drz, gripper)`` on panda_hand."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls"
        ),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.0]
        ),
    )
    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": PANDA_FINGER_OPEN_M},
        close_command_expr={"panda_finger_joint.*": PANDA_FINGER_CLOSED_M},
    )


def arm_joint_pos(env, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    idx = [
        i for i, n in enumerate(robot.data.joint_names) if n.startswith("panda_joint")
    ]
    return _to_torch(robot.data.joint_pos)[:, idx]


def gripper_pos(env, asset_cfg=SceneEntityCfg("robot")):
    """0 = open, 1 = closed (RoboLab's droid.py normalisation)."""
    robot = env.scene[asset_cfg.name]
    idx = [
        i
        for i, n in enumerate(robot.data.joint_names)
        if n.startswith("panda_finger_joint")
    ]
    width = _to_torch(robot.data.joint_pos)[:, idx].sum(dim=-1, keepdim=True)
    return 1.0 - width / PANDA_MAX_WIDTH_M


def ee_pos(env, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    i = robot.data.body_names.index("panda_hand")
    return _to_torch(robot.data.body_pos_w)[:, i, :] - env.scene.env_origins[:, 0:3]


def ee_quat(env, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    i = robot.data.body_names.index("panda_hand")
    return _to_torch(robot.data.body_quat_w)[:, i, :]


@configclass
class FrankaProprioCfg(ObsGroup):
    arm_joint_pos = ObsTerm(func=arm_joint_pos)
    gripper_pos = ObsTerm(
        func=gripper_pos, noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1)
    )
    ee_pos = ObsTerm(func=ee_pos)
    ee_quat = ObsTerm(func=ee_quat)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False
