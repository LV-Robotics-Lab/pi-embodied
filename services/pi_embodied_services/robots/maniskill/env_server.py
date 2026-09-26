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
the arm's 0.1 m bound, and the gripper in pi's convention (> 0 open, < 0 close) whatever
the ``--robot``; ``RobotSpec.action`` turns it into the robot's own action (the Robotiq's
+1 closes). Observations
carry the agentview (``base_camera``; ``external_cam`` on the RLinf rigs) and, on a robot
with a wrist camera, the wrist (``hand_camera``) RGB, the TCP pose and the gripper
opening; ``info`` is flattened to plain scalars (``success``, ``is_grasped``, ...). An env id of ./scenes.py (BlockPAP-v1,
BlockStack-v1) runs that rig with Show-Harness's calibrated cameras, reset and views; the
other ids (``ENV_IDS``) are stock ManiSkill tabletop tasks on the shared oblique camera.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from pi_embodied_services.components.code_api import register_code_api
from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.robots.maniskill.primitives import MANISKILL_PRIMITIVES
from pi_embodied_services.utils import ground_truth, reach
from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.perception import (
    add_perception_arguments,
    install_perception,
    render_view,
)
from pi_embodied_services.utils.rpc.main_thread_serve import MainThreadServeMixin

logger = get_logger("env_server")

#: Show-Harness core/sim/maniskill_scenes.py SCENES (stock ManiSkill rows), then the stock
#: tabletop tasks of OpenETA's ManiSkill table (every registered env minus locomotion,
#: humanoid and dexterous hands) that a translation-only Panda with a gripper can attempt on
#: the shared table scene: the task text.
INSTRUCTIONS = {
    # PickCube's success is the cube inside a goal sphere (goal_thresh 2.5 cm, up to 0.3 m above
    # the table), not the lift alone: the text names it, and SHOW_GOALS renders the sphere.
    "PickCube-v1": "pick up the red cube and move it into the green goal sphere",
    "StackCube-v1": "stack the red cube on top of the green cube",
    "PushCube-v1": "push the cube to the goal marker",
    "PullCube-v1": "pull the cube to the goal marker",
    "PokeCube-v1": "poke the cube to the goal marker",
    "LiftPegUpright-v1": "lift the peg upright",
    # PlaceSphere succeeds with the ball resting on the bin's floor within 5 mm of its centre.
    "PlaceSphere-v1": "pick up the blue ball and place it inside the small bin",
    # StackPyramid: cubes A (red) and B (green) side by side, cube C (blue) resting on both.
    "StackPyramid-v1": "put the red and green cubes side by side, touching, then stack the blue cube on top so it rests on both",
    # PullCubeTool: the cube starts out of reach; success is the cube within 0.6 m of the base.
    "PullCubeTool-v1": "pick up the L-shaped tool and use its hook to pull the blue cube toward the robot, within reach",
    # PegInsertionSide: the peg's head into the hole on the side of the box.
    "PegInsertionSide-v1": "pick up the peg and insert its head sideways into the hole of the box",
    # PlugCharger: position within 5 mm and orientation within 0.2 rad of the receptacle's slots.
    "PlugCharger-v1": "pick up the charger and plug its two prongs into the receptacle",
    # PickSingleYCB samples one YCB object per episode (ycb assets); the goal sphere is SHOW_GOALS.
    "PickSingleYCB-v1": "pick up the object on the table and move it into the green goal sphere",
    # FMBAssembly1Easy: success is the bridge's centre within 5 mm of its slot on the board
    # (position only); it starts lying on its side next to the reorienting fixture.
    "FMBAssembly1Easy-v1": "pick up the green bridge piece and assemble it onto the board so it straddles the yellow peg between the purple and blue pieces; the grey fixture can help stand it upright",
    # PickCubeWidowXAI registers PickCube with the WidowX AI fixed (--robot widowxai only).
    "PickCubeWidowXAI-v1": "pick up the red cube and move it into the green goal sphere",
    # The panda_stick tasks (--robot panda_stick: a stick instead of a gripper). PushT
    # succeeds when the T covers >= 90 % of the goal outline (position and turn).
    "PushT-v1": "push the red T-shaped block with the stick so it lies exactly on the grey T-shaped outline",
    # Drawing: the stick paints a red dot at every control step its tip is within ~8 mm
    # of the canvas; success is a dot within 2.5 cm (triangle) / 10 cm (svg) of every
    # point of the outline.
    "DrawTriangle-v1": "draw the outlined triangle on the white canvas: trace all three edges with the stick's tip touching the canvas",
    "DrawSVG-v1": "draw the outlined shape on the white canvas: trace its whole outline with the stick's tip touching the canvas",
    # The two-robot scenes (--robot panda_pair). TwoRobotPickCube: the cube starts on the
    # left arm's side, the goal on the right's; success is the cube inside the goal
    # sphere (2.5 cm) with the right arm still.
    "TwoRobotPickCube-v1": "pick up the red cube with the left arm, hand it over to the right arm, and hold it still with the right arm inside the green goal sphere",
    # TwoRobotStackCube: the green cube inside the target, the blue cube on it, neither held.
    "TwoRobotStackCube-v1": "with the right arm, place the green cube on the red-and-white target; with the left arm, stack the blue cube on top of the green cube; then release both",
    # The BridgeData V2 digital twins (--robot widowx250s, SIMPLER's scenes): their own
    # WidowX 250 S, camera and real-image background; the texts are the scenes' own
    # instructions. Success is the source object resting on the target.
    "PutCarrotOnPlateInScene-v1": "put carrot on plate",
    "PutEggplantInBasketScene-v1": "put eggplant into yellow basket",
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1": "stack the green block on the yellow block",
    "PutSpoonOnTableClothInScene-v1": "put the spoon on the towel",
}
#: --env-id accepts these: the RLinf rigs (./scenes.py) and the stock tasks above.
ENV_IDS = ["BlockPAP-v1", "BlockStack-v1", *INSTRUCTIONS]
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
    "PlaceSphere-v1": ["obj", "bin"],
    "StackPyramid-v1": ["cubeA", "cubeB", "cubeC"],
    "PullCubeTool-v1": ["cube", "l_shape_tool"],
    "PegInsertionSide-v1": ["peg", "box"],
    "PlugCharger-v1": ["charger", "receptacle"],
    "PickSingleYCB-v1": ["obj", "goal_site"],
    "FMBAssembly1Easy-v1": ["bridge", "board", "peg"],
    "PickCubeWidowXAI-v1": ["cube", "goal_site"],
    "PushT-v1": ["tee", "goal_tee"],
    "DrawTriangle-v1": ["canvas", "goal_tri"],
    "DrawSVG-v1": ["canvas", "goal_outline"],
    "TwoRobotPickCube-v1": ["cube", "goal_site"],
    "TwoRobotStackCube-v1": ["cubeA", "cubeB", "goal_region"],
    # The bridge scenes keep their objects in ``objs`` (by model id): ``objs/<id>``. The
    # eggplant's target is an invisible plane in the sink's basket: the sink stands in.
    "PutCarrotOnPlateInScene-v1": [
        "objs/bridge_carrot_generated_modified",
        "objs/bridge_plate_objaverse_larger",
    ],
    "PutEggplantInBasketScene-v1": ["objs/eggplant", "sink"],
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1": [
        "objs/baked_green_cube_3cm",
        "objs/baked_yellow_cube_3cm",
    ],
    "PutSpoonOnTableClothInScene-v1": [
        "objs/bridge_spoon_generated_modified",
        "objs/table_cloth_generated_shorter",
    ],
}
#: Success markers a task keeps in ``_hidden_objects`` (drawn for the human viewer only, never
#: in the sensor cameras) although its success depends on them: shown to the cameras at every
#: reset, so the model can see the goal (and ``check_visible`` can require it).
SHOW_GOALS = {
    "PickCube-v1": ["goal_site"],
    "PickSingleYCB-v1": ["goal_site"],
    "PickCubeWidowXAI-v1": ["goal_site"],
    "TwoRobotPickCube-v1": ["goal_site"],
}
#: The bridge digital twins (their robot, camera and control mode are the scene's).
BRIDGE_ENVS = (
    "PutCarrotOnPlateInScene-v1",
    "PutEggplantInBasketScene-v1",
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
    "PutSpoonOnTableClothInScene-v1",
)
#: Env ids whose registration fixes the robot (``gym.make`` must not pass ``robot_uids``).
FIXED_ROBOT_ENVS = ("PickCubeWidowXAI-v1", *BRIDGE_ENVS)
#: The table top of a scene not at z = 0 (meta ``table_z``): the bridge twins' table
#: (measured: the plate rests at 0.870), and the sink's floor under the eggplant.
TABLE_Z = {
    "PutCarrotOnPlateInScene-v1": 0.87,
    "PutEggplantInBasketScene-v1": 0.91,
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1": 0.87,
    "PutSpoonOnTableClothInScene-v1": 0.87,
}
#: Fewest agentview pixels (640x480 sensor) a task actor may show; a 4 cm cube at the far
#: edge of the workspace covers ~60.
MIN_VISIBLE_PX = 20
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


def _actor(env, name: str):
    """A TASK_ACTORS entry of ``env``: an attribute, or ``objs/<model id>`` in its
    ``objs`` dict (the bridge twins)."""
    if name.startswith("objs/"):
        return env.objs[name[len("objs/") :]]
    return getattr(env, name)


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


#: ik_server's model of each arm (``--robot``) and its arm joint count; an arm without one
#: answers env.preview_reach with "unknown".
IK_MODELS = {"panda": "panda", "xarm6_robotiq": "xarm6"}
IK_JOINTS = {"panda": 7, "xarm6": 6}


def _place_depth(depth: np.ndarray, nh: int, nw: int, size: int) -> np.ndarray:
    """``depth`` resized to ``nh`` x ``nw`` (nearest: no depth is blended across an edge) and
    centred in a ``size`` square of zeros (no depth), as the letterboxes place the image."""
    h, w = depth.shape
    rows = np.minimum((np.arange(nh) + 0.5) * h / nh, h - 1).astype(int)
    cols = np.minimum((np.arange(nw) + 0.5) * w / nw, w - 1).astype(int)
    out = np.zeros((size, size), dtype=np.float32)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = depth[rows][:, cols]
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


@dataclass(frozen=True)
class RobotSpec:
    """One ``--robot``: a ManiSkill agent with a parallel gripper, driven in
    ``pd_ee_delta_pos`` (base-frame TCP translation, orientation held by the IK).

    ``gripper`` is the (open, close) gripper action held while the command lasts, None
    for a tool without one (the action is the translation alone); ``width`` how
    ``gripper_width`` is measured: ``("qpos",)`` the sum of the last two joint positions
    (two prismatic fingers), ``("pads", link_a, link_b, offset)`` the distance between two
    finger links minus their distance when closed on nothing, ``("none",)`` always 0.
    ``envs`` are the stock env ids its reset, visibility gate and reach were checked on
    (the rigs fix their own Panda). ``wrist`` is the hand camera: ``mount`` ("centered":
    the Panda's patched D415 mount; otherwise the agent's own link it is added on, with
    ``pose`` [p, q wxyz] from ManiSkill's ``*_wristcam`` variant), and the view transform;
    None: the robot has no camera link, and observations carry the agentview alone.
    ``ee_joints`` adds a ``pd_ee_delta_pos`` mode (ManiSkill's PDEEPosController on these
    joints and the agent's TCP link) to an agent that ships joint control only. ``arms``
    names the agents of a multi-robot uid (a tuple), in ManiSkill's agent order: each
    servo call drives one arm while the others hold still with their last gripper
    command."""

    uid: str | tuple[str, ...]
    name: str
    gripper: Optional[tuple[float, float]]
    width: tuple
    envs: tuple[str, ...]
    wrist: Optional[dict]
    ee_joints: Optional[tuple[str, ...]] = None
    arms: Optional[tuple[str, ...]] = None
    #: The scene's own controller when it ships one (the bridge twins): ``pos_scale`` m per
    #: normalised unit (an unnormalised controller) and ``rot_dims`` zero rotation entries
    #: between the translation and the gripper (a pose controller held at its orientation).
    control_mode: str = "pd_ee_delta_pos"
    pos_scale: float = 1.0
    rot_dims: int = 0
    #: The TCP link of an agent without a ``tcp`` attribute.
    tcp_link: Optional[str] = None
    #: The agent's own camera used as the agentview (no oblique camera is added).
    agentview: Optional[str] = None
    #: The env's reward mode when the task's default cannot be computed for this robot
    #: (None: the task's default).
    reward_mode: Optional[str] = None
    #: ``(agent uids, the scene's mode)``: ``control_mode`` is added to those agents as a copy
    #: of that mode acting on the current pose (``use_target`` off). A target-based mode sums
    #: every commanded delta into its target, so a closed-loop servo pressing against the
    #: table winds the target up and the arm lunges when released.
    untargeted_from: Optional[tuple[tuple[str, ...], str]] = None

    @property
    def open(self) -> Optional[float]:
        return self.gripper[0] if self.gripper else None

    def gripper_action(self, command: float) -> Optional[float]:
        """``command`` > 0 opens, <= 0 closes (the servo's and pi's convention); None
        without a gripper."""
        if not self.gripper:
            return None
        return self.gripper[0] if command > 0 else self.gripper[1]

    def action(self, delta: np.ndarray, command: float) -> np.ndarray:
        """One control action: the normalised translation, then the gripper action (a
        robot with one)."""
        g = self.gripper_action(command)
        if self.pos_scale != 1.0 or self.rot_dims:
            delta = np.concatenate([delta * self.pos_scale, np.zeros(self.rot_dims)])
        return delta if g is None else np.append(delta, g)


#: Stock tasks another robot owns: registered with it fixed (FIXED_ROBOT_ENVS) or built for
#: it alone (the panda_stick scenes: no gripper).
_OTHER_ROBOT = {
    "PickCubeWidowXAI-v1": "widowxai",
    "PushT-v1": "panda_stick",
    "DrawTriangle-v1": "panda_stick",
    "DrawSVG-v1": "panda_stick",
    "TwoRobotPickCube-v1": "panda_pair",
    "TwoRobotStackCube-v1": "panda_pair",
    "PutCarrotOnPlateInScene-v1": "widowx250s",
    "PutEggplantInBasketScene-v1": "widowx250s",
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1": "widowx250s",
    "PutSpoonOnTableClothInScene-v1": "widowx250s",
}
#: The stock env ids the Panda's reset was checked on; PickCube-v1 has a per-robot
#: layout (ManiSkill's PICK_CUBE_CONFIGS: cube size, spawn area, goal height).
_ALL_STOCK = tuple(e for e in INSTRUCTIONS if e not in _OTHER_ROBOT)
#: ``--robot``: the ManiSkill 3.0.1 agents with a parallel gripper that the stock table
#: scene places (TableSceneBuilder) and that reach the tasks' objects in
#: ``pd_ee_delta_pos``. Measured on the box (reset, visibility gate, reach, MV_* probe):
#: see ``_ALL_STOCK`` for the Panda and ``envs`` for the others.
ROBOTS: dict[str, RobotSpec] = {
    # The default: panda_wristcam with Show-Harness's centred wrist mount (``main``).
    "panda": RobotSpec(
        uid="panda_wristcam",
        name="Franka Panda",
        gripper=(1.0, -1.0),
        width=("qpos",),
        envs=_ALL_STOCK,
        wrist={"mount": "centered", "rotation": 270, "flip": "none"},
    ),
    # Robotiq 2F-85 in delta mode: +1 closes by 0.15 rad per control step, -1 opens; held,
    # the command drives the knuckle to its limit (0.81 closed) or presses the object.
    # ``eef`` (the TCP) sits between the fingertips. The hand camera is
    # xarm6_robotiq_wristcam's, added on ``camera_link`` of xarm6_robotiq (the wristcam
    # uid is not placed by the table scene: its arm spawns inside the table).
    "xarm6_robotiq": RobotSpec(
        uid="xarm6_robotiq",
        name="UFactory xArm6 with a Robotiq 2F-85 gripper",
        gripper=(-1.0, 1.0),
        width=("pads", "left_inner_finger_pad", "right_inner_finger_pad", 0.0067),
        # PushCube / PokeCube put the goal (and PokeCube the cube) beyond its reach (x > 0.1,
        # 0.62 m from the base); PegInsertionSide resets a Panda's joint vector only;
        # PickSingleYCB has no xarm6 layout. PullCubeTool succeeds with the cube within
        # 0.6 m of the base, and the xArm6's base (x = -0.522) is 9 cm nearer the cube
        # than the Panda's: ~8 % of seeds start solved.
        envs=(
            "PickCube-v1",
            "StackCube-v1",
            "PullCube-v1",
            "LiftPegUpright-v1",
            "PlaceSphere-v1",
            "StackPyramid-v1",
            "PlugCharger-v1",
        ),
        # The dense rewards of StackCube / PlaceSphere divide by twice the last joint's upper
        # limit, the Panda's finger width; the Robotiq's last joint is the right inner finger,
        # limit 0 (measured): the ungrasp term is a division by zero. Success-based instead.
        reward_mode="sparse",
        wrist={
            "mount": "camera_link",
            "pose": [[0.0, 0.0, -0.05], [0.70710678, 0.0, 0.70710678, 0.0]],
            # Measured (PickCube / StackCube seed 0, world axes projected through the
            # camera): raw image left = +x, down = +y; 90 deg CCW gives the Panda's
            # convention (right = +y, bottom = +x) with the fingertips at the top corners.
            "rotation": 90,
            "flip": "none",
        },
    ),
    # Joint control only in ManiSkill: ``ee_joints`` adds pd_ee_delta_pos on its six arm
    # joints and ``ee_gripper_link``. wxai_base.urdf has no camera link (the D405 is on
    # widowxai_wristcam's wxai_follower.urdf, which PickCube has no layout for).
    "widowxai": RobotSpec(
        uid="widowxai",
        name="Trossen WidowX AI",
        gripper=(1.0, -1.0),
        width=("qpos",),
        # With the gripper held pointing down its reach ends ~0.37 m from the base (the
        # wrist pitch hits its limit): only PickCube's own layout (cube at x -0.25, goal up
        # to 0.2 m) is within it; the other scenes' objects sit at x ~ 0, 0.6 m away.
        envs=("PickCube-v1", "PickCubeWidowXAI-v1"),
        wrist=None,
        ee_joints=tuple(f"joint_{i}" for i in range(6)),
    ),
    # A Panda holding a stick (ManiSkill's panda_stick: pd_ee_delta_pos on the arm, no
    # gripper, no camera link): the pushing and drawing scenes built for it.
    "panda_stick": RobotSpec(
        uid="panda_stick",
        name="Franka Panda with a stick",
        gripper=None,
        width=("none",),
        envs=("PushT-v1", "DrawTriangle-v1", "DrawSVG-v1"),
        wrist=None,
    ),
    # Two Pandas facing each other across the table (the two-robot scenes): agent 0 at
    # y = -0.75 (the agentview's left), agent 1 at y = +0.75. The plain panda uid: the
    # table scene places ("panda", "panda") like the scenes' wristcam pair, and the
    # agentview shows both arms.
    "panda_pair": RobotSpec(
        uid=("panda", "panda"),
        name="two Franka Pandas",
        gripper=(1.0, -1.0),
        width=("qpos",),
        envs=("TwoRobotPickCube-v1", "TwoRobotStackCube-v1"),
        wrist=None,
        arms=("left", "right"),
    ),
    # The bridge twins' WidowX 250 S (the scene picks its flat-table or sink variant): its
    # arm_pd_ee_target_delta_pose_align2 controller, copied without its target (a Muse
    # episode pressed the gripper into the table, the summed target sent the arm 0.43 m on
    # the next release and left it at an unreachable target), takes an unnormalised
    # [dx, dy, dz] in m and a rotation (held at zero), then the normalised mimic gripper
    # (+1 open); the TCP is ee_gripper_link; the agentview is the scene's 3rd_view_camera
    # (with the real background composited behind the objects).
    "widowx250s": RobotSpec(
        uid="widowx250s",
        name="WidowX 250 S",
        gripper=(1.0, -1.0),
        width=("qpos",),
        envs=BRIDGE_ENVS,
        wrist=None,
        control_mode="pi_ee_delta_pose_gripper_pd_joint_pos",
        untargeted_from=(
            (
                "widowx250s_bridgedataset_flat_table",
                "widowx250s_bridgedataset_sink",
            ),
            "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        ),
        pos_scale=DELTA_BOUND_M,
        rot_dims=3,
        tcp_link="ee_gripper_link",
        agentview="3rd_view_camera",
    ),
}


def add_wrist_camera(agent_cls, link: str, pose: list) -> None:
    """Give ``agent_cls`` a 256 px ``hand_camera`` on its ``link`` (in place, like
    ``center_wrist_camera``: the table scene only places uids it knows)."""
    import sapien
    from mani_skill.sensors.camera import CameraConfig

    agent_cls._sensor_configs = property(
        lambda self: [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=pose[0], q=pose[1]),
                width=256,
                height=256,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map[link],
            )
        ]
    )


def add_ee_control(agent_cls, joints: tuple[str, ...]) -> None:
    """Add ``pd_ee_delta_pos`` to ``agent_cls``: ManiSkill's PDEEPosController (0.1 m
    bound, the arm's own gains) on ``joints`` and the agent's TCP link, with the
    gripper controller of its ``pd_joint_pos`` mode. Idempotent."""
    from mani_skill.agents.controllers import PDEEPosControllerConfig

    base = agent_cls._controller_configs
    # The mark sits on the wrapping property's getter (a property takes no attributes).
    if getattr(base.fget, "_pi_ee", False):
        return

    def configs(self):
        cfg = base.fget(self)
        arm = PDEEPosControllerConfig(
            joint_names=list(joints),
            pos_lower=-DELTA_BOUND_M,
            pos_upper=DELTA_BOUND_M,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        rest = {k: v for k, v in cfg["pd_joint_pos"].items() if k != "arm"}
        return {**cfg, "pd_ee_delta_pos": {"arm": arm, **rest}}

    prop = property(configs)
    prop.fget._pi_ee = True  # type: ignore[attr-defined]
    agent_cls._controller_configs = prop


def robot_spec(robot: str, env_id: str, rig: bool) -> RobotSpec:
    """The ``--robot`` spec, refused unless it was checked on ``env_id``; a rig
    (BlockPAP-v1 / BlockStack-v1) runs its own Panda only."""
    if robot not in ROBOTS:
        raise ValueError(f"unknown robot {robot!r}; one of {list(ROBOTS)}")
    spec = ROBOTS[robot]
    if rig:
        if robot != "panda":
            raise ValueError(
                f"{env_id} is a real2sim rig with its own Panda; --robot panda only"
            )
    elif env_id not in spec.envs:
        raise ValueError(f"--robot {robot} runs {list(spec.envs)}, not {env_id}")
    return spec


def add_untargeted_mode(agent_cls, base: str, name: str) -> None:
    """Add mode ``name`` to ``agent_cls``: a copy of its mode ``base`` whose arm acts on the
    current end-effector pose (``use_target`` off). Idempotent."""
    prop = agent_cls._controller_configs
    if getattr(prop.fget, "_pi_untargeted", False):
        return

    def configs(self):
        cfg = prop.fget(self)
        mode = dict(cfg[base])
        arm = copy.deepcopy(mode["arm"])
        arm.use_target = False
        return {**cfg, name: {**mode, "arm": arm}}

    configs._pi_untargeted = True  # type: ignore[attr-defined]
    agent_cls._controller_configs = property(configs)


def prepare_robot(spec: RobotSpec, wrist_mount: str) -> None:
    """Patch the agent class before ``gym.make``: the wrist camera and the EE mode."""
    from mani_skill.agents.registration import REGISTERED_AGENTS

    if spec.untargeted_from:
        uids, base = spec.untargeted_from
        for uid in uids:
            add_untargeted_mode(
                REGISTERED_AGENTS[uid].agent_cls, base, spec.control_mode
            )
        return
    if spec.arms or (spec.wrist is None and not spec.ee_joints):
        return  # the agent as it is: no wrist camera, its own EE mode
    cls = REGISTERED_AGENTS[spec.uid].agent_cls
    if spec.ee_joints:
        add_ee_control(cls, spec.ee_joints)
    if spec.wrist is None:
        return
    if spec.wrist["mount"] == "centered":
        if wrist_mount == "centered":
            center_wrist_camera()
    else:
        add_wrist_camera(cls, spec.wrist["mount"], spec.wrist["pose"])


class ManiskillEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """One ManiSkill env (``num_envs=1``); every call runs on the main thread."""

    SERVICE_NAME = "maniskill-env"

    def __init__(
        self,
        *,
        env_id: str,
        seed: int,
        robot: str = "panda",
        wrist_mount: str = "centered",
        control_mode: str = "pd_ee_delta_pos",
        sim_backend: str = "physx_cpu",
        agentview: str = "oblique",
        view_size: int = 256,
        max_episode_steps: int = 100_000,
        settle_steps: int = 8,
        wrist_rotation: int | None = None,
        wrist_flip: str | None = None,
        scene: dict | None = None,
    ):
        super().__init__()
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 -- registers the stock env ids

        from pi_embodied_services.robots.maniskill import scenes

        if env_id not in ENV_IDS:
            raise ValueError(f"unknown env id {env_id!r}; one of {ENV_IDS}")
        #: An RLinf real2sim rig (./scenes.py) and its options, or None for a stock scene.
        self._rig = scenes.SCENES.get(env_id)
        #: The --robot (ROBOTS): gripper actions, width, wrist camera.
        self._robot = robot_spec(robot, env_id, bool(self._rig))
        wrist = self._robot.wrist or {}
        if wrist.get("mount") != "centered":
            # The Panda's --wrist-mount choice; another robot's camera is its own.
            wrist_mount = wrist.get("mount", "none")
        if wrist_rotation is None:
            wrist_rotation = wrist.get("rotation", 0)
        if wrist_flip is None:
            wrist_flip = wrist.get("flip", "none")
        robot_uids = self._robot.uid
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
                obs_mode="rgb+depth+segmentation",
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
            table_z = TABLE_Z.get(env_id, 0.0)
            prepare_robot(self._robot, wrist_mount)
            if self._robot.control_mode != "pd_ee_delta_pos":
                control_mode = self._robot.control_mode
            if self._robot.agentview:
                # The scene's own camera is the agentview; no camera is added.
                agentview = self._robot.agentview
                self._cameras = {"agentview": agentview}
            # An env id that fixes its robot (FIXED_ROBOT_ENVS) takes no robot_uids.
            fixed = env_id in FIXED_ROBOT_ENVS
            self._env = gym.make(
                env_id,
                num_envs=1,
                obs_mode="rgb+depth+segmentation",
                control_mode=control_mode,
                **({} if fixed else {"robot_uids": robot_uids}),
                **(
                    {"reward_mode": self._robot.reward_mode}
                    if self._robot.reward_mode
                    else {}
                ),
                sim_backend=sim_backend,
                max_episode_steps=int(max_episode_steps),
                sensor_configs=(
                    {}
                    if self._robot.agentview
                    else self._sensor_configs(
                        agentview, wrist=self._robot.wrist is not None
                    )
                ),
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
            "robot": robot,
            "robot_uids": robot_uids,
            "control_mode": control_mode,
            "reward_mode": self._env.unwrapped.reward_mode,
            "sim_backend": sim_backend,
            "agentview": agentview,
            "view_size": self._view_size,
            "settle_steps": self._settle_steps,
            "wrist_mount": wrist_mount,
            "wrist_rotation": self._wrist_rotation,
            "wrist_flip": wrist_flip,
            "scene": self._scene,
            "table_z": table_z,
            # A multi-robot env's action space is a dict: its per-agent shapes.
            "action_space": (
                list(self._env.action_space.shape)
                if self._env.action_space.shape is not None
                else {
                    k: list(v.shape) for k, v in self._env.action_space.spaces.items()
                }
            ),
            "gripper_action": (
                {"open": self._robot.gripper[0], "close": self._robot.gripper[1]}
                if self._robot.gripper
                else None
            ),
            "wrist": self._robot.wrist is not None or bool(self._rig),
            "arms": list(self._robot.arms) if self._robot.arms else None,
        }
        #: The last gripper command of each arm (> 0 open), held while another arm moves.
        self._grip = [1.0] * len(self._robot.arms or [None])

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["env.state"] = self.state
        self._rpc["env.servo"] = self.servo
        self._rpc["env.ground_truth_poses"] = self.ground_truth_poses
        register_code_api(self, MANISKILL_PRIMITIVES)

    # ---- helpers ----

    @staticmethod
    def _sensor_configs(agentview: str, wrist: bool = True) -> dict:
        """The agentview pose (``oblique`` = AGENTVIEW, or the scene's ``stock`` camera)
        and a 256 px wrist render (on a robot with a wrist camera)."""
        cfg: dict = {"hand_camera": {"width": 256, "height": 256}} if wrist else {}
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

    @property
    def _agents(self) -> list:
        """The arm agents, in ``RobotSpec.arms`` order (the one agent of a single arm)."""
        agent = self._agent
        return list(agent.agents) if self._robot.arms else [agent]

    def _arm(self, arm: Optional[str]) -> int:
        """The index of ``arm`` (required on a multi-arm robot, refused on one arm)."""
        arms = self._robot.arms
        if not arms:
            if arm is not None:
                raise ValueError(f"--robot {self._meta['robot']} has one arm; no arm")
            return 0
        if arm not in arms:
            raise ValueError(f"arm must be one of {list(arms)}, not {arm!r}")
        return arms.index(arm)

    def _env_action(self, actions: list) -> Any:
        """Per-arm actions as the env takes them: the array (one arm) or a dict keyed by
        the multi-agent action space's agent ids, in agent order."""
        if not self._robot.arms:
            return actions[0]
        return dict(zip(self._env.action_space.spaces.keys(), actions))

    def _hold(self) -> Any:
        """Every arm still, each with its last gripper command."""
        return self._env_action(
            [self._robot.action(np.zeros(3), g).astype(np.float32) for g in self._grip]
        )

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

    def _depth(self, obs: dict, name: str) -> np.ndarray:
        """Metric depth (float32 m, 0 = none) of one view, through the same orientation,
        crop and letterbox as :meth:`_rgb`, so pixel (row, col) of both is one ray."""
        raw = _np(obs["sensor_data"][self._cameras[name]]["depth"])[0]
        depth = raw.reshape(raw.shape[:2]).astype(np.float32) / 1000.0  # int16 mm
        if self._rig:
            from pi_embodied_services.robots.maniskill import scenes

            v = scenes.VIEWS[name]
            depth = _orient(depth, v["rotation"], v["flip"])
            h, w = depth.shape
            crop = v["crop"]
            if crop and abs(w / h - crop) >= 1e-6:
                if w / h > crop:
                    nw = int(round(h * crop))
                    depth = depth[:, (w - nw) // 2 : (w - nw) // 2 + nw]
                else:
                    nh = int(round(w / crop))
                    depth = depth[(h - nh) // 2 : (h - nh) // 2 + nh]
            if not self._view_size:
                return np.ascontiguousarray(depth)
            h, w = depth.shape
            scale = min(self._view_size / w, self._view_size / h)
            return _place_depth(depth, int(h * scale), int(w * scale), self._view_size)
        if name == "wrist":
            depth = _orient(depth, self._wrist_rotation, self._wrist_flip)
        if not self._view_size:
            return np.ascontiguousarray(depth)
        h, w = depth.shape
        scale = self._view_size / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        return _place_depth(depth, nh, nw, self._view_size)

    def _gripper_width(self, qpos: np.ndarray, agent=None) -> float:
        """The finger opening, m (``RobotSpec.width``); about 0 closed on nothing."""
        width = self._robot.width
        if width[0] == "none":
            return 0.0
        if width[0] == "qpos":
            return float(qpos[-1] + qpos[-2])
        links = (agent or self._agent).robot.links_map
        a, b = (_np(links[n].pose.p).reshape(-1) for n in width[1:3])
        return max(0.0, float(np.linalg.norm(a - b)) - width[3])

    @property
    def _has_wrist(self) -> bool:
        return self._meta["wrist"]

    def _tcp(self, agent):
        """The agent's TCP pose (``RobotSpec.tcp_link`` for an agent without ``tcp``)."""
        if self._robot.tcp_link:
            return agent.robot.links_map[self._robot.tcp_link].pose
        return agent.tcp.pose

    def _arm_state(self, agent) -> dict:
        tcp = self._tcp(agent)
        qpos = _np(agent.robot.get_qpos()).reshape(-1)
        return {
            "tcp_pos": _np(tcp.p).reshape(-1).astype(np.float32),
            "tcp_quat_wxyz": _np(tcp.q).reshape(-1).astype(np.float32),
            "gripper_width": self._gripper_width(qpos, agent),
            "qpos": qpos.astype(np.float32),
        }

    def _state(self) -> dict:
        """The arm's TCP pose, gripper opening and joints; a multi-arm robot's per arm
        under ``arms``."""
        if not self._robot.arms:
            return self._arm_state(self._agent)
        return {
            "arms": {
                name: self._arm_state(agent)
                for name, agent in zip(self._robot.arms, self._agents)
            }
        }

    def _pack(self, obs: dict) -> dict:
        """The agentview, the wrist view (a robot with a wrist camera only) and the state."""
        self._obs = obs
        views = {"agentview": self._rgb(obs, "agentview")}
        if self._has_wrist:
            views["wrist"] = self._rgb(obs, "wrist")
        return {**views, **self._state()}

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
        if isinstance(action, dict):
            a = {
                k: np.asarray(v, dtype=np.float32).reshape(1, -1)
                for k, v in action.items()
            }
        else:
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
        self._grip = [1.0] * len(self._grip)
        hold = self._hold()
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
            ids = _np(_actor(env, name).per_scene_id).reshape(-1)
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

    def _split(self, action) -> Any:
        """A flat code-mode action in pi's convention, ``[dx, dy, dz, gripper]`` per arm
        (``[dx, dy, dz]`` without a gripper; arms concatenated in arm order), as the env
        takes it: the gripper value keeps its magnitude with the robot's sign (the
        Robotiq's +1 closes, so pi's +1 open becomes -1), the translation the robot's
        scale and zero rotation (``RobotSpec.action``). The Panda's is unchanged."""
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        n = len(self._robot.arms or [None])
        out = []
        for arm in a.reshape(n, -1):
            native = self._robot.action(arm[:3], 1.0).astype(np.float32)
            if self._robot.gripper:
                native[-1] = arm[3] * self._robot.gripper[0]
            out.append(native)
        return self._env_action(out)

    def step(self, action):
        obs, rew, term, trunc, info = self._step(self._split(action))
        return self._pack(obs), rew, term, trunc, info

    def chunk_step(self, actions, *, return_all_frames: bool = False):
        """Run ``actions`` [N, 4] in one call; stops early on ``stop``, termination,
        truncation or success. Arrays hold one entry per executed action."""
        frames, rews, terms, truncs = [], [], [], []
        info: dict = {}
        dim = (3 + (self._robot.gripper is not None)) * len(self._robot.arms or [None])
        for action in np.asarray(actions, dtype=np.float32).reshape(-1, dim):
            if self.stop_requested():
                info["cancelled"] = True
                break
            obs, rew, term, trunc, info = self._step(self._split(action))
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
        arm: Optional[str] = None,
    ):
        """Drive the TCP to ``target_xyz`` (world, m) with the gripper command held (> 0
        open, <= 0 close; the robot's gripper action, ``RobotSpec.gripper``): each
        control step commands ``clip(error * gain / 0.1)``, until the error is below
        ``tol_m`` (after ``min_steps``), ``max_steps``, success or ``stop``. The closed-loop
        2 cm execution of Show-Harness's real2sim tokenizer; open-loop steps fall short when
        the arm reverses (PD lag). A multi-arm robot drives ``arm``; the others hold still
        with their last gripper command. Returns ``[frames, info]`` with one frame per step,
        each with the action it applied (flat: every arm's in agent order) and its
        ``success`` (a call that ran no step returns the current frame alone, without them)."""
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        idx = self._arm(arm)
        agent = self._agents[idx]
        self._grip[idx] = float(gripper)
        frames: list = []
        info: dict = {}
        for k in range(int(max_steps)):
            if self.stop_requested():
                info["cancelled"] = True
                break
            err = target - _np(self._tcp(agent).p).reshape(-1).astype(np.float32)
            if k >= int(min_steps) and np.linalg.norm(err) < tol_m:
                break
            actions = [
                self._robot.action(np.zeros(3), g).astype(np.float32)
                for g in self._grip
            ]
            actions[idx] = self._robot.action(
                np.clip(self._to_root(err, agent) * gain / DELTA_BOUND_M, -1, 1),
                float(gripper),
            )
            a = self._env_action(actions)
            obs, _rew, term, trunc, info = self._step(a)
            # The env action of the step and its success, for the Flywheel recorder.
            frames.append(
                {
                    **self._pack(obs),
                    "action": np.concatenate(
                        [np.asarray(x, dtype=np.float32).reshape(-1) for x in actions]
                    ),
                    "success": bool(info.get("success")),
                }
            )
            if term or trunc or info.get("success"):
                break
        if not frames:
            frames.append(self._pack(self._obs))
        return frames, info

    def _to_root(self, vec: np.ndarray, agent=None) -> np.ndarray:
        """A world-frame vector in the robot base's frame, where ``pd_ee_delta_pos``
        acts: unchanged for a base at the identity rotation (the one-arm table scenes),
        rotated for a base turned about z (the two-robot scenes' arms)."""
        q = _np((agent or self._agent).robot.pose.q).reshape(-1)
        if np.allclose(q, [1.0, 0.0, 0.0, 0.0]):
            return vec
        from scipy.spatial.transform import Rotation

        r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # xyzw
        return r.inv().apply(vec)

    def state(self) -> dict:
        """TCP pose, gripper opening and the current success flags (no stepping)."""
        info = self._info(self._env.unwrapped.evaluate())
        return {**self._state(), "info": info}

    def ground_truth_poses(self, names=None) -> dict:
        """World poses of ``names`` (default all) from the scene's own object list
        (``--privileged``): its actors (goal markers included) and its articulations other
        than the robot."""
        env = self._env.unwrapped
        # A multi-robot scene's agent holds its arms in ``agents``.
        robots = [a.robot for a in getattr(env.agent, "agents", [env.agent])]
        objects = {
            **env.scene.actors,
            **{
                k: a
                for k, a in env.scene.articulations.items()
                if not any(a is r for r in robots)
            },
        }
        return ground_truth.respond(
            {
                name: ground_truth.pose(
                    _np(o.pose.p).reshape(-1), _np(o.pose.q).reshape(-1)
                )
                for name, o in objects.items()
            },
            names,
        )

    def _camera(self, camera_name: str) -> str:
        if camera_name == "wrist" and not self._has_wrist:
            raise ValueError(f"--robot {self._meta['robot']} has no wrist camera")
        return self._cameras[camera_name]

    def render_camera(
        self, camera_name: str = "agentview", depth: bool = False, **_: Any
    ):
        """Latest frame of ``agentview`` or ``wrist``, as the model sees it; with ``depth``,
        ``[rgb, depth_m]`` (float32 metres, 0 = none: the letterbox bars) on the same pixels."""
        self._camera(camera_name)
        rgb = self._rgb(self._obs, camera_name)
        return [rgb, self._depth(self._obs, camera_name)] if depth else rgb

    def get_camera_meta(self, camera_name: str = "agentview", **_: Any) -> dict:
        """OpenCV intrinsics and camera-to-world extrinsic of a sensor camera (raw sensor
        pixels, before the orientation and letterbox of ``render_camera``)."""
        param = self._obs["sensor_param"][self._camera(camera_name)]
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
    p.add_argument("--env-id", choices=ENV_IDS, default="BlockPAP-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--robot",
        choices=list(ROBOTS),
        default="panda",
        help="the arm (ROBOTS); the rigs run their own Panda (panda only)",
    )
    p.add_argument(
        "--wrist-mount",
        choices=["centered", "camera_link"],
        default="centered",
        help="the Panda's wrist camera mount; another robot's is its own",
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
    # Another robot's wrist transform (ROBOTS[...].wrist) was measured the same way; the
    # default is the robot's.
    p.add_argument(
        "--wrist-rotation", type=int, choices=[0, 90, 180, 270], default=None
    )
    p.add_argument(
        "--wrist-flip",
        choices=["none", "vertical", "horizontal", "both"],
        default=None,
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
    add_perception_arguments(p, sam3=True)
    reach.add_ik_argument(p)
    args = p.parse_args()

    facade = ManiskillEnvFacade(
        env_id=args.env_id,
        seed=args.seed,
        robot=args.robot,
        sim_backend=args.sim_backend,
        agentview=args.agentview,
        view_size=args.view_size,
        settle_steps=args.settle_steps,
        wrist_mount=args.wrist_mount,
        wrist_rotation=args.wrist_rotation,
        wrist_flip=args.wrist_flip,
        scene=dict(kv.split("=", 1) for kv in args.scene.split(",") if kv) or None,
    )
    # --ik: env.preview_reach through the ik service's model of the arm (ik_server ROBOTS; the other
    # arms answer "unknown").
    ik_model = IK_MODELS.get(facade._meta["robot"])

    def _wxyz_to_xyzw(q) -> list[float]:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]

    def _base() -> dict:
        pose = facade._agent.robot.pose
        return {"pos": _np(pose.p).reshape(-1), "quat_xyzw": _wxyz_to_xyzw(_np(pose.q))}

    reach.install_preview_reach(
        facade,
        reach.reach_from_args(args, ik_model) if ik_model else None,
        joints=lambda: _np(facade._agent.robot.get_qpos()).reshape(-1)[
            : IK_JOINTS[ik_model]
        ],
        eef_quat_xyzw=lambda: _wxyz_to_xyzw(facade._state()["tcp_quat_wxyz"]),
        base_pose=_base,
    )
    # --sam3 / --unidepth: env.detect, env.select_detection, env.reject_detection, env.enhance_depth
    # (no rendered depth here: enhance_depth supplies it). The views are letterboxed: no K.
    install_perception(
        facade,
        args,
        cameras=["agentview", "wrist"],
        view=render_view(facade, intrinsics=False),
        mutating=("env.servo",),
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
