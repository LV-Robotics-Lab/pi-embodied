/**
 * ManiSkill robot for pi: a Panda in ManiSkill 3's translation-only `pd_ee_delta_pos` mode.
 *
 *   pi -e packages/embodied/src/robots/maniskill --seed 0                    (BlockPAP-v1, Show-Harness's default scene)
 *   pi -e packages/embodied/src/robots/maniskill --units --scene table_tex=white,cam_t=0302 --seed 3
 *   pi -e packages/embodied/src/robots/maniskill --env-id StackCube-v1 --seed 0   (a stock ManiSkill scene)
 *   pi -e packages/embodied/src/robots/maniskill --env-id PlaceSphere-v1 --seed 0  (one of OpenETA's tasks, ENV_IDS)
 *   pi -e packages/embodied/src/robots/maniskill --robot xarm6_robotiq --env-id PickCube-v1 --seed 0  (another arm, ROBOTS)
 *   pi -e packages/embodied/src/robots/maniskill --env-id PickCube-v1 --seed 0 --code=true   (run_code over the env server's registry)
 *
 * BlockPAP-v1 / BlockStack-v1 are RLinf's real2sim replicas of the real Franka rig (services/.../
 * robots/maniskill/scenes.py; fetch_real2sim.sh installs them): their calibrated front RealSense
 * and Show-Harness's wrist transform, reset like the training episodes, so a seed reproduces the
 * Show-Harness-Data frames. `--scene` passes the rig's options (table_tex, cam_t og|0302|0303,
 * traj_id, layout wide|none).
 *
 * Starts one ManiSkill env server per session (services/.../robots/maniskill/env_server.py, the
 * `maniskill` venv; rendering needs a GPU). `--robot` picks the arm (ROBOTS: the Panda by default, an xArm6
 * with a Robotiq gripper, a WidowX AI without a wrist camera); the rigs run their own Panda. `move_delta`, the units hook `apply`
 * and code mode's `move_delta` run one server method (env.move_delta): a base-frame delta in metres becomes ~2 cm
 * decisions, each a closed-loop servo of 2-8 control steps to its waypoint (Show-Harness's step calibration and
 * real2sim execution), with the per-call cap, stop and the latched success on the server;
 * every result carries the agentview and wrist images and the state; success is ManiSkill's own `success` flag, recorded in `robot_result`.
 * --collect-flywheel-data records every control step of a motion, per arm (services robots/maniskill/flywheel.py).
 * Tools and code primitives: ../../primitives/manifests/maniskill.json (the env server reads it too).
 *
 * Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: interpreters/maniskill_atomic_controller.py, core/sim/maniskill_scenes.py
 * and configs/robot_maniskill.yaml ported as a pi robot.
 */

import { writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { anchorPlane, type CameraMeta, pixelOnPlane, unletterbox } from "../../capabilities/flash/plane.ts";
import { recipeFlash } from "../../capabilities/flash/recipe.ts";
import type { FlywheelObs, FlywheelSpec } from "../../capabilities/flywheel.ts";
import { MOLMO, SAM3 } from "../../infra/model-services.ts";
import { trackFlags } from "../../infra/params.ts";
import { encodePng } from "../../infra/png.ts";
import { NdArray, type RpcClient } from "../../infra/rpc.ts";
import { MOVE_UNITS, type MoveUnit, type Vec3 } from "../../modes/units/index.ts";
import { template } from "../../planner/context-version.ts";
import { detectionActive, detectionArgs, detectionTools, registerDetectionFlags } from "../../primitives/detections.ts";
import { mountGraspTool } from "../../primitives/grasp.ts";
import { ikArgs, type Reach, registerIkFlag } from "../../primitives/ik.ts";
import { pointActive, pointTool, registerPointFlags } from "../../primitives/pointing.ts";
import { attach, defineRobot, type Json, rgbOf, SERVICES, toolResult } from "../../robot.ts";

const read = (name: string) => template(new URL(name, import.meta.url));
const SYSTEM = read("./SYSTEM.md");
const EXPLORE = read("./explore.md");
/** Keep (`on`) or drop a `[name]...[/name]` block of SYSTEM.md; the markers themselves always go. */
const section = (text: string, name: string, on: boolean) =>
	text.replace(new RegExp(`\\[${name}\\]\\n([\\s\\S]*?)\\[/${name}\\]\\n`, "g"), on ? "$1" : "");
/** A memory cell tag part: the scene options (`table_tex=white`) with the characters a tag may not hold replaced. */
const tagPart = (s: string) => s.replace(/[^\w.-]+/g, "-");

/**
 * `--env-id` takes one of these: the RLinf rigs (BlockPAP-v1 default, BlockStack-v1) and the stock
 * ManiSkill tabletop tasks the env server has a task text and a visibility list for (its INSTRUCTIONS
 * / TASK_ACTORS; from PlaceSphere-v1 on, OpenETA's ManiSkill table, sim/envs/maniskill at 7d4a0a1: the
 * Panda's, then those another `--robot` owns, OTHER_ROBOT_ENVS). The same list, in the same order.
 */
export const ENV_IDS = [
	"BlockPAP-v1",
	"BlockStack-v1",
	"PickCube-v1",
	"StackCube-v1",
	"PushCube-v1",
	"PullCube-v1",
	"PokeCube-v1",
	"LiftPegUpright-v1",
	"PlaceSphere-v1",
	"StackPyramid-v1",
	"PullCubeTool-v1",
	"PegInsertionSide-v1",
	"PlugCharger-v1",
	"PickSingleYCB-v1",
	"FMBAssembly1Easy-v1",
	"PickCubeWidowXAI-v1",
	"PushT-v1",
	"DrawTriangle-v1",
	"DrawSVG-v1",
	"TwoRobotPickCube-v1",
	"TwoRobotStackCube-v1",
	"PutCarrotOnPlateInScene-v1",
	"PutEggplantInBasketScene-v1",
	"StackGreenCubeOnYellowCubeBakedTexInScene-v1",
	"PutSpoonOnTableClothInScene-v1",
] as const;
export type EnvId = (typeof ENV_IDS)[number];
/** The RLinf rigs among ENV_IDS: they fix their own robot (a Panda). */
const RIGS: readonly string[] = ENV_IDS.slice(0, 2);
/** Stock tasks built for another robot than the Panda (env server _OTHER_ROBOT): only that `--robot` runs them. */
export const OTHER_ROBOT_ENVS: Partial<Record<EnvId, string>> = {
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
};

/** configs/robot_maniskill.yaml `move_vectors`: +x away from the base, -y = MV_LEFT, +z up. */
export const VECTORS: Record<MoveUnit, Vec3> = {
	MV_FWD: [1, 0, 0],
	MV_BACK: [-1, 0, 0],
	MV_LEFT: [0, -1, 0],
	MV_RIGHT: [0, 1, 0],
	MV_UP: [0, 0, 1],
	MV_DOWN: [0, 0, -1],
};
/**
 * Physical metres per decision (the MVTOKEN 2 cm convention). The waypoint split, the servo gain and
 * steps, each robot's gripper hold and the per-call cap live on the env server (env.move_delta,
 * env_server.py STEP_M / GAIN / SERVO_* / RobotSpec.gripper_steps / MAX_MOVE_M).
 */
export const STEP_M = 0.02;
/** Largest translation one call may command, m (the env server's MAX_MOVE_M, which enforces it). */
export const MAX_MOVE_M = 0.2;
/** Closed-and-empty gripper width, m (yaml `empty_width_m`, measured in sim). */
export const EMPTY_WIDTH_M = 0.005;

/**
 * How the views look, measured on PickCube seed 0 (units stepped, the cube projected through the
 * wrist calibration): the agentview (env server AGENTVIEW) sits low in front of the robot, turned
 * 15 deg toward the robot's left; the wrist camera (Show-Harness's centred mount, rotated 270 deg)
 * looks straight down with the fingertips at the left edge and the point under the TCP at
 * mid-height, 31-41 % of the width from the left (lower gripper = further left). Both are
 * letterboxed to 256x256.
 */
export const VIEWS = `Each result shows the third-person view, then the wrist view (both 256x256, black bars are padding). In BOTH views MV_LEFT / MV_RIGHT move the gripper toward the image left / right, MV_FWD toward the image bottom, MV_BACK toward the image top.
- Third-person view: it looks at the robot from in front of the table, slightly from the robot's left side, so the robot base is at the top and the directions above are tilted about 15 degrees; judge the gripper against the target directly.
- Wrist view: it looks straight down; the two fingertips stay fixed at the left edge (one near the top, one near the bottom), and the grasp point is between them, at mid-height, about a third of the width from the left edge. A target right of the grasp point needs MV_RIGHT, left of it MV_LEFT, below it MV_FWD, above it MV_BACK; a target on the grasp point is under the gripper: MV_DOWN. The camera moves with the gripper, so after MV_RIGHT the scene shifts left.`;

/** One arm's proprioception. */
type ArmObs = { tcp_pos: NdArray; tcp_quat_wxyz: NdArray; gripper_width: number; qpos: NdArray };
type Obs = {
	agentview: NdArray;
	/** Absent on a robot without a wrist camera (ManiskillRobot.setup.wrist_mount "none"). */
	wrist?: NdArray;
	/** A two-arm robot's arms (ManiskillRobot.arms), in place of the one arm's fields. */
	arms?: Record<string, ArmObs>;
} & Partial<ArmObs>;
type Info = Record<string, unknown>;
/** A servo control step: its observation, the env action it applied and its success (absent when none ran). */
type Frame = Obs & { action?: NdArray; success?: boolean };
/** env.move_delta's answer: the call's result, every control step, the last observation and step info. */
type MoveReturn = { result: Record<string, unknown>; frames: Frame[]; obs: Obs; info: Info };
/**
 * The RLinf rigs (BlockPAP-v1 / BlockStack-v1), measured on BlockPAP seed 20000 by stepping each
 * unit 4 cm and tracking the block's segmentation centroid: agentview = the calibrated front
 * RealSense facing the robot (image right = +y, down = toward the camera); wrist = the centred
 * mount turned 180 deg, fingertips at the top corners, MV_RIGHT shifts the scene 32 px left and
 * MV_FWD 30 px up; the point under the TCP is horizontally centred, 28 % down at the block's height
 * and 42 % down 0.2 m above it. Both letterboxed to 256x256 exactly as Show-Harness stores them.
 */
export const RIG_VIEWS = `Each result shows the third-person view, then the wrist view (both 256x256, black bars are padding). In BOTH views MV_LEFT / MV_RIGHT move the gripper toward the image left / right, MV_FWD toward the image bottom, MV_BACK toward the image top.
- Third-person view: a fixed camera in front of the table facing the robot, so the robot base is at the top and the table edge nearest the camera at the bottom; MV_FWD brings the gripper toward the camera (down in the image, and larger).
- Wrist view: it looks down past the gripper; the two fingertips stay fixed at the top corners, and the point under the gripper is horizontally centred, about 30% down from the top edge when the gripper is low and about 40% down when it is high. A target right of that point needs MV_RIGHT, left of it MV_LEFT, below it MV_FWD, above it MV_BACK; a target on it is under the gripper: MV_DOWN. The camera moves with the gripper, so after MV_RIGHT the scene shifts left.`;

/** SYSTEM.md's scene-dependent sentences: the stock scenes, then the RLinf rigs. */
const SCENE_TEXT = {
	stock: {
		table: "The table top is at z = 0",
		object: "a 4 cm cube's centre is at z = 0.02",
		views: "the agentview (third-person, 256x256 with black padding bars: it looks at the robot from in front of the table, turned about 15 degrees toward the robot's left; the robot base is at the top, image right is roughly +y, image bottom roughly +x) and the wrist view (looking straight down: the fingertips are fixed at its left edge and the point under the gripper is at mid-height, about a third of the width from the left; image right is +y, image bottom is +x there too)",
	},
	rig: {
		table: "The table top is at z = 0.03",
		object: "the 4 x 4 x 6 cm upright block's centre is at z = 0.06",
		views: "the agentview (third-person, 256x256 with black padding bars: a fixed camera in front of the table facing the robot; the robot base is at the top, image right is +y, image bottom is toward the camera, +x) and the wrist view (looking down past the gripper: the fingertips are fixed at its top corners and the point under the gripper is horizontally centred, 30-40% down from the top; image right is +y, image bottom is +x there too)",
	},
};

type Meta = {
	env_id: string;
	seed: number;
	scene: Record<string, unknown> | null;
	table_z?: number;
	/** The letterbox square the views are sent as (env server --view-size; 0 = raw). */
	view_size?: number;
	/** The env server's --robot (absent: a server from before it, a Panda). */
	robot?: string;
	capabilities?: { perception?: { segment?: boolean; enhance_depth?: boolean } };
} & Partial<typeof VIEW_SETUP>;
/** The raw agentview frame both scene kinds render (env server AGENTVIEW, the rigs' external_cam), letterboxed to view_size. */
export const AGENTVIEW_PX = { width: 640, height: 480 };
/** The env server camera setup VIEWS describes; a server rendering anything else is refused. */
export const VIEW_SETUP = { agentview: "oblique", wrist_mount: "centered", wrist_rotation: 270, wrist_flip: "none" };
/** The same for the RLinf rigs (RIG_VIEWS): their calibrated external_cam and the 180 deg wrist. */
export const RIG_VIEW_SETUP = {
	agentview: "external_cam",
	wrist_mount: "centered",
	wrist_rotation: 0,
	wrist_flip: "both",
};

/** The third-person sentences every stock-scene robot shares (the oblique AGENTVIEW); VIEWS opens with them. */
const THIRD_PERSON =
	"- Third-person view: it looks at the robot from in front of the table, slightly from the robot's left side, so the robot base is at the top and the directions above are tilted about 15 degrees; judge the gripper against the target directly.";

/**
 * The xArm6 + Robotiq 2F-85 (env server ROBOTS["xarm6_robotiq"]): measured on PickCube / StackCube seed 0 with
 * pi's motion path, each MV_* 4 units from the reset: 19.7 mm per unit along its axis (cos 1.000, the Panda's
 * 19.7-20.0 mm), so the Panda's vectors, step and gain. The Robotiq closes from 86 mm to 0 in 5 control steps
 * and opens in 6 (the server's 6-step hold); closed on a 4 cm cube it reads ~50 mm (pad links), on nothing 0. Its wrist
 * camera (xarm6_robotiq_wristcam's, on camera_link) renders +x to the image left and +y down; turned 90 deg
 * the directions match the agentview's, the fingertips sit at the top corners and the point under the TCP is
 * horizontally centred, ~41 % down at the table and ~25 % down at the TCP.
 */
export const XARM6_VIEWS = `Each result shows the third-person view, then the wrist view (both 256x256, black bars are padding). In BOTH views MV_LEFT / MV_RIGHT move the gripper toward the image left / right, MV_FWD toward the image bottom, MV_BACK toward the image top.
${THIRD_PERSON}
- Wrist view: it looks straight down; the two fingertips stay fixed at the top corners, and the point under the gripper is horizontally centred, about 40% down from the top edge at the table's height (higher objects appear nearer the top). A target right of that point needs MV_RIGHT, left of it MV_LEFT, below it MV_FWD, above it MV_BACK; a target on it is under the gripper: MV_DOWN. The camera moves with the gripper, so after MV_RIGHT the scene shifts left.`;
const XARM6_ROBOTIQ: ManiskillRobot = {
	arm: "UFactory xArm6 arm with a Robotiq 2F-85 gripper",
	envs: [
		"PickCube-v1",
		"StackCube-v1",
		"PullCube-v1",
		"LiftPegUpright-v1",
		"PlaceSphere-v1",
		"StackPyramid-v1",
		"PlugCharger-v1",
	],
	vectors: VECTORS,
	stepM: STEP_M,
	emptyWidthM: EMPTY_WIDTH_M,
	setup: { agentview: "oblique", wrist_mount: "camera_link", wrist_rotation: 90, wrist_flip: "none" },
	views: XARM6_VIEWS,
	scene: {
		...SCENE_TEXT.stock,
		views: "the agentview (third-person, 256x256 with black padding bars: it looks at the robot from in front of the table, turned about 15 degrees toward the robot's left; the robot base is at the top, image right is roughly +y, image bottom roughly +x) and the wrist view (looking straight down: the fingertips are fixed at its top corners and the point under the gripper is horizontally centred, about 40% down from the top at the table's height; image right is +y, image bottom is +x there too)",
	},
};

/**
 * The WidowX AI (env server ROBOTS["widowxai"], pd_ee_delta_pos added on its six arm joints): measured on
 * PickCube seed 0, each MV_* 19.7 mm per unit along its axis (cos 1.000). Its carriages close from 87 mm
 * to 4 mm in 6 control steps and under 1 mm only by the 10th, hence 10 hold steps; closed on PickCube's
 * 3.6 cm cube it reads ~41 mm. wxai_base.urdf has no camera link: observations carry the agentview alone.
 */
export const WIDOWXAI_VIEWS = `Each result shows the third-person view (256x256, black bars are padding); this robot has NO wrist camera, so there is no wrist view. MV_LEFT / MV_RIGHT move the gripper toward the image left / right, MV_FWD toward the image bottom, MV_BACK toward the image top.
${THIRD_PERSON}
- With no wrist view, judge alignment from the gripper's position against the target in the third-person view, and descend in small steps.`;
const WIDOWXAI: ManiskillRobot = {
	arm: "Trossen WidowX AI arm",
	envs: ["PickCube-v1", "PickCubeWidowXAI-v1"],
	vectors: VECTORS,
	stepM: STEP_M,
	emptyWidthM: EMPTY_WIDTH_M,
	setup: { agentview: "oblique", wrist_mount: "none", wrist_rotation: 0, wrist_flip: "none" },
	views: WIDOWXAI_VIEWS,
	scene: {
		table: SCENE_TEXT.stock.table,
		object: "a 3.6 cm cube's centre is at z = 0.018",
		views: "the agentview alone (third-person, 256x256 with black padding bars: it looks at the robot from in front of the table, turned about 15 degrees toward the robot's left; the robot base is at the top, image right is roughly +y, image bottom roughly +x). This robot has no wrist camera, so there is no wrist view",
	},
};

/**
 * The Panda with a stick (env server ROBOTS["panda_stick"]): ManiSkill's panda_stick in its own pd_ee_delta_pos,
 * no gripper and no camera link, on the pushing and drawing scenes built for it. `tcp_pos` is the stick's tip.
 */
export const PANDA_STICK_VIEWS = `Each result shows the third-person view (256x256, black bars are padding); this robot has NO wrist camera and NO gripper: it holds a stick pointing straight down. MV_LEFT / MV_RIGHT move the stick's tip toward the image left / right, MV_FWD toward the image bottom, MV_BACK toward the image top.
${THIRD_PERSON}
- With no wrist view, judge the tip's position against the target in the third-person view, and descend in small steps.`;
const PANDA_STICK: ManiskillRobot = {
	arm: "Franka Panda arm holding a stick (no gripper)",
	envs: ["PushT-v1", "DrawTriangle-v1", "DrawSVG-v1"],
	vectors: VECTORS,
	stepM: STEP_M,
	emptyWidthM: EMPTY_WIDTH_M,
	gripper: false,
	setup: { agentview: "oblique", wrist_mount: "none", wrist_rotation: 0, wrist_flip: "none" },
	views: PANDA_STICK_VIEWS,
	scene: {
		table: SCENE_TEXT.stock.table,
		object:
			"PushT's T block is about 4 cm tall; the drawing canvas's top is at z = 0.02 and the stick paints a dot there whenever its tip is below z = 0.028",
		views: "the agentview alone (third-person, 256x256 with black padding bars: it looks at the robot from in front of the table, turned about 15 degrees toward the robot's left; the robot base is at the top, image right is roughly +y, image bottom roughly +x). This robot has no wrist camera, so there is no wrist view",
	},
};

/**
 * Two Pandas facing each other across the table (env server ROBOTS["panda_pair"], the two-robot scenes): the
 * left arm's base at y = -0.75 (the agentview's left), the right arm's at y = +0.75, each in its own
 * pd_ee_delta_pos (the server turns the world-frame move into each base's frame). No wrist cameras.
 */
export const PANDA_PAIR_VIEWS = `Each result shows the third-person view (256x256, black bars are padding); there is no wrist view. The left arm stands at the image left, the right arm at the image right, facing each other across the table. MV_LEFT / MV_RIGHT move the chosen arm's gripper toward the image left / right (toward the left / right arm's base), MV_FWD toward the image bottom, MV_BACK toward the image top, for either arm.
- Third-person view: it looks at the table from in front of it, turned about 15 degrees; judge each gripper against its target directly, and descend in small steps.`;
const PANDA_PAIR: ManiskillRobot = {
	arm: "pair of Franka Panda arms (left and right)",
	envs: ["TwoRobotPickCube-v1", "TwoRobotStackCube-v1"],
	vectors: VECTORS,
	stepM: STEP_M,
	emptyWidthM: EMPTY_WIDTH_M,
	arms: ["left", "right"],
	setup: { agentview: "oblique", wrist_mount: "none", wrist_rotation: 0, wrist_flip: "none" },
	views: PANDA_PAIR_VIEWS,
	scene: {
		table: SCENE_TEXT.stock.table,
		object: SCENE_TEXT.stock.object,
		views: "the agentview alone (third-person, 256x256 with black padding bars: it looks at the table from in front, turned about 15 degrees; the left arm stands at the image left, the right arm at the image right; image right is +y, image bottom roughly +x). There are no wrist cameras",
	},
};

/** The move frame SYSTEM.md and `move_delta` state for a one-arm robot standing at the table's -x end. */
export const BASE_FRAME =
	"a base-frame `[dx, dy, dz]` in metres: +x away from the robot base, +y toward the robot's left, +z up";

/**
 * The BridgeData V2 twins' WidowX 250 S (env server ROBOTS["widowx250s"], SIMPLER's scenes): the scene's own
 * robot, controller and 3rd_view_camera (a real photo composited behind the objects), no wrist camera. Its base
 * stands at the table's +x end facing -x, so moves are world-frame: measured on the four scenes, each MV_* 19.5-
 * 20.3 mm per unit along the world axis (cos 1.000), and the camera shows +x toward the image bottom, +y to the
 * right. The mimic gripper closes from 74 mm (the fingers' joint sum) to 30 mm on nothing in 3-4 control steps
 * (5 Hz) and reads 50 mm closed on the 3 cm cube.
 */
export const WIDOWX250S_VIEWS = `Each result shows the third-person view (256x256, black bars are padding): the scene's own camera beside the robot, looking over the gripper at the table (a real photo behind the simulated objects); there is no wrist view. The gripper hangs down from the top of the image; MV_FWD moves it toward the image bottom (toward the camera), MV_BACK toward the image top, MV_LEFT / MV_RIGHT toward the image left / right.
- Judge the gripper against the target directly in this view, and descend in small steps: nearer objects appear lower in the image.`;
const WIDOWX250S: ManiskillRobot = {
	arm: "WidowX 250 S arm (a BridgeData V2 real-to-sim scene)",
	envs: [
		"PutCarrotOnPlateInScene-v1",
		"PutEggplantInBasketScene-v1",
		"StackGreenCubeOnYellowCubeBakedTexInScene-v1",
		"PutSpoonOnTableClothInScene-v1",
	],
	vectors: VECTORS,
	stepM: STEP_M,
	emptyWidthM: 0.031,
	frame: "a world-frame `[dx, dy, dz]` in metres: +x toward the camera and the robot base (the image bottom), +y toward the image right, +z up",
	setup: { agentview: "3rd_view_camera", wrist_mount: "none", wrist_rotation: 0, wrist_flip: "none" },
	views: WIDOWX250S_VIEWS,
	scene: {
		table: "The table top is at about z = 0.87 (the sink's floor at about z = 0.91 in the eggplant scene)",
		object: "a 3 cm block's centre is about 1.5 cm above the table",
		views: "the agentview alone (third-person, 256x256 with black padding bars: the scene's camera beside the robot looking over the gripper at the table; image bottom is +x, toward the camera, image right is +y). This robot has no wrist camera, so there is no wrist view",
	},
};

/** One `--robot`: the env server's ROBOTS row (services/.../maniskill/env_server.py) and what pi needs of it. */
export type ManiskillRobot = {
	/** How SYSTEM.md names it: "You control a <arm> in the ManiSkill simulator". */
	arm: string;
	/** The stock env ids it runs (reset, visibility gate and reach measured); the rigs run their own Panda only. */
	envs: readonly EnvId[];
	/** Base-frame MV_* vectors (measured: each unit moves ~stepM along its axis) and metres per unit. */
	vectors: Record<MoveUnit, Vec3>;
	stepM: number;
	/** The closed-and-empty gripper width, m. */
	emptyWidthM: number;
	/** `false`: no gripper (a stick): the action is the translation alone and `gripper` commands are refused. */
	gripper?: false;
	/** Two arms (the server's RobotSpec.arms): `move_delta` and `act` take `arm`; moves are world-frame. */
	arms?: readonly string[];
	/** The move frame a one-arm robot states (default BASE_FRAME): a scene whose base faces another way moves in the world frame. */
	frame?: string;
	/** The env server's camera setup (VIEW_SETUP); `wrist_mount: "none"`: no wrist camera, the agentview alone. */
	setup: { agentview: string; wrist_mount: string; wrist_rotation: number; wrist_flip: string };
	/** The units `views` text, and SYSTEM.md's scene sentences (SCENE_TEXT.stock with this robot's views and object). */
	views: string;
	scene: { table: string; object: string; views: string };
};

/** A robot without a wrist camera: SYSTEM.md's "both images" / "wrist image" become the one agentview. */
const ONE_VIEW = { images: "the image", grasp_view: "image" };
const TWO_VIEWS = { images: "both images", grasp_view: "wrist image" };

/**
 * `--robot`: the arms the env server's ROBOTS table drives in pd_ee_delta_pos (ManiSkill 3.0.1 agents with a
 * parallel gripper that the stock table scene places), measured on the box with the pi motion path (4 units
 * of each MV_* from the PickCube / StackCube seed 0 reset, a closed-loop servo per 2 cm waypoint). The same
 * ids in the same order as the server's table; panda is the default and keeps every existing constant.
 */
export const ROBOTS = {
	panda: {
		arm: "Franka Panda arm",
		envs: ENV_IDS.slice(2).filter((e) => !OTHER_ROBOT_ENVS[e]),
		vectors: VECTORS,
		stepM: STEP_M,
		emptyWidthM: EMPTY_WIDTH_M,
		setup: VIEW_SETUP,
		views: VIEWS,
		scene: SCENE_TEXT.stock,
	},
	xarm6_robotiq: XARM6_ROBOTIQ,
	widowxai: WIDOWXAI,
	panda_stick: PANDA_STICK,
	panda_pair: PANDA_PAIR,
	widowx250s: WIDOWX250S,
} satisfies Record<string, ManiskillRobot>;
export type RobotId = keyof typeof ROBOTS;
export const ROBOT_IDS = Object.keys(ROBOTS) as RobotId[];
/** Whether a robot's observations carry a wrist view. */
export const hasWrist = (r: ManiskillRobot) => r.setup.wrist_mount !== "none";
/** Whether a robot has a gripper (the stick has none). */
export const hasGripper = (r: ManiskillRobot) => r.gripper !== false;

/**
 * The robot an episode runs: `--robot` checked against ROBOTS and the env id (a rig runs its own Panda; a stock
 * scene must be one the robot was measured on). Throws with the choices otherwise.
 */
export function robotFor(robot: string, envId: string): ManiskillRobot {
	if (!Object.hasOwn(ROBOTS, robot)) throw new Error(`unknown --robot ${robot}; one of ${ROBOT_IDS.join(", ")}`);
	const spec: ManiskillRobot = ROBOTS[robot as RobotId];
	if (RIGS.includes(envId)) {
		if (robot !== "panda") throw new Error(`${envId} is a real2sim rig with its own Panda; --robot panda only`);
	} else if (!spec.envs.includes(envId as EnvId)) {
		const owner = OTHER_ROBOT_ENVS[envId as EnvId];
		throw new Error(
			`--robot ${robot} runs ${spec.envs.join(", ")}, not ${envId}${owner ? ` (${envId} runs on --robot ${owner})` : ""}`,
		);
	}
	return spec;
}

const round = (v: number, d = 4) => Number(v.toFixed(d));
/** The --robot values whose env server serves env.preview_reach from an IK model (env server IK_MODELS). */
export const IK_ROBOTS: readonly string[] = ["panda", "xarm6_robotiq"];

/**
 * The env action width of each `--robot` (services robots/maniskill/flywheel.py SPACES): pd_ee_delta_pos and
 * the gripper, the stick's translation alone, the WidowX 250 S's pose action, the pair's two arms.
 */
export const FLYWHEEL_ACTION: Record<RobotId, number> = {
	panda: 4,
	xarm6_robotiq: 4,
	widowxai: 4,
	panda_stick: 3,
	panda_pair: 8,
	widowx250s: 7,
};

/** One arm's TCP pose and finger opening. */
const armState = (a: Partial<ArmObs>): number[] => [
	...(a.tcp_pos?.toArray() ?? []),
	...(a.tcp_quat_wxyz?.toArray() ?? []),
	a.gripper_width ?? 0,
];

/**
 * The views, the TCP pose and finger opening (services robots/maniskill/flywheel.py; a robot without a wrist
 * camera has none); a two-arm robot's `arms`, in their order, one after the other.
 */
export const flyObs = (o: Obs, arms?: readonly string[]): FlywheelObs => ({
	images: { agentview_images: o.agentview, ...(o.wrist ? { wrist_images: o.wrist } : {}) },
	state: arms ? arms.flatMap((a) => armState(o.arms?.[a] ?? {})) : armState(o),
});

/** ManiSkill's grasp flag: `is_grasped` (PickCube), `is_cubeA_grasped` (StackCube), ... */
export const grasped = (info: Record<string, unknown>) =>
	Object.entries(info).some(([k, v]) => /^is_.*grasped$/.test(k) && Boolean(v));

/** Two HxWx3 uint8 images of the same height, side by side (the episode video's frame). */
export function sideBySide(a: NdArray, b: NdArray): NdArray {
	const [h, wa] = a.shape;
	const wb = b.shape[1];
	if (b.shape[0] !== h) throw new Error(`views differ in height: ${a.shape} vs ${b.shape}`);
	const out = Buffer.alloc(h * (wa + wb) * 3);
	for (let y = 0; y < h; y++) {
		a.data.copy(out, y * (wa + wb) * 3, y * wa * 3, (y + 1) * wa * 3);
		b.data.copy(out, (y * (wa + wb) + wa) * 3, y * wb * 3, (y + 1) * wb * 3);
	}
	return new NdArray("uint8", [h, wa + wb, 3], out);
}
/** One unit's probe: the TCP displacement after `n` units from the reset pose. */
export type Probe = { unit: MoveUnit; n: number; moved: Vec3 };
/** Units each MV_* is repeated from the reset pose (Show-Harness probe_move_axes: 8 control steps). */
export const PROBE_UNITS = 4;
/** Least cosine between a unit's commanded and measured direction; below it the axis is wrong. */
export const PROBE_MIN_COS = 0.95;

/**
 * Show-Harness `--probe-axes` (core/sim/maniskill_task.probe_move_axes), as a calibration: from each
 * unit's measured displacement, its metres per unit and direction cosine, and the vectors that make
 * one unit travel `stepM` (each commanded direction scaled by stepM / measured). A direction off by
 * more than acos(PROBE_MIN_COS) (a mirrored or swapped axis) is an error, not something to scale.
 */
export function calibrate(vectors: Record<MoveUnit, Vec3>, stepM: number, probes: Probe[]) {
	const units: Record<string, unknown> = {};
	const calibrated = { ...vectors };
	for (const { unit, n, moved } of probes) {
		const v = vectors[unit];
		const len = Math.hypot(...moved);
		const cos = len > 0 ? moved.reduce((a, m, k) => a + m * v[k], 0) / (len * Math.hypot(...v)) : 0;
		const perUnit = len / n;
		if (!(cos >= PROBE_MIN_COS))
			throw new Error(`${unit} moved ${JSON.stringify(moved)} for ${JSON.stringify(v)}: cos ${cos.toFixed(3)}`);
		calibrated[unit] = v.map((x) => (x * stepM) / perUnit) as Vec3;
		units[unit] = {
			commanded: v.map((x) => x * stepM * n),
			measured: moved.map((x) => Number(x.toFixed(4))),
			per_unit_m: Number(perUnit.toFixed(4)),
			cos: Number(cos.toFixed(4)),
			scale: Number((stepM / perUnit).toFixed(4)),
		};
	}
	return { units, vectors: calibrated };
}

export default function maniskill(pi: ExtensionAPI) {
	// Every flag this robot registers is tracked: numbers fail closed, the result records them (../../infra/params.ts).
	trackFlags(pi);
	const flag = (name: string, fallback: string) => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("env-id", {
		type: "string",
		default: "BlockPAP-v1",
		description: `ManiSkill env id: ${ENV_IDS.join(", ")} (BlockPAP-v1 / BlockStack-v1 are the RLinf rigs)`,
	});
	pi.registerFlag("scene", {
		type: "string",
		default: "",
		description:
			"RLinf rig options key=value,...: table_tex (006 wood | white | black | 001-021), cam_t (og | 0302 | 0303), traj_id (random | 0 | 15 | 25 | 40 | 45), layout (wide | none)",
	});
	pi.registerFlag("robot", {
		type: "string",
		default: "panda",
		description: `The arm: ${ROBOT_IDS.join(", ")} (the RLinf rigs run their own Panda)`,
	});
	pi.registerFlag("seed", { type: "string", default: "0", description: "Reset seed (the object layout)" });
	pi.registerFlag("env", { type: "string", description: "Attach to a running env server instead of starting one" });
	// --detections / --unidepth: detect, select_detection, reject_detection, enhance_depth (../primitives/detections.ts).
	// --ik: preview_reach over the env server's IK check (../ik.ts; the Panda and the xArm6).
	registerIkFlag(pi);
	registerDetectionFlags(pi, { sam3: true });
	// --point: Molmo's point over --molmo (../primitives/pointing.ts).
	registerPointFlags(pi);
	pi.registerFlag("probe-axes", {
		type: "boolean",
		default: false,
		description:
			"Before the episode, step each MV_* unit from the reset pose, write calibration.json next to the session and use the measured vectors (Show-Harness --probe-axes)",
	});
	pi.registerFlag("services", {
		type: "string",
		default: process.env.PI_EMBODIED_SERVICES ?? SERVICES,
		description: "pi-embodied services dir",
	});
	pi.registerFlag("python", {
		type: "string",
		default: process.env.PI_EMBODIED_PYTHON ?? "python",
		description: "Python for the env server (the maniskill venv)",
	});

	let env: RpcClient;
	let obs: Obs;
	let info: Info = {};
	let success = false;
	let everGrasped = false;
	let envStep = 0;
	/** pi's gripper command per arm ("" = the one arm): +1 open, -1 close; the server maps it to each robot's action. */
	let grippers: Record<string, number> = {};
	const gripOf = (a?: string) => grippers[a ?? ""] ?? 1;
	let language = "";
	/** The server runs an RLinf rig (BlockPAP-v1 / BlockStack-v1). */
	let rig = false;
	let tableZ = 0;
	/** The views' letterbox square (meta.view_size), for Flash's pixel -> raw pixel mapping. */
	let viewSize = 256;
	/** The --robot of this episode (ROBOTS), fixed at start. */
	let robotId: RobotId = "panda";
	const arm = (): ManiskillRobot => ROBOTS[robotId];
	const text = () => (rig ? SCENE_TEXT.rig : arm().scene);
	const wrist = () => rig || hasWrist(arm());
	const gripping = () => rig || hasGripper(arm());
	/** The --robot flag's arm, read at load (the `move_delta` schema, units' arms) before the robot starts. */
	const flagRobot = (): ManiskillRobot | undefined => ROBOTS[String(pi.getFlag("robot") ?? "panda") as RobotId];
	/** A two-arm robot's arm names (undefined: one arm). */
	const arms = () => (rig ? undefined : arm().arms);
	/** One arm's proprioception (`a` undefined: the one arm). */
	const armObs = (a?: string): ArmObs => (a ? obs.arms![a] : (obs as ArmObs));
	/** The scene's raw-path part: its options as a tag, or `default`. */
	const sceneTag = () => (robot.task.scene ? tagPart(robot.task.scene) : "default");
	/** Every control step: the robot's views, its TCP state and the pd_ee_delta_pos action; one space per arm. */
	const FLYWHEEL: FlywheelSpec = {
		robot: "maniskill",
		get space() {
			return robotId;
		},
		get images(): FlywheelSpec["images"] {
			return wrist() ? { agentview_images: null, wrist_images: null } : { agentview_images: null };
		},
		/** Each arm's TCP pose and finger opening. */
		get state() {
			return 8 * (arms()?.length ?? 1);
		},
		get action() {
			return FLYWHEEL_ACTION[robotId];
		},
	};
	/** raw/maniskill/<robot>/<env-id>/<scene>/seed_NNN (services robots/maniskill/flywheel.py). */
	const flyMeta = () => ({
		path: [robotId, robot.task["env-id"], sceneTag(), `seed_${robot.task.seed.padStart(3, "0")}`],
		metadata: {
			maniskill_robot: robotId,
			env_id: robot.task["env-id"],
			scene: robot.task.scene ?? "",
			seed: Number(robot.task.seed),
			task_language: language,
		},
	});
	/** The --probe-axes vectors (undefined: VECTORS) and their calibration record. */
	let vectors: Record<MoveUnit, Vec3> | undefined;
	let calibration: Record<string, unknown> | undefined;

	// Another arm's memory is its own cell: `maniskill_<robot>_<env-id>...` (the Panda keeps `maniskill_<env-id>...`).
	const tag = (seed: string) =>
		`maniskill_${robotId === "panda" ? "" : `${robotId}_`}${tagPart(robot.task["env-id"])}${robot.task.scene ? `_${tagPart(robot.task.scene)}` : ""}_s${seed}`;
	const robot = defineRobot(pi, {
		name: "maniskill",
		// Tools and code primitives: ../../primitives/manifests/maniskill.json (the env server reads it too).
		manifest: "maniskill",
		// Read at load and at session start (from --robot, like the move frame the description states).
		vars: () => {
			const r = flagRobot();
			return {
				max_move: MAX_MOVE_M,
				cameras: ["agentview", "wrist"],
				// `arm` exists on a two-arm robot only (one-arm robots' tools and primitives leave it out).
				arms: r?.arms ? [...r.arms] : [],
				move_subject: r?.arms
					? `ONE arm's gripper (\`arm\`: ${r.arms.join(" or ")}; the other holds still)`
					: "the gripper",
				move_frame: r?.arms
					? "a world-frame [dx, dy, dz] in metres (+x toward the camera, +y toward the right arm's base, +z up)"
					: r?.frame
						? r.frame.replaceAll("`", "")
						: "a base-frame [dx, dy, dz] in metres (+x away from the base, +y toward the robot's left, +z up)",
			};
		},
		// Must agree with the env server's `_has` (code mode refuses a server whose code.api differs).
		capabilities: (c) =>
			({
				sam3: pi.getFlag("detections") === true && Boolean(flag("sam3", "")),
				unidepth: Boolean(String(pi.getFlag("unidepth") ?? "").trim()),
				// env.preview_reach answers from an IK model the Panda and the xArm6 have (env server IK_MODELS).
				ik: Boolean(String(pi.getFlag("ik") ?? "").trim()) && IK_ROBOTS.includes(flag("robot", "panda")),
			})[c] ?? false,
		services: { models: [SAM3, MOLMO] },
		task: ["env-id", "seed", "scene"],
		// The env server's primitive registry (code.api), recorded per episode.
		codeApi: () => env,
		// Code mode (../code): the env server runs the program against that registry; the result
		// carries the control steps, the success latched since the reset, the new observation and the video frames.
		code: {
			rpc: () => env,
			instruction: () => language,
			refuse: () => (success ? "the task is already solved; call finish" : undefined),
			observe: async (r) => {
				for (const f of (r.frames as NdArray[] | undefined) ?? []) video.frame(f);
				const steps = Number(r.steps) || 0;
				envStep += steps;
				if (!arms() && (r.gripper === 1 || r.gripper === -1)) grippers[""] = r.gripper;
				success ||= r.success === true;
				if (r.obs) absorb(r.obs as Obs, (r.info as Info | undefined) ?? {});
				return observe({ name: "run_code", status: r.status, env_steps: steps });
			},
		},
		keepImages: 4,
		video: true,
		flywheel: { spec: FLYWHEEL, select: () => `${robotId}/${robot.task["env-id"]}/${sceneTag()}` },
		// Observations carry the agentview then the wrist image (the agentview alone on a robot without one).
		vdm: () => {
			const r = ROBOTS[String(pi.getFlag("robot") ?? "panda") as RobotId];
			return r && !hasWrist(r) ? { views: 1 } : { views: 2, wrist: 1 };
		},
		groundTruth: (names) => call("env.ground_truth_poses", { names: names ?? null }),
		// No corpus is published for ManiSkill: memory is what exploration writes locally.
		memory: {
			cell: () => ({ tag: tag(robot.task.seed), reference: tag("0") }),
			primitives: ["move_delta", "act"],
			published: false,
		},
		// Flash replays a solved episode's plan (../flash/generate.ts --session: move_delta waypoints with their
		// absolute end positions). Anchors are pointed at by Molmo in the agentview and met with the plane at
		// their recorded height (else table_z + HALF_HEIGHT_M) through the calibrated camera (no depth here),
		// after undoing the view's letterbox; anchored waypoints then move
		// with them, as deltas from the live TCP position, in moves of at most MAX_MOVE_M.
		flash: recipeFlash(pi, {
			names: () => [tag(robot.task.seed), tag("0")],
			memory: () => robot.mem?.render("{{memory_dir}}") ?? "",
			observe: "view_env_state",
			targets: {
				move_delta: {
					delta: "delta_xyz",
					position: (json) => (json.state as { tcp_pos?: number[] } | undefined)?.tcp_pos,
					maxStep: MAX_MOVE_M,
				},
			},
			// The agentview is a fixed calibrated camera: anchors re-localize against the plan's recorded view.
			fixedCamera: true,
			// The pointed image's own square (the recorded view's as recorded), else this server's view_size.
			backProject: async (_fr, pixel, anchor, size) => {
				const z = anchorPlane(anchor.xyz, tableZ);
				if (z === undefined) return undefined;
				const meta = await call<CameraMeta>("env.get_camera_meta", { camera_name: "agentview" });
				return pixelOnPlane(meta, unletterbox(pixel, { ...AGENTVIEW_PX, size: size?.[0] ?? viewSize }), z);
			},
			over: (latest) => latest.json.success === true,
			solved: (latest) => latest.json.success === true,
		}),
		explore: {
			// The exploration `reset` tool is not a robot.tool, so robot.signal is unset here; use its own signal.
			reset: async (result, _ctx, signal) => {
				const [o, i] = await env.call<[Obs, Info]>("env.reset", {}, 300_000, [], signal);
				success = everGrasped = false;
				envStep = 0;
				grippers = {};
				absorb(o, i);
				fly.reset(flyObs(o, arms()), flyMeta());
				return observe({ ...result, reset: true });
			},
			prompt: () =>
				EXPLORE.replaceAll("study both images", wrist() ? "study both images" : "study the image")
					.replaceAll("{{env_id}}", robot.task["env-id"])
					.replaceAll("{{seed}}", robot.task.seed)
					.replaceAll("{{scene}}", robot.task.scene || "stock"),
			rewrite: [
				[
					/This is a single episode\. You may recover within it \(re-position, re-grasp\), but you cannot restart it\./,
					"This is an exploration run: `reset` starts a fresh attempt (see Exploration). Within an attempt, recover in place (re-position, re-grasp).",
				],
			],
		},
		start: startEpisode,
		prompt: () =>
			(
				[
					["gripper", gripping()],
					["stick", !gripping()],
					["one_arm", !arms()],
					["two_arms", Boolean(arms())],
				] as [string, boolean][]
			)
				.reduce((t, [name, on]) => section(t, name, on), SYSTEM)
				.replaceAll("{{task_language}}", language)
				.replaceAll("{{frame}}", (rig ? undefined : arm().frame) ?? BASE_FRAME)
				.replaceAll("{{arm}}", rig ? ROBOTS.panda.arm : arm().arm)
				.replaceAll("{{images}}", (wrist() ? TWO_VIEWS : ONE_VIEW).images)
				.replaceAll("{{grasp_view}}", (wrist() ? TWO_VIEWS : ONE_VIEW).grasp_view)
				.replaceAll("{{table}}", text().table)
				.replaceAll("{{object}}", text().object)
				.replaceAll("{{views}}", text().views),
		result: () => ({
			env_id: robot.task["env-id"],
			seed: Number(robot.task.seed),
			...(robot.task.scene ? { scene: robot.task.scene } : {}),
			// Another arm than the Panda (--robot); `robot` itself names this pi robot, "maniskill".
			...(robotId !== "panda" ? { maniskill_robot: robotId } : {}),
			...(calibration ? { calibration } : {}),
			success,
			ever_grasped: everGrasped,
			env_steps: envStep,
		}),
		status: () => ({ language, step: envStep, solved: success }),
		finish: {
			description:
				"End the episode after checking the latest state. Success is ManiSkill's success flag, not this call.",
			parameters: Type.Object({
				status: StringEnum(["success", "failure"] as const),
				summary: Type.String(),
			}),
			result: (params) => ({
				content: [{ type: "text", text: `Episode finished (success=${success}).` }],
				details: params,
			}),
		},
		units: {
			get vectors() {
				return vectors ?? arm().vectors;
			},
			get stepM() {
				return arm().stepM;
			},
			instruction: () => language,
			get views() {
				return rig ? RIG_VIEWS : arm().views;
			},
			get emptyWidthM() {
				return arm().emptyWidthM;
			},
			// From --robot (as `vdm`), so it holds before the robot starts: the WidowX AI has no wrist camera.
			wrist: () => {
				const r = ROBOTS[String(pi.getFlag("robot") ?? "panda") as RobotId];
				return !r || hasWrist(r);
			},
			// The same for the gripper: the stick has none (no GRASP / RELEASE units).
			gripper: () => {
				const r = flagRobot();
				return !r || hasGripper(r);
			},
			// Two arms (panda_pair): `act` takes `arm`, read at load like the wrist view.
			get arms() {
				return flagRobot()?.arms;
			},
			apply: async (m, signal) => {
				if (m.yaw) throw new Error("this robot has no yaw (pd_ee_delta_pos holds the orientation)");
				return observe(
					await move(
						{
							delta_xyz: m.delta,
							...(m.gripper ? { gripper: m.gripper } : {}),
							...(m.arm ? { arm: m.arm } : {}),
						},
						signal,
					),
				);
			},
			state: async (a) => ({
				eef_xyz: armObs(a)
					.tcp_pos!.toArray()
					.map((v) => round(v)),
				...(gripping() ? { gripper_width: round(armObs(a).gripper_width!) } : {}),
				table_z: tableZ,
				...(gripping() ? { is_grasped: grasped(info) } : {}),
			}),
		},
	});
	const { video } = robot;
	const fly = robot.fly!;

	const call = <T = unknown>(
		method: string,
		kwargs: Record<string, unknown> = {},
		args: unknown[] = [],
		signal = robot.signal,
	) => env.call<T>(method, kwargs, 120_000, args, signal);

	function absorb(o: Obs, i: Info) {
		obs = o;
		info = i;
		success ||= Boolean(i.success);
		everGrasped ||= grasped(i);
	}

	/**
	 * Run one move on the env server (env.move_delta: the ~2 cm waypoints, the gripper hold, the per-call cap,
	 * stop and the latched success live there); every control step goes to the video and the Flywheel.
	 * A two-arm robot moves `arm` (world frame), the other arm holding still.
	 */
	async function move(params: Json, signal: AbortSignal | undefined) {
		const r = await env.call<MoveReturn>("env.move_delta", params, 300_000, [], signal);
		for (const f of r.frames) {
			video.frame(f.wrist ? sideBySide(f.agentview, f.wrist) : f.agentview);
			// Flywheel: every control step the servo ran (a leg that ran none returns no action).
			if (fly.recording && f.action)
				fly.transition(f.action.toArray(), flyObs(f, arms()), f.success ? 1 : 0, Boolean(f.success), false);
		}
		envStep += Number(r.result.env_steps) || 0;
		absorb(r.obs, r.info);
		if (r.result.gripper === "open" || r.result.gripper === "close")
			grippers[(params.arm as string | undefined) ?? ""] = r.result.gripper === "open" ? 1 : -1;
		return r.result;
	}

	/** The motion result with the new state, then the agentview and (a robot with one) the wrist image. */
	function observe(result: Record<string, unknown>) {
		const views = obs.wrist ? [obs.agentview, obs.wrist] : [obs.agentview];
		const images = views.map((a) => encodePng(a.data, a.shape[1], a.shape[0]));
		const details = {
			result,
			step: envStep,
			success,
			terminated: success,
			task_language: language,
			state: arms()
				? {
						arms: Object.fromEntries(
							arms()!.map((a) => [
								a,
								{
									tcp_pos: armObs(a)
										.tcp_pos!.toArray()
										.map((v) => round(v)),
									gripper_width: round(armObs(a).gripper_width!),
									gripper_command: gripOf(a) > 0 ? "open" : "close",
								},
							]),
						),
						is_grasped: grasped(info),
					}
				: {
						tcp_pos: obs.tcp_pos!.toArray().map((v) => round(v)),
						// A stick has no gripper to report.
						...(gripping()
							? {
									gripper_width: round(obs.gripper_width!),
									gripper_command: gripOf() > 0 ? "open" : "close",
									is_grasped: grasped(info),
								}
							: {}),
					},
			images: [
				`agentview ${obs.agentview.shape[1]}x${obs.agentview.shape[0]}`,
				...(obs.wrist ? [`wrist ${obs.wrist.shape[1]}x${obs.wrist.shape[0]}`] : []),
			],
		};
		return {
			content: [
				{ type: "text" as const, text: JSON.stringify(details) },
				...images.map((png) => ({ type: "image" as const, data: png.toString("base64"), mimeType: "image/png" })),
			],
			details,
		};
	}

	robot.primitive("view_env_state", async () => observe({}));

	// The manifest's parameters go to the server as they are (manifests/maniskill.json: delta_xyz, gripper, arm).
	robot.primitive("move_delta", async (params, signal) => observe(await move(params as Json, signal)));

	/** Show-Harness probe_move_axes: each unit PROBE_UNITS times from a fresh reset, no video. */
	async function probeAxes(ctx: ExtensionContext) {
		const probes: Probe[] = [];
		const r = arm();
		if (arms()) throw new Error("--probe-axes measures one arm; not on a two-arm --robot");
		for (const unit of MOVE_UNITS) {
			const [o] = await env.call<[Obs, Info]>("env.reset", {}, 300_000);
			const start = o.tcp_pos!.toArray();
			const delta = r.vectors[unit].map((x) => x * r.stepM * PROBE_UNITS) as Vec3;
			// The move's ~2 cm waypoints on the server, the gripper held open from the reset, no video.
			const m = await env.call<MoveReturn>("env.move_delta", { delta_xyz: delta }, 300_000);
			const end = m.obs.tcp_pos!.toArray();
			probes.push({ unit, n: PROBE_UNITS, moved: end.map((v, k) => v - start[k]) as Vec3 });
		}
		const c = calibrate(r.vectors, r.stepM, probes);
		vectors = c.vectors;
		calibration = {
			env_id: robot.task["env-id"],
			seed: Number(robot.task.seed),
			...(robotId !== "panda" ? { robot: robotId } : {}),
			step_m: r.stepM,
			units: c.units,
		};
		const file = ctx.sessionManager.getSessionFile();
		if (file) writeFileSync(join(dirname(file), "calibration.json"), `${JSON.stringify(calibration, null, 2)}\n`);
	}

	// Molmo pointing on the current images (active with --point).
	mountGraspTool(
		robot.tool,
		pointTool(pi, {
			cameras: ["agentview", "wrist"],
			frame: async (c) => {
				const a = c === "wrist" ? obs.wrist : obs.agentview;
				if (!a) throw new Error(`${arm().arm} has no ${c} camera`);
				return rgbOf(a);
			},
			signal: () => robot.signal,
		}),
	);

	// --ik: the env server's IK check (xyz, quat_xyzw as the manifest declares them).
	robot.primitive("preview_reach", async (params) =>
		toolResult((await env.call<Reach>("env.preview_reach", params as Json, 60_000, [], robot.signal)) as Json),
	);

	// SAM3 masks with ids and UniDepth over the env server's perception (active with --detections / --unidepth).
	for (const d of detectionTools(pi, {
		call: (method, kwargs, timeoutMs) =>
			env.call<Record<string, any>>(method, kwargs, timeoutMs ?? 120_000, [], robot.signal),
		cameras: ["agentview", "wrist"],
	}))
		mountGraspTool(robot.tool, d);

	async function startEpisode(ctx: ExtensionContext) {
		const envId = robot.task["env-id"];
		if (!(ENV_IDS as readonly string[]).includes(envId))
			throw new Error(`unknown --env-id ${envId}; one of ${ENV_IDS.join(", ")}`);
		const seed = robot.task.seed;
		const scene = robot.task.scene ?? "";
		const wanted = flag("robot", "panda");
		robotFor(wanted, envId);
		robotId = wanted as RobotId;
		const endpoint = pi.getFlag("env") as string | undefined;
		if (endpoint) env = await attach(endpoint);
		else {
			const services = flag("services", SERVICES);
			env = await robot.serve({
				python: flag("python", "python"),
				args: [
					"-m",
					"pi_embodied_services.robots.maniskill.env_server",
					"--env-id",
					envId,
					"--seed",
					seed,
					...(scene ? ["--scene", scene] : []),
					...(robotId !== "panda" ? ["--robot", robotId] : []),
					...ikArgs(pi.getFlag("ik")),
					...detectionArgs(pi, flag("sam3", "")),
				],
				cwd: services,
				env: { ...process.env, PYTHONPATH: services },
				log: (port) => join(tmpdir(), `pi-embodied-maniskill-${envId}-s${seed}-${port}.log`),
				readyMs: 600_000,
			});
		}
		const meta = await env.call<Meta>("env.get_env_meta");
		if (meta.env_id !== envId || meta.seed !== Number(seed))
			throw new Error(`env server runs ${meta.env_id} seed ${meta.seed}, not ${envId} seed ${seed}`);
		if ((meta.robot ?? "panda") !== robotId)
			throw new Error(`env server runs --robot ${meta.robot ?? "panda"}, not ${robotId}`);
		rig = Boolean(meta.scene);
		tableZ = meta.table_z ?? 0;
		viewSize = meta.view_size ?? 256;
		const want = rig ? RIG_VIEW_SETUP : arm().setup;
		const setup = Object.entries(want).filter(([k, v]) => meta[k as keyof typeof VIEW_SETUP] !== v);
		if (setup.length)
			throw new Error(
				`env server cameras (${setup.map(([k]) => `${k}=${meta[k as keyof typeof VIEW_SETUP]}`).join(", ")}) differ from the ones the prompt describes (${JSON.stringify(want)}); update the services dir`,
			);
		success = everGrasped = false;
		envStep = 0;
		grippers = {};
		vectors = calibration = undefined;
		if (pi.getFlag("probe-axes") === true) await probeAxes(ctx);
		const [o, i] = await env.call<[Obs, Info]>("env.reset", {}, 300_000);
		absorb(o, i);
		language = await env.call<string>("env.get_task_language");
		fly.reset(flyObs(o, arms()), flyMeta());
		return [
			"view_env_state",
			"move_delta",
			"finish",
			...(String(pi.getFlag("ik") ?? "").trim() ? ["preview_reach"] : []),
			...detectionActive(pi, meta.capabilities?.perception),
			...pointActive(pi),
		];
	}
}
