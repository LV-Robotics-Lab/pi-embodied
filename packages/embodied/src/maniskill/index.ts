/**
 * ManiSkill robot for pi: a Panda in ManiSkill 3's translation-only `pd_ee_delta_pos` mode.
 *
 *   pi -e packages/embodied/src/maniskill --seed 0                    (BlockPAP-v1, Show-Harness's default scene)
 *   pi -e packages/embodied/src/maniskill --units --scene table_tex=white,cam_t=0302 --seed 3
 *   pi -e packages/embodied/src/maniskill --env-id StackCube-v1 --seed 0   (a stock ManiSkill scene)
 *   pi -e packages/embodied/src/maniskill --env-id PlaceSphere-v1 --seed 0  (one of OpenETA's tasks, ENV_IDS)
 *   pi -e packages/embodied/src/maniskill --robot xarm6_robotiq --env-id PickCube-v1 --seed 0  (another arm, ROBOTS)
 *
 * BlockPAP-v1 / BlockStack-v1 are RLinf's real2sim replicas of the real Franka rig (services/.../
 * robots/maniskill/scenes.py; fetch_real2sim.sh installs them): their calibrated front RealSense
 * and Show-Harness's wrist transform, reset like the training episodes, so a seed reproduces the
 * Show-Harness-Data frames. `--scene` passes the rig's options (table_tex, cam_t og|0302|0303,
 * traj_id, layout wide|none).
 *
 * Starts one ManiSkill env server per session (services/.../robots/maniskill/env_server.py, the
 * `maniskill` venv; rendering needs a GPU). `--robot` picks the arm (ROBOTS: the Panda by default, an xArm6
 * with a Robotiq gripper, a WidowX AI without a wrist camera); the rigs run their own Panda. `move_delta` and the units hook `apply` share one
 * motion path: a base-frame delta in metres becomes ~2 cm decisions, each a closed-loop servo of
 * 2-8 control steps to its waypoint (Show-Harness's step calibration and real2sim execution);
 * every result carries the agentview and wrist images and the state; success is ManiSkill's own `success` flag, recorded in `robot_result`.
 *
 * Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: interpreters/maniskill_atomic_controller.py, core/sim/maniskill_scenes.py
 * and configs/robot_maniskill.yaml ported as a pi robot.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { anchorPlane, type CameraMeta, pixelOnPlane, unletterbox } from "../flash/plane.ts";
import { recipeFlash } from "../flash/recipe.ts";
import { encodePng } from "../png.ts";
import { attach, defineRobot, SERVICES } from "../robot.ts";
import { NdArray, type RpcClient } from "../rpc.ts";
import { MOVE_UNITS, type MoveUnit, type Vec3 } from "../units/index.ts";

const read = (name: string) => readFileSync(new URL(name, import.meta.url), "utf8");
const SYSTEM = read("./SYSTEM.md");
const MEMORY = read("./memory.md");
const EXPLORE = read("./explore.md");
/** A memory cell tag part: the scene options (`table_tex=white`) with the characters a tag may not hold replaced. */
const tagPart = (s: string) => s.replace(/[^\w.-]+/g, "-");

/**
 * `--env-id` takes one of these: the RLinf rigs (BlockPAP-v1 default, BlockStack-v1) and the stock
 * ManiSkill tabletop tasks the env server has a task text and a visibility list for (its INSTRUCTIONS
 * / TASK_ACTORS; the last six are OpenETA's ManiSkill table, sim/envs/maniskill at 7d4a0a1, minus
 * what a one-arm translation-only Panda cannot attempt). The same list, in the same order.
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
] as const;
export type EnvId = (typeof ENV_IDS)[number];
/** The RLinf rigs among ENV_IDS: they fix their own robot (a Panda). */
const RIGS: readonly string[] = ENV_IDS.slice(0, 2);

/** configs/robot_maniskill.yaml `move_vectors`: +x away from the base, -y = MV_LEFT, +z up. */
export const VECTORS: Record<MoveUnit, Vec3> = {
	MV_FWD: [1, 0, 0],
	MV_BACK: [-1, 0, 0],
	MV_LEFT: [0, -1, 0],
	MV_RIGHT: [0, 1, 0],
	MV_UP: [0, 0, 1],
	MV_DOWN: [0, 0, -1],
};
/** Physical metres per decision (the MVTOKEN 2 cm convention). */
export const STEP_M = 0.02;
/**
 * Servo gain, command per metre of remaining error: the yaml's step_m 0.026 per 2 cm decision
 * (0.026 x 2 control steps measured 20.2 mm on BlockPAP, 20.6 mm on PickCube from rest).
 */
export const GAIN = 0.026 / 0.02;
/**
 * Control steps per decision: at least the yaml's `sim_steps_per_decision` (2), and up to 8 until
 * the TCP is within 2 mm of the waypoint. Open-loop 2-step decisions fell short after a reversal
 * (measured on PickCube: MV_RIGHT right after MV_LEFT moved 0.6 of 2 cm, MV_UP after MV_DOWN 0.9).
 */
export const SERVO = { minSteps: 2, maxSteps: 8, tolM: 0.002 };
/** Control steps a gripper toggle holds still: closing takes 3 steps, opening ~6 (measured). */
export const GRIPPER_STEPS = 6;
/** Largest translation one call may command, m. */
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

type Obs = {
	agentview: NdArray;
	/** Absent on a robot without a wrist camera (ManiskillRobot.setup.wrist_mount "none"). */
	wrist?: NdArray;
	tcp_pos: NdArray;
	tcp_quat_wxyz: NdArray;
	gripper_width: number;
	qpos: NdArray;
};
type Info = Record<string, unknown>;
type ServoReturn = [Obs[], Info];
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
 * and opens in 6 (GRIPPER_STEPS); closed on a 4 cm cube it reads ~50 mm (pad links), on nothing 0. Its wrist
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
		"PullCubeTool-v1",
		"PlugCharger-v1",
	],
	vectors: VECTORS,
	stepM: STEP_M,
	gain: GAIN,
	gripperSteps: GRIPPER_STEPS,
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
	envs: ["PickCube-v1"],
	vectors: VECTORS,
	stepM: STEP_M,
	gain: GAIN,
	gripperSteps: 10,
	emptyWidthM: EMPTY_WIDTH_M,
	setup: { agentview: "oblique", wrist_mount: "none", wrist_rotation: 0, wrist_flip: "none" },
	views: WIDOWXAI_VIEWS,
	scene: {
		table: SCENE_TEXT.stock.table,
		object: "a 3.6 cm cube's centre is at z = 0.018",
		views: "the agentview alone (third-person, 256x256 with black padding bars: it looks at the robot from in front of the table, turned about 15 degrees toward the robot's left; the robot base is at the top, image right is roughly +y, image bottom roughly +x). This robot has no wrist camera, so there is no wrist view",
	},
};

/** One `--robot`: the env server's ROBOTS row (services/.../maniskill/env_server.py) and what pi needs of it. */
export type ManiskillRobot = {
	/** How SYSTEM.md names it: "You control a <arm> in the ManiSkill simulator". */
	arm: string;
	/** The stock env ids it runs (reset, visibility gate and reach measured); the rigs run their own Panda only. */
	envs: readonly EnvId[];
	/** Base-frame MV_* vectors (measured: each unit moves ~stepM along its axis), metres per unit and the servo gain. */
	vectors: Record<MoveUnit, Vec3>;
	stepM: number;
	gain: number;
	/** Control steps a gripper toggle holds still; the closed-and-empty gripper width, m. */
	gripperSteps: number;
	emptyWidthM: number;
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
		envs: ENV_IDS.slice(2),
		vectors: VECTORS,
		stepM: STEP_M,
		gain: GAIN,
		gripperSteps: GRIPPER_STEPS,
		emptyWidthM: EMPTY_WIDTH_M,
		setup: VIEW_SETUP,
		views: VIEWS,
		scene: SCENE_TEXT.stock,
	},
	xarm6_robotiq: XARM6_ROBOTIQ,
	widowxai: WIDOWXAI,
} satisfies Record<string, ManiskillRobot>;
export type RobotId = keyof typeof ROBOTS;
export const ROBOT_IDS = Object.keys(ROBOTS) as RobotId[];
/** Whether a robot's observations carry a wrist view. */
export const hasWrist = (r: ManiskillRobot) => r.setup.wrist_mount !== "none";

/**
 * The robot an episode runs: `--robot` checked against ROBOTS and the env id (a rig runs its own Panda; a stock
 * scene must be one the robot was measured on). Throws with the choices otherwise.
 */
export function robotFor(robot: string, envId: string): ManiskillRobot {
	if (!Object.hasOwn(ROBOTS, robot)) throw new Error(`unknown --robot ${robot}; one of ${ROBOT_IDS.join(", ")}`);
	const spec: ManiskillRobot = ROBOTS[robot as RobotId];
	if (RIGS.includes(envId)) {
		if (robot !== "panda") throw new Error(`${envId} is a real2sim rig with its own Panda; --robot panda only`);
	} else if (!spec.envs.includes(envId as EnvId))
		throw new Error(`--robot ${robot} runs ${spec.envs.join(", ")}, not ${envId}`);
	return spec;
}

const round = (v: number, d = 4) => Number(v.toFixed(d));

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
/**
 * The waypoints of one base-frame move: one per ~2 cm decision, ceil(|delta| / STEP_M) of them,
 * evenly spaced from `start` (a pure gripper command or STOP is one waypoint at `start`).
 */
export function waypoints(start: number[], delta: Vec3, stepM = STEP_M): number[][] {
	const n = Math.max(1, Math.ceil(Math.hypot(...delta) / stepM - 1e-9));
	return Array.from({ length: n }, (_, i) => start.map((p, k) => p + (delta[k] * (i + 1)) / n));
}

/** One closed-loop servo call: drive to `target` in `minSteps`..`maxSteps` control steps. */
export type Phase = { target: number[]; minSteps: number; maxSteps: number };
/**
 * The servo phases of one call. A gripper change is its own phase first, holding still at `start`
 * for GRIPPER_STEPS until the fingers settle (Show-Harness's GRASP / RELEASE are separate tokens:
 * the fingers never close while the arm travels); then the ~2 cm waypoints of the move. A call
 * that neither moves nor changes the gripper (STOP) holds for one decision. `gripperSteps` / `stepM`: the robot's.
 */
export function phases(
	start: number[],
	delta: Vec3,
	gripperChanged: boolean,
	gripperSteps = GRIPPER_STEPS,
	stepM = STEP_M,
): Phase[] {
	const hold = (steps: number): Phase => ({ target: start, minSteps: steps, maxSteps: steps });
	const out: Phase[] = gripperChanged ? [hold(gripperSteps)] : [];
	if (Math.hypot(...delta) > 0)
		out.push(
			...waypoints(start, delta, stepM).map((target) => ({
				target,
				minSteps: SERVO.minSteps,
				maxSteps: SERVO.maxSteps,
			})),
		);
	return out.length ? out : [hold(SERVO.minSteps)];
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
	/** Panda mimic gripper command: +1 open, -1 close. */
	let gripper = 1;
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
	/** The --probe-axes vectors (undefined: VECTORS) and their calibration record. */
	let vectors: Record<MoveUnit, Vec3> | undefined;
	let calibration: Record<string, unknown> | undefined;

	// Another arm's memory is its own cell: `maniskill_<robot>_<env-id>...` (the Panda keeps `maniskill_<env-id>...`).
	const tag = (seed: string) =>
		`maniskill_${robotId === "panda" ? "" : `${robotId}_`}${tagPart(robot.task["env-id"])}${robot.task.scene ? `_${tagPart(robot.task.scene)}` : ""}_s${seed}`;
	const robot = defineRobot(pi, {
		name: "maniskill",
		task: ["env-id", "seed", "scene"],
		// The env server's primitive registry (code.api), recorded per episode.
		codeApi: () => env,
		keepImages: 4,
		video: true,
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
			backProject: async (_fr, pixel, anchor) => {
				const z = anchorPlane(anchor.xyz, tableZ);
				if (z === undefined) return undefined;
				const meta = await call<CameraMeta>("env.get_camera_meta", { camera_name: "agentview" });
				return pixelOnPlane(meta, unletterbox(pixel, { ...AGENTVIEW_PX, size: viewSize }), z);
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
				gripper = 1;
				absorb(o, i);
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
			SYSTEM.replaceAll("{{task_language}}", language)
				.replaceAll("{{arm}}", rig ? ROBOTS.panda.arm : arm().arm)
				.replaceAll("{{images}}", (wrist() ? TWO_VIEWS : ONE_VIEW).images)
				.replaceAll("{{grasp_view}}", (wrist() ? TWO_VIEWS : ONE_VIEW).grasp_view)
				.replaceAll("{{table}}", text().table)
				.replaceAll("{{object}}", text().object)
				.replaceAll("{{views}}", text().views)
				.replaceAll("{{memory}}", pi.getFlag("explore") === true ? "" : robot.mem!.render(MEMORY).trim()),
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
			apply: async (m, signal) => {
				if (m.yaw) throw new Error("this robot has no yaw (pd_ee_delta_pos holds the orientation)");
				return observe(await move(m.delta, m.gripper, signal));
			},
			state: async () => ({
				eef_xyz: obs.tcp_pos.toArray().map((v) => round(v)),
				gripper_width: round(obs.gripper_width),
				table_z: tableZ,
				is_grasped: grasped(info),
			}),
		},
	});
	const { video } = robot;

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

	/** Run one base-frame move (m) with an optional gripper command; every control step goes to the video. */
	async function move(delta: Vec3, grip: "open" | "close" | null, signal: AbortSignal | undefined) {
		const norm = Math.hypot(...delta);
		if (!(norm <= MAX_MOVE_M))
			throw new Error(`delta moves ${round(norm)} m; the limit is ${MAX_MOVE_M} m per call. Split the motion.`);
		const before = gripper;
		if (grip) gripper = grip === "open" ? 1 : -1;
		const start = obs.tcp_pos.toArray();
		let steps = 0;
		const r = arm();
		for (const { target, minSteps, maxSteps } of phases(start, delta, gripper !== before, r.gripperSteps, r.stepM)) {
			if (success) break;
			const [frames, i] = await call<ServoReturn>(
				"env.servo",
				{ gain: r.gain, tol_m: SERVO.tolM, min_steps: minSteps, max_steps: maxSteps },
				[target, gripper],
				signal,
			);
			for (const f of frames) video.frame(f.wrist ? sideBySide(f.agentview, f.wrist) : f.agentview);
			steps += frames.length;
			envStep += frames.length;
			absorb(frames[frames.length - 1], i);
			if (i.cancelled) break;
		}
		const end = obs.tcp_pos.toArray();
		return {
			commanded_m: delta.map((v) => round(v)),
			moved_m: end.map((v, k) => round(v - start[k])),
			gripper: gripper > 0 ? "open" : "close",
			env_steps: steps,
		};
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
			state: {
				tcp_pos: obs.tcp_pos.toArray().map((v) => round(v)),
				gripper_width: round(obs.gripper_width),
				gripper_command: gripper > 0 ? "open" : "close",
				is_grasped: grasped(info),
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

	robot.tool(
		"view_env_state",
		"Current state with the agentview (third-person) and wrist images.",
		Type.Object({}),
		async () => observe({}),
	);

	const xyz = Type.Array(Type.Number(), { minItems: 3, maxItems: 3 });
	robot.tool(
		"move_delta",
		`Translate the gripper by a base-frame [dx, dy, dz] in metres (+x away from the base, +y toward the robot's left, +z up; at most ${MAX_MOVE_M} m per call), optionally opening or closing the gripper first (the arm holds still until the fingers settle, then moves). The orientation is locked. Returns the new state and images.`,
		Type.Object({
			delta_xyz: xyz,
			gripper: Type.Optional(StringEnum(["open", "close"] as const)),
		}),
		async ({ delta_xyz, gripper: g }, signal) => {
			if (success) return observe({ error: "the task is already solved; call finish" });
			return observe(await move(delta_xyz as Vec3, g ?? null, signal));
		},
	);

	/** Show-Harness probe_move_axes: each unit PROBE_UNITS times from a fresh reset, no video. */
	async function probeAxes(ctx: ExtensionContext) {
		const probes: Probe[] = [];
		const r = arm();
		for (const unit of MOVE_UNITS) {
			const [o] = await env.call<[Obs, Info]>("env.reset", {}, 300_000);
			const start = o.tcp_pos.toArray();
			let end = start;
			const delta = r.vectors[unit].map((x) => x * r.stepM * PROBE_UNITS) as Vec3;
			for (const target of waypoints(start, delta, r.stepM)) {
				const [frames] = await call<ServoReturn>(
					"env.servo",
					{ gain: r.gain, tol_m: SERVO.tolM, min_steps: SERVO.minSteps, max_steps: SERVO.maxSteps },
					[target, 1],
				);
				end = frames[frames.length - 1].tcp_pos.toArray();
			}
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
		gripper = 1;
		vectors = calibration = undefined;
		if (pi.getFlag("probe-axes") === true) await probeAxes(ctx);
		const [o, i] = await env.call<[Obs, Info]>("env.reset", {}, 300_000);
		absorb(o, i);
		language = await env.call<string>("env.get_task_language");
		return ["view_env_state", "move_delta", "finish"];
	}
}
