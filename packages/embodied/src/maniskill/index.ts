/**
 * ManiSkill robot for pi: a Panda in ManiSkill 3's translation-only `pd_ee_delta_pos` mode.
 *
 *   pi -e packages/embodied/src/maniskill --seed 0                    (BlockPAP-v1, Show-Harness's default scene)
 *   pi -e packages/embodied/src/maniskill --units --scene table_tex=white,cam_t=0302 --seed 3
 *   pi -e packages/embodied/src/maniskill --env-id StackCube-v1 --seed 0   (a stock ManiSkill scene)
 *
 * BlockPAP-v1 / BlockStack-v1 are RLinf's real2sim replicas of the real Franka rig (services/.../
 * robots/maniskill/scenes.py; fetch_real2sim.sh installs them): their calibrated front RealSense
 * and Show-Harness's wrist transform, reset like the training episodes, so a seed reproduces the
 * Show-Harness-Data frames. `--scene` passes the rig's options (table_tex, cam_t og|0302|0303,
 * traj_id, layout wide|none).
 *
 * Starts one ManiSkill env server per session (services/.../robots/maniskill/env_server.py, the
 * `maniskill` venv; rendering needs a GPU). `move_delta` and the units hook `apply` share one
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
import { encodePng } from "../png.ts";
import { attach, defineRobot, SERVICES } from "../robot.ts";
import { NdArray, type RpcClient } from "../rpc.ts";
import { MOVE_UNITS, type MoveUnit, type Vec3 } from "../units/index.ts";

const SYSTEM = readFileSync(new URL("./SYSTEM.md", import.meta.url), "utf8");

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
	wrist: NdArray;
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

type Meta = { env_id: string; seed: number; scene: Record<string, unknown> | null; table_z?: number } & Partial<
	typeof VIEW_SETUP
>;
/** The env server camera setup VIEWS describes; a server rendering anything else is refused. */
export const VIEW_SETUP = { agentview: "oblique", wrist_mount: "centered", wrist_rotation: 270, wrist_flip: "none" };
/** The same for the RLinf rigs (RIG_VIEWS): their calibrated external_cam and the 180 deg wrist. */
export const RIG_VIEW_SETUP = {
	agentview: "external_cam",
	wrist_mount: "centered",
	wrist_rotation: 0,
	wrist_flip: "both",
};

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
export function waypoints(start: number[], delta: Vec3): number[][] {
	const n = Math.max(1, Math.ceil(Math.hypot(...delta) / STEP_M - 1e-9));
	return Array.from({ length: n }, (_, i) => start.map((p, k) => p + (delta[k] * (i + 1)) / n));
}

/** One closed-loop servo call: drive to `target` in `minSteps`..`maxSteps` control steps. */
export type Phase = { target: number[]; minSteps: number; maxSteps: number };
/**
 * The servo phases of one call. A gripper change is its own phase first, holding still at `start`
 * for GRIPPER_STEPS until the fingers settle (Show-Harness's GRASP / RELEASE are separate tokens:
 * the fingers never close while the arm travels); then the ~2 cm waypoints of the move. A call
 * that neither moves nor changes the gripper (STOP) holds for one decision.
 */
export function phases(start: number[], delta: Vec3, gripperChanged: boolean): Phase[] {
	const hold = (steps: number): Phase => ({ target: start, minSteps: steps, maxSteps: steps });
	const out: Phase[] = gripperChanged ? [hold(GRIPPER_STEPS)] : [];
	if (Math.hypot(...delta) > 0)
		out.push(
			...waypoints(start, delta).map((target) => ({ target, minSteps: SERVO.minSteps, maxSteps: SERVO.maxSteps })),
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
		description: "ManiSkill env id: BlockPAP-v1 / BlockStack-v1 (RLinf rigs) or a stock task (PickCube-v1, ...)",
	});
	pi.registerFlag("scene", {
		type: "string",
		default: "",
		description:
			"RLinf rig options key=value,...: table_tex (006 wood | white | black | 001-021), cam_t (og | 0302 | 0303), traj_id (random | 0 | 15 | 25 | 40 | 45), layout (wide | none)",
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
	const text = () => SCENE_TEXT[rig ? "rig" : "stock"];
	/** The --probe-axes vectors (undefined: VECTORS) and their calibration record. */
	let vectors: Record<MoveUnit, Vec3> | undefined;
	let calibration: Record<string, unknown> | undefined;

	const robot = defineRobot(pi, {
		name: "maniskill",
		task: ["env-id", "seed", "scene"],
		keepImages: 4,
		video: true,
		// Observations carry the agentview then the wrist image.
		vdm: { views: 2, wrist: 1 },
		groundTruth: (names) => call("env.ground_truth_poses", { names: names ?? null }),
		start: startEpisode,
		prompt: () =>
			SYSTEM.replaceAll("{{task_language}}", language)
				.replaceAll("{{table}}", text().table)
				.replaceAll("{{object}}", text().object)
				.replaceAll("{{views}}", text().views),
		result: () => ({
			env_id: robot.task["env-id"],
			seed: Number(robot.task.seed),
			...(robot.task.scene ? { scene: robot.task.scene } : {}),
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
				return vectors ?? VECTORS;
			},
			stepM: STEP_M,
			instruction: () => language,
			get views() {
				return rig ? RIG_VIEWS : VIEWS;
			},
			emptyWidthM: EMPTY_WIDTH_M,
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
		for (const { target, minSteps, maxSteps } of phases(start, delta, gripper !== before)) {
			if (success) break;
			const [frames, i] = await call<ServoReturn>(
				"env.servo",
				{ gain: GAIN, tol_m: SERVO.tolM, min_steps: minSteps, max_steps: maxSteps },
				[target, gripper],
				signal,
			);
			for (const f of frames) video.frame(sideBySide(f.agentview, f.wrist));
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

	/** The motion result with the new state, then the agentview and wrist images. */
	function observe(result: Record<string, unknown>) {
		const images = [obs.agentview, obs.wrist].map((a) => encodePng(a.data, a.shape[1], a.shape[0]));
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
				`wrist ${obs.wrist.shape[1]}x${obs.wrist.shape[0]}`,
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
		for (const unit of MOVE_UNITS) {
			const [o] = await env.call<[Obs, Info]>("env.reset", {}, 300_000);
			const start = o.tcp_pos.toArray();
			let end = start;
			const delta = VECTORS[unit].map((x) => x * STEP_M * PROBE_UNITS) as Vec3;
			for (const target of waypoints(start, delta)) {
				const [frames] = await call<ServoReturn>(
					"env.servo",
					{ gain: GAIN, tol_m: SERVO.tolM, min_steps: SERVO.minSteps, max_steps: SERVO.maxSteps },
					[target, 1],
				);
				end = frames[frames.length - 1].tcp_pos.toArray();
			}
			probes.push({ unit, n: PROBE_UNITS, moved: end.map((v, k) => v - start[k]) as Vec3 });
		}
		const c = calibrate(VECTORS, STEP_M, probes);
		vectors = c.vectors;
		calibration = { env_id: robot.task["env-id"], seed: Number(robot.task.seed), step_m: STEP_M, units: c.units };
		const file = ctx.sessionManager.getSessionFile();
		if (file) writeFileSync(join(dirname(file), "calibration.json"), `${JSON.stringify(calibration, null, 2)}\n`);
	}

	async function startEpisode(ctx: ExtensionContext) {
		const envId = robot.task["env-id"];
		const seed = robot.task.seed;
		const scene = robot.task.scene ?? "";
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
		rig = Boolean(meta.scene);
		tableZ = meta.table_z ?? 0;
		const want = rig ? RIG_VIEW_SETUP : VIEW_SETUP;
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
