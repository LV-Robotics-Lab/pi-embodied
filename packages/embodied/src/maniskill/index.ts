/**
 * ManiSkill robot for pi: a Panda in ManiSkill 3's translation-only `pd_ee_delta_pos` mode.
 *
 *   pi -e packages/embodied/src/maniskill --env-id StackCube-v1 --seed 0
 *   pi -e packages/embodied/src/maniskill --units --env-id PickCube-v1 --seed 3   (Show-Harness action units)
 *
 * Starts one ManiSkill env server per session (services/.../robots/maniskill/env_server.py, the
 * `maniskill` venv; rendering needs a GPU). `move_delta` and the units hook `apply` share one
 * motion path: a base-frame delta in metres becomes ~2 cm decisions, each a closed-loop servo of
 * 2-8 control steps to its waypoint (Show-Harness's step calibration and real2sim execution);
 * every result carries the agentview and wrist images and the state; success is ManiSkill's own `success` flag, recorded in `robot_result`.
 *
 * Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: interpreters/maniskill_atomic_controller.py, the stock scenes of
 * core/sim/maniskill_scenes.py and configs/robot_maniskill.yaml ported as a pi robot.
 */

import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { encodePng } from "../png.ts";
import { attach, defineRobot, SERVICES } from "../robot.ts";
import { NdArray, type RpcClient } from "../rpc.ts";
import type { MoveUnit, Vec3 } from "../units/index.ts";

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
type Meta = { env_id: string; seed: number } & Partial<typeof VIEW_SETUP>;
/** The env server camera setup VIEWS describes; a server rendering anything else is refused. */
export const VIEW_SETUP = { agentview: "oblique", wrist_mount: "centered", wrist_rotation: 270, wrist_flip: "none" };

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

export default function maniskill(pi: ExtensionAPI) {
	const flag = (name: string, fallback: string) => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("env-id", { type: "string", default: "PickCube-v1", description: "ManiSkill env id" });
	pi.registerFlag("seed", { type: "string", default: "0", description: "Reset seed (the object layout)" });
	pi.registerFlag("env", { type: "string", description: "Attach to a running env server instead of starting one" });
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

	const robot = defineRobot(pi, {
		name: "maniskill",
		task: ["env-id", "seed"],
		keepImages: 4,
		video: true,
		start: startEpisode,
		prompt: () => SYSTEM.replaceAll("{{task_language}}", language),
		result: () => ({
			env_id: robot.task["env-id"],
			seed: Number(robot.task.seed),
			success,
			ever_grasped: everGrasped,
			env_steps: envStep,
		}),
		status: () => ({ language, step: envStep, solved: success }),
		finish: {
			description:
				"End the episode after checking the latest state. Success is ManiSkill's success flag, not this call.",
			parameters: Type.Object({
				status: Type.Union([Type.Literal("success"), Type.Literal("failure")]),
				summary: Type.String(),
			}),
			result: (params) => ({
				content: [{ type: "text", text: `Episode finished (success=${success}).` }],
				details: params,
			}),
		},
		units: {
			vectors: VECTORS,
			stepM: STEP_M,
			instruction: () => language,
			views: VIEWS,
			emptyWidthM: EMPTY_WIDTH_M,
			apply: async (m, signal) => {
				if (m.yaw) throw new Error("this robot has no yaw (pd_ee_delta_pos holds the orientation)");
				return observe(await move(m.delta, m.gripper, signal));
			},
			state: async () => ({
				eef_xyz: obs.tcp_pos.toArray().map((v) => round(v)),
				gripper_width: round(obs.gripper_width),
				table_z: 0,
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
		// A pure gripper toggle holds still until the fingers settle; STOP holds one decision.
		const hold = gripper !== before ? GRIPPER_STEPS : SERVO.minSteps;
		const servo = norm > 0 ? SERVO : { ...SERVO, minSteps: hold, maxSteps: hold };
		let steps = 0;
		for (const target of waypoints(start, delta)) {
			if (success) break;
			const [frames, i] = await call<ServoReturn>(
				"env.servo",
				{ gain: GAIN, tol_m: servo.tolM, min_steps: servo.minSteps, max_steps: servo.maxSteps },
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
		`Translate the gripper by a base-frame [dx, dy, dz] in metres (+x away from the base, +y toward the robot's left, +z up; at most ${MAX_MOVE_M} m per call), optionally opening or closing the gripper first. The orientation is locked. Returns the new state and images.`,
		Type.Object({
			delta_xyz: xyz,
			gripper: Type.Optional(Type.Union([Type.Literal("open"), Type.Literal("close")])),
		}),
		async ({ delta_xyz, gripper: g }, signal) => {
			if (success) return observe({ error: "the task is already solved; call finish" });
			return observe(await move(delta_xyz as Vec3, g ?? null, signal));
		},
	);

	async function startEpisode() {
		const envId = robot.task["env-id"];
		const seed = robot.task.seed;
		const endpoint = pi.getFlag("env") as string | undefined;
		if (endpoint) env = await attach(endpoint);
		else {
			const services = flag("services", SERVICES);
			env = await robot.serve({
				python: flag("python", "python"),
				args: ["-m", "pi_embodied_services.robots.maniskill.env_server", "--env-id", envId, "--seed", seed],
				cwd: services,
				env: { ...process.env, PYTHONPATH: services },
				log: (port) => join(tmpdir(), `pi-embodied-maniskill-${envId}-s${seed}-${port}.log`),
				readyMs: 600_000,
			});
		}
		const meta = await env.call<Meta>("env.get_env_meta");
		if (meta.env_id !== envId || meta.seed !== Number(seed))
			throw new Error(`env server runs ${meta.env_id} seed ${meta.seed}, not ${envId} seed ${seed}`);
		const setup = Object.entries(VIEW_SETUP).filter(([k, v]) => meta[k as keyof typeof VIEW_SETUP] !== v);
		if (setup.length)
			throw new Error(
				`env server cameras (${setup.map(([k]) => `${k}=${meta[k as keyof typeof VIEW_SETUP]}`).join(", ")}) differ from the ones the prompt describes (${JSON.stringify(VIEW_SETUP)}); update the services dir`,
			);
		success = everGrasped = false;
		envStep = 0;
		gripper = 1;
		const [o, i] = await env.call<[Obs, Info]>("env.reset", {}, 300_000);
		absorb(o, i);
		language = await env.call<string>("env.get_task_language");
		return ["view_env_state", "move_delta", "finish"];
	}
}
