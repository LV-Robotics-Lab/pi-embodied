/**
 * RoboLab robot for pi: one NVIDIA RoboLab (Isaac Lab) benchmark task on a Franka + Panda hand.
 *
 *   pi -e packages/embodied/src/robolab --task BananaInBowlTask --seed 0 --cuda-device 1
 *   pi -e packages/embodied/src/robolab --units --task RubiksCubeTask   (Show-Harness action units)
 *
 * Starts one RoboLab env server per session (services/.../robots/robolab/env_server.py, the
 * venv from robots/robolab/install_isaac61.sh: Isaac Sim 6.1 / Isaac Lab 3.0 with RoboLab patched by
 * robolab-isaac61.patch; needs ROBOLAB_ROOT, and ROBOLAB_ISAAC_ASSETS for an offline copy of Isaac's
 * Franka USDs). `--instruction-type` picks RoboLab's default/vague/specific phrasing, `--subtask`
 * records its subtask progress (score) in the tool-result details and `robot_result`, never in the
 * planner's context. Isaac Sim takes about a minute to come up (longer
 * on a cold shader cache). `move_delta`, `rotate_delta` and the units hook `apply` share one
 * motion path: a base-frame delta in metres runs as ~2 cm relative-IK decisions with the orientation
 * held (Show-Harness's calibration), a yaw turns the hold's reference about the base vertical
 * (ROTATE_CW = +yaw, counter-clockwise seen from above, as LIBERO_TURNS); every result carries the
 * front and wrist images and the state; success is the task's own RoboLab termination predicate,
 * recorded in `robot_result`.
 *
 * Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: interpreters/robolab_atomic_controller.py, core/sim/robolab_*.py and
 * configs/robot_robolab.yaml ported as a pi robot.
 */

import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { type CameraMeta, pixelOnPlane } from "../flash/plane.ts";
import { recipeFlash } from "../flash/recipe.ts";
import { encodePng } from "../png.ts";
import { attach, defineRobot, SERVICES } from "../robot.ts";
import type { NdArray, RpcClient } from "../rpc.ts";
import type { MoveUnit, Vec3 } from "../units/index.ts";

const read = (name: string) => readFileSync(new URL(name, import.meta.url), "utf8");
const SYSTEM = read("./SYSTEM.md");
const MEMORY = read("./memory.md");
const EXPLORE = read("./explore.md");

/** configs/robot_robolab.yaml `move_vectors`: +x away from the base, -y = MV_LEFT, +z up. */
export const VECTORS: Record<MoveUnit, Vec3> = {
	MV_FWD: [1, 0, 0],
	MV_BACK: [-1, 0, 0],
	MV_LEFT: [0, -1, 0],
	MV_RIGHT: [0, 1, 0],
	MV_UP: [0, 0, 1],
	MV_DOWN: [0, 0, -1],
};
/** Physical metres per decision (the MVTOKEN 2 cm convention; the server applies the IK gain). */
export const STEP_M = 0.02;
/** Largest translation one call may command, m (the server refuses more). */
export const MAX_MOVE_M = 0.3;
/**
 * Radians per ROTATE_CW (LIBERO_TURNS' step); ROTATE_CW is +yaw about base +z, counter-clockwise seen
 * from above, as Show-Harness executes it (see ../libero LIBERO_TURNS), and the wrist view turns with it.
 */
export const YAW_STEP_RAD = 0.15;
/** Largest yaw one call may command, rad (the server clips more; the units layer splits longer turns). */
export const MAX_ROTATE_RAD = 0.3;
/** A closed Panda hand at or below this width holds nothing, m. */
export const EMPTY_WIDTH_M = 0.005;

/** How the front and wrist cameras look: verified on Isaac Sim 6.1 frames (BananaInBowlTask seed 0, 10 cm probe moves along each axis). */
export const VIEWS = `Each result shows the front view (a fixed camera in front of the robot, facing it; the robot base is at the top of the image), then the wrist view (it looks straight down from the gripper, rotated so the fingers are at the top of the image).
- Front view: MV_LEFT / MV_RIGHT move toward the image left / right, MV_FWD toward the image bottom (toward the camera), MV_BACK toward the image top (toward the robot base).
- Wrist view: MV_LEFT / MV_RIGHT move the gripper toward the image left / right, MV_FWD toward the image bottom, MV_BACK toward the image top; an object centered in the image is under the gripper: MV_DOWN.
- ROTATE_CW turns the gripper counter-clockwise seen from above, so the scene turns clockwise in the wrist view; ROTATE_CCW the opposite. The front view and the MV_* directions stay in the base frame.`;

type Obs = {
	agentview: NdArray;
	wrist: NdArray;
	eef_pos: NdArray;
	eef_quat_wxyz: NdArray;
	tilt_deg: number;
	/** Heading from the reset pose, deg (+ = counter-clockwise seen from above). */
	yaw_deg: number;
	gripper_width: number;
	gripper_command: "open" | "close";
	success: boolean;
	terminated: boolean;
	truncated: boolean;
	env_steps: number;
	/** RoboLab's subtask progress (`--subtask`): completed/total subtasks and the partial-credit score. */
	subtask?: { completed: number; total: number; score: number; info: string };
};
type Motion = Obs & {
	moved_m: number[];
	decisions: number;
	control_steps: number;
	frames?: NdArray[];
	cancelled?: boolean;
	error?: string;
};
type Moved = Motion & { commanded_m: number[] };
type Rotated = Motion & { requested_yaw: number; commanded_yaw: number; yaw: number; clipped?: boolean };
type Meta = {
	task: string;
	seed: number;
	instruction: string;
	instruction_type: string;
	subtask: boolean;
	episode_length_s: number;
	control_hz: number;
};

const round = (v: number, d = 4) => Number(v.toFixed(d));

export default function robolab(pi: ExtensionAPI) {
	const flag = (name: string, fallback: string) => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("task", { type: "string", default: "BananaInBowlTask", description: "RoboLab task class name" });
	pi.registerFlag("seed", { type: "string", default: "0", description: "Env seed" });
	pi.registerFlag("instruction-type", {
		type: "string",
		default: "default",
		description: "Task phrasing: default, vague or specific (RoboLab's per-task instruction variants)",
	});
	pi.registerFlag("subtask", {
		type: "boolean",
		default: false,
		description: "Track RoboLab's subtask progress (partial-credit score in results; extra physics queries)",
	});
	pi.registerFlag("cuda-device", {
		type: "string",
		default: "0",
		description: "GPU for Isaac Sim (physics, rendering; Vulkan ignores CUDA_VISIBLE_DEVICES)",
	});
	pi.registerFlag("env", { type: "string", description: "Attach to a running env server instead of starting one" });
	pi.registerFlag("services", {
		type: "string",
		default: process.env.PI_EMBODIED_SERVICES ?? SERVICES,
		description: "pi-embodied services dir",
	});
	pi.registerFlag("python", {
		type: "string",
		default: process.env.PI_EMBODIED_PYTHON ?? "python",
		description: "Python for the env server (the robolab venv)",
	});

	let env: RpcClient;
	let obs: Obs;
	let meta: Meta;

	/** The cell: task, instruction type (when not the default one) and seed. */
	const tag = (seed: string) => {
		const type = flag("instruction-type", "default");
		return `robolab_${robot.task.task}${type === "default" ? "" : `_${type}`}_s${seed}`;
	};
	const robot = defineRobot(pi, {
		name: "robolab",
		task: ["task", "seed"],
		// The env server's primitive registry (code.api), recorded per episode.
		codeApi: () => env,
		keepImages: 4,
		video: true,
		// Observations carry the front then the wrist image.
		vdm: { views: 2, wrist: 1 },
		groundTruth: (names) => env.call("env.ground_truth_poses", { names: names ?? null }, 60_000, [], robot.signal),
		// No corpus is published for RoboLab: memory is what exploration writes locally.
		memory: {
			cell: () => ({ tag: tag(robot.task.seed), reference: tag("0") }),
			primitives: ["move_delta", "rotate_delta", "act"],
			published: false,
		},
		// Flash replays a solved episode's plan (../flash/generate.ts --session --position state.eef_pos
		// --turn rotate_delta=yaw --heading state.yaw_deg: move_delta waypoints with their absolute end
		// positions, rotate_delta turns from the heading changes). Anchors are pointed at by Molmo in the front image (sent raw, 640x480) and met
		// with the plane at their recorded height through the front camera's calibration (env.get_camera_meta:
		// camera -> base, the frame of eef_pos); anchored waypoints then move with them, as deltas from the live
		// eef position, in moves of at most MAX_MOVE_M.
		flash: recipeFlash(pi, {
			names: () => [tag(robot.task.seed), tag("0")],
			memory: () => robot.mem?.render("{{memory_dir}}") ?? "",
			observe: "view_env_state",
			targets: {
				move_delta: {
					delta: "delta_xyz",
					position: (json) => (json.state as { eef_pos?: number[] } | undefined)?.eef_pos,
					maxStep: MAX_MOVE_M,
				},
			},
			backProject: async (_fr, pixel, anchor) => {
				const meta = await env.call<CameraMeta>(
					"env.get_camera_meta",
					{ camera_name: "agentview" },
					60_000,
					[],
					robot.signal,
				);
				return pixelOnPlane(meta, pixel, anchor.xyz[2]);
			},
			over: (latest) => latest.json.success === true,
			solved: (latest) => latest.json.success === true,
		}),
		explore: {
			// The exploration `reset` tool is not a robot.tool, so robot.signal is unset here; use its own signal.
			reset: async (result, _ctx, signal) => {
				[obs] = await env.call<[Obs, unknown]>("env.reset", {}, 300_000, [], signal);
				return observe({ ...result, reset: true });
			},
			prompt: () =>
				EXPLORE.replaceAll("{{task}}", robot.task.task)
					.replaceAll("{{seed}}", robot.task.seed)
					.replaceAll("{{instruction_type}}", flag("instruction-type", "default")),
			rewrite: [
				[
					/This is a single episode with a time limit\. You may recover within it \(re-position, re-grasp\), but you cannot restart it\./,
					"This is an exploration run with a time limit per attempt: `reset` starts a fresh attempt (see Exploration). Within an attempt, recover in place (re-position, re-grasp).",
				],
			],
		},
		start: startEpisode,
		prompt: () =>
			SYSTEM.replaceAll("{{task_language}}", meta.instruction).replaceAll(
				"{{memory}}",
				pi.getFlag("explore") === true ? "" : robot.mem!.render(MEMORY).trim(),
			),
		result: () => ({
			task: robot.task.task,
			seed: Number(robot.task.seed),
			instruction: meta?.instruction ?? null,
			instruction_type: meta?.instruction_type ?? flag("instruction-type", "default"),
			success: obs?.success ?? false,
			...(obs?.subtask ? { subtask: obs.subtask } : {}),
			truncated: obs?.truncated ?? false,
			env_steps: obs?.env_steps ?? 0,
		}),
		status: () => ({ language: meta.instruction, step: obs.env_steps, solved: obs.success }),
		finish: {
			description:
				"End the episode after checking the latest state. Success is RoboLab's task predicate, not this call.",
			parameters: Type.Object({
				status: StringEnum(["success", "failure"] as const),
				summary: Type.String(),
			}),
			result: (params) => ({
				content: [{ type: "text", text: `Episode finished (success=${obs.success}).` }],
				details: params,
			}),
		},
		units: {
			vectors: VECTORS,
			stepM: STEP_M,
			instruction: () => meta.instruction,
			views: VIEWS,
			emptyWidthM: EMPTY_WIDTH_M,
			yawStepRad: YAW_STEP_RAD,
			maxYawRad: () => MAX_ROTATE_RAD,
			// A grounded unit is a move or a turn (the rotation plugin's realign is a bare yaw); STOP is an empty move.
			apply: async (m, signal) => {
				const moved = m.yaw && !m.gripper && !Math.hypot(...m.delta) ? {} : await move(m.delta, m.gripper, signal);
				return observe(m.yaw ? { ...moved, rotate: await rotate(m.yaw, signal) } : moved);
			},
			state: async () => ({
				eef_xyz: obs.eef_pos.toArray().map((v) => round(v)),
				gripper_width: obs.gripper_width,
				tilt_deg: obs.tilt_deg,
				yaw_deg: obs.yaw_deg,
			}),
		},
	});
	const { video } = robot;

	/** Run one base-frame move (m) with an optional gripper command first; frames go to the video. */
	async function move(delta: Vec3, gripper: "open" | "close" | null, signal: AbortSignal | undefined) {
		const norm = Math.hypot(...delta);
		if (!(norm <= MAX_MOVE_M))
			throw new Error(`delta moves ${round(norm)} m; the limit is ${MAX_MOVE_M} m per call. Split the motion.`);
		const r = await env.call<Moved>(
			"env.move_delta",
			{ gripper, return_frames: true },
			300_000,
			[delta],
			signal ?? robot.signal,
		);
		for (const f of r.frames ?? []) video.frame(f);
		const { frames: _frames, commanded_m, moved_m, decisions, control_steps, cancelled, error, ...o } = r;
		obs = o;
		return {
			commanded_m: commanded_m.map((v) => round(v)),
			moved_m,
			decisions,
			control_steps,
			...(cancelled ? { cancelled } : {}),
			...(error ? { error } : {}),
		};
	}

	/** Turn the gripper by `yaw` (rad, + = counter-clockwise seen from above) about the base vertical; frames go to the video. */
	async function rotate(yaw: number, signal: AbortSignal | undefined) {
		if (!(Math.abs(yaw) <= MAX_ROTATE_RAD))
			throw new Error(`yaw ${round(yaw)} rad; the limit is ${MAX_ROTATE_RAD} rad per call. Split the turn.`);
		const r = await env.call<Rotated>(
			"env.rotate_delta",
			{ yaw, return_frames: true },
			300_000,
			[],
			signal ?? robot.signal,
		);
		for (const f of r.frames ?? []) video.frame(f);
		const {
			frames: _frames,
			requested_yaw: _requested,
			commanded_yaw,
			yaw: turned,
			clipped,
			moved_m,
			decisions,
			control_steps,
			cancelled,
			error,
			...o
		} = r;
		obs = o;
		return {
			commanded_yaw: round(commanded_yaw),
			yaw: turned,
			...(clipped ? { clipped } : {}),
			moved_m,
			decisions,
			control_steps,
			...(cancelled ? { cancelled } : {}),
			...(error ? { error } : {}),
		};
	}

	/** The result with the new state, then the front and wrist images. */
	function observe(result: Record<string, unknown>) {
		const images = [obs.agentview, obs.wrist].map((a) => encodePng(a.data, a.shape[1], a.shape[0]));
		const details = {
			result,
			step: obs.env_steps,
			success: obs.success,
			// The solved signal exploration and the memory recipe read (../memory, ../explore.ts), as ../maniskill sets it.
			terminated: obs.success,
			truncated: obs.truncated,
			task_language: meta.instruction,
			state: {
				eef_pos: obs.eef_pos.toArray().map((v) => round(v)),
				gripper_width: obs.gripper_width,
				gripper_command: obs.gripper_command,
				tilt_deg: obs.tilt_deg,
				yaw_deg: obs.yaw_deg,
			},
			images: [
				`front ${obs.agentview.shape[1]}x${obs.agentview.shape[0]}`,
				`wrist ${obs.wrist.shape[1]}x${obs.wrist.shape[0]}`,
			],
		};
		return {
			content: [
				{ type: "text" as const, text: JSON.stringify(details) },
				...images.map((png) => ({ type: "image" as const, data: png.toString("base64"), mimeType: "image/png" })),
			],
			// RoboLab's subtask judgement is the evaluator's, like `success` in Show-Harness: it goes to the
			// session (details, robot_result), never into the planner's context.
			details: obs.subtask ? { ...details, subtask: obs.subtask } : details,
		};
	}

	robot.tool(
		"view_env_state",
		"Current state with the front (third-person) and wrist images.",
		Type.Object({}),
		async () => observe({}),
	);

	robot.tool(
		"move_delta",
		`Translate the gripper by a base-frame [dx, dy, dz] in metres (+x away from the base, +y toward the robot's left, +z up; at most ${MAX_MOVE_M} m per call), optionally opening or closing the gripper first. The orientation is held. Returns the new state and images.`,
		Type.Object({
			delta_xyz: Type.Array(Type.Number(), { minItems: 3, maxItems: 3 }),
			gripper: Type.Optional(StringEnum(["open", "close"] as const)),
		}),
		async ({ delta_xyz, gripper }, signal) => {
			if (obs.success) return observe({ error: "the task is already solved; call finish" });
			return observe(await move(delta_xyz as Vec3, gripper ?? null, signal));
		},
	);

	robot.tool(
		"rotate_delta",
		`Turn the gripper by \`yaw\` radians about the vertical axis through it (+ = counter-clockwise seen from above, - = clockwise; at most ${MAX_ROTATE_RAD} rad per call), holding its position and downward tilt. The wrist view turns with it. Returns the new state and images.`,
		Type.Object({ yaw: Type.Number() }),
		async ({ yaw }, signal) => {
			if (obs.success) return observe({ error: "the task is already solved; call finish" });
			return observe(await rotate(yaw, signal));
		},
	);

	async function startEpisode() {
		const { task, seed } = robot.task;
		const endpoint = pi.getFlag("env") as string | undefined;
		if (endpoint) env = await attach(endpoint);
		else {
			const services = flag("services", SERVICES);
			env = await robot.serve({
				python: flag("python", "python"),
				args: [
					...["-m", "pi_embodied_services.robots.robolab.env_server"],
					...["--task", task, "--seed", seed, "--cuda-device", flag("cuda-device", "0")],
					...["--instruction-type", flag("instruction-type", "default")],
					...(pi.getFlag("subtask") ? ["--enable-subtask"] : []),
				],
				cwd: services,
				env: { ...process.env, PYTHONPATH: services, OMNI_KIT_ACCEPT_EULA: "YES" },
				log: (port) => join(tmpdir(), `pi-embodied-robolab-${task}-s${seed}-${port}.log`),
				// Isaac Sim's cold start (and a cold shader cache) comes before the server binds.
				readyMs: 1_200_000,
			});
		}
		meta = await env.call<Meta>("env.get_env_meta");
		const instructionType = flag("instruction-type", "default");
		const subtask = Boolean(pi.getFlag("subtask"));
		if (
			meta.task !== task ||
			meta.seed !== Number(seed) ||
			meta.instruction_type !== instructionType ||
			meta.subtask !== subtask
		)
			throw new Error(
				`env server runs ${meta.task} seed ${meta.seed} (${meta.instruction_type}, subtask ${meta.subtask}), not ${task} seed ${seed} (${instructionType}, subtask ${subtask})`,
			);
		// The server comes up reset; a new session on an attached server resets it again.
		[obs] = await env.call<[Obs, unknown]>("env.reset", {}, 300_000);
		return ["view_env_state", "move_delta", "rotate_delta", "finish"];
	}
}
