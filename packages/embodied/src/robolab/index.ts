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
 * on a cold shader cache). `move_delta` and the units hook `apply` share one motion path: a
 * base-frame delta in metres runs as ~2 cm relative-IK decisions with the orientation locked
 * (Show-Harness's calibration); every result carries the front and wrist images and the state;
 * success is the task's own RoboLab termination predicate, recorded in `robot_result`.
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
import { encodePng } from "../png.ts";
import { attach, defineRobot, SERVICES } from "../robot.ts";
import type { NdArray, RpcClient } from "../rpc.ts";
import type { MoveUnit, Vec3 } from "../units/index.ts";

const SYSTEM = readFileSync(new URL("./SYSTEM.md", import.meta.url), "utf8");

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
/** A closed Panda hand at or below this width holds nothing, m. */
export const EMPTY_WIDTH_M = 0.005;

/** How the front and wrist cameras look: verified on Isaac Sim 6.1 frames (BananaInBowlTask seed 0, 10 cm probe moves along each axis). */
export const VIEWS = `Each result shows the front view (a fixed camera in front of the robot, facing it; the robot base is at the top of the image), then the wrist view (it looks straight down from the gripper, rotated so the fingers are at the top of the image).
- Front view: MV_LEFT / MV_RIGHT move toward the image left / right, MV_FWD toward the image bottom (toward the camera), MV_BACK toward the image top (toward the robot base).
- Wrist view: MV_LEFT / MV_RIGHT move the gripper toward the image left / right, MV_FWD toward the image bottom, MV_BACK toward the image top; an object centered in the image is under the gripper: MV_DOWN.`;

type Obs = {
	agentview: NdArray;
	wrist: NdArray;
	eef_pos: NdArray;
	eef_quat_wxyz: NdArray;
	tilt_deg: number;
	gripper_width: number;
	gripper_command: "open" | "close";
	success: boolean;
	terminated: boolean;
	truncated: boolean;
	env_steps: number;
	/** RoboLab's subtask progress (`--subtask`): completed/total subtasks and the partial-credit score. */
	subtask?: { completed: number; total: number; score: number; info: string };
};
type Moved = Obs & {
	commanded_m: number[];
	moved_m: number[];
	decisions: number;
	control_steps: number;
	frames?: NdArray[];
	cancelled?: boolean;
	error?: string;
};
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
		start: startEpisode,
		prompt: () => SYSTEM.replaceAll("{{task_language}}", meta.instruction),
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
			apply: async (m, signal) => {
				if (m.yaw) throw new Error("this robot has no yaw (the relative IK holds the orientation)");
				return observe(await move(m.delta, m.gripper, signal));
			},
			state: async () => ({
				eef_xyz: obs.eef_pos.toArray().map((v) => round(v)),
				gripper_width: obs.gripper_width,
				tilt_deg: obs.tilt_deg,
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

	/** The result with the new state, then the front and wrist images. */
	function observe(result: Record<string, unknown>) {
		const images = [obs.agentview, obs.wrist].map((a) => encodePng(a.data, a.shape[1], a.shape[0]));
		const details = {
			result,
			step: obs.env_steps,
			success: obs.success,
			truncated: obs.truncated,
			task_language: meta.instruction,
			state: {
				eef_pos: obs.eef_pos.toArray().map((v) => round(v)),
				gripper_width: obs.gripper_width,
				gripper_command: obs.gripper_command,
				tilt_deg: obs.tilt_deg,
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
		`Translate the gripper by a base-frame [dx, dy, dz] in metres (+x away from the base, +y toward the robot's left, +z up; at most ${MAX_MOVE_M} m per call), optionally opening or closing the gripper first. The orientation is locked. Returns the new state and images.`,
		Type.Object({
			delta_xyz: Type.Array(Type.Number(), { minItems: 3, maxItems: 3 }),
			gripper: Type.Optional(StringEnum(["open", "close"] as const)),
		}),
		async ({ delta_xyz, gripper }, signal) => {
			if (obs.success) return observe({ error: "the task is already solved; call finish" });
			return observe(await move(delta_xyz as Vec3, gripper ?? null, signal));
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
		if (meta.task !== task || meta.seed !== Number(seed) || meta.instruction_type !== instructionType)
			throw new Error(
				`env server runs ${meta.task} seed ${meta.seed} (${meta.instruction_type}), not ${task} seed ${seed} (${instructionType})`,
			);
		// The server comes up reset; a new session on an attached server resets it again.
		[obs] = await env.call<[Obs, unknown]>("env.reset", {}, 300_000);
		return ["view_env_state", "move_delta", "finish"];
	}
}
