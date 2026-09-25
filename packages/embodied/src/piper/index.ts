/**
 * One physical AgileX Piper arm for pi, ported from Show-Harness (github.com/showlab/Show-Harness).
 *
 *   pi -e packages/embodied/src/piper --operator --task banana_plate --robot-config my_piper.yaml
 *
 * Starts the env server (pi_embodied_services.robots.piper.env_server, ROS topics to the AgileX
 * arm node and the Orbbec cameras) or attaches to one with --robot-env. The server owns the
 * safety limits from the robot YAML: per-call step and yaw refusal, the Z floor, the optional
 * workspace box, the joint-stream speed, and the divergence and dropped-gripper guards. A real
 * robot needs an operator: pi must have a UI, --operator must be on (the base then asks for a
 * verdict before `finish`), and the operator confirms the reset before any motion. Motion tools
 * record a state step (robot state, front and wrist RGB) under --out and return it with both
 * images. The robot opts into the shared action-unit layer (../units) with the Show-Harness
 * primitives of configs/primitives_piper.yaml.
 */

import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { encodePng } from "../png.ts";
import {
	attach,
	checkMove,
	defineRobot,
	type Json,
	message,
	plain,
	type RobotSpec,
	rgbOf,
	round,
	SERVICES,
	type Services,
	servicesEnv,
	toolResult,
	vec,
} from "../robot.ts";
import { NdArray, type RpcClient, RpcUnavailable } from "../rpc.ts";
import { compensate, type MoveUnit, type State, type UnitsSpec, type Vec3 } from "../units/index.ts";

const SYSTEM = readFileSync(new URL("./SYSTEM.md", import.meta.url), "utf8");

type Task = { instruction: string; success_criteria?: string };
type Meta = {
	arm: string;
	units_frame: "base" | "heading";
	limits: { max_step_m: number; max_yaw_rad: number; z_floor_m: number | null; empty_width_m: number | null };
	has_begin_pose: boolean;
	tasks: Record<string, Task>;
};
type Step = { blob: Json; images: Record<string, string> };

/**
 * The action-unit grounding of Show-Harness configs/primitives_piper.yaml: MV_* unit vectors
 * (x toward the far field, y left, z up), 2 cm per unit. The env server interprets them in the
 * frame its config names (`motion.units_frame`: `heading` is Show-Harness's `motion_frame: wrist`).
 * No ROTATE_* units: Show-Harness never offers rotation on the Piper (core/launch.py), and the
 * rotation plugin would turn heading-frame moves a second time.
 */
export const PIPER_UNITS = {
	vectors: {
		MV_FWD: [1, 0, 0],
		MV_BACK: [-1, 0, 0],
		MV_LEFT: [0, 1, 0],
		MV_RIGHT: [0, -1, 0],
		MV_UP: [0, 0, 1],
		MV_DOWN: [0, 0, -1],
	} as Record<MoveUnit, Vec3>,
	stepM: 0.02,
};

/** The base-frame translation of a heading-frame delta: rotated by the state's `heading_yaw_rad`, as the server does. */
export function headingToBase(delta: Vec3, state: State | undefined): Vec3 {
	const heading = state?.heading_yaw_rad;
	return typeof heading === "number" ? compensate(delta, heading) : delta;
}

/**
 * How the Piper rig's views look, from Show-Harness plugins/ego (the rig's `is_ego` fix, learned on
 * hardware): the front camera and the wrist camera do not share one forward/back convention. With
 * `units_frame: heading` the front-view directions follow the gripper heading (plugins/wrist_frame):
 * image-edge directions are wrong there once the gripper is yawed.
 */
const FRONT_BASE = `- Image 1, FRONT camera: faces the arm, which enters from the TOP of the image. MV_FWD moves the gripper toward the image bottom, MV_BACK toward the image top, MV_LEFT / MV_RIGHT toward image left / right.`;
const FRONT_HEADING = `- Image 1, FRONT camera: faces the arm, which enters from the TOP of the image. Moves follow the GRIPPER HEADING, the direction the gripper points, visible in this view: MV_FWD moves further ahead along the heading, MV_BACK back against it, MV_LEFT / MV_RIGHT to the heading's left / right. Only while the gripper points straight toward the image bottom are these the image bottom / top / left / right.`;
const WRIST_VIEWS = `- Image 2, WRIST camera: looks along the gripper at the fingertips (bottom of the image). MV_FWD advances the gripper, so a target near the image TOP needs MV_FWD and one between the image top and the fingers MV_BACK; MV_LEFT / MV_RIGHT move toward image left / right; MV_DOWN brings the fingers down onto what is centered between them.
- MV_UP / MV_DOWN change the gripper's height in both views.`;
export const piperViews = (frame: "base" | "heading") =>
	`${frame === "heading" ? FRONT_HEADING : FRONT_BASE}\n${WRIST_VIEWS}`;

const TOOLS = ["view_env_state", "move_delta", "rotate_yaw", "open_gripper", "close_gripper", "finish"];

export default function piper(pi: ExtensionAPI) {
	const flag = (name: string, fallback = "") => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("task", {
		type: "string",
		default: "banana_plate",
		description: "Task name from the robot YAML's `tasks:`",
	});
	pi.registerFlag("robot-config", {
		type: "string",
		description: "Robot YAML (default: services/pi_embodied_services/robots/piper/config/example.yaml)",
	});
	pi.registerFlag("robot-env", {
		type: "string",
		description: "Attach to a running Piper env server instead of starting one",
	});
	pi.registerFlag("robot-ros-setup", {
		type: "string",
		default: "",
		description:
			"Colon-separated setup.bash files sourced before the env server starts (e.g. /opt/ros/noetic/setup.bash:~/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash); empty = inherit pi's environment",
	});
	pi.registerFlag("services", {
		type: "string",
		default: process.env.PI_EMBODIED_SERVICES ?? SERVICES,
		description: "pi-embodied services dir",
	});
	pi.registerFlag("python", {
		type: "string",
		default: process.env.PI_EMBODIED_PYTHON ?? "python",
		description: "Python with the services' [piper] extra",
	});
	pi.registerFlag("out", {
		type: "string",
		description: "Step artifact directory (default: a new directory under the OS temp dir)",
	});
	pi.registerFlag("max-move", {
		type: "string",
		default: "0.05",
		description: "Largest translation per call, m (the server's limits.max_step_m applies if tighter)",
	});
	pi.registerFlag("max-yaw", {
		type: "string",
		default: "0.2",
		description: "Largest |yaw| per call, rad (the server's limits.max_yaw_rad applies if tighter)",
	});

	let env: RpcClient | undefined;
	let meta: Meta | undefined;
	let task: Task | undefined;
	let out = "";
	const steps: Step[] = [];
	const taskName = () => robot.task.task;

	const units: UnitsSpec = {
		...PIPER_UNITS,
		// The units layer runs its own recovery, so an empty close stays closed for it to see.
		apply: (move, signal) =>
			act({ action: "unit", ...move }, () =>
				guardedStep(move.delta, move.yaw, move.gripper, unitsFrame(), signal, false),
			),
		state: async () => proprio(await call<Json>("env.get_robot_state")),
		// The stall check compares commanded and measured motion in the base frame.
		baseDelta: (delta, state) => (unitsFrame() === "heading" ? headingToBase(delta, state) : delta),
		instruction: () => task?.instruction ?? "",
		get views() {
			return piperViews(unitsFrame());
		},
		get emptyWidthM() {
			return meta?.limits.empty_width_m ?? 0.005;
		},
	};
	const spec: RobotSpec = {
		name: "piper",
		task: ["task"],
		keepImages: 4,
		operator: { step: () => steps.length, reset: resetArm },
		start: startRobot,
		stop: () => {
			env = meta = task = undefined;
		},
		prompt: () => {
			if (!task || !meta) return undefined;
			const vars: Record<string, string> = {
				task_name: taskName(),
				instruction: task.instruction,
				success_criteria: task.success_criteria ?? "Judged by the operator.",
				max_move: String(maxMove()),
				max_yaw: String(maxYaw()),
			};
			return SYSTEM.replace(/\{\{(\w+)\}\}/g, (m, k: string) => vars[k] ?? m);
		},
		result: () => ({ task: taskName(), arm: meta?.arm ?? null, steps: steps.length, out }),
		status: () => ({ language: task?.instruction, step: steps.length - 1 }),
		finish: {
			description:
				"Call when the task is complete or unrecoverable. Halts the agent loop. With --operator, the operator gives a verdict first.",
			parameters: Type.Object({
				status: Type.String({ description: "Outcome, e.g. 'success', 'failure', or 'stuck'." }),
				summary: Type.String({ description: "Short natural-language summary of the run." }),
			}),
			result: (params) => toolResult({ _finish: true, ...params }),
		},
		units,
	};
	const robot = defineRobot(pi, spec);
	const { op } = robot;

	const maxMove = () => Math.min(Number(flag("max-move", "0.05")), meta?.limits.max_step_m ?? Infinity);
	const maxYaw = () => Math.min(Number(flag("max-yaw", "0.2")), meta?.limits.max_yaw_rad ?? Infinity);
	const unitsFrame = () => meta?.units_frame ?? "base";

	function call<T = Json>(method: string, kwargs: Json = {}, timeoutMs = 30_000, signal?: AbortSignal) {
		if (!env) throw new Error("piper is not initialized; see the session start error");
		return env.call<T>(method, kwargs, timeoutMs, [], signal);
	}

	/** Refuse a step beyond the limits, then run it on the server (which checks them again). */
	async function guardedStep(
		delta: number[],
		yaw: number,
		gripper: "open" | "close" | null,
		frame: "base" | "heading",
		signal?: AbortSignal,
		reopenEmpty = true,
	): Promise<Json> {
		op.check();
		if (signal?.aborted) throw new Error("tool operation interrupted");
		if (delta.length !== 3 || !delta.every(Number.isFinite)) throw new Error("delta must be 3 finite numbers");
		if (!Number.isFinite(yaw)) throw new Error("yaw must be finite");
		checkMove(delta, maxMove());
		if (Math.abs(yaw) > maxYaw())
			throw new Error(`yaw ${round(yaw, 4)} rad exceeds the limit of ${maxYaw()} rad per call.`);
		return call("env.step", { delta_xyz: delta, yaw, gripper, frame, reopen_empty: reopenEmpty }, 120_000, signal);
	}

	/** Proprioception for prompts and the units plugins (`eef_xyz`, `gripper_width`, `table_z`). */
	function proprio(s: Json): Record<string, unknown> {
		return {
			eef_xyz: vec(s.eef_pos),
			gripper_width: Number(s.gripper_width_m),
			// The Z floor is the EEF height with the gripper resting on the table (Show-Harness high_above_table).
			...(typeof s.z_floor_m === "number" ? { table_z: s.z_floor_m } : {}),
			eef_euler_xyz: vec(s.eef_euler_xyz).map((v) => round(v, 3)),
			heading_yaw_rad: typeof s.heading_yaw_rad === "number" ? s.heading_yaw_rad : null,
			gripper_closed: s.gripper_closed,
		};
	}

	// ---- state steps

	/** Record the current observation as step N (PNG per camera + states.jsonl) and return it. */
	async function record(command: Json | null, result: Json | null, elapsed: number | null): Promise<Step> {
		const obs = await call<Json>("env.get_observation");
		const idx = steps.length;
		const dir = join(out, `step_${String(idx).padStart(4, "0")}`);
		mkdirSync(dir, { recursive: true });
		const images: Record<string, string> = {};
		for (const name of ["front", "wrist"]) {
			const v = obs.images?.[name];
			if (!(v instanceof NdArray)) continue;
			const img = rgbOf(v);
			images[name] = join(dir, `${name}.png`);
			writeFileSync(images[name], encodePng(img.rgb, img.width, img.height));
		}
		const blob: Json = { step_idx: idx, state: plain(obs.robot_state), images };
		if (command) blob.command = command;
		if (result) blob.result = plain(result);
		if (elapsed !== null) blob.elapsed_s = elapsed;
		appendFileSync(join(out, "states.jsonl"), `${JSON.stringify(blob)}\n`);
		const step = { blob, images };
		steps.push(step);
		return step;
	}

	/** The step blob plus the front then wrist image. */
	function view(s: Step): AgentToolResult<unknown> {
		const pngs = ["front", "wrist"].filter((k) => s.images[k]).map((k) => readFileSync(s.images[k]));
		return toolResult(s.blob, pngs);
	}

	/** Run one motion, then record and return the new state; errors are returned, not thrown. */
	async function act(command: Json, run: () => Promise<Json>): Promise<AgentToolResult<unknown>> {
		const started = performance.now();
		let result: Json;
		try {
			result = await run();
		} catch (err) {
			// A server that stopped answering ends the episode (the base's tool wrapper handles it).
			if (err instanceof RpcUnavailable) throw err;
			return toolResult({ error: message(err), command });
		}
		const elapsed = round((performance.now() - started) / 1000, 2);
		try {
			return view(await record(command, result, elapsed));
		} catch (err) {
			if (err instanceof RpcUnavailable) throw err;
			return toolResult({ ...result, error: `failed to capture state after the motion: ${message(err)}` });
		}
	}

	robot.tool(
		"view_env_state",
		"Read a Piper state step (eef pose, gripper, limits) and its front and wrist images.",
		Type.Object({ step: Type.Optional(Type.Integer({ description: "State step (default -1 = latest)" })) }),
		async ({ step = -1 }) => {
			const s = steps[step < 0 ? steps.length + step : step];
			if (!s) return toolResult({ error: `step ${step} is not recorded (have 0..${steps.length - 1})` });
			return view(s);
		},
	);

	const xyz = Type.Array(Type.Number(), { minItems: 3, maxItems: 3 });
	robot.tool(
		"move_delta",
		"Move the Piper gripper by a bounded base-frame xyz delta in meters (x forward, y left, z up).",
		Type.Object({ delta_xyz: xyz }),
		async ({ delta_xyz }, signal) =>
			act({ action: "move_delta", delta_xyz }, () => guardedStep(delta_xyz, 0, null, "base", signal)),
	);
	robot.tool(
		"rotate_yaw",
		"Rotate the Piper gripper about the base z axis by a bounded angle in radians.",
		Type.Object({ yaw: Type.Number() }),
		async ({ yaw }, signal) =>
			act({ action: "rotate_yaw", yaw }, () => guardedStep([0, 0, 0], yaw, null, "base", signal)),
	);
	robot.tool(
		"open_gripper",
		"Open the Piper gripper and wait for it to settle.",
		Type.Object({}),
		async (_p, signal) => act({ action: "open_gripper" }, () => guardedStep([0, 0, 0], 0, "open", "base", signal)),
	);
	robot.tool(
		"close_gripper",
		"Close the Piper gripper and wait for it to settle; an empty close reopens and says so in notes.",
		Type.Object({}),
		async (_p, signal) => act({ action: "close_gripper" }, () => guardedStep([0, 0, 0], 0, "close", "base", signal)),
	);

	// ---- lifecycle

	/** Operator-confirmed reset (request_scene_reset): open the gripper, move to the begin pose. */
	async function resetArm(): Promise<Json> {
		const r = await call<Json>("env.reset", {}, 120_000);
		if (!r.ok) throw new Error(`reset did not reach the begin pose: ${JSON.stringify(plain(r.move))}`);
		await record({ action: "reset" }, r, null);
		return { ok: true, step: steps.length - 1 };
	}

	async function startRobot(ctx: ExtensionContext): Promise<string[]> {
		if (!ctx.hasUI)
			throw new Error("piper drives a real robot: run pi interactively (or over RPC) so an operator is present");
		if (pi.getFlag("operator") !== true)
			throw new Error("piper drives a real robot: start pi with --operator so an operator judges every episode");
		const r: Services = { root: flag("services"), python: flag("python", "python") };
		const configFlag = flag("robot-config");
		const config = configFlag ? resolve(ctx.cwd, configFlag) : "";
		const stamp = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
		out = resolve(ctx.cwd, flag("out") || join(tmpdir(), "pi-embodied", `piper_${taskName()}_${stamp}`));
		mkdirSync(out, { recursive: true });
		steps.length = 0;
		const endpoint = flag("robot-env");
		const setups = flag("robot-ros-setup").split(":").filter(Boolean);
		const server = [
			"-m",
			"pi_embodied_services.robots.piper.env_server",
			...(config ? ["--robot-config", config] : []),
		];
		// Source the ROS workspaces in a shell that then execs Python; serve appends the transport flags.
		const sourced = setups.map((f) => `. ${JSON.stringify(f.replace(/^~(?=\/)/, "$HOME"))}`).join(" && ");
		const rpc = endpoint
			? await attach(endpoint)
			: await robot.serve({
					python: setups.length ? "bash" : r.python,
					args: setups.length ? ["-c", `${sourced} && exec "$@"`, "piper-env", r.python, ...server] : server,
					cwd: r.root,
					env: servicesEnv(r),
					log: () => join(out, "piper_env_server.log"),
					readyMs: 60_000,
				});
		const m = await rpc.call<Meta>("env.get_env_meta", {}, 30_000);
		const t = m.tasks?.[taskName()];
		if (!t?.instruction)
			throw new Error(
				`task '${taskName()}' is not in the robot config's tasks (have: ${Object.keys(m.tasks ?? {}).join(", ") || "none"})`,
			);
		if (!m.has_begin_pose)
			throw new Error("calibration.begin_joints is not set in the robot config; reset would fail");
		const go = await ctx.ui.confirm(
			"Move the Piper arm?",
			`The ${m.arm} arm will open its gripper and move to its begin pose, then the agent drives it for: ${t.instruction}. Clear the workspace and keep the emergency stop in reach.`,
		);
		if (!go) throw new Error("operator declined the reset; the Piper tools stay disabled");
		env = rpc;
		meta = m;
		try {
			await resetArm();
		} catch (err) {
			env = meta = undefined;
			throw err;
		}
		task = t;
		ctx.ui.notify(`Piper ready: ${taskName()} (${m.arm} arm); steps under ${out}`, "info");
		return TOOLS;
	}
}
