/**
 * One physical AgileX Piper arm, or both arms of the Cobot Magic rig, for pi, ported from
 * Show-Harness (github.com/showlab/Show-Harness).
 *
 *   pi -e packages/embodied/src/piper --operator --task banana_plate --robot-config my_piper.yaml
 *   pi -e packages/embodied/src/piper/dual.ts --operator --task banana_handover --robot-config my_dual.yaml
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
 *
 * Two arms (./dual.ts, a server config with an `arms:` block): every motion names its arm (the
 * units' `arm`; STILL leaves the other arm alone), each arm has its own Z floor and workspace box on
 * the server, a faulted arm is halted there until its reset while the other keeps working, and
 * halt_arm stops one arm on request. The images are front, left wrist, right wrist.
 *
 * Views and motion frames (Show-Harness plugins/wrist_frame and plugins/view_select): with
 * `motion.units_frame: heading` (Show-Harness `motion_frame: wrist`, the default config) the front
 * view's directions are given relative to the gripper heading. --view-select (config
 * `units_frame: base`) lets the model report which view guided each move (`act`'s `view`, the units
 * view hook): WRIST runs it in the heading frame, FRONT in the base frame, and the directions stay in
 * the base convention. The robot refuses to start with --view-select when the units module it runs
 * with has no view hook (every move would silently run in the base frame).
 *
 * Reset: on two arms the start and the operator's scene reset reset both arms (the operator
 * confirmed it); a gripper that holds an object is opened only after the operator confirms that too.
 */

import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type TSchema, Type } from "typebox";
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
import {
	compensate,
	type MoveUnit,
	type State,
	UNITS_EVENT,
	type UnitsHandle,
	type UnitsSpec,
	type Vec3,
} from "../units/index.ts";

const SYSTEM = readFileSync(new URL("./SYSTEM.md", import.meta.url), "utf8");
const SYSTEM_DUAL = readFileSync(new URL("./SYSTEM_DUAL.md", import.meta.url), "utf8");

/** The dual rig's arms, as the env server names them. */
export const PIPER_ARMS = ["left", "right"] as const;

type Frame = "base" | "heading";
type Task = { instruction: string; success_criteria?: string };
type Meta = {
	arm: string;
	/** Two arms: ["left", "right"]; one arm: []. */
	arms?: string[];
	cameras?: string[];
	units_frame: Frame;
	limits: { max_step_m: number; max_yaw_rad: number; z_floor_m: number | null; empty_width_m: number | null };
	has_begin_pose: boolean;
	tasks: Record<string, Task>;
	/** The server's smooth joint stream (`enabled`, `blend`: chaining on). */
	smooth?: { enabled?: boolean; blend?: boolean };
	/** One arm: the backend; two arms: per arm. */
	motion_backend?: string | Record<string, string>;
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

/** The act `view` of Show-Harness plugins/view_select: which view guided the move. */
export type GuideView = "WRIST" | "FRONT";

/**
 * The frame a unit move runs in (Show-Harness plugins/view_select `frame_for`): with view select on,
 * a WRIST-guided move runs in the heading frame (the wrist view's directions are exact at any yaw)
 * and a FRONT-guided one in the base frame (the front view's image-edge directions are exact); a
 * move without a view keeps the configured frame.
 */
export function motionFrame(view: unknown, configured: Frame, viewSelect: boolean): Frame {
	if (!viewSelect) return configured;
	if (view === "WRIST") return "heading";
	if (view === "FRONT") return "base";
	return configured;
}

/**
 * How the Piper rig's views look, from Show-Harness plugins/ego (the rig's `is_ego` fix, learned on
 * hardware): the front camera and the wrist camera do not share one forward/back convention.
 * `frame` is the frame the units run in: in the heading frame the front view's image-edge mapping
 * is wrong once the gripper is yawed, so it is rewritten to follow the gripper heading
 * (Show-Harness plugins/wrist_frame); the wrist view's directions are exact in that frame already.
 * `viewSelect` keeps the base convention for both views and asks for each move's guiding view.
 */
export function piperViews(frame: Frame, arms: readonly string[] = [], viewSelect = false): string {
	const dual = arms.length > 0;
	const enters = dual ? "arms, which enter" : "arm, which enters";
	const front =
		frame === "heading" && !viewSelect
			? `FRONT camera: faces the ${enters} from the TOP of the image. Moves follow the GRIPPER HEADING, the direction ${dual ? "that arm's" : "the"} gripper points, visible in this view: MV_FWD moves further ahead along the heading, MV_BACK back against it, MV_LEFT / MV_RIGHT to the heading's left / right. Only while the gripper points straight toward the image bottom are these the image bottom / top / left / right.`
			: `FRONT camera: faces the ${enters} from the TOP of the image. MV_FWD moves the gripper toward the image bottom, MV_BACK toward the image top, MV_LEFT / MV_RIGHT toward image left / right.`;
	const wrist =
		"looks along the gripper at the fingertips (bottom of the image). MV_FWD advances the gripper, so a target near the image TOP needs MV_FWD and one between the image top and the fingers MV_BACK; MV_LEFT / MV_RIGHT move toward image left / right; MV_DOWN brings the fingers down onto what is centered between them.";
	const lines = dual
		? [
				`- Image 1, ${front}`,
				`- Image 2, LEFT WRIST camera (the left arm's): ${wrist}`,
				`- Image 3, RIGHT WRIST camera (the right arm's): ${wrist}`,
				"- MV_UP / MV_DOWN change the gripper's height in every view. Each unit moves only the arm named by `arm`.",
			]
		: [
				`- Image 1, ${front}`,
				`- Image 2, WRIST camera: ${wrist}`,
				"- MV_UP / MV_DOWN change the gripper's height in both views.",
			];
	if (viewSelect)
		lines.push(
			"- VIEW SELECT: with every MV_* set `view` to the view that guided it: WRIST when the target is in the wrist view and you judged it there, FRONT when you judged it in the front view. A WRIST move runs along the gripper's heading and a FRONT move in the front view's image directions, so the directions above are exact for the view you name.",
		);
	return lines.join("\n");
}

/** Whether the server chains `continuous` moves: smooth + blend on the joint-stream backend (every arm). */
export function chains(meta: Pick<Meta, "smooth" | "motion_backend"> | undefined): boolean {
	const b = meta?.motion_backend;
	const backends = typeof b === "string" ? [b] : Object.values(b ?? {});
	return (
		meta?.smooth?.enabled === true &&
		meta.smooth.blend === true &&
		backends.length > 0 &&
		backends.every((x) => x === "joint_stream")
	);
}

/** The robot's own tools; two arms add halt_arm. */
const TOOLS = ["view_env_state", "move_delta", "rotate_yaw", "open_gripper", "close_gripper", "finish"];

/** One Piper arm (the default entry). */
export default function piper(pi: ExtensionAPI) {
	piperRobot(pi, false);
}

/** The Piper robot on one arm, or on both arms of the dual rig (`dual`, ./dual.ts). */
export function piperRobot(pi: ExtensionAPI, dual: boolean) {
	const arms: readonly string[] = dual ? PIPER_ARMS : [];
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
	pi.registerFlag("view-select", {
		type: "boolean",
		default: false,
		description:
			"Units: the model reports each move's guiding view (act's `view`): WRIST runs it in the gripper-heading frame, FRONT in the base frame (Show-Harness plugins/view_select; needs motion.units_frame: base)",
	});

	let env: RpcClient | undefined;
	let meta: Meta | undefined;
	let task: Task | undefined;
	let out = "";
	const steps: Step[] = [];
	const taskName = () => robot.task.task;
	const viewSelect = () => pi.getFlag("view-select") === true;
	/** The frame of the last unit move: the stall check converts that move's delta to the base frame. */
	let lastFrame: Frame = "base";
	/** The units module published a handle with the view hook this session (`UnitsHandle.viewSelect`). */
	let unitsViewSelect = false;
	/** The operator's dialogs (the reset confirms opening a gripper that holds something). */
	let ui: ExtensionContext["ui"] | undefined;
	// Registered before the robot (and its units) so it clears before the units publish their handle.
	pi.on("session_start", () => {
		unitsViewSelect = false;
	});
	pi.events.on(UNITS_EVENT, (h) => {
		unitsViewSelect = (h as UnitsHandle).viewSelect === true;
	});

	const units: UnitsSpec = {
		...PIPER_UNITS,
		// Asks ../units for `act`'s `view` while --view-select is on, passed on as `Move.view`.
		viewSelect,
		...(dual ? { arms } : {}),
		// The units layer runs its own recovery, so an empty close stays closed for it to see.
		apply: (move, signal) => {
			const view = move.view;
			const frame = motionFrame(view, unitsFrame(), viewSelect());
			const command = { action: "unit", ...move, ...(viewSelect() && view ? { frame } : {}) };
			return act(command, () => {
				lastFrame = frame;
				return guardedStep(move.delta, move.yaw, move.gripper, frame, signal, false, move.arm, move.continuous);
			});
		},
		state: async (arm) => proprio(await call<Json>("env.get_robot_state", dual ? { arm: armName(arm) } : {})),
		// The stall check compares commanded and measured motion in the base frame.
		baseDelta: (delta, state) => (lastFrame === "heading" ? headingToBase(delta, state) : delta),
		instruction: () => task?.instruction ?? "",
		get views() {
			return piperViews(viewSelect() ? "base" : unitsFrame(), arms, viewSelect());
		},
		get emptyWidthM() {
			return meta?.limits.empty_width_m ?? 0.005;
		},
		// The server's smooth stream chains `continuous` moves: they return before the arm settles.
		chains: () => chains(meta),
	};
	const spec: RobotSpec = {
		name: dual ? "piper_dual" : "piper",
		task: ["task"],
		keepImages: 4,
		// Observations carry cameras() in order; the front view has to lead to be the main one.
		vdm: () => {
			const c = cameras();
			if (c[0] !== "front") return undefined;
			return { views: c.length, wrist: c.flatMap((name, i) => (name.startsWith("wrist") ? [i] : [])) };
		},
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
			return (dual ? SYSTEM_DUAL : SYSTEM).replace(/\{\{(\w+)\}\}/g, (m, k: string) => vars[k] ?? m);
		},
		result: () => ({ task: taskName(), arm: meta?.arm ?? null, ...(dual ? { arms } : {}), steps: steps.length, out }),
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
	const unitsFrame = (): Frame => meta?.units_frame ?? "base";
	/** The arm a motion drives: required (and checked) on two arms, none on one. */
	function armName(arm: string | undefined): string | undefined {
		if (!dual) {
			if (arm !== undefined) throw new Error("this Piper robot drives one arm; omit `arm`");
			return undefined;
		}
		if (arm === undefined) throw new Error(`two arms: name the arm (${arms.join(" or ")})`);
		if (!arms.includes(arm)) throw new Error(`unknown arm '${arm}' (have ${arms.join(", ")})`);
		return arm;
	}
	const cameras = () =>
		meta?.cameras?.filter((c) => c === "front" || c.startsWith("wrist")) ??
		(dual ? ["front", "wrist_left", "wrist_right"] : ["front", "wrist"]);

	function call<T = Json>(method: string, kwargs: Json = {}, timeoutMs = 30_000, signal?: AbortSignal) {
		if (!env) throw new Error("piper is not initialized; see the session start error");
		return env.call<T>(method, kwargs, timeoutMs, [], signal);
	}

	/** Refuse a step beyond the limits, then run it on the server (which checks them again). */
	async function guardedStep(
		delta: number[],
		yaw: number,
		gripper: "open" | "close" | null,
		frame: Frame,
		signal?: AbortSignal,
		reopenEmpty = true,
		arm?: string,
		continuous = false,
	): Promise<Json> {
		const side = armName(arm);
		op.check();
		if (signal?.aborted) throw new Error("tool operation interrupted");
		if (delta.length !== 3 || !delta.every(Number.isFinite)) throw new Error("delta must be 3 finite numbers");
		if (!Number.isFinite(yaw)) throw new Error("yaw must be finite");
		checkMove(delta, maxMove());
		if (Math.abs(yaw) > maxYaw())
			throw new Error(`yaw ${round(yaw, 4)} rad exceeds the limit of ${maxYaw()} rad per call.`);
		const kwargs: Json = {
			delta_xyz: delta,
			yaw,
			gripper,
			frame,
			reopen_empty: reopenEmpty,
			// The same MV_* follows in this act call: the server's smooth stream flows through the join.
			...(continuous ? { continuous: true } : {}),
		};
		return call("env.step", side ? { ...kwargs, arm: side } : kwargs, 120_000, signal);
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
			...(dual ? { arm: s.arm, halted: s.halted ?? null } : {}),
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
		for (const name of cameras()) {
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

	/** The step blob plus the front then the wrist image(s) (two arms: left, then right). */
	function view(s: Step): AgentToolResult<unknown> {
		const pngs = cameras()
			.filter((k) => s.images[k])
			.map((k) => readFileSync(s.images[k]));
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
		dual
			? "Read a Piper state step (both arms' eef pose, gripper, limits) and its front, left wrist and right wrist images."
			: "Read a Piper state step (eef pose, gripper, limits) and its front and wrist images.",
		Type.Object({ step: Type.Optional(Type.Integer({ description: "State step (default -1 = latest)" })) }),
		async ({ step = -1 }) => {
			const s = steps[step < 0 ? steps.length + step : step];
			if (!s) return toolResult({ error: `step ${step} is not recorded (have 0..${steps.length - 1})` });
			return view(s);
		},
	);

	const xyz = Type.Array(Type.Number(), { minItems: 3, maxItems: 3 });
	/** Two arms: every motion tool takes the required `arm`; the other arm holds still. */
	const armParam: Record<string, TSchema> = dual
		? {
				arm: StringEnum(arms, { description: "Which arm moves; the other holds still" }),
			}
		: {};
	type ArmP = { arm?: string };
	const armOf = (p: ArmP) => (dual ? { arm: p.arm } : {});
	const the = dual ? "one Piper arm's" : "the Piper";
	robot.tool(
		"move_delta",
		`Move ${the} gripper by a bounded base-frame xyz delta in meters (x forward, y left, z up${dual ? ", in that arm's own base frame" : ""}).`,
		Type.Object({ delta_xyz: xyz, ...armParam }),
		async (p, signal) =>
			act({ action: "move_delta", delta_xyz: p.delta_xyz, ...armOf(p as ArmP) }, () =>
				guardedStep(p.delta_xyz, 0, null, "base", signal, true, (p as ArmP).arm),
			),
	);
	robot.tool(
		"rotate_yaw",
		`Rotate ${the} gripper about the base z axis by a bounded angle in radians.`,
		Type.Object({ yaw: Type.Number(), ...armParam }),
		async (p, signal) =>
			act({ action: "rotate_yaw", yaw: p.yaw, ...armOf(p as ArmP) }, () =>
				guardedStep([0, 0, 0], p.yaw, null, "base", signal, true, (p as ArmP).arm),
			),
	);
	robot.tool(
		"open_gripper",
		`Open ${the} gripper and wait for it to settle.`,
		Type.Object({ ...armParam }),
		async (p, signal) =>
			act({ action: "open_gripper", ...armOf(p as ArmP) }, () =>
				guardedStep([0, 0, 0], 0, "open", "base", signal, true, (p as ArmP).arm),
			),
	);
	robot.tool(
		"close_gripper",
		`Close ${the} gripper and wait for it to settle; an empty close reopens and says so in notes.`,
		Type.Object({ ...armParam }),
		async (p, signal) =>
			act({ action: "close_gripper", ...armOf(p as ArmP) }, () =>
				guardedStep([0, 0, 0], 0, "close", "base", signal, true, (p as ArmP).arm),
			),
	);
	if (dual)
		robot.tool(
			"halt_arm",
			"Stop one arm for the rest of the episode (its part of the task is done, or it is in trouble): the server holds it where it is and refuses its motion until the operator resets it. The other arm keeps working.",
			Type.Object({ ...armParam, reason: Type.String({ description: "Why the arm stops" }) }),
			async (p) =>
				act({ action: "halt_arm", ...armOf(p as ArmP), reason: p.reason }, async () => {
					const side = armName((p as ArmP).arm);
					op.check();
					return call("env.halt_arm", { arm: side ?? null, reason: p.reason });
				}),
		);

	// ---- lifecycle

	/**
	 * Operator-confirmed reset (the start dialog, request_scene_reset): open the gripper(s), move to
	 * the begin pose; both arms on the dual rig. A gripper that holds an object opens only after the
	 * operator confirms it (the server refuses otherwise).
	 */
	async function resetArm(): Promise<Json> {
		const s = await call<Json>("env.get_robot_state");
		const states: Json[] = dual ? Object.values(s.arms ?? {}) : [s];
		const held = states.filter((a) => a.holding_object === true).map((a) => String(a.arm ?? meta?.arm));
		if (held.length) {
			const which = `${held.join(" and ")} gripper${held.length > 1 ? "s hold" : " holds"}`;
			const release = await ui?.confirm(
				"Open a gripper that holds an object?",
				`The ${which} an object; the reset opens it and drops it. Take the object out or secure it, then confirm.`,
			);
			if (!release)
				throw new Error(`reset refused: the ${which} an object and the operator did not confirm opening it`);
		}
		const r = await call<Json>(
			"env.reset",
			{ ...(dual ? { both: true } : {}), ...(held.length ? { release: true } : {}) },
			120_000,
		);
		if (!r.ok) throw new Error(`reset did not reach the begin pose: ${JSON.stringify(plain(r.move))}`);
		await record({ action: "reset" }, r, null);
		return { ok: true, step: steps.length - 1 };
	}

	async function startRobot(ctx: ExtensionContext): Promise<string[]> {
		if (!ctx.hasUI)
			throw new Error("piper drives a real robot: run pi interactively (or over RPC) so an operator is present");
		if (pi.getFlag("operator") !== true)
			throw new Error("piper drives a real robot: start pi with --operator so an operator judges every episode");
		for (const name of ["max-move", "max-yaw"])
			if (!(Number(flag(name)) > 0)) throw new Error(`--${name} must be a positive number (got '${flag(name)}')`);
		// Without the units view hook `act` has no `view`: every move would run in the base frame while
		// the prompt says wrist-judged moves follow the gripper heading.
		if (viewSelect() && !unitsViewSelect)
			throw new Error(
				"--view-select needs the units view hook (act's `view`, UnitsHandle.viewSelect), which the units module in use does not have: every move would run in the base frame. Start without --view-select",
			);
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
		const served = m.arms ?? [];
		if (dual !== served.length > 0)
			throw new Error(
				dual
					? "piper/dual.ts drives both arms, but the env server's config has no `arms:` block (use a dual config, e.g. config/dual_example.yaml, or the single-arm entry packages/embodied/src/piper)"
					: "the env server drives both arms (`arms:` in its config): use packages/embodied/src/piper/dual.ts",
			);
		if (dual && served.join() !== arms.join())
			throw new Error(`the env server drives arms ${served.join(", ")}; expected ${arms.join(", ")}`);
		if (!m.has_begin_pose)
			throw new Error(
				`${dual ? "arms.<side>.calibration" : "calibration"}.begin_joints is not set in the robot config; reset would fail`,
			);
		// Show-Harness refuses view_select with motion_frame: wrist: the heading-frame prompt rewrite would contradict FRONT-guided base-frame moves.
		if (pi.getFlag("view-select") === true && m.units_frame === "heading")
			throw new Error(
				"--view-select picks each move's frame from its guiding view, so the config needs motion.units_frame: base (it is heading)",
			);
		const go = await ctx.ui.confirm(
			dual ? "Move both Piper arms?" : "Move the Piper arm?",
			dual
				? `The left arm, then the right arm, will open its gripper and move to its begin pose, then the agent drives them for: ${t.instruction}. Clear the workspace and keep both emergency stops in reach.`
				: `The ${m.arm} arm will open its gripper and move to its begin pose, then the agent drives it for: ${t.instruction}. Clear the workspace and keep the emergency stop in reach.`,
		);
		if (!go) throw new Error("operator declined the reset; the Piper tools stay disabled");
		env = rpc;
		meta = m;
		ui = ctx.ui;
		try {
			await resetArm();
		} catch (err) {
			env = meta = undefined;
			throw err;
		}
		task = t;
		ctx.ui.notify(`Piper ready: ${taskName()} (${dual ? "both arms" : `${m.arm} arm`}); steps under ${out}`, "info");
		return dual ? [...TOOLS, "halt_arm"] : TOOLS;
	}
}
