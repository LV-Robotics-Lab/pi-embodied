/**
 * One physical AgileX Piper arm, or both arms of the Cobot Magic rig, for pi, ported from
 * Show-Harness (github.com/showlab/Show-Harness).
 *
 *   pi -e packages/embodied/src/robots/piper --operator --task banana_plate --robot-config my_piper.yaml
 *   pi -e packages/embodied/src/robots/piper/dual.ts --operator --task banana_handover --robot-config my_dual.yaml
 *
 * Starts the env server (pi_embodied_services.robots.piper.env_server, ROS topics to the AgileX
 * arm node and the Orbbec cameras) or attaches to one with --env-url. The server owns the
 * safety limits: pi's --max-move / --max-yaw (passed at spawn; an attached server must enforce
 * them or tighter ones, ../../primitives/motion.ts servedLimits) and the robot YAML's per-call
 * step and yaw refusal, the Z floor, the optional workspace box, the joint-stream speed, and the
 * divergence and dropped-gripper guards. The tools and code primitives are the manifest's
 * (../../primitives/manifests/piper.json): move_delta / rotate_yaw / open_gripper /
 * close_gripper run the server methods of the same names, which a program calls too. A real
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
 * Code mode (../code, `--code=true --code-real --operator`): the env server runs with --code; a
 * program's calls meet the same server-side limits as the tools; every program is confirmed by the
 * operator and becomes the next state step.
 *
 * Reset: on two arms the start and the operator's scene reset reset both arms (the operator
 * confirmed it); a gripper that holds an object is opened only after the operator confirms that too.
 */

import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { dir, python, rosSetup, service, servicesDir } from "../../infra/config.ts";
import { trackFlags } from "../../infra/params.ts";
import { encodePng } from "../../infra/png.ts";
import { NdArray, type RpcClient, RpcUnavailable } from "../../infra/rpc.ts";
import {
	compensate,
	type Move,
	type MoveUnit,
	type State,
	UNITS_EVENT,
	type UnitsHandle,
	type UnitsSpec,
	type Vec3,
} from "../../modes/units/index.ts";
import { template } from "../../planner/context-version.ts";
import {
	detectionActive,
	detectionArgs,
	detectionTools,
	type PerceptionCaps,
	registerDetectionFlags,
} from "../../primitives/detections.ts";
import { mountGraspTool } from "../../primitives/grasp.ts";
import { limitArgs, type MotionLimits, servedLimits } from "../../primitives/motion.ts";
import { pointActive, pointTool, registerPointFlags } from "../../primitives/pointing.ts";
import { gripperCommand, poseDelta, toWxyz, type XPolicySpec } from "../../primitives/xpolicy.ts";
import {
	attach,
	defineRobot,
	frameOf,
	type Json,
	message,
	plain,
	type Rgb,
	type RobotSpec,
	rgbOf,
	round,
	type Services,
	servicesEnv,
	toolResult,
	vec,
} from "../../robot.ts";

const SYSTEM = template(new URL("./SYSTEM.md", import.meta.url));
const SYSTEM_DUAL = template(new URL("./SYSTEM_DUAL.md", import.meta.url));
const EXPLORE = template(new URL("./explore.md", import.meta.url));
/** The motion tools: the recipe of a solved exploration attempt. */
const MOTION = ["move_delta", "rotate_yaw", "open_gripper", "close_gripper", "act"];
const SUCCESS_REFUSAL =
	"motion refused: the operator judged this attempt a success. Write the audit and memory drafts, then call finish.";

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
	/** pi's limits as the server enforces them (services utils/code_real.py). */
	motion_limits?: MotionLimits;
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
 * `cameras` are the streamed views in image order (default: the front camera and each arm's wrist
 * camera); a wrist camera the server does not stream is not described, and without any wrist camera
 * there is no view to select.
 */
export function piperViews(
	frame: Frame,
	arms: readonly string[] = [],
	viewSelect = false,
	cameras: readonly string[] = arms.length ? ["front", "wrist_left", "wrist_right"] : ["front", "wrist"],
): string {
	const dual = arms.length > 0;
	const enters = dual ? "arms, which enter" : "arm, which enters";
	const front =
		frame === "heading" && !viewSelect
			? `FRONT camera: faces the ${enters} from the TOP of the image. Moves follow the GRIPPER HEADING, the direction ${dual ? "that arm's" : "the"} gripper points, visible in this view: MV_FWD moves further ahead along the heading, MV_BACK back against it, MV_LEFT / MV_RIGHT to the heading's left / right. Only while the gripper points straight toward the image bottom are these the image bottom / top / left / right.`
			: `FRONT camera: faces the ${enters} from the TOP of the image. MV_FWD moves the gripper toward the image bottom, MV_BACK toward the image top, MV_LEFT / MV_RIGHT toward image left / right.`;
	const wrist =
		"looks along the gripper at the fingertips (bottom of the image). MV_FWD advances the gripper, so a target near the image TOP needs MV_FWD and one between the image top and the fingers MV_BACK; MV_LEFT / MV_RIGHT move toward image left / right; MV_DOWN brings the fingers down onto what is centered between them.";
	const side = (name: string) => arms.find((a) => name === `wrist_${a}`);
	const label = (name: string) => {
		const a = side(name);
		if (a) return `${a.toUpperCase()} WRIST camera (the ${a} arm's)`;
		return name === "wrist" && !dual ? "WRIST camera" : `WRIST camera (${name})`;
	};
	const lines = cameras.flatMap((name, i) =>
		name === "front"
			? [`- Image ${i + 1}, ${front}`]
			: name.startsWith("wrist")
				? [`- Image ${i + 1}, ${label(name)}: ${wrist}`]
				: [],
	);
	const wrists = cameras.filter((c) => c.startsWith("wrist")).length;
	lines.push(
		`- MV_UP / MV_DOWN change the gripper's height in ${wrists ? (wrists > 1 ? "every view" : "both views") : "the front view"}.${dual ? " Each unit moves only the arm named by `arm`." : ""}`,
	);
	if (!wrists) lines.push("- There is no wrist camera: judge every move in the front view.");
	if (viewSelect && wrists)
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
	// Every flag this robot registers is tracked: numbers fail closed, the result records them (../../infra/params.ts).
	trackFlags(pi);
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
	pi.registerFlag("env-url", {
		type: "string",
		description: "Attach to a running Piper env server instead of starting one",
	});
	// --detections / --depth unidepth: the env server's SAM3 masks with ids and UniDepth depth for the webcams (../primitives/detections.ts).
	registerDetectionFlags(pi);
	// --point: Molmo's point over services.molmo (../primitives/pointing.ts).
	registerPointFlags(pi);
	pi.registerFlag("max-move", {
		type: "string",
		default: "0.05",
		description:
			"Largest translation per call, m, enforced by the env server (its limits.max_step_m applies if tighter)",
	});
	pi.registerFlag("max-yaw", {
		type: "string",
		default: "0.2",
		description:
			"Largest |yaw| per call, rad, enforced by the env server (its limits.max_yaw_rad applies if tighter)",
	});
	pi.registerFlag("view-select", {
		type: "boolean",
		default: false,
		description:
			"Units: the model reports each move's guiding view (act's `view`): WRIST runs it in the gripper-heading frame, FRONT in the base frame (Show-Harness plugins/view_select; needs motion.units_frame: base)",
	});

	let env: RpcClient | undefined;
	let meta: Meta | undefined;
	/** The limits the env server enforces (at least pi's flags). */
	let enforced: MotionLimits | undefined;
	let task: Task | undefined;
	let out = "";
	const steps: Step[] = [];
	/** Each camera's latest frame (point reads it). */
	const frames = new Map<string, Rgb>();
	const cameras = () =>
		meta?.cameras?.filter((c) => c === "front" || c.startsWith("wrist")) ??
		(dual ? ["front", "wrist_left", "wrist_right"] : ["front", "wrist"]);
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
		// Two arms: a paired step runs both arms' steps at once on the server (env.step_pair), as
		// Show-Harness's dual runners step both arms together.
		...(dual
			? {
					applyPair: (moves: Move[], signal: AbortSignal | undefined) => {
						const frames = moves.map((m) => motionFrame(m.view, unitsFrame(), viewSelect()));
						return act({ action: "unit_pair", moves }, () => {
							const steps = moves.map((m, i) =>
								stepArgs(m.delta, m.yaw, m.gripper, frames[i], signal, false, m.arm, false),
							);
							lastFrame = frames[0];
							return call("env.step_pair", { steps }, 120_000, signal);
						});
					},
				}
			: {}),
		state: async (arm) => proprio(await call<Json>("env.get_robot_state", dual ? { arm: armName(arm) } : {})),
		// The stall check compares commanded and measured motion in the base frame.
		baseDelta: (delta, state) => (lastFrame === "heading" ? headingToBase(delta, state) : delta),
		instruction: () => task?.instruction ?? "",
		get views() {
			return piperViews(viewSelect() ? "base" : unitsFrame(), arms, viewSelect(), cameras());
		},
		get emptyWidthM() {
			return meta?.limits.empty_width_m ?? 0.005;
		},
		// The server's smooth stream chains `continuous` moves: they return before the arm settles.
		chains: () => chains(meta),
		// The wrist camera(s) the server streams (front + wrist, or front + wrist_left/right, by default).
		wrist: () => cameras().some((c) => c.startsWith("wrist")),
	};
	const spec: RobotSpec = {
		name: dual ? "piper_dual" : "piper",
		// Tools and code primitives: ../../primitives/manifests/piper.json (the env server reads it too).
		manifest: "piper",
		vars: () => ({
			// `arm` exists on the dual rig only (the one-arm tools and primitives leave it out).
			arms: [...arms],
			cameras: cameras(),
			max_move: String(maxMove()),
			max_yaw: String(maxYaw()),
			the: dual ? "one Piper arm's" : "the Piper",
			arm_frame: dual ? ", in that arm's own base frame" : "",
			state_fields: dual ? "both arms' eef pose, gripper, limits" : "eef pose, gripper, limits",
			image_names: dual ? "front, left wrist and right wrist" : "front and wrist",
		}),
		// Must agree with the env server's _has (code mode refuses otherwise).
		capabilities: (c) =>
			({
				dual,
				sam3: pi.getFlag("detections") === true && Boolean(service(pi, "sam3")),
				unidepth: Boolean(String(pi.getFlag("depth") ?? "").trim()),
				xpolicy: Boolean(flag("xpolicy").trim()),
			})[c] ?? false,
		task: ["task"],
		// The env server's primitive registry (code.api), recorded per episode.
		codeApi: () => env,
		// Code mode (../code) on the real arm(s): --code-real and --operator, every program confirmed.
		code: {
			real: true,
			rpc: () => env as RpcClient,
			instruction: () => task?.instruction ?? "",
			refuse: () => (successJudged() ? SUCCESS_REFUSAL : undefined),
			observe: async (r) => {
				for (const f of (r.frames as NdArray[] | undefined) ?? []) robot.video.frame(frameOf(f));
				return view(await record({ action: "run_code" }, { status: r.status, motions: r.motions ?? 0 }, null));
			},
		},
		keepImages: 4,
		video: true,
		// Observations carry cameras() in order; the front view has to lead to be the main one.
		vdm: () => {
			const c = cameras();
			if (c[0] !== "front") return undefined;
			return { views: c.length, wrist: c.flatMap((name, i) => (name.startsWith("wrist") ? [i] : [])) };
		},
		operator: { step: () => steps.length, reset: resetArm },
		// No corpus is published for the Piper: memory is what exploration writes locally; the guard
		// also opens the step images the prompt points at.
		memory: {
			cell: () => ({ tag: `${dual ? "piper_dual" : "piper"}_${taskName()}`, reference: "" }),
			primitives: MOTION,
			readable: () => [out],
			published: false,
		},
		explore: {
			// The operator restores the scene; a failed or unconfirmed reset throws and starts no attempt.
			reset: async (result, ctx, signal) => {
				const r: Json = await op.sceneReset(ctx, String(result.reason ?? ""), "", signal);
				if (r.error) throw new Error(JSON.stringify(r));
				return view(steps[steps.length - 1], {
					...result,
					robot_reset: r.robot_reset,
					scene_reset_confirmed: true,
				});
			},
			prompt: () =>
				EXPLORE.replace(/\{\{(task_id|task_name|instruction)\}\}/g, (_, k: string) =>
					k === "instruction" ? (task?.instruction ?? "") : taskName(),
				),
			rewrite: [
				[
					/^4\. Ask for the operator's verdict \(request_operator_verdict\) when you believe the task is done, then finish\.$/m,
					"4. This is an exploration run: follow the Exploration workflow below. Success is only the operator's verdict.",
				],
			],
			// Every attempt costs the operator a manual scene reset (RPent's real-robot defaults).
			budget: { sessions: 1, attempts: 3 },
			operatorJudged: true,
		},
		start: startRobot,
		stop: () => {
			env = meta = task = enforced = undefined;
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
		// XPolicyLab policies (--xpolicy, the dual rig: env_cfg piper is two-armed): ee targets run as guarded steps;
		// the manifest's xpolicy_act (requires dual, xpolicy) activates the tool.
		...(dual ? { xpolicy: xpolicySpec() } : {}),
	};
	const robot = defineRobot(pi, spec);
	const { op } = robot;

	/** Exploration: the operator judged this attempt a success; no more motion. */
	const successJudged = () => pi.getFlag("explore") === true && (op.result() as Json).operator_verdict === "success";
	/** Code mode is on (--code): the env server serves code.run. */
	const coding = () => (pi.getFlag("code") ?? "false") !== "false";
	/** pi's limits (its flags), which the env server enforces. */
	const wanted = (): MotionLimits => ({
		max_move_m: Number(flag("max-move", "0.05")),
		max_yaw_rad: Number(flag("max-yaw", "0.2")),
	});
	/** The per-call limits in force (for the prompt): the served ones and the config's caps. */
	const maxMove = () =>
		Math.min(enforced?.max_move_m ?? wanted().max_move_m ?? Infinity, meta?.limits.max_step_m ?? Infinity);
	const maxYaw = () =>
		Math.min(enforced?.max_yaw_rad ?? wanted().max_yaw_rad ?? Infinity, meta?.limits.max_yaw_rad ?? Infinity);
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

	function call<T = Json>(method: string, kwargs: Json = {}, timeoutMs = 30_000, signal?: AbortSignal) {
		if (!env) throw new Error("piper is not initialized; see the session start error");
		return env.call<T>(method, kwargs, timeoutMs, [], signal);
	}

	/** One arm's step on the server (which holds it to the limits). */
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
		const kwargs = stepArgs(delta, yaw, gripper, frame, signal, reopenEmpty, arm, continuous);
		return call("env.step", kwargs, 120_000, signal);
	}

	/** The `env.step` arguments of one arm's step, after the gates (the operator's verdict). */
	function stepArgs(
		delta: number[],
		yaw: number,
		gripper: "open" | "close" | null,
		frame: Frame,
		signal: AbortSignal | undefined,
		reopenEmpty: boolean,
		arm: string | undefined,
		continuous: boolean,
	): Json {
		const side = armName(arm);
		gate(signal);
		if (delta.length !== 3 || !delta.every(Number.isFinite)) throw new Error("delta must be 3 finite numbers");
		if (!Number.isFinite(yaw)) throw new Error("yaw must be finite");
		const kwargs: Json = {
			delta_xyz: delta,
			yaw,
			gripper,
			frame,
			reopen_empty: reopenEmpty,
			// The same MV_* follows in this act call: the server's smooth stream flows through the join.
			...(continuous ? { continuous: true } : {}),
		};
		return side ? { ...kwargs, arm: side } : kwargs;
	}

	/** The gates every motion passes before the server call: the operator, an abort, a judged success. */
	function gate(signal: AbortSignal | undefined) {
		op.check();
		if (signal?.aborted) throw new Error("tool operation interrupted");
		if (successJudged()) throw new Error(SUCCESS_REFUSAL);
	}

	/** A motion tool: its manifest params straight to the server method of the same name (after the gates). */
	function motion(method: string, p: Json, signal: AbortSignal | undefined): Promise<Json> {
		const side = armName(p.arm as string | undefined);
		gate(signal);
		const { arm: _arm, ...rest } = p;
		return call(method, side ? { ...rest, arm: side } : rest, 120_000, signal);
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
		let framed = false;
		for (const name of cameras()) {
			const v = obs.images?.[name];
			if (!(v instanceof NdArray)) continue;
			// The episode video follows the first camera (the front one).
			if (!framed) robot.video.frame(frameOf(v));
			framed = true;
			const img = rgbOf(v);
			frames.set(name, img);
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
	function view(s: Step, extra: Json = {}): AgentToolResult<unknown> {
		const pngs = cameras()
			.filter((k) => s.images[k])
			.map((k) => readFileSync(s.images[k]));
		return toolResult({ ...s.blob, ...extra }, pngs);
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

	// The tools' schemas and descriptions are the manifest's (manifests/piper.json).
	robot.tool("view_env_state", "", Type.Object({}), async (p: Json) => {
		const step = typeof p.step === "number" ? p.step : -1;
		const s = steps[step < 0 ? steps.length + step : step];
		if (!s) return toolResult({ error: `step ${step} is not recorded (have 0..${steps.length - 1})` });
		return view(s);
	});

	/** The recorded command of a motion tool: its name and parameters. */
	const command = (action: string, p: Json): Json => ({ action, ...p });
	for (const name of ["move_delta", "rotate_yaw", "open_gripper", "close_gripper"])
		robot.tool(name, "", Type.Object({}), async (p: Json, signal) =>
			act(command(name, p), () => motion(`env.${name}`, p, signal)),
		);
	if (dual)
		// Stopping an arm stays allowed after a judged success (only the operator gate applies).
		robot.tool("halt_arm", "", Type.Object({}), async (p: Json) =>
			act(command("halt_arm", p), async () => {
				const side = armName(p.arm as string | undefined);
				op.check();
				return call("env.halt_arm", { arm: side ?? null, reason: p.reason ?? "" });
			}),
		);

	/**
	 * XPolicyLab (--xpolicy, dual rig: env_cfg piper, two arms): the front camera as cam_head and the wrist
	 * cameras as cam_left_wrist / cam_right_wrist, each arm's joints, gripper width and eef pose. An ee
	 * action runs per arm as one guarded step (the same limits as move_delta): the translation, the
	 * rotation about base z (the Piper step has no roll or pitch, so those are dropped) and an open/close
	 * of the gripper (gripperCommand). No joint actions: the env server has no joint-position command.
	 * XPolicyLab publishes no Piper weights: a policy has to be fine-tuned on this rig's own data.
	 */
	function xpolicySpec(): XPolicySpec {
		const widest: Record<string, number> = {};
		const arm = (s: Json, side: string) => (s.arms?.[side] ?? {}) as Json;
		return {
			envCfgType: "piper",
			actions: ["ee"],
			observe: async () => {
				const obs = await call<Json>("env.get_observation");
				const vision: Record<string, { color: NdArray }> = {};
				const cams = { cam_head: "front", cam_left_wrist: "wrist_left", cam_right_wrist: "wrist_right" };
				for (const [name, cam] of Object.entries(cams))
					if (obs.images?.[cam] instanceof NdArray) vision[name] = { color: obs.images[cam] };
				const state: Record<string, number[]> = {};
				for (const side of arms) {
					const a = arm(obs.robot_state as Json, side);
					widest[side] = Math.max(widest[side] ?? 0, Number(a.gripper_width_m));
					state[`${side}_arm_joint_state`] = vec(a.joints);
					state[`${side}_ee_joint_state`] = [Number(a.gripper_width_m)];
					state[`${side}_ee_pose`] = toWxyz(vec(a.eef_pose));
				}
				return { instruction: task?.instruction ?? "", vision, state };
			},
			act: async (action, signal) => {
				const s = await call<Json>("env.get_robot_state");
				for (const side of arms) {
					const target = action.arms[`${side}_`];
					if (!target) continue;
					const a = arm(s, side);
					const move = target.pose
						? poseDelta(toWxyz(vec(a.eef_pose)), target.pose)
						: { delta: [0, 0, 0], rpy: [0, 0, 0] };
					const grip = target.ee
						? gripperCommand(target.ee[0], widest[side] ?? 0, a.gripper_closed === true)
						: null;
					const r = await guardedStep(move.delta, move.rpy[2], grip, "base", signal, true, side);
					if (r.ok === false) throw new Error(`the ${side} Piper step failed: ${JSON.stringify(plain(r))}`);
				}
			},
			over: () => false,
			present: async (run) => view(await record({ action: "xpolicy_act" }, run, null)),
		};
	}

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

	// Molmo pointing on the current images (active with --point).
	mountGraspTool(
		robot.tool,
		pointTool(pi, {
			cameras: [],
			defaultCamera: () => cameras()[0] ?? "front",
			frame: async (c) => {
				const f = frames.get(c);
				if (!f) throw new Error(`no current image of camera ${c} (cameras: ${cameras().join(", ")})`);
				return f;
			},
			signal: () => robot.signal,
		}),
	);

	// SAM3 masks with ids and UniDepth over the env server's perception, on its latest frames.
	for (const d of detectionTools(pi, {
		call: (method, kwargs, timeoutMs) => call<Json>(method, kwargs, timeoutMs ?? 120_000, robot.signal),
		cameras: [],
		defaultCamera: () => cameras()[0] ?? "front",
	}))
		mountGraspTool(robot.tool, d);

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
		const r: Services = { root: servicesDir(pi), python: python(pi, "piper") };
		const configFlag = flag("robot-config");
		const config = configFlag ? resolve(ctx.cwd, configFlag) : "";
		const stamp = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
		out = resolve(ctx.cwd, dir(pi, "artifacts") || join(tmpdir(), "pi-embodied", `piper_${taskName()}_${stamp}`));
		mkdirSync(out, { recursive: true });
		steps.length = 0;
		const endpoint = flag("env-url");
		const setups = rosSetup(pi);
		const server = [
			"-m",
			"pi_embodied_services.robots.piper.env_server",
			...(config ? ["--robot-config", config] : []),
			// pi's per-call limits, enforced by the server for tools and programs alike.
			...limitArgs(wanted()),
			...detectionArgs(pi, { sam3: true }),
			...(coding() ? ["--code"] : []),
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
		// An attached server must enforce pi's limits (or tighter ones); throws otherwise.
		const limits = servedLimits(m.motion_limits, wanted());
		const t = m.tasks?.[taskName()];
		if (!t?.instruction)
			throw new Error(
				`task '${taskName()}' is not in the robot config's tasks (have: ${Object.keys(m.tasks ?? {}).join(", ") || "none"})`,
			);
		const served = m.arms ?? [];
		if (dual !== served.length > 0)
			throw new Error(
				dual
					? "piper/dual.ts drives both arms, but the env server's config has no `arms:` block (use a dual config, e.g. config/dual_example.yaml, or the single-arm entry packages/embodied/src/robots/piper)"
					: "the env server drives both arms (`arms:` in its config): use packages/embodied/src/robots/piper/dual.ts",
			);
		if (dual && served.join() !== arms.join())
			throw new Error(`the env server drives arms ${served.join(", ")}; expected ${arms.join(", ")}`);
		if (!m.has_begin_pose)
			throw new Error(
				`${dual ? "arms.<side>.calibration" : "calibration"}.begin_joints is not set in the robot config; reset would fail`,
			);
		// view_select chooses between the wrist and the front view: without a wrist camera there is no choice.
		if (pi.getFlag("view-select") === true && !(m.cameras ?? ["wrist"]).some((c) => c.startsWith("wrist")))
			throw new Error(
				`--view-select picks each move's guiding view (WRIST or FRONT), but the env server streams no wrist camera (cameras: ${(m.cameras ?? []).join(", ") || "none"}); start without --view-select`,
			);
		// Show-Harness refuses view_select with motion_frame: wrist: the heading-frame prompt rewrite would contradict FRONT-guided base-frame moves.
		if (pi.getFlag("view-select") === true && m.units_frame === "heading")
			throw new Error(
				"--view-select picks each move's frame from its guiding view, so the config needs motion.units_frame: base (it is heading)",
			);
		// Code mode: a server that would refuse every program fails here, before the operator confirms and the arm resets.
		await robot.codePreflight(rpc);
		const go = await ctx.ui.confirm(
			dual ? "Move both Piper arms?" : "Move the Piper arm?",
			dual
				? `The left arm, then the right arm, will open its gripper and move to its begin pose, then the agent drives them for: ${t.instruction}. Clear the workspace and keep both emergency stops in reach.`
				: `The ${m.arm} arm will open its gripper and move to its begin pose, then the agent drives it for: ${t.instruction}. Clear the workspace and keep the emergency stop in reach.`,
		);
		if (!go) throw new Error("operator declined the reset; the Piper tools stay disabled");
		env = rpc;
		meta = m;
		enforced = limits;
		ui = ctx.ui;
		try {
			await resetArm();
		} catch (err) {
			env = meta = enforced = undefined;
			throw err;
		}
		task = t;
		ctx.ui.notify(`Piper ready: ${taskName()} (${dual ? "both arms" : `${m.arm} arm`}); steps under ${out}`, "info");
		const perception = (m as { capabilities?: { perception?: PerceptionCaps } }).capabilities?.perception;
		return [...TOOLS, ...(dual ? ["halt_arm"] : []), ...detectionActive(pi, perception), ...pointActive(pi)];
	}
}
