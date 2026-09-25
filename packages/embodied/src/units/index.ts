/**
 * Show-Harness action units for any robot: the model drives the arm one semantic unit at a time.
 *
 *   pi -e packages/embodied/src/libero --units=true --suite libero_object_swap --task 0 --seed 1
 *   pi -e packages/embodied/src/libero --units=true --stateless ...   (the paper's no-history setting)
 *   pi -e packages/embodied/src/libero --units=both ...              (units next to the robot's tools)
 *
 * A robot opts in with `units` in its defineRobot spec (base-frame vectors per MV_* unit, the step,
 * an optional yaw step, `apply` and `state`); ../robot.ts mounts this module. `--units=true` hides
 * the robot's own tools: only `act`, `finish` and the enabled plugins' `point` / `plan` remain
 * (Show-Harness's pure mode) and the system prompt is ./SYSTEM.md. `--units=both` adds them to the
 * robot's tools and appends the units section to the robot's prompt. `--stateless` keeps only the
 * first user message and the latest observation turn in context. `act`'s parameters follow the
 * enabled plugins. The episode state (gripper, accumulated yaw, plan, move history) is a
 * `units_state` session entry whenever it changes, rebuilt at session start on resume and fork.
 *
 * Plugins (`--units-plugins`, default the robot's `plugins` or Show-Harness's zero-shot Franka set):
 * - recovery: reopen after a GRASP that closed on nothing.
 * - auto_release: reopen a closed gripper whose object slipped out.
 * - proprioception: height, width and blocked moves in every result.
 * - variable_step: a coarse step for MV_UP, high above the table, or while the target is not in
 *   the wrist view (`act`'s `target_in_wrist`, Show-Harness's `WRIST: YES/NO` marker).
 * - action_chunk: while the target is not in the wrist view, `act` may commit `plan`, up to 3 MV_*
 *   moves run in order.
 * - rotation (robots with a yaw step, which always get ROTATE_CW/CCW, capped at 150 deg accumulated
 *   yaw): wrist-judged MV_* are rotated by the accumulated yaw (the wrist camera turns with the
 *   gripper), and MV_UP while holding first turns back (in commands within the robot's `maxYawRad`).
 * - plan: subgoal stages, with deepplan's REASON checkpoint.
 * - point: affordance pixels -> world xyz (robots with `point`).
 * - mem_text: "Recent moves, newest first" in every result, with the history rules (no oscillation,
 *   no GRASP in place after an empty one); off, the results carry no move history at all.
 *
 * Side VLM calls (./vlm.ts, `--units-vlm-model`, default the session's model):
 * - `--units-verify=true|false|auto` (auto: on for dual-arm robots, as Show-Harness's dual runner): a `finish`
 *   claiming success is checked once against the task on the latest camera images; a NOT complete
 *   verdict refuses it with the verifier's reason and the agent replans (at most once per episode).
 *   Every check is a `units_verify` session entry.
 * - `--units-video-ref <mp4>`: `--units-video-ref-frames` frames sampled uniformly are distilled
 *   into an ordered demo brief (arm, grasp part, destination) that the prompt tells the agent to
 *   replicate; the brief is a `units_video_ref` session entry. A failed extraction fails closed.
 *
 * Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: core/action_units.py, interpreters/ (unit -> base-frame motion) and
 * plugins/{recovery,auto_release,proprioception,variable_step,action_chunk,rotation,affordance,
 * subgoal,deepplan,mem_text} ported as pi tools; the final task check and plugins/video_ref in ./vlm.ts.
 */

import { readFileSync } from "node:fs";
import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import { type ImageContent, StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";
import {
	askVlm,
	type DemoBrief,
	parseJson,
	parseVerdict,
	renderBrief,
	sampleFrames,
	VIDEO_REF_FRAMES,
	validateBrief,
	verifyPrompt,
	videoRefPrompt,
} from "./vlm.ts";

/** `pi.events` channel on which this module publishes the robot's `UnitsHandle` at every session start. */
export const UNITS_EVENT = "pi-embodied:units";
export type UnitsHandle = {
	/** The agent's unit tool (`act`). */
	tool: string;
	/** Arm names on a dual-arm robot (`act`'s `arm`), [] on one arm. */
	arms: readonly string[];
	/** Units `act` accepts. */
	vocabulary: readonly string[];
	stepM: number;
	yawStepRad?: number;
	/** What one `act` call does (grounding, `apply`, recovery / auto_release, the units header), without the model. */
	/** `operator: true` for a human's unit (GUMI): exactly what was pressed, no recovery/auto_release/variable step/rotation assists. */
	run: (
		params: { unit: string; n?: number; arm?: string; operator?: boolean },
		signal?: AbortSignal,
	) => Promise<AgentToolResult<unknown>>;
	/** The robot's proprioception (`eef_xyz`, `gripper_width`, ...), per arm on two arms. */
	state?: (arm?: string) => Promise<Record<string, unknown>>;
	/** Every robot tool (`act` and the robot's own): ../gumi holds them all while the operator drives. */
	tools: () => readonly string[];
	/** Why an operator's unit may not run now (the gates an `act` call passes), else undefined. */
	refuse: () => string | undefined;
	/** view_select is on this session: `act` takes `view` and passes it on as `Move.view`. */
	viewSelect?: boolean;
};

// ---------------------------------------------------------------------------
// the vocabulary (core/action_units.py)

export const MOVE_UNITS = ["MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN"] as const;
export const ROTATE_UNITS = ["ROTATE_CW", "ROTATE_CCW"] as const;
/** STOP holds the setpoint for one step; STILL (dual arm) leaves an arm alone; DONE ends the task. */
export const UNITS = [...MOVE_UNITS, ...ROTATE_UNITS, "STOP", "GRASP", "RELEASE", "DONE", "STILL"] as const;
export type MoveUnit = (typeof MOVE_UNITS)[number];
export type Unit = (typeof UNITS)[number];
export type Vec3 = [number, number, number];

export const PLUGINS = [
	"recovery",
	"auto_release",
	"proprioception",
	"variable_step",
	"action_chunk",
	"rotation",
	"plan",
	"point",
	"mem_text",
] as const;
export type Plugin = (typeof PLUGINS)[number];
/**
 * Show-Harness configs/robot_franka.yaml (zero-shot, mem_text on), plus rotation: here ROTATE_* are offered
 * whenever the robot has a yaw step, and the plugin keeps wrist-judged moves right after a turn.
 * Affordance (point) is off there too.
 */
export const DEFAULT_PLUGINS: readonly Plugin[] = [
	"recovery",
	"auto_release",
	"proprioception",
	"variable_step",
	"action_chunk",
	"rotation",
	"plan",
	"mem_text",
];

/**
 * One grounded unit: a base-frame translation (m), a yaw about base +z (rad), a gripper command.
 * `continuous`: the same MV_* unit runs next in this act call, so the robot may flow through the join.
 */
export type Move = {
	delta: Vec3;
	yaw: number;
	gripper: "open" | "close" | null;
	arm?: string;
	continuous?: boolean;
	/** view_select: the view that guided the move (act's `view`, robots with `viewSelect`). */
	view?: GuideView;
};
/** Show-Harness plugins/view_select: WRIST (rule A, the wrist view) or FRONT (rule B, the third-person view). */
export const GUIDE_VIEWS = ["WRIST", "FRONT"] as const;
export type GuideView = (typeof GUIDE_VIEWS)[number];
type Result = AgentToolResult<unknown>;
export type State = Record<string, unknown>;

export type UnitsSpec = {
	/** Base-frame unit vector of each MV_* unit (calibrated so each matches its look in VIEWS). */
	vectors: Record<MoveUnit, Vec3>;
	/** Metres per MV_* unit (Show-Harness: 0.02). */
	stepM: number;
	/** Radians per ROTATE_CW (ROTATE_CCW is the negative); omit on robots without yaw. */
	yawStepRad?: number;
	/** Execute one move through the robot's own safety checks; return the new observation (images + state). */
	apply: (move: Move, signal: AbortSignal | undefined) => Promise<Result>;
	/**
	 * Proprioception. The plugins read `eef_xyz` (base frame, m), `gripper_width` (m) and
	 * `table_z` (m, optional); everything else is shown as-is.
	 */
	state?: (arm?: string) => Promise<State>;
	/** The task text for the prompt (default: the episode's task flags). */
	instruction?: () => string;
	/** How the camera images look and which way each MV_* unit moves in them (default: DEFAULT_VIEWS). */
	views?: string;
	/** Dual-arm robots: the arm names `act` chooses between. */
	arms?: readonly string[];
	/** A closed gripper at or below this width (m) holds nothing (recovery, auto_release). */
	emptyWidthM?: number;
	/** Default of --units-plugins. */
	plugins?: readonly Plugin[];
	/** variable_step: the coarse step (default 0.04 m) and the "high above the table" gap (default 0.08 m). */
	coarseStepM?: number;
	highAboveTableM?: number;
	/** rotation: +1 rotates wrist-judged moves by +yaw (flip if a post-rotation move goes the wrong way). */
	yawCompensationSign?: number;
	/** The robot's largest yaw per command, rad: longer turns are split into commands within it. */
	maxYawRad?: () => number;
	/** The robot's largest translation per call, m: one `act` call travels at most this in total. */
	maxMoveM?: () => number;
	/**
	 * The robot flows through `continuous` moves: it returns before the arm settles, so the measured
	 * position lags the command and the stall check waits for the chain's last move.
	 */
	chains?: () => boolean;
	/**
	 * view_select (Show-Harness plugins/view_select), read at session start: when true, `act` takes
	 * `view` (which view guided the move) and passes it on in `Move.view`; the robot picks the move's
	 * frame from it and explains it in `views`.
	 */
	viewSelect?: () => boolean;
	/** Robots that execute `delta` in another frame: the base-frame translation it becomes from `state` (proprioception). */
	baseDelta?: (delta: Vec3, state: State | undefined) => Vec3;
	/** Affordance: mark [row, col] fractions (0..1) in a camera image; the marked PNG and world xyz per point. */
	point?: {
		cameras: readonly string[];
		locate: (
			camera: string,
			points: [number, number][],
			signal: AbortSignal | undefined,
		) => Promise<{ image?: Buffer; xyz: (number[] | null)[] }>;
	};
};

/** The robot base's tool registrar (terminate-with-finish, abort signal, env failure). */
export type ToolRegistrar = <P extends TSchema>(
	name: string,
	description: string,
	parameters: P,
	run: (params: Static<P>, signal: AbortSignal | undefined, ctx: ExtensionContext) => Promise<Result>,
) => void;

type Stage = {
	motion: string;
	target: string;
	affordance?: string;
	description?: string;
	completion: string;
	arm?: string;
};
type Target = { label: string; camera: string; point: [number, number]; xyz: number[] | null };

/**
 * Show-Harness's image convention (prompts/controller.txt), which configs/primitives_<robot>.yaml
 * calibrate the unit vectors to: a third-person view facing the robot, then the wrist view.
 */
export const DEFAULT_VIEWS = `Each result shows the third-person view (it faces the robot), then the wrist view (the gripper fingers stay fixed in it).
- Third-person view: MV_LEFT / MV_RIGHT move toward the image left / right, MV_FWD toward the image bottom, MV_BACK toward the image top.
- Wrist view: a target to the fingers' left / right needs MV_LEFT / MV_RIGHT, one near the image bottom and far from the fingers needs MV_FWD, one between the image top and the fingers needs MV_BACK; centered between the fingers: MV_DOWN.`;

/** A MV_* that travelled less than this fraction of the command did not move freely (proprioception). */
const STALL_RATIO = 0.7;
/** Largest repeat count per `act` call. */
const MAX_REPEAT = 10;
/** action_chunk: the most moves one call may commit (action_chunk_step_num). */
const CHUNK_STEPS = 3;
/** rotation: soft guard on the accumulated yaw (Franka joint 7 is about +-166 deg), and "back at neutral". */
const MAX_YAW = (150 * Math.PI) / 180;
const NEUTRAL_YAW = (2 * Math.PI) / 180;
/** The most commands one turn is split into (the 150 deg cap in 5 deg commands). */
const MAX_YAW_PIECES = 30;
/** mem_text: moves the "Recent moves" line shows (configs/robot_franka.yaml mem_text_len). */
const MEM_LEN = 5;
/** mem_text: the history entry of a GRASP that closed on nothing (core/runners/real.py EMPTY_GRASP_LABEL). */
const EMPTY_GRASP = "GRASP(empty)";
/** Verifier: NOT complete verdicts that refuse `finish` per episode (v0.max_replans). */
const MAX_REPLANS = 1;
/** Session entries of the verifier's checks and the video_ref brief. */
export const VERIFY_ENTRY = "units_verify";
export const VIDEO_REF_ENTRY = "units_video_ref";
/** Session entry of the episode state (gripper, accumulated yaw, plan, history), rebuilt on resume and fork. */
export const STATE_ENTRY = "units_state";
const NOTES = {
	empty_grasp: "Empty close; do not retry on an edge/corner. Recenter body and confirm depth.",
	lost_grasp: "Grasp lost; return to GRASP, recenter the object body, then confirm depth.",
};

const TEMPLATE = readFileSync(new URL("./SYSTEM.md", import.meta.url), "utf8").replace(/^<!--[\s\S]*?-->\n/, "");
const r3 = (v: number) => Number(v.toFixed(3));
const text = (s: string) => ({ type: "text" as const, text: s });
const isMove = (u: string): u is MoveUnit => (MOVE_UNITS as readonly string[]).includes(u);

/** Keep every `[name]...[/name]` block when `on`, drop them otherwise. */
function section(prompt: string, name: string, on: boolean) {
	const re = new RegExp(`\\[${name}\\]\\n([\\s\\S]*?)\\[/${name}\\]\\n`, "g");
	return prompt.replace(re, on ? "$1" : "");
}

/** Ground a unit into a move (the interpreters' job), or undefined for units that do not move. */
export function ground(
	spec: Pick<UnitsSpec, "vectors" | "stepM" | "yawStepRad">,
	unit: Unit,
	stepM = spec.stepM,
): Move | undefined {
	if (isMove(unit)) return { delta: spec.vectors[unit].map((x) => x * stepM) as Vec3, yaw: 0, gripper: null };
	if (unit === "ROTATE_CW" || unit === "ROTATE_CCW") {
		if (!spec.yawStepRad) throw new Error(`${unit}: this robot has no yaw`);
		return { delta: [0, 0, 0], yaw: unit === "ROTATE_CW" ? spec.yawStepRad : -spec.yawStepRad, gripper: null };
	}
	if (unit === "GRASP" || unit === "RELEASE")
		return { delta: [0, 0, 0], yaw: 0, gripper: unit === "GRASP" ? "close" : "open" };
	if (unit === "STOP") return { delta: [0, 0, 0], yaw: 0, gripper: null };
	return undefined;
}

/** rotation: a base-frame vector rotated about +z by `yaw` (plugins/rotation compensate_move). */
export function compensate(delta: Vec3, yaw: number): Vec3 {
	if (Math.abs(yaw) < 1e-6) return delta;
	const [c, s] = [Math.cos(yaw), Math.sin(yaw)];
	return [c * delta[0] - s * delta[1], s * delta[0] + c * delta[1], delta[2]];
}

/**
 * The paper's no-history context: the first user message (the task) and the latest observation
 * turn, i.e. everything from the last assistant message whose tool results carry an image.
 */
export function latestTurn<M extends { role: string; content?: unknown }>(messages: M[]): M[] | undefined {
	const hasImage = (m: M) =>
		m.role === "toolResult" &&
		Array.isArray(m.content) &&
		m.content.some((c: { type?: string }) => c?.type === "image");
	let last = -1;
	for (let i = messages.length - 1; i >= 0 && last < 0; i--) {
		if (!hasImage(messages[i])) continue;
		for (let j = i - 1; j >= 0; j--)
			if (messages[j].role === "assistant") {
				last = j;
				break;
			}
	}
	const first = messages.findIndex((m) => m.role === "user");
	if (last < 0 || first < 0 || first >= last) return undefined;
	const kept = [messages[first], ...messages.slice(last)];
	return kept.length === messages.length ? undefined : kept;
}

/** Register the units flags and tools; `mode()`, `tools()` and `prompt()` are read by ../robot.ts. */
export function units(
	pi: ExtensionAPI,
	spec: UnitsSpec,
	tool: ToolRegistrar,
	task: () => Record<string, string> = () => ({}),
	base: Pick<UnitsHandle, "tools" | "refuse"> = { tools: () => ["act"], refuse: () => undefined },
) {
	const instruction = () =>
		spec.instruction?.() ??
		Object.entries(task())
			.map(([k, v]) => `${k} ${v}`)
			.join(", ");
	pi.registerFlag("units", {
		type: "string",
		default: "false",
		description: "Show-Harness action units: true = only act/finish (+ point/plan), both = next to the robot's tools",
	});
	pi.registerFlag("stateless", {
		type: "boolean",
		default: false,
		description: "Units mode: keep only the task and the latest observation turn in context",
	});
	pi.registerFlag("units-plugins", {
		type: "string",
		default: (spec.plugins ?? DEFAULT_PLUGINS).join(","),
		description: `Units plugins, comma-separated (${PLUGINS.join(", ")}; "" = none)`,
	});
	pi.registerFlag("units-coarse-step", {
		type: "string",
		default: String(spec.coarseStepM ?? 0.04),
		description: "variable_step: the coarse MV_* step, m",
	});
	// A string flag: pi sets a boolean flag to true whatever its value, so a default-on check could not be turned off.
	pi.registerFlag("units-verify", {
		type: "string",
		default: "auto",
		description:
			"Units mode: check a success `finish` with one VLM call on the latest images, NOT complete refuses it once (true, false, auto = dual-arm robots)",
	});
	pi.registerFlag("units-vlm-model", {
		type: "string",
		default: "",
		description: "Model (provider/id) of the verifier and video_ref calls (default: the session's model)",
	});
	pi.registerFlag("units-video-ref", {
		type: "string",
		default: "",
		description: "Units mode: a demo video (mp4) distilled into an ordered brief the agent replicates",
	});
	pi.registerFlag("units-video-ref-frames", {
		type: "string",
		default: String(VIDEO_REF_FRAMES),
		description: "video_ref: frames sampled uniformly from the demo video",
	});
	/** "pure" (--units / --units=true), "both", or undefined (off). */
	const mode = (): "pure" | "both" | undefined => {
		const v = pi.getFlag("units");
		if (v === true || v === "true" || v === "pure") return "pure";
		return v === "both" ? "both" : undefined;
	};
	const plugin = (name: Plugin) =>
		String(pi.getFlag("units-plugins") ?? "")
			.split(",")
			.map((s) => s.trim())
			.includes(name) &&
		(name !== "point" || spec.point !== undefined) &&
		(name !== "rotation" || Boolean(spec.yawStepRad));
	const armNames = spec.arms ?? [];
	/** --units-verify: on, off, or auto (Show-Harness's dual runner verifies, its single-arm runners do not). */
	const verifying = () => {
		const v = String(pi.getFlag("units-verify") ?? "auto");
		return v === "auto" ? armNames.length > 1 : v === "true";
	};
	const coarse = () => Number(pi.getFlag("units-coarse-step")) || spec.stepM;
	const high = spec.highAboveTableM ?? 0.08;
	const wristSignal = () => plugin("variable_step") || plugin("action_chunk") || plugin("rotation");
	const vocab = UNITS.filter(
		(u) =>
			(spec.yawStepRad || !(ROTATE_UNITS as readonly string[]).includes(u)) && (u !== "STILL" || armNames.length),
	);

	/** Per-arm episode state ("" = the single arm). */
	let closed = new Map<string, boolean>();
	let yaw = new Map<string, number>();
	let recent: string[] = [];
	let note = "";
	let stages: Stage[] = [];
	let stage = 0;
	let targets: Target[] = [];
	/** Verifier: refusals so far and the latest refusal's reason (shown until a new plan). */
	let replans = 0;
	let verdict = "";
	/** The latest camera images a robot tool returned (the verifier's view). */
	let images: ImageContent[] = [];
	/** video_ref: the brief (extracted once per video and frame count) and why it failed. */
	let demo: { key: string; brief: DemoBrief; indices: number[]; model: string } | undefined;
	let demoError: string | undefined;
	/** A new episode or scene reset: the arm is back at its start heading with the gripper open, no plan. */
	const reset = () => {
		closed = new Map();
		yaw = new Map();
		recent = [];
		note = "";
		stages = [];
		stage = 0;
		targets = [];
		replans = 0;
		verdict = "";
		images = [];
	};
	/** mem_text: record a unit in the move history (newest last). */
	const remember = (u: string) => {
		recent = [...recent, u].slice(-MEM_LEN);
	};
	/** The state a resumed or forked session continues from (not the verifier's images, which the next result renews). */
	const snapshot = () =>
		JSON.stringify({
			closed: Object.fromEntries(closed),
			yaw: Object.fromEntries(yaw),
			recent,
			note,
			stages,
			stage,
			targets,
			replans,
			verdict,
		});
	let saved = snapshot();
	/** Append the state entry when it changed: the accumulated yaw must survive a resume (the wrist stays turned). */
	const save = () => {
		const now = snapshot();
		if (now === saved) return;
		saved = now;
		pi.appendEntry(STATE_ENTRY, JSON.parse(now));
	};
	pi.on("session_start", (_event, ctx) => {
		reset();
		demoError = undefined;
		const branch = ctx.sessionManager.getBranch();
		const custom = (type: string) =>
			branch
				.filter((e) => e.type === "custom" && e.customType === type)
				.map((e) => (e.type === "custom" ? (e.data as Record<string, unknown>) : {}));
		const last = custom(STATE_ENTRY).pop() as Partial<Record<string, unknown>> | undefined;
		if (last) {
			const numbers = (o: unknown) =>
				new Map(
					Object.entries((o ?? {}) as Record<string, unknown>).filter(([, v]) => typeof v === "number"),
				) as Map<string, number>;
			closed = new Map(
				Object.entries((last.closed ?? {}) as Record<string, unknown>).map(([k, v]) => [k, v === true]),
			);
			yaw = numbers(last.yaw);
			recent = Array.isArray(last.recent) ? last.recent.map(String) : [];
			note = String(last.note ?? "");
			stages = Array.isArray(last.stages) ? (last.stages as Stage[]) : [];
			stage = Number(last.stage) || 0;
			targets = Array.isArray(last.targets) ? (last.targets as Target[]) : [];
			replans = Number(last.replans) || 0;
			verdict = String(last.verdict ?? "");
		}
		saved = snapshot();
		// A brief extracted earlier in this branch is reused (same video and frame count).
		const brief = custom(VIDEO_REF_ENTRY)
			.filter((e) => e.brief)
			.pop();
		if (brief)
			demo = {
				key: `${brief.video_path}#${brief.num_frames}`,
				brief: brief.brief as DemoBrief,
				indices: (brief.sampled_indices as number[]) ?? [],
				model: String(brief.model ?? ""),
			};
	});

	/** Proprioception, or undefined when the robot has none or cannot read it now. */
	const read = async (arm: string | undefined) => spec.state?.(arm).catch(() => undefined);
	const width = (s: State | undefined) => (typeof s?.gripper_width === "number" ? s.gripper_width : undefined);
	const eef = (s: State | undefined) =>
		Array.isArray(s?.eef_xyz) && s.eef_xyz.length >= 3 ? (s.eef_xyz as number[]).map(Number) : undefined;
	const gap = (s: State | undefined) => {
		const p = eef(s);
		return p && typeof s?.table_z === "number" ? p[2] - s.table_z : undefined;
	};
	const empty = (s: State | undefined) => {
		const w = width(s);
		return spec.emptyWidthM !== undefined && w !== undefined && w <= spec.emptyWidthM;
	};
	const reopen = async (arm: string | undefined, signal: AbortSignal | undefined) => {
		closed.set(arm ?? "", false);
		return spec.apply({ delta: [0, 0, 0], yaw: 0, gripper: "open", ...(arm ? { arm } : {}) }, signal);
	};

	/** variable_step: coarse for MV_UP, high above the table, or the target not in the wrist view. */
	const stepFor = (unit: MoveUnit, st: State | undefined, inWrist: boolean | undefined) => {
		if (!plugin("variable_step")) return spec.stepM;
		const g = gap(st);
		return unit === "MV_UP" || (g !== undefined && g > high) || inWrist === false ? coarse() : spec.stepM;
	};

	/** The units block that heads every `act` result. */
	async function header(lines: string[], arm: string | undefined) {
		const out = [...lines];
		if (plugin("plan") && stages.length) {
			const s = stages[stage];
			out.push(
				s
					? `STAGE ${stage + 1}/${stages.length} [${s.motion}]${s.arm ? ` (${s.arm} arm)` : ""}: target ${s.target}${s.affordance ? `; affordance ${s.affordance}` : ""}${s.description ? `; ${s.description}` : ""}; DONE WHEN ${s.completion}`
					: "STAGE: all planned stages are done; check the task and DONE, or send a new plan.",
			);
			if (s?.motion.toUpperCase() === "REASON")
				out.push("Decision point: judge the rule from the images now and send the concrete stages with `plan`.");
		}
		const st = await read(arm);
		if (plugin("proprioception") && st) {
			const p = eef(st);
			const g = gap(st);
			const w = width(st);
			const held = closed.get(arm ?? "") === true;
			const parts: string[] = [];
			if (g !== undefined) parts.push(`the gripper is ${(g * 100).toFixed(1)} cm above the table`);
			else if (p) parts.push(`gripper at [${p.map(r3).join(", ")}] m`);
			if (w !== undefined) parts.push(`width ${(w * 100).toFixed(1)} cm, commanded ${held ? "CLOSE" : "OPEN"}`);
			parts.push(
				plugin("variable_step")
					? `each step moves ~${(spec.stepM * 100).toFixed(0)} cm (${(coarse() * 100).toFixed(0)} cm)`
					: `each step moves ~${(spec.stepM * 100).toFixed(0)} cm`,
			);
			out.push(`Proprioception: ${parts.join("; ")}.`);
			if (g !== undefined)
				out.push(
					held
						? "Holding an object: lift until clear of the table; descend only to place."
						: `If height > ${(high * 100).toFixed(0)} cm, MV_DOWN first.`,
				);
		}
		if (plugin("rotation") && Math.abs(yaw.get(arm ?? "") ?? 0) > NEUTRAL_YAW)
			out.push(
				`Gripper turned ${Math.round(((yaw.get(arm ?? "") ?? 0) * 180) / Math.PI)} deg from its start heading.`,
			);
		if (plugin("point") && targets.length) {
			const p = eef(st);
			for (const t of targets)
				out.push(
					`Point ${t.label} (${t.camera} [${t.point}]): ${t.xyz ? `xyz [${t.xyz.map(r3).join(", ")}] m${p ? `, offset from gripper [${t.xyz.map((v, i) => r3(v - p[i])).join(", ")}] m` : ""}` : "no depth"}`,
				);
		}
		if (note) out.push(`Recovery: ${note}`);
		if (verdict) out.push(`Verifier: the task is NOT complete: ${verdict}`);
		if (plugin("mem_text")) out.push(`Recent moves, newest first: ${[...recent].reverse().join(", ") || "none"}`);
		out.push(`TASK: ${instruction()}`);
		return text(out.join("\n"));
	}

	/** `act`'s parameters for the enabled plugins (a disabled plugin's parameter is not offered). */
	function actSchema() {
		const props: Record<string, TSchema> = {
			unit: StringEnum(vocab, { description: "The action unit" }),
			n: Type.Optional(Type.Integer({ minimum: 1, maximum: MAX_REPEAT, description: "Repeat count (default 1)" })),
		};
		if (wristSignal())
			props.target_in_wrist = Type.Optional(
				Type.Boolean({ description: "WRIST CHECK: is the TARGET visible in the wrist view?" }),
			);
		if (plugin("action_chunk"))
			props.plan = Type.Optional(
				Type.Array(StringEnum(MOVE_UNITS), {
					maxItems: CHUNK_STEPS,
					description: `Only with target_in_wrist false: up to ${CHUNK_STEPS} MV_* moves run in order (replaces unit and n)`,
				}),
			);
		if (armNames.length) props.arm = StringEnum(armNames, { description: "Which arm the unit drives" });
		if (spec.viewSelect?.())
			props.view = Type.Optional(
				StringEnum(GUIDE_VIEWS, {
					description:
						"VIEW SELECT: the view that guided this move, WRIST (the wrist view) or FRONT (the front view)",
				}),
			);
		return Type.Object(props);
	}
	let registered = "";
	/** (Re-)register `act` when its schema changed: at load from the flag defaults, at session start from the flags. */
	function registerAct() {
		const schema = actSchema();
		if (JSON.stringify(schema) === registered) return;
		registered = JSON.stringify(schema);
		tool(
			"act",
			`Execute one action unit (${vocab.join(", ")}), repeated n times. MV_* move the gripper ~${Math.round(spec.stepM * 100)} cm${spec.yawStepRad ? `, ROTATE_* turn it ~${Math.round((spec.yawStepRad * 180) / Math.PI)} deg` : ""}; GRASP closes, RELEASE opens, STOP holds one step, DONE means the task is complete (call finish). Returns the new images and state.`,
			schema,
			(params, signal) => actAndSave(params as ActParams, signal),
		);
	}
	registerAct();
	pi.on("session_start", registerAct);

	type ActParams = {
		unit: string;
		n?: number;
		arm?: string;
		target_in_wrist?: boolean;
		plan?: string[];
		operator?: boolean;
		view?: string;
	};
	/** `act`, then the state entry (the agent's and the operator's units both change the episode state). */
	async function actAndSave(params: ActParams, signal: AbortSignal | undefined) {
		try {
			return await act(params, signal);
		} finally {
			save();
		}
	}
	/** `act`'s body; ../gumi (dashboard teleop, DAgger takeover) runs it too, through the handle below. */
	async function act(params: ActParams, signal: AbortSignal | undefined): Promise<Result> {
		const p = params as {
			unit: Unit;
			n?: number;
			arm?: string;
			target_in_wrist?: boolean;
			plan?: MoveUnit[];
			operator?: boolean;
			view?: GuideView;
		};
		const { unit, arm, target_in_wrist: inWrist } = p;
		// A human's unit (GUMI teleop) runs as pressed: the agent-side assists would override the operator
		// (recovery reopens a GRASP that closed on air), as in Show-Harness's collectors.
		const assist = (name: Plugin) => !p.operator && plugin(name);
		const key = arm ?? "";
		// The schema offers these only with their plugins; a stale or hand-written call is refused, not silently dropped.
		if (p.plan !== undefined && !plugin("action_chunk"))
			throw new Error("act: `plan` needs the action_chunk plugin (--units-plugins)");
		if (p.view !== undefined && !(spec.viewSelect?.() && (GUIDE_VIEWS as readonly string[]).includes(p.view)))
			throw new Error("act: `view` needs the robot's view select (WRIST or FRONT)");
		if (p.target_in_wrist !== undefined && !wristSignal())
			throw new Error(
				"act: `target_in_wrist` needs the variable_step, action_chunk or rotation plugin (--units-plugins)",
			);
		if (demoError) return { content: [text(`video_ref failed: ${demoError}`)], details: { unit, error: demoError } };
		if (unit === "DONE")
			return {
				content: [text("DONE: if the images show the task complete, call `finish` now; otherwise keep acting.")],
				details: { unit },
			};
		if (unit === "STILL") return { content: [text(`STILL: the ${arm ?? ""} arm holds.`)], details: { unit } };
		const lines: string[] = [];
		let queue: Unit[] = Array(Math.max(1, Math.min(MAX_REPEAT, Math.floor(p.n ?? 1)))).fill(unit);
		if (p.plan?.length && plugin("action_chunk")) {
			if (inWrist === false) queue = p.plan.filter(isMove).slice(0, CHUNK_STEPS);
			else lines.push("plan ignored: plans run only while target_in_wrist is false; one unit ran.");
			if (inWrist !== false) queue = [unit];
		}
		let last: Result | undefined;
		const ran: string[] = [];
		const halted = (r: Result) => {
			const d = r.details as { error?: unknown; terminated?: unknown } | undefined;
			return Boolean(d?.error || d?.terminated);
		};
		/** Path length so far: one call travels at most the robot's per-call translation limit. */
		let travelled = 0;
		// A recovery note lasts until the next GRASP.
		if (queue.includes("GRASP")) note = "";
		for (const [index, u] of queue.entries()) {
			const before = await read(arm);
			const acc = yaw.get(key) ?? 0;
			let label: string = u;
			let move = ground(spec, u, isMove(u) && !p.operator ? stepFor(u, before, inWrist) : spec.stepM) as Move;
			// Holding and turned: MV_UP first turns back to the start heading.
			if (assist("rotation") && u === "MV_UP" && closed.get(key) && Math.abs(acc) > NEUTRAL_YAW) {
				move = { delta: [0, 0, 0], yaw: -acc, gripper: null };
				label = "MV_UP(realign)";
			} else if (assist("rotation") && isMove(u) && inWrist !== false)
				move.delta = compensate(move.delta, (spec.yawCompensationSign ?? 1) * acc);
			else if (move.yaw && Math.abs(acc + move.yaw) > MAX_YAW) {
				// The accumulated-yaw guard holds with or without the rotation plugin.
				lines.push(`${u} refused: the gripper is already turned ${Math.round((acc * 180) / Math.PI)} deg.`);
				break;
			}
			const dist = Math.hypot(...move.delta);
			const maxMove = spec.maxMoveM?.();
			if (maxMove !== undefined && travelled > 0 && !(travelled + dist <= maxMove + 1e-9)) {
				lines.push(`${u} not run: one act call moves at most ${maxMove} m in total.`);
				break;
			}
			// A turn beyond the robot's per-command limit runs as equal commands within it.
			const maxYaw = spec.maxYawRad?.();
			const parts = move.yaw && maxYaw !== undefined ? Math.ceil(Math.abs(move.yaw) / maxYaw - 1e-9) : 1;
			if (!(parts >= 1 && parts <= MAX_YAW_PIECES)) {
				lines.push(`${label} refused: the robot's per-call rotation limit is ${maxYaw} rad.`);
				break;
			}
			if (arm) move.arm = arm;
			if (p.view) move.view = p.view;
			if (isMove(u) && label === u && queue[index + 1] === u) move.continuous = true;
			for (let i = 0; i < parts; i++) {
				const piece: Move = i ? { ...move, delta: [0, 0, 0], gripper: null } : move;
				last = await spec.apply(parts > 1 ? { ...piece, yaw: move.yaw / parts } : piece, signal);
				if (halted(last)) break;
				if (move.yaw) yaw.set(key, (yaw.get(key) ?? 0) + move.yaw / parts);
			}
			travelled += dist;
			ran.push(label);
			// mem_text records moves, turns and grasps (not STOP / RELEASE), as core/runners/real.py.
			if (isMove(u) || u === "GRASP" || u === "ROTATE_CW" || u === "ROTATE_CCW") remember(u);
			if (move.gripper) closed.set(key, move.gripper === "close");
			const after = await read(arm);
			if (last && halted(last)) break;
			// proprioception: a MV_* that barely moved is blocked (contact, floor, workspace limit).
			const [p0, p1] = [eef(before), eef(after)];
			const base = spec.baseDelta?.(move.delta, before) ?? move.delta;
			const commanded = Math.hypot(...base);
			// A chained move returned before the arm settled: its lagging position is not a stall.
			const settling = move.continuous === true && spec.chains?.() === true;
			if (plugin("proprioception") && !settling && p0 && p1 && commanded > 0) {
				const moved = [0, 1, 2].reduce((s, k) => s + (p1[k] - p0[k]) * base[k], 0) / commanded;
				if (moved < commanded * STALL_RATIO) {
					lines.push(
						u === "MV_DOWN"
							? `Last MV_DOWN lowered ${(moved * 100).toFixed(1)} of ${(commanded * 100).toFixed(1)} cm -> already in contact, do NOT MV_DOWN again`
							: `Last ${u} moved ${(moved * 100).toFixed(1)} of ${(commanded * 100).toFixed(1)} cm -> blocked`,
					);
					break;
				}
			}
			// recovery: a GRASP that closed on nothing is reopened at once.
			if (u === "GRASP" && assist("recovery") && empty(after)) {
				last = await reopen(arm, signal);
				// The fresh approach starts from a clean history holding only the failed GRASP.
				recent = [EMPTY_GRASP];
				note = NOTES.empty_grasp;
				break;
			}
			// auto_release: a closed gripper that collapsed (the object slipped out) is reopened.
			if (u !== "GRASP" && assist("auto_release") && closed.get(key) && empty(after)) {
				last = await reopen(arm, signal);
				note = NOTES.lost_grasp;
				break;
			}
		}
		const what =
			queue.every((u) => u === queue[0]) && ran.every((u) => u === ran[0])
				? `${ran[0] ?? queue[0]} x${ran.length}`
				: ran.join(", ");
		lines.unshift(
			`units: ${what}${arm ? ` (${arm} arm)` : ""}${ran.length < queue.length ? ` of ${queue.length} (stopped early)` : ""}`,
		);
		if (!last) return { content: [await header(lines, arm)], details: { unit } };
		return { ...last, content: [await header(lines, arm), ...last.content] };
	}
	// The robot's unit layer for ../gumi, published every session (the dashboard operator drives through it).
	const handle: UnitsHandle = {
		tool: "act",
		arms: armNames,
		vocabulary: vocab,
		stepM: spec.stepM,
		yawStepRad: spec.yawStepRad,
		run: actAndSave,
		state: spec.state,
		viewSelect: false,
		...base,
	};
	pi.on("session_start", () => pi.events.emit(UNITS_EVENT, { ...handle, viewSelect: spec.viewSelect?.() === true }));

	if (spec.point) {
		const { cameras, locate } = spec.point;
		tool(
			"point",
			"Affordance: mark the exact gripper contact point(s) in one camera's current image, [y, x] on a 0-1000 grid (y from the top, x from the left). Returns the marked image and the world xyz per point where the robot has depth; later act results report the gripper-to-point offset. A new call with the same label replaces that point.",
			Type.Object({
				camera: StringEnum(cameras, { description: `Camera (${cameras.join(", ")})` }),
				points: Type.Array(
					Type.Object({
						label: Type.String({ description: "What the point is, e.g. 'bowl rim' or 'place spot'" }),
						yx: Type.Array(Type.Number({ minimum: 0, maximum: 1000 }), { minItems: 2, maxItems: 2 }),
					}),
					{ minItems: 1, maxItems: 4 },
				),
			}),
			async ({ camera, points }, signal) => {
				const fr = points.map((q) => [q.yx[0] / 1000, q.yx[1] / 1000] as [number, number]);
				const { image, xyz } = await locate(camera, fr, signal);
				const marked = points.map((q, i) => ({
					label: q.label,
					camera,
					point: [Math.round(q.yx[0]), Math.round(q.yx[1])] as [number, number],
					xyz: xyz[i] ? xyz[i].map(r3) : null,
				}));
				targets = [...targets.filter((t) => !marked.some((m) => m.label === t.label)), ...marked].slice(-4);
				save();
				const content: Result["content"] = [text(JSON.stringify({ points: marked }))];
				if (image) content.push({ type: "image", data: image.toString("base64"), mimeType: "image/png" });
				return { content, details: { points: marked } };
			},
		);
	}

	tool(
		"plan",
		"Subgoal plan: `stages` replaces the plan from the current stage on (ordered GRASP/LIFT/MOVE/PLACE/RELEASE/RETREAT/REASON stages, each with a visible DONE WHEN); `done: true` marks the current stage complete. The current stage is shown in every act result.",
		Type.Object({
			stages: Type.Optional(
				Type.Array(
					Type.Object({
						motion: Type.String({ description: "GRASP, LIFT, MOVE, PLACE, RELEASE, RETREAT or REASON" }),
						target: Type.String(),
						affordance: Type.Optional(Type.String({ description: "The one visible part to aim at" })),
						description: Type.Optional(
							Type.String({ description: "Visual strategy; for REASON the complete IF ... THEN rule" }),
						),
						completion: Type.String({ description: "DONE WHEN: a condition visible in the images" }),
						arm: Type.Optional(Type.String({ description: "Dual-arm robots: the arm of this stage" })),
					}),
				),
			),
			done: Type.Optional(Type.Boolean({ description: "The current stage's DONE WHEN is visible" })),
		}),
		async ({ stages: next, done }) => {
			if (done && stage < stages.length) stage++;
			if (next?.length) stages = [...stages.slice(0, stage), ...next];
			// A new stage starts with a clean move history (core/runners/real.py); a new plan answers the verifier.
			if (done || next?.length) recent = [];
			if (next?.length) verdict = "";
			save();
			const lines = stages.map(
				(s, i) =>
					`${i === stage ? ">" : i < stage ? "x" : " "} ${i + 1}. [${s.motion}] ${s.target}: ${s.completion}`,
			);
			return { content: [text(lines.length ? lines.join("\n") : "No plan yet.")], details: { stage, stages } };
		},
	);

	// The verifier judges the latest camera images: the newest robot tool result that carries any
	// (`point`'s marked image is not a camera view).
	pi.on("tool_result", (event) => {
		if (!mode() || event.toolName === "point" || !Array.isArray(event.content)) return undefined;
		const shown = event.content.filter((c): c is ImageContent => c.type === "image");
		if (shown.length) images = shown;
		return undefined;
	});

	// The final task check (core/runners/dual.py _completion_outcome): a success claim is judged once
	// more from the images; NOT complete refuses `finish` (with the reason) while the replan budget lasts.
	pi.on("tool_call", async (event, ctx) => {
		if (event.toolName !== "finish" || !mode() || !verifying()) return undefined;
		const status = (event.input as { status?: unknown }).status;
		if (status !== undefined && status !== "success") return undefined;
		const entry: Record<string, unknown> = { status, replans, cameras: images.length };
		let refuse = false;
		if (!images.length) entry.skipped = "no camera images yet";
		else {
			const started = Date.now();
			try {
				const reply = await askVlm(
					ctx,
					String(pi.getFlag("units-vlm-model") ?? ""),
					pi.getThinkingLevel(),
					verifyPrompt(instruction(), armNames, images.length),
					images,
					ctx.signal,
				);
				const v = parseVerdict(reply.text);
				Object.assign(entry, { model: reply.model, complete: v.complete, reason: v.reason, raw: reply.text });
				if (!v.available) entry.unavailable = true;
				refuse = !v.complete && replans < MAX_REPLANS;
			} catch (err) {
				// A verifier failure never turns a finished episode into a replan.
				Object.assign(entry, { complete: true, error: err instanceof Error ? err.message : String(err) });
			}
			entry.ms = Date.now() - started;
		}
		entry.refused = refuse;
		pi.appendEntry(VERIFY_ENTRY, entry);
		if (!refuse) return undefined;
		replans++;
		verdict = String(entry.reason ?? "");
		// Replan from the live scene: the old plan and move history no longer apply.
		stages = [];
		stage = 0;
		recent = [];
		note = "";
		save();
		return {
			block: true,
			reason: `finish refused: the verifier judged the task NOT complete (${verdict}). Replan the remaining work from the live images${plugin("plan") ? " (send new stages with `plan`)" : ""} and continue with \`act\`; call \`finish\` again once the task is visibly complete. This is the only refusal.`,
		};
	});

	// video_ref: extract the demo brief once, before the first prompt that needs it (plugins/video_ref).
	pi.on("before_agent_start", async (_event, ctx) => {
		const path = String(pi.getFlag("units-video-ref") ?? "");
		if (!mode() || !path) return undefined;
		const frames = Number(pi.getFlag("units-video-ref-frames")) || VIDEO_REF_FRAMES;
		const key = `${path}#${frames}`;
		if (demo?.key === key || demoError) return undefined;
		try {
			const { indices, images: shown } = await sampleFrames(path, frames, String(pi.getFlag("ffmpeg") || "ffmpeg"));
			const prompt = videoRefPrompt(shown.length, armNames);
			const errors: string[] = [];
			// A reply that is not a usable brief is asked once more (their guided-JSON try, then free JSON).
			for (let i = 0; i < 2 && demo?.key !== key; i++) {
				const reply = await askVlm(
					ctx,
					String(pi.getFlag("units-vlm-model") ?? ""),
					pi.getThinkingLevel(),
					prompt,
					shown,
					ctx.signal,
				);
				try {
					demo = { key, brief: validateBrief(parseJson(reply.text), armNames), indices, model: reply.model };
				} catch (err) {
					errors.push(err instanceof Error ? err.message : String(err));
				}
			}
			if (!demo || demo.key !== key) throw new Error(errors.join(" | "));
			pi.appendEntry(VIDEO_REF_ENTRY, {
				video_path: path,
				num_frames: frames,
				sampled_indices: demo.indices,
				model: demo.model,
				brief: demo.brief,
			});
		} catch (err) {
			// A replication run without the brief would silently become ordinary planning: fail closed.
			demoError = `could not extract a demo brief from ${path}: ${err instanceof Error ? err.message : String(err)}`;
			pi.appendEntry(VIDEO_REF_ENTRY, { video_path: path, num_frames: frames, error: demoError });
			if (ctx.hasUI) ctx.ui.notify(`video_ref: ${demoError}`, "error");
			else {
				console.error(`[units] video_ref: ${demoError}`);
				process.exitCode = 1;
				ctx.shutdown();
			}
		}
		return undefined;
	});

	pi.on("context", (event) => {
		if (!mode() || pi.getFlag("stateless") !== true) return undefined;
		const kept = latestTurn(event.messages);
		return kept ? { messages: kept } : undefined;
	});

	return {
		mode,
		/** A scene reset: the arm is back at its start heading with the gripper open, no plan. */
		reset: () => {
			reset();
			save();
		},
		/** The units tools: act and the enabled plugins' tools (pure mode adds finish). */
		tools: () => ["act", ...(plugin("point") ? ["point"] : []), ...(plugin("plan") ? ["plan"] : [])],
		/** Pure mode: the whole prompt. Both mode: the section appended to the robot's prompt. */
		prompt: () => {
			const m = mode();
			let p = section(TEMPLATE, "pure", m === "pure");
			p = section(p, "both", m === "both");
			p = section(p, "arms", armNames.length > 0);
			p = section(p, "yaw", Boolean(spec.yawStepRad));
			p = section(p, "wrist", wristSignal());
			for (const name of PLUGINS)
				p = section(
					p,
					name,
					plugin(name) && (!["recovery", "auto_release"].includes(name) || spec.emptyWidthM !== undefined),
				);
			p = section(p, "stateless", pi.getFlag("stateless") === true);
			const brief = demo?.key.startsWith(`${pi.getFlag("units-video-ref")}#`) ? demo.brief : undefined;
			p = section(p, "video_ref", brief !== undefined);
			const vars: Record<string, string> = {
				arm: armNames.length ? `with ${armNames.length} arms` : "arm",
				task: instruction(),
				views: (spec.views ?? DEFAULT_VIEWS).trim(),
				step_cm: (spec.stepM * 100).toFixed(0),
				coarse_cm: (coarse() * 100).toFixed(0),
				high_cm: (high * 100).toFixed(0),
				chunk: String(CHUNK_STEPS),
				yaw_deg: String(Math.round(((spec.yawStepRad ?? 0) * 180) / Math.PI)),
				arms: armNames.join(", "),
				proprio_note: plugin("proprioception") ? ", the gripper's height and width, blocked moves" : "",
				mem_note: plugin("mem_text") ? ", the recent moves (newest first)" : "",
				video_ref: brief ? renderBrief(brief, armNames) : "",
			};
			return p.replace(/\{\{(\w+)\}\}/g, (m, k: string) => vars[k] ?? m).trim();
		},
	};
}
