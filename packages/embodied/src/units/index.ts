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
 * first user message and the latest observation turn in context.
 *
 * Plugins (`--units-plugins`, default the robot's `plugins` or Show-Harness's zero-shot Franka set):
 * - recovery: reopen after a GRASP that closed on nothing.
 * - auto_release: reopen a closed gripper whose object slipped out.
 * - proprioception: height, width and blocked moves in every result.
 * - variable_step: a coarse step for MV_UP, high above the table, or while the target is not in
 *   the wrist view (`act`'s `target_in_wrist`, Show-Harness's `WRIST: YES/NO` marker).
 * - action_chunk: while the target is not in the wrist view, `act` may commit `plan`, up to 3 MV_*
 *   moves run in order.
 * - rotation (robots with a yaw step, which always get ROTATE_CW/CCW): wrist-judged MV_* are rotated
 *   by the accumulated yaw (the wrist camera turns with the gripper), MV_UP while holding first turns
 *   back, and a soft guard caps the accumulated yaw at 150 deg.
 * - plan: subgoal stages, with deepplan's REASON checkpoint.
 * - point: affordance pixels -> world xyz (robots with `point`).
 *
 * Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: core/action_units.py, interpreters/ (unit -> base-frame motion) and
 * plugins/{recovery,auto_release,proprioception,variable_step,action_chunk,rotation,affordance,
 * subgoal,deepplan} ported as pi tools.
 */

import { readFileSync } from "node:fs";
import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";

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
	run: (params: { unit: string; n?: number; arm?: string }, signal?: AbortSignal) => Promise<AgentToolResult<unknown>>;
	/** The robot's proprioception (`eef_xyz`, `gripper_width`, ...), per arm on two arms. */
	state?: (arm?: string) => Promise<Record<string, unknown>>;
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
] as const;
export type Plugin = (typeof PLUGINS)[number];
/**
 * Show-Harness configs/robot_franka.yaml (zero-shot), plus rotation: here ROTATE_* are offered
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
];

/** One grounded unit: a base-frame translation (m), a yaw about base +z (rad), a gripper command. */
export type Move = { delta: Vec3; yaw: number; gripper: "open" | "close" | null; arm?: string };
type Result = AgentToolResult<unknown>;
type State = Record<string, unknown>;

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
	pi.on("session_start", () => {
		closed = new Map();
		yaw = new Map();
		recent = [];
		note = "";
		stages = [];
		stage = 0;
		targets = [];
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
		out.push(`Recent units: ${recent.slice(-5).join(", ") || "none"}`);
		out.push(`TASK: ${instruction()}`);
		return text(out.join("\n"));
	}

	const props: Record<string, TSchema> = {
		unit: Type.Union(
			vocab.map((u) => Type.Literal(u)),
			{ description: "The action unit" },
		),
		n: Type.Optional(Type.Integer({ minimum: 1, maximum: MAX_REPEAT, description: "Repeat count (default 1)" })),
	};
	if (wristSignal())
		props.target_in_wrist = Type.Optional(
			Type.Boolean({ description: "WRIST CHECK: is the TARGET visible in the wrist view?" }),
		);
	if (plugin("action_chunk"))
		props.plan = Type.Optional(
			Type.Array(Type.Union(MOVE_UNITS.map((u) => Type.Literal(u))), {
				maxItems: CHUNK_STEPS,
				description: `Only with target_in_wrist false: up to ${CHUNK_STEPS} MV_* moves run in order (replaces unit and n)`,
			}),
		);
	if (armNames.length)
		props.arm = Type.Union(
			armNames.map((a) => Type.Literal(a)),
			{ description: "Which arm the unit drives" },
		);
	tool(
		"act",
		`Execute one action unit (${vocab.join(", ")}), repeated n times. MV_* move the gripper ~${Math.round(spec.stepM * 100)} cm${spec.yawStepRad ? `, ROTATE_* turn it ~${Math.round((spec.yawStepRad * 180) / Math.PI)} deg` : ""}; GRASP closes, RELEASE opens, STOP holds one step, DONE means the task is complete (call finish). Returns the new images and state.`,
		Type.Object(props),
		(params, signal) => act(params as ActParams, signal),
	);

	type ActParams = { unit: string; n?: number; arm?: string; target_in_wrist?: boolean; plan?: string[] };
	/** `act`'s body; ../gumi (dashboard teleop, DAgger takeover) runs it too, through the handle below. */
	async function act(params: ActParams, signal: AbortSignal | undefined): Promise<Result> {
		const p = params as { unit: Unit; n?: number; arm?: string; target_in_wrist?: boolean; plan?: MoveUnit[] };
		const { unit, arm, target_in_wrist: inWrist } = p;
		const key = arm ?? "";
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
		// A recovery note lasts until the next GRASP.
		if (queue.includes("GRASP")) note = "";
		for (const u of queue) {
			const before = await read(arm);
			const acc = yaw.get(key) ?? 0;
			let label: string = u;
			let move = ground(spec, u, isMove(u) ? stepFor(u, before, inWrist) : spec.stepM) as Move;
			if (plugin("rotation")) {
				// Holding and turned: MV_UP first turns back to the start heading.
				if (u === "MV_UP" && closed.get(key) && Math.abs(acc) > NEUTRAL_YAW) {
					move = { delta: [0, 0, 0], yaw: -acc, gripper: null };
					label = "MV_UP(realign)";
				} else if (isMove(u) && inWrist !== false)
					move.delta = compensate(move.delta, (spec.yawCompensationSign ?? 1) * acc);
				else if (move.yaw && Math.abs(acc + move.yaw) > MAX_YAW) {
					lines.push(`${u} refused: the gripper is already turned ${Math.round((acc * 180) / Math.PI)} deg.`);
					break;
				}
			}
			if (arm) move.arm = arm;
			last = await spec.apply(move, signal);
			ran.push(label);
			recent.push(label);
			if (move.yaw) yaw.set(key, acc + move.yaw);
			if (move.gripper) closed.set(key, move.gripper === "close");
			const after = await read(arm);
			const details = last.details as { error?: unknown; terminated?: unknown } | undefined;
			if (details?.error || details?.terminated) break;
			// proprioception: a MV_* that barely moved is blocked (contact, floor, workspace limit).
			const [p0, p1] = [eef(before), eef(after)];
			const commanded = Math.hypot(...move.delta);
			if (plugin("proprioception") && p0 && p1 && commanded > 0) {
				const moved = [0, 1, 2].reduce((s, k) => s + (p1[k] - p0[k]) * move.delta[k], 0) / commanded;
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
			if (u === "GRASP" && plugin("recovery") && empty(after)) {
				last = await reopen(arm, signal);
				recent.push("RELEASE(recovery)");
				note = NOTES.empty_grasp;
				break;
			}
			// auto_release: a closed gripper that collapsed (the object slipped out) is reopened.
			if (u !== "GRASP" && plugin("auto_release") && closed.get(key) && empty(after)) {
				last = await reopen(arm, signal);
				recent.push("RELEASE(auto)");
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
		run: act,
		state: spec.state,
	};
	pi.on("session_start", () => pi.events.emit(UNITS_EVENT, handle));

	if (spec.point) {
		const { cameras, locate } = spec.point;
		tool(
			"point",
			"Affordance: mark the exact gripper contact point(s) in one camera's current image, [y, x] on a 0-1000 grid (y from the top, x from the left). Returns the marked image and the world xyz per point where the robot has depth; later act results report the gripper-to-point offset. A new call with the same label replaces that point.",
			Type.Object({
				camera: Type.Union(
					cameras.map((c) => Type.Literal(c)),
					{ description: `Camera (${cameras.join(", ")})` },
				),
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
			const lines = stages.map(
				(s, i) =>
					`${i === stage ? ">" : i < stage ? "x" : " "} ${i + 1}. [${s.motion}] ${s.target}: ${s.completion}`,
			);
			return { content: [text(lines.length ? lines.join("\n") : "No plan yet.")], details: { stage, stages } };
		},
	);

	pi.on("context", (event) => {
		if (!mode() || pi.getFlag("stateless") !== true) return undefined;
		const kept = latestTurn(event.messages);
		return kept ? { messages: kept } : undefined;
	});

	return {
		mode,
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
			};
			return p.replace(/\{\{(\w+)\}\}/g, (m, k: string) => vars[k] ?? m).trim();
		},
	};
}
