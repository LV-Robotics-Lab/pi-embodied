/**
 * GUMI (GUI-based Manipulation Interface) for pi-embodied: drive the robot one action unit per key
 * press from the dashboard, record every step in GUMI's rollout format, and take over from the agent
 * mid-run (DAgger). Ported from Show-Harness (github.com/showlab/Show-Harness @137d571):
 * gumi/web_teleop{,_dual} (key map, `w*3 a g` sequences, one synchronized (left, right) pair per
 * dual-arm step with STILL for the idle arm), core/teleop/{single,dual}.py (RolloutRecorder,
 * DualRolloutRecorder: the layout train/data_preparation/rollouts_to_alpaca.py reads) and
 * plugins/dagger (operator takeover; a decision made before the takeover is dropped, and each
 * recorded step says who acted).
 *
 * Copyright 2026 Show Lab, National University of Singapore. Licensed under the Apache License 2.0.
 * Modified by pi-embodied: rewritten in TypeScript on the shared action-unit layer (../units); the
 * units go through the robot's `units.apply` like the `act` tool's, and the observation recorded
 * with each unit is the tool result the policy sees (its PNG images, byte for byte).
 *
 * Runs go to `<--gumi-record>/<MMDD>/task_<id>/<HH-MM-SS>/` (core/record/episode_logger.py's run_dir, UTC+8):
 *   images/agentview/0000.png, images/wrist/0000.png   the tool result's PNGs, byte for byte
 *                     (two arms: images/wrist_left/, images/wrist_right/ instead of one side-by-side wrist png)
 *   steps.jsonl       episode_logger records: {i, stage, act, eef, w, grip, grip_measured, ts, src[, dagger, n]}
 *                     (two arms: {i, left: {act, eef, w, grip, src}, right: {...}, ts[, dagger]}); steps.json at close
 *   actions.jsonl     the GUMI collectors' records (core/teleop/{single,dual}.py), which
 *                     train/data_preparation/rollouts_to_alpaca.py reads: {step, token, kind, gripper_closed,
 *                     ee_pose, gripper_width, gripper_closed_measured, agentview, wrist, time, src[, dagger, n]}
 *                     (two arms: {step, agentview, wrist_left, wrist_right, time, left: {token, ...}, right: {...}})
 *                     `gripper_closed` / `grip` is the commanded state (the label, as GUMI records it);
 *                     ee_pose, gripper_width and `gripper_closed_measured` / `grip_measured` are what the robot
 *                     measured at obs_t (null when it reports no gripper state)
 *   metadata.json     task, robot, step size, tokens, sources, success; summary.json {success, steps, end_reason, ...}
 * Each record is (obs_t, a_t): the observation the unit was decided on, then the unit. `src` is "human" or
 * "agent"; operator steps taken over from a running agent carry `dagger` (true, or "L|R" on two arms), as
 * Show-Harness's runners write it. `rollouts_to_alpaca.py <root>/<MMDD>/task_<id>` converts every run of a task.
 */

import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type RobotStatus, STATUS_EVENT } from "../robot.ts";
import { UNITS_EVENT, type UnitsHandle } from "../units/index.ts";

export { UNITS_EVENT, type UnitsHandle };

// ---------------------------------------------------------------------------
// vocabulary and the command grammar (gumi/web_teleop/backend.py, web_teleop_dual/dual_backend.py)

export const MOVES = ["MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN"] as const;
export const ROTATES = ["ROTATE_CW", "ROTATE_CCW"] as const;
export const GRIPPERS = ["GRASP", "RELEASE"] as const;
export const STILL = "STILL";
/** One arm's slot on a single-arm robot (Show-Harness's plugins.dagger.SINGLE_SIDE). */
export const ARM = "arm";
export const DUAL = ["left", "right"] as const;

/**
 * The key bindings, by `KeyboardEvent.code`: the one table the dashboard's key handler and pad labels
 * (sent in the GUMI state as `keys`) and the typed sequence aliases below are made from.
 * One arm (web_teleop/static): WASD QE move, Z/X rotate, G grasp, R release, arrows as a second move
 * cluster. Two arms (web_teleop_dual/static/index.html): the left hand's cluster for the left arm,
 * IJKL UO NM Shift-R . for the right; Ctrl holds that arm STILL.
 */
const LEFT_KEYS: Record<string, string> = {
	KeyW: "MV_FWD",
	KeyS: "MV_BACK",
	KeyA: "MV_LEFT",
	KeyD: "MV_RIGHT",
	KeyQ: "MV_UP",
	KeyE: "MV_DOWN",
	KeyZ: "ROTATE_CCW",
	KeyX: "ROTATE_CW",
	KeyG: "GRASP",
	KeyR: "RELEASE",
};
const RIGHT_KEYS: Record<string, string> = {
	KeyI: "MV_FWD",
	KeyK: "MV_BACK",
	KeyJ: "MV_LEFT",
	KeyL: "MV_RIGHT",
	KeyU: "MV_UP",
	KeyO: "MV_DOWN",
	KeyN: "ROTATE_CCW",
	KeyM: "ROTATE_CW",
	ShiftRight: "GRASP",
	Period: "RELEASE",
	ControlRight: STILL,
};
export const KEYS = {
	single: {
		...LEFT_KEYS,
		ArrowUp: "MV_UP",
		ArrowDown: "MV_DOWN",
		ArrowLeft: "MV_LEFT",
		ArrowRight: "MV_RIGHT",
	} as Record<string, string>,
	dual: {
		left: { ...LEFT_KEYS, ShiftLeft: "GRASP", ControlLeft: STILL } as Record<string, string>,
		right: RIGHT_KEYS,
	},
};
/** The keys `arms` are driven with: code -> [arm, unit]. */
export function keyMap(arms: readonly string[]): Record<string, [string, string]> {
	const bound =
		arms.length > 1
			? arms.map((a, i) => [a, i ? KEYS.dual.right : KEYS.dual.left] as const)
			: [[arms[0], KEYS.single] as const];
	return Object.fromEntries(
		bound.flatMap(([arm, keys]) =>
			Object.entries(keys).map(([code, unit]): [string, [string, string]] => [code, [arm, unit]]),
		),
	);
}
/** A cluster's typeable keys (letters and `.`) as sequence aliases. */
const typed = (keys: Record<string, string>) =>
	Object.fromEntries(
		Object.entries(keys).flatMap(([code, unit]) => {
			const k = code === "Period" ? "." : /^Key([A-Z])$/.exec(code)?.[1];
			return k ? [[k, unit]] : [];
		}),
	);
/** The keyboard keys, typed: `w*3 a g` is pressing W three times, then A, then G. */
export const ALIASES: Record<string, string> = { ...typed(LEFT_KEYS), STAY: STILL };
/** The dual rig's right-hand cluster, accepted after `R:`. */
export const RIGHT_ALIASES: Record<string, string> = typed(RIGHT_KEYS);
export const MAX_REPEAT = 64;
/** Steps one request may carry (single arm: MAX_BATCH_TOKENS; two arms: MAX_PAIRS). */
export const MAX_STEPS = { single: 64, dual: 24 };

/** One time step: one unit per arm (STILL = that arm holds). */
export type Step = Record<string, string>;

export const kindOf = (unit: string) =>
	unit === STILL
		? "still"
		: (GRIPPERS as readonly string[]).includes(unit)
			? "gripper"
			: (ROTATES as readonly string[]).includes(unit)
				? "rotate"
				: "move";

/** Words (`MV_FWD`, `MV_FWD*3`, `w`, `w*3`) to units; aliases and repeats expanded, case-insensitive. */
function expand(words: string[], aliases: Record<string, string>): string[] {
	const out: string[] = [];
	for (const raw of words) {
		const m = /^([A-Z_.]+)(?:\*(\d+))?$/.exec(raw.toUpperCase());
		if (!m) throw new Error(`cannot parse '${raw}'; expected NAME or NAME*N (e.g. MV_FWD*3, w*3)`);
		const n = m[2] === undefined ? 1 : Number(m[2]);
		if (!(n >= 1 && n <= MAX_REPEAT)) throw new Error(`repeat count in '${raw}' must be 1..${MAX_REPEAT}`);
		out.push(...Array<string>(n).fill(aliases[m[1]] ?? m[1]));
	}
	return out;
}

const words = (v: unknown): string[] => {
	if (v === undefined || v === null) return [];
	if (typeof v === "string") return v.replace(/,/g, " ").split(/\s+/).filter(Boolean);
	if (Array.isArray(v) && v.every((x) => typeof x === "string")) return v.flatMap(words);
	throw new Error("expected a unit string or a list of unit strings");
};

/**
 * A teleop request to steps. `arms` is [ARM] or DUAL; `vocabulary` the units the robot accepts.
 *   {command: "w*3 a g"}                     the sequence box; two arms: "L:w*3 R:i g" (an omitted arm is STILL)
 *   {units: "MV_FWD*2 GRASP"} / {unit: "w"}  one arm
 *   {left: "MV_FWD*3", right: "MV_UP"}       two arms, zipped and STILL-padded
 * Throws with the reason on anything else.
 */
export function parseSteps(body: Record<string, unknown>, arms: readonly string[], vocabulary: readonly string[]) {
	const dual = arms.length > 1;
	let sides: Record<string, string[]>;
	const armed = "left" in body || "right" in body;
	const plain = "unit" in body || "units" in body;
	if ([armed, plain, "command" in body].filter(Boolean).length !== 1)
		throw new Error("send exactly one of {command}, {unit|units} or {left, right}");
	if ("command" in body) {
		if (typeof body.command !== "string") throw new Error("'command' must be a string");
		const all = words(body.command);
		if (!all.length) throw new Error("empty command");
		if (!all.some((w) => /^[LR]:/i.test(w))) sides = { [dual ? "none" : ARM]: all };
		else {
			sides = { L: [], R: [] };
			let side: string | undefined;
			for (const w of all) {
				const m = /^([LR]):(.*)$/i.exec(w);
				if (m) {
					side = m[1].toUpperCase();
					if (m[2]) sides[side].push(m[2]);
				} else if (side) sides[side].push(w);
				else throw new Error(`'${w}' comes before any arm prefix; write e.g. 'L:w*2 R:STILL'`);
			}
			if (!dual) throw new Error("this robot has one arm: drop the L:/R: prefixes");
			sides = { left: sides.L, right: sides.R };
		}
	} else if (armed) {
		if (!dual) throw new Error("this robot has one arm: send {units} or {command}, not {left, right}");
		sides = { left: words(body.left), right: words(body.right) };
	} else sides = { [dual ? "none" : ARM]: words(body.units ?? body.unit) };
	if ("none" in sides)
		throw new Error("two arms: prefix every group with an arm, e.g. 'L:w*3 R:i' (omitted arm = STILL)");

	const units = Object.fromEntries(
		Object.entries(sides).map(([side, w]) => [
			side,
			expand(w, side === "right" ? { ...ALIASES, ...RIGHT_ALIASES } : ALIASES),
		]),
	);
	const allowed = new Set([...vocabulary, ...(dual ? [STILL] : [])]);
	const bad = [...new Set(Object.values(units).flat())].filter((u) => !allowed.has(u));
	if (bad.length)
		throw new Error(
			`not a unit here: ${bad.join(", ")}; allowed: ${[...allowed].join(", ")} (keys: ${Object.entries(ALIASES)
				.filter(([, u]) => allowed.has(u))
				.map(([k, u]) => `${k.toLowerCase()}=${u}`)
				.join(" ")})`,
		);
	const n = Math.max(...arms.map((a) => units[a]?.length ?? 0));
	const steps: Step[] = Array.from({ length: n }, (_, i) =>
		Object.fromEntries(arms.map((a) => [a, units[a]?.[i] ?? STILL])),
	);
	if (!steps.length) throw new Error('no units given; e.g. {"command": "w*3 a g"}');
	if (steps.some((s) => arms.every((a) => s[a] === STILL)))
		throw new Error("every step needs at least one arm to act");
	const max = dual ? MAX_STEPS.dual : MAX_STEPS.single;
	if (steps.length > max) throw new Error(`too many steps (${steps.length}); at most ${max} per request`);
	return steps;
}

// ---------------------------------------------------------------------------
// observations: the images of a robot tool result, by camera

type Image = { type: "image"; data: string; mimeType: string };
/** A robot tool result's images, their camera labels and its JSON part (state), if any. */
export type Observation = { images: Image[]; labels: string[]; json: Record<string, unknown> | undefined };

/** The images and their labels (the result JSON's `images` list, as the dashboard reads it). */
export function observation(result: Pick<AgentToolResult<unknown>, "content">): Observation | undefined {
	const images = result.content.filter((c): c is Image => c.type === "image");
	if (!images.length) return undefined;
	let json: Record<string, unknown> | undefined;
	// `act` puts its units header before the robot's own JSON text.
	for (const c of result.content) {
		if (json || c.type !== "text") continue;
		try {
			const v = JSON.parse(c.text);
			if (v && typeof v === "object" && !Array.isArray(v)) json = v;
		} catch {}
	}
	const listed = Array.isArray(json?.images) ? (json.images as unknown[]) : [];
	const labels = images.map((_, k) => {
		const v = listed[k];
		return typeof v === "string" ? v : String((v as { camera?: unknown })?.camera ?? "");
	});
	return { images, labels, json };
}

/** Which image is which GUMI view: by label (wrist / left / right), the rest in order (agentview first). */
export function views(obs: Observation, arms: readonly string[]): Record<string, Image> {
	const wrist = /wrist|hand/i;
	const want: Record<string, (label: string) => boolean> =
		arms.length > 1
			? {
					agentview: (l) => l !== "" && !wrist.test(l),
					wrist_left: (l) => wrist.test(l) && /left/i.test(l),
					wrist_right: (l) => wrist.test(l) && /right/i.test(l),
				}
			: { agentview: (l) => l !== "" && !wrist.test(l), wrist: (l) => wrist.test(l) };
	const out: Record<string, Image> = {};
	const used = new Set<number>();
	for (const [name, test] of Object.entries(want)) {
		const k = obs.images.findIndex((_, i) => !used.has(i) && test(obs.labels[i] ?? ""));
		if (k < 0) continue;
		out[name] = obs.images[k];
		used.add(k);
	}
	const rest = obs.images.map((_, i) => i).filter((i) => !used.has(i));
	for (const name of Object.keys(want)) if (!out[name] && rest.length) out[name] = obs.images[rest.shift() as number];
	return out;
}

/** Proprioception for a record: the first matching key anywhere in `state` (depth-first). */
function find(state: unknown, keys: string[], depth = 0): unknown {
	if (!state || typeof state !== "object" || depth > 4) return undefined;
	const o = state as Record<string, unknown>;
	for (const k of keys) if (k in o && o[k] !== null) return o[k];
	for (const v of Object.values(o)) {
		const hit = find(v, keys, depth + 1);
		if (hit !== undefined) return hit;
	}
	return undefined;
}

const EEF = ["eef_xyz", "ee_pose", "eef_pose", "tcp_pose", "eef_pos", "robot0_eef_pos", "eef", "ee_pos", "tcp_pos"];
const WIDTH = ["gripper_width", "gripper_opening", "gripper_open_width", "robot0_gripper_qpos", "gripper"];
/** The robot's measured gripper state: closed (`gripper_closed`) or open (`gripper_open`, negated). */
const CLOSED = ["gripper_closed", "gripper_is_closed"];
const OPEN = ["gripper_open", "gripper_is_open"];
const r5 = (v: number) => Number(Number(v).toFixed(5));

/**
 * The measured {ee_pose, gripper_width, gripper_closed_measured} of one arm from a state object
 * (per-arm sub-object when there is one); `gripper_closed_measured` is null when the robot reports
 * no gripper state.
 */
export function armState(state: unknown, arm: string): ArmState {
	const own = state && typeof state === "object" && arm in state ? (state as Record<string, unknown>)[arm] : state;
	const eef = find(own, EEF);
	const w = find(own, WIDTH);
	const width = Array.isArray(w) ? w.reduce((s: number, x) => s + Math.abs(Number(x)), 0) : Number(w);
	const closed = find(own, CLOSED);
	const open = find(own, OPEN);
	return {
		ee_pose: Array.isArray(eef) ? eef.map((x) => r5(Number(x))) : [],
		gripper_width: Number.isFinite(width) ? r5(width) : 0,
		gripper_closed_measured: typeof closed === "boolean" ? closed : typeof open === "boolean" ? !open : null,
	};
}
export type ArmState = { ee_pose: number[]; gripper_width: number; gripper_closed_measured: boolean | null };
const NO_STATE: ArmState = { ee_pose: [], gripper_width: 0, gripper_closed_measured: null };

// ---------------------------------------------------------------------------
// the recorder (core/record/episode_logger.py run dirs, with core/teleop/{single,dual}.py actions.jsonl)

export type Src = "human" | "agent";
type StepInfo = {
	src: Src;
	/** A human step taken over from a running agent (DAgger). */
	dagger: boolean;
	/** Commanded gripper state of each arm when the step was taken (the recorded training label). */
	closed: Record<string, boolean>;
	/** Measured proprioception per arm at obs_t (the robot's `units.state`). */
	state: Record<string, ArmState>;
	/** `act({unit, n})` with n > 1: one record for the decision, the count alongside. */
	n?: number;
};

/** The run directory's name parts, as core/record/episode_logger.py makes them (UTC+8). */
function stamp(now = new Date()) {
	const t = new Date(now.getTime() + 8 * 3600_000).toISOString();
	return {
		date: `${t.slice(5, 7)}${t.slice(8, 10)}`,
		time: `${t.slice(11, 13)}-${t.slice(14, 16)}-${t.slice(17, 19)}`,
	};
}
const r3 = (v: number) => Number(Number(v).toFixed(3));

export class Recorder {
	dir: string | undefined;
	steps: Record<string, unknown>[] = [];
	readonly root: string;
	readonly arms: readonly string[];
	private meta: Record<string, unknown> = {};
	private full: Record<string, unknown>[] = [];
	constructor(root: string, arms: readonly string[]) {
		this.root = root;
		this.arms = arms;
	}
	get active() {
		return this.dir !== undefined;
	}
	get dual() {
		return this.arms.length > 1;
	}
	/** Open `<root>/<MMDD>/task_<id>/<HH-MM-SS>/` (episode_logger's run_dir). */
	start(meta: Record<string, unknown>, taskId: string | number = 0, now = new Date()) {
		if (this.dir) throw new Error(`already recording ${this.dir}`);
		const { date, time } = stamp(now);
		let dir = join(this.root, date, `task_${taskId}`, time);
		for (let k = 1; existsSync(dir); k++) dir = join(this.root, date, `task_${taskId}`, `${time}_${k}`);
		for (const v of this.dual ? ["agentview", "wrist_left", "wrist_right"] : ["agentview", "wrist"])
			mkdirSync(join(dir, "images", v), { recursive: true });
		writeFileSync(join(dir, "steps.jsonl"), "");
		writeFileSync(join(dir, "actions.jsonl"), "");
		this.dir = dir;
		this.steps = [];
		this.full = [];
		this.meta = { ...meta, run_dir: dir, started: now.toISOString() };
		writeFileSync(join(dir, "metadata.json"), JSON.stringify(this.meta, null, 2));
		return dir;
	}

	/** Append (obs_t, a_t): the observation the step was taken FROM, then the unit(s) executed. */
	add(obs: Observation, step: Step, info: StepInfo) {
		const dir = this.dir;
		if (!dir) return;
		const n = this.steps.length;
		const files: Record<string, string> = {};
		for (const [view, img] of Object.entries(views(obs, this.arms))) {
			const file = `images/${view}/${String(n).padStart(4, "0")}.${img.mimeType === "image/jpeg" ? "jpg" : "png"}`;
			writeFileSync(join(dir, file), Buffer.from(img.data, "base64"));
			files[view] = file;
		}
		const ts = Date.now() / 1000;
		const human = info.dagger && info.src === "human";
		const dagger = this.dual
			? this.arms
					.filter((a) => step[a] !== STILL)
					.map((a) => a[0].toUpperCase())
					.join("|")
			: true;
		// steps.jsonl: core/runners/mvtoken.py's record (dual: core/runners/dual.py's per-arm records).
		const logged = (a: string) => {
			const st = info.state[a] ?? NO_STATE;
			const measured = st.gripper_closed_measured;
			return {
				act: step[a],
				eef: st.ee_pose.slice(0, 3).map(r3),
				w: st.gripper_width,
				grip: info.closed[a] ? "CLOSED" : "OPEN",
				grip_measured: measured === null ? null : measured ? "CLOSED" : "OPEN",
				src: info.src,
			};
		};
		const record: Record<string, unknown> = this.dual
			? { i: n, ...Object.fromEntries(this.arms.map((a) => [a, logged(a)])), ts: Number(ts.toFixed(3)) }
			: { i: n, stage: "-", ...logged(ARM), ts: Number(ts.toFixed(3)) };
		if (human) record.dagger = dagger;
		if (info.n && info.n > 1) record.n = info.n;
		this.full.push(record);
		writeFileSync(join(dir, "steps.jsonl"), `${JSON.stringify(record)}\n`, { flag: "a" });
		// actions.jsonl: the GUMI collectors' record (core/teleop/{single,dual}.py), what
		// train/data_preparation/rollouts_to_alpaca.py reads.
		const arm = (a: string) => ({
			token: step[a],
			kind: kindOf(step[a]),
			gripper_closed: info.closed[a] ?? false,
			...(info.state[a] ?? NO_STATE),
		});
		const time = Number(ts.toFixed(3));
		const action: Record<string, unknown> = this.dual
			? { step: n, ...files, time, ...Object.fromEntries(this.arms.map((a) => [a, { ...arm(a), src: info.src }])) }
			: { step: n, ...arm(ARM), ...files, time, src: info.src };
		if (human) action.dagger = dagger;
		if (info.n && info.n > 1) action.n = info.n;
		this.steps.push(action);
		writeFileSync(join(dir, "actions.jsonl"), `${JSON.stringify(action)}\n`, { flag: "a" });
		return action;
	}

	/** Close the run: steps.json, metadata.json and summary.json (episode_logger.close + write_summary). */
	stop(extra: { success?: boolean } & Record<string, unknown>) {
		const dir = this.dir;
		if (!dir) throw new Error("not recording");
		const tokens = (a: string) => this.steps.map((s) => (this.dual ? (s[a] as { token: string }).token : s.token));
		const meta: Record<string, unknown> = {
			...this.meta,
			num_steps: this.steps.length,
			...(this.dual ? { tokens_left: tokens("left"), tokens_right: tokens("right") } : { tokens: tokens(ARM) }),
			sources: this.steps.map((s) => (this.dual ? (s.left as { src: string }).src : s.src)),
			...extra,
		};
		writeFileSync(join(dir, "steps.json"), JSON.stringify(this.full, null, 2));
		writeFileSync(join(dir, "metadata.json"), JSON.stringify(meta, null, 2));
		writeFileSync(
			join(dir, "summary.json"),
			JSON.stringify(
				{
					success: extra.success === true,
					steps: this.steps.length,
					end_reason: String(extra.end_reason ?? (extra.success ? "done" : "stopped")),
					run_dir: dir,
					control_mode: "gumi",
					task: this.meta.task ?? null,
				},
				null,
				2,
			),
		);
		this.dir = undefined;
		this.steps = [];
		this.full = [];
		return { dir, meta };
	}

	/** Delete the run in progress. */
	discard() {
		const dir = this.dir;
		if (!dir) throw new Error("not recording");
		rmSync(dir, { recursive: true, force: true });
		this.dir = undefined;
		this.steps = [];
		this.full = [];
		return dir;
	}
}

// ---------------------------------------------------------------------------
// DAgger takeover (plugins/dagger + core/runners/preemption.py)

export type Mode = "agent" | "requested" | "human";

/**
 * Who drives. The agent's robot tool calls pass `gate()` first: while the operator has the robot, or
 * an operator batch (`begin()` .. `end()`) is running, they wait (the agent is paused between
 * steps). A call is stale when the operator acted after the latest robot result the agent had when
 * its model request started (`seen()` on robot results and drop notices, `decide()` at each
 * request): it is dropped, as Show-Harness's runner drops a decision whose input generation
 * changed. A non-robot result (e.g. `read`) carries no observation, so it never refreshes a
 * decision. `take()` during an agent call waits for it to finish ("requested") before the operator
 * may act; `release()` during an operator batch takes effect when the batch ends.
 */
export class Takeover {
	mode: Mode = "agent";
	/** Operator units since the agent's latest observation, for the drop message. */
	driven: string[] = [];
	/** Operator steps since boot. */
	generation = 0;
	/** An operator batch is running: nothing of the agent's runs until it ends. */
	operating = false;
	/** The generation the agent's current decision rests on (its model request's). */
	private observed = 0;
	/** The generation of the latest robot result or drop notice the agent received. */
	private informed = 0;
	private busy = false;
	private handBack = false;
	private waiters: (() => void)[] = [];

	take() {
		this.handBack = false;
		if (this.mode === "agent") this.mode = this.busy ? "requested" : "human";
		return this.mode;
	}
	/** Hand back to the agent: now, or when the running operator batch ends. */
	release() {
		if (this.operating) this.handBack = true;
		else this.mode = "agent";
		for (const w of this.waiters.splice(0)) w();
	}
	/** An operator batch starts; it holds the agent until `end()`, whoever has control. */
	begin() {
		this.operating = true;
	}
	end() {
		this.operating = false;
		if (this.handBack) this.mode = "agent";
		this.handBack = false;
		for (const w of this.waiters.splice(0)) w();
	}
	/** The operator may drive now (the agent is not in the middle of a unit). */
	get human() {
		return this.mode === "human";
	}
	humanStep(units: string[]) {
		this.generation++;
		this.driven.push(...units);
	}
	/** The agent received the robot's current state (a robot tool result, or a drop notice). */
	seen() {
		this.informed = this.generation;
		this.driven = [];
	}
	/** A model request starts: its decisions rest on what the agent had received so far. */
	decide() {
		this.observed = this.informed;
	}

	/** Before an agent robot call: wait while the operator drives; `stale` when they acted since the agent last looked. */
	async gate(signal?: AbortSignal): Promise<{ stale: boolean; driven: string[] }> {
		while (this.mode !== "agent" || this.operating) {
			if (signal?.aborted) throw new Error("aborted while the operator had the robot");
			await new Promise<void>((resolve) => {
				const done = () => {
					signal?.removeEventListener("abort", done);
					resolve();
				};
				this.waiters.push(done);
				signal?.addEventListener("abort", done, { once: true });
			});
		}
		// Every later call of a dropped request is stale too: it rests on the same observation.
		const stale = this.generation !== this.observed;
		const driven = [...this.driven];
		if (stale) this.seen();
		else this.busy = true;
		return { stale, driven };
	}
	/** After an agent robot call: a pending takeover becomes the operator's. */
	done() {
		this.busy = false;
		if (this.mode === "requested") this.mode = "human";
	}
}

// ---------------------------------------------------------------------------
// the robot's unit layer, as ../units publishes it on pi.events

/**
 * What an `act` call asked for, as one step: `{unit, n}`, on two arms `{unit, arm, n}` (the other arm
 * holds STILL). STOP (hold and look) and DONE move nothing and are not recorded, like the rollouts'
 * own vocabulary (rollouts_to_alpaca.py synthesizes DONE).
 */
export function actSteps(
	input: Record<string, unknown>,
	arms: readonly string[],
): { step: Step; n: number } | undefined {
	const n = Math.max(1, Math.floor(Number(input.n ?? 1)) || 1);
	const unit = typeof input.unit === "string" ? input.unit.trim().toUpperCase() : "";
	if (!unit || unit === "STOP" || unit === "DONE") return undefined;
	if (arms.length === 1) return { step: { [arms[0]]: unit }, n };
	const arm = String(input.arm ?? "").toLowerCase();
	if (!arms.includes(arm)) return undefined;
	return { step: Object.fromEntries(arms.map((a) => [a, a === arm ? unit : STILL])), n };
}

// ---------------------------------------------------------------------------
// the controller the dashboard drives

export type GumiState = {
	available: boolean;
	robot: string | null;
	arms: readonly string[];
	vocabulary: readonly string[];
	mode: Mode;
	busy: boolean;
	recording: string | null;
	steps: number;
	saved: number;
	closed: Record<string, boolean>;
	last: string | null;
	message: string;
	root: string | null;
	/** The key bindings for these arms (`keyMap`): KeyboardEvent.code -> [arm, unit]. */
	keys: Record<string, [string, string]>;
};

/**
 * Mount GUMI on this runtime: `--gumi-record <dir>`, the agent-side DAgger gate and agent-step
 * recording (tool_call / tool_result of the unit tool), and the operator's calls for the dashboard.
 * `onStep` hands each operator step's result to the dashboard (camera panel, timeline).
 */
export function gumi(
	pi: ExtensionAPI,
	o: {
		onState?: (s: GumiState) => void;
		onStep?: (label: string, step: Step, result: AgentToolResult<unknown>, isError: boolean) => void;
	} = {},
) {
	pi.registerFlag("gumi-record", {
		type: "string",
		description: "GUMI: record teleop / DAgger episodes (Show-Harness episode_logger layout) under this dir",
	});
	let handle: UnitsHandle | undefined;
	/** [ARM], or the robot's two arm names. */
	let arms: readonly string[] = [ARM];
	let ctx: ExtensionContext | undefined;
	let recorder: Recorder | undefined;
	let latest: Observation | undefined;
	let closed: Record<string, boolean> = {};
	let saved = 0;
	let last: string | null = null;
	let message = "";
	let task: Record<string, unknown> = {};
	let solved = false;
	let robotName = "robot";
	let pending:
		| { step: Step; n: number; obs: Observation | undefined; state: StepInfo["state"]; closed: StepInfo["closed"] }
		| undefined;
	const takeover = new Takeover();
	/** Stops the running operator batch (`stop()`), also while the agent is idle. */
	let batch: AbortController | undefined;

	const root = () => {
		const v = pi.getFlag("gumi-record");
		return typeof v === "string" && v ? v : undefined;
	};
	const state = (): GumiState => ({
		available: handle !== undefined,
		robot: handle ? robotName : null,
		arms,
		vocabulary: handle?.vocabulary ?? [],
		mode: takeover.mode,
		busy: takeover.operating,
		recording: recorder?.dir ?? null,
		steps: recorder?.steps.length ?? 0,
		saved,
		closed,
		last,
		message,
		root: root() ?? null,
		keys: keyMap(arms),
	});
	const publish = (msg?: string) => {
		if (msg !== undefined) message = msg;
		o.onState?.(state());
	};

	pi.events.on(UNITS_EVENT, (data) => {
		handle = data as UnitsHandle;
		arms = handle.arms.length > 1 ? handle.arms : [ARM];
		closed = Object.fromEntries(arms.map((a) => [a, false]));
		recorder = root() ? new Recorder(root() as string, arms) : undefined;
		publish();
	});
	// The robot's status (../robot.ts STATUS_EVENT): the task language and LIBERO's success flag.
	pi.events.on(STATUS_EVENT, (data) => {
		const s = data as RobotStatus;
		task = { robot: s.robot, robot_task: s.task, ...(s.language ? { task: s.language } : {}) };
		solved = s.solved === true;
		robotName = s.robot;
	});

	pi.on("session_start", (_event, c) => {
		ctx = c;
		latest = undefined;
		takeover.release();
	});
	pi.on("session_shutdown", () => {
		// A rollout still open when the episode ends is kept, marked unfinished (GUMI saves on shutdown too).
		if (recorder?.active) recorder.stop({ success: false, end_reason: "session_shutdown" });
		takeover.release();
	});
	pi.on("agent_end", () => {
		takeover.release();
		publish();
	});

	/** Proprioception at obs_t: the robot's `units.state`, else the observation's own state JSON. */
	async function stateNow(obs: Observation | undefined) {
		const out: StepInfo["state"] = {};
		for (const a of arms) {
			let s: unknown;
			try {
				s = handle?.state ? await handle.state(a === ARM ? undefined : a) : undefined;
			} catch {}
			out[a] = armState(s ?? obs?.json, a);
		}
		return out;
	}
	/** One step through `act`'s own path: each acting arm's unit (two arms: one after the other). */
	async function execute(step: Step, signal: AbortSignal | undefined) {
		let result: AgentToolResult<unknown> | undefined;
		for (const a of arms) {
			if (step[a] === STILL) continue;
			result = await (handle as UnitsHandle).run(
				a === ARM ? { unit: step[a], operator: true } : { unit: step[a], arm: a, operator: true },
				signal,
			);
		}
		return result as AgentToolResult<unknown>;
	}
	function track(step: Step) {
		for (const [a, u] of Object.entries(step)) if (u === "GRASP" || u === "RELEASE") closed[a] = u === "GRASP";
	}

	// The DAgger gate: every robot tool (`act`, and the robot's own motion and perception tools)
	// waits while the operator drives, and a call decided before the operator acted is dropped; the
	// agent gets the current observation instead.
	pi.on("tool_call", async (event, c) => {
		if (!handle?.tools().includes(event.toolName)) return undefined;
		const { stale, driven } = await takeover.gate(c.signal);
		if (stale) {
			publish();
			if (!driven.length)
				return {
					block: true,
					reason: `This ${event.toolName} call was decided before the operator's steps (reported above) and was not executed; decide again from the current observation.`,
				};
			const summary = `The operator took over and executed ${driven.length} step(s): ${driven.join(" ")}.`;
			if (latest)
				pi.sendUserMessage(
					[
						{ type: "text", text: `${summary} This is the current observation (${latest.labels.join(", ")}).` },
						...latest.images,
					],
					{ deliverAs: "steer" },
				);
			return {
				block: true,
				reason: `${summary} This ${event.toolName} call was decided on an older observation and was not executed; decide again from the current observation.`,
			};
		}
		if (event.toolName !== handle.tool) return undefined;
		const parsed = actSteps(event.input, arms);
		pending = parsed ? { ...parsed, obs: latest, state: await stateNow(latest), closed: { ...closed } } : undefined;
		publish();
		return undefined;
	});

	pi.on("tool_result", (event) => {
		const obs = observation(event);
		if (handle && event.toolName === handle.tool) {
			const p = pending;
			pending = undefined;
			if (p && !event.isError) {
				if (recorder?.active && p.obs)
					recorder.add(p.obs, p.step, { src: "agent", dagger: false, closed: p.closed, state: p.state, n: p.n });
				track(p.step);
				last = `agent ${Object.values(p.step).join("/")}${p.n > 1 ? `×${p.n}` : ""}`;
			}
			publish();
		}
		// The cameras (the unit tool's result, or a robot observation that lists its `images`); not, e.g.,
		// `point`'s marked image, which is no camera frame.
		if (obs && (event.toolName === handle?.tool || Array.isArray(obs.json?.images))) latest = obs;
		// A robot tool's result is the robot's current state as the agent sees it next.
		if (handle?.tools().includes(event.toolName)) takeover.seen();
		return undefined;
	});

	// A model request decides on what the agent has received by now.
	pi.on("context", () => {
		takeover.decide();
		return undefined;
	});

	// Every robot call ends here, also one a later tool_call handler blocked (it gets no tool_result).
	pi.on("tool_execution_end", (event) => {
		if (!handle?.tools().includes(event.toolName)) return;
		takeover.done();
		publish();
	});

	/** The operator's calls (the dashboard's /gumi/* endpoints). Errors carry an HTTP status. */
	const fail = (status: number, msg: string) => Object.assign(new Error(msg), { status });
	return {
		state,
		takeover,
		/** Parse and run a teleop request: each step records (obs_t, a_t) then executes. */
		async step(body: Record<string, unknown>) {
			if (!handle || !ctx) throw fail(409, "no robot with action units is up (run the robot with --units)");
			const steps = (() => {
				try {
					return parseSteps(
						body,
						arms,
						handle.vocabulary.filter((u) => u !== "DONE"),
					);
				} catch (e) {
					throw fail(422, (e as Error).message);
				}
			})();
			if (!ctx.isIdle() && !takeover.human)
				throw fail(
					409,
					takeover.mode === "requested"
						? "waiting for the agent's current unit to finish"
						: "the agent is driving: take over first",
				);
			if (takeover.operating) throw fail(409, "busy with the previous step");
			// The gates an agent's call passes: robot up and not broken, episode not over, budget left, scene confirmed.
			const refused = handle.refuse();
			if (refused) throw fail(409, refused);
			takeover.begin();
			batch = new AbortController();
			// The agent's abort stops the batch too, as before.
			const signal = ctx.signal ? AbortSignal.any([batch.signal, ctx.signal]) : batch.signal;
			publish();
			const results: { step: Step; ok: boolean; error?: string }[] = [];
			try {
				for (const step of steps) {
					const label =
						arms.length > 1 ? arms.map((a) => `${a[0].toUpperCase()}:${step[a]}`).join(" ") : step[ARM];
					const why = signal.aborted ? "stopped" : handle.refuse();
					if (why) {
						results.push({ step, ok: false, error: why });
						publish(`${label} refused: ${why}`);
						break;
					}
					// obs_t: the observation the policy would see now (the latest robot result); before any
					// result exists, STOP (hold one step and look) produces it.
					if (!latest && recorder?.active && handle.vocabulary.includes("STOP")) {
						const first = arms[0];
						try {
							const look = await handle.run(
								first === ARM ? { unit: "STOP" } : { unit: "STOP", arm: first },
								signal,
							);
							latest = observation(look);
						} catch (e) {
							results.push({ step, ok: false, error: (e as Error).message });
							publish(`${label} failed: ${(e as Error).message}`);
							break;
						}
					}
					const obs = latest;
					const info: StepInfo = {
						src: "human",
						dagger: !ctx.isIdle(),
						closed: { ...closed },
						state: await stateNow(obs),
					};
					const units = arms.map((a) => step[a]);
					let result: AgentToolResult<unknown>;
					try {
						result = await execute(step, signal);
					} catch (e) {
						const error = (e as Error).message;
						results.push({ step, ok: false, error });
						o.onStep?.(label, step, { content: [{ type: "text", text: error }], details: {} }, true);
						publish(`${label} failed: ${error}`);
						break;
					}
					// STOP (hold and look) only refreshes the observation: not a training step.
					const moved = arms.some((a) => step[a] !== STILL && step[a] !== "STOP");
					if (recorder?.active && moved) {
						if (obs) recorder.add(obs, step, info);
						else message = "no observation before this step; not recorded";
					}
					track(step);
					if (moved) takeover.humanStep(units.filter((u) => u !== STILL));
					const next = observation(result);
					if (next) latest = next;
					last = `human ${label}`;
					results.push({ step, ok: true });
					o.onStep?.(label, step, result, false);
					publish(`executed ${label}`);
				}
			} finally {
				batch = undefined;
				// A hand-back asked for during the batch takes effect now.
				takeover.end();
				publish();
			}
			const failed = results.find((r) => !r.ok);
			return {
				ok: !failed,
				executed: results.filter((r) => r.ok).length,
				requested: steps.length,
				results,
				state: state(),
			};
		},
		/** start | save (success marked) | discard. */
		record(action: string, success?: unknown) {
			if (!recorder) throw fail(409, "recording is off: start pi with --gumi-record <dir>");
			try {
				if (action === "start") {
					const id = (task.robot_task as Record<string, string> | undefined)?.task ?? "0";
					const dir = recorder.start(
						{
							...task,
							teleop: "pi-embodied dashboard",
							arms: handle?.arms,
							step_m: handle?.stepM,
							yaw_step_rad: handle?.yawStepRad,
						},
						/^[\w.-]+$/.test(id) ? id : "0",
					);
					publish(`recording ${dir}`);
					return { ok: true, dir, state: state() };
				}
				if (action === "save") {
					const ok = typeof success === "boolean" ? success : solved;
					const { dir, meta } = recorder.stop({ success: ok, env_solved: solved, end_reason: "saved" });
					saved++;
					publish(`saved ${dir} (${meta.num_steps} steps, success=${ok})`);
					return { ok: true, dir, meta, state: state() };
				}
				if (action === "discard") {
					const dir = recorder.discard();
					publish(`discarded ${dir}`);
					return { ok: true, dir, state: state() };
				}
			} catch (e) {
				throw fail(409, (e as Error).message);
			}
			throw fail(422, "action must be start, save or discard");
		},
		/** Stop the running operator batch (the robot's RPC gets `stop`); false when none runs. */
		stop() {
			if (!batch) return false;
			batch.abort();
			publish("stopping the operator batch");
			return true;
		},
		/** take | release. */
		control(action: string) {
			if (action === "take") {
				takeover.take();
				publish(takeover.mode === "human" ? "operator has the robot" : "takeover requested");
			} else if (action === "release") {
				takeover.release();
				publish(
					takeover.operating
						? "handing back to the agent after the current batch"
						: `handed back to the agent after ${takeover.driven.length} operator unit(s)`,
				);
			} else throw fail(422, "action must be take or release");
			return { ok: true, state: state() };
		},
	};
}

export type Gumi = ReturnType<typeof gumi>;
