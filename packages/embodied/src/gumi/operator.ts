/**
 * The VLM operator of the GUMI teleop page (Show-Harness gumi/gpt_web_operator.py,
 * gumi/gpt_operator/operator.py): a vision model watches the teleop page's live observation and
 * drives the robot through the page's own command path (`w*3 a g`, `L:.. R:..`), so what it makes is a
 * demonstration in exactly the human teleop format, recorded with `src: "gpt-operator"`. That is the
 * difference from a pi agent driving `act`: the operator is a teleoperator, not the planner.
 *
 *   pi -e packages/embodied/src/libero -e packages/embodied/src/dashboard --dashboard --units=true \
 *     --gumi-record runs/gumi --gumi-operator selfhost/muse-glimmer-30b ...
 *
 * `--gumi-operator <provider/model>` (or `session`: the session's model) mounts it; the dashboard's
 * teleop panel gets Run / Pause / Step once, and it starts paused. One cycle: observe (the latest
 * observation; a STOP look before the first), one VLM decision (./operator.md: the robot's VIEWS, the
 * task, the teleop state, the recent commands) as JSON {phase, evidence, next_goal, command,
 * confidence, finish, pause}, then the safety gates of the upstream operator before anything moves:
 * malformed or out-of-grammar decisions fail the cycle, a confidence below
 * --gumi-operator-confidence pauses, one unit per arm repeated at most 3 times (only horizontal moves
 * and MV_UP repeat), an inverse-command oscillation pauses, a pause pressed during the call drops the
 * decision, and a teleop state that changed during the call (another step, the gripper, recording)
 * discards it as stale. finish is accepted only on the robot's own success flag (a simulator's); then
 * the rollout is saved as a success. It starts recording by itself when --gumi-record is on. Every
 * decision is a `gumi_operator` session entry; --gumi-operator-max-steps caps the executed steps.
 *
 * Copyright 2026 Show Lab, National University of Singapore. Licensed under the Apache License 2.0.
 * Modified by pi-embodied: the operator loop and its decision checks (normalize_decision,
 * _oscillates, state_marker, run_cycle) rewritten in TypeScript over ../gumi and the units layer; the
 * decision's per-arm action objects replaced by one command in the teleop grammar; the model call
 * through pi's model registry (../units/vlm.ts askVlm), its cost on the episode budget.
 */

import { readFileSync } from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { UNITS_EVENT, type UnitsHandle } from "../units/index.ts";
import { askVlm, parseJson, VLM_COST_EVENT } from "../units/vlm.ts";
import { ARM, type Gumi, parseSteps, STILL, type Step } from "./index.ts";

export const PHASES = [
	"search",
	"approach",
	"align",
	"descend",
	"grasp",
	"lift",
	"transport",
	"place",
	"verify",
	"recover",
] as const;
/** Units the upstream operator lets repeat within one decision (clear translation or lift). */
const REPEATABLE = new Set<string>(["MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP"]);
const INVERSE: Record<string, string> = {
	MV_FWD: "MV_BACK",
	MV_BACK: "MV_FWD",
	MV_LEFT: "MV_RIGHT",
	MV_RIGHT: "MV_LEFT",
	MV_UP: "MV_DOWN",
	MV_DOWN: "MV_UP",
	ROTATE_CCW: "ROTATE_CW",
	ROTATE_CW: "ROTATE_CCW",
	STILL: "STILL",
};
/** operator.py OperatorConfig: max_repeat 3, confidence_threshold 0.55, max_steps 150, interval 0.25 s. */
export const MAX_UNITS = 3;
const INTERVAL_MS = 250;
/** The session entry of every decision. */
export const OPERATOR_ENTRY = "gumi_operator";

const TEMPLATE = readFileSync(new URL("./operator.md", import.meta.url), "utf8").replace(/^<!--[\s\S]*?-->\n/, "");

export class DecisionError extends Error {}

export type Decision = {
	phase: string;
	evidence: string;
	next_goal: string;
	command: string;
	confidence: number;
	finish: boolean;
	pause: boolean;
};

const oneLine = (v: unknown) =>
	String(v ?? "")
		.split(/\s+/)
		.join(" ")
		.trim();

/**
 * normalize_decision: validate a model decision before it can reach the teleop path. Each arm runs one
 * unit, repeated at most `maxUnits` times (units other than clear moves once); finish and pause send
 * nothing; a confidence below the threshold pauses. Returns the decision and its teleop steps.
 */
export function normalizeDecision(
	raw: unknown,
	o: { arms: readonly string[]; vocabulary: readonly string[]; threshold: number; rt?: boolean; maxUnits?: number },
): { decision: Decision; steps: Step[] } {
	if (!raw || typeof raw !== "object" || Array.isArray(raw))
		throw new DecisionError("model output must be a JSON object");
	const r = raw as Record<string, unknown>;
	const phase = oneLine(r.phase).toLowerCase();
	if (!(PHASES as readonly string[]).includes(phase))
		throw new DecisionError(`invalid phase: ${JSON.stringify(r.phase)}`);
	const evidence = oneLine(r.evidence).slice(0, 500);
	const next_goal = oneLine(r.next_goal).slice(0, 300);
	if (!evidence || !next_goal) throw new DecisionError("evidence and next_goal must be non-empty");
	const confidence = Number(r.confidence);
	if (typeof r.confidence !== "number" || !(confidence >= 0 && confidence <= 1))
		throw new DecisionError("confidence must be a number between 0 and 1");
	const finish = r.finish === true;
	let pause = r.pause === true;
	let command = oneLine(r.command);
	const max = o.maxUnits ?? MAX_UNITS;
	let steps: Step[] = [];
	if (finish || pause) command = "";
	else {
		if (!command) throw new DecisionError("a decision that neither finishes nor pauses needs a command");
		try {
			steps = parseSteps({ command }, o.arms, o.vocabulary, o.rt === true);
		} catch (err) {
			throw new DecisionError(`command ${JSON.stringify(command)}: ${(err as Error).message}`);
		}
		for (const a of o.arms) {
			const units = steps.map((s) => s[a]).filter((u) => u !== STILL);
			if (new Set(units).size > 1)
				throw new DecisionError(
					`one unit per arm per decision (repeated at most ${max} times); got ${units.join(" ")}`,
				);
			if (units.length > max) throw new DecisionError(`at most ${max} repeats per arm; got ${units.length}`);
		}
		// Closed-loop precision: descent, turns and the gripper need another image before repeating.
		for (const a of o.arms) {
			const u = steps.find((st) => st[a] !== STILL)?.[a];
			if (u === undefined || REPEATABLE.has(u)) continue;
			let seen = false;
			for (const st of steps)
				if (st[a] !== STILL) {
					if (seen) st[a] = STILL;
					seen = true;
				}
		}
		steps = steps.filter((st) => o.arms.some((a) => st[a] !== STILL));
	}
	if (confidence < o.threshold && !finish) {
		pause = true;
		command = "";
		steps = [];
	}
	return { decision: { phase, evidence, next_goal, command, confidence, finish, pause }, steps };
}

/** _oscillates: the last four executed commands alternate between this one and its inverse. */
export function oscillates(recent: readonly string[][], next: readonly string[]): boolean {
	const inverse = next.map((u) => INVERSE[u] ?? u);
	if (inverse.every((u, i) => u === next[i]) || recent.length < 4) return false;
	const key = (s: readonly string[]) => s.join(" ");
	const tail = recent.slice(-4).map(key);
	return tail.join("|") === [next, inverse, next, inverse].map(key).join("|");
}

export type OperatorState = {
	available: boolean;
	model: string;
	mode: "paused" | "continuous";
	busy: boolean;
	status: string;
	cycle: number;
	executed: number;
	lastDecision: Decision | null;
	lastError: string | null;
};

/**
 * Mount the operator on this runtime's GUMI (`g`); `publish` re-sends the teleop state (which carries
 * the operator's) to the dashboard. Registers its flags; nothing runs until Run or Step once.
 */
export function vlmOperator(pi: ExtensionAPI, g: Gumi, publish: () => void) {
	pi.registerFlag("gumi-operator", {
		type: "string",
		default: "",
		description:
			"GUMI VLM operator: the model (provider/id, or `session`) that drives the teleop page and records demonstrations as gpt-operator",
	});
	pi.registerFlag("gumi-operator-confidence", {
		type: "string",
		default: "0.55",
		description: "GUMI VLM operator: pause on a decision below this confidence",
	});
	pi.registerFlag("gumi-operator-max-steps", {
		type: "string",
		default: "150",
		description: "GUMI VLM operator: the most teleop steps it executes",
	});
	const modelRef = () => {
		const v = String(pi.getFlag("gumi-operator") ?? "").trim();
		return v === "session" ? "" : v;
	};
	const enabled = () => String(pi.getFlag("gumi-operator") ?? "").trim() !== "";
	let handle: UnitsHandle | undefined;
	pi.events.on(UNITS_EVENT, (h) => {
		handle = h as UnitsHandle;
	});
	let mode: OperatorState["mode"] = "paused";
	let pendingOnce = 0;
	let busy = false;
	let running = false;
	let abortAction = false;
	let status = "Paused: review the views, then Run or Step once";
	let cycle = 0;
	let executed = 0;
	let lastDecision: Decision | null = null;
	let lastError: string | null = null;
	let recent: string[][] = [];
	pi.on("session_start", () => {
		mode = "paused";
		pendingOnce = 0;
		cycle = executed = 0;
		recent = [];
		lastDecision = null;
		lastError = null;
		status = "Paused: review the views, then Run or Step once";
	});
	pi.on("session_shutdown", () => {
		mode = "paused";
		pendingOnce = 0;
		abortAction = true;
	});

	const state = (): OperatorState => ({
		available: enabled(),
		model: String(pi.getFlag("gumi-operator") ?? ""),
		mode,
		busy,
		status,
		cycle,
		executed,
		lastDecision,
		lastError,
	});
	const pause = (msg: string) => {
		mode = "paused";
		pendingOnce = 0;
		status = msg;
		publish();
	};
	const record = (event: Record<string, unknown>) => {
		pi.appendEntry(OPERATOR_ENTRY, event);
	};

	/** The teleop state that must not change while the model decides (state_marker). */
	const marker = () => {
		const s = g.state();
		return JSON.stringify([Boolean(s.recording), s.steps, s.closed, s.last]);
	};

	function prompt(task: string, arms: readonly string[], dual: boolean) {
		const s = g.state();
		const vocabulary = s.vocabulary.filter((u) => u !== "DONE");
		const units = dual ? [...vocabulary, STILL] : vocabulary;
		const vars: Record<string, string> = {
			views: handle?.guide?.() ?? "Image A is the third-person view; the next images are the wrist views.",
			step_cm: String(Math.round((handle?.stepM ?? 0.02) * 100)),
			units: units.join(", "),
			max_units: String(MAX_UNITS),
			dual_grammar: dual
				? `- Two arms (${arms.join(", ")}): prefix each arm's units with L: or R:, e.g. "L:MV_UP*2 R:MV_LEFT"; an omitted arm holds STILL, and the two arms' units run together step by step.\n`
				: "",
			dual_rules: dual
				? [
						"",
						"DUAL-ARM COORDINATION",
						"- Move both arms together only when their paths and destinations are clearly independent.",
						"- For one small shared destination, place with one arm at a time; keep the other arm clear (omit it).",
						"- Do not pair GRASP or RELEASE on one arm with motion of the other unless the views make the action",
						"  obviously collision-free. For handover, the giver holds STILL until the receiver visibly grasps.",
						"",
					].join("\n")
				: "",
			example: dual ? "L:MV_LEFT" : "MV_LEFT",
			task,
		};
		const dynamic = {
			mode: dual
				? `DUAL ARM: the images are the robot's (${arms.join(", ")}).`
				: "SINGLE ARM: send one arm's units, no L:/R: prefixes.",
			state: {
				recording: Boolean(s.recording),
				steps: s.steps,
				gripper_closed: s.closed,
				last: s.last,
				message: s.message,
				solved: g.observe().solved,
			},
			recent_executed_commands: recent.slice(-6).map((c) => c.join(" ")),
		};
		const text = TEMPLATE.replace(/\{\{(\w+)\}\}/g, (m, k: string) => vars[k] ?? m).trim();
		return `${text}\n\nCURRENT INPUT (variable):\n${JSON.stringify(dynamic)}`;
	}

	/** One observe-decide-(act) transaction (run_cycle). */
	async function runCycle() {
		cycle++;
		const started = Date.now();
		const s = g.state();
		if (!s.available) throw new Error("no robot with action units is up (run the robot with --units)");
		const max = Number(pi.getFlag("gumi-operator-max-steps")) || 150;
		if (executed >= max) throw new Error(`operator max steps (${max}) reached`);
		if (s.root && !s.recording) g.record("start");
		const arms = s.arms;
		const dual = arms.length > 1;
		const seen = g.observe();
		if (!seen.ctx) throw new Error("the session has not started");
		// Nothing to look at yet: STOP holds one step and returns the first observation (not recorded).
		if (!seen.obs) await g.look();
		const obs = g.observe().obs;
		if (!obs) throw new Error("the robot returned no observation to decide on");
		const task = String((seen.task.task as string | undefined) ?? JSON.stringify(seen.task.robot_task ?? {}));
		const text = prompt(task, arms, dual);
		const before = marker();
		status = "The operator model is analyzing the current views";
		publish();
		const ask = async () => {
			const reply = await askVlm(
				seen.ctx as NonNullable<typeof seen.ctx>,
				modelRef(),
				pi.getThinkingLevel(),
				text,
				obs.images,
				seen.ctx?.signal,
			);
			pi.events.emit(VLM_COST_EVENT, reply.cost);
			return reply;
		};
		// A truncated or non-JSON answer is asked once more (_complete_decision).
		let reply = await ask();
		let attempts = 1;
		if (parseJson(reply.text) === undefined) {
			reply = await ask();
			attempts = 2;
		}
		const threshold = Number(pi.getFlag("gumi-operator-confidence"));
		const { decision, steps } = normalizeDecision(parseJson(reply.text), {
			arms: dual ? arms : [ARM],
			vocabulary: s.vocabulary.filter((u) => u !== "DONE"),
			threshold: Number.isFinite(threshold) ? threshold : 0.55,
			rt: s.rt,
		});
		lastDecision = decision;
		const event: Record<string, unknown> = {
			cycle,
			time: new Date().toISOString(),
			kind: "decision",
			model: reply.model,
			decision,
			decision_attempts: attempts,
			latency_ms: Date.now() - started,
			state_before: { steps: s.steps, closed: s.closed, recording: s.recording },
		};
		const done = (outcome: string, message: string, paused = true) => {
			record({ ...event, outcome, message });
			if (paused) pause(`Paused: ${message}`);
			else status = message;
		};
		if (decision.pause)
			return done(
				"paused",
				decision.confidence >= threshold
					? `the operator asked for a pause: ${decision.next_goal}`
					: "confidence below threshold",
			);
		if (decision.finish) {
			if (!g.observe().solved)
				return done("finish_rejected", "the operator reported completion, but the robot's success flag is not set");
			if (g.state().recording) {
				const saved = g.record("save", true);
				record({ ...event, outcome: "saved", dir: saved.dir });
				return pause(`rollout saved: ${saved.dir}`);
			}
			return done("complete_not_saved", "the operator verified completion; recording is off");
		}
		const units = (dual ? arms : [ARM]).map((a) => steps[0][a]);
		if (oscillates(recent, units)) return done("oscillation_rejected", "inverse-command oscillation detected");
		if (abortAction) return done("discarded_by_pause", "paused before the action; the decision was dropped", false);
		if (marker() !== before)
			return done("stale_observation", "the teleop state changed while the model decided; re-observing", false);
		status = `Executing ${decision.command}`;
		publish();
		const result = await g.step(
			dual ? stepsBody(steps, arms) : { units: steps.map((st) => st[ARM]) },
			"gpt-operator",
		);
		executed += result.executed;
		recent = [...recent, units].slice(-8);
		record({
			...event,
			outcome: result.ok ? "executed" : "failed",
			request: decision.command,
			executed: result.executed,
		});
		if (!result.ok) throw new Error(result.results.find((x) => !x.ok)?.error ?? "teleop step failed");
		status = `Executed ${decision.command}`;
	}

	/** The worker: continuous cycles, or the queued single ones. */
	async function loop() {
		if (running) return;
		running = true;
		try {
			while (mode === "continuous" || pendingOnce > 0) {
				const single = mode !== "continuous";
				if (single) pendingOnce--;
				busy = true;
				publish();
				try {
					await runCycle();
				} catch (err) {
					lastError = err instanceof Error ? err.message : String(err);
					record({ cycle, time: new Date().toISOString(), kind: "error", message: lastError });
					pause(`Paused after error: ${lastError}`);
				} finally {
					busy = false;
					publish();
				}
				if (mode === "continuous") await new Promise((r) => setTimeout(r, INTERVAL_MS));
			}
		} finally {
			running = false;
		}
	}

	return {
		state,
		/** run | pause | step | status (the dashboard's /gumi/operator; status only reads). */
		control(action: string) {
			if (!enabled())
				throw Object.assign(new Error("the VLM operator is off (--gumi-operator <model>)"), { status: 409 });
			if (action === "status") return { ok: true, operator: state() };
			if (action === "run") {
				abortAction = false;
				mode = "continuous";
				lastError = null;
				status = "Running";
			} else if (action === "step") {
				abortAction = false;
				pendingOnce++;
				lastError = null;
				status = "One operator step queued";
			} else if (action === "pause") {
				abortAction = true;
				pause("Paused by the dashboard");
				return { ok: true, operator: state() };
			} else throw Object.assign(new Error("action must be run, pause, step or status"), { status: 422 });
			publish();
			void loop();
			return { ok: true, operator: state() };
		},
		/** Resolves when no cycle runs (tests). */
		idle: async () => {
			while (running) await new Promise((r) => setTimeout(r, 5));
		},
	};
}

/** Two arms: the validated steps as the teleop page's per-arm body. */
function stepsBody(steps: Step[], arms: readonly string[]) {
	return Object.fromEntries(arms.map((a) => [a, steps.map((s) => s[a])]));
}
