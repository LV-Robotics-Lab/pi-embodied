/**
 * HumanCLAW's plan-and-skill planner with verifier (agent/planner.py HumanClawBenchPSVPlanSkillPlanner,
 * prompt v4 + verifier v3, evaluation/evaluator.py `_history_item`), ported so that every request is
 * the one the Python sends: the same prompt text, the same retries at the same state (5 attempts,
 * backoff 1/2/4/8 s; after 5 planner failures one Walk<forward><slow> without the verifier, after 5
 * verifier failures the planner's proposal), the same verifier routes and replacement, the same
 * turn-for-sit gating and the same bounded text history (no images).
 *
 * It knows nothing of pi: `ask(prompt)` sends one request (the prompt text, then the current ego
 * image) and returns the reply text and usage. ./provider.ts wires it to a base model as the
 * `humanclaw-psv/<base>` planner.
 *
 * Copyright 2026 The HumanCLAW Authors (github.com/Human-CLAW/HumanCLAW @c4f9351).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: agent/planner.py, agent/verifiers/v3.py, agent/prompts/v4.py render,
 * agent/utils.py parse_json_loose / clip_text and evaluator._history_item ported to TypeScript.
 */

import { chooserAction, pyStr, pyStrip, type SkillCall, skillToText, toJson, truthy } from "./actions.ts";
import {
	V3_ACTION_SPACE,
	V3_CLIMB_UP_GUIDELINE,
	V3_CLIMB_UP_RETURN_SCHEMA,
	V3_COMMON_GUIDELINE,
	V3_SITTING_ACTION_SPACE,
	V3_STOP_AFTER_SIT_GUIDELINE,
	V3_STOP_AFTER_SIT_RETURN_SCHEMA,
	V3_STOP_GUIDELINE,
	V3_STOP_RETURN_SCHEMA,
	V3_SYSTEM_PROMPT,
	V3_TASK_TEMPLATE,
	V3_TURN_FOR_SIT_GUIDELINE,
	V3_TURN_FOR_SIT_RETURN_SCHEMA,
	V3_WALK_GUIDELINE,
	V3_WALK_RETURN_SCHEMA,
	V4_TEMPLATE,
} from "./templates.ts";

type Json = Record<string, unknown>;
export const MAX_ATTEMPTS = 5;
export const BACKOFF_S = [1.0, 2.0, 4.0, 8.0];
export const PLAN_HORIZON_STEPS = 6;
export const MAX_HISTORY = 10;

/** One VLM stage (types.py PSVStageOutput). */
export type Stage = { stage: string; raw: Json; raw_output: string; prompt: string; error?: string; usage: Json };
/** One decision (types.py PlannerResult). */
export type Decision = {
	raw_plan: Json;
	action: SkillCall;
	planner_skill: Json;
	verifier: Json;
	stages: Stage[];
};
/** One request to the base model and what it answered. */
export type Ask = (prompt: string, stage: string, signal?: AbortSignal) => Promise<{ text: string; usage?: Json }>;

// ---------------------------------------------------------------------------
// agent/utils.py

/** clip_text: one line, at most `limit` characters (code points) with a "..." marker. */
export function clipText(value: unknown, limit: number): string {
	const text = value === undefined || value === null ? "" : pyStrip(pyStr(value)).replaceAll("\n", " ");
	const chars = Array.from(text);
	return chars.length <= limit ? text : `${chars.slice(0, limit - 3).join("")}...`;
}

/** The end index of the JSON value starting at `i` (json.JSONDecoder.raw_decode's extent), or -1. */
function valueEnd(s: string, i: number): number {
	let depth = 0;
	let inString = false;
	for (let j = i; j < s.length; j++) {
		const c = s[j];
		if (inString) {
			if (c === "\\") j++;
			else if (c === '"') inString = false;
			continue;
		}
		if (c === '"') inString = true;
		else if (c === "{" || c === "[") depth++;
		else if (c === "}" || c === "]") {
			depth--;
			if (depth === 0) return j + 1;
		}
	}
	return -1;
}
const isObject = (v: unknown): v is Json => typeof v === "object" && v !== null && !Array.isArray(v);

/** parse_json_loose: the reply as a JSON object, tolerating <think> blocks, fences and prose. */
export function parseJsonLoose(text: string): Json {
	const clean = pyStrip(
		pyStrip(text.replace(/<think>[\s\S]*?<\/think>/g, ""))
			.replaceAll("```json", "")
			.replaceAll("```", ""),
	);
	try {
		const parsed = JSON.parse(clean);
		if (isObject(parsed)) return parsed;
	} catch {}
	for (let i = clean.indexOf("{"); i >= 0; i = clean.indexOf("{", i + 1)) {
		const end = valueEnd(clean, i);
		if (end < 0) continue;
		try {
			const parsed = JSON.parse(clean.slice(i, end));
			if (isObject(parsed)) return parsed;
		} catch {}
	}
	throw new Error(`ValueError: Could not parse JSON from model output: ${Array.from(text).slice(0, 300).join("")}`);
}

/** json.dumps(value, ensure_ascii=False, indent=2) for the JSON the prompts carry. */
const dumps = (value: unknown) => JSON.stringify(value, null, 2);

/** str.format over a template: {name} fields, {{ and }} escapes. */
function pyFormat(template: string, fields: Record<string, string>): string {
	return template.replace(/\{\{|\}\}|\{(\w+)\}/g, (m, name: string | undefined) => {
		if (m === "{{") return "{";
		if (m === "}}") return "}";
		const v = fields[name as string];
		if (v === undefined) throw new Error(`KeyError: ${name}`);
		return v;
	});
}

// ---------------------------------------------------------------------------
// prompts/v4.py and verifiers/v3.py

export const renderPlanner = (input: Json) => pyFormat(V4_TEMPLATE, { input_json: dumps(input) });

type Route = { name: string; actionSpace: string; guideline: string; questions: string; schema: string };
const truthyFlag = (v: unknown) => v === true || (typeof v === "string" && pyStrip(v).toLowerCase() === "true");
const infoOf = (plan: Json | undefined): Json => (isObject(plan?.additional_info) ? plan.additional_info : {});

/** verifiers/v3.py route_prompt. */
export function routePrompt(proposed: SkillCall, tokens: string[], skiller: Json | undefined): Route | undefined {
	const info = infoOf(skiller);
	if (proposed.skill === "turn" && truthyFlag(info.turn_for_sit))
		return {
			name: "turn_for_sit",
			actionSpace: V3_SITTING_ACTION_SPACE,
			guideline: V3_COMMON_GUIDELINE + V3_TURN_FOR_SIT_GUIDELINE,
			questions:
				"1. Is the target sitting surface touching the body at zero distance, or is there still distance to cover?\n" +
				"2. Does this Turn rotate the body away so the target will be behind for Sit down?\n" +
				"3. Should the proposed Turn be accepted or replaced with a movement that continues toward the target?",
			schema: V3_TURN_FOR_SIT_RETURN_SCHEMA,
		};
	if (proposed.skill === "stand" && truthyFlag(info.stop_after_sit))
		return {
			name: "stop_after_sit",
			actionSpace: V3_SITTING_ACTION_SPACE,
			guideline: V3_COMMON_GUIDELINE + V3_STOP_AFTER_SIT_GUIDELINE,
			questions:
				"1. Is there clear evidence that sitting failed?\n" +
				"2. Does the image show the back of the blue legs contacting the object rather than floating or unsupported?\n" +
				"3. Should Stop/Stand be accepted or replaced?",
			schema: V3_STOP_AFTER_SIT_RETURN_SCHEMA,
		};
	if (proposed.skill === "walk_forward") {
		const speed = tokens.length > 1 ? tokens[1] : "slow";
		return {
			name: "walk",
			actionSpace: V3_ACTION_SPACE,
			guideline: V3_COMMON_GUIDELINE + V3_WALK_GUIDELINE,
			questions:
				"1. What is in the straight-forward walking lane?\n" +
				"2. How far is the nearest object or obstacle in that lane, in meters?\n" +
				`3. If the humanoid executes Walk<forward><${speed}> for the next 0.5 seconds, would it unnecessarily collide with anything?\n` +
				"4. If this walk is not appropriate, what should the final action be?",
			schema: V3_WALK_RETURN_SCHEMA.replaceAll("{lane_checked}", "forward"),
		};
	}
	if (proposed.skill === "stand")
		return {
			name: "stop",
			actionSpace: V3_ACTION_SPACE,
			guideline: V3_COMMON_GUIDELINE + V3_STOP_GUIDELINE,
			questions:
				"1. What is the high-level goal in the instruction, and is it a navigation goal, or an environment-interaction goal?\n" +
				"2. Has the high-level goal been fully completed according to the current image? Give the reason whether the answer is yes or no.\n" +
				"3. If this is navigation: has the target been found? How far is it to the target? Are you touching it (~zero distance)?\n" +
				"4. If this is environment interaction: has that interaction been completed?\n" +
				"5. If the goal is complete, accept Stop/Stand. Otherwise, choose one other non-Stop skill from the action space that best continues the goal.",
			schema: V3_STOP_RETURN_SCHEMA,
		};
	if (proposed.skill === "step_climb_up")
		return {
			name: "climb_up",
			actionSpace: V3_ACTION_SPACE,
			guideline: V3_COMMON_GUIDELINE + V3_CLIMB_UP_GUIDELINE,
			questions:
				"1. What is the approximate angle between the visible stair/riser edge and the horizontal image direction, in degrees?\n" +
				"2. Is the humanoid facing the stairs from the front, or is it side-facing the stairs?\n" +
				"3. How far is the first stair/riser from the feet, in meters?\n" +
				"4. What angle range counts as facing the stairs from the front?",
			schema: V3_CLIMB_UP_RETURN_SCHEMA,
		};
	return undefined;
}

/** verifiers/v3.py render: the system text once, then the task. */
export const renderVerifier = (route: Route, proposedName: string, input: Json) =>
	`${V3_SYSTEM_PROMPT}\n\n${pyFormat(V3_TASK_TEMPLATE, {
		proposed_action_name: proposedName,
		action_space: route.actionSpace,
		action_guideline: route.guideline,
		input_json: dumps(input),
		questions: route.questions,
		return_json: route.schema,
	})}`;

/** verifiers/v3.py verifier_action: the approved or corrected action. */
export function verifierAction(plan: Json, proposed: SkillCall): SkillCall {
	const verdict = pyStrip(pyStr(truthy(plan.verdict) ? plan.verdict : "accept")).toLowerCase();
	const finalName = pyStrip(pyStr(truthy(plan.final_action_name) ? plan.final_action_name : ""));
	if (verdict === "replace" && finalName && !finalName.includes("<final action>"))
		return chooserAction({ action_id: plan.final_action_id, action_name: finalName });
	return proposed;
}

/** verifiers/v3.py normalize_verifier_plan, plus the planner's verifier_version. */
export function normalizeVerifier(plan: Json, proposed: SkillCall, final: SkillCall): Json {
	const n: Json = { ...plan };
	if (!("verdict" in n)) n.verdict = "accept";
	if (!("reason" in n)) n.reason = "";
	n.visual_state_description = truthy(n.lane_observation) ? n.lane_observation : truthy(n.reason) ? n.reason : "";
	n.reasoning_and_reflection = n.reason;
	n.executable_plan = [{ action_id: final.action_id, action_name: final.action_name || skillToText(final) }];
	n.at_target = false;
	n.proposed_action = toJson(proposed);
	n.final_action = toJson(final);
	n.verifier_version = "v3";
	return n;
}

/** evaluator._history_item: what the next planner call reads of a decision. */
export function historyItem(step: number, d: Decision): Json {
	return {
		step,
		action: toJson(d.action),
		action_text: skillToText(d.action),
		visual_state_description: d.raw_plan.visual_state_description ?? "",
		reasoning_and_reflection: d.raw_plan.reasoning_and_reflection ?? "",
		current_subgoal: d.raw_plan.current_subgoal ?? "",
		language_plan: d.raw_plan.language_plan ?? "",
		planner_skill: d.planner_skill,
		verifier: d.verifier,
	};
}

const or = (...values: unknown[]) => values.find(truthy) ?? values[values.length - 1];

/** The planner (HumanClawBenchPSVPlanSkillPlanner) for one episode at a time. */
export class PsvPlanner {
	instruction: string | undefined;
	currentPlan: string | null = null;
	currentStep = 0;
	turnForSitVerified = false;

	private readonly ask: Ask;
	private readonly sleep: (s: number, signal?: AbortSignal) => Promise<void>;
	readonly maxHistory: number;
	readonly planHorizon: number;
	constructor(
		ask: Ask,
		sleep: (s: number, signal?: AbortSignal) => Promise<void> = abortableSleep,
		maxHistory = MAX_HISTORY,
		planHorizon = PLAN_HORIZON_STEPS,
	) {
		this.ask = ask;
		this.sleep = sleep;
		this.maxHistory = maxHistory;
		this.planHorizon = planHorizon;
	}

	reset(instruction: string) {
		this.instruction = instruction;
		this.currentPlan = null;
		this.currentStep = 0;
		this.turnForSitVerified = false;
	}

	/** The planner state (session entries restore it on resume). */
	state() {
		return {
			current_plan: this.currentPlan,
			current_step: this.currentStep,
			turn_for_sit_sequence_verified: this.turnForSitVerified,
		};
	}
	restore(s: { current_plan: string | null; current_step: number; turn_for_sit_sequence_verified: boolean }) {
		this.currentPlan = s.current_plan;
		this.currentStep = s.current_step;
		this.turnForSitVerified = s.turn_for_sit_sequence_verified;
	}

	private async call(prompt: string, stage: string, signal?: AbortSignal): Promise<Stage> {
		let reply: { text: string; usage?: Json };
		try {
			reply = await this.ask(prompt, stage, signal);
		} catch (err) {
			if (signal?.aborted) throw err;
			const e = err as Error;
			return {
				stage,
				raw: {},
				raw_output: "",
				prompt,
				error: `${e?.name ?? "Error"}: ${e?.message ?? err}`,
				usage: {},
			};
		}
		try {
			return { stage, raw: parseJsonLoose(reply.text), raw_output: reply.text, prompt, usage: reply.usage ?? {} };
		} catch (err) {
			return {
				stage,
				raw: {},
				raw_output: reply.text,
				prompt,
				error: (err as Error).message,
				usage: reply.usage ?? {},
			};
		}
	}

	/** _call_with_retries: up to 5 attempts at this state, backing off between them. */
	async callWithRetries(prompt: string, stage: string, signal?: AbortSignal): Promise<Stage[]> {
		const attempts: Stage[] = [];
		for (let i = 0; i < MAX_ATTEMPTS; i++) {
			const a = await this.call(prompt, stage, signal);
			attempts.push(a);
			if (!a.error) break;
			if (i < MAX_ATTEMPTS - 1) await this.sleep(BACKOFF_S[i], signal);
		}
		return attempts;
	}

	/** _plan_skill_history: the bounded semantic rows of earlier decisions. */
	planHistory(history: Json[]): Json[] {
		return history.slice(-this.maxHistory).map((item) => {
			const ps = isObject(item.planner_skill) ? item.planner_skill : {};
			const row: Json = {
				step: "step" in item ? item.step : "?",
				visible_state: clipText(or(ps.visible_state, item.visual_state_description ?? ""), 220),
				mid_level_progress_analysis: clipText(
					or(ps.mid_level_progress_analysis, item.reasoning_and_reflection ?? ""),
					220,
				),
				mid_level_goal: clipText(or(ps.mid_level_goal, item.language_plan, item.current_subgoal ?? ""), 160),
				low_level_action_reasoning: clipText(ps.low_level_action_reasoning ?? "", 180),
				action_name: clipText(or(ps.action_name, item.action_text ?? ""), 80),
			};
			const rejection = rejectionSummary(item);
			if (rejection) row.verifier_rejection = clipText(rejection, 240);
			return row;
		});
	}

	/** _recent_skill_history_text: the executed actions, runs grouped ("X x3; Y x1"). */
	recentActions(history: Json[]): string {
		const actions: string[] = [];
		for (const item of history.slice(-this.maxHistory)) {
			let text = pyStrip(pyStr(truthy(item.action_text) ? item.action_text : ""));
			if (!text && isObject(item.action)) text = pyStrip(pyStr(or(item.action.action_name, item.action.skill, "")));
			if (text) actions.push(clipText(text, 80));
		}
		if (!actions.length) return "none";
		const grouped: [string, number][] = [];
		for (const a of actions) {
			const last = grouped[grouped.length - 1];
			if (last && last[0] === a) last[1]++;
			else grouped.push([a, 1]);
		}
		return grouped.map(([a, n]) => `${a} x${n}`).join("; ");
	}

	/** The planner prompt of the next decision (_plan_skill_prompt). */
	plannerPrompt(history: Json[]): string {
		if (this.instruction === undefined) throw new Error("Call reset(task) before act().");
		return renderPlanner({
			goal: this.instruction,
			current_ego_view_image: "attached",
			current_step: history.length,
			history: this.planHistory(history),
		});
	}

	/** act(): planner, then the verifier where its route asks for one; the final action. */
	async act(history: Json[], signal?: AbortSignal): Promise<Decision> {
		if (this.instruction === undefined) throw new Error("Call reset(task) before act().");
		const stages: Stage[] = [];
		const planAttempts = await this.callWithRetries(this.plannerPrompt(history), "percept_mid_low", signal);
		stages.push(...planAttempts);
		const planStage = planAttempts[planAttempts.length - 1];
		const fallback = Boolean(planStage.error);
		let plan: Json;
		if (fallback) {
			planStage.error = `${planStage.error}; planner retry limit reached; executing Walk<forward><slow>`;
			plan = {
				visible_state: "",
				mid_level_progress_analysis:
					"Planner unavailable after five attempts; use one short forward step and re-plan from the next observation.",
				mid_level_goal: this.currentPlan || "Continue toward the task goal.",
				low_level_action_reasoning: "Retry-exhausted recovery action.",
				action_id: 0,
				action_name: "Walk<forward><slow>",
				additional_info: { fallback: "walk_slow_after_retry_exhausted" },
			};
		} else plan = planStage.raw;

		// _apply_plan_skill_output
		let planText = pyStrip(pyStr(or(plan.mid_level_goal, this.currentPlan, "")));
		if (!planText) planText = "Stop/Stand";
		if (this.currentPlan === null || planText !== this.currentPlan || this.currentStep >= this.planHorizon) {
			this.currentPlan = planText;
			this.currentStep = 0;
		} else this.currentPlan = planText;

		// _planner_from_plan_skill / _skiller_from_plan_skill
		let goal = pyStrip(pyStr(truthy(plan.mid_level_goal) ? plan.mid_level_goal : ""));
		if (!goal) goal = this.currentPlan || "Stop/Stand";
		const planner: Json = {
			visible_state: pyStr(truthy(plan.visible_state) ? plan.visible_state : ""),
			plan_status: isFinishGoal(plan) ? "finish" : "continue",
			plan: goal,
			reason: pyStr(truthy(plan.mid_level_progress_analysis) ? plan.mid_level_progress_analysis : ""),
		};
		let skiller: Json = {
			plan_following_state: pyStr(truthy(plan.mid_level_progress_analysis) ? plan.mid_level_progress_analysis : ""),
			skill_reason: pyStr(truthy(plan.low_level_action_reasoning) ? plan.low_level_action_reasoning : ""),
			current_subgoal: pyStr(truthy(plan.mid_level_goal) ? plan.mid_level_goal : ""),
			action_id: plan.action_id ?? null,
			action_name: pyStr(truthy(plan.action_name) ? plan.action_name : ""),
			additional_info: isObject(plan.additional_info) ? plan.additional_info : {},
		};
		const proposed = chooserAction(skiller);

		// _gate_turn_for_sit_verifier
		const info: Json = { ...infoOf(skiller) };
		const requested = proposed.skill === "turn" && truthyFlagInfo(info.turn_for_sit);
		if (requested && this.turnForSitVerified) {
			delete info.turn_for_sit;
			skiller = { ...skiller, additional_info: info };
		}

		let verifierPlan: Json;
		let final = proposed;
		const prompt = fallback ? "" : this.verifierPrompt(skiller, proposed, history);
		if (fallback) {
			verifierPlan = normalizeVerifier(
				{
					verdict: "accept",
					reason: "planner retry exhausted; executed Walk<forward><slow>",
					fallback: "walk_slow_after_retry_exhausted",
				},
				proposed,
				final,
			);
		} else if (!prompt) {
			verifierPlan = normalizeVerifier(
				{ verdict: "accept", reason: "no verifier for this action type" },
				proposed,
				final,
			);
		} else {
			const attempts = await this.callWithRetries(prompt, "verifier", signal);
			stages.push(...attempts);
			const last = attempts[attempts.length - 1];
			let raw: Json;
			if (last.error) {
				raw = {
					verdict: "accept",
					reason: "verifier unavailable after 5 attempts; accepted the planner-proposed action",
					fallback: "accept_after_retry_exhausted",
				};
			} else {
				raw = last.raw;
				final = verifierAction(raw, proposed);
			}
			verifierPlan = normalizeVerifier(raw, proposed, final);
		}

		// _update_turn_for_sit_state
		this.turnForSitVerified = requested && final.skill === "turn";
		if (this.currentPlan) this.currentStep++;
		return {
			raw_plan: combinedRawPlan(plan, planner, skiller, verifierPlan),
			action: final,
			planner_skill: plan,
			verifier: verifierPlan,
			stages,
		};
	}

	/** _verifier_prompt: the route's prompt for the proposal, or "" when it has no route. */
	verifierPrompt(skiller: Json, proposed: SkillCall, history: Json[]): string {
		const name = proposed.action_name || skillToText(proposed);
		const tokens = [...name.matchAll(/<([^>]+)>/g)].map((m) => pyStrip(m[1]).toLowerCase().replaceAll(" ", "_"));
		const shown = name.replaceAll("left_forward", "left forward").replaceAll("right_forward", "right forward");
		const input: Json = { goal: this.instruction, proposed_action: shown };
		const info = infoOf(skiller);
		if (truthy(info)) input.additional_info = info;
		if (proposed.skill === "stand" && truthyFlagInfo(info.stop_after_sit))
			input.recent_actions = this.recentActions(history);
		const route = routePrompt(proposed, tokens, skiller);
		return route ? renderVerifier(route, shown, input) : "";
	}
}

/** _is_additional_info_true (bool, or a "true" string). */
const truthyFlagInfo = (v: unknown) =>
	typeof v === "boolean" ? v : typeof v === "string" && pyStrip(v).toLowerCase() === "true";

function isFinishGoal(plan: Json): boolean {
	const goal = pyStrip(pyStr(truthy(plan.mid_level_goal) ? plan.mid_level_goal : "")).toLowerCase();
	return ["stop/stand", "stop stand", "stop", "stand"].includes(goal);
}

/** _previous_verifier_rejection_summary. */
function rejectionSummary(item: Json): string {
	const v = item.verifier;
	if (!isObject(v)) return "";
	if (pyStrip(pyStr(truthy(v.verdict) ? v.verdict : "")).toLowerCase() !== "replace") return "";
	const nameOf = (a: unknown) => (isObject(a) ? pyStrip(pyStr(or(a.action_name, a.skill, ""))) : "");
	const proposed = nameOf(v.proposed_action);
	const final = nameOf(v.final_action);
	const reason = pyStrip(pyStr(truthy(v.reason) ? v.reason : ""));
	const pieces = ["Verifier replaced"];
	if (proposed) pieces.push(proposed);
	if (final) pieces.push(`with ${final}`);
	if (reason) pieces.push(`because ${reason}`);
	return pyStrip(pieces.join(" "));
}

/** _combined_raw_plan. */
function combinedRawPlan(plan: Json, planner: Json, skiller: Json, verifier: Json): Json {
	const get = (o: Json, k: string, d: unknown) => (k in o ? o[k] : d);
	const parts = [
		`planner: ${pyStr(get(planner, "reason", ""))}`,
		`skiller: ${pyStr(get(skiller, "skill_reason", get(skiller, "reasoning_and_reflection", "")))}`,
		`verifier: ${pyStr(get(verifier, "reason", get(verifier, "reasoning_and_reflection", "")))}`,
	];
	const final = isObject(verifier.final_action) ? verifier.final_action : {};
	const actionName = pyStrip(pyStr(or(final.action_name, final.name, "")));
	const skill = pyStrip(pyStr(truthy(final.skill) ? final.skill : ""));
	const stand = skill === "stand" || actionName === "Stop/Stand" || final.action_id === 1;
	return {
		visual_state_description: or(
			verifier.visual_state_description,
			skiller.visual_state_description,
			skiller.plan_following_state,
			planner.visible_state,
			"",
		),
		reasoning_and_reflection: parts.filter((p) => pyStrip(p)).join(" | "),
		current_subgoal: or(skiller.current_subgoal, skiller.action_name, planner.plan ?? ""),
		language_plan: pyStr(truthy(plan.mid_level_goal) ? plan.mid_level_goal : ""),
		at_target: Boolean(truthy(verifier.at_target ?? false) || (isFinishGoal(plan) && stand)),
	};
}

/** time.sleep that an abort ends. */
export function abortableSleep(seconds: number, signal?: AbortSignal): Promise<void> {
	return new Promise((resolve, reject) => {
		if (signal?.aborted) return reject(new Error("aborted"));
		const t = setTimeout(resolve, seconds * 1000);
		signal?.addEventListener(
			"abort",
			() => {
				clearTimeout(t);
				reject(new Error("aborted"));
			},
			{ once: true },
		);
	});
}
