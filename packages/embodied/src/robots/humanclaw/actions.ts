/**
 * HumanCLAW's action space: the units vocabulary (WALK / TURN_* / SIDE_* / STEP_BACK / CLIMB_UP /
 * WALK_DOWN / SIT / STOP), and HumanCLAW's own mapping from an action to the motion skill it runs,
 * ported byte for byte from agent/planner.py `_chooser_action` (clamps `_clamp_degree`,
 * `_clamp_short_distance`, `_clamp_side_step_distance`; Python's round-half-even and "%.2f";
 * `int()` of whatever action_id a provider sends) and agent/skills.py `SkillCall` / `skill_to_text`.
 *
 *   chooserAction({action_id: 2, action_name: "Turn<left><37.5>"})
 *     -> {skill: "turn", cond: 37.5, action_id: null, action_name: "Turn<left><38>"}
 *   unitOf(call) -> {unit: "TURN_LEFT", param: 37.5};  skillCall("TURN_LEFT", 37.5) -> the same call
 *
 * Copyright 2026 The HumanCLAW Authors (github.com/Human-CLAW/HumanCLAW @c4f9351).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: agent/planner.py and agent/skills.py ported to TypeScript.
 */

import type { CustomUnit } from "../../modes/units/index.ts";

/** One executable motion skill (agent/skills.py SkillCall; source_index is always 0 here). */
export type SkillCall = {
	skill: string;
	cond: number | number[] | null;
	action_id: number | null;
	action_name: string | null;
	source_index: 0;
};
const call = (skill: string, cond: SkillCall["cond"], action_name: string, action_id: number | null = null) =>
	({ skill, cond, action_id, action_name, source_index: 0 }) as SkillCall;

/** SkillCall.to_json(). */
export const toJson = (a: SkillCall) => ({
	action_id: a.action_id,
	action_name: a.action_name,
	skill: a.skill,
	cond: a.cond,
	source_index: a.source_index,
});

/** skill_to_text: the action text of prompts, logs and history. */
export function skillToText(a: SkillCall): string {
	if (a.action_name) return a.action_id === null ? a.action_name : `action id ${a.action_id}: ${a.action_name}`;
	if (a.cond === null) return a.skill;
	return `${a.skill} cond=${pyStr(a.cond)}`;
}

export const STAND: SkillCall = call("stand", null, "Stop/Stand");

// ---------------------------------------------------------------------------
// Python semantics the port depends on

/** Python truthiness (`x or y`). */
export const truthy = (v: unknown) =>
	!(
		v === undefined ||
		v === null ||
		v === false ||
		v === 0 ||
		v === "" ||
		(Array.isArray(v) && v.length === 0) ||
		(typeof v === "object" && !Array.isArray(v) && Object.keys(v as object).length === 0)
	);

/** Python str() of a JSON value (repr for containers is approximated by JSON). */
export function pyStr(v: unknown): string {
	if (v === undefined || v === null) return "None";
	if (v === true) return "True";
	if (v === false) return "False";
	if (typeof v === "string") return v;
	if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(v);
	return JSON.stringify(v);
}

/** str.isspace() characters (str.strip()'s set; JS trim() differs on \x1c-\x1f, \x85 and ﻿). */
const WS = "\\t\\n\\x0b\\x0c\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const STRIP = new RegExp(`^[${WS}]+|[${WS}]+$`, "g");
export const pyStrip = (s: string) => s.replace(STRIP, "");

/** int(v) for a JSON value; throws where Python raises. */
export function pyInt(v: unknown): number {
	if (typeof v === "boolean") return v ? 1 : 0;
	if (typeof v === "number") {
		if (!Number.isFinite(v)) throw new Error("cannot convert float to integer");
		return Math.trunc(v);
	}
	if (typeof v === "string") {
		const s = pyStrip(v);
		if (!/^[+-]?\d+(?:_\d+)*$/.test(s)) throw new Error(`invalid literal for int(): ${v}`);
		return Number.parseInt(s.replaceAll("_", ""), 10);
	}
	throw new Error("int() argument must be a string or a number");
}

/** A finite double as mantissa * 2^exponent (both exact). */
function decompose(x: number): [bigint, bigint] {
	const view = new DataView(new ArrayBuffer(8));
	view.setFloat64(0, Math.abs(x));
	const bits = view.getBigUint64(0);
	const exp = Number((bits >> 52n) & 0x7ffn);
	const frac = bits & ((1n << 52n) - 1n);
	return exp === 0 ? [frac, BigInt(-1074)] : [frac | (1n << 52n), BigInt(exp - 1075)];
}

/**
 * x rounded to `digits` decimals half-to-even on its exact binary value, as Python's round() and
 * "%.{digits}f" do (JS toFixed rounds exact ties up: 0.375 -> "0.38" in both, 0.625 -> "0.63" vs "0.62").
 */
export function pyFixed(x: number, digits: number): string {
	const [mant, exp] = decompose(x);
	const scale = 10n ** BigInt(digits);
	if (exp < 0n) {
		const denom = 1n << -exp;
		const scaled = mant * scale;
		const rem = scaled % denom;
		if (rem * 2n === denom) {
			let n = scaled / denom;
			if (n % 2n === 1n) n += 1n;
			const s = n.toString().padStart(digits + 1, "0");
			const out = digits ? `${s.slice(0, s.length - digits)}.${s.slice(s.length - digits)}` : s;
			return x < 0 ? `-${out}` : out;
		}
	}
	return x.toFixed(digits);
}
/** int(round(x)). */
export const pyRound = (x: number) => Number(pyFixed(x, 0));

// ---------------------------------------------------------------------------
// agent/planner.py

/** _angle_bracket_tokens: lower-cased <...> arguments, spaces as underscores. */
export const angleTokens = (name: string) =>
	[...name.matchAll(/<([^>]+)>/g)].map((m) => pyStrip(m[1]).toLowerCase().replaceAll(" ", "_"));

const clampDegree = (v: number) => Math.max(10.0, Math.min(120.0, v));
const clampShortDistance = (v: number) => Math.max(0.1, Math.min(0.6, v));
const clampSideStep = (v: number) => Math.max(0.1, Math.min(0.5, v));
const firstNumber = (text: string, signed: boolean) => {
	const m = (signed ? /-?\d+(?:\.\d+)?/ : /\d+(?:\.\d+)?/).exec(text);
	return m ? Number.parseFloat(m[0]) : undefined;
};

/** The action family an action_name names when action_id is missing or not an int (the planner's recovery path). */
function familyOfName(lowered: string): number {
	if (lowered.includes("downstairs") || lowered.includes("climb down")) return 7;
	if (lowered.startsWith("walk")) return 0;
	if (lowered.includes("stop") || lowered.includes("stand")) return 1;
	if (lowered.startsWith("turn")) return 2;
	if (lowered.includes("climb")) return 3;
	if (lowered.includes("sit")) return 4;
	if (lowered.includes("step back") || lowered.startsWith("back")) return 5;
	if (lowered.includes("side step") || lowered.includes("sidestep") || lowered.includes("side walk")) return 6;
	return 1;
}

/** _chooser_action: planner (or verifier) JSON -> the SkillCall it runs. */
export function chooserAction(plan: Record<string, unknown>): SkillCall {
	const actionName = pyStrip(pyStr(truthy(plan.action_name) ? plan.action_name : ""));
	let family: number;
	try {
		family = pyInt(plan.action_id);
	} catch {
		family = familyOfName(actionName.toLowerCase());
	}
	const tokens = angleTokens(actionName);
	if (family === 0) {
		const t = tokens.length >= 2 ? tokens[1] : "slow";
		const speed = t === "normal" || t === "fast" ? t : "slow";
		const d = { slow: 0.2, normal: 0.4, fast: 0.6 }[speed];
		return call("walk_forward", [0.0, d, 0.0], `Walk<forward><${speed}>`);
	}
	if (family === 2) {
		const direction = tokens.length >= 1 && tokens[0] === "right" ? "right" : "left";
		const n = firstNumber(tokens.length >= 2 ? tokens[1] : actionName, true);
		const degree = n === undefined ? 10.0 : clampDegree(n);
		return call("turn", direction === "left" ? degree : -degree, `Turn<${direction}><${pyRound(degree)}>`);
	}
	if (family === 3) return call("step_climb_up", [0.28, 0.3], "Climb upstairs<normal>");
	if (family === 4) {
		const n = firstNumber(actionName, false);
		const h = Math.max(0.15, Math.min(0.85, n ?? 0.5));
		return call("sit", h, `Sit down<${pyFixed(h, 2)}>`, 4);
	}
	if (family === 5) {
		const n = firstNumber(actionName, false);
		const d = n === undefined ? 0.25 : clampShortDistance(n);
		return call("step_back", [0.0, -d], `Step back<${pyFixed(d, 2)}>`, 5);
	}
	if (family === 6) {
		const direction = tokens.includes("right") || actionName.toLowerCase().includes("right") ? "right" : "left";
		const n = firstNumber(tokens.length >= 2 ? tokens.slice(1).join(" ") : actionName, false);
		const d = n === undefined ? 0.25 : clampSideStep(n);
		return call("side_walk", direction === "left" ? d : -d, `Side step<${direction}><${pyFixed(d, 2)}>`, 6);
	}
	if (family === 7) return call("step_climb_down", [0.2, 0.4], "Walk downstairs<normal>", 7);
	return STAND;
}

// ---------------------------------------------------------------------------
// the units vocabulary

/** The units: one per HumanCLAW action family, the direction of a turn or side step in the name. */
export const UNITS: readonly CustomUnit[] = [
	{
		name: "WALK",
		description:
			"Walk straight forward for 0.5 s (slow 0.2 m, normal 0.4 m, fast 0.6 m), keeping the facing direction.",
		param: { name: "speed", kind: "enum", values: ["slow", "normal", "fast"], default: "slow" },
	},
	{
		name: "TURN_LEFT",
		description: "Turn left in place by this many degrees within 0.5 s.",
		param: { name: "degree", kind: "number", min: 10, max: 120, unit: "deg" },
	},
	{
		name: "TURN_RIGHT",
		description: "Turn right in place by this many degrees within 0.5 s.",
		param: { name: "degree", kind: "number", min: 10, max: 120, unit: "deg" },
	},
	{
		name: "SIDE_LEFT",
		description: "Side step to the left like a crab-walk, keeping the facing direction.",
		param: { name: "distance", kind: "number", min: 0.1, max: 0.5, unit: "m", default: 0.25 },
	},
	{
		name: "SIDE_RIGHT",
		description: "Side step to the right like a crab-walk, keeping the facing direction.",
		param: { name: "distance", kind: "number", min: 0.1, max: 0.5, unit: "m", default: 0.25 },
	},
	{
		name: "STEP_BACK",
		description: "Step backward, keeping the facing direction.",
		param: { name: "distance", kind: "number", min: 0.1, max: 0.6, unit: "m", default: 0.25 },
	},
	{
		name: "CLIMB_UP",
		description: "Climb upstairs; only when the next riser is at the feet and the stairs are straight ahead.",
	},
	{
		name: "WALK_DOWN",
		description: "Walk downstairs; only when the next step down is at the feet and the stairs are straight ahead.",
	},
	{
		name: "SIT",
		description: "Sit down in place onto a surface of this height (the target must be right behind you).",
		param: { name: "height", kind: "number", min: 0.15, max: 0.85, unit: "m", default: 0.5 },
	},
	{ name: "STOP", description: "Stop/Stand: the task is fully complete.", terminal: true },
];

/** GUMI's keys (the human baseline): WASD walk / step back / turn, Q E side steps, Z sit, X stop. */
export const KEYS: Record<string, { unit: string; param?: string | number }> = {
	KeyW: { unit: "WALK", param: "normal" },
	KeyR: { unit: "WALK", param: "slow" },
	KeyF: { unit: "WALK", param: "fast" },
	KeyS: { unit: "STEP_BACK", param: 0.25 },
	KeyA: { unit: "TURN_LEFT", param: 30 },
	KeyD: { unit: "TURN_RIGHT", param: 30 },
	KeyQ: { unit: "SIDE_LEFT", param: 0.25 },
	KeyE: { unit: "SIDE_RIGHT", param: 0.25 },
	KeyZ: { unit: "SIT", param: 0.5 },
	KeyX: { unit: "STOP" },
};

/**
 * The HumanCLAW action name of a unit with its (checked) parameter. The number is written in full,
 * so `chooserAction` reads back exactly the value that was checked.
 */
export function actionName(unit: string, param: string | number | undefined): { id: number; name: string } {
	const p = param === undefined ? "" : String(param);
	switch (unit) {
		case "WALK":
			return { id: 0, name: `Walk<forward><${p || "slow"}>` };
		case "TURN_LEFT":
			return { id: 2, name: `Turn<left><${p}>` };
		case "TURN_RIGHT":
			return { id: 2, name: `Turn<right><${p}>` };
		case "SIDE_LEFT":
			return { id: 6, name: `Side step<left><${p}>` };
		case "SIDE_RIGHT":
			return { id: 6, name: `Side step<right><${p}>` };
		case "STEP_BACK":
			return { id: 5, name: `Step back<${p}>` };
		case "CLIMB_UP":
			return { id: 3, name: "Climb upstairs<normal>" };
		case "WALK_DOWN":
			return { id: 7, name: "Walk downstairs<normal>" };
		case "SIT":
			return { id: 4, name: `Sit down<${p}>` };
		case "STOP":
			return { id: 1, name: "Stop/Stand" };
	}
	throw new Error(`not a HumanCLAW unit: ${unit}`);
}

/** The SkillCall a unit runs: its action name through HumanCLAW's own `_chooser_action`. */
export const skillCall = (unit: string, param: string | number | undefined): SkillCall => {
	const { id, name } = actionName(unit, param);
	return chooserAction({ action_id: id, action_name: name });
};

/** The unit and parameter that run a SkillCall (the inverse of `skillCall` on its value). */
export function unitOf(a: SkillCall): { unit: string; param?: string | number } {
	switch (a.skill) {
		case "walk_forward": {
			const d = (a.cond as number[])[1];
			return { unit: "WALK", param: d >= 0.55 ? "fast" : d >= 0.35 ? "normal" : "slow" };
		}
		case "turn": {
			const deg = a.cond as number;
			return deg >= 0 ? { unit: "TURN_LEFT", param: deg } : { unit: "TURN_RIGHT", param: -deg };
		}
		case "side_walk": {
			const x = a.cond as number;
			return x >= 0 ? { unit: "SIDE_LEFT", param: x } : { unit: "SIDE_RIGHT", param: -x };
		}
		case "step_back":
			return { unit: "STEP_BACK", param: -(a.cond as number[])[1] };
		case "step_climb_up":
			return { unit: "CLIMB_UP" };
		case "step_climb_down":
			return { unit: "WALK_DOWN" };
		case "sit":
			return { unit: "SIT", param: a.cond as number };
		default:
			return { unit: "STOP" };
	}
}
