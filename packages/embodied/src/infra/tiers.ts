/**
 * `--tier` (CaP-X's eight tiers; handoff 2.9): one flag instead of the handful it stands for. A tier
 * is four orthogonal axes, each of which is an existing flag the user may also set alone:
 *
 *   turns      single = --max-turns 1 (one program, then the environment's verdict); multi = any other budget
 *   feedback   none / text = --keep-images 0, no --anchor-image (the model sees no camera frame; recordings keep them),
 *              image = frames kept, vdm = --vdm --keep-images 0 (the VLM's words, not the frames),
 *              image+vdm = --vdm (frames and words; not one of the eight)
 *   api        --code-api high | low | low-noexamples | raw, in code mode (--code=true)
 *   privileged --privileged (the simulator's ground truth)
 *
 * The expansion is not a mechanism of its own: ./params.ts's tracker reads a flag the tier expands
 * to as the tier's value unless that flag was given itself, so the robot, its modules and `params`
 * (what the result records) all see the same values. A flag given with another value than the
 * tier's is a start-time error naming both (`configError`), never a silent override; an M tier leaves
 * --privileged free and is recorded as `M2+privileged`, a combination, not one of the eight. A tier
 * the robot cannot serve (no code mode, no high-tier primitives, no VDM, no ground truth) is refused
 * at start too. The result records `tier` and `axes` (the expanded turns / feedback / api / privileged).
 * `src/scripts/params-match.mjs` and `tier-flags.mjs` import the same table, so the eval scripts
 * expand a tier exactly as the robot does.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type FlagSpec, trackFlags } from "./params.ts";

export type Turns = "single" | "multi";
export type Feedback = "none" | "text" | "image" | "vdm" | "image+vdm";
export type Api = "high" | "low" | "low-noexamples" | "raw";
/** The four axes a tier expands to; `privileged` undefined = left to the flag (an M tier). */
export type Axes = { turns: Turns; feedback: Feedback; api: Api; privileged?: boolean };

/** CaP-X's eight tiers (paper section 3, table 1): S = single turn, no feedback; M = multi-turn with feedback. */
export const TIERS: Record<string, Axes> = {
	S1: { turns: "single", feedback: "none", api: "high", privileged: true },
	S2: { turns: "single", feedback: "none", api: "high", privileged: false },
	S3: { turns: "single", feedback: "none", api: "low", privileged: false },
	S4: { turns: "single", feedback: "none", api: "low-noexamples", privileged: false },
	M1: { turns: "multi", feedback: "text", api: "high" },
	M2: { turns: "multi", feedback: "image", api: "high" },
	M3: { turns: "multi", feedback: "vdm", api: "high" },
	M4: { turns: "multi", feedback: "vdm", api: "low" },
};
export const TIER_NAMES = Object.keys(TIERS);

/**
 * The flags a choice sets: a string or boolean is the value the flag reads as (and must have when
 * given); `null` keeps a string flag at its registered default (giving it at all is a conflict).
 */
export type Expansion = Record<string, string | boolean | null>;

/** The flags CaP-X's axes stand for (code mode throughout: the tiers are code-mode settings). */
export function axesFlags(axes: Axes): Expansion {
	const x: Expansion = { code: "true", units: null, "code-api": axes.api };
	if (axes.turns === "single") x["max-turns"] = "1";
	const vdm = axes.feedback === "vdm" || axes.feedback === "image+vdm";
	x.vdm = vdm;
	if (!vdm) x["vdm-video"] = false;
	if (axes.feedback === "none" || axes.feedback === "text" || axes.feedback === "vdm") {
		x["keep-images"] = "0";
		x["anchor-image"] = false;
	}
	if (axes.privileged !== undefined) x.privileged = axes.privileged;
	return x;
}

/** What `--tier` / `--preset` chose, or the reason the choice is invalid. */
export type Choice =
	| { kind: "tier"; name: string; axes: Axes; flags: Expansion }
	| { kind: "preset"; name: string; axes: Axes; flags: Expansion }
	| { error: string };

/** The choice `--tier` and `--preset` make (undefined: neither given). */
export function choose(tier: unknown, preset: unknown): Choice | undefined {
	const t = String(tier ?? "").trim();
	const p = String(preset ?? "").trim();
	if (t && p) return { error: `--tier ${t} and --preset ${p} are mutually exclusive: a preset is a tier of its own` };
	if (t) {
		const axes = TIERS[t];
		if (!axes) return { error: `--tier ${t}: not a CaP-X tier; the tiers are ${TIER_NAMES.join(", ")}` };
		return { kind: "tier", name: t, axes, flags: axesFlags(axes) };
	}
	if (p) return choosePreset(p);
	return undefined;
}

/** `--preset <name>` (the second half of handoff 2.9, see below). */
function choosePreset(p: string): Choice {
	return { error: `--preset ${p}: no such preset` };
}

/** `--code=pure` and a bare `--code` are `true`; a boolean flag's value is its own name. */
export const norm = (v: unknown): string =>
	v === true || v === "pure" ? "true" : v === false ? "false" : String(v ?? "");

/** Whether a flag read as `raw` was given (differs from what it was registered with). */
export const given = (raw: unknown, spec: FlagSpec | undefined) =>
	spec !== undefined && raw !== undefined && norm(raw) !== norm(spec.default);

/**
 * Why the flags given (`explicit`: name to value, the ones on the command line) contradict the
 * choice: one message per flag, exact names and values; empty when they agree. Shared by the robot
 * (which knows what was given from each flag's default) and tier-flags.mjs (which reads pi's argv).
 */
export function conflicts(c: Choice, explicit: Map<string, unknown>): string[] {
	if ("error" in c) return [c.error];
	const flag = `--${c.kind} ${c.name}`;
	const out: string[] = [];
	for (const [name, want] of Object.entries(c.flags)) {
		if (!explicit.has(name)) continue;
		const got = explicit.get(name);
		const shown = typeof got === "boolean" ? `--${name}` : `--${name}=${norm(got)}`;
		// S2 with ground truth is S1: say so, instead of only refusing.
		const hint =
			name === "privileged" && c.kind === "tier" && c.name !== "S1"
				? " (S1 is S2 with the simulator's ground truth)"
				: "";
		if (want === null) out.push(`${flag} leaves --${name} at its default, but ${shown} was given`);
		else if (want === false) {
			if (norm(got) !== "false") out.push(`${flag} runs without --${name}, but ${shown} was given${hint}`);
		} else if (want === true) {
			if (norm(got) !== "true") out.push(`${flag} sets --${name}, but ${shown} was given`);
		} else if (norm(got) !== norm(want)) out.push(`${flag} sets --${name}=${want}, but ${shown} was given`);
	}
	// The axes a flag's default already satisfies have one value that contradicts them.
	if (c.axes.turns === "multi" && explicit.has("max-turns") && Number(explicit.get("max-turns")) === 1)
		out.push(`${flag} is multi-turn, but --max-turns=1 was given`);
	if (
		(c.axes.feedback === "image" || c.axes.feedback === "image+vdm") &&
		explicit.has("keep-images") &&
		Number(explicit.get("keep-images")) === 0
	)
		out.push(`${flag} feeds the camera frames back, but --keep-images=0 was given`);
	return out;
}

/** What a robot can serve, read by `tiers` to refuse a tier it cannot run. */
export type Caps = {
	robot: string;
	code: boolean;
	units: boolean;
	vdm: boolean;
	groundTruth: boolean;
	/** The base tiers the robot's manifest declares code primitives in (high, low, raw). */
	codeTiers: () => string[];
};

/** Why `caps` cannot run the choice, else undefined. */
export function unsupported(c: Choice, caps: Caps): string | undefined {
	if ("error" in c) return c.error;
	const flag = `--${c.kind} ${c.name}`;
	if (c.kind !== "tier") return undefined;
	if (!caps.code) return `${flag} runs code mode (run_code), which ${caps.robot} does not mount`;
	const have = caps.codeTiers();
	const base = c.axes.api === "low-noexamples" ? "low" : c.axes.api;
	if (!have.includes(base)) {
		const can = TIER_NAMES.filter((n) => have.includes(TIERS[n].api === "low-noexamples" ? "low" : TIERS[n].api));
		return `${flag} needs ${base}-tier code primitives; ${caps.robot}'s manifest has ${have.join(", ") || "none"} (its tiers: ${can.join(", ") || "none"})`;
	}
	if (c.axes.privileged && !caps.groundTruth)
		return `${flag} needs the simulator's ground truth (--privileged), which ${caps.robot} has none of: S1 is simulation only`;
	if ((c.axes.feedback === "vdm" || c.axes.feedback === "image+vdm") && !caps.vdm)
		return `${flag} describes each change with VDM (--vdm), which ${caps.robot} does not mount`;
	return undefined;
}

/**
 * Register `--tier` (and `--preset`) on `pi` and hook the expansion into the flag tracker: call it
 * right after `trackFlags` in the robot base. `configError()` is the start-time check (robot.ts
 * fails closed on it); `result()` the fields the result records.
 */
export function tiers(pi: ExtensionAPI, caps: Caps) {
	const t = trackFlags(pi);
	pi.registerFlag("tier", {
		type: "string",
		default: "",
		description: `CaP-X tier (${TIER_NAMES.join(", ")}): sets --code, --code-api, --max-turns, --keep-images, --vdm and --privileged; a flag given with another value is an error`,
	});
	pi.registerFlag("preset", {
		type: "string",
		default: "",
		description:
			"A source repository's native setting (see the README's preset table); mutually exclusive with --tier",
	});
	const choice = () => choose(t.raw("tier"), t.raw("preset"));
	t.expand = (name, raw, spec) => {
		const c = choice();
		if (!c || "error" in c || !(name in c.flags)) return raw;
		const want = c.flags[name];
		if (want === null || given(raw, spec)) return raw;
		return want;
	};
	/** The flags the robot registered that were given on the command line, with their raw values. */
	const explicit = () => {
		const m = new Map<string, unknown>();
		for (const [name, spec] of t.names) {
			const raw = t.raw(name);
			if (given(raw, spec)) m.set(name, raw);
		}
		return m;
	};
	return {
		/** Why the robot must not start with these flags, else undefined. */
		configError: (): string | undefined => {
			const c = choice();
			if (!c) return undefined;
			if ("error" in c) return c.error;
			return unsupported(c, caps) ?? (conflicts(c, explicit()).join("; ") || undefined);
		},
		/** `tier` or `preset` (with `+privileged` when ground truth was stacked on a choice that leaves it free) and the expanded `axes`. */
		result: (): Record<string, unknown> => {
			const c = choice();
			if (!c || "error" in c) return {};
			const privileged = pi.getFlag("privileged") === true;
			const stacked = c.axes.privileged === undefined && privileged;
			return {
				[c.kind]: stacked ? `${c.name}+privileged` : c.name,
				axes: { ...c.axes, privileged },
			};
		},
	};
}
