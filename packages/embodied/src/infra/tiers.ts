/**
 * `--tier` (CaP-X's eight tiers) and `--preset` (a source repository's native setting; handoff 2.9):
 * one flag instead of the handful it stands for. A tier is four orthogonal axes, each of which is an
 * existing flag the user may also set alone:
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
 *
 * `--preset <name>` (`PRESETS`) is the same for the other ported repositories, which do not write
 * code but call a tool or pick an action each step, so their settings are multi-turn with image
 * feedback and differ in the primitives' level (the `api` axis by analogy): Show-Harness's action
 * units, HumanCLAW's paper mode, RPent's tools with memory and a VLA skill, OpenETA's tools with
 * visual differencing, XPolicyLab's policy (`capx-<tier>` is `--tier <tier>` under the preset
 * spelling). A preset needs the modules it stands for mounted, is mutually exclusive with `--tier`,
 * and is recorded as `preset` with the same `axes`. `src/scripts/params-match.mjs` and
 * `tier-flags.mjs` import the same tables, so the eval scripts expand a choice exactly as the robot does.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type FlagSpec, trackFlags } from "./params.ts";

export type Turns = "single" | "multi";
export type Feedback = "none" | "text" | "image" | "vdm" | "image+vdm";
/** The primitives' level: CaP-X's code tiers, or `policy` when an XPolicyLab policy acts (--xpolicy). */
export type Api = "high" | "low" | "low-noexamples" | "raw" | "policy";
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

/** What a preset needs the robot to have (`Caps`). */
export type Need = "units" | "humanclaw" | "memory" | "vla" | "vdm" | "xpolicy";
type Preset = {
	axes: Axes;
	flags: Expansion;
	needs: Need[];
	/** A flag the preset needs given (its value is the run's): --xpolicy ws://... */
	requires?: { flag: string; hint: string };
};

/** The ported repositories' native settings (handoff 2.9.4), by analogy with CaP-X's axes. */
export const PRESETS: Record<string, Preset> = {
	showharness: {
		axes: { turns: "multi", feedback: "image", api: "raw" },
		flags: { units: "true", code: null, vdm: false, "vdm-video": false },
		needs: ["units"],
	},
	humanclaw: {
		axes: { turns: "multi", feedback: "image", api: "raw" },
		flags: { units: "both", "humanclaw-mode": "paper", vdm: false, "vdm-video": false },
		needs: ["humanclaw"],
	},
	rpent: {
		axes: { turns: "multi", feedback: "image", api: "low" },
		flags: { code: null, units: null, vdm: false, "vdm-video": false, stateless: false },
		needs: ["memory", "vla"],
	},
	openeta: {
		axes: { turns: "multi", feedback: "image+vdm", api: "low" },
		flags: { vdm: true, "anchor-image": true, code: null, units: null },
		needs: ["vdm"],
	},
	xpolicylab: {
		axes: { turns: "multi", feedback: "image", api: "policy" },
		flags: { code: null, units: null, vdm: false, "vdm-video": false },
		needs: ["xpolicy"],
		requires: { flag: "xpolicy", hint: "ws://host:port (the policy server)" },
	},
};
export const PRESET_NAMES = [...Object.keys(PRESETS), ...TIER_NAMES.map((n) => `capx-${n}`)];

/** What `--tier` / `--preset` chose (`flag` names it in messages), or the reason the choice is invalid. */
export type Choice =
	| {
			kind: "tier" | "preset";
			name: string;
			flag: string;
			axes: Axes;
			flags: Expansion;
			needs: Need[];
			requires?: { flag: string; hint: string };
	  }
	| { error: string };

/** The choice `--tier` and `--preset` make (undefined: neither given). */
export function choose(tier: unknown, preset: unknown): Choice | undefined {
	const t = String(tier ?? "").trim();
	const p = String(preset ?? "").trim();
	if (t && p) return { error: `--tier ${t} and --preset ${p} are mutually exclusive: a preset is a tier of its own` };
	if (t) {
		const axes = TIERS[t];
		if (!axes) return { error: `--tier ${t}: not a CaP-X tier; the tiers are ${TIER_NAMES.join(", ")}` };
		return { kind: "tier", name: t, flag: `--tier ${t}`, axes, flags: axesFlags(axes), needs: [] };
	}
	if (!p) return undefined;
	// capx-S3 is --tier S3 under the preset spelling: the same tier, recorded as one.
	if (p.startsWith("capx-") && TIERS[p.slice(5)]) {
		const name = p.slice(5);
		return { kind: "tier", name, flag: `--preset ${p}`, axes: TIERS[name], flags: axesFlags(TIERS[name]), needs: [] };
	}
	const pre = PRESETS[p];
	if (!pre) return { error: `--preset ${p}: no such preset; the presets are ${PRESET_NAMES.join(", ")}` };
	return {
		kind: "preset",
		name: p,
		flag: `--preset ${p}`,
		axes: pre.axes,
		flags: pre.flags,
		needs: pre.needs,
		requires: pre.requires,
	};
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
	const { flag } = c;
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

/** What a robot can serve, read by `tiers` to refuse a choice it cannot run. */
export type Caps = {
	robot: string;
	code: boolean;
	units: boolean;
	vdm: boolean;
	groundTruth: boolean;
	memory: boolean;
	xpolicy: boolean;
	/** The manifest declares the pi0_pick VLA skill (RPent's). */
	vla: boolean;
	/** The base tiers the robot's manifest declares code primitives in (high, low, raw). */
	codeTiers: () => string[];
};

const NEEDS: Record<Need, (caps: Caps) => string | undefined> = {
	units: (caps) =>
		caps.units
			? undefined
			: `drives the arm with Show-Harness action units (--units), which ${caps.robot} does not mount`,
	humanclaw: (caps) =>
		caps.robot === "humanclaw"
			? undefined
			: `is the HumanCLAW robot's paper setting (--units=both --humanclaw-mode paper); ${caps.robot} is not it`,
	memory: (caps) => (caps.memory ? undefined : `plans with RPent's memory, which ${caps.robot} does not mount`),
	vla: (caps) =>
		caps.vla ? undefined : `calls a VLA skill (pi0_pick), which ${caps.robot}'s manifest does not declare`,
	vdm: (caps) => (caps.vdm ? undefined : `describes each change with VDM (--vdm), which ${caps.robot} does not mount`),
	xpolicy: (caps) =>
		caps.xpolicy ? undefined : `runs an XPolicyLab policy (--xpolicy), which ${caps.robot} does not mount`,
};

/** Why `caps` cannot run the choice, else undefined. */
export function unsupported(c: Choice, caps: Caps): string | undefined {
	if ("error" in c) return c.error;
	const { flag } = c;
	if (c.kind === "preset") {
		for (const need of c.needs) {
			const why = NEEDS[need](caps);
			if (why) return `${flag} ${why}`;
		}
		return undefined;
	}
	if (!caps.code) return `${flag} runs code mode (run_code), which ${caps.robot} does not mount`;
	const have = caps.codeTiers();
	const base = c.axes.api === "low-noexamples" ? "low" : c.axes.api;
	if (!have.includes(base)) {
		const can = TIER_NAMES.filter((n) => have.includes(TIERS[n].api === "low-noexamples" ? "low" : TIERS[n].api));
		return `${flag} needs ${base}-tier code primitives; ${caps.robot}'s manifest has ${have.join(", ") || "none"} (its tiers: ${can.join(", ") || "none"})`;
	}
	if (c.axes.privileged && !caps.groundTruth)
		return `${flag} needs the simulator's ground truth (--privileged), which ${caps.robot} has none of: S1 is simulation only`;
	if (c.axes.feedback === "vdm" || c.axes.feedback === "image+vdm") {
		const why = NEEDS.vdm(caps);
		if (why) return `${flag} ${why}`;
	}
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
		description: `A source repository's native setting (${PRESET_NAMES.join(", ")}): sets the flags it stands for; mutually exclusive with --tier`,
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
			const cannot = unsupported(c, caps);
			if (cannot) return cannot;
			if (c.requires && !norm(t.raw(c.requires.flag)).trim())
				return `${c.flag} needs --${c.requires.flag} ${c.requires.hint}`;
			return conflicts(c, explicit()).join("; ") || undefined;
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
