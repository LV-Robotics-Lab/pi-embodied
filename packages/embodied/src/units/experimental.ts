/**
 * The units module's experimental answer protocols and prompt transforms (Show-Harness `plugins/mcq`
 * and `plugins/action_ablation`; `plugins/coords` is a [coords] section of ./SYSTEM.md). Pure
 * functions and one per-episode record; ./index.ts mounts them on `act`, its results and the prompt.
 *
 * - mcq: the model answers `act` with an option letter (A, B, ...), mapped back to its unit before
 *   anything runs, so results, history and recordings keep the unit names.
 * - action_ablation (`--units-ablation`): the paper's action-representation ablation, over the six
 *   direction units only (GRASP, RELEASE, DONE and the turns keep their names):
 *     bare           MV_* names, every explanation of what a direction does removed;
 *     letters        opaque ACT_A..ACT_F symbols with the explanations (symbolized);
 *     letters_blind  opaque symbols, told only that the six ARE the six directions; the model
 *                    blind-picks, reviews each symbol's effect from a before/after frame pair and
 *                    keeps its own symbol -> effect table (`note`), shown back every step.
 *   Everything the model reads passes one funnel (`filter`): the system prompt, `act`'s description
 *   and every `act` result's units block, as Show-Harness's `filter_final` runs on the assembled prompt.
 *
 * Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: plugins/mcq/{plugin.py,mcq.txt} and plugins/action_ablation/
 * {plugin.py,action_ablation.txt} ported to pi's `act` tool (enum answers instead of decoded tokens,
 * the NOTE line as `act`'s `note`), their fragments rewritten for pi's prompt and tool names.
 */

// ---------------------------------------------------------------------------
// mcq (plugins/mcq)

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/** The letter <-> unit map over `units` in order (plugins/mcq McqPlugin). */
export function mcqOptions(units: readonly string[]) {
	if (units.length > LETTERS.length) throw new Error("mcq: too many units to label with single letters");
	const letters = units.map((_, i) => LETTERS[i]);
	return {
		letters,
		/** "A) MV_FWD  B) MV_BACK  ..." (options_block). */
		block: units.map((u, i) => `${letters[i]}) ${u}`).join("  "),
		/** A letter's unit, or undefined. */
		unit: (letter: string) => units[letters.indexOf(String(letter).trim().toUpperCase())],
	};
}

// ---------------------------------------------------------------------------
// action_ablation (plugins/action_ablation)

export const ABLATION_MODES = ["bare", "letters", "letters_blind"] as const;
export type AblationMode = (typeof ABLATION_MODES)[number];

/** The fixed direction -> symbol map (only the six directions are ever blinded). */
export const SYMBOLS: Record<string, string> = {
	MV_FWD: "ACT_A",
	MV_BACK: "ACT_B",
	MV_LEFT: "ACT_C",
	MV_RIGHT: "ACT_D",
	MV_UP: "ACT_E",
	MV_DOWN: "ACT_F",
};
const TOKENS_OF = Object.fromEntries(Object.entries(SYMBOLS).map(([t, s]) => [s, t]));
export const SYMBOL_LIST = Object.values(SYMBOLS);
/** A symbol's unit, or the name unchanged (GRASP, RELEASE, ...). */
export const unsymbol = (s: string) => TOKENS_OF[s] ?? s;
/** A unit's symbol in a letters mode, or the name unchanged. */
export const symbol = (u: string) => SYMBOLS[u] ?? u;

const NOTE_MAX = 90;
/** "(Pairs: MV_LEFT/MV_RIGHT, ...)"-style direction-pair hints: an explanation in the no-explanation modes. */
const PAIRS_RE = /\s*\((?:Pairs?|MV_LEFT \/ MV_RIGHT, MV_FWD \/ MV_BACK)[^)]*\)/g;
const DIRECTION_RE = /\b(?:MV_FWD|MV_BACK|MV_LEFT|MV_RIGHT|MV_UP|MV_DOWN)\b/;
/** Lines that may keep a direction name after the blind scrub: the move history (symbolized next), and task/stage fields with the name deleted. */
const HISTORY_RE = /^\s*(?:Recent moves|units:)/i;
const FIELD_RE = /^\s*(?:TASK|STAGE|TARGET|AFFORD|Stage goal|DONE WHEN):/i;
/** The views text of the modes without explanations: which images come, not which way a unit moves in them. */
export const NEUTRAL_VIEWS =
	"Each result shows the robot's camera images: the third-person view first, then the wrist view(s) if the robot has them.";
/** The movement-units line of the modes without explanations. */
export const NEUTRAL_MOVES =
	"- MV_FWD, MV_BACK, MV_LEFT, MV_RIGHT, MV_UP, MV_DOWN: movement units, one fixed step each.";

/**
 * Tier 1 of the blind scrub: sentences of other plugins that pair a direction with advice keep their
 * neutral facts (plugins/action_ablation _BLIND_REWRITES, rewritten for ./SYSTEM.md and the units block).
 */
const BLIND_REWRITES: [RegExp, string][] = [
	// proprioception: keep the height, drop the descend directive.
	[/\s*If height > [^\n]*? cm, MV_DOWN first\.?/g, ""],
	// mem_text: keep the no-regrasp rule, drop the escape-direction list.
	[/first reposition with MV_UP, MV_BACK, MV_DOWN or MV_FWD/g, "first move away"],
	// action_chunk: keep the plan protocol, neutralize the descent example.
	[
		/\(e\.g\. never plan more MV_DOWN than the height above the table allows\)/g,
		"(mind the height above the table and the step size)",
	],
];

/** The DIRECTION section each mode puts in the prompt (action_ablation.txt [bare_direction], [blind_direction]). */
export function ablationDirection(mode: AblationMode): string | undefined {
	if (mode === "bare")
		return "DIRECTION:\nChoose exactly one movement unit from the action list (MV_FWD, MV_BACK, MV_LEFT, MV_RIGHT, MV_UP, MV_DOWN). No description of what any movement unit does is provided.";
	if (mode === "letters_blind")
		return [
			"DIRECTION:",
			"ACT_A, ACT_B, ACT_C, ACT_D, ACT_E, ACT_F are the gripper's six movement directions -- up, down, left, right, forward, backward -- exactly one symbol per direction, but WHICH symbol is WHICH direction is NOT told.",
			"Every `act` result shows YOUR ACTION RECORDS: the table you write yourself, one line per symbol.",
			"How to choose:",
			"- If a recorded symbol matches the direction you need now, choose it directly.",
			"- Otherwise pick one of the (unknown) symbols at random to try -- a blind pick is expected, not an error.",
			"- After a single movement symbol the result asks you to REVIEW it: it attaches the third-person frame from BEFORE that action as the LAST image. Compare it with the current (first) image and pass your record with the next `act` as `note`: NOTE[ACT_X]: <one short clause on what it did>.",
			"All other actions (e.g. GRASP, RELEASE, DONE) keep their usual meaning.",
		].join("\n");
	return undefined;
}

/** The per-episode state of one ablation setting, and its funnel over everything the model reads. */
export class Ablation {
	readonly mode: AblationMode;
	/** letters_blind: symbol -> the model's latest note, and every note event in order. */
	notes = new Map<string, string>();
	history: { symbol: string; note: string }[] = [];
	/** letters_blind: the symbol under review this step (armed after a single blinded move), else undefined. */
	review: string | undefined;

	constructor(mode: AblationMode) {
		this.mode = mode;
	}

	/** The mode answers in symbols (the mcq-style slot). */
	get symbolic() {
		return this.mode !== "bare";
	}

	reset() {
		this.notes = new Map();
		this.history = [];
		this.review = undefined;
	}

	/** The model's text (system prompt, tool description, a units block) in this setting (filter_final). */
	filter(text: string): string {
		let out = text;
		if (this.mode !== "letters") out = out.replace(PAIRS_RE, "");
		if (this.mode === "letters_blind") out = scrub(out);
		if (this.symbolic) {
			out = out.replace(/\(MV_ units only\)/g, "(movement symbols only)");
			for (const [t, s] of Object.entries(SYMBOLS)) out = out.replace(new RegExp(`\\b${t}\\b`, "g"), s);
		}
		return out.replace(/\n{3,}/g, "\n\n");
	}

	/**
	 * letters_blind: take the reviewed symbol's NOTE from `act`'s `note` (only the symbol under review:
	 * a note without its before/after pair is a guess). Returns what was recorded, if anything.
	 */
	harvest(note: string | undefined): string | undefined {
		if (this.mode !== "letters_blind" || !this.review || !note) return undefined;
		const review = this.review;
		for (const m of note.matchAll(/NOTE\[\s*(ACT_[A-F])\s*\]\s*:\s*([^\n"]+?)(?=\s*NOTE\[|["\n]|$)/gi)) {
			if (m[1].toUpperCase() !== review) continue;
			const clause = m[2]
				.split(/\s+/)
				.join(" ")
				.split(/(?<=[.;!?])\s/)[0]
				.slice(0, NOTE_MAX);
			this.notes.set(review, clause);
			this.history.push({ symbol: review, note: clause });
			return `${review}: ${clause}`;
		}
		return undefined;
	}

	/** letters_blind: the record table every result shows. */
	table(): string {
		return [
			'YOUR ACTION RECORDS so far (written by you in earlier steps; "(unknown)" = never tried or not yet described):',
			...SYMBOL_LIST.map((s) => `${s}: ${this.notes.get(s) ?? "(unknown)"}`),
		].join("\n");
	}

	/** letters_blind: the review request after a single blinded move (action_ablation.txt [blind_review]). */
	reviewText(s: string) {
		return `REVIEW YOUR LAST ACTION: you just executed ${s}. The LAST attached image is the third-person frame captured BEFORE ${s}; the first image is the current third-person view (AFTER it). Compare the two frames, judge what ${s} did to the gripper, and pass your record with the next \`act\` as \`note\`: NOTE[${s}]: <one short clause on what it did>`;
	}

	/** The run record (metadata): the setting and, blind, the learned table; never the true mapping for blind. */
	record() {
		return {
			mode: this.mode,
			...(this.mode === "letters_blind"
				? {
						notes: Object.fromEntries(SYMBOL_LIST.map((s) => [s, this.notes.get(s) ?? null])),
						note_history: [...this.history],
					}
				: { symbols: { ...SYMBOLS } }),
		};
	}
}

/** letters_blind: tier 1 rewrites, then every other line pairing a direction with a meaning goes. */
function scrub(text: string): string {
	let out = text;
	for (const [re, to] of BLIND_REWRITES) out = out.replace(re, to);
	const kept: string[] = [];
	for (let line of out.split("\n")) {
		if (DIRECTION_RE.test(line) && !HISTORY_RE.test(line)) {
			if (!FIELD_RE.test(line)) continue;
			line = line.replace(new RegExp(DIRECTION_RE.source, "g"), "").replace(/[ \t]{2,}/g, " ");
		}
		kept.push(line);
	}
	return kept.join("\n");
}
