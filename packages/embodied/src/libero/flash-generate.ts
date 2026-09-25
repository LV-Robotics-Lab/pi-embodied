/**
 * Generate one LIBERO Flash plan from one successful trace.
 *
 *   node src/libero/flash-generate.ts --audit <mem>/task_only/goal_swap_t3_s7.json \
 *     --recipe <mem>/task_only/goal_swap_t3_s7_recipe.jsonl --destination memory/libero/flash
 *
 * Inputs are the episode audit (JSON) and its primitive recipe (JSONL); `segment_*.json` readings
 * are optional. Writes `<family>_<suite>_t<task>_{plan,anchors}.json` and `_trace.md`. Waypoints
 * are stored as XY offsets from the nearest anchor; anchors come from saved segment readings, else
 * from the task's goal relations and the recipe's pick/release (or articulation) transactions.
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

type Json = Record<string, unknown>;
type XY = [number, number];
export type Goal = { predicate: string; subject: string; object: string | null };
export type Anchor = {
	phrase: string;
	locator: "segment" | "molmo";
	camera: string;
	score: number | null;
	readings: number[][];
	median_xy: XY;
	z_top: number;
	z_span: number;
};
export type PlanEntry = {
	action: string;
	arguments: Json;
	anchor?: string;
	offset?: XY;
	anchor_distance?: number;
};

const CELL_TAG = /^(10|goal|object|spatial)_(task|swap)_t([0-9])_s(\d+)$/;
const MOVE_ACTIONS = new Set(["move_to", "move_pose"]);
const ARTICULATION_VERBS = /^\s*(?:turn|switch|open|close)\b/i;
const MAX_ATTACH = 0.2;
const SAME_PLACE = 0.03;

const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const round4 = (v: number) => Number(v.toFixed(4));
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

// ---------------------------------------------------------------- relations (relations.py)

const collapse = (s: string) => s.replace(/\s+/g, " ");
const clean = (language: string) => collapse(language.trim().replace(/[. ]+$/, ""));
const full = (pattern: string, text: string) => new RegExp(`^(?:${pattern})$`, "i").exec(text);

function entity(raw: string): string {
	const value = collapse(raw.replace(/^[ .,_-]+|[ .,_-]+$/g, "")).trim();
	return value.replace(/^(?:the|a|an)\s+/i, "").trim();
}

function graph(language: string, goals: Goal[]) {
	const entities: string[] = [];
	for (const g of goals) for (const e of [g.subject, g.object]) if (e && !entities.includes(e)) entities.push(e);
	return { language: language.trim(), entities, goals };
}
export type TaskGraph = ReturnType<typeof graph>;
const rel = (predicate: string, subject: string, object: string | null = null): Goal => ({
	predicate,
	subject,
	object,
});

/** One LIBERO-Goal/Object/Spatial instruction as ordered relations; unsupported text is rejected. */
export function extractGoalRelations(language: string): TaskGraph {
	const text = clean(language);
	if (!text) throw new Error("task language is empty");
	const compound = full(
		"open\\s+(?<fixture>.+?)\\s+and\\s+(?:" +
			"(?:pick(?:\\s+up)?|lift|grab)\\s+(?<picked>.+?)\\s+(?:and\\s+)?" +
			"(?:put|place|set)\\s+(?:it\\s+)?inside(?:\\s+it)?|" +
			"(?:put|place|set)\\s+(?<placed>.+?)\\s+inside(?:\\s+it)?)",
		text,
	);
	if (compound?.groups) {
		const fixture = entity(compound.groups.fixture);
		const item = entity(compound.groups.picked ?? compound.groups.placed ?? "");
		return graph(language, [rel("open", fixture), rel("in", item, fixture)]);
	}
	const unary: [string, string][] = [
		["open", "open\\s+(?<subject>.+)"],
		["turn_on", "turn\\s+on\\s+(?<subject>.+)"],
		["turn_on", "switch\\s+on\\s+(?<subject>.+)"],
		["turn_off", "turn\\s+off\\s+(?<subject>.+)"],
		["turn_off", "switch\\s+off\\s+(?<subject>.+)"],
	];
	for (const [predicate, pattern] of unary) {
		const m = full(pattern, text);
		if (m?.groups) return graph(language, [rel(predicate, entity(m.groups.subject))]);
	}
	const relations: [string, string][] = [
		["in_front_of", "(?:to\\s+)?(?:the\\s+)?front\\s+of"],
		["in", "(?:in|inside|into)"],
		["on", "(?:on\\s+(?:the\\s+)?top\\s+of|onto|on)"],
	];
	const sentences = [
		"(?:put|place|set)\\s+(?<subject>.+?)\\s+REL\\s+(?<object>.+)",
		"(?:pick(?:\\s+up)?|lift|grab)\\s+(?<subject>.+?)\\s+and\\s+(?:put|place|set)\\s+it\\s+REL\\s+(?<object>.+)",
		"push\\s+(?<subject>.+?)\\s+REL\\s+(?<object>.+)",
	];
	for (const [predicate, relation] of relations) {
		for (const sentence of sentences) {
			const m = full(sentence.replace("REL", relation), text);
			if (m?.groups) {
				return graph(language, [rel(predicate, entity(m.groups.subject), entity(m.groups.object))]);
			}
		}
	}
	throw new Error(`unsupported LIBERO-Goal instruction: ${JSON.stringify(language)}`);
}

/** One canonical LIBERO-10 instruction as ordered transactions. */
export function extractLongRelations(language: string): TaskGraph {
	const text = clean(language);
	let m = full("put both (.+?)s on the stove", text);
	if (m) {
		const item = entity(m[1]);
		return graph(language, [rel("on", `left ${item}`, "stove"), rel("on", `right ${item}`, "stove")]);
	}
	m = full("put both (.+?) and (.+?) in the basket", text);
	if (m) return graph(language, [rel("in", entity(m[1]), "basket"), rel("in", entity(m[2]), "basket")]);
	m = full("turn on the stove and put (.+?) on it", text);
	if (m) return graph(language, [rel("turn_on", "stove"), rel("on", entity(m[1]), "stove")]);
	m = full("put (.+?) in the bottom drawer of the cabinet and close it", text);
	if (m) {
		const drawer = "bottom drawer of the cabinet";
		return graph(language, [rel("in", entity(m[1]), drawer), rel("close", drawer)]);
	}
	m = full("put (.+?) in the microwave and close it", text);
	if (m) return graph(language, [rel("in", entity(m[1]), "microwave"), rel("close", "microwave")]);
	m = full("put (.+?) on the left plate and put (.+?) on the right plate", text);
	if (m) return graph(language, [rel("on", entity(m[1]), "left plate"), rel("on", entity(m[2]), "right plate")]);
	m = full("pick up (.+?) and place it in the back compartment of the caddy", text);
	if (m) return graph(language, [rel("in", entity(m[1]), "back compartment of the caddy")]);
	m = full("put (.+?) on the plate and put (.+?) to the right of the plate", text);
	if (m) return graph(language, [rel("on", entity(m[1]), "plate"), rel("right_of", entity(m[2]), "plate")]);
	m = full("put (.+?) on the stove", text);
	if (m) return graph(language, [rel("on", entity(m[1]), "stove")]);
	throw new Error(`unsupported LIBERO-10 instruction: ${JSON.stringify(language)}`);
}

export function extractRelations(family: string, language: string): TaskGraph {
	if (family === "10") return extractLongRelations(language);
	if (["goal", "object", "spatial"].includes(family)) return extractGoalRelations(language);
	throw new Error(`unsupported LIBERO family: ${family}`);
}

// ---------------------------------------------------------------- anchors (generate.py)

function readJson(path: string, what: string): unknown {
	try {
		return JSON.parse(readFileSync(path, "utf8"));
	} catch (err) {
		throw new Error(`cannot read ${what} ${path}: ${err}`);
	}
}

function readRecipe(path: string): Json[] {
	let lines: string[];
	try {
		lines = readFileSync(path, "utf8").split("\n");
	} catch (err) {
		throw new Error(`cannot read recipe JSONL ${path}: ${err}`);
	}
	const plan: Json[] = [];
	lines.forEach((line, i) => {
		if (!line.trim()) return;
		let entry: unknown;
		try {
			entry = JSON.parse(line);
		} catch (err) {
			throw new Error(`invalid recipe JSONL line ${i + 1}: ${err}`);
		}
		if (!entry || typeof entry !== "object" || Array.isArray(entry) || typeof (entry as Json).action !== "string")
			throw new Error(`recipe line ${i + 1} must be an object with an action`);
		plan.push(entry as Json);
	});
	if (!plan.length) throw new Error("recipe JSONL contains no actions");
	return plan;
}

function segmentAnchors(dir: string | undefined): Anchor[] {
	if (!dir || !existsSync(dir)) return [];
	const anchors: Anchor[] = [];
	const files = readdirSync(dir)
		.filter((f) => f.startsWith("segment_") && f.endsWith(".json"))
		.sort();
	for (const file of files) {
		let data: Json;
		try {
			data = JSON.parse(readFileSync(join(dir, file), "utf8")) as Json;
		} catch {
			continue;
		}
		const xyz = data.world_xyz;
		const phrase = String(data.prompt ?? "").trim();
		if (data.found !== true || !phrase || !Array.isArray(xyz) || xyz.length < 3 || !xyz.slice(0, 3).every(isNum))
			continue;
		const point = (xyz as number[]).slice(0, 3);
		if (anchors.some((a) => a.phrase === phrase)) continue;
		if (anchors.some((a) => dist(point, a.median_xy) < SAME_PLACE)) continue;
		anchors.push({
			phrase,
			locator: data.mode === "text" ? "segment" : "molmo",
			camera: String(data.camera ?? "agentview"),
			score: isNum(data.score) ? data.score : null,
			readings: [point],
			median_xy: [point[0], point[1]],
			z_top: point[2],
			z_span: 0,
		});
	}
	return anchors;
}

function moveXY(entry: Json): XY | undefined {
	const xyz = entry.xyz;
	return Array.isArray(xyz) && xyz.length === 3 && xyz.every(isNum) ? [xyz[0], xyz[1]] : undefined;
}

const isArticulation = (e: Json) =>
	e.action === "pi0_doubled" || (e.action === "pi0_pick" && ARTICULATION_VERBS.test(String(e.prompt ?? "")));

function semanticPhrase(entityName: string, predicate: string, subject: boolean): string {
	if (subject) return `the ${entityName}`;
	if (predicate === "in") return `the inside of the ${entityName}`;
	if (predicate === "on" && entityName.includes("drawer")) return `the top surface of the ${entityName}`;
	if (predicate === "on" && entityName.includes("stove")) return "the stove burner";
	if (predicate === "in_front_of" || predicate === "right_of")
		return `the ${predicate === "in_front_of" ? "front" : "right side"} of the ${entityName}`;
	return `the ${entityName}`;
}

/** Semantic reference points recovered from the goal graph and the recipe's ordered transactions. */
function fallbackAnchors(g: TaskGraph, plan: Json[]): Anchor[] {
	const moves: [number, XY][] = [];
	plan.forEach((e, i) => {
		const xy = moveXY(e);
		if (xy) moves.push([i, xy]);
	});
	if (!moves.length) return [];
	const indices = (pred: (e: Json) => boolean) => plan.flatMap((e, i) => (pred(e) ? [i] : []));
	const policyPicks = indices((e) => e.action === "pi0_pick" && !isArticulation(e));
	const releases = indices((e) => e.action === "release");
	// Recipes that grasp with move + set_gripper: the first close after each release boundary.
	const manualPicks: number[] = [];
	let previousRelease = -1;
	for (const release of releases) {
		const closing = plan.findIndex(
			(e, i) => previousRelease < i && i < release && e.action === "set_gripper" && e.gripper === 1,
		);
		if (closing >= 0) manualPicks.push(closing);
		previousRelease = release;
	}
	const picks = [...new Set([...policyPicks, ...manualPicks])].sort((a, b) => a - b);
	const interactions = indices(isArticulation);
	const lastBefore = (boundary: number, after = -1) => moves.filter(([i]) => after < i && i < boundary).at(-1)?.[1];

	let grasp = 0;
	let interaction = 0;
	const anchors: Anchor[] = [];
	for (const goal of g.goals) {
		let pairs: [string, XY | undefined][];
		if (["open", "close", "turn_on", "turn_off"].includes(goal.predicate)) {
			const boundary = interaction < interactions.length ? interactions[interaction] : plan.length;
			interaction++;
			const suffix = ["drawer", "door"].some((x) => goal.subject.includes(x)) ? " handle" : " knob";
			pairs = [[`the ${goal.subject}${suffix}`, lastBefore(boundary) ?? moves[0][1]]];
		} else if (goal.predicate === "in_front_of") {
			pairs = [[semanticPhrase(goal.subject, goal.predicate, true), moves[0][1]]];
		} else {
			const pick = grasp < picks.length ? picks[grasp] : plan.length;
			const release = releases.find((r) => r > pick) ?? plan.length;
			pairs = [[semanticPhrase(goal.subject, goal.predicate, true), lastBefore(pick) ?? moves[0][1]]];
			if (goal.object)
				pairs.push([
					semanticPhrase(goal.object, goal.predicate, false),
					lastBefore(release, pick) ?? lastBefore(release),
				]);
			grasp++;
		}
		for (const [phrase, xy] of pairs) {
			if (!xy || anchors.some((a) => a.phrase === phrase)) continue;
			anchors.push({
				phrase,
				locator: "molmo",
				camera: "agentview",
				score: null,
				readings: [[xy[0], xy[1], 0]],
				median_xy: xy,
				z_top: 0,
				z_span: 0,
			});
		}
	}
	return anchors;
}

function mergeAnchors(measured: Anchor[], inferred: Anchor[]): Anchor[] {
	const anchors = [...measured];
	for (const c of inferred) {
		if (anchors.some((a) => a.phrase === c.phrase)) continue;
		if (anchors.some((a) => dist(c.median_xy, a.median_xy) < SAME_PLACE)) continue;
		anchors.push(c);
	}
	return anchors;
}

function attachMoves(plan: Json[], anchors: Anchor[]): PlanEntry[] {
	return plan.map(({ action, ...args }) => {
		const entry: PlanEntry = { action: String(action), arguments: args };
		const xy = moveXY(args);
		if (xy && anchors.length) {
			const anchor = anchors.reduce((best, a) => (dist(xy, a.median_xy) < dist(xy, best.median_xy) ? a : best));
			const d = dist(xy, anchor.median_xy);
			if (d <= MAX_ATTACH) {
				entry.anchor = anchor.phrase;
				entry.offset = [round4(xy[0] - anchor.median_xy[0]), round4(xy[1] - anchor.median_xy[1])];
				entry.anchor_distance = round4(d);
			}
		}
		return entry;
	});
}

export function generateFlashPlan(options: {
	audit: string;
	recipe: string;
	destination: string;
	segments?: string;
	/** Task text when the audit does not carry it (pi-embodied audits may omit task_language). */
	language?: string;
}) {
	const tag = basename(options.audit).replace(/\.json$/, "");
	const cell = CELL_TAG.exec(tag);
	if (!cell)
		throw new Error(
			`audit filename ${basename(options.audit)} must be <10|goal|object|spatial>_<task|swap>_t<0-9>_s<seed>.json`,
		);
	const [, family, suite, taskText, seedText] = cell;
	const [task, seed] = [Number(taskText), Number(seedText)];
	if (basename(options.recipe) !== `${tag}_recipe.jsonl`)
		throw new Error(`recipe filename ${basename(options.recipe)} does not match audit; expected ${tag}_recipe.jsonl`);

	const audit = readJson(options.audit, "audit JSON") as Json;
	if (!audit || typeof audit !== "object" || Array.isArray(audit))
		throw new Error("audit JSON must contain one object");
	// Explore audits write libero_terminated; evaluate audits write terminated. Either must be true.
	if (audit.libero_terminated !== true && audit.terminated !== true)
		throw new Error("Flash plans can only be generated from libero_terminated=true traces");
	for (const [field, expected] of [
		["suite", `libero_${family}_${suite}`],
		["task_id", task],
		["seed", seed],
	] as const) {
		if (field in audit && audit[field] !== expected)
			throw new Error(`audit ${field}=${JSON.stringify(audit[field])} does not match filename (${expected})`);
	}
	const finalState = (audit.final_state ?? {}) as Json;
	const language = String(
		audit.perturbed_task_language || audit.task_language || finalState.task_language || options.language || "",
	).trim();
	if (!language) throw new Error("audit JSON has no task language; pass --language");

	const g = extractRelations(family, language);
	const raw = readRecipe(options.recipe);
	const anchors = mergeAnchors(
		segmentAnchors(options.segments ?? join(dirname(options.audit), "segments")),
		fallbackAnchors(g, raw),
	);
	const plan = attachMoves(raw, anchors);
	const key = `${suite}_t${task}`;
	const taskName = `${family}/${key}`;
	const source = { cell: tag, audit: options.audit, recipe: options.recipe };
	const attached = plan.filter((e) => e.anchor !== undefined).length;
	const moves = plan.filter((e) => MOVE_ACTIONS.has(e.action)).length;
	const trace =
		`# ${taskName}\n\nTask: ${language}\n\nSource: \`${tag}\`\n\nAudit: \`${options.audit}\`\n\n` +
		`Recipe: \`${options.recipe}\`\n\nGoals: \`${JSON.stringify(g.goals)}\`\n\n` +
		`Anchors: ${anchors.length}\n\nActions: ${plan.length}\n\nAnchored waypoints: ${attached}/${moves}\n`;

	mkdirSync(options.destination, { recursive: true });
	const name = `${family}_${key}`;
	const write = (suffix: string, doc: unknown) =>
		writeFileSync(join(options.destination, `${name}_${suffix}`), `${JSON.stringify(doc, null, 2)}\n`);
	write("plan.json", { task: taskName, language, source, goal_graph: g, plan });
	write("anchors.json", { task: taskName, language, source, anchors, prompts_without_readings: [] });
	writeFileSync(join(options.destination, `${name}_trace.md`), trace);
	return { family, suite, task, key, source: tag, language, anchors: anchors.length, actions: plan.length };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
	const { values } = parseArgs({
		options: {
			audit: { type: "string" },
			recipe: { type: "string" },
			destination: { type: "string" },
			segments: { type: "string" },
			language: { type: "string" },
		},
	});
	if (!values.audit || !values.recipe || !values.destination) {
		console.error(
			"usage: flash-generate.ts --audit <cell>.json --recipe <cell>_recipe.jsonl --destination <dir> [--segments <dir>] [--language <text>]",
		);
		process.exit(2);
	}
	const row = generateFlashPlan({
		audit: values.audit,
		recipe: values.recipe,
		destination: values.destination,
		segments: values.segments,
		language: values.language,
	});
	console.log(JSON.stringify(row, null, 2));
}
