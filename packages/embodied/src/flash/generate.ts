/**
 * Anchor a robot's memory recipe, or a solved session, into a Flash plan (./recipe.ts).
 *
 *   node src/flash/generate.ts --recipe <memory>/task_only/<cell>_recipe.jsonl --anchors anchors.json \
 *     --targets move_to=xyz,scripted_grasp=xyz,navigate_to=xy --destination <memory>/flash [--name <cell>]
 *   node src/flash/generate.ts --session <session>.jsonl --anchors anchors.json --name <cell> \
 *     --targets move_delta=delta_xyz --position state.tcp_pos --destination <memory>/flash
 *
 * `anchors.json` is `[{phrase, xyz}]`: what the targets were relative to in the recorded episode,
 * each a phrase Molmo can point at and its recorded world position (read off the recording, e.g.
 * its back-projections or the grasp point), or, in simulation, a saved `ground_truth_poses` result
 * of the recorded scene (`{poses: {name: {pos}}}`, each name's underscores read as spaces). Every
 * target within 0.2 m (x/y) of an anchor is stored as an x/y offset from the nearest one; the rest
 * replay as recorded. Writes `<name>_plan.json` (default name: the recipe's cell).
 *
 * A recipe holds a delta move's [dx, dy, dz] only, which no anchor can carry. `--session` reads the
 * whole recording instead: every motion result (`--motion`, default `act`, plus the relative
 * target's tool) carries the end-effector position after it (`--position`, a path in the result's
 * JSON) and the gripper command (`--gripper`), so each move becomes a waypoint: the recorded delta,
 * the absolute end position (`to`) and a `gripper` argument where the command changed. The other
 * `--targets` tools are kept as called; everything else (observations, planning tools) is dropped.
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { parseSessionEntries, type SessionEntry } from "@earendil-works/pi-coding-agent";
import { toolCalls } from "../memory/index.ts";
import { type Anchor, type AnchoredEntry, loadProgram, type RelativeTarget, type Target, targetOf } from "./recipe.ts";

type Json = Record<string, unknown>;
/** Beyond this a target was not written relative to the anchor. */
export const MAX_ATTACH = 0.2;
const r4 = (v: number) => Number(v.toFixed(4));

/** `plan` with every target near an anchor stored relative to the nearest one. */
export function anchorPlan(plan: AnchoredEntry[], anchors: Anchor[], targets: Record<string, Target>) {
	return plan.map((entry): AnchoredEntry => {
		const where = targets[entry.action];
		const t = where ? targetOf(entry, where) : undefined;
		const kept = { action: entry.action, arguments: entry.arguments, ...(entry.to ? { to: entry.to } : {}) };
		if (!t || !anchors.length) return kept;
		const d = (a: Anchor) => Math.hypot(t[0] - a.xyz[0], t[1] - a.xyz[1]);
		const nearest = anchors.reduce((best, a) => (d(a) < d(best) ? a : best));
		if (d(nearest) > MAX_ATTACH) return kept;
		return { ...kept, anchor: nearest.phrase, offset: [r4(t[0] - nearest.xyz[0]), r4(t[1] - nearest.xyz[1])] };
	});
}

/** A reader of the dotted `path` (`state.tcp_pos`) in a result's JSON. */
export function pathOf(path: string): (json: Json) => unknown {
	const keys = path.split(".").filter(Boolean);
	return (json) => keys.reduce<unknown>((v, k) => (v && typeof v === "object" ? (v as Json)[k] : undefined), json);
}
const isVec3 = (v: unknown): v is number[] =>
	Array.isArray(v) && v.length >= 3 && v.slice(0, 3).every((x) => typeof x === "number" && Number.isFinite(x));

/**
 * `move_to=xyz,navigate_to=xy,move_delta=delta_xyz` as a target table: `xyz` / `xy` name the world
 * target argument, any other name the delta argument of a relative target, whose end-effector
 * position is read by `position`.
 */
export function parseTargets(spec: string, position?: (json: Json) => number[] | undefined): Record<string, Target> {
	return Object.fromEntries(
		spec
			.split(",")
			.filter(Boolean)
			.map((pair): [string, Target] => {
				const [tool, where] = pair.split("=");
				if (!tool || !where) throw new Error(`bad target ${pair}: use <tool>=xyz|xy|<delta argument>`);
				if (where === "xyz" || where === "xy") return [tool, where];
				if (!position)
					throw new Error(`target ${tool}=${where} is a delta: --position names the end-effector position`);
				return [tool, { delta: where, position }];
			}),
	);
}

/**
 * The plan a solved session recorded: after its last successful `reset`, each result of a motion
 * tool becomes a waypoint of the relative target's tool (the delta from the previous end-effector
 * position, `to` = the new one, `gripper` when the command changed); the absolute-target tools are
 * kept as called. Throws when no result reports the task solved (`terminated` or `success`).
 */
export function sessionPlan(
	entries: SessionEntry[],
	o: {
		targets: Record<string, Target>;
		/** Tools besides the relative target's whose results are waypoints (the units' `act`). */
		motion: readonly string[];
		gripper: (json: Json) => unknown;
	},
): AnchoredEntry[] {
	const relative = Object.entries(o.targets).filter((e): e is [string, RelativeTarget] => typeof e[1] !== "string");
	if (relative.length !== 1) throw new Error("a session plan needs exactly one relative target (its waypoints' tool)");
	const [tool, where] = relative[0];
	const motion = new Set([tool, ...o.motion]);
	const calls = toolCalls(entries);
	const ok = (c: (typeof calls)[number]) => !c.isError && !c.details?.error && !c.details?.result?.error;
	let start = 0;
	calls.forEach((c, i) => {
		if (c.name === "reset" && ok(c)) start = i + 1;
	});
	const episode = calls.slice(start).filter(ok);
	const json = (c: (typeof calls)[number]) => (c.details ?? {}) as Json;
	if (!episode.some((c) => json(c).terminated === true || json(c).success === true))
		throw new Error("the session did not solve the task (no result reports terminated or success)");
	const plan: AnchoredEntry[] = [];
	let pos: number[] | undefined;
	let grip: unknown;
	for (const c of episode) {
		const p = where.position(json(c));
		const g = o.gripper(json(c));
		if (motion.has(c.name) && isVec3(p)) {
			if (pos) {
				const delta = p.map((v, k) => r4(v - (pos as number[])[k]));
				const changed = g !== undefined && g !== grip;
				if (Math.hypot(...delta) > 1e-4 || changed)
					plan.push({
						action: tool,
						arguments: { [where.delta]: delta, ...(changed ? { gripper: g } : {}) },
						to: p.map(r4),
					});
			}
			pos = p;
		} else if (c.name in o.targets) plan.push({ action: c.name, arguments: c.args });
		else if (isVec3(p)) pos = p;
		if (g !== undefined && isVec3(p)) grip = g;
	}
	if (!plan.length) throw new Error("the session holds no motion results with an end-effector position");
	return plan;
}

export function generate(o: {
	recipe?: string;
	session?: string;
	anchors: string;
	targets: string;
	position?: string;
	gripper?: string;
	motion?: string;
	destination: string;
	name?: string;
}) {
	const source = o.session ?? o.recipe;
	if (!source || (o.session && o.recipe)) throw new Error("one of --recipe or --session is required");
	const name = o.name ?? (o.recipe ? basename(o.recipe).replace(/_recipe\.jsonl$/, "") : undefined);
	if (!name) throw new Error("--name is required with --session");
	const doc = JSON.parse(readFileSync(o.anchors, "utf8")) as Anchor[] | { poses?: Record<string, { pos: number[] }> };
	const anchors: Anchor[] = Array.isArray(doc)
		? doc
		: Object.entries(doc.poses ?? {}).map(([k, p]) => ({ phrase: k.replace(/_/g, " "), xyz: p.pos }));
	if (!Array.isArray(anchors) || anchors.some((a) => !a.phrase || !Array.isArray(a.xyz) || a.xyz.length < 2))
		throw new Error(`${o.anchors} must be [{phrase, xyz}]`);
	const read = o.position ? pathOf(o.position) : undefined;
	const position = read
		? (json: Json) => {
				const p = read(json);
				return isVec3(p) ? p : undefined;
			}
		: undefined;
	const targets = parseTargets(o.targets, position);
	const plan = o.session
		? sessionPlan(parseSessionEntries(readFileSync(o.session, "utf8")) as SessionEntry[], {
				targets,
				motion: (o.motion ?? "act").split(",").filter(Boolean),
				gripper: pathOf(o.gripper ?? "state.gripper_command"),
			})
		: loadProgram(o.recipe as string, name).plan;
	const anchored = anchorPlan(plan, anchors, targets);
	mkdirSync(o.destination, { recursive: true });
	const path = join(o.destination, `${name}_plan.json`);
	writeFileSync(path, `${JSON.stringify({ name, source, anchors, plan: anchored }, null, 2)}\n`);
	return { path, calls: anchored.length, anchored: anchored.filter((e) => e.anchor).length, anchors: anchors.length };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
	const { values } = parseArgs({
		options: {
			recipe: { type: "string" },
			session: { type: "string" },
			anchors: { type: "string" },
			targets: { type: "string" },
			position: { type: "string" },
			gripper: { type: "string" },
			motion: { type: "string" },
			destination: { type: "string" },
			name: { type: "string" },
		},
	});
	if ((!values.recipe && !values.session) || !values.anchors || !values.targets || !values.destination) {
		console.error(
			"usage: generate.ts (--recipe <cell>_recipe.jsonl | --session <session>.jsonl --name <name> --position <json path> [--gripper <json path>] [--motion act,...]) --anchors anchors.json --targets <tool>=xyz|xy|<delta arg>,... --destination <dir> [--name <name>]",
		);
		process.exit(2);
	}
	console.log(JSON.stringify(generate(values as Parameters<typeof generate>[0]), null, 2));
}
