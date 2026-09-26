/**
 * Anchor a robot's memory recipe into a Flash plan (./recipe.ts).
 *
 *   node src/flash/generate.ts --recipe <memory>/task_only/<cell>_recipe.jsonl --anchors anchors.json \
 *     --targets move_to=xyz,scripted_grasp=xyz,navigate_to=xy --destination <memory>/flash [--name <cell>]
 *
 * `anchors.json` is `[{phrase, xyz}]`: what the targets were relative to in the recorded episode,
 * each a phrase Molmo can point at and its recorded world position (read off the recording, e.g.
 * its back-projections), or, in simulation, a saved `ground_truth_poses` result of the recorded
 * scene (`{poses: {name: {pos}}}`, each name's underscores read as spaces). Every target within 0.2 m (x/y) of
 * an anchor is stored as an x/y offset from the nearest one; the rest replay as recorded. Writes
 * `<name>_plan.json` (default name: the recipe's cell).
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { type Anchor, type AnchoredEntry, loadProgram, type Target, targetOf } from "./recipe.ts";

/** Beyond this a target was not written relative to the anchor. */
export const MAX_ATTACH = 0.2;
const r4 = (v: number) => Number(v.toFixed(4));

/** `plan` with every target near an anchor stored relative to the nearest one. */
export function anchorPlan(plan: AnchoredEntry[], anchors: Anchor[], targets: Record<string, Target>) {
	return plan.map((entry): AnchoredEntry => {
		const where = targets[entry.action];
		const t = where ? targetOf(entry.arguments, where) : undefined;
		if (!t || !anchors.length) return { action: entry.action, arguments: entry.arguments };
		const d = (a: Anchor) => Math.hypot(t[0] - a.xyz[0], t[1] - a.xyz[1]);
		const nearest = anchors.reduce((best, a) => (d(a) < d(best) ? a : best));
		if (d(nearest) > MAX_ATTACH) return { action: entry.action, arguments: entry.arguments };
		return {
			action: entry.action,
			arguments: entry.arguments,
			anchor: nearest.phrase,
			offset: [r4(t[0] - nearest.xyz[0]), r4(t[1] - nearest.xyz[1])],
		};
	});
}

/** `move_to=xyz,navigate_to=xy` as a target table. */
export function parseTargets(spec: string): Record<string, Target> {
	return Object.fromEntries(
		spec
			.split(",")
			.filter(Boolean)
			.map((pair) => {
				const [tool, where] = pair.split("=");
				if (!tool || (where !== "xyz" && where !== "xy")) throw new Error(`bad target ${pair}: use <tool>=xyz|xy`);
				return [tool, where];
			}),
	);
}

export function generate(o: { recipe: string; anchors: string; targets: string; destination: string; name?: string }) {
	const name = o.name ?? basename(o.recipe).replace(/_recipe\.jsonl$/, "");
	const doc = JSON.parse(readFileSync(o.anchors, "utf8")) as Anchor[] | { poses?: Record<string, { pos: number[] }> };
	const anchors: Anchor[] = Array.isArray(doc)
		? doc
		: Object.entries(doc.poses ?? {}).map(([k, p]) => ({ phrase: k.replace(/_/g, " "), xyz: p.pos }));
	if (!Array.isArray(anchors) || anchors.some((a) => !a.phrase || !Array.isArray(a.xyz) || a.xyz.length < 2))
		throw new Error(`${o.anchors} must be [{phrase, xyz}]`);
	const program = loadProgram(o.recipe, name);
	const plan = anchorPlan(program.plan, anchors, parseTargets(o.targets));
	mkdirSync(o.destination, { recursive: true });
	const path = join(o.destination, `${name}_plan.json`);
	writeFileSync(path, `${JSON.stringify({ name, source: o.recipe, anchors, plan }, null, 2)}\n`);
	return { path, calls: plan.length, anchored: plan.filter((e) => e.anchor).length, anchors: anchors.length };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
	const { values } = parseArgs({
		options: {
			recipe: { type: "string" },
			anchors: { type: "string" },
			targets: { type: "string" },
			destination: { type: "string" },
			name: { type: "string" },
		},
	});
	if (!values.recipe || !values.anchors || !values.targets || !values.destination) {
		console.error(
			"usage: generate.ts --recipe <cell>_recipe.jsonl --anchors anchors.json --targets <tool>=xyz|xy,... --destination <dir> [--name <name>]",
		);
		process.exit(2);
	}
	console.log(JSON.stringify(generate(values as Parameters<typeof generate>[0]), null, 2));
}
