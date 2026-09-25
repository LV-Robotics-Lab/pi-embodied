/**
 * GUMI recordings -> Show-Harness rollouts that train/data_preparation/rollouts_to_alpaca.py reads.
 *
 *   node --experimental-strip-types packages/embodied/src/finetuned/prepare.ts \
 *     --out /root/autodl-tmp/data/finetuned/mydata/rollouts <gumi record dir>...
 *
 * Finds every single-arm run (a directory with actions.jsonl) under the given directories and writes
 * `<out>/<task>/rollout_NNN/` with `agentview/NNNN.png`, `wrist/NNNN.png`, `actions.jsonl` ({token,
 * gripper_closed, agentview, wrist} plus the recorded fields) and `metadata.json` (`task_text` for
 * train/scripts/prepare_dataset.sh). Each frame goes through the SAME camera transform the provider
 * applies at inference (./views.ts: the recording robot's ROBOT_VIEWS, or --agentview / --wrist), so
 * training and deployment images are identical pixel for pixel. Rows are kept in order as recorded;
 * tokens outside the single-arm vocabulary (ROTATE_*, STILL) are left for the converter to skip.
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { encodePng } from "../png.ts";
import { ROBOT_VIEWS } from "./index.ts";
import { decodePng, formatView, parseView, prepareView, type ViewSpec } from "./views.ts";

type Json = Record<string, any>;

/** Every directory under `root` (itself included) that holds an actions.jsonl. */
export function findRuns(root: string): string[] {
	if (existsSync(join(root, "actions.jsonl"))) return [root];
	return readdirSync(root)
		.map((d) => join(root, d))
		.filter((d) => statSync(d).isDirectory())
		.sort()
		.flatMap(findRuns);
}

const slug = (s: string) =>
	s
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "_")
		.replace(/^_|_$/g, "")
		.slice(0, 80) || "task";

/** Convert one run into `<taskDir>/rollout_NNN`; the number of rows written and warnings. */
export function convertRun(
	run: string,
	taskDir: string,
	index: number,
	views: { agentview: ViewSpec; wrist: ViewSpec },
	task: string,
	meta: Json,
) {
	const rows = readFileSync(join(run, "actions.jsonl"), "utf8")
		.split("\n")
		.filter((l) => l.trim())
		.map((l) => JSON.parse(l) as Json);
	const out = join(taskDir, `rollout_${String(index).padStart(3, "0")}`);
	for (const v of ["agentview", "wrist"]) mkdirSync(join(out, v), { recursive: true });
	const warnings: string[] = [];
	const lines: string[] = [];
	let repeats = 0;
	const missing: number[] = [];
	rows.forEach((row, i) => {
		if (typeof row.token !== "string") throw new Error(`${run}/actions.jsonl row ${i}: not a single-arm GUMI step`);
		// The converter needs both views of every sample; a step recorded without one is dropped.
		if (typeof row.agentview !== "string" || typeof row.wrist !== "string") {
			missing.push(i);
			return;
		}
		const files: Record<string, string> = {};
		for (const v of ["agentview", "wrist"] as const) {
			const img = prepareView(decodePng(readFileSync(join(run, row[v]))), views[v]);
			const file = `${v}/${String(lines.length).padStart(4, "0")}.png`;
			writeFileSync(join(out, file), encodePng(img.rgb, img.width, img.height));
			files[v] = file;
		}
		if (Number(row.n) > 1) repeats++;
		lines.push(JSON.stringify({ ...row, ...files, gripper_closed: row.gripper_closed === true }));
	});
	if (missing.length)
		warnings.push(`${run}: dropped steps ${missing.join(", ")} (recorded without both camera views)`);
	if (repeats)
		warnings.push(
			`${run}: ${repeats} steps executed their unit n>1 times from one frame; each stays one sample (one history entry)`,
		);
	writeFileSync(join(out, "actions.jsonl"), `${lines.join("\n")}\n`);
	writeFileSync(
		join(out, "metadata.json"),
		JSON.stringify(
			{
				...meta,
				task_text: task,
				source_run: run,
				views: { agentview: formatView(views.agentview), wrist: formatView(views.wrist) },
			},
			null,
			2,
		),
	);
	return { out, rows: lines.length, warnings };
}

function main() {
	const { values, positionals } = parseArgs({
		allowPositionals: true,
		options: {
			out: { type: "string" },
			task: { type: "string" },
			robot: { type: "string" },
			agentview: { type: "string" },
			wrist: { type: "string" },
		},
	});
	if (!values.out || !positionals.length) {
		console.error(
			"usage: prepare.ts --out <dir> [--task T] [--robot R] [--agentview spec] [--wrist spec] <gumi dir>...",
		);
		process.exit(2);
	}
	const runs = positionals.flatMap((p) => findRuns(resolve(p)));
	if (!runs.length) throw new Error(`no actions.jsonl under ${positionals.join(", ")}`);
	const counters = new Map<string, number>();
	let total = 0;
	for (const run of runs) {
		const meta: Json = existsSync(join(run, "metadata.json"))
			? JSON.parse(readFileSync(join(run, "metadata.json"), "utf8"))
			: {};
		const first = readFileSync(join(run, "actions.jsonl"), "utf8").split("\n", 1)[0];
		if (!first.trim()) {
			console.error(`skip ${run}: no steps (a run still recording, or discarded)`);
			continue;
		}
		if ("left" in JSON.parse(first)) {
			console.error(`skip ${run}: dual-arm run (convert with rollouts_to_alpaca.py --dual)`);
			continue;
		}
		const task = values.task ?? meta.task_text ?? meta.task;
		if (typeof task !== "string" || !task) throw new Error(`${run}: no task in metadata.json; pass --task`);
		const robot = values.robot ?? meta.robot ?? "";
		const d = ROBOT_VIEWS[robot];
		if (!d && !(values.agentview && values.wrist))
			throw new Error(`${run}: no camera transform for robot "${robot}"; pass --agentview and --wrist`);
		const views = {
			agentview: values.agentview ? parseView(values.agentview) : d.agentview,
			wrist: values.wrist ? parseView(values.wrist) : d.wrist,
		};
		const taskDir = join(resolve(values.out), slug(task));
		const n = counters.get(taskDir) ?? 0;
		counters.set(taskDir, n + 1);
		const r = convertRun(run, taskDir, n, views, task, meta);
		for (const w of r.warnings) console.error(`warning: ${w}`);
		console.log(`${basename(run)} -> ${r.out} (${r.rows} steps, "${task}")`);
		total += r.rows;
	}
	console.log(`${total} steps in ${counters.size} task dir(s) under ${values.out}`);
}

if (process.argv[1] && resolve(process.argv[1]) === new URL(import.meta.url).pathname) main();
