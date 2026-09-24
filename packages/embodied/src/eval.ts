/**
 * Batch evaluator CLI: run a robot task over seeds with a pi agent and record
 * environment-judged success. The episode and batch logic lives in `runner.ts`.
 *
 *   node --import tsx packages/embodied/src/eval.ts --robot <name> --endpoint http://127.0.0.1:18100 \
 *     --tasks 0,1 --seeds 0-2 --model selfhost/muse-glimmer-30b --thinking medium
 *
 * Outputs <out>/<robot>_t<task>_s<seed>[_retryN]/{result.json,trace.json,<timestamp>_<id>.jsonl,frames/}
 * and <out>/summary.json (rewritten after every episode).
 */

import { join, resolve } from "node:path";
import type { ThinkingLevel } from "@earendil-works/pi-agent-core";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { makeRobot } from "./robots/index.ts";
import { type RunOptions, runBatch } from "./runner.ts";

interface Options extends RunOptions {
	robot: string;
	endpoint: string;
	tasks: number[];
	seeds: number[];
	model: string;
}

function parseRange(spec: string): number[] {
	return spec.split(",").flatMap((part) => {
		const [a, b] = part.split("-").map(Number);
		if (Number.isNaN(a) || (b !== undefined && Number.isNaN(b))) throw new Error(`bad range: ${spec}`);
		return b === undefined ? [a] : Array.from({ length: b - a + 1 }, (_, i) => a + i);
	});
}

function parseArgs(argv: string[]): Options {
	const get = (name: string, fallback?: string): string => {
		const i = argv.indexOf(`--${name}`);
		if (i >= 0 && i + 1 < argv.length) return argv[i + 1];
		if (fallback === undefined) throw new Error(`missing --${name}`);
		return fallback;
	};
	const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
	return {
		robot: get("robot"),
		endpoint: get("endpoint", "http://127.0.0.1:18100"),
		tasks: parseRange(get("tasks", "0")),
		seeds: parseRange(get("seeds", "0")),
		model: get("model", "selfhost/muse-glimmer-30b"),
		thinking: get("thinking", "medium") as ThinkingLevel,
		maxTurns: Number(get("max-turns", "40")),
		timeoutS: Number(get("timeout-s", "900")),
		retryInfra: Number(get("retry-infra", "1")),
		imageBudgetMb: Number(get("image-budget-mb", "4")),
		out: resolve(get("out", `runs/${stamp}`)),
	};
}

async function main(): Promise<void> {
	const opts = parseArgs(process.argv.slice(2));
	const robot = makeRobot(opts.robot, opts.endpoint);
	const modelRuntime = await ModelRuntime.create();
	const [provider, ...rest] = opts.model.split("/");
	const model = modelRuntime.getModel(provider, rest.join("/"));
	if (!model) throw new Error(`model not found: ${opts.model}`);
	const summary = await runBatch(robot, model, modelRuntime, opts);
	console.log(
		`\nsuccess ${summary.successes}/${summary.episodes} (${(summary.successRate * 100).toFixed(1)}%), infra errors excluded ${summary.infraErrors}, env errors excluded ${summary.envErrors}, claimed-but-failed ${summary.claimedButFailed}, mean turns ${summary.meanTurns.toFixed(1)}, cost $${summary.totalCostUsd}\n${join(opts.out, "summary.json")}`,
	);
}

await main();
