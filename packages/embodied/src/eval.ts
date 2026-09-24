/**
 * Batch evaluator: run a robot task over seeds with a pi agent and record
 * environment-judged success.
 *
 *   node --import tsx packages/embodied/src/eval.ts --robot <name> --endpoint http://127.0.0.1:18100 \
 *     --tasks 0,1 --seeds 0-2 --model selfhost/muse-glimmer-30b --thinking medium
 *
 * Outputs <out>/<robot>_t<task>_s<seed>/{result.json,trace.json,session.jsonl,frames/}
 * and <out>/summary.json.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import type { ThinkingLevel } from "@earendil-works/pi-agent-core";
import {
	createAgentSession,
	DefaultResourceLoader,
	ModelRuntime,
	SessionManager,
	SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { episodeExtension } from "./extension.ts";
import { makeRobot } from "./robots/index.ts";
import { Episode, type Robot } from "./toolkit.ts";

interface Options {
	robot: string;
	endpoint: string;
	tasks: number[];
	seeds: number[];
	model: string;
	thinking: ThinkingLevel;
	maxTurns: number;
	timeoutS: number;
	retryInfra: number;
	imageBudgetMb: number;
	out: string;
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

interface EpisodeResult {
	robot: string;
	task: number;
	seed: number;
	language: string;
	model: string;
	thinking: string;
	success: boolean;
	claimed?: string;
	summary?: string;
	/** `infra_error`: the run ended on a provider/transport failure, not on the agent's behavior. */
	stopReason: "finish" | "max_turns" | "timeout" | "no_tool_call" | "error" | "infra_error";
	attempt: number;
	error?: string;
	turns: number;
	toolCalls: number;
	imagesPruned: number;
	elapsedS: number;
	tokens: { input: number; output: number; cacheRead: number };
	costUsd: number;
}

async function runEpisode(
	robot: Robot,
	opts: Options,
	modelRuntime: ModelRuntime,
	taskId: number,
	seed: number,
	attempt: number,
): Promise<EpisodeResult> {
	const [provider, ...rest] = opts.model.split("/");
	const model = modelRuntime.getModel(provider, rest.join("/"));
	if (!model) throw new Error(`model not found: ${opts.model}`);

	const task = await robot.begin(taskId, seed);
	const dir = join(opts.out, `${robot.name}_t${taskId}_s${seed}${attempt ? `_retry${attempt}` : ""}`);
	const episode = new Episode(dir);
	const started = Date.now();
	const resourceLoader = new DefaultResourceLoader({
		cwd: dir,
		agentDir: dir,
		noExtensions: true,
		noSkills: true,
		noPromptTemplates: true,
		noThemes: true,
		noContextFiles: true,
		systemPrompt: robot.systemPrompt(task),
		extensionFactories: [
			{
				name: "embodied",
				factory: episodeExtension(robot, episode, { images: { maxBytes: opts.imageBudgetMb * 1024 * 1024 } }),
			},
		],
	});
	await resourceLoader.reload();
	const { session } = await createAgentSession({
		cwd: dir,
		model,
		thinkingLevel: opts.thinking,
		modelRuntime,
		resourceLoader,
		noTools: "builtin",
		sessionManager: SessionManager.create(dir, dir),
		settingsManager: SettingsManager.inMemory({
			compaction: { enabled: false },
			retry: { enabled: true, maxRetries: 2 },
		}),
	});

	let turns = 0;
	let stopReason: EpisodeResult["stopReason"] = "no_tool_call";
	let error: string | undefined;
	session.subscribe((event) => {
		if (event.type === "turn_end") {
			turns++;
			if (turns >= opts.maxTurns && !episode.finish) {
				stopReason = "max_turns";
				void session.abort();
			}
		}
		if (event.type === "tool_execution_start") {
			process.stdout.write(`  [t${taskId}s${seed}] ${event.toolName} ${JSON.stringify(event.args)}\n`);
		}
	});
	const timer = setTimeout(() => {
		stopReason = "timeout";
		void session.abort();
	}, opts.timeoutS * 1000);

	try {
		await session.prompt(robot.userPrompt(task));
	} catch (err) {
		stopReason = "error";
		error = err instanceof Error ? err.message : String(err);
	} finally {
		clearTimeout(timer);
	}
	if (episode.finish) stopReason = "finish";

	const tokens = { input: 0, output: 0, cacheRead: 0 };
	let costUsd = 0;
	for (const msg of session.messages) {
		if (msg.role === "assistant" && msg.usage) {
			tokens.input += msg.usage.input;
			tokens.output += msg.usage.output;
			tokens.cacheRead += msg.usage.cacheRead;
			costUsd += msg.usage.cost?.total ?? 0;
		}
		if (msg.role === "assistant" && msg.stopReason === "error" && !error) error = msg.errorMessage;
	}
	session.dispose();
	if (!episode.finish && !robot.solved() && error && isInfraError(error)) stopReason = "infra_error";

	const result: EpisodeResult = {
		robot: robot.name,
		task: taskId,
		seed,
		language: task.language,
		model: opts.model,
		thinking: opts.thinking,
		success: robot.solved(),
		claimed: episode.finish?.status,
		summary: episode.finish?.summary,
		stopReason,
		attempt,
		error,
		turns,
		toolCalls: episode.trace.length,
		imagesPruned: episode.imagesPruned,
		elapsedS: Math.round((Date.now() - started) / 100) / 10,
		tokens,
		costUsd: Number(costUsd.toFixed(6)),
	};
	writeFileSync(join(dir, "result.json"), `${JSON.stringify(result, null, 2)}\n`);
	return result;
}

/** Provider / gateway / network failures that say nothing about the agent. */
function isInfraError(message: string): boolean {
	return /rejected illegal short-input|rate.?limit|overloaded|\b(429|500|502|503|504)\b|ECONNRESET|ECONNREFUSED|ETIMEDOUT|fetch failed|socket hang up|upstream|no credits/i.test(
		message,
	);
}

async function main(): Promise<void> {
	const opts = parseArgs(process.argv.slice(2));
	mkdirSync(opts.out, { recursive: true });
	const robot = makeRobot(opts.robot, opts.endpoint);
	const modelRuntime = await ModelRuntime.create();
	const results: EpisodeResult[] = [];
	for (const taskId of opts.tasks) {
		for (const seed of opts.seeds) {
			console.log(`\n=== ${opts.robot} task ${taskId} seed ${seed} (${opts.model}, thinking ${opts.thinking})`);
			let r = await runEpisode(robot, opts, modelRuntime, taskId, seed, 0);
			for (let attempt = 1; r.stopReason === "infra_error" && attempt <= opts.retryInfra; attempt++) {
				console.log(`=== infra error (${r.error}); retrying (${attempt}/${opts.retryInfra})`);
				r = await runEpisode(robot, opts, modelRuntime, taskId, seed, attempt);
			}
			results.push(r);
			console.log(
				`=== ${r.success ? "SUCCESS" : "FAIL"} claimed=${r.claimed ?? "-"} stop=${r.stopReason} turns=${r.turns} tools=${r.toolCalls} ${r.elapsedS}s${r.error ? ` error=${r.error}` : ""}`,
			);
		}
	}
	const scored = results.filter((r) => r.stopReason !== "infra_error");
	const successes = scored.filter((r) => r.success).length;
	const summary = {
		robot: opts.robot,
		model: opts.model,
		thinking: opts.thinking,
		episodes: scored.length,
		infraErrors: results.length - scored.length,
		successes,
		successRate: scored.length ? successes / scored.length : 0,
		claimedButFailed: scored.filter((r) => r.claimed === "success" && !r.success).length,
		meanTurns: scored.length ? scored.reduce((a, r) => a + r.turns, 0) / scored.length : 0,
		meanElapsedS: scored.length ? scored.reduce((a, r) => a + r.elapsedS, 0) / scored.length : 0,
		totalCostUsd: Number(results.reduce((a, r) => a + r.costUsd, 0).toFixed(6)),
		results,
	};
	writeFileSync(join(opts.out, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
	console.log(
		`\nsuccess ${successes}/${scored.length} (${(summary.successRate * 100).toFixed(1)}%), infra errors excluded ${summary.infraErrors}, claimed-but-failed ${summary.claimedButFailed}, mean turns ${summary.meanTurns.toFixed(1)}, cost $${summary.totalCostUsd}\n${join(opts.out, "summary.json")}`,
	);
}

await main();
