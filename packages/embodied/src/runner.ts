/**
 * Episode and batch runner: drives one robot task per pi agent session and
 * records environment-judged success.
 *
 * Per-episode failures never throw: an environment RPC failure in
 * `robot.begin` becomes `env_error`, a session setup or prompt failure becomes
 * `error`, and provider/transport failures become `infra_error`. The batch
 * summary is rewritten after every episode, so a crash keeps what ran so far.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { ThinkingLevel } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import {
	type AgentSession,
	createAgentSession,
	DefaultResourceLoader,
	type ModelRuntime,
	SessionManager,
	SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { episodeExtension } from "./extension.ts";
import { Episode, type Robot, type TaskInfo } from "./toolkit.ts";

export interface RunOptions {
	thinking: ThinkingLevel;
	maxTurns: number;
	timeoutS: number;
	/** Retries for `infra_error` and `env_error` episodes. */
	retryInfra: number;
	imageBudgetMb: number;
	/** Absolute output directory. */
	out: string;
}

export interface EpisodeResult {
	robot: string;
	task: number;
	seed: number;
	language: string;
	/** `${provider}/${id}` */
	model: string;
	thinking: string;
	success: boolean;
	claimed?: string;
	summary?: string;
	/**
	 * `infra_error`: the run ended on a provider/transport failure, not on the agent's behavior.
	 * `env_error`: the environment failed to start the episode (`robot.begin`).
	 */
	stopReason: "finish" | "max_turns" | "timeout" | "no_tool_call" | "error" | "infra_error" | "env_error";
	attempt: number;
	error?: string;
	turns: number;
	toolCalls: number;
	imagesPruned: number;
	elapsedS: number;
	tokens: { input: number; output: number; cacheRead: number };
	costUsd: number;
}

export interface Summary {
	robot: string;
	model: string;
	thinking: string;
	/** Scored episodes: excludes `infra_error` and `env_error`. */
	episodes: number;
	infraErrors: number;
	envErrors: number;
	successes: number;
	successRate: number;
	claimedButFailed: number;
	meanTurns: number;
	meanElapsedS: number;
	totalCostUsd: number;
	results: EpisodeResult[];
}

/** Provider / gateway / network failures that say nothing about the agent. */
export function isInfraError(message: string): boolean {
	return /rejected illegal short-input|rate.?limit|overloaded|\b(429|500|502|503|504)\b|ECONNRESET|ECONNREFUSED|ETIMEDOUT|fetch failed|socket hang up|upstream|no credits/i.test(
		message,
	);
}

function errorMessage(err: unknown): string {
	return err instanceof Error ? err.message : String(err);
}

function writeJson(path: string, value: unknown): void {
	writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`);
}

async function createSession(
	robot: Robot,
	model: Model<Api>,
	modelRuntime: ModelRuntime,
	opts: RunOptions,
	task: TaskInfo,
	episode: Episode,
	dir: string,
): Promise<AgentSession> {
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
	return session;
}

/** Run one episode. Never throws for per-episode failures; writes `<dir>/result.json`. */
export async function runEpisode(
	robot: Robot,
	model: Model<Api>,
	modelRuntime: ModelRuntime,
	opts: RunOptions,
	taskId: number,
	seed: number,
	attempt: number,
	log: (line: string) => void = console.log,
): Promise<EpisodeResult> {
	const dir = join(opts.out, `${robot.name}_t${taskId}_s${seed}${attempt ? `_retry${attempt}` : ""}`);
	const episode = new Episode(dir);
	const started = Date.now();
	const result: EpisodeResult = {
		robot: robot.name,
		task: taskId,
		seed,
		language: "",
		model: `${model.provider}/${model.id}`,
		thinking: opts.thinking,
		success: false,
		stopReason: "no_tool_call",
		attempt,
		turns: 0,
		toolCalls: 0,
		imagesPruned: 0,
		elapsedS: 0,
		tokens: { input: 0, output: 0, cacheRead: 0 },
		costUsd: 0,
	};
	const finalize = (): EpisodeResult => {
		result.elapsedS = Math.round((Date.now() - started) / 100) / 10;
		writeJson(join(dir, "result.json"), result);
		return result;
	};

	let task: TaskInfo;
	try {
		task = await robot.begin(taskId, seed);
	} catch (err) {
		result.stopReason = "env_error";
		result.error = errorMessage(err);
		return finalize();
	}
	result.language = task.language;

	let session: AgentSession;
	try {
		session = await createSession(robot, model, modelRuntime, opts, task, episode, dir);
	} catch (err) {
		result.stopReason = "error";
		result.error = errorMessage(err);
		return finalize();
	}

	let stopReason: EpisodeResult["stopReason"] = "no_tool_call";
	let error: string | undefined;
	let turns = 0;
	const timer = setTimeout(() => {
		stopReason = "timeout";
		void session.abort();
	}, opts.timeoutS * 1000);
	try {
		session.subscribe((event) => {
			if (event.type === "turn_end") {
				turns++;
				if (turns >= opts.maxTurns && !episode.finish) {
					stopReason = "max_turns";
					void session.abort();
				}
			}
			if (event.type === "tool_execution_start") {
				log(`  [t${taskId}s${seed}] ${event.toolName} ${JSON.stringify(event.args)}`);
			}
		});
		try {
			await session.prompt(robot.userPrompt(task));
		} catch (err) {
			stopReason = "error";
			error = errorMessage(err);
		}

		let lastAssistantError = false;
		for (const msg of session.messages) {
			if (msg.role !== "assistant") continue;
			if (msg.usage) {
				result.tokens.input += msg.usage.input;
				result.tokens.output += msg.usage.output;
				result.tokens.cacheRead += msg.usage.cacheRead;
				result.costUsd += msg.usage.cost?.total ?? 0;
			}
			lastAssistantError = msg.stopReason === "error";
			if (lastAssistantError && !error) error = msg.errorMessage;
		}
		// A non-retryable provider error (e.g. HTTP 400) ends the session without `prompt` throwing.
		if (stopReason === "no_tool_call" && lastAssistantError) stopReason = "error";
	} catch (err) {
		stopReason = "error";
		error ??= errorMessage(err);
	} finally {
		clearTimeout(timer);
		try {
			session.dispose();
		} catch (err) {
			error ??= errorMessage(err);
		}
	}

	if (episode.finish) stopReason = "finish";
	let success = false;
	try {
		success = robot.solved();
	} catch (err) {
		error ??= errorMessage(err);
	}
	if (
		(stopReason === "error" || stopReason === "no_tool_call") &&
		!episode.finish &&
		!success &&
		error &&
		isInfraError(error)
	) {
		stopReason = "infra_error";
	}

	result.success = success;
	result.claimed = episode.finish?.status;
	result.summary = episode.finish?.summary;
	result.stopReason = stopReason;
	result.error = error;
	result.turns = turns;
	result.toolCalls = episode.trace.length;
	result.imagesPruned = episode.imagesPruned;
	result.costUsd = Number(result.costUsd.toFixed(6));
	return finalize();
}

function summarize(robot: Robot, model: Model<Api>, thinking: string, results: EpisodeResult[]): Summary {
	const scored = results.filter((r) => r.stopReason !== "infra_error" && r.stopReason !== "env_error");
	const successes = scored.filter((r) => r.success).length;
	const mean = (f: (r: EpisodeResult) => number): number =>
		scored.length ? scored.reduce((a, r) => a + f(r), 0) / scored.length : 0;
	return {
		robot: robot.name,
		model: `${model.provider}/${model.id}`,
		thinking,
		episodes: scored.length,
		infraErrors: results.filter((r) => r.stopReason === "infra_error").length,
		envErrors: results.filter((r) => r.stopReason === "env_error").length,
		successes,
		successRate: scored.length ? successes / scored.length : 0,
		claimedButFailed: scored.filter((r) => r.claimed === "success" && !r.success).length,
		meanTurns: mean((r) => r.turns),
		meanElapsedS: mean((r) => r.elapsedS),
		totalCostUsd: Number(results.reduce((a, r) => a + r.costUsd, 0).toFixed(6)),
		results,
	};
}

/** Run tasks x seeds, retrying infra/env errors, and rewrite `<out>/summary.json` after every episode. */
export async function runBatch(
	robot: Robot,
	model: Model<Api>,
	modelRuntime: ModelRuntime,
	opts: RunOptions & { tasks: number[]; seeds: number[] },
	log: (line: string) => void = console.log,
): Promise<Summary> {
	mkdirSync(opts.out, { recursive: true });
	const modelName = `${model.provider}/${model.id}`;
	const results: EpisodeResult[] = [];
	let summary = summarize(robot, model, opts.thinking, results);
	for (const taskId of opts.tasks) {
		for (const seed of opts.seeds) {
			log(`\n=== ${robot.name} task ${taskId} seed ${seed} (${modelName}, thinking ${opts.thinking})`);
			let r = await runEpisode(robot, model, modelRuntime, opts, taskId, seed, 0, log);
			for (
				let attempt = 1;
				(r.stopReason === "infra_error" || r.stopReason === "env_error") && attempt <= opts.retryInfra;
				attempt++
			) {
				log(`=== ${r.stopReason.replace("_", " ")} (${r.error}); retrying (${attempt}/${opts.retryInfra})`);
				r = await runEpisode(robot, model, modelRuntime, opts, taskId, seed, attempt, log);
			}
			results.push(r);
			log(
				`=== ${r.success ? "SUCCESS" : "FAIL"} claimed=${r.claimed ?? "-"} stop=${r.stopReason} turns=${r.turns} tools=${r.toolCalls} ${r.elapsedS}s${r.error ? ` error=${r.error}` : ""}`,
			);
			summary = summarize(robot, model, opts.thinking, results);
			writeJson(join(opts.out, "summary.json"), summary);
		}
	}
	writeJson(join(opts.out, "summary.json"), summary);
	return summary;
}
