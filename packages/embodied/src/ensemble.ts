/**
 * Ensemble: a planner whose every turn is N candidate turns plus one synthesis (CaP-X's
 * query_single_model_ensemble / query_model_ensemble, capx/llm/client.py).
 *
 *   pi -p -e src/<robot> --model ensemble/<base> [--ensemble-temps 0.3,0.7,1.0] ... "Solve the task."
 *   pi -p -e src/<robot> --model ensemble/selfhost/muse-glimmer-30b --ensemble-models relay/gpt-6-astra,selfhost/muse-glimmer-30b ...
 *
 * `ensemble/<base>` is a model whose turn asks the candidates in parallel with the same transcript
 * (system prompt, tools, images): the base model at each `--ensemble-temps` (default 0.3,0.7,1.0),
 * or each model of `--ensemble-models` (at its own default temperature; with `--ensemble-temps`
 * too, every model at every temperature). Then `--ensemble-synth-model` (default the base) is asked
 * once with the same transcript plus a user note listing the candidates that answered (their text
 * and proposed tool calls, none of which ran); its reply, tool calls included, is the turn. A
 * candidate that fails is left out; the turn fails only when all of them fail (with the first
 * candidate's error, so pi's own retry judges it as usual) or the synthesis fails. An abort ends
 * the turn as aborted. Each turn writes a `planner_ensemble` entry {turn, candidates: [{model,
 * temperature, ok, error?, text, tool_calls, cost_usd, tokens, ms}], synth, cost_usd}; `result()`
 * gives `ensemble` {candidates, synth, turns, candidate_calls, candidate_failures, synth_failures,
 * cost_usd} for the `robot_result` row.
 *
 * Cost: the turn's message is the synthesis reply with `usage.cost` replaced by the sum over every
 * call of the turn (the candidates, failed ones that reported usage included, and the synthesis);
 * its token counts stay the synthesis call's own, since pi sizes the context from them. So
 * `--max-cost` (../robot.ts sums `usage.cost` at message_end) and pi's session cost see every call
 * exactly once, and nothing goes through VLM_COST_EVENT. Streaming is per turn: the synthesis
 * events are relayed once it has completed.
 *
 * `--max-api-concurrency` (./api-gate.ts): the gate holds one slot for the turn; the first candidate
 * lane and the synthesis use it, and each further parallel lane takes a slot of its own through
 * `acquire`, so with n = 1 the candidates run one after another.
 *
 * Like ./fallback.ts, the module reads `--model` from `argv` (pi resolves the model before it reads
 * the flags) and registers nothing unless it is `ensemble/<base>`; `defineRobot` mounts it, or
 * `-e packages/embodied/src/ensemble.ts` without a robot. At session start the session's model object
 * is sized to the smallest delegate's contextWindow and maxTokens.
 */

import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEvent,
	type AssistantMessageEventStream,
	createAssistantMessageEventStream,
	createProvider,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type TranscriptContext,
	type Usage,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI, ModelRegistry } from "@earendil-works/pi-coding-agent";
import { API_GATE_EVENT, acquire, release } from "./api-gate.ts";

/** The custom entry written for each ensemble turn. */
export const ENSEMBLE_ENTRY = "planner_ensemble";
export const DEFAULT_TEMPS = [0.3, 0.7, 1.0];

/** `--model ensemble/<base>` from `argv` (`--model value` or `--model=value`). */
export function ensembleArgs(argv: readonly string[]): { base?: string } {
	const i = argv.findIndex((a) => a === "--model" || a.startsWith("--model="));
	const model = i < 0 ? undefined : argv[i] === "--model" ? argv[i + 1] : argv[i].slice("--model=".length);
	return { base: model?.startsWith("ensemble/") ? model.slice("ensemble/".length) : undefined };
}

export type Candidate = { model: string; temperature?: number };

/** The candidate calls: every model (default the base) at every temperature (default DEFAULT_TEMPS, none with --ensemble-models). */
export function candidates(base: string, models?: string, temps?: string): Candidate[] {
	const list = (s?: string) =>
		(s ?? "")
			.split(",")
			.map((x) => x.trim())
			.filter(Boolean);
	const ms = list(models);
	const ts = list(temps).map(Number);
	if (ts.some((t) => !Number.isFinite(t) || t < 0)) throw new Error(`ensemble: bad --ensemble-temps ${temps}`);
	const temperatures: (number | undefined)[] = ts.length ? ts : ms.length ? [undefined] : DEFAULT_TEMPS;
	return (ms.length ? ms : [base]).flatMap((model) =>
		temperatures.map((temperature) => (temperature === undefined ? { model } : { model, temperature })),
	);
}

type Outcome = { message: AssistantMessage; events: AssistantMessageEvent[]; ms: number };

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
const USAGE: Usage = { ...ZERO_COST, totalTokens: 0, cost: { ...ZERO_COST, total: 0 } };
const brief = (s: string, n: number) => (s.length > n ? `${s.slice(0, n)}...` : s);
const label = (c: Candidate) => (c.temperature === undefined ? c.model : `${c.model}@${c.temperature}`);
const textOf = (m: AssistantMessage) =>
	m.content
		.filter((c) => c.type === "text")
		.map((c) => c.text)
		.join("\n")
		.trim();
const callsOf = (m: AssistantMessage) =>
	m.content.flatMap((c) => (c.type === "toolCall" ? [{ name: c.name, arguments: c.arguments }] : []));
const ok = (m: AssistantMessage) => m.stopReason !== "error" && m.stopReason !== "aborted";

/** The user note the synthesis model sees after the transcript: the candidates' text and proposed calls. */
export function synthesisPrompt(answered: { candidate: Candidate; message: AssistantMessage }[]): string {
	const parts = answered.map(({ candidate, message }, i) => {
		const calls = callsOf(message);
		const shown = calls.length
			? calls.map((c) => `- ${c.name} ${JSON.stringify(c.arguments)}`).join("\n")
			: "(no tool call)";
		return `<candidate index="${i + 1}" model="${label(candidate)}">\ntext: ${textOf(message) || "(none)"}\ntool calls:\n${shown}\n</candidate>`;
	});
	return [
		`[ensemble] ${answered.length} candidate next steps were proposed for the conversation above. None of their tool calls has run.`,
		...parts,
		"Synthesize the best next step. Analyze critically and assume no candidate is fully correct; prefer explicit checks over assumptions; combine the best ideas; where candidates disagree fundamentally, choose the more robust approach. Reply as the assistant's next turn and make the tool calls yourself.",
	].join("\n\n");
}

/** Register `ensemble/<base>` when `argv` asks for it; returns `result()` for the robot's row, or undefined. */
export function ensemble(pi: ExtensionAPI, argv: readonly string[] = process.argv) {
	const { base } = ensembleArgs(argv);
	if (!base) return undefined;
	pi.registerFlag("ensemble-temps", {
		type: "string",
		description: `Candidate temperatures for --model ensemble/<base> (comma list; default ${DEFAULT_TEMPS.join(",")}, none with --ensemble-models)`,
	});
	pi.registerFlag("ensemble-models", {
		type: "string",
		description: "Candidate models (provider/model, comma list) instead of the base model",
	});
	pi.registerFlag("ensemble-synth-model", {
		type: "string",
		description: "Model (provider/model) that synthesizes the candidates into the turn (default the base)",
	});

	let registry: ModelRegistry | undefined;
	let gate: { dir: string; n: number } | undefined;
	pi.events.on(API_GATE_EVENT, (g) => {
		gate = g as { dir: string; n: number };
	});
	const counts = { turns: 0, candidate_calls: 0, candidate_failures: 0, synth_failures: 0, cost_usd: 0 };
	const config = () => {
		const list = candidates(
			base,
			pi.getFlag("ensemble-models") as string | undefined,
			pi.getFlag("ensemble-temps") as string | undefined,
		);
		return { list, synth: String(pi.getFlag("ensemble-synth-model") || base) };
	};

	const errored = (model: { api: Api; provider: string; id: string }, err: unknown, aborted = false) =>
		({
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: { ...USAGE, cost: { ...USAGE.cost } },
			stopReason: aborted ? "aborted" : "error",
			errorMessage: err instanceof Error ? err.message : String(err),
			timestamp: Date.now(),
		}) as AssistantMessage;
	const find = (ref: string) => {
		const i = ref.indexOf("/");
		return i > 0 ? registry?.find(ref.slice(0, i), ref.slice(i + 1)) : undefined;
	};
	const delegate = (ref: string) => {
		const m = find(ref);
		if (!m) throw new Error(`ensemble: unknown model ${ref}`);
		return m;
	};

	/** One buffered delegate call. */
	async function call(model: Model<Api>, messages: Message[], options: SimpleStreamOptions): Promise<Outcome> {
		const started = Date.now();
		const events: AssistantMessageEvent[] = [];
		let message: AssistantMessage | undefined;
		try {
			if (!registry) throw new Error("ensemble: no session");
			for await (const ev of registry.streamSimple(model, { messages }, options)) {
				events.push(ev);
				if (ev.type === "done") message = ev.message;
				if (ev.type === "error") message = ev.error;
				if (message) break;
			}
		} catch (err) {
			message = errored(model, err, options.signal?.aborted);
		}
		message ??= errored(model, new Error("stream ended without a result"));
		return { message, events, ms: Date.now() - started };
	}

	/** Run the candidates on at most as many lanes as the gate allows; lane 0 is the turn's own slot. */
	async function fanOut(
		list: Candidate[],
		models: Model<Api>[],
		context: TranscriptContext,
		rest: SimpleStreamOptions,
	): Promise<Outcome[]> {
		const out: (Outcome | undefined)[] = list.map(() => undefined);
		let next = 0;
		const drained = new AbortController();
		const worker = async () => {
			for (let i = next++; i < list.length; i = next++) {
				const t = list[i].temperature;
				out[i] = await call(models[i], context.messages, t === undefined ? rest : { ...rest, temperature: t });
			}
		};
		const extra = async () => {
			let slot: string | undefined;
			try {
				if (gate) {
					const signals = [drained.signal, rest.signal].filter((s): s is AbortSignal => s !== undefined);
					slot = await acquire(gate.dir, gate.n, 250, AbortSignal.any(signals));
				}
			} catch {
				return; // the queue ran dry (or the turn was aborted) before a slot came free
			}
			try {
				await worker();
			} finally {
				if (slot) release(slot);
			}
		};
		const lanes = Math.min(list.length, gate ? gate.n : list.length);
		const first = worker().finally(() => drained.abort());
		await Promise.all([first, ...Array.from({ length: lanes - 1 }, extra)]);
		return out.map((o, i) => o ?? { message: errored(models[i], new Error("aborted"), true), events: [], ms: 0 });
	}

	async function run(
		context: TranscriptContext,
		options: SimpleStreamOptions | undefined,
		out: AssistantMessageEventStream,
	) {
		const end = (message: AssistantMessage) => {
			out.push({ type: "error", reason: message.stopReason === "aborted" ? "aborted" : "error", error: message });
			out.end();
		};
		const { apiKey: _apiKey, headers: _headers, env: _env, ...rest } = options ?? {};
		let list: Candidate[];
		let synthRef: string;
		let models: Model<Api>[];
		let synthModel: Model<Api>;
		try {
			({ list, synth: synthRef } = config());
			models = list.map((c) => delegate(c.model));
			synthModel = delegate(synthRef);
		} catch (err) {
			return end(errored(MODEL, err));
		}
		const turn = ++counts.turns;
		const results = await fanOut(list, models, context, rest);
		const usd = (m: AssistantMessage) => m.usage?.cost?.total ?? 0;
		let spent = results.reduce((a, r) => a + usd(r.message), 0);
		const entry = {
			turn,
			candidates: results.map((r, i) => ({
				model: list[i].model,
				temperature: list[i].temperature ?? null,
				ok: ok(r.message),
				...(ok(r.message) ? {} : { error: brief(r.message.errorMessage ?? r.message.stopReason, 300) }),
				text: brief(textOf(r.message), 500),
				tool_calls: callsOf(r.message).map((c) => ({
					name: c.name,
					arguments: brief(JSON.stringify(c.arguments), 500),
				})),
				cost_usd: Number(usd(r.message).toFixed(6)),
				tokens: r.message.usage?.totalTokens ?? 0,
				ms: r.ms,
			})),
			synth: null as Record<string, unknown> | null,
			cost_usd: 0,
		};
		counts.candidate_calls += results.length;
		counts.candidate_failures += results.filter((r) => !ok(r.message)).length;
		const answered = results.flatMap((r, i) => (ok(r.message) ? [{ candidate: list[i], message: r.message }] : []));
		const finish = (message: AssistantMessage, events?: AssistantMessageEvent[]) => {
			// One message carries the whole turn's cost; its tokens stay the synthesis call's own.
			const cost = { ...USAGE.cost, ...message.usage?.cost };
			for (const r of results)
				for (const k of ["input", "output", "cacheRead", "cacheWrite", "total"] as const)
					cost[k] += r.message.usage?.cost?.[k] ?? 0;
			message.usage = { ...(message.usage ?? USAGE), cost };
			entry.cost_usd = Number(spent.toFixed(6));
			counts.cost_usd += spent;
			pi.appendEntry(ENSEMBLE_ENTRY, entry);
			if (!events) return end(message);
			for (const ev of events) out.push(ev);
			out.end();
		};
		if (rest.signal?.aborted) return finish(errored(MODEL, new Error("aborted"), true));
		if (!answered.length) {
			const first = results[0].message;
			return finish(
				errored(
					models[0],
					`${first.errorMessage ?? "no reply"} (ensemble: all ${results.length} candidates failed)`,
				),
			);
		}
		const note: Message = {
			role: "user",
			content: [{ type: "text", text: synthesisPrompt(answered) }],
			timestamp: Date.now(),
		};
		const s = await call(synthModel, [...context.messages, note], rest);
		spent += usd(s.message);
		entry.synth = {
			model: synthRef,
			ok: ok(s.message),
			...(ok(s.message) ? {} : { error: brief(s.message.errorMessage ?? s.message.stopReason, 300) }),
			tool_calls: callsOf(s.message).map((c) => c.name),
			cost_usd: Number(usd(s.message).toFixed(6)),
			ms: s.ms,
		};
		if (!ok(s.message)) {
			if (s.message.stopReason === "error") counts.synth_failures++;
			return finish(s.message);
		}
		// `done` carries the final message (also the shared partial): it is patched before the relay.
		finish(s.message, s.events);
	}

	function streamSimple(_model: Model<Api>, context: TranscriptContext, options?: SimpleStreamOptions) {
		const out = createAssistantMessageEventStream();
		void run(context, options, out);
		return out;
	}

	const MODEL: Model<"ensemble"> = {
		id: base,
		name: `Ensemble: ${base}`,
		api: "ensemble",
		provider: "ensemble",
		baseUrl: "",
		reasoning: true,
		input: ["text", "image"],
		cost: ZERO_COST,
		contextWindow: 128_000,
		maxTokens: 16_384,
	};
	pi.registerProvider(
		createProvider({
			id: "ensemble",
			name: "Ensemble",
			auth: { apiKey: { name: "Ensemble", resolve: async () => ({ auth: {} }) } },
			models: [MODEL],
			api: { stream: streamSimple, streamSimple },
		}),
	);

	pi.on("session_start", (_event, ctx) => {
		registry = ctx.modelRegistry;
		counts.turns = counts.candidate_calls = counts.candidate_failures = counts.synth_failures = counts.cost_usd = 0;
		if (ctx.model?.provider !== "ensemble") return;
		try {
			const { list, synth } = config();
			const all = [...list.map((c) => c.model), synth].map(find);
			if (all.some((m) => !m)) return;
			ctx.model.contextWindow = Math.min(...all.map((m) => m!.contextWindow));
			ctx.model.maxTokens = Math.min(...all.map((m) => m!.maxTokens));
		} catch {}
	});

	return {
		result: () => {
			let setup: { candidates: string[]; synth: string } | { error: string };
			try {
				const { list, synth } = config();
				setup = { candidates: list.map(label), synth };
			} catch (err) {
				setup = { error: String(err) };
			}
			return { ensemble: { ...setup, ...counts, cost_usd: Number(counts.cost_usd.toFixed(6)) } };
		},
	};
}

export default function ensembleExtension(pi: ExtensionAPI) {
	ensemble(pi);
}
