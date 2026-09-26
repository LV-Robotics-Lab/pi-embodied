/**
 * Fallback: a planner that fails over from a primary model to a backup (OpenETA's 主备端点切换).
 *
 *   pi -p -e src/<robot> --model fallback/<primary> --fallback-model <backup> ... "Solve the task."
 *   pi -p -e src/<robot> --model fallback/relay/gpt-6-astra --fallback-model selfhost/muse-glimmer-30b ...
 *
 * In pi the planner is the model, so the switch is a provider: `fallback/<primary>` is a model whose
 * every turn is planned by the primary (`relay/gpt-6-astra`), or by `--fallback-model` once the primary
 * has failed `--fallback-after` turns in a row (default 2; a failed turn is retried on the primary until
 * then). The backup keeps planning for the rest of the episode, or until `--fallback-retry-primary M`
 * of its turns have passed (default 0: never), when the primary is tried again. Each switch writes a
 * `planner_fallback` entry {from, to, reason, turn} and, without a UI, a `[fallback] ...` stderr line.
 *
 * A failure is a turn that ends in an error pi-ai deems retryable (5xx, 429, network and transport
 * errors, timeouts), one whose message matches `--fallback-on` (default `RELAY_ERRORS`: a relay's 4xx
 * "not supported" / "no credits" answers), or one that streams nothing for `--fallback-timeout` s
 * (default 120; 0 = none). Any other error (a bad request, missing auth) ends the turn as it would
 * without this provider; so does the backup failing. An abort is never a failure.
 *
 * The delegate's events are relayed once its turn has completed, not as they stream: a turn that fails
 * midway leaves nothing behind and is redone on the other model inside the same request, so pi's own
 * retry never sees it and nothing is counted twice. The relayed reply is the delegate's, with its
 * usage, cost and model id, so `--max-cost` (../robot.ts reads `usage.cost` from assistant messages)
 * and pricing stay right. `result()` gives `planner_models`, the turns each model planned, for the
 * `robot_result` row.
 *
 * pi resolves `--model` when it loads the extensions, before any session runs, and reads the flags into
 * the runtime only after that; so the module reads `--model` and `--fallback-model` from `argv` itself
 * and registers nothing without `--fallback-model`. `defineRobot` mounts it; without a robot, mount it
 * with `-e packages/embodied/src/fallback.ts`. pi compacts against this model's window (128k), not the
 * delegates' own.
 */

import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEvent,
	type AssistantMessageEventStream,
	createAssistantMessageEventStream,
	createProvider,
	isRetryableAssistantError,
	type Model,
	type SimpleStreamOptions,
	type TranscriptContext,
	type Usage,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI, ModelRegistry } from "@earendil-works/pi-coding-agent";

/** The custom entry written at each switch: {from, to, reason, turn}. */
export const FALLBACK_ENTRY = "planner_fallback";
/** Default `--fallback-on`: a relay's 4xx answers that mean this endpoint cannot serve the request now. */
export const RELAY_ERRORS =
	/not supported|unsupported|no credits|credit|quota|insufficient|balance|unavailable|no available|无可用|额度|余额/i;

export type FallbackArgs = { primary?: string; fallback?: string };

/** `--model fallback/<primary>` and `--fallback-model <backup>` from `argv` (`--flag value` or `--flag=value`). */
export function fallbackArgs(argv: readonly string[]): FallbackArgs {
	const value = (flag: string) => {
		const i = argv.findIndex((a) => a === flag || a.startsWith(`${flag}=`));
		return i < 0 ? undefined : argv[i] === flag ? argv[i + 1] : argv[i].slice(flag.length + 1);
	};
	const model = value("--model");
	return {
		primary: model?.startsWith("fallback/") ? model.slice("fallback/".length) : undefined,
		fallback: value("--fallback-model"),
	};
}

type Role = "primary" | "fallback";
type Attempt = { events: AssistantMessageEvent[]; message: AssistantMessage; timedOut: boolean };

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
const USAGE: Usage = { ...ZERO_COST, totalTokens: 0, cost: { ...ZERO_COST, total: 0 } };
const brief = (s: string | undefined) => (s ?? "").replace(/\s+/g, " ").slice(0, 200);

/**
 * Register the `fallback/<primary>` model when `argv` has `--fallback-model`; returns `result()` for the
 * robot's result row, or undefined when nothing was registered.
 */
export function fallback(pi: ExtensionAPI, argv: readonly string[] = process.argv) {
	const args = fallbackArgs(argv);
	if (!args.fallback) return undefined;
	pi.registerFlag("fallback-model", {
		type: "string",
		description: "Backup planner (provider/model) for --model fallback/<primary>",
	});
	pi.registerFlag("fallback-after", {
		type: "string",
		default: "2",
		description: "Consecutive primary failures before switching to --fallback-model",
	});
	pi.registerFlag("fallback-retry-primary", {
		type: "string",
		default: "0",
		description: "Try the primary again after this many fallback turns (0 = never)",
	});
	pi.registerFlag("fallback-on", {
		type: "string",
		default: RELAY_ERRORS.source,
		description: "Regex (case-insensitive) of error messages that count as a failure, besides pi's retryable errors",
	});
	pi.registerFlag("fallback-timeout", {
		type: "string",
		default: "120",
		description: "Seconds without a stream event before a turn counts as a failure (0 = none)",
	});

	const refs: Record<Role, string> = { primary: args.primary ?? "", fallback: args.fallback };
	let registry: ModelRegistry | undefined;
	let hasUI = false;
	let warned = false;
	let active: Role = "primary";
	let failures = 0;
	/** Turns the backup has planned since the last switch. */
	let onFallback = 0;
	const turns: Record<Role, number> = { primary: 0, fallback: 0 };

	const note = (line: string) => {
		if (!hasUI) console.error(`[fallback] ${line}`);
	};
	const errored = (model: Model<Api>, err: unknown, stopReason: "error" | "aborted" = "error"): AssistantMessage => ({
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: { ...USAGE, cost: { ...USAGE.cost } },
		stopReason,
		errorMessage: err instanceof Error ? err.message : String(err),
		timestamp: Date.now(),
	});
	function delegate(ref: string): Model<Api> {
		const i = ref.indexOf("/");
		const model = i > 0 ? registry?.find(ref.slice(0, i), ref.slice(i + 1)) : undefined;
		if (!model) throw new Error(`fallback: unknown model ${ref || "(--model is not fallback/<primary>)"}`);
		return model;
	}
	function switchTo(to: Role, reason: string, turn: number) {
		const from = active;
		active = to;
		onFallback = 0;
		pi.appendEntry(FALLBACK_ENTRY, { from: refs[from], to: refs[to], reason, turn });
		note(`turn ${turn}: ${refs[from]} -> ${refs[to]} (${reason})`);
	}

	/** One delegate turn, buffered; a turn that streams nothing for --fallback-timeout is aborted and reported as an error. */
	async function attempt(
		model: Model<Api>,
		context: TranscriptContext,
		options?: SimpleStreamOptions,
	): Promise<Attempt> {
		const given: SimpleStreamOptions = options ?? {};
		const { apiKey: _apiKey, headers: _headers, env: _env, signal, ...rest } = given;
		const own = new AbortController();
		const onAbort = () => own.abort();
		if (signal?.aborted) own.abort();
		else signal?.addEventListener("abort", onAbort, { once: true });
		const idle = Number(pi.getFlag("fallback-timeout")) * 1000;
		let timedOut = false;
		let timer: ReturnType<typeof setTimeout> | undefined;
		const arm = () => {
			if (!(idle > 0)) return;
			clearTimeout(timer);
			timer = setTimeout(() => {
				timedOut = true;
				own.abort();
			}, idle);
			timer.unref();
		};
		const events: AssistantMessageEvent[] = [];
		let message: AssistantMessage | undefined;
		try {
			arm();
			if (!registry) throw new Error("fallback: no session");
			for await (const ev of registry.streamSimple(
				model,
				{ messages: context.messages },
				{ ...rest, signal: own.signal },
			)) {
				arm();
				events.push(ev);
				if (ev.type === "done") message = ev.message;
				if (ev.type === "error") message = ev.error;
				if (message) break;
			}
		} catch (err) {
			message = errored(model, err);
		} finally {
			clearTimeout(timer);
			signal?.removeEventListener("abort", onAbort);
		}
		message ??= errored(model, new Error("stream ended without a result"));
		if (timedOut && !signal?.aborted)
			message = errored(model, new Error(`no reply from ${model.provider}/${model.id} for ${idle / 1000} s`));
		return { events, message, timedOut };
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
		let models: Record<Role, Model<Api>>;
		try {
			models = { primary: delegate(refs.primary), fallback: delegate(refs.fallback) };
		} catch (err) {
			return end(errored(MODEL, err));
		}
		const after = Math.max(1, Number(pi.getFlag("fallback-after")) || 2);
		const retryPrimary = Number(pi.getFlag("fallback-retry-primary")) || 0;
		const on = new RegExp(String(pi.getFlag("fallback-on") ?? RELAY_ERRORS.source), "i");
		const turn = turns.primary + turns.fallback + 1;
		if (active === "fallback" && retryPrimary > 0 && onFallback >= retryPrimary)
			switchTo("primary", `retrying the primary after ${onFallback} fallback turns`, turn);
		for (;;) {
			const a = await attempt(models[active], context, options);
			const m = a.message;
			if (m.stopReason !== "error" && m.stopReason !== "aborted") {
				if (active === "primary") failures = 0;
				else onFallback++;
				turns[active]++;
				for (const ev of a.events) out.push(ev);
				out.end();
				return;
			}
			const counts =
				m.stopReason === "error" && (a.timedOut || isRetryableAssistantError(m) || on.test(m.errorMessage ?? ""));
			if (!counts || options?.signal?.aborted || active === "fallback") return end(m);
			failures++;
			if (failures < after) {
				note(`turn ${turn}: ${refs.primary} failed (${failures}/${after}): ${brief(m.errorMessage)}`);
				continue;
			}
			switchTo("fallback", `${failures} consecutive failures; last: ${brief(m.errorMessage)}`, turn);
		}
	}

	function streamSimple(_model: Model<Api>, context: TranscriptContext, options?: SimpleStreamOptions) {
		const out = createAssistantMessageEventStream();
		void run(context, options, out);
		return out;
	}

	const MODEL: Model<"fallback"> = {
		id: refs.primary,
		name: `Fallback: ${refs.primary} -> ${refs.fallback}`,
		api: "fallback",
		provider: "fallback",
		baseUrl: "",
		reasoning: true,
		input: ["text", "image"],
		cost: ZERO_COST,
		contextWindow: 128_000,
		maxTokens: 16_384,
	};
	pi.registerProvider(
		createProvider({
			id: "fallback",
			name: "Fallback",
			auth: { apiKey: { name: "Fallback", resolve: async () => ({ auth: {} }) } },
			models: refs.primary ? [MODEL] : [],
			api: { stream: streamSimple, streamSimple },
		}),
	);

	pi.on("session_start", (_event, ctx) => {
		registry = ctx.modelRegistry;
		hasUI = ctx.hasUI;
		warned = false;
		active = "primary";
		failures = onFallback = turns.primary = turns.fallback = 0;
	});
	pi.on("before_agent_start", (_event, ctx) => {
		if (ctx.model?.provider === "fallback" || warned) return;
		warned = true;
		const s = `--fallback-model is set but the model is not fallback/<primary>; nothing fails over`;
		if (ctx.hasUI) ctx.ui.notify(s, "warning");
		else console.error(`[fallback] ${s}`);
	});

	return { result: () => ({ planner_models: { ...turns } }) };
}

export default function fallbackExtension(pi: ExtensionAPI) {
	fallback(pi);
}
