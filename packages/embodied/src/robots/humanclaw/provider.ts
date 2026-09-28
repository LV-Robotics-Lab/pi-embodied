/**
 * `humanclaw-psv/<base>`: HumanCLAW's paper planner (prompt v4 + verifier v3, ./psv.ts) as a pi model.
 *
 *   pi -e packages/embodied/src/robots/humanclaw --units=both --humanclaw-mode paper \
 *     --model humanclaw-psv/selfhost/muse-glimmer-30b --episode one "Solve the task."
 *
 * Every turn is one decision of HumanCLAW's evaluator loop: the latest ego image from the robot's
 * `look` / `act` result, the text history this provider keeps (evaluator._history_item rows, one
 * `humanclaw_psv_step` session entry per decision, rebuilt on resume), the planner request to the
 * base model (the prompt text, then the image; temperature 0, --humanclaw-max-tokens, a JSON-object
 * response format, no system prompt, no tools, no transcript), up to 5 attempts at the same state,
 * the verifier where its route asks for one, and the final action as one `act` call (unit + param,
 * which the robot maps back through HumanCLAW's own `_chooser_action`). The decision record goes to
 * the robot on DECISION_EVENT so the env server's metric recorder and step JSON get the planner's
 * raw JSON. The first turn is `look`; a done episode ends with `finish`. Like ../../planner/ensemble.ts
 * it reads --model from argv and registers nothing unless it is `humanclaw-psv/<base>`.
 */

import {
	type Api,
	type AssistantMessage,
	createAssistantMessageEventStream,
	createProvider,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type ToolCall,
	type TranscriptContext,
	type Usage,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI, ModelRegistry } from "@earendil-works/pi-coding-agent";
import { unitOf } from "./actions.ts";
import { type Decision, historyItem, PsvPlanner } from "./psv.ts";

/** pi.events channel: the decision the next `act` executes (the robot records it on the env server). */
export const DECISION_EVENT = "pi-embodied:humanclaw-decision";
/** One session entry per decision: the planner/verifier record, the history row and the planner state. */
export const PSV_ENTRY = "humanclaw_psv_step";
type Json = Record<string, unknown>;

const ZERO = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
const USAGE: Usage = { ...ZERO, totalTokens: 0, cost: { ...ZERO, total: 0 } };

/** The base model of `--model humanclaw-psv/<provider>/<id>`, from argv (pi resolves models before flags). */
export function psvBase(argv: readonly string[] = process.argv): string | undefined {
	for (let i = 0; i < argv.length; i++) {
		const v = argv[i] === "--model" ? argv[i + 1] : argv[i].startsWith("--model=") ? argv[i].slice(8) : undefined;
		if (v?.startsWith("humanclaw-psv/")) return v.slice("humanclaw-psv/".length);
	}
	return undefined;
}

export function mountPsv(pi: ExtensionAPI, base: string, ask?: (m: Json[]) => Promise<{ text: string; usage?: Json }>) {
	let registry: ModelRegistry | undefined;
	let history: Json[] = [];
	let pending: string | undefined;
	let over = false;
	let ids = 0;
	let image: { data: string; mimeType: string } | undefined;
	let restored: Parameters<PsvPlanner["restore"]>[0] | undefined;

	/** One base-model request: the prompt text, then the current ego image (planner._message). */
	const request = async (prompt: string, _stage: string, signal?: AbortSignal) => {
		const content: Json[] = [
			{ type: "text", text: prompt },
			...(image ? [{ type: "image", data: image.data, mimeType: image.mimeType }] : []),
		];
		if (ask) return ask(content);
		if (!registry) throw new Error("humanclaw-psv: no session");
		const i = base.indexOf("/");
		const model = registry.find(base.slice(0, i), base.slice(i + 1));
		if (!model) throw new Error(`humanclaw-psv: unknown base model ${base}`);
		const json = pi.getFlag("humanclaw-json-format") !== false;
		let message: AssistantMessage | undefined;
		for await (const ev of registry.streamSimple(
			model,
			{ messages: [{ role: "user", content, timestamp: Date.now() } as unknown as Message] },
			{
				temperature: 0,
				// The paper leaves reasoning at each model's default; --humanclaw-reasoning sets a level.
				...(String(pi.getFlag("humanclaw-reasoning") ?? "")
					? { reasoning: String(pi.getFlag("humanclaw-reasoning")) as SimpleStreamOptions["reasoning"] }
					: {}),
				maxTokens: Number(pi.getFlag("humanclaw-max-tokens") ?? 4096) || 4096,
				signal,
				// The release's adapter asks for a JSON object (configs/models/vllm_openai_compatible.json).
				onPayload: (payload) =>
					json && model.api === "openai-completions"
						? { ...(payload as Json), response_format: { type: "json_object" } }
						: undefined,
			},
		)) {
			if (ev.type === "done") message = ev.message;
			if (ev.type === "error") throw new Error(ev.error.errorMessage ?? "base model error");
		}
		if (!message) throw new Error("base model stream ended without a reply");
		const text = message.content.flatMap((c) => (c.type === "text" ? [c.text] : [])).join("");
		return { text, usage: { prompt_tokens: message.usage.input, completion_tokens: message.usage.output } };
	};
	const planner = new PsvPlanner(request);

	const toolCall = (name: string, args: Json): ToolCall => ({
		type: "toolCall",
		id: `psv_${++ids}`,
		name,
		arguments: args as ToolCall["arguments"],
	});

	async function next(messages: Message[], signal?: AbortSignal): Promise<{ text: string; calls: ToolCall[] }> {
		if (over) return { text: "The HumanCLAW planner already ended this episode.", calls: [] };
		if (!pending) {
			const c = toolCall("look", {});
			pending = c.id;
			return { text: "look (the first ego image)", calls: [c] };
		}
		const result = messages.find((m) => m.role === "toolResult" && m.toolCallId === pending);
		if (!result || result.role !== "toolResult") {
			over = true;
			return { text: `no result for ${pending}; the planner stops.`, calls: [] };
		}
		const details = (result.details ?? {}) as Json;
		if (result.isError) {
			over = true;
			const t = result.content.flatMap((c) => (c.type === "text" ? [c.text] : [])).join(" ");
			return { text: `the robot refused: ${t.slice(0, 300)}; the planner stops.`, calls: [] };
		}
		if (details.done === true) {
			over = true;
			const c = toolCall("finish", {
				status: details.stopped ? "success" : "failure",
				summary: details.stopped ? "Stop/Stand chosen by the planner" : `max steps reached (${details.step})`,
			});
			return { text: "episode over", calls: [c] };
		}
		const img = [...result.content].reverse().find((c) => c.type === "image") as
			| { data: string; mimeType: string }
			| undefined;
		if (!img) throw new Error("humanclaw-psv: the observation carries no ego image");
		image = img;
		if (planner.instruction === undefined) {
			planner.reset(String(details.instruction ?? ""));
			if (restored) planner.restore(restored);
		}
		const step = history.length;
		const decision: Decision = await planner.act(history, signal);
		const item = historyItem(step, decision);
		history.push(item);
		pi.appendEntry(PSV_ENTRY, { step, decision, history_item: item, state: planner.state() });
		pi.events.emit(DECISION_EVENT, decision);
		const { unit, param } = unitOf(decision.action);
		const c = toolCall("act", param === undefined ? { unit } : { unit, param });
		pending = c.id;
		return { text: `${decision.action.action_name}`, calls: [c] };
	}

	function streamSimple(m: Model<Api>, context: TranscriptContext, options?: SimpleStreamOptions) {
		const stream = createAssistantMessageEventStream();
		const message: AssistantMessage = {
			role: "assistant",
			content: [],
			api: m.api,
			provider: m.provider,
			model: m.id,
			usage: { ...USAGE, cost: { ...USAGE.cost } },
			stopReason: "stop",
			timestamp: Date.now(),
		};
		next(context.messages as Message[], options?.signal).then(
			(turn) => {
				stream.push({ type: "start", partial: message });
				if (turn.text) {
					message.content.push({ type: "text", text: turn.text });
					stream.push({ type: "text_start", contentIndex: 0, partial: message });
					stream.push({ type: "text_delta", contentIndex: 0, delta: turn.text, partial: message });
					stream.push({ type: "text_end", contentIndex: 0, content: turn.text, partial: message });
				}
				for (const c of turn.calls) {
					const contentIndex = message.content.push(c) - 1;
					stream.push({ type: "toolcall_start", contentIndex, partial: message });
					stream.push({ type: "toolcall_end", contentIndex, toolCall: c, partial: message });
				}
				message.stopReason = turn.calls.length ? "toolUse" : "stop";
				stream.push({ type: "done", reason: message.stopReason, message });
				stream.end();
			},
			(err) => {
				message.stopReason = options?.signal?.aborted ? "aborted" : "error";
				message.errorMessage = err instanceof Error ? err.message : String(err);
				stream.push({ type: "error", reason: message.stopReason, error: message });
				stream.end();
			},
		);
		return stream;
	}

	pi.registerProvider(
		createProvider({
			id: "humanclaw-psv",
			name: "HumanCLAW plan-skill-verify",
			auth: { apiKey: { name: "HumanCLAW PSV", resolve: async () => ({ auth: {} }) } },
			models: [
				{
					id: base,
					name: `HumanCLAW PSV: ${base}`,
					api: "humanclaw-psv",
					provider: "humanclaw-psv",
					baseUrl: "",
					reasoning: false,
					input: ["text", "image"],
					cost: ZERO,
					contextWindow: 100_000_000,
					maxTokens: 16_384,
				} as Model<"humanclaw-psv">,
			],
			api: { stream: streamSimple, streamSimple },
		}),
	);

	pi.on("session_start", (_e, ctx) => {
		registry = ctx.modelRegistry;
		// Resume: the history and planner state of the branch's decisions.
		const entries = ctx.sessionManager
			.getBranch()
			.filter((e) => e.type === "custom" && e.customType === PSV_ENTRY)
			.map((e) => (e.type === "custom" ? (e.data as Json) : {}));
		history = entries.map((e) => e.history_item as Json);
		pending = undefined;
		over = false;
		const last = entries[entries.length - 1];
		planner.instruction = undefined;
		restored = last?.state as typeof restored;
	});
	return { planner, history: () => history, streamSimple };
}
