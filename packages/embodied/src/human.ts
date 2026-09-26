/**
 * Human-as-VLM (OpenETA's manual VLM console, tools/manual_vlm_proxy.py + docs/manual-vlm-harness-debugger.md):
 * a person answers the model's requests. In pi the planner and the side VLMs are models, so this is a
 * provider: `human/operator` is a model whose every reply the operator writes.
 *
 *   pi -e packages/embodied/src/libero -e packages/embodied/src/dashboard --dashboard --model human/operator
 *   pi ... --model selfhost/muse-glimmer-30b --attach-vlm-model human/operator   # only check_attached asks a person
 *
 * Each request is shown to the operator through pi's extension UI (`ctx.ui`): a widget with the
 * request (the latest prompt or tool results, the images it carries, the tools on offer), then
 * `ctx.ui.select` for how to answer (a text reply, or one of the offered tools) and `ctx.ui.input`
 * for the text or the tool's JSON arguments (its required parameters as the placeholder). The TUI shows these
 * dialogs; an RPC client answers them as `extension_ui_request`s (docs/rpc-extension-ui.md). The
 * dashboard (../dashboard, --dashboard) shows the same request with its images and a composer, and
 * whichever answers first wins; the other is cancelled. A tool call must name an offered tool and
 * carry a JSON object; otherwise the operator is asked again. Without pi's UI and without the
 * dashboard there is nobody to ask, and the turn fails.
 *
 * pi resolves `--model` before extension flags are read, so, like ./fallback.ts, the provider is
 * registered only when a `human/...` model appears on the command line (any flag's value), and the
 * robot base (../robot.ts) mounts it. Replies cost nothing; each is a normal assistant message in the session.
 */

import { randomBytes } from "node:crypto";
import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	createAssistantMessageEventStream,
	createProvider,
	getCurrentSystemMessage,
	getCurrentTools,
	type ImageContent,
	type Model,
	type SimpleStreamOptions,
	type ToolCall,
	type TranscriptContext,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

/** `pi.events` channel of each pending request (read by the dashboard, which sets `claimed`). */
export const HUMAN_EVENT = "pi-embodied:human-request";
export const HUMAN_PROVIDER = "human";

export type HumanTool = { name: string; description: string; parameters: unknown };
export type HumanReply = { text?: string; tool?: { name: string; arguments: unknown } };
export type HumanRequest = {
	id: string;
	turn: number;
	/** "planner" when tools are on offer, "vlm" for a side question (no tools: a text answer). */
	kind: "planner" | "vlm";
	/** The request's system prompt (the robot prompt, or a side VLM's instructions). */
	system: string;
	/** What is new since the last reply: the prompt, or the tool results, as text. */
	text: string;
	images: ImageContent[];
	tools: HumanTool[];
	/** Answer it; returns why the reply was refused, or undefined when it was taken. */
	answer: (reply: HumanReply) => string | undefined;
	/** Set by a dashboard that shows the request. */
	claimed?: boolean;
	/** Emitted once more with `done` when the request is answered, dismissed or aborted. */
	done?: boolean;
};

/** Whether `argv` names a `human/...` model (`--flag human/x` or `--flag=human/x`). */
export function humanRequested(argv: readonly string[]): boolean {
	return argv.some((a) => a.startsWith(`${HUMAN_PROVIDER}/`) || /^--[\w-]+=human\//.test(a));
}

const ZERO = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

/** A one-line JSON skeleton of a tool's required parameters (optional ones named after `//`). */
export function argumentSkeleton(parameters: unknown): string {
	const props = ((parameters as { properties?: Record<string, { type?: string }> })?.properties ?? {}) as Record<
		string,
		{ type?: string; anyOf?: unknown }
	>;
	const required = new Set(((parameters as { required?: string[] })?.required ?? []) as string[]);
	const sample = (s: { type?: string }) =>
		s.type === "number" || s.type === "integer"
			? 0
			: s.type === "boolean"
				? false
				: s.type === "array"
					? []
					: s.type === "object"
						? {}
						: "";
	const out: Record<string, unknown> = {};
	for (const [k, s] of Object.entries(props)) if (required.has(k)) out[k] = sample(s);
	const optional = Object.keys(props).filter((k) => !required.has(k));
	return `${JSON.stringify(out)}${optional.length ? ` // optional: ${optional.join(", ")}` : ""}`;
}

/** Validate a reply against the offered tools: the ToolCall content, or the reason it is refused. */
export function checkReply(reply: HumanReply, tools: HumanTool[]): string | undefined {
	if (reply.tool) {
		if (!tools.some((t) => t.name === reply.tool?.name)) return `"${reply.tool.name}" is not an offered tool`;
		const a = reply.tool.arguments;
		if (!a || typeof a !== "object" || Array.isArray(a)) return "arguments must be a JSON object";
		return undefined;
	}
	return reply.text?.trim() ? undefined : "empty reply";
}

/** Parse the typed JSON; a trailing `// ...` comment (the placeholder's) is ignored. */
export function parseArguments(raw: string): unknown {
	const body = raw.replace(/\s\/\/[^"]*$/, "").trim();
	return JSON.parse(body || "{}");
}

/** What the model would see as new: the trailing user message or tool results after the last assistant reply. */
export function digest(context: TranscriptContext): { text: string; images: ImageContent[] } {
	const msgs = context.messages;
	let i = msgs.length;
	while (i > 0 && msgs[i - 1].role !== "assistant") i--;
	const parts: string[] = [];
	const images: ImageContent[] = [];
	for (const m of msgs.slice(i)) {
		if (m.role !== "user" && m.role !== "toolResult") continue;
		const content = typeof m.content === "string" ? [{ type: "text" as const, text: m.content }] : m.content;
		const texts = content.filter((c) => c.type === "text").map((c) => (c as { text: string }).text);
		for (const c of content) if (c.type === "image") images.push(c as ImageContent);
		const head = m.role === "toolResult" ? `[${m.toolName}${m.isError ? " error" : ""}] ` : "";
		parts.push(head + texts.join("\n"));
	}
	return { text: parts.join("\n\n").slice(0, 20_000), images };
}

export function human(pi: ExtensionAPI, argv: readonly string[] = process.argv) {
	if (!humanRequested(argv)) return;
	let ctx: ExtensionContext | undefined;
	let turn = 0;
	pi.on("session_start", (_e, c) => {
		ctx = c;
		turn = 0;
	});
	pi.on("session_shutdown", () => {
		ctx = undefined;
	});

	const MODEL: Model<"human"> = {
		id: "operator",
		name: "Human operator",
		api: "human",
		provider: HUMAN_PROVIDER,
		baseUrl: "",
		reasoning: false,
		input: ["text", "image"],
		cost: ZERO,
		contextWindow: 1_000_000,
		maxTokens: 32_000,
	};

	function message(model: Model<Api>, content: AssistantMessage["content"], extra: Partial<AssistantMessage> = {}) {
		return {
			role: "assistant",
			content,
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: { ...ZERO, totalTokens: 0, cost: { ...ZERO, total: 0 } },
			stopReason: content.some((c) => c.type === "toolCall") ? "toolUse" : "stop",
			timestamp: Date.now(),
			...extra,
		} as AssistantMessage;
	}

	/** Ask through ctx.ui until a valid reply, the dashboard's answer, dismissal or abort. */
	async function dialogs(c: ExtensionContext, req: HumanRequest, signal: AbortSignal) {
		const TEXT = "reply with text";
		for (;;) {
			const choice =
				req.tools.length === 0
					? TEXT
					: await c.ui.select(`Turn ${req.turn}: answer as the model`, [TEXT, ...req.tools.map((t) => t.name)], {
							signal,
						});
			if (choice === undefined || signal.aborted) return;
			if (choice === TEXT) {
				const text = await c.ui.input(req.kind === "vlm" ? "Answer the question" : "Reply", "", { signal });
				if (text === undefined || signal.aborted) return;
				const why = req.answer({ text });
				if (!why) return;
				c.ui.notify(why, "warning");
				continue;
			}
			const tool = req.tools.find((t) => t.name === choice) as HumanTool;
			const raw = await c.ui.input(`${tool.name} arguments (JSON object)`, argumentSkeleton(tool.parameters), {
				signal,
			});
			if (raw === undefined || signal.aborted) return;
			let args: unknown;
			try {
				args = parseArguments(raw);
			} catch (err) {
				c.ui.notify(`invalid JSON: ${(err as Error).message}`, "warning");
				continue;
			}
			const why = req.answer({ tool: { name: tool.name, arguments: args } });
			if (!why) return;
			c.ui.notify(why, "warning");
		}
	}

	async function run(
		model: Model<Api>,
		context: TranscriptContext,
		options: SimpleStreamOptions | undefined,
		out: AssistantMessageEventStream,
	) {
		const fail = (why: string, aborted = false) => {
			const m = message(model, [], { stopReason: aborted ? "aborted" : "error", errorMessage: why });
			out.push({ type: "error", reason: aborted ? "aborted" : "error", error: m });
			out.end();
		};
		const tools: HumanTool[] = getCurrentTools(context.messages).map((t) => ({
			name: t.name,
			description: t.description,
			parameters: t.parameters,
		}));
		const system = getCurrentSystemMessage(context.messages);
		const sys =
			typeof system?.content === "string" ? system.content : (system?.content ?? []).map((c) => c.text).join("\n");
		const { text, images } = digest(context);
		const done = new AbortController();
		const signal = options?.signal;
		let reply: HumanReply | undefined;
		const req: HumanRequest = {
			id: randomBytes(6).toString("hex"),
			turn: ++turn,
			kind: tools.length ? "planner" : "vlm",
			system: sys,
			text,
			images,
			tools,
			answer: (r) => {
				if (reply || done.signal.aborted) return "the request is already answered";
				const why = checkReply(r, tools);
				if (why) return why;
				reply = r;
				done.abort();
				return undefined;
			},
		};
		const c = ctx;
		pi.events.emit(HUMAN_EVENT, req);
		if (!c?.hasUI && !req.claimed)
			return fail("human/operator has nobody to ask: run pi interactively or over RPC, or with --dashboard");
		const onAbort = () => done.abort();
		signal?.addEventListener("abort", onAbort, { once: true });
		try {
			if (c?.hasUI) {
				c.ui.setWidget("human", [
					`Turn ${req.turn} (${req.kind}): answer as the model. ${images.length} image(s)${req.claimed ? "; also on the dashboard" : ""}.`,
					...text.split("\n").slice(-12),
				]);
				await dialogs(c, req, done.signal);
			}
			// Dismissed dialogs (or no UI): the dashboard may still answer; abort ends it.
			if (!reply && !done.signal.aborted && req.claimed)
				await new Promise<void>((resolve) =>
					done.signal.addEventListener("abort", () => resolve(), { once: true }),
				);
		} finally {
			signal?.removeEventListener("abort", onAbort);
			c?.hasUI && c.ui.setWidget("human", undefined);
			done.abort();
			pi.events.emit(HUMAN_EVENT, { ...req, done: true });
		}
		if (signal?.aborted) return fail("aborted", true);
		if (!reply) return fail("the operator dismissed the request");
		const content: AssistantMessage["content"] = [];
		if (reply.tool) {
			const call: ToolCall = {
				type: "toolCall",
				id: `human_${randomBytes(6).toString("hex")}`,
				name: reply.tool.name,
				arguments: reply.tool.arguments as ToolCall["arguments"],
			};
			content.push(call);
		} else content.push({ type: "text", text: String(reply.text).trim() });
		const final = message(model, content);
		const partial = message(model, []);
		out.push({ type: "start", partial });
		content.forEach((part, i) => {
			if (part.type === "toolCall") {
				out.push({ type: "toolcall_start", contentIndex: i, partial });
				out.push({ type: "toolcall_end", contentIndex: i, toolCall: part, partial: final });
			} else if (part.type === "text") {
				out.push({ type: "text_start", contentIndex: i, partial });
				out.push({ type: "text_delta", contentIndex: i, delta: part.text, partial: final });
				out.push({ type: "text_end", contentIndex: i, content: part.text, partial: final });
			}
		});
		out.push({ type: "done", reason: final.stopReason === "toolUse" ? "toolUse" : "stop", message: final });
		out.end();
	}

	function streamSimple(model: Model<Api>, context: TranscriptContext, options?: SimpleStreamOptions) {
		const out = createAssistantMessageEventStream();
		void run(model, context, options, out).catch((err) => {
			const m = message(model, [], { stopReason: "error", errorMessage: String(err) });
			out.push({ type: "error", reason: "error", error: m });
			out.end();
		});
		return out;
	}

	pi.registerProvider(
		createProvider({
			id: HUMAN_PROVIDER,
			name: "Human operator",
			auth: { apiKey: { name: "Human operator", resolve: async () => ({ auth: {} }) } },
			models: [MODEL],
			api: { stream: streamSimple, streamSimple },
		}),
	);
}

export default function humanExtension(pi: ExtensionAPI) {
	human(pi);
}
