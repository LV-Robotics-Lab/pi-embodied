import assert from "node:assert/strict";
import { test } from "node:test";
import { convertMessages } from "@earendil-works/pi-ai/api/openai-completions";
import { normalizeContext } from "@earendil-works/pi-ai/utils/transcript";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { dropReplayedThinking, unsigned } from "../src/planner/replay-thinking.ts";
import { defineRobot, RESULT_ENTRY } from "../src/robot.ts";
import { deployFlags } from "./helpers/deployment.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A toy robot on a stub pi, like context-images.test.ts: `emit` runs the handlers of one event. */
function stubPi(values: Record<string, unknown> = {}) {
	values = deployFlags(values);
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const entries: { type: string; data: any }[] = [];
	const stderr: string[] = [];
	const api: Record<string, unknown> = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: { emit: () => {}, on: () => () => {} },
	};
	const pi = new Proxy(api, { get: (t, k: string) => t[k] ?? (() => {}) }) as unknown as ExtensionAPI;
	const ctx = { hasUI: true, ui: { notify: () => {} }, sessionManager: { getBranch: () => [] } };
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let out: unknown;
		for (const fn of handlers.get(name) ?? []) out = (await fn({ type: name, ...event }, ctx)) ?? out;
		return out;
	}
	pi.registerFlag("seed", { type: "string", default: "0" });
	defineRobot(pi, {
		name: "toy",
		task: ["seed"],
		keepImages: 2,
		start: async () => ["look"],
		result: () => ({ success: false }),
		finish: {
			description: "finish",
			parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
			result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
		},
	});
	const log = console.error;
	console.error = (line: string) => stderr.push(String(line));
	const restore = () => {
		console.error = log;
	};
	return { flags, entries, stderr, emit, restore };
}

const usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};
/** An openai-completions planner turn: thinking as NInfer / vLLM return it (`reasoning_content`), text, one `act` call. */
const turn = (n: number, thinking: Record<string, unknown> = { thinkingSignature: "reasoning_content" }) => ({
	role: "assistant",
	api: "openai-completions",
	provider: "selfhost",
	model: "qwen3.8-27b",
	content: [
		{ type: "thinking", thinking: `reasoning ${n}`, ...thinking },
		{ type: "text", text: `step ${n}` },
		{ type: "toolCall", id: `c${n}`, name: "act", arguments: { unit: "MV_DOWN" } },
	],
	usage,
	stopReason: "toolUse",
	timestamp: n,
});
const result = (n: number) => ({
	role: "toolResult",
	toolCallId: `c${n}`,
	toolName: "act",
	content: [{ type: "text", text: `ok ${n}` }],
	isError: false,
	timestamp: n,
});
const thinkingOf = (messages: any[]) =>
	messages.map((m) =>
		m.role === "assistant" ? m.content.filter((c: any) => c.type === "thinking").map((c: any) => c.thinking) : null,
	);

/** A robot episode as pi records it: one user message, then tool loops. The last turn's `act` is being answered. */
const episode = [
	{ role: "user", content: "Solve the task.", timestamp: 0 },
	turn(1),
	result(1),
	turn(2),
	result(2),
	turn(3),
	result(3),
];

test("earlier turns' unsigned thinking leaves the request; the latest turn's stays, and so do text and tool calls", () => {
	const before = structuredClone(episode);
	const out = dropReplayedThinking(episode as any);
	assert.ok(out);
	assert.deepEqual(thinkingOf(out.messages), [null, [], null, [], null, ["reasoning 3"], null]);
	assert.deepEqual(
		out.messages.map((m: any) => (m.role === "assistant" ? m.content.map((c: any) => c.type) : m.role)),
		[
			"user",
			["text", "toolCall"],
			"toolResult",
			["text", "toolCall"],
			"toolResult",
			["thinking", "text", "toolCall"],
			"toolResult",
		],
	);
	assert.deepEqual(episode, before, "the session's messages are not edited in place");
	// A block that came in with no signature at all (an endpoint answering in `reasoning`) is text-only too.
	assert.ok(unsigned({ type: "thinking", thinking: "x" }));
	assert.ok(unsigned({ type: "thinking", thinking: "x", thinkingSignature: "reasoning" }));
});

test("nothing to drop leaves the context alone", () => {
	assert.equal(dropReplayedThinking([episode[0], turn(1), result(1)] as any), undefined, "only the latest turn");
	assert.equal(dropReplayedThinking([episode[0]] as any), undefined);
	const noThinking = [episode[0], { ...turn(1), content: turn(1).content.slice(1) }, result(1), turn(2), result(2)];
	assert.equal(dropReplayedThinking(noThinking as any), undefined);
});

test("signed thinking and other apis are never touched: Anthropic, OpenAI Responses, redacted and reasoning_details blocks", () => {
	const anthropic = (n: number) => ({
		...turn(n, { thinkingSignature: "EqQBCkYIBRgCKkCx..." }),
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-sonnet-4-5",
	});
	const responses = (n: number) => ({
		...turn(n, { thinking: "", thinkingSignature: '{"id":"rs_1","encrypted_content":"gAAAA"}' }),
		api: "openai-responses",
		provider: "openai",
		model: "gpt-5.4",
	});
	const redacted = turn(4, { redacted: true, thinkingSignature: "" });
	const details = turn(5, { thinkingSignature: '[{"type":"reasoning.encrypted","data":"abc"}]' });
	const unknownApi = { ...turn(6), api: undefined };
	const messages = [
		episode[0],
		anthropic(1),
		result(1),
		responses(2),
		result(2),
		redacted,
		result(4),
		details,
		result(5),
		unknownApi,
		result(6),
		turn(7),
		result(7),
	];
	assert.equal(
		dropReplayedThinking(messages as any),
		undefined,
		"every earlier block is signed or of another api: kept",
	);
	// Mixed: only the unsigned openai-completions block goes.
	const mixed = [episode[0], anthropic(1), result(1), turn(2), result(2), turn(3), result(3)];
	const out = dropReplayedThinking(mixed as any);
	assert.ok(out);
	assert.deepEqual(thinkingOf(out.messages), [null, ["reasoning 1"], null, [], null, ["reasoning 3"], null]);
	assert.equal((out.messages[1] as any).content[0].thinkingSignature, "EqQBCkYIBRgCKkCx...");
});

test("the robot's context handler: --replay-thinking false (default) drops, true replays, and params records the value", async () => {
	const off = stubPi();
	try {
		const out = (await off.emit("context", { messages: episode })) as { messages: any[] };
		assert.deepEqual(thinkingOf(out.messages), [null, [], null, [], null, ["reasoning 3"], null]);
		await off.emit("session_start");
		await off.emit("agent_start");
		await off.emit("session_shutdown");
		const r = off.entries.find((e) => e.type === RESULT_ENTRY)?.data;
		assert.equal(r.params["replay-thinking"], "false");
		assert.equal(r.params_default["replay-thinking"], "false");
	} finally {
		off.restore();
	}
	const on = stubPi({ "replay-thinking": "true" });
	try {
		assert.equal(await on.emit("context", { messages: episode }), undefined, "pi's own replay");
		await on.emit("session_start");
		await on.emit("agent_start");
		await on.emit("session_shutdown");
		assert.equal(on.entries.find((e) => e.type === RESULT_ENTRY)?.data.params["replay-thinking"], "true");
	} finally {
		on.restore();
	}
	// A bare --replay-thinking reads true (pi gives a flag without a value `true`).
	const bare = stubPi({ "replay-thinking": true });
	try {
		assert.equal(await bare.emit("context", { messages: episode }), undefined);
	} finally {
		bare.restore();
	}
});

test("another value than true / false fails closed before the robot starts", async () => {
	const f = stubPi({ "replay-thinking": "maybe" });
	try {
		await f.emit("session_start");
		await f.emit("session_shutdown");
		const r = f.entries.find((e) => e.type === RESULT_ENTRY)?.data;
		assert.equal(r.env_error, true);
		assert.match(r.error, /--replay-thinking must be true or false, got "maybe"/);
	} finally {
		f.restore();
	}
});

test("the outgoing openai-completions request carries no reasoning_content for earlier turns when dropped, and does when replayed", () => {
	const model = {
		id: "qwen3.8-27b",
		name: "Qwen3.8 27B",
		api: "openai-completions",
		provider: "selfhost",
		baseUrl: "http://127.0.0.1:8000/v1",
		reasoning: true,
		input: ["text", "image"],
		contextWindow: 65536,
		maxTokens: 16384,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		compat: { thinkingFormat: "qwen", supportsReasoningEffort: true, supportsStrictMode: false },
	} as any;
	// The fields convertMessages reads of the resolved compat (pi-ai keeps its resolver private).
	const compat = {
		requiresAssistantAfterToolResult: false,
		requiresThinkingAsText: false,
		requiresReasoningContentOnAssistantMessages: false,
		supportsDeveloperRole: false,
		supportsMidConvoSystemMessages: false,
		supportsMidConvoToolAdditions: false,
	} as unknown as Parameters<typeof convertMessages>[2];
	const request = (messages: any[]) =>
		convertMessages(model, normalizeContext({ systemPrompt: "robot", messages }), compat)
			.filter((m) => m.role === "assistant")
			.map((m: any) => m.reasoning_content ?? null);
	assert.deepEqual(
		request(episode),
		["reasoning 1", "reasoning 2", "reasoning 3"],
		"pi's own request replays every turn",
	);
	const dropped = dropReplayedThinking(episode as any);
	assert.ok(dropped);
	assert.deepEqual(request(dropped.messages), [null, null, "reasoning 3"]);
	// The rest of the request is the same: text and tool calls of every turn.
	const shape = (messages: any[]) =>
		convertMessages(model, normalizeContext({ systemPrompt: "robot", messages }), compat).map((m: any) => [
			m.role,
			m.content ?? null,
			m.tool_calls?.map((t: any) => t.id) ?? null,
		]);
	assert.deepEqual(shape(dropped.messages), shape(episode));
});
