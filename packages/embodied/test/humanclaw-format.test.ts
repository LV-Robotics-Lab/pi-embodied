import assert from "node:assert/strict";
import { test } from "node:test";
import type { Api, AssistantMessage, Message, Model, SimpleStreamOptions, ToolCall } from "@earendil-works/pi-ai";
import { normalizeContext } from "@earendil-works/pi-ai";
import type { ModelRegistry } from "@earendil-works/pi-coding-agent";
import { mountPsv } from "../src/robots/humanclaw/provider.ts";
import { stubPi } from "./sim-stub.ts";

test("PSV retries NInfer's unsupported JSON format at the same observation and remembers text mode", async () => {
	const s = stubPi();
	const requests: unknown[] = [];
	const contexts: unknown[] = [];
	const options: SimpleStreamOptions[] = [];
	const model = { id: "qwen", api: "openai-completions", provider: "selfhost" } as Model<Api>;
	const registry = {
		find: () => model,
		async *streamSimple(_model: Model<Api>, context: unknown, opts: SimpleStreamOptions) {
			const payload = await opts.onPayload?.({}, model);
			requests.push(payload);
			contexts.push(context);
			options.push(opts);
			if (requests.length === 1) {
				yield {
					type: "error",
					error: { errorMessage: '400: {"code":"response_format_not_supported","param":"response_format"}' },
				};
				return;
			}
			yield {
				type: "done",
				message: {
					content: [
						{
							type: "text",
							text: JSON.stringify({
								visible_state: "The couch is visible.",
								action_name: "Turn<left><30>",
								action_id: 2,
							}),
						},
					],
					usage: { input: 10, output: 20 },
				},
			};
		},
	} as unknown as ModelRegistry;
	s.pi.on("session_start", (_event, ctx) => {
		Object.assign(ctx, { modelRegistry: registry });
	});
	const psv = mountPsv(s.pi, "selfhost/qwen");
	await s.emit("session_start");
	const turn = async (messages: Message[]): Promise<ToolCall[]> => {
		let result: AssistantMessage | undefined;
		const wrapped = { ...model, api: "humanclaw-psv", provider: "humanclaw-psv" } as Model<Api>;
		for await (const ev of psv.streamSimple(wrapped, normalizeContext({ messages }), {})) {
			if (ev.type === "error") assert.fail(ev.error.errorMessage);
			if (ev.type === "done") result = ev.message;
		}
		assert.ok(result);
		return result.content.filter((c): c is ToolCall => c.type === "toolCall");
	};
	const observation = (call: ToolCall, step: number): Message => ({
		role: "toolResult",
		toolCallId: call.id,
		toolName: call.name,
		content: [{ type: "image", data: "AAAA", mimeType: "image/png" }],
		details: { instruction: "Find the couch.", step },
		isError: false,
		timestamp: 0,
	});
	const [look] = await turn([]);
	const [act] = await turn([observation(look, 0)]);
	assert.deepEqual(act.arguments, { unit: "TURN_LEFT", param: 30 });
	assert.deepEqual(requests, [{ response_format: { type: "json_object" } }, undefined]);
	assert.deepEqual(contexts[0], contexts[1], "format negotiation must keep the same prompt and image");
	await turn([observation(act, 1)]);
	assert.equal(requests.length, 3);
	assert.equal(requests[2], undefined, "later steps must not repeat the rejected format");
	assert.equal(s.entries.filter((e) => e.type === "humanclaw_response_format").length, 1);
	// Every request is the paper's: temperature 0, max_tokens 4096 by default and the OpenAI SDK's two retries.
	for (const o of options) {
		assert.equal(o.temperature, 0);
		assert.equal(o.maxTokens, 4096);
		assert.equal(o.maxRetries, 2, "the paper's adapter is the SDK at max_retries=2");
	}
});
