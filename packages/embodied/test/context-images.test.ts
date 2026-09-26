import assert from "node:assert/strict";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { defineRobot, RESULT_ENTRY } from "../src/robot.ts";

type Handler = (event: any, ctx: any) => unknown;

function stubPi(values: Record<string, unknown> = {}) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const entries: { type: string; data: any }[] = [];
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
	return { flags, entries, emit };
}

const image = (id: string) => ({ type: "image", data: id, mimeType: "image/png" });
/** A tool result with the main view then the wrist view of observation `n`. */
const look = (n: number) => ({
	role: "toolResult",
	toolName: "look",
	content: [{ type: "text", text: `step ${n}` }, image(`main${n}`), image(`wrist${n}`)],
});
const kept = (messages: any[]) =>
	messages.flatMap((m) =>
		(m.role === "toolResult" ? m.content : []).filter((p: any) => p.type === "image").map((p: any) => p.data),
	);

const messages = [{ role: "user", content: "Solve the task." }, look(0), look(1), look(2)];

test("--keep-images keeps the latest frames and stubs the rest", async () => {
	const f = stubPi();
	const out = (await f.emit("context", { messages })) as { messages: any[] };
	assert.deepEqual(kept(out.messages), ["main2", "wrist2"]);
	assert.equal(out.messages[1].content[1].text, "[older camera frame omitted]");
});

test("--anchor-image also keeps the episode's first main frame, and the result records it", async () => {
	const f = stubPi({ "anchor-image": true });
	const out = (await f.emit("context", { messages })) as { messages: any[] };
	assert.deepEqual(kept(out.messages), ["main0", "main2", "wrist2"], "the anchor is extra to --keep-images");
	assert.equal(out.messages[1].content[2].text, "[older camera frame omitted]", "only the first frame of it");
	// Nothing to prune yet: the context is left alone.
	assert.equal(await f.emit("context", { messages: messages.slice(0, 2) }), undefined);
	await f.emit("session_start");
	await f.emit("agent_start");
	await f.emit("session_shutdown");
	assert.equal(f.entries.find((e) => e.type === RESULT_ENTRY)?.data.anchor_image, true);
});

test("without --anchor-image the result does not mention it", async () => {
	const f = stubPi();
	await f.emit("session_start");
	await f.emit("agent_start");
	await f.emit("session_shutdown");
	const result = f.entries.find((e) => e.type === RESULT_ENTRY);
	assert.ok(result);
	assert.equal("anchor_image" in result.data, false);
});
