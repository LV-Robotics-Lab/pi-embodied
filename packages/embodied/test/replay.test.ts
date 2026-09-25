import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import replay, { loadRecording, REPLAY_ENTRY, taskDiff } from "../src/replay/index.ts";

type Handler = (event: any, ctx: any) => unknown;

const assistant = (id: string, parentId: string, content: unknown[], stopReason = "toolUse") => ({
	type: "message",
	id,
	parentId,
	message: { role: "assistant", content, stopReason },
});
const call = (name: string, args: Record<string, unknown>) => ({
	type: "toolCall",
	id: `rec_${name}`,
	name,
	arguments: args,
});
const result = (id: string, parentId: string) => ({
	type: "message",
	id,
	parentId,
	message: { role: "toolResult", toolCallId: "x", toolName: "x", content: [], isError: false },
});

/** A recorded session: a task, an abandoned branch, an errored reply, a multi-call turn, and (optionally) finish. */
function recordSession(finish = true) {
	const dir = mkdtempSync(join(tmpdir(), "replay-"));
	const lines: unknown[] = [
		{ type: "session", version: 3, id: "s", cwd: dir },
		{
			type: "custom",
			id: "t",
			parentId: null,
			customType: "robot_task",
			data: { robot: "arm", task: "3", seed: "1" },
		},
		{ type: "message", id: "u", parentId: "t", message: { role: "user", content: "Solve." } },
		assistant("a1", "u", [{ type: "thinking", thinking: "look first" }, call("view", {})]),
		result("r1", "a1"),
		assistant("dead", "r1", [call("move", { xyz: [9, 9, 9] })]), // abandoned: the leaf's branch skips it
		assistant("err", "r1", [call("move", { xyz: [8, 8, 8] })], "error"),
		assistant("a2", "err", [
			{ type: "text", text: "two at once" },
			call("move", { xyz: [1, 2, 3] }),
			call("grip", { v: 1 }),
		]),
		result("r2", "a2"),
		assistant("a3", "r2", [call("write", { path: "/tmp/x", content: "y" })]),
		result("r3", "a3"),
		assistant("a4", "r3", finish ? [call("finish", { status: "success", summary: "ok" })] : [call("view", {})]),
	];
	writeFileSync(join(dir, "2026-01-01_s.jsonl"), `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
	writeFileSync(join(dir, "arm_recipe.jsonl"), `${JSON.stringify({ action: "view" })}\n`);
	return dir;
}

function fakePi(flags: Record<string, unknown>, task: Record<string, string>) {
	const handlers = new Map<string, Handler[]>();
	const entries: { type: string; data: any }[] = [];
	const stderr: string[] = [];
	let provider: any;
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: () => {},
		getFlag: (name: string) => flags[name],
		registerProvider: (p: any) => {
			provider = p;
		},
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
	} as unknown as ExtensionAPI;
	const ctx = {
		hasUI: false,
		cwd: "/",
		model: { provider: "replay", id: "session" },
		ui: { notify: () => {} },
		sessionManager: { getBranch: () => [{ type: "custom", customType: "robot_task", data: task }] },
	};
	const emit = async (name: string) => {
		for (const fn of handlers.get(name) ?? []) await fn({ type: name }, ctx);
	};
	const log = console.error;
	console.error = (line: string) => stderr.push(line);
	replay(pi);
	const model = provider.getModels()[0];
	/** One model turn over `messages`, as pi's agent loop asks for it. */
	const turn = (messages: unknown[] = [], signal?: AbortSignal) =>
		provider.streamSimple(model, { messages }, { signal }).result();
	return {
		emit,
		turn,
		entries,
		stderr,
		model,
		restore: () => {
			console.error = log;
		},
	};
}

const calls = (m: any) => m.content.filter((c: any) => c.type === "toolCall");
const text = (m: any) =>
	m.content
		.filter((c: any) => c.type === "text")
		.map((c: any) => c.text)
		.join("\n");
const results = (m: any, isError = false) =>
	calls(m).map((c: any) => ({
		role: "toolResult",
		toolCallId: c.id,
		toolName: c.name,
		content: [{ type: "text", text: isError ? `${c.name} blew up` : "{}" }],
		isError,
	}));

test("loadRecording follows the leaf's branch, drops errored replies, and picks the session file in a dir", () => {
	const rec = loadRecording(recordSession());
	assert.match(rec.path, /2026-01-01_s\.jsonl$/);
	assert.deepEqual(rec.task, { robot: "arm", task: "3", seed: "1" });
	assert.deepEqual(
		rec.turns.map((t) => t.calls.map((c) => c.name)),
		[["view"], ["move", "grip"], ["write"], ["finish"]],
	);
	assert.equal(rec.turns[0].thinking, "look first");
	assert.equal(rec.finished, true);
	assert.deepEqual(taskDiff({ robot: "arm", seed: "1" }, { robot: "arm", seed: 2 }), ["seed: 1 -> 2"]);
});

test("turns replay in order with recorded arguments; multi-call turns together; errors noted, writes skipped", async () => {
	const p = fakePi({ replay: recordSession() }, { robot: "arm", task: "3", seed: "1" });
	try {
		await p.emit("session_start");
		await p.emit("before_agent_start");
		assert.equal(p.model.id, "session");
		const t1 = await p.turn();
		assert.equal(t1.stopReason, "toolUse");
		assert.deepEqual(t1.usage.totalTokens, 0);
		assert.deepEqual(
			calls(t1).map((c: any) => c.name),
			["view"],
		);
		assert.deepEqual(p.entries[0].type, REPLAY_ENTRY);
		const history = [t1, ...results(t1)];
		const t2 = await p.turn(history);
		assert.deepEqual(
			calls(t2).map((c: any) => [c.name, c.arguments]),
			[
				["move", { xyz: [1, 2, 3] }],
				["grip", { v: 1 }],
			],
		);
		assert.match(text(t2), /two at once/);
		// The move errors; the replay notes it, skips the recorded write, and goes on to finish.
		history.push(t2, ...results(t2, true));
		const t3 = await p.turn(history);
		assert.deepEqual(
			calls(t3).map((c: any) => [c.name, c.arguments]),
			[["finish", { status: "success", summary: "ok" }]],
		);
		assert.match(text(t3), /move failed: move blew up; continuing/);
		assert.match(text(t3), /skipped write/);
		assert.ok(p.stderr.some((l) => /\[replay\] move failed/.test(l)));
		assert.ok(!p.stderr.some((l) => /differs/.test(l)));
		// pi stops after finish; if asked again, the recording is spent.
		history.push(t3, ...results(t3));
		const t4 = await p.turn(history);
		assert.equal(t4.stopReason, "stop");
		assert.equal(calls(t4).length, 0);
		await p.emit("agent_end");
		assert.match(text(await p.turn()), /already ran/);
	} finally {
		p.restore();
	}
});

test("a recording without finish ends with a text turn", async () => {
	const p = fakePi({ replay: recordSession(false) }, { robot: "arm", task: "3", seed: "1" });
	try {
		await p.emit("session_start");
		let m = await p.turn();
		const history: unknown[] = [];
		for (let i = 0; i < 5 && m.stopReason === "toolUse"; i++) {
			history.push(m, ...results(m));
			m = await p.turn(history);
		}
		assert.equal(m.stopReason, "stop");
		assert.match(text(m), /no finish/);
	} finally {
		p.restore();
	}
});

test("an abort ends the replay, and a task mismatch is warned on stderr", async () => {
	const p = fakePi({ replay: recordSession() }, { robot: "arm", task: "3", seed: "7" });
	try {
		await p.emit("session_start");
		await p.emit("before_agent_start");
		assert.ok(p.stderr.some((l) => /differs from the recorded robot_task: seed: 1 -> 7/.test(l)));
		await p.turn();
		const ac = new AbortController();
		ac.abort();
		const aborted = await p.turn([], ac.signal);
		assert.equal(aborted.stopReason, "aborted");
		const after = await p.turn();
		assert.equal(calls(after).length, 0);
		assert.match(text(after), /already ran/);
		// A new session replays from the start.
		await p.emit("session_start");
		assert.deepEqual(
			calls(await p.turn()).map((c: any) => c.name),
			["view"],
		);
	} finally {
		p.restore();
	}
});

test("without --replay the model says so and calls nothing", async () => {
	const p = fakePi({}, { robot: "arm" });
	try {
		await p.emit("session_start");
		const m = await p.turn();
		assert.equal(m.stopReason, "stop");
		assert.match(text(m), /no --replay/);
	} finally {
		p.restore();
	}
});
