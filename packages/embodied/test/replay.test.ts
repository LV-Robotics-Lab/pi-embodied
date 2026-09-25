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
let callIds = 0;
const call = (name: string, args: Record<string, unknown>) => ({
	type: "toolCall",
	id: `rec_${name}_${++callIds}`,
	name,
	arguments: args,
});
/** A turn and one result entry per call, chained. */
function turnWithResults(id: string, parentId: string, content: unknown[], errors: Record<string, string> = {}) {
	const a = assistant(id, parentId, content);
	const calls = content.filter((c: any) => c.type === "toolCall") as { id: string; name: string }[];
	const rs = calls.map((c, i) => ({
		type: "message",
		id: `${id}_r${i}`,
		parentId: i ? `${id}_r${i - 1}` : id,
		message: {
			role: "toolResult",
			toolCallId: c.id,
			toolName: c.name,
			content: [{ type: "text", text: errors[c.name] ?? "{}" }],
			isError: c.name in errors,
		},
	}));
	return { entries: [a, ...rs], leaf: rs.at(-1)?.id ?? id };
}
const STALE =
	"The operator took over and executed 2 step(s): FORWARD GRASP. This move call was decided on an older observation and was not executed; decide again from the current observation.";

/**
 * A recorded session: a task, an abandoned branch, an errored reply, a multi-call turn, a move the
 * operator's takeover dropped as stale, a turn refused by the robot base next to a grip that
 * really failed, a write, and (optionally) finish.
 */
function recordSession(finish = true) {
	const dir = mkdtempSync(join(tmpdir(), "replay-"));
	const t1 = turnWithResults("a1", "u", [{ type: "thinking", thinking: "look first" }, call("view", {})]);
	const t2 = turnWithResults("a2", "err", [
		{ type: "text", text: "two at once" },
		call("move", { xyz: [1, 2, 3] }),
		call("grip", { v: 1 }),
	]);
	const t3 = turnWithResults("a3", t2.leaf, [call("move", { xyz: [5, 5, 5] })], { move: STALE });
	const t4 = turnWithResults("a4", t3.leaf, [call("move", { xyz: [6, 6, 6] }), call("grip", { v: 0 })], {
		move: "Planner turns budget exhausted; the episode is over.",
		grip: "gripper jammed",
	});
	const t5 = turnWithResults("a5", t4.leaf, [call("write", { path: "/tmp/x", content: "y" })]);
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
		...t1.entries,
		assistant("dead", t1.leaf, [call("move", { xyz: [9, 9, 9] })]), // abandoned: the leaf's branch skips it
		assistant("err", t1.leaf, [call("move", { xyz: [8, 8, 8] })], "error"),
		...t2.entries,
		...t3.entries,
		...t4.entries,
		...t5.entries,
		assistant("a6", t5.leaf, finish ? [call("finish", { status: "success", summary: "ok" })] : [call("view", {})]),
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
	/** Run `name`'s handlers; the first defined handler result. */
	const emit = async (name: string, event: Record<string, unknown> = {}) => {
		let out: unknown;
		for (const fn of handlers.get(name) ?? []) out ??= await fn({ type: name, ...event }, ctx);
		return out as { block?: boolean; reason?: string } | undefined;
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
/** This run's results of turn `m`'s calls; `errors` maps a call name to its error text. */
const results = (m: any, errors: Record<string, string> = {}) =>
	calls(m).map((c: any) => ({
		role: "toolResult",
		toolCallId: c.id,
		toolName: c.name,
		content: [{ type: "text", text: errors[c.name] ?? "{}" }],
		isError: c.name in errors,
	}));

test("loadRecording follows the leaf's branch, drops errored replies and calls that never ran, and picks the session file in a dir", () => {
	const rec = loadRecording(recordSession());
	assert.match(rec.path, /2026-01-01_s\.jsonl$/);
	assert.deepEqual(rec.task, { robot: "arm", task: "3", seed: "1" });
	assert.deepEqual(
		rec.turns.map((t) => t.calls.map((c) => c.name)),
		[["view"], [], ["move", "grip"], [], ["grip"], ["write"], ["finish"]],
	);
	assert.deepEqual(
		rec.turns.map((t) => t.blocked.map((b) => b.name)),
		[[], ["move"], [], ["move"], ["move"], [], []],
	);
	assert.match(rec.turns[1].blocked[0].reason, /stopReason "error"/);
	assert.equal(rec.turns[4].calls[0].error, "gripper jammed");
	assert.equal(rec.turns[2].calls[0].error, undefined);
	assert.equal(rec.turns[0].thinking, "look first");
	assert.equal(rec.finished, true);
	assert.deepEqual(taskDiff({ robot: "arm", seed: "1" }, { robot: "arm", seed: 2 }), ["seed: 1 -> 2"]);
});

async function start(p: ReturnType<typeof fakePi>) {
	await p.emit("session_start");
	await p.emit("before_agent_start");
	const t1 = await p.turn();
	const history = [t1, ...results(t1)];
	const t2 = await p.turn(history);
	return { t1, t2, history };
}

test("turns replay in order with recorded arguments; blocked and stale calls and writes skipped; recorded errors continue", async () => {
	const p = fakePi({ replay: recordSession() }, { robot: "arm", task: "3", seed: "1" });
	try {
		const { t1, t2, history } = await start(p);
		assert.equal(p.model.id, "session");
		assert.equal(t1.stopReason, "toolUse");
		assert.deepEqual(t1.usage.totalTokens, 0);
		assert.deepEqual(
			calls(t1).map((c: any) => c.name),
			["view"],
		);
		assert.deepEqual(p.entries[0].type, REPLAY_ENTRY);
		assert.deepEqual(
			calls(t2).map((c: any) => [c.name, c.arguments]),
			[
				["move", { xyz: [1, 2, 3] }],
				["grip", { v: 1 }],
			],
		);
		assert.match(text(t2), /two at once/);
		// The stale-dropped move and the budget-refused move never ran in the recording: not sent.
		history.push(t2, ...results(t2));
		const t3 = await p.turn(history);
		assert.deepEqual(
			calls(t3).map((c: any) => [c.name, c.arguments]),
			[["grip", { v: 0 }]],
		);
		assert.match(text(t3), /skipped move \(it never ran in the recording: The operator took over/);
		assert.match(text(t3), /skipped move \(it never ran in the recording: Planner turns budget exhausted/);
		// The grip fails again, as it did in the recording: noted, and the replay goes on.
		history.push(t3, ...results(t3, { grip: "gripper jammed" }));
		const t4 = await p.turn(history);
		assert.deepEqual(
			calls(t4).map((c: any) => [c.name, c.arguments]),
			[["finish", { status: "success", summary: "ok" }]],
		);
		assert.match(text(t4), /grip failed, as in the recording: gripper jammed; continuing/);
		assert.match(text(t4), /skipped write/);
		assert.ok(p.stderr.some((l) => /\[replay\] grip failed/.test(l)));
		assert.ok(!p.stderr.some((l) => /differs/.test(l)));
		// pi stops after finish; if asked again, the recording is spent.
		history.push(t4, ...results(t4));
		const t5 = await p.turn(history);
		assert.equal(t5.stopReason, "stop");
		assert.equal(calls(t5).length, 0);
		await p.emit("agent_end");
		assert.match(text(await p.turn()), /already ran/);
	} finally {
		p.restore();
	}
});

for (const error of ["robot error: joint limit", "http://127.0.0.1:1/call: connect ECONNREFUSED"]) {
	test(`a call that fails where the recording's succeeded stops the replay (${error})`, async () => {
		const p = fakePi({ replay: recordSession() }, { robot: "arm", task: "3", seed: "1" });
		try {
			const { t2, history } = await start(p);
			history.push(t2, ...results(t2, { move: error }));
			const t3 = await p.turn(history);
			assert.equal(t3.stopReason, "stop");
			assert.equal(calls(t3).length, 0);
			assert.match(text(t3), /move failed: .*; it succeeded in the recording/);
			assert.match(text(t3), /Replay stopped: .*3 of 7 turns replayed/);
			// No further recorded motion, however often pi asks.
			assert.equal(calls(await p.turn([...history, t3])).length, 0);
		} finally {
			p.restore();
		}
	});
}

test("a recording without finish ends with a text turn", async () => {
	const p = fakePi({ replay: recordSession(false) }, { robot: "arm", task: "3", seed: "1" });
	try {
		await p.emit("session_start");
		let m = await p.turn();
		const history: unknown[] = [];
		for (let i = 0; i < 5 && m.stopReason === "toolUse"; i++) {
			// The recording's grip {v: 0} jammed; this run's does too.
			const jammed = calls(m).some((c: any) => c.name === "grip" && c.arguments.v === 0);
			history.push(m, ...results(m, jammed ? { grip: "gripper jammed" } : {}));
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

/** A session of `turns` (content, stopReason, recorded errors by call name), each followed by its results. */
function sessionOf(turns: { content: unknown[]; stopReason?: string; errors?: Record<string, string> }[]) {
	const dir = mkdtempSync(join(tmpdir(), "replay-"));
	const lines: unknown[] = [
		{ type: "session", version: 3, id: "s", cwd: dir },
		{ type: "message", id: "u", parentId: null, message: { role: "user", content: "Solve." } },
	];
	let leaf = "u";
	turns.forEach((t, i) => {
		const r = turnWithResults(`x${i}`, leaf, t.content, t.errors);
		(r.entries[0] as any).message.stopReason = t.stopReason ?? "toolUse";
		lines.push(...r.entries);
		leaf = r.leaf;
	});
	writeFileSync(join(dir, "s.jsonl"), `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
	return dir;
}

/** Execute turn `m`'s calls as pi does for a sequential batch: tool_call hooks, then the result; `errors` fail calls by name. */
async function runBatch(p: ReturnType<typeof fakePi>, m: any, errors: Record<string, string> = {}) {
	const out: any[] = [];
	const sent: string[] = [];
	for (const c of calls(m)) {
		const blocked = await p.emit("tool_call", { toolCallId: c.id, toolName: c.name, input: c.arguments });
		const failed = blocked?.block ? (blocked.reason ?? "blocked") : errors[c.name];
		if (!blocked?.block) sent.push(c.name);
		const content = [{ type: "text", text: failed ?? "{}" }];
		if (!blocked?.block)
			await p.emit("tool_result", { toolCallId: c.id, input: c.arguments, content, isError: failed !== undefined });
		out.push({ role: "toolResult", toolCallId: c.id, toolName: c.name, content, isError: failed !== undefined });
	}
	return { results: out, sent };
}

test("a reply cut at the output token limit (or otherwise abnormal) has its calls skipped with a note", async () => {
	const dir = sessionOf([
		{ content: [call("view", {})] },
		{
			content: [call("move", { xyz: [1, 2] })],
			stopReason: "length",
			errors: {
				move: 'Tool call "move" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.',
			},
		},
		{ content: [call("grip", { v: 1 })], stopReason: "pause_turn" },
		{ content: [call("move", { xyz: [1, 2, 3] })] },
	]);
	const rec = loadRecording(dir);
	assert.deepEqual(
		rec.turns.map((t) => [t.calls.map((c) => c.name), t.blocked.map((b) => b.name)]),
		[
			[["view"], []],
			[[], ["move"]],
			[[], ["grip"]],
			[["move"], []],
		],
	);
	assert.match(rec.turns[1].blocked[0].reason, /stopReason "length", so its arguments may be truncated/);
	const p = fakePi({ replay: dir }, {});
	try {
		await p.emit("session_start");
		const t1 = await p.turn();
		const t2 = await p.turn([t1, ...(await runBatch(p, t1)).results]);
		// The truncated move is never sent; the next real turn is.
		assert.deepEqual(
			calls(t2).map((c: any) => [c.name, c.arguments]),
			[["move", { xyz: [1, 2, 3] }]],
		);
		assert.match(text(t2), /skipped move \(it never ran in the recording: the reply ended with stopReason "length"/);
		assert.match(
			text(t2),
			/skipped grip \(it never ran in the recording: the reply ended with stopReason "pause_turn"/,
		);
	} finally {
		p.restore();
	}
});

test("a failure inside a multi-call turn stops the replay at that call: the rest of the turn is not sent", async () => {
	const dir = sessionOf([
		{ content: [call("move", { xyz: [1, 0, 0] }), call("move", { xyz: [2, 0, 0] }), call("grasp", {})] },
		{ content: [call("finish", { status: "success" })] },
	]);
	const p = fakePi({ replay: dir }, {});
	try {
		await p.emit("session_start");
		const t1 = await p.turn();
		assert.equal(calls(t1).length, 3);
		const { results: r1, sent } = await runBatch(p, t1, { move: "robot error: IK failed" });
		assert.deepEqual(sent, ["move"]);
		assert.match(
			r1[1].content[0].text,
			/replay stopped \(move failed: robot error: IK failed; it succeeded in the recording\)/,
		);
		const t2 = await p.turn([t1, ...r1]);
		assert.equal(calls(t2).length, 0);
		assert.match(text(t2), /move not sent/);
		assert.match(text(t2), /grasp not sent/);
		assert.match(text(t2), /Replay stopped/);
		assert.equal(calls(await p.turn([t1, ...r1, t2])).length, 0);
	} finally {
		p.restore();
	}
});

test("a call that failed in the recording but succeeds now stops the replay", async () => {
	const dir = sessionOf([
		{ content: [call("grip", { v: 1 }), call("move", { xyz: [1, 0, 0] })], errors: { grip: "gripper jammed" } },
		{ content: [call("move", { xyz: [2, 0, 0] })] },
	]);
	const p = fakePi({ replay: dir }, {});
	try {
		await p.emit("session_start");
		const t1 = await p.turn();
		const { results: r1, sent } = await runBatch(p, t1);
		assert.deepEqual(sent, ["grip"]);
		const t2 = await p.turn([t1, ...r1]);
		assert.equal(calls(t2).length, 0);
		assert.match(text(t2), /grip succeeded; it failed in the recording \(gripper jammed\)/);
		assert.match(text(t2), /Replay stopped/);
	} finally {
		p.restore();
	}
});
