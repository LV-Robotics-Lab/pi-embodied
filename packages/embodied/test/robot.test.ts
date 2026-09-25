import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, realpathSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { defineRobot, RESULT_ENTRY, type RobotSpec, STATUS_EVENT, TASK_ENTRY } from "../src/robot.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi that runs handlers in registration order, like pi's runner, and records what the base does. */
function fakePi(flagValues: Record<string, unknown> = {}, branch: unknown[] = []) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const commands = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	const events: { channel: string; data: any }[] = [];
	const stderr: string[] = [];
	let active: string[] = ["everything"];
	let shutdown = false;
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: (name: string, c: any) => commands.set(name, c),
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: { emit: (channel: string, data: any) => events.push({ channel, data }), on: () => () => {} },
	} as unknown as ExtensionAPI;
	const dir = realpathSync(mkdtempSync(join(tmpdir(), "robot-")));
	const ctx = {
		hasUI: false,
		cwd: dir,
		ui: { notify: () => {} },
		shutdown: () => {
			shutdown = true;
		},
		sessionManager: { getBranch: () => branch, getSessionDir: () => dir },
	};
	/** Emit like pi: every handler in order; a tool_call block or an input "handled" stops the chain. */
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) {
			const r = await fn({ type: name, ...event }, ctx);
			if (r !== undefined) result = r;
			if (r && ((name === "tool_call" && r.block) || (name === "input" && r.action === "handled"))) return r;
		}
		return result;
	}
	const log = console.error;
	console.error = (line: string) => stderr.push(line);
	const restore = () => {
		console.error = log;
		process.exitCode = undefined;
	};
	return {
		pi,
		emit,
		tools,
		commands,
		entries,
		events,
		stderr,
		dir,
		restore,
		active: () => active,
		stopped: () => shutdown,
	};
}

const finish: RobotSpec["finish"] = {
	description: "finish",
	parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
	result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
};

function toy(pi: ExtensionAPI, start: RobotSpec["start"], extra: Partial<RobotSpec> = {}) {
	pi.registerFlag("suite", { type: "string", default: "flag-suite" });
	pi.registerFlag("seed", { type: "string", default: "0" });
	let steps = 0;
	const robot = defineRobot(pi, {
		name: "toy",
		task: ["suite", "seed"],
		keepImages: 1,
		start,
		result: () => ({ steps }),
		status: () => ({ step: steps }),
		finish,
		...extra,
	});
	robot.tool("move", "move", Type.Object({ n: Type.Number() }), async (p) => {
		steps += p.n;
		return { content: [{ type: "text", text: "moved" }], details: {} };
	});
	return robot;
}

const assistant = (...calls: string[]) => ({
	message: { role: "assistant", stopReason: "toolUse", content: calls.map((name) => ({ type: "toolCall", name })) },
});

test("a failed start fails closed and writes one env_error result", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	toy(f.pi, async () => {
		throw new Error("no simulator");
	});
	await f.emit("session_start");
	assert.deepEqual(f.active(), []);
	assert.equal(process.exitCode, 1);
	assert.ok(f.stopped(), "a run without a UI shuts down");
	assert.deepEqual(f.stderr, ["[toy] unavailable: no simulator"]);
	assert.deepEqual(await f.emit("input", { text: "go" }), { action: "handled" });
	assert.equal((await f.emit("tool_call", { toolName: "move" }))?.terminate, true);
	await f.emit("session_shutdown");
	await f.emit("session_shutdown");
	const results = f.entries.filter((e) => e.type === RESULT_ENTRY);
	assert.deepEqual(results, [
		{
			type: RESULT_ENTRY,
			data: { robot: "toy", suite: "flag-suite", seed: "0", env_error: true, error: "no simulator" },
		},
	]);
	assert.equal(f.stderr.length, 2);
	assert.equal(f.events.at(-1)?.channel, STATUS_EVENT);
	assert.equal(f.events.at(-1)?.data.ready, false);
});

test("the task entry overrides the flags and is resolved before the robot starts", async (t) => {
	const branch = [
		{ type: "custom", customType: TASK_ENTRY, data: { robot: "toy", suite: "entry-suite", seed: "7" } },
		{ type: "custom", customType: TASK_ENTRY, data: { robot: "other", suite: "no", seed: "9" } },
	];
	const f = fakePi({}, branch);
	t.after(f.restore);
	let seen: Record<string, string> = {};
	const robot = toy(f.pi, async () => {
		seen = { ...robot.task };
		return ["move", "finish"];
	});
	await f.emit("session_start");
	assert.deepEqual(seen, { suite: "entry-suite", seed: "7" });
	assert.deepEqual(f.active(), ["move", "finish"]);
	assert.equal(f.entries.filter((e) => e.type === TASK_ENTRY).length, 0, "an existing entry is not recorded again");

	const g = fakePi({ suite: "cli" });
	t.after(g.restore);
	toy(g.pi, async () => ["move"]);
	await g.emit("session_start");
	assert.deepEqual(g.entries[0], { type: TASK_ENTRY, data: { robot: "toy", suite: "cli", seed: "0" } });
});

test("memory's session_start sees the resolved task", async (t) => {
	const f = fakePi({ "memory-profile": "local" });
	t.after(f.restore);
	const memoryDir = join(f.dir, "memory", "toy");
	mkdirSync(memoryDir, { recursive: true });
	writeFileSync(join(memoryDir, "MEMORY.md"), "# memory\n");
	const robot = toy(f.pi, async () => ["move", "finish"], {
		memory: {
			home: () => join(f.dir, "memory"),
			cell: () => ({ tag: `cell_${robot.task.suite}`, reference: "" }),
			primitives: ["move"],
		},
	});
	await f.emit("session_start");
	assert.deepEqual(f.active(), ["move", "finish", "read", "ls", "grep", "find", "write"]);
	assert.equal(await f.emit("tool_call", { toolName: "write", input: { path: "cell_flag-suite.json" } }), undefined);
	assert.ok((await f.emit("tool_call", { toolName: "read", input: { path: "/etc/passwd" } }))?.block);
});

test("finish ends the episode, terminates its batch, and yields exactly one result", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	toy(f.pi, async () => ["move", "finish"]);
	await f.emit("session_start");
	await f.emit("agent_start");
	const move = f.tools.get("move");
	await f.emit("message_end", assistant("move"));
	assert.equal((await move.execute("1", { n: 1 }, undefined, undefined, {})).terminate, false);
	await f.emit("message_end", assistant("move", "finish"));
	assert.equal((await move.execute("2", { n: 2 }, undefined, undefined, {})).terminate, true);
	assert.equal(await f.emit("tool_call", { toolName: "finish" }), undefined);
	const done = await f.tools.get("finish").execute("3", { status: "success", summary: "ok" });
	assert.equal(done.terminate, true);
	assert.match((await f.emit("tool_call", { toolName: "move" }))?.reason, /episode is finished/);
	assert.equal(await f.emit("tool_call", { toolName: "finish" }), undefined, "finish may still run");
	await f.emit("agent_end");
	await f.emit("agent_end");
	await f.emit("session_shutdown");
	const results = f.entries.filter((e) => e.type === RESULT_ENTRY).map((e) => e.data);
	assert.deepEqual(results, [
		{
			robot: "toy",
			steps: 3,
			claimed: "success",
			summary: "ok",
			turns: 0,
			planner_budget_exhausted: null,
			planner_error: null,
			env_error: false,
		},
	]);
	assert.deepEqual(f.stderr, [`[toy] ${JSON.stringify(results[0])}`]);
	assert.equal(f.events.at(-1)?.data.claimed, "success");
	assert.equal(f.events.at(-1)?.data.step, 3);
});

test("a spent turn budget ends the episode; an unended episode reports at shutdown", async (t) => {
	const f = fakePi({ "max-turns": "1" });
	t.after(f.restore);
	toy(f.pi, async () => ["move", "finish"]);
	await f.emit("session_start");
	await f.emit("agent_start");
	await f.emit("before_agent_start");
	assert.equal(await f.emit("tool_call", { toolName: "move" }), undefined);
	await f.emit("turn_end");
	const blocked = await f.emit("tool_call", { toolName: "move" });
	assert.deepEqual(blocked, {
		block: true,
		reason: "Planner turn or time budget exhausted; the episode is over.",
		terminate: true,
	});
	await f.emit("message_end", {
		message: { role: "assistant", stopReason: "error", errorMessage: "503", content: [] },
	});
	await f.emit("agent_end");
	const [result] = f.entries.filter((e) => e.type === RESULT_ENTRY).map((e) => e.data);
	assert.equal(result.planner_budget_exhausted, "turns");
	assert.equal(result.planner_error, "503");

	const g = fakePi();
	t.after(g.restore);
	toy(g.pi, async () => ["move"]);
	await g.emit("session_start");
	await g.emit("agent_start");
	await g.emit("agent_end");
	assert.equal(g.entries.filter((e) => e.type === RESULT_ENTRY).length, 0, "not ended: no result yet");
	await g.emit("session_shutdown");
	assert.equal(g.entries.filter((e) => e.type === RESULT_ENTRY).length, 1);
});
