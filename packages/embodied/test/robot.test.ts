import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, realpathSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { RpcUnavailable } from "../src/infra/rpc.ts";
import { UNITS_EVENT, type UnitsHandle, type UnitsSpec } from "../src/modes/units/index.ts";
import { CLOSED_LOOP } from "../src/planner/closed-loop.ts";
import {
	defineRobot,
	leakedToolCall,
	RESULT_ENTRY,
	type RobotSpec,
	STATUS_EVENT,
	SYSTEM_PROMPT_ENTRY,
	serverError,
	TASK_ENTRY,
	toolSections,
} from "../src/robot.ts";

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
	const providers: any[] = [];
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
		registerProvider: (p: any) => providers.push(p),
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
		abort: () => {},
		sessionManager: { getBranch: () => branch, getSessionDir: () => dir },
		// Like pi: the prompt the last before_agent_start handler forced (see emit).
		getSystemPrompt: () => forced ?? "pi default prompt",
	};
	let forced: string | undefined;
	/** Emit like pi: every handler in order; a tool_call block or an input "handled" stops the chain. */
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) {
			const r: any = await fn({ type: name, ...event }, ctx);
			if (r !== undefined) result = r;
			if (name === "before_agent_start" && r?.systemPrompt !== undefined) forced = r.systemPrompt;
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
		providers,
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
	// Every robot gets /robot-task and /robot-check from the base.
	assert.deepEqual([...g.commands.keys()].sort(), ["robot-check", "robot-task"]);
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

test("pure units mode keeps memory: the file tools stay and memory's section follows the units prompt", async (t) => {
	const f = fakePi({ "memory-profile": "local", units: "true", "units-plugins": "" });
	t.after(f.restore);
	const memoryDir = join(f.dir, "memory", "toy");
	mkdirSync(memoryDir, { recursive: true });
	writeFileSync(join(memoryDir, "MEMORY.md"), "# memory\n");
	const robot = toy(f.pi, async () => ["move", "finish"], {
		memory: {
			home: () => join(f.dir, "memory"),
			cell: () => ({ tag: `cell_${robot.task.suite}`, reference: "cell_ref" }),
			primitives: ["move"],
		},
		units: {
			vectors: {
				MV_FWD: [1, 0, 0],
				MV_BACK: [-1, 0, 0],
				MV_LEFT: [0, -1, 0],
				MV_RIGHT: [0, 1, 0],
				MV_UP: [0, 0, 1],
				MV_DOWN: [0, 0, -1],
			},
			stepM: 0.02,
			apply: async () => ({ content: [], details: {} }),
		},
	});
	await f.emit("session_start");
	assert.deepEqual(f.active(), ["act", "read", "ls", "grep", "find", "write", "finish"]);
	const prompt = (await f.emit("before_agent_start", { systemPrompt: "" }))?.systemPrompt as string;
	assert.match(prompt, /ACTION UNITS/);
	assert.match(prompt, /# Memory\nCell `cell_flag-suite`/);
	assert.ok(prompt.includes(`${memoryDir}/MEMORY.md`));
	assert.match(prompt, /task_only\/cell_ref_recipe\.jsonl/);
	assert.match(prompt, /when your tools are not the ones it names \(action units, code\), carry its stages over/);
	assert.equal(prompt.split("Reading memory is a required step").length, 2, "one memory section");
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
	assert.deepEqual(f.stderr, [`[toy] ${JSON.stringify(results[0])}`]);
	// The context version (../src/context-version.ts) is checked in runtime-mech.test.ts.
	assert.equal(typeof results[0]?.context_version?.git_commit, "string");
	assert.deepEqual(
		results.map(({ context_version: _, ...r }) => r),
		[
			{
				robot: "toy",
				steps: 3,
				claimed: "success",
				summary: "ok",
				tool_calls: 0,
				planner_tokens: { input: 0, output: 0 },
				max_tool_calls: 0,
				max_tokens: 0,
				turns: 0,
				planner_budget_exhausted: null,
				cost_usd: 0,
				planner_error: null,
				env_error: false,
			},
		],
	);
	assert.equal(f.events.at(-1)?.data.claimed, "success");
	assert.equal(f.events.at(-1)?.data.step, 3);
});

test("the time-limit abort ends the episode as a failure, not a planner error", async (t) => {
	const f = fakePi({ "time-limit": "0.01" });
	t.after(f.restore);
	toy(f.pi, async () => ["move", "finish"]);
	await f.emit("session_start");
	await f.emit("agent_start");
	await f.emit("before_agent_start");
	await new Promise((r) => setTimeout(r, 50));
	// The reply the deadline cut off: some providers report the abort as an error.
	await f.emit("message_end", {
		message: { role: "assistant", stopReason: "error", errorMessage: "This operation was aborted", content: [] },
	});
	await f.emit("agent_end");
	const [result] = f.entries.filter((e) => e.type === RESULT_ENTRY).map((e) => e.data);
	assert.equal(result.planner_budget_exhausted, "time");
	assert.equal(result.planner_error, null);
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
		reason: "Planner turns budget exhausted; the episode is over.",
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

test("a Flash replay with no program to load ends as an env_error, not a failed episode", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	toy(f.pi, async () => ["move"], {
		flash: {
			load: () => {
				throw new Error("no Flash program for this episode; looked for /plans/cell_plan.json");
			},
			start: async () => ({ localized: 0, rewrite: () => "stop" as const }),
			over: () => false,
			solved: () => false,
		},
	});
	await f.emit("session_start");
	await f.emit("agent_start");
	const provider = f.providers.find((p) => p.id === "flash");
	const turn = await provider.streamSimple(provider.getModels()[0], { messages: [] }, {}).result();
	const call = turn.content.find((c: any) => c.type === "toolCall");
	assert.equal(call.name, "finish");
	await f.emit("agent_end");
	await f.emit("session_shutdown");
	const results = f.entries.filter((e) => e.type === RESULT_ENTRY).map((e) => e.data);
	assert.equal(results.length, 1);
	assert.equal(results[0].env_error, true);
	assert.match(results[0].error, /^flash error: no program to replay: no Flash program for this episode/);
});

test("a service that stops answering mid-episode ends it as an env_error", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	const robot = toy(f.pi, async () => ["move", "render"]);
	robot.tool("render", "render", Type.Object({}), async () => {
		throw new RpcUnavailable("env.render: timed out after 5 ms; the server is still running it");
	});
	await f.emit("session_start");
	await f.emit("agent_start");
	await f.tools.get("move").execute("1", { n: 2 }, undefined, undefined, {});
	await assert.rejects(f.tools.get("render").execute("2", {}, undefined, undefined, {}), RpcUnavailable);
	const blocked = await f.emit("tool_call", { toolName: "move" });
	assert.equal(blocked?.terminate, true);
	assert.match(blocked?.reason, /The robot failed: env\.render: timed out/);
	await f.emit("agent_end");
	await f.emit("session_shutdown");
	const results = f.entries.filter((e) => e.type === RESULT_ENTRY).map((e) => e.data);
	assert.equal(results.length, 1);
	assert.equal(results[0].env_error, true);
	assert.match(results[0].error, /env\.render: timed out/);
	assert.equal(results[0].steps, 2);
});

test("a spent cost budget ends the episode and the result carries the cost", async (t) => {
	const f = fakePi({ "max-cost": "0.05" });
	t.after(f.restore);
	toy(f.pi, async () => ["move", "finish"]);
	await f.emit("session_start");
	await f.emit("agent_start");
	const reply = (usd: number) => ({
		message: { role: "assistant", stopReason: "toolUse", content: [], usage: { cost: { total: usd } } },
	});
	await f.emit("message_end", reply(0.03));
	assert.equal(await f.emit("tool_call", { toolName: "move" }), undefined);
	await f.emit("message_end", reply(0.03));
	assert.match((await f.emit("tool_call", { toolName: "move" }))?.reason, /Planner cost budget exhausted/);
	await f.emit("agent_end");
	const [result] = f.entries.filter((e) => e.type === RESULT_ENTRY).map((e) => e.data);
	assert.equal(result.planner_budget_exhausted, "cost");
	assert.equal(result.cost_usd, 0.06);
});

const unitsSpec: UnitsSpec = {
	vectors: {
		MV_FWD: [1, 0, 0],
		MV_BACK: [-1, 0, 0],
		MV_LEFT: [0, -1, 0],
		MV_RIGHT: [0, 1, 0],
		MV_UP: [0, 0, 1],
		MV_DOWN: [0, 0, -1],
	},
	stepM: 0.02,
	apply: async () => ({ content: [{ type: "text", text: "moved" }], details: {} }),
};
const unitsHandle = (f: ReturnType<typeof fakePi>) =>
	f.events.filter((e) => e.channel === UNITS_EVENT).at(-1)?.data as UnitsHandle;

test("an operator's unit passes the gates a tool call passes; every robot tool is listed for the takeover", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	toy(
		f.pi,
		async () => {
			throw new Error("no simulator");
		},
		{ units: unitsSpec },
	);
	await f.emit("session_start");
	assert.equal(unitsHandle(f).refuse(), "toy is not available.");

	const g = fakePi({ "max-turns": "1" });
	t.after(g.restore);
	const robot = toy(g.pi, async () => ["move", "finish"], { units: unitsSpec });
	robot.tool("render", "render", Type.Object({}), async () => {
		throw new RpcUnavailable("env.render: timed out");
	});
	await g.emit("session_start");
	const handle = unitsHandle(g);
	assert.deepEqual([...handle.tools()].sort(), ["act", "move", "plan", "render", "request_scene_reset", "reset"]);
	assert.equal(handle.refuse(), undefined);
	// Operator units are no planner turns; the budget the agent spent still binds them.
	await g.emit("turn_end");
	assert.equal(handle.refuse(), "Planner turns budget exhausted; the episode is over.");

	const h = fakePi();
	t.after(h.restore);
	const other = toy(h.pi, async () => ["move", "finish"], { units: unitsSpec });
	other.tool("render", "render", Type.Object({}), async () => {
		throw new RpcUnavailable("env.render: timed out");
	});
	await h.emit("session_start");
	await assert.rejects(h.tools.get("render").execute("1", {}, undefined, undefined, {}), RpcUnavailable);
	assert.match(unitsHandle(h).refuse() ?? "", /The robot failed: env\.render: timed out/);

	const k = fakePi();
	t.after(k.restore);
	toy(k.pi, async () => ["move", "finish"], { units: unitsSpec });
	await k.emit("session_start");
	await k.tools.get("finish").execute("1", { status: "success", summary: "ok" });
	assert.equal(unitsHandle(k).refuse(), "The episode is finished.");
});

test("the prompt the planner saw is recorded at turn start, once per change and per session", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	let prompt = "Move the block.";
	toy(f.pi, async () => ["move", "finish"], { prompt: () => prompt });
	// Every prompt closes with the shared closed-loop rules (../src/closed-loop.md).
	const closed = (text: string) => `${text}\n\n${toolSections(CLOSED_LOOP, ["move", "finish"])}`;
	// A later handler rewriting the robot's prompt, as explore.ts does: the entry holds the rewrite.
	f.pi.on("before_agent_start", () => ({ systemPrompt: `${closed(prompt)}\n\nExplore.` }));
	const recorded = () => f.entries.filter((e) => e.type === SYSTEM_PROMPT_ENTRY).map((e) => e.data.text);
	await f.emit("session_start");
	const first = await f.emit("before_agent_start");
	assert.deepEqual(first, { systemPrompt: `${closed("Move the block.")}\n\nExplore.` });
	assert.deepEqual(recorded(), [], "nothing before the turn starts");
	await f.emit("turn_start");
	await f.emit("turn_start");
	prompt = "Move the block carefully.";
	await f.emit("before_agent_start");
	await f.emit("turn_start");
	assert.deepEqual(recorded(), [
		`${closed("Move the block.")}\n\nExplore.`,
		`${closed("Move the block carefully.")}\n\nExplore.`,
	]);
	// A new session records its prompt again, even an unchanged one.
	await f.emit("session_start");
	await f.emit("before_agent_start");
	await f.emit("turn_start");
	assert.equal(recorded().length, 3);
});

test("a robot without a prompt of its own records nothing", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	toy(f.pi, async () => ["move", "finish"]);
	await f.emit("session_start");
	await f.emit("before_agent_start");
	await f.emit("turn_start");
	assert.deepEqual(
		f.entries.filter((e) => e.type === SYSTEM_PROMPT_ENTRY),
		[],
	);
});

test("a reply whose text holds an unparsed tool call is a planner error", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	toy(f.pi, async () => ["move"]);
	await f.emit("session_start");
	await f.emit("agent_start");
	await f.emit("before_agent_start");
	const leaked =
		'">\n<atem:parameter name="path">/root/.pi/embodied/memory/maniskill</atem:parameter>\n</atem:invoke>\n</atem:function_calls>';
	await f.emit("message_end", {
		message: { role: "assistant", stopReason: "stop", content: [{ type: "text", text: leaked }] },
	});
	await f.emit("turn_end");
	await f.emit("agent_end");
	await f.emit("session_shutdown");
	const [result] = f.entries.filter((e) => e.type === RESULT_ENTRY).map((e) => e.data);
	assert.equal(result.planner_error, "unparsed tool call in the reply text");
	assert.equal(leakedToolCall([{ type: "text", text: "The cube is left of the gripper." }]), undefined);
	assert.equal(leakedToolCall([{ type: "text", text: "<invoke>" }, { type: "toolCall" }]), undefined);
});

const UNITS: RobotSpec["units"] = {
	vectors: {
		MV_FWD: [1, 0, 0],
		MV_BACK: [-1, 0, 0],
		MV_LEFT: [0, -1, 0],
		MV_RIGHT: [0, 1, 0],
		MV_UP: [0, 0, 1],
		MV_DOWN: [0, 0, -1],
	},
	stepM: 0.02,
	apply: async () => ({ content: [], details: {} }),
};

/** A toy robot with memory on a local corpus and the given prompt; returns the forced system prompt. */
async function memoryPrompt(
	t: { after: (fn: () => void) => void },
	prompt: string,
	o: {
		memory?: { inPrompt?: boolean; prompt?: () => string };
		values?: Record<string, unknown>;
		extra?: Partial<RobotSpec>;
		active?: (names: string[]) => string[];
	} = {},
) {
	const f = fakePi({ "memory-profile": "local", ...o.values });
	t.after(f.restore);
	const memoryDir = join(f.dir, "memory", "toy");
	mkdirSync(memoryDir, { recursive: true });
	writeFileSync(join(memoryDir, "MEMORY.md"), "# memory\n");
	const robot = toy(f.pi, async () => ["move", "finish"], {
		prompt: () => prompt,
		memory: {
			home: () => join(f.dir, "memory"),
			cell: () => ({ tag: `cell_${robot.task.suite}`, reference: "cell_ref" }),
			primitives: ["move"],
			...o.memory,
		},
		...o.extra,
	});
	await f.emit("session_start");
	if (o.active) f.pi.setActiveTools(o.active(f.active()));
	return { text: (await f.emit("before_agent_start", { systemPrompt: "" }))?.systemPrompt as string, memoryDir };
}

const sections = (text: string) => text.split("Reading memory is a required step").length - 1;

test("the evaluation prompt reads memory once: memory's section at the robot's {{memory}}, else at its end", async (t) => {
	const slot = await memoryPrompt(t, "# Robot\nintro\n\n{{memory}}\n\n# Task\nMove.");
	assert.match(
		slot.text,
		/^# Robot\nintro\n\n# Memory\nCell `cell_flag-suite`[\s\S]*Reading memory is a required step[^\n]*\n\n# Task\nMove\./,
	);
	assert.ok(slot.text.includes(`${slot.memoryDir}/task_only/cell_ref_recipe.jsonl`));
	assert.doesNotMatch(slot.text, /\{\{\w+\}\}|\[\/?tool:/);
	assert.equal(sections(slot.text), 1);
	const end = await memoryPrompt(t, "# Task\nMove.");
	assert.match(end.text, /^# Task\nMove\.\n\n# Memory\nCell `cell_flag-suite`/);
	assert.equal(sections(end.text), 1);
	// A robot's own section (memory.prompt) takes the default's place and is closed the same way.
	const own = await memoryPrompt(t, "{{memory}}\n# Task", {
		memory: { prompt: () => "# Memory\nRead {{memory_dir}}/notes.md." },
	});
	assert.match(own.text, /^# Memory\nRead \S+\/notes\.md\.\n\nReading memory is a required step/);
	assert.doesNotMatch(own.text, /Cell `cell_flag-suite`/);
});

test("memory's section is left out while exploring, without memory's read tool, and for a robot whose prompt has its own", async (t) => {
	const own = await memoryPrompt(t, "# Task\nRead MEMORY.md first.", { memory: { inPrompt: true } });
	assert.equal(sections(own.text), 0);
	assert.doesNotMatch(own.text, /# Memory/);
	const noRead = await memoryPrompt(t, "{{memory}}\n# Task", { active: (names) => names.filter((n) => n !== "read") });
	assert.equal(sections(noRead.text), 0);
	assert.doesNotMatch(noRead.text, /# Memory|\{\{memory\}\}/);
	const exploring = await memoryPrompt(t, "{{memory}}\n# Task", {
		values: { explore: true },
		extra: {
			explore: { reset: async () => ({ content: [], details: {} }), prompt: () => "# Explore\nRead the inbox." },
		},
	});
	assert.equal(sections(exploring.text), 0);
	assert.match(exploring.text, /# Explore\nRead the inbox\./);
	assert.doesNotMatch(exploring.text, /\{\{memory\}\}/);
	// A mode that replaces the robot's prompt adds the section, even for a robot whose own prompt carried one.
	const pure = await memoryPrompt(t, "# Task\nRead MEMORY.md first.", {
		memory: { inPrompt: true },
		values: { units: "true", "units-plugins": "" },
		extra: { units: UNITS },
	});
	assert.equal(sections(pure.text), 1);
	assert.doesNotMatch(pure.text, /Read MEMORY\.md first/);
});

test("a server error reaches the model as one line with the server's message, never a traceback", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	const robot = toy(f.pi, async () => ["move", "finish", "step"]);
	robot.tool("step", "step", Type.Object({}), async () => {
		throw new Error(
			'env.move_delta: Traceback (most recent call last):\n  File "env_server.py", line 176, in move_delta\n    raise ValueError(\nValueError: delta moves 0.300 m; the limit is 0.2 m per call. Split the motion.\n',
		);
	});
	await f.emit("session_start");
	await assert.rejects(f.tools.get("step").execute("1", {}, undefined, undefined, {}), (err: Error) => {
		assert.equal(
			err.message,
			"env.move_delta: the server refused the call: ValueError: delta moves 0.300 m; the limit is 0.2 m per call. Split the motion.",
		);
		return true;
	});
	// An error without a traceback is the server's message already: unchanged.
	assert.equal(serverError(new Error("env.move_delta: delta moves 0.300 m")), undefined);
	assert.equal(serverError(new RpcUnavailable(`x: ${"Traceback (most recent call last):"}`)), undefined);
});
