import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import {
	type Helper,
	ORACLE_MAX_CALLS,
	ORACLE_TIMEOUT_S,
	oracleTierHint,
	type RunResult,
	renderHelpers,
	renderPrimitives,
} from "../src/modes/code/index.ts";
import type { UnitsSpec } from "../src/modes/units/index.ts";
import { codeApiReply } from "./helpers/code-api.ts";

// The toy robot's primitive manifest (./fixtures/manifests/toy.json).
process.env.PI_EMBODIED_MANIFESTS = new URL("./fixtures/manifests/", import.meta.url).pathname;

import { defineRobot, RESULT_ENTRY } from "../src/robot.ts";
import { deployFlags } from "./helpers/deployment.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi that runs handlers in registration order and records flags, tools and entries. */
function fakePi(flagValues: Record<string, unknown> = {}, hasUI = true, confirm = async () => true) {
	flagValues = deployFlags(flagValues);
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	let active: string[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const confirmed: string[] = [];
	const stderr: string[] = [];
	let shutdown = false;
	const ctx = {
		hasUI,
		ui: {
			notify: () => {},
			confirm: async (_title: string, message: string) => {
				confirmed.push(message);
				return confirm();
			},
		},
		shutdown: () => {
			shutdown = true;
		},
		sessionManager: { getBranch: () => [] },
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) {
			const r: any = await fn({ type: name, ...event }, ctx);
			if (r !== undefined) result = r;
		}
		return result;
	}
	const run = async (name: string, params: unknown, signal?: AbortSignal) =>
		(await tools.get(name).execute("id", params, signal, undefined, ctx)) as any;
	return { pi, flags, tools, entries, emit, run, confirmed, stderr, active: () => active, stopped: () => shutdown };
}

const HELPERS: Helper[] = [{ name: "normalize_vector", signature: "(v)", doc: "v / |v|" }];

/** A fake env server: `code.api` answers with API (+ helpers, + ground truth when privileged); `code.run` with `answer`. */
function fakeEnv(
	answer: (kwargs: Record<string, unknown>) => Partial<RunResult> | Promise<Partial<RunResult>>,
	codeApi?: () => unknown,
) {
	const calls: { method: string; kwargs: Record<string, unknown>; signal?: AbortSignal }[] = [];
	let interrupts = 0;
	/** While set, `code.run` waits in the client's queue (behind another call) until it resolves. */
	const queue: { hold?: Promise<void>; preflight?: Record<string, unknown> | Error } = {};
	const rpc = {
		interrupt: async () => {
			interrupts++;
		},
		call: async <T>(
			method: string,
			kwargs: Record<string, unknown> = {},
			_t?: number,
			_a?: unknown[],
			signal?: AbortSignal,
			onSent?: () => void,
		): Promise<T> => {
			if (method === "code.run" && queue.hold) {
				await queue.hold;
				// RpcClient: a call released while it waited is never sent.
				if (signal?.aborted)
					throw new Error("code.run: aborted waiting for env.step, which the server is still running");
			}
			onSent?.();
			calls.push({ method, kwargs, signal });
			if (method === "code.api") return (codeApi?.() ?? codeApiReply("toy", kwargs.tier as string | undefined)) as T;
			if (method === "code.helpers") return HELPERS as T;
			if (method === "code.preflight") {
				if (queue.preflight instanceof Error) throw queue.preflight;
				return (queue.preflight ?? { isolated: true, error: null }) as T;
			}
			if (method === "code.run")
				return {
					status: "ran",
					stdout: "",
					stderr: "",
					traceback: null,
					error: null,
					result: null,
					calls: [],
					n_calls: 0,
					move_m: 0,
					ms: 1,
					...(await answer(kwargs)),
				} as T;
			throw new Error(`unexpected ${method}`);
		},
	};
	return { rpc, calls, interrupts: () => interrupts, queue };
}

const UNITS: UnitsSpec = {
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

/** A toy robot with code mode; `observe` absorbs the run and returns an image. */
async function toyRobot(
	flags: Record<string, unknown>,
	o: {
		answer?: (kwargs: Record<string, unknown>) => Partial<RunResult> | Promise<Partial<RunResult>>;
		real?: boolean;
		units?: boolean;
		hasUI?: boolean;
		confirm?: () => Promise<boolean>;
		ended?: boolean;
		/** `observe` fails on an aborted signal, as a robot's image calls do. */
		observeNeedsSignal?: boolean;
		/** Override the server's code.api reply. */
		codeApi?: () => unknown;
		/** What the env's `code.preflight` answers (an Error: the call is rejected with it). */
		preflight?: Record<string, unknown> | Error;
		/** The fixture manifest (default toy: high and low tiers). */
		manifest?: string;
	} = {},
) {
	const f = fakePi(flags, o.hasUI ?? true, o.confirm);
	const env = fakeEnv(o.answer ?? (() => ({})), o.codeApi);
	if (o.preflight) env.queue.preflight = o.preflight;
	const observed: RunResult[] = [];
	defineRobot(f.pi, {
		name: "toy",
		manifest: o.manifest ?? "toy",
		task: [],
		keepImages: 2,
		start: async () => ["move_to", "segment"],
		prompt: () => "ROBOT PROMPT",
		result: () => ({}),
		...(o.units ? { units: UNITS } : {}),
		...(o.real ? { operator: { step: () => 0 } } : {}),
		code: {
			rpc: () => env.rpc,
			instruction: () => "put the cube in the bowl",
			refuse: () => (o.ended ? "Episode already ended." : undefined),
			real: o.real,
			observe: async (r, signal) => {
				observed.push(r);
				if (o.observeNeedsSignal && signal?.aborted) throw new Error("env.raw_obs: aborted");
				return {
					content: [
						{ type: "text", text: JSON.stringify({ step: r.steps }) },
						{ type: "image", data: "img", mimeType: "image/png" },
					],
					details: { step: r.steps },
				};
			},
		},
		finish: {
			description: "finish",
			parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
			result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
		},
	});
	await f.emit("session_start");
	// A refused start without a UI sets pi's exit code (fail closed); this process is the test runner.
	process.exitCode = undefined;
	return { ...f, env, observed };
}

test("--code=true leaves only run_code and finish, fetches the tier's API and renders the code prompt", async () => {
	const f = await toyRobot({ code: true });
	assert.deepEqual(f.active(), ["run_code", "finish"]);
	assert.deepEqual(
		f.env.calls.map((c) => [c.method, c.kwargs]),
		[
			["code.api", { tier: "high" }],
			["code.preflight", {}],
		],
	);
	const prompt = (await f.emit("before_agent_start")).systemPrompt as string;
	assert.match(prompt, /^You control a robot arm by writing Python programs/);
	assert.match(prompt, /TASK: put the cube in the bowl/);
	assert.match(prompt, /You are perception-isolated: object coordinates are never given/, "without --privileged");
	assert.match(
		prompt,
		/PRIMITIVES \(high tier\):\ndef move_to\(xyz: vec3, gripper: number = None\):\n {4}Servo to xyz\.\n\n {4}Args:\n {8}xyz \(vec3\): target/,
	);
	assert.match(prompt, / {8}gripper \(number, optional\): -1 opens, \+1 closes\n\n {4}Moves the robot\./);
	assert.match(prompt, /def get_state\(\):\n {4}Proprioception\./);
	assert.doesNotMatch(prompt, /move_delta|ground_truth_poses/, "only the tier's primitives");
	assert.match(prompt, /timeout of 60 s .* at most 50 primitive calls, at most 3 m/);
	assert.doesNotMatch(prompt, /HELPERS/);
	assert.doesNotMatch(prompt, /ROBOT PROMPT/);
	assert.doesNotMatch(prompt, /\[\/?\w+\]/, "no section markers left");
	const tool = f.tools.get("run_code");
	assert.match(tool.description, /high tier.*move_to, get_state/);
	assert.match(JSON.stringify(tool.parameters), /timeout_s/);
});

test("--code=both adds run_code to the robot's tools and appends the code section; off registers nothing", async () => {
	const both = await toyRobot({ code: "both", "code-api": "low", "code-helpers": true });
	assert.deepEqual(both.active(), ["move_to", "segment", "run_code"]);
	assert.deepEqual(
		both.env.calls.map((c) => [c.method, c.kwargs]),
		[
			["code.api", { tier: "low" }],
			["code.preflight", {}],
			["code.helpers", {}],
		],
	);
	const prompt = (await both.emit("before_agent_start")).systemPrompt as string;
	assert.match(prompt, /^ROBOT PROMPT\n\n# Code mode\n/);
	assert.match(prompt, /PRIMITIVES \(low tier\):\ndef move_delta\(dxyz: vec3\):/);
	assert.match(prompt, / {4}Example:\n {8}move_delta\(\[0, 0, 0\.05\]\)/, "the low tier shows the examples");
	assert.doesNotMatch(prompt, /def move_to/);
	assert.match(prompt, /HELPERS .*\ndef normalize_vector\(v\):\n {4}v \/ \|v\|/);
	assert.match(both.tools.get("run_code").description, /low tier.*move_delta and the numpy helpers/);
	assert.doesNotMatch(prompt, /one `run_code` call per reply/);
	const off = await toyRobot({});
	assert.deepEqual(off.active(), ["move_to", "segment"]);
	assert.equal(off.tools.has("run_code"), false, "off: no run_code registered");
	assert.equal(off.env.calls.length, 0, "off: the env is never asked for code.api");
	// Off, the prompt is the robot's, closed by the shared closed-loop rules (../src/closed-loop.md).
	assert.match((await off.emit("before_agent_start")).systemPrompt, /^ROBOT PROMPT\n\n## Closed-loop rules\n/);
});

test("run_code passes the code, the clamped timeout and the limits, and reports the run before the observation", async () => {
	let seen: Record<string, unknown> = {};
	const f = await toyRobot(
		{ code: true, "code-timeout": "30", "code-max-calls": "7", "code-max-move": "1.5" },
		{
			answer: (kw) => {
				seen = kw;
				return {
					stdout: "hi\n",
					result: { pos: [1, 2] },
					calls: [{ name: "move_to" }],
					n_calls: 1,
					move_m: 0.2,
					steps: 12,
				};
			},
		},
	);
	const r = await f.run("run_code", { code: "RESULT = move_to([1, 2, 3])", timeout_s: 100 });
	assert.deepEqual(seen, {
		code: "RESULT = move_to([1, 2, 3])",
		timeout_s: 30,
		tier: "high",
		max_calls: 7,
		max_move_m: 1.5,
		helpers: false,
	});
	assert.equal(r.details.status, "ran");
	assert.equal(r.details.step, 12, "the robot absorbed the run");
	assert.equal(r.details.run.stdout, "hi\n");
	assert.match(r.content[0].text, /^run_code: \{\n "status": "ran"/);
	assert.match(r.content[0].text, /"pos": \[\n {3}1,\n {3}2\n {2}\]/);
	assert.equal(r.content[1].text, '{"step":12}');
	assert.equal(r.content[2].type, "image", "the latest camera image follows, like a motion tool's");
	assert.equal(f.observed.length, 1);
	// The default timeout is 60 s or the cap, whichever is smaller.
	await f.run("run_code", { code: "pass" });
	assert.equal(seen.timeout_s, 30);
});

test("error and timeout runs keep their status, error and traceback; a refused episode does not reach the env", async () => {
	const f = await toyRobot(
		{ code: true },
		{
			answer: (kw) =>
				String(kw.code).includes("loop")
					? { status: "timeout", error: "the program ran past its 60 s timeout and was killed", stop_issued: true }
					: {
							status: "error",
							error: "KeyError: 'x'",
							traceback: "File \"<run_code>\", line 1\nKeyError: 'x'",
							limit: "max_calls",
						},
		},
	);
	const err = await f.run("run_code", { code: "raise KeyError('x')" });
	assert.equal(err.details.status, "error");
	assert.match(err.content[0].text, /"error": "KeyError: 'x'"/);
	assert.match(err.content[0].text, /"traceback": "File/);
	assert.match(err.content[0].text, /"limit": "max_calls"/);
	const out = await f.run("run_code", { code: "while True: loop" });
	assert.equal(out.details.status, "timeout");
	assert.match(out.content[0].text, /killed/);
	const ended = await toyRobot({ code: true }, { ended: true });
	const r = await ended.run("run_code", { code: "pass" });
	assert.equal(r.details.status, "error");
	assert.match(r.content[0].text, /Episode already ended/);
	assert.equal(ended.env.calls.filter((c) => c.method === "code.run").length, 0);
});

test("pi's abort stops the server's run but waits for its result, so the steps it took are absorbed", async () => {
	const ac = new AbortController();
	const f = await toyRobot(
		{ code: true },
		{
			answer: async () => {
				ac.abort(); // the operator aborts while the program runs
				await new Promise((r) => setTimeout(r, 20));
				return { status: "error", cancelled: true, steps: 7, success_step: 3 };
			},
		},
	);
	const r = await f.run("run_code", { code: "pass" }, ac.signal);
	const run = f.env.calls.find((c) => c.method === "code.run");
	assert.equal(run?.signal?.aborted, false, "the sent call is not abandoned");
	assert.equal(f.env.interrupts(), 1, "the server was told to stop");
	assert.equal(r.details.status, "error");
	assert.equal(r.details.run.cancelled, true);
	assert.equal(f.observed[0]?.steps, 7, "the effects before the stop are absorbed");
	// An abort that already happened: nothing runs.
	const g = await toyRobot({ code: true });
	const done = new AbortController();
	done.abort();
	const out = await g.run("run_code", { code: "pass" }, done.signal);
	assert.equal(g.env.calls.filter((c) => c.method === "code.run").length, 0);
	assert.match(out.content[0].text, /aborted before it ran/);
});

test("--privileged asks for the ground-truth primitive and the prompt says so", async () => {
	const f = fakePi({ code: true, privileged: true });
	const env = fakeEnv(() => ({}));
	defineRobot(f.pi, {
		name: "sim",
		manifest: "toy",
		task: [],
		keepImages: 1,
		start: async () => ["move"],
		result: () => ({}),
		groundTruth: async () => ({}),
		code: { rpc: () => env.rpc, observe: async () => ({ content: [], details: {} }) },
		finish: { description: "f", parameters: Type.Object({}), result: () => ({ content: [], details: {} }) },
	});
	await f.emit("session_start");
	// --privileged runs the registry's privileged tier (high plus ground truth), whatever --code-api says.
	assert.deepEqual(env.calls[0].kwargs, { tier: "privileged" });
	assert.deepEqual(f.active(), ["run_code", "finish", "ground_truth_poses"]);
	const prompt = (await f.emit("before_agent_start")).systemPrompt as string;
	assert.match(prompt, /PRIMITIVES \(privileged tier\)/);
	assert.match(prompt, /def ground_truth_poses\(names: array = None\)/);
	assert.match(prompt, /def move_to\(/);
	assert.match(prompt, /privileged ground truth/);
	assert.doesNotMatch(prompt, /perception-isolated|never given/, "the prompt does not forbid the poses it offers");
	assert.match(f.tools.get("run_code").description, /ground_truth_poses/);
	const tool = f.tools.get("run_code");
	const calls: Record<string, unknown>[] = [];
	env.rpc.call = async <T>(method: string, kwargs: Record<string, unknown> = {}) => {
		calls.push({ method, ...kwargs });
		return {
			status: "ran",
			stdout: "",
			stderr: "",
			traceback: null,
			error: null,
			result: null,
			calls: [],
			n_calls: 0,
			move_m: 0,
			ms: 1,
		} as T;
	};
	await tool.execute("id", { code: "pass" }, undefined, undefined, { hasUI: true, ui: {} });
	assert.equal(calls[0].tier, "privileged");
});

test("--code and --units refuse to start together; a bad tier refuses too", async () => {
	const both = await toyRobot({ code: true, units: true }, { units: true, hasUI: false });
	assert.deepEqual(both.active(), []);
	assert.ok(both.stopped(), "a non-interactive run exits");
	const r = both.entries.find((e) => e.type === RESULT_ENTRY)?.data ?? (await result(both));
	assert.match(String(r.error), /mutually exclusive/);
	const bad = await toyRobot({ code: true, "code-api": "medium" }, { hasUI: false });
	assert.deepEqual(bad.active(), []);
	const units = await toyRobot({ units: true }, { units: true });
	assert.deepEqual(units.active(), ["act", "plan", "finish"], "units alone still start");
});

const result = async (f: Awaited<ReturnType<typeof toyRobot>>) => {
	await f.emit("agent_start");
	await f.emit("session_shutdown");
	return f.entries.find((e) => e.type === RESULT_ENTRY)?.data;
};

test("a real robot registers nothing without --code-real and --operator, and confirms every program with both", async () => {
	for (const flags of [{ code: true }, { code: true, "code-real": true }, { code: true, operator: true }]) {
		const f = await toyRobot(flags, { real: true, hasUI: false });
		assert.equal(f.tools.has("run_code"), false, JSON.stringify(flags));
		assert.deepEqual(f.active(), []);
		assert.equal(f.env.calls.length, 0, "the env is never asked for code.api");
		assert.match(String((await result(f)).error), /--code-real and --operator/);
	}
	const off = await toyRobot({}, { real: true });
	assert.deepEqual(off.active(), ["move_to", "segment"], "code off on a real robot: the robot's own tools");
	let yes = true;
	const on = await toyRobot(
		// --approval off: code mode's own confirmation (with --approval human the gate asks instead, runtime-mech.test.ts).
		{ code: true, "code-real": true, operator: true, approval: "off" },
		{ real: true, confirm: async () => yes },
	);
	assert.deepEqual(on.active(), ["run_code", "finish", "request_operator_verdict"]);
	assert.match(
		(await on.emit("before_agent_start")).systemPrompt,
		/This is a real robot: every program is shown to the operator/,
	);
	await on.run("run_code", { code: "move_to([0, 0, 0.3])" });
	assert.deepEqual(on.confirmed, ["move_to([0, 0, 0.3])"]);
	assert.equal(on.env.calls.filter((c) => c.method === "code.run").length, 1);
	yes = false;
	const declined = await on.run("run_code", { code: "move_to([0, 0, 0])" });
	assert.equal(declined.details.status, "error");
	assert.match(declined.content[0].text, /operator declined/);
	assert.equal(on.env.calls.filter((c) => c.method === "code.run").length, 1, "a declined program never runs");
});

test("the robot result records the code mode and tier", async () => {
	const f = await toyRobot({ code: "both", "code-api": "low" });
	const r = await result(f);
	assert.equal(r.code, "both");
	assert.equal(r.code_api, "low");
	assert.deepEqual(r.code_budget, { timeout_s: 60, max_calls: 50, max_move_m: 3, helpers: false });
	assert.equal(r.code_budget_flags, "timeout=+max_calls=+max_move=+helpers=false");
	const flagged = await result(await toyRobot({ code: true, "code-max-calls": "7", "code-oracle": tempOracle() }));
	assert.equal(flagged.code_budget_flags, "timeout=+max_calls=7+max_move=+helpers=false+oracle");
	const off = await toyRobot({});
	const plain = await result(off);
	assert.equal("code" in plain, false);
	const pure = await toyRobot({ code: true });
	assert.equal((await result(pure)).code, "true");
});

test("--stateless keeps the task and the latest observation turn in code mode", async () => {
	const f = await toyRobot({ code: true, stateless: true });
	const messages = [
		{ role: "user", content: "Solve." },
		{ role: "assistant", content: [{ type: "toolCall", id: "1", name: "run_code", arguments: {} }] },
		{ role: "toolResult", toolCallId: "1", content: [{ type: "image" }] },
		{ role: "assistant", content: [{ type: "toolCall", id: "2", name: "run_code", arguments: {} }] },
		{ role: "toolResult", toolCallId: "2", content: [{ type: "image" }] },
	];
	const kept = await f.emit("context", { messages });
	assert.deepEqual(kept.messages, [messages[0], messages[3], messages[4]]);
	assert.match((await f.emit("before_agent_start")).systemPrompt, /Only your latest step stays in context/);
	const stateful = await toyRobot({ code: true });
	assert.equal(await stateful.emit("context", { messages }), undefined);
});

test("renderPrimitives and renderHelpers lay each entry out as a def with its doc indented", () => {
	assert.equal(
		renderPrimitives([
			{
				name: "f",
				method: "env.f",
				doc: "Does a.",
				params: {
					a: { type: "integer", description: "the a", required: true },
					b: { type: "string", description: "", required: false },
				},
				mutating: true,
				tier: "high",
			},
		]),
		"def f(a: integer, b: string = None):\n    Does a.\n\n    Args:\n        a (integer): the a\n        b (string, optional):\n\n    Moves the robot.",
	);
	assert.equal(
		renderPrimitives([{ name: "g", method: "env.g", doc: "G.", params: {}, mutating: false, tier: "low" }]),
		"def g():\n    G.",
	);
	assert.equal(
		renderHelpers([{ name: "h", signature: "(v)", doc: "Does h.\n\nArgs:\n    v: x" }]),
		"def h(v):\n    Does h.\n\n    Args:\n        v: x",
	);
	assert.equal(renderHelpers([{ name: "k", signature: "()", doc: "" }]), "def k():");
});

test("an abort while run_code is still queued cancels it: it is never sent", async () => {
	const ac = new AbortController();
	const f = await toyRobot({ code: true });
	let release!: () => void;
	f.env.queue.hold = new Promise<void>((r) => {
		release = r;
	});
	const pending = f.run("run_code", { code: "move_to([0, 0, 0.3])" }, ac.signal);
	await new Promise((r) => setTimeout(r, 10));
	ac.abort();
	release();
	const out = await pending;
	assert.equal(
		f.env.calls.filter((c) => c.method === "code.run").length,
		0,
		"the queued run never reached the server",
	);
	assert.match(out.content[0].text, /aborted before it ran/);
	assert.equal(f.observed.length, 0);
});

test("after an abort the run's effects are absorbed and the result is clean, without images", async () => {
	const ac = new AbortController();
	const f = await toyRobot(
		{ code: true },
		{
			observeNeedsSignal: true,
			answer: async () => {
				ac.abort();
				return { status: "error", cancelled: true, error: "stopped: the run was aborted", steps: 4 };
			},
		},
	);
	const r = await f.run("run_code", { code: "pass" }, ac.signal);
	assert.equal(f.observed[0]?.steps, 4, "absorbed");
	assert.equal(r.details.status, "error");
	assert.equal(r.details.run.cancelled, true);
	assert.doesNotMatch(JSON.stringify(r.content), /env\.raw_obs: aborted/);
	assert.match(r.content[0].text, /"cancelled": true/);
});

test("S4 (--code-api=low-noexamples): the manifest's examples show in the low tier and not in S4", async () => {
	const s3 = await toyRobot({ code: true, "code-api": "low" });
	assert.match((await s3.emit("before_agent_start")).systemPrompt, /Example:\n {8}move_delta/);
	const s4 = await toyRobot({ code: true, "code-api": "low-noexamples" });
	assert.deepEqual(s4.active(), ["run_code", "finish"]);
	assert.deepEqual(s4.env.calls[0], { method: "code.api", kwargs: { tier: "low-noexamples" }, signal: undefined });
	const prompt = (await s4.emit("before_agent_start")).systemPrompt as string;
	assert.match(prompt, /def move_delta\(dxyz: vec3\):\n {4}Move by dxyz\./);
	assert.doesNotMatch(prompt, /Example:/);
	const r = await result(s4);
	assert.equal(r.code_api, "low-noexamples");
});

test("code mode refuses a server with another manifest, or one that disagrees on what is available", async () => {
	const other = await toyRobot(
		{ code: true },
		{ hasUI: false, codeApi: () => ({ ...codeApiReply("toy", "high"), manifest_digest: "0".repeat(64) }) },
	);
	assert.deepEqual(other.active(), []);
	assert.match(String((await result(other)).error), /runs primitive manifest 000000000000/);
	const disagree = await toyRobot(
		{ code: true },
		{ hasUI: false, codeApi: () => ({ ...codeApiReply("toy", "high"), available: ["move_to"] }) },
	);
	assert.match(String((await result(disagree)).error), /disagree on the high primitives/);
});

test("--code-oracle needs pure code mode and a simulator", async () => {
	const privilegedOracle = join(mkdtempSync(join(tmpdir(), "oracle-")), "gt.py");
	writeFileSync(privilegedOracle, "# tier: privileged\nprint(1)\n");
	for (const [flags, o, why] of [
		[{ "code-oracle": "x" }, {}, /needs --code=true/],
		[{ code: "both", "code-oracle": "x" }, {}, /needs --code=true/],
		[{ code: true, "code-oracle": "x", "code-real": true, operator: true }, { real: true }, /for simulators/],
		[{ code: true, "code-oracle": "x" }, {}, /no oracle x/],
		[
			{ code: true, "code-oracle": privilegedOracle },
			{},
			/privileged tier, this episode runs high \(add --privileged with --code-api=high\)/,
		],
	] as const) {
		const f = await toyRobot(flags, { ...o, hasUI: false });
		assert.deepEqual(f.active(), [], JSON.stringify(flags));
		assert.match(String((await result(f)).error), why);
	}
});

test("the result records the code budget; an oracle run relaxes the caps its flags leave unset", async () => {
	const plain = await toyRobot({ code: true });
	await plain.run("run_code", { code: "pass" });
	const run = plain.env.calls.find((c) => c.method === "code.run");
	assert.equal(run?.kwargs.max_calls, 50);
	const set = await toyRobot({ code: true, "code-max-calls": "7", "code-helpers": true });
	await set.run("run_code", { code: "pass" });
	assert.equal(set.env.calls.find((c) => c.method === "code.run")?.kwargs.max_calls, 7);
	const oracle = await toyRobot({ code: true, "code-oracle": tempOracle(), "code-max-move": "2" });
	await oracle.run("run_code", { code: "pass" });
	const o = oracle.env.calls.find((c) => c.method === "code.run");
	assert.equal(o?.kwargs.max_calls, ORACLE_MAX_CALLS);
	assert.equal(o?.kwargs.max_move_m, 2, "a given flag is kept");
	assert.ok((o?.kwargs.timeout_s as number) <= ORACLE_TIMEOUT_S);
});

/** A reference program on disk, for --code-oracle (a path is taken as is). */
function tempOracle(): string {
	const dir = mkdtempSync(join(tmpdir(), "oracle-"));
	const path = join(dir, "noop.py");
	writeFileSync(path, "RESULT = 1\n");
	return path;
}

test("code mode's preflight refuses a server that cannot isolate a program, before the episode", async () => {
	const f = await toyRobot(
		{ code: true },
		{ hasUI: false, preflight: { isolated: false, error: "run_code refused: run the server as root" } },
	);
	assert.ok(!f.active().includes("run_code"), "the robot did not start in code mode");
	const r = f.entries.find((e) => e.type === RESULT_ENTRY)?.data ?? (await result(f));
	assert.match(String(r.error), /run_code refused: run the server as root/);
	assert.equal(f.env.calls.filter((c) => c.method === "code.run").length, 0);
	// Audit 92245e3 CM-3: pi no longer tells the server whether it is remote; the server reads its peer address.
	assert.deepEqual(f.env.calls.find((c) => c.method === "code.preflight")?.kwargs, {});
});

test("only a server without code.preflight is let through; its other rejections fail the start", async () => {
	// Audit 92245e3 S4: every rejection (a token refusal, a busy server, a timeout) was read as "no such method".
	const old = await toyRobot(
		{ code: true },
		{ hasUI: false, preflight: new Error("code.preflight: unknown RPC method: 'code.preflight'") },
	);
	assert.ok(old.active().includes("run_code"), "an older server's code.run still refuses on its own");
	for (const why of [
		"code.preflight: refused: this server requires its RPC token",
		"code.preflight: refused: a run_code program is running on this server",
		"code.preflight: timed out after 30000 ms",
	]) {
		const f = await toyRobot({ code: true }, { hasUI: false, preflight: new Error(why) });
		assert.ok(!f.active().includes("run_code"), why);
		const r = f.entries.find((e) => e.type === RESULT_ENTRY)?.data ?? (await result(f));
		assert.equal(r.error, why);
	}
});

test("a real robot preflights the server before it confirms or resets: once, and a refusal stops it there", async () => {
	// Audit 92245e3 CM-2: the preflight sat in code mode's start(), after the robot's confirm + reset.
	for (const refused of [false, true]) {
		const f = fakePi({ code: true, "code-real": true, operator: true }, true);
		const env = fakeEnv(() => ({}));
		if (refused) env.queue.preflight = { isolated: false, error: "run_code refused: run the server as root" };
		const order: string[] = [];
		const robot = defineRobot(f.pi, {
			name: "toy",
			manifest: "toy",
			task: [],
			keepImages: 2,
			operator: { step: () => 0 },
			start: async (ctx) => {
				order.push("connected");
				await robot.codePreflight(env.rpc as never);
				order.push("confirm");
				await ctx.ui.confirm("Reset the arm?", "it moves");
				order.push("reset");
				return ["move_to"];
			},
			result: () => ({}),
			code: { rpc: () => env.rpc, real: true, observe: async () => ({ content: [], details: {} }) },
			finish: {
				description: "finish",
				parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
				result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
			},
		});
		await f.emit("session_start");
		if (refused) {
			assert.deepEqual(order, ["connected"], "neither confirmed nor reset");
			assert.deepEqual(f.confirmed, []);
			assert.ok(!f.active().includes("run_code"));
		} else {
			assert.deepEqual(order, ["connected", "confirm", "reset"]);
			assert.ok(f.active().includes("run_code"));
			assert.equal(env.calls.filter((c) => c.method === "code.preflight").length, 1, "start() did not ask again");
		}
	}
});

test("--code-api defaults to the robot's highest tier; an explicit tier it lacks is refused naming its tiers", async () => {
	const toy = await toyRobot({ code: true });
	assert.deepEqual(toy.env.calls.find((c) => c.method === "code.api")?.kwargs, { tier: "high" });
	const t = await result(toy);
	assert.equal(t.code_api, "high");
	assert.equal(t.code_api_auto, true);
	const low = await toyRobot({ code: true }, { manifest: "toylow", codeApi: () => codeApiReply("toylow", "low") });
	assert.deepEqual(low.env.calls.find((c) => c.method === "code.api")?.kwargs, { tier: "low" }, "no high tier: low");
	const l = await result(low);
	assert.equal(l.code_api, "low");
	assert.equal(l.code_api_auto, true);
	const explicit = await result(await toyRobot({ code: true, "code-api": "low" }));
	assert.equal(explicit.code_api, "low");
	assert.equal(explicit.code_api_auto, false);
	const bad = await toyRobot({ code: true, "code-api": "high" }, { manifest: "toylow", hasUI: false });
	assert.ok(!bad.active().includes("run_code"));
	const r = bad.entries.find((e) => e.type === RESULT_ENTRY)?.data ?? (await result(bad));
	assert.match(String(r.error), /--code-api=high: this robot has no high-tier code primitives; its tiers are low/);
});

test("an oracle's tier hint names flags --code-api accepts", () => {
	// low+privileged is --code-api=low with --privileged; --code-api itself refuses "low+privileged".
	assert.equal(oracleTierHint("low+privileged", true), "pass --code-api=low");
	assert.equal(oracleTierHint("low+privileged", false), "add --privileged with --code-api=low");
	assert.equal(oracleTierHint("privileged", false), "add --privileged with --code-api=high");
	assert.equal(oracleTierHint("privileged", true), "pass --code-api=high");
	assert.equal(oracleTierHint("low", true), "pass --code-api=low without --privileged");
	assert.equal(oracleTierHint("high", false), "pass --code-api=high");
});
