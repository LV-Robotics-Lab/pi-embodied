import assert from "node:assert/strict";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { DEPLOYMENT, NUMERIC, numberError, params, trackedFlags } from "../src/infra/params.ts";

/** Every robot extension, loaded into a stub pi (flags at their defaults). */
const ROBOTS: Record<string, () => Promise<{ default: (pi: ExtensionAPI) => unknown }>> = {
	behavior: () => import("../src/robots/behavior/index.ts"),
	dual_franka: () => import("../src/robots/dual_franka/index.ts"),
	franka: () => import("../src/robots/franka/index.ts"),
	genesis: () => import("../src/robots/genesis/index.ts"),
	humanclaw: () => import("../src/robots/humanclaw/index.ts"),
	libero: () => import("../src/robots/libero/index.ts"),
	maniskill: () => import("../src/robots/maniskill/index.ts"),
	metaworld: () => import("../src/robots/metaworld/index.ts"),
	piper: () => import("../src/robots/piper/index.ts"),
	piper_dual: () => import("../src/robots/piper/dual.ts"),
	robocasa: () => import("../src/robots/robocasa/index.ts"),
	robodojo: () => import("../src/robots/robodojo/index.ts"),
	robolab: () => import("../src/robots/robolab/index.ts"),
	robosuite: () => import("../src/robots/robosuite/index.ts"),
	robotwin: () => import("../src/robots/robotwin/index.ts"),
	ur5e: () => import("../src/robots/ur5e/index.ts"),
};

function stubPi(values: Record<string, unknown> = {}) {
	const flags: Record<string, unknown> = {};
	const pi = {
		on: () => {},
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: () => {},
		registerCommand: () => {},
		registerProvider: () => {},
		registerMessageRenderer: () => {},
		registerShortcut: () => {},
		setActiveTools: () => {},
		getActiveTools: () => [],
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	return { pi, flags };
}

for (const [robot, load] of Object.entries(ROBOTS))
	test(`${robot}: every flag has one owner, the numeric ones parse at their defaults, and params names them`, async () => {
		const { pi } = stubPi();
		(await load()).default(pi);
		const t = trackedFlags(pi);
		assert.ok(t, `${robot} tracks its flags (trackFlags first in its extension)`);
		assert.deepEqual(t.duplicates, [], `${robot} registers a flag twice`);
		for (const name of t.names.keys()) assert.equal(numberError(name, pi.getFlag(name)), undefined, name);
		const p = params(pi);
		assert.ok(Object.keys(p.params).length > 5);
		for (const k of Object.keys(p.params)) assert.ok(!DEPLOYMENT.has(k), `${k} is deployment, not a parameter`);
		assert.deepEqual(Object.keys(p.params), Object.keys(p.params_default));
	});

test("a number that does not parse or is out of range is refused, never defaulted", () => {
	assert.match(String(numberError("max-turns", "abc")), /must be a number/);
	assert.match(String(numberError("max-turns", "1.5")), /must be an integer/);
	assert.match(String(numberError("max-turns", "-1")), /at least 0/);
	assert.match(String(numberError("gumi-operator-confidence", "2")), /at most 1/);
	assert.equal(numberError("max-turns", "12"), undefined);
	assert.equal(numberError("max-turns", ""), undefined, "unset keeps the default");
	assert.equal(numberError("not-numeric", "abc"), undefined);
	assert.ok(Object.keys(NUMERIC).includes("code-max-calls"));
});

test("the result records the extras that were on, every experiment flag (params) and their defaults", async () => {
	const { defineRobot, RESULT_ENTRY } = await import("../src/robot.ts");
	const { Type } = await import("typebox");
	const handlers = new Map<string, ((e: unknown, c: unknown) => unknown)[]>();
	const entries: { type: string; data: any }[] = [];
	const values: Record<string, unknown> = { "object-memory": true, "max-turns": "7" };
	const flags: Record<string, unknown> = {};
	const pi = {
		on: (n: string, fn: (e: unknown, c: unknown) => unknown) => handlers.set(n, [...(handlers.get(n) ?? []), fn]),
		registerFlag: (n: string, o: { default?: unknown }) => {
			flags[n] = n in values ? values[n] : o.default;
		},
		getFlag: (n: string) => flags[n],
		registerTool: () => {},
		registerCommand: () => {},
		setActiveTools: () => {},
		getActiveTools: () => [],
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	pi.registerFlag("object-memory", { type: "boolean", default: false });
	defineRobot(pi, {
		name: "toy",
		task: [],
		keepImages: 2,
		start: async () => [],
		result: () => ({}),
		finish: { description: "f", parameters: Type.Object({}), result: () => ({ content: [], details: {} }) },
	});
	const ctx = { hasUI: true, ui: { notify: () => {} }, sessionManager: { getBranch: () => [] }, shutdown: () => {} };
	for (const n of ["session_start", "agent_start", "session_shutdown"])
		for (const fn of handlers.get(n) ?? []) await fn({ type: n }, ctx);
	const r = entries.find((e) => e.type === RESULT_ENTRY)?.data;
	assert.deepEqual(r.extras, ["object-memory"]);
	assert.equal(r.params["max-turns"], "7");
	assert.equal(r.params_default["max-turns"], "0");
	assert.equal(r.params["object-memory"], true);
	assert.equal(r.params_default["object-memory"], false);
});

test("a robot refuses to start on a number that does not parse", async () => {
	const { defineRobot } = await import("../src/robot.ts");
	const { Type } = await import("typebox");
	const handlers = new Map<string, ((e: unknown, c: unknown) => unknown)[]>();
	const flags: Record<string, unknown> = {};
	const notes: string[] = [];
	const pi = {
		on: (n: string, fn: (e: unknown, c: unknown) => unknown) => handlers.set(n, [...(handlers.get(n) ?? []), fn]),
		registerFlag: (n: string, o: { default?: unknown }) => {
			flags[n] = n === "max-turns" ? "ten" : o.default;
		},
		getFlag: (n: string) => flags[n],
		registerTool: () => {},
		registerCommand: () => {},
		setActiveTools: () => {},
		getActiveTools: () => [],
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	let started = false;
	defineRobot(pi, {
		name: "toy",
		task: [],
		keepImages: 2,
		start: async () => {
			started = true;
			return [];
		},
		result: () => ({}),
		finish: { description: "f", parameters: Type.Object({}), result: () => ({ content: [], details: {} }) },
	});
	const ctx = { hasUI: true, ui: { notify: (m: string) => notes.push(m) }, sessionManager: { getBranch: () => [] } };
	for (const fn of handlers.get("session_start") ?? []) await fn({ type: "session_start" }, ctx);
	assert.equal(started, false);
	assert.match(notes.join("\n"), /--max-turns must be a number, got "ten"/);
});
