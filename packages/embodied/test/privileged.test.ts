import assert from "node:assert/strict";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import dualFranka from "../src/dual_franka/index.ts";
import franka from "../src/franka/index.ts";
import libero from "../src/libero/index.ts";
import maniskill from "../src/maniskill/index.ts";
import piper from "../src/piper/index.ts";
import robocasa from "../src/robocasa/index.ts";
import { defineRobot, RESULT_ENTRY } from "../src/robot.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi that runs handlers in registration order and records flags, tools and entries. */
function stubPi(values: Record<string, unknown> = {}) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	let active: string[] = [];
	const api: Record<string, unknown> = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
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
	};
	// Loading a whole robot also registers providers, renderers and the like: no-ops here.
	const pi = new Proxy(api, { get: (t, k: string) => t[k] ?? (() => {}) }) as unknown as ExtensionAPI;
	const ctx = { hasUI: true, ui: { notify: () => {} }, sessionManager: { getBranch: () => [] } };
	async function emit(name: string, event: Record<string, unknown> = {}) {
		for (const fn of handlers.get(name) ?? []) await fn({ type: name, ...event }, ctx);
	}
	return { pi, flags, tools, entries, emit, active: () => active };
}

/** A simulated toy robot whose env answers `env.ground_truth_poses` with the names it was asked for. */
function sim(values: Record<string, unknown>) {
	const f = stubPi(values);
	const asked: (string[] | undefined)[] = [];
	f.pi.registerFlag("seed", { type: "string", default: "0" });
	defineRobot(f.pi, {
		name: "toy",
		task: ["seed"],
		keepImages: 1,
		start: async () => ["move"],
		result: () => ({ success: false }),
		groundTruth: async (names) => {
			asked.push(names);
			return { frame: "world", poses: { cube: { pos: [0.1, 0, 0.02], quat_xyzw: [0, 0, 0, 1] } } };
		},
		finish: {
			description: "finish",
			parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
			result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
		},
	});
	return { ...f, asked };
}

const result = async (f: ReturnType<typeof stubPi>) => {
	await f.emit("agent_start");
	await f.emit("session_shutdown");
	return f.entries.find((e) => e.type === RESULT_ENTRY)?.data;
};

test("--privileged off: no ground_truth_poses tool, and the result is not marked", async () => {
	const f = sim({});
	assert.equal(f.flags.privileged, false);
	await f.emit("session_start");
	assert.equal(f.tools.has("ground_truth_poses"), false);
	assert.deepEqual(f.active(), ["move"]);
	const r = await result(f);
	assert.equal(r.success, false);
	assert.equal("privileged" in r, false);
});

test("--privileged on: ground_truth_poses is registered and active, calls the env, and the result is marked", async () => {
	const f = sim({ privileged: true });
	await f.emit("session_start");
	assert.deepEqual(f.active(), ["move", "ground_truth_poses"]);
	const tool = f.tools.get("ground_truth_poses");
	const out = await tool.execute("id", { names: ["cube"] }, undefined, undefined, {});
	assert.deepEqual(f.asked, [["cube"]]);
	assert.deepEqual(out.details.poses.cube.pos, [0.1, 0, 0.02]);
	assert.equal(out.details.frame, "world");
	await tool.execute("id", {}, undefined, undefined, {});
	assert.deepEqual(f.asked[1], undefined, "no names: every object");
	// A second session start does not register it again.
	await f.emit("session_start");
	assert.deepEqual(f.active(), ["move", "ground_truth_poses"]);
	assert.equal((await result(f)).privileged, true);
});

test("only the simulated robots register --privileged; the real ones have no such flag", () => {
	for (const [name, robot, sim] of [
		["libero", libero, true],
		["robocasa", robocasa, true],
		["maniskill", maniskill, true],
		["franka", franka, false],
		["dual_franka", dualFranka, false],
		["piper", piper, false],
	] as const) {
		const f = stubPi();
		robot(f.pi);
		assert.equal("privileged" in f.flags, sim, name);
		assert.equal(f.tools.has("ground_truth_poses"), false, `${name} registers nothing at load`);
	}
});
