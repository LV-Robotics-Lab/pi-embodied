import assert from "node:assert/strict";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { Static, TSchema } from "typebox";
import { NdArray } from "../src/infra/rpc.ts";
import {
	checkRotate,
	checkRoute,
	limitArgs,
	type MotionLimits,
	type MotionRig,
	moveDelta,
	rotateDelta,
	servedLimits,
	setGripper,
} from "../src/primitives/motion.ts";
import { viewCameraMeta, viewEnvState } from "../src/primitives/perception.ts";
import { getStep, outcome, type Step, type StepsIO, type ToolDef } from "../src/primitives/steps.ts";
import type { Json } from "../src/robot.ts";
import dualFranka from "../src/robots/dual_franka/index.ts";
import franka from "../src/robots/franka/index.ts";

/** A rig whose env is a recorder: every motion call lands in `calls` tagged with this rig's name. */
function rig(name: string, o: Partial<MotionRig> = {}) {
	const calls: { rig: string; method: string; kwargs: Json }[] = [];
	const r: MotionRig = {
		check: () => {},
		motion: async (method, kwargs) => {
			calls.push({ rig: name, method, kwargs });
			return { ok: true, rig: name };
		},
		...o,
	};
	return { r, calls };
}
const plain = (a: unknown) => (a instanceof NdArray ? a.toArray() : a);
/** Run a primitive as pi would, with params as the model sends them. */
const exec = <P extends TSchema>(d: ToolDef<P>, p: Json) => d.run(p as Static<P>, undefined);

test("one primitive mounted on two robots calls each robot's own env, with that robot's parameters", async () => {
	const one = rig("one");
	const two = rig("two", { arm: (v) => String(v).trim().toLowerCase() });
	const a = moveDelta(one.r);
	const b = moveDelta(two.r);
	assert.deepEqual(await exec(a, { delta_xyz: [0.0625, 0, 0] }), { ok: true, rig: "one" });
	assert.deepEqual(await exec(b, { arm: "Left", delta_xyz: [0, 0.03125, 0] }), { ok: true, rig: "two" });
	assert.deepEqual(await exec(setGripper(two.r, false), { arm: "right" }), { ok: true, rig: "two" });
	assert.deepEqual(await exec(setGripper(one.r, true), {}), { ok: true, rig: "one" });
	await exec(rotateDelta(two.r), { arm: "left", delta_rpy: [0, 0, 0.125] });
	assert.deepEqual(
		one.calls.map((c) => [c.method, Object.fromEntries(Object.entries(c.kwargs).map(([k, v]) => [k, plain(v)]))]),
		[
			["env.move_delta", { delta_xyz: [0.0625, 0, 0] }],
			["env.open_gripper", {}],
		],
	);
	assert.deepEqual(
		two.calls.map((c) => [c.method, Object.fromEntries(Object.entries(c.kwargs).map(([k, v]) => [k, plain(v)]))]),
		[
			["env.move_delta", { arm: "left", delta_xyz: [0, 0.03125, 0] }],
			["env.close_gripper", { arm: "right" }],
			["env.rotate_delta", { arm: "left", delta_rpy: [0, 0, 0.125] }],
		],
	);
	// The kwargs go to the env as float32 vectors, arm first.
	assert.ok(two.calls[0].kwargs.delta_xyz instanceof NdArray);
	assert.deepEqual(Object.keys(two.calls[0].kwargs), ["arm", "delta_xyz"]);
});

test("the motion tools leave pi's limits to the server; finiteness and the operator gate stay in pi", async () => {
	const { r, calls } = rig("r", { arm: (v) => String(v) });
	const move = moveDelta(r);
	// A large move reaches the env: the server enforces --max-move (services utils/code_real.py).
	await assert.doesNotReject(exec(move, { arm: "left", delta_xyz: [0.6, 0, 0] }));
	await assert.rejects(exec(move, { arm: "left", delta_xyz: [Number.NaN, 0, 0] }), /finite/);
	await assert.rejects(exec(move, { arm: "left", delta_xyz: [0, 0] }), /exactly 3 values/);
	const gated = setGripper(
		{
			...r,
			check: () => {
				throw new Error("operator paused the robot");
			},
		},
		true,
	);
	await assert.rejects(exec(gated, { arm: "left" }), /operator paused/);
	assert.equal(calls.length, 1);
});

test("pi's limits: the spawn arguments, an attached server's check, a multi-call route's pre-check", () => {
	const wanted: MotionLimits = {
		max_move_m: 0.04,
		max_rotate_rad: 0.5,
		z_floor_m: 0.14,
		workspace_xy: [0.2, 1, -0.5, 0.5],
	};
	assert.deepEqual(limitArgs(wanted), [
		"--max-move",
		"0.04",
		"--max-rotate",
		"0.5",
		"--z-floor",
		"0.14",
		"--workspace-xy",
		"0.2,1,-0.5,0.5",
	]);
	assert.deepEqual(limitArgs({ max_move_m: 0.1, workspace_xy: null }), ["--max-move", "0.1"]);
	assert.deepEqual(servedLimits({ ...wanted, max_move_m: 0.03 }, wanted).max_move_m, 0.03, "tighter is fine");
	assert.throws(() => servedLimits(undefined, wanted), /enforces none of pi's per-call limits/);
	assert.throws(() => servedLimits({ ...wanted, max_move_m: 0.1 }, wanted), /--max-move is 0.1/);
	assert.throws(() => servedLimits({ ...wanted, z_floor_m: 0.1 }, wanted), /--z-floor is 0.1/);
	assert.throws(() => servedLimits({ ...wanted, workspace_xy: null }, wanted), /--workspace-xy is off/);
	assert.throws(() => servedLimits({ ...wanted, workspace_xy: [0.1, 1, -0.5, 0.5] }, wanted), /--workspace-xy is/);
	assert.throws(() => checkRoute([0.5, 0, 0.3], [[0.05, 0, 0]], wanted), /the limit is 0.04 m per call/);
	assert.throws(() => checkRoute([0.5, 0, 0.16], [[0, 0, -0.03]], wanted), /outside the workspace/);
	assert.doesNotThrow(() => checkRoute([0.5, 0, 0.1], [[0, 0, 0.02]], wanted), "back toward the box");
	assert.throws(() => checkRotate([0.1, 0.1, 0.1], 0.1), /rotates 0\.1732 rad; the limit is 0\.1 rad/);
	assert.doesNotThrow(() => checkRotate([3, 0, 0], null));
});

test("the recorded-state layer: steps by index, views with images, a mutating tool records a step", async () => {
	type S = Step & { tag: string };
	const steps: S[] = [
		{ blob: { step_idx: 0, artifacts: [] }, dir: "/d/0", meta: { cameras: {} }, tag: "a" },
		{ blob: { step_idx: 1, artifacts: [] }, dir: "/d/1", meta: null, tag: "b" },
	];
	assert.equal(getStep(steps).tag, "b");
	assert.equal(getStep(steps, -2).tag, "a");
	assert.equal(getStep(steps, 0).tag, "a");
	assert.throws(() => getStep(steps, 2), /step 2 is not recorded \(have 0\.\.1\)/);
	let ready = false;
	const io: StepsIO<S> = {
		steps,
		ready: () => ready,
		dump: async (command, result, elapsed) => {
			const s = {
				blob: { step_idx: steps.length, artifacts: [], command, result, elapsed_s: elapsed },
				dir: "",
				meta: null,
				tag: "c",
			};
			steps.push(s);
			return s;
		},
		view: (s) => ({ output: { ...s.blob, tag: s.tag }, pngs: [Buffer.from(`png-${s.tag}`)] }),
		headline: (result) => (result.jam ? { jammed: true } : undefined),
	};
	const view = viewEnvState(io, "view");
	assert.deepEqual(await exec(view, { step: 0 }), {
		step_idx: 0,
		artifacts: [],
		tag: "a",
		_pngs: [Buffer.from("png-a")],
	});
	const meta = viewCameraMeta(io, "meta");
	assert.deepEqual(await exec(meta, {}), { error: "camera metadata is unavailable", step: -1 });
	assert.deepEqual(await exec(meta, { step: 0 }), { step: 0, camera_meta: { cameras: {} } });

	// Not up: nothing runs.
	let ran = 0;
	const body = async () => {
		ran++;
		return { moved: true };
	};
	assert.match((await outcome(io, "move_delta", { delta_xyz: [0, 0, 0] }, body)).details.error, /not initialized/);
	assert.equal(ran, 0);
	ready = true;
	// Read-only: the result, its `_pngs` attached as images.
	const read = await outcome(io, "view_env_state", {}, () => exec(view, { step: 1 }), false);
	assert.deepEqual(read.details, { step_idx: 1, artifacts: [], tag: "b" });
	assert.equal(read.content.length, 2);
	assert.equal(read.content[1].type, "image");
	// Mutating: a fresh step is recorded with the command, and returned with its image.
	const moved = await outcome(io, "move_delta", { delta_xyz: [0, 0, 0.01] }, body);
	assert.equal(steps.length, 3);
	assert.deepEqual(moved.details.command, { action: "move_delta", delta_xyz: [0, 0, 0.01] });
	assert.deepEqual(moved.details.result, { moved: true });
	assert.equal(moved.details.step_idx, 2);
	assert.equal(typeof moved.details.agent_elapsed_s, "number");
	assert.equal(moved.content[1].type, "image");
	// A failing body still records a step, keeps its error, and the headline leads.
	const failed = await outcome(io, "close_gripper", {}, async () => {
		throw new Error("fingers stuck");
	});
	assert.equal(failed.details.error, "fingers stuck");
	assert.equal(steps.length, 4);
	const jammed = await outcome(io, "close_gripper", {}, async () => ({ jam: true }));
	assert.deepEqual(Object.keys(jammed.details).slice(0, 2), ["jammed", "step_idx"]);
});

/** A stub pi that keeps the registered tools so a test can execute them. */
function fakePi() {
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const pi = {
		on: () => {},
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		setActiveTools: () => {},
		getActiveTools: () => [],
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	return { pi, tools };
}

test("franka and dual_franka mount the shared primitives behind their own not-initialized guard", async () => {
	for (const [load, params] of [
		[franka, { delta_xyz: [0.0625, 0, 0] }],
		[dualFranka, { arm: "left", delta_xyz: [0.0625, 0, 0] }],
	] as const) {
		const { pi, tools } = fakePi();
		load(pi);
		for (const name of [
			"view_env_state",
			"view_camera_meta",
			"move_delta",
			"rotate_delta",
			"open_gripper",
			"close_gripper",
		])
			assert.ok(tools.has(name), `${name} registered`);
		const r = await tools.get("move_delta").execute("id", params, undefined, undefined, {});
		assert.match(r.details.error, /robot not initialized/);
	}
});
