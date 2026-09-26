import assert from "node:assert/strict";
import { test } from "node:test";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { Static, TSchema } from "typebox";
import dualFranka from "../src/dual_franka/index.ts";
import franka from "../src/franka/index.ts";
import { checkRotate, type MotionRig, moveDelta, rotateDelta, setGripper } from "../src/primitives/motion.ts";
import { viewCameraMeta, viewEnvState } from "../src/primitives/perception.ts";
import { getStep, outcome, type Step, type StepsIO, type ToolDef } from "../src/primitives/steps.ts";
import type { Json } from "../src/robot.ts";
import { NdArray } from "../src/rpc.ts";

/** A rig whose env is a recorder: every motion call lands in `calls` tagged with this rig's name. */
function rig(name: string, o: Partial<MotionRig> = {}) {
	const calls: { rig: string; method: string; kwargs: Json }[] = [];
	const r: MotionRig = {
		check: () => {},
		motion: async (method, kwargs) => {
			calls.push({ rig: name, method, kwargs });
			return { ok: true, rig: name };
		},
		maxMove: () => 0.1,
		maxRotate: () => 0.5,
		constraints: () => undefined,
		workspace: () => {},
		...o,
	};
	return { r, calls };
}
const plain = (a: unknown) => (a instanceof NdArray ? a.toArray() : a);
/** Run a primitive as pi would, with params as the model sends them. */
const exec = <P extends TSchema>(d: ToolDef<P>, p: Json) => d.run(p as Static<P>, undefined);

test("one primitive mounted on two robots calls each robot's own env, with that robot's parameters", async () => {
	const one = rig("one");
	const two = rig("two", {
		arm: { schema: StringEnum(["left", "right"] as const), name: (v) => String(v).trim().toLowerCase() },
	});
	const a = moveDelta(one.r, "one arm");
	const b = moveDelta(two.r, "two arms");
	assert.deepEqual(Object.keys(a.parameters.properties), ["delta_xyz"]);
	assert.deepEqual(Object.keys(b.parameters.properties), ["arm", "delta_xyz"]);
	assert.deepEqual(await exec(a, { delta_xyz: [0.0625, 0, 0] }), { ok: true, rig: "one" });
	assert.deepEqual(await exec(b, { arm: "Left", delta_xyz: [0, 0.03125, 0] }), { ok: true, rig: "two" });
	assert.deepEqual(await exec(setGripper(two.r, false, "close"), { arm: "right" }), { ok: true, rig: "two" });
	assert.deepEqual(await exec(setGripper(one.r, true, "open"), {}), { ok: true, rig: "one" });
	await exec(rotateDelta(two.r, "turn"), { arm: "left", delta_rpy: [0, 0, 0.125] });
	assert.deepEqual(
		one.calls.map((c) => [c.method, Object.fromEntries(Object.entries(c.kwargs).map(([k, v]) => [k, plain(v)]))]),
		[
			["env.move_delta", { delta_xyz: [0.0625, 0, 0] }],
			["env.set_gripper", { open: true }],
		],
	);
	assert.deepEqual(
		two.calls.map((c) => [c.method, Object.fromEntries(Object.entries(c.kwargs).map(([k, v]) => [k, plain(v)]))]),
		[
			["env.move_delta", { arm: "left", delta_xyz: [0, 0.03125, 0] }],
			["env.set_gripper", { arm: "right", open: false }],
			["env.rotate_delta", { arm: "left", delta_rpy: [0, 0, 0.125] }],
		],
	);
	// The kwargs go to the env as float32 vectors, arm first.
	assert.ok(two.calls[0].kwargs.delta_xyz instanceof NdArray);
	assert.deepEqual(Object.keys(two.calls[0].kwargs), ["arm", "delta_xyz"]);
});

test("the motion primitives refuse before the env: limits, workspace, finiteness, operator gate", async () => {
	const seen: unknown[] = [];
	const { r, calls } = rig("r", {
		maxMove: () => 0.05,
		maxRotate: () => 0.2,
		constraints: () => ["Keep translation commands at or below 0.02 m per call"],
		workspace: (delta, arm) => {
			seen.push([delta, arm]);
			if (delta[2] < 0) throw new Error("below the floor");
		},
		arm: { schema: StringEnum(["left", "right"] as const), name: (v) => String(v) },
	});
	const move = moveDelta(r, "m");
	// The task's documented 0.02 m is tighter than --max-move.
	await assert.rejects(exec(move, { arm: "left", delta_xyz: [0.03, 0, 0] }), /the limit is 0\.02 m per call/);
	await assert.rejects(exec(move, { arm: "left", delta_xyz: [0, 0, -0.6] }), /the limit is 0\.02 m per call/);
	await assert.rejects(exec(move, { arm: "left", delta_xyz: [Number.NaN, 0, 0] }), /finite/);
	await assert.rejects(exec(move, { arm: "left", delta_xyz: [0, 0] }), /exactly 3 values/);
	assert.deepEqual(seen, [], "nothing reached the workspace check");
	const loose = moveDelta({ ...r, constraints: () => [] }, "m");
	await assert.rejects(exec(loose, { arm: "right", delta_xyz: [0, 0, -0.9] }), /the limit is 0\.05 m per call/);
	await assert.rejects(exec(loose, { arm: "right", delta_xyz: [0, 0, -0.02] }), /below the floor/);
	await assert.doesNotReject(exec(loose, { arm: "right", delta_xyz: [0, 0, 0.02] }));
	assert.deepEqual(seen, [
		[[0, 0, -0.02], "right"],
		[[0, 0, 0.02], "right"],
	]);
	const turn = rotateDelta(r, "t");
	await assert.rejects(
		exec(turn, { arm: "left", delta_rpy: [0, 0, 0.3] }),
		/delta_rpy rotates 0\.3 rad; the limit is 0\.2 rad per call/,
	);
	assert.throws(() => checkRotate([0.1, 0.1, 0.1], 0.1), /rotates 0\.1732 rad; the limit is 0\.1 rad/);
	const gated = setGripper(
		{
			...r,
			check: () => {
				throw new Error("operator paused the robot");
			},
		},
		true,
		"o",
	);
	await assert.rejects(exec(gated, { arm: "left" }), /operator paused/);
	assert.equal(calls.length, 1, "only the in-limit move reached the env");
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
