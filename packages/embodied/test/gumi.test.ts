import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	ARM,
	actSteps,
	armState,
	DUAL,
	gumi,
	type Observation,
	observation,
	parseSteps,
	Recorder,
	Takeover,
	UNITS_EVENT,
	type UnitsHandle,
	views,
} from "../src/gumi/index.ts";
import { encodePng } from "../src/png.ts";
import { STATUS_EVENT } from "../src/robot.ts";

const SINGLE = ["MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN", "STOP", "GRASP", "RELEASE", "DONE"];
const WITH_YAW = [...SINGLE, "ROTATE_CW", "ROTATE_CCW"];
const units = (steps: Record<string, string>[], arm = ARM) => steps.map((s) => s[arm]);

test("sequence box: keys, names and repeats expand to one unit per step", () => {
	assert.deepEqual(units(parseSteps({ command: "w*3 a g" }, [ARM], SINGLE)), [
		"MV_FWD",
		"MV_FWD",
		"MV_FWD",
		"MV_LEFT",
		"GRASP",
	]);
	assert.deepEqual(units(parseSteps({ command: "MV_UP*2, r q e s d" }, [ARM], SINGLE)), [
		"MV_UP",
		"MV_UP",
		"RELEASE",
		"MV_UP",
		"MV_DOWN",
		"MV_BACK",
		"MV_RIGHT",
	]);
	assert.deepEqual(units(parseSteps({ units: ["mv_fwd*2", "G"] }, [ARM], SINGLE)), ["MV_FWD", "MV_FWD", "GRASP"]);
	assert.deepEqual(units(parseSteps({ unit: "x" }, [ARM], WITH_YAW)), ["ROTATE_CW"]);
});

test("sequence box: refuses what the robot cannot do, with the reason", () => {
	const bad = (body: Record<string, unknown>, re: RegExp, arms: readonly string[] = [ARM], vocab = SINGLE) =>
		assert.throws(() => parseSteps(body, arms, vocab), re);
	bad({ command: "z" }, /not a unit here: ROTATE_CCW/); // no yaw on this robot
	bad({ command: "w*0" }, /repeat count/);
	bad({ command: "w*65" }, /repeat count/);
	bad({ command: "w*40 s*40" }, /too many steps/);
	bad({ command: "w-3" }, /cannot parse/);
	bad({ command: "" }, /empty command/);
	bad({ command: "L:w R:s" }, /one arm/);
	bad({ left: "w" }, /one arm/);
	bad({ command: "w", unit: "w" }, /exactly one/);
	bad({ command: "STILL" }, /not a unit here: STILL/); // STILL is a two-arm unit
	bad({ command: "w" }, /prefix every group/, DUAL, [...SINGLE, "STILL"]);
	bad({ command: "L:STILL R:STAY" }, /at least one arm/, DUAL, [...SINGLE, "STILL"]);
});

test("two arms: one synchronized (left, right) pair per step, the idle arm STILL", () => {
	const vocab = [...SINGLE, "STILL"];
	assert.deepEqual(parseSteps({ command: "L:w*2 R:i u" }, DUAL, vocab), [
		{ left: "MV_FWD", right: "MV_FWD" },
		{ left: "MV_FWD", right: "MV_UP" },
	]);
	assert.deepEqual(parseSteps({ command: "R:. l*2" }, DUAL, vocab), [
		{ left: "STILL", right: "RELEASE" },
		{ left: "STILL", right: "MV_RIGHT" },
		{ left: "STILL", right: "MV_RIGHT" },
	]);
	assert.deepEqual(parseSteps({ left: "g", right: ["STAY", "e"] }, DUAL, vocab), [
		{ left: "GRASP", right: "STILL" },
		{ left: "STILL", right: "MV_DOWN" },
	]);
	assert.deepEqual(actSteps({ unit: "mv_up", arm: "right", n: 2 }, DUAL), {
		step: { left: "STILL", right: "MV_UP" },
		n: 2,
	});
	assert.deepEqual(actSteps({ unit: "GRASP" }, [ARM]), { step: { arm: "GRASP" }, n: 1 });
	assert.equal(actSteps({ unit: "STOP" }, [ARM]), undefined);
	assert.equal(actSteps({ unit: "DONE" }, [ARM]), undefined);
});

// ---------------------------------------------------------------------------
// recording

const png = (value: number) => encodePng(Buffer.alloc(4 * 4 * 3, value), 4, 4).toString("base64");
const image = (value: number) => ({ type: "image" as const, data: png(value), mimeType: "image/png" });
function result(values: number[], labels: string[], state: Record<string, unknown> = {}) {
	return {
		content: [
			{ type: "text" as const, text: "units: MV_FWD x1\nRecent units: MV_FWD" },
			{ type: "text" as const, text: JSON.stringify({ result: {}, state, images: labels }) },
			...values.map(image),
		],
		details: {},
	};
}

test("observations: views by camera label, else by position; state from the robot JSON", () => {
	const o = observation(result([1, 2], ["wrist_high 1024x1024", "agentview_high 1024x1024"])) as Observation;
	assert.deepEqual(o.labels, ["wrist_high 1024x1024", "agentview_high 1024x1024"]);
	const v = views(o, [ARM]);
	assert.equal(v.agentview.data, png(2));
	assert.equal(v.wrist.data, png(1));
	const unlabelled = views(observation(result([5, 6], [])) as Observation, [ARM]);
	assert.equal(unlabelled.agentview.data, png(5));
	assert.equal(unlabelled.wrist.data, png(6));
	const d = views(observation(result([7, 8, 9], ["d455", "right_wrist", "left_wrist"])) as Observation, DUAL);
	assert.deepEqual([d.agentview.data, d.wrist_left.data, d.wrist_right.data], [png(7), png(9), png(8)]);
	assert.equal(observation({ content: [{ type: "text", text: "no image" }] }), undefined);
	assert.deepEqual(armState({ state: { robot0_eef_pos: [0.1, 0.2, 0.3], robot0_gripper_qpos: [0.02, -0.02] } }, ARM), {
		ee_pose: [0.1, 0.2, 0.3],
		gripper_width: 0.04,
	});
	assert.deepEqual(armState({ eef_xyz: [1, 2, 3], gripper_width: 0.05 }, ARM), {
		ee_pose: [1, 2, 3],
		gripper_width: 0.05,
	});
});

test("recorder: episode_logger run dir with steps.jsonl, and the actions.jsonl rollouts_to_alpaca.py reads", () => {
	const root = mkdtempSync(join(tmpdir(), "gumi-"));
	const rec = new Recorder(root, [ARM]);
	const at = new Date("2026-09-25T07:04:05Z"); // 15:04:05 in UTC+8
	const dir = rec.start({ task: "put the bowl on the plate", robot: "libero" }, 2, at);
	assert.equal(dir, join(root, "0925", "task_2", "15-04-05"));
	const obs = observation(result([10, 20], ["agentview", "wrist"])) as Observation;
	const info = (src: "human" | "agent", dagger = false) => ({
		src,
		dagger,
		closed: { [ARM]: false },
		state: { [ARM]: { ee_pose: [0.1, 0.2, 0.3], gripper_width: 0.08 } },
	});
	rec.add(obs, { [ARM]: "MV_FWD" }, info("agent"));
	rec.add(obs, { [ARM]: "GRASP" }, info("human", true));
	const { meta } = rec.stop({ success: true });
	const read = (f: string) =>
		readFileSync(join(dir, f), "utf8")
			.trim()
			.split("\n")
			.map((l) => JSON.parse(l));
	const lines = read("actions.jsonl");
	assert.equal(lines.length, 2);
	assert.deepEqual(
		{ ...lines[1], time: 0 },
		{
			step: 1,
			token: "GRASP",
			kind: "gripper",
			gripper_closed: false,
			ee_pose: [0.1, 0.2, 0.3],
			gripper_width: 0.08,
			agentview: "images/agentview/0001.png",
			wrist: "images/wrist/0001.png",
			time: 0,
			src: "human",
			dagger: true,
		},
	);
	const logged = read("steps.jsonl");
	assert.deepEqual(
		{ ...logged[1], ts: 0 },
		{
			i: 1,
			stage: "-",
			act: "GRASP",
			eef: [0.1, 0.2, 0.3],
			w: 0.08,
			grip: "OPEN",
			src: "human",
			ts: 0,
			dagger: true,
		},
	);
	assert.deepEqual(JSON.parse(readFileSync(join(dir, "steps.json"), "utf8")), logged);
	// The PNGs are the tool result's bytes, exactly what the policy was shown.
	assert.equal(readFileSync(join(dir, "images/agentview/0000.png")).toString("base64"), png(10));
	assert.equal(readFileSync(join(dir, "images/wrist/0001.png")).toString("base64"), png(20));
	assert.deepEqual(meta.tokens, ["MV_FWD", "GRASP"]);
	assert.equal(JSON.parse(readFileSync(join(dir, "metadata.json"), "utf8")).task, "put the bowl on the plate");
	const summary = JSON.parse(readFileSync(join(dir, "summary.json"), "utf8"));
	assert.deepEqual([summary.success, summary.steps, summary.control_mode], [true, 2, "gumi"]);
	// A second run in the same second gets its own dir; discard deletes it.
	const next = rec.start({}, 2, at);
	assert.equal(next, join(root, "0925", "task_2", "15-04-05_1"));
	rec.discard();
	assert.equal(existsSync(next), false);
});

test("recorder, two arms: agentview + wrist_left/right and per-arm token records", () => {
	const root = mkdtempSync(join(tmpdir(), "gumi-"));
	const rec = new Recorder(root, DUAL);
	const dir = rec.start({});
	const obs = observation(result([1, 2, 3], ["d455", "left_wrist", "right_wrist"])) as Observation;
	const state = { left: { ee_pose: [1], gripper_width: 0.01 }, right: { ee_pose: [2], gripper_width: 0.02 } };
	rec.add(obs, { left: "MV_FWD", right: "STILL" }, { src: "human", dagger: true, closed: { left: true }, state });
	const [line] = readFileSync(join(dir, "actions.jsonl"), "utf8")
		.trim()
		.split("\n")
		.map((l) => JSON.parse(l));
	assert.equal(line.wrist_left, "images/wrist_left/0000.png");
	assert.equal(line.wrist_right, "images/wrist_right/0000.png");
	assert.deepEqual(line.left, {
		token: "MV_FWD",
		kind: "move",
		gripper_closed: true,
		ee_pose: [1],
		gripper_width: 0.01,
		src: "human",
	});
	assert.equal(line.right.token, "STILL");
	assert.equal(line.right.kind, "still");
	assert.equal(line.dagger, "L");
	assert.deepEqual(readdirSync(join(dir, "images")).sort(), ["agentview", "wrist_left", "wrist_right"]);
	const [logged] = readFileSync(join(dir, "steps.jsonl"), "utf8")
		.trim()
		.split("\n")
		.map((l) => JSON.parse(l));
	assert.deepEqual(
		[logged.left.act, logged.right.act, logged.left.grip, logged.dagger],
		["MV_FWD", "STILL", "CLOSED", "L"],
	);
	const { meta } = rec.stop({});
	assert.deepEqual([meta.tokens_left, meta.tokens_right], [["MV_FWD"], ["STILL"]]);
});

// ---------------------------------------------------------------------------
// takeover

test("takeover: the agent waits while the operator drives; a decision older than their steps is stale", async () => {
	const t = new Takeover();
	assert.deepEqual(await t.gate(), { stale: false, driven: [] });
	// take() during an agent unit waits for it.
	assert.equal(t.take(), "requested");
	assert.equal(t.human, false);
	t.done();
	assert.equal(t.mode, "human");
	let passed: { stale: boolean; driven: string[] } | undefined;
	const gate = t.gate().then((r) => {
		passed = r;
	});
	await new Promise((r) => setTimeout(r, 10));
	assert.equal(passed, undefined, "the agent is paused while the operator has the robot");
	t.humanStep(["MV_FWD"]);
	t.humanStep(["GRASP"]);
	t.release();
	await gate;
	assert.deepEqual(passed, { stale: true, driven: ["MV_FWD", "GRASP"] });
	// After the drop the agent decides on a fresh observation.
	assert.deepEqual(await t.gate(), { stale: false, driven: [] });
	t.done();
	// Taking over and handing back without acting drops nothing.
	t.take();
	assert.equal(t.mode, "human");
	t.release();
	assert.equal((await t.gate()).stale, false);
	t.done();
	// Operator steps before the agent's next observation make its pending decision stale.
	t.humanStep(["MV_UP"]);
	assert.equal((await t.gate()).stale, true);
	t.humanStep(["MV_UP"]);
	t.seen();
	assert.equal((await t.gate()).stale, false);
	t.done();
	// An abort while paused ends the wait.
	t.take();
	const ac = new AbortController();
	const waiting = t.gate(ac.signal);
	ac.abort();
	await assert.rejects(waiting, /aborted/);
});

// ---------------------------------------------------------------------------
// gumi on a fake pi

type Handler = (event: any, ctx: any) => unknown;
function fakePi(flags: Record<string, unknown>) {
	const handlers = new Map<string, Handler[]>();
	const listeners = new Map<string, ((d: unknown) => void)[]>();
	const sent: unknown[] = [];
	let idle = true;
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] ??= o.default;
		},
		getFlag: (name: string) => flags[name],
		sendUserMessage: (content: unknown, options: unknown) => sent.push({ content, options }),
		events: {
			emit: (channel: string, data: unknown) => {
				for (const fn of listeners.get(channel) ?? []) fn(data);
			},
			on: (channel: string, fn: (d: unknown) => void) =>
				listeners.set(channel, [...(listeners.get(channel) ?? []), fn]),
		},
	} as unknown as ExtensionAPI;
	const ctx = {
		isIdle: () => idle,
		signal: undefined,
	};
	const emit = async (name: string, event: Record<string, unknown> = {}) => {
		let out: unknown;
		for (const fn of handlers.get(name) ?? []) out = (await fn({ type: name, ...event }, ctx)) ?? out;
		return out;
	};
	return {
		pi,
		emit,
		sent,
		setIdle: (v: boolean) => {
			idle = v;
		},
	};
}

function fakeRobot() {
	const calls: { unit: string; arm?: string }[] = [];
	let frame = 100;
	const handle: UnitsHandle = {
		tool: "act",
		arms: [],
		vocabulary: SINGLE,
		stepM: 0.02,
		run: async (params) => {
			calls.push(params);
			frame++;
			return result([frame, frame + 50], ["agentview_high", "wrist_high"], {
				robot0_eef_pos: [0, 0, frame / 1000],
				robot0_gripper_qpos: [0.04, -0.04],
			});
		},
		state: async () => ({ eef_xyz: [0, 0, frame / 1000], gripper_width: 0.08 }),
	};
	return { handle, calls };
}

test("gumi: teleop records (obs_t, a_t) through units.run; agent steps too; save marks success", async () => {
	const root = mkdtempSync(join(tmpdir(), "gumi-"));
	const f = fakePi({ "gumi-record": root });
	const g = gumi(f.pi);
	const robot = fakeRobot();
	await f.emit("session_start");
	f.pi.events.emit(UNITS_EVENT, robot.handle);
	f.pi.events.emit(STATUS_EVENT, {
		robot: "libero",
		task: { suite: "libero_10" },
		language: "open the drawer",
		solved: false,
	});
	g.record("start");
	// No observation yet: the first recorded step looks first (STOP), which is not recorded itself.
	const out = await g.step({ command: "w*2 g" });
	assert.equal(out.executed, 3);
	assert.deepEqual(
		robot.calls.map((c) => c.unit),
		["STOP", "MV_FWD", "MV_FWD", "GRASP"],
	);
	assert.deepEqual(g.state().closed, { arm: true });
	// The agent starts after the operator drove: its first call predates their steps and is dropped,
	// with the current observation delivered instead; the next one runs and is recorded as the agent's.
	f.setIdle(false);
	await f.emit("agent_start");
	assert.equal(((await f.emit("tool_call", { toolName: "act", input: { unit: "STOP" } })) as any).block, true);
	assert.equal(f.sent.length, 1);
	assert.equal(await f.emit("tool_call", { toolName: "act", input: { unit: "MV_UP" } }), undefined);
	await f.emit("tool_result", {
		toolName: "act",
		input: { unit: "MV_UP" },
		...result([7, 8], ["agentview", "wrist"]),
	});
	// The agent is driving: the operator must take over first.
	await assert.rejects(g.step({ unit: "w" }), (e: any) => e.status === 409);
	f.setIdle(true);
	await assert.rejects(g.step({ command: "q*999" }), (e: any) => e.status === 422);
	const saved = g.record("save", true);
	const dir = saved.dir;
	const lines = readFileSync(join(dir, "actions.jsonl"), "utf8")
		.trim()
		.split("\n")
		.map((l) => JSON.parse(l));
	assert.deepEqual(
		lines.map((l) => [l.token, l.src]),
		[
			["MV_FWD", "human"],
			["MV_FWD", "human"],
			["GRASP", "human"],
			["MV_UP", "agent"],
		],
	);
	// obs_t: each step's images are the ones before it ran (STOP's result for the first MV_FWD).
	assert.equal(readFileSync(join(dir, lines[0].agentview)).toString("base64"), png(101));
	assert.equal(readFileSync(join(dir, lines[1].agentview)).toString("base64"), png(102));
	assert.equal(readFileSync(join(dir, lines[3].wrist)).toString("base64"), png(154));
	assert.equal(lines[2].gripper_closed, false);
	assert.equal(lines[3].gripper_closed, true);
	assert.deepEqual(lines[0].ee_pose, [0, 0, 0.101]);
	const meta = JSON.parse(readFileSync(join(dir, "metadata.json"), "utf8"));
	assert.equal(meta.task, "open the drawer");
	assert.equal(meta.success, true);
	assert.deepEqual(meta.sources, ["human", "human", "human", "agent"]);
});

test("gumi: DAgger takeover pauses the agent between units, drops its stale call, records who acted", async () => {
	const root = mkdtempSync(join(tmpdir(), "gumi-"));
	const f = fakePi({ "gumi-record": root });
	const g = gumi(f.pi);
	const robot = fakeRobot();
	await f.emit("session_start");
	f.pi.events.emit(UNITS_EVENT, robot.handle);
	g.record("start");
	f.setIdle(false);
	await f.emit("agent_start");
	await f.emit("tool_result", { toolName: "act", input: { unit: "STOP" }, ...result([1, 2], []) });
	// The agent's unit is running when the operator asks: the takeover waits for it.
	await f.emit("tool_call", { toolName: "act", input: { unit: "MV_LEFT" } });
	g.control("take");
	assert.equal(g.state().mode, "requested");
	await assert.rejects(g.step({ unit: "w" }), (e: any) => e.status === 409);
	await f.emit("tool_result", { toolName: "act", input: { unit: "MV_LEFT" }, ...result([3, 4], []) });
	assert.equal(g.state().mode, "human");
	// The agent's next call is held while the operator drives.
	let held = true;
	const call = f.emit("tool_call", { toolName: "act", input: { unit: "MV_DOWN" } }).finally(() => {
		held = false;
	});
	await g.step({ command: "s e" });
	await new Promise((r) => setTimeout(r, 10));
	assert.equal(held, true);
	g.control("release");
	const decision = (await call) as { block: boolean; reason: string };
	assert.equal(decision.block, true);
	assert.match(decision.reason, /operator took over and executed 2 step\(s\): MV_BACK MV_DOWN/);
	// The agent gets the current observation (the operator's last result) as a steering message.
	const [msg] = f.sent as { content: any[]; options: any }[];
	assert.equal(msg.options.deliverAs, "steer");
	assert.equal(msg.content[1].data, png(102));
	// A marked `point` image is no camera frame: it does not become the next step's observation.
	await f.emit("tool_result", { toolName: "point", input: {}, content: [{ type: "text", text: "{}" }, image(99)] });
	// It decides again: that call runs and is recorded as the agent's.
	assert.equal(await f.emit("tool_call", { toolName: "act", input: { unit: "MV_DOWN" } }), undefined);
	await f.emit("tool_result", { toolName: "act", input: { unit: "MV_DOWN" }, ...result([5, 6], []) });
	const { dir } = g.record("save", false);
	const lines = readFileSync(join(dir, "actions.jsonl"), "utf8")
		.trim()
		.split("\n")
		.map((l) => JSON.parse(l));
	assert.deepEqual(
		lines.map((l) => [l.token, l.src, l.dagger ?? null]),
		[
			["MV_LEFT", "agent", null],
			["MV_BACK", "human", true],
			["MV_DOWN", "human", true],
			["MV_DOWN", "agent", null],
		],
	);
	assert.equal(readFileSync(join(dir, lines[3].agentview)).toString("base64"), png(102));
});
