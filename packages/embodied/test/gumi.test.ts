import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { runSucceeded } from "../src/finetuned/prepare.ts";
import {
	ALIASES,
	ARM,
	actSteps,
	armState,
	DUAL,
	gumi,
	KEYS,
	keyMap,
	type Observation,
	observation,
	parseSteps,
	Recorder,
	RIGHT_ALIASES,
	RIGHT_RT_ALIASES,
	RT_ALIASES,
	RT_KEYS,
	Takeover,
	UNITS_EVENT,
	type UnitsHandle,
	views,
} from "../src/gumi/index.ts";
import { LIBERO_TURNS, rotvecToMatrix, yawOf } from "../src/libero/index.ts";
import { encodePng } from "../src/png.ts";
import { STATUS_EVENT } from "../src/robot.ts";
import { ground, RT_UNITS, type Unit } from "../src/units/index.ts";

const SINGLE = ["MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN", "STOP", "GRASP", "RELEASE", "DONE"];
const WITH_YAW = [...SINGLE, "ROTATE_CW", "ROTATE_CCW"];
/** --units-rt (RT_* instead of ROTATE_*) on a robot with every axis (LIBERO), and on one without pitch. */
const WITH_RT = [...SINGLE, ...RT_UNITS];
const NO_PITCH = SINGLE.concat(RT_UNITS.filter((u) => !u.startsWith("RT_PITCH")));
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
		gripper_closed_measured: null,
	});
	assert.deepEqual(armState({ eef_xyz: [1, 2, 3], gripper_width: 0.05, gripper_closed: true }, ARM), {
		ee_pose: [1, 2, 3],
		gripper_width: 0.05,
		gripper_closed_measured: true,
	});
	// dual_franka reports `gripper_open`.
	assert.equal(armState({ left: { gripper_open: true } }, "left").gripper_closed_measured, false);
});

test("keys: one table for the dashboard's key handler and the typed sequence aliases", () => {
	assert.deepEqual(ALIASES, {
		W: "MV_FWD",
		S: "MV_BACK",
		A: "MV_LEFT",
		D: "MV_RIGHT",
		Q: "MV_UP",
		E: "MV_DOWN",
		Z: "ROTATE_CCW",
		X: "ROTATE_CW",
		G: "GRASP",
		R: "RELEASE",
		STAY: "STILL",
	});
	assert.deepEqual(RIGHT_ALIASES, {
		I: "MV_FWD",
		K: "MV_BACK",
		J: "MV_LEFT",
		L: "MV_RIGHT",
		U: "MV_UP",
		O: "MV_DOWN",
		N: "ROTATE_CCW",
		M: "ROTATE_CW",
		".": "RELEASE",
	});
	const one = keyMap([ARM]);
	assert.deepEqual([one.KeyW, one.ArrowUp, one.KeyI], [[ARM, "MV_FWD"], [ARM, "MV_UP"], undefined]);
	const two = keyMap(DUAL);
	assert.deepEqual(
		[two.KeyW, two.ShiftLeft, two.KeyI, two.Period, two.ControlRight, two.ArrowUp],
		[["left", "MV_FWD"], ["left", "GRASP"], ["right", "MV_FWD"], ["right", "RELEASE"], ["right", "STILL"], undefined],
	);
	// Every typed right alias is a bound key of the right arm, and vice versa for its letters.
	for (const [k, u] of Object.entries(RIGHT_ALIASES))
		assert.deepEqual(two[k === "." ? "Period" : `Key${k}`], ["right", u]);
	assert.equal(Object.keys(KEYS.single).length, 14);
});

test("keys: --units-rt swaps the rotate keys for RT_* pairs, only for the axes the robot turns about", () => {
	// v3 mode (--units-rt off): Z/X rotate, no RT key or alias.
	const v3 = keyMap([ARM]);
	assert.deepEqual([v3.KeyZ, v3.KeyX, v3.Digit1], [[ARM, "ROTATE_CCW"], [ARM, "ROTATE_CW"], undefined]);
	assert.deepEqual(units(parseSteps({ command: "z" }, [ARM], WITH_YAW)), ["ROTATE_CCW"]);
	assert.throws(() => parseSteps({ command: "1" }, [ARM], WITH_YAW), /not a unit here: 1;/);
	// RT mode, every axis: Z/X yaw by physical direction (X = RT_YAW_CCW, the +z turn ROTATE_CW makes),
	// 1/2 roll, 3/4 pitch (positive turn first); nothing is ROTATE_*.
	const rt = keyMap([ARM], WITH_RT);
	assert.deepEqual(
		["KeyZ", "KeyX", "Digit1", "Digit2", "Digit3", "Digit4"].map((k) => rt[k]?.[1]),
		["RT_YAW_CW", "RT_YAW_CCW", "RT_ROLL_LEFT", "RT_ROLL_RIGHT", "RT_PITCH_FWD", "RT_PITCH_BACK"],
	);
	assert.ok(!Object.values(rt).some(([, u]) => u.startsWith("ROTATE")));
	assert.equal(Object.keys(rt).length, 14 + 4);
	assert.deepEqual(units(parseSteps({ command: "1*2 4 x RT_YAW_CW" }, [ARM], WITH_RT, true)), [
		"RT_ROLL_LEFT",
		"RT_ROLL_LEFT",
		"RT_PITCH_BACK",
		"RT_YAW_CCW",
		"RT_YAW_CW",
	]);
	assert.throws(() => parseSteps({ command: "ROTATE_CW" }, [ARM], WITH_RT, true), /not a unit here: ROTATE_CW/);
	// An axis the robot cannot turn about: its pair is unbound, typing it is refused.
	const noPitch = keyMap([ARM], NO_PITCH);
	assert.deepEqual(
		[noPitch.Digit1, noPitch.Digit3, noPitch.Digit4, noPitch.KeyZ],
		[[ARM, "RT_ROLL_LEFT"], undefined, undefined, [ARM, "RT_YAW_CW"]],
	);
	assert.throws(() => parseSteps({ command: "1 3" }, [ARM], NO_PITCH, true), /not a unit here: 3;.* 1=RT_ROLL_LEFT/);
	assert.throws(() => parseSteps({ command: "RT_PITCH_FWD" }, [ARM], NO_PITCH, true), /not a unit here: RT_PITCH_FWD/);
	// No yaw axis: Z/X are unbound (not ROTATE_*, which RT mode does not offer).
	const noYaw = SINGLE.concat(RT_UNITS.filter((u) => !u.startsWith("RT_YAW")));
	assert.deepEqual([keyMap([ARM], noYaw).KeyZ, keyMap([ARM], noYaw).Digit1], [undefined, [ARM, "RT_ROLL_LEFT"]]);
	assert.throws(() => parseSteps({ command: "z" }, [ARM], noYaw, true), /not a unit here: Z;/);
	// Two arms: the left hand's Z/X 1-4, the right hand's N/M 7 8 9 0; typed after L: / R:.
	const dualRt = [...WITH_RT, "STILL"];
	const two = keyMap(DUAL, dualRt);
	assert.deepEqual(
		[two.KeyZ, two.Digit4, two.KeyN, two.KeyM, two.Digit7, two.Digit0],
		[
			["left", "RT_YAW_CW"],
			["left", "RT_PITCH_BACK"],
			["right", "RT_YAW_CW"],
			["right", "RT_YAW_CCW"],
			["right", "RT_ROLL_LEFT"],
			["right", "RT_PITCH_BACK"],
		],
	);
	assert.deepEqual([keyMap(DUAL).KeyN, keyMap(DUAL).Digit7], [["right", "ROTATE_CCW"], undefined]);
	assert.deepEqual(parseSteps({ command: "L:3 R:m 7*2" }, DUAL, dualRt, true), [
		{ left: "RT_PITCH_FWD", right: "RT_YAW_CCW" },
		{ left: "STILL", right: "RT_ROLL_LEFT" },
		{ left: "STILL", right: "RT_ROLL_LEFT" },
	]);
	// Every RT alias is its bound key, typed.
	const code = (k: string) => (/\d/.test(k) ? `Digit${k}` : `Key${k}`);
	for (const [k, u] of Object.entries(RT_ALIASES)) assert.deepEqual(two[code(k)], ["left", u]);
	for (const [k, u] of Object.entries(RIGHT_RT_ALIASES)) assert.deepEqual(two[code(k)], ["right", u]);
	assert.equal(Object.keys(RT_KEYS.dual.right).length, 6);
});

test("keys: on LIBERO (base +z up) each rotate key turns the gripper the same physical way in both modes", () => {
	const spec = {
		vectors: {
			MV_FWD: [1, 0, 0],
			MV_BACK: [-1, 0, 0],
			MV_LEFT: [0, -1, 0],
			MV_RIGHT: [0, 1, 0],
			MV_UP: [0, 0, 1],
			MV_DOWN: [0, 0, -1],
		} as const,
		stepM: 0.02,
		...LIBERO_TURNS,
	};
	// LIBERO's yaw servo (move.yaw) measures the right-hand angle about base +z.
	assert.ok(Math.abs(yawOf([0, 0, Math.sin(0.15), Math.cos(0.15)]) - 0.3) < 1e-12);
	/** One unit's turn about base +z, rad: move.yaw, or the z-yaw of an RT_* world rotation vector. */
	const turn = (unit: string) => {
		const m = ground(spec as never, unit as Unit);
		if (!m?.rot) return m?.yaw ?? 0;
		const r = rotvecToMatrix(m.rot);
		return Math.atan2(r[1][0], r[0][0]);
	};
	// Show-Harness: ROTATE_CW is +yaw about base +Z (configs/primitives_franka.yaml `ROTATE_CW: 1.0`).
	assert.equal(turn("ROTATE_CW"), 0.15);
	assert.equal(turn("ROTATE_CCW"), -0.15);
	// RT_YAW_CCW: counter-clockwise seen from above, +z (unverified against the v5 dataset).
	assert.ok(Math.abs(turn("RT_YAW_CCW") - Math.PI / 18) < 1e-12);
	assert.ok(Math.abs(turn("RT_YAW_CW") + Math.PI / 18) < 1e-12);
	const v3 = { ...keyMap([ARM]), ...keyMap(DUAL) };
	const rt = { ...keyMap([ARM], [...SINGLE, ...RT_UNITS]), ...keyMap(DUAL, [...SINGLE, ...RT_UNITS, "STILL"]) };
	for (const k of ["KeyZ", "KeyX", "KeyN", "KeyM"]) {
		assert.ok(v3[k][1].startsWith("ROTATE") && rt[k][1].startsWith("RT_YAW"), k);
		assert.equal(Math.sign(turn(v3[k][1])), Math.sign(turn(rt[k][1])), `${k}: ${v3[k][1]} vs ${rt[k][1]}`);
	}
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
		state: { [ARM]: { ee_pose: [0.1, 0.2, 0.3], gripper_width: 0.08, gripper_closed_measured: src === "human" } },
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
			gripper_closed_measured: true,
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
			grip_measured: "CLOSED",
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
	const state = {
		left: { ee_pose: [1], gripper_width: 0.01, gripper_closed_measured: null },
		right: { ee_pose: [2], gripper_width: 0.02, gripper_closed_measured: null },
	};
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
		gripper_closed_measured: null,
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
	// The rest of that request rests on the same observation: dropped too, with no new steps.
	assert.deepEqual(await t.gate(), { stale: true, driven: [] });
	// The drop notice informed the agent; its next request decides on a fresh observation.
	t.decide();
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
	// A robot result alone does not refresh a decision already made; the next request's does.
	assert.equal((await t.gate()).stale, true);
	t.decide();
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
	/** Set to hold each unit until `next()` lets it finish. */
	let slow = false;
	const releases: (() => void)[] = [];
	const g = {
		refusal: undefined as string | undefined,
		slow: (v: boolean) => {
			slow = v;
		},
		/** Let the oldest held unit finish. */
		next: async () => {
			while (!releases.length) await new Promise((r) => setTimeout(r, 1));
			(releases.shift() as () => void)();
		},
	};
	const handle: UnitsHandle = {
		tool: "act",
		arms: [],
		vocabulary: SINGLE,
		stepM: 0.02,
		tools: () => ["act", "move_to"],
		refuse: () => g.refusal,
		run: async (params, signal) => {
			calls.push(params);
			if (slow)
				await new Promise<void>((r, reject) => {
					releases.push(r);
					signal?.addEventListener("abort", () => reject(new Error("stopped")), { once: true });
				});
			frame++;
			return result([frame, frame + 50], ["agentview_high", "wrist_high"], {
				robot0_eef_pos: [0, 0, frame / 1000],
				robot0_gripper_qpos: [0.04, -0.04],
			});
		},
		// The measured gripper never closes (the GRASP missed), whatever was commanded.
		state: async () => ({ eef_xyz: [0, 0, frame / 1000], gripper_width: 0.08, gripper_closed: false }),
	};
	return Object.assign(g, { handle, calls });
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
	await f.emit("context", { messages: [] });
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
	// After GRASP: commanded closed (the label), measured open (the robot's state), side by side.
	assert.deepEqual(
		[lines[3].gripper_closed, lines[3].gripper_closed_measured, lines[3].gripper_width],
		[true, false, 0.08],
	);
	// The dashboard gets its key bindings from the state.
	assert.deepEqual(g.state().keys.KeyG, [ARM, "GRASP"]);
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
	await f.emit("tool_execution_end", { toolName: "act" });
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
	// It decides again (a new model request): that call runs and is recorded as the agent's.
	await f.emit("context", { messages: [] });
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

/** Resolves to the tool_call decision; `held()` is true while it has not come back. */
function pending(f: ReturnType<typeof fakePi>, toolName: string, input: Record<string, unknown> = {}) {
	let done = false;
	const decision = f.emit("tool_call", { toolName, input }).finally(() => {
		done = true;
	}) as Promise<{ block: boolean; reason: string } | undefined>;
	const held = async () => {
		await new Promise((r) => setTimeout(r, 10));
		return !done;
	};
	return { decision, held };
}

test("gumi: during a takeover every robot tool waits, and is dropped as stale if the operator acted", async () => {
	const f = fakePi({});
	const g = gumi(f.pi);
	const robot = fakeRobot();
	await f.emit("session_start");
	f.pi.events.emit(UNITS_EVENT, robot.handle);
	f.setIdle(false);
	await f.emit("agent_start");
	await f.emit("tool_result", { toolName: "act", input: { unit: "STOP" }, ...result([1, 2], []) });
	g.control("take");
	assert.equal(g.state().mode, "human");
	// The robot's own motion tool (not only `act`) is held; a tool that is no robot tool is not.
	const move = pending(f, "move_to", { x: 0.4 });
	assert.equal(await move.held(), true);
	assert.equal(await f.emit("tool_call", { toolName: "read", input: {} }), undefined);
	await g.step({ command: "w g" });
	assert.equal(await move.held(), true);
	g.control("release");
	const decision = await move.decision;
	assert.equal(decision?.block, true);
	assert.match(
		decision?.reason ?? "",
		/executed 2 step\(s\): MV_FWD GRASP\. This move_to call was decided on an older/,
	);
	assert.equal(f.sent.length, 1, "the current observation is steered in");
	await f.emit("tool_execution_end", { toolName: "move_to" });
	// Taken and handed back without acting: the held call (of the next model request) runs.
	await f.emit("context", { messages: [] });
	g.control("take");
	const again = pending(f, "move_to");
	assert.equal(await again.held(), true);
	g.control("release");
	assert.equal(await again.decision, undefined);
	// A call a later handler blocked gets no tool_result; its end still frees the takeover.
	await f.emit("tool_execution_end", { toolName: "move_to" });
	g.control("take");
	assert.equal(g.state().mode, "human");
});

test("gumi: operator units pass the robot's gates (not ready, broken, ended, scene), refused with 409", async () => {
	const f = fakePi({});
	const g = gumi(f.pi);
	const robot = fakeRobot();
	await f.emit("session_start");
	f.pi.events.emit(UNITS_EVENT, robot.handle);
	for (const why of [
		"toy is not available.",
		"The robot failed: env server exited (1). The episode is over.",
		"The episode is finished.",
		"refused; request_scene_reset and obtain operator confirmation first",
	]) {
		robot.refusal = why;
		await assert.rejects(g.step({ command: "w" }), (e: any) => e.status === 409 && e.message === why);
	}
	assert.equal(robot.calls.length, 0, "nothing ran");
	assert.equal(g.state().busy, false);
	// The robot breaking mid-batch stops the batch before its next unit.
	robot.refusal = undefined;
	robot.slow(true);
	const batch = g.step({ command: "w*3" });
	await robot.next();
	robot.refusal = "The robot failed: env server exited (1). The episode is over.";
	const out = await batch;
	assert.deepEqual([out.ok, out.executed, out.results.at(-1)?.error], [false, 1, robot.refusal]);
	assert.deepEqual(
		robot.calls.map((c) => c.unit),
		["MV_FWD"],
	);
});

test("gumi: a hand-back during an operator batch takes effect when the batch ends", async () => {
	const f = fakePi({});
	const g = gumi(f.pi);
	const robot = fakeRobot();
	await f.emit("session_start");
	f.pi.events.emit(UNITS_EVENT, robot.handle);
	f.setIdle(false);
	await f.emit("agent_start");
	await f.emit("tool_result", { toolName: "act", input: { unit: "STOP" }, ...result([1, 2], []) });
	g.control("take");
	const call = pending(f, "act", { unit: "MV_DOWN" });
	robot.slow(true);
	const batch = g.step({ command: "w*2 a" });
	await robot.next();
	// Released after the first unit: the batch keeps the robot, the agent stays held.
	assert.equal(g.control("release").state.mode, "human");
	assert.equal(g.state().message, "handing back to the agent after the current batch");
	assert.equal(await call.held(), true);
	await robot.next();
	assert.equal(await call.held(), true);
	await robot.next();
	assert.equal((await batch).executed, 3);
	assert.equal(g.state().mode, "agent");
	assert.match((await call.decision)?.reason ?? "", /executed 3 step\(s\): MV_FWD MV_FWD MV_LEFT/);
	await f.emit("tool_execution_end", { toolName: "act" });
	// Without a takeover (the agent idle), an agent started mid-batch waits for the batch too.
	f.setIdle(true);
	const idle = g.step({ command: "q e" });
	await robot.next();
	const early = pending(f, "move_to");
	assert.equal(await early.held(), true);
	await robot.next();
	await idle;
	assert.equal((await early.decision)?.block, true);
});

test("gumi: a non-robot result does not refresh a decision the operator's steps overtook", async () => {
	const f = fakePi({});
	const g = gumi(f.pi);
	const robot = fakeRobot();
	await f.emit("session_start");
	f.pi.events.emit(UNITS_EVENT, robot.handle);
	f.setIdle(false);
	await f.emit("agent_start");
	await f.emit("tool_result", { toolName: "act", input: { unit: "STOP" }, ...result([1, 2], []) });
	// The model is thinking (its request started) when the operator takes over and drives.
	await f.emit("context", { messages: [] });
	g.control("take");
	await g.step({ command: "w" });
	g.control("release");
	// Its reply is [read, act]: read is no robot tool and carries no observation.
	assert.equal(await f.emit("tool_call", { toolName: "read", input: {} }), undefined);
	await f.emit("tool_result", { toolName: "read", input: {}, content: [{ type: "text", text: "notes" }] });
	const act = (await f.emit("tool_call", { toolName: "act", input: { unit: "MV_DOWN" } })) as any;
	assert.equal(act?.block, true);
	assert.match(act.reason, /executed 1 step\(s\): MV_FWD/);
	assert.deepEqual(
		robot.calls.map((c) => c.unit),
		["MV_FWD"],
	);
});

test("gumi: stop() ends an operator batch while the agent is idle", async () => {
	const f = fakePi({});
	const g = gumi(f.pi);
	const robot = fakeRobot();
	await f.emit("session_start");
	f.pi.events.emit(UNITS_EVENT, robot.handle);
	assert.equal(g.stop(), false, "no batch runs");
	robot.slow(true);
	const batch = g.step({ command: "w*5" });
	while (!robot.calls.length) await new Promise((r) => setTimeout(r, 1));
	assert.equal(g.stop(), true);
	const out = await batch;
	assert.deepEqual([out.ok, out.executed, out.results.at(-1)?.error], [false, 0, "stopped"]);
	assert.equal(robot.calls.length, 1, "no unit after the stop");
	assert.equal(g.state().busy, false);
	assert.equal(g.stop(), false);
});

test("prepare converts only runs saved as successful, unless --include-failures", () => {
	const root = mkdtempSync(join(tmpdir(), "gumi-"));
	const rec = new Recorder(root, [ARM]);
	const obs = observation(result([10, 20], ["agentview", "wrist"])) as Observation;
	const info = { src: "human" as const, dagger: false, closed: { [ARM]: false }, state: {} };
	const run = (i: number, stop?: Record<string, unknown>) => {
		const dir = rec.start({ task: "open the drawer", robot: "libero" }, i);
		rec.add(obs, { [ARM]: "MV_FWD" }, info);
		if (stop) rec.stop(stop);
		else rec.dir = undefined; // still recording when the process died: no summary.json
		return dir;
	};
	const ok = run(1, { success: true });
	const failed = run(2, { success: false, end_reason: "saved" });
	const shutdown = run(3, { success: false, end_reason: "session_shutdown" });
	const open = run(4);
	assert.deepEqual([ok, failed, shutdown, open].map(runSucceeded), [true, false, false, false]);
	const prepare = (...extra: string[]) => {
		const out = mkdtempSync(join(tmpdir(), "rollouts-"));
		const r = spawnSync(
			process.execPath,
			[
				"--experimental-strip-types",
				join(import.meta.dirname, "../src/finetuned/prepare.ts"),
				"--out",
				out,
				"--agentview",
				"raw",
				"--wrist",
				"raw",
				...extra,
				root,
			],
			{ encoding: "utf8" },
		);
		assert.equal(r.status, 0, r.stderr);
		return { rollouts: readdirSync(join(out, "open_the_drawer")).length, stderr: r.stderr };
	};
	const only = prepare();
	assert.equal(only.rollouts, 1);
	assert.match(only.stderr, /3 run\(s\) skipped as not successful/);
	assert.equal(prepare("--include-failures").rollouts, 4);
});

test("gumi: RT_* keys with --units-rt; recorded token unchanged; prepare keeps them for v5, drops them for v3", async () => {
	const root = mkdtempSync(join(tmpdir(), "gumi-"));
	const f = fakePi({ "gumi-record": root });
	const g = gumi(f.pi);
	const robot = fakeRobot();
	await f.emit("session_start");
	f.pi.events.emit(UNITS_EVENT, robot.handle);
	assert.deepEqual([g.state().rt, g.state().keys.Digit1], [false, undefined], "--units-rt off: no RT key");
	f.pi.events.emit(UNITS_EVENT, { ...robot.handle, vocabulary: NO_PITCH, rt: true });
	assert.equal(g.state().rt, true);
	assert.deepEqual(
		[g.state().keys.Digit1, g.state().keys.Digit3, g.state().keys.KeyX],
		[[ARM, "RT_ROLL_LEFT"], undefined, [ARM, "RT_YAW_CCW"]],
	);
	f.pi.events.emit(STATUS_EVENT, { robot: "libero", task: {}, language: "turn the mug", solved: false });
	g.record("start");
	const out = await g.step({ command: "w 1*2 x" });
	assert.equal(out.executed, 4);
	assert.deepEqual(
		robot.calls.map((c) => c.unit),
		["STOP", "MV_FWD", "RT_ROLL_LEFT", "RT_ROLL_LEFT", "RT_YAW_CCW"],
	);
	await assert.rejects(g.step({ command: "3" }), (e: any) => e.status === 422 && /not a unit here: 3/.test(e.message));
	const { dir } = g.record("save", true);
	const read = (file: string) =>
		readFileSync(join(dir, file), "utf8")
			.trim()
			.split("\n")
			.map((l) => JSON.parse(l));
	const actions = read("actions.jsonl");
	assert.deepEqual(
		actions.map((l) => [l.token, l.kind]),
		[
			["MV_FWD", "move"],
			["RT_ROLL_LEFT", "rotate"],
			["RT_ROLL_LEFT", "rotate"],
			["RT_YAW_CCW", "rotate"],
		],
	);
	assert.deepEqual(
		read("steps.jsonl").map((l) => l.act),
		["MV_FWD", "RT_ROLL_LEFT", "RT_ROLL_LEFT", "RT_YAW_CCW"],
	);
	assert.deepEqual(JSON.parse(readFileSync(join(dir, "metadata.json"), "utf8")).tokens, [
		"MV_FWD",
		"RT_ROLL_LEFT",
		"RT_ROLL_LEFT",
		"RT_YAW_CCW",
	]);
	const prepare = (...extra: string[]) => {
		const out = mkdtempSync(join(tmpdir(), "rollouts-"));
		const r = spawnSync(
			process.execPath,
			[
				"--experimental-strip-types",
				join(import.meta.dirname, "../src/finetuned/prepare.ts"),
				"--out",
				out,
				"--agentview",
				"raw",
				"--wrist",
				"raw",
				...extra,
				dir,
			],
			{ encoding: "utf8" },
		);
		assert.equal(r.status, 0, r.stderr);
		const rollout = join(out, "turn_the_mug", "rollout_000");
		const tokens = readFileSync(join(rollout, "actions.jsonl"), "utf8")
			.trim()
			.split("\n")
			.map((l) => JSON.parse(l).token);
		return { tokens, stderr: r.stderr, meta: JSON.parse(readFileSync(join(rollout, "metadata.json"), "utf8")) };
	};
	const v3 = prepare();
	assert.deepEqual(v3.tokens, ["MV_FWD"]);
	assert.match(v3.stderr, /dropped RT_\* steps 1, 2, 3 \(v3 has no turns/);
	const v5 = prepare("--prompt", "v5");
	assert.deepEqual(v5.tokens, ["MV_FWD", "RT_ROLL_LEFT", "RT_ROLL_LEFT", "RT_YAW_CCW"]);
	assert.doesNotMatch(v5.stderr, /dropped/);
	assert.equal(v5.meta.prompt_version, "v5");
	const bad = spawnSync(
		process.execPath,
		[
			"--experimental-strip-types",
			join(import.meta.dirname, "../src/finetuned/prepare.ts"),
			"--out",
			root,
			"--prompt",
			"v9",
			dir,
		],
		{ encoding: "utf8" },
	);
	assert.notEqual(bad.status, 0);
	assert.match(bad.stderr, /unknown prompt version v9/);
});
