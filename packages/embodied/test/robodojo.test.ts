import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import robodojo, { ARMS, FLYWHEEL, MAX_MOVE_M, STEP_M, VECTORS, YAW_STEP_RAD } from "../src/robodojo/index.ts";
import { ground, MOVE_UNITS } from "../src/units/index.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi that records flags and tools and runs handlers in registration order (no env server is started). */
function stubPi(values: Record<string, unknown> = {}) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	let active: string[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		registerProvider: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const ctx = {
		hasUI: false,
		cwd: tmpdir(),
		ui: { notify: () => {} },
		shutdown: () => {},
		sessionManager: {
			getBranch: () => [],
			getSessionDir: () => tmpdir(),
			getSessionFile: () => undefined,
			getSessionId: () => "sess",
		},
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) result = (await fn({ type: name, ...event }, ctx)) ?? result;
		return result;
	}
	const run = async (name: string, params: unknown) =>
		(await tools.get(name).execute("id", params, undefined, undefined, ctx)) as any;
	return { pi, flags, tools, emit, run, active: () => active };
}

/** A fake RoboDojo env server (the wire protocol of ../src/rpc.ts): moves land exactly, `solveAfter` motions succeed. */
async function fakeEnv(o: { layouts?: number; task?: string; solveAfter?: number; resetError?: string } = {}) {
	const calls: { method: string; args: unknown[]; kwargs: Record<string, unknown> }[] = [];
	const pos: Record<string, number[]> = { left: [-0.3, -0.15, 0.97], right: [0.3, -0.15, 0.97] };
	let steps = 0;
	let motions = 0;
	const nd = (dtype: string, shape: number[], data: Buffer) => ({
		__ndarray__: data.toString("base64"),
		dtype,
		shape,
	});
	const f32 = (v: number[]) => nd("float32", [v.length], Buffer.from(Float32Array.from(v).buffer));
	const img = () => nd("uint8", [2, 2, 3], Buffer.alloc(12));
	const solved = () => o.solveAfter !== undefined && motions >= o.solveAfter;
	const arm = (a: string) => ({
		eef_pos: f32(pos[a]),
		eef_quat_wxyz: f32([0, 0.6, 0.8, 0]),
		tcp_pos: f32([pos[a][0], pos[a][1] + 0.145, pos[a][2]]),
		joints: f32([0, 0, 0, 0, 0, 0]),
		joints_command: f32([0, 0, 0, 0, 0, 0]),
		gripper: 1,
		gripper_command: 1,
	});
	const state = () => ({
		arms: { left: arm("left"), right: arm("right") },
		success: solved(),
		ended: solved(),
		truncated: false,
		score: solved() ? 1 : 0.15,
		env_steps: steps,
		step_lim: 800,
		seed: 0,
	});
	const obs = () => ({ head: img(), left_wrist: img(), right_wrist: img(), ...state() });
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, args = [], kwargs = {} } = JSON.parse(body);
			calls.push({ method, args, kwargs });
			const meta = {
				task: o.task ?? "stack_bowls",
				seed: 0,
				eval_seed: 0,
				dimension: "generalization",
				layouts: o.layouts ?? 85,
				instruction: "Stack the three bowls together.",
				step_lim: 800,
			};
			let result: unknown = { status: "ok" };
			if (method === "code.api") result = { tier: null, primitives: [], digest: "d" };
			else if (method === "env.get_env_meta") result = meta;
			else if (method === "env.reset")
				result = [obs(), { instruction: meta.instruction, ...(o.resetError ? { error: o.resetError } : {}) }];
			else if (method === "env.set_recording") result = null;
			else if (method === "env.get_obs")
				result = {
					instruction: meta.instruction,
					vision: {
						cam_head: { color: img() },
						cam_left_wrist: { color: img() },
						cam_right_wrist: { color: img() },
					},
					state: {
						left_arm_joint_state: f32([0, 0, 0, 0, 0, 0]),
						left_ee_joint_state: [1],
						left_ee_pose: f32([...pos.left, 0, 0.6, 0.8, 0]),
						right_arm_joint_state: f32([0, 0, 0, 0, 0, 0]),
						right_ee_joint_state: [1],
						right_ee_pose: f32([...pos.right, 0, 0.6, 0.8, 0]),
					},
				};
			else if (method === "env.step") {
				steps++;
				result = [obs(), 0, false, false, {}];
			} else if (method === "env.state") result = state();
			else if (method === "env.render_camera") result = img();
			else if (method === "env.back_project")
				result = { frame: "env", xyz: (kwargs.pixels as number[][]).map(([c, r]) => [c / 100, r / 100, 0.75]) };
			else if (method.startsWith("env.")) {
				motions++;
				steps += 10;
				const a = String(kwargs.arm ?? "left");
				if (method === "env.move_delta") pos[a] = pos[a].map((v, i) => v + (kwargs.delta_xyz as number[])[i]);
				if (method === "env.move_to") pos[a] = kwargs.xyz as number[];
				result = { ...obs(), arm: a, moved_m: [0, 0, 0], executed: 1, control_steps: 10, frames: [img()] };
			}
			res.end(JSON.stringify({ ok: true, result }));
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
	const close = () => {
		server.closeAllConnections();
		server.close();
	};
	const motion = () => calls.filter((c) => /^env\.(move|rotate|set_gripper|go_home)/.test(c.method));
	return { url, calls, motion, close };
}

async function start(values: Record<string, unknown>, env: Awaited<ReturnType<typeof fakeEnv>>) {
	const s = stubPi({ env: env.url, ...values });
	robodojo(s.pi);
	const errors: string[] = [];
	const stderr = console.error;
	console.error = (m: string) => errors.push(String(m));
	try {
		await s.emit("session_start");
	} finally {
		console.error = stderr;
	}
	process.exitCode = 0;
	return { ...s, errors };
}

test("each MV_* unit is one 2 cm step in the env frame (+y away from the robot, -x the robot's left)", () => {
	for (const unit of MOVE_UNITS) {
		const move = ground({ vectors: VECTORS, stepM: STEP_M }, unit)!;
		assert.deepEqual(
			move.delta.map((v) => Number(v.toFixed(9))),
			VECTORS[unit].map((v) => v * 0.02),
		);
		assert.equal(Math.hypot(...VECTORS[unit]), 1);
	}
	assert.deepEqual(VECTORS.MV_FWD, [0, 1, 0]);
	assert.deepEqual(VECTORS.MV_LEFT, [-1, 0, 0]);
	assert.equal(ground({ vectors: VECTORS, stepM: STEP_M, yawStepRad: YAW_STEP_RAD }, "ROTATE_CW")?.yaw, 0.15);
});

test("the Flywheel records RoboDojo's joint space: 2 x (6 joints + gripper) and the three cameras", () => {
	assert.equal(FLYWHEEL.state, 14);
	assert.equal(FLYWHEEL.action, 14);
	assert.deepEqual(Object.keys(FLYWHEEL.images), ["head_images", "left_wrist_images", "right_wrist_images"]);
});

test("a session resets the cell's layout and activates the per-arm tools", async (t) => {
	const env = await fakeEnv();
	t.after(env.close);
	const s = await start({ seed: "3" }, env);
	// The robot's tools, then memory's file tools.
	assert.deepEqual(s.active().slice(0, 8), [
		"view_env_state",
		"move_to",
		"move_delta",
		"rotate_delta",
		"set_gripper",
		"go_home",
		"locate",
		"finish",
	]);
	const reset = env.calls.find((c) => c.method === "env.reset")!;
	assert.deepEqual(reset.kwargs, { seed: 3 });
	assert.deepEqual(env.calls.find((c) => c.method === "env.set_recording")?.kwargs, { on: false });
	const r = await s.run("move_delta", { arm: "right", delta_xyz: [0, 0.05, -0.02] });
	const call = env.motion().at(-1)!;
	assert.equal(call.method, "env.move_delta");
	assert.deepEqual(call.kwargs, { arm: "right", delta_xyz: [0, 0.05, -0.02], gripper: null, return_frames: true });
	const details = JSON.parse(r.content[0].text);
	assert.deepEqual(details.state.right.eef_xyz, [0.3, -0.1, 0.95]);
	assert.equal(details.remaining_steps, 790);
	assert.equal(r.content.filter((c: any) => c.type === "image").length, 3);
	// RoboDojo's partial-credit score is the evaluator's: in the details, never in the planner's text.
	assert.equal(r.details.score, 0.15);
	assert.equal("score" in details, false);
	await assert.rejects(s.run("move_delta", { arm: "left", delta_xyz: [0, 0.6, 0] }), /limit is 0.5 m per call/);
	assert.equal(MAX_MOVE_M, 0.5);
});

test("units: a unit moves the arm it names, its gripper first", async (t) => {
	const env = await fakeEnv();
	t.after(env.close);
	const s = await start({ units: "true", "units-plugins": "" }, env);
	assert.deepEqual(s.active(), ["act", "finish"]);
	await s.run("act", { unit: "MV_FWD", arm: "left" });
	assert.deepEqual(env.motion().at(-1)?.kwargs, {
		arm: "left",
		delta_xyz: [0, 0.02, 0],
		gripper: null,
		return_frames: true,
	});
	await s.run("act", { unit: "ROTATE_CW", arm: "right" });
	assert.deepEqual(env.motion().at(-1)?.kwargs, { arm: "right", yaw: 0.15, return_frames: true });
	assert.deepEqual(ARMS, ["left", "right"]);
});

test("the episode ends on RoboDojo's success: motions are answered without a call", async (t) => {
	const env = await fakeEnv({ solveAfter: 1 });
	t.after(env.close);
	const s = await start({}, env);
	await s.run("go_home", {});
	const n = env.motion().length;
	const r = await s.run("set_gripper", { arm: "left", value: 1 });
	assert.equal(env.motion().length, n);
	assert.match(r.content[0].text, /the task is solved; call finish/);
});

test("locate back-projects head pixels through the server", async (t) => {
	const env = await fakeEnv();
	t.after(env.close);
	const s = await start({}, env);
	const r = await s.run("locate", { pixels: [[320, 240]] });
	assert.deepEqual(r.details.points, [{ pixel: [320, 240], xyz: [3.2, 2.4, 0.75] }]);
});

test("a server running another task, a seed beyond its layouts or an unstable layout fails closed", async (t) => {
	for (const [values, o, why] of [
		[{}, { task: "push_T" }, /runs push_T \(eval seed 0\), not stack_bowls/],
		[{ seed: "25" }, { layouts: 25 }, /--seed 25: stack_bowls has eval layouts 0..24/],
		[{}, { resetError: "layout 0 is unstable" }, /RoboDojo reset: layout 0 is unstable/],
	] as const) {
		const env = await fakeEnv(o);
		t.after(env.close);
		const s = await start(values, env);
		assert.deepEqual(s.active(), []);
		assert.match(s.errors.join("\n"), why);
	}
});

test("flags default to the benchmark's first layout set on GPU 0", () => {
	const s = stubPi();
	robodojo(s.pi);
	assert.equal(s.flags.task, "stack_bowls");
	assert.equal(s.flags.seed, "0");
	assert.equal(s.flags["eval-seed"], "0");
	assert.equal(s.flags["cuda-device"], "0");
	assert.equal(s.flags.privileged, false, "a simulator: --privileged is registered");
	const schema = JSON.stringify(s.tools.get("move_to").parameters);
	assert.doesNotMatch(schema, /anyOf/);
	assert.match(schema, /"enum":\["left","right"\]/);
});
