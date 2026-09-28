import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { ground, MOVE_UNITS } from "../src/modes/units/index.ts";
import robodojo, { ARMS, FLYWHEEL, MAX_MOVE_M, STEP_M, VECTORS, YAW_STEP_RAD } from "../src/robots/robodojo/index.ts";
import { codeApiReply } from "./helpers/code-api.ts";
import { checkDetections, checkPoint, perceptionAnswers } from "./sim-stub.ts";

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
async function fakeEnv(
	o: {
		layouts?: number;
		task?: string;
		solveAfter?: number;
		resetError?: string;
		perception?: boolean;
		/** Recorded control steps per motion: RoboDojo's success after each (Flywheel policy frames). */
		frames?: boolean[];
	} = {},
) {
	const calls: { method: string; args: unknown[]; kwargs: Record<string, unknown> }[] = [];
	const frame = (success: boolean) => ({
		head: img(),
		left_wrist: img(),
		right_wrist: img(),
		state: f32(Array(14).fill(0)),
		action: f32(Array(14).fill(0)),
		success,
		ended: success,
		truncated: false,
	});
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
				...(o.perception ? { capabilities: { perception: { segment: true, enhance_depth: true } } } : {}),
			};
			let result: unknown = { status: "ok" };
			const perceived = o.perception ? perceptionAnswers({ method, args, kwargs }) : undefined;
			if (method === "code.api") result = codeApiReply("robodojo", kwargs.tier);
			else if (method === "code.run") {
				// A program that solved the task in 12 control steps.
				motions = o.solveAfter ?? motions;
				steps += 12;
				result = {
					status: "ran",
					stdout: "",
					stderr: "",
					traceback: null,
					error: null,
					result: null,
					calls: [],
					n_calls: 1,
					move_m: 0.05,
					ms: 5,
					steps: 12,
					success: solved(),
					ended: solved(),
					truncated: false,
					obs: obs(),
					frames: [img(), img()],
				};
			} else if (perceived !== undefined) result = perceived;
			else if (method === "env.get_env_meta") result = meta;
			else if (method === "env.reset")
				result = [obs(), { instruction: meta.instruction, ...(o.resetError ? { error: o.resetError } : {}) }];
			else if (method === "env.set_recording") result = kwargs.on ? frame(false) : null;
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
				result = {
					...obs(),
					arm: a,
					moved_m: [0, 0, 0],
					executed: 1,
					control_steps: 10,
					frames: [img()],
					...(o.frames ? { policy_frames: o.frames.map(frame) } : {}),
				};
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
		"back_project",
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
	// move_to checks the same limit against the gripper's current position, before calling the server.
	const before = env.motion().length;
	await assert.rejects(s.run("move_to", { arm: "right", xyz: [0.3, 0.5, 0.95] }), /0\.5 m per call/);
	assert.equal(env.motion().length, before);
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

test("back_project (the tool once named locate) back-projects head pixels through the server", async (t) => {
	const env = await fakeEnv();
	t.after(env.close);
	const s = await start({}, env);
	const r = await s.run("back_project", { pixels: [[320, 240]] });
	assert.deepEqual(r.details.points, [{ pixel: [320, 240], xyz: [3.2, 2.4, 0.75] }]);
});

test("tool schemas come from the manifest; program-only options and code primitives are not tools", async (t) => {
	const env = await fakeEnv();
	t.after(env.close);
	const s = await start({}, env);
	const props = (name: string) => s.tools.get(name)!.parameters.properties;
	for (const name of ["move_to", "move_delta", "rotate_delta", "set_gripper"]) {
		assert.deepEqual(props(name).arm.enum, ["left", "right"], name);
		assert.ok(s.tools.get(name)!.parameters.required.includes("arm"), name);
		assert.equal(props(name).return_frames, undefined, `${name}: return_frames is a program's option`);
	}
	assert.equal(props("back_project").camera_name, undefined);
	assert.equal(props("back_project").pixels.maxItems, 32);
	for (const name of ["locate", "solve_ik", "move_to_joints", "traj_plan", "move_along_trajectory", "step"])
		assert.ok(!s.tools.has(name), name);
	assert.match(s.tools.get("move_to")!.description, /at most 0\.5 m/);
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

test("--detections / --unidepth / --point: the env server's perception and Molmo; the head's detections and points get env-frame xyz", async (t) => {
	const env = await fakeEnv({ perception: true });
	t.after(env.close);
	const s = await checkDetections({
		load: robodojo,
		values: { env: env.url },
		calls: env.calls as never,
		camera: "head",
	});
	const d = await s.run("detect", { prompt: "bowl" });
	assert.deepEqual(d.details.detections[0].centroid_xyz, [0.01, 0.01, 0.75]);
	const { one } = await checkPoint({
		load: robodojo,
		values: { env: env.url },
		url: env.url,
		calls: env.calls as never,
		cameras: ["head", "left_wrist"],
	});
	assert.deepEqual(one.details.world_xyz, [0.01, 0.01, 0.75]);
});

test("--code=true: run_code runs on the env server; its observation and success come back, then it refuses", async (t) => {
	const env = await fakeEnv({ solveAfter: 1 });
	t.after(env.close);
	const s = await start({ code: "true", "code-api": "low" }, env);
	assert.deepEqual(s.errors, []);
	assert.deepEqual(s.active(), ["run_code", "finish"]);
	assert.deepEqual(
		env.calls.filter((c) => c.method === "code.api").map((c) => c.kwargs.tier),
		[undefined, "low"],
		"the episode's registry, then code mode's tier",
	);
	await s.emit("agent_start");
	const r = await s.run("run_code", { code: "move_delta('left', [0, 0, 0.05])" });
	assert.equal(env.calls.find((c) => c.method === "code.run")?.kwargs.tier, "low");
	assert.equal(r.details.status, "ran");
	assert.equal(r.details.success, true);
	assert.equal(r.details.step, 12, "the server's step count, absorbed from the run's obs");
	assert.deepEqual(
		r.content.map((c: { type: string }) => c.type),
		["text", "text", "image", "image", "image"],
	);
	assert.doesNotMatch(r.content[1].text, /"score"/, "the evaluator's score stays out of the planner's text");
	const again = await s.run("run_code", { code: "state()" });
	assert.match(again.content[0].text, /solved/);
	assert.equal(env.calls.filter((c) => c.method === "code.run").length, 1);
});

/** A fake xpolicy bridge (`POST /call`, ../src/primitives/xpolicy.ts): `chunks` are get_action's replies in order. */
async function fakeBridge(chunks: unknown[][]) {
	const calls: { method: string; kwargs: Record<string, any> }[] = [];
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, kwargs } = JSON.parse(body);
			if (method !== "healthz") calls.push({ method, kwargs });
			const reply = (result: unknown) => res.end(JSON.stringify({ ok: true, result }));
			if (method === "healthz") return reply({ status: "ok" });
			if (method === "xpolicy.action_dims") return reply({ robot: "dual_x5", arm_dim: [6, 6], ee_dim: [1, 1] });
			if (method === "xpolicy.connect") return reply({ server_instance_id: "s", xpolicylab_rev: "d6332bf", ms: 1 });
			if (method === "xpolicy.get_action") return reply({ actions: chunks.shift() ?? [], ms: 5 });
			reply({ result: null, ms: 1 });
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
	const close = () => {
		server.closeAllConnections();
		server.close();
	};
	return { url, calls, close };
}

test("xpolicy_act sends RoboDojo's native observation and runs each action as one native action dict", async (t) => {
	const env = await fakeEnv();
	t.after(env.close);
	const q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
	const bridge = await fakeBridge([
		[
			{ left_arm_joint_state: q, left_ee_joint_state: [0.2] },
			{ action_type: "ee", right_ee_pose: [0.3, 0, 0.9, 1, 0, 0, 0] },
		],
	]);
	t.after(bridge.close);
	const s = await start(
		{ xpolicy: "ws://policy:19000", "xpolicy-bridge": bridge.url, "xpolicy-action": "joint" },
		env,
	);
	assert.ok(s.active().includes("xpolicy_act"), s.errors.join("\n"));
	assert.equal(bridge.calls.find((c) => c.method === "xpolicy.action_dims")?.kwargs.env_cfg_type, "arx_x5");
	await s.run("xpolicy_act", { chunks: 1 });
	const obs = bridge.calls.find((c) => c.method === "xpolicy.update_obs")!.kwargs.obs;
	assert.deepEqual(Object.keys(obs.vision).sort(), ["cam_head", "cam_left_wrist", "cam_right_wrist"]);
	assert.ok("left_ee_pose" in obs.state && "right_arm_joint_state" in obs.state);
	const steps = env.calls
		.filter((c) => c.method === "env.step")
		.map((c) => c.kwargs.action as Record<string, number[]>);
	assert.equal(steps.length, 2);
	// A joint action: the arm the policy left out holds its commanded joints and gripper.
	assert.deepEqual(steps[0], {
		left_arm_joint_state: q,
		left_ee_joint_state: [0.2],
		right_arm_joint_state: [0, 0, 0, 0, 0, 0],
		right_ee_joint_state: [1],
	});
	// An ee action: the other arm holds its current end-effector pose ([x, y, z, qw, qx, qy, qz]).
	const round = (v: number[]) => v.map((x) => Number(x.toFixed(4)));
	assert.deepEqual(round(steps[1].left_ee_pose), [-0.3, -0.15, 0.97, 0, 0.6, 0.8, 0]);
	assert.deepEqual(steps[1].right_ee_pose, [0.3, 0, 0.9, 1, 0, 0, 0]);
	assert.deepEqual([steps[1].left_ee_joint_state, steps[1].right_ee_joint_state], [[1], [1]]);
});

test("the Flywheel takes success per control step: a success midway through go_home keeps its whole return", async (t) => {
	const env = await fakeEnv({ frames: [false, false, true] });
	t.after(env.close);
	const root = mkdtempSync(join(tmpdir(), "robodojo-fly-"));
	const s = await start({ "collect-flywheel-data": true, "flywheel-root": root }, env);
	await s.run("go_home", {});
	await s.emit("session_shutdown");
	const dir = join(root, "raw", "robodojo", "stack_bowls", "seed_000");
	const [episode] = readdirSync(dir);
	const meta = JSON.parse(readFileSync(join(dir, episode, "episode.json"), "utf8"));
	// Stamped with the call's final success, every step would be terminated and training would keep one.
	assert.deepEqual([meta.step_count, meta.training_step_count, meta.is_success], [3, 3, true]);
});

/** eval.sh with a stand-in pi: layout 1 does not settle, the task has layouts 0..3, every other episode succeeds. */
function evalRun(seeds: string) {
	const dir = mkdtempSync(join(tmpdir(), "robodojo-eval-"));
	const pi = join(dir, "pi");
	writeFileSync(
		pi,
		`#!/usr/bin/env bash
while [ $# -gt 0 ]; do case $1 in --seed) seed=$2 ;; --session-dir) sd=$2 ;; esac; shift; done
if [ "$seed" = 1 ]; then echo "[robodojo] unavailable: RoboDojo reset: layout 1 is unstable in simulation (RoboDojo skips it); reset another seed" >&2; exit 1; fi
if [ "$seed" -ge 4 ]; then echo "[robodojo] unavailable: --seed $seed: stack_bowls has eval layouts 0..3" >&2; exit 1; fi
echo '{"type":"custom","customType":"robot_result","data":{"robot":"robodojo","success":true,"score":1,"env_steps":10}}' > "$sd/s.jsonl"
`,
	);
	chmodSync(pi, 0o755);
	const script = new URL("../src/robots/robodojo/eval.sh", import.meta.url).pathname;
	const r = spawnSync("bash", [script, join(dir, "out"), "stack_bowls", seeds], {
		env: { ...process.env, PI: pi, TIME_LIMIT: "0" },
		encoding: "utf8",
	});
	const status = (seed: number) => {
		try {
			return JSON.parse(readFileSync(join(dir, "out", `stack_bowls_s${seed}`, "result.json"), "utf8")).status;
		} catch {
			return undefined;
		}
	};
	return { r, status };
}

test("eval.sh follows RoboDojo's SeedManager: an unstable layout is not scored and the next one replaces it", () => {
	const { r, status } = evalRun("0-2");
	assert.equal(r.status, 0, r.stdout + r.stderr);
	assert.deepEqual([0, 1, 2, 3].map(status), ["success", "unstable", "success", "success"]);
	assert.match(r.stderr, /seed 1 is unstable .* layout 3 replaces it/);
	assert.match(r.stdout, /success 3\/3 \(100.0%\).*unstable 1 \(replaced\), invalid 0/);
	// When the task has no layout left to draw, the selection ends one short, and nothing is invalid.
	const short = evalRun("0-3");
	assert.equal(short.r.status, 0, short.r.stdout + short.r.stderr);
	assert.deepEqual([0, 1, 2, 3, 4].map(short.status), ["success", "unstable", "success", "success", undefined]);
	assert.match(short.r.stderr, /no layout after 3 to replace an unstable one/);
	assert.match(short.r.stdout, /success 3\/3 .*unstable 1 \(replaced\), invalid 0/);
});
