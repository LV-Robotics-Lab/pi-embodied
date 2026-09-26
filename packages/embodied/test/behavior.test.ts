import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import behavior, { project, TASKS } from "../src/behavior/index.ts";
import { RESULT_ENTRY, toolSections } from "../src/robot.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi that records flags, tools and entries and runs handlers in registration order. */
function stubPi(values: Record<string, unknown> = {}) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	let active: string[] = [];
	const pi = {
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
	return { pi, flags, tools, entries, emit, run, active: () => active };
}

const nd = (dtype: string, shape: number[], data: Buffer) => ({ __ndarray__: data.toString("base64"), dtype, shape });
const f32 = (v: number[]) => nd("float32", [v.length], Buffer.from(Float32Array.from(v).buffer));

/**
 * A fake BEHAVIOR env server (the wire protocol of ../src/rpc.ts): 4x4 frames, a radio the grasp
 * picks up (privileged `in_hand`, `picked`), BDDL success scripted by `solveAt`.
 */
async function fakeEnv() {
	const calls: { method: string; kwargs: Record<string, unknown> }[] = [];
	const state = { steps: 0, success: false, q: 0, held: null as string | null, picked: false, solveAt: Number.NaN };
	const H = 4;
	const W = 4;
	const rgb = () => nd("uint8", [H, W, 3], Buffer.alloc(H * W * 3, 7));
	// Depth 2 m everywhere but a 1 m patch at (1, 1) and no hit at (0, 0).
	const depth = () => {
		const d = new Float32Array(H * W).fill(2);
		d[1 * W + 1] = 1;
		d[0] = Number.POSITIVE_INFINITY;
		return nd("float32", [H, W], Buffer.from(d.buffer));
	};
	const eef = (w: number) => ({
		pos: f32([0.3, 0.2, 0.9]),
		quat_xyzw: f32([0, Math.SQRT1_2, 0, Math.SQRT1_2]),
		gripper_width: w,
	});
	const obs = () => ({
		head: rgb(),
		head_depth: depth(),
		left_wrist: rgb(),
		left_wrist_depth: depth(),
		right_wrist: rgb(),
		right_wrist_depth: depth(),
		base_pos: f32([0, 0, 0]),
		base_quat_xyzw: f32([0, 0, 0, 1]),
		base_yaw: 0,
		eef: { left: eef(state.held ? 0.02 : 0.05), right: eef(0.05) },
		success: state.success,
		q_score: state.q,
		goals: { satisfied: state.success ? 1 : 0, total: 1 },
		terminated: state.success,
		truncated: false,
		env_steps: state.steps,
		privileged: { in_hand: { left: state.held, right: null }, picked: state.picked },
	});
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, kwargs = {} } = JSON.parse(body);
			calls.push({ method, kwargs });
			let result: unknown = { status: "ok" };
			const motion = (primitive: string, extra: Record<string, unknown> = {}) => {
				state.steps += 3;
				if (state.steps >= state.solveAt) {
					state.success = true;
					state.q = 1;
				}
				return { ...obs(), primitive, ok: true, phase: "done", steps: 3, ...extra };
			};
			if (method === "code.api") result = { tier: null, primitives: [], digest: "d" };
			else if (method === "env.get_env_meta")
				result = {
					task: "turning_on_radio",
					seed: 0,
					instruction: "Turn on the radio.",
					image_size: 4,
					grasping_mode: "sticky",
				};
			else if (method === "env.reset") result = [obs(), {}];
			else if (method === "env.navigate_to_pose")
				result = motion("navigate_to_pose", { reached_pos: [kwargs.x, kwargs.y, 0], reached_yaw: kwargs.yaw });
			else if (method === "env.move_hand") result = motion("move_hand", { arm: kwargs.arm, distance_left_m: 0 });
			else if (method === "env.grasp_object") {
				state.held = "radio_89";
				state.picked = true;
				result = motion("grasp_object", { arm: kwargs.arm, grasping_mode: "sticky", gripper_width: 0.02 });
			} else if (method === "env.open_gripper") {
				state.held = null;
				state.picked = false;
				result = motion("open_gripper", { arm: kwargs.arm, gripper_width: 0.05 });
			} else if (method === "env.close_gripper")
				result = motion("close_gripper", { arm: kwargs.arm, gripper_width: 0 });
			else if (method === "env.get_robot_position")
				result = { pos: [0, 0, 0], quat_xyzw: [0, 0, 0, 1], yaw: 0, eef: {} };
			else if (method === "env.get_camera_meta")
				result = {
					camera: kwargs.camera_name,
					intrinsic_K: [
						[2, 0, 2],
						[0, 2, 2],
						[0, 0, 1],
					],
					// The camera sits 1 m up, looking along world -z... as OpenGL: identity means looking along -z.
					extrinsic_cam2world: [
						[1, 0, 0, 0],
						[0, 1, 0, 0],
						[0, 0, 1, 1],
						[0, 0, 0, 1],
					],
					convention: "opengl",
					width: W,
					height: H,
				};
			else if (method === "env.ground_truth_poses")
				result = {
					frame: "world",
					poses: { "radio_receiver.n.01_1": { pos: [1, 0.5, 0.45], quat_xyzw: [0, 0, 0, 1] } },
				};
			res.end(JSON.stringify({ ok: true, result }));
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
	const close = () => {
		server.closeAllConnections();
		server.close();
	};
	return { url, calls, state, close };
}

const text = (r: any) => JSON.parse(r.content[0].text);

test("the task list is the env server's (services/.../behavior/tasks.py), CaP-X's two tasks first", () => {
	const py = readFileSync(
		new URL("../../../services/pi_embodied_services/robots/behavior/tasks.py", import.meta.url),
		"utf8",
	);
	const names = [...py.matchAll(/\(\s*"([a-zA-Z_]+)",\s*"/g)].map((m) => m[1]);
	assert.deepEqual([...TASKS], names);
	assert.equal(TASKS.length, 50);
	assert.deepEqual(TASKS.slice(0, 2), ["turning_on_radio", "picking_up_trash"]);
});

test("the tools are CaP-X's primitive set plus perception; motions carry the three camera images", async (t) => {
	const env = await fakeEnv();
	t.after(env.close);
	const s = stubPi({ env: env.url });
	behavior(s.pi);
	assert.equal(s.flags.task, "turning_on_radio");
	assert.equal(s.flags.privileged, false, "a simulated robot: --privileged exists, off by default");
	await s.emit("session_start");
	assert.deepEqual(s.active(), [
		"view_env_state",
		"get_robot_position",
		"navigate_to_pose",
		"move_hand",
		"grasp_object",
		"open_gripper",
		"close_gripper",
		"segment",
		"point",
		"back_project",
		"finish",
	]);
	assert.deepEqual(
		env.calls.map((c) => c.method).filter((m) => m !== "code.api"),
		["healthz", "env.get_env_meta", "env.reset"],
	);
	const r = await s.run("navigate_to_pose", { x: 1, y: 0.5, yaw: 1.57 });
	assert.deepEqual(env.calls.at(-1)?.kwargs, { x: 1, y: 0.5, yaw: 1.57 });
	const d = text(r);
	assert.equal(d.result.primitive, "navigate_to_pose");
	assert.equal(d.result.ok, true);
	assert.deepEqual(d.result.reached_pos, [1, 0.5, 0]);
	assert.equal(d.step, 3);
	assert.deepEqual(d.images, ["head 4x4", "left_wrist 4x4", "right_wrist 4x4"]);
	assert.equal(r.content.filter((c: any) => c.type === "image").length, 3);
	// move_hand asks the server for the arm and the pose, nothing about obstacles.
	await s.run("move_hand", { arm: "right", position: [0.5, -0.3, 1.0] });
	assert.deepEqual(env.calls.at(-1)?.kwargs, { arm: "right", position: [0.5, -0.3, 1.0], quat_xyzw: null });
	assert.equal("ignore_all_obstacles" in (env.calls.at(-1)?.kwargs ?? {}), false);
	// String choices are plain enums.
	const schema = JSON.stringify(s.tools.get("move_hand").parameters);
	assert.doesNotMatch(schema, /anyOf/);
	assert.match(schema, /"enum":\["left","right"\]/);
});

test("success is BDDL's; q_score and the reference `picked` go to the result, in_hand only under --privileged", async (t) => {
	const env = await fakeEnv();
	t.after(env.close);
	const s = stubPi({ env: env.url });
	behavior(s.pi);
	await s.emit("session_start");
	assert.equal(s.tools.has("ground_truth_poses"), false);
	let d = text(await s.run("grasp_object", { arm: "left", position: [1, 0.5, 0.45] }));
	assert.deepEqual(env.calls.at(-1)?.kwargs, {
		arm: "left",
		position: [1, 0.5, 0.45],
		quat_xyzw: null,
		pregrasp_offset_m: 0.1,
	});
	// The radio is in hand and lifted: CaP-X would call this success. BDDL does not.
	assert.equal(d.success, false);
	assert.equal(d.q_score, 0);
	assert.equal(d.state.eef.left.gripper_width, 0.02);
	assert.equal("in_hand" in d.state, false, "the object in hand is simulator knowledge");
	assert.equal(JSON.stringify(d).includes("picked"), false);
	await s.emit("agent_start");
	await s.emit("session_shutdown");
	let result = s.entries.find((e) => e.type === RESULT_ENTRY)?.data;
	assert.equal(result.success, false);
	assert.equal(result.q_score, 0);
	assert.deepEqual(result.reference, { picked: true, in_hand: { left: "radio_89", right: null } });
	assert.equal("privileged" in result, false);

	// A later BDDL success ends the episode: success true, q_score 1, and the motion tools refuse.
	const p = stubPi({ env: env.url, privileged: true });
	behavior(p.pi);
	await p.emit("session_start");
	assert.ok(p.active().includes("ground_truth_poses"));
	env.state.solveAt = env.state.steps + 1;
	d = text(await p.run("open_gripper", { arm: "left" }));
	assert.equal(d.success, true);
	assert.equal(d.q_score, 1);
	assert.deepEqual(d.state.in_hand, { left: null, right: null }, "--privileged shows the object in hand");
	d = text(await p.run("close_gripper", { arm: "left" }));
	assert.equal(d.result.error, "the task is already solved; call finish");
	const gt = await p.run("ground_truth_poses", {});
	assert.deepEqual(gt.details.poses["radio_receiver.n.01_1"].pos, [1, 0.5, 0.45]);
	await p.emit("agent_start");
	await p.emit("session_shutdown");
	result = p.entries.find((e) => e.type === RESULT_ENTRY)?.data;
	assert.equal(result.success, true);
	assert.equal(result.q_score, 1);
	assert.equal(result.privileged, true);
});

test("back_project uses the OpenGL camera convention through the camera's metric depth", async (t) => {
	// K = [[2,0,2],[0,2,2]]: pixel (row 1, col 1) at depth 1 -> camera (-0.5, +0.5, -1); the camera is 1 m up.
	const meta = {
		intrinsic_K: [
			[2, 0, 2],
			[0, 2, 2],
			[0, 0, 1],
		],
		extrinsic_cam2world: [
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 1, 1],
			[0, 0, 0, 1],
		],
		convention: "opengl",
		width: 4,
		height: 4,
	};
	const depth = new Float32Array(16).fill(2);
	depth[5] = 1;
	depth[0] = Number.POSITIVE_INFINITY;
	const xyz = project(depth, 4, 4, meta);
	assert.deepEqual(
		[...xyz.subarray(15, 18)].map((v) => Number(v.toFixed(3))),
		[-0.5, 0.5, 0],
	);
	assert.ok(Number.isNaN(xyz[0]), "no hit is NaN");
	// Row 3 (bottom) is below the optical centre: -y in the OpenGL camera.
	assert.ok(xyz[(3 * 4 + 2) * 3 + 1] < 0);

	const env = await fakeEnv();
	t.after(env.close);
	const s = stubPi({ env: env.url });
	behavior(s.pi);
	await s.emit("session_start");
	const r = await s.run("back_project", { row: 1, col: 1, camera: "left_wrist" });
	assert.equal(env.calls.at(-1)?.method, "env.get_camera_meta");
	assert.deepEqual(env.calls.at(-1)?.kwargs, { camera_name: "left_wrist" });
	// The 3x3 window around (1,1) mixes the 1 m and 2 m pixels; the median is the 2 m ring.
	assert.deepEqual(r.details.pixel, [1, 1]);
	assert.equal(r.details.world_xyz.length, 3);
	assert.equal(r.details.camera, "left_wrist");
	const bad = await s.run("back_project", { row: 9, col: 0 });
	assert.match(bad.details.error, /out of bounds/);
	const region = await s.run("back_project", { row_range: [0, 4], col_range: [0, 4] });
	assert.equal(region.details.mode, "region");
	assert.equal(region.details.n_valid, 15);
	// The world map is cached per env step: a second call on the same camera asks no meta again.
	const n = env.calls.length;
	await s.run("back_project", { row: 2, col: 2 });
	assert.equal(env.calls.length, n);
});

test("an unknown task or a non-integer seed fails closed before any server starts", async () => {
	for (const [values, why] of [
		[{ task: "make_coffee" }, /--task make_coffee is not a BEHAVIOR-1K challenge task/],
		[{ seed: "x" }, /--seed must be a task instance id/],
	] as const) {
		const s = stubPi({ ...values });
		behavior(s.pi);
		const errors: string[] = [];
		const stderr = console.error;
		console.error = (m: string) => errors.push(String(m));
		try {
			await s.emit("session_start");
		} finally {
			console.error = stderr;
		}
		assert.deepEqual(s.active(), []);
		assert.match(errors.join("\n"), why);
	}
	process.exitCode = 0;
});

test("attaching to a server with another task or instance fails closed", async (t) => {
	const env = await fakeEnv();
	t.after(env.close);
	const s = stubPi({ env: env.url, seed: "3" });
	behavior(s.pi);
	const errors: string[] = [];
	const stderr = console.error;
	console.error = (m: string) => errors.push(String(m));
	try {
		await s.emit("session_start");
	} finally {
		console.error = stderr;
	}
	assert.deepEqual(s.active(), []);
	assert.match(errors.join("\n"), /runs turning_on_radio instance 0, not turning_on_radio instance 3/);
	process.exitCode = 0;
});

test("the system prompt describes only the active tools", () => {
	const prompt = readFileSync(new URL("../src/behavior/SYSTEM.md", import.meta.url), "utf8");
	const all = toolSections(prompt, [
		"view_env_state",
		"navigate_to_pose",
		"move_hand",
		"grasp_object",
		"open_gripper",
		"close_gripper",
		"segment",
		"point",
		"back_project",
	]);
	assert.match(all, /grasp_object/);
	assert.match(all, /Molmo/);
	const noGrasp = toolSections(prompt, ["view_env_state", "navigate_to_pose", "move_hand", "segment", "back_project"]);
	assert.doesNotMatch(noGrasp, /grasp_object/);
	assert.doesNotMatch(noGrasp, /Molmo/);
	assert.doesNotMatch(noGrasp, /open_gripper/);
	assert.doesNotMatch(noGrasp, /\[tool:/);
});
