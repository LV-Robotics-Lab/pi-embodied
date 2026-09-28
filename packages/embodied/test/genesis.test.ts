import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { ground, MOVE_UNITS } from "../src/modes/units/index.ts";
import genesis, { CAMERAS, maskPixels, medianPoint, STEP_M, TASKS, VECTORS } from "../src/robots/genesis/index.ts";
import {
	checkDetections,
	checkPoint,
	checkSimExplore,
	f32,
	fakeEnv,
	nd,
	perceptionAnswers,
	rgb,
	stubPi as simPi,
	withPerception,
} from "./sim-stub.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi that records flags, tools, entries and the active set, and runs handlers and tools. */
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
	const pi = new Proxy(api, { get: (t, k: string) => t[k] ?? (() => {}) }) as unknown as ExtensionAPI;
	const ctx = {
		hasUI: false,
		cwd: tmpdir(),
		ui: { notify: () => {} },
		shutdown: () => {},
		abort: () => {},
		sessionManager: {
			getBranch: () => [],
			getEntries: () => [],
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

test("the task flag defaults to cube_pick, the only task, and the simulator registers --privileged", () => {
	const f = stubPi();
	genesis(f.pi);
	assert.deepEqual([...TASKS], ["cube_pick"]);
	assert.equal(f.flags.task, "cube_pick");
	assert.equal(f.flags.seed, "0");
	assert.equal(f.flags.backend, "gpu");
	assert.equal("privileged" in f.flags, true);
	assert.equal(f.tools.has("ground_truth_poses"), false, "nothing privileged at load");
	for (const name of [
		"view_env_state",
		"view_camera_meta",
		"segment",
		"back_project",
		"move_delta",
		"gripper",
		"finish",
	])
		assert.ok(f.tools.has(name), name);
	assert.deepEqual([...CAMERAS], ["agentview", "wrist"]);
});

test("each MV_* unit is one 2 cm decision along the base-frame vector; MV_LEFT is -y", () => {
	for (const unit of MOVE_UNITS) {
		const move = ground({ vectors: VECTORS, stepM: STEP_M }, unit)!;
		assert.ok(
			move.delta.every((v, k) => Math.abs(v - VECTORS[unit][k] * 0.02) < 1e-12),
			unit,
		);
	}
	assert.deepEqual(VECTORS.MV_LEFT, [0, -1, 0]);
	assert.deepEqual(VECTORS.MV_UP, [0, 0, 1]);
});

test("segment subsamples the mask evenly and takes the median of the back-projected points", () => {
	const w = 8;
	const data = new Uint8Array(w * w);
	for (let r = 2; r < 6; r++) for (let c = 1; c < 5; c++) data[r * w + c] = 255; // a 4x4 blob
	const m = maskPixels({ width: w, height: w, data }, 4);
	assert.equal(m.n, 16);
	assert.deepEqual(m.centroid, [4, 3]); // median row / col (rounded)
	assert.equal(m.pixels.length, 4);
	assert.deepEqual(m.pixels[0], [2, 1]);
	assert.ok(m.pixels.every(([r, c]) => data[r * w + c] === 255));
	assert.deepEqual(maskPixels({ width: w, height: w, data: new Uint8Array(w * w) }), {
		n: 0,
		centroid: null,
		pixels: [],
	});
	const pts = Array.from({ length: 12 }, (_, i) => [0.5 + i * 0.001, -0.1, 0.02]);
	assert.deepEqual(medianPoint([...pts, null, [Number.NaN, 0, 0]]), [0.5055, -0.1, 0.02]);
	assert.equal(medianPoint(pts.slice(0, 5)), null, "too few points");
});

/** A fake Genesis env server running cube_pick at seed 0. */
async function fakeGenesis(perception = false) {
	const obs = () => ({
		agentview: rgb(),
		wrist: rgb(),
		tcp_pos: f32([0.4, 0, 0.3]),
		tcp_quat_wxyz: f32([0, 1, 0, 0]),
		gripper_width: 0.08,
		gripper_command: "open",
		qpos: f32([0]),
		success: false,
		is_grasped: false,
		lift_m: 0,
		env_steps: 0,
	});
	return fakeEnv((c) => {
		const p = perception ? perceptionAnswers(c) : undefined;
		if (p !== undefined) return p;
		if (c.method === "env.back_project") return [[0.4, 0, 0.02]];
		if (c.method === "env.get_env_meta")
			return (perception ? withPerception : (m: Record<string, unknown>) => m)({
				task: "cube_pick",
				seed: 0,
				instruction: "pick up the cube",
				workspace: { min: [0, 0, 0], max: [1, 1, 1] },
				z_floor_m: 0,
				max_move_m: 0.2,
				lift_m: 0.08,
			});
		if (c.method === "env.reset") return [obs(), {}];
		if (c.method === "env.move_delta")
			return { ...obs(), commanded_m: [0, 0, 0], moved_m: [0, 0, 0], decisions: 1, control_steps: 1 };
		return undefined;
	});
}

test("memory and exploration: reset restarts the seeded scene, the cell is genesis_<task>_s<seed>", async (t) => {
	const env = await fakeGenesis();
	t.after(env.close);
	await checkSimExplore({
		load: genesis,
		values: { env: env.url, task: "cube_pick", seed: "0" },
		tag: "genesis_cube_pick_s0",
		resets: () => env.calls.filter((c) => c.method === "env.reset").length,
		observe: "view_env_state",
	});
});

test("VDM is mounted over the two images every observation carries (front, then wrist)", async (t) => {
	const f = stubPi();
	genesis(f.pi);
	for (const name of ["vdm", "vdm-model", "vdm-wrist"]) assert.ok(name in f.flags, name);
	const env = await fakeGenesis();
	t.after(env.close);
	const s = simPi({ env: env.url });
	genesis(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	const r = await s.run("view_env_state", {});
	assert.deepEqual(
		r.content.map((c: { type: string }) => c.type),
		["text", "image", "image"],
	);
	assert.deepEqual(r.details.images, ["front 2x2", "wrist 2x2"]);
});

test("--detections / --unidepth: the env server's perception primitives; detect locates the centroid through the depth", async (t) => {
	const env = await fakeGenesis(true);
	t.after(env.close);
	const s = await checkDetections({ load: genesis, values: { env: env.url }, calls: env.calls, camera: "wrist" });
	const r = await s.run("detect", { prompt: "cube" });
	assert.deepEqual(r.details.detections[0].centroid_world_xyz, [0.4, 0, 0.02]);
	assert.deepEqual(env.calls.filter((c) => c.method === "env.back_project").at(-1)?.kwargs, {
		camera_name: "agentview",
		pixels: [[1, 1]],
	});
});

test("--point: Molmo on the current images; the pixel's world xyz where the robot has depth", async (t) => {
	const env = await fakeGenesis(true);
	t.after(env.close);
	const { one } = await checkPoint({
		load: genesis,
		values: { env: env.url },
		url: env.url,
		calls: env.calls,
		cameras: ["agentview", "wrist"],
	});
	assert.deepEqual(one.details.world_xyz, [0.4, 0, 0.02]);
});

test("--contact-graspnet: plan_grasp and execute_grasp, the claimed path run as move_delta calls of at most 0.2 m", async (t) => {
	let tcp = [0.4, 0, 0.3];
	const moves: number[][] = [];
	const env = await fakeEnv((c) => {
		const obs = () => ({
			agentview: rgb(),
			wrist: rgb(),
			tcp_pos: f32(tcp),
			tcp_quat_wxyz: f32([0, 1, 0, 0]),
			gripper_width: 0.08,
			gripper_command: "open",
			qpos: f32([0]),
			success: false,
			is_grasped: false,
			lift_m: 0,
			env_steps: 0,
		});
		if (c.method === "env.get_env_meta")
			return {
				task: "cube_pick",
				seed: 0,
				instruction: "pick",
				workspace: {},
				z_floor_m: 0,
				max_move_m: 0.2,
				lift_m: 0.08,
			};
		if (c.method === "env.reset") return [obs(), {}];
		if (c.method === "env.plan_grasp") return { active: "g1", candidates: [{ id: "g1" }], expired_ids: [] };
		if (c.method === "env.resolve_grasp") return { approach: [0, 0, -1] };
		if (c.method === "env.claim_waypoints")
			return {
				kind: "grasp",
				waypoints: { pre_grasp: [0.5, 0.1, 0.12], grasp: [0.5, 0.1, 0.02], lift: [0.5, 0.1, 0.12] },
				steps: [
					{ to: "pre_grasp", gripper: -1 },
					{ to: "grasp", gripper: -1 },
					{ gripper: 1 },
					{ to: "lift", gripper: 1 },
				],
				eef_yaw: 0,
				expired_ids: [],
			};
		if (c.method === "env.move_delta") {
			const d = c.args[0] as number[];
			moves.push(d);
			tcp = tcp.map((v, i) => v + d[i]);
			return { ...obs(), commanded_m: d, moved_m: d, decisions: 1, control_steps: 1 };
		}
		if (c.method === "env.set_gripper") return { ...obs(), control_steps: 5 };
		return undefined;
	});
	t.after(env.close);
	const s = simPi({ env: env.url, "contact-graspnet": "http://127.0.0.1:1" });
	genesis(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	for (const name of ["plan_grasp", "plan_place", "check_attached", "execute_grasp", "execute_place"])
		assert.ok(s.active().includes(name), name);
	assert.equal((await s.run("plan_grasp", { object: "cube" })).details.active, "g1");
	const r = await s.run("execute_grasp", { grasp_id: "g1" });
	assert.deepEqual(
		r.details.result.legs.map((l: any) => l.to ?? l.gripper),
		["pre_grasp", "grasp", "close", "lift"],
	);
	for (const d of moves) assert.ok(Math.hypot(...d) <= 0.2 + 1e-9);
	assert.deepEqual(
		env.calls.filter((c) => c.method === "env.set_gripper").map((c) => c.kwargs.open),
		[false],
	);
	assert.ok(Math.abs(tcp[2] - 0.12) < 1e-6);
	assert.match((await s.emit("before_agent_start")).systemPrompt as string, /`execute_grasp`/);

	const off = simPi({ env: env.url });
	genesis(off.pi);
	await off.emit("session_start");
	process.exitCode = undefined;
	assert.ok(!off.active().includes("execute_grasp") && !off.active().includes("plan_grasp"));
});

test("--ik: preview_reach asks env.preview_reach and nothing moves; without --ik it is inactive", async (t) => {
	const env = await fakeGenesis();
	t.after(env.close);
	const s = simPi({ env: env.url, ik: "http://127.0.0.1:1" });
	genesis(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.ok(s.active().includes("preview_reach"));
	await s.run("preview_reach", { xyz: [0.5, 0, 0.1] });
	assert.deepEqual(env.calls.find((c) => c.method === "env.preview_reach")?.kwargs, {
		pos: [0.5, 0, 0.1],
		quat_xyzw: null,
	});
	assert.ok(!env.calls.some((c) => c.method === "env.move_delta"));
	const off = simPi({ env: env.url });
	genesis(off.pi);
	await off.emit("session_start");
	process.exitCode = undefined;
	assert.ok(!off.active().includes("preview_reach"));
});

/** A fake Genesis env server (`--env`): cube_pick seed 0, and a `code.run` whose run lifts the cube. */
async function fakeCodeEnv() {
	const calls: { method: string; kwargs: Record<string, unknown> }[] = [];
	const obs = (success: boolean, steps: number) => ({
		agentview: nd("uint8", [2, 2, 3], Buffer.alloc(12)),
		wrist: nd("uint8", [2, 2, 3], Buffer.alloc(12)),
		tcp_pos: f32([0.5, 0, 0.2]),
		tcp_quat_wxyz: f32([0, 1, 0, 0]),
		gripper_width: success ? 0.038 : 0.08,
		gripper_command: success ? "close" : "open",
		qpos: f32([0, 0, 0, 0, 0, 0, 0, 0.04, 0.04]),
		success,
		is_grasped: success,
		lift_m: success ? 0.09 : 0,
		env_steps: steps,
	});
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, kwargs = {} } = JSON.parse(body);
			calls.push({ method, kwargs });
			let result: unknown = { ok: true };
			if (method === "code.api") result = { tier: kwargs.tier ?? null, primitives: [], digest: "d" };
			else if (method === "env.get_env_meta")
				result = { task: "cube_pick", seed: 0, instruction: "Pick up the red cube from the table and lift it." };
			else if (method === "env.reset") result = [obs(false, 0), {}];
			else if (method === "code.run")
				result = {
					status: "ran",
					stdout: "",
					stderr: "",
					traceback: null,
					error: null,
					result: null,
					calls: [],
					n_calls: 3,
					move_m: 0.33,
					ms: 5,
					steps: 120,
					success: true,
					obs: obs(true, 120),
					frames: [nd("uint8", [2, 4, 3], Buffer.alloc(24))],
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
	return { url, calls, close };
}

test("--code=true: run_code runs on the env server; its obs is absorbed into the state and robot_result", async (t) => {
	const env = await fakeCodeEnv();
	t.after(env.close);
	const s = stubPi({ env: env.url, code: "true", "code-api": "low" });
	genesis(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(s.active(), ["run_code", "finish"]);
	await s.emit("agent_start");
	const r = await s.run("run_code", { code: "move_delta([0, 0, -0.18], gripper='open')" });
	const run = env.calls.find((c) => c.method === "code.run")!;
	assert.equal(run.kwargs.tier, "low");
	assert.equal(run.kwargs.code, "move_delta([0, 0, -0.18], gripper='open')");
	assert.equal(r.details.status, "ran");
	assert.equal(r.details.success, true);
	assert.equal(r.details.step, 120, "the server's step count, absorbed from the run's obs");
	assert.equal(r.details.state.is_grasped, true);
	assert.deepEqual(
		r.content.map((c: { type: string }) => c.type),
		["text", "text", "image", "image"],
	);
	// The task is solved: like move_delta, run_code refuses and nothing reaches the server.
	const again = await s.run("run_code", { code: "move_delta([0, 0, 0.05])" });
	assert.match(again.content[0].text, /already solved/);
	assert.equal(env.calls.filter((c) => c.method === "code.run").length, 1);
	await s.run("finish", { status: "success", summary: "lifted" });
	await s.emit("agent_end", { messages: [] });
	const result = s.entries.find((e) => e.type === "robot_result")?.data;
	assert.equal(result.success, true);
	assert.equal(result.ever_grasped, true);
	assert.equal(result.env_steps, 120);
	assert.equal(result.code, "true");
	assert.equal(result.code_api, "low");
});
