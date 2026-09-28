import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { ground, MOVE_UNITS } from "../src/modes/units/index.ts";
import metaworld, {
	backProject,
	EMPTY_WIDTH_M,
	STEP_M,
	TASKS,
	VECTORS,
	VIEW_SETUP,
	VIEW_SIZE,
} from "../src/robots/metaworld/index.ts";
import {
	checkDetections,
	checkPoint,
	checkSimExplore,
	f32,
	fakeEnv,
	nd,
	perceptionAnswers,
	rgb,
	withPerception,
} from "./sim-stub.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi that records flags, tools and handlers; `values` override flag defaults. */
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
	const errors: string[] = [];
	// A UI context: a failed start notifies instead of setting the process exit code.
	const ctx = {
		hasUI: true,
		ui: { notify: (msg: string) => errors.push(msg) },
		shutdown: () => {},
		sessionManager: {
			getBranch: () => [],
			getSessionFile: () => undefined,
			getSessionDir: () => undefined,
			getSessionId: () => "s",
		},
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		for (const fn of handlers.get(name) ?? []) await fn({ type: name, ...event }, ctx);
	}
	const run = async (name: string, params: unknown) =>
		(await tools.get(name).execute("id", params, undefined, undefined, ctx)) as any;
	return { pi, flags, tools, entries, emit, run, errors, active: () => active };
}

const close = (a: number[], b: number[]) => a.every((x, k) => Math.abs(x - b[k]) < 1e-9);

test("the task table is Metaworld's MT50: 50 distinct *-v3 names, matching the env server's", () => {
	assert.equal(TASKS.length, 50);
	assert.equal(new Set(TASKS).size, 50);
	for (const t of TASKS) assert.match(t, /^[a-z-]+-v3$/);
	const server = readFileSync(
		new URL("../../../services/pi_embodied_services/robots/metaworld/env_server.py", import.meta.url),
		"utf8",
	);
	const table = server.slice(
		server.indexOf("INSTRUCTIONS: dict[str, str] = {"),
		server.indexOf("TASKS = list(INSTRUCTIONS)"),
	);
	const names = [...table.matchAll(/^ {4}"([a-z-]+-v3)": "/gm)].map((m) => m[1]);
	assert.deepEqual(names, [...TASKS]);
	// The three tasks the box runs are in the table.
	for (const t of ["reach-v3", "pick-place-v3", "button-press-v3"])
		assert.ok((TASKS as readonly string[]).includes(t));
});

test("flags: --task and --seed name the episode, --privileged is registered, nothing is active before start", () => {
	const f = stubPi();
	metaworld(f.pi);
	assert.equal(f.flags.task, "reach-v3");
	assert.equal(f.flags.seed, "0");
	assert.equal(f.flags.privileged, false);
	assert.equal(f.flags.units, "false");
	assert.match(String(f.flags.sam3), /^http/);
	assert.deepEqual(f.active(), []);
	for (const name of [
		"view_env_state",
		"view_camera_meta",
		"segment",
		"back_project",
		"move_delta",
		"gripper",
		"finish",
		"act",
	])
		assert.ok(f.tools.has(name), name);
	assert.equal(f.tools.has("ground_truth_poses"), false, "registered only with --privileged at start");
});

test("tool schemas: string enums for the gripper command, cameras and resolutions; a 3-vector delta", () => {
	const f = stubPi();
	metaworld(f.pi);
	const schema = (name: string) => JSON.stringify(f.tools.get(name).parameters);
	assert.match(schema("move_delta"), /"enum":\["open","close"\]/);
	assert.match(schema("move_delta"), /"minItems":3,"maxItems":3/);
	assert.match(schema("gripper"), /"enum":\["open","close"\]/);
	assert.match(schema("segment"), /"enum":\["agentview","wrist"\]/);
	assert.match(schema("back_project"), /"enum":\["low","high"\]/);
	assert.match(schema("finish"), /"enum":\["success","failure"\]/);
});

test("an unknown --task fails the start closed: no tools, an error notice", async () => {
	const f = stubPi({ task: "reach-v9" });
	metaworld(f.pi);
	await f.emit("session_start");
	assert.deepEqual(f.active(), []);
	assert.match(f.errors.join("\n"), /unknown Metaworld task "reach-v9"/);
});

test("units grounding: each MV_* is one 2 cm step along the Sawyer's world axes (+y away, -x = robot-left)", () => {
	for (const unit of MOVE_UNITS) {
		const move = ground({ vectors: VECTORS, stepM: STEP_M }, unit)!;
		assert.ok(
			close(
				move.delta,
				VECTORS[unit].map((x) => x * 0.02),
			),
			`${unit} ${move.delta}`,
		);
		assert.equal(Math.hypot(...move.delta), STEP_M);
	}
	assert.deepEqual(VECTORS.MV_FWD, [0, 1, 0]);
	assert.deepEqual(VECTORS.MV_LEFT, [-1, 0, 0]);
	assert.deepEqual(VECTORS.MV_UP, [0, 0, 1]);
	assert.ok(EMPTY_WIDTH_M > 0.023 && EMPTY_WIDTH_M < 0.04, "above the empty-close pad distance, below a held puck");
});

test("back_project: a metric depth map goes through OpenCV intrinsics and the camera-to-world transform", () => {
	// A 2x2 camera at the origin looking along +z (identity extrinsic), f = 1, principal point (1, 1).
	const k = [
		[1, 0, 1],
		[0, 1, 1],
		[0, 0, 1],
	];
	const eye = [
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1],
	];
	const xyz = backProject([2, 2, 2, 2], 2, k, eye);
	// pixel (row 0, col 0): x = (0 - 1) * 2 / 1 = -2, y = -2, z = 2
	assert.deepEqual([...xyz.slice(0, 3)], [-2, -2, 2]);
	// pixel (row 1, col 1) is on the principal point.
	assert.deepEqual([...xyz.slice(9, 12)], [0, 0, 2]);
	// A translated camera shifts every point.
	const moved = eye.map((r, i) => (i < 3 ? [...r.slice(0, 3), [0.5, 0, 1][i]] : r));
	assert.deepEqual([...backProject([2, 2, 2, 2], 2, k, moved).slice(9, 12)], [0.5, 0, 3]);
});

/** A fake Metaworld env server running reach-v3 seed 0, with the perception primitives when `perception`. */
async function fakeMetaworld(perception = false) {
	const obs = () => ({
		agentview: rgb(),
		wrist: rgb(),
		tcp_pos: f32([0, 0.6, 0.2]),
		gripper_width: 0.09,
		obs: f32([0]),
	});
	return fakeEnv((c) => {
		const p = perception ? perceptionAnswers(c) : undefined;
		if (p !== undefined) return p;
		if (c.method === "env.get_env_meta") {
			const meta = {
				task: "reach-v3",
				seed: 0,
				metaworld: "3.1.1",
				workspace: { min: [0, 0, 0], max: [1, 1, 1] },
				...VIEW_SETUP,
			};
			return perception ? withPerception(meta) : meta;
		}
		if (c.method === "env.reset") return [obs(), {}];
		if (c.method === "env.get_task_language") return "reach the goal";
		if (c.method === "env.render_camera")
			return [rgb(VIEW_SIZE, VIEW_SIZE), f32(new Array(VIEW_SIZE * VIEW_SIZE).fill(1))];
		// As the server sends it: numpy arrays (float64), not nested lists.
		if (c.method === "env.get_camera_meta")
			return {
				intrinsic_K: nd("float64", [3, 3], Buffer.from(Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]).buffer)),
				extrinsic_cam2world: nd(
					"float64",
					[4, 4],
					Buffer.from(Float64Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).buffer),
				),
			};
		return undefined;
	});
}

test("memory and exploration: reset restarts the seeded layout, the cell is metaworld_<task>_s<seed>", async (t) => {
	const env = await fakeMetaworld();
	t.after(env.close);
	await checkSimExplore({
		load: metaworld,
		values: { env: env.url, task: "reach-v3", seed: "0" },
		tag: "metaworld_reach-v3_s0",
		resets: () => env.calls.filter((c) => c.method === "env.reset").length,
		observe: "view_env_state",
	});
});

test("--detections / --unidepth: the env server's perception primitives; detect locates the centroid through the world map", async (t) => {
	const env = await fakeMetaworld(true);
	t.after(env.close);
	const s = await checkDetections({
		load: metaworld,
		values: { env: env.url, task: "reach-v3" },
		calls: env.calls,
		camera: "wrist",
	});
	const r = await s.run("detect", { prompt: "puck" });
	assert.deepEqual(r.details.detections[0].centroid_world_xyz, [1, 1, 1]);
});

test("--point: Molmo on the current images; the pixel's world xyz where the robot has depth", async (t) => {
	const env = await fakeMetaworld(true);
	t.after(env.close);
	const { one } = await checkPoint({
		load: metaworld,
		values: { env: env.url, task: "reach-v3" },
		url: env.url,
		calls: env.calls,
		cameras: ["agentview", "wrist"],
	});
	assert.deepEqual(one.details.world_xyz, [1, 1, 1]);
});

/** A fake Metaworld env server (`--env`): reach-v3 seed 0, and a `code.run` that reports `run`'s fields. */
async function fakeCodeEnv(run: Record<string, unknown>) {
	const calls: { method: string; kwargs: Record<string, unknown> }[] = [];
	const nd = (dtype: string, shape: number[], data: Buffer) => ({
		__ndarray__: data.toString("base64"),
		dtype,
		shape,
	});
	const f32 = (v: number[]) => nd("float32", [v.length], Buffer.from(Float32Array.from(v).buffer));
	const obs = (z: number) => ({
		agentview: nd("uint8", [2, 2, 3], Buffer.alloc(12)),
		wrist: nd("uint8", [2, 2, 3], Buffer.alloc(12)),
		tcp_pos: f32([0, 0.6, z]),
		gripper_width: 0.03,
		obs: f32(new Array(39).fill(0)),
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
				result = {
					task: "reach-v3",
					seed: 0,
					metaworld: "3.1.1",
					agentview: "corner4",
					wrist: "gripperPOV",
					view_size: 256,
				};
			else if (method === "env.reset") result = [obs(0.2), { success: false }];
			else if (method === "env.get_task_language") result = "reach the red ball";
			else if (method === "code.run")
				result = {
					status: "ran",
					stdout: "",
					stderr: "",
					traceback: null,
					error: null,
					result: null,
					calls: [],
					n_calls: 2,
					move_m: 0.1,
					ms: 5,
					steps: 23,
					success: true,
					obs: obs(0.1),
					info: { success: true, success_once: true, grasp_success: true },
					gripper: "close",
					frames: [nd("uint8", [2, 4, 3], Buffer.alloc(24))],
					...run,
				};
			res.end(JSON.stringify({ ok: true, result }));
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
	return {
		url,
		calls,
		close: () => {
			server.closeAllConnections();
			server.close();
		},
	};
}

test("--code=true: run_code runs on the env server and its result becomes the observation and the success", async (t) => {
	const env = await fakeCodeEnv({});
	t.after(env.close);
	const s = stubPi({ env: env.url, task: "reach-v3", code: "true", "code-api": "low" });
	metaworld(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(s.errors, []);
	assert.deepEqual(s.active(), ["run_code", "finish"]);
	assert.deepEqual(
		env.calls.filter((c) => c.method === "code.api").map((c) => c.kwargs.tier),
		[undefined, "low"],
		"the episode's registry, then code mode's tier",
	);
	await s.emit("agent_start");
	const r = await s.run("run_code", { code: "move_delta([0, 0, -0.1])" });
	const run = env.calls.find((c) => c.method === "code.run")!;
	assert.equal(run.kwargs.tier, "low");
	assert.equal(r.details.status, "ran");
	assert.equal(r.details.success, true);
	assert.equal(r.details.step, 23, "the run's control steps");
	assert.deepEqual(r.details.state.tcp_pos, [0, 0.6, 0.1], "the run's new observation");
	assert.equal(r.details.state.gripper_command, "close");
	assert.equal(r.details.state.grasp_success, true);
	assert.deepEqual(
		r.content.map((c: { type: string }) => c.type),
		["text", "text", "image", "image"],
	);
	// The episode is solved: the next program is refused, as the motion tools refuse.
	const again = await s.run("run_code", { code: "state()" });
	assert.match(again.content[0].text, /already solved/);
	assert.equal(env.calls.filter((c) => c.method === "code.run").length, 1);
	await s.run("finish", { status: "success", summary: "reached" });
	await s.emit("agent_end", { messages: [] });
	const result = s.entries.find((e) => e.type === "robot_result")?.data;
	assert.equal(result.success, true);
	assert.equal(result.env_steps, 23);
	assert.equal(result.code, "true");
	assert.equal(result.code_api, "low");
});
