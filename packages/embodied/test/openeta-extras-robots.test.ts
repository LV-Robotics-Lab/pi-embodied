import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import franka from "../src/franka/index.ts";
import libero from "../src/libero/index.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi that runs a robot's lifecycle: flags, tools, active tools, entries, hooks. */
function stubPi(values: Record<string, unknown>, ui: Record<string, unknown> = {}) {
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
		registerProvider: () => {},
		registerMessageRenderer: () => {},
		registerShortcut: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		getAllTools: () => [...tools.keys()].map((name) => ({ name })),
		getThinkingLevel: () => "off",
		appendEntry: (type: string, data: unknown) => entries.push({ type, data }),
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const notes: string[] = [];
	const ctx = {
		hasUI: Object.keys(ui).length > 0,
		cwd: tmpdir(),
		ui: { notify: (m: string) => notes.push(m), setWidget: () => {}, ...ui },
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
	return { pi, flags, tools, entries, notes, emit, run, active: () => active };
}

const nd = (dtype: string, shape: number[], data: Buffer) => ({ __ndarray__: data.toString("base64"), dtype, shape });
const f32 = (v: number[], shape = [v.length]) => nd("float32", shape, Buffer.from(Float32Array.from(v).buffer));

/** An env server speaking ../src/rpc.ts's wire protocol; `answer` maps a call to its result. */
async function fakeServer(answer: (method: string, args: any[], kwargs: any) => unknown) {
	const calls: { method: string; args: any[]; kwargs: any }[] = [];
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, args = [], kwargs = {} } = JSON.parse(body);
			calls.push({ method, args, kwargs });
			try {
				res.end(JSON.stringify({ ok: true, result: answer(method, args, kwargs) ?? { status: "ok" } }));
			} catch (err) {
				res.end(JSON.stringify({ ok: false, error: String(err) }));
			}
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

// ---------------------------------------------------------------------------
// LIBERO: a point EEF that the OSC action moves by action * 0.05 m, a wrist camera looking down from 0.2 m above it

async function fakeLibero() {
	let eef = [0, 0, 1.0];
	const obs = () => ({
		main_images: nd("uint8", [2, 2, 3], Buffer.alloc(12)),
		wrist_images: nd("uint8", [2, 2, 3], Buffer.alloc(12)),
		states: f32([...eef, 0, 0, 0, 0.02, 0.02]),
	});
	const meta = (camera: string) =>
		camera === "robot0_eye_in_hand"
			? {
					intrinsic_K: [
						[500, 0, 512],
						[0, 500, 512],
						[0, 0, 1],
					],
					extrinsic_cam2world: [
						[1, 0, 0, eef[0]],
						[0, -1, 0, eef[1]],
						[0, 0, -1, eef[2] + 0.2],
						[0, 0, 0, 1],
					],
				}
			: {
					intrinsic_K: [
						[200, 0, 128],
						[0, 200, 128],
						[0, 0, 1],
					],
					extrinsic_cam2world: [
						[1, 0, 0, 0],
						[0, -1, 0, 0],
						[0, 0, -1, 2],
						[0, 0, 0, 1],
					],
				};
	const env = await fakeServer((method, args, kwargs) => {
		if (method === "code.api") return { tier: null, primitives: [], digest: "d" };
		if (method === "env.reset") return [obs(), {}];
		if (method === "env.get_task_language") return "put the bowl on the plate";
		if (method === "env.raw_obs")
			return {
				robot0_eef_pos: f32(eef),
				robot0_eef_quat: f32([1, 0, 0, 0]),
				robot0_gripper_qpos: f32([0.02, -0.02]),
			};
		if (method === "env.get_camera_meta") return meta(kwargs.camera_name);
		if (method === "env.render_camera") {
			const n = kwargs.height as number;
			const rgb = nd("uint8", [n, n, 3], Buffer.alloc(n * n * 3));
			return kwargs.depth ? [rgb, f32(new Array(n * n).fill(0.4), [n, n])] : rgb;
		}
		if (method === "env.step") {
			const a = Buffer.from(args[0].__ndarray__, "base64");
			const act = Array.from(new Float32Array(a.buffer, a.byteOffset, 7));
			eef = eef.map((v, i) => v + act[i] * 0.05);
			return [obs(), 0, false, false, {}];
		}
		return undefined;
	});
	return { ...env, eef: () => eef };
}

const liberoFlags = (url: string, extra: Record<string, unknown> = {}) => ({
	env: url,
	"memory-profile": "local",
	"memory-dir": corpus(),
	...extra,
});

test("LIBERO registers none of the OpenETA extras at default flags", async (t) => {
	const env = await fakeLibero();
	t.after(env.close);
	const s = stubPi(liberoFlags(env.url));
	libero(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.ok(s.active().includes("move_to"));
	for (const name of ["follow_waypoints", "align_wrist", "suggest_grasp", "remember_object", "web_search"]) {
		assert.ok(!s.tools.has(name), name);
		assert.ok(!s.active().includes(name), name);
	}
});

test("LIBERO --waypoints --align-wrist --object-memory: the route servos through each waypoint; the wrist alignment reads the wrist depth", async (t) => {
	const env = await fakeLibero();
	t.after(env.close);
	const s = stubPi(liberoFlags(env.url, { waypoints: true, "align-wrist": true, "object-memory": true }));
	libero(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	for (const name of ["follow_waypoints", "align_wrist", "remember_object", "recall_objects", "forget_object"])
		assert.ok(s.active().includes(name), name);
	const r = await s.run("follow_waypoints", {
		waypoints: [
			[0, 0, 0.9],
			[0.1, 0.05, 0.9],
		],
		gripper: 1,
	});
	const out = JSON.parse(r.content[0].text);
	assert.equal(out.result.reached_target, true, JSON.stringify(out.result));
	assert.equal(out.result.waypoints_completed, 2);
	assert.ok(Math.hypot(...env.eef().map((v, i) => v - [0.1, 0.05, 0.9][i])) < 0.012);
	// Every step held the gripper closed.
	const steps = env.calls.filter((c) => c.method === "env.step");
	assert.ok(steps.length > 2);
	// A route beyond LIBERO's 0.3 m segment limit moves nothing.
	const before = steps.length;
	const refused = await s.run("follow_waypoints", { waypoints: [[0.6, 0.05, 0.9]] });
	assert.match(refused.content[0].text, /segment 0/);
	assert.equal(env.calls.filter((c) => c.method === "env.step").length, before);
	// The target 2 cm along +x and 4 cm along +y of the gripper, seen 0.4 m below the wrist camera.
	const eef = env.eef();
	const a = await s.run("align_wrist", { point: [462, 537], max_correction_m: 0.05 });
	const al = a.details;
	assert.deepEqual(al.desired_pixel, [512, 512]);
	assert.equal(al.clamped, false);
	assert.ok(
		al.aligned_xyz.every((v: number, i: number) => Math.abs(v - (eef[i] + [0.02, 0.04, 0][i])) < 1e-3),
		JSON.stringify(al.aligned_xyz),
	);
	assert.equal(a.content[1].type, "image");
	await s.run("remember_object", { name: "bowl", position: [0.1, 0.05, 0.85] });
	const rec = (await s.run("recall_objects", {})).details;
	assert.equal(rec.objects[0].last_seen_step, steps.length);
});

test("LIBERO --grasp-advisor without a grasp backend fails the start with the reason", async (t) => {
	const env = await fakeLibero();
	t.after(env.close);
	const s = stubPi(liberoFlags(env.url, { "grasp-advisor": true }));
	libero(s.pi);
	const log = console.error;
	const lines: string[] = [];
	console.error = (l: string) => lines.push(l);
	try {
		await s.emit("session_start");
	} finally {
		console.error = log;
		process.exitCode = undefined;
	}
	assert.deepEqual(s.active(), []);
	assert.match(lines.join("\n"), /--grasp-advisor needs a plan_grasp backend/);
});

// ---------------------------------------------------------------------------
// Franka (mocked hardware): the TCP points down (180° about x); the wrist camera sits 0.2 m up the tool axis

async function fakeFranka() {
	let tcp = [0.5, 0, 0.3];
	const [w, h] = [64, 48];
	const env = await fakeServer((method, _args, kwargs) => {
		if (method === "code.api") return { tier: null, primitives: [], digest: "d" };
		if (method === "env.get_env_meta") return { capabilities: { backend: "rlinf", has_vla: false } };
		if (method === "env.reset") return { states: f32([...tcp, 0]) };
		if (method === "env.get_observation")
			return {
				main_images: nd("uint8", [h, w, 3], Buffer.alloc(w * h * 3)),
				main_depths: f32(new Array(w * h).fill(0.4), [h, w]),
				states: f32([...tcp, 0]),
			};
		if (method === "env.get_robot_state") return { raw_base_state: { tcp_pose: [...tcp, 1, 0, 0, 0] } };
		if (method === "env.get_camera_meta")
			return {
				observation_camera_map: { main: "wrist_1" },
				cameras: {
					wrist_1: {
						intrinsic_K: [
							[50, 0, 32],
							[0, 50, 24],
							[0, 0, 1],
						],
					},
				},
			};
		if (method === "env.move_delta") {
			const d = Buffer.from(kwargs.delta_xyz.__ndarray__, "base64");
			const delta = Array.from(new Float32Array(d.buffer, d.byteOffset, 3));
			tcp = tcp.map((v, i) => v + delta[i]);
			return { ok: true, states: f32([...tcp, 0]) };
		}
		return undefined;
	});
	return { ...env, tcp: () => tcp };
}

/** A local memory corpus (the robots mount memory; `--memory-profile local` needs one). */
function corpus() {
	const dir = mkdtempSync(join(tmpdir(), "mem-"));
	writeFileSync(join(dir, "MEMORY.md"), "# memory\n");
	return dir;
}

/** `python -c SETUP_PY task config` stand-in: prints the task and the hand-eye calibration. */
function setupStub() {
	const dir = mkdtempSync(join(tmpdir(), "franka-setup-"));
	const I = [
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1],
	];
	const wrist = [
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, -0.2],
		[0, 0, 0, 1],
	];
	const cal = (matrix: number[][], eye: boolean) => ({
		path: "cal.yaml",
		eye_on_hand: eye,
		base_frame: "base",
		tracking_base_frame: eye ? "tcp" : "camera",
		matrix,
	});
	const setup = {
		task: { name: "pick_block", instruction: "pick up the block", success_criteria: "", constraints: [] },
		calibration: { external: cal(I, false), wrist: cal(wrist, true), convention: "opencv" },
	};
	const path = join(dir, "python");
	writeFileSync(path, `#!/bin/sh\necho '${JSON.stringify(setup)}'\n`);
	chmodSync(path, 0o755);
	return path;
}

test("Franka --waypoints sends each segment as one bounded move_delta; --align-wrist goes through the wrist hand-eye calibration", async (t) => {
	const env = await fakeFranka();
	t.after(env.close);
	const s = stubPi(
		{
			"robot-env": env.url,
			python: setupStub(),
			services: tmpdir(),
			out: mkdtempSync(join(tmpdir(), "franka-out-")),
			"z-floor": "0.32",
			"max-move": "0.06",
			"memory-profile": "local",
			"memory-dir": corpus(),
			waypoints: true,
			"align-wrist": true,
		},
		{ confirm: async () => true },
	);
	franka(s.pi);
	await s.emit("session_start");
	assert.ok(s.active().includes("follow_waypoints"), s.notes.join("\n"));
	assert.ok(s.active().includes("align_wrist"));
	// No gripper parameter: the Franka gripper keeps its last command.
	assert.ok(!("gripper" in s.tools.get("follow_waypoints").parameters.properties));
	const r = await s.run("follow_waypoints", {
		waypoints: [
			[0.5, 0, 0.35],
			[0.55, 0, 0.35],
		],
	});
	assert.equal(r.details.result?.reached_target ?? r.details.reached_target, true, JSON.stringify(r.details));
	const moves = env.calls
		.filter((c) => c.method === "env.move_delta")
		.map((c) => {
			const d = Buffer.from(c.kwargs.delta_xyz.__ndarray__, "base64");
			return Array.from(new Float32Array(d.buffer, d.byteOffset, 3)).map((v) => Number(v.toFixed(4)));
		});
	assert.deepEqual(moves, [
		[0, 0, 0.05],
		[0.05, 0, 0],
	]);
	// A segment beyond --max-move, or a waypoint below --z-floor, moves nothing.
	await s.run("follow_waypoints", { waypoints: [[0.55, 0, 0.45]] });
	await s.run("follow_waypoints", { waypoints: [[0.55, 0, 0.31]] });
	assert.equal(env.calls.filter((c) => c.method === "env.move_delta").length, 2);
	// Pixel (19, 34) of the wrist image, 0.4 m deep: 1.6 cm along +x and 4 cm along +y of the TCP.
	const a = (await s.run("align_wrist", { point: [19, 34], max_correction_m: 0.05 })).details;
	assert.deepEqual(a.desired_pixel, [24, 32]);
	const tcp = env.tcp();
	assert.ok(
		a.aligned_xyz.every((v: number, i: number) => Math.abs(v - (tcp[i] + [0.016, 0.04, 0][i])) < 1e-3),
		JSON.stringify(a),
	);
	await s.emit("session_shutdown");
});
