import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import robolab, { MAX_ROTATE_RAD, STEP_M, VECTORS, YAW_STEP_RAD } from "../src/robolab/index.ts";
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

/** A fake env server (the wire protocol of ../src/rpc.ts) whose hand turns by exactly the yaw it is asked. */
async function fakeEnv() {
	const calls: { method: string; args: unknown[]; kwargs: Record<string, unknown> }[] = [];
	let yaw = 0;
	const nd = (dtype: string, shape: number[], data: Buffer) => ({
		__ndarray__: data.toString("base64"),
		dtype,
		shape,
	});
	const f32 = (v: number[]) => nd("float32", [v.length], Buffer.from(Float32Array.from(v).buffer));
	const obs = () => ({
		agentview: nd("uint8", [2, 2, 3], Buffer.alloc(12)),
		wrist: nd("uint8", [2, 2, 3], Buffer.alloc(12)),
		eef_pos: f32([0.4, 0, 0.3]),
		eef_quat_wxyz: f32([0, 1, 0, 0]),
		tilt_deg: 0,
		yaw_deg: (yaw * 180) / Math.PI,
		gripper_width: 0.08,
		gripper_command: "open",
		success: false,
		terminated: false,
		truncated: false,
		env_steps: calls.length,
	});
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, args = [], kwargs = {} } = JSON.parse(body);
			calls.push({ method, args, kwargs });
			const meta = {
				task: "BananaInBowlTask",
				seed: 0,
				instruction: "put the banana in the bowl",
				instruction_type: "default",
				subtask: false,
				episode_length_s: 60,
				control_hz: 15,
			};
			let result: unknown = { status: "ok" };
			if (method === "code.api") result = { tier: null, primitives: [], digest: "d" };
			else if (method === "env.get_env_meta") result = meta;
			else if (method === "env.reset") result = [obs(), {}];
			else if (method === "env.move_delta")
				result = { ...obs(), commanded_m: args[0], moved_m: args[0], decisions: 1, control_steps: 8 };
			else if (method === "env.rotate_delta") {
				yaw += Number(kwargs.yaw);
				result = {
					...obs(),
					requested_yaw: kwargs.yaw,
					commanded_yaw: kwargs.yaw,
					yaw: kwargs.yaw,
					moved_m: [0, 0, 0],
					decisions: 2,
					control_steps: 16,
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
	const motions = () => calls.filter((c) => c.method === "env.move_delta" || c.method === "env.rotate_delta");
	return { url, calls, motions, close };
}

test("each MV_* unit is one 2 cm step along the base-frame vector (-y is MV_LEFT, +x away from the base)", () => {
	for (const unit of MOVE_UNITS) {
		const move = ground({ vectors: VECTORS, stepM: STEP_M }, unit)!;
		assert.deepEqual(
			move.delta.map((v) => Number(v.toFixed(9))),
			VECTORS[unit].map((v) => v * 0.02),
		);
		assert.equal(Math.hypot(...VECTORS[unit]), 1);
	}
	assert.deepEqual(VECTORS.MV_LEFT, [0, -1, 0]);
	assert.deepEqual(VECTORS.MV_FWD, [1, 0, 0]);
});

test("ROTATE_CW grounds to +yaw about base +z (counter-clockwise seen from above), as on LIBERO; one unit is one call", () => {
	const spec = { vectors: VECTORS, stepM: STEP_M, yawStepRad: YAW_STEP_RAD };
	assert.deepEqual(ground(spec, "ROTATE_CW"), { delta: [0, 0, 0], yaw: 0.15, gripper: null });
	assert.deepEqual(ground(spec, "ROTATE_CCW"), { delta: [0, 0, 0], yaw: -0.15, gripper: null });
	assert.ok(YAW_STEP_RAD <= MAX_ROTATE_RAD);
});

test("units: ROTATE_* run as env.rotate_delta with the grounded yaw, MV_* as env.move_delta; the 150 deg cap holds", async (t) => {
	const env = await fakeEnv();
	t.after(env.close);
	const s = stubPi({ units: "true", "units-plugins": "", env: env.url });
	robolab(s.pi);
	await s.emit("session_start");
	assert.deepEqual(s.active(), ["act", "finish"]);
	assert.deepEqual(
		env.calls.map((c) => c.method).filter((m) => m !== "code.api"),
		["healthz", "env.get_env_meta", "env.reset"],
	);
	let r = await s.run("act", { unit: "ROTATE_CW", n: 2 });
	assert.match(r.content[0].text, /^units: ROTATE_CW x2\n/);
	assert.deepEqual(
		env.motions().map((c) => [c.method, c.kwargs.yaw]),
		[
			["env.rotate_delta", 0.15],
			["env.rotate_delta", 0.15],
		],
	);
	r = await s.run("act", { unit: "ROTATE_CCW" });
	assert.equal(env.motions().at(-1)?.kwargs.yaw, -0.15);
	assert.match(r.content[1].text, /"yaw_deg":-?[0-9.]+/, "the heading is part of the reported state");
	await s.run("act", { unit: "MV_FWD" });
	const fwd = env.motions().at(-1)!;
	assert.equal(fwd.method, "env.move_delta");
	assert.deepEqual(fwd.args, [[0.02, 0, 0]]);
	assert.deepEqual(fwd.kwargs, { gripper: null, return_frames: true });
	// The accumulated-yaw guard: 0.15 rad units stop at 150 deg (17 more turns from +0.15).
	for (let i = 0; i < 2; i++) await s.run("act", { unit: "ROTATE_CW", n: 10 });
	const turns = env.motions().filter((c) => c.method === "env.rotate_delta");
	assert.equal(turns.length, 3 + 16);
	const total = turns.reduce((a, c) => a + Number(c.kwargs.yaw), 0);
	assert.ok(total <= (150 * Math.PI) / 180 && total > 2.5, `total ${total} rad`);
	r = await s.run("act", { unit: "ROTATE_CW" });
	assert.match(r.content[0].text, /ROTATE_CW refused: the gripper is already turned 146 deg/);
	assert.equal(env.motions().filter((c) => c.method === "env.rotate_delta").length, 19);
	// The robot's own tool refuses a turn beyond the per-call limit before calling the server.
	await assert.rejects(s.run("rotate_delta", { yaw: 0.5 }), /the limit is 0.3 rad per call/);
	const ok = await s.run("rotate_delta", { yaw: -0.3 });
	assert.equal(env.motions().at(-1)?.kwargs.yaw, -0.3);
	assert.match(ok.content[0].text, /"commanded_yaw":-0.3/);
});

test("attaching to a server with another task, phrasing or subtask setting fails closed", async (t) => {
	const env = await fakeEnv();
	t.after(env.close);
	for (const [values, why] of [
		[{ subtask: true }, /subtask false\), not BananaInBowlTask seed 0 \(default, subtask true\)/],
		[{ "instruction-type": "vague" }, /\(default, subtask false\), not BananaInBowlTask seed 0 \(vague/],
		[{ seed: "3" }, /not BananaInBowlTask seed 3/],
	] as const) {
		const s = stubPi({ env: env.url, ...values });
		robolab(s.pi);
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

test("instruction phrasing and subtask tracking are flags, default as Show-Harness (default, off)", () => {
	const s = stubPi();
	robolab(s.pi);
	assert.equal(s.flags["instruction-type"], "default");
	assert.equal(s.flags.subtask, false);
	assert.equal(s.flags.task, "BananaInBowlTask");
	const vague = stubPi({ "instruction-type": "vague", subtask: true });
	robolab(vague.pi);
	assert.equal(vague.flags["instruction-type"], "vague");
});

test("string choices in tool schemas are plain string enums (no anyOf of literals)", () => {
	const s = stubPi();
	robolab(s.pi);
	const move = s.tools.get("move_delta");
	assert.ok(move, [...s.tools.keys()].join(","));
	const schema = JSON.stringify(move.parameters);
	assert.doesNotMatch(schema, /anyOf/);
	assert.match(schema, /"enum":\["open","close"\]/);
	assert.deepEqual(Object.keys(s.tools.get("rotate_delta").parameters.properties), ["yaw"]);
});
