import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import franka, { DETECTIONS_ENTRY } from "../src/franka/index.ts";

/** A stub pi that only records registrations (no robot starts: tools are inspected, not run). */
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
	return { pi, flags, tools };
}

test("franka segment takes an `all` flag and its selection tools take a detection id", () => {
	const { pi, flags, tools } = fakePi();
	franka(pi);
	const segment = tools.get("segment");
	assert.ok(segment, "segment is registered");
	const props = segment.parameters.properties;
	assert.deepEqual(Object.keys(props).sort(), ["all", "camera", "min_score", "point", "prompt"]);
	assert.equal(props.all.type, "boolean");
	assert.equal(props.point.minItems, 2);
	assert.deepEqual(props.camera.enum, ["wrist", "third_person"]);
	assert.deepEqual(segment.parameters.required ?? [], [], "prompt and point are both optional");
	assert.match(segment.description, /d3/, "the description names the short id form");
	for (const name of ["select_detection", "reject_detection"]) {
		const t = tools.get(name);
		assert.ok(t, `${name} is registered`);
		assert.deepEqual(Object.keys(t.parameters.properties), ["id"]);
		assert.equal(t.parameters.properties.id.type, "string");
		assert.deepEqual(t.parameters.required, ["id"]);
	}
	assert.deepEqual(Object.keys(tools.get("enhance_depth").parameters.properties), ["camera"]);
	// The perception services are opt-in flags; off by default.
	assert.equal(flags["robot-sam3"], undefined);
	assert.equal(flags["robot-unidepth"], undefined);
	assert.equal(DETECTIONS_ENTRY, "robot_detections");
});

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi with an operator UI: `confirms` answers ui.confirm in order (the reset, then each program). */
function operatorPi(values: Record<string, unknown>, confirms: boolean[]) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	const asked: string[] = [];
	const notes: string[] = [];
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
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const dir = mkdtempSync(join(tmpdir(), "franka-code-"));
	const ctx = {
		hasUI: true,
		cwd: dir,
		ui: {
			notify: (m: string) => notes.push(m),
			setWidget: () => {},
			select: async () => undefined,
			confirm: async (title: string) => {
				asked.push(title);
				return confirms.shift() ?? false;
			},
		},
		shutdown: () => {},
		abort: () => {},
		sessionManager: {
			getBranch: () => [],
			getEntries: () => [],
			getSessionDir: () => dir,
			getSessionFile: () => undefined,
			getSessionId: () => "sess",
		},
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) {
			const r: any = await fn({ type: name, ...event }, ctx);
			if (r !== undefined) result = r;
		}
		return result;
	}
	const run = async (name: string, params: unknown) =>
		(await tools.get(name).execute("id", params, undefined, undefined, ctx)) as any;
	return { pi, emit, run, entries, asked, notes, dir, active: () => active };
}

/** A fake franka env server (`--robot-env`, RLinf capabilities) whose `code.run` made two motions. */
async function fakeFranka() {
	const calls: { method: string; kwargs: Record<string, any> }[] = [];
	const nd = (shape: number[]) => ({
		__ndarray__: Buffer.alloc(shape.reduce((a, b) => a * b, 1)).toString("base64"),
		dtype: "uint8",
		shape,
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
			if (method === "env.get_env_meta") result = { ok: true, capabilities: { backend: "rlinf", has_vla: true } };
			else if (method === "env.reset") result = { ok: true, states: [1, 2] };
			else if (method === "env.get_observation")
				result = { main_images: nd([2, 2, 3]), extra_view_images: nd([1, 2, 2, 3]) };
			else if (method === "env.get_robot_state")
				result = { raw_base_state: { tcp_pose: [0.5, 0, 0.3, 1, 0, 0, 0], gripper_open: true } };
			else if (method === "env.get_camera_meta") result = null;
			else if (method === "code.api") result = { tier: kwargs.tier ?? null, primitives: [], digest: "d" };
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
					move_m: 0.02,
					ms: 5,
					motions: 2,
					states: [3, 4],
					frames: [nd([2, 2, 3]), nd([2, 2, 3])],
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

/** A stand-in for the services' Python: SETUP_PY's JSON (task 1, no calibration). */
function fakePython(dir: string) {
	const py = join(dir, "python");
	const setup = {
		task: {
			name: "pick_block",
			instruction: "pick up the block",
			success_criteria: "block lifted",
			constraints: ["Keep translation commands at or below 0.04 m per call."],
		},
		calibration_error: "none in this test",
	};
	writeFileSync(py, `#!/usr/bin/env bash\necho '${JSON.stringify(setup)}'\n`);
	chmodSync(py, 0o755);
	return py;
}

const CODE_FLAGS = { code: "true", "code-real": true, operator: true, task: "1", "z-floor": "0.14" };

test("franka --code: pi's limits reach the server, every program is confirmed and the run is a state step", async (t) => {
	const env = await fakeFranka();
	t.after(env.close);
	const dir = mkdtempSync(join(tmpdir(), "franka-py-"));
	const f = operatorPi({ ...CODE_FLAGS, "robot-env": env.url, python: fakePython(dir), out: dir }, [
		true,
		true,
		false,
	]);
	franka(f.pi);
	await f.emit("session_start");
	process.exitCode = undefined;
	assert.ok(f.active().includes("run_code"), f.active().join(","));
	const limits = env.calls.find((c) => c.method === "code.set_limits")?.kwargs;
	assert.deepEqual(limits, {
		max_move_m: 0.04,
		max_rotate_rad: 0.5,
		z_floor_m: 0.14,
		workspace_xy: [0.159, 1.159, -0.456, 0.544],
	});
	await f.emit("agent_start");
	const r = await f.run("run_code", { code: "move_delta([0, 0, -0.02])" });
	assert.equal(f.asked[1], "Run this program on the robot?");
	assert.equal(r.details.status, "ran");
	const run = env.calls.find((c) => c.method === "code.run")!;
	assert.equal(run.kwargs.code, "move_delta([0, 0, -0.02])");
	const text = r.content.map((c: any) => c.text ?? "").join("\n");
	assert.match(text, /"action": "run_code"/, "the run is recorded as the next state step");
	assert.match(text, /"wrapped_state_vector": \[\s*3,\s*4\s*\]/, "the run's states refresh pi's cache");
	// The operator declines the next program: it never reaches the server.
	const no = await f.run("run_code", { code: "move_delta([0, 0, 0.02])" });
	assert.match(no.content[0].text, /operator declined/);
	assert.equal(env.calls.filter((c) => c.method === "code.run").length, 1);
});

test("franka --code without --code-real or --operator refuses before touching the robot", async (t) => {
	const env = await fakeFranka();
	t.after(env.close);
	const dir = mkdtempSync(join(tmpdir(), "franka-py-"));
	for (const drop of ["code-real", "operator"]) {
		const flags: Record<string, unknown> = { ...CODE_FLAGS, "robot-env": env.url, python: fakePython(dir), out: dir };
		delete flags[drop];
		const f = operatorPi(flags, [true]);
		franka(f.pi);
		await f.emit("session_start");
		assert.match(f.notes.join("\n"), /needs both --code-real and --operator/);
		assert.ok(!f.active().includes("run_code"));
	}
	assert.equal(env.calls.filter((c) => c.method === "env.reset").length, 0);
});
