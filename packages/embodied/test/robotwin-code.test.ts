import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Duplex } from "node:stream";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import robotwin from "../src/robotwin/index.ts";

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
		registerProvider: () => {},
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

const nd = (dtype: string, shape: number[], data: Buffer) => ({ __ndarray__: data.toString("base64"), dtype, shape });
const f64 = (v: number[], shape = [v.length]) => nd("float64", shape, Buffer.from(Float64Array.from(v).buffer));
const info = (count: number, success: boolean) => ({
	robot_state: {
		left_eef_pose: f64([-0.2, 0, 0.9, 1, 0, 0, 0]),
		right_eef_pose: f64([0.2, 0, 0.9, 1, 0, 0, 0]),
		left_gripper: 1,
		right_gripper: 1,
		qpos_target14: f64(Array(14).fill(0)),
	},
	episode_status: { eval_success: success, take_action_cnt: count, step_lim: 10000, actual_seed: 100000 },
});

/** msgpack of LingBot's metadata (fixmap of fixstr keys; fixstr values and one positive fixint). */
function metadata(): Buffer {
	const str = (s: string) => Buffer.concat([Buffer.from([0xa0 | s.length]), Buffer.from(s)]);
	const fields: [string, string | number][] = [
		["runtime", "lingbotvla"],
		["policy_name", "robotwin_eef"],
		["state_layout", "eef16"],
		["action_layout", "eef16"],
		["use_length", 50],
	];
	return Buffer.concat([
		Buffer.from([0x80 | fields.length]),
		...fields.flatMap(([k, v]) => [str(k), typeof v === "number" ? Buffer.from([v]) : str(v)]),
	]);
}

/**
 * A fake RoboTwin env server (the wire protocol of ../src/rpc.ts) that also answers the LingBot
 * WebSocket handshake with the policy metadata; `code.run` reports a program that solved the task
 * in 12 native actions.
 */
async function fakeEnv() {
	const calls: { method: string; args: unknown[]; kwargs: Record<string, unknown> }[] = [];
	const sockets: Duplex[] = [];
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, args = [], kwargs = {} } = JSON.parse(body);
			calls.push({ method, args, kwargs });
			let result: unknown = { status: "ok" };
			if (method === "code.api") result = { tier: kwargs.tier ?? null, primitives: [], digest: "d" };
			else if (method === "env.get_env_meta")
				result = { task_name: "beat_block_hammer", task_config: "demo_randomized", seed: 100000 };
			else if (method === "env.reset") result = [{}, { ...info(0, false), instruction: "beat the block" }];
			else if (method === "env.render_camera")
				result = [nd("uint8", [2, 2, 3], Buffer.alloc(12)), nd("float32", [2, 2], Buffer.alloc(16))];
			else if (method === "env.get_camera_meta")
				result = {
					intrinsic_K: f64([1, 0, 1, 0, 1, 1, 0, 0, 1], [3, 3]),
					cam2world_gl: f64([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [4, 4]),
					width: 2,
					height: 2,
				};
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
					move_m: 0.2,
					ms: 5,
					steps: 12,
					info: info(12, true),
					success: true,
					budget_exhausted: false,
					frames: [nd("uint8", [2, 2, 3], Buffer.alloc(12))],
				};
			res.end(JSON.stringify({ ok: true, result }));
		});
	});
	server.on("upgrade", (req, socket) => {
		sockets.push(socket);
		const accept = createHash("sha1")
			.update(`${req.headers["sec-websocket-key"]}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`)
			.digest("base64");
		socket.write(
			`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`,
		);
		const payload = metadata();
		socket.write(Buffer.concat([Buffer.from([0x82, payload.length]), payload]));
		socket.on("error", () => {});
		// Answer the client's close frame, so its close handshake ends at once.
		socket.on("data", (d: Buffer) => {
			if ((d[0] & 0x0f) === 0x8) socket.end(Buffer.from([0x88, 0]));
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	const port = (server.address() as AddressInfo).port;
	return {
		url: `http://127.0.0.1:${port}`,
		ws: `ws://127.0.0.1:${port}`,
		calls,
		close: () => {
			for (const s of sockets) s.destroy();
			server.closeAllConnections();
			server.close();
		},
	};
}

test("--code=true: run_code runs on the env server and its result becomes a recorded state and the success", async (t) => {
	const env = await fakeEnv();
	t.after(env.close);
	// A local memory corpus: the published one would be fetched from Hugging Face.
	const base = mkdtempSync(join(tmpdir(), "robotwin-code-"));
	t.after(() => rmSync(base, { recursive: true, force: true }));
	const memory = join(base, "robotwin");
	mkdirSync(memory);
	writeFileSync(join(memory, "MEMORY.md"), "# RoboTwin\n");
	const s = stubPi({
		env: env.url,
		lingbot: env.ws,
		code: "true",
		"code-api": "low",
		"memory-profile": "local",
		"memory-dir": memory,
	});
	robotwin(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(s.active(), ["run_code", "finish"]);
	assert.deepEqual(
		env.calls.filter((c) => c.method === "code.api").map((c) => c.kwargs.tier),
		[undefined, "low"],
		"the episode's registry, then code mode's tier",
	);
	await s.emit("agent_start");
	const r = await s.run("run_code", { code: "step(list(get_state()['qpos_target']))" });
	const run = env.calls.find((c) => c.method === "code.run")!;
	assert.equal(run.kwargs.tier, "low");
	assert.equal(run.kwargs.code, "step(list(get_state()['qpos_target']))");
	assert.equal(r.details.status, "ran");
	assert.equal(r.details.eval_success, true, "the success of the run's episode status");
	assert.equal(r.details.step, 1, "a new recorded state after the reset's");
	assert.deepEqual(
		r.content.map((c: { type: string }) => c.type),
		["text", "text", "image", "image", "image"],
	);
	// As the motion tools: nothing runs once the episode is terminal.
	const again = await s.run("run_code", { code: "step(list(get_state()['qpos_target']))" });
	assert.match(again.content[0].text, /^Episode is terminal \(eval_success=true/);
	assert.equal(env.calls.filter((c) => c.method === "code.run").length, 1);
	await s.run("finish", { status: "success", summary: "hammered" });
	await s.emit("agent_end", { messages: [] });
	const result = s.entries.find((e) => e.type === "robot_result")?.data;
	assert.equal(result.success, true);
	assert.equal(result.take_action_cnt, 12);
	assert.equal(result.native_actions, 12);
	assert.equal(result.code, "true");
	assert.equal(result.code_api, "low");
	await s.emit("session_shutdown");
});
