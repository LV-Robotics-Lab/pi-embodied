import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { chmodSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { RpcClient } from "../src/infra/rpc.ts";
import { parsePorts } from "../src/observation/viser.ts";
import { defineRobot, type RobotStatus, STATUS_EVENT } from "../src/robot.ts";
import { deployFlags } from "./helpers/deployment.ts";

type Handler = (event: any, ctx: any) => unknown;

function fakePi(flagValues: Record<string, unknown>) {
	flagValues = deployFlags(flagValues);
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const bus = new EventEmitter();
	const notes: string[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: () => {},
		registerCommand: () => {},
		setActiveTools: () => {},
		getActiveTools: () => [],
		appendEntry: () => {},
		getThinkingLevel: () => "low",
		events: {
			emit: (channel: string, data: unknown) => bus.emit(channel, data),
			on: (channel: string, fn: (data: unknown) => void) => {
				bus.on(channel, fn);
				return () => bus.off(channel, fn);
			},
		},
	} as unknown as ExtensionAPI;
	const ctx = {
		hasUI: true,
		ui: { notify: (m: string) => notes.push(m), setWidget: () => {} },
		shutdown: () => {},
		signal: undefined,
		sessionManager: { getBranch: () => [], getSessionDir: () => "/tmp" },
		model: { provider: "relay", id: "planner" },
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) {
			const r: any = await fn({ type: name, ...event }, ctx);
			if (r !== undefined) result = r;
		}
		return result;
	}
	return { pi, emit, handlers, bus, notes };
}

/**
 * A stand-in for viser_view.py: a node script that records its argv, prints the two listening
 * lines, answers POST /call (recording each call) and exits when stdin closes (--parent-watch).
 */
function fakeViewer(dir: string) {
	const script = join(dir, "python");
	writeFileSync(
		script,
		`#!/usr/bin/env node
const { createServer } = require("node:http");
const { appendFileSync, writeFileSync } = require("node:fs");
writeFileSync(${JSON.stringify(join(dir, "argv"))}, JSON.stringify(process.argv.slice(2)));
writeFileSync(${JSON.stringify(join(dir, "token"))}, process.env.PI_EMBODIED_ENV_TOKEN ?? "");
const server = createServer((req, res) => {
	let body = "";
	req.on("data", (c) => (body += c));
	req.on("end", () => {
		appendFileSync(${JSON.stringify(join(dir, "calls"))}, body + "\\n");
		res.end(JSON.stringify({ ok: true, result: { drawn: 1 } }));
	});
});
server.listen(0, "127.0.0.1", () => {
	console.log("viser listening on http://0.0.0.0:18123");
	console.error("RPC server listening on http://127.0.0.1:" + server.address().port);
});
process.stdin.on("end", () => {
	writeFileSync(${JSON.stringify(join(dir, "exited"))}, "1");
	process.exit(0);
});
process.stdin.resume();
`,
	);
	chmodSync(script, 0o755);
	return script;
}

const finish = {
	description: "finish",
	parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
	result: (p: any) => ({ content: [{ type: "text" as const, text: p.status }], details: p }),
};

async function toy(flags: Record<string, unknown>, viser = true) {
	const f = fakePi(flags);
	const statuses: RobotStatus[] = [];
	f.bus.on(STATUS_EVENT, (s) => statuses.push(s as RobotStatus));
	defineRobot(f.pi, {
		name: "toy",
		task: [],
		keepImages: 4,
		start: async () => ["finish"],
		result: () => ({}),
		finish,
		...(viser
			? {
					viser: {
						source: "libero" as const,
						env: () => Object.assign(new RpcClient("http://127.0.0.1:4567"), { token: "f00d" }),
					},
				}
			: {}),
	});
	await f.emit("session_start");
	return { ...f, statuses };
}

test("parsePorts reads the page's and the RPC server's ports", () => {
	assert.deepEqual(parsePorts("x\nviser listening on http://0.0.0.0:8081\n"), { viser: 8081, rpc: undefined });
	assert.deepEqual(parsePorts("RPC server listening on http://127.0.0.1:40000 viser listening on http://h:8080"), {
		viser: 8080,
		rpc: 40000,
	});
});

test("without --viser nothing starts and no hook is registered; a robot without the spec has no flag", async () => {
	const dir = mkdtempSync(join(tmpdir(), "viser-"));
	const f = await toy({ "viser-python": fakeViewer(dir) });
	assert.equal(existsSync(join(dir, "argv")), false, "no view process");
	assert.equal(f.handlers.get("tool_result"), undefined);
	assert.equal(
		f.statuses.some((s) => "viser_port" in s),
		false,
	);
	const bare = await toy({}, false);
	assert.equal(bare.pi.getFlag("viser"), undefined);
});

test("--viser starts the view on the env server, links its port, draws plan results and stops with the session", async () => {
	const dir = mkdtempSync(join(tmpdir(), "viser-"));
	const f = await toy({ viser: true, "viser-python": fakeViewer(dir), "viser-port": "8123" });
	const argv = JSON.parse(readFileSync(join(dir, "argv"), "utf8")) as string[];
	const arg = (k: string) => argv[argv.indexOf(k) + 1];
	assert.deepEqual(argv.slice(0, 2), ["-m", "pi_embodied_services.components.viser_view"]);
	assert.equal(arg("--robot"), "libero");
	assert.equal(arg("--env"), "http://127.0.0.1:4567");
	assert.equal(arg("--viser-port"), "8123");
	assert.ok(argv.includes("--parent-watch"));
	assert.equal(readFileSync(join(dir, "token"), "utf8"), "f00d", "the env server's token, in the environment");
	assert.equal(argv.join(" ").includes("f00d"), false, "never on the command line");
	assert.equal(f.statuses.at(-1)?.viser_port, 18123, "the dashboard gets the page's port");
	assert.match(f.notes.join("\n"), /viser 3D view: http:\/\/0\.0\.0\.0:18123\//);

	const candidates = [{ id: "g1", position: [0.5, 0, 0.1] }];
	const planned = { toolName: "plan_grasp", isError: false, content: [], details: { candidates, active: "g1" } };
	assert.equal(await f.emit("tool_result", planned), undefined, "the result is untouched");
	await f.emit("tool_result", { ...planned, toolName: "move_to" });
	await f.emit("tool_result", { ...planned, isError: true });
	const calls = readFileSync(join(dir, "calls"), "utf8")
		.trim()
		.split("\n")
		.map((l) => JSON.parse(l));
	assert.deepEqual(
		calls.map((c) => [c.method, c.kwargs]),
		[["viser.grasps", { candidates, active: "g1" }]],
	);

	await f.emit("session_shutdown");
	assert.equal(existsSync(join(dir, "exited")), true, "stdin EOF ended it");
});

test("a view that cannot start is a warning, not a failed episode", async () => {
	const dir = mkdtempSync(join(tmpdir(), "viser-"));
	const bad = join(dir, "python");
	writeFileSync(bad, "#!/bin/sh\necho no viser >&2\nexit 3\n");
	chmodSync(bad, 0o755);
	const f = await toy({ viser: true, "viser-python": bad });
	assert.match(f.notes.join("\n"), /\[viser\] not started: viser view exited \(3\)/);
	assert.equal(f.statuses.at(-1)?.ready, true, "the robot is up");
	assert.equal(f.handlers.get("tool_result"), undefined);
});
