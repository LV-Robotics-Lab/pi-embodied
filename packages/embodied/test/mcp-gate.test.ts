/**
 * The MCP server's own operator gate on real robots (../src/integrations/mcp/gate.ts): whatever the
 * host does (Codex 0.160 runs no plugin hooks), a motion tool or `reset` on a Franka or UR5e runs
 * only when the operator authorised it outside the model: `PI_EMBODIED_MOTION_CONFIRMED=1` at launch
 * for the session, or a per-call ticket (the tool's name) written into the `--confirm-file`, which
 * exactly one matching call consumes. A simulator is not gated. Against a fake env server, in-process.
 */

import assert from "node:assert/strict";
import { chmodSync, existsSync, mkdtempSync, readFileSync, utimesSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { McpClient } from "@earendil-works/pi-mcp";
import { createInMemoryTransportPair } from "@earendil-works/pi-mcp/testing";
import { CONFIRMED_ENV, expandHome, OperatorGate, TICKET_TTL_MS } from "../src/integrations/mcp/gate.ts";
import { McpToolServer } from "../src/integrations/mcp/protocol.ts";
import { connect, parseArgs, session } from "../src/integrations/mcp/server.ts";
import { REAL_ROBOTS } from "../src/integrations/mcp/tools.ts";
import { codeApiReply } from "./helpers/code-api.ts";
import { useDeployment } from "./helpers/deployment.ts";

/** A fake env server for `robot`: healthz, code.api (nothing optional), the motions the tests call; records the calls. */
function fakeEnv(robot: string) {
	const calls: string[] = [];
	const methods: Record<string, (kw: Record<string, unknown>) => unknown> = {
		healthz: () => ({ status: "ok", service: `${robot}-env`, version: "test", pid: 4242 }),
		"code.api": (kw) => codeApiReply(robot, (kw.tier as string) ?? null, () => false),
		"env.move_delta": (kw) => ({ ok: true, delta_xyz: kw.delta_xyz }),
		"env.open_gripper": () => ({ ok: true }),
		"env.set_gripper": (kw) => ({ ok: true, gripper: kw.gripper }),
		"env.reset": () => [{ eef_pos: [0, 0, 0] }, {}],
		stop: () => ({ ok: true }),
	};
	const server: Server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const call = JSON.parse(body) as { method: string; kwargs?: Record<string, unknown> };
			calls.push(call.method);
			const fn = methods[call.method];
			const reply = fn
				? { ok: true, result: fn(call.kwargs ?? {}) }
				: { ok: false, error: `unknown RPC method: '${call.method}'` };
			res.writeHead(200, { "Content-Type": "application/json" });
			res.end(JSON.stringify(reply));
		});
	});
	return {
		calls,
		motions: () => calls.filter((m) => m.startsWith("env.")),
		async listen() {
			await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
			return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
		},
		close() {
			server.closeAllConnections();
			server.close();
		},
	};
}

/** A session on `robot` through the server's own parseArgs/connect/session, and an in-process MCP client on it. */
async function serve(robot: string, url: string, argv: string[] = [], env: NodeJS.ProcessEnv = {}) {
	const a = parseArgs(["--robot", robot, "--env", url, ...argv], env);
	const s = session(a, await connect(a), { log: () => {} });
	const pair = createInMemoryTransportPair();
	await new McpToolServer(s, pair.server).start();
	const c = new McpClient({ name: "test", version: "0" });
	await c.connect(pair.client);
	return { s, c, close: () => c.close() };
}

const textOf = (r: { content: { type: string; text?: string }[] }) =>
	r.content.find((c) => c.type === "text")?.text ?? "";
const ticketDir = () => mkdtempSync(join(tmpdir(), "pi-embodied-ticket-"));

test("a real robot without authorisation: every motion and reset is refused with both channels named; status, stop and resume run", async () => {
	useDeployment({});
	for (const robot of ["franka", "ur5e"]) {
		const env = fakeEnv(robot);
		const url = await env.listen();
		try {
			const { s, c, close } = await serve(robot, url);
			try {
				assert.match(s.instructions, /real robot.*operator's authorisation/);
				const motions = s
					.entries()
					.filter((e) => e.mutating)
					.map((e) => e.name);
				assert.ok(motions.includes("move_delta") && motions.includes("open_gripper"), motions.join(","));
				for (const name of ["move_delta", "open_gripper", "reset"]) {
					const r = await c.callTool(name, name === "move_delta" ? { delta_xyz: [0.001, 0, 0] } : {});
					assert.equal(r.isError, true, `${robot}: ${name} refused`);
					const why = textOf(r);
					assert.match(
						why,
						new RegExp(`^${name} moves a real robot \\(${robot}\\) and the operator has not authorised it`),
					);
					assert.match(why, /PI_EMBODIED_MOTION_CONFIRMED=1/, "names the session channel");
					assert.match(why, /--confirm-file/, "names the ticket channel");
					assert.match(why, /outside the model/);
				}
				assert.deepEqual(env.motions(), [], "nothing reached the server");
				// The gate holds motions only: robot_status, stop and resume run.
				const status = await c.callTool("robot_status", {});
				const gate = JSON.parse(textOf(status)).operator_gate;
				assert.deepEqual(gate, {
					real: true,
					gated: true,
					session_confirmed: false,
					confirm_file: null,
					ticket: null,
				});
				assert.equal((await c.callTool("stop", {})).isError, undefined);
				assert.equal((await c.callTool("resume", {})).isError, undefined);
			} finally {
				await close();
			}
		} finally {
			env.close();
		}
	}
});

test("PI_EMBODIED_MOTION_CONFIRMED=1 in the server's environment at launch authorises the whole session", async () => {
	useDeployment({});
	const env = fakeEnv("ur5e");
	const url = await env.listen();
	try {
		const { c, close } = await serve("ur5e", url, [], { [CONFIRMED_ENV]: "1" });
		try {
			for (let i = 0; i < 3; i++) {
				const r = await c.callTool("move_delta", { delta_xyz: [0.001, 0, 0] });
				assert.equal(r.isError, undefined, textOf(r));
			}
			assert.equal((await c.callTool("reset", {})).isError, undefined);
			assert.deepEqual(env.motions(), ["env.move_delta", "env.move_delta", "env.move_delta", "env.reset"]);
			assert.equal(JSON.parse(textOf(await c.callTool("robot_status", {}))).operator_gate.session_confirmed, true);
		} finally {
			await close();
		}
		// Whitespace is not a confirmation; the variable is read from the environment given, not the process's.
		assert.equal(parseArgs(["--robot", "ur5e", "--env", url], { [CONFIRMED_ENV]: "  " }).confirmed, false);
		assert.equal(parseArgs(["--robot", "ur5e", "--env", url], {}).confirmed, false);
	} finally {
		env.close();
	}
});

test("--confirm-file: a ticket naming the tool admits exactly one call of it and is consumed; other tools, empty and stale tickets refuse", async () => {
	useDeployment({});
	const env = fakeEnv("franka");
	const url = await env.listen();
	const file = join(ticketDir(), "confirm");
	try {
		const { c, close } = await serve("franka", url, ["--confirm-file", file]);
		try {
			const move = () => c.callTool("move_delta", { delta_xyz: [0.001, 0, 0] });
			// No ticket yet.
			let r = await move();
			assert.equal(r.isError, true);
			assert.match(textOf(r), /holds no ticket/);
			assert.match(textOf(r), new RegExp(`writes "move_delta" into ${file.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`));
			// A ticket for another tool is left in place and the call refused.
			writeFileSync(file, "open_gripper\n");
			r = await move();
			assert.equal(r.isError, true);
			assert.match(textOf(r), /ticket for "open_gripper", not move_delta/);
			assert.ok(existsSync(file), "the other tool's ticket is left for its call");
			assert.equal(
				JSON.parse(textOf(await c.callTool("robot_status", {}))).operator_gate.ticket.tool,
				"open_gripper",
			);
			// Its own call consumes it.
			r = await c.callTool("open_gripper", {});
			assert.equal(r.isError, undefined, textOf(r));
			assert.ok(!existsSync(file), "the ticket is consumed");
			r = await c.callTool("open_gripper", {});
			assert.equal(r.isError, true, "a second call needs a second ticket");
			assert.deepEqual(env.motions(), ["env.open_gripper"]);
			// The matching ticket admits exactly one move; a trailing newline and spaces are fine.
			writeFileSync(file, "  move_delta  \n");
			r = await move();
			assert.equal(r.isError, undefined, textOf(r));
			r = await move();
			assert.equal(r.isError, true);
			assert.deepEqual(env.motions(), ["env.open_gripper", "env.move_delta"]);
			// Reset follows the same rule.
			r = await c.callTool("reset", {});
			assert.equal(r.isError, true);
			writeFileSync(file, "reset");
			r = await c.callTool("reset", {});
			assert.equal(r.isError, undefined, textOf(r));
			assert.ok(!existsSync(file));
			// An empty file (a touch) is not a ticket.
			writeFileSync(file, "\n");
			r = await move();
			assert.equal(r.isError, true);
			assert.match(textOf(r), /is empty: a ticket is the tool's name/);
			// A stale ticket is discarded, not honoured.
			writeFileSync(file, "move_delta");
			const old = (Date.now() - TICKET_TTL_MS - 60_000) / 1000;
			utimesSync(file, old, old);
			r = await move();
			assert.equal(r.isError, true);
			assert.match(textOf(r), /s old \(over 600 s\) and was discarded/);
			assert.ok(!existsSync(file));
			assert.deepEqual(env.motions(), ["env.open_gripper", "env.move_delta", "env.reset"]);
			// A stopped session refuses before the ticket is looked at: the ticket survives for after resume.
			writeFileSync(file, "move_delta");
			await c.callTool("stop", {});
			r = await move();
			assert.match(textOf(r), /stop was issued/);
			assert.equal(readFileSync(file, "utf8"), "move_delta", "the ticket was not spent on a refused call");
			await c.callTool("resume", {});
			assert.equal((await move()).isError, undefined);
		} finally {
			await close();
		}
	} finally {
		env.close();
	}
});

test("two concurrent calls of the ticket's tool: one runs, the other is refused", async () => {
	useDeployment({});
	const env = fakeEnv("ur5e");
	const url = await env.listen();
	const file = join(ticketDir(), "confirm");
	try {
		const { c, close } = await serve("ur5e", url, ["--confirm-file", file]);
		try {
			writeFileSync(file, "open_gripper");
			const [a, b] = await Promise.all([c.callTool("open_gripper", {}), c.callTool("open_gripper", {})]);
			assert.equal([a, b].filter((r) => r.isError).length, 1);
			assert.deepEqual(env.motions(), ["env.open_gripper"]);
		} finally {
			await close();
		}
	} finally {
		env.close();
	}
});

test("a simulator is not gated: motions run with neither channel; --confirm-file is accepted and idle", async () => {
	useDeployment({});
	const env = fakeEnv("libero");
	const url = await env.listen();
	const file = join(ticketDir(), "confirm");
	try {
		const { s, c, close } = await serve("libero", url, ["--confirm-file", file]);
		try {
			assert.doesNotMatch(s.instructions, /real robot/);
			assert.equal((await c.callTool("set_gripper", { gripper: 1 })).isError, undefined);
			assert.equal((await c.callTool("reset", {})).isError, undefined);
			assert.ok(env.motions().includes("env.set_gripper") && env.motions().includes("env.reset"));
			const gate = JSON.parse(textOf(await c.callTool("robot_status", {}))).operator_gate;
			assert.equal(gate.real, false);
			assert.equal(gate.gated, false);
			assert.equal(gate.confirm_file, file);
		} finally {
			await close();
		}
	} finally {
		env.close();
	}
});

test("OperatorGate alone: describe/warning/status; a ticket that cannot be removed authorises nothing; every REAL_ROBOT is gated", () => {
	// The documented `--config confirm_file=~/.pi/agent/confirm-ur5e` reaches the server with a literal tilde.
	assert.equal(
		new OperatorGate({ robot: "ur5e", real: true, sessionConfirmed: false, confirmFile: "~/.pi/agent/confirm-ur5e" })
			.confirmFile,
		join(homedir(), ".pi/agent/confirm-ur5e"),
	);
	assert.equal(expandHome("~"), homedir());
	assert.equal(expandHome("/abs/~/x"), "/abs/~/x");
	const none = new OperatorGate({ robot: "ur5e", real: true, sessionConfirmed: false });
	assert.match(none.describe(), /every motion tool and reset will be refused/);
	assert.equal(none.warning(), undefined);
	const sim = new OperatorGate({ robot: "libero", real: false, sessionConfirmed: false });
	assert.equal(sim.gated, false);
	assert.equal(sim.authorise("move_to"), undefined);
	const confirmed = new OperatorGate({ robot: "franka", real: true, sessionConfirmed: true });
	assert.equal(confirmed.gated, false);
	assert.match(confirmed.describe(), /every motion of this session is authorised/);
	for (const robot of REAL_ROBOTS)
		assert.equal(new OperatorGate({ robot, real: true, sessionConfirmed: false }).gated, true, robot);
	// A confirm file under the temp dir is where a sandboxed model may write: warned about.
	const inTmp = new OperatorGate({
		robot: "ur5e",
		real: true,
		sessionConfirmed: false,
		confirmFile: join(tmpdir(), "x"),
	});
	assert.match(inTmp.warning() ?? "", /under the temp dir/);
	// A ticket in a directory the server cannot write to cannot be consumed: refused, and the file stays.
	const dir = ticketDir();
	const file = join(dir, "confirm");
	writeFileSync(file, "move_delta");
	const g = new OperatorGate({ robot: "ur5e", real: true, sessionConfirmed: false, confirmFile: file });
	assert.deepEqual(g.status().ticket, { tool: "move_delta", age_s: 0 });
	if (process.getuid?.() !== 0) {
		chmodSync(dir, 0o500);
		try {
			assert.match(g.authorise("move_delta") ?? "", /cannot be removed.*refusing/);
			assert.ok(existsSync(file));
		} finally {
			chmodSync(dir, 0o700);
		}
	}
	assert.equal(g.authorise("move_delta"), undefined);
	assert.ok(!existsSync(file));
});
