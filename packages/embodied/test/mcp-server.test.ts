/**
 * The MCP server (../src/integrations/mcp) against a fake env server: an in-process MCP client
 * (pi-mcp's McpClient over its in-memory transport pair) lists the tools a manifest yields and calls
 * them; the gates (finish, stop, another process on the port, unknown tools, bad arguments) refuse
 * as the pi session does; images come back as MCP image content; the stdio entry serves the same
 * list to a client that spawns it.
 */

import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { McpClient, StdioTransport } from "@earendil-works/pi-mcp";
import { createInMemoryTransportPair } from "@earendil-works/pi-mcp/testing";
import { RpcClient } from "../src/infra/rpc.ts";
import { McpToolServer } from "../src/integrations/mcp/protocol.ts";
import { connect, listOnly, parseArgs, session } from "../src/integrations/mcp/server.ts";
import { RobotSession } from "../src/integrations/mcp/session.ts";
import { loadManifest } from "../src/primitives/manifest.ts";
import { codeApiReply } from "./helpers/code-api.ts";
import { useDeployment } from "./helpers/deployment.ts";

type Call = { method: string; kwargs: Record<string, unknown>; token?: string };

/** A fake env server (libero by default): healthz, code.api (with SAM3), a few env methods; records every call. */
function fakeEnv(o: { robot?: string; pid?: () => number; has?: (c: string) => boolean } = {}) {
	const calls: Call[] = [];
	const robot = o.robot ?? "libero";
	const pid = o.pid ?? (() => 4242);
	const image = () => ({
		__ndarray__: Buffer.alloc(4 * 4 * 3, 200).toString("base64"),
		dtype: "uint8",
		shape: [4, 4, 3],
	});
	const methods: Record<string, (kw: Record<string, unknown>) => unknown> = {
		healthz: () => ({ status: "ok", service: `${robot}-env`, version: "test", pid: pid() }),
		"code.api": (kw) => codeApiReply(robot, (kw.tier as string) ?? null, o.has ?? ((c) => c === "sam3")),
		"env.set_gripper": (kw) => ({ ok: true, gripper: kw.gripper, steps_used: 3 }),
		"env.move_to": (kw) => ({ ok: true, xyz: kw.xyz, final_dist_m: 0.001, steps_used: 12 }),
		"env.get_observation": () => ({ agentview: { rgb: image() }, wrist: { rgb: image() }, eef_pos: [0.1, 0.2, 0.3] }),
		"env.render_camera": () => image(),
		"env.segment": (kw) => ({ masks: [], prompt: kw.prompt }),
		"env.reset": () => [{ eef_pos: [0, 0, 0] }, {}],
		"env.move_delta": (kw) => ({ ok: true, delta_xyz: kw.delta_xyz }),
		stop: () => ({ ok: true, stop_generation: 1, call_in_progress: false }),
	};
	const server: Server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const call = JSON.parse(body) as { method: string; kwargs?: Record<string, unknown>; token?: string };
			calls.push({ method: call.method, kwargs: call.kwargs ?? {}, token: call.token });
			const fn = methods[call.method];
			res.writeHead(200, { "Content-Type": "application/json" });
			res.end(
				JSON.stringify(
					fn
						? { ok: true, result: fn(call.kwargs ?? {}) }
						: { ok: false, error: `unknown RPC method: '${call.method}'` },
				),
			);
		});
	});
	return {
		calls,
		methods,
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

/** An MCP client connected in-process to a session on the fake server. */
async function client(s: RobotSession) {
	const pair = createInMemoryTransportPair();
	const server = new McpToolServer(s, pair.server);
	await server.start();
	const c = new McpClient({ name: "test", version: "0" });
	await c.connect(pair.client);
	return { c, close: () => c.close() };
}

const text = (r: { content: { type: string; text?: string }[] }) =>
	JSON.parse(r.content.find((c) => c.type === "text")?.text ?? "null");

test("tools/list: the manifest's env tools under the requires rule, plus the built-ins; ts and module entries stay pi-only", async () => {
	useDeployment({});
	const env = fakeEnv();
	const url = await env.listen();
	try {
		const a = parseArgs(["--robot", "libero", "--env", url, "--var", "cameras=agentview,wrist"]);
		const c = await connect(a);
		assert.equal(c.pid, 4242);
		// connect resets the env once, after the manifest check, as every robot's start does.
		assert.deepEqual(
			env.calls.map((x) => x.method),
			["healthz", "healthz", "code.api", "env.reset"],
		);
		const s = session(a, c);
		const { c: mcp, close } = await client(s);
		try {
			const names = (await mcp.listTools()).map((t) => t.name);
			for (const n of ["move_to", "set_gripper", "release", "observe", "finish", "stop", "resume", "robot_status"])
				assert.ok(names.includes(n), `${n} served: ${names.join(", ")}`);
			// SAM3 is inferred from code.api (the server lists code primitives that require it).
			assert.ok(names.includes("segment"), "segment served with sam3 from code.api");
			// Not served: `requires` the server does not meet, ts tools, module-owned entries, privileged variants, code primitives.
			for (const n of [
				"preview_reach",
				"execute_grasp",
				"view_env_state",
				"pi0_pick",
				"follow_waypoints",
				"act",
				"run_code",
				"ground_truth_poses",
				"get_state",
				"render_camera",
			])
				assert.ok(!names.includes(n), `${n} not served`);
			assert.equal(names.filter((n) => n === "finish").length, 1);
			const move = (await mcp.listTools()).find((t) => t.name === "move_to");
			assert.equal(move?.inputSchema.type, "object");
			assert.deepEqual((move?.inputSchema as { required?: string[] }).required, ["xyz"]);
			assert.equal(move?.annotations?.destructiveHint, true);
			assert.equal(move?.description, loadManifest("libero").primitives.find((e) => e.name === "move_to")?.doc.tool);
		} finally {
			await close();
		}
	} finally {
		env.close();
	}
});

test("tools/call forwards to the manifest's method with the validated kwargs; bad arguments and unknown tools are refused", async () => {
	useDeployment({});
	const env = fakeEnv();
	const url = await env.listen();
	try {
		const a = parseArgs(["--robot", "libero", "--env", url]);
		const s = session(a, await connect(a));
		const { c: mcp, close } = await client(s);
		try {
			const r = await mcp.callTool("set_gripper", { gripper: 1 });
			assert.equal(r.isError, undefined);
			assert.deepEqual(text(r), { ok: true, gripper: 1, steps_used: 3 });
			const sent = env.calls.find((c) => c.method === "env.set_gripper");
			assert.deepEqual(sent?.kwargs, { gripper: 1 });
			// The motion guard asked healthz first.
			const i = env.calls.findIndex((c) => c.method === "env.set_gripper");
			assert.equal(env.calls[i - 1]?.method, "healthz");
			const bad = await mcp.callTool("move_to", { xyz: [1, 2] });
			assert.equal(bad.isError, true);
			assert.match(bad.content[0].type === "text" ? bad.content[0].text : "", /move_to/);
			assert.ok(!env.calls.some((c) => c.method === "env.move_to"), "an invalid call never reaches the server");
			await assert.rejects(mcp.callTool("no_such_tool", {}), /Unknown tool/);
		} finally {
			await close();
		}
	} finally {
		env.close();
	}
});

test("observe returns the cameras as image content; finish ends the episode; stop latches motion until resume", async () => {
	useDeployment({});
	const env = fakeEnv();
	const url = await env.listen();
	try {
		const a = parseArgs(["--robot", "libero", "--env", url]);
		const s = session(a, await connect(a));
		const { c: mcp, close } = await client(s);
		try {
			const o = await mcp.callTool("observe", {});
			assert.equal(o.content.filter((c) => c.type === "image").length, 2);
			assert.deepEqual(text(o).eef_pos, [0.1, 0.2, 0.3]);
			assert.deepEqual(text(o).agentview.rgb, { image: 1, shape: [4, 4, 3] });
			const one = await mcp.callTool("observe", { camera: "wrist" });
			assert.equal(one.content.filter((c) => c.type === "image").length, 1);
			assert.deepEqual(env.calls.at(-1), {
				method: "env.render_camera",
				kwargs: { camera_name: "wrist" },
				token: undefined,
			});

			await mcp.callTool("stop", {});
			const halted = await mcp.callTool("set_gripper", { gripper: -1 });
			assert.equal(halted.isError, true);
			assert.match(halted.content[0].type === "text" ? halted.content[0].text : "", /stop was issued/);
			await mcp.callTool("resume", {});
			assert.equal((await mcp.callTool("set_gripper", { gripper: -1 })).isError, undefined);

			const fin = await mcp.callTool("finish", { status: "success", summary: "the bowl is on the plate" });
			assert.equal(text(fin).claimed, "success");
			const after = await mcp.callTool("set_gripper", { gripper: 1 });
			assert.equal(after.isError, true);
			assert.equal(after.content[0].type === "text" ? after.content[0].text : "", "The episode is finished.");
			assert.equal((await mcp.callTool("observe", {})).isError, undefined);
			assert.equal(text(await mcp.callTool("robot_status", {})).claimed.status, "success");
		} finally {
			await close();
		}
	} finally {
		env.close();
	}
});

test("a motion refuses when another process answers on the port, and ends the episode when the server is gone", async () => {
	useDeployment({});
	let pid = 4242;
	const env = fakeEnv({ pid: () => pid });
	const url = await env.listen();
	try {
		const a = parseArgs(["--robot", "libero", "--env", url, "--no-reset"]);
		const s = session(a, await connect(a));
		assert.ok(!env.calls.some((x) => x.method === "env.reset"), "--no-reset leaves the env as found");
		const { c: mcp, close } = await client(s);
		try {
			pid = 9999;
			const r = await mcp.callTool("set_gripper", { gripper: 1 });
			assert.equal(r.isError, true);
			assert.match(r.content[0].type === "text" ? r.content[0].text : "", /pid 9999, not the pid 4242/);
			assert.ok(!env.calls.some((c) => c.method === "env.set_gripper"));
			assert.equal(text(await mcp.callTool("robot_status", {})).same_process, false);
			env.close();
			const gone = await mcp.callTool("set_gripper", { gripper: 1 });
			assert.equal(gone.isError, true);
			const later = await mcp.callTool("observe", {});
			assert.match(
				later.content[0].type === "text" ? later.content[0].text : "",
				/^The robot failed: .*The episode is over\.$/,
			);
		} finally {
			await close();
		}
	} finally {
		env.close();
	}
});

test("a real robot connects without env.reset; its reset is a tool under the motion guard and the stop latch", async () => {
	useDeployment({});
	const env = fakeEnv({ robot: "ur5e" });
	const url = await env.listen();
	try {
		const a = parseArgs(["--robot", "ur5e", "--env", url]);
		assert.equal(a.reset, undefined, "unset: the robot decides");
		const c = await connect(a);
		assert.deepEqual(
			env.calls.map((x) => x.method),
			["healthz", "healthz", "code.api"],
			"connecting to a real arm sends no env.reset (pi's start asks the operator first)",
		);
		const s = session(a, c);
		const { c: mcp, close } = await client(s);
		try {
			const reset = (await mcp.listTools()).find((t) => t.name === "reset");
			assert.equal(reset?.annotations?.destructiveHint, true, "reset is a motion");
			assert.equal(reset?.annotations?.readOnlyHint, false);
			const r = await mcp.callTool("reset", {});
			assert.equal(r.isError, undefined);
			const i = env.calls.findIndex((x) => x.method === "env.reset");
			assert.ok(i > 0, "reset reached the server only through the tool");
			assert.equal(env.calls[i - 1]?.method, "healthz", "the motion guard ran first");
			await mcp.callTool("stop", {});
			const halted = await mcp.callTool("reset", {});
			assert.equal(halted.isError, true);
			assert.match(halted.content[0].type === "text" ? halted.content[0].text : "", /stop was issued/);
			assert.equal(env.calls.filter((x) => x.method === "env.reset").length, 1);
			await mcp.callTool("finish", { status: "aborted", summary: "done" });
			assert.equal((await mcp.callTool("reset", {})).isError, true, "no reset after finish");
		} finally {
			await close();
		}
		// --no-reset still skips a simulator's reset; a simulator resets by default (the first test).
		assert.equal(parseArgs(["--robot", "ur5e", "--env", url, "--no-reset"]).reset, false);
	} finally {
		env.close();
	}
});

test("--tier and --privileged select entries as pi's manifestTool does; a server of another manifest version is refused", async () => {
	useDeployment({});
	const env = fakeEnv();
	const url = await env.listen();
	try {
		const high = session(parseArgs(["--robot", "libero", "--env", url, "--tier", "high", "--privileged"]), {
			rpc: new RpcClient(url),
			served: codeApiReply("libero", "privileged").available,
		});
		const names = high.list().map((t) => t.name);
		assert.ok(names.includes("goto_pose") && names.includes("ground_truth_poses"), names.join(","));
		assert.ok(!names.includes("move_to"), "low-tier move_to is not in the high tier");
		assert.equal(high.entries().find((e) => e.name === "get_object_pose")?.method, "env.get_object_pose_privileged");
		const plain = new RobotSession({
			robot: "libero",
			manifest: loadManifest("libero"),
			rpc: new RpcClient(url),
			privileged: false,
		});
		assert.equal(
			plain.entries().find((e) => e.name === "get_object_pose"),
			undefined,
			"requires sam3 without a source: left out",
		);
		assert.ok(!plain.list().some((t) => t.name === "ground_truth_poses"));

		env.methods["code.api"] = () => ({ tier: null, manifest_digest: "0".repeat(64), available: [], digest: "x" });
		await assert.rejects(connect(parseArgs(["--robot", "libero", "--env", url])), /runs manifest 000000000000/);
	} finally {
		env.close();
	}
});

test("--list prints the tool list without a server; --capabilities stands in for code.api", () => {
	const none = listOnly(parseArgs(["--robot", "libero", "--list"]));
	assert.ok(none.tools.some((t) => t.name === "move_to"));
	assert.ok(!none.tools.some((t) => t.name === "segment"));
	const sam = listOnly(parseArgs(["--robot", "libero", "--list", "--capabilities", "sam3,grasp"]));
	assert.ok(sam.tools.some((t) => t.name === "segment") && sam.tools.some((t) => t.name === "execute_grasp"));
	assert.throws(() => parseArgs(["--robot", "libero", "--tier", "privileged"]), /--tier privileged/);
	assert.throws(() => parseArgs(["--env", "x"]), /--robot/);
	assert.equal(parseArgs(["--robot", "libero", "--no-reset"]).reset, false);
	assert.equal(parseArgs(["--robot", "libero"]).reset, undefined);
	assert.throws(() => parseArgs(["--robot", "libero", "--env", "x", "--serve"]), /exclusive/);
});

test("stdio: a client that spawns the server lists the same tools and calls one", async () => {
	const config = useDeployment({});
	const env = fakeEnv();
	const url = await env.listen();
	const transport = new StdioTransport({
		command: process.execPath,
		args: [
			"--experimental-strip-types",
			fileURLToPath(new URL("../src/integrations/mcp/server.ts", import.meta.url)),
			"--robot",
			"libero",
			"--env",
			url,
		],
		env: { ...process.env, PI_EMBODIED_CONFIG: config, NODE_NO_WARNINGS: "1" },
		stderr: "pipe",
	} as ConstructorParameters<typeof StdioTransport>[0]);
	const c = new McpClient({ name: "test", version: "0" });
	try {
		const init = await c.connect(transport);
		assert.equal(init.serverInfo.name, "pi-embodied-libero");
		assert.match(init.instructions ?? "", /Observe before the first motion/);
		const names = (await c.listTools()).map((t) => t.name);
		assert.ok(names.includes("move_to") && names.includes("observe"), names.join(","));
		const r = await c.callTool("set_gripper", { gripper: 1 });
		assert.deepEqual(text(r), { ok: true, gripper: 1, steps_used: 3 });
	} finally {
		await c.close().catch(() => {});
		env.close();
	}
});
