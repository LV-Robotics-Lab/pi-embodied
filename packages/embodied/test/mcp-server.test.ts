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
import { decodePng } from "../src/infra/png.ts";
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
		// The facade's signature (utils/wrist_alignment.py): row and col, not the tool's point.
		"env.align_wrist": (kw) => {
			const unknown = Object.keys(kw).filter((k) => !["row", "col", "max_correction_m", "execute"].includes(k));
			if (unknown.length) throw new Error(`align_wrist() got an unexpected keyword argument '${unknown[0]}'`);
			if (!("row" in kw) || !("col" in kw))
				throw new Error("align_wrist() missing required arguments: 'row', 'col'");
			return { desired_pixel: [8, 8], target_pixel: [kw.row, kw.col], aligned_xyz: [0.1, 0.2, 0.3] };
		},
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
			// A method may answer later (a test holds healthz to order the calls) or throw (a facade's error).
			void (async () => {
				let reply: unknown;
				try {
					reply = fn
						? { ok: true, result: await fn(call.kwargs ?? {}) }
						: { ok: false, error: `unknown RPC method: '${call.method}'` };
				} catch (err) {
					reply = { ok: false, error: err instanceof Error ? err.message : String(err) };
				}
				res.writeHead(200, { "Content-Type": "application/json" });
				res.end(JSON.stringify(reply));
			})();
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
			// segment's `step` (pi's recorded states) is not offered; the method's own point is.
			const segment = (await mcp.listTools()).find((t) => t.name === "segment");
			const props = Object.keys((segment?.inputSchema as { properties?: object }).properties ?? {});
			assert.ok(props.includes("point") && !props.includes("step"), props.join(","));
		} finally {
			await close();
		}
	} finally {
		env.close();
	}
});

/** A 4x4 image whose first row is dark and the rest bright: a flip puts the dark row last. */
const rowsImage = () => {
	const data = Buffer.alloc(4 * 4 * 3, 200);
	data.fill(10, 0, 4 * 3);
	return { __ndarray__: data.toString("base64"), dtype: "uint8", shape: [4, 4, 3] };
};
/** The first PNG of a result, decoded to its top-left and bottom-left pixel values. */
const corners = (r: { content: { type: string; data?: string }[] }) => {
	const png = r.content.find((c) => c.type === "image")?.data ?? "";
	const { data, width, height, channels } = decodePng(Buffer.from(png, "base64"));
	return { top: data[0], bottom: data[(height - 1) * width * channels] };
};

test("observe on RoboCasa: no get_observation; render_camera(camera_name, height, width, depth) per camera with the robot's names and size, rows flipped; get_state", async () => {
	useDeployment({});
	const env = fakeEnv({ robot: "robocasa", has: () => false });
	delete env.methods["env.get_observation"];
	// The facade's signature (robocasa/env_server.py): every argument required, the sim's camera names.
	env.methods["env.render_camera"] = (kw) => {
		for (const k of ["camera_name", "height", "width", "depth"])
			if (!(k in kw)) throw new Error(`render_camera() missing 1 required positional argument: '${k}'`);
		const unknown = Object.keys(kw).filter((k) => !["camera_name", "height", "width", "depth"].includes(k));
		if (unknown.length) throw new Error(`render_camera() got an unexpected keyword argument '${unknown[0]}'`);
		if (!["robot0_agentview_left", "mobilebase0_navview", "robot0_eye_in_hand"].includes(kw.camera_name as string))
			throw new Error(`unknown camera ${kw.camera_name}`);
		return rowsImage();
	};
	env.methods["env.get_state"] = () => ({ eef_pos: [0.4, 0.5, 0.6], gripper: 0.1 });
	const url = await env.listen();
	try {
		const a = parseArgs(["--robot", "robocasa", "--env", url]);
		const s = session(a, await connect(a));
		const { c: mcp, close } = await client(s);
		try {
			const o = await mcp.callTool("observe", {});
			assert.equal(o.isError, undefined, JSON.stringify(o.content[0]));
			assert.equal(o.content.filter((c) => c.type === "image").length, 3);
			assert.deepEqual(Object.keys(text(o)), ["agentview", "navview", "wrist", "state"]);
			assert.deepEqual(text(o).state.eef_pos, [0.4, 0.5, 0.6]);
			const renders = env.calls.filter((c) => c.method === "env.render_camera").map((c) => c.kwargs);
			assert.deepEqual(renders, [
				{ camera_name: "robot0_agentview_left", height: 256, width: 256, depth: false },
				{ camera_name: "mobilebase0_navview", height: 256, width: 256, depth: false },
				{ camera_name: "robot0_eye_in_hand", height: 256, width: 256, depth: false },
			]);
			assert.ok(
				!env.calls.some((c) => c.method === "env.get_observation"),
				"never asked for a method RoboCasa lacks",
			);
			// MuJoCo renders bottom-up: the dark first row of the render is the image's last row.
			assert.deepEqual(corners(o), { top: 200, bottom: 10 });
			const one = await mcp.callTool("observe", { camera: "navview" });
			assert.equal(one.content.filter((c) => c.type === "image").length, 1);
			assert.deepEqual(env.calls.at(-1)?.kwargs, {
				camera_name: "mobilebase0_navview",
				height: 256,
				width: 256,
				depth: false,
			});
			assert.ok(!env.calls.slice(-1).some((c) => c.method === "env.get_state"), "one camera: no state");
		} finally {
			await close();
		}
	} finally {
		env.close();
	}
});

test("observe on RoboTwin: render_camera(camera_name) per declared view, upright, and the state from env.policy_frame", async () => {
	useDeployment({});
	const env = fakeEnv({ robot: "robotwin", has: () => false });
	delete env.methods["env.get_observation"];
	delete env.methods["env.get_state"];
	env.methods["env.render_camera"] = (kw) => {
		if (!("camera_name" in kw)) throw new Error("render_camera() missing 1 required argument: 'camera_name'");
		if (!["head", "left_wrist", "right_wrist"].includes(kw.camera_name as string))
			throw new Error(`no camera ${kw.camera_name}`);
		return rowsImage();
	};
	env.methods["env.policy_frame"] = () => ({ qpos: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] });
	const url = await env.listen();
	try {
		const a = parseArgs(["--robot", "robotwin", "--env", url]);
		const s = session(a, await connect(a));
		const { c: mcp, close } = await client(s);
		try {
			const o = await mcp.callTool("observe", {});
			assert.equal(o.isError, undefined, JSON.stringify(o.content[0]));
			assert.deepEqual(Object.keys(text(o)), ["head", "left_wrist", "right_wrist", "state"]);
			assert.deepEqual(
				env.calls.filter((c) => c.method === "env.render_camera").map((c) => c.kwargs),
				[{ camera_name: "head" }, { camera_name: "left_wrist" }, { camera_name: "right_wrist" }],
			);
			assert.equal(env.calls.at(-1)?.method, "env.policy_frame", "RoboTwin's get_state is env.policy_frame");
			assert.deepEqual(corners(o), { top: 10, bottom: 200 }, "SAPIEN renders upright: no flip");
			const head = await mcp.callTool("observe", { camera: "head" });
			assert.deepEqual(Object.keys(text(head)), ["head"]);
		} finally {
			await close();
		}
	} finally {
		env.close();
	}
});

test("a tool whose parameters differ from the method's is adapted as pi adapts it: align_wrist's point reaches the facade as row, col", async () => {
	useDeployment({});
	const env = fakeEnv({ has: (c) => c === "sam3" || c === "align_wrist" });
	const url = await env.listen();
	try {
		const a = parseArgs(["--robot", "libero", "--env", url]);
		const s = session(a, await connect(a));
		const { c: mcp, close } = await client(s);
		try {
			const tool = (await mcp.listTools()).find((t) => t.name === "align_wrist");
			assert.ok(tool, "align_wrist served (the server's code.api lists it)");
			assert.deepEqual(Object.keys((tool.inputSchema as { properties: object }).properties).sort(), [
				"execute",
				"max_correction_m",
				"point",
			]);
			const r = await mcp.callTool("align_wrist", { point: [100, 200], max_correction_m: 0.02 });
			assert.equal(r.isError, undefined, JSON.stringify(r.content));
			assert.deepEqual(text(r).target_pixel, [100, 200]);
			const sent = env.calls.find((c) => c.method === "env.align_wrist");
			assert.deepEqual(sent?.kwargs, { row: 100, col: 200, max_correction_m: 0.02 });
			const bad = await mcp.callTool("align_wrist", { point: [1] });
			assert.equal(bad.isError, true, "a one-element point fails the schema");
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
			// One camera: LIBERO's own path (robots/libero OBSERVATION): the facade's camera name, the tools' size.
			const one = await mcp.callTool("observe", { camera: "wrist" });
			assert.equal(one.content.filter((c) => c.type === "image").length, 1);
			assert.deepEqual(text(one).wrist, { image: 1, shape: [4, 4, 3] });
			assert.deepEqual(env.calls.at(-1), {
				method: "env.render_camera",
				kwargs: { camera_name: "robot0_eye_in_hand", height: 1024, width: 1024 },
				token: undefined,
			});
			const none = await mcp.callTool("observe", { camera: "overhead" });
			assert.equal(none.isError, true);
			assert.match(none.content[0].type === "text" ? none.content[0].text : "", /cameras are agentview, wrist/);

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

test("a motion that waited on the guard is refused when stop (or stop and resume) went by; stop latches before its RPC", async () => {
	useDeployment({});
	const env = fakeEnv();
	const url = await env.listen();
	try {
		const a = parseArgs(["--robot", "libero", "--env", url]);
		const s = session(a, await connect(a));
		const signal = new AbortController().signal;
		const healthz = env.methods.healthz;
		/** Hold the next healthz (the motion guard) until released; resolves when the motion reached it. */
		const hold = () => {
			let arrived = () => {};
			let release = () => {};
			const reached = new Promise<void>((r) => {
				arrived = r;
			});
			const released = new Promise<void>((r) => {
				release = r;
			});
			env.methods.healthz = async (kw) => {
				env.methods.healthz = healthz;
				arrived();
				await released;
				return healthz(kw);
			};
			return { reached, release };
		};
		const moved = () => env.calls.filter((c) => c.method === "env.set_gripper").length;

		// stop arrives while the motion waits on the guard: the motion is refused, the server never sees it.
		let gate = hold();
		let motion = s.call("set_gripper", { gripper: 1 }, signal);
		await gate.reached;
		const stopping = s.call("stop", {}, signal);
		assert.ok(s.refusal("set_gripper"), "the latch is set synchronously, before stop's RPC answers");
		await stopping;
		gate.release();
		let r = await motion;
		assert.equal(r.isError, true);
		assert.match(r.content[0].type === "text" ? r.content[0].text : "", /stop was issued/);
		assert.equal(moved(), 0, "the motion never reached the server");
		assert.ok(
			env.calls.some((c) => c.method === "stop"),
			"the server's stop was sent",
		);

		// stop then resume while the motion waits: the generation changed, so it is still refused.
		await s.call("resume", {}, signal);
		gate = hold();
		motion = s.call("set_gripper", { gripper: 1 }, signal);
		await gate.reached;
		await s.call("stop", {}, signal);
		await s.call("resume", {}, signal);
		gate.release();
		r = await motion;
		assert.equal(r.isError, true);
		assert.match(r.content[0].type === "text" ? r.content[0].text : "", /stop was issued while this call waited/);
		assert.equal(moved(), 0);

		// A motion admitted after the resume runs.
		assert.equal((await s.call("set_gripper", { gripper: 1 }, signal)).isError, undefined);
		assert.equal(moved(), 1);
		assert.equal(text(await s.call("robot_status", {}, signal)).stops, 2);
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
