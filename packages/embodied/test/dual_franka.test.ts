import assert from "node:assert/strict";
import { chmodSync, mkdirSync, mkdtempSync, realpathSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { defineRobot } from "../src/robot.ts";
import { inlineWrist } from "../src/robots/dual_franka/config.ts";
import dualFranka from "../src/robots/dual_franka/index.ts";
import { codeApiReply } from "./helpers/code-api.ts";
import { deployFlags } from "./helpers/deployment.ts";

type Handler = (event: any, ctx: any) => unknown;

/**
 * A stub pi (as in robot.test.ts) whose UI is an operator answering `select` from `answers`.
 * No robot starts: tests either drive tools that act before the env is needed, or a toy robot.
 */
function fakePi(flagValues: Record<string, unknown> = {}) {
	flagValues = deployFlags(flagValues);
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const commands = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	const branch: unknown[] = [];
	const dialogs: { prompt: string; options: string[] }[] = [];
	const answers: string[] = [];
	const confirms: boolean[] = [];
	let active: string[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: (name: string, c: any) => commands.set(name, c),
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const dir = realpathSync(mkdtempSync(join(tmpdir(), "dual-franka-")));
	const ctx = {
		hasUI: true,
		cwd: dir,
		ui: {
			notify: () => {},
			setWidget: () => {},
			input: async () => "",
			confirm: async () => confirms.shift() ?? true,
			select: async (prompt: string, options: string[]) => {
				dialogs.push({ prompt, options });
				return answers.shift();
			},
		},
		shutdown: () => {},
		sessionManager: {
			getBranch: () => branch,
			getSessionDir: () => dir,
			getSessionFile: () => join(dir, "session.jsonl"),
			getSessionId: () => "session",
		},
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) {
			const r: any = await fn({ type: name, ...event }, ctx);
			if (r !== undefined) result = r;
			if (r && name === "tool_call" && r.block) return r;
		}
		return result;
	}
	/** Execute a registered tool as pi would, with this UI. */
	const run = (name: string, params: Record<string, unknown> = {}) =>
		tools.get(name).execute("id", params, undefined, undefined, ctx);
	/** Record a tool result on the branch, as pi does after execution. */
	const record = (toolName: string, details: Record<string, unknown>, isError = false) =>
		branch.push({ type: "message", message: { role: "toolResult", toolName, toolCallId: "id", details, isError } });
	const kinds = () => entries.filter((e) => e.type === "operator_event").map((e) => e.data.kind);
	return { pi, emit, run, record, tools, commands, flags, dialogs, answers, dir, kinds, active: () => active };
}

test("dual_franka mounts exploration with RPent's real-robot budget", () => {
	const f = fakePi();
	dualFranka(f.pi);
	assert.equal(f.flags.explore, false);
	assert.equal(f.flags["explore-sessions"], "1");
	assert.equal(f.flags["explore-attempts-per-session"], "3");
	assert.ok(f.commands.has("explore"));
	for (const name of ["reset", "request_scene_reset", "request_operator_verdict", "finish"])
		assert.ok(f.tools.has(name), name);
});

test("request_scene_reset asks the operator; only a confirmed scene resets the robot", async () => {
	const declined = fakePi({ operator: true });
	dualFranka(declined.pi);
	declined.answers.push("abort");
	const no = await declined.run("request_scene_reset", { reason: "bowl dropped", expected_scene_state: "bowl left" });
	assert.deepEqual(declined.dialogs[0].options, ["done", "abort"]);
	assert.match(declined.dialogs[0].prompt, /bowl dropped[\s\S]*bowl left/);
	assert.deepEqual(no.details, { error: "scene reset not confirmed", operator_aborted: true });
	assert.deepEqual(declined.kinds(), ["reset_requested", "reset_response"], "the robot was not reset");

	const confirmed = fakePi({ operator: true });
	dualFranka(confirmed.pi);
	confirmed.answers.push("done");
	const yes = await confirmed.run("request_scene_reset", { reason: "bowl dropped" });
	// The robot's reset ran (and, with no env server in this test, failed); the attempt did not advance.
	assert.equal(yes.details.error, "robot reset failed");
	assert.match(yes.details.robot_reset, /dual_franka is not initialized/);
	assert.deepEqual(confirmed.kinds(), ["reset_requested", "reset_response", "reset_failed"]);
});

test("exploration's reset is the operator's scene reset, within the archive and budget rules", async () => {
	// --memory-dir keeps the exploration inbox in the test's temp dir, not the user's memory home.
	const f = fakePi({
		operator: true,
		explore: true,
		"output-dir": "run",
		"memory-dir": "memory",
		python: "/nonexistent/python",
	});
	dualFranka(f.pi);
	await f.emit("session_start"); // the robot cannot start (no Python); memory and exploration can
	const attempts = join(f.dir, "run", "attempts", "dual_franka_t0");

	await assert.rejects(f.run("reset", { reason: "slipped" }), /Close out attempt 1 first/);
	assert.equal(f.dialogs.length, 0, "an unarchived attempt never reaches the operator");

	mkdirSync(attempts, { recursive: true });
	writeFileSync(join(attempts, "attempt_1_failed.json"), "{}");
	f.answers.push("done");
	await assert.rejects(f.run("reset", { reason: "slipped" }), /robot reset failed/);
	assert.match(f.dialogs[0].prompt, /Scene reset requested: slipped/);
	assert.deepEqual(f.dialogs[0].options, ["done", "abort"]);

	f.record("reset", { action: "reset" });
	f.record("reset", { action: "reset" });
	writeFileSync(join(attempts, "attempt_2_failed.json"), "{}");
	writeFileSync(join(attempts, "attempt_3_failed.json"), "{}");
	await assert.rejects(f.run("reset", { reason: "again" }), /attempt budget is spent \(3 attempts\)/);
	assert.equal(f.dialogs.length, 1);

	f.pi.setActiveTools(["move_delta", "request_operator_verdict", "request_scene_reset"]);
	const started = await f.emit("before_agent_start", { systemPrompt: "base" });
	assert.ok(f.active().includes("reset") && !f.active().includes("request_scene_reset"));
	assert.match(started.systemPrompt, /REAL-ROBOT EXPLORATION\. You are agent 1 of up to 1 on `dual_franka_t0`/);
	assert.match(started.systemPrompt, /suite_dual_franka_real_t0/);
});

test("only an operator success verdict marks an exploration attempt solved", async () => {
	const f = fakePi({ operator: true, explore: true });
	dualFranka(f.pi);
	const judge = async (answer: string) => {
		f.answers.push(answer);
		const r = await f.run("request_operator_verdict");
		return f.emit("tool_result", { toolName: "request_operator_verdict", details: r.details, isError: false });
	};
	const solved = await judge("success");
	assert.equal(solved.details.status, "success");
	assert.equal(solved.details.terminated, true);
	assert.match(solved.content[0].text, /"terminated":true/);
	assert.equal(await judge("failure"), undefined);
	assert.equal(await judge("continue"), undefined);

	const g = fakePi({ operator: true });
	dualFranka(g.pi);
	g.answers.push("success");
	const r = await g.run("request_operator_verdict");
	assert.equal(await g.emit("tool_result", { toolName: "request_operator_verdict", details: r.details }), undefined);
});

/** The composition dual_franka relies on (operator + exploration + memory), on a robot that starts. */
function realRobotToy(f: ReturnType<typeof fakePi>, resets: string[]) {
	let steps = 0;
	const robot = defineRobot(f.pi, {
		name: "toy",
		task: [],
		keepImages: 1,
		memory: { home: () => join(f.dir, "memory"), cell: () => ({ tag: "toy_t0", reference: "" }), primitives: [] },
		operator: {
			step: () => steps,
			reset: async () => {
				resets.push("robot");
				return { ok: true };
			},
		},
		explore: {
			reset: async (result, ctx, signal) => {
				const r = (await robot.op.sceneReset(ctx, String(result.reason), "", signal)) as Record<string, unknown>;
				if (r.error) throw new Error(JSON.stringify(r));
				return { content: [], details: result };
			},
			prompt: () => "",
			budget: { sessions: 1, attempts: 3 },
		},
		start: async () => ["move"],
		result: () => ({}),
		finish: {
			description: "finish",
			parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
			result: (p) => ({ content: [], details: p }),
		},
	});
	robot.tool("move", "move", Type.Object({}), async () => {
		steps++;
		return { content: [], details: { terminated: false } };
	});
}

test("finish needs an operator verdict and, with attempts left, a success; an operator abort ends the run", async () => {
	const f = fakePi({ operator: true, explore: true, "output-dir": "run" });
	const resets: string[] = [];
	realRobotToy(f, resets);
	await f.emit("session_start");
	// Attempt 1 is archived (<output>/attempts/<tag>/) before it may be judged and reset.
	mkdirSync(join(f.dir, "run", "attempts", "toy_t0"), { recursive: true });
	writeFileSync(join(f.dir, "run", "attempts", "toy_t0", "attempt_1_failed.json"), "{}");

	f.answers.push("failure");
	const failed = await f.emit("tool_call", { toolName: "finish" });
	assert.match(f.dialogs[0].prompt, /The agent wants to finish/);
	assert.match(failed.reason, /2 of its 3 attempts left/);

	f.answers.push("done");
	await f.run("reset", { reason: "retry" });
	assert.deepEqual(resets, ["robot"]);
	f.record("reset", { action: "reset" });
	f.answers.push("success");
	const verdict = await f.run("request_operator_verdict");
	f.record("request_operator_verdict", { ...verdict.details, terminated: true });
	assert.equal(await f.emit("tool_call", { toolName: "finish" }), undefined, "a judged success may finish");

	const g = fakePi({ operator: true, explore: true, "output-dir": "run" });
	realRobotToy(g, []);
	await g.emit("session_start");
	g.answers.push("abort");
	assert.equal((await g.run("request_scene_reset", { reason: "stuck" })).details.operator_aborted, true);
	assert.match((await g.emit("tool_call", { toolName: "move" })).reason, /operator aborted/);
	assert.equal(await g.emit("tool_call", { toolName: "finish" }), undefined, "finish despite attempts left");
});

test("dual Franka has a wrist view only when an inline camera is a wrist camera (the D455 alone by default)", () => {
	assert.equal(inlineWrist({}), false, "the default inline camera is the D455");
	assert.equal(inlineWrist(null), false);
	assert.equal(inlineWrist({ agent_observation: { inline_cameras: ["d455", "left_wrist"] } }), true);
	assert.equal(inlineWrist({ agent_observation: { inline_cameras: ["d455", "base"] } }), false);
	assert.equal(inlineWrist({ agent_observation: { inline_cameras: [] } }), false);
});

/** A fake dual-Franka env server (`--env-url`) whose `code.run` made one motion. */
async function fakeDualEnv() {
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
			if (method === "env.get_env_meta")
				result = {
					ok: true,
					motion_limits: {
						max_move_m: 0.1,
						max_rotate_rad: 0.5,
						z_floor_m: 0.02,
						workspace_xy: [0.1, 1.15, -0.85, 0.85],
					},
				};
			else if (method === "env.reset") result = { ok: true, states: [1] };
			else if (method === "env.get_observation") result = { states: [1], d455_images: nd([2, 2, 3]) };
			else if (method === "env.get_robot_state")
				result = {
					left_arm: { tcp_pose: [0.5, 0.3, 0.3, 1, 0, 0, 0] },
					right_arm: { tcp_pose: [0.5, -0.3, 0.3, 1, 0, 0, 0] },
				};
			else if (method === "env.get_camera_meta") result = null;
			else if (method === "code.api") result = codeApiReply("dual_franka", kwargs.tier);
			else if (method === "code.run")
				result = {
					status: "ran",
					stdout: "",
					stderr: "",
					traceback: null,
					error: null,
					result: null,
					calls: [],
					n_calls: 1,
					move_m: 0.03,
					ms: 5,
					motions: 1,
					states: [2],
					frames: [],
				};
			res.end(JSON.stringify({ ok: true, result }));
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	return {
		url: `http://127.0.0.1:${(server.address() as AddressInfo).port}`,
		calls,
		close: () => {
			server.closeAllConnections();
			server.close();
		},
	};
}

test("dual_franka --code: the server enforces pi's limits, the program is confirmed and the run is a state step", async (t) => {
	const env = await fakeDualEnv();
	t.after(env.close);
	const dir = mkdtempSync(join(tmpdir(), "dual-py-"));
	const py = join(dir, "python");
	const setup = {
		task: {
			name: "handover",
			instruction: "hand the cup over",
			success_criteria: "cup in the right gripper",
			constraints: [],
			setup: "cup on the left",
			vla_instruction: null,
		},
	};
	writeFileSync(py, `#!/usr/bin/env bash\necho '${JSON.stringify(setup)}'\n`);
	chmodSync(py, 0o755);
	const f = fakePi({
		operator: true,
		code: "true",
		"code-real": true,
		task: "3",
		"z-floor": "0.02",
		"env-url": env.url,
		python: py,
		out: dir,
	});
	dualFranka(f.pi);
	await f.emit("session_start");
	assert.ok(f.active().includes("run_code"), f.active().join(","));
	// The attached server enforces pi's limits (its motion_limits were checked); pi sends none.
	assert.ok(!env.calls.some((c) => c.method === "code.set_limits"));
	await f.emit("agent_start");
	const r = await f.run("run_code", { code: "move_delta('left', [0, 0, 0.03])" });
	assert.equal(r.details.status, "ran");
	assert.equal(env.calls.filter((c) => c.method === "code.run").length, 1);
	assert.match(r.content.map((c: any) => c.text ?? "").join("\n"), /"action": "run_code"/);
});
