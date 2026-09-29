import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import "../src/robots/libero/index.ts";
import { memory } from "../src/capabilities/memory/index.ts";
import { APPROVAL_ENTRY, highRisk, parseReview, reviewPrompt } from "../src/capabilities/operator.ts";
import { VLM_COST_EVENT } from "../src/modes/units/vlm.ts";
import { CLOSED_LOOP, CLOSED_LOOP_ENTRY, NON_MOTION, unknownOutcome } from "../src/planner/closed-loop.ts";
import {
	CONTEXT_VERSION_ENTRY,
	gitCommit,
	gitDirty,
	normalizeTemplates,
	sha256,
	templateId,
	usedTemplates,
} from "../src/planner/context-version.ts";
import { defineRobot, RESULT_ENTRY, type RobotSpec } from "../src/robot.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi running handlers in registration order (a tool_call block stops the chain), with a stub model registry. */
function fakePi(
	flagValues: Record<string, unknown> = {},
	o: { hasUI?: boolean; confirm?: boolean; review?: string } = {},
) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const entries: { type: string; data: any }[] = [];
	const events: { channel: string; data: any }[] = [];
	const listeners = new Map<string, ((data: any) => void)[]>();
	const asked: { system?: string; content: any[] }[] = [];
	const confirms: string[] = [];
	const branch: unknown[] = [];
	let active: string[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, f: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : f.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: () => {},
		registerCommand: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		getThinkingLevel: () => "off",
		getAllTools: () => [{ name: "move", description: "Move the gripper. gripper -1 = open", parameters: {} }],
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		// A real bus: the robot adds the reviewer's cost (VLM_COST_EVENT) to its budget.
		events: {
			emit: (channel: string, data: any) => {
				events.push({ channel, data });
				for (const fn of listeners.get(channel) ?? []) fn(data);
			},
			on: (channel: string, fn: (data: any) => void) => {
				listeners.set(channel, [...(listeners.get(channel) ?? []), fn]);
				return () => {};
			},
		},
	} as unknown as ExtensionAPI;
	const dir = realpathSync(mkdtempSync(join(tmpdir(), "runtime-")));
	const ctx = {
		hasUI: o.hasUI ?? false,
		cwd: dir,
		ui: {
			notify: () => {},
			confirm: async (title: string) => {
				confirms.push(title);
				return o.confirm ?? false;
			},
		},
		shutdown: () => {},
		getSystemPrompt: () => "the system prompt",
		model: { provider: "faux", id: "planner", reasoning: false },
		modelRegistry: {
			find: (provider: string, id: string) => ({ provider, id, reasoning: false }),
			streamSimple: (_model: unknown, req: { systemPrompt?: string; messages: { content: any[] }[] }) => ({
				result: async () => {
					asked.push({ system: req.systemPrompt, content: req.messages[0].content });
					return {
						stopReason: "stop",
						content: [{ type: "text", text: o.review ?? '{"decision":"approve","reason":"fine"}' }],
						usage: { cost: { total: 0.25 } },
					};
				},
			}),
		},
		sessionManager: { getBranch: () => branch, getSessionDir: () => dir },
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
	const log = console.error;
	console.error = () => {};
	const restore = () => {
		console.error = log;
		process.exitCode = undefined;
	};
	return { pi, emit, entries, events, asked, confirms, branch, dir, restore, flags };
}

const finish: RobotSpec["finish"] = {
	description: "finish",
	parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
	result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
};

/** A robot with one motion tool (`move`) and one observation tool (`view_env_state`). */
function toy(f: ReturnType<typeof fakePi>, tools = ["move", "view_env_state"], extra: Partial<RobotSpec> = {}) {
	f.pi.registerFlag("seed", { type: "string", default: "0" });
	const robot = defineRobot(f.pi, {
		name: "toy",
		task: ["seed"],
		keepImages: 1,
		start: async () => tools,
		prompt: () => "Toy robot.",
		result: () => ({}),
		status: () => ({ language: "put the cube in the bowl" }),
		finish,
		...extra,
	});
	const ok = async () => ({ content: [{ type: "text" as const, text: "ok" }], details: {} });
	robot.tool("move", "move", Type.Object({}), ok);
	robot.tool("view_env_state", "look", Type.Object({}), ok);
	return robot;
}

const call = (f: ReturnType<typeof fakePi>, toolName: string, input: Record<string, unknown> = {}) =>
	f.emit("tool_call", { toolName, input });
const result = (f: ReturnType<typeof fakePi>) => f.entries.find((e) => e.type === RESULT_ENTRY)?.data;
async function end(f: ReturnType<typeof fakePi>) {
	await f.emit("agent_start");
	await f.emit("session_shutdown");
	return result(f);
}

// ---------------------------------------------------------------------------
// closed-loop rules and the re-observe gate

test("every robot prompt ends with the closed-loop rules, their tool blocks following the active tools", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	toy(f);
	await f.emit("session_start");
	const out = (await f.emit("before_agent_start")) as { systemPrompt: string };
	assert.ok(out.systemPrompt.startsWith("Toy robot.\n\n## Closed-loop rules"));
	assert.match(out.systemPrompt, /three separate judgements/);
	assert.match(out.systemPrompt, /until you do, the next motion is refused/);
	assert.doesNotMatch(out.systemPrompt, /\[\/?tool:/);
	const g = fakePi();
	t.after(g.restore);
	toy(g, ["move"]);
	await g.emit("session_start");
	const blind = (await g.emit("before_agent_start")) as { systemPrompt: string };
	assert.doesNotMatch(blind.systemPrompt, /next motion is refused/, "no observation tool: no refusal is promised");
	assert.match(CLOSED_LOOP, /look at the new frame/);
});

test("unknownOutcome: timeouts and ambiguous results, not refusals", () => {
	const err = (text: string) => ({ isError: true, content: [{ type: "text", text }] });
	assert.match(unknownOutcome(err("env.move_to timed out after 120000 ms")) ?? "", /timed out/);
	assert.ok(unknownOutcome(err("LingBot reply timed out after 30000 ms")));
	assert.equal(unknownOutcome({ isError: false, content: [], details: { status: "timeout" } }), "timed out");
	assert.ok(unknownOutcome({ isError: false, content: [], details: { error: "servo timeout; pose unknown" } }));
	assert.ok(unknownOutcome({ isError: false, content: [], details: { result: { error: "outcome is unknown" } } }));
	assert.equal(unknownOutcome(err("delta_xyz moves 0.3 m; the limit is 0.02 m per call.")), undefined);
	assert.equal(unknownOutcome({ isError: false, content: [], details: { status: "ran" } }), undefined);
});

test("after a motion of unknown outcome the next motion is refused until an observation ran", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	toy(f);
	await f.emit("session_start");
	const timedOut = { isError: true, content: [{ type: "text", text: "env.step timed out after 5000 ms" }] };
	await f.emit("tool_result", { toolName: "move", ...timedOut });
	const refused = await call(f, "move");
	assert.equal(refused?.block, true);
	assert.match(refused.reason, /unknown outcome .*Re-observe with view_env_state/);
	assert.equal(await call(f, "view_env_state"), undefined, "looking is allowed");
	await f.emit("tool_result", { toolName: "view_env_state", isError: false, content: [] });
	assert.equal(await call(f, "move"), undefined, "re-observed: motion runs again");
	assert.deepEqual(
		f.entries.filter((e) => e.type === CLOSED_LOOP_ENTRY).map((e) => e.data.kind),
		["unknown_outcome", "refused", "reobserved"],
	);
	const r = await end(f);
	assert.equal(r.unknown_outcomes, 1);
	assert.equal(r.reobserve_refusals, 1);
});

test("without an active observation tool the re-observe gate stays open", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	toy(f, ["move"]);
	await f.emit("session_start");
	await f.emit("tool_result", { toolName: "move", isError: false, content: [], details: { status: "timeout" } });
	assert.equal(await call(f, "move"), undefined);
});

test("advice and look-up tools are not motion", () => {
	for (const tool of ["suggest_grasp", "back_project", "view_env_state", "plan_grasp"])
		assert.ok(NON_MOTION.has(tool), tool);
});

test("motion classification over every robot's tools", () => {
	const fixture = JSON.parse(readFileSync(new URL("./fixtures/tool-schemas.json", import.meta.url), "utf8"))
		.robots as Record<string, { name: string }[]>;
	const names = new Set(Object.values(fixture).flatMap((tools) => tools.map((t) => t.name)));
	// Tools the robot base or a capability registers, not robot.tool: never counted as motion (../robot.ts moves()).
	const notRobotTools = new Set(["finish", "request_operator_verdict", "write_audit"]);
	const motion = [...names].filter((n) => !notRobotTools.has(n) && !NON_MOTION.has(n)).sort();
	assert.deepEqual(motion, [
		"act",
		"close_gripper",
		"execute_grasp",
		"execute_place",
		"go_home",
		"goto_pose",
		"gr00t_act",
		"grasp_object",
		"home_pose",
		"lingbot_act",
		"move_base",
		"move_delta",
		"move_grip",
		"move_hand",
		"move_pose",
		"move_to",
		"navigate_to",
		"navigate_to_pose",
		"open_gripper",
		"openvla_act",
		"openvla_oft_act",
		"pi0_doubled",
		"pi0_pick",
		"recover_joint_posture",
		"release",
		"request_scene_reset",
		"reset",
		"rldx_arm",
		"rldx_skill",
		"rotate_delta",
		"rotate_pitch",
		"rotate_wrist",
		"rotate_yaw",
		"scripted_grasp",
		"set_gripper",
		"vla_grasp",
		"vla_handoff",
		"vla_left_place",
		"vla_right_grasp",
	]);
});

// ---------------------------------------------------------------------------
// --approval

/** The toy plus the given motion tools. */
function toyWith(f: ReturnType<typeof fakePi>, names: string[], extra: Partial<RobotSpec> = {}) {
	const robot = toy(f, ["move", "view_env_state", ...names], extra);
	const ok = async () => ({ content: [{ type: "text" as const, text: "ok" }], details: {} });
	for (const n of names) robot.tool(n, n, Type.Object({}), ok);
	return robot;
}

test("--approval standard asks only about high-risk motions in simulation", async (t) => {
	const f = fakePi({ approval: "standard" }, { hasUI: true, confirm: false });
	t.after(f.restore);
	toyWith(f, ["move_delta", "execute_grasp", "reset", "move_to"]);
	await f.emit("session_start");
	assert.equal(await call(f, "move_delta", { delta_xyz: [0.02, 0, 0] }), undefined, "a small move runs");
	assert.equal(await call(f, "view_env_state"), undefined);
	for (const [tool, input] of [
		["move_delta", { delta_xyz: [0.2, 0, 0] }],
		["execute_grasp", { grasp_id: "g1" }],
		["reset", {}],
		["move_to", { xyz: [0.1, 0, 0.3] }],
	] as const)
		assert.match((await call(f, tool, input))?.reason ?? "", /not approved by the operator/, tool);
	assert.deepEqual(f.confirms, [
		"Approve move_delta?",
		"Approve execute_grasp?",
		"Approve reset?",
		"Approve move_to?",
	]);
	assert.equal((await end(f)).approval_requests, 4);
	assert.equal(highRisk("move_delta", { delta_xyz: [0.05, 0, 0] }, false, 0.1), false);
	assert.equal(highRisk("move_delta", { delta_xyz: [0.05, 0, 0] }, true, 0.1), true, "real: every motion");
	assert.equal(highRisk("run_code", { code: "" }, false, 0.1), true);
	// Stage B's names: move_grip is a move to a target, the renamed gripper tools are not high risk in simulation.
	assert.equal(highRisk("move_grip", { xyz: [0.1, 0, 0.3] }, false, 0.1), true);
	for (const tool of ["set_gripper", "open_gripper", "close_gripper"])
		assert.equal(highRisk(tool, {}, false, 0.1), false, tool);
});

test("a real robot defaults to --approval human; one prompt covers a real-robot program", async (t) => {
	const f = fakePi({}, { hasUI: true, confirm: true });
	t.after(f.restore);
	toy(f, ["move", "view_env_state"], { explore: undefined, code: undefined });
	await f.emit("session_start");
	assert.equal(await call(f, "move"), undefined);
	assert.deepEqual(f.confirms, [], "simulation: off");
	const g = fakePi({}, { hasUI: true, confirm: true });
	t.after(g.restore);
	const real = { real: true, rpc: () => ({}) as never, observe: async () => ({ content: [], details: {} }) };
	toy(g, ["move", "view_env_state"], { code: real as RobotSpec["code"] });
	await g.emit("session_start");
	assert.equal(await call(g, "move"), undefined);
	assert.deepEqual(g.confirms, ["Approve move?"], "real: human");
	// Asked for explicitly without a UI, the robot does not start.
	const h = fakePi({ approval: "standard" });
	t.after(h.restore);
	toy(h, ["move", "view_env_state"], { code: real as RobotSpec["code"] });
	await h.emit("session_start");
	await h.emit("session_shutdown");
	assert.match(result(h)?.error ?? "", /--approval standard needs an operator UI/);
	// run_code: the gate shows the program under code mode's title; code mode then asks no more.
	const k = fakePi({ approval: "human" }, { hasUI: true, confirm: false });
	t.after(k.restore);
	toyWith(k, ["run_code"], { code: real as RobotSpec["code"] });
	await k.emit("session_start");
	await call(k, "run_code", { code: "move_to([0, 0, 0.3])" });
	assert.deepEqual(k.confirms, ["Run this program on the robot?"]);
});

test("--approval off (the simulation default) reviews nothing and leaves the result without approval fields", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	toy(f);
	await f.emit("session_start");
	assert.equal(await call(f, "move"), undefined);
	assert.equal(f.asked.length, 0);
	const r = await end(f);
	assert.equal("approval" in r, false);
	assert.equal(f.entries.filter((e) => e.type === APPROVAL_ENTRY).length, 0);
});

test("--approval reviewed before any tool result has a frame (units mode): the reviewer gets a fresh observation", async (t) => {
	const f = fakePi({ approval: "reviewed" });
	t.after(f.restore);
	const robot = toy(f);
	let looks = 0;
	// The robot's look-only tool returns the current camera frame (called directly, not by the planner).
	robot.tool("view_env_state", "look", Type.Object({}), async () => {
		looks++;
		return { content: [{ type: "image", data: "fresh", mimeType: "image/png" }], details: {} };
	});
	await f.emit("session_start");
	assert.equal(f.branch.length, 0, "no tool result yet");
	assert.equal(await call(f, "move"), undefined, "approved");
	assert.equal(looks, 1);
	assert.equal(f.asked[0].content[1].data, "fresh");
	assert.match(f.asked[0].content[0].text, /The 1 image\(s\) are the robot's latest camera views/);
	const entry = f.entries.find((e) => e.type === APPROVAL_ENTRY)?.data;
	assert.equal(entry.images, 1);
	assert.equal(entry.images_from, "observation");
});

test("--approval reviewed: a rejection blocks the motion with the reviewer's reason; looking is not reviewed", async (t) => {
	const f = fakePi(
		{ approval: "reviewed", "approval-model": "faux/reviewer" },
		{ review: 'Sure. {"decision":"reject","reason":"the gripper is above the wrong bowl"}' },
	);
	t.after(f.restore);
	toy(f);
	const shot = (toolName: string, data: string) => ({
		type: "message",
		message: { role: "toolResult", toolName, content: [{ type: "image", data, mimeType: "image/png" }] },
	});
	// The latest camera view, not the later segmentation overlay.
	f.branch.push(shot("view_env_state", "main"), shot("segment", "overlay"));
	await f.emit("session_start");
	const blocked = await call(f, "move", { xyz: [0.1, 0, 0.2] });
	assert.equal(blocked?.block, true);
	assert.match(blocked.reason, /not approved by the reviewer \(reject\): the gripper is above the wrong bowl/);
	assert.equal(await call(f, "view_env_state"), undefined);
	assert.equal(f.asked.length, 1, "one review, for the motion only");
	const [q] = f.asked;
	assert.match(q.system ?? "", /independent action reviewer/);
	assert.match(
		q.content[0].text,
		/TASK: put the cube in the bowl[\s\S]*TOOL CONTRACT \(move\): Move the gripper\. gripper -1 = open[\s\S]*PROPOSED CALL: move[\s\S]*0\.2/,
	);
	assert.equal(q.content[1].data, "main", "the latest camera image goes with it");
	assert.deepEqual(
		f.events.filter((e) => e.channel === VLM_COST_EVENT).map((e) => e.data),
		[0.25],
	);
	const entry = f.entries.find((e) => e.type === APPROVAL_ENTRY)?.data;
	assert.equal(entry.decision, "reject");
	assert.equal(entry.model, "faux/reviewer");
	const r = await end(f);
	assert.equal(r.approval, "reviewed");
	assert.equal(r.approval_rejected, 1);
	assert.equal(r.approval_cost_usd, 0.25);
	assert.equal(r.cost_usd, 0.25, "the review counts toward --max-cost");
});

test("--approval reviewed: an approval lets the motion run; an unparseable reply blocks it", async (t) => {
	const f = fakePi({ approval: "reviewed" });
	t.after(f.restore);
	toy(f);
	await f.emit("session_start");
	assert.equal(await call(f, "move"), undefined);
	const g = fakePi({ approval: "reviewed" }, { review: "looks good to me" });
	t.after(g.restore);
	toy(g);
	await g.emit("session_start");
	const blocked = await call(g, "move");
	assert.match(blocked?.reason ?? "", /not a decision/);
	assert.equal((await end(g)).approval_errors, 1);
	assert.deepEqual(parseReview('{"decision":"Abstain"}'), {
		decision: "abstain",
		reason: "reviewer decision: abstain",
	});
	assert.match(reviewPrompt("t", "run_code", { code: "x" }, 0), /No camera image/);
});

test("--approval human: the operator confirms each motion; without a UI the robot does not start", async (t) => {
	const f = fakePi({ approval: "human" }, { hasUI: true, confirm: false });
	t.after(f.restore);
	toy(f);
	await f.emit("session_start");
	const blocked = await call(f, "move");
	assert.match(blocked?.reason ?? "", /not approved by the operator/);
	assert.deepEqual(f.confirms, ["Approve move?"]);
	const g = fakePi({ approval: "human" }, { hasUI: true, confirm: true });
	t.after(g.restore);
	toy(g);
	await g.emit("session_start");
	assert.equal(await call(g, "move"), undefined);
	for (const [flags, why] of [
		[{ approval: "human" }, /needs an operator UI/],
		[{ approval: "yolo" }, /--approval must be one of off, standard, human, reviewed/],
	] as const) {
		const h = fakePi(flags);
		t.after(h.restore);
		toy(h);
		await h.emit("session_start");
		await h.emit("session_shutdown");
		const r = result(h);
		assert.equal(r?.env_error, true);
		assert.match(r?.error ?? "", why);
	}
});

// ---------------------------------------------------------------------------
// budgets

test("--max-tool-calls ends the episode once the planner made that many calls (finish excluded)", async (t) => {
	const f = fakePi({ "max-tool-calls": "2" });
	t.after(f.restore);
	toy(f);
	await f.emit("session_start");
	assert.equal(await call(f, "view_env_state"), undefined);
	assert.equal(await call(f, "move"), undefined);
	const spent = await call(f, "move");
	assert.deepEqual(spent, {
		block: true,
		reason: "Planner tool_calls budget exhausted; the episode is over.",
		terminate: true,
	});
	const r = await end(f);
	assert.equal(r.planner_budget_exhausted, "tool_calls");
	assert.equal(r.tool_calls, 2);
	assert.equal(r.max_tool_calls, 2);
});

test("--max-tokens counts the planner's input (cache included) and output tokens", async (t) => {
	const f = fakePi({ "max-tokens": "200" });
	t.after(f.restore);
	toy(f);
	await f.emit("session_start");
	const usage = { input: 100, cacheRead: 40, cacheWrite: 10, output: 60, cost: { total: 0 } };
	await f.emit("message_end", { message: { role: "assistant", content: [], usage, stopReason: "toolUse" } });
	const spent = await call(f, "move");
	assert.match(spent?.reason ?? "", /tokens budget exhausted/);
	const r = await end(f);
	assert.deepEqual(r.planner_tokens, { input: 150, output: 60 });
	assert.equal(r.planner_budget_exhausted, "tokens");
	assert.equal(r.max_tokens, 200);
});

// ---------------------------------------------------------------------------
// the context version record

test("the result and a session entry record the context version", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	toy(f);
	await f.emit("session_start");
	const reply = (model: string, responseModel?: string) => ({
		message: { role: "assistant", provider: "faux", model, responseModel, content: [], stopReason: "toolUse" },
	});
	await f.emit("message_end", reply("m1"));
	await f.emit("message_end", reply("alias", "m1"));
	const r = await end(f);
	const v = r.context_version;
	assert.equal(v.system_prompt_sha256, sha256("the system prompt"));
	assert.equal(
		v.templates["closed-loop.md"],
		sha256(readFileSync(new URL("../src/planner/closed-loop.md", import.meta.url))),
	);
	assert.equal(v.templates["libero/explore.md"], undefined, "not exploring");
	assert.deepEqual(v.memory_files, {});
	assert.equal(v.code_api_digest, null);
	assert.match(v.git_commit, /^([0-9a-f]{40}|unknown)$/);
	assert.equal(v.git_commit, gitCommit());
	assert.equal(v.git_dirty, gitDirty());
	assert.ok(v.git_dirty === null || typeof v.git_dirty === "boolean");
	assert.deepEqual(v.planner_models, ["faux/m1"]);
	assert.deepEqual(f.entries.find((e) => e.type === CONTEXT_VERSION_ENTRY)?.data, v);
});

test("template ids are stable across the src/ re-layering", () => {
	for (const [path, id] of [
		["robots/libero/SYSTEM.md", "libero/SYSTEM.md"],
		["libero/SYSTEM.md", "libero/SYSTEM.md"],
		["robots/libero/compact/memory-hf.md", "libero/compact/memory-hf.md"],
		["modes/code/SYSTEM.md", "code/SYSTEM.md"],
		["planner/closed-loop.md", "closed-loop.md"],
		["closed-loop.md", "closed-loop.md"],
	])
		assert.equal(templateId(path), id, path);
	assert.deepEqual(normalizeTemplates({ "planner/closed-loop.md": "a", "libero/SYSTEM.md": "b" }), {
		"closed-loop.md": "a",
		"libero/SYSTEM.md": "b",
	});
});

test("usedTemplates keeps the templates this mode uses", () => {
	const plain = usedTemplates({ explore: false, memoryProfile: "hf", code: false, units: false });
	assert.ok(plain["libero/SYSTEM.md"]);
	assert.ok(plain["libero/compact/memory-hf.md"]);
	for (const unused of [
		"libero/compact/memory-local.md",
		"libero/explore.md",
		"libero/distil.md",
		"libero/compact/explore.md",
		"code/SYSTEM.md",
		"units/SYSTEM.md",
	])
		assert.equal(plain[unused], undefined, unused);
	const exploring = usedTemplates({ explore: true, memoryProfile: "local", code: true, units: false });
	for (const used of ["libero/compact/memory-local.md", "libero/explore.md", "libero/distil.md", "code/SYSTEM.md"])
		assert.ok(exploring[used], used);
	assert.equal(exploring["libero/compact/memory-hf.md"], undefined);
});

test("memory records the corpus files the agent read, with their digest", async (t) => {
	const f = fakePi({ "memory-profile": "local" });
	t.after(f.restore);
	const root = join(f.dir, "corpus");
	mkdirSync(join(root, "suite"), { recursive: true });
	writeFileSync(join(root, "MEMORY.md"), "# index\n");
	writeFileSync(join(root, "suite", "lesson.md"), "grasp low\n");
	writeFileSync(join(f.dir, "elsewhere.md"), "not memory\n");
	const mem = memory(f.pi, { cell: () => ({ tag: "cell", reference: "ref" }), home: () => f.dir });
	f.flags["memory-dir"] = root;
	await f.emit("session_start");
	for (const path of [join(root, "suite", "lesson.md"), join(f.dir, "elsewhere.md")])
		await f.emit("tool_result", { toolName: "read", isError: false, input: { path }, content: [] });
	assert.deepEqual(mem.loaded(), { "suite/lesson.md": sha256("grasp low\n") });
});
