import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, realpathSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import dualFranka from "../src/dual_franka/index.ts";
import franka from "../src/franka/index.ts";
import { defineRobot } from "../src/robot.ts";
import {
	compensate,
	ground,
	latestTurn,
	type Move,
	STATE_ENTRY,
	UNITS_EVENT,
	type UnitsHandle,
	type UnitsSpec,
	VERIFY_ENTRY,
	VIDEO_REF_ENTRY,
} from "../src/units/index.ts";
import { parseVerdict, renderBrief, validateBrief } from "../src/units/vlm.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi that runs handlers in registration order, like pi's runner; `vlm` answers side model calls in order. */
function fakePi(flagValues: Record<string, unknown> = {}, vlm: (string | Error)[] = [], branch: any[] = []) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	let active: string[] = [];
	const emitted = new Map<string, unknown>();
	const entries: { customType: string; data: any }[] = [];
	/** The side model calls: the model and the user message. */
	const asked: { model: string; content: any[] }[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: (customType: string, data: unknown) => {
			entries.push({ customType, data });
			branch.push({ type: "custom", customType, data });
		},
		getThinkingLevel: () => "low",
		events: { emit: (channel: string, data: unknown) => emitted.set(channel, data), on: () => () => {} },
	} as unknown as ExtensionAPI;
	const notes: string[] = [];
	const dir = realpathSync(mkdtempSync(join(tmpdir(), "units-")));
	const ctx = {
		hasUI: true,
		cwd: dir,
		ui: {
			notify: (m: string) => notes.push(m),
			setWidget: () => {},
			select: async () => "done",
		},
		shutdown: () => {},
		sessionManager: { getBranch: () => branch, getSessionDir: () => dir },
		model: { provider: "relay", id: "planner" },
		modelRegistry: {
			find: (provider: string, id: string) => ({ provider, id }),
			streamSimple: (model: { provider: string; id: string }, context: { messages: { content: any[] }[] }) => ({
				result: async () => {
					asked.push({ model: `${model.provider}/${model.id}`, content: context.messages[0].content });
					const reply = vlm.shift() ?? "";
					if (reply instanceof Error) return { stopReason: "error", errorMessage: reply.message, content: [] };
					return { stopReason: "stop", content: [{ type: "text", text: reply }] };
				},
			}),
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
	/** A flag given on the command line: pi parses flags after the extensions load. */
	const setFlag = (name: string, value: unknown) => {
		flags[name] = value;
	};
	return { pi, emit, run, tools, notes, emitted, entries, asked, branch, setFlag, active: () => active };
}

const VECTORS: UnitsSpec["vectors"] = {
	MV_FWD: [1, 0, 0],
	MV_BACK: [-1, 0, 0],
	MV_LEFT: [0, -1, 0],
	MV_RIGHT: [0, 1, 0],
	MV_UP: [0, 0, 1],
	MV_DOWN: [0, 0, -1],
};

/** A toy arm: `apply` integrates the moves (or not, when blocked); `onClose` decides the closed width. */
async function toyRobot(
	flags: Record<string, unknown>,
	o: {
		vlm?: (string | Error)[];
		branch?: any[];
		yaw?: number;
		onClose?: () => number;
		blocked?: boolean;
		prompt?: string;
		maxYaw?: number;
		maxMove?: number;
		chains?: boolean;
		arms?: string[];
		viewSelect?: boolean;
		/** The robot refuses the verifier's retreat (e.g. an episode that already ended). */
		refuseRetreat?: boolean;
		reset?: () => Promise<Record<string, unknown>>;
	} = {},
) {
	// The Show-Harness step rules: fixed 2 cm steps unless a test turns variable_step on.
	const f = fakePi(
		{ units: true, "units-plugins": "recovery,auto_release,proprioception,plan,mem_text", ...flags },
		o.vlm,
		o.branch,
	);
	const moves: Move[] = [];
	const pos = [0.5, 0, 0.2];
	let width = 0.08;
	const spec: UnitsSpec = {
		vectors: VECTORS,
		stepM: 0.02,
		...(o.yaw ? { yawStepRad: o.yaw } : {}),
		...(o.maxYaw !== undefined ? { maxYawRad: () => o.maxYaw as number } : {}),
		...(o.maxMove !== undefined ? { maxMoveM: () => o.maxMove as number } : {}),
		...(o.chains !== undefined ? { chains: () => o.chains as boolean } : {}),
		...(o.arms ? { arms: o.arms } : {}),
		...(o.viewSelect !== undefined ? { viewSelect: () => o.viewSelect as boolean } : {}),
		apply: async (move) => {
			if (move.retreat && o.refuseRetreat)
				return { content: [{ type: "text", text: "Episode already ended." }], details: { terminated: true } };
			moves.push(move);
			if (!o.blocked) for (let i = 0; i < 3; i++) pos[i] += move.delta[i];
			if (move.gripper === "open") width = 0.08;
			if (move.gripper === "close") width = o.onClose?.() ?? 0.03;
			return {
				content: [
					{ type: "text", text: "obs" },
					{ type: "image", data: `m${moves.length}`, mimeType: "image/png" },
				],
				details: {},
			};
		},
		state: async () => ({ eef_xyz: [...pos], gripper_width: width, table_z: 0 }),
		instruction: () => "put the cube in the bowl",
		emptyWidthM: 0.005,
	};
	defineRobot(f.pi, {
		name: "toy",
		task: [],
		keepImages: 2,
		start: async () => ["move_to", "segment", "finish"],
		...(o.prompt ? { prompt: () => o.prompt } : {}),
		...(o.reset ? { operator: { step: () => 0, reset: o.reset } } : {}),
		result: () => ({}),
		finish: {
			description: "finish",
			parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
			result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
		},
		units: spec,
	});
	await f.emit("session_start");
	const setWidth = (w: number) => {
		width = w;
	};
	return { ...f, moves, pos, setWidth };
}

const head = (r: any) => r.content[0].text as string;

test("units ground into base-frame moves; rotation only with a yaw step", () => {
	assert.deepEqual(ground({ vectors: VECTORS, stepM: 0.02 }, "MV_LEFT"), {
		delta: [0, -0.02, 0],
		yaw: 0,
		gripper: null,
	});
	assert.deepEqual(ground({ vectors: VECTORS, stepM: 0.02 }, "GRASP")?.gripper, "close");
	assert.deepEqual(ground({ vectors: VECTORS, stepM: 0.02 }, "RELEASE")?.gripper, "open");
	assert.deepEqual(ground({ vectors: VECTORS, stepM: 0.02 }, "STOP"), { delta: [0, 0, 0], yaw: 0, gripper: null });
	assert.equal(ground({ vectors: VECTORS, stepM: 0.02, yawStepRad: 0.15 }, "ROTATE_CCW")?.yaw, -0.15);
	assert.throws(() => ground({ vectors: VECTORS, stepM: 0.02 }, "ROTATE_CW"), /no yaw/);
	assert.equal(ground({ vectors: VECTORS, stepM: 0.02 }, "DONE"), undefined);
});

test("--units hides the robot's tools and swaps in the units prompt", async () => {
	const f = await toyRobot({});
	assert.deepEqual(f.active(), ["act", "plan", "finish"]);
	const prompt = (await f.emit("before_agent_start")).systemPrompt as string;
	assert.match(prompt, /TASK: put the cube in the bowl/);
	assert.doesNotMatch(prompt, /ROTATE_CW/, "no rotation units without a yaw step");
	assert.doesNotMatch(prompt, /\[\/?\w+\]/, "no section markers left");
	const schema = JSON.stringify(f.tools.get("act").parameters);
	assert.doesNotMatch(schema, /ROTATE/);
	const withYaw = await toyRobot({ "units-plugins": "rotation" }, { yaw: 0.15 });
	assert.match(JSON.stringify(withYaw.tools.get("act").parameters), /ROTATE_CW/);
	assert.match(
		(await withYaw.emit("before_agent_start")).systemPrompt,
		/ROTATE_CW, ROTATE_CCW: turn the gripper about 9 degrees/,
	);
	assert.match(
		JSON.stringify((await toyRobot({ "units-plugins": "" }, { yaw: 0.15 })).tools.get("act").parameters),
		/ROTATE/,
		"a yaw step is enough for ROTATE_*",
	);
	const off = await toyRobot({ units: "false" });
	assert.deepEqual(off.active(), ["move_to", "segment", "finish"]);
	const both = await toyRobot({ units: "both" }, { prompt: "ROBOT PROMPT" });
	assert.deepEqual(both.active(), ["move_to", "segment", "finish", "act", "plan"]);
	const bothPrompt = (await both.emit("before_agent_start")).systemPrompt as string;
	assert.match(bothPrompt, /^ROBOT PROMPT\n\n# Action units\n/);
	assert.doesNotMatch(bothPrompt, /one `act` call per reply/);
});

test("act repeats a unit n times and reports proprioception", async () => {
	const f = await toyRobot({});
	const r = await f.run("act", { unit: "MV_LEFT", n: 3 });
	assert.equal(f.moves.length, 3);
	assert.deepEqual(f.moves[0], { delta: [0, -0.02, 0], yaw: 0, gripper: null, continuous: true });
	assert.deepEqual(
		f.moves.map((m) => m.continuous),
		[true, true, undefined],
		"continuous while the same MV_* unit runs next in this call",
	);
	assert.match(head(r), /^units: MV_LEFT x3\n/);
	assert.match(head(r), /20\.0 cm above the table; width 8\.0 cm, commanded OPEN/);
	assert.match(head(r), /Recent moves, newest first: MV_LEFT, MV_LEFT, MV_LEFT/);
	assert.equal(r.content[1].text, "obs", "the robot's observation follows the units block");
	const done = await f.run("act", { unit: "DONE" });
	assert.match(done.content[0].text, /call `finish`/);
	assert.equal(f.moves.length, 3, "DONE does not move");
});

test("a blocked move stops the repeat and says so", async () => {
	const f = await toyRobot({}, { blocked: true });
	const r = await f.run("act", { unit: "MV_DOWN", n: 4 });
	assert.equal(f.moves.length, 1);
	assert.match(head(r), /x1 of 4 \(stopped early\)/);
	assert.match(head(r), /Last MV_DOWN lowered 0\.0 of 2\.0 cm -> already in contact, do NOT MV_DOWN again/);
});

test("recovery reopens a GRASP that closed on nothing", async () => {
	const f = await toyRobot({}, { onClose: () => 0.001 });
	const r = await f.run("act", { unit: "GRASP" });
	assert.deepEqual(
		f.moves.map((m) => m.gripper),
		["close", "open"],
	);
	assert.match(head(r), /Recovery: Empty close/);
	assert.match(head(r), /commanded OPEN/);
	// Without the plugin the empty close stays closed.
	const g = await toyRobot({ "units-plugins": "proprioception" }, { onClose: () => 0.001 });
	await g.run("act", { unit: "GRASP" });
	assert.deepEqual(
		g.moves.map((m) => m.gripper),
		["close"],
	);
	// A human's GRASP (GUMI teleop) stays closed even with the plugin on.
	const h = await toyRobot({}, { onClose: () => 0.001 });
	await h.run("act", { unit: "GRASP", operator: true });
	assert.deepEqual(
		h.moves.map((m) => m.gripper),
		["close"],
	);
});

test("auto_release reopens a closed gripper whose object slipped out", async () => {
	const f = await toyRobot({});
	await f.run("act", { unit: "GRASP" });
	const held = await f.run("act", { unit: "MV_UP" });
	assert.doesNotMatch(head(held), /Recovery/);
	f.setWidth(0.0005);
	const r = await f.run("act", { unit: "MV_UP", n: 3 });
	assert.deepEqual(
		f.moves.map((m) => m.gripper),
		["close", null, null, "open"],
	);
	assert.match(head(r), /x1 of 3/);
	assert.match(head(r), /Recovery: Grasp lost/);
});

test("plan tracks the current stage in every act result", async () => {
	const f = await toyRobot({});
	await f.run("plan", {
		stages: [
			{ motion: "GRASP", target: "cube", completion: "cube between the fingers" },
			{ motion: "REASON", target: "bowls", description: "IF ... THEN ...", completion: "both bowls visible" },
		],
	});
	assert.match(
		head(await f.run("act", { unit: "STOP" })),
		/STAGE 1\/2 \[GRASP\]: target cube; DONE WHEN cube between/,
	);
	await f.run("plan", { done: true });
	const r = head(await f.run("act", { unit: "STOP" }));
	assert.match(r, /STAGE 2\/2 \[REASON\]/);
	assert.match(r, /Decision point/);
});

test("--stateless keeps the task and the latest observation turn", async () => {
	const img = { type: "image", data: "", mimeType: "image/png" };
	const messages = [
		{ role: "user", content: "Solve the task." },
		{ role: "assistant", content: [{ type: "toolCall", name: "act" }] },
		{ role: "toolResult", content: [{ type: "text", text: "a" }, img] },
		{ role: "assistant", content: [{ type: "toolCall", name: "act" }] },
		{ role: "toolResult", content: [{ type: "text", text: "b" }, img] },
		{ role: "assistant", content: [{ type: "toolCall", name: "plan" }] },
		{ role: "toolResult", content: [{ type: "text", text: "plan" }] },
	];
	assert.deepEqual(latestTurn(messages), [messages[0], ...messages.slice(3)]);
	assert.equal(latestTurn(messages.slice(0, 3)), undefined, "nothing to prune");
	const f = await toyRobot({ stateless: true });
	const out = await f.emit("context", { messages });
	assert.equal(out.messages.length, 5);
	assert.match((await f.emit("before_agent_start")).systemPrompt, /Only your latest step stays in context/);
	const g = await toyRobot({});
	const kept = await g.emit("context", { messages });
	assert.ok(!kept || kept.messages.length === messages.length, "without --stateless the history stays");
});

test("Franka and dual Franka mount units on their move primitives, with the workspace flags", async () => {
	for (const [robot, box] of [
		[franka, "0.159,1.159,-0.456,0.544"],
		[dualFranka, "0.1,1.15,-0.85,0.85"],
	] as const) {
		const f = fakePi({ units: "true", "units-plugins": "rotation" });
		robot(f.pi);
		assert.equal(f.pi.getFlag("workspace-xy"), box);
		assert.equal(f.pi.getFlag("z-floor"), "");
		const schema = JSON.stringify(f.tools.get("act").parameters);
		assert.match(schema, /ROTATE_CW/);
		assert.equal(/"arm"/.test(schema), robot === dualFranka, "only the dual arm picks an arm");
		// Before the robot is up, a unit goes through the robot's own tool path and is refused.
		const r = await f.run("act", { unit: "MV_FWD", ...(robot === dualFranka ? { arm: "left" } : {}) });
		assert.match(r.content.at(-1).text, /robot not initialized/);
	}
});

test("variable_step: coarse for MV_UP, high above the table, or the target not in the wrist view", async () => {
	const f = await toyRobot({ "units-plugins": "variable_step", "units-coarse-step": "0.05" });
	assert.equal(f.pi.getFlag("units-coarse-step"), "0.05");
	// 20 cm above the table (> 8 cm): coarse even with the target in the wrist view.
	await f.run("act", { unit: "MV_LEFT", target_in_wrist: true });
	assert.deepEqual(f.moves.at(-1)?.delta, [0, -0.05, 0]);
	await f.run("act", { unit: "MV_DOWN", n: 3, target_in_wrist: true });
	// 5 cm above: fine with the target in view, coarse without it, and coarse for MV_UP.
	await f.run("act", { unit: "MV_FWD", target_in_wrist: true });
	assert.deepEqual(f.moves.at(-1)?.delta, [0.02, 0, 0]);
	await f.run("act", { unit: "MV_FWD", target_in_wrist: false });
	assert.deepEqual(f.moves.at(-1)?.delta, [0.05, 0, 0]);
	await f.run("act", { unit: "MV_UP", target_in_wrist: true });
	assert.deepEqual(f.moves.at(-1)?.delta, [0, 0, 0.05]);
	const prompt = (await f.emit("before_agent_start")).systemPrompt as string;
	assert.match(prompt, /WRIST CHECK/);
	assert.match(prompt, /coarse \(~5 cm\) for MV_UP, while the gripper is more than 8 cm above the table/);
	const off = await toyRobot({});
	assert.ok(!("target_in_wrist" in off.tools.get("act").parameters.properties), "no wrist signal without its plugins");
});

test("action_chunk: a plan of distinct moves runs only while the target is not in the wrist view", async () => {
	const f = await toyRobot({ "units-plugins": "action_chunk" });
	const far = await f.run("act", { unit: "MV_FWD", target_in_wrist: false, plan: ["MV_FWD", "MV_LEFT", "MV_DOWN"] });
	assert.deepEqual(
		f.moves.map((m) => m.delta),
		[
			[0.02, 0, 0],
			[0, -0.02, 0],
			[0, 0, -0.02],
		],
	);
	assert.match(head(far), /^units: MV_FWD, MV_LEFT, MV_DOWN\n/);
	assert.ok(
		f.moves.every((m) => !m.continuous),
		"distinct moves do not chain",
	);
	const near = await f.run("act", { unit: "MV_LEFT", target_in_wrist: true, plan: ["MV_FWD", "MV_FWD"] });
	assert.equal(f.moves.length, 4, "near the target one unit runs");
	assert.deepEqual(f.moves.at(-1)?.delta, [0, -0.02, 0]);
	assert.match(head(near), /plan ignored/);
	assert.match((await f.emit("before_agent_start")).systemPrompt, /ACTION PLAN, only when `target_in_wrist` is false/);
});

test("rotation: wrist-judged moves follow the turned gripper; MV_UP while holding turns back first", async () => {
	assert.deepEqual(
		compensate([0.02, 0, 0], Math.PI / 2).map((v) => Number(v.toFixed(6))),
		[0, 0.02, 0],
	);
	const f = await toyRobot({ "units-plugins": "rotation" }, { yaw: Math.PI / 4 });
	await f.run("act", { unit: "ROTATE_CW", n: 2 });
	assert.deepEqual(
		f.moves.map((m) => m.yaw),
		[Math.PI / 4, Math.PI / 4],
	);
	// Judged in the wrist view (turned 90 deg): MV_FWD becomes base +y; judged in the agentview it stays +x.
	await f.run("act", { unit: "MV_FWD", target_in_wrist: true });
	assert.deepEqual(
		f.moves.at(-1)?.delta.map((v) => Number(v.toFixed(6))),
		[0, 0.02, 0],
	);
	await f.run("act", { unit: "MV_FWD", target_in_wrist: false });
	assert.deepEqual(f.moves.at(-1)?.delta, [0.02, 0, 0]);
	// The soft joint-travel guard (150 deg) refuses a third and fourth turn beyond it.
	await f.run("act", { unit: "ROTATE_CW" });
	const refused = await f.run("act", { unit: "ROTATE_CW" });
	assert.match(head(refused), /ROTATE_CW refused: the gripper is already turned 135 deg/);
	await f.run("act", { unit: "GRASP" });
	const up = await f.run("act", { unit: "MV_UP" });
	assert.deepEqual(f.moves.at(-1), { delta: [0, 0, 0], yaw: (-3 * Math.PI) / 4, gripper: null });
	assert.match(head(up), /MV_UP\(realign\)/);
	await f.run("act", { unit: "MV_UP" });
	assert.deepEqual(f.moves.at(-1)?.delta, [0, 0, 0.02], "back at the start heading, MV_UP lifts");
	await f.emit("session_start");
	await f.run("act", { unit: "MV_FWD", target_in_wrist: true });
	assert.deepEqual(f.moves.at(-1)?.delta, [0.02, 0, 0], "an episode reset clears the accumulated yaw");
});

test("rotation: no rotate command exceeds the robot's per-call limit; the realign runs in pieces", async () => {
	const f = await toyRobot({ "units-plugins": "rotation" }, { yaw: Math.PI / 4, maxYaw: 0.5 });
	await f.run("act", { unit: "ROTATE_CW", n: 3 });
	assert.equal(f.moves.length, 6, "each 45 deg unit runs as two 22.5 deg commands");
	await f.run("act", { unit: "GRASP" });
	const up = await f.run("act", { unit: "MV_UP" });
	assert.match(head(up), /MV_UP\(realign\)/);
	const turns = f.moves.map((m) => m.yaw).filter((y) => y !== 0);
	assert.ok(
		turns.every((y) => Math.abs(y) <= 0.5 + 1e-9),
		`every command within 0.5 rad: ${turns}`,
	);
	const back = turns.filter((y) => y < 0);
	assert.equal(back.length, 5);
	assert.ok(Math.abs(back.reduce((a, b) => a + b, 0) + (3 * Math.PI) / 4) < 1e-9, "the realign undoes the whole turn");
	await f.run("act", { unit: "MV_UP" });
	assert.deepEqual(f.moves.at(-1)?.delta, [0, 0, 0.02], "back at the start heading");
	// An invalid limit refuses the turn instead of sending it whole.
	const bad = await toyRobot({ "units-plugins": "" }, { yaw: 0.15, maxYaw: Number.NaN });
	assert.match(
		head(await bad.run("act", { unit: "ROTATE_CW" })),
		/ROTATE_CW refused: the robot's per-call rotation limit/,
	);
	assert.equal(bad.moves.length, 0);
});

test("the 150 deg accumulated-yaw cap holds without the rotation plugin", async () => {
	const f = await toyRobot({ "units-plugins": "" }, { yaw: 0.15 });
	for (let i = 0; i < 3; i++) await f.run("act", { unit: "ROTATE_CW", n: 10 });
	const total = f.moves.reduce((a, m) => a + m.yaw, 0);
	assert.ok(total <= (150 * Math.PI) / 180, `total ${total} rad`);
	assert.equal(f.moves.length, 17);
	const refused = await f.run("act", { unit: "ROTATE_CW" });
	assert.match(head(refused), /ROTATE_CW refused: the gripper is already turned 146 deg/);
	await f.run("act", { unit: "ROTATE_CCW", n: 2 });
	assert.equal(f.moves.length, 19, "turning back is allowed");
});

test("one act call travels at most the robot's per-call move limit in total", async () => {
	const f = await toyRobot({}, { maxMove: 0.05 });
	const r = await f.run("act", { unit: "MV_DOWN", n: 10 });
	assert.equal(f.moves.length, 2);
	assert.match(head(r), /MV_DOWN x2 of 10 \(stopped early\)/);
	assert.match(head(r), /MV_DOWN not run: one act call moves at most 0\.05 m in total/);
	const chunk = await toyRobot({ "units-plugins": "action_chunk" }, { maxMove: 0.03 });
	await chunk.run("act", { unit: "MV_FWD", target_in_wrist: false, plan: ["MV_FWD", "MV_LEFT", "MV_DOWN"] });
	assert.equal(chunk.moves.length, 1, "a chunk counts toward the same total");
	const nan = await toyRobot({}, { maxMove: Number.NaN });
	await nan.run("act", { unit: "MV_UP", n: 3 });
	assert.equal(nan.moves.length, 1, "an invalid limit stops the repeat (the robot refuses the first unit itself)");
});

test("a scene reset clears the units state (yaw, gripper, plan)", async () => {
	let fail = false;
	const f = await toyRobot(
		{ "units-plugins": "rotation,plan", operator: true },
		{
			yaw: Math.PI / 2,
			reset: async () => {
				if (fail) throw new Error("reset failed");
				return { ok: true };
			},
		},
	);
	await f.run("plan", { stages: [{ motion: "GRASP", target: "cube", completion: "held" }] });
	await f.run("act", { unit: "ROTATE_CW" });
	await f.run("act", { unit: "GRASP" });
	fail = true;
	await f.run("request_scene_reset", { reason: "retry" });
	await f.run("act", { unit: "MV_FWD", target_in_wrist: true });
	assert.deepEqual(
		f.moves.at(-1)?.delta.map((v) => Number(v.toFixed(6))),
		[0, 0.02, 0],
		"a failed reset keeps the state",
	);
	fail = false;
	const r = await f.run("request_scene_reset", { reason: "retry" });
	assert.match(r.content[0].text, /scene_reset_confirmed/);
	await f.run("act", { unit: "MV_FWD", target_in_wrist: true });
	assert.deepEqual(f.moves.at(-1)?.delta, [0.02, 0, 0], "yaw cleared");
	const up = await f.run("act", { unit: "MV_UP" });
	assert.doesNotMatch(head(up), /realign|STAGE/, "gripper and plan cleared");
	assert.deepEqual(f.moves.at(-1)?.delta, [0, 0, 0.02]);
});

test("Franka and dual Franka refuse to start without a valid Z floor and workspace box", async () => {
	for (const robot of [franka, dualFranka]) {
		for (const [flags, error] of [
			[{}, /--z-floor must be/],
			[{ "z-floor": "abc" }, /--z-floor must be/],
			[{ "z-floor": "0.14", "workspace-xy": "0.2,0.8,-0.4" }, /--workspace-xy must be/],
			[{ "z-floor": "0.14", "workspace-xy": "0.2,0.8,-0.4,nan" }, /--workspace-xy must be/],
			[{ "z-floor": "0.14", "workspace-xy": "0.8,0.2,-0.4,0.4" }, /--workspace-xy must be/],
		] as const) {
			const f = fakePi({ operator: true, python: "/nonexistent/python", ...flags });
			robot(f.pi);
			await f.emit("session_start");
			assert.match(f.notes.join("\n"), error, JSON.stringify(flags));
			assert.deepEqual(f.active(), []);
		}
		const ok = fakePi({ operator: true, python: "/nonexistent/python", "z-floor": "0.14" });
		robot(ok.pi);
		await ok.emit("session_start");
		assert.doesNotMatch(ok.notes.join("\n"), /--z-floor|--workspace-xy/, "valid limits pass the check");
	}
});

test("an operator's unit waits for the operator's scene confirmation, like the agent's act", async () => {
	let fail = true;
	const f = await toyRobot(
		{ operator: true },
		{
			reset: async () => {
				if (fail) throw new Error("reset failed");
				return { ok: true };
			},
		},
	);
	const handle = f.emitted.get(UNITS_EVENT) as UnitsHandle;
	assert.equal(handle.refuse(), undefined);
	// A reset that did not complete leaves the scene unconfirmed: the agent's act and the operator's units are refused.
	await f.run("request_scene_reset", { reason: "retry" });
	const refusal = "refused; request_scene_reset and obtain operator confirmation first";
	assert.equal((await f.emit("tool_call", { toolName: "act", input: { unit: "MV_FWD" } }))?.reason, refusal);
	assert.equal(handle.refuse(), refusal);
	fail = false;
	await f.run("request_scene_reset", { reason: "retry" });
	assert.equal(handle.refuse(), undefined);
});

test("mem_text: recent moves newest first, the history rules, and resets on empty grasps and stages", async () => {
	const f = await toyRobot({}, { onClose: () => 0.001 });
	await f.run("act", { unit: "MV_LEFT" });
	await f.run("act", { unit: "STOP" });
	const r = head(await f.run("act", { unit: "MV_FWD", n: 2 }));
	assert.match(r, /Recent moves, newest first: MV_FWD, MV_FWD, MV_LEFT\n/, "STOP is not a move");
	// An empty GRASP leaves only itself in the history, so the model sees it and does not grasp in place again.
	assert.match(head(await f.run("act", { unit: "GRASP" })), /Recent moves, newest first: GRASP\(empty\)\n/);
	for (let i = 0; i < 6; i++) await f.run("act", { unit: "MV_UP" });
	assert.match(head(await f.run("act", { unit: "STOP" })), /newest first: MV_UP, MV_UP, MV_UP, MV_UP, MV_UP\n/);
	await f.run("plan", { stages: [{ motion: "GRASP", target: "cube", completion: "held" }] });
	assert.match(head(await f.run("act", { unit: "STOP" })), /newest first: none\n/, "a new stage starts clean");
	const prompt = (await f.emit("before_agent_start")).systemPrompt as string;
	assert.match(
		prompt,
		/If the recent moves show GRASP\(empty\) \(a GRASP that closed on nothing\), do not GRASP in place again/,
	);
	assert.match(
		prompt,
		/Do not undo the newest recent move \(MV_LEFT \/ MV_RIGHT, MV_FWD \/ MV_BACK\) unless the images show it overshot/,
	);
	assert.match(prompt, /units block: what ran, the recent moves \(newest first\), the gripper's height/);
	// Off: no move history anywhere.
	const off = await toyRobot({ "units-plugins": "proprioception" });
	assert.doesNotMatch(head(await off.run("act", { unit: "MV_LEFT" })), /Recent/);
	assert.doesNotMatch((await off.emit("before_agent_start")).systemPrompt, /undo the newest|recent moves/);
	// Stateless: the history lives in the observation, so it stays in the kept turn.
	const s = await toyRobot({ stateless: true });
	await s.run("act", { unit: "MV_LEFT" });
	assert.match(head(await s.run("act", { unit: "MV_BACK" })), /newest first: MV_BACK, MV_LEFT/);
	assert.match(
		(await s.emit("before_agent_start")).systemPrompt,
		/Do not undo the newest recent move[\s\S]*latest result \(task, stage, recent moves/,
	);
	assert.match(String(s.pi.getFlag("units-plugins")), /mem_text/);
	const defaults = fakePi({ units: "true" });
	franka(defaults.pi);
	assert.match(
		String(defaults.pi.getFlag("units-plugins")),
		/(^|,)mem_text(,|$)/,
		"on by default, as in Show-Harness",
	);
});

test("verifier: a success finish is checked once on the latest images; NOT complete refuses it once", async () => {
	const no = '```json\n{"complete": false, "reason": "the soup can is beside the basket"}\n```';
	const f = await toyRobot({ "units-verify": "true", "units-vlm-model": "selfhost/muse" }, { vlm: [no, no] });
	await f.run("plan", { stages: [{ motion: "GRASP", target: "soup", completion: "held" }] });
	const obs = await f.run("act", { unit: "MV_LEFT" });
	await f.emit("tool_result", { toolName: "act", content: obs.content });
	await f.emit("tool_result", {
		toolName: "point",
		content: [{ type: "image", data: "marked", mimeType: "image/png" }],
	});
	const first = await f.emit("tool_call", { toolName: "finish", input: { status: "success", summary: "done" } });
	assert.equal(first?.block, true);
	assert.match(first.reason, /verifier judged the task NOT complete \(the soup can is beside the basket\)/);
	assert.match(first.reason, /send new stages with `plan`/);
	assert.equal(f.asked.length, 1);
	assert.equal(f.asked[0].model, "selfhost/muse");
	assert.match(f.asked[0].content[0].text, /TASK: put the cube in the bowl[\s\S]*Return JSON only/);
	// The retreat ran first: 5 MV_UP (10 cm in 2 cm steps) through apply, and the check saw its images.
	assert.deepEqual(
		f.moves.slice(1).map((m) => [m.delta[2], m.retreat]),
		Array(5).fill([0.02, true]),
	);
	assert.deepEqual(
		f.asked[0].content.slice(1).map((c: any) => c.data),
		["m6"],
		"the camera images after the retreat, not point's marked image",
	);
	const verified = f.entries.find((e) => e.customType === VERIFY_ENTRY);
	assert.ok(verified, "a units_verify entry");
	const [retreated] = verified.data.retreat;
	assert.equal(retreated.ran, "units: MV_UP x5");
	// The replan starts from a clean plan and history, and the reason stays in the observation.
	const r = head(await f.run("act", { unit: "MV_RIGHT" }));
	assert.doesNotMatch(r, /STAGE/);
	assert.match(r, /Verifier: the task is NOT complete: the soup can is beside the basket/);
	assert.match(r, /newest first: MV_RIGHT\n/);
	// The replan budget is one: the second NOT complete is recorded but lets finish through.
	assert.equal(
		await f.emit("tool_call", { toolName: "finish", input: { status: "success", summary: "done" } }),
		undefined,
	);
	const checks = f.entries.filter((e) => e.customType === VERIFY_ENTRY).map((e) => e.data);
	assert.deepEqual(
		checks.map((c) => [c.complete, c.refused, c.model]),
		[
			[false, true, "selfhost/muse"],
			[false, false, "selfhost/muse"],
		],
	);
	// A failure claim is not checked; without the flag nothing is checked.
	assert.equal(
		await f.emit("tool_call", { toolName: "finish", input: { status: "failure", summary: "x" } }),
		undefined,
	);
	assert.equal(f.asked.length, 2);
	const off = await toyRobot({}, { vlm: [no] });
	await off.emit("tool_result", { toolName: "act", content: obs.content });
	assert.equal(
		await off.emit("tool_call", { toolName: "finish", input: { status: "success", summary: "" } }),
		undefined,
	);
	assert.equal(off.asked.length, 0);
});

test("verifier: an unavailable or failed check accepts the finish; dual-arm robots verify by default", async () => {
	for (const reply of ["I think it is done.", new Error("503")]) {
		const f = await toyRobot({ "units-verify": "true" }, { vlm: [reply] });
		const obs = await f.run("act", { unit: "STOP" });
		await f.emit("tool_result", { toolName: "act", content: obs.content });
		assert.equal(
			await f.emit("tool_call", { toolName: "finish", input: { status: "success", summary: "" } }),
			undefined,
		);
		const [check] = f.entries.filter((e) => e.customType === VERIFY_ENTRY).map((e) => e.data);
		assert.equal(check.complete, true);
		assert.equal(check.refused, false);
		assert.equal(f.asked[0].model, "relay/planner", "the session's model by default");
	}
	const none = await toyRobot(
		{ "units-verify": "true" },
		{ vlm: ['{"complete": false, "reason": "x"}'], refuseRetreat: true },
	);
	assert.equal(
		await none.emit("tool_call", { toolName: "finish", input: { status: "success", summary: "" } }),
		undefined,
	);
	assert.equal(none.entries.find((e) => e.customType === VERIFY_ENTRY)?.data.skipped, "no camera images yet");
	assert.deepEqual(parseVerdict('noise "complete": false, more'), {
		complete: false,
		reason: 'noise "complete": false, more',
		available: true,
	});
	// auto: dual-arm robots verify, single-arm ones do not; the toy robot has one arm.
	const img = [{ type: "image", data: "", mimeType: "image/png" }];
	const claim = { toolName: "finish", input: { status: "success", summary: "" } };
	const one = await toyRobot({}, { vlm: ['{"complete": false, "reason": "x"}'] });
	await one.emit("tool_result", { toolName: "act", content: img });
	assert.equal(await one.emit("tool_call", claim), undefined);
	assert.equal(one.asked.length, 0);
	assert.equal(one.pi.getFlag("units-verify"), "auto");
	const two = await toyRobot({}, { arms: ["left", "right"], vlm: ['{"complete": false, "reason": "x"}'] });
	await two.emit("tool_result", { toolName: "act", content: img });
	assert.equal((await two.emit("tool_call", claim))?.block, true);
	const off = await toyRobot({ "units-verify": "false" }, { arms: ["left", "right"], vlm: ["{}"] });
	await off.emit("tool_result", { toolName: "act", content: img });
	assert.equal(await off.emit("tool_call", claim), undefined);
	assert.equal(off.asked.length, 0);
});

const hasFfmpeg = (() => {
	try {
		execFileSync("ffmpeg", ["-version"], { stdio: "ignore" });
		return true;
	} catch {
		return false;
	}
})();

test(
	"video_ref: uniform frames -> one analyst call -> the demo brief in the prompt",
	{ skip: !hasFfmpeg },
	async () => {
		const dir = realpathSync(mkdtempSync(join(tmpdir(), "video-ref-")));
		const mp4 = join(dir, "demo.mp4");
		execFileSync("ffmpeg", [
			"-v",
			"error",
			"-f",
			"lavfi",
			"-i",
			"testsrc=size=640x360:rate=20",
			"-frames:v",
			"30",
			mp4,
		]);
		const brief = JSON.stringify({
			task: "put the soup in the basket",
			operations: [
				{ action: "pick up", object: "red soup can", grasp: "body", destination: "-" },
				{ action: "place", object: "red soup can", grasp: "-", destination: "inside the basket" },
			],
		});
		const f = await toyRobot({ "units-video-ref": mp4 }, { vlm: [brief] });
		const prompt = (await f.emit("before_agent_start")).systemPrompt as string;
		assert.match(prompt, /REFERENCE DEMO -- a demonstration video of this task was analyzed\. REPLICATE it\./);
		assert.match(prompt, /1\. pick up red soup can -- grasp body\n2\. place red soup can -> inside the basket\n/);
		assert.equal(f.asked.length, 1);
		const images = f.asked[0].content.filter((c: any) => c.type === "image");
		assert.equal(images.length, 8);
		assert.match(f.asked[0].content[0].text, /ROLE: DemoVideoAnalyst\nYou see 8 frames[\s\S]*performed by ONE arm/);
		assert.doesNotMatch(f.asked[0].content[0].text, /"arm"/);
		const width = Buffer.from(images[0].data, "base64").readUInt32BE(16);
		assert.equal(width, 512, "frames are downscaled to 512 px");
		const [entry] = f.entries.filter((e) => e.customType === VIDEO_REF_ENTRY).map((e) => e.data);
		assert.deepEqual(entry.sampled_indices, [0, 4, 8, 12, 17, 21, 25, 29]);
		// Once per video: the next prompt reuses the brief.
		await f.emit("before_agent_start");
		assert.equal(f.asked.length, 1);
		// Two unusable replies fail closed: act refuses and the error is shown.
		const bad = await toyRobot({ "units-video-ref": mp4 }, { vlm: ["no json", '{"task": ""}'] });
		await bad.emit("before_agent_start");
		assert.equal(bad.asked.length, 2);
		assert.match(bad.notes.join("\n"), /video_ref: could not extract a demo brief/);
		assert.match(head(await bad.run("act", { unit: "MV_LEFT" })), /video_ref failed/);
		assert.equal(bad.moves.length, 0);
		const missing = await toyRobot({ "units-video-ref": join(dir, "nope.mp4") }, { vlm: [brief] });
		await missing.emit("before_agent_start");
		assert.equal(missing.asked.length, 0);
		assert.match(missing.notes.join("\n"), /could not extract/);
	},
);

test("video_ref: dual-arm briefs carry the arm and the cross-arm sequencing rule", () => {
	const b = validateBrief(
		{
			task: "t",
			operations: [
				{ arm: "LEFT", action: "hold", object: "box", grasp: "edge", destination: "-" },
				{ arm: "middle", action: "open", object: "lid" },
				"junk",
			],
		},
		["left", "right"],
	);
	assert.deepEqual(
		b.operations.map((o) => o.arm),
		["left", "both"],
	);
	const text = renderBrief(b, ["left", "right"]);
	assert.match(text, /1\. LEFT hold box -- grasp edge\n2\. BOTH open lid\n/);
	assert.match(text, /give this arm a WAIT stage first/);
	assert.throws(() => validateBrief({ task: "t", operations: [] }, []), /missing task\/operations/);
});

test("act's schema follows the plugins the flags enable, with string enums; disabled params are refused", async () => {
	// pi parses flags after load: the schema built from the defaults is rebuilt at session start.
	const f = fakePi({ units: "true" });
	franka(f.pi);
	const props = () => f.tools.get("act").parameters.properties;
	assert.ok(props().plan && props().target_in_wrist, "the default plugins offer plan and target_in_wrist");
	assert.equal(props().unit.type, "string");
	assert.ok(props().unit.enum.includes("ROTATE_CW"));
	assert.equal(props().unit.anyOf, undefined, "no anyOf literal unions");
	assert.deepEqual(props().plan.items.enum, ["MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN"]);
	f.setFlag("units-plugins", "proprioception");
	await f.emit("session_start");
	assert.equal(props().plan, undefined);
	assert.equal(props().target_in_wrist, undefined);
	const d = fakePi({ units: "true" });
	dualFranka(d.pi);
	assert.deepEqual(d.tools.get("act").parameters.properties.arm.enum, ["left", "right"]);
	const t = await toyRobot({ "units-plugins": "proprioception" });
	await assert.rejects(t.run("act", { unit: "MV_FWD", plan: ["MV_FWD"] }), /`plan` needs the action_chunk plugin/);
	await assert.rejects(
		t.run("act", { unit: "MV_FWD", target_in_wrist: true }),
		/`target_in_wrist` needs the variable_step, action_chunk or rotation plugin/,
	);
	assert.equal(t.moves.length, 0);
});

test("the units state survives a resume or fork: turned wrist, closed gripper, plan, history", async () => {
	const f = await toyRobot({ "units-plugins": "rotation,plan,mem_text,proprioception" }, { yaw: Math.PI / 4 });
	await f.run("plan", { stages: [{ motion: "LIFT", target: "cube", completion: "cube above the table" }] });
	await f.run("act", { unit: "ROTATE_CW", n: 2 });
	await f.run("act", { unit: "GRASP" });
	const states = f.entries.filter((e) => e.customType === STATE_ENTRY);
	assert.deepEqual(states.at(-1)?.data.yaw, { "": Math.PI / 2 });
	assert.deepEqual(states.at(-1)?.data.closed, { "": true });
	const count = states.length;
	await f.run("act", { unit: "STOP" });
	assert.equal(f.entries.filter((e) => e.customType === STATE_ENTRY).length, count, "unchanged state adds no entry");
	// Resume: a new process on the same branch.
	const g = await toyRobot(
		{ "units-plugins": "rotation,plan,mem_text,proprioception" },
		{ yaw: Math.PI / 4, branch: [...f.branch] },
	);
	const r = head(await g.run("act", { unit: "MV_FWD", target_in_wrist: true }));
	assert.match(r, /Gripper turned 90 deg from its start heading/, "the model is told the wrist is still turned");
	assert.deepEqual(
		g.moves[0].delta.map((v) => Number(v.toFixed(6))),
		[0, 0.02, 0],
		"wrist-judged moves still compensate the turn",
	);
	assert.match(r, /STAGE 1\/1 \[LIFT\]/);
	assert.match(r, /commanded CLOSE/);
	assert.match(r, /newest first: MV_FWD, GRASP, ROTATE_CW, ROTATE_CW/);
	assert.match(
		head(await g.run("act", { unit: "MV_UP" })),
		/MV_UP\(realign\)/,
		"holding and turned: MV_UP turns back first",
	);
	// A scene reset is recorded too, so a later resume starts straight.
	const h = await toyRobot(
		{ "units-plugins": "rotation,plan,mem_text", operator: true },
		{ yaw: Math.PI / 4, branch: [...f.branch], reset: async () => ({ ok: true }) },
	);
	await h.run("request_scene_reset", { reason: "retry" });
	const k = await toyRobot({ "units-plugins": "rotation,plan,mem_text" }, { yaw: Math.PI / 4, branch: [...h.branch] });
	const fresh = head(await k.run("act", { unit: "MV_FWD", target_in_wrist: true }));
	assert.doesNotMatch(fresh, /Gripper turned|STAGE/);
	assert.deepEqual(k.moves[0].delta, [0.02, 0, 0]);
});

test("a chaining robot's continuous moves skip the stall check until the chain's last move", async () => {
	// A blocked arm: the robot that chains returns before settling, so only the last move is judged.
	const f = await toyRobot({}, { blocked: true, chains: true });
	const r = await f.run("act", { unit: "MV_DOWN", n: 3 });
	assert.deepEqual(
		f.moves.map((m) => m.continuous),
		[true, true, undefined],
	);
	assert.match(head(r), /MV_DOWN x3\n/);
	assert.match(head(r), /Last MV_DOWN lowered 0\.0 of 2\.0 cm -> already in contact/);
	// A robot that settles every move is judged per move (Show-Harness keeps chaining off by default).
	const g = await toyRobot({}, { blocked: true, chains: false });
	await g.run("act", { unit: "MV_DOWN", n: 3 });
	assert.equal(g.moves.length, 1);
});

test("view_select: act's `view` reaches the robot as Move.view, only when the robot selects views", async () => {
	const f = await toyRobot({}, { viewSelect: true });
	assert.deepEqual(f.tools.get("act").parameters.properties.view.enum, ["WRIST", "FRONT"]);
	assert.equal((f.emitted.get(UNITS_EVENT) as UnitsHandle).viewSelect, true, "robots detect the hook on the handle");
	await f.run("act", { unit: "MV_FWD", view: "WRIST" });
	await f.run("act", { unit: "MV_LEFT" });
	assert.deepEqual(
		f.moves.map((m) => m.view),
		["WRIST", undefined],
	);
	await assert.rejects(f.run("act", { unit: "MV_FWD", view: "SIDE" }), /`view` needs the robot's view select/);
	const off = await toyRobot({}, { viewSelect: false });
	assert.equal(off.tools.get("act").parameters.properties.view, undefined);
	assert.equal((off.emitted.get(UNITS_EVENT) as UnitsHandle).viewSelect, false);
	await assert.rejects(off.run("act", { unit: "MV_FWD", view: "WRIST" }), /`view` needs the robot's view select/);
	assert.equal(off.moves.length, 0);
});

test("verifier retreat: before the VLM call; a closed gripper stays; a refused retreat still verifies, noted", async () => {
	const no = '{"complete": false, "reason": "x"}';
	// Holding: no lift (a held object would leave its destination); the check runs on the latest images.
	const held = await toyRobot({ "units-verify": "true" }, { vlm: [no] });
	await held.run("act", { unit: "GRASP" });
	const obs = await held.run("act", { unit: "STOP" });
	await held.emit("tool_result", { toolName: "act", content: obs.content });
	await held.emit("tool_call", { toolName: "finish", input: { status: "success", summary: "" } });
	assert.equal(held.moves.length, 2);
	const heldEntry = held.entries.find((e) => e.customType === VERIFY_ENTRY)?.data;
	assert.match(heldEntry.retreat[0].skipped, /gripper closed/);
	assert.equal(held.asked.length, 1);
	// Refused (the episode already ended): recorded, and the check uses the images at hand.
	const ended = await toyRobot({ "units-verify": "true" }, { vlm: [no], refuseRetreat: true });
	const last = await ended.run("act", { unit: "MV_LEFT" });
	await ended.emit("tool_result", { toolName: "act", content: last.content });
	assert.equal(
		(await ended.emit("tool_call", { toolName: "finish", input: { status: "success", summary: "" } }))?.block,
		true,
	);
	const endedEntry = ended.entries.find((e) => e.customType === VERIFY_ENTRY)?.data;
	assert.equal(endedEntry.retreat[0].refused, "Episode already ended.");
	assert.deepEqual(
		ended.asked[0].content.slice(1).map((c: any) => c.data),
		["m1"],
	);
	// A per-call travel limit shortens the lift; variable_step lifts in coarse steps.
	const short = await toyRobot(
		{ "units-verify": "true", "units-plugins": "variable_step,proprioception" },
		{ vlm: [no], maxMove: 0.08 },
	);
	await short.emit("tool_call", { toolName: "finish", input: { status: "success", summary: "" } });
	assert.deepEqual(
		short.moves.map((m) => m.delta[2]),
		[0.04, 0.04],
	);
	assert.match(short.entries.find((e) => e.customType === VERIFY_ENTRY)?.data.retreat[0].ran, /MV_UP x2 of 3/);
	// Two arms: each open gripper lifts, one arm after the other.
	const two = await toyRobot({ "units-verify": "true" }, { arms: ["left", "right"], vlm: [no] });
	await two.emit("tool_call", { toolName: "finish", input: { status: "success", summary: "" } });
	assert.deepEqual([...new Set(two.moves.map((m) => m.arm))], ["left", "right"]);
	assert.equal(two.moves.length, 10);
});
