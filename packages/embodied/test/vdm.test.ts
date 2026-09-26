import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdtempSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { API_GATE_EVENT } from "../src/api-gate.ts";
import { defineRobot, RESULT_ENTRY, type RobotSpec } from "../src/robot.ts";
import type { UnitsSpec } from "../src/units/index.ts";
import { HEADERS, VDM_ENTRY } from "../src/vdm.ts";

type Handler = (event: any, ctx: any) => unknown;
/** A faux VLM reply: its text and cost, an error, or "hang" (answers only when the call's signal aborts). */
type Reply = { text: string; usd?: number } | Error | "hang";

/** A stub pi that runs handlers in registration order, with a real event bus; `vlm` answers side model calls in order. */
function fakePi(flagValues: Record<string, unknown>, vlm: Reply[] = []) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	const asked: { model: string; system?: string; content: any[] }[] = [];
	const bus = new EventEmitter();
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		setActiveTools: () => {},
		getActiveTools: () => [],
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		getThinkingLevel: () => "low",
		events: {
			emit: (channel: string, data: unknown) => bus.emit(channel, data),
			on: (channel: string, fn: (data: unknown) => void) => {
				bus.on(channel, fn);
				return () => bus.off(channel, fn);
			},
		},
	} as unknown as ExtensionAPI;
	/** Runs inside every side model call (to observe what the call holds). */
	let during: (() => void) | undefined;
	const ctx = {
		hasUI: true,
		ui: { notify: () => {}, setWidget: () => {} },
		shutdown: () => {},
		signal: undefined as AbortSignal | undefined,
		sessionManager: { getBranch: () => [], getSessionDir: () => "/tmp" },
		model: { provider: "relay", id: "planner" },
		modelRegistry: {
			find: (provider: string, id: string) => ({ provider, id }),
			streamSimple: (
				model: { provider: string; id: string },
				context: { systemPrompt?: string; messages: { content: any[] }[] },
				options: { signal?: AbortSignal },
			) => ({
				result: async () => {
					asked.push({
						model: `${model.provider}/${model.id}`,
						system: context.systemPrompt,
						content: context.messages[0].content,
					});
					during?.();
					const reply = vlm.shift() ?? { text: "" };
					if (reply === "hang") {
						if (!options.signal?.aborted)
							await new Promise<void>((resolve) => options.signal?.addEventListener("abort", () => resolve()));
						return { stopReason: "aborted", content: [] };
					}
					if (reply instanceof Error) return { stopReason: "error", errorMessage: reply.message, content: [] };
					return {
						stopReason: "stop",
						content: [{ type: "text", text: reply.text }],
						usage: { cost: { total: reply.usd ?? 0 } },
					};
				},
			}),
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
	return {
		pi,
		emit,
		tools,
		entries,
		asked,
		handlers,
		ctx,
		bus,
		setDuring: (fn: () => void) => {
			during = fn;
		},
	};
}

const finish: RobotSpec["finish"] = {
	description: "finish",
	parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
	result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
};

/** A LIBERO-shaped robot: observations carry the agentview then the wrist view; `segment` returns one overlay. */
async function toy(flags: Record<string, unknown>, vlm: Reply[] = [], extra: Partial<RobotSpec> = {}) {
	const f = fakePi(flags, vlm);
	const robot = defineRobot(f.pi, {
		name: "toy",
		task: [],
		keepImages: 4,
		start: async () => ["move", "segment", "finish"],
		result: () => ({}),
		status: () => ({ language: "put the bowl on the plate" }),
		finish,
		vdm: { views: 2, wrist: 1 },
		...extra,
	});
	for (const name of ["move", "segment", "look"])
		robot.tool(name, name, Type.Object({}), async () => ({ content: [], details: {} }));
	await f.emit("session_start");
	await f.emit("agent_start");
	return f;
}

const image = (data: string) => ({ type: "image", data, mimeType: "image/png" });
/** A robot tool result as pi's tool_result event carries it. */
const observed = (toolName: string, ...images: string[]) => ({
	toolName,
	toolCallId: "id",
	input: {},
	isError: false,
	details: {},
	content: [{ type: "text", text: "{}" }, ...images.map(image)],
});
const texts = (content: any[]) => content.filter((c) => c.type === "text").map((c) => c.text);
const images = (content: any[]) => content.filter((c) => c.type === "image").map((c) => c.data);
const result = (f: Awaited<ReturnType<typeof toy>>) =>
	f.entries.filter((e) => e.type === RESULT_ENTRY).map((e) => e.data)[0];

test("--vdm off: no hook, results are untouched, no model is asked, nothing is written or reported", async () => {
	const f = await toy({});
	assert.equal(f.handlers.get("tool_result"), undefined, "no tool_result hook is registered");
	assert.equal(await f.emit("tool_result", observed("move", "a0", "w0")), undefined);
	assert.equal(await f.emit("tool_result", observed("move", "a1", "w1")), undefined);
	assert.equal(f.asked.length, 0);
	assert.deepEqual(
		f.entries.filter((e) => e.type === VDM_ENTRY),
		[],
	);
	await f.tools.get("finish").execute("id", { status: "success", summary: "" });
	await f.emit("agent_end");
	assert.equal("vdm" in result(f), false, "no vdm fields in the robot result");
	assert.equal(result(f).vdm_calls, undefined);
});

test("a robot without a vdm spec registers no vdm flags", async () => {
	const f = await toy({}, [], { vdm: undefined });
	assert.equal(f.pi.getFlag("vdm"), undefined);
	assert.equal(await f.emit("tool_result", observed("move", "a0", "w0")), undefined);
});

test("--vdm describes the first observation, then appends each diff and writes a vdm entry per call", async () => {
	const f = await toy({ vdm: true }, [
		{ text: "A bowl and a plate on the table.", usd: 0.01 },
		{ text: "The gripper moved above the bowl. Not complete.", usd: 0.02 },
	]);
	const first = await f.emit("tool_result", observed("move", "a0", "w0"));
	assert.deepEqual(images(first.content), ["a0", "w0"], "the camera images stay");
	assert.equal(texts(first.content).at(-1), `${HEADERS.initial}\nA bowl and a plate on the table.`);
	// The initial description: CaP-X's system prompt, the task, the main view only (no --vdm-wrist).
	assert.equal(f.asked[0].model, "relay/planner");
	assert.match(f.asked[0].system ?? "", /describes the initial state of the environment/);
	assert.equal(f.asked[0].content[0].text, "put the bowl on the plate");
	assert.deepEqual(images(f.asked[0].content), ["a0"]);

	// A segmentation overlay (one image) and a non-robot tool are not observations.
	assert.equal(await f.emit("tool_result", observed("segment", "overlay")), undefined);
	assert.equal(await f.emit("tool_result", observed("read", "x", "y")), undefined);

	const second = await f.emit("tool_result", observed("move", "a1", "w1"));
	assert.equal(texts(second.content).at(-1), `${HEADERS.diff}\nThe gripper moved above the bowl. Not complete.`);
	assert.match(f.asked[1].system ?? "", /difference between the current state/);
	assert.deepEqual(images(f.asked[1].content), ["a0", "a1"], "previous then current agentview");
	assert.deepEqual(texts(f.asked[1].content).slice(2), [
		"Previous state (main camera):",
		"Current state (main camera):",
	]);
	assert.equal(f.asked.length, 2);

	const logged = f.entries.filter((e) => e.type === VDM_ENTRY).map((e) => e.data);
	assert.deepEqual(
		logged.map((e) => [e.kind, e.tool, e.model, e.text, e.cost_usd, e.wrist]),
		[
			["initial", "move", "relay/planner", "A bowl and a plate on the table.", 0.01, false],
			["diff", "move", "relay/planner", "The gripper moved above the bowl. Not complete.", 0.02, false],
		],
	);
	await f.tools.get("finish").execute("id", { status: "success", summary: "" });
	await f.emit("agent_end");
	const r = result(f);
	assert.equal(r.vdm, true);
	assert.equal(r.vdm_calls, 2);
	assert.equal(r.vdm_errors, 0);
	assert.equal(r.vdm_cost_usd, 0.03);
	assert.equal(r.cost_usd, 0.03, "VDM calls count toward the episode's cost");
});

test("--vdm-wrist adds the wrist pair and --vdm-model picks the model", async () => {
	const f = await toy({ vdm: true, "vdm-wrist": true, "vdm-model": "selfhost/muse" }, [
		{ text: "scene" },
		{ text: "diff" },
	]);
	await f.emit("tool_result", observed("move", "a0", "w0"));
	await f.emit("tool_result", observed("move", "a1", "w1"));
	assert.deepEqual(
		f.asked.map((a) => a.model),
		["selfhost/muse", "selfhost/muse"],
	);
	assert.deepEqual(images(f.asked[0].content), ["a0", "w0"]);
	assert.deepEqual(texts(f.asked[0].content).slice(2), ["Main camera view:", "Wrist camera view:"]);
	assert.deepEqual(images(f.asked[1].content), ["a0", "a1", "w0", "w1"]);
	assert.equal(f.entries.find((e) => e.type === VDM_ENTRY)?.data.wrist, true);
});

test("a failed VDM call is noted in the result and the episode goes on", async () => {
	const f = await toy({ vdm: true }, [new Error("402 no credits"), { text: "the bowl moved" }]);
	const first = await f.emit("tool_result", observed("move", "a0", "w0"));
	assert.equal(texts(first.content).at(-1), "[visual differencing unavailable: 402 no credits]");
	assert.deepEqual(images(first.content), ["a0", "w0"]);
	assert.equal(await f.emit("tool_call", { toolName: "move" }), undefined, "the next robot call runs");
	// The failed call's frame is still the previous one: the next result is a diff against it.
	const second = await f.emit("tool_result", observed("move", "a1", "w1"));
	assert.equal(texts(second.content).at(-1), `${HEADERS.diff}\nthe bowl moved`);
	assert.deepEqual(images(f.asked[1].content), ["a0", "a1"]);
	const logged = f.entries.filter((e) => e.type === VDM_ENTRY).map((e) => e.data);
	assert.equal(logged[0].error, "402 no credits");
	assert.equal(logged[0].kind, "initial");
	await f.tools.get("finish").execute("id", { status: "success", summary: "" });
	await f.emit("agent_end");
	assert.equal(result(f).vdm_errors, 1);
	assert.equal(result(f).env_error, false);
	assert.equal(result(f).planner_error, null);
});

test("VDM spend exhausts --max-cost like the planner's replies", async () => {
	const f = await toy({ vdm: true, "max-cost": "0.05" }, [
		{ text: "scene", usd: 0.03 },
		{ text: "diff", usd: 0.03 },
	]);
	await f.emit("tool_result", observed("move", "a0", "w0"));
	assert.equal(await f.emit("tool_call", { toolName: "move" }), undefined);
	await f.emit("tool_result", observed("move", "a1", "w1"));
	assert.match((await f.emit("tool_call", { toolName: "move" }))?.reason, /Planner cost budget exhausted/);
	await f.emit("agent_end");
	assert.equal(result(f).planner_budget_exhausted, "cost");
	assert.equal(result(f).cost_usd, 0.06);
});

test("the units verifier's VLM call counts toward the episode's cost", async () => {
	const units: UnitsSpec = {
		vectors: {
			MV_FWD: [1, 0, 0],
			MV_BACK: [-1, 0, 0],
			MV_LEFT: [0, -1, 0],
			MV_RIGHT: [0, 1, 0],
			MV_UP: [0, 0, 1],
			MV_DOWN: [0, 0, -1],
		},
		stepM: 0.02,
		apply: async () => ({ content: [{ type: "text", text: "obs" }, image("m")] as any, details: {} }),
		state: async () => ({ eef_xyz: [0.5, 0, 0.3], gripper_width: 0.08, table_z: 0 }),
		instruction: () => "put the cube in the bowl",
	};
	const f = await toy(
		{ units: true, "units-verify": "true" },
		[{ text: '{"complete":true,"reason":"ok"}', usd: 0.04 }],
		{
			units,
			vdm: undefined,
		},
	);
	await f.emit("tool_result", observed("act", "a0"));
	assert.equal(
		await f.emit("tool_call", { toolName: "finish", input: { status: "success", summary: "" } }),
		undefined,
	);
	assert.equal(f.asked.length, 1);
	await f.tools.get("finish").execute("id", { status: "success", summary: "" });
	await f.emit("agent_end");
	assert.equal(result(f).finish_verified, true);
	assert.equal(result(f).cost_usd, 0.04);
});

test("a two-armed robot's --vdm-wrist gives the model both wrist pairs, numbered", async () => {
	const f = await toy({ vdm: true, "vdm-wrist": true }, [{ text: "scene" }, { text: "diff" }], {
		vdm: { views: 3, wrist: [1, 2] },
	});
	await f.emit("tool_result", observed("move", "h0", "l0", "r0"));
	await f.emit("tool_result", observed("move", "h1", "l1", "r1"));
	assert.deepEqual(images(f.asked[0].content), ["h0", "l0", "r0"]);
	assert.deepEqual(texts(f.asked[0].content).slice(2), [
		"Main camera view:",
		"Wrist camera 1 view:",
		"Wrist camera 2 view:",
	]);
	assert.deepEqual(images(f.asked[1].content), ["h0", "h1", "l0", "l1", "r0", "r1"]);
	assert.deepEqual(texts(f.asked[1].content).slice(4), [
		"Previous state (wrist camera 1):",
		"Current state (wrist camera 1):",
		"Previous state (wrist camera 2):",
		"Current state (wrist camera 2):",
	]);
	await f.tools.get("finish").execute("id", { status: "success", summary: "" });
	await f.emit("agent_end");
	assert.equal(result(f).vdm_wrist, true);
});

test("a robot whose cameras decide the views passes a function; undefined skips the result", async () => {
	let views: { views: number; wrist?: number } | undefined;
	const f = await toy({ vdm: true }, [{ text: "scene" }], { vdm: () => views });
	assert.equal(await f.emit("tool_result", observed("move", "a0")), undefined, "no cameras known yet");
	views = { views: 1 };
	const first = await f.emit("tool_result", observed("move", "a0"));
	assert.equal(texts(first.content).at(-1), `${HEADERS.initial}\nscene`);
	await f.tools.get("finish").execute("id", { status: "success", summary: "" });
	await f.emit("agent_end");
	assert.equal(result(f).vdm_wrist, false, "--vdm-wrist is off");
});

test("a look-only tool records the frame without a call; a scene reset starts a new initial scene", async () => {
	const f = await toy(
		{ vdm: true },
		[{ text: "scene" }, { text: "moved" }, { text: "scene again" }, { text: "moved again" }, { text: "third scene" }],
		{ vdm: { views: 2, wrist: 1, observe: ["look"] } },
	);
	// The first observation, from a look, is the initial scene.
	const first = await f.emit("tool_result", observed("look", "a0", "w0"));
	assert.equal(texts(first.content).at(-1), `${HEADERS.initial}\nscene`);
	// Looking again: no call, the frame is the new previous one.
	assert.equal(await f.emit("tool_result", observed("look", "a1", "w1")), undefined);
	assert.equal(f.asked.length, 1);
	const moved = await f.emit("tool_result", observed("move", "a2", "w2"));
	assert.equal(texts(moved.content).at(-1), `${HEADERS.diff}\nmoved`);
	assert.deepEqual(images(f.asked[1].content), ["a1", "a2"], "diffed against the latest look, not the first");
	// A reset whose result carries the views is described as the initial scene, not as a change.
	const reset = await f.emit("tool_result", observed("reset", "a3", "w3"));
	assert.equal(texts(reset.content).at(-1), `${HEADERS.initial}\nscene again`);
	assert.deepEqual(images(f.asked[2].content), ["a3"]);
	const after = await f.emit("tool_result", observed("move", "a4", "w4"));
	assert.equal(texts(after.content).at(-1), `${HEADERS.diff}\nmoved again`);
	assert.deepEqual(images(f.asked[3].content), ["a3", "a4"]);
	// A reset without images (the operator's): the next observation is initial.
	assert.equal(
		await f.emit("tool_result", { ...observed("request_scene_reset"), content: [{ type: "text", text: "{}" }] }),
		undefined,
	);
	const third = await f.emit("tool_result", observed("look", "a5", "w5"));
	assert.equal(texts(third.content).at(-1), `${HEADERS.initial}\nthird scene`);
	assert.deepEqual(
		f.entries.filter((e) => e.type === VDM_ENTRY).map((e) => [e.data.kind, e.data.tool]),
		[
			["initial", "look"],
			["diff", "move"],
			["initial", "reset"],
			["diff", "move"],
			["initial", "look"],
		],
	);
});

test("an aborted call is recorded, not counted as an error; a slow call times out on its own clock", async () => {
	const f = await toy({ vdm: true, "vdm-timeout": "0.05" }, ["hang", "hang", { text: "back" }]);
	f.ctx.signal = AbortSignal.abort();
	assert.equal(await f.emit("tool_result", observed("move", "a0", "w0")), undefined, "no note for an aborted call");
	f.ctx.signal = undefined;
	const slow = await f.emit("tool_result", observed("move", "a1", "w1"));
	assert.equal(texts(slow.content).at(-1), "[visual differencing unavailable: timed out after 0.05 s]");
	const fine = await f.emit("tool_result", observed("move", "a2", "w2"));
	assert.equal(texts(fine.content).at(-1), `${HEADERS.diff}\nback`);
	const logged = f.entries.filter((e) => e.type === VDM_ENTRY).map((e) => e.data);
	assert.deepEqual(
		logged.map((e) => [e.kind, e.aborted, e.error]),
		[
			["initial", true, undefined],
			["diff", undefined, "timed out after 0.05 s"],
			["diff", undefined, undefined],
		],
	);
	await f.tools.get("finish").execute("id", { status: "success", summary: "" });
	await f.emit("agent_end");
	assert.equal(result(f).vdm_calls, 3);
	assert.equal(result(f).vdm_errors, 1, "the abort is not an error");
});

test("with eval-parallel's api gate loaded, a VDM call holds one of its slots", async () => {
	const f = await toy({ vdm: true }, [{ text: "scene" }]);
	const dir = mkdtempSync(join(tmpdir(), "vdm-gate-"));
	f.bus.emit(API_GATE_EVENT, { dir, n: 1 });
	let held: string[] = [];
	f.setDuring(() => {
		held = readdirSync(dir);
	});
	await f.emit("tool_result", observed("move", "a0", "w0"));
	assert.deepEqual(held, ["slot-0"], "the slot is held during the call");
	assert.deepEqual(readdirSync(dir), [], "and given back after it");
});
