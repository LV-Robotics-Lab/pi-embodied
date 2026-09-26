import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { test } from "node:test";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import dualFranka from "../src/dual_franka/index.ts";
import franka from "../src/franka/index.ts";
import libero, { type Claim, runClaim } from "../src/libero/index.ts";
import {
	attachedPrompt,
	CHECK_ATTACHED_ENTRY,
	DETECTIONS_EXPIRED_ENTRY,
	GRASP_TOOLS,
	graspActive,
	graspArgs,
	graspTools,
	isStale,
	parseAttached,
	registerGraspFlags,
} from "../src/primitives/grasp.ts";
import { VLM_COST_EVENT } from "../src/units/vlm.ts";

type Json = Record<string, any>;

/** A stub pi: flags, tools, entries, a real event bus, and a faux VLM answering `replies` in order. */
function fakePi(flagValues: Record<string, unknown> = {}, replies: { text: string; usd?: number }[] = []) {
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const entries: { type: string; data: Json }[] = [];
	const asked: { model: string; system?: string; content: any[] }[] = [];
	const bus = new EventEmitter();
	const costs: number[] = [];
	bus.on(VLM_COST_EVENT, (usd) => costs.push(Number(usd)));
	const api: Record<string, unknown> = {
		on: () => {},
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		setActiveTools: () => {},
		getActiveTools: () => [],
		appendEntry: (type: string, data: Json) => entries.push({ type, data }),
		getThinkingLevel: () => "low",
		events: {
			emit: (channel: string, data: unknown) => bus.emit(channel, data),
			on: (channel: string, fn: (data: unknown) => void) => {
				bus.on(channel, fn);
				return () => bus.off(channel, fn);
			},
		},
	};
	const pi = new Proxy(api, { get: (t, k: string) => t[k] ?? (() => {}) }) as unknown as ExtensionAPI;
	const ctx = {
		hasUI: true,
		model: { provider: "relay", id: "planner" },
		modelRegistry: {
			find: (provider: string, id: string) => ({ provider, id }),
			streamSimple: (
				model: { provider: string; id: string },
				context: { systemPrompt?: string; messages: { content: any[] }[] },
			) => ({
				result: async () => {
					asked.push({
						model: `${model.provider}/${model.id}`,
						system: context.systemPrompt,
						content: context.messages[0].content,
					});
					const reply = replies.shift() ?? { text: "" };
					return {
						stopReason: "stop",
						content: [{ type: "text", text: reply.text }],
						usage: { cost: { total: reply.usd ?? 0 } },
					};
				},
			}),
		},
	};
	return { pi, flags, tools, entries, asked, costs, ctx };
}

/** A fake env: answers `env.*` from `answers`, records the calls. */
function fakeEnv(answers: Record<string, Json | Error>) {
	const calls: { method: string; kwargs: Json }[] = [];
	const call = async (method: string, kwargs: Json) => {
		calls.push({ method, kwargs });
		const a = answers[method];
		if (a === undefined) throw new Error(`unexpected ${method}`);
		if (a instanceof Error) throw a;
		return a;
	};
	return { call, calls };
}

const PNG = Buffer.from(
	"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
	"base64",
);

test("the grasp flags are off by default: no env args and no active tool", () => {
	const f = fakePi();
	registerGraspFlags(f.pi);
	assert.deepEqual(Object.keys(f.flags).sort(), ["anygrasp", "anyplace", "attach-vlm-model", "graspgenx", "graspnet"]);
	assert.deepEqual(graspArgs(f.pi), []);
	assert.deepEqual(graspActive(f.pi), []);
	const g = fakePi({ graspnet: "http://127.0.0.1:8120", anyplace: "http://127.0.0.1:8123" });
	registerGraspFlags(g.pi);
	assert.deepEqual(graspArgs(g.pi), ["--graspnet", "http://127.0.0.1:8120", "--anyplace", "http://127.0.0.1:8123"]);
	assert.deepEqual(graspActive(g.pi), [...GRASP_TOOLS]);
});

test("every robot registers the three grasp tools at load, with the arm parameter only on the dual arm", () => {
	for (const [name, load, arm] of [
		["libero", libero, false],
		["franka", franka, false],
		["dual_franka", dualFranka, true],
	] as const) {
		const f = fakePi();
		load(f.pi);
		for (const tool of GRASP_TOOLS) assert.ok(f.tools.has(tool), `${name} registers ${tool}`);
		const props = f.tools.get("plan_grasp").parameters.properties;
		assert.equal("arm" in props, arm, `${name} plan_grasp arm parameter`);
		assert.ok("object" in props && "mask_id" in props && "next_after" in props);
		assert.ok("grasp_id" in f.tools.get("plan_place").parameters.properties);
		assert.ok("object" in f.tools.get("check_attached").parameters.properties);
		assert.ok("graspnet" in f.flags && "anyplace" in f.flags, `${name} registers the flags`);
	}
	// LIBERO executes a planned id in one tool (one resolution); its free motions take xyz only.
	const l = fakePi();
	libero(l.pi);
	for (const tool of ["move_to", "move_pose"]) {
		const props = l.tools.get(tool).parameters.properties;
		assert.ok(!("grasp_id" in props) && !("standoff" in props), `${tool} takes no grasp_id`);
		assert.ok(l.tools.get(tool).parameters.required?.includes("xyz"), `${tool}: xyz is required`);
	}
	assert.deepEqual(l.tools.get("execute_grasp").parameters.required, ["grasp_id"]);
	assert.deepEqual(l.tools.get("execute_place").parameters.required, ["place_id"]);
});

test("a claimed grasp runs leg by leg with the claim's orientation and stops at a stalled leg", async () => {
	const claim: Claim = {
		id: "g2",
		waypoints: { pre_grasp: [0, 0, 0.3], grasp: [0, 0, 0.2], lift: [0, 0, 0.3] },
		steps: [
			{ to: "pre_grasp", gripper: -1 },
			{ to: "grasp", gripper: -1 },
			{ gripper: 1 },
			{ to: "lift", gripper: 1 },
		],
		eef_yaw: 1.5,
		eef_pitch: 0.2,
	};
	const log: string[] = [];
	let width = 0.08;
	const io = (stallAt?: string) => ({
		servo: async (target: number[], pitch: number, yaw: number, g: number) => {
			log.push(`servo ${target.join(",")} p${pitch} y${yaw} g${g}`);
			const stalled = claim.waypoints[stallAt ?? ""] === target;
			return { steps: 5, final_dist_m: stalled ? 0.08 : 0.004 };
		},
		actuate: async (g: number) => {
			log.push(`grip ${g}`);
			width = g > 0 ? 0.02 : 0.08;
			return 4;
		},
		width: () => width,
		ended: () => false,
	});
	const ok = await runClaim(claim, io());
	assert.deepEqual(log, [
		"servo 0,0,0.3 p0.2 y1.5 g-1",
		"servo 0,0,0.2 p0.2 y1.5 g-1",
		"grip 1",
		"servo 0,0,0.3 p0.2 y1.5 g1",
	]);
	assert.equal(ok.steps_used, 19);
	assert.deepEqual(ok.legs[2], { gripper: 1, gripper_width: 0.02 });
	assert.equal(ok.error, undefined);
	log.length = 0;
	const stalled = await runClaim(claim, io("grasp"));
	assert.equal(stalled.stalled, "grasp");
	assert.match(stalled.error ?? "", /stalled 0.08 m short of grasp/);
	assert.equal(log.length, 2, "nothing after the stalled leg");
	const ended = await runClaim(claim, { ...io(), ended: () => true });
	assert.deepEqual(ended, { legs: [], steps_used: 0 });
});

test("plan_grasp relays the env call, records expired ids and a stale refusal", async () => {
	const f = fakePi();
	const env = fakeEnv({
		"env.plan_grasp": { observation: 3, candidates: [{ id: "g4" }], active: "g4", expired_ids: ["d1", "g2"] },
		"env.next_grasp": { active: "g5", rejected: { id: "g4", reason: "collision" }, expired_ids: [] },
	});
	const [planGrasp] = graspTools(f.pi, { call: env.call, cameras: ["agentview", "wrist"], task: () => "task" });
	assert.equal(planGrasp.name, "plan_grasp");
	const out = await planGrasp.run({ object: " black bowl ", backend: "graspgenx" } as any, undefined, f.ctx as any);
	assert.equal(out.active, "g4");
	assert.deepEqual(env.calls[0], { method: "env.plan_grasp", kwargs: { object: "black bowl", backend: "graspgenx" } });
	assert.deepEqual(f.entries, [
		{ type: DETECTIONS_EXPIRED_ENTRY, data: { tool: "plan_grasp", ids: ["d1", "g2"], observation: 3 } as Json },
	]);
	// The greedy policy goes through env.next_grasp, no re-planning.
	const next = await planGrasp.run({ next_after: "g4", reason: "collision" } as any, undefined, f.ctx as any);
	assert.equal(next.active, "g5");
	assert.deepEqual(env.calls[1], { method: "env.next_grasp", kwargs: { grasp_id: "g4", reason: "collision" } });
	assert.equal(f.entries.length, 1, "nothing expired: no entry");
	// Neither object nor mask id: refused before any call.
	assert.match((await planGrasp.run({} as any, undefined, f.ctx as any)).error, /object .* or mask_id/);
	// A stale id refused by the server is recorded.
	const stale = fakeEnv({
		"env.plan_grasp": new Error(
			"env.plan_grasp: detection d1 is stale: it belongs to observation 2, the current observation is 3",
		),
	});
	const [pg] = graspTools(f.pi, { call: stale.call, cameras: ["agentview"], task: () => "" });
	const r = await pg.run({ mask_id: "d1" } as any, undefined, f.ctx as any);
	assert.match(r.error, /stale/);
	assert.equal(f.entries.at(-1)?.type, DETECTIONS_EXPIRED_ENTRY);
	assert.match(f.entries.at(-1)?.data.error, /stale/);
	assert.ok(isStale(new Error("g3 is stale: ...")) && !isStale(new Error("no grasp backend")));
});

test("plan_place segments the region text into a mask id first; the object defaults to the grasp's mask", async () => {
	const f = fakePi();
	const env = fakeEnv({
		"env.segment_mask": { found: true, id: "d7" },
		"env.plan_place": { observation: 1, candidates: [{ id: "p1" }], active: "p1", expired_ids: [] },
	});
	const arm = { schema: StringEnum(["left", "right"] as const), name: (v: unknown) => String(v) };
	const [, planPlace] = graspTools(f.pi, { call: env.call, task: () => "task", arm });
	const out = await planPlace.run({ region: "plate", grasp_id: "g1" } as any, undefined, f.ctx as any);
	assert.equal(out.active, "p1");
	assert.deepEqual(
		env.calls.map((c) => c.method),
		["env.segment_mask", "env.plan_place"],
	);
	// The object is never re-segmented: that would give a new id, not the one the grasp was planned on.
	assert.deepEqual(env.calls[1].kwargs, { region_mask_id: "d7", grasp_id: "g1" });
	await planPlace.run({ region_mask_id: "d2", grasp_id: "g1", object_mask_id: "d1" } as any, undefined, f.ctx as any);
	assert.deepEqual(env.calls[2].kwargs, { region_mask_id: "d2", grasp_id: "g1", object_mask_id: "d1" });
	const missing = await planPlace.run({ grasp_id: "g1", object_mask_id: "d1" } as any, undefined, f.ctx as any);
	assert.match(missing.error, /region/);
	// Without cameras the camera parameter is free text (the dual arm's registered views).
	assert.equal((planPlace.parameters as Json).properties.camera.type, "string");
});

test("check_attached asks the VLM with the env's frames, parses the verdict and records the entry and cost", async () => {
	const f = fakePi({}, [
		{
			text: 'Sure: {"attached": true, "confidence": 0.8, "reason": "the bowl hangs   from the fingers"}',
			usd: 0.002,
		},
	]);
	const png = PNG.toString("base64");
	const env = fakeEnv({
		"env.attachment_frames": {
			observation: 5,
			frames: [
				{ camera: "wrist", png_base64: png, crop_rc: null },
				{ camera: "agentview", png_base64: png, crop_rc: [10, 20, 110, 120] },
			],
		},
	});
	const [, , check] = graspTools(f.pi, {
		call: env.call,
		cameras: ["agentview", "wrist"],
		task: () => "put the bowl on the plate",
	});
	const out = await check.run({ object: "black bowl" } as any, undefined, f.ctx as any);
	assert.equal(out.attached, true);
	assert.equal(out.confidence, 0.8);
	assert.equal(out.reason, "the bowl hangs from the fingers");
	assert.equal(out.model, "relay/planner");
	assert.equal((out._pngs as Buffer[]).length, 2, "the frames come back as images");
	assert.deepEqual(out.frames, [
		{ camera: "wrist", crop_rc: null },
		{ camera: "agentview", crop_rc: [10, 20, 110, 120] },
	]);
	assert.equal(f.asked.length, 1);
	assert.match(f.asked[0].system ?? "", /JSON only/);
	assert.equal(f.asked[0].content.filter((c: any) => c.type === "image").length, 2);
	assert.match(f.asked[0].content[1].text, /black bowl/);
	assert.deepEqual(f.costs, [0.002]);
	const entry = f.entries.find((e) => e.type === CHECK_ATTACHED_ENTRY);
	assert.equal(entry?.data.attached, true);
	assert.equal(entry?.data.observation, 5);
	assert.equal(entry?.data.cost_usd, 0.002);
	// The model picks --attach-vlm-model, else --units-vlm-model.
	const g = fakePi({ "attach-vlm-model": "vision/eyes" }, [{ text: '{"attached": false}' }]);
	registerGraspFlags(g.pi);
	const [, , check2] = graspTools(g.pi, { call: env.call, cameras: ["agentview", "wrist"], task: () => "" });
	const r = await check2.run({ object: "bowl" } as any, undefined, g.ctx as any);
	assert.equal(r.attached, false);
	assert.equal(g.asked[0].model, "vision/eyes");
});

test("the attachment verdict is parsed leniently and never throws", () => {
	assert.deepEqual(parseAttached('{"attached": true, "confidence": 2, "reason": "x"}'), {
		attached: true,
		confidence: 1,
		reason: "x",
	});
	assert.equal(parseAttached('prose "attached": false prose').attached, false);
	assert.equal(parseAttached("no idea").attached, false);
	assert.equal(parseAttached("no idea").confidence, 0);
	const p = attachedPrompt("task", "cup", ["wrist", "agentview (cropped around the gripper)"]);
	assert.match(p.content[1].text, /2 image\(s\)/);
});
