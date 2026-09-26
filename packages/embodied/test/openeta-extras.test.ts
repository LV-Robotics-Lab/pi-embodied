import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { type AssistantMessage, createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import dashboard from "../src/dashboard/index.ts";
import {
	argumentSkeleton,
	checkReply,
	HUMAN_EVENT,
	type HumanRequest,
	human,
	humanRequested,
	parseArguments,
} from "../src/human.ts";
import { OBJECT_ENTRY, objectMemory, replay } from "../src/objects.ts";
import { GRASP_SUGGESTION_ENTRY, graspAdvisorTool, parseAdvice } from "../src/primitives/advisor.ts";
import { checkRoute, followWaypoints, type WaypointRig, waypointsTool } from "../src/primitives/waypoints.ts";
import { alignWrist, compose, projectPoints, wristAlignment } from "../src/primitives/wrist.ts";
import type { Json, Mat } from "../src/robot.ts";
import { webTools } from "../src/web.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi: flags, tools, providers, entries, hooks and a working pi.events. */
function fakePi(flagValues: Record<string, unknown> = {}, loaded: string[] = []) {
	const handlers = new Map<string, Handler[]>();
	const listeners = new Map<string, ((data: unknown) => void)[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const providers: any[] = [];
	const entries: { type: string; customType: string; data: any }[] = [];
	const notes: string[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		getThinkingLevel: () => "off",
		registerTool: (t: any) => tools.set(t.name, t),
		registerProvider: (p: any) => providers.push(p),
		registerCommand: () => {},
		getAllTools: () => [...loaded, ...tools.keys()].map((name) => ({ name })),
		appendEntry: (customType: string, data: any) => entries.push({ type: "custom", customType, data }),
		events: {
			emit: (channel: string, data: unknown) => {
				for (const fn of listeners.get(channel) ?? []) fn(data);
			},
			on: (channel: string, fn: (data: unknown) => void) => {
				listeners.set(channel, [...(listeners.get(channel) ?? []), fn]);
				return () => {};
			},
		},
	} as unknown as ExtensionAPI;
	const ctx: any = {
		hasUI: false,
		ui: { notify: (m: string) => notes.push(m), setWidget: () => {} },
		sessionManager: { getBranch: () => entries },
		isIdle: () => true,
	};
	const emit = async (name: string, event: Record<string, unknown> = {}) => {
		let out: unknown;
		for (const fn of handlers.get(name) ?? []) out = (await fn({ type: name, ...event }, ctx)) ?? out;
		return out;
	};
	const run = async (name: string, params: Json) => {
		const r = await tools.get(name).execute("id", params, undefined, undefined, ctx);
		return r.details as Json;
	};
	return { pi, ctx, emit, flags, tools, providers, entries, notes, run };
}

// ---------------------------------------------------------------------------
// follow_waypoints (OpenETA follow_eef_trajectory)

function waypointRig(o: Partial<WaypointRig> = {}) {
	const moves: number[][][] = [];
	const rig: WaypointRig = {
		current: () => [0, 0, 0.3],
		maxSegment: () => 0.1,
		maxPath: () => 0.25,
		segment: async (from, to) => {
			moves.push([from, to]);
			return { reached: true };
		},
		...o,
	};
	return { rig, moves };
}

test("a route is checked whole before anything moves: count, each segment, total length, workspace", () => {
	const { rig } = waypointRig({ workspace: (t) => assert.ok(t[0] <= 0.12, "outside the box") });
	const start = [0, 0, 0.3];
	assert.deepEqual(
		checkRoute(start, [[0.1, 0, 0.3]], rig).map((v) => Number(v.toFixed(6))),
		[0.1],
	);
	assert.throws(() => checkRoute(start, [], rig), /1 to 5/);
	assert.throws(() => checkRoute(start, Array(6).fill(start), rig), /1 to 5/);
	assert.throws(
		() => checkRoute(start, [[0.2, 0, 0.3]], rig),
		/segment 0: .*limit is 0.1 m.*Add waypoints in between/,
	);
	assert.throws(
		() =>
			checkRoute(
				start,
				[
					[0, 0.1, 0.3],
					[0, 0.2, 0.3],
					[0, 0.3, 0.3],
				],
				rig,
			),
		/route is 0.3 m long; the limit is 0.25 m/,
	);
	assert.throws(
		() =>
			checkRoute(
				start,
				[
					[0.1, 0, 0.3],
					[0.19, 0, 0.3],
				],
				rig,
			),
		/outside the box/,
	);
	// The task's documented per-call limit is tighter than the robot's.
	const documented = waypointRig({ constraints: () => ["Keep translation commands at or below 0.02 m per call."] });
	assert.throws(() => checkRoute(start, [[0.05, 0, 0.3]], documented.rig), /limit is 0.02 m/);
});

test("follow_waypoints runs segments in order and stops at the first one not reached or failing", async () => {
	const a = waypointRig();
	const t = followWaypoints(a.rig);
	const ok = await t.run(
		{
			waypoints: [
				[0, 0, 0.35],
				[0.05, 0, 0.35],
			],
			gripper: 1,
		},
		undefined,
	);
	assert.equal(ok.reached_target, true);
	assert.equal(ok.waypoints_completed, 2);
	assert.deepEqual(a.moves, [
		[
			[0, 0, 0.3],
			[0, 0, 0.35],
		],
		[
			[0, 0, 0.35],
			[0.05, 0, 0.35],
		],
	]);
	let n = 0;
	const b = waypointRig({
		segment: async () => (++n === 1 ? { reached: false, final_dist_m: 0.03 } : { reached: true }),
	});
	const stopped = await followWaypoints(b.rig).run(
		{
			waypoints: [
				[0, 0, 0.35],
				[0, 0, 0.4],
			],
		},
		undefined,
	);
	assert.equal(stopped.stop_reason, "not_reached");
	assert.equal(stopped.segments.length, 1);
	const c = waypointRig({
		segment: async () => {
			throw new Error("reflex");
		},
	});
	const failed = await followWaypoints(c.rig).run({ waypoints: [[0, 0, 0.35]] }, undefined);
	assert.equal(failed.stop_reason, "error");
	assert.equal(failed.segments[0].error, "reflex");
	// A route over the limit moves nothing.
	const d = waypointRig();
	const refused = await followWaypoints(d.rig).run({ waypoints: [[0.5, 0, 0.3]] }, undefined);
	assert.match(refused.error, /segment 0: .*Add waypoints in between/);
	assert.equal(refused.moved, false);
	assert.equal(d.moves.length, 0);
	// A real arm's gripper keeps its last command: no gripper parameter.
	const props = (r: WaypointRig) => (followWaypoints(r).parameters as { properties: object }).properties;
	assert.ok("gripper" in props(a.rig));
	assert.ok(!("gripper" in props({ ...a.rig, gripper: false })));
});

test("--waypoints registers nothing when off and mounts the tool once when on", () => {
	const mounted: string[] = [];
	const off = fakePi();
	assert.deepEqual(waypointsTool(off.pi, waypointRig().rig, (d) => mounted.push(d.name))(), [] as string[]);
	assert.equal(mounted.length, 0);
	const on = fakePi({ waypoints: true });
	const activate = waypointsTool(on.pi, waypointRig().rig, (d) => mounted.push(d.name));
	assert.deepEqual(activate(), ["follow_waypoints"]);
	assert.deepEqual(activate(), ["follow_waypoints"]);
	assert.deepEqual(mounted, ["follow_waypoints"]);
});

// ---------------------------------------------------------------------------
// align_wrist (OpenETA compute_wrist_alignment)

/** A wrist camera 0.2 m above the gripper centre... looking straight down: camera x = world x, camera y = -world y. */
const K: Mat = [
	[500, 0, 320],
	[0, 500, 240],
	[0, 0, 1],
];
const DOWN: Mat = [
	[1, 0, 0, 0],
	[0, -1, 0, 0],
	[0, 0, -1, 0.5],
	[0, 0, 0, 1],
];

test("wristAlignment moves the gripper centre's pixel onto the target at the target's depth, clamped", () => {
	// Gripper centre 0.2 m below the camera on its axis; target 0.4 m below, 1 cm along +x and 2 cm along +y.
	const a = wristAlignment({
		K,
		cam2world: DOWN,
		gripper: [0, 0, 0.3],
		target: [0.01, 0.02, 0.1],
		maxCorrection: 0.03,
	});
	assert.deepEqual(a.desired_pixel, [240, 320]);
	assert.deepEqual(a.target_pixel, [215, 333]);
	assert.equal(a.target_depth_m, 0.4);
	assert.deepEqual(
		a.delta_world.map((v) => Number(v.toFixed(4))),
		[0.01, 0.02, 0],
	);
	assert.equal(a.clamped, false);
	assert.deepEqual(a.aligned_xyz, [0.01, 0.02, 0.3]);
	const far = wristAlignment({
		K,
		cam2world: DOWN,
		gripper: [0, 0, 0.3],
		target: [0.3, 0.4, 0.1],
		maxCorrection: 0.03,
	});
	assert.equal(far.clamped, true);
	assert.equal(Number(Math.hypot(...far.delta_world).toFixed(4)), 0.03);
	assert.throws(
		() => wristAlignment({ K, cam2world: DOWN, gripper: [0, 0, 0.6], target: [0, 0, 0.1], maxCorrection: 0.03 }),
		/gripper centre does not project/,
	);
	assert.deepEqual(
		projectPoints(K, DOWN, [
			[0.01, 0.02, 0.1],
			[0, 0, 0.9],
		]),
		[[215, 333], null],
	);
	// compose: a translation after the camera pose.
	const shifted = compose(
		[
			[1, 0, 0, 0.1],
			[0, 1, 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1],
		],
		DOWN,
	);
	assert.deepEqual(shifted[0], [1, 0, 0, 0.1]);
});

test("align_wrist returns the aligned position and the marked wrist image; it moves nothing", async () => {
	const rgb = Buffer.alloc(640 * 480 * 3);
	const t = alignWrist({
		moveWith: "move_to xyz",
		gripper: () => [0, 0, 0.3],
		view: async (row, col) => {
			assert.deepEqual([row, col], [215, 333]);
			return { K, cam2world: DOWN, target: [0.01, 0.02, 0.1], image: { width: 640, height: 480, rgb } };
		},
	});
	const r = await t.run({ point: [215, 333] }, undefined);
	assert.deepEqual(r.aligned_xyz, [0.01, 0.02, 0.3]);
	assert.equal(r.max_correction_m, 0.03);
	assert.equal(r._pngs.length, 1);
	assert.ok(t.description.includes("move_to xyz"));
});

// ---------------------------------------------------------------------------
// suggest_grasp (OpenETA grasp_pose_advisor)

test("parseAdvice keeps only offered ids and turns anything malformed into an abstention", () => {
	const ids = ["g1", "g2", "g3"];
	const ok = parseAdvice(
		'Sure: {"decision":"recommend","recommended_candidate_id":"g2","alternatives":["g3","g9","g2"],"confidence":0.8,"reasons":["deep side contact"],"rejected":{"g1":"on the rim","g7":"x"}}',
		ids,
	);
	assert.equal(ok.recommended_candidate_id, "g2");
	assert.deepEqual(ok.alternatives, ["g3"]);
	assert.deepEqual(ok.rejected, { g1: "on the rim" });
	assert.equal(ok.warnings.length, 2);
	const invented = parseAdvice('{"decision":"recommend","recommended_candidate_id":"g8","confidence":0.9}', ids);
	assert.equal(invented.decision, "abstain");
	assert.equal(invented.confidence, 0.5);
	assert.equal(parseAdvice("no idea", ids).decision, "abstain");
});

test("suggest_grasp draws the latest plan's candidates, asks the advisor model and records the advice", async () => {
	const f = fakePi({ "grasp-advisor": true, "grasp-advisor-model": "vlm/judge" });
	let stamp = 7;
	const asked: any[] = [];
	f.ctx.modelRegistry = {
		find: (provider: string, id: string) => ({ provider, id, api: "x", reasoning: false }),
		streamSimple(model: any, context: any) {
			asked.push({ model: `${model.provider}/${model.id}`, context });
			const s = createAssistantMessageEventStream();
			const message = {
				role: "assistant",
				content: [
					{
						type: "text",
						text: '{"decision":"recommend","recommended_candidate_id":"g2","alternatives":[],"confidence":0.7,"reasons":["broad body"],"rejected":{}}',
					},
				],
				usage: { cost: { total: 0.002 } },
				stopReason: "stop",
			} as unknown as AssistantMessage;
			s.push({ type: "done", reason: "stop", message });
			s.end();
			return s;
		},
	};
	const mounted: any[] = [];
	const activate = graspAdvisorTool(
		f.pi,
		{
			image: async () => ({ width: 640, height: 480, rgb: Buffer.alloc(640 * 480 * 3) }),
			project: async (_c, points) => projectPoints(K, DOWN, points),
			stamp: () => stamp,
			task: () => "pick up the can",
		},
		(d) => mounted.push(d),
	);
	assert.throws(() => activate(false), /needs a plan_grasp backend/);
	assert.deepEqual(activate(true), ["suggest_grasp"]);
	const tool = mounted[0];
	assert.match(String((await tool.run({}, undefined, f.ctx)).error), /call plan_grasp first/);
	const cand = (id: string, x: number) => ({
		id,
		rank: Number(id.slice(1)) - 1,
		score: 0.9,
		position: [x, 0, 0.1],
		approach: [0, 0, -1],
		width_m: 0.05,
		contact_points: [
			[x, -0.02, 0.1],
			[x, 0.02, 0.1],
		],
	});
	await f.emit("tool_result", {
		toolName: "plan_grasp",
		isError: false,
		input: { object: "can" },
		details: { camera: "agentview", candidates: [cand("g1", 0), cand("g2", 0.05), cand("g3", 0.3)] },
	});
	await f.emit("tool_result", {
		toolName: "plan_grasp",
		isError: false,
		input: {},
		details: { rejected: { id: "g1" } },
	});
	const r = await tool.run({}, undefined, f.ctx);
	assert.equal(r.recommended_candidate_id, "g2");
	assert.deepEqual(
		r.candidates.map((c: Json) => [c.id, c.color]),
		[
			["g2", "cyan"],
			["g3", "yellow"],
		],
	);
	assert.equal(r._pngs.length, 1);
	assert.equal(asked[0].model, "vlm/judge");
	assert.match(JSON.stringify(asked[0].context.messages[0].content[0]), /OBJECT: can/);
	const entry = f.entries.find((e) => e.customType === GRASP_SUGGESTION_ENTRY)?.data;
	assert.deepEqual(entry.offered, ["g2", "g3"]);
	stamp = 8;
	assert.match(String((await tool.run({}, undefined, f.ctx)).error), /moved since that plan_grasp/);
});

// ---------------------------------------------------------------------------
// object memory

test("object memory: records by name with the sighting's step, rebuilt from the branch", async () => {
	let step = 3;
	const f = fakePi({ "object-memory": true });
	const om = objectMemory(f.pi, {
		robot: "libero",
		scene: () => ({ suite: "libero_10", task: "2" }),
		step: () => step,
	});
	await f.emit("session_start");
	assert.deepEqual(om.tools(), ["remember_object", "recall_objects", "forget_object"]);
	await f.run("remember_object", { name: "Red Mug", position: [0.1, 0.2, 0.05], note: "on the plate" });
	step = 9;
	const upd = await f.run("remember_object", { name: "red  mug", position: [0.3, 0.2, 0.05] });
	assert.equal(upd.updated, true);
	await f.run("remember_object", { name: "bowl", position: [0, 0, 0] });
	const all = await f.run("recall_objects", {});
	assert.deepEqual(
		all.objects.map((o: Json) => [o.name, o.position, o.last_seen_step]),
		[
			["red  mug", [0.3, 0.2, 0.05], 9],
			["bowl", [0, 0, 0], 9],
		],
	);
	assert.deepEqual((await f.run("recall_objects", { names: ["BOWL", "plate"] })).unknown, ["plate"]);
	await f.run("forget_object", { name: "Bowl" });
	assert.deepEqual(
		[...replay(f.entries).values()].map((r) => r.name),
		["red  mug"],
	);
	assert.ok(f.entries.every((e) => e.customType === OBJECT_ENTRY));
	const off = fakePi();
	assert.deepEqual(objectMemory(off.pi, { robot: "x", scene: () => ({}), step: () => undefined }).tools(), []);
	assert.equal(off.tools.size, 0);
});

test("--object-memory-dir carries a scene's records into its next episode, marked as earlier", async () => {
	const dir = mkdtempSync(join(tmpdir(), "objmem-"));
	const scene = () => ({ task: "1" });
	const a = fakePi({ "object-memory": true, "object-memory-dir": dir });
	const oa = objectMemory(a.pi, { robot: "franka", scene, step: () => 1 });
	await a.emit("session_start");
	oa.tools();
	await a.run("remember_object", { name: "block", position: [0.5, 0, 0.02] });
	const file = JSON.parse(readFileSync(join(dir, "franka", "task-1.json"), "utf8"));
	assert.deepEqual(
		file.records.map((r: Json) => r.name),
		["block"],
	);
	const b = fakePi({ "object-memory": true, "object-memory-dir": dir });
	const ob = objectMemory(b.pi, { robot: "franka", scene, step: () => 0 });
	await b.emit("session_start");
	ob.tools();
	const got = await b.run("recall_objects", {});
	assert.equal(got.objects[0].from_earlier_episode, true);
	// Another scene starts empty.
	const c = fakePi({ "object-memory": true, "object-memory-dir": dir });
	const oc = objectMemory(c.pi, { robot: "franka", scene: () => ({ task: "2" }), step: () => 0 });
	await c.emit("session_start");
	oc.tools();
	assert.deepEqual((await c.run("recall_objects", {})).objects, []);
});

// ---------------------------------------------------------------------------
// web tools from pi packages

test("--web-tools keeps the pi packages' web_search / web_fetch active, and fails closed when one is missing", () => {
	assert.deepEqual(webTools(fakePi().pi).tools(), []);
	assert.deepEqual(webTools(fakePi({ "web-tools": true }, ["web_search", "web_fetch"]).pi).tools(), [
		"web_search",
		"web_fetch",
	]);
	assert.throws(
		() => webTools(fakePi({ "web-tools": true }, ["web_search"]).pi).tools(),
		/web_fetch not loaded; pi install npm:@zeldrisho\/pi-web-fetch/,
	);
});

// ---------------------------------------------------------------------------
// human/operator

const TOOLS = [
	{
		name: "move_to",
		description: "servo",
		parameters: {
			type: "object",
			properties: { xyz: { type: "array" }, gripper: { type: "number" } },
			required: ["xyz"],
		},
	},
];
const CONTEXT = {
	messages: [
		{ role: "system", content: "robot prompt", toolsAdded: TOOLS, timestamp: 1 },
		{
			role: "user",
			content: [
				{ type: "text", text: "Solve the task." },
				{ type: "image", data: Buffer.from("png").toString("base64"), mimeType: "image/png" },
			],
			timestamp: 2,
		},
	],
};
async function turn(provider: any, signal?: AbortSignal) {
	const stream = provider.streamSimple(provider.getModels()[0], CONTEXT, { signal });
	for await (const _ of stream);
	return (await stream.result()) as AssistantMessage;
}

test("human/operator is registered only when a human/... model is on the command line", () => {
	assert.equal(humanRequested(["pi", "--model", "human/operator"]), true);
	assert.equal(humanRequested(["pi", "--attach-vlm-model=human/operator"]), true);
	assert.equal(humanRequested(["pi", "--model", "selfhost/muse"]), false);
	const f = fakePi();
	human(f.pi, ["pi", "--model", "selfhost/muse"]);
	assert.equal(f.providers.length, 0);
	assert.equal(argumentSkeleton(TOOLS[0].parameters), '{"xyz":[]} // optional: gripper');
	assert.deepEqual(parseArguments('{"xyz":[1,2,3]} // optional: gripper'), { xyz: [1, 2, 3] });
	assert.equal(checkReply({ tool: { name: "fly", arguments: {} } }, TOOLS), '"fly" is not an offered tool');
	assert.equal(checkReply({ tool: { name: "move_to", arguments: [] } }, TOOLS), "arguments must be a JSON object");
});

test("the operator answers through pi's UI dialogs; an invalid answer is asked again", async () => {
	const f = fakePi();
	human(f.pi, ["pi", "--model", "human/operator"]);
	const inputs = ["not json", '{"xyz":[0.1,0.2,0.3]}'];
	const asked: string[] = [];
	f.ctx.hasUI = true;
	f.ctx.ui = {
		...f.ctx.ui,
		setWidget: (_k: string, lines?: string[]) => lines && asked.push(lines[0]),
		select: async (_t: string, options: string[]) => {
			assert.deepEqual(options, ["reply with text", "move_to"]);
			return "move_to";
		},
		input: async (title: string, placeholder: string) => {
			assert.equal(title, "move_to arguments (JSON object)");
			assert.equal(placeholder, '{"xyz":[]} // optional: gripper');
			return inputs.shift();
		},
	};
	await f.emit("session_start");
	const m = await turn(f.providers[0]);
	assert.equal(m.stopReason, "toolUse");
	assert.deepEqual(m.content[0], {
		...m.content[0],
		type: "toolCall",
		name: "move_to",
		arguments: { xyz: [0.1, 0.2, 0.3] },
	});
	assert.match(asked[0], /Turn 1 \(planner\).*1 image/);
	assert.ok(f.notes.some((n) => n.startsWith("invalid JSON")));
});

test("without pi's UI the dashboard answers (request, images and reply over HTTP); nobody to ask fails the turn", async () => {
	const alone = fakePi();
	human(alone.pi, ["pi", "--model", "human/operator"]);
	await alone.emit("session_start");
	const failed = await turn(alone.providers[0]);
	assert.equal(failed.stopReason, "error");
	assert.match(String(failed.errorMessage), /nobody to ask/);

	const f = fakePi({ dashboard: true });
	human(f.pi, ["pi", "--model", "human/operator"]);
	dashboard(f.pi);
	let url = "";
	f.ctx.ui.notify = (m: string) => {
		if (m.startsWith("Dashboard: ")) url = m.slice(11);
	};
	f.ctx.hasUI = true;
	f.ctx.ui.select = (_t: string, _o: string[], o: { signal: AbortSignal }) =>
		new Promise((resolve) => o.signal.addEventListener("abort", () => resolve(undefined)));
	await f.emit("session_start");
	let pending: HumanRequest | undefined;
	f.pi.events.on(HUMAN_EVENT, (d) => {
		if (!(d as HumanRequest).done) pending = d as HumanRequest;
	});
	const answered = turn(f.providers[0]);
	while (!pending) await new Promise((r) => setTimeout(r, 5));
	assert.equal(pending.claimed, true);
	const post = (body: Json) =>
		fetch(`${url}human/reply`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(body),
		});
	const img = await fetch(`${url}human/image/${pending.id}/0`);
	assert.equal(Buffer.from(await img.arrayBuffer()).toString(), "png");
	assert.equal((await post({ id: pending.id, tool: "fly", arguments: "{}" })).status, 422);
	assert.equal((await post({ id: "other", text: "hi" })).status, 409);
	assert.equal((await post({ id: pending.id, text: "I see a red mug on the left." })).status, 200);
	const m = await answered;
	assert.deepEqual(m.content, [{ type: "text", text: "I see a red mug on the left." }]);
	assert.equal(m.stopReason, "stop");
	await f.emit("session_shutdown", { reason: "quit" });
});

test("an abort ends a pending human turn as aborted", async () => {
	const f = fakePi();
	human(f.pi, ["pi", "--model", "human/operator"]);
	f.ctx.hasUI = true;
	f.ctx.ui.select = (_t: string, _o: string[], o: { signal: AbortSignal }) =>
		new Promise((resolve) => o.signal.addEventListener("abort", () => resolve(undefined)));
	await f.emit("session_start");
	const ac = new AbortController();
	const p = turn(f.providers[0], ac.signal);
	setTimeout(() => ac.abort(), 10);
	assert.equal((await p).stopReason, "aborted");
});
