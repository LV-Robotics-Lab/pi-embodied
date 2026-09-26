import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI, SessionEntry } from "@earendil-works/pi-coding-agent";
import { anchorPlan, generate, parseTargets, pathOf, sessionPlan } from "../src/flash/generate.ts";
import { type FlashCall, type FlashHook, type FlashReply, type FlashRobot, runFlash } from "../src/flash/index.ts";
import { pixelOnPlane, projectPixel, unletterbox } from "../src/flash/plane.ts";
import {
	type AnchoredEntry,
	loadProgram,
	type RecipeProgram,
	recipeFlash,
	splitMove,
	startRecipe,
} from "../src/flash/recipe.ts";
import { NdArray, type RpcClient } from "../src/rpc.ts";

const RECIPE = [
	{ action: "move_to", xyz: [0.5, 0.1, 0.9], gripper: "open" },
	{ action: "scripted_grasp", xyz: [0.52, 0.12, 0.85] },
	{ action: "navigate_to", xy: [2.0, 1.0] },
	{ action: "move_to", xyz: [1.5, -0.5, 0.95] },
	{ action: "release" },
];
const recipeFile = (dir: string, name: string) => {
	const path = join(dir, `${name}_recipe.jsonl`);
	writeFileSync(path, RECIPE.map((r) => `${JSON.stringify(r)}\n`).join(""));
	return path;
};
const TARGETS = parseTargets("move_to=xyz,scripted_grasp=xyz,navigate_to=xy");
const tcp = (json: Record<string, unknown>) => (json.state as { tcp_pos?: number[] } | undefined)?.tcp_pos;
/** The robots' delta move: `state.tcp_pos` after it, at most 0.2 m per call. */
const DELTA = { move_delta: { delta: "delta_xyz", position: tcp, maxStep: 0.2 } };

test("a recipe loads as recorded; the generator anchors targets within 0.2 m of an anchor", () => {
	const dir = mkdtempSync(join(tmpdir(), "flash-"));
	const program = loadProgram(recipeFile(dir, "cell"), "cell");
	assert.deepEqual(program.plan[1], { action: "scripted_grasp", arguments: { xyz: [0.52, 0.12, 0.85] } });
	const plan = anchorPlan(program.plan, [{ phrase: "the mug", xyz: [0.5, 0.1, 0.8] }], TARGETS);
	assert.deepEqual(
		plan.map((e) => [e.anchor, e.offset]),
		[
			["the mug", [0, 0]],
			["the mug", [0.02, 0.02]],
			[undefined, undefined],
			[undefined, undefined],
			[undefined, undefined],
		],
	);
	// A saved ground_truth_poses result serves as the anchors, names read as phrases.
	const gt = join(dir, "gt.json");
	writeFileSync(gt, JSON.stringify({ frame: "world", poses: { red_mug: { pos: [0.5, 0.1, 0.8] } } }));
	const out = generate({
		recipe: recipeFile(dir, "cell"),
		anchors: gt,
		targets: "move_to=xyz",
		destination: join(dir, "flash"),
	});
	assert.equal(out.anchored, 1);
	const doc = JSON.parse(readFileSync(out.path, "utf8"));
	assert.equal(doc.plan[0].anchor, "red mug");
	assert.throws(() => parseTargets("move_to=abc"), /--position/);
	assert.throws(() => parseTargets("move_to"), /bad target/);
	assert.deepEqual(Object.keys(parseTargets("move_delta=delta_xyz", tcp)), ["move_delta"]);
});

/** A robot whose observation shows one image and whose back-projection answers `world`. */
function fakeRobot(world: number[] | undefined, state: Record<string, unknown> = {}) {
	const notes: string[] = [];
	const moves: FlashCall[] = [];
	let latest: FlashReply = { json: {}, images: [] };
	const robot: FlashRobot = {
		act: async () => [{ json: {}, images: [] }],
		move: async (call) => {
			moves.push(call);
			latest = { json: { state }, images: ["IMG"] };
			return latest;
		},
		latest: () => latest,
		note: (l) => notes.push(l),
	};
	const backProject = async () => world;
	return { robot, notes, moves, backProject };
}
const molmo = { call: async () => ({ point_xy: [10, 20] }) } as unknown as RpcClient;
const anchored = {
	name: "cell",
	anchors: [{ phrase: "the mug", xyz: [0.5, 0.1, 0.8] }],
	plan: [
		{
			action: "move_to",
			arguments: { xyz: [0.5, 0.1, 0.9] },
			anchor: "the mug",
			offset: [0.01, -0.02] as [number, number],
		},
		{ action: "release", arguments: {} },
	],
};

test("a replay moves anchored targets with the live anchor in x/y and keeps the recorded height", async () => {
	const f = fakeRobot([0.7, 0.3, 0.8]);
	const r = await startRecipe(
		anchored,
		f.robot,
		{ observe: "view_env_state", targets: TARGETS, backProject: f.backProject },
		molmo,
	);
	assert.equal(f.moves[0].name, "view_env_state");
	assert.equal(r.localized, 1);
	assert.deepEqual(r.rewrite(anchored.plan[0]), { name: "move_to", arguments: { xyz: [0.71, 0.28, 0.9] } });
	assert.deepEqual(r.rewrite(anchored.plan[1]), { name: "release", arguments: {} });
});

test("an anchor that is not found stops the replay; without Molmo or back-projection anchors stay recorded", async () => {
	const lost = fakeRobot(undefined);
	const r = await startRecipe(
		anchored,
		lost.robot,
		{ observe: "view_env_state", targets: TARGETS, backProject: lost.backProject },
		molmo,
	);
	assert.equal(r.rewrite(anchored.plan[0]), "stop");
	assert.match(lost.notes.join("\n"), /the mug not located/);
	for (const [backProject, client] of [
		[fakeRobot([9, 9, 9]).backProject, undefined],
		[undefined, molmo],
	] as const) {
		const f = fakeRobot([9, 9, 9]);
		const s = await startRecipe(
			anchored,
			f.robot,
			{ observe: "view_env_state", targets: TARGETS, backProject },
			client,
		);
		assert.equal(s.localized, 0);
		assert.deepEqual(s.rewrite(anchored.plan[0]), { name: "move_to", arguments: { xyz: [0.51, 0.08, 0.9] } });
	}
});

/** A delta-move plan: approach the block (anchored), lift (anchored, same x/y), a free transit. */
const DELTA_PLAN: RecipeProgram = {
	name: "cell",
	anchors: [{ phrase: "orange block", xyz: [0.55, -0.1, 0.06] }],
	plan: [
		{
			action: "move_delta",
			arguments: { delta_xyz: [0.19, -0.11, -0.2] },
			to: [0.55, -0.1, 0.06],
			anchor: "orange block",
			offset: [0, 0],
		},
		{
			action: "move_delta",
			arguments: { delta_xyz: [0, 0, 0], gripper: "close" },
			to: [0.55, -0.1, 0.06],
			anchor: "orange block",
			offset: [0, 0],
		},
		{
			action: "move_delta",
			arguments: { delta_xyz: [0, 0, 0.1] },
			to: [0.55, -0.1, 0.16],
			anchor: "orange block",
			offset: [0, 0],
		},
		{ action: "move_delta", arguments: { delta_xyz: [-0.1, -0.1, 0] } },
	],
};

test("a relative target's anchored waypoint is replayed as the live anchor minus the live TCP, split by the per-call limit", async () => {
	// The block moved 0.1 m in +y; the arm starts at the recorded start.
	const f = fakeRobot([0.55, 0.0, 0.06], { tcp_pos: [0.36, 0.01, 0.26] });
	const r = await startRecipe(
		DELTA_PLAN,
		f.robot,
		{ observe: "view_env_state", targets: DELTA, backProject: f.backProject },
		molmo,
	);
	assert.equal(f.moves[0].name, "view_env_state");
	// 0.55-0.36 = 0.19, 0.0-0.01 = -0.01, 0.06-0.26 = -0.2: 0.276 m, split in two 0.138 m moves.
	assert.deepEqual(r.rewrite(DELTA_PLAN.plan[0]), [
		{ name: "move_delta", arguments: { delta_xyz: [0.095, -0.005, -0.1] } },
		{ name: "move_delta", arguments: { delta_xyz: [0.095, -0.005, -0.1] } },
	]);
	// The fake's TCP never moves, so the grasp waypoint is the same 0.276 m away; the gripper command rides on the first move only.
	assert.deepEqual(r.rewrite(DELTA_PLAN.plan[1]), [
		{ name: "move_delta", arguments: { delta_xyz: [0.095, -0.005, -0.1], gripper: "close" } },
		{ name: "move_delta", arguments: { delta_xyz: [0.095, -0.005, -0.1] } },
	]);
	// A waypoint without an anchor keeps its recorded delta.
	assert.deepEqual(r.rewrite(DELTA_PLAN.plan[3]), [{ name: "move_delta", arguments: { delta_xyz: [-0.1, -0.1, 0] } }]);
	// Without Molmo the anchor stays recorded: the waypoint is the recorded absolute position minus the live TCP.
	const g = fakeRobot(undefined, { tcp_pos: [0.36, 0.01, 0.26] });
	const s = await startRecipe(DELTA_PLAN, g.robot, { observe: "view_env_state", targets: DELTA }, undefined);
	assert.deepEqual(s.rewrite(DELTA_PLAN.plan[2]), [
		{ name: "move_delta", arguments: { delta_xyz: [0.095, -0.055, -0.05] } },
		{ name: "move_delta", arguments: { delta_xyz: [0.095, -0.055, -0.05] } },
	]);
	// A lost anchor stops before the first waypoint that needs it.
	const lost = fakeRobot(undefined, { tcp_pos: [0.36, 0.01, 0.26] });
	const l = await startRecipe(
		DELTA_PLAN,
		lost.robot,
		{ observe: "view_env_state", targets: DELTA, backProject: lost.backProject },
		molmo,
	);
	assert.equal(l.rewrite(DELTA_PLAN.plan[0]), "stop");
	const entry: AnchoredEntry = { action: "move_delta", arguments: { delta_xyz: [0.3, 0, 0], gripper: "open" } };
	assert.deepEqual(
		splitMove(entry, DELTA.move_delta, [0.3, 0, 0]).map((c) => c.arguments),
		[{ delta_xyz: [0.15, 0, 0], gripper: "open" }, { delta_xyz: [0.15, 0, 0] }],
	);
});

test("runFlash sends a split move as consecutive calls and stops once the episode is over", async () => {
	const sent: FlashCall[] = [];
	let over = false;
	const robot: FlashRobot = {
		act: async () => [],
		move: async (call) => {
			sent.push(call);
			over ||= call.name === "finish_move";
			return { json: { over }, images: [] };
		},
		latest: () => ({ json: { over }, images: [] }),
		note: () => {},
	};
	const hook: FlashHook = {
		load: () => ({ name: "p", plan: [] }),
		start: async () => ({
			localized: 0,
			rewrite: (e) =>
				e.action === "split"
					? [
							{ name: "a", arguments: { i: 1 } },
							{ name: "a", arguments: { i: 2 } },
						]
					: { name: e.action, arguments: {} },
		}),
		over: (l) => l.json.over === true,
		solved: (l) => l.json.over === true,
	};
	const out = await runFlash(
		hook,
		{
			name: "p",
			plan: [
				{ action: "split", arguments: {} },
				{ action: "finish_move", arguments: {} },
				{ action: "never", arguments: {} },
			],
		},
		robot,
	);
	assert.deepEqual(
		sent.map((c) => [c.name, c.arguments.i]),
		[
			["a", 1],
			["a", 2],
			["finish_move", undefined],
		],
	);
	assert.equal(out.done, true);
});

/** A session of `act` units on ManiSkill: STOP observes, moves, GRASP, lift, the last result solves. */
function fakeSession(): SessionEntry[] {
	const steps: [string, Record<string, unknown>, number[], string, boolean][] = [
		["act", { unit: "STOP" }, [0.36, 0.01, 0.29], "open", false],
		["plan", {}, [], "", false],
		["act", { unit: "MV_FWD" }, [0.4, 0.01, 0.29], "open", false],
		["act", { unit: "MV_DOWN", n: 2 }, [0.4, 0.01, 0.21], "open", false],
		["act", { unit: "GRASP" }, [0.4, 0.01, 0.21], "close", false],
		["act", { unit: "MV_UP" }, [0.4, 0.01, 0.25], "close", false],
		["move_delta", { delta_xyz: [0.05, 0, 0] }, [0.45, 0.01, 0.25], "close", false],
		["act", { unit: "RELEASE" }, [0.45, 0.01, 0.25], "open", true],
		["finish", { status: "success" }, [], "", false],
	];
	return steps.flatMap(([name, args, pos, grip, terminated], i): SessionEntry[] => {
		const id = `c${i}`;
		const details =
			name === "act" || name === "move_delta"
				? { terminated, success: terminated, state: { tcp_pos: pos, gripper_command: grip } }
				: {};
		return [
			{
				type: "message",
				message: { role: "assistant", content: [{ type: "toolCall", id, name, arguments: args }] },
			} as unknown as SessionEntry,
			{
				type: "message",
				message: { role: "toolResult", toolCallId: id, toolName: name, content: [], details, isError: false },
			} as unknown as SessionEntry,
		];
	});
}

test("a session becomes delta waypoints with absolute end positions, gripper changes and the recorded absolute calls", () => {
	const plan = sessionPlan(fakeSession(), {
		targets: parseTargets("move_delta=delta_xyz", tcp),
		motion: ["act"],
		gripper: pathOf("state.gripper_command"),
	});
	assert.deepEqual(plan, [
		{ action: "move_delta", arguments: { delta_xyz: [0.04, 0, 0] }, to: [0.4, 0.01, 0.29] },
		{ action: "move_delta", arguments: { delta_xyz: [0, 0, -0.08] }, to: [0.4, 0.01, 0.21] },
		{ action: "move_delta", arguments: { delta_xyz: [0, 0, 0], gripper: "close" }, to: [0.4, 0.01, 0.21] },
		{ action: "move_delta", arguments: { delta_xyz: [0, 0, 0.04] }, to: [0.4, 0.01, 0.25] },
		{ action: "move_delta", arguments: { delta_xyz: [0.05, 0, 0] }, to: [0.45, 0.01, 0.25] },
		{ action: "move_delta", arguments: { delta_xyz: [0, 0, 0], gripper: "open" }, to: [0.45, 0.01, 0.25] },
	]);
	// Anchors attach through `to`: the grasp and the release are each within 0.2 m of one.
	const anchors = [
		{ phrase: "block", xyz: [0.4, 0.0, 0.02] },
		{ phrase: "coaster", xyz: [0.46, 0.02, 0.005] },
	];
	assert.deepEqual(
		anchorPlan(plan, anchors, parseTargets("move_delta=delta_xyz", tcp)).map((e) => [e.anchor, e.offset]),
		[
			["block", [0, 0.01]],
			["block", [0, 0.01]],
			["block", [0, 0.01]],
			["block", [0, 0.01]],
			["coaster", [-0.01, -0.01]],
			["coaster", [-0.01, -0.01]],
		],
	);
	// An unsolved session is refused.
	assert.throws(
		() =>
			sessionPlan(fakeSession().slice(0, 12), {
				targets: parseTargets("move_delta=delta_xyz", tcp),
				motion: ["act"],
				gripper: pathOf("state.gripper_command"),
			}),
		/did not solve/,
	);
	// The CLI path: --session with --position writes the plan.
	const dir = mkdtempSync(join(tmpdir(), "flash-"));
	const session = join(dir, "s.jsonl");
	writeFileSync(
		session,
		fakeSession()
			.map((e) => `${JSON.stringify(e)}\n`)
			.join(""),
	);
	writeFileSync(join(dir, "anchors.json"), JSON.stringify(anchors));
	const out = generate({
		session,
		anchors: join(dir, "anchors.json"),
		targets: "move_delta=delta_xyz",
		position: "state.tcp_pos",
		destination: join(dir, "flash"),
		name: "cell",
	});
	assert.deepEqual([out.calls, out.anchored, out.anchors], [6, 6, 2]);
	assert.equal(JSON.parse(readFileSync(out.path, "utf8")).plan[2].to[2], 0.21);
	assert.throws(() => generate({ session, anchors: "x", targets: "a=xyz", destination: dir }), /--name/);
});

/** RLinf's calibrated front D435 as ../robolab franka.py states it: camera -> base, OpenCV, 640x480. */
const FRONT = {
	intrinsic_K: [
		[607.875, 0, 348.961],
		[0, 607.719, 270.486],
		[0, 0, 1],
	],
	extrinsic_cam2world: [
		[0.02816316, 0.2178868, -0.97556762, 1.1002696],
		[0.99959024, -0.00114196, 0.0286016, -0.00701879],
		[0.00511786, -0.97597338, -0.21782968, 0.2589829],
		[0, 0, 0, 1],
	],
};

test("a pixel ray meets the table plane where the projected point was; the letterbox maps back to raw pixels", () => {
	// Round trip through the front camera: a point on the block's centre plane projects, then unprojects.
	const p = [0.5458, -0.1025, 0.06];
	const px = projectPixel(FRONT, p);
	assert.ok(px);
	// The camera faces the robot from +x: a point left of the base (-y) lands on the image's left, the near table low.
	assert.ok(px[0] < 320 && px[1] > 240, `pixel ${px}`);
	const back = pixelOnPlane(FRONT, px, 0.06);
	assert.ok(back);
	for (let i = 0; i < 3; i++) assert.ok(Math.abs(back[i] - p[i]) < 1e-6, `${back} vs ${p}`);
	// A 1 cm plane error moves the point about 2.8 cm along x at this elevation (about 20 deg).
	const low = pixelOnPlane(FRONT, px, 0.05);
	assert.ok(low && Math.abs(low[0] - p[0]) > 0.02 && Math.abs(low[0] - p[0]) < 0.035, `${low}`);
	// A ray above the horizon never meets the plane; the same for a plane behind the camera.
	assert.equal(pixelOnPlane(FRONT, [320, 0], 0.06), undefined);
	assert.equal(pixelOnPlane(FRONT, px, 1.0), undefined);
	// The RPC's NdArray form is accepted.
	const nd = {
		intrinsic_K: new NdArray("float64", [3, 3], Buffer.from(Float64Array.from(FRONT.intrinsic_K.flat()).buffer)),
		extrinsic_cam2world: new NdArray(
			"float64",
			[4, 4],
			Buffer.from(Float64Array.from(FRONT.extrinsic_cam2world.flat()).buffer),
		),
	};
	assert.deepEqual(pixelOnPlane(nd, px, 0.06), back);
	// 640x480 letterboxed into 256: a 0.4 scale, 32 black rows top and bottom; centres map to centres.
	const box = { width: 640, height: 480, size: 256 };
	assert.deepEqual(unletterbox([127.5, 127.5], box), [319.5, 239.5]);
	const [c, r] = unletterbox([0, 32], box);
	assert.ok(Math.abs(c - 0.75) < 1e-9 && Math.abs(r - 0.75) < 1e-9, `${c},${r}`);
	assert.deepEqual(unletterbox([10, 20], { ...box, size: 0 }), [10, 20]);
});

test("the hook finds the cell's program first, then the reference's, in flash/ then task_only/", () => {
	const memory = mkdtempSync(join(tmpdir(), "memory-"));
	for (const d of ["flash", "task_only"]) mkdirSync(join(memory, d));
	recipeFile(join(memory, "task_only"), "ref");
	const flags: Record<string, unknown> = { molmo: "off" };
	const pi = { registerFlag: () => {}, getFlag: (n: string) => flags[n] } as unknown as ExtensionAPI;
	const hook = recipeFlash(pi, {
		names: () => ["cell", "ref"],
		memory: () => memory,
		observe: "view_env_state",
		targets: TARGETS,
		over: () => false,
		solved: () => false,
	});
	assert.equal((hook.load("/") as { name: string }).name, "ref");
	writeFileSync(join(memory, "flash", "cell_plan.json"), JSON.stringify(anchored));
	const program = hook.load("/") as { name: string; anchors: unknown[] };
	assert.deepEqual([program.name, program.anchors.length], ["cell", 1]);
	flags["flash-plans"] = join(memory, "none");
	assert.throws(() => hook.load("/"), /no Flash program for this episode/);
});
