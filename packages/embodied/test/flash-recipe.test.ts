import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { anchorPlan, generate, parseTargets } from "../src/flash/generate.ts";
import type { FlashCall, FlashReply, FlashRobot } from "../src/flash/index.ts";
import { loadProgram, recipeFlash, startRecipe } from "../src/flash/recipe.ts";
import type { RpcClient } from "../src/rpc.ts";

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
	assert.throws(() => parseTargets("move_to=abc"), /bad target/);
});

/** A robot whose observation shows one image and whose back-projection answers `world`. */
function fakeRobot(world: number[] | undefined) {
	const notes: string[] = [];
	const moves: FlashCall[] = [];
	let latest: FlashReply = { json: {}, images: [] };
	const robot: FlashRobot = {
		act: async () => [{ json: {}, images: [] }],
		move: async (call) => {
			moves.push(call);
			latest = { json: {}, images: ["IMG"] };
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
