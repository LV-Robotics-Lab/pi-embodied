import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { toolSections } from "../src/robot.ts";

/** Every simulator prompt of a robot with `--privileged` (its `groundTruth`), and the LIBERO variants. */
const PROMPTS = [
	"robots/robotwin/SYSTEM.md",
	"robots/libero/SYSTEM.md",
	"robots/libero/explore.md",
	"robots/libero/compact/SYSTEM.md",
	"robots/behavior/SYSTEM.md",
	"robots/robodojo/SYSTEM.md",
	"robots/metaworld/SYSTEM.md",
	"robots/maniskill/SYSTEM.md",
	"robots/robocasa/SYSTEM.md",
	"robots/robosuite/SYSTEM.md",
	"robots/robolab/SYSTEM.md",
	"robots/genesis/SYSTEM.md",
];
const NO_POSES =
	/never given|NOT given object|inspect task source, evaluator implementation, hidden rewards, object poses/;

test("the perception-isolation rule follows --privileged: no prompt forbids the ground truth it offers", () => {
	for (const f of PROMPTS) {
		const text = readFileSync(new URL(`../src/${f}`, import.meta.url), "utf8");
		const off = toolSections(text, ["finish"]);
		const on = toolSections(text, ["finish", "ground_truth_poses"]);
		assert.match(off, NO_POSES, `${f}: without --privileged the prompt withholds object poses`);
		assert.doesNotMatch(off, /ground_truth_poses/, `${f}: without --privileged the tool is not named`);
		assert.doesNotMatch(on, NO_POSES, `${f}: with --privileged the prompt does not forbid object poses`);
		assert.match(on, /`ground_truth_poses` gives the simulator's object poses/, f);
	}
});
