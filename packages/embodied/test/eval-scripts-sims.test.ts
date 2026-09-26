import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

/**
 * The eval.sh scripts of the four newer simulators (Robosuite, Metaworld, Genesis, BEHAVIOR) record
 * the fallback planner like the five older ones (test/eval-scripts.test.ts FALLBACK_ROBOTS), plus
 * Robosuite's --max-move and BEHAVIOR's --grasping-mode, and never mix them in one out dir.
 */
const CELLS: [string, string[], string][] = [
	["robosuite", ["Lift", "0"], "Lift_s0"],
	["metaworld", ["reach-v3", "0"], "reach-v3_s0"],
	["genesis", ["cube_pick", "0"], "cube_pick_s0"],
	["behavior", ["turning_on_radio", "0"], "turning_on_radio_s0"],
];

/** Run `<robot>/eval.sh <out> ...positional` with a stand-in pi that records its argv and fails. */
function run(robot: string, positional: string[], cell: string, args: string[]) {
	const dir = mkdtempSync(join(tmpdir(), "eval-"));
	const pi = join(dir, "pi");
	writeFileSync(pi, `#!/usr/bin/env bash\nprintf '%s\\n' "$@" > "$(dirname "$0")/argv"\nexit 1\n`);
	chmodSync(pi, 0o755);
	const script = new URL(`../src/${robot}/eval.sh`, import.meta.url).pathname;
	const r = spawnSync("bash", [script, join(dir, "out"), ...positional, ...args], {
		env: { ...process.env, PI: pi, TIME_LIMIT: "0" },
		encoding: "utf8",
	});
	const read = (p: string) => {
		try {
			return readFileSync(join(dir, p), "utf8");
		} catch {
			return undefined;
		}
	};
	const result = read(`out/${cell}/result.json`);
	return {
		status: r.status,
		stderr: r.stderr,
		argv: read("argv")?.trimEnd().split("\n"),
		result: result && JSON.parse(result),
	};
}

/** eval.sh twice into one out dir, with a stand-in pi that records one successful episode: `first` args, then `second`. */
function rerun(
	robot: string,
	positional: string[],
	first: string[],
	second: string[],
	data: Record<string, unknown> = {},
) {
	const dir = mkdtempSync(join(tmpdir(), "eval-"));
	const pi = join(dir, "pi");
	const entry = JSON.stringify({
		type: "custom",
		customType: "robot_result",
		data: { robot, terminated: true, success: true, env_error: false, planner_error: null, ...data },
	});
	writeFileSync(
		pi,
		`#!/usr/bin/env bash\nwhile [ $# -gt 0 ]; do [ "$1" = --session-dir ] && dir=$2; shift; done\necho '${entry}' > "$dir/s.jsonl"\n`,
	);
	chmodSync(pi, 0o755);
	const script = new URL(`../src/${robot}/eval.sh`, import.meta.url).pathname;
	const once = (args: string[]) =>
		spawnSync("bash", [script, join(dir, "out"), ...positional, ...args], {
			env: { ...process.env, PI: pi, TIME_LIMIT: "0" },
			encoding: "utf8",
		});
	return [once(first), once(second)];
}

for (const [robot, positional, cell] of CELLS) {
	test(`${robot}/eval.sh records the fallback planner, never mixes it with runs without it, and totals planner_models`, () => {
		const run1 = (args: string[]) => run(robot, positional, cell, args);
		const plain = run1([]).result;
		assert.deepEqual(
			[plain?.fallback_model, plain?.fallback_after, plain?.fallback_retry_primary],
			[null, null, null],
		);
		const on = run1(["--fallback-model", "selfhost/muse", "--fallback-after", "3", "--fallback-retry-primary=5"]);
		assert.deepEqual(
			[on.result?.fallback_model, on.result?.fallback_after, on.result?.fallback_retry_primary],
			["selfhost/muse", 3, 5],
		);
		assert.ok(on.argv?.includes("--fallback-model") && on.argv?.includes("selfhost/muse"), String(on.argv));
		// The defaults of src/fallback.ts are recorded when only the model is given; without a model the rest is moot.
		const defaults = run1(["--fallback-model=selfhost/muse"]).result;
		assert.deepEqual([defaults?.fallback_after, defaults?.fallback_retry_primary], [2, 0]);
		assert.equal(run1(["--fallback-after", "3"]).result?.fallback_after, null);
		for (const [first, second] of [
			[["--fallback-model", "selfhost/muse"], []],
			[[], ["--fallback-model=selfhost/muse"]],
			[
				["--fallback-model", "selfhost/muse"],
				["--fallback-model", "selfhost/muse", "--fallback-after", "3"],
			],
		]) {
			const [a, b] = rerun(robot, positional, first, second);
			assert.equal(a.status, 0, a.stdout + a.stderr);
			assert.equal(b.status, 1, `${first} then ${second}`);
			assert.match(b.stderr, /fallback/);
		}
		const planned = { planner_models: { primary: 1, fallback: 2 } };
		const [, same] = rerun(
			robot,
			positional,
			["--fallback-model", "selfhost/muse"],
			["--fallback-model=selfhost/muse"],
			planned,
		);
		assert.equal(same.status, 0, same.stdout + same.stderr);
		assert.match(same.stdout, /\/fallback=selfhost\/muse:2:0/);
		assert.match(same.stdout, /planner_models primary=1 fallback=2/);
	});
}

test("robosuite/eval.sh records --max-move (the robot's default without it) and never mixes caps in one out dir", () => {
	const positional = ["Lift", "0"];
	const run1 = (args: string[]) => run("robosuite", positional, "Lift_s0", args);
	assert.equal(run1([]).result?.max_move, 0.3);
	const on = run1(["--max-move", "0.15"]);
	assert.equal(on.result?.max_move, 0.15);
	assert.ok(on.argv?.includes("--max-move") && on.argv?.includes("0.15"), "pi runs with the cap");
	assert.equal(run1(["--max-move=0.2"]).result?.max_move, 0.2);
	for (const [first, second] of [
		[["--max-move", "0.15"], []],
		[[], ["--max-move=0.15"]],
	]) {
		const [a, b] = rerun("robosuite", positional, first, second);
		assert.equal(a.status, 0, a.stdout + a.stderr);
		assert.equal(b.status, 1, `${first} then ${second}`);
		assert.match(b.stderr, /--max-move/);
	}
	// A result written before --max-move was recorded ran with the default cap.
	const [, same] = rerun("robosuite", positional, [], ["--max-move", "0.30"]);
	assert.equal(same.status, 0, same.stdout + same.stderr);
	assert.match(same.stdout, /\/max_move=0.3: success 1\/1/);
});

test("behavior/eval.sh records --grasping-mode (sticky without it) and never mixes modes in one out dir", () => {
	const positional = ["turning_on_radio", "0"];
	const run1 = (args: string[]) => run("behavior", positional, "turning_on_radio_s0", args);
	assert.equal(run1([]).result?.grasping_mode, "sticky");
	const on = run1(["--grasping-mode", "assisted"]);
	assert.equal(on.result?.grasping_mode, "assisted");
	assert.ok(on.argv?.includes("--grasping-mode") && on.argv?.includes("assisted"), "pi runs with the mode");
	assert.equal(run1(["--grasping-mode=assisted"]).result?.grasping_mode, "assisted");
	for (const [first, second] of [
		[["--grasping-mode", "assisted"], []],
		[[], ["--grasping-mode=assisted"]],
	]) {
		const [a, b] = rerun("behavior", positional, first, second);
		assert.equal(a.status, 0, a.stdout + a.stderr);
		assert.equal(b.status, 1, `${first} then ${second}`);
		assert.match(b.stderr, /--grasping-mode/);
	}
	const [, same] = rerun("behavior", positional, ["--grasping-mode", "sticky"], []);
	assert.equal(same.status, 0, same.stdout + same.stderr);
	assert.match(same.stdout, /\/grasp=sticky: success 1\/1/);
});
