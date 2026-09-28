/**
 * eval.sh keys code mode's budget (--code-timeout / -max-calls / -max-move / -helpers, and an
 * oracle's relaxed caps) into the configuration: results of another budget are never mixed.
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

/** Run eval.sh twice into one out dir with a fake pi that records `flags` as the robot's result. */
function twice(robot: string, positional: string[], flags: string, first: string[], second: string[]) {
	const dir = mkdtempSync(join(tmpdir(), "eval-budget-"));
	const pi = join(dir, "pi");
	const entry = JSON.stringify({
		type: "custom",
		customType: "robot_result",
		data: { robot, terminated: true, success: true, env_error: false, planner_error: null, code_budget_flags: flags },
	});
	writeFileSync(
		pi,
		`#!/usr/bin/env bash\nwhile [ $# -gt 0 ]; do [ "$1" = --session-dir ] && dir=$2; shift; done\necho '${entry}' > "$dir/s.jsonl"\n`,
	);
	chmodSync(pi, 0o755);
	const script = new URL(`../src/robots/${robot}/eval.sh`, import.meta.url).pathname;
	const once = (args: string[]) =>
		spawnSync("bash", [script, join(dir, "out"), ...positional, ...args], {
			env: { ...process.env, PI: pi, TIME_LIMIT: "0" },
			encoding: "utf8",
		});
	return [once(first), once(second)];
}

for (const [robot, positional] of [
	["libero", ["libero_10_task", "0", "0"]],
	["maniskill", ["PickCube-v1", "0"]],
] as const) {
	test(`${robot}/eval.sh never mixes code budgets in one out dir`, () => {
		const flags = "timeout=+max_calls=7+max_move=+helpers=false";
		const base = ["--code", "--code-max-calls", "7"];
		const [a, same] = twice(robot, [...positional], flags, base, ["--code", "--code-max-calls=7"]);
		assert.equal(a.status, 0, a.stderr);
		assert.equal(same.status, 0, "the same budget keeps its result");
		for (const other of [
			["--code"],
			["--code", "--code-max-calls", "8"],
			[...base, "--code-helpers"],
			[...base, "--code-max-move", "5"],
			[...base, "--code-timeout=120"],
		]) {
			const [, b] = twice(robot, [...positional], flags, base, other);
			assert.equal(b.status, 1, `${base.join(" ")} then ${other.join(" ")}: ${b.stderr}`);
		}
	});
}
