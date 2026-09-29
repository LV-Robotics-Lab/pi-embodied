import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

const SCRIPT = new URL("../src/robots/humanclaw/eval.sh", import.meta.url).pathname;
const CELL = "sceneA_ep1_cup";

/** eval.sh into `out` with a stand-in pi that records one successful episode. */
function run(out: string, args: string[]) {
	const dir = mkdtempSync(join(tmpdir(), "hc-eval-"));
	const pi = join(dir, "pi");
	const entry = JSON.stringify({
		type: "custom",
		customType: "robot_result",
		data: { robot: "humanclaw", success: true, env_error: false, planner_error: null },
	});
	writeFileSync(
		pi,
		`#!/usr/bin/env bash\nwhile [ $# -gt 0 ]; do [ "$1" = --session-dir ] && dir=$2; shift; done\necho '${entry}' > "$dir/s.jsonl"\n`,
	);
	chmodSync(pi, 0o755);
	return spawnSync("bash", [SCRIPT, out, "--episodes", CELL, "--mode", "pi", "--model", "p/m", ...args], {
		env: { ...process.env, PI: pi },
		encoding: "utf8",
	});
}
const config = (out: string) => JSON.parse(readFileSync(join(out, CELL, "result.json"), "utf8")).config as string;

test("humanclaw/eval.sh keys --approval, --max-tool-calls and --max-tokens only when set", () => {
	const plain = mkdtempSync(join(tmpdir(), "hc-out-"));
	assert.equal(run(plain, []).status, 0);
	assert.doesNotMatch(config(plain), /approval|tool_calls|tokens/, "the key of an out dir from before is unchanged");
	const reviewed = mkdtempSync(join(tmpdir(), "hc-out-"));
	const r = run(reviewed, ["--approval", "reviewed", "--max-tool-calls=40", "--max-tokens", "900000"]);
	assert.equal(r.status, 0, r.stderr);
	assert.match(config(reviewed), /\/approval=reviewed\/tool_calls=40\/tokens=900000$/);
	// Another approval mode in the same out dir is another configuration.
	const mixed = run(reviewed, []);
	assert.equal(mixed.status, 1);
	assert.match(mixed.stderr, /holds a result of another configuration/);
	assert.equal(run(reviewed, ["--approval=reviewed", "--max-tool-calls", "40", "--max-tokens=900000"]).status, 0);
});
