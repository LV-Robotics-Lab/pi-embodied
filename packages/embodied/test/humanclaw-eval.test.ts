import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

const SCRIPT = new URL("../src/robots/humanclaw/eval.sh", import.meta.url).pathname;
const CELL = "sceneA_ep1_cup";

/**
 * eval.sh into `out` with a stand-in pi that records one successful episode; `env` adds to the
 * environment, `mode` picks --mode (paper runs the humanclaw-psv model).
 */
function run(out: string, args: string[], env: Record<string, string> = {}, mode: "pi" | "paper" = "pi") {
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
	const model = mode === "paper" ? "humanclaw-psv/p/m" : "p/m";
	return spawnSync("bash", [SCRIPT, out, "--episodes", CELL, "--mode", mode, "--model", model, ...args], {
		env: { ...process.env, PI: pi, ...env },
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

test("humanclaw/eval.sh run serially with LOCK takes it per episode (flock around pi); without LOCK it does not", () => {
	const bin = mkdtempSync(join(tmpdir(), "hc-bin-"));
	const log = join(bin, "flock.log");
	// A flock that logs its lock file and runs the command, as the real one does once it holds the lock.
	writeFileSync(join(bin, "flock"), `#!/usr/bin/env bash\necho "$1" >>"${log}"\nshift\nexec "$@"\n`);
	chmodSync(join(bin, "flock"), 0o755);
	const env = { PATH: `${bin}:${process.env.PATH}` };
	const locked = mkdtempSync(join(tmpdir(), "hc-out-"));
	const lock = join(bin, "gpu1.lock");
	const r = run(locked, [], { ...env, LOCK: lock });
	assert.equal(r.status, 0, r.stderr);
	assert.deepEqual(readFileSync(log, "utf8").trim().split("\n"), [lock]);
	assert.equal(JSON.parse(readFileSync(join(locked, CELL, "result.json"), "utf8")).status, "success");
	const plain = mkdtempSync(join(tmpdir(), "hc-out-"));
	const r2 = run(plain, [], { ...env, LOCK: "" });
	assert.equal(r2.status, 0, r2.stderr);
	assert.deepEqual(readFileSync(log, "utf8").trim().split("\n"), [lock], "no LOCK: no flock");
});

test("humanclaw/eval.sh keys paper mode's request parameters: both reasoning forms and --humanclaw-max-tokens", () => {
	const out = mkdtempSync(join(tmpdir(), "hc-out-"));
	const r = run(out, ["--humanclaw-reasoning=low", "--humanclaw-max-tokens", "1200"], {}, "paper");
	assert.equal(r.status, 0, r.stderr);
	assert.match(
		config(out),
		/^mode=paper\/preset=humanclaw\/model=humanclaw-psv\/p\/m\/.*\/reasoning=low\/request_tokens=1200$/,
	);
	// The same request contract in the other spelling is the same configuration; dropping --humanclaw-max-tokens is another.
	assert.equal(run(out, ["--humanclaw-reasoning", "low", "--humanclaw-max-tokens=1200"], {}, "paper").status, 0);
	const mixed = run(out, ["--humanclaw-reasoning", "low"], {}, "paper");
	assert.equal(mixed.status, 1);
	assert.match(mixed.stderr, /holds a result of another configuration/);
	// pi mode plans with the model's own tools: paper mode's request parameters are refused, not silently ignored.
	const pi = run(mkdtempSync(join(tmpdir(), "hc-out-")), ["--humanclaw-max-tokens", "1200"]);
	assert.equal(pi.status, 2);
	assert.match(pi.stderr, /paper mode's request parameters/);
	// The JSON response format is a boolean pi cannot turn off: the flag is refused rather than keyed as a no-op.
	const json = run(mkdtempSync(join(tmpdir(), "hc-out-")), ["--humanclaw-json-format=false"], {}, "paper");
	assert.equal(json.status, 2);
	assert.match(json.stderr, /cannot be turned off/);
});

test("humanclaw/eval.sh keys pi mode's planner flags as eval-options.sh parses them", () => {
	const out = mkdtempSync(join(tmpdir(), "hc-out-"));
	const r = run(out, [
		"--thinking",
		"low",
		"--max-turns=40",
		"--time-limit",
		"900",
		"--units-plugins",
		"plan",
		"--units-stage-steps=3",
		"--anchor-image",
		"--vdm",
		"--vdm-model=v/m",
		"--fallback-model",
		"f/m",
		"--fallback-after=3",
	]);
	assert.equal(r.status, 0, r.stderr);
	assert.match(
		config(out),
		/\/vdm=true\/.*\/thinking=low\/turns=40\/limit=900\/units=both\+plugins=plan\+stage-steps=3\/anchor\/vdm_model=v\/m\/fallback=f\/m:3:0$/,
	);
	const other = run(out, ["--thinking", "high"]);
	assert.equal(other.status, 1);
	assert.match(other.stderr, /holds a result of another configuration/);
	// pi's boolean flags are on whatever value they are given: eval-options.sh refuses the misleading forms.
	const bad = run(mkdtempSync(join(tmpdir(), "hc-out-")), ["--anchor-image=false"]);
	assert.equal(bad.status, 2);
});

test("humanclaw/eval.sh takes --humanclaw-max-steps only as a --smoke, and keys it", () => {
	const out = mkdtempSync(join(tmpdir(), "hc-out-"));
	const refused = run(out, ["--humanclaw-max-steps", "5"]);
	assert.equal(refused.status, 2);
	assert.match(refused.stderr, /--smoke/);
	const smoke = run(out, ["--smoke", "--humanclaw-max-steps=5"]);
	assert.equal(smoke.status, 0, smoke.stderr);
	assert.match(config(out), /\/max_steps=5$/);
	const argv = JSON.parse(readFileSync(join(out, CELL, "result.json"), "utf8"));
	assert.equal(argv.status, "success");
	// A full run does not share the smoke's out dir.
	assert.equal(run(out, []).status, 1);
});

test("humanclaw/eval.sh keys --humanclaw-collision-feedback (pi mode, with --metrics only)", () => {
	const out = mkdtempSync(join(tmpdir(), "hc-out-"));
	// --metrics ends with the humanclaw venv's aggregate_metric_files(): a no-op python here.
	const py = { HUMANCLAW_PYTHON: "true" };
	const r = run(out, ["--metrics", "--humanclaw-collision-feedback"], py);
	assert.equal(r.status, 0, r.stderr);
	assert.match(config(out), /\/metrics=true\/.*\/collision_feedback=true$/);
	const plain = run(out, ["--metrics"], py);
	assert.equal(plain.status, 1);
	assert.match(plain.stderr, /holds a result of another configuration/);
	assert.equal(
		run(mkdtempSync(join(tmpdir(), "hc-out-")), ["--humanclaw-collision-feedback"]).status,
		2,
		"needs --metrics",
	);
	const paper = run(
		mkdtempSync(join(tmpdir(), "hc-out-")),
		["--metrics", "--humanclaw-collision-feedback"],
		py,
		"paper",
	);
	assert.equal(paper.status, 2);
});
