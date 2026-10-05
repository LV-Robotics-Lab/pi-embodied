import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

const SCRIPT = new URL("../src/robots/humanclaw/eval.sh", import.meta.url).pathname;
const CELL = "sceneA_ep1_cup";
type Options = {
	/** --mode (paper runs the humanclaw-psv model). */
	mode?: "pi" | "paper";
	/** Pass --metrics (default): the stand-in's episode then has a verdict. */
	metrics?: boolean;
	env?: Record<string, string>;
};

/**
 * eval.sh into `out` with a stand-in pi that records one episode: `success: true` when it was started
 * with --humanclaw-metrics (the robot's NavSR verdict comes from metrics.json), else `success: null`
 * as the robot records an unmeasured episode. --metrics ends with the humanclaw venv's
 * aggregate_metric_files(): HUMANCLAW_PYTHON is a no-op here.
 */
function run(out: string, args: string[], o: Options = {}) {
	const dir = mkdtempSync(join(tmpdir(), "hc-eval-"));
	const pi = join(dir, "pi");
	const entry = (success: string) =>
		`{"type":"custom","customType":"robot_result","data":{"robot":"humanclaw","success":${success},"env_error":false,"planner_error":null}}`;
	writeFileSync(
		pi,
		`#!/usr/bin/env bash\nok=null\nwhile [ $# -gt 0 ]; do [ "$1" = --session-dir ] && dir=$2; [ "$1" = --humanclaw-metrics ] && ok=true; shift; done\n` +
			`[ "$ok" = true ] && echo '${entry("true")}' > "$dir/s.jsonl" || echo '${entry("null")}' > "$dir/s.jsonl"\n`,
	);
	chmodSync(pi, 0o755);
	const mode = o.mode ?? "pi";
	const model = mode === "paper" ? "humanclaw-psv/p/m" : "p/m";
	const metrics = o.metrics === false ? [] : ["--metrics"];
	return spawnSync("bash", [SCRIPT, out, "--episodes", CELL, "--mode", mode, "--model", model, ...metrics, ...args], {
		env: { ...process.env, PI: pi, HUMANCLAW_PYTHON: "true", ...o.env },
		encoding: "utf8",
	});
}
const result = (out: string) => JSON.parse(readFileSync(join(out, CELL, "result.json"), "utf8"));
const config = (out: string) => result(out).config as string;
const fresh = () => mkdtempSync(join(tmpdir(), "hc-out-"));

test("humanclaw/eval.sh keys --approval, --max-tool-calls and --max-tokens only when set", () => {
	const plain = fresh();
	assert.equal(run(plain, []).status, 0);
	assert.doesNotMatch(config(plain), /approval|tool_calls|tokens/, "the key of an out dir from before is unchanged");
	const reviewed = fresh();
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
	const locked = fresh();
	const lock = join(bin, "gpu1.lock");
	const r = run(locked, [], { env: { ...env, LOCK: lock } });
	assert.equal(r.status, 0, r.stderr);
	assert.deepEqual(readFileSync(log, "utf8").trim().split("\n"), [lock]);
	assert.equal(result(locked).status, "success");
	const r2 = run(fresh(), [], { env: { ...env, LOCK: "" } });
	assert.equal(r2.status, 0, r2.stderr);
	assert.deepEqual(readFileSync(log, "utf8").trim().split("\n"), [lock], "no LOCK: no flock");
});

test("humanclaw/eval.sh keys paper mode's request parameters: both reasoning forms and --humanclaw-max-tokens", () => {
	const out = fresh();
	const paper = { mode: "paper" as const };
	const r = run(out, ["--humanclaw-reasoning=low", "--humanclaw-max-tokens", "1200"], paper);
	assert.equal(r.status, 0, r.stderr);
	assert.match(
		config(out),
		/^mode=paper\/preset=humanclaw\/model=humanclaw-psv\/p\/m\/.*\/reasoning=low\/request_tokens=1200$/,
	);
	// The same request contract in the other spelling is the same configuration; dropping max-tokens is another.
	assert.equal(run(out, ["--humanclaw-reasoning", "low", "--humanclaw-max-tokens=1200"], paper).status, 0);
	const mixed = run(out, ["--humanclaw-reasoning", "low"], paper);
	assert.equal(mixed.status, 1);
	assert.match(mixed.stderr, /holds a result of another configuration/);
	// pi mode plans with the model's own tools: paper mode's request parameters are refused, not silently ignored.
	const pi = run(fresh(), ["--humanclaw-max-tokens", "1200"]);
	assert.equal(pi.status, 2);
	assert.match(pi.stderr, /paper mode's request parameters/);
	// The JSON response format is a boolean pi cannot turn off: the flag is refused rather than keyed as a no-op.
	const json = run(fresh(), ["--humanclaw-json-format=false"], paper);
	assert.equal(json.status, 2);
	assert.match(json.stderr, /cannot be turned off/);
});

test("humanclaw/eval.sh keys pi mode's planner flags as eval-options.sh parses them", () => {
	const out = fresh();
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
		"--aux-model=v/m",
		"--fallback-model",
		"f/m",
		"--fallback-after=3",
	]);
	assert.equal(r.status, 0, r.stderr);
	assert.match(
		config(out),
		/\/vdm=true\/.*\/thinking=low\/turns=40\/limit=900\/units=both\+plugins=plan\+stage-steps=3\/anchor\/aux_model=v\/m\/fallback=f\/m:3:0$/,
	);
	const other = run(out, ["--thinking", "high"]);
	assert.equal(other.status, 1);
	assert.match(other.stderr, /holds a result of another configuration/);
	// pi's boolean flags are on whatever value they are given: eval-options.sh refuses the misleading forms.
	assert.equal(run(fresh(), ["--anchor-image=false"]).status, 2);
});

test("humanclaw/eval.sh takes --humanclaw-max-steps only as a --smoke, and keys it", () => {
	const out = fresh();
	const refused = run(out, ["--humanclaw-max-steps", "5"]);
	assert.equal(refused.status, 2);
	assert.match(refused.stderr, /--smoke/);
	const smoke = run(out, ["--smoke", "--humanclaw-max-steps=5"]);
	assert.equal(smoke.status, 0, smoke.stderr);
	assert.match(config(out), /\/max_steps=5$/);
	assert.equal(result(out).status, "success");
	// A full run does not share the smoke's out dir.
	assert.equal(run(out, []).status, 1);
});

test("humanclaw/eval.sh keys --humanclaw-collision-feedback (pi mode, with --metrics only)", () => {
	const out = fresh();
	const r = run(out, ["--humanclaw-collision-feedback"]);
	assert.equal(r.status, 0, r.stderr);
	assert.match(config(out), /\/metrics=true\/.*\/collision_feedback=true$/);
	const plain = run(out, []);
	assert.equal(plain.status, 1);
	assert.match(plain.stderr, /holds a result of another configuration/);
	assert.equal(run(fresh(), ["--humanclaw-collision-feedback"], { metrics: false }).status, 2, "needs --metrics");
	assert.equal(run(fresh(), ["--humanclaw-collision-feedback"], { mode: "paper" }).status, 2);
});

test("humanclaw/eval.sh without --metrics records every episode unscored, never failure, and does not rerun it", () => {
	const out = fresh();
	const r = run(out, [], { metrics: false });
	assert.equal(r.status, 1, "an unscored run is not a scored run");
	assert.match(r.stderr, /without --metrics the benchmark measures nothing/);
	assert.equal(result(out).status, "unscored");
	assert.equal(result(out).success, null);
	assert.match(r.stdout, /scored 0\/1 \(success 0\), unscored 1, invalid 0/);
	assert.match(r.stdout, /score them in another out dir with --metrics/);
	// The cell is kept as it is: the same configuration again runs no pi (its rollout artifacts stay).
	const again = run(out, [], { metrics: false });
	assert.doesNotMatch(again.stdout, /== sceneA_ep1_cup/);
	assert.equal(result(out).status, "unscored");
	// Scoring needs --metrics, which is another configuration: another out dir.
	const scored = run(out, []);
	assert.equal(scored.status, 1);
	assert.match(scored.stderr, /holds a result of another configuration/);
	const measured = fresh();
	assert.equal(run(measured, []).status, 0);
	assert.equal(result(measured).status, "success");
	assert.match(run(measured, []).stdout, /scored 1\/1 \(success 1\), unscored 0, invalid 0/);
});

test("humanclaw/eval.sh keys the OpenETA extras a run turns on, as eval-options.sh collects them", () => {
	const out = fresh();
	const r = run(out, ["--web-tools", "--object-memory=true"]);
	assert.equal(r.status, 0, r.stderr);
	assert.match(config(out), /\/extras=object-memory,web-tools$/);
	const other = run(out, ["--web-tools"]);
	assert.equal(other.status, 1);
	assert.match(other.stderr, /holds a result of another configuration/);
	assert.equal(run(fresh(), ["--object-memory=false"]).status, 2, "pi would turn it on: refused");
});
