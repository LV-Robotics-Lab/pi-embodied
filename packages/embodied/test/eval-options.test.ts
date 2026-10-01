import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

const helper = new URL("../src/scripts/eval-options.sh", import.meta.url).pathname;

function parse(args: string[], defaults = "") {
	const dir = mkdtempSync(join(tmpdir(), "eval-options-"));
	try {
		const harness = join(dir, "parse.sh");
		writeFileSync(
			harness,
			`#!/usr/bin/env bash
set -uo pipefail
source "$1"
shift
eval_options_defaults ${defaults}
robot=panda
eval_robot_option() { case $1 in --robot) robot=$2 ;; --robot=*) robot=\${1#*=} ;; esac; }
eval_parse_options "$@"
eval_normalize_options
node -e 'console.log(JSON.stringify(process.argv.slice(1)))' "$model" "$turns" "$units" "$CODE_BUDGET_FLAGS" "$vdm_video" "$robot" "$@"
`,
		);
		return spawnSync("bash", [harness, helper, ...args], {
			encoding: "utf8",
			env: { ...process.env, units_opts: "+stale=true", units_plugins: "stale" },
		});
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
}

test("shared eval options preserve pi argv and robot options while normalizing equivalent spellings", () => {
	const variants = [
		[
			"--model",
			"provider/model",
			"--max-turns",
			"12",
			"--units",
			"pure",
			"--units-plugins",
			"",
			"--code-max-calls",
			"7",
			"--vdm-video",
			"--vdm-video-frames",
			"4",
			"--robot",
			"fetch",
		],
		[
			"--model=provider/model",
			"--max-turns=12",
			"--units=pure",
			"--units-plugins=",
			"--code-max-calls=7",
			"--vdm-video=true",
			"--vdm-video-frames=4",
			"--robot=fetch",
		],
	];
	for (const flags of variants) {
		const args = [...flags, "--future-option=value", "@task.md"];
		const result = parse(args);
		assert.equal(result.status, 0, result.stderr);
		assert.deepEqual(JSON.parse(result.stdout), [
			"provider/model",
			"12",
			"true+plugins=",
			"timeout=+max_calls=7+max_move=+helpers=false",
			"4",
			"fetch",
			...args,
		]);
	}
});

test("shared defaults exclude inherited experiment suffixes and preserve protocol turn defaults", () => {
	for (const [defaults, turns] of [
		["", "0"],
		["100", "100"],
	]) {
		const result = parse(["--units"], defaults);
		assert.equal(result.status, 0, result.stderr);
		const values = JSON.parse(result.stdout);
		assert.equal(values[1], turns);
		assert.equal(values[2], "true");
	}
});

test("explicit planner options override protocol defaults and repeated scalar flags use the last value", () => {
	const result = parse(["--max-turns=10", "--max-turns", "20", "--code-oracle=grasp"], "100");
	assert.equal(result.status, 0, result.stderr);
	const values = JSON.parse(result.stdout);
	assert.equal(values[1], "20");
	assert.equal(values[3], "timeout=+max_calls=+max_move=+helpers=false+oracle");
});

test("boolean values that pi would interpret differently are rejected before running a task", () => {
	for (const flag of ["stateless", "privileged", "anchor-image", "vdm", "vdm-wrist", "vdm-video"]) {
		for (const args of [[`--${flag}=false`], [`--${flag}`, "false"]]) {
			const result = parse(args);
			assert.equal(result.status, 2, `${args.join(" ")}: ${result.stderr}`);
			assert.equal(result.stdout, "");
		}
	}
});
