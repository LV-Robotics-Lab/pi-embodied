import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";

const here = (p: string) => new URL(p, import.meta.url).pathname;

/**
 * fixed-regression.sh in a stand-in checkout: the real script and old-flags.sh at their paths, a
 * `git` that reports a clean tree, a built cli.js, and each robot's eval.sh replaced by a stub that
 * refuses the old flags like the real ones do, then records its arguments and the memory it was given.
 */
function checkout() {
	const root = mkdtempSync(join(tmpdir(), "fixed-regression-"));
	const scripts = join(root, "packages/embodied/src/scripts");
	mkdirSync(scripts, { recursive: true });
	copyFileSync(here("../src/scripts/fixed-regression.sh"), join(scripts, "fixed-regression.sh"));
	copyFileSync(here("../src/scripts/old-flags.sh"), join(scripts, "old-flags.sh"));
	for (const robot of ["metaworld", "maniskill", "libero"]) {
		const eval_ = join(root, `packages/embodied/src/robots/${robot}/eval.sh`);
		mkdirSync(dirname(eval_), { recursive: true });
		writeFileSync(
			eval_,
			[
				"#!/usr/bin/env bash",
				"set -euo pipefail",
				'here=$(cd "$(dirname "$0")" && pwd)',
				'. "$here/../../scripts/old-flags.sh"',
				'old_flags "$@" || exit 2',
				'out=$1; shift; mkdir -p "$out"',
				`printf '%s\\n' "$@" > "$out/argv"`,
				`printf '%s\\n' "\${PI_EMBODIED_DIRS_MEMORY-unset}" > "$out/memory"`,
				`[ -f "\${PI_EMBODIED_DIRS_MEMORY-/nonexistent}/MEMORY.md" ]`,
				"",
			].join("\n"),
		);
		chmodSync(eval_, 0o755);
	}
	mkdirSync(join(root, "packages/coding-agent/dist"), { recursive: true });
	writeFileSync(join(root, "packages/coding-agent/dist/cli.js"), "");
	const bin = join(root, "bin");
	mkdirSync(bin);
	writeFileSync(
		join(bin, "git"),
		'#!/usr/bin/env bash\ncase " $* " in *" status "*) exit 0 ;; *" rev-parse "*) echo 0123abcd ;; esac\n',
	);
	chmodSync(join(bin, "git"), 0o755);
	const env = join(root, "env");
	mkdirSync(env);
	for (const robot of ["metaworld", "maniskill", "libero"])
		writeFileSync(join(env, `${robot}-env.sh`), "# host configuration\n");
	return { root, scripts, bin, env };
}

test("fixed-regression.sh runs its six cells with flags the eval scripts accept, and a fresh memory per cell through dirs.memory", () => {
	const c = checkout();
	const out = join(c.root, "results");
	const r = spawnSync("bash", [join(c.scripts, "fixed-regression.sh"), out, c.env], {
		env: { ...process.env, PATH: `${c.bin}:${process.env.PATH}` },
		encoding: "utf8",
	});
	assert.equal(r.status, 0, r.stderr);
	const status = readFileSync(join(out, "status.txt"), "utf8").trimEnd().split("\n");
	assert.deepEqual(status, [
		"metaworld 0 0",
		"metaworld 1 0",
		"maniskill 0 0",
		"maniskill 1 0",
		"libero 0 0",
		"libero 1 0",
	]);
	for (const [robot, cells] of [
		["metaworld", ["reach-v3"]],
		["maniskill", ["PickCube-v1"]],
		["libero", ["libero_spatial", "0"]],
	] as const) {
		for (const seed of ["0", "1"]) {
			const cell = join(out, `${robot}-${seed}`);
			const argv = readFileSync(join(cell, "argv"), "utf8").trimEnd().split("\n");
			assert.deepEqual(argv.slice(0, cells.length + 1), [...cells, seed], `${robot} ${seed}: the cell`);
			assert.ok(!argv.some((a) => a.startsWith("--memory-dir")), `${robot} ${seed}: --memory-dir is gone`);
			for (const flag of ["--units=true", "--code=false", "--thinking", "--memory-profile"])
				assert.ok(argv.includes(flag), flag);
			assert.equal(argv[argv.indexOf("--model") + 1], "relay/gpt-6-astra", "the default model");
			const memory = join(out, `memory-${robot}-${seed}`);
			assert.equal(readFileSync(join(cell, "memory"), "utf8").trim(), memory);
			assert.match(readFileSync(join(memory, "MEMORY.md"), "utf8"), /No stored task recipes/);
		}
	}
	assert.equal(readFileSync(join(out, "commit.txt"), "utf8"), readFileSync(join(out, "commit-after.txt"), "utf8"));
	assert.equal(readFileSync(join(out, "model.txt"), "utf8").trim(), "relay/gpt-6-astra");
	assert.ok(existsSync(join(out, "completed.txt")));
	// The stand-in eval.sh refuses the old flag like the real ones: the test would catch its return.
	const old = spawnSync(
		"bash",
		[
			join(c.root, "packages/embodied/src/robots/metaworld/eval.sh"),
			join(c.root, "x"),
			"reach-v3",
			"0",
			"--memory-dir",
			"m",
		],
		{
			encoding: "utf8",
		},
	);
	assert.equal(old.status, 2);
	assert.match(old.stderr, /--memory-dir is gone/);
});

test("fixed-regression.sh takes the model as its third argument", () => {
	const c = checkout();
	const out = join(c.root, "results");
	const r = spawnSync("bash", [join(c.scripts, "fixed-regression.sh"), out, c.env, "selfhost/qwen3.8-27b"], {
		env: { ...process.env, PATH: `${c.bin}:${process.env.PATH}` },
		encoding: "utf8",
	});
	assert.equal(r.status, 0, r.stderr);
	const argv = readFileSync(join(out, "libero-1/argv"), "utf8").trimEnd().split("\n");
	assert.equal(argv[argv.indexOf("--model") + 1], "selfhost/qwen3.8-27b");
	assert.equal(readFileSync(join(out, "model.txt"), "utf8").trim(), "selfhost/qwen3.8-27b");
});
