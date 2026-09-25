import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

/** Run `<robot>/eval.sh` for one cell with a stand-in pi that records its argv and fails. */
function run(robot: string, cell: string, args: string[]) {
	const dir = mkdtempSync(join(tmpdir(), "eval-"));
	const pi = join(dir, "pi");
	writeFileSync(pi, `#!/usr/bin/env bash\nprintf '%s\\n' "$@" > "$(dirname "$0")/argv"\nexit 1\n`);
	chmodSync(pi, 0o755);
	const script = new URL(`../src/${robot}/eval.sh`, import.meta.url).pathname;
	const r = spawnSync("bash", [script, join(dir, "out"), cell, "0", ...args], {
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
	const result = read(`out/${cell}_s0/result.json`);
	return {
		status: r.status,
		stderr: r.stderr,
		argv: read("argv")?.trimEnd().split("\n"),
		result: result && JSON.parse(result),
	};
}

for (const [robot, cell] of [
	["maniskill", "PickCube-v1"],
	["robolab", "BananaInBowlTask"],
]) {
	test(`${robot}/eval.sh records --stateless as pi runs it and refuses a value pi would ignore`, () => {
		// pi sets a boolean flag to true whatever its value: `--stateless=false` would run stateless.
		for (const args of [["--stateless=false"], ["--stateless", "false"], ["--stateless=0"]]) {
			const r = run(robot, cell, args);
			assert.equal(r.status, 2, args.join(" "));
			assert.match(r.stderr, /stateless/);
			assert.equal(r.argv, undefined, "pi never ran");
		}
		for (const args of [["--stateless"], ["--stateless=true"], ["--stateless", "--model", "m/x"]]) {
			const r = run(robot, cell, args);
			assert.equal(r.result?.stateless, true, args.join(" "));
			// The prompt precedes the user's args, so a trailing bare flag cannot swallow it.
			const prompt = r.argv?.indexOf("Solve the task.") ?? -1;
			assert.ok(prompt >= 0 && prompt < (r.argv?.indexOf(args[0]) ?? -1), String(r.argv));
		}
		assert.equal(run(robot, cell, []).result?.stateless, false);
	});
}
