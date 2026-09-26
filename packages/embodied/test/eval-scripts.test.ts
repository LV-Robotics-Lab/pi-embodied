import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

/** Run `<robot>/eval.sh <out> ...positional` with a stand-in pi that records its argv and fails; `cell` is the result's directory. */
function run(robot: string, positional: string[], cell: string, args: string[], env: Record<string, string> = {}) {
	const dir = mkdtempSync(join(tmpdir(), "eval-"));
	const pi = join(dir, "pi");
	writeFileSync(pi, `#!/usr/bin/env bash\nprintf '%s\\n' "$@" > "$(dirname "$0")/argv"\nexit 1\n`);
	chmodSync(pi, 0o755);
	const script = new URL(`../src/${robot}/eval.sh`, import.meta.url).pathname;
	const r = spawnSync("bash", [script, join(dir, "out"), ...positional, ...args], {
		env: { ...process.env, PI: pi, TIME_LIMIT: "0", ...env },
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

/** One cell per robot: its positional args, the result's directory, the environment narrowing the matrix. */
const CELLS: [string, string[], string, Record<string, string>?][] = [
	["maniskill", ["PickCube-v1", "0"], "PickCube-v1_s0"],
	["robolab", ["BananaInBowlTask", "0"], "BananaInBowlTask_s0"],
	["libero", ["libero_10_task", "0", "0"], "libero_10_task_t0_s0"],
];
const eval_ = (path: string) =>
	JSON.parse(readFileSync(new URL(`../../../services/pi_embodied_services/robots/${path}`, import.meta.url), "utf8"));
{
	// RoboTwin runs every seed of a task; RoboCasa's matrix is narrowed to one cell.
	const [task, [{ seed }]] = Object.entries(eval_("robotwin/eval/demo_randomized.json").tasks)[0] as [
		string,
		{ seed: number }[],
	];
	CELLS.push(["robotwin", [task], `${task}_s${seed}`]);
	const { tasks, seeds } = eval_("robocasa/eval/target50.json").splits.atomic as { tasks: string[]; seeds: number[] };
	CELLS.push([
		"robocasa",
		["atomic"],
		`atomic/${tasks[0]}_s${seeds[0]}`,
		{ TASKS: tasks[0], SEEDS: String(seeds[0]) },
	]);
}

for (const [robot, positional, cell, env] of CELLS) {
	const run1 = (args: string[]) => run(robot, positional, cell, args, env);
	test(`${robot}/eval.sh records --units and --stateless as pi runs them and refuses a value pi would ignore`, () => {
		// pi sets a boolean flag to true whatever its value: `--stateless=false` would run stateless.
		for (const args of [["--stateless=false"], ["--stateless", "false"], ["--stateless=0"]]) {
			const r = run1(args);
			assert.equal(r.status, 2, args.join(" "));
			assert.match(r.stderr, /stateless/);
			assert.equal(r.argv, undefined, "pi never ran");
		}
		for (const args of [["--stateless"], ["--stateless=true"], ["--stateless", "--model", "m/x"]]) {
			const r = run1(args);
			assert.equal(r.result?.stateless, true, args.join(" "));
			// The prompt precedes the user's args, so a trailing bare flag cannot swallow it.
			const prompt = r.argv?.findIndex((a) => a.startsWith("Solve the task.")) ?? -1;
			assert.ok(prompt >= 0 && prompt < (r.argv?.indexOf(args[0]) ?? -1), String(r.argv));
		}
		const plain = run1([]).result;
		assert.equal(plain?.stateless, false);
		assert.equal(plain?.units, "false");
		for (const [args, units] of [
			[["--units"], "true"],
			[["--units=true"], "true"],
			[["--units", "both"], "both"],
			[["--units=pure"], "true"],
		] as const)
			assert.equal(run1([...args]).result?.units, units, args.join(" "));
	});
}

test("libero/eval.sh records --vdm, --vdm-model and --vdm-wrist and refuses a value pi would ignore", () => {
	const run1 = (args: string[]) => run("libero", ["libero_10_task", "0", "0"], "libero_10_task_t0_s0", args);
	for (const args of [["--vdm=false"], ["--vdm", "false"], ["--vdm-wrist=0"]]) {
		const r = run1(args);
		assert.equal(r.status, 2, args.join(" "));
		assert.match(r.stderr, /vdm/);
		assert.equal(r.argv, undefined, "pi never ran");
	}
	const plain = run1([]).result;
	assert.deepEqual([plain?.vdm, plain?.vdm_model, plain?.vdm_wrist], [false, null, false]);
	const on = run1(["--vdm", "--vdm-model", "selfhost/muse", "--vdm-wrist"]).result;
	assert.deepEqual([on?.vdm, on?.vdm_model, on?.vdm_wrist], [true, "selfhost/muse", true]);
});

/** eval.sh twice into one out dir, with a stand-in pi that records one successful episode: `first` args, then `second`. */
function rerun(robot: string, positional: string[], env: Record<string, string>, first: string[], second: string[]) {
	const dir = mkdtempSync(join(tmpdir(), "eval-"));
	const pi = join(dir, "pi");
	const entry = JSON.stringify({
		type: "custom",
		customType: "robot_result",
		data: { robot, terminated: true, success: true, env_error: false, planner_error: null },
	});
	writeFileSync(
		pi,
		`#!/usr/bin/env bash\nwhile [ $# -gt 0 ]; do [ "$1" = --session-dir ] && dir=$2; shift; done\necho '${entry}' > "$dir/s.jsonl"\n`,
	);
	chmodSync(pi, 0o755);
	const script = new URL(`../src/${robot}/eval.sh`, import.meta.url).pathname;
	const once = (args: string[]) =>
		spawnSync("bash", [script, join(dir, "out"), ...positional, ...args], {
			env: { ...process.env, PI: pi, TIME_LIMIT: "0", ...env },
			encoding: "utf8",
		});
	return [once(first), once(second)];
}

for (const [robot, positional, cell, env] of CELLS.filter(([r]) => ["libero", "robocasa", "maniskill"].includes(r))) {
	test(`${robot}/eval.sh records --privileged and never mixes it with runs without it in one out dir`, () => {
		const run1 = (args: string[]) => run(robot, positional, cell, args, env);
		for (const args of [["--privileged=false"], ["--privileged", "false"]]) {
			const r = run1(args);
			assert.equal(r.status, 2, args.join(" "));
			assert.match(r.stderr, /privileged/);
			assert.equal(r.argv, undefined, "pi never ran");
		}
		assert.equal(run1([]).result?.privileged, false);
		const on = run1(["--privileged"]);
		assert.equal(on.result?.privileged, true);
		assert.ok(on.argv?.includes("--privileged"), "pi runs privileged");
		for (const [first, second] of [
			[["--privileged"], []],
			[[], ["--privileged"]],
		]) {
			const [a, b] = rerun(robot, positional, env ?? {}, first, second);
			assert.equal(a.status, 0, a.stdout + a.stderr);
			assert.equal(b.status, 1);
			assert.match(
				b.stderr,
				/--privileged or --anchor-image; use another out dir|--privileged or --anchor-image \(or an older/,
			);
		}
		// The same configuration keeps its valid result.
		const [, same] = rerun(robot, positional, env ?? {}, ["--privileged"], ["--privileged"]);
		assert.equal(same.status, 0, same.stdout + same.stderr);
		assert.match(same.stdout, /\/privileged/);
	});
}
