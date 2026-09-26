import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
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
	["metaworld", ["reach-v3", "0"], "reach-v3_s0"],
	["robolab", ["BananaInBowlTask", "0"], "BananaInBowlTask_s0"],
	["libero", ["libero_10_task", "0", "0"], "libero_10_task_t0_s0"],
	["robosuite", ["Lift", "0"], "Lift_s0"],
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

for (const [robot, positional, cell, env] of CELLS) {
	test(`${robot}/eval.sh records --vdm, --vdm-model and --vdm-wrist and refuses a value pi would ignore`, () => {
		const run1 = (args: string[]) => run(robot, positional, cell, args, env);
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
}

test("libero/eval.sh records --unit-tol and drops --vdm-model when --vdm is off", () => {
	const run1 = (args: string[]) => run("libero", ["libero_10_task", "0", "0"], "libero_10_task_t0_s0", args);
	// Without --vdm no VDM call runs: the model is not part of the configuration.
	assert.equal(run1(["--vdm-model", "selfhost/muse"]).result?.vdm_model, null);
	// --unit-tol is recorded (the robot's default without it).
	assert.equal(run1([]).result?.unit_tol, 0.004);
	assert.equal(run1(["--unit-tol", "0.002"]).result?.unit_tol, 0.002);
	assert.equal(run1(["--unit-tol=0.006"]).result?.unit_tol, 0.006);
});

test("robotwin/eval.sh records the env server's planner and never mixes results of another one in one out dir", () => {
	const [, positional, cell] = CELLS.find(([r]) => r === "robotwin") as (typeof CELLS)[number];
	const contract = readFileSync(
		new URL("../../../services/pi_embodied_services/robots/robotwin/contract.py", import.meta.url),
		"utf8",
	);
	const planner = /^ROBOTWIN_PLANNER = "([^"]+)"$/m.exec(contract)?.[1];
	assert.ok(planner);
	assert.equal(run("robotwin", positional, cell, []).result?.planner, planner);
	// A result of the earlier planner (no `planner`, or another one) is another configuration.
	for (const old of [{}, { planner: "curobo_lbfgs=cuda" }]) {
		const dir = mkdtempSync(join(tmpdir(), "eval-"));
		mkdirSync(join(dir, "out", cell), { recursive: true });
		writeFileSync(
			join(dir, "out", cell, "result.json"),
			JSON.stringify({
				status: "success",
				model: null,
				thinking: null,
				max_turns: 0,
				time_limit: 0,
				units: "false",
				stateless: false,
				...old,
			}),
		);
		const script = new URL("../src/robotwin/eval.sh", import.meta.url).pathname;
		const r = spawnSync("bash", [script, join(dir, "out"), ...positional], {
			env: { ...process.env, PI: "false", TIME_LIMIT: "0" },
			encoding: "utf8",
		});
		assert.equal(r.status, 1);
		assert.match(r.stderr, /planner \(an older env server\), .*; use another out dir/);
	}
});

test("libero/eval.sh records --code and --code-api as pi runs them and never mixes code mode with tool runs", () => {
	const run1 = (args: string[]) => run("libero", ["libero_10_task", "0", "0"], "libero_10_task_t0_s0", args);
	const plain = run1([]).result;
	assert.deepEqual([plain?.code, plain?.code_api], ["false", null]);
	for (const [args, code, api] of [
		[["--code"], "true", "high"],
		[["--code=true"], "true", "high"],
		[["--code", "both"], "both", "high"],
		[["--code=pure"], "true", "high"],
		[["--code", "--code-api", "low"], "true", "low"],
		[["--code=both", "--code-api=low"], "both", "low"],
		[["--code=false", "--code-api", "low"], "false", null],
	] as const) {
		const r = run1([...args]);
		assert.deepEqual([r.result?.code, r.result?.code_api], [code, api], args.join(" "));
		// The prompt precedes the user's args, so a trailing bare --code cannot swallow it.
		const prompt = r.argv?.findIndex((a) => a.startsWith("Solve the task.")) ?? -1;
		assert.ok(prompt >= 0 && prompt < (r.argv?.indexOf(args[0]) ?? -1), String(r.argv));
	}
	const positional = ["libero_10_task", "0", "0"];
	for (const [first, second] of [
		[["--code"], []],
		[[], ["--code=both"]],
		[
			["--code", "--code-api", "low"],
			["--code", "--code-api", "high"],
		],
	]) {
		const [a, b] = rerun("libero", positional, {}, first, second);
		assert.equal(a.status, 0, a.stderr);
		assert.equal(b.status, 1, `${first} then ${second}`);
		assert.match(b.stderr, /code mode/);
	}
	const [, same] = rerun("libero", positional, {}, ["--code", "--code-api", "low"], ["--code", "--code-api", "low"]);
	assert.equal(same.status, 0, same.stderr);
	assert.match(same.stdout, /\/code=true:low/);
});

test("robolab/eval.sh records --instruction-type and --subtask and never mixes them in one out dir", () => {
	const positional = ["BananaInBowlTask", "0"];
	const run1 = (args: string[]) => run("robolab", positional, "BananaInBowlTask_s0", args);
	for (const args of [["--subtask=false"], ["--subtask", "false"]]) {
		const r = run1(args);
		assert.equal(r.status, 2, args.join(" "));
		assert.match(r.stderr, /subtask/);
		assert.equal(r.argv, undefined, "pi never ran");
	}
	const plain = run1([]).result;
	assert.equal(plain?.instruction_type, "default");
	assert.equal(plain?.subtask, false);
	const on = run1(["--instruction-type", "vague", "--subtask"]);
	assert.equal(on.result?.instruction_type, "vague");
	assert.equal(on.result?.subtask, true);
	assert.ok(on.argv?.includes("--subtask") && on.argv?.includes("vague"), String(on.argv));
	assert.equal(run1(["--instruction-type=specific"]).result?.instruction_type, "specific");
	// A phrasing is a different task; a subtask run a different configuration.
	for (const [first, second] of [
		[["--instruction-type", "vague"], []],
		[[], ["--instruction-type=specific"]],
		[[], ["--subtask"]],
	]) {
		const [a, b] = rerun("robolab", positional, {}, first, second);
		assert.equal(a.status, 0, a.stdout + a.stderr);
		assert.equal(b.status, 1);
		assert.match(b.stderr, /--instruction-type or --subtask \(or an older result without them\)/);
	}
	const [, same] = rerun(
		"robolab",
		positional,
		{},
		["--instruction-type", "vague", "--subtask"],
		["--instruction-type=vague", "--subtask"],
	);
	assert.equal(same.status, 0, same.stdout + same.stderr);
	assert.match(same.stdout, /\/vague\/subtask: success 1\/1/);
});

/** eval.sh twice into one out dir, with a stand-in pi that records one successful episode: `first` args, then `second`. */
function rerun(
	robot: string,
	positional: string[],
	env: Record<string, string>,
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
			env: { ...process.env, PI: pi, TIME_LIMIT: "0", ...env },
			encoding: "utf8",
		});
	return [once(first), once(second)];
}

for (const [robot, positional, cell, env] of CELLS) {
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

// The five scripts that record the fallback planner (src/fallback.ts); newer robots' scripts are their own.
const FALLBACK_ROBOTS = ["libero", "robocasa", "robotwin", "maniskill", "robolab"];
for (const [robot, positional, cell, env] of CELLS.filter(([r]) => FALLBACK_ROBOTS.includes(r))) {
	test(`${robot}/eval.sh records the fallback planner, never mixes it with runs without it, and totals planner_models`, () => {
		const run1 = (args: string[]) => run(robot, positional, cell, args, env);
		const plain = run1([]).result;
		assert.deepEqual(
			[plain?.fallback_model, plain?.fallback_after, plain?.fallback_retry_primary],
			[null, null, null],
		);
		const on = run1([
			"--fallback-model",
			"selfhost/muse",
			"--fallback-after",
			"3",
			"--fallback-retry-primary=5",
		]).result;
		assert.deepEqual([on?.fallback_model, on?.fallback_after, on?.fallback_retry_primary], ["selfhost/muse", 3, 5]);
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
			const [a, b] = rerun(robot, positional, env ?? {}, first, second);
			assert.equal(a.status, 0, a.stdout + a.stderr);
			assert.equal(b.status, 1, `${first} then ${second}`);
			assert.match(b.stderr, /fallback/);
		}
		const planned = { planner_models: { primary: 1, fallback: 2 } };
		const [, same] = rerun(
			robot,
			positional,
			env ?? {},
			["--fallback-model", "selfhost/muse"],
			["--fallback-model=selfhost/muse"],
			planned,
		);
		assert.equal(same.status, 0, same.stdout + same.stderr);
		assert.match(same.stdout, /\/fallback=selfhost\/muse:2:0/);
		// Summed over the cells the script ran (robotwin runs every seed of the task).
		const totals = /planner_models primary=(\d+) fallback=(\d+)/.exec(same.stdout);
		assert.ok(totals && Number(totals[1]) >= 1 && Number(totals[2]) === 2 * Number(totals[1]), same.stdout);
	});
}
