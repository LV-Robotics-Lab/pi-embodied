import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { envId, loadTable, nearMatches, resolveCell } from "../src/robocasa/tasks.ts";

const SERVICES = new URL("../../../services", import.meta.url).pathname;
const SCRIPT = new URL("../src/robocasa/eval.sh", import.meta.url).pathname;
const table = loadTable(SERVICES);

test("the RoboCasa365 table has 317 tasks in two splits with 50 distinct manifest scenes each", () => {
	assert.equal(table.tasks.length, 317);
	assert.deepEqual(table.splits, ["pretrain", "target"]);
	assert.equal(table.scenes_per_task, 50);
	const names = table.tasks.map((t) => t.name);
	assert.deepEqual(names, [...new Set(names)].sort());
	assert.equal(table.tasks.filter((t) => t.kind === "atomic").length, 65);
	assert.equal(table.tasks.filter((t) => t.target50).length, 50);
	for (const t of table.tasks) {
		assert.equal(new Set(t.manifest).size, 50, t.name);
		assert.ok(
			t.manifest.every((s) => Number.isInteger(s) && s >= 0 && s < 2 ** 31),
			t.name,
		);
		assert.ok(t.horizon > 0 && t.instruction.length > 0, t.name);
	}
	const ids = new Set(table.splits.flatMap((s) => names.map((n) => envId(n, s))));
	assert.equal(ids.size, 634);
	// The Target50 protocol's tasks are all in the table.
	const t50 = JSON.parse(
		readFileSync(join(SERVICES, "pi_embodied_services/robots/robocasa/eval/target50.json"), "utf8"),
	);
	for (const split of Object.values(t50.splits) as { tasks: string[] }[])
		for (const task of split.tasks) assert.ok(table.tasks.find((t) => t.name === task)?.target50, task);
});

test("resolveCell checks the task, split and scene flags and names near matches", () => {
	const open = table.tasks.find((t) => t.name === "OpenDrawer")!;
	assert.deepEqual(resolveCell(table, { task: "OpenDrawer", split: "target", scene: "" }), {
		task: open,
		split: "target",
		envId: "robocasa365/target/OpenDrawer",
	});
	// `all` has no manifest: seed mode only, no env id.
	assert.deepEqual(resolveCell(table, { task: "OpenDrawer", split: "all", scene: "" }), { task: open, split: "all" });
	assert.deepEqual(resolveCell(table, { task: "OpenDrawer", split: "pretrain", scene: "3" }), {
		task: open,
		split: "pretrain",
		scene: 3,
		seed: open.manifest[3],
		envId: "robocasa365/pretrain/OpenDrawer",
	});
	assert.throws(
		() => resolveCell(table, { task: "OpenDrawr", split: "target", scene: "" }),
		/not one of the 317 RoboCasa365 tasks; near matches: .*OpenDrawer/,
	);
	assert.throws(() => resolveCell(table, { task: "OpenDrawer", split: "test", scene: "" }), /--split must be one of/);
	assert.throws(() => resolveCell(table, { task: "OpenDrawer", split: "all", scene: "0" }), /--scene needs --split/);
	for (const scene of ["50", "-1", "x", "1.5"])
		assert.throws(() => resolveCell(table, { task: "OpenDrawer", split: "target", scene }), /manifest index 0\.\.49/);
	assert.deepEqual(nearMatches("drawer", ["OpenDrawer", "CloseDrawer", "OpenFridge"], 5), [
		"CloseDrawer",
		"OpenDrawer",
		"OpenFridge",
	]);
	assert.equal(
		nearMatches(
			"PrepareCofee",
			table.tasks.map((t) => t.name),
		)[0],
		"PrepareCoffee",
	);
});

/** eval.sh into `dir/out` with a stand-in pi that records its argv and writes `entry` as the episode's robot_result (or fails without one). */
function evalSh(dir: string, positional: string[], args: string[], env: Record<string, string>, entry?: object) {
	const pi = join(dir, "pi");
	const body = entry
		? `while [ $# -gt 0 ]; do [ "$1" = --session-dir ] && dir=$2; shift; done\necho '${JSON.stringify({ type: "custom", customType: "robot_result", data: entry })}' > "$dir/s.jsonl"\n`
		: `printf '%s\\n' "$@" > "$(dirname "$0")/argv"\nexit 1\n`;
	writeFileSync(pi, `#!/usr/bin/env bash\n${body}`);
	chmodSync(pi, 0o755);
	const r = spawnSync("bash", [SCRIPT, join(dir, "out"), ...positional, ...args], {
		env: { ...process.env, PI: pi, TIME_LIMIT: "0", ...env },
		encoding: "utf8",
	});
	const read = (p: string) => (existsSync(join(dir, p)) ? readFileSync(join(dir, p), "utf8") : undefined);
	return { status: r.status, stdout: r.stdout, stderr: r.stderr, argv: read("argv")?.trimEnd().split("\n"), read };
}
const ok = { robot: "robocasa", terminated: true, success: true, env_error: false, planner_error: null };
const t50 = JSON.parse(readFileSync(join(SERVICES, "pi_embodied_services/robots/robocasa/eval/target50.json"), "utf8"));
const T50_CELL = { TASKS: t50.splits.atomic.tasks[0], SEEDS: String(t50.splits.atomic.seeds[0]) };

test("robocasa/eval.sh manifest mode runs task x scene cells with --scene and writes robocasa365 results", () => {
	const dir = mkdtempSync(join(tmpdir(), "eval365-"));
	const r = evalSh(dir, ["pretrain"], ["--units", "--model", "m/x"], { TASKS: "OpenDrawer", SCENES: "7" });
	assert.ok(r.argv, r.stderr);
	const at = (flag: string) => r.argv?.[r.argv.indexOf(flag) + 1];
	assert.equal(at("--task-name"), "OpenDrawer");
	assert.equal(at("--split"), "pretrain");
	assert.equal(at("--scene"), "7");
	assert.ok(!r.argv?.includes("--seed"), "the scene's seed comes from the table");
	const result = JSON.parse(r.read("out/pretrain/OpenDrawer_m7/result.json") ?? "{}");
	const open = table.tasks.find((t) => t.name === "OpenDrawer")!;
	assert.equal(result.protocol, "robocasa365");
	assert.equal(result.protocol_id, undefined, "no Target50 field");
	assert.deepEqual(
		[result.split, result.task_name, result.scene, result.seed, result.env_id, result.horizon],
		["pretrain", "OpenDrawer", 7, open.manifest[7], "robocasa365/pretrain/OpenDrawer", open.horizon],
	);
	assert.deepEqual([result.units, result.stateless, result.privileged, result.model], ["true", false, false, "m/x"]);
	assert.equal(result.status, "env_error");
	// A bad TASKS or SCENES value never runs pi.
	const bads: Record<string, string>[] = [
		{ TASKS: "OpenDrawr" },
		{ TASKS: "OpenDrawer", SCENES: "50" },
		{ TASKS: "OpenDrawer", SCENES: "0" },
	];
	for (const env of bads) {
		const bad = evalSh(
			mkdtempSync(join(tmpdir(), "eval365-")),
			[env.SCENES === "0" ? "atomic_unseen" : "target"],
			[],
			env,
		);
		assert.equal(bad.status, 1, JSON.stringify(env));
		assert.equal(bad.argv, undefined);
	}
});

test("robocasa/eval.sh summarizes per split, keeps valid cells on rerun and refuses another configuration", () => {
	const dir = mkdtempSync(join(tmpdir(), "eval365-"));
	const env = { TASKS: "OpenDrawer,CloseDrawer", SCENES: "0-1" };
	const a = evalSh(dir, ["target"], ["--model", "m/x"], env, ok);
	assert.equal(a.status, 0, a.stdout + a.stderr);
	assert.match(a.stdout, /robocasa365 target: success 4\/4 \(100\.0%\) over 2 tasks, task-weighted 100\.0%/);
	const summary = JSON.parse(a.read("out/robocasa365-target-summary.json") ?? "{}");
	assert.deepEqual([summary.protocol, summary.split], ["robocasa365", "target"]);
	assert.deepEqual([summary.tasks, summary.cells, summary.successes, summary.success_rate], [2, 4, 4, 1]);
	assert.equal(a.read("out/robocasa365-pretrain-summary.json"), undefined);
	assert.ok(!existsSync(join(dir, "out/target50.pi.json")), "no Target50 derived manifest");
	// The same configuration keeps its results (pi is not run again); another one is refused.
	const same = evalSh(dir, ["target"], ["--model", "m/x"], env);
	assert.equal(same.status, 0, same.stderr);
	assert.equal(same.argv, undefined, "pi never ran");
	const other = evalSh(dir, ["target"], ["--model", "m/y"], env);
	assert.equal(other.status, 1);
	assert.match(other.stderr, /another model, thinking level/);
	// Both splits, each summarized on its own.
	const both = evalSh(
		mkdtempSync(join(tmpdir(), "eval365-")),
		["pretrain,target"],
		[],
		{ TASKS: "OpenDrawer", SCENES: "0" },
		ok,
	);
	assert.equal(both.status, 0, both.stdout + both.stderr);
	assert.match(both.stdout, /robocasa365 pretrain: success 1\/1/);
	assert.match(both.stdout, /robocasa365 target: success 1\/1/);
	for (const split of ["pretrain", "target"])
		assert.equal(JSON.parse(both.read(`out/robocasa365-${split}-summary.json`) ?? "{}").split, split);
	// A later run of one split into the same out dir leaves the other split's summary alone.
	const dir2 = mkdtempSync(join(tmpdir(), "eval365-"));
	assert.equal(evalSh(dir2, ["pretrain"], [], { TASKS: "OpenDrawer", SCENES: "0" }, ok).status, 0);
	const pre = evalSh(dir2, ["target"], [], { TASKS: "OpenDrawer", SCENES: "0" }, ok).read(
		"out/robocasa365-pretrain-summary.json",
	);
	assert.equal(JSON.parse(pre ?? "{}").successes, 1);
});

test("robocasa/eval.sh never mixes Target50 and robocasa365 results in one out dir", () => {
	// Target50 first, then a manifest run into the same dir: refused, and vice versa.
	const dir = mkdtempSync(join(tmpdir(), "eval365-"));
	const t = evalSh(dir, ["atomic"], [], T50_CELL, ok);
	assert.equal(t.status, 0, t.stdout + t.stderr);
	assert.equal(
		JSON.parse(t.read(`out/atomic/${T50_CELL.TASKS}_s${T50_CELL.SEEDS}/result.json`) ?? "{}").protocol,
		undefined,
	);
	const m = evalSh(dir, ["target"], [], { TASKS: "OpenDrawer", SCENES: "0" }, ok);
	assert.equal(m.status, 1);
	assert.match(m.stderr, /holds target50 results; a robocasa365 run needs another out dir/);
	assert.ok(!existsSync(join(dir, "out/target/OpenDrawer_m0")), "no manifest cell was run");

	const dir2 = mkdtempSync(join(tmpdir(), "eval365-"));
	const m2 = evalSh(dir2, ["target"], [], { TASKS: "OpenDrawer", SCENES: "0" }, ok);
	assert.equal(m2.status, 0, m2.stdout + m2.stderr);
	const t2 = evalSh(dir2, ["atomic"], [], T50_CELL, ok);
	assert.equal(t2.status, 1);
	assert.match(t2.stderr, /holds robocasa365 results; a target50 run needs another out dir/);
	assert.ok(!existsSync(join(dir2, `out/atomic/${T50_CELL.TASKS}_s${T50_CELL.SEEDS}`)), "no Target50 cell was run");
});
