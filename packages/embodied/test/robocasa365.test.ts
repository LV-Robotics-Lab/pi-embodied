import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import robocasa from "../src/robocasa/index.ts";
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

test("robocasa/eval.sh records code mode in both protocols and never mixes it with tool runs", () => {
	for (const [positional, env, cell] of [
		[["target"], { TASKS: "OpenDrawer", SCENES: "0" }, "out/target/OpenDrawer_m0/result.json"],
		[["atomic"], T50_CELL, `out/atomic/${T50_CELL.TASKS}_s${T50_CELL.SEEDS}/result.json`],
	] as const) {
		const dir = mkdtempSync(join(tmpdir(), "eval-code-"));
		const a = evalSh(dir, [...positional], ["--code=true", "--code-api", "low"], env, ok);
		const result = JSON.parse(a.read(cell) ?? "{}");
		assert.deepEqual([result.code, result.code_api, result.code_oracle], ["true", "low", null], a.stderr);
		assert.match(a.stdout, /\/code=true:low/);
		// The same configuration keeps its result; a tool run (or another tier) into the dir is refused.
		assert.equal(evalSh(dir, [...positional], ["--code=true", "--code-api=low"], env).argv, undefined);
		for (const other of [[], ["--code=true"]]) {
			const r = evalSh(dir, [...positional], other, env);
			assert.equal(r.status, 1);
			assert.match(r.stderr, /code mode/);
		}
	}
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

type Handler = (event: any, ctx: any) => unknown;
type Tool = { name: string; description: string; parameters: any; execute: (...a: any[]) => Promise<any> };

/** A stub pi recording the flags, tools and entries the robot registers (flags at their defaults, or `values`). */
function stubPi(values: Record<string, unknown> = {}) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, { default?: unknown; description?: string }> = {};
	const tools = new Map<string, Tool>();
	const entries: { type: string; data: any }[] = [];
	let active: string[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown; description?: string }) => {
			flags[name] = o;
		},
		getFlag: (name: string) => (name in values ? values[name] : flags[name]?.default),
		registerTool: (t: Tool) => tools.set(t.name, t),
		registerCommand: () => {},
		registerProvider: () => {},
		registerMessageRenderer: () => {},
		registerShortcut: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const ctx = {
		hasUI: false,
		cwd: tmpdir(),
		ui: { notify: () => {} },
		shutdown: () => {},
		abort: () => {},
		sessionManager: {
			getBranch: () => [],
			getEntries: () => [],
			getSessionDir: () => tmpdir(),
			getSessionFile: () => undefined,
			getSessionId: () => "sess",
		},
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) {
			const r: any = await fn({ type: name, ...event }, ctx);
			if (r !== undefined) result = r;
		}
		return result;
	}
	const run = async (name: string, params: unknown) =>
		(await tools.get(name)!.execute("id", params, undefined, undefined, ctx)) as any;
	return { pi, tools, entries, emit, run, active: () => active };
}

/**
 * A fake RoboCasa env server (`--env`, also the `--rldx` session server): OpenDrawer target seed 0,
 * robot observations, renders and a `code.run` that stepped 12 times and solved the task.
 */
async function fakeEnv() {
	const calls: { method: string; kwargs: Record<string, unknown>; args: unknown[] }[] = [];
	const nd = (dtype: string, shape: number[], data: Buffer) => ({
		__ndarray__: data.toString("base64"),
		dtype,
		shape,
	});
	const f32 = (v: number[]) => nd("float32", [v.length], Buffer.from(Float32Array.from(v).buffer));
	const f64 = (v: number[], shape = [v.length]) => nd("float64", shape, Buffer.from(Float64Array.from(v).buffer));
	const raw = (z: number) => ({
		robot0_eef_pos: f32([0.5, 0, z]),
		robot0_eef_quat: f32([0, 0, 0, 1]),
		robot0_gripper_qpos: f32([0.04, -0.04]),
		robot0_base_pos: f32([0, 0, 0]),
		robot0_base_quat: f32([0, 0, 0, 1]),
		robot0_base_to_eef_pos: f32([0.5, 0, z]),
		robot0_base_to_eef_quat: f32([0, 0, 0, 1]),
	});
	let solved = false;
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, kwargs = {}, args = [] } = JSON.parse(body);
			calls.push({ method, kwargs, args });
			let result: unknown = { ok: true };
			const size = Number(kwargs.height ?? 256);
			if (method === "code.api") result = { tier: kwargs.tier ?? null, primitives: [], digest: "d" };
			else if (method === "env.get_env_meta")
				result = { task_name: "OpenDrawer", split: "target", seed: 0, scene: null, env_id: null };
			else if (method === "env.reset") result = raw(1);
			else if (method === "env.get_task_language") result = "open the drawer";
			else if (method === "env.get_success_criteria_text") result = "drawer open";
			else if (method === "env.check_success") result = solved;
			else if (method === "env.get_task_progress") result = { open: solved };
			else if (method === "env.render_camera") {
				const img = nd("uint8", [size, size, 3], Buffer.alloc(size * size * 3));
				result = kwargs.depth ? [img, nd("float32", [size, size], Buffer.alloc(size * size * 4))] : img;
			} else if (method === "env.get_camera_transform")
				result = f64([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [4, 4]);
			else if (method === "code.run") {
				solved = true;
				result = {
					status: "ran",
					stdout: "",
					stderr: "",
					traceback: null,
					error: null,
					result: null,
					calls: [],
					n_calls: 12,
					move_m: 0.3,
					ms: 5,
					steps: 12,
					success: true,
					obs: raw(1.2),
					frames: [nd("uint8", [2, 2, 3], Buffer.alloc(12))],
				};
			}
			res.end(JSON.stringify({ ok: true, result }));
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
	return {
		url,
		calls,
		close: () => {
			server.closeAllConnections();
			server.close();
		},
	};
}

test("--code=true: run_code runs on the env server; the run becomes a state, its steps and success the result's", async (t) => {
	const env = await fakeEnv();
	t.after(env.close);
	const s = stubPi({ env: env.url, rldx: env.url, code: "true", "code-api": "low", services: SERVICES });
	robocasa(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(s.active(), ["run_code", "finish"]);
	await s.emit("agent_start");
	const r = await s.run("run_code", { code: "step([0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, -1])" });
	const run = env.calls.find((c) => c.method === "code.run")!;
	assert.equal(run.kwargs.tier, "low");
	assert.equal(r.details.status, "ran");
	assert.equal(r.details.success, true);
	const view = JSON.parse(r.content[1].text);
	assert.equal(view.step, 1, "the run is the next numbered state");
	assert.deepEqual(view.state.robot0_eef_pos, [0.5, 0, 1.2], "the run's robot obs, absorbed");
	assert.equal(view.robocasa_terminated, true);
	await s.run("finish", { status: "success", summary: "opened" });
	await s.emit("agent_end", { messages: [] });
	const result = s.entries.find((e) => e.type === "robot_result")?.data;
	assert.equal(result.success, true);
	assert.equal(result.env_steps, 12);
	assert.equal(result.code, "true");
	assert.equal(result.code_api, "low");
});
