import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import {
	chmodSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readdirSync,
	readFileSync,
	symlinkSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { acquire, release } from "../src/api-gate.ts";

const RUNNER = new URL("../src/eval-parallel.sh", import.meta.url).pathname;

/**
 * A stand-in pi for the real eval.sh scripts: it logs each call (session dir and GPU environment),
 * then writes a `robot_result` that succeeds on even seeds or with `--good`. A seed in FAKE_FAIL
 * exits without a result (env_error); seed 2 hangs while the FAKE_HANG file exists.
 */
const FAKE_PI = `#!/usr/bin/env bash
all=("$@") dir="" seed="" good=false
while [ $# -gt 0 ]; do case $1 in --session-dir) dir=$2 && shift ;; --seed) seed=$2 && shift ;; --good) good=true ;; esac; shift; done
printf '%s\\n' "\${all[@]}" >"$dir/argv"
echo $$ >"$dir/pid"
echo "$dir CVD=\${CUDA_VISIBLE_DEVICES-unset} EGL=\${MUJOCO_EGL_DEVICE_ID-unset} ORDER=\${CUDA_DEVICE_ORDER-unset}" >>"$FAKE_LOG"
if [ "$seed" = 2 ] && [ -n "\${FAKE_HANG:-}" ] && [ -e "$FAKE_HANG" ]; then sleep 30 & wait; fi
case ",\${FAKE_FAIL:-}," in *",$seed,"*) exit 1 ;; esac
[ $((seed % 2)) = 0 ] && good=true
echo '{"type":"custom","customType":"robot_result","data":{"success":'$good',"terminated":'$good',"claimed":"success","env_steps":1}}' >"$dir/s.jsonl"
`;

function sandbox() {
	const dir = mkdtempSync(join(tmpdir(), "eval-parallel-"));
	const bin = join(dir, "bin");
	mkdirSync(bin);
	const pi = join(bin, "pi");
	writeFileSync(pi, FAKE_PI);
	chmodSync(pi, 0o755);
	const env: Record<string, string> = {
		...(process.env as Record<string, string>),
		PATH: `${bin}:${process.env.PATH}`,
		PI: pi,
		TIME_LIMIT: "0",
		FAKE_LOG: join(dir, "calls.log"),
	};
	delete env.CUDA_VISIBLE_DEVICES;
	delete env.LOCK;
	const tool = (name: string, body: string) => {
		writeFileSync(join(bin, name), `#!/usr/bin/env bash\n${body}\n`);
		chmodSync(join(bin, name), 0o755);
	};
	const run = (args: string[], extra: Record<string, string> = {}) =>
		spawnSync("bash", [RUNNER, ...args], { env: { ...env, ...extra }, encoding: "utf8" });
	const calls = () =>
		existsSync(env.FAKE_LOG)
			? readFileSync(env.FAKE_LOG, "utf8")
					.trim()
					.split("\n")
					.map((l) => l.split(" "))
			: [];
	return { dir, env, tool, run, calls };
}

/** Every cell directory under `out` with its result status, relative to `out`. */
function cells(out: string) {
	const found: Record<string, string> = {};
	const walk = (d: string, rel: string) => {
		for (const e of readdirSync(d, { withFileTypes: true })) {
			if (!e.isDirectory() || e.name === ".parallel") continue;
			const path = join(d, e.name);
			const result = join(path, "result.json");
			if (existsSync(result)) found[`${rel}${e.name}`] = JSON.parse(readFileSync(result, "utf8")).status;
			else walk(path, `${rel}${e.name}/`);
		}
	};
	walk(out, "");
	return found;
}

/** Whether `pid` has exited. A zombie not yet reaped counts: kill(pid, 0) still succeeds on it. */
function gone(pid: number) {
	try {
		process.kill(pid, 0);
		const state =
			process.platform === "linux"
				? readFileSync(`/proc/${pid}/stat`, "utf8").replace(/^.*\) /, "")[0]
				: spawnSync("ps", ["-o", "stat=", "-p", String(pid)], { encoding: "utf8" }).stdout.trim()[0];
		return state === "Z" || state === undefined;
	} catch {
		return true;
	}
}

async function untilGone(pid: number, ms = 3000) {
	for (let t = 0; t < ms && !gone(pid); t += 50) await new Promise((r) => setTimeout(r, 50));
	return gone(pid);
}

const groups = (out: string) =>
	readdirSync(join(out, ".parallel"))
		.filter((f) => /^w\d+\.pid$/.test(f))
		.map((f) => Number(readFileSync(join(out, ".parallel", f), "utf8")));

const summary = (out: string) =>
	(JSON.parse(readFileSync(join(out, "summary.json"), "utf8")).variants as Record<string, unknown>[]).map(
		({ dir: _dir, ...v }) => v,
	);

test("N=1 and N=4 run the same cells, each once, with the same summary", () => {
	const s = sandbox();
	const args = (n: number) => [
		"-j",
		String(n),
		"--variant",
		"base=",
		"--variant",
		"good=--good",
		"maniskill",
		join(s.dir, `n${n}`),
		"PickCube-v1,StackCube-v1",
		"0-3",
		"--model",
		"m/x",
	];
	const one = s.run(args(1));
	assert.equal(one.status, 0, one.stderr + one.stdout);
	const four = s.run(args(4));
	assert.equal(four.status, 0, four.stderr + four.stdout);
	const a = cells(join(s.dir, "n1"));
	assert.equal(Object.keys(a).length, 16);
	assert.deepEqual(cells(join(s.dir, "n4")), a);
	assert.deepEqual(summary(join(s.dir, "n4")), summary(join(s.dir, "n1")));
	// Every cell ran exactly once under N=4: workers never share a unit.
	const n4 = s.calls().filter(([d]) => d.includes("/n4/"));
	assert.equal(n4.length, 16);
	assert.equal(new Set(n4.map(([d]) => d)).size, 16);
});

test("variants run the same seeds into their own dirs; success, Pass@k and --min-success per variant", () => {
	const s = sandbox();
	const out = join(s.dir, "ab");
	const r = s.run([
		"-j",
		"3",
		"--variant",
		"base=",
		"--variant",
		"good=--good --thinking low",
		"--min-success",
		"5",
		"libero",
		out,
		"libero_10_task",
		"0-1",
		"0-3",
		"--model",
		"m/x",
	]);
	assert.equal(r.status, 1, "base has 4 successes, below 5");
	assert.match(r.stdout, /base: below --min-success 5/);
	assert.doesNotMatch(r.stdout, /good: below/);
	const c = cells(out);
	const seeds = (v: string) =>
		Object.keys(c)
			.filter((k) => k.startsWith(`${v}/`))
			.map((k) => k.slice(v.length + 1))
			.sort();
	assert.deepEqual(seeds("base"), seeds("good"));
	assert.equal(seeds("base").length, 8);
	const argv = readFileSync(join(out, "good/libero_10_task_t0_s1/argv"), "utf8").split("\n");
	assert.ok(argv.includes("--good") && argv.includes("low"), String(argv));
	const [base, good] = summary(out) as {
		variant: string;
		success: number;
		success_rate: number;
		config: string;
		pass_at_k: { k: number; value: number; tasks: number }[];
	}[];
	assert.equal(base.variant, "base");
	assert.equal(base.success, 4);
	assert.equal(base.success_rate, 0.5);
	// Two successes of four seeds per task: pass@1 = 1/2, pass@4 = 1.
	assert.deepEqual(
		base.pass_at_k.map((p) => [p.k, p.value, p.tasks]),
		[
			[1, 0.5, 2],
			[4, 1, 2],
		],
	);
	assert.equal(good.success, 8);
	assert.match(good.config, /low/);
	// Pass@2 of 2/4: 1 - C(2,2)/C(4,2) = 5/6.
	const k2 = s.run([
		"--variant",
		"base=",
		"--pass-k",
		"2",
		"libero",
		out,
		"libero_10_task",
		"0-1",
		"0-3",
		"--model",
		"m/x",
	]);
	assert.equal(k2.status, 0, k2.stdout);
	assert.match(k2.stdout, /pass@2 83\.3% \(2\/2 tasks\)/);
});

test("a rerun fills only the invalid cells; infrastructure failures are counted apart", () => {
	const s = sandbox();
	const out = join(s.dir, "out");
	const args = ["-j", "2", "maniskill", out, "PickCube-v1", "0-5", "--model", "m/x"];
	const first = s.run(args, { FAKE_FAIL: "1,4" });
	assert.equal(first.status, 1);
	assert.match(first.stdout, /success 2\/4 \(50\.0%\).*invalid 2 \(env_error 2,/);
	const before = s.calls().length;
	const second = s.run(args);
	assert.equal(second.status, 0, second.stdout);
	assert.deepEqual(
		s
			.calls()
			.slice(before)
			.map(([d]) => d.split("/").pop())
			.sort(),
		["PickCube-v1_s1", "PickCube-v1_s4"],
	);
	assert.match(second.stdout, /success 3\/6 \(50\.0%\).*invalid 0/);
	// A result of another configuration stops the run before any episode.
	const third = s.run(["maniskill", out, "PickCube-v1", "0-5", "--model", "m/other"]);
	assert.notEqual(third.status, 0);
	assert.match(third.stderr, /use another out dir/);
	assert.equal(s.calls().length, before + 2);
});

test("an interrupted run stops its episodes, and the rerun only fills the cells without a valid result", async () => {
	const s = sandbox();
	const out = join(s.dir, "out");
	const hang = join(s.dir, "hang");
	writeFileSync(hang, "");
	const args = ["-j", "2", "maniskill", out, "PickCube-v1", "0-5", "--model", "m/x"];
	const child = spawn("bash", [RUNNER, ...args], { env: { ...s.env, FAKE_HANG: hang }, stdio: "ignore" });
	const exited = new Promise<number | null>((r) => child.once("exit", (code) => r(code)));
	const pidFile = join(out, "PickCube-v1_s2/pid");
	for (let i = 0; i < 200 && !existsSync(pidFile); i++) await new Promise((r) => setTimeout(r, 50));
	await new Promise((r) => setTimeout(r, 200));
	const hung = Number(readFileSync(pidFile, "utf8"));
	const pgids = groups(out);
	assert.equal(pgids.length, 2, "one process group per worker");
	child.kill("SIGTERM");
	assert.equal(await exited, 130);
	assert.ok(await untilGone(hung), "the hung episode was stopped");
	assert.deepEqual(groups(out), [], "the groups are forgotten once they are gone");
	for (const g of pgids) assert.notEqual(spawnSync("pgrep", ["-g", String(g)]).status, 0, `group ${g} is empty`);
	const done = cells(out);
	assert.equal(done["PickCube-v1_s2"], undefined);
	const before = s.calls().length;
	const rerun = s.run(args);
	assert.equal(rerun.status, 0, rerun.stdout + rerun.stderr);
	const redone = s
		.calls()
		.slice(before)
		.map(([d]) => d.split("/").pop());
	const expected = [0, 1, 2, 3, 4, 5]
		.map((i) => `PickCube-v1_s${i}`)
		.filter((c) => done[c] !== "success" && done[c] !== "failure");
	assert.deepEqual(redone.sort(), expected);
	assert.equal(Object.keys(cells(out)).length, 6);
});

test("a rerun refuses to start while an earlier run's worker groups are alive", async () => {
	const s = sandbox();
	const out = join(s.dir, "out");
	const hang = join(s.dir, "hang");
	writeFileSync(hang, "");
	const args = ["maniskill", out, "PickCube-v1", "2", "--model", "m/x"];
	const child = spawn("bash", [RUNNER, ...args], { env: { ...s.env, FAKE_HANG: hang }, stdio: "ignore" });
	const pidFile = join(out, "PickCube-v1_s2/pid");
	for (let i = 0; i < 200 && !existsSync(pidFile); i++) await new Promise((r) => setTimeout(r, 50));
	const hung = Number(readFileSync(pidFile, "utf8"));
	// The shell dies without its stop handler; the worker's group (eval.sh, pi, the episode) lives on.
	child.kill("SIGKILL");
	await new Promise<void>((r) => child.once("exit", () => r()));
	const [pgid, ...rest] = groups(out);
	assert.deepEqual(rest, []);
	assert.ok(!gone(hung));
	const refused = s.run(args);
	assert.equal(refused.status, 2);
	assert.match(refused.stderr, /still has episodes of an earlier run/);
	assert.ok(!gone(hung), "the rerun did not touch the running episode");
	process.kill(-pgid, "SIGTERM");
	assert.ok(await untilGone(hung));
	for (let t = 0; t < 3000 && spawnSync("pgrep", ["-g", String(pgid)]).status === 0; t += 50)
		await new Promise((r) => setTimeout(r, 50));
	const rerun = s.run(args);
	assert.equal(rerun.status, 0, rerun.stdout + rerun.stderr);
	assert.deepEqual(cells(out), { "PickCube-v1_s2": "success" });
});

test("workers get their GPU and the EGL device on its PCI bus; only a heavy robot's run takes the GPU lock, once", () => {
	const s = sandbox();
	s.tool("nvidia-smi", `printf '0, 00000000:98:00.0\\n1, 00000000:C8:00.0\\n'`);
	const dri = join(s.dir, "by-path");
	mkdirSync(dri);
	symlinkSync("../card5", join(dri, "pci-0000:98:00.0-card"));
	symlinkSync("../card8", join(dri, "pci-0000:c8:00.0-card"));
	// EGL lists card8 (GPU 1) first; the CUDA ordinal must not be used as the EGL index.
	const py = join(s.dir, "bin/egl-python");
	writeFileSync(py, "#!/usr/bin/env bash\nprintf '0 card8\\n1 card5\\n2 card8\\n3 -\\n'\n");
	chmodSync(py, 0o755);
	const flocks = join(s.dir, "flock.log");
	s.tool("flock", `printf "%s\\n" "$*" >>"${flocks}"`);
	const env = { DRI_BY_PATH: dri, EGL_PYTHON: py, LOCK: join(s.dir, "gpu1.lock") };
	const r = s.run(["-j", "3", "--gpus", "1", "maniskill", join(s.dir, "one"), "PickCube-v1", "0-5"], env);
	assert.equal(r.status, 0, r.stdout + r.stderr);
	const seen = new Set(s.calls().map(([, ...e]) => e.join(" ")));
	// Only GPU 1 is exposed; the EGL index (0) is not added to CUDA_VISIBLE_DEVICES for robosuite's check.
	assert.deepEqual([...seen], ["CVD=1 EGL=0 ORDER=PCI_BUS_ID"]);
	// ManiSkill is a light job: it shares the GPU and never takes LOCK.
	assert.ok(!existsSync(flocks), "a light robot takes no lock");
	assert.match(r.stderr, /maniskill is a light job .* LOCK is for robolab and robotwin/);
	const heavy = s.run(["-j", "2", "--gpus", "1", "robolab", join(s.dir, "heavy"), "BananaInBowlTask", "0-1"], env);
	assert.equal(heavy.status, 0, heavy.stdout + heavy.stderr);
	assert.deepEqual(
		readFileSync(flocks, "utf8").trim().split("\n"),
		["-n 9"],
		"the lock is taken once, not per worker",
	);
	const two = s.run(["-j", "2", "--gpus", "0,1", "maniskill", join(s.dir, "two"), "PickCube-v1", "0-3"], env);
	assert.equal(two.status, 0, two.stdout + two.stderr);
	const byGpu = new Set(
		s
			.calls()
			.filter(([d]) => d.includes("/two/"))
			.map(([, cvd, egl]) => `${cvd} ${egl}`),
	);
	assert.deepEqual([...byGpu].sort(), ["CVD=0 EGL=1", "CVD=1 EGL=0"]);
	// The MuJoCo env servers pin the GPU themselves (and drop CUDA_VISIBLE_DEVICES before importing robosuite).
	const lib = s.run(["--gpus", "1", "libero", join(s.dir, "lib"), "libero_10_task", "0", "0"], env);
	assert.equal(lib.status, 0, lib.stdout + lib.stderr);
	const argv = readFileSync(join(s.dir, "lib/libero_10_task_t0_s0/argv"), "utf8").split("\n");
	assert.equal(argv[argv.indexOf("--cuda-device") + 1], "1");
	const plain = s.run(["maniskill", join(s.dir, "none"), "PickCube-v1", "0"]);
	assert.equal(plain.status, 0);
	assert.equal(s.calls().at(-1)?.slice(1).join(" "), "CVD=unset EGL=unset ORDER=unset");
});

test("--max-api-concurrency and --dashboard-ports reach every pi call", () => {
	const s = sandbox();
	const out = join(s.dir, "out");
	const r = s.run([
		"-j",
		"2",
		"--max-api-concurrency",
		"1",
		"--dashboard-ports",
		"8765,8767",
		"maniskill",
		out,
		"PickCube-v1",
		"0-3",
	]);
	assert.equal(r.status, 0, r.stdout + r.stderr);
	const ports = new Set<string>();
	for (const cell of Object.keys(cells(out))) {
		const argv = readFileSync(join(out, cell, "argv"), "utf8").split("\n");
		assert.ok(argv.some((a) => a.endsWith("/api-gate.ts")));
		assert.equal(argv[argv.indexOf("--max-api-concurrency") + 1], "1");
		assert.equal(argv[argv.indexOf("--api-slots") + 1], join(out, ".parallel/api"));
		ports.add(argv[argv.indexOf("--dashboard-port") + 1]);
	}
	assert.deepEqual([...ports].sort(), ["8765", "8767"]);
	assert.equal(s.run(["-j", "3", "--dashboard-ports", "8765", "maniskill", out, "PickCube-v1", "0"]).status, 2);
});

test("api-gate admits n model calls at once and takes over the slot of a dead process", async () => {
	const dir = mkdtempSync(join(tmpdir(), "api-gate-"));
	const a = await acquire(dir, 1, 10);
	let second: string | undefined;
	const waiting = acquire(dir, 1, 10).then((p) => {
		second = p;
	});
	await new Promise((r) => setTimeout(r, 60));
	assert.equal(second, undefined, "the only slot is held");
	release(a);
	await waiting;
	assert.equal(second, a);
	release(a);
	// A slot left by a pid that no longer exists is free.
	const dead = spawnSync("true").pid;
	writeFileSync(join(dir, "slot-0"), String(dead));
	assert.equal(await acquire(dir, 1, 10), join(dir, "slot-0"));
	assert.equal(readFileSync(join(dir, "slot-0"), "utf8"), String(process.pid));
	// Two slots: a second holder does not wait.
	assert.equal(await acquire(dir, 2, 10), join(dir, "slot-1"));
	// A waiter whose agent aborts stops waiting instead of taking a slot later.
	const ac = new AbortController();
	setTimeout(() => ac.abort(), 30);
	await assert.rejects(acquire(dir, 2, 10, ac.signal), /aborted/);
});
