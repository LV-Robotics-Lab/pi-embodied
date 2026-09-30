/**
 * Model-service auto-start (RPent's `robots/runtime.py`): `--serve-models vla,sam3,molmo` launches
 * the robot's model servers itself instead of attaching to ones `serve.sh` started. Each listed
 * service is started on the port of the robot's own endpoint flag (`--vla`, `--sam3`, `--molmo`,
 * ...: a loopback URL), so the robot, Flash and /robot-check use it unchanged. All of them start at
 * once and are waited for (healthz); if any exits or is not ready in time, every one started here is
 * stopped and the robot fails closed. They run with `--parent-watch` and are stopped at the next
 * session start and at session end, like the env server.
 *
 * `--serve-lock <file>` (default $PI_EMBODIED_GPU_LOCK) holds an exclusive flock(1) on that file from
 * before the first model loads until the services stop, the convention of a shared GPU box
 * (`/root/autodl-tmp/locks/gpu1.lock`): a heavy job queues for it instead of loading next to another.
 * `--serve-min-free <MiB>` first polls the GPU's free memory without the lock and checks it again
 * once the lock is held (the box's `gpu1_take`). Never combine --serve-lock with eval-parallel.sh's
 * LOCK on the same file: that run already holds it, and the flock would wait out --serve-timeout.
 * Off (the default) registers nothing but the flags.
 */

import { type ChildProcess, execFile, spawn } from "node:child_process";
import { closeSync, mkdirSync, openSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { SERVICES, servicesEnv, shutdown } from "../robot.ts";
import { RpcClient } from "./rpc.ts";

/** One model server a robot can start: `python -m module ...args` on the port of its endpoint flag. */
export type ModelService = {
	/** Its name in --serve-models and --serve-python. */
	name: string;
	/** The robot flag that holds its endpoint (http://127.0.0.1:<port>). */
	flag: string;
	module: string;
	/** Arguments before the transport ones, read at start (a checkpoint from the environment). */
	args?: () => string[];
	/** Extra environment, read at start. */
	env?: () => Record<string, string>;
};

export type ModelServicesSpec = {
	models: ModelService[];
	/** The robot's services Python (default: its --python flag); --serve-python overrides it per service. */
	python?: () => string;
};

export const SAM3: ModelService = { name: "sam3", flag: "sam3", module: "pi_embodied_services.components.sam3_server" };
/** Molmo needs its own venv (its transformers pin): pass `--serve-python molmo=<venv>/bin/python`. */
export const MOLMO: ModelService = {
	name: "molmo",
	flag: "molmo",
	module: "pi_embodied_services.components.molmo_server",
	args: () => (process.env.MOLMO_MODEL ? ["--model", process.env.MOLMO_MODEL] : []),
};
/** The Pi0.5 VLA server with one embodiment preset. */
export const pi05 = (embodiment: string): ModelService => ({
	name: "vla",
	flag: "vla",
	module: "pi_embodied_services.components.pi05_vla_server",
	args: () => ["--embodiment", embodiment],
});

const LOOPBACK = new Set(["127.0.0.1", "localhost", "[::1]"]);

/** The loopback port a service flag names, or why it cannot be served here. */
export function servicePort(flag: string, url: string): number {
	let u: URL;
	try {
		u = new URL(url.includes("://") ? url : `http://${url}`);
	} catch {
		throw new Error(`--${flag} ${url || "(unset)"} is not a URL to start a service on`);
	}
	if (u.protocol !== "http:" || !LOOPBACK.has(u.hostname) || !u.port)
		throw new Error(
			`--${flag} ${url}: a started service binds http://127.0.0.1:<port>; name one, or attach without --serve-models`,
		);
	return Number(u.port);
}

/** Parse `a=x,b=y`. */
function pairs(value: string): Map<string, string> {
	const out = new Map<string, string>();
	for (const item of value
		.split(",")
		.map((s) => s.trim())
		.filter(Boolean)) {
		const i = item.indexOf("=");
		if (i <= 0) throw new Error(`--serve-python expects name=python, got ${item}`);
		out.set(item.slice(0, i), item.slice(i + 1));
	}
	return out;
}

const tail = (path: string) => {
	try {
		return readFileSync(path, "utf8").trim().split("\n").slice(-15).join("\n");
	} catch {
		return "";
	}
};

/**
 * Take an exclusive flock on `path` (waiting at most `seconds`), held by a child that exits (and so
 * releases it) when its stdin closes: at `release`, or when pi dies.
 */
export async function lockFile(path: string, seconds: number): Promise<{ release: () => void }> {
	const holder = spawn(
		"flock",
		["-x", "-w", String(Math.max(1, Math.round(seconds))), path, "-c", "echo locked; exec cat"],
		{
			stdio: ["pipe", "pipe", "pipe"],
		},
	);
	let err = "";
	holder.stderr?.on("data", (c: Buffer) => {
		err += c.toString();
	});
	await new Promise<void>((resolve, reject) => {
		holder.stdout?.on("data", (c: Buffer) => {
			if (c.toString().includes("locked")) resolve();
		});
		holder.once("error", (e) => reject(new Error(`--serve-lock needs flock(1): ${e.message}`)));
		holder.once("exit", (code) =>
			reject(
				new Error(`--serve-lock ${path}: not acquired within ${seconds} s (flock exited ${code}) ${err.trim()}`),
			),
		);
	});
	return {
		release: () => {
			holder.stdin?.end();
		},
	};
}

/** Free memory of GPU `gpu` (nvidia-smi's physical ordinal), MiB. */
export async function gpuFreeMiB(gpu: string): Promise<number> {
	const { stdout } = await promisify(execFile)("nvidia-smi", [
		"--query-gpu=memory.used,memory.total",
		"--format=csv,noheader,nounits",
		"-i",
		gpu,
	]);
	const [used, total] = stdout.trim().split(",").map(Number);
	if (!Number.isFinite(used) || !Number.isFinite(total)) throw new Error(`nvidia-smi -i ${gpu}: ${stdout.trim()}`);
	return total - used;
}

/** Poll until GPU `gpu` has `mib` free, without any lock; fails at `deadline`. */
async function waitForMemory(gpu: string, mib: number, deadline: number, pollMs = 15_000) {
	for (;;) {
		const free = await gpuFreeMiB(gpu);
		if (free >= mib) return;
		if (Date.now() + pollMs > deadline)
			throw new Error(`--serve-min-free: GPU ${gpu} has ${free} MiB free, not ${mib}, within --serve-timeout`);
		await new Promise((r) => setTimeout(r, pollMs));
	}
}

/** The pid a service's healthz reports (services/PROTOCOL.md), or undefined when nothing answers. */
async function answeringPid(rpc: RpcClient): Promise<number | null | undefined> {
	return rpc.call<{ pid?: number }>("healthz", {}, 3_000).then(
		(h) => (typeof h?.pid === "number" ? h.pid : null),
		() => undefined,
	);
}

/** Start `python -m module` on `port` and wait for healthz; rejects when it exits or `deadline` passes. */
async function startOne(o: {
	python: string;
	module: string;
	args: string[];
	port: number;
	cwd: string;
	env: NodeJS.ProcessEnv;
	log: string;
	deadline: number;
	started: (proc: ChildProcess, rpc: RpcClient) => void;
}): Promise<void> {
	const argv = ["-m", o.module, ...o.args, "--transport", "http", "--host", "127.0.0.1", "--port", String(o.port)];
	const fd = openSync(o.log, "a");
	const proc = spawn(o.python, [...argv, "--parent-watch"], { cwd: o.cwd, env: o.env, stdio: ["pipe", fd, fd] });
	closeSync(fd);
	const rpc = new RpcClient(`http://127.0.0.1:${o.port}`);
	o.started(proc, rpc);
	let gone: Error | undefined;
	proc.once("exit", (code, sig) => {
		gone = new Error(`exited (${code ?? sig}); ${o.log}:\n${tail(o.log)}`);
	});
	proc.once("error", (e) => {
		gone = new Error(`failed to start ${o.python}: ${e.message}`);
	});
	// Poll healthz until it answers; stop polling as soon as the process is gone or time is up.
	for (;;) {
		if (gone) throw gone;
		if (Date.now() > o.deadline) throw new Error(`not ready by --serve-timeout; ${o.log}:\n${tail(o.log)}`);
		// Ready means OUR process answers: another pi's server on the same port (bound while this one
		// was still loading its model) must not be taken for it.
		const pid = await answeringPid(rpc);
		if (pid !== undefined && pid === proc.pid) return;
		if (pid !== undefined)
			throw new Error(
				`:${o.port} is answered by another process (pid ${pid}), not the one started here (pid ${proc.pid})`,
			);
		await new Promise((r) => setTimeout(r, 500));
	}
}

/**
 * Register the --serve-* flags and return the start/stop pair the robot base calls around the
 * robot's own start (before it attaches to the services) and at its stop.
 */
export function modelServices(pi: ExtensionAPI, spec: ModelServicesSpec) {
	const names = spec.models.map((m) => m.name);
	pi.registerFlag("serve-models", {
		type: "string",
		default: "",
		description: `Start these model services here and wait for them (${names.join(", ")}, or all), each on its endpoint flag's loopback port; any failure stops them all`,
	});
	pi.registerFlag("serve-python", {
		type: "string",
		default: "",
		description: "Per-service Python for --serve-models, name=path,... (default: the robot's --python)",
	});
	pi.registerFlag("serve-cuda-device", {
		type: "string",
		default: "",
		description: "GPU ordinal for the --serve-models services (default: the robot's --cuda-device)",
	});
	pi.registerFlag("serve-lock", {
		type: "string",
		default: process.env.PI_EMBODIED_GPU_LOCK ?? "",
		description: "flock file held while the --serve-models services load and run (a shared GPU's lock)",
	});
	pi.registerFlag("serve-min-free", {
		type: "string",
		default: "0",
		description:
			"MiB the --serve-models GPU must have free: polled without the lock, checked again once --serve-lock is held",
	});
	pi.registerFlag("serve-timeout", {
		type: "string",
		default: "900",
		description: "Seconds for --serve-lock and every --serve-models service to be ready",
	});
	pi.registerFlag("serve-log-dir", {
		type: "string",
		default: join(tmpdir(), "pi-embodied"),
		description: "Logs of the --serve-models services (<name>-<port>.log)",
	});
	const flag = (name: string) => String(pi.getFlag(name) ?? "").trim();

	let running: { name: string; proc: ChildProcess; rpc: RpcClient; ready: boolean }[] = [];
	let lock: { release: () => void } | undefined;

	/**
	 * A ready service whose healthz still reports its pid gets the env server's orderly shutdown;
	 * any other (still loading, or its port answered by someone else) is signalled directly, never
	 * over the port: SIGTERM, and SIGKILL 5 s later.
	 */
	async function stopOne(s: (typeof running)[number]) {
		// The orderly shutdown goes over the port: only while our own process still answers there.
		if (s.ready && (await answeringPid(s.rpc)) === s.proc.pid) return shutdown(s.proc, s.rpc);
		if (s.proc.exitCode !== null || s.proc.signalCode !== null) return;
		const exited = new Promise<boolean>((resolve) => s.proc.once("exit", () => resolve(true)));
		s.proc.kill("SIGTERM");
		const late = new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 5_000).unref());
		if (!(await Promise.race([exited, late]))) s.proc.kill("SIGKILL");
	}

	async function stop() {
		const r = running;
		running = [];
		await Promise.all(r.reverse().map(stopOne));
		lock?.release();
		lock = undefined;
	}

	/** The services named by --serve-models, validated. */
	function selected(): ModelService[] {
		const want = flag("serve-models");
		if (!want) return [];
		const list =
			want === "all"
				? names
				: want
						.split(",")
						.map((s) => s.trim())
						.filter(Boolean);
		const unknown = list.filter((n) => !names.includes(n));
		if (unknown.length)
			throw new Error(`--serve-models: unknown ${unknown.join(", ")}; this robot serves ${names.join(", ")}`);
		return spec.models.filter((m) => list.includes(m.name));
	}

	async function start() {
		await stop();
		const models = selected();
		if (!models.length) return;
		const pythons = pairs(flag("serve-python"));
		const unknownPy = [...pythons.keys()].filter((n) => !models.some((m) => m.name === n));
		if (unknownPy.length) throw new Error(`--serve-python: ${unknownPy.join(", ")} not in --serve-models`);
		const ports = models.map((m) => servicePort(m.flag, flag(m.flag)));
		// A server already answering there is not ours to load or stop.
		await Promise.all(
			models.map(async (m, i) => {
				const probe = new RpcClient(`http://127.0.0.1:${ports[i]}`);
				const up = await probe.call("healthz", {}, 2_000).then(
					() => true,
					() => false,
				);
				if (up)
					throw new Error(
						`--serve-models ${m.name}: ${flag(m.flag)} already serves; drop it from --serve-models to attach to it`,
					);
			}),
		);
		const seconds = Number(flag("serve-timeout")) || 900;
		const deadline = Date.now() + seconds * 1000;
		const cuda = flag("serve-cuda-device") || flag("cuda-device");
		const root = flag("services") || SERVICES;
		const logDir = flag("serve-log-dir") || join(tmpdir(), "pi-embodied");
		mkdirSync(logDir, { recursive: true });
		const lockPath = flag("serve-lock");
		const minFree = Number(flag("serve-min-free")) || 0;
		// The box's rule (its gpu1_take): poll the free memory without the lock, take the lock, check again.
		for (;;) {
			const gpu =
				cuda || (/^\d+$/.test(process.env.CUDA_VISIBLE_DEVICES ?? "") ? process.env.CUDA_VISIBLE_DEVICES : "0");
			if (minFree) await waitForMemory(String(gpu), minFree, deadline);
			if (lockPath) lock = await lockFile(lockPath, Math.max(1, (deadline - Date.now()) / 1000));
			if (!minFree || (await gpuFreeMiB(String(gpu))) >= minFree) break;
			lock?.release();
			lock = undefined;
		}
		try {
			await Promise.all(
				models.map((m, i) =>
					startOne({
						python: pythons.get(m.name) ?? spec.python?.() ?? (flag("python") || "python"),
						module: m.module,
						args: [...(m.args?.() ?? []), ...(cuda ? ["--cuda-device", cuda] : [])],
						port: ports[i],
						cwd: root,
						env: servicesEnv({ root, python: "", env: m.env?.() }),
						log: join(logDir, `${m.name}-${ports[i]}.log`),
						deadline,
						started: (proc, rpc) => running.push({ name: m.name, proc, rpc, ready: false }),
					}).then(
						() => {
							const s = running.find((r) => r.name === m.name);
							if (s) s.ready = true;
						},
						(err: unknown) => {
							throw new Error(
								`--serve-models ${m.name} on :${ports[i]}: ${err instanceof Error ? err.message : err}`,
							);
						},
					),
				),
			);
		} catch (err) {
			await stop();
			throw err;
		}
	}

	return { start, stop, running: () => running.map((s) => s.name) };
}
