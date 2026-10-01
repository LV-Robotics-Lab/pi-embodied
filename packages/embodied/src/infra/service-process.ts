/** Process ownership and readiness for Python RPC services. No robot or planner policy. */
import { type ChildProcess, execFile, spawn } from "node:child_process";
import { closeSync, openSync, writeSync } from "node:fs";
import { promisify } from "node:util";
import { parseEndpoint, RpcClient } from "./rpc.ts";

export type Services = { root: string; python: string; env?: Record<string, string> };
export type ServiceOptions = {
	python: string;
	args: string[];
	cwd: string;
	env: NodeJS.ProcessEnv;
	log: (port: number) => string;
	readyMs?: number;
};

/** Return ownership immediately, so a failed startup is still cleaned up by its owner. */
export function startService(o: ServiceOptions) {
	// The server binds port 0 and prints the port it got: probing a free port here and passing it on
	// would race every other process on the box for it between the probe and the bind.
	const argv = [...o.args, "--transport", "http", "--host", "127.0.0.1", "--port", "0", "--parent-watch"];
	const proc = spawn(o.python, argv, { cwd: o.cwd, env: o.env, stdio: ["pipe", "pipe", "pipe"] });
	const rpc = new RpcClient("http://127.0.0.1:0");
	// Its output goes to the log file, whose name carries the port, so it is held until the port is known.
	let fd: number | undefined;
	let held = "";
	let log = "";
	const listening = new Promise<number>((resolve) => {
		const sink = (chunk: Buffer) => {
			if (fd !== undefined) {
				writeSync(fd, chunk);
				return;
			}
			held += chunk.toString();
			const m = /RPC server listening on http:\/\/[^\s:]+:(\d+)(?: \(token ([0-9a-f]+)\))?\r?\n/.exec(held);
			if (!m) return;
			if (m[2]) rpc.token = m[2];
			log = o.log(Number(m[1]));
			fd = openSync(log, "a");
			// The token stays out of the log file (a run_code program may be able to read it).
			writeSync(fd, m[2] ? held.replace(m[2], "<redacted>") : held);
			held = "";
			resolve(Number(m[1]));
		};
		proc.stdout?.on("data", sink);
		proc.stderr?.on("data", sink);
	});
	proc.once("close", () => {
		if (fd !== undefined) closeSync(fd);
	});
	const where = () => (log ? `see ${log}` : `it printed:\n${held.trim()}`);
	const exited = new Promise<never>((_, reject) => {
		proc.once("exit", (code) => reject(new Error(`env server exited (${code}); ${where()}`)));
		proc.once("error", (err) => reject(new Error(`env server failed to start: ${err.message}`)));
	});
	exited.catch(() => {});
	const readyMs = o.readyMs ?? 300_000;
	const deadline = Date.now() + readyMs;
	const late = new Promise<never>((_, reject) => {
		setTimeout(() => reject(new Error(`env server bound no port in ${readyMs} ms; ${where()}`)), readyMs).unref();
	});
	late.catch(() => {});
	const ready = (async () => {
		const port = await Promise.race([listening, exited, late]);
		rpc.url = rpc.url.replace(":0/", `:${port}/`);
		await Promise.race([rpc.ready(deadline - Date.now()), exited]);
		return log;
	})();
	return { proc, rpc, ready };
}

/**
 * Stop a service so it cleans up (the services' `close()` / `close_env()`: the sim, or the real
 * arm's RLinf worker): `stop` interrupts a running call where the server has that method, the
 * built-in `shutdown` (queued behind any call still running) closes the env and exits. A server
 * still up 30 s later gets EOF on stdin (--parent-watch: close without waiting for a running call),
 * and is killed 5 s after that. A bare kill would skip the cleanup.
 */
export async function shutdown(proc: ChildProcess, rpc: RpcClient) {
	if (proc.exitCode !== null || proc.signalCode !== null) return;
	const exited = new Promise<boolean>((resolve) => proc.once("exit", () => resolve(true)));
	const wait = (ms: number) => new Promise<boolean>((resolve) => setTimeout(() => resolve(false), ms).unref());
	await rpc.interrupt();
	void rpc.call("shutdown", {}, 30_000).catch(() => {});
	if (await Promise.race([exited, wait(30_000)])) return;
	proc.stdin?.end();
	if (!(await Promise.race([exited, wait(5_000)]))) proc.kill("SIGKILL");
}

/** The environment of a service process: the services dir on PYTHONPATH, plus `r.env`. */
export function servicesEnv(r: Services): NodeJS.ProcessEnv {
	return { ...process.env, PYTHONPATH: [r.root, process.env.PYTHONPATH].filter(Boolean).join(":"), ...r.env };
}

/** Run `python -c code ...args` in the services dir; the last stdout line is JSON. */
export async function servicesJson<T>(r: Services, code: string, args: string[]): Promise<T> {
	const { stdout } = await promisify(execFile)(r.python, ["-c", code, ...args], {
		cwd: r.root,
		env: servicesEnv(r),
		maxBuffer: 64 << 20,
	});
	return JSON.parse(stdout.trim().split("\n").pop() ?? "") as T;
}

/** Attach to a running service and wait for healthz. */
export async function attach(endpoint: string, readyMs = 300_000): Promise<RpcClient> {
	// A server that requires its RPC token (services/PROTOCOL.md) is attached as `URL#token=HEX`.
	const { url, token } = parseEndpoint(endpoint);
	const rpc = new RpcClient(url);
	if (token) rpc.token = token;
	await rpc.ready(readyMs);
	return rpc;
}
