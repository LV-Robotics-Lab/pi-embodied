/**
 * The pi-embodied MCP server: a robot's primitive manifest as MCP tools for Codex, Claude Code or any
 * MCP client, over stdio.
 *
 *   node --experimental-strip-types src/integrations/mcp/server.ts --robot <name>
 *        [--deployment <name>] [--tier high|low|raw] [--privileged]
 *        [--env URL[#token=HEX] | --serve [-- <env server args>]] [--serve-module <python module>]
 *        [--capabilities a,b] [--var name=value[,value]]... [--timeout <ms>] [--no-reset]
 *        [--confirm-file <path>] [--list]
 *
 * The deployment config (../../infra/config.ts: ~/.pi/agent/embodied.json, $PI_EMBODIED_CONFIG,
 * <cwd>/.pi/embodied.json; `--deployment` as pi's flag) gives the python, the services tree and the
 * CUDA device the env server is started with (`--serve`: `python -m
 * pi_embodied_services.robots.<robot>.env_server <args> --transport http --port 0`, the robots'
 * own startup, ../../infra/service-process.ts) or the server is attached by URL (`--env`, as pi's
 * `--env-url`). A server for a held arm exits at startup with the hardware lock's refusal
 * (services/.../utils/hardware_lock.py): the start fails and nothing is served. The server's
 * `code.api` must carry this manifest's digest (a server of another version is refused, as pi
 * refuses it) and tells which `requires` are met (./tools.ts). A simulator's env is reset once at
 * connect, as every robot's start does (`--no-reset` leaves it as found); a real arm (./tools.ts
 * REAL_ROBOTS) is never reset at connect, since pi's own start asks the operator before that motion:
 * its `reset` tool is the explicit, gated motion that does it. On a real arm the server itself gates
 * every motion and `reset` (./gate.ts), whatever the host runs: `PI_EMBODIED_MOTION_CONFIRMED=1` in
 * the environment at launch authorises the session, `--confirm-file <path>` takes the operator's
 * per-call tickets; without either every motion is refused.
 *
 * pi stays the entry point for everything beyond tools: units, code mode, VDM, memory, exploration,
 * evaluation and their results are pi's and are not served here (README "Using the robots from
 * Codex or Claude Code"). `--list` prints the tool list as JSON and exits (no server needed).
 */

import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { configProblem, cudaDevice, python, servicesDir } from "../../infra/config.ts";
import type { RpcClient } from "../../infra/rpc.ts";
import { attach, shutdown, startService } from "../../infra/service-process.ts";
import { loadManifest, type Vars } from "../../primitives/manifest.ts";
import { fetchCodeApi } from "../../primitives/registry.ts";
import { CONFIRMED_ENV } from "./gate.ts";
import { McpToolServer, StdioServerTransport } from "./protocol.ts";
import { RobotSession, type SessionOptions } from "./session.ts";
import { capabilitiesFrom, isReal, type McpTier, manifestTools, parseTier, parseVar } from "./tools.ts";

export const VERSION = "0.0.1";
const log = (line: string) => console.error(`[pi-embodied-mcp] ${line}`);

export type Args = {
	robot: string;
	deployment: string;
	tier?: McpTier;
	privileged: boolean;
	env?: string;
	serve: boolean;
	serveModule?: string;
	serveArgs: string[];
	capabilities: string[];
	vars: Vars;
	timeoutMs?: number;
	list: boolean;
	/**
	 * `env.reset` once the server is up: `--no-reset` sets false; unset, a simulator resets (as every
	 * robot's start does) and a real robot never does (`connect`).
	 */
	reset?: boolean;
	/** `PI_EMBODIED_MOTION_CONFIRMED` was set in the environment: a real robot's session is authorised. */
	confirmed: boolean;
	/** `--confirm-file`: the operator's per-call tickets for a real robot (./gate.ts). */
	confirmFile?: string;
};

export function parseArgs(argv: readonly string[], env: NodeJS.ProcessEnv = process.env): Args {
	const a: Args = {
		robot: "",
		deployment: "",
		privileged: false,
		serve: false,
		serveArgs: [],
		capabilities: [],
		vars: {},
		list: false,
		confirmed: Boolean(env[CONFIRMED_ENV]?.trim()),
	};
	const vars: Record<string, string | readonly string[]> = {};
	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		const next = () => {
			const v = argv[++i];
			if (v === undefined) throw new Error(`${arg} needs a value`);
			return v;
		};
		switch (arg) {
			case "--robot":
				a.robot = next();
				break;
			case "--deployment":
				a.deployment = next();
				break;
			case "--tier":
				a.tier = parseTier(next());
				break;
			case "--privileged":
				a.privileged = true;
				break;
			case "--env":
				a.env = next();
				break;
			case "--serve":
				a.serve = true;
				break;
			case "--serve-module":
				a.serveModule = next();
				break;
			case "--capabilities":
				a.capabilities.push(
					...next()
						.split(",")
						.map((s) => s.trim())
						.filter(Boolean),
				);
				break;
			case "--var":
				parseVar(next(), vars);
				break;
			case "--timeout":
				a.timeoutMs = Number(next());
				if (!Number.isFinite(a.timeoutMs) || a.timeoutMs <= 0) throw new Error("--timeout: milliseconds > 0");
				break;
			case "--list":
				a.list = true;
				break;
			case "--no-reset":
				a.reset = false;
				break;
			case "--confirm-file":
				a.confirmFile = next();
				break;
			case "--":
				a.serveArgs = argv.slice(i + 1);
				i = argv.length;
				break;
			default:
				throw new Error(`unknown argument ${arg}`);
		}
	}
	if (!a.robot) throw new Error("--robot <name> is required");
	if (a.env && a.serve) throw new Error("--env and --serve are exclusive: attach to a server or start one");
	a.vars = vars;
	return a;
}

/** The deployment config reads `pi.getFlag("deployment")`; this is the one flag it needs here. */
export const configReader = (deployment: string) =>
	({ getFlag: (name: string) => (name === "deployment" ? deployment : undefined) }) as unknown as ExtensionAPI;

export type Connected = {
	rpc: RpcClient;
	pid?: number;
	served?: string[];
	/** Stop the env server this process started (undefined when attached). */
	close?: () => Promise<void>;
};

/** Attach (`--env`) or start (`--serve`) the env server, check its manifest and read its registry. */
export async function connect(a: Args, manifest = loadManifest(a.robot)): Promise<Connected> {
	const pi = configReader(a.deployment);
	const problem = configProblem(pi);
	if (problem) throw new Error(problem);
	let rpc: RpcClient;
	let close: (() => Promise<void>) | undefined;
	if (a.env) rpc = await attach(a.env);
	else if (a.serve) {
		const services = servicesDir(pi);
		const cuda = cudaDevice(pi);
		const owned = startService({
			python: python(pi, a.robot),
			args: ["-m", a.serveModule ?? `pi_embodied_services.robots.${a.robot}.env_server`, ...a.serveArgs],
			cwd: services,
			env: { ...process.env, PYTHONPATH: services, ...(cuda ? { CUDA_VISIBLE_DEVICES: cuda } : {}) },
			log: (port) => join(tmpdir(), `pi-embodied-mcp-${a.robot}-${port}.log`),
		});
		try {
			const logFile = await owned.ready;
			log(`env server started (pid ${owned.proc.pid}); log ${logFile}`);
		} catch (err) {
			// A busy arm (hardware lock, exit 3) or any other startup failure: nothing is served.
			await shutdown(owned.proc, owned.rpc).catch(() => {});
			throw err;
		}
		rpc = owned.rpc;
		close = () => shutdown(owned.proc, owned.rpc);
	} else throw new Error("give --env URL[#token=HEX] (a running env server) or --serve [-- <env server args>]");
	const health = await rpc.call<{ pid?: number; service?: string }>("healthz", {}, 10_000);
	const api = await fetchCodeApi(rpc);
	if (api && api.manifest_digest !== manifest.digest) {
		await close?.();
		throw new Error(
			`env server ${health.service ?? ""} runs manifest ${api.manifest_digest.slice(0, 12)}, this checkout has ${manifest.digest.slice(0, 12)}: update one side`,
		);
	}
	if (!api) log("env server serves no code.api: only --capabilities decides which `requires` are met");
	// A simulator's start resets the env before the first observation (it has none until then). The
	// same here, unless --no-reset. A real arm's reset is a motion (the UR5e opens the gripper and
	// moves to its begin pose) that pi's start lets the operator confirm first: never at connect,
	// only through the session's `reset` tool, which the operator gate holds like every motion.
	const reset = a.reset ?? !isReal(a.robot);
	if (reset) {
		try {
			await rpc.call("env.reset", {}, 600_000);
		} catch (err) {
			await close?.();
			throw new Error(
				`env.reset failed: ${err instanceof Error ? err.message : err} (--no-reset leaves the env as found)`,
			);
		}
	} else if (isReal(a.robot)) log(`${a.robot} is a real robot: the env is left as found (its reset tool resets it)`);
	return { rpc, pid: health.pid, served: api?.available, close };
}

/** The session for parsed args and a connected server (tests build one on a fake server). */
export function session(a: Args, c: Connected, o: Partial<SessionOptions> = {}): RobotSession {
	const s = new RobotSession({
		robot: a.robot,
		manifest: loadManifest(a.robot),
		rpc: c.rpc,
		tier: a.tier,
		privileged: a.privileged,
		capabilities: a.capabilities,
		served: c.served,
		pid: c.pid,
		vars: a.vars,
		timeoutMs: a.timeoutMs,
		confirm: { session: a.confirmed, file: a.confirmFile },
		version: VERSION,
		log,
		...o,
	});
	for (const l of s.leftOut) log(`tool ${l.name} left out: ${l.reason} (give --var)`);
	log(s.gate.describe());
	const warning = s.gate.warning();
	if (warning) log(`WARNING: ${warning}`);
	return s;
}

/** `--list`: the tools this configuration would serve, with no server (requires are --capabilities' only). */
export function listOnly(a: Args) {
	const m = loadManifest(a.robot);
	const has = capabilitiesFrom(m, undefined, a.capabilities, a.privileged);
	const { tools, leftOut } = manifestTools(m, { tier: a.tier, privileged: a.privileged }, has, a.vars);
	return { robot: a.robot, manifest_digest: m.digest, tools: tools.map((t) => t.tool), left_out: leftOut };
}

export async function main(argv = process.argv.slice(2)): Promise<void> {
	const a = parseArgs(argv);
	if (a.list) {
		process.stdout.write(`${JSON.stringify(listOnly(a), null, 2)}\n`);
		return;
	}
	const c = await connect(a);
	const s = session(a, c);
	log(`serving ${s.list().length} tools for ${a.robot} on ${c.rpc.url}`);
	const transport = new StdioServerTransport();
	const server = new McpToolServer(s, transport);
	let closing = false;
	const close = async () => {
		if (closing) return;
		closing = true;
		await c.close?.().catch((err) => log(`env server shutdown: ${err}`));
		process.exit(0);
	};
	transport.onClose(() => void close());
	process.on("SIGINT", () => void close());
	process.on("SIGTERM", () => void close());
	await server.start();
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
	main().catch((err) => {
		log(err instanceof Error ? err.message : String(err));
		process.exit(1);
	});
