/**
 * Preflight check for a robot run (Show-Harness scripts/check_setup.py): read-only, nothing moves.
 *
 *   node packages/embodied/src/check.ts libero [--python P] [--services DIR] [--units] \
 *     [--model provider/id] [--dashboard-port 8779] [--vla URL] [--sam3 URL] [--endpoint name=URL ...]
 *   pi -e packages/embodied/src/libero ...   then /robot-check   (every robot: ./robot.ts registers it)
 *
 * Checks, per robot (SPECS): the services Python (its version, and importing the robot's env server
 * module and simulator packages the way the env server will), the checkpoint / asset paths the
 * services read from the environment, GPU visibility (nvidia-smi and CUDA_VISIBLE_DEVICES), the model
 * servers the robot attaches to (the services' RPC `healthz`, then a real read-only request where the
 * server has one: an env server's `get_env_meta`; a ws:// server by TCP connect; no VLA server offers
 * a dry-run act, so a VLA gets healthz only), the planner (a real 1-token completion: in pi through
 * its model registry, `llmCheck`; standalone to an OpenAI-compatible provider of ~/.pi/agent/models.json
 * whose key resolves, else `<baseUrl>/models`) and whether the dashboard port is free. Flags take the robot's own names and fall back to the same
 * environment variables and defaults as the robot. Model servers the robot only needs for its own
 * tools (VLA, SAM3) are warnings with --units=true, whose `act` tool does not use them.
 *
 * Prints a PASS / WARN / FAIL / SKIP table; the exit code is 1 when any check FAILs.
 */

import { execFile } from "node:child_process";
import { existsSync, readFileSync, statSync } from "node:fs";
import { createServer, Socket } from "node:net";
import { homedir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { Api, Model } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

const SERVICES = fileURLToPath(new URL("../../../services", import.meta.url));

export type Status = "PASS" | "WARN" | "FAIL" | "SKIP";
export type Row = { status: Status; check: string; detail: string };
export type Flags = Record<string, string | true>;

/** A path the services read from the environment. */
type PathSpec = {
	env: string;
	/** The robot flag that overrides the variable. */
	flag?: string;
	kind: "file" | "dir";
	required: boolean;
	why: string;
	fallback?: string;
	contains?: string;
};
/**
 * A server the robot attaches to: `flag` (and its default) names the endpoint. `calls`: read-only
 * methods sent after healthz, each a real request the robot makes too (an env server's `get_env_meta`).
 */
type EndpointSpec = { flag: string; default?: string; why: string; toolsOnly?: boolean; calls?: string[] };
/** What a running env server is asked beyond healthz. */
const ENV_CALLS = ["get_env_meta"];

export type RobotCheckSpec = {
	/** The flag naming the services Python, and the environment variables it defaults to. */
	python: { flag: string; env: string[] };
	/** Modules imported with PYTHONPATH=<services>, as the env server imports them. */
	imports: string[];
	/** Packages only located (importlib.util.find_spec), where importing needs a running app (Isaac Sim). */
	find?: string[];
	/** Environment the robot gives its env server. */
	pyEnv?: Record<string, string>;
	paths?: PathSpec[];
	endpoints?: EndpointSpec[];
	/** Needs a GPU for rendering or models. */
	gpu: boolean;
};

const PY = (...env: string[]) => ({ flag: "python", env: [...env, "PI_EMBODIED_PYTHON"] });
const ENV_SERVER = (robot: string) => `pi_embodied_services.robots.${robot}.env_server`;

export const SPECS: Record<string, RobotCheckSpec> = {
	libero: {
		python: PY(),
		imports: [ENV_SERVER("libero"), "rlinf.envs.libero.libero_env", "mujoco"],
		pyEnv: { MUJOCO_GL: "egl", ROBOT_PLATFORM: "LIBERO" },
		paths: [
			{ env: "PI05_CHECKPOINT_PATH", kind: "dir", required: false, why: "Pi0.5 VLA server (serve.sh)" },
			{ env: "SAM3_CHECKPOINT_PATH", kind: "file", required: false, why: "SAM3 server (serve.sh)" },
		],
		endpoints: [
			{ flag: "vla", default: "http://127.0.0.1:18200", why: "Pi0.5 VLA (pi0_pick)", toolsOnly: true },
			{ flag: "sam3", default: "http://127.0.0.1:18300", why: "SAM3 (segment)", toolsOnly: true },
		],
		gpu: true,
	},
	maniskill: {
		python: PY(),
		imports: [ENV_SERVER("maniskill"), "mani_skill.envs", "sapien"],
		paths: [{ env: "VK_ICD_FILENAMES", kind: "file", required: false, why: "SAPIEN's Vulkan renderer" }],
		gpu: true,
	},
	metaworld: {
		python: PY(),
		imports: [ENV_SERVER("metaworld"), "metaworld.env_dict", "mujoco"],
		pyEnv: { MUJOCO_GL: "egl" },
		gpu: true,
	},
	robosuite: {
		python: PY(),
		imports: [ENV_SERVER("robosuite"), "robosuite", "mujoco"],
		pyEnv: { MUJOCO_GL: "egl" },
		endpoints: [{ flag: "sam3", default: "http://127.0.0.1:18300", why: "SAM3 (segment)", toolsOnly: true }],
		gpu: true,
	},
	robolab: {
		python: PY(),
		imports: [ENV_SERVER("robolab")],
		find: ["isaaclab", "isaacsim"],
		paths: [
			{
				env: "ROBOLAB_ROOT",
				kind: "dir",
				required: true,
				why: "RoboLab checkout",
				fallback: join(homedir(), "RoboLab"),
				contains: "robolab",
			},
			{
				env: "ROBOLAB_ISAAC_ASSETS",
				kind: "dir",
				required: false,
				why: "local Isaac assets (else fetched from S3)",
			},
		],
		pyEnv: { OMNI_KIT_ACCEPT_EULA: "YES" },
		gpu: true,
	},
	robodojo: {
		python: PY(),
		imports: [ENV_SERVER("robodojo")],
		find: ["isaaclab", "isaacsim", "curobo"],
		paths: [
			{
				env: "ROBODOJO_ROOT",
				kind: "dir",
				required: true,
				why: "RoboDojo checkout (patched by robodojo-isaac61.patch) with its Assets/",
				fallback: join(homedir(), "RoboDojo"),
				contains: "Assets",
			},
		],
		pyEnv: { OMNI_KIT_ACCEPT_EULA: "YES" },
		gpu: true,
	},
	robocasa: {
		python: { flag: "robocasa-python", env: ["ROBOCASA_PYTHON", "PI_EMBODIED_PYTHON"] },
		imports: [ENV_SERVER("robocasa"), "robocasa", "robosuite"],
		endpoints: [{ flag: "rldx", default: "http://127.0.0.1:18500", why: "RLDX-1 VLA" }],
		gpu: true,
	},
	robotwin: {
		python: PY(),
		imports: [ENV_SERVER("robotwin")],
		find: ["sapien", "robotwin"],
		paths: [{ env: "ROBOTWIN_ASSETS_PATH", flag: "assets", kind: "dir", required: true, why: "RoboTwin assets" }],
		endpoints: [{ flag: "lingbot", default: "ws://127.0.0.1:18400", why: "LingBot-VLA", toolsOnly: true }],
		gpu: true,
	},
	franka: {
		python: PY(),
		imports: [ENV_SERVER("franka")],
		endpoints: [
			{ flag: "robot-env", why: "running env server", calls: ENV_CALLS },
			{ flag: "robot-vla", why: "Pi0.5 VLA", toolsOnly: true },
		],
		gpu: false,
	},
	dual_franka: {
		python: PY(),
		imports: [ENV_SERVER("dual_franka")],
		endpoints: [
			{ flag: "robot-env", why: "running env server", calls: ENV_CALLS },
			{ flag: "robot-vla", why: "Pi0.5 VLA", toolsOnly: true },
			{ flag: "robot-sam3", why: "SAM3", toolsOnly: true },
		],
		gpu: false,
	},
	piper: {
		python: PY(),
		imports: [ENV_SERVER("piper")],
		endpoints: [{ flag: "robot-env", why: "running env server", calls: ENV_CALLS }],
		gpu: false,
	},
};

/** `--name value` / `--name=value` / `--name` (true) from an argv; later ones win. Positional words are skipped. */
export function parseFlags(argv: readonly string[]): Flags {
	const out: Flags = {};
	for (let i = 0; i < argv.length; i++) {
		const m = /^--([\w-]+)(?:=(.*))?$/.exec(argv[i]);
		if (!m) continue;
		if (m[2] !== undefined) out[m[1]] = m[2];
		else if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) out[m[1]] = argv[++i];
		else out[m[1]] = true;
	}
	return out;
}

const str = (v: string | true | undefined) => (typeof v === "string" ? v : undefined);
const clip = (s: string, n = 160) => (s.length > n ? `${s.slice(0, n - 1)}~` : s);
const run = (cmd: string, args: string[], o: { env?: NodeJS.ProcessEnv; cwd?: string; timeoutMs: number }) =>
	new Promise<{ code: number | null; stdout: string; stderr: string; error?: Error & { killed?: boolean } }>(
		(resolve) => {
			execFile(
				cmd,
				args,
				{ env: o.env, cwd: o.cwd, timeout: o.timeoutMs, maxBuffer: 16 << 20, killSignal: "SIGKILL" },
				(error, stdout, stderr) =>
					resolve({
						code: error ? (typeof error.code === "number" ? error.code : null) : 0,
						stdout: String(stdout),
						stderr: String(stderr),
						error: error ?? undefined,
					}),
			);
		},
	);

const IMPORT_CHECK = `
import importlib, importlib.util, json, sys, time
mods, finds = json.loads(sys.argv[1]), json.loads(sys.argv[2])
out = {"python": sys.version.split()[0], "executable": sys.executable, "imports": {}, "find": {}}
for m in mods:
    t = time.time()
    try:
        importlib.import_module(m)
        out["imports"][m] = [True, "%.1f s" % (time.time() - t)]
    except BaseException as e:
        out["imports"][m] = [False, "%s: %s" % (type(e).__name__, e)]
for m in finds:
    try:
        spec = importlib.util.find_spec(m)
        out["find"][m] = [spec is not None, (spec.origin or "namespace") if spec else "not found"]
    except BaseException as e:
        out["find"][m] = [False, "%s: %s" % (type(e).__name__, e)]
print("\\n" + json.dumps(out))
`;

type ImportReport = {
	python: string;
	executable: string;
	imports: Record<string, [boolean, string]>;
	find: Record<string, [boolean, string]>;
};

async function pythonRows(spec: RobotCheckSpec, flags: Flags, timeoutMs: number): Promise<Row[]> {
	const python = str(flags[spec.python.flag]) ?? spec.python.env.map((k) => process.env[k]).find(Boolean) ?? "python";
	const services = str(flags.services) ?? process.env.PI_EMBODIED_SERVICES ?? SERVICES;
	if (!existsSync(join(services, "pi_embodied_services")))
		return [{ status: "FAIL", check: "services", detail: `${services} has no pi_embodied_services package` }];
	const rows: Row[] = [{ status: "PASS", check: "services", detail: services }];
	const env = {
		...process.env,
		PYTHONPATH: [services, process.env.PYTHONPATH].filter(Boolean).join(":"),
		...spec.pyEnv,
	};
	const r = await run(python, ["-c", IMPORT_CHECK, JSON.stringify(spec.imports), JSON.stringify(spec.find ?? [])], {
		env,
		cwd: services,
		timeoutMs,
	});
	let report: ImportReport | undefined;
	try {
		report = JSON.parse(r.stdout.trim().split("\n").pop() ?? "");
	} catch {}
	if (!report) {
		const why = r.error?.message.includes("ENOENT")
			? `${python} not found`
			: r.error?.killed
				? `timed out after ${timeoutMs / 1000} s`
				: clip((r.stderr || r.stdout || String(r.error)).trim().split("\n").pop() ?? "");
		return [...rows, { status: "FAIL", check: "python", detail: `${python}: ${why}` }];
	}
	const [major, minor] = report.python.split(".").map(Number);
	rows.push({
		status: major === 3 && minor >= 10 ? "PASS" : "FAIL",
		check: "python",
		detail: `${report.executable} (${report.python})`,
	});
	for (const [m, [ok, detail]] of Object.entries(report.imports))
		rows.push({ status: ok ? "PASS" : "FAIL", check: `import ${m}`, detail: clip(detail) });
	for (const [m, [ok, detail]] of Object.entries(report.find))
		rows.push({ status: ok ? "PASS" : "FAIL", check: `find ${m}`, detail: clip(detail) });
	return rows;
}

export function pathRows(paths: readonly PathSpec[], flags: Flags = {}, env: NodeJS.ProcessEnv = process.env): Row[] {
	return paths.map((p) => {
		const given = (p.flag ? str(flags[p.flag]) : undefined) || env[p.env];
		const value = given || p.fallback;
		const check = p.flag && str(flags[p.flag]) ? `--${p.flag}` : `$${p.env}`;
		const miss = (detail: string): Row => ({ status: p.required ? "FAIL" : "WARN", check, detail });
		if (!value) return miss(`not set (${p.why})`);
		let st: ReturnType<typeof statSync> | undefined;
		try {
			st = statSync(value);
		} catch {}
		if (!st) return miss(`${value} does not exist (${p.why})`);
		if (p.kind === "dir" ? !st.isDirectory() : !st.isFile()) return miss(`${value} is not a ${p.kind} (${p.why})`);
		if (p.contains && !existsSync(join(value, p.contains))) return miss(`${value} has no ${p.contains}/ (${p.why})`);
		return { status: "PASS", check, detail: `${value}${given ? "" : " (default)"}` };
	});
}

async function gpuRows(needed: boolean): Promise<Row[]> {
	const r = await run(
		"nvidia-smi",
		["--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"],
		{ timeoutMs: 20_000 },
	);
	const visible = process.env.CUDA_VISIBLE_DEVICES;
	if (r.code !== 0)
		return [
			{
				status: needed ? "FAIL" : "SKIP",
				check: "gpu",
				detail: r.error?.message.includes("ENOENT")
					? "nvidia-smi not found"
					: clip(r.stderr.trim() || "nvidia-smi failed"),
			},
		];
	const gpus = r.stdout
		.trim()
		.split("\n")
		.map((l) => l.split(",").map((s) => s.trim()))
		.map(([index, name, used, total]) => ({ index, name, free: Number(total) - Number(used), total: Number(total) }));
	const ids =
		visible === undefined
			? gpus.map((g) => g.index)
			: visible
					.split(",")
					.map((s) => s.trim())
					.filter(Boolean);
	const shown = gpus.filter((g) => ids.includes(g.index));
	const describe = shown.map(
		(g) => `${g.index}: ${g.name}, ${(g.free / 1024).toFixed(1)} of ${(g.total / 1024).toFixed(1)} GB free`,
	);
	if (!shown.length)
		return [
			{
				status: needed ? "FAIL" : "WARN",
				check: "gpu",
				detail: `CUDA_VISIBLE_DEVICES=${visible ?? ""} selects none of ${gpus.map((g) => g.index).join(",")}`,
			},
		];
	return [
		{
			status: "PASS",
			check: "gpu",
			detail: `${visible === undefined ? "all visible" : `CUDA_VISIBLE_DEVICES=${visible}`}; ${describe.join("; ")}`,
		},
	];
}

/** One services RPC call (`POST <base>/call`): its result, or why it failed. */
async function rpcCall(
	base: string,
	method: string,
	timeoutMs: number,
): Promise<{ ok: true; result: unknown; ms: number } | { ok: false; detail: string }> {
	const t0 = Date.now();
	try {
		const res = await fetch(`${base.replace(/\/$/, "")}/call`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ method, kwargs: {}, args: [] }),
			signal: AbortSignal.timeout(timeoutMs),
		});
		const text = await res.text();
		let body: { ok?: unknown; result?: unknown; error?: unknown } | undefined;
		try {
			body = JSON.parse(text);
		} catch {}
		if (res.ok && body?.ok === true) return { ok: true, result: body.result, ms: Date.now() - t0 };
		return { ok: false, detail: `HTTP ${res.status} ${clip(String(body?.error ?? text), 100)}` };
	} catch (err) {
		const e = err as Error & { cause?: { code?: string } };
		return { ok: false, detail: e.cause?.code ?? e.message };
	}
}

/**
 * A services RPC server's healthz (`POST <url>/call {"method":"healthz"}`), then each of `calls` (read-only
 * methods, e.g. an env server's `get_env_meta`); or a TCP connect for ws:// servers.
 */
export async function probeEndpoint(
	endpoint: string,
	timeoutMs = 3000,
	calls: readonly string[] = [],
): Promise<{ ok: boolean; detail: string }> {
	const base = endpoint.includes("://") ? endpoint : `http://${endpoint}`;
	const url = new URL(base);
	if (url.protocol === "ws:" || url.protocol === "wss:") {
		const port = Number(url.port || (url.protocol === "wss:" ? 443 : 80));
		return new Promise((resolve) => {
			const sock = new Socket();
			const done = (ok: boolean, detail: string) => {
				sock.destroy();
				resolve({ ok, detail });
			};
			sock.setTimeout(timeoutMs, () => done(false, "connect timed out"));
			sock.once("error", (e) => done(false, e.message));
			sock.connect(port, url.hostname, () => done(true, "accepts connections"));
		});
	}
	const health = await rpcCall(base, "healthz", timeoutMs);
	if (!health.ok) return health;
	const parts = [`healthz ok ${clip(JSON.stringify(health.result ?? ""), 80)}`];
	for (const method of calls) {
		// A real request can take longer than healthz (the env server answers between env steps).
		const r = await rpcCall(base, method, Math.max(timeoutMs, 10_000));
		if (!r.ok) return { ok: false, detail: `${parts.join("; ")}; ${method} failed: ${r.detail}` };
		parts.push(`${method} ok in ${r.ms} ms ${clip(JSON.stringify(r.result ?? ""), 80)}`);
	}
	return { ok: true, detail: parts.join("; ") };
}

async function endpointRows(spec: RobotCheckSpec, flags: Flags): Promise<Row[]> {
	const units = flags.units === true || flags.units === "true" || flags.units === "pure";
	const named = (str(flags.endpoint) ?? "").split(",").filter((s) => s.includes("="));
	const wanted = [
		...(spec.endpoints ?? []).map((e) => ({ ...e, url: str(flags[e.flag]) ?? e.default })),
		...(str(flags.env)
			? [{ flag: "env", why: "running env server", url: str(flags.env), toolsOnly: false, calls: ENV_CALLS }]
			: []),
		...named.map((s) => ({
			flag: s.slice(0, s.indexOf("=")),
			why: "extra endpoint",
			url: s.slice(s.indexOf("=") + 1),
			toolsOnly: false,
		})),
	];
	return Promise.all(
		wanted.map(async (e): Promise<Row> => {
			const check = `--${e.flag}`;
			if (!e.url) return { status: "SKIP", check, detail: `not set (${e.why})` };
			const r = await probeEndpoint(e.url, 3000, "calls" in e ? (e.calls ?? []) : []);
			if (r.ok) return { status: "PASS", check, detail: `${e.url} ${r.detail}` };
			const soft = e.toolsOnly && units;
			return {
				status: soft ? "WARN" : "FAIL",
				check,
				detail: `${e.url} unreachable: ${r.detail} (${e.why}${soft ? "; not used by --units=true" : ""})`,
			};
		}),
	);
}

/** The outcome of one real minimal completion: whether the model answered, its latency, and the reply or error. */
export type LlmCheck = { ok: boolean; model: string; ms: number; detail: string };

/**
 * Send `model` one real minimal completion (at most one output token) through pi's model registry, the
 * way the agent's own requests go (provider, key, headers), and report whether it answered and how fast.
 */
export async function llmCheck(
	registry: ExtensionContext["modelRegistry"],
	model: Model<Api>,
	timeoutMs = 30_000,
): Promise<LlmCheck> {
	const name = `${model.provider}/${model.id}`;
	const t0 = Date.now();
	try {
		const reply = await registry
			.streamSimple(
				model,
				{ messages: [{ role: "user", content: "Reply with the single word OK.", timestamp: Date.now() }] },
				{ maxTokens: 1, signal: AbortSignal.timeout(timeoutMs) },
			)
			.result();
		const ms = Date.now() - t0;
		if (reply.stopReason === "error" || reply.stopReason === "aborted")
			return { ok: false, model: name, ms, detail: clip(reply.errorMessage ?? reply.stopReason, 300) };
		const text = reply.content
			.map((c) => (c.type === "text" ? c.text : ""))
			.join("")
			.trim();
		return { ok: true, model: name, ms, detail: `answered in ${ms} ms${text ? ` ("${clip(text, 40)}")` : ""}` };
	} catch (err) {
		return {
			ok: false,
			model: name,
			ms: Date.now() - t0,
			detail: clip(err instanceof Error ? err.message : String(err), 300),
		};
	}
}

type ProviderConfig = { baseUrl?: string; api?: string; apiKey?: string };

/** A models.json `apiKey`: the named environment variable's value, else the literal; a `!command` key is not run here. */
function resolveKey(key: string | undefined, env: NodeJS.ProcessEnv): string | undefined {
	if (!key || key.startsWith("!")) return undefined;
	return env[key] ?? key;
}

/**
 * The planner. In pi (`complete`, ../robot.ts's /robot-check): one real 1-token completion through pi's
 * model registry (`llmCheck`). Standalone: the provider from models.json (`--model provider/id`); an
 * OpenAI-compatible one whose key resolves gets a real 1-token `chat/completions` request, any other
 * a `GET <baseUrl>/models` (any answer below 500 means the gateway is up).
 */
export async function plannerRow(
	model: string | undefined,
	baseUrl?: string,
	modelsJson = join(process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent"), "models.json"),
	complete?: () => Promise<LlmCheck>,
	env: NodeJS.ProcessEnv = process.env,
): Promise<Row> {
	if (complete) {
		const r = await complete();
		return { status: r.ok ? "PASS" : "FAIL", check: "planner", detail: `${r.model}: ${r.detail}` };
	}
	if (!model) return { status: "SKIP", check: "planner", detail: "no --model given" };
	const provider = model.split("/")[0];
	let cfg: ProviderConfig | undefined;
	try {
		cfg = (JSON.parse(readFileSync(modelsJson, "utf8")) as { providers?: Record<string, ProviderConfig> })
			.providers?.[provider];
	} catch {}
	const url = baseUrl ?? cfg?.baseUrl;
	if (!url)
		return { status: "SKIP", check: "planner", detail: `${model}: no baseUrl for "${provider}" in ${modelsJson}` };
	const base = url.replace(/\/$/, "");
	const key = resolveKey(cfg?.apiKey, env);
	try {
		if (key && (cfg?.api ?? "openai-completions") === "openai-completions") {
			const t0 = Date.now();
			const res = await fetch(`${base}/chat/completions`, {
				method: "POST",
				headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
				body: JSON.stringify({
					model: model.slice(provider.length + 1),
					messages: [{ role: "user", content: "Reply with the single word OK." }],
					max_tokens: 1,
				}),
				signal: AbortSignal.timeout(30_000),
			});
			const text = await res.text();
			const ms = Date.now() - t0;
			return {
				status: res.ok ? "PASS" : "FAIL",
				check: "planner",
				detail: `${model} via ${url}: ${res.ok ? `1-token completion in ${ms} ms` : `HTTP ${res.status} ${clip(text, 120)}`}`,
			};
		}
		const res = await fetch(`${base}/models`, { signal: AbortSignal.timeout(5000) });
		await res.body?.cancel();
		return {
			status: res.status < 500 ? "PASS" : "FAIL",
			check: "planner",
			detail: `${model} via ${url}: HTTP ${res.status}${res.status === 401 || res.status === 403 ? " (up; needs the key)" : ""} (no key: gateway only, no completion sent)`,
		};
	} catch (err) {
		const e = err as Error & { cause?: { code?: string } };
		return { status: "FAIL", check: "planner", detail: `${model} via ${url}: ${e.cause?.code ?? e.message}` };
	}
}

export function portRow(port: number, host = "127.0.0.1"): Promise<Row> {
	const check = "dashboard port";
	if (!(port > 0)) return Promise.resolve({ status: "SKIP", check, detail: "--dashboard-port 0 (any free port)" });
	return new Promise((resolve) => {
		const srv = createServer();
		srv.once("error", (e: NodeJS.ErrnoException) =>
			resolve({
				status: "FAIL",
				check,
				detail: `${host}:${port} ${e.code === "EADDRINUSE" ? "is in use" : e.message}`,
			}),
		);
		srv.listen(port, host, () =>
			srv.close(() => resolve({ status: "PASS", check, detail: `${host}:${port} is free` })),
		);
	});
}

/** Every check for `robot`, in table order. `planner` overrides the models.json lookup (a running pi's model). */
export async function runChecks(
	robot: string,
	flags: Flags,
	o: {
		spec?: RobotCheckSpec;
		planner?: { model: string; baseUrl?: string; complete?: () => Promise<LlmCheck> };
		dashboard?: "skip";
		timeoutMs?: number;
	} = {},
): Promise<Row[]> {
	const spec = o.spec ?? SPECS[robot];
	if (!spec)
		return [
			{
				status: "FAIL",
				check: "robot",
				detail: `unknown robot "${robot}"; one of ${Object.keys(SPECS).join(", ")}`,
			},
		];
	const port = Number(str(flags["dashboard-port"]) ?? 0);
	const [py, gpu, endpoints, planner, dash] = await Promise.all([
		pythonRows(spec, flags, o.timeoutMs ?? Number(str(flags.timeout) ?? 300) * 1000),
		gpuRows(spec.gpu),
		endpointRows(spec, flags),
		plannerRow(o.planner?.model ?? str(flags.model), o.planner?.baseUrl, undefined, o.planner?.complete),
		o.dashboard === "skip"
			? Promise.resolve<Row>({ status: "SKIP", check: "dashboard port", detail: "served by this pi" })
			: portRow(port, str(flags["dashboard-host"]) === "0.0.0.0" ? "0.0.0.0" : "127.0.0.1"),
	]);
	return [
		{ status: "PASS", check: "robot", detail: robot },
		...py,
		...pathRows(spec.paths ?? [], flags),
		...gpu,
		...endpoints,
		planner,
		dash,
	];
}

export function formatTable(rows: readonly Row[]): string {
	const w = Math.max(...rows.map((r) => r.check.length));
	const lines = rows.map((r) => `${r.status.padEnd(4)}  ${r.check.padEnd(w)}  ${r.detail}`);
	const n = (s: Status) => rows.filter((r) => r.status === s).length;
	lines.push(
		`${n("FAIL") ? "NOT READY" : "READY"}: ${n("PASS")} pass, ${n("WARN")} warn, ${n("FAIL")} fail, ${n("SKIP")} skip`,
	);
	return lines.join("\n");
}

/** `/robot-check [robot]`: the checks for this pi's robot (default `robot`), with its command-line flags and running model. */
export function robotCheck(pi: ExtensionAPI, robot?: string) {
	pi.registerCommand("robot-check", {
		description: `Preflight check of the robot's services, paths, GPU, model servers and planner: /robot-check [${Object.keys(SPECS).join("|")}]`,
		handler: async (args, ctx: ExtensionContext) => {
			const entry = ctx.sessionManager
				.getBranch()
				.filter((e) => e.type === "custom" && e.customType === "robot_task")
				.pop();
			const named = entry?.type === "custom" ? String((entry.data as { robot?: unknown })?.robot ?? "") : "";
			const which = args.trim() || named || robot;
			if (!which) {
				ctx.ui.notify(`Usage: /robot-check <${Object.keys(SPECS).join("|")}>`, "error");
				return;
			}
			ctx.ui.notify(`Checking ${which}...`, "info");
			const dashboard = (globalThis as Record<symbol, unknown>)[Symbol.for("pi-embodied.dashboard")]
				? "skip"
				: undefined;
			const rows = await runChecks(which, parseFlags(process.argv.slice(2)), {
				planner: ctx.model
					? {
							model: `${ctx.model.provider}/${ctx.model.id}`,
							baseUrl: ctx.model.baseUrl,
							complete: () => llmCheck(ctx.modelRegistry, ctx.model as Model<Api>),
						}
					: undefined,
				dashboard,
			});
			ctx.ui.notify(formatTable(rows), rows.some((r) => r.status === "FAIL") ? "error" : "info");
		},
	});
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
	const robot = process.argv.slice(2).find((a) => !a.startsWith("--"));
	if (!robot || process.argv.includes("--help")) {
		console.log(`usage: node check.ts <${Object.keys(SPECS).join("|")}> [--python P] [--services DIR] [--units]
  [--model provider/id] [--dashboard-port N] [--<endpoint flag> URL] [--endpoint name=URL,...] [--timeout s]`);
		process.exit(robot ? 0 : 2);
	}
	const rows = await runChecks(robot, parseFlags(process.argv.slice(2)));
	console.log(formatTable(rows));
	process.exit(rows.some((r) => r.status === "FAIL") ? 1 : 0);
}
