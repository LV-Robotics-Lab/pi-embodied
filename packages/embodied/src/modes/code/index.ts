/**
 * Code mode (CaP-X's "write code, execute it") for any robot whose env server serves `code.run`
 * over its primitive registry (`code.api`, ../primitives/registry.ts; services/PROTOCOL.md; the
 * shared runner is services/.../utils/code_exec.py).
 *
 *   pi -e packages/embodied/src/robots/libero --code=true --code-api=high --suite libero_spatial --task 0 --seed 0
 *   pi -e packages/embodied/src/robots/libero --code=both ...          (run_code next to the robot's tools)
 *   pi -e packages/embodied/src/robots/libero --code=true --stateless   (the paper's no-history setting)
 *
 * A robot opts in with `code` in its defineRobot spec (the env RPC, and how a run's result becomes
 * the robot's observation); ../robot.ts mounts this module. `--code=true` hides the robot's own
 * tools: only `run_code` and `finish` remain and the system prompt is ./SYSTEM.md, rendered with the
 * registry's declaration of the episode's tier (`--code-api`: high = CaP-X's S2, perception plus
 * pose-level motion; low = S3, relative moves, with the primitives' usage examples;
 * low-noexamples = S4, the same primitives without them; `--privileged` runs the registry's
 * privileged tier, high plus the simulator's ground truth = S1). `--code=both` adds `run_code` to the robot's tools
 * and appends the code section to its prompt. `--units` and `--code` are mutually exclusive.
 *
 * `run_code({code, timeout_s?})` is registered through the robot base's `tool`, so the operator gate,
 * the budgets, GUMI's takeover and pi's abort apply. The program runs in a spawned subprocess on the
 * env server that holds no env object, only stubs whose calls the server resolves through the
 * registry (`CodeApi.resolve`: declared name, parameters and tier) to the facade methods the robot's
 * tools reach, with their limits and stop handling; it is killed at `--code-timeout` (and a stop
 * issued), refused past `--code-max-calls` primitive calls or `--code-max-move` metres of commanded
 * translation, and an abort (`stop` on the server) kills it. `--code-helpers` injects CaP-X's nine
 * numpy helpers (pure computation, listed by the server's `code.helpers`). The result carries stdout,
 * stderr, the traceback, `RESULT`, the primitive log and, like every motion tool, the latest camera
 * images; `details.status` is ran, error or timeout. `--stateless` keeps only the task and the
 * latest observation turn.
 *
 * Real robots refuse code mode unless `--code-real` and `--operator` are both on, and every program
 * is confirmed by the operator (`ui.confirm`) before it runs.
 *
 * `--code-oracle <file>` (simulators, with `--code=true`) runs a human-written reference program
 * instead of asking the model: CaP-X's oracles (`env_configs/human_oracle_code`), ported onto the
 * registry's primitives in `../<robot>/oracle/` (a bare name resolves there). The first prompt runs
 * it once through `code.run`, exactly as a `run_code` call would, and nothing is sent to the model;
 * the result records `code_oracle` (the file) and its sha256. The file's header comments
 * (`# key: value`) name the tier it needs (`tier:`, checked against --code-api / --privileged), the
 * task fields it was written for (checked against the episode's) and an optional `prelude:` file in
 * the same directory that is prepended (the CaP-X API names over the registry's primitives).
 */

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { hostname } from "node:os";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import type { RpcClient } from "../../infra/rpc.ts";
import { template } from "../../planner/context-version.ts";
import type { Manifest, Vars } from "../../primitives/manifest.ts";
import {
	type CodeApi,
	type CodeApiPrimitive,
	type CodeApiTier,
	codePrimitives,
	fetchCodeApi,
} from "../../primitives/registry.ts";
import { latestTurn, type ToolRegistrar } from "../units/index.ts";

/** `--code-api` values (CaP-X's S2, S3, S4); `--privileged` runs the registry's privileged tier (S1) instead. */
export const TIERS = ["high", "low", "low-noexamples", "raw"] as const;
/** The session entry of an oracle run (`--code-oracle`). */
export const ORACLE_ENTRY = "code_oracle";
export type Tier = (typeof TIERS)[number];
/** A helper as the server's `code.helpers` lists it (services/.../utils/code_exec.py). */
export type Helper = { name: string; signature: string; doc: string };
/** `code.run`'s result (services/.../utils/code_exec.py `CodeRunner.run`), plus the robot's `finish` fields. */
export type RunResult = {
	status: "ran" | "error" | "timeout";
	stdout: string;
	stderr: string;
	traceback: string | null;
	error: string | null;
	result: unknown;
	calls: Record<string, unknown>[];
	n_calls: number;
	move_m: number;
	limit?: string;
	cancelled?: boolean;
	stop_issued?: boolean;
	ms: number;
	[key: string]: unknown;
};
type Result = AgentToolResult<unknown>;
type Json = Record<string, unknown>;

export const DEFAULT_TIMEOUT_S = 60;
export const DEFAULT_MAX_CALLS = 50;
export const DEFAULT_MAX_MOVE_M = 3;
/**
 * An oracle run's (`--code-oracle`) caps where the flag is not given: a reference program must not
 * be cut short by the model's defaults (CaP-X's two_arm_lift makes 52 calls, wipe about 83 and 5 m).
 */
export const ORACLE_MAX_CALLS = 1000;
export const ORACLE_MAX_MOVE_M = 50;
export const ORACLE_TIMEOUT_S = 600;

/** A `--code-oracle` program: its file, header fields (`# key: value`) and the code that runs (prelude first). */
export type Oracle = { name: string; path: string; header: Record<string, string>; code: string; sha256: string };

/** Read an oracle file (a path, or a name in `dir`) with its header and prelude; throws when missing. */
export function loadOracle(ref: string, dir: string): Oracle {
	const path = existsSync(ref) ? ref : join(dir, ref.endsWith(".py") ? ref : `${ref}.py`);
	if (!existsSync(path)) throw new Error(`--code-oracle: no oracle ${ref} (looked in ${dir})`);
	const text = readFileSync(path, "utf8");
	const header: Record<string, string> = {};
	for (const line of text.split("\n")) {
		if (!line.startsWith("#")) break;
		const m = /^#\s*([\w-]+):\s*(.*?)\s*$/.exec(line);
		if (m) header[m[1]] = m[2];
	}
	const prelude = header.prelude ? readFileSync(join(dirname(path), header.prelude), "utf8") : "";
	const code = prelude ? `${prelude}\n${text}` : text;
	return { name: basename(path), path, header, code, sha256: createHash("sha256").update(code).digest("hex") };
}

export type CodeSpec = {
	/** The env server that serves `code.api` / `code.run` (up after the robot's `start`). */
	rpc: () => Pick<RpcClient, "call" | "interrupt">;
	/**
	 * Turn a `code.run` result into the robot's observation: absorb the new state (steps, success,
	 * the latest obs, frames) and return the latest camera images and state, as the robot's motion
	 * tools do. The run's own report is prepended by this module.
	 */
	observe: (result: RunResult, signal: AbortSignal | undefined) => Promise<Result>;
	/** Why no program may run now (the episode ended), else undefined. */
	refuse?: () => string | undefined;
	/** The task text for the prompt (default: the episode's task flags). */
	instruction?: () => string;
	/** A real robot: code mode needs --code-real and --operator, and every program is confirmed. */
	real?: boolean;
	/** Default of --code-max-move, m (the accumulated translation one program may command). */
	maxMoveM?: number;
	/** Default of --code-timeout and of a program's timeout_s, s (primitives that take minutes, e.g. BEHAVIOR's). */
	timeoutS?: number;
};

const TEMPLATE = template(new URL("./SYSTEM.md", import.meta.url)).replace(/^<!--[\s\S]*?-->\n/, "");
const text = (s: string) => ({ type: "text" as const, text: s });

/** Whether an endpoint URL is on this host (a spawned server, or one attached on loopback). */
export function isLocalEndpoint(url: string): boolean {
	try {
		const host = new URL(url.includes("://") ? url : `http://${url}`).hostname.replace(/^\[|\]$/g, "");
		return ["127.0.0.1", "localhost", "::1", "0.0.0.0", ""].includes(host) || host === hostname();
	} catch {
		return true;
	}
}

/** Keep every `[name]...[/name]` block when `on`, drop them otherwise. */
function section(prompt: string, name: string, on: boolean) {
	const re = new RegExp(`\\[${name}\\]\\n([\\s\\S]*?)\\[/${name}\\]\\n`, "g");
	return prompt.replace(re, on ? "$1" : "");
}

const indent = (s: string) => s.trim().replace(/^(?=.)/gm, "    ");

/**
 * The registry's primitives as the prompt shows them: a `def` line with the declared parameters
 * (`name: type` required, `name: type = None` optional), the doc, an `Args:` block and whether it
 * moves the robot.
 */
export function renderPrimitives(primitives: CodeApiPrimitive[]): string {
	return primitives
		.map((p) => {
			const params = Object.entries(p.params)
				.map(([k, v]) => `${k}: ${v.type}${v.required ? "" : " = None"}`)
				.join(", ");
			const args = Object.entries(p.params).map(([k, v]) =>
				`    ${k} (${v.type}${v.required ? "" : ", optional"}): ${v.description}`.trimEnd(),
			);
			const body = [
				p.doc.trim(),
				...(args.length ? ["", "Args:", ...args] : []),
				...(p.mutating ? ["", "Moves the robot."] : []),
			];
			return `def ${p.name}(${params}):\n${indent(body.join("\n"))}`;
		})
		.join("\n\n");
}

/** The server's helpers as the prompt shows them: a `def` line and the doc. */
export function renderHelpers(helpers: Helper[]): string {
	return helpers.map((h) => `def ${h.name}${h.signature}:${h.doc.trim() ? `\n${indent(h.doc)}` : ""}`).join("\n\n");
}

/** Register the code-mode flags and `run_code`; `mode()`, `start()`, `tools()`, `prompt()` and `result()` are read by ../robot.ts. */
export function code(
	pi: ExtensionAPI,
	spec: CodeSpec,
	tool: ToolRegistrar,
	task: () => Record<string, string> = () => ({}),
	base: {
		unitsOn: () => boolean;
		privileged: () => boolean;
		/** The robot's name: `--code-oracle <name>` resolves in `../<robot>/oracle/`. */
		robot?: string;
		/** Whether the robot is up (an oracle runs only then). */
		ready?: () => boolean;
		/** An oracle ran instead of the model: the episode ran and is over. */
		oracleRan?: () => void;
		/** The robot's primitive manifest: the prompt is rendered from it, and the server must run the same one. */
		manifest?: Manifest;
		/** Whether the run has a capability a manifest entry requires (the robot's view; the server must agree). */
		has?: (capability: string) => boolean;
		/** The robot's manifest variables (`{{name}}` in the code docs). */
		vars?: () => Vars;
	} = { unitsOn: () => false, privileged: () => false },
) {
	pi.registerFlag("code", {
		type: "string",
		default: "false",
		description: "Code mode (run_code): true = only run_code/finish, both = next to the robot's tools",
	});
	pi.registerFlag("code-api", {
		type: "string",
		default: "high",
		description: `Code mode primitive tier: ${TIERS.join(", ")} (CaP-X's S2, S3, S4; --privileged runs the privileged tier, S1)`,
	});
	pi.registerFlag("code-oracle", {
		type: "string",
		default: "",
		description:
			"Code mode (simulators): run this reference program (a path, or a name in the robot's oracle/ dir) instead of the model",
	});
	pi.registerFlag("code-timeout", {
		type: "string",
		default: "",
		description: `Code mode: the most wall-clock seconds one program may run (default ${spec.timeoutS ?? DEFAULT_TIMEOUT_S}; ${ORACLE_TIMEOUT_S} for --code-oracle)`,
	});
	pi.registerFlag("code-max-calls", {
		type: "string",
		default: "",
		description: `Code mode: primitive calls one program may make (default ${DEFAULT_MAX_CALLS}; ${ORACLE_MAX_CALLS} for --code-oracle)`,
	});
	pi.registerFlag("code-max-move", {
		type: "string",
		default: "",
		description: `Code mode: metres of translation one program may command in total (default ${spec.maxMoveM ?? DEFAULT_MAX_MOVE_M}; ${ORACLE_MAX_MOVE_M} for --code-oracle)`,
	});
	pi.registerFlag("code-helpers", {
		type: "boolean",
		default: false,
		description: "Code mode: inject CaP-X's numpy helpers (pure computation) into the program",
	});
	pi.registerFlag("code-real", {
		type: "boolean",
		default: false,
		description: "Allow code mode on a real robot (with --operator; every program is confirmed first)",
	});
	// Units registers the same flag; both modules honour it (they are mutually exclusive).
	pi.registerFlag("stateless", {
		type: "boolean",
		default: false,
		description: "Units / code mode: keep only the task and the latest observation turn in context",
	});

	/** "pure" (--code / --code=true), "both", or undefined (off). */
	const mode = (): "pure" | "both" | undefined => {
		const v = pi.getFlag("code");
		if (v === true || v === "true" || v === "pure") return "pure";
		return v === "both" ? "both" : undefined;
	};
	/** The registry tier this episode runs: --privileged wins over --code-api. */
	const tier = (): CodeApiTier | string =>
		base.privileged()
			? String(pi.getFlag("code-api") ?? "high") === "high"
				? "privileged"
				: `${String(pi.getFlag("code-api"))}+privileged`
			: String(pi.getFlag("code-api") ?? "high");
	const defaultTimeout = spec.timeoutS ?? DEFAULT_TIMEOUT_S;
	/** A budget flag as given (empty when not): eval.sh keys its configuration on these. */
	const rawFlag = (name: string) => String(pi.getFlag(name) ?? "").trim();
	// An oracle run (--code-oracle) relaxes every cap its flag does not set.
	const oracleOn = () => !!String(pi.getFlag("code-oracle") ?? "").trim();
	const timeoutCap = () =>
		Number(rawFlag("code-timeout")) || (oracleOn() ? Math.max(ORACLE_TIMEOUT_S, defaultTimeout) : defaultTimeout);
	const maxCalls = () =>
		Math.max(1, Math.floor(Number(rawFlag("code-max-calls")) || (oracleOn() ? ORACLE_MAX_CALLS : DEFAULT_MAX_CALLS)));
	const maxMove = () => {
		const v = Number(rawFlag("code-max-move"));
		if (Number.isFinite(v) && v > 0) return v;
		return oracleOn() ? ORACLE_MAX_MOVE_M : (spec.maxMoveM ?? DEFAULT_MAX_MOVE_M);
	};
	const helpersOn = () => pi.getFlag("code-helpers") === true;
	const oracleRef = () => String(pi.getFlag("code-oracle") ?? "").trim();
	const oracleDir = () => fileURLToPath(new URL(`../../robots/${base.robot ?? "_"}/oracle/`, import.meta.url));
	const instruction = () =>
		spec.instruction?.() ||
		Object.entries(task())
			.map(([k, v]) => `${k} ${v}`)
			.join(", ");

	/** The registry's declaration for this session's tier and the helpers (fetched at `start`). */
	let api: CodeApi | undefined;
	/** The tier's primitives as the prompt shows them (from the manifest, checked against the server). */
	let primitives: CodeApiPrimitive[] = [];
	let helpers: Helper[] = [];
	let registered = "";

	function registerTool() {
		const names = primitives.map((p) => p.name);
		const description = `Execute a Python program on the robot (code mode, ${tier()} tier). It may call the primitives ${names.join(", ")}${helpersOn() ? " and the numpy helpers" : ""}; assign RESULT to report a value. Returns stdout, stderr, the traceback, RESULT, the primitive call log and the new camera images and state. Limits: ${timeoutCap()} s, ${maxCalls()} primitive calls, ${maxMove()} m of translation per call.`;
		if (description === registered) return;
		registered = description;
		tool(
			"run_code",
			description,
			Type.Object({
				code: Type.String({ description: "The Python program" }),
				timeout_s: Type.Optional(
					Type.Number({
						minimum: 1,
						description: `Wall-clock timeout, s (default ${Math.min(defaultTimeout, timeoutCap())}, at most ${timeoutCap()})`,
					}),
				),
			}),
			runCode,
		);
	}

	async function runCode(
		params: { code: string; timeout_s?: number },
		signal: AbortSignal | undefined,
		ctx: ExtensionContext,
	): Promise<Result> {
		const refused = spec.refuse?.();
		if (refused) return { content: [text(refused)], details: { status: "error", error: refused } };
		const cap = timeoutCap();
		const timeout_s = Math.min(params.timeout_s ?? Math.min(defaultTimeout, cap), cap);
		if (spec.real) {
			// A real robot: the operator reads the program before it moves anything.
			const go = ctx.hasUI ? await ctx.ui.confirm("Run this program on the robot?", params.code) : false;
			if (!go) {
				const why = ctx.hasUI
					? "the operator declined this program; it did not run"
					: "no operator UI to confirm the program; it did not run";
				return { content: [text(why)], details: { status: "error", error: why } };
			}
		}
		// An abort stops the server (its stop kills the program), but a sent call itself is waited
		// out: the run's result carries the steps it took and any success before the stop, which
		// an abandoned call would lose. A call still queued behind another one is never sent.
		const rpc = spec.rpc();
		if (signal?.aborted) return { content: [text("run_code: aborted before it ran")], details: { status: "error" } };
		let sent = false;
		const unsent = new AbortController();
		const onAbort = () => {
			void rpc.interrupt();
			if (!sent) unsent.abort();
		};
		signal?.addEventListener("abort", onAbort, { once: true });
		let r: RunResult;
		try {
			r = await rpc.call<RunResult>(
				"code.run",
				{
					code: params.code,
					timeout_s,
					tier: tier(),
					max_calls: maxCalls(),
					max_move_m: maxMove(),
					helpers: helpersOn(),
				},
				(timeout_s + 60) * 1000,
				[],
				unsent.signal,
				() => {
					sent = true;
				},
			);
		} catch (err) {
			if (!sent && unsent.signal.aborted)
				return { content: [text("run_code: aborted before it ran")], details: { status: "error" } };
			throw err;
		} finally {
			signal?.removeEventListener("abort", onAbort);
		}
		// After an abort the run's effects are absorbed, but the robot's observation (camera
		// images) is not fetched: its calls would fail on the aborted signal.
		let observed: Result;
		try {
			observed = await spec.observe(r, signal);
		} catch (err) {
			if (!signal?.aborted) throw err;
			observed = { content: [text("aborted: the latest images were not fetched")], details: {} };
		}
		const report: Json = {
			status: r.status,
			...(r.error ? { error: r.error } : {}),
			...(r.limit ? { limit: r.limit } : {}),
			...(r.cancelled ? { cancelled: true } : {}),
			stdout: r.stdout,
			...(r.stderr ? { stderr: r.stderr } : {}),
			...(r.traceback ? { traceback: r.traceback } : {}),
			result: r.result ?? null,
			calls: r.calls,
			n_calls: r.n_calls,
			move_m: r.move_m,
			ms: r.ms,
		};
		const details = (observed.details ?? {}) as Json;
		return {
			...observed,
			content: [text(`run_code: ${JSON.stringify(report, null, 1)}`), ...observed.content],
			details: { ...details, status: r.status, run: report },
		};
	}

	pi.on("context", (event) => {
		if (!mode() || pi.getFlag("stateless") !== true) return undefined;
		const kept = latestTurn(event.messages);
		return kept ? { messages: kept } : undefined;
	});

	/** The oracle this session ran (`--code-oracle`), once it ran. */
	let oracleRun: Oracle | undefined;
	pi.on("session_start", () => {
		oracleRun = undefined;
	});
	// --code-oracle: the first prompt runs the reference program once, as run_code would; the model is never asked.
	pi.on("input", async (_event, ctx) => {
		if (!oracleRef() || !mode() || !(base.ready?.() ?? false)) return undefined;
		if (oracleRun) return { action: "handled" as const };
		const o = loadOracle(oracleRef(), oracleDir());
		oracleRun = o;
		const r = await runCode({ code: o.code, timeout_s: timeoutCap() }, undefined, ctx);
		const details = (r.details ?? {}) as Json;
		const entry = { file: o.name, sha256: o.sha256, header: o.header, run: details.run ?? null };
		pi.appendEntry(ORACLE_ENTRY, entry);
		// pi writes no session file for a session without an assistant message: the run's full report
		// (stdout, traceback, the primitive log) is kept next to where the session would be.
		try {
			const dir = ctx.sessionManager.getSessionDir();
			if (dir) {
				mkdirSync(dir, { recursive: true });
				writeFileSync(join(dir, "code_oracle.json"), `${JSON.stringify(entry, null, 1)}\n`);
			}
		} catch {}
		base.oracleRan?.();
		const summary = `code oracle ${o.name}: ${details.status ?? "error"}`;
		if (ctx.hasUI) ctx.ui.notify(summary, details.status === "ran" ? "info" : "warning");
		else
			console.error(
				`[code-oracle] ${summary}${details.run ? ` ${JSON.stringify(details.run).slice(0, 2000)}` : ""}`,
			);
		return { action: "handled" as const };
	});

	/** Why --code-oracle cannot run in this episode (mode, robot, file, tier, task), else undefined. */
	function oracleError(): string | undefined {
		if (!oracleRef()) return undefined;
		if (mode() !== "pure") return "--code-oracle runs a program instead of the model: it needs --code=true";
		if (spec.real) return "--code-oracle is for simulators; a real robot runs no unattended program";
		let o: Oracle;
		try {
			o = loadOracle(oracleRef(), oracleDir());
		} catch (err) {
			return err instanceof Error ? err.message : String(err);
		}
		const want = o.header.tier;
		if (want && want !== tier())
			return `--code-oracle ${o.name} is written for the ${want} tier, this episode runs ${tier()} (${want === "privileged" ? (base.privileged() ? "pass --code-api=high" : "add --privileged with --code-api=high") : `pass --code-api=${want}${base.privileged() ? " without --privileged" : ""}`})`;
		const fields = task();
		for (const [k, v] of Object.entries(fields))
			if (o.header[k] !== undefined && o.header[k] !== v)
				return `--code-oracle ${o.name} is written for ${k} ${o.header[k]}, this episode runs ${k} ${v}`;
		return undefined;
	}

	return {
		mode,
		/** Why the robot must not start with these flags, else undefined. */
		configError: (): string | undefined => {
			if (!mode())
				return oracleRef() ? "--code-oracle runs a program instead of the model: it needs --code=true" : undefined;
			if (base.unitsOn()) return "--code and --units are mutually exclusive: pick one mode";
			const t = String(pi.getFlag("code-api") ?? "high");
			if (!(TIERS as readonly string[]).includes(t))
				return `--code-api must be one of ${TIERS.join(", ")}, got "${t}"`;
			if (spec.real && !(pi.getFlag("code-real") === true && pi.getFlag("operator") === true))
				return "code mode on a real robot needs both --code-real and --operator (every program is then confirmed by the operator)";
			return oracleError();
		},
		/** After the robot is up: fetch this tier's registry (and the helpers) and register `run_code`; the tools to activate. */
		start: async (): Promise<string[]> => {
			if (!mode()) return [];
			const client = spec.rpc();
			api = await fetchCodeApi(client as RpcClient, tier() as CodeApiTier);
			if (!api) throw new Error("code mode needs an env server with a primitive registry (code.api)");
			const m = base.manifest;
			if (!m) throw new Error("code mode needs the robot's primitive manifest");
			if (api.manifest_digest !== m.digest)
				throw new Error(
					`the env server runs primitive manifest ${api.manifest_digest.slice(0, 12)}, this pi ${m.digest.slice(0, 12)}: start the server from this checkout`,
				);
			primitives = codePrimitives(m, tier() as CodeApiTier, base.has ?? (() => false), base.vars?.() ?? {});
			const mine = primitives.map((p) => p.name).sort();
			const theirs = [...api.available].sort();
			if (JSON.stringify(mine) !== JSON.stringify(theirs))
				throw new Error(
					`pi and the env server disagree on the ${tier()} primitives: pi has ${mine.join(", ") || "none"}, the server ${theirs.join(", ") || "none"}`,
				);
			if (!primitives.length)
				throw new Error(`this robot has no ${tier()}-tier code primitives; pick another --code-api`);
			// Preflight before the episode starts (no reset, no operator confirmation wasted): a server
			// that cannot isolate a program from this host's processes refuses code mode here. A
			// server on another host (URL#token) cannot expose this process: its refusal is waived.
			const remote = !isLocalEndpoint((client as RpcClient).url ?? "");
			const pre = await client
				.call<{ error?: string | null }>("code.preflight", { remote }, 30_000)
				// A server without code.preflight (older, or a stand-in): its code.run still refuses on its own.
				.catch((): { error?: string | null } => ({}));
			if (pre.error) throw new Error(pre.error);
			helpers = helpersOn() ? await client.call<Helper[]>("code.helpers", {}, 30_000) : [];
			registerTool();
			return ["run_code"];
		},
		/** The code-mode tool (pure mode adds finish). */
		tools: () => (mode() ? ["run_code"] : []),
		/** Pure mode: the whole prompt. Both mode: the section appended to the robot's prompt. */
		prompt: () => {
			const m = mode();
			let p = section(TEMPLATE, "pure", m === "pure");
			p = section(p, "both", m === "both");
			p = section(p, "helpers", helpersOn());
			p = section(p, "privileged", base.privileged());
			p = section(p, "stateless", pi.getFlag("stateless") === true);
			p = section(p, "real", spec.real === true);
			const vars: Record<string, string> = {
				task: instruction(),
				tier: tier(),
				timeout: String(timeoutCap()),
				max_calls: String(maxCalls()),
				max_move: String(maxMove()),
				api: renderPrimitives(primitives) || "(none)",
				helpers: renderHelpers(helpers),
			};
			return p.replace(/\{\{(\w+)\}\}/g, (match, k: string) => vars[k] ?? match).trim();
		},
		/** The robot result's code-mode fields: the mode and the tier the programs ran with. */
		result: () =>
			mode()
				? {
						code: mode() === "pure" ? "true" : "both",
						code_api: tier(),
						// The budget the programs ran under (effective) and as flagged (eval.sh's configuration key).
						code_budget: {
							timeout_s: timeoutCap(),
							max_calls: maxCalls(),
							max_move_m: maxMove(),
							helpers: helpersOn(),
						},
						code_budget_flags: `timeout=${rawFlag("code-timeout")}+max_calls=${rawFlag("code-max-calls")}+max_move=${rawFlag("code-max-move")}+helpers=${helpersOn()}${oracleOn() ? "+oracle" : ""}`,
						// The digest of the tier the programs ran with (robot.ts's code_api_digest is the episode's whole registry).
						...(api ? { code_tier_digest: api.digest } : {}),
						// A reference program ran, not the model: never comparable with a planner's run.
						...(oracleRun ? { code_oracle: oracleRun.name, code_oracle_sha256: oracleRun.sha256 } : {}),
					}
				: {},
	};
}
