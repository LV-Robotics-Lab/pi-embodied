/**
 * Code mode (CaP-X's "write code, execute it") for any robot whose env server serves `code.run`
 * over its primitive registry (`code.api`, ../primitives/registry.ts; services/PROTOCOL.md; the
 * shared runner is services/.../utils/code_exec.py).
 *
 *   pi -e packages/embodied/src/libero --code=true --code-api=high --suite libero_spatial --task 0 --seed 0
 *   pi -e packages/embodied/src/libero --code=both ...          (run_code next to the robot's tools)
 *   pi -e packages/embodied/src/libero --code=true --stateless   (the paper's no-history setting)
 *
 * A robot opts in with `code` in its defineRobot spec (the env RPC, and how a run's result becomes
 * the robot's observation); ../robot.ts mounts this module. `--code=true` hides the robot's own
 * tools: only `run_code` and `finish` remain and the system prompt is ./SYSTEM.md, rendered with the
 * registry's declaration of the episode's tier (`--code-api`: high = CaP-X's S2, perception plus
 * pose-level motion; low = S3, relative moves; `--privileged` runs the registry's privileged tier,
 * high plus the simulator's ground truth = S1). `--code=both` adds `run_code` to the robot's tools
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
 */

import { readFileSync } from "node:fs";
import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { type CodeApi, type CodeApiPrimitive, type CodeApiTier, fetchCodeApi } from "../primitives/registry.ts";
import type { RpcClient } from "../rpc.ts";
import { latestTurn, type ToolRegistrar } from "../units/index.ts";

/** `--code-api` values; `--privileged` runs the registry's third tier instead. */
export const TIERS = ["high", "low"] as const;
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

export type CodeSpec = {
	/** The env server that serves `code.api` / `code.run` (up after the robot's `start`). */
	rpc: () => Pick<RpcClient, "call">;
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
};

const TEMPLATE = readFileSync(new URL("./SYSTEM.md", import.meta.url), "utf8").replace(/^<!--[\s\S]*?-->\n/, "");
const text = (s: string) => ({ type: "text" as const, text: s });

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
	base: { unitsOn: () => boolean; privileged: () => boolean } = { unitsOn: () => false, privileged: () => false },
) {
	pi.registerFlag("code", {
		type: "string",
		default: "false",
		description: "Code mode (run_code): true = only run_code/finish, both = next to the robot's tools",
	});
	pi.registerFlag("code-api", {
		type: "string",
		default: "high",
		description: `Code mode primitive tier: ${TIERS.join(", ")} (CaP-X's S2, S3; --privileged runs the privileged tier, S1)`,
	});
	pi.registerFlag("code-timeout", {
		type: "string",
		default: String(DEFAULT_TIMEOUT_S),
		description: "Code mode: the most wall-clock seconds one program may run (its default timeout_s is 60 or this)",
	});
	pi.registerFlag("code-max-calls", {
		type: "string",
		default: String(DEFAULT_MAX_CALLS),
		description: "Code mode: primitive calls one program may make",
	});
	pi.registerFlag("code-max-move", {
		type: "string",
		default: String(spec.maxMoveM ?? DEFAULT_MAX_MOVE_M),
		description: "Code mode: metres of translation one program may command in total",
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
		base.privileged() ? "privileged" : String(pi.getFlag("code-api") ?? "high");
	const timeoutCap = () => Number(pi.getFlag("code-timeout")) || DEFAULT_TIMEOUT_S;
	const maxCalls = () => Math.max(1, Math.floor(Number(pi.getFlag("code-max-calls")) || DEFAULT_MAX_CALLS));
	const maxMove = () => {
		const v = Number(pi.getFlag("code-max-move"));
		return Number.isFinite(v) && v > 0 ? v : (spec.maxMoveM ?? DEFAULT_MAX_MOVE_M);
	};
	const helpersOn = () => pi.getFlag("code-helpers") === true;
	const instruction = () =>
		spec.instruction?.() ||
		Object.entries(task())
			.map(([k, v]) => `${k} ${v}`)
			.join(", ");

	/** The registry's declaration for this session's tier and the helpers (fetched at `start`). */
	let api: CodeApi | undefined;
	let helpers: Helper[] = [];
	let registered = "";

	function registerTool() {
		const names = (api?.primitives ?? []).map((p) => p.name);
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
						description: `Wall-clock timeout, s (default ${Math.min(DEFAULT_TIMEOUT_S, timeoutCap())}, at most ${timeoutCap()})`,
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
		const timeout_s = Math.min(params.timeout_s ?? Math.min(DEFAULT_TIMEOUT_S, cap), cap);
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
		const r = await spec.rpc().call<RunResult>(
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
			signal,
		);
		const observed = await spec.observe(r, signal);
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

	return {
		mode,
		/** Why the robot must not start with these flags, else undefined. */
		configError: (): string | undefined => {
			if (!mode()) return undefined;
			if (base.unitsOn()) return "--code and --units are mutually exclusive: pick one mode";
			const t = String(pi.getFlag("code-api") ?? "high");
			if (!(TIERS as readonly string[]).includes(t))
				return `--code-api must be one of ${TIERS.join(", ")}, got "${t}"`;
			if (spec.real && !(pi.getFlag("code-real") === true && pi.getFlag("operator") === true))
				return "code mode on a real robot needs both --code-real and --operator (every program is then confirmed by the operator)";
			return undefined;
		},
		/** After the robot is up: fetch this tier's registry (and the helpers) and register `run_code`; the tools to activate. */
		start: async (): Promise<string[]> => {
			if (!mode()) return [];
			const client = spec.rpc();
			api = await fetchCodeApi(client as RpcClient, tier() as CodeApiTier);
			if (!api) throw new Error("code mode needs an env server with a primitive registry (code.api)");
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
				api: renderPrimitives(api?.primitives ?? []) || "(none)",
				helpers: renderHelpers(helpers),
			};
			return p.replace(/\{\{(\w+)\}\}/g, (match, k: string) => vars[k] ?? match).trim();
		},
		/** The robot result's code-mode fields: the mode and the tier the programs ran with. */
		result: () => (mode() ? { code: mode() === "pure" ? "true" : "both", code_api: tier() } : {}),
	};
}
