/**
 * The robot base. `defineRobot(pi, spec)` gives every robot the same episode lifecycle and mounts
 * the shared modules it asks for; the robot itself only registers its flags, tools and
 * observations. Also the helpers robots use for the Python services (../../../services): RPC
 * servers, the services' Python definitions, numpy payloads, camera frames, rigid transforms and
 * tool results.
 */

import { type ChildProcess, execFile, spawn } from "node:child_process";
import { closeSync, openSync, writeSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";
import { robotCheck } from "./check.ts";
import { type CodeSpec, code } from "./code/index.ts";
import { explore } from "./explore.ts";
import { fallback } from "./fallback.ts";
import { type FlashHook, flash } from "./flash/index.ts";
import { type FlywheelSpec, flywheel } from "./flywheel.ts";
import { type MemoryOptions, memory } from "./memory/index.ts";
import { operator } from "./operator.ts";
import { CODE_API_ENTRY, CODE_API_EVENT, type CodeApi, fetchCodeApi } from "./primitives/registry.ts";
import { forgetUnresponsive, NdArray, RpcClient, RpcUnavailable } from "./rpc.ts";
import { type UnitsSpec, units } from "./units/index.ts";
import { VLM_COST_EVENT } from "./units/vlm.ts";
import { type VdmSpec, vdm } from "./vdm.ts";
import { episodeVideo } from "./video.ts";

export type Json = Record<string, any>;
export type Mat = number[][];
export type Rgb = { width: number; height: number; rgb: Buffer };
export type Grid = { height: number; width: number; data: Float32Array };
/** The repository's services/ directory (the `pi_embodied_services` package). */
export const SERVICES = fileURLToPath(new URL("../../../services", import.meta.url));
/** The services directory and the Python that has their dependencies. */
export type Services = { root: string; python: string; env?: Record<string, string> };

export const round = (v: number, d = 5) => Number(Number(v).toFixed(d));
export const roundAll = (v: number[], d = 5) => v.map((x) => round(x, d));
export const median = (v: number[]) => {
	const s = [...v].sort((a, b) => a - b);
	return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};
export const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

/**
 * Keep every `[tool:name]...[/tool:name]` block of a robot's system prompt when that tool is
 * active, drop it otherwise, so an excluded tool is not described. `[tool:a|b]` is kept when any of
 * them is active; nested blocks need all of theirs. Markers on their own lines wrap whole lines,
 * markers within a line wrap that text. An unpaired marker, or a block nested in one of the same
 * name, throws: the template is broken.
 */
export function toolSections(prompt: string, active: readonly string[]): string {
	const re = /\[tool:([\w|]+)\](\n?)([\s\S]*?)\[\/tool:\1\](\n?)/g;
	let out = prompt;
	for (let prev = ""; prev !== out; ) {
		prev = out;
		out = out.replace(re, (_, names: string, open: string, body: string, close: string) => {
			if (body.includes(`[tool:${names}]`)) throw new Error(`[tool:${names}] nested in itself in the system prompt`);
			return (names.split("|").some((n) => active.includes(n)) ? body : "") + (open ? "" : close);
		});
	}
	const left = out.match(/\[\/?tool:[^\]]*\]/);
	if (left) throw new Error(`unpaired ${left[0]} in the system prompt`);
	return out;
}

// ---------------------------------------------------------------------------
// the robot base

/** Session entry naming the episode: `{ robot, <task flag>: value, ... }`. */
export const TASK_ENTRY = "robot_task";
/** Session entry with the episode's outcome: `{ robot, ..., claimed, summary, env_error }`. */
export const RESULT_ENTRY = "robot_result";
/** `pi.events` channel on which the robot publishes its `RobotStatus` (read by the dashboard). */
export const STATUS_EVENT = "pi-embodied:robot";
export type RobotStatus = {
	robot: string;
	/** The task flags in /robot-task order, and their values for this episode. */
	fields: readonly string[];
	task: Record<string, string>;
	ready: boolean;
	ended: boolean;
	claimed: string | null;
	summary: string | null;
	language?: string;
	step?: number;
	solved?: boolean;
};
type Result = AgentToolResult<unknown>;
/** What the agent claims in `finish`. */
type Claim = { status: string; summary: string };
type Ended = "agent_end" | "shutdown";

export type RobotSpec = {
	/** `robot` in the task and result entries, the stderr prefix `[name]`, the memory corpus name. */
	name: string;
	/** Flags (registered by the robot) that name the episode; /robot-task takes their values in this order. */
	task: readonly string[];
	/** Default of --keep-images, and the text that replaces older camera frames. */
	keepImages: number;
	imageStub?: string;
	/** Defaults of --max-turns and --time-limit (seconds from the first prompt); 0 = no limit. */
	budget?: { turns: number; seconds: number };
	/** Bring the robot up for this session and return the tools to activate. Throwing fails closed. */
	start: (ctx: ExtensionContext) => Promise<string[]>;
	/** Release what `start` acquired beyond the `serve`d env server (before a start and at shutdown). */
	stop?: () => Promise<void> | void;
	/** The system prompt, filled for this episode; its `[tool:name]` blocks follow the active tools (`toolSections`). */
	prompt?: () => string | undefined;
	/** The episode's outcome for the result entry; the base adds robot, claimed, summary, turns, budget, operator fields, env_error. */
	result: (ended: Ended) => Json;
	/** Live status for the dashboard, called only once the robot is up. */
	status?: () => Pick<RobotStatus, "language" | "step" | "solved">;
	/** The `finish` tool: status/summary parameters and the result text. It always ends the episode. */
	finish: { description: string; parameters: TSchema; result: (params: Claim) => Result };
	/** Mount memory (../memory): the cell and the primitives that make up its recipe. */
	memory?: Omit<MemoryOptions, "robot" | "explore">;
	/** Mount exploration (../explore.ts, needs memory): restart the episode, and the robot's exploration prompt. */
	explore?: {
		reset: (result: Json, ctx: ExtensionContext, signal?: AbortSignal) => Promise<Result>;
		prompt: () => string;
		distil?: string;
		rewrite?: [RegExp, string][];
		budget?: { sessions: number; attempts: number };
		/**
		 * A real robot: `reset` is the operator's scene reset (so request_scene_reset is hidden), and an
		 * attempt is solved by the operator's success verdict, which is marked `terminated` for exploration.
		 */
		operatorJudged?: boolean;
	};
	/** Mount the human-in-the-loop operator (../operator.ts). */
	operator?: { step: () => number; reset?: () => Promise<Json> };
	/** Mount code mode (../code, `--code`): the server's `code.run` over its primitive registry (`codeApi`), and how a run becomes an observation. */
	code?: CodeSpec;
	/** Mount the episode video (../video.ts). */
	video?: boolean;
	/**
	 * Mount the Flywheel recorder (../flywheel.ts, --collect-flywheel-data): what the robot's VLA
	 * observations and actions hold, and the default /flywheel-export selection (the current task).
	 */
	flywheel?: { spec: FlywheelSpec; select: () => string };
	/** Mount Show-Harness action units (../units): --units=true leaves only `act`, `finish` and its plugins' tools; --units=both adds them. */
	units?: UnitsSpec;
	/** Mount Flash (../flash, `--model flash/replay`): the robot's plans and how it re-localizes them. */
	flash?: FlashHook;
	/**
	 * Mount visual differencing (../vdm.ts, `--vdm`): how many camera images an observation result
	 * carries, and the wrist one(s); a function when that depends on the robot's cameras (undefined: none yet).
	 */
	vdm?: VdmSpec | (() => VdmSpec | undefined);
	/**
	 * Simulation only (CaP-X's S1 tier): the env call behind `ground_truth_poses` (the server's
	 * `env.ground_truth_poses`). It registers `--privileged`; real robots leave it unset, so they have no such flag.
	 */
	groundTruth?: (names: string[] | undefined) => Promise<unknown>;
	/**
	 * The env server whose primitive registry (`code.api`, ./primitives/registry.ts) this robot runs
	 * on, once `start` connected it. The base fetches the declaration for the episode's tier
	 * (`privileged` under --privileged), records it and puts its digest in the result.
	 */
	codeApi?: () => RpcClient | undefined;
};

/**
 * Define a robot on `pi`; call it before the robot registers its own handlers. The base owns:
 * - The task: the latest `robot_task` entry (from /robot-task or the dashboard) overrides the task
 *   flags; without one the flags are recorded as that entry. It is resolved before any mounted
 *   module's session_start runs.
 * - Startup, failing closed: every session starts with no active tools, and prompts and tool calls
 *   are refused until `start` succeeds. A failed start is shown in the UI, or without one printed
 *   to stderr as `[name] unavailable: ...` with exit code 1.
 * - Finish rules: pi stops early only when every result in a batch terminates, so tools registered
 *   with `tool` terminate when their batch also calls `finish`. `finish`, or a spent --max-turns /
 *   --time-limit / --max-cost budget, ends the episode, after which only `finish` runs.
 * - One `robot_result` entry per episode (and, without a UI, one `[name] {...}` stderr line): at the
 *   first agent_end after the episode ended, otherwise at session_shutdown once an agent ran, or
 *   `env_error: true` when the start failed. The robot breaking mid-episode (its env server exits,
 *   or a service stops answering: `RpcUnavailable` from a tool) ends the episode with
 *   `env_error: true` too. `planner_error` is set when the model's last reply was an error;
 *   evaluations treat both as invalid episodes, whatever the outcome.
 * - The env server started with `serve`, pruning of older camera frames, the system prompt, the
 *   /robot-task and /robot-check (../check.ts) commands, the status published on `pi.events`, and
 *   the mounted modules.
 */
export function defineRobot(pi: ExtensionAPI, spec: RobotSpec) {
	const { name } = spec;
	let task: Record<string, string> = {};
	let ready = false;
	let failed: string | undefined;
	/** The episode's primitive registry, when the robot declares `codeApi` and its server serves one. */
	let api: CodeApi | undefined;
	/** The robot broke during the episode: its env server exited, or a service stopped answering. */
	let broken: string | undefined;
	let ran = false;
	let ended = false;
	let reported = false;
	let finishing = false;
	let claimed: Claim | undefined;
	let signal: AbortSignal | undefined;
	let server: { proc: ChildProcess; rpc: RpcClient } | undefined;
	let turns = 0;
	let started: number | undefined;
	let outOfBudget: "turns" | "time" | "cost" | undefined;
	/** USD of this episode's model replies, as pi prices them from models.json. */
	let cost = 0;
	let plannerError: string | undefined;
	/** Ends the episode when the --time-limit wall-clock budget runs out, even mid-call. */
	let deadline: ReturnType<typeof setTimeout> | undefined;
	pi.registerFlag("keep-images", {
		type: "string",
		default: String(spec.keepImages),
		description: "Camera frames kept in context",
	});
	pi.registerFlag("anchor-image", {
		type: "boolean",
		default: false,
		description: "Also keep the episode's first camera frame (OpenETA's visual-history anchor) beyond --keep-images",
	});
	const anchored = () => pi.getFlag("anchor-image") === true;
	pi.registerFlag("max-turns", {
		type: "string",
		default: String(spec.budget?.turns ?? 0),
		description: "Planner turn budget (0 = none)",
	});
	pi.registerFlag("time-limit", {
		type: "string",
		default: String(spec.budget?.seconds ?? 0),
		description: "Planner wall-time budget from the first prompt, s (0 = none)",
	});
	pi.registerFlag("max-cost", {
		type: "string",
		default: "0",
		description:
			"Planner cost budget in USD (the planner's replies and the side VLM calls), as pi prices them from models.json (0 = none)",
	});
	if (spec.groundTruth)
		pi.registerFlag("privileged", {
			type: "boolean",
			default: false,
			description: "Simulation only: add ground_truth_poses (object poses from the simulator) and mark the result",
		});
	const privileged = () => spec.groundTruth !== undefined && pi.getFlag("privileged") === true;

	// Registered before the modules, so the task is resolved before memory's session_start reads it.
	pi.on("session_start", (_event, ctx) => {
		ready = ran = ended = reported = finishing = false;
		failed = broken = claimed = started = plannerError = outOfBudget = undefined;
		clearTimeout(deadline);
		deadline = undefined;
		// A service that stopped answering ended the last episode; this one may find it restarted.
		forgetUnresponsive();
		turns = cost = 0;
		pi.setActiveTools([]);
		const picked = ctx.sessionManager
			.getBranch()
			.filter((e) => e.type === "custom" && e.customType === TASK_ENTRY && (e.data as Json)?.robot === name)
			.pop();
		const data = picked?.type === "custom" ? (picked.data as Json) : undefined;
		task = Object.fromEntries(spec.task.map((k) => [k, String(data ? data[k] : (pi.getFlag(k) ?? ""))]));
		if (!picked) pi.appendEntry(TASK_ENTRY, { robot: name, ...task });
	});

	const mem = spec.memory
		? memory(pi, {
				robot: name,
				...spec.memory,
				explore: spec.explore ? () => pi.getFlag("explore") === true : undefined,
			})
		: undefined;
	const video = spec.video ? episodeVideo(pi) : { frame: (_image: NdArray) => {} };
	const fly = spec.flywheel ? flywheel(pi, spec.flywheel.spec, spec.flywheel.select) : undefined;
	if (spec.flash) flash(pi, spec.flash);
	// The `fallback/<primary>` planner (../fallback.ts), only when `--fallback-model` is on the command line.
	const fb = fallback(pi);
	/** A successful scene reset also restarts the units state (accumulated yaw, gripper, plan). */
	const resetsUnits =
		<A extends unknown[], R>(reset: (...args: A) => Promise<R>) =>
		async (...args: A) => {
			const r = await reset(...args);
			un?.reset();
			return r;
		};
	const operatorReset = spec.operator?.reset;
	const op = spec.operator
		? operator(pi, { ...spec.operator, ...(operatorReset ? { reset: resetsUnits(operatorReset) } : {}) })
		: {
				tools: () => [],
				check: () => {},
				result: () => ({}),
				sceneReset: async (..._args: unknown[]) => ({ error: "no operator is mounted" }),
				refuse: (_name: string): string | undefined => undefined,
			};

	const status = (): RobotStatus => ({
		robot: name,
		fields: spec.task,
		task,
		ready,
		ended,
		claimed: claimed?.status ?? null,
		summary: claimed?.summary ?? null,
		...(ready ? spec.status?.() : {}),
	});

	/** Every tool registered with `tool` (the robot's own and the units'): ../gumi holds them all during a takeover. */
	const robotTools: string[] = [];
	/** Register a sequential robot tool; its result terminates the batch when the batch also calls `finish`. */
	function tool<P extends TSchema>(
		toolName: string,
		description: string,
		parameters: P,
		run: (params: Static<P>, signal: AbortSignal | undefined, ctx: ExtensionContext) => Promise<Result>,
	) {
		robotTools.push(toolName);
		pi.registerTool({
			name: toolName,
			label: toolName,
			description,
			parameters,
			executionMode: "sequential",
			async execute(_id, params, sig, _onUpdate, ctx) {
				signal = sig;
				try {
					return { ...(await run(params, sig, ctx)), terminate: finishing };
				} catch (err) {
					if (err instanceof RpcUnavailable) fail(err.message);
					throw err;
				} finally {
					signal = undefined;
				}
			},
		});
	}
	// An operator's unit (../gumi) passes the gates an `act` call passes, without counting as a planner turn.
	const un = spec.units
		? units(pi, spec.units, tool, () => task, {
				// Exploration's `reset` and the operator's `request_scene_reset` move the robot too.
				tools: () => [...robotTools, "reset", "request_scene_reset"],
				refuse: () => refusal("act") ?? op.refuse("act"),
			})
		: undefined;
	const co = spec.code
		? code(pi, spec.code, tool, () => task, { unitsOn: () => un?.mode() !== undefined, privileged })
		: undefined;
	const vd = spec.vdm
		? vdm(
				pi,
				spec.vdm,
				() => [...robotTools, "reset", "request_scene_reset"],
				() =>
					spec.status?.().language ||
					Object.entries(task)
						.map(([k, v]) => `${k} ${v}`)
						.join(", "),
			)
		: undefined;
	let groundTruthRegistered = false;
	/** `--privileged`: register `ground_truth_poses` at the first start that asks for it (off registers nothing), and name it. */
	function groundTruth(): string[] {
		const poses = spec.groundTruth;
		if (!poses || !privileged()) return [];
		if (!groundTruthRegistered) {
			groundTruthRegistered = true;
			tool(
				"ground_truth_poses",
				"Privileged simulator ground truth: world-frame poses of the scene's objects, as pos [x, y, z] in m and quat_xyzw. Omit names for every object; names must come from that list.",
				Type.Object({
					names: Type.Optional(Type.Array(Type.String(), { description: "Object names (default: all)" })),
				}),
				async ({ names }) => toolResult((await poses(names)) as Record<string, unknown>),
			);
		}
		return ["ground_truth_poses"];
	}
	const publish = () => pi.events.emit(STATUS_EVENT, status());

	async function stop() {
		const s = server;
		server = undefined;
		if (s) await shutdown(s.proc, s.rpc);
		await spec.stop?.();
	}

	pi.on("session_start", async (_event, ctx) => {
		await stop();
		try {
			// A flag the robot cannot honour fails closed, before the robot boots.
			const misconfigured = un?.configError() ?? co?.configError();
			if (misconfigured) throw new Error(misconfigured);
			api = undefined;
			const tools = await spec.start(ctx);
			// The robot's configuration (its cameras) is known now: the units read its wrist view again.
			un?.started(ctx);
			const client = spec.codeApi?.();
			if (client) {
				api = await fetchCodeApi(client, privileged() ? "privileged" : undefined);
				if (api) pi.appendEntry(CODE_API_ENTRY, api);
			}
			pi.events.emit(CODE_API_EVENT, api);
			// Code mode fetches its tier of the registry once the robot is up and registers run_code.
			const coded = (await co?.start()) ?? [];
			// Pure units or code mode hides the robot's own tools and memory's (Show-Harness's pure mode).
			const mode = un?.mode() ?? co?.mode();
			const own =
				mode === "pure"
					? [...(un?.tools() ?? []), ...coded, "finish"]
					: [...tools, ...(mem?.tools ?? []), ...(mode === "both" ? [...(un?.tools() ?? []), ...coded] : [])];
			pi.setActiveTools([...new Set([...own, ...op.tools(), ...groundTruth()])]);
			ready = true;
		} catch (err) {
			// Without a robot there is nothing to act on: no tools, and a non-interactive run exits.
			failed = message(err);
			pi.setActiveTools([]);
			if (ctx.hasUI) ctx.ui.notify(`${name} unavailable: ${failed}`, "error");
			else {
				console.error(`[${name}] unavailable: ${failed}`);
				process.exitCode = 1;
				ctx.shutdown();
			}
		}
		publish();
	});

	/** The robot broke mid-episode: end the episode; its result is an env_error. */
	function fail(why: string) {
		broken ??= why;
		ended = true;
		publish();
	}

	pi.on("before_agent_start", (_event, ctx) => {
		if (started === undefined) {
			started = Date.now();
			// The budget is wall-clock: a hung model or tool call must not outlive it. Aborting stops the
			// running tool (its RPC gets `stop`), and the result records a planner timeout.
			const limit = Number(pi.getFlag("time-limit"));
			if (limit > 0) {
				deadline = setTimeout(() => {
					deadline = undefined;
					if (ended) return;
					outOfBudget = "time";
					ended = true;
					publish();
					ctx.abort();
				}, limit * 1000);
				deadline.unref();
			}
		}
		// Units and code mode are mutually exclusive (configError): at most one of them shapes the prompt.
		const mod = un?.mode() ? un : co?.mode() ? co : undefined;
		const mode = mod?.mode();
		// The prompt (the robot's and the mode's) describes only the tools left active by --tools/--exclude-tools and --units/--code.
		const own = spec.prompt?.();
		const systemPrompt =
			mode === "pure" ? mod?.prompt() : mode === "both" ? `${own ?? ""}\n\n${mod?.prompt()}`.trim() : own;
		return systemPrompt === undefined ? undefined : { systemPrompt: toolSections(systemPrompt, pi.getActiveTools()) };
	});
	pi.on("message_end", (event) => {
		const m = event.message;
		if (m.role !== "assistant") return;
		cost += m.usage?.cost?.total ?? 0;
		finishing = m.content.some((c) => c.type === "toolCall" && c.name === "finish");
		// The model failing (after pi's own retries) makes the episode's outcome meaningless, whatever it is.
		plannerError = m.stopReason === "error" ? (m.errorMessage ?? "model error") : undefined;
	});
	pi.on("turn_end", () => {
		turns++;
	});
	// Side VLM calls (the units verifier and video_ref, ../vdm.ts) spend from the same budget.
	pi.events.on(VLM_COST_EVENT, (usd) => {
		cost += Number(usd) || 0;
	});
	/** Why `toolName` may not run now (robot not up or broken, episode over, budget spent), else undefined. */
	function refusal(toolName: string): string | undefined {
		if (!ready) return `${name} is not available.`;
		if (broken !== undefined) return `The robot failed: ${broken}. The episode is over.`;
		if (toolName === "finish") return undefined;
		if (ended) return "The episode is finished.";
		const maxTurns = Number(pi.getFlag("max-turns"));
		const limit = Number(pi.getFlag("time-limit"));
		const late = limit > 0 && started !== undefined && Date.now() - started > limit * 1000;
		const maxCost = Number(pi.getFlag("max-cost"));
		const spent = maxCost > 0 && cost >= maxCost;
		if (!(maxTurns > 0 && turns >= maxTurns) && !late && !spent) return undefined;
		outOfBudget = late ? "time" : spent ? "cost" : "turns";
		ended = true;
		return `Planner ${outOfBudget} budget exhausted; the episode is over.`;
	}
	pi.on("tool_call", (event) => {
		const reason = refusal(event.toolName);
		return reason === undefined ? undefined : { block: true, reason, terminate: true };
	});
	/**
	 * Older camera frames are replaced by a stub, keeping the latest --keep-images. With --anchor-image
	 * the episode's first frame stays too (OpenETA's bounded visual history): the first image of the
	 * earliest tool result that carries one, which is the robot's first observation, main view first.
	 */
	pi.on("context", (event) => {
		let keep = Number(pi.getFlag("keep-images"));
		const anchor = anchored()
			? event.messages.find((m) => m.role === "toolResult" && m.content.some((p) => p.type === "image"))
			: undefined;
		let pruned = false;
		const messages = [...event.messages].reverse().map((m) => {
			if (m.role !== "toolResult") return m;
			let first = m === anchor;
			const content = m.content.map((part) => {
				if (part.type !== "image") return part;
				if (first) {
					first = false;
					return part;
				}
				if (keep-- > 0) return part;
				pruned = true;
				return { type: "text" as const, text: spec.imageStub ?? "[older camera frame omitted]" };
			});
			return { ...m, content };
		});
		return pruned ? { messages: messages.reverse() } : undefined;
	});
	pi.on("input", (_event, ctx) => {
		if (ready) return undefined;
		if (ctx.hasUI) ctx.ui.notify(`${name} is not available; nothing was sent to the model.`, "error");
		return { action: "handled" as const };
	});

	function report(hasUI: boolean, when: Ended) {
		if (reported || (failed === undefined && !(ready && ran))) return;
		reported = true;
		// Ground truth was on offer (--privileged): not comparable with a run without it.
		const mark = { ...(privileged() ? { privileged: true } : {}), ...(anchored() ? { anchor_image: true } : {}) };
		const r =
			failed !== undefined
				? { robot: name, ...task, ...mark, env_error: true, error: failed }
				: {
						robot: name,
						...spec.result(when),
						...mark,
						// The units verifier: whether the success finish was checked, and the call's error.
						...un?.result(),
						// Code mode (../code): `code` and `code_api`, so evaluations never mix it with tool runs.
						...co?.result(),
						...vd?.result(),
						// With --fallback-model: the turns each planner model planned (../fallback.ts).
						...fb?.result(),
						// Which primitive API (code.api) the episode ran with.
						...(api ? { code_api_digest: api.digest, code_api_tier: api.tier } : {}),
						claimed: claimed?.status ?? null,
						summary: claimed?.summary ?? null,
						turns,
						// Which budget ended the episode: "turns", "time" (a planner timeout), "cost", or null.
						planner_budget_exhausted: outOfBudget ?? null,
						cost_usd: Number(cost.toFixed(6)),
						// An operator verdict (/success /failure /abort) aborts the model mid-request; that ends the run, it is not a planner failure.
						planner_error: (op.result() as Json).operator_finished === true ? null : (plannerError ?? null),
						...op.result(),
						// A robot that broke mid-episode makes its outcome meaningless, like a failed start.
						env_error: broken !== undefined,
						...(broken !== undefined ? { error: broken } : {}),
					};
		try {
			pi.appendEntry(RESULT_ENTRY, r);
		} catch {}
		if (!hasUI) console.error(`[${name}] ${JSON.stringify(r)}`);
	}
	pi.on("agent_start", () => {
		ran = true;
	});
	pi.on("tool_execution_end", publish);
	pi.on("agent_end", (_event, ctx) => {
		if (ended) report(ctx.hasUI, "agent_end");
		publish();
	});
	// The result first: stopping the env server can take a while.
	pi.on("session_shutdown", (_event, ctx) => report(ctx.hasUI, "shutdown"));
	pi.on("session_shutdown", () => {
		clearTimeout(deadline);
		deadline = undefined;
	});
	pi.on("session_shutdown", stop);

	// After the robot's before_agent_start and tool_call gate: exploration rewrites that prompt and guards `finish`.
	if (spec.explore && mem)
		explore(pi, {
			...spec.explore,
			reset: resetsUnits(spec.explore.reset),
			render: mem.render,
			tools: mem.tools,
			aborted: () => (op.result() as Json).operator_aborted === true,
		});
	const exploring = () => spec.explore !== undefined && pi.getFlag("explore") === true;
	if (spec.explore?.operatorJudged) {
		// Exploration's success signal is `terminated` (../explore.ts, the memory recipe); here it is the operator's success verdict.
		pi.on("tool_result", (event) => {
			if (!exploring() || event.toolName !== "request_operator_verdict" || event.isError) return undefined;
			const details = event.details as Json | undefined;
			if (details?.status !== "success") return undefined;
			const marked = { ...details, terminated: true };
			return { details: marked, content: [{ type: "text" as const, text: JSON.stringify(marked) }] };
		});
		// In exploration `reset` is the scene reset: it counts attempts and bounds the recipe.
		pi.on("before_agent_start", () => {
			if (exploring()) pi.setActiveTools(pi.getActiveTools().filter((t) => t !== "request_scene_reset"));
		});
	}

	pi.registerTool({
		name: "finish",
		label: "finish",
		description: spec.finish.description,
		parameters: spec.finish.parameters,
		executionMode: "sequential",
		async execute(_id, params) {
			claimed = params as Claim;
			ended = true;
			return { ...spec.finish.result(claimed), terminate: true };
		},
	});

	const usage = `/robot-task ${spec.task.map((k) => `<${k}>`).join(" ")}`;
	pi.registerCommand("robot-task", {
		description: `Start a new ${name} episode in a new session: ${usage}`,
		handler: async (args, ctx) => {
			const values = args.trim().split(/\s+/).filter(Boolean);
			if (values.length !== spec.task.length) {
				ctx.ui.notify(`Usage: ${usage}`, "error");
				return;
			}
			await ctx.newSession({
				setup: async (sm) => {
					sm.appendCustomEntry(TASK_ENTRY, {
						robot: name,
						...Object.fromEntries(spec.task.map((k, i) => [k, values[i]])),
					});
				},
				withSession: async (next) => {
					next.sendUserMessage("Solve the task.").catch((err) => next.ui.notify(String(err), "error"));
				},
			});
		},
	});

	robotCheck(pi, name);

	return {
		/** This episode's task flag values (the /robot-task entry, else the flags). */
		get task() {
			return task;
		},
		/** The running tool's abort signal; pass it to every robot RPC so an abort stops motion between calls. */
		get signal() {
			return signal;
		},
		/** Register a sequential robot tool; its result terminates the batch when the batch also calls `finish`. */
		tool,
		/** The episode's primitive registry (`codeApi`), once the robot is up; undefined without one. */
		get codeApi() {
			return api;
		},
		/**
		 * Start `python ...args --transport http --host 127.0.0.1 --port 0 --parent-watch` (a service RPC
		 * server; it picks its port and exits with pi) and wait for healthz. It is stopped at the next start and at shutdown.
		 */
		async serve(o: {
			python: string;
			args: string[];
			cwd: string;
			env: NodeJS.ProcessEnv;
			log: (port: number) => string;
			readyMs?: number;
		}): Promise<RpcClient> {
			// The server binds port 0 and prints the port it got: probing a free port here and passing it on
			// would race every other process on the box for it between the probe and the bind.
			const argv = [...o.args, "--transport", "http", "--host", "127.0.0.1", "--port", "0", "--parent-watch"];
			const proc = spawn(o.python, argv, { cwd: o.cwd, env: o.env, stdio: ["pipe", "pipe", "pipe"] });
			const rpc = new RpcClient("http://127.0.0.1:0");
			server = { proc, rpc };
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
				setTimeout(
					() => reject(new Error(`env server bound no port in ${readyMs} ms; ${where()}`)),
					readyMs,
				).unref();
			});
			late.catch(() => {});
			const port = await Promise.race([listening, exited, late]);
			rpc.url = rpc.url.replace(":0/", `:${port}/`);
			await Promise.race([rpc.ready(deadline - Date.now()), exited]);
			// Stopping it clears `server` first; any other exit is the robot breaking mid-episode.
			proc.once("exit", (code, sig) => {
				if (server?.proc === proc) fail(`env server exited (${code ?? sig}); see ${log}`);
			});
			return rpc;
		},
		video,
		fly,
		op,
		mem,
	};
}

// ---------------------------------------------------------------------------
// service processes

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
	const rpc = new RpcClient(endpoint);
	await rpc.ready(readyMs);
	return rpc;
}

/**
 * Refuse a real-robot `move_delta` larger than one call may move, in meters. The services clip each
 * servo step on the server (0.02 m) but bound no call's total; their tasks document the per-call
 * limit only in the prompt ("Keep translation commands at or below 0.02 m per call"). The
 * tighter of that and `cap` applies.
 */
export function checkMove(delta: number[], cap: number, constraints: string[] = []) {
	const limit = moveLimit(cap, constraints);
	const norm = Math.hypot(...delta);
	if (!(norm <= limit))
		throw new Error(
			`delta_xyz moves ${round(norm, 4)} m; the limit is ${limit} m per call. Split the motion into smaller calls.`,
		);
}

/**
 * The --workspace-xy box (off when empty) and the required --z-floor, validated: a malformed box or a
 * missing or non-finite floor throws (the robot does not start, and no move runs without a floor).
 */
export function workspaceLimits(xy: string, zFloor: string): { box?: number[]; floor: number } {
	const box = xy.trim() ? xy.split(",").map((v) => (v.trim() ? Number(v) : Number.NaN)) : undefined;
	if (box && (box.length !== 4 || !box.every(Number.isFinite) || !(box[0] < box[1] && box[2] < box[3])))
		throw new Error(`--workspace-xy must be "xmin,xmax,ymin,ymax" (finite, min < max), got "${xy}"`);
	const floor = zFloor.trim() ? Number(zFloor) : Number.NaN;
	if (!Number.isFinite(floor))
		throw new Error(
			`--z-floor must be the lowest safe TCP z in m (e.g. Show-Harness's empty-table 0.14), got "${zFloor}"`,
		);
	return { ...(box ? { box } : {}), floor };
}

/** The per-call translation limit `checkMove` applies: the tighter of `cap` and the task's documented limit. */
export function moveLimit(cap: number, constraints: string[] = []) {
	const documented = constraints.map((c) => /translation commands at or below ([\d.]+) m per call/i.exec(c)?.[1]);
	return Math.min(cap, ...documented.filter((v) => v !== undefined).map(Number));
}

// ---------------------------------------------------------------------------
// numpy payloads

/** Elements of an array of any numeric or bool dtype, in C order. */
export function numbers(a: NdArray): number[] {
	const b = new Uint8Array(a.data).buffer;
	switch (a.dtype) {
		case "float32":
			return Array.from(new Float32Array(b));
		case "float64":
			return Array.from(new Float64Array(b));
		case "int8":
			return Array.from(new Int8Array(b));
		case "int16":
			return Array.from(new Int16Array(b));
		case "int32":
			return Array.from(new Int32Array(b));
		case "int64":
			return Array.from(new BigInt64Array(b), Number);
		case "uint16":
			return Array.from(new Uint16Array(b));
		case "uint32":
			return Array.from(new Uint32Array(b));
		case "uint64":
			return Array.from(new BigUint64Array(b), Number);
		case "uint8":
		case "bool":
			return Array.from(new Uint8Array(b));
		default:
			throw new Error(`unsupported dtype ${a.dtype}`);
	}
}

/** A list, NdArray or scalar as plain numbers (for poses and state vectors). */
export function vec(v: unknown): number[] {
	if (v instanceof NdArray) return numbers(v);
	if (Array.isArray(v)) return v.map(Number);
	return v === null || v === undefined ? [] : [Number(v)];
}

export const size = (a: NdArray) => a.shape.reduce((x, y) => x * y, 1);
export const f32 = (a: NdArray) => (a.dtype === "float32" ? a : NdArray.f32(numbers(a), a.shape));
export const u8 = (a: NdArray) =>
	a.dtype === "uint8" ? a : new NdArray("uint8", a.shape, Buffer.from(Uint8Array.from(numbers(a))));

/** Element `i` along the first axis. */
export function sub(a: NdArray, i: number): NdArray {
	const n = a.data.length / a.shape[0];
	return new NdArray(a.dtype, a.shape.slice(1), a.data.subarray(i * n, (i + 1) * n));
}

function nest(flat: unknown[], shape: number[]): unknown {
	if (shape.length === 0) return flat[0];
	if (shape.length === 1) return flat;
	const n = flat.length / shape[0];
	return Array.from({ length: shape[0] }, (_, i) => nest(flat.slice(i * n, (i + 1) * n), shape.slice(1)));
}

/** JSON-safe copy: small arrays become nested lists (numpy `tolist`), large ones a shape stub. */
export function plain(v: unknown): unknown {
	if (v instanceof NdArray) {
		if (size(v) > 256) return { ndarray: v.dtype, shape: v.shape };
		const flat = v.dtype === "bool" ? numbers(v).map(Boolean) : numbers(v);
		return nest(flat, v.shape);
	}
	if (Buffer.isBuffer(v)) return `<${v.length} bytes>`;
	if (Array.isArray(v)) return v.map(plain);
	if (v && typeof v === "object") return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, plain(x)]));
	return v;
}

// ---------------------------------------------------------------------------
// camera frames

/** An `[H, W, C>=3]` image as 8-bit RGB. */
export function rgbOf(a: NdArray): Rgb {
	if (a.shape.length !== 3 || a.shape[2] < 3) throw new Error(`expected an [H,W,3] image, got [${a.shape}]`);
	const [height, width, c] = a.shape;
	const src = u8(a).data;
	if (c === 3) return { width, height, rgb: Buffer.from(src) };
	const rgb = Buffer.alloc(width * height * 3);
	for (let i = 0; i < width * height; i++) src.copy(rgb, i * 3, i * c, i * c + 3);
	return { width, height, rgb };
}

/** An `[H, W, C>=3]` image as the `[H, W, 3]` uint8 frame the episode video (../video.ts) takes. */
export function frameOf(a: NdArray): NdArray {
	const { width, height, rgb } = rgbOf(a);
	return new NdArray("uint8", [height, width, 3], rgb);
}

/** A depth map with singleton axes squeezed, as float32 meters. */
export function gridOf(a: NdArray): Grid {
	const shape = a.shape.filter((d) => d !== 1);
	if (shape.length !== 2) throw new Error(`expected 2D depth, got [${a.shape}]`);
	return { height: shape[0], width: shape[1], data: Float32Array.from(numbers(a)) };
}

/** Copy of `img` with a crosshair and ring at pixel (row, col). */
export function mark(img: Rgb, row: number, col: number, color: [number, number, number]): Buffer {
	const out = Buffer.from(img.rgb);
	const { width: w, height: h } = img;
	const x = Math.max(0, Math.min(w - 1, col));
	const y = Math.max(0, Math.min(h - 1, row));
	const put = (px: number, py: number) => {
		if (px < 0 || py < 0 || px >= w || py >= h) return;
		out[(py * w + px) * 3] = color[0];
		out[(py * w + px) * 3 + 1] = color[1];
		out[(py * w + px) * 3 + 2] = color[2];
	};
	for (let d = -15; d <= 15; d++)
		for (let t = 0; t < 2; t++) {
			put(x + d, y + t);
			put(x + t, y + d);
		}
	for (let dy = -13; dy <= 13; dy++)
		for (let dx = -13; dx <= 13; dx++) {
			const r = Math.hypot(dx, dy);
			if (r >= 11 && r <= 13) put(x + dx, y + dy);
		}
	return out;
}

// ---------------------------------------------------------------------------
// rigid transforms

/** Rotation matrix of an xyzw quaternion (normalized first). */
export function quatMat(q: number[]): Mat {
	const n = Math.hypot(q[0], q[1], q[2], q[3]);
	if (!(n > 0)) throw new Error("zero-norm quaternion");
	const [x, y, z, w] = q.map((v) => v / n);
	return [
		[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
		[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
		[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
	];
}

/** Homogeneous transform of an `[x, y, z, qx, qy, qz, qw]` pose. */
export function pose7(p: number[]): Mat {
	if (p.length !== 7) throw new Error(`expected tcp_pose shape (7,), got (${p.length},)`);
	const r = quatMat(p.slice(3));
	return [
		[...r[0], p[0]],
		[...r[1], p[1]],
		[...r[2], p[2]],
		[0, 0, 0, 1],
	];
}

/** First three rows of `t @ [p, 1]` (or `t @ p` for a 3x3 `t`). */
export const apply = (t: Mat, p: number[]): number[] =>
	[0, 1, 2].map((i) => t[i][0] * p[0] + t[i][1] * p[1] + t[i][2] * p[2] + (t[i][3] ?? 0));

export function inv3(m: Mat): Mat {
	const [[a, b, c], [d, e, f], [g, h, i]] = m;
	const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
	if (!det) throw new Error("singular intrinsic_K");
	return [
		[(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
		[(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
		[(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
	];
}

// ---------------------------------------------------------------------------
// pi plumbing

/** Tool result: JSON text (capped at 60 kB) followed by PNG images. */
export function toolResult(out: Record<string, unknown>, pngs: Buffer[] = []) {
	const details = plain(out) as Json;
	let text = JSON.stringify(details, null, 2) ?? "null";
	if (Buffer.byteLength(text) > 60_000) text = `${Buffer.from(text).subarray(0, 60_000).toString()}\n[truncated]`;
	return {
		content: [
			{ type: "text" as const, text },
			...pngs.map((png) => ({ type: "image" as const, data: png.toString("base64"), mimeType: "image/png" })),
		],
		details,
	};
}
