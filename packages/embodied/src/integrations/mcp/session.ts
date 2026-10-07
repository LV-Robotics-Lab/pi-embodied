/**
 * One MCP session on one robot env server: the manifest's env tools (./tools.ts) forwarded to the
 * server over our RPC (../../infra/rpc.ts), each call checked against its schema as pi checks an
 * operator's manual call (`validateToolArguments`, as ../../robot.ts `prepare` does) and its
 * arguments adapted to the method's as pi's tool wrappers adapt them (../../primitives/arguments.ts
 * `rpcArguments`: `point: [row, col]` → `row`, `col`), and the built-in tools that mirror the pi
 * session where they do not depend on pi's agent loop:
 *
 *   observe       the current camera images and state along the robot's observation path
 *                 (../../observation/path.ts: the manifest's `get_observation`, else `render_camera`
 *                 per camera and `get_state`, with the robot's own cameras, size and orientation),
 *                 images as MCP image content
 *   reset         `env.reset`: a simulator's new episode, a real arm's start-pose motion (the gripper
 *                 opens, the arm moves). A motion like the manifest's: the guard, the latch and the
 *                 operator gate apply; a real arm is never reset at connect (./server.ts)
 *   finish        the agent's claim `{status, summary}`; afterwards only observe, robot_status, stop
 *                 and finish run ("The episode is finished.", as the robot base refuses)
 *   stop          the server's `stop` (what pi's abort sends; services/PROTOCOL.md "stop semantics",
 *                 not an emergency stop) and a latch: motion tools refuse until `resume`
 *   resume        clears the latch
 *   robot_status  healthz, the server pid against the one attached, the manifest digest, the latch
 *
 * Fail closed: a motion tool runs only when the server answers `healthz` now and is the process this
 * session attached (another server on the port is refused), the stop latch is clear, the episode
 * is not finished and, on a real robot, the operator authorised it outside the model (./gate.ts:
 * `PI_EMBODIED_MOTION_CONFIRMED=1` at launch for the session, or a per-call ticket in the
 * `--confirm-file`; the hosts' hooks are only a second layer, and Codex 0.160 runs none). Calls run
 * concurrently (MCP requests are independent), so `stop` latches synchronously, before its RPC, and
 * counts a stop generation: a motion checks the latch and the generation it was admitted under again
 * right before its dispatch, with no await in between, so a motion that waited on the guard while a
 * stop (or a stop and a resume) went by is refused and never reaches the server; the ticket check
 * sits in that same stretch, so a ticket is spent only by the call it admits. A server that stopped answering (`RpcUnavailable`) ends the episode: every tool
 * but robot_status answers "The robot failed: ...", as the robot base does. The hardware lock
 * (services/.../utils/hardware_lock.py) is the env server's: a second server for a held arm exits at
 * startup, which ./server.ts reports instead of serving. Result text is the robot base's `plain`
 * rendering (small arrays inline, large ones as shape stubs, 60 kB cap); `[H, W, 3+]` uint8 arrays
 * become PNG image content.
 */

import { type Tool as AiTool, validateToolArguments } from "@earendil-works/pi-ai";
import {
	type CallToolResult,
	type ContentBlock,
	JSON_RPC_ERROR_CODES,
	McpError,
	type Tool,
} from "@earendil-works/pi-mcp";
import { Type } from "typebox";
import { encodePng } from "../../infra/png.ts";
import { NdArray, type RpcClient, RpcUnavailable } from "../../infra/rpc.ts";
import {
	flipRows,
	type ObservationDecl,
	type ObservationPath,
	observationPath,
	robotObservation,
} from "../../observation/path.ts";
import { rpcArguments } from "../../primitives/arguments.ts";
import type { Manifest, ManifestEntry, Vars } from "../../primitives/manifest.ts";
import { message, plain, rgbOf, serverError } from "../../robot.ts";
import { CONFIRMED_ENV, OperatorGate } from "./gate.ts";
import type { ToolProvider } from "./protocol.ts";
import {
	BUILTIN_MOTIONS,
	capabilitiesFrom,
	isReal,
	type LeftOut,
	type McpTier,
	type McpTool,
	manifestTools,
	moves,
} from "./tools.ts";

export type SessionOptions = {
	robot: string;
	manifest: Manifest;
	rpc: RpcClient;
	tier?: McpTier;
	privileged: boolean;
	/** `--capabilities`: what the env server was started with, when it serves no `code.api` to tell. */
	capabilities?: readonly string[];
	/** The server's `code.api` reply (available names) at attach, when it serves one. */
	served?: string[];
	/** The server's pid at attach (`healthz`); a motion call refuses another pid on the port. */
	pid?: number;
	vars?: Vars;
	/** Per-call RPC timeout, ms (the client's default otherwise). */
	timeoutMs?: number;
	/**
	 * The operator's authorisation for a real robot's motions (./gate.ts): `session` when
	 * `PI_EMBODIED_MOTION_CONFIRMED` was set at launch, `file` the `--confirm-file` path. Neither on a
	 * real robot: every motion and reset is refused. Ignored on a simulator.
	 */
	confirm?: { session?: boolean; file?: string };
	/** The robot's observation declaration (default: `OBSERVATION` of ../../robots/<robot>/index.ts). */
	observation?: ObservationDecl;
	log?: (line: string) => void;
	version?: string;
};

type Claim = { status: string; summary: string };
const BUILTIN = ["observe", "reset", "finish", "stop", "resume", "robot_status"] as const;
type Builtin = (typeof BUILTIN)[number];
/** Built-ins that run after finish, under the latch, and (robot_status) after a failure. */
const AFTER_FINISH: ReadonlySet<string> = new Set(["observe", "finish", "stop", "robot_status"]);

const schema = (s: unknown) => JSON.parse(JSON.stringify(s)) as Record<string, unknown>;

const BUILTINS: Record<Builtin, Tool> = {
	observe: {
		name: "observe",
		description:
			"Look before and after acting: the robot's current camera images and state (the env server's get_observation, else render_camera per camera and get_state, as the robot's own observe does). Images come back as image content, arrays as shapes. Give camera to render one camera only.",
		inputSchema: schema(
			Type.Object({ camera: Type.Optional(Type.String({ description: "A camera name (default: every camera)" })) }),
		),
		annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
	},
	reset: {
		name: "reset",
		description:
			"Reset the robot's env (the env server's reset): a simulator starts a new episode of its task; a real arm opens its gripper (releasing anything held) and moves to its start pose. This moves the robot: the operator gate applies, and nothing is reset when the server connects to a real arm. Observe afterwards.",
		inputSchema: schema(Type.Object({})),
		annotations: { title: "reset", readOnlyHint: false, destructiveHint: true, openWorldHint: false },
	},
	finish: {
		name: "finish",
		description:
			"End the episode with your claim: status (e.g. success, failure, blocked) and a one-paragraph summary of what was done and what the last observation showed. After finish only observe, robot_status and stop run.",
		inputSchema: schema(
			Type.Object({
				status: Type.String({ description: "success | failure | blocked | aborted" }),
				summary: Type.String({ description: "What was done and the evidence (last observation)" }),
			}),
		),
		annotations: { readOnlyHint: false, destructiveHint: false, openWorldHint: false },
	},
	stop: {
		name: "stop",
		description:
			"Interrupt the running robot call at its next step boundary (the server's stop; not an emergency stop, use the hardware E-stop for that) and latch the motion tools off until resume.",
		inputSchema: schema(Type.Object({})),
		annotations: { readOnlyHint: false, destructiveHint: false, openWorldHint: false },
	},
	resume: {
		name: "resume",
		description: "Clear the stop latch so motion tools run again. Observe first.",
		inputSchema: schema(Type.Object({})),
		annotations: { readOnlyHint: false, destructiveHint: false, openWorldHint: false },
	},
	robot_status: {
		name: "robot_status",
		description:
			"The env server's health (healthz: service, version, pid), whether it is the process this session attached, the manifest digest match, the tier, the stop latch, the operator gate (a real robot: session authorisation or the ticket on file), the episode claim and the tools served.",
		inputSchema: schema(Type.Object({})),
		annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
	},
};

/** Walk a decoded RPC result: `[H, W, 3+]` uint8 arrays out as PNGs, a stub in their place. */
function extractImages(value: unknown, pngs: Buffer[]): unknown {
	if (value instanceof NdArray) {
		if (value.shape.length === 3 && value.shape[2] >= 3 && value.dtype === "uint8") {
			try {
				const rgb = rgbOf(value);
				pngs.push(encodePng(rgb.rgb, rgb.width, rgb.height));
				return { image: pngs.length, shape: value.shape };
			} catch {
				return value;
			}
		}
		return value;
	}
	if (Array.isArray(value)) return value.map((v) => extractImages(v, pngs));
	if (value && typeof value === "object")
		return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, extractImages(v, pngs)]));
	return value;
}

/** An RPC result as MCP content: JSON text (images replaced by `{image: n, shape}`) then the PNGs. */
export function renderResult(result: unknown): CallToolResult {
	const pngs: Buffer[] = [];
	const details = plain(extractImages(result, pngs));
	let text = JSON.stringify(details, null, 2) ?? "null";
	if (Buffer.byteLength(text) > 60_000) text = `${Buffer.from(text).subarray(0, 60_000).toString()}\n[truncated]`;
	const content: ContentBlock[] = [
		{ type: "text", text },
		...pngs.map((png) => ({ type: "image" as const, data: png.toString("base64"), mimeType: "image/png" })),
	];
	const structured =
		details && typeof details === "object" && !Array.isArray(details)
			? { structuredContent: details as Record<string, unknown> }
			: {};
	return { content, ...structured };
}

const failure = (text: string): CallToolResult => ({ content: [{ type: "text", text }], isError: true });

/** A rendered image (or `[rgb, depth]`) with its rows in top-down order (a sim that renders bottom-up). */
function flipped(value: unknown): unknown {
	if (value instanceof NdArray)
		return value.shape.length >= 2
			? new NdArray(value.dtype, value.shape, flipRows(value.data, value.shape[0]))
			: value;
	if (Array.isArray(value)) return value.map(flipped);
	return value;
}

export class RobotSession implements ToolProvider {
	readonly info: { name: string; version: string; title: string };
	readonly instructions: string;
	private readonly byName = new Map<string, McpTool>();
	readonly leftOut: LeftOut[];
	private claimed: Claim | undefined;
	private halted = false;
	/** Counts the stops issued; a motion admitted under one generation is refused under another. */
	private stops = 0;
	private broken: string | undefined;
	private readonly rpc: RpcClient;
	private readonly timeoutMs: number | undefined;
	private readonly log: (line: string) => void;
	private readonly o: SessionOptions;
	private readonly has: (capability: string) => boolean;
	private path: Promise<ObservationPath> | undefined;
	/** The cameras active on the configured robot (the path's `active`, asked once). */
	private cameras: Promise<string[]> | undefined;
	/** The operator gate on a real robot's motions. */
	readonly gate: OperatorGate;

	constructor(o: SessionOptions) {
		this.o = o;
		this.rpc = o.rpc;
		this.timeoutMs = o.timeoutMs;
		this.log = o.log ?? (() => {});
		const has = capabilitiesFrom(o.manifest, o.served, o.capabilities ?? [], o.privileged);
		this.has = has;
		const { tools, leftOut } = manifestTools(o.manifest, { tier: o.tier, privileged: o.privileged }, has, o.vars);
		for (const t of tools) this.byName.set(t.tool.name, t);
		this.leftOut = leftOut;
		this.gate = new OperatorGate({
			robot: o.robot,
			real: isReal(o.robot),
			sessionConfirmed: Boolean(o.confirm?.session),
			confirmFile: o.confirm?.file,
		});
		this.info = { name: `pi-embodied-${o.robot}`, version: o.version ?? "0.0.1", title: `pi-embodied ${o.robot}` };
		this.instructions = [
			`Robot ${o.robot}: ${this.byName.size} manifest tools${o.tier ? ` (tier ${o.tier})` : ""}${o.privileged ? " with the simulator's ground truth" : ""}, plus observe, reset, finish, stop, resume, robot_status.`,
			"Observe before the first motion and after every motion: a motion's result reports what the controller did, not what the scene is.",
			"Tools that move the robot are marked destructiveHint; call them one at a time and read the result's refusal or stopped fields.",
			"When the task is done or cannot be done, call finish with your evidence.",
			...(this.gate.real
				? [
						`${o.robot} is a real robot: a motion tool or reset runs only with the operator's authorisation given outside this session (${CONFIRMED_ENV}=1 when the server was started, or a ticket the operator writes into the server's confirm file before the call). A refusal is final until the operator acts: report it and wait.`,
					]
				: []),
		].join(" ");
	}

	/** The tools served (manifest, then built-ins). */
	list(): Tool[] {
		return [...[...this.byName.values()].map((t) => t.tool), ...BUILTIN.map((b) => BUILTINS[b])];
	}

	/** The manifest entries served (for the hook and tests). */
	entries(): ManifestEntry[] {
		return [...this.byName.values()].map((t) => t.entry);
	}

	/** Why `name` may not run now, else undefined (the robot base's `refusal`, without pi's budgets). */
	refusal(name: string): string | undefined {
		if (this.broken !== undefined && name !== "robot_status")
			return `The robot failed: ${this.broken}. The episode is over.`;
		if (this.claimed && !AFTER_FINISH.has(name)) return "The episode is finished.";
		if (this.moves(name) && this.halted)
			return "Motion is stopped (stop was issued): observe, then call resume to re-arm the motion tools.";
		return undefined;
	}

	/** Whether `name` moves the robot: a `mutating` manifest tool or a built-in motion (reset). */
	private moves(name: string): boolean {
		const t = this.byName.get(name);
		return t ? moves(t.entry) : BUILTIN_MOTIONS.includes(name);
	}

	/** The server broke (stopped answering): end the episode. */
	private fail(why: string) {
		this.broken ??= why;
		this.log(`robot failed: ${why}`);
	}

	private async rpcCall<T = unknown>(
		method: string,
		kwargs: Record<string, unknown>,
		signal: AbortSignal,
		timeoutMs?: number,
	) {
		try {
			return await this.rpc.call<T>(method, kwargs, timeoutMs ?? this.timeoutMs, [], signal);
		} catch (err) {
			if (err instanceof RpcUnavailable) this.fail(err.message);
			throw err;
		}
	}

	/** Before a motion: the server answers now and is the one attached (fail closed). */
	private async motionGuard(): Promise<string | undefined> {
		let h: { pid?: number; status?: string };
		try {
			h = await this.rpc.call<{ pid?: number; status?: string }>("healthz", {}, 5_000);
		} catch (err) {
			const why = `env server did not answer healthz before the motion: ${message(err)}`;
			if (err instanceof RpcUnavailable) this.fail(why);
			return why;
		}
		if (this.o.pid !== undefined && h.pid !== undefined && h.pid !== this.o.pid)
			return `env server at ${this.rpc.url} is pid ${h.pid}, not the pid ${this.o.pid} this session attached: refusing to move it`;
		return undefined;
	}

	async call(name: string, args: Record<string, unknown>, signal: AbortSignal): Promise<CallToolResult> {
		if ((BUILTIN as readonly string[]).includes(name)) return this.builtin(name as Builtin, args, signal);
		const t = this.byName.get(name);
		if (!t) throw new McpError(JSON_RPC_ERROR_CODES.invalidParams, `Unknown tool: ${name}`);
		const why = this.refusal(name);
		if (why) return failure(why);
		let kwargs: Record<string, unknown>;
		try {
			const tool: AiTool = {
				name,
				description: t.tool.description ?? "",
				parameters: t.tool.inputSchema as AiTool["parameters"],
			};
			const params = validateToolArguments(tool, {
				type: "toolCall",
				id: "mcp",
				name,
				arguments: args as Parameters<typeof validateToolArguments>[1]["arguments"],
			}) as Record<string, unknown>;
			kwargs = rpcArguments(t.entry, params);
		} catch (err) {
			return failure(`${name}: ${message(err)}`);
		}
		const dispatch = () => this.rpcCall(t.entry.method as string, kwargs, signal);
		return moves(t.entry) ? this.motion(name, dispatch) : this.forward(dispatch);
	}

	/**
	 * Run a motion admitted by `refusal` now: the guard (healthz now, the pid attached), then, with no
	 * await in between, the latch and the stop generation again, then the dispatch.
	 */
	private async motion(name: string, dispatch: () => Promise<unknown>): Promise<CallToolResult> {
		const admitted = this.stops;
		const blocked = await this.motionGuard();
		if (blocked) return failure(blocked);
		const late =
			this.stops !== admitted
				? "Motion is stopped (stop was issued while this call waited for the guard): observe, then call it again if it still applies."
				: this.refusal(name);
		if (late) return failure(late);
		// The operator's authorisation last, synchronously: a ticket is consumed by the call it admits.
		const unauthorised = this.gate.authorise(name);
		if (unauthorised) return failure(unauthorised);
		return this.forward(dispatch);
	}

	/** Dispatch a call; the server's error is the result text. */
	private async forward(dispatch: () => Promise<unknown>): Promise<CallToolResult> {
		try {
			return renderResult(await dispatch());
		} catch (err) {
			return failure(message(serverError(err) ?? err));
		}
	}

	private async builtin(name: Builtin, args: Record<string, unknown>, signal: AbortSignal): Promise<CallToolResult> {
		const why = this.refusal(name);
		if (why) return failure(why);
		switch (name) {
			case "robot_status":
				return this.status();
			case "reset":
				// A simulator's reset re-renders the scene; a real arm's moves: minutes, as connect allows.
				return this.motion(name, () => this.rpcCall("env.reset", {}, signal, 600_000));
			case "stop": {
				// Latch first, synchronously: a motion past its refusal check sees it before its dispatch.
				this.halted = true;
				this.stops++;
				await this.rpc.interrupt();
				return renderResult({
					ok: true,
					halted: true,
					note: "stop sent; the server interrupts the running call at its next step boundary. Motion tools refuse until resume.",
				});
			}
			case "resume":
				this.halted = false;
				return renderResult({ ok: true, halted: false });
			case "finish": {
				const status = typeof args.status === "string" ? args.status : "";
				const summary = typeof args.summary === "string" ? args.summary : "";
				if (!status || !summary) return failure("finish: status and summary are required strings");
				this.claimed = { status, summary };
				return renderResult({ robot: this.o.robot, claimed: status, summary, finished: true });
			}
			case "observe":
				return this.observe(typeof args.camera === "string" ? args.camera : undefined, signal);
		}
	}

	/** The robot's observation path, resolved once (its declaration is read from its module). */
	private observationPath(): Promise<ObservationPath> {
		this.path ??= (async () =>
			observationPath(
				this.o.manifest,
				this.has,
				this.o.vars ?? {},
				this.o.observation ?? (await robotObservation(this.o.robot)),
			))();
		return this.path;
	}

	private async observe(camera: string | undefined, signal: AbortSignal): Promise<CallToolResult> {
		try {
			const path = await this.observationPath();
			if (camera === undefined && path.observation)
				return renderResult(await this.rpcCall(path.observation, {}, signal));
			const render = path.render;
			if (!render)
				return failure(
					`${this.o.robot}'s manifest declares no render_camera${camera === undefined ? " or get_observation" : ""}: nothing to observe${camera === undefined ? "" : " per camera"}`,
				);
			// The cameras active on this robot (ManiSkill's widowxai has no wrist), by name or by the facade's camera_name.
			// Only a successful discovery is kept: a cancelled or failed one is retried by the next observe.
			this.cameras ??= render
				.active((method, kwargs) => this.rpcCall(method, kwargs, signal))
				.catch((e: unknown) => {
					this.cameras = undefined;
					throw e;
				});
			const active = await this.cameras;
			const names =
				camera === undefined ? active : active.filter((k) => k === camera || render.cameras[k] === camera);
			if (names.length === 0) return failure(`camera ${camera}: ${this.o.robot}'s cameras are ${active.join(", ")}`);
			const out: Record<string, unknown> = {};
			for (const name of names) {
				const image = await this.rpcCall(render.method, render.kwargs(render.cameras[name]), signal);
				out[name] = render.flip ? flipped(image) : image;
			}
			if (camera === undefined && path.state) out.state = await this.rpcCall(path.state, {}, signal);
			return renderResult(out);
		} catch (err) {
			return failure(message(serverError(err) ?? err));
		}
	}

	private async status(): Promise<CallToolResult> {
		let healthz: unknown;
		let reachable = true;
		try {
			healthz = await this.rpc.call("healthz", {}, 5_000);
		} catch (err) {
			reachable = false;
			healthz = { error: message(err) };
		}
		const pid = (healthz as { pid?: number } | undefined)?.pid;
		return renderResult({
			robot: this.o.robot,
			endpoint: this.rpc.url,
			reachable,
			healthz,
			attached_pid: this.o.pid ?? null,
			same_process: this.o.pid === undefined || pid === undefined ? null : pid === this.o.pid,
			manifest_digest: this.o.manifest.digest,
			tier: this.o.tier ?? null,
			privileged: this.o.privileged,
			halted: this.halted,
			stops: this.stops,
			operator_gate: this.gate.status(),
			claimed: this.claimed ?? null,
			broken: this.broken ?? null,
			tools: [...this.byName.keys()],
			left_out: this.leftOut,
		});
	}
}
