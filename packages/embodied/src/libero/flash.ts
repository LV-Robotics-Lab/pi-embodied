/**
 * LIBERO Flash mode: replay a recorded plan with live grounding and no LLM.
 *
 *   pi -p -e src/libero --model flash/replay --suite libero_object_swap --task 3 --seed 0 \
 *     --molmo http://127.0.0.1:18400 "Solve the task."
 *
 * Flash is the planner, and in pi the planner is the model: the LIBERO extension registers a
 * `flash/replay` provider whose every turn is the plan's next tool call (with zero usage; an abort
 * ends the replay). The LIBERO tools execute it, so the session, `finish`, and the `robot_result`
 * row are exactly those of an LLM run.
 *
 * Each anchor is re-read the way it was recorded: `segment` anchors by SAM3 (the segment tool), the
 * rest by Molmo pointing in the opening agentview image, profiled through back_project; the arm
 * then parks over each Molmo anchor and asks again from the wrist, kept only within 5 cm of the
 * coarse reading. Waypoints are replayed as offsets from their live anchor.
 *
 * Plans live in `--flash-plans` (default memory/libero/flash) as `<family>_<suite>_t<task>_{plan,anchors}.json`,
 * from flash-generate.ts or the HF memory dataset. Molmo runs in its own env:
 * `python -m pi_embodied_services.components.molmo_server` (see services/README.md).
 */

import { readFileSync } from "node:fs";
import { isAbsolute, join, resolve } from "node:path";
import {
	type Api,
	type AssistantMessage,
	createAssistantMessageEventStream,
	createProvider,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type ToolCall,
	type TranscriptContext,
	type Usage,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { RpcClient } from "../rpc.ts";
import type { PlanEntry } from "./flash-generate.ts";

type Json = Record<string, unknown>;
type XY = [number, number];
type Call = { name: string; arguments: Json };
type Reply = { json: Json; images: string[]; error?: string };
type Program = { plan: PlanEntry[]; reference: Map<string, XY>; locatorOf: Map<string, string> };

/** A close reading further than this from the coarse one has found something else. */
const REFINE_ACCEPT = 0.05;
/** Beyond this a waypoint was not written relative to any located object. */
const MAX_ATTACH = 0.2;
/** The reachable workspace; a reading outside it is not a position. */
const REACH = 0.45;
/** How far a held object can plausibly sit from the gripper holding it. */
const MAX_HELD = 0.06;
/** Height-dependent parallax of a wrist reading of a held object. */
const PARALLAX = { x: [0.0231, 0.061], y: [-0.0029, 0.2056] } as const;
/** A pick that did not take hold is retried in place by replaying its approach. */
const PICK_ATTEMPTS = 3;
const FLASH_PICK_THRESHOLDS = {
	lift_thresh: 0.04,
	gripper_closed_thresh: 0.07,
	gripper_open_thresh: 0.003,
	descent_thresh: 0.0,
};
const PROMPTS = {
	survey: (o: string) => o,
	refine: (o: string) => `the center of the ${o} directly below the gripper`,
	held: (o: string) => `the body of the ${o} held in the gripper`,
};
const IMAGE = 1024;
const SUITE = /^libero_(10|goal|object|spatial)_(task|swap)$/;

const median = (v: number[]) => {
	const s = [...v].sort((a, b) => a - b);
	return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};
const medianXY = (pts: number[][]): XY => [median(pts.map((p) => p[0])), median(pts.map((p) => p[1]))];
const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const r4 = (v: number) => Number(v.toFixed(4));
const worldOf = (j: Json) => {
	const w = j.world_xyz;
	return Array.isArray(w) && w.length >= 3 && w.slice(0, 3).every((v) => typeof v === "number" && Number.isFinite(v))
		? (w.slice(0, 3) as number[])
		: undefined;
};
const parallax = (xy: XY, span: number): XY => [
	xy[0] - (PARALLAX.x[0] + PARALLAX.x[1] * span),
	xy[1] - (PARALLAX.y[0] + PARALLAX.y[1] * span),
];

function load(dir: string, name: string): Program {
	const read = (suffix: string) => {
		const path = join(dir, `${name}_${suffix}.json`);
		try {
			return JSON.parse(readFileSync(path, "utf8")) as Json;
		} catch (err) {
			throw new Error(`no complete Flash plan ${path} (plan and anchors are both required): ${err}`);
		}
	};
	const plan = read("plan").plan as PlanEntry[];
	const anchors = read("anchors").anchors as { phrase: string; locator: string; median_xy: XY }[];
	return {
		plan,
		reference: new Map(anchors.map((a) => [a.phrase, a.median_xy])),
		locatorOf: new Map(anchors.map((a) => [a.phrase, a.locator])),
	};
}

/**
 * The robot as the replay sees it: `act` hands tool calls to the model turn and resolves with their
 * results. Motion results carry `{result, terminated, state}` and the agentview + wrist images.
 */
async function replay(
	act: (calls: Call[]) => Promise<Reply[]>,
	molmo: RpcClient,
	program: Program,
	note: (s: string) => void,
) {
	const { plan, reference, locatorOf } = program;
	let latest: Reply = { json: {}, images: [] };
	const finished = () => latest.json.terminated === true || latest.json.truncated === true;
	const solved = () => latest.json.terminated === true;

	/** One motion tool; its result becomes the latest observation. Tool errors end the replay. */
	async function move(name: string, args: Json): Promise<Json> {
		const [reply] = await act([{ name, arguments: args }]);
		if (reply.error !== undefined) throw new Error(`${name} failed: ${reply.error}`);
		latest = reply;
		return (reply.json.result ?? {}) as Json;
	}

	/** Molmo's point for `query` in a 1024 camera image, as [col, row] in that image. */
	async function point(image: string | undefined, query: string): Promise<XY | undefined> {
		if (!image) return undefined;
		const res = await molmo.call<{ point_xy?: number[]; image_size?: number[] }>(
			"molmo.ground",
			{ image_base64: image, query },
			180_000,
		);
		if (!res.point_xy) return undefined;
		const [w, h] = res.image_size ?? [IMAGE, IMAGE];
		return [(res.point_xy[0] * IMAGE) / w, (res.point_xy[1] * IMAGE) / h];
	}

	/** World points of pixels in the current image of `camera`. */
	async function project(camera: string, pixels: XY[]): Promise<number[][]> {
		const replies = await act(
			pixels.map(([col, row]) => ({
				name: "back_project",
				arguments: { row: Math.round(row), col: Math.round(col), camera, resolution: "high" },
			})),
		);
		return replies.flatMap((r) => (r.error === undefined && worldOf(r.json) ? [worldOf(r.json) as number[]] : []));
	}

	/** A phrase's pixel profiled down a vertical line; readings below the top 3 cm left the object. */
	async function locate(camera: "agentview" | "wrist", query: string): Promise<XY | undefined> {
		const px = await point(latest.images[camera === "agentview" ? 0 : 1], query);
		if (!px) return undefined;
		const line = Array.from({ length: 9 }, (_, i): XY => [px[0], px[1] - 45 + i * 11.25]);
		const pts = await project(camera, line);
		if (!pts.length) return undefined;
		const top = Math.max(...pts.map((p) => p[2]));
		return medianXY(pts.filter((p) => p[2] > top - 0.03));
	}

	/** What is in the gripper, sampled on a grid, corrected for parallax. */
	async function heldBody(query: string): Promise<XY | undefined> {
		const px = await point(latest.images[1], query);
		if (!px) return undefined;
		const grid: XY[] = [];
		for (const dc of [-40, 0, 40]) for (const dr of [-40, 0, 40]) grid.push([px[0] + dc, px[1] + dr]);
		const pts = await project("wrist", grid);
		if (pts.length < 4) return undefined;
		const centre = medianXY(pts);
		const near = pts.filter((p) => dist(p, centre) < 0.05);
		if (near.length < 3) return undefined;
		const z = pts.map((p) => p[2]);
		return parallax(medianXY(near), Math.max(...z) - Math.min(...z));
	}

	// Coarse survey of the opening frame.
	await move("view_env_state", {});
	const live = new Map<string, XY>();
	for (const phrase of reference.keys()) {
		let xy: XY | undefined;
		if (locatorOf.get(phrase) === "segment") {
			const [r] = await act([
				{ name: "segment", arguments: { prompt: phrase, camera: "agentview", min_score: 0.2 } },
			]);
			const w = r.error === undefined ? worldOf(r.json) : undefined;
			if (!w) note(`${phrase} segmentation failed: ${r.error ?? r.json.error ?? "no world_xyz"}`);
			xy = w ? [w[0], w[1]] : undefined;
		} else {
			xy = await locate("agentview", PROMPTS.survey(phrase));
		}
		if (!xy || Math.max(Math.abs(xy[0]), Math.abs(xy[1])) > REACH) {
			note(`${phrase} not located, or out of reach`);
			continue;
		}
		if ([...live.values()].some((a) => dist(a, xy) < 0.03)) continue;
		live.set(phrase, xy);
	}
	note(`survey ${[...live].map(([p, a]) => `${p}=(${a[0].toFixed(3)},${a[1].toFixed(3)})`).join("  ")}`);

	// Park over each point-grounded anchor and read it again from the wrist.
	const zs = plan.flatMap((s) => {
		const xyz = s.arguments.xyz;
		const z = Array.isArray(xyz) ? Number(xyz[2]) : Number.NaN;
		return (s.action === "move_to" || s.action === "move_pose") && Number.isFinite(z) ? [z] : [];
	});
	const hover = zs.length ? Math.max(...zs) : 0.72;
	for (const [phrase, coarse] of [...live]) {
		if (locatorOf.get(phrase) === "segment") continue;
		try {
			await move("move_to", {
				xyz: [r4(coarse[0]), r4(coarse[1]), hover],
				gripper: -1,
				step_clip: 0.02,
				max_steps: 150,
				tol: 0.012,
			});
		} catch {
			continue;
		}
		const close = await locate("wrist", PROMPTS.refine(phrase));
		if (!close) continue;
		const gap = dist(close, coarse);
		if (gap > REFINE_ACCEPT) {
			note(`${phrase} close reading ${gap.toFixed(3)} away, rejected`);
			continue;
		}
		note(`${phrase} refined by ${gap.toFixed(3)}`);
		live.set(phrase, close);
	}

	// The plan, with every anchored waypoint moved to its live anchor.
	let offset: XY = [0, 0];
	let heldPhrase = "object";
	let recent: Call[] = [];
	let skipSuffix = false;
	for (const entry of plan) {
		if (finished()) break;
		const name = entry.action;
		const args: Json = { ...entry.arguments };
		if (skipSuffix) {
			// A pick that never took hold must not fall through into its carry.
			if (name === "release") {
				skipSuffix = false;
				recent = [];
				continue;
			}
			if (name !== "pi0_pick") continue;
			skipSuffix = false;
		}
		if (name === "move_to" || name === "move_pose") {
			const xyz = Array.isArray(args.xyz) ? (args.xyz as number[]) : [];
			if (xyz.length !== 3) continue;
			let target: XY = [xyz[0], xyz[1]];
			const phrase = entry.anchor;
			const attached = phrase !== undefined && (entry.anchor_distance ?? 9) <= MAX_ATTACH;
			if (attached && !live.has(phrase)) {
				note(`${phrase} unavailable; stopping replay`);
				break;
			}
			if (attached) {
				const a = live.get(phrase) as XY;
				const o = entry.offset ?? [0, 0];
				target = [a[0] + o[0], a[1] + o[1]];
			}
			if ((args.gripper ?? -1) === 1) target = [target[0] - offset[0], target[1] - offset[1]];
			if (Math.max(Math.abs(target[0]), Math.abs(target[1])) > REACH) continue;
			args.xyz = [r4(target[0]), r4(target[1]), xyz[2]];
			await move(name, args);
		} else if (name === "segment" || name === "segment_point") {
			continue;
		} else if (name === "pi0_pick" || name === "pi0_doubled") {
			const stripped = String(args.prompt ?? "").replace(/^(pick up|grasp)\s+the\s+/i, "");
			heldPhrase = stripped.split(/\b(?:on|in|into|inside|by|and)\b/)[0].trim();
			if (name === "pi0_pick") Object.assign(args, FLASH_PICK_THRESHOLDS);
			let result = await move(name, args);
			if (name === "pi0_pick" && result.success !== true) {
				for (let attempt = 1; attempt < PICK_ATTEMPTS && !finished(); attempt++) {
					for (const again of recent) await move(again.name, { ...again.arguments });
					result = await move(name, { ...args });
					if (result.success === true) break;
				}
				if (result.success !== true) {
					note("pick unconfirmed, skipping its carry");
					skipSuffix = true;
				}
			}
		} else if (name === "set_gripper") {
			await move(name, args);
			const body = await heldBody(PROMPTS.held(heldPhrase));
			const eef = (latest.json.state as { robot0_eef_pos?: number[] } | undefined)?.robot0_eef_pos;
			const candidate: XY | undefined = body && eef ? [body[0] - eef[0], body[1] - eef[1]] : undefined;
			if (candidate && Math.hypot(...candidate) <= MAX_HELD) {
				offset = candidate;
				note(`held offset (${offset[0].toFixed(4)},${offset[1].toFixed(4)})`);
			} else offset = [0, 0];
		} else if (name === "release") {
			await move(name, args);
			offset = [0, 0];
		} else {
			await move(name, args);
		}
		if (name === "release" || name === "pi0_pick" || name === "pi0_doubled") recent = [];
		else if (["move_to", "move_pose", "set_gripper", "rotate_wrist"].includes(name))
			recent = [...recent, { name, arguments: args }].slice(-6);
	}
	return { done: solved(), anchors: live.size, plan: plan.length };
}

/** Parse the tool results for `ids` out of the transcript the model turn received. */
function repliesFor(messages: Message[], ids: string[]): Reply[] {
	const byId = new Map<string, Message>();
	for (const m of messages) if (m.role === "toolResult") byId.set(m.toolCallId, m);
	return ids.map((id) => {
		const m = byId.get(id);
		if (!m || m.role !== "toolResult") return { json: {}, images: [], error: "no tool result" };
		const text = m.content.flatMap((c) => (c.type === "text" ? [c.text] : [])).join("\n");
		const images = m.content.flatMap((c) => (c.type === "image" ? [c.data] : []));
		let json: Json;
		try {
			json = JSON.parse(text) as Json;
		} catch {
			// Motion tools answer "Episode already ended (terminated=.., truncated=..)" once LIBERO is done.
			const ended = /^Episode already ended/.test(text);
			const flags = { terminated: /terminated=true/.test(text), truncated: /truncated=true/.test(text) };
			return ended ? { json: flags, images } : { json: {}, images, error: text };
		}
		return m.isError ? { json, images, error: text } : { json, images };
	});
}

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
/** Flash runs no model, so every turn reports zero usage. */
const USAGE: Usage = { ...ZERO_COST, totalTokens: 0, cost: { ...ZERO_COST, total: 0 } };
const MODEL: Model<"flash"> = {
	id: "replay",
	name: "Flash replay",
	api: "flash",
	provider: "flash",
	baseUrl: "",
	reasoning: false,
	input: ["text", "image"],
	cost: ZERO_COST,
	contextWindow: 100_000_000,
	maxTokens: 16_384,
};

/** Register the `flash/replay` model on the LIBERO extension; `cell` reads its --suite and --task. */
export function registerFlash(pi: ExtensionAPI, cell: () => { suite: string; task: string }) {
	pi.registerFlag("molmo", { type: "string", default: "http://127.0.0.1:18400", description: "Molmo server" });
	pi.registerFlag("flash-plans", {
		type: "string",
		default: "memory/libero/flash",
		description: "Directory of Flash plans",
	});

	type Turn = { text: string; calls: ToolCall[] };
	let toModel: { resolve: (turn: Turn) => void; reject: (err: Error) => void } | undefined;
	let toReplay: { resolve: (messages: Message[]) => void; reject: (err: Error) => void } | undefined;
	let pending: string[] = [];
	let notes: string[] = [];
	let started = false;
	let over = false;
	let stopped: Error | undefined;
	let calls = 0;
	let cwd = process.cwd();

	const say = (turn: Turn) => {
		const f = toModel;
		toModel = undefined;
		f?.resolve(turn);
	};
	const flush = () => {
		const text = notes.join("\n");
		notes = [];
		return text;
	};
	const toolCall = (c: Call): ToolCall => ({
		type: "toolCall",
		id: `flash_${++calls}`,
		name: c.name,
		arguments: c.arguments as ToolCall["arguments"],
	});

	/** An aborted turn ends the replay: the waiting turn fails, and the plan stops at its next call. */
	function stop() {
		stopped ??= new Error("Flash replay aborted");
		over = true;
		toModel?.reject(stopped);
		toReplay?.reject(stopped);
		toModel = toReplay = undefined;
	}

	async function act(calls: Call[]): Promise<Reply[]> {
		if (stopped) throw stopped;
		const toolCalls = calls.map(toolCall);
		pending = toolCalls.map((c) => c.id);
		const results = new Promise<Message[]>((resolve, reject) => {
			toReplay = { resolve, reject };
		});
		say({ text: flush(), calls: toolCalls });
		return repliesFor(await results, pending);
	}

	async function run() {
		const t0 = Date.now();
		const { suite, task } = cell();
		let status: "success" | "failure" = "failure";
		let summary: string;
		try {
			const match = SUITE.exec(suite);
			if (!match) throw new Error(`Flash plans cover libero_{10,goal,object,spatial}_{task,swap}, not ${suite}`);
			const program = `${match[1]}_${match[2]}_t${task}`;
			const dir = String(pi.getFlag("flash-plans") ?? "");
			const plans = isAbsolute(dir) ? dir : resolve(cwd, dir);
			const loaded = load(plans, program);
			const molmo = new RpcClient(String(pi.getFlag("molmo") ?? ""));
			await molmo.ready(30_000);
			notes.push(`replaying the ${program} program`);
			const out = await replay(act, molmo, loaded, (s) => notes.push(s));
			status = out.done ? "success" : "failure";
			summary =
				`replayed the ${program} program: ${out.plan} actions, ${out.anchors} anchors re-localized, ` +
				`${((Date.now() - t0) / 1000).toFixed(1)} s`;
		} catch (err) {
			if (stopped) return;
			summary = `flash error: ${err instanceof Error ? err.message : String(err)}`;
		}
		over = true;
		say({ text: flush(), calls: [toolCall({ name: "finish", arguments: { status, summary } })] });
	}

	/** The next turn of the replay: the plan's next tool calls, or `finish` once it is done. */
	function next(messages: Message[], signal?: AbortSignal): Promise<Turn> {
		if (signal?.aborted) stop();
		if (over) return Promise.resolve({ text: "Flash replay already ran in this session.", calls: [] });
		const turn = new Promise<Turn>((resolve, reject) => {
			toModel = { resolve, reject };
		});
		signal?.addEventListener("abort", stop, { once: true });
		if (!started) {
			started = true;
			void run();
		} else {
			const f = toReplay;
			toReplay = undefined;
			f?.resolve(messages);
		}
		return turn;
	}

	function streamSimple(model: Model<Api>, context: TranscriptContext, options?: SimpleStreamOptions) {
		const stream = createAssistantMessageEventStream();
		const message: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: { ...USAGE, cost: { ...USAGE.cost } },
			stopReason: "stop",
			timestamp: Date.now(),
		};
		next(context.messages as Message[], options?.signal).then(
			(turn) => {
				stream.push({ type: "start", partial: message });
				if (turn.text) {
					message.content.push({ type: "text", text: turn.text });
					stream.push({ type: "text_start", contentIndex: 0, partial: message });
					stream.push({ type: "text_delta", contentIndex: 0, delta: turn.text, partial: message });
					stream.push({ type: "text_end", contentIndex: 0, content: turn.text, partial: message });
				}
				for (const call of turn.calls) {
					const contentIndex = message.content.push(call) - 1;
					stream.push({ type: "toolcall_start", contentIndex, partial: message });
					stream.push({ type: "toolcall_end", contentIndex, toolCall: call, partial: message });
				}
				message.stopReason = turn.calls.length ? "toolUse" : "stop";
				stream.push({ type: "done", reason: message.stopReason, message });
				stream.end();
			},
			(err: Error) => {
				message.stopReason = options?.signal?.aborted ? "aborted" : "error";
				message.errorMessage = err.message;
				stream.push({ type: "error", reason: message.stopReason, error: message });
				stream.end();
			},
		);
		return stream;
	}

	pi.registerProvider(
		createProvider({
			id: "flash",
			name: "Flash",
			auth: { apiKey: { name: "Flash", resolve: async () => ({ auth: {} }) } },
			models: [MODEL],
			api: { stream: streamSimple, streamSimple },
		}),
	);

	pi.on("session_start", (_event, ctx) => {
		cwd = ctx.cwd;
		started = over = false;
		stopped = toModel = toReplay = undefined;
		pending = [];
		notes = [];
	});
}
