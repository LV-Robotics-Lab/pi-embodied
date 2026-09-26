/**
 * Live dashboard for pi-embodied (`--dashboard`), for any robot defined with ../robot.ts.
 *
 *   pi -e packages/embodied/src/libero -e packages/embodied/src/dashboard --dashboard \
 *     [--dashboard-host 0.0.0.0] [--dashboard-port 8765] [--dashboard-language zh-cn]
 *
 * One static page served by node:http follows pi's own events over Server-Sent Events:
 * streamed thinking and text, tool calls with their arguments and results, the camera
 * frames each tool returned, and the episode status the robot publishes on `pi.events`
 * (STATUS_EVENT). The page state is rebuilt from the session branch at every session_start,
 * so reload, resume and fork show the right history.
 *
 * The camera panel can also show a live stream (`GET /live`, multipart/x-mixed-replace PNG, at most
 * --dashboard-live-fps frames per second, downscaled to `?w=`): the robot's env frames as it records
 * them for the episode video (../video.ts FRAME_EVENT), so it follows every env step of a motion call
 * or an operator's unit, not only tool results. It adds no robot RPC: a simulator only renders when
 * it steps, and the stream sends the latest frame, skipping any a client is too slow for.
 *
 * The new-task form sends `/robot-task <values>` for the robot's task fields, which starts a
 * new pi session carrying a `robot_task` entry; the robot reads it in place of its flags.
 *
 * Operator tools (RPent's dashboard, github.com/RLinf/RPent @eecf206 rpent/dashboard/server.py):
 * - A message sent while the agent runs goes into pi's steering queue; until the agent takes it
 *   (its user message starts) `POST /message/withdraw {id}` takes it back out (`ctx.withdrawQueuedMessage`).
 * - `GET /primitives` lists the robot's active tools with their parameter schemas, and
 *   `POST /primitive {name, arguments}` runs one through the robot's own tool path (../robot.ts
 *   PRIMITIVES_EVENT: schema check, the robot's and the operator's gates, the same execute), only in
 *   operator mode: the agent is idle or the operator took over (GUMI), and no operator batch runs.
 * - `POST /llm-check` sends the session's model one real 1-token completion through pi's model
 *   registry (../check.ts llmCheck) and reports its latency or error.
 * - `GET /download/session?format=jsonl|html` is pi's /export of the current session;
 *   `GET /downloads` lists the episode videos (../video.ts) of the sessions this process served, and
 *   `GET /download/video/<session>/<file>.mp4` sends one. episode.mp4 is written when its session ends.
 *
 * pi rebuilds extension runtimes, and with them `pi.events`, on every session switch, so the
 * HTTP server (whose port and open browser connections must survive the switch) is the one
 * process-wide object: a promise in a `globalThis` slot, re-attached to each new runtime.
 */

import { createReadStream, existsSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { hostname, tmpdir } from "node:os";
import { basename, join } from "node:path";
import { inflateSync } from "node:zlib";
import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import type {
	ExtensionAPI,
	ExtensionContext,
	MessageEndEvent,
	MessageUpdateEvent,
} from "@earendil-works/pi-coding-agent";
import { llmCheck } from "../check.ts";
import { type Gumi, type GumiState, gumi, observation, type Step as UnitStep } from "../gumi/index.ts";
import { encodePng } from "../png.ts";
import {
	PRIMITIVES_EVENT,
	type PrimitivesHandle,
	RESULT_ENTRY,
	type RobotStatus,
	rgbOf,
	STATUS_EVENT,
	TASK_ENTRY,
} from "../robot.ts";
import type { NdArray } from "../rpc.ts";
import { FRAME_EVENT, NOTE_EVENT, VIDEO_DIR_EVENT, type VideoNote } from "../video.ts";

const HUB = Symbol.for("pi-embodied.dashboard");
/** The episode videos ../video.ts writes: episode.mp4, episode_overlay.mp4, action_<n>_<tool>.mp4. */
const VIDEO_FILE = /^[\w.-]+\.mp4$/;

type Message = MessageEndEvent["message"];
type StreamEvent = MessageUpdateEvent["assistantMessageEvent"];
type Image = { type: "image"; data: string; mimeType: string };
type Item = {
	id: number;
	kind: "user" | "thinking" | "text" | "tool_call" | "tool_result" | "meta";
	text: string;
	name?: string;
	callId?: string;
	args?: unknown;
	isError?: boolean;
	step?: number;
};
type Step = {
	n: number;
	name: string;
	args: unknown;
	result: unknown;
	isError: boolean;
	envStep: number | null;
	solved: boolean;
	ms: number | null;
	frames: string[];
};
/** A message the operator sent while the agent ran: queued in pi until the agent takes it, or withdrawn. */
type Queued = { id: number; text: string; status: "queued" | "delivered" | "withdrawn" | "dropped" };
/** The robot's published status plus what the dashboard tracks itself. */
type Episode = Omit<RobotStatus, "step" | "solved"> & {
	gen: number;
	attached: boolean;
	running: boolean;
	envStep: number | null;
	solved: boolean;
	usage: { input: number; output: number; tools: number };
	/** `provider/model · thinking` of the running agent. */
	model: string | null;
};

const modelLabel = (pi: ExtensionAPI, ctx: ExtensionContext) =>
	ctx.model ? `${ctx.model.provider}/${ctx.model.id} · ${pi.getThinkingLevel()}` : null;

/** Process-wide dashboard; it outlives extension runtimes, which pi rebuilds on every session switch. */
type Hub = ReturnType<typeof createHub>;

/** Before the robot publishes its status: its name and task from the branch's latest `robot_task` entry. */
function taskEntry(ctx: ExtensionContext): Pick<RobotStatus, "robot" | "fields" | "task"> {
	const e = ctx.sessionManager
		.getBranch()
		.filter((x) => x.type === "custom" && x.customType === TASK_ENTRY)
		.pop();
	const { robot, ...task } = (e?.type === "custom" ? e.data : {}) as Record<string, unknown>;
	return {
		robot: String(robot ?? "robot"),
		fields: Object.keys(task),
		task: Object.fromEntries(Object.entries(task).map(([k, v]) => [k, String(v)])),
	};
}

const textOf = (content: string | { type: string; text?: string }[]) =>
	typeof content === "string"
		? content
		: content
				.filter((c) => c.type === "text")
				.map((c) => c.text)
				.join("\n");

const clip = (s: string, n = 4000) => (s.length > n ? `${s.slice(0, n)}… [${s.length - n} more chars]` : s);

/** Downscale 8-bit RGB pixels by the integer box filter that brings them to at most `maxWidth`. */
function shrinkRgb(rgb: Buffer, width: number, height: number, maxWidth: number) {
	if (width <= maxWidth) return { rgb, width, height };
	const f = Math.ceil(width / maxWidth);
	const [w, h] = [Math.floor(width / f), Math.floor(height / f)];
	const out = Buffer.alloc(w * h * 3);
	for (let y = 0; y < h; y++)
		for (let x = 0; x < w; x++)
			for (let c = 0; c < 3; c++) {
				let sum = 0;
				for (let dy = 0; dy < f; dy++)
					for (let dx = 0; dx < f; dx++) sum += rgb[((y * f + dy) * width + x * f + dx) * 3 + c];
				out[(y * w + x) * 3 + c] = Math.round(sum / (f * f));
			}
	return { rgb: out, width: w, height: h };
}

/** Downscale a PNG from ../png.ts (8-bit RGB, filter 0) by an integer box filter; anything else is served as is. */
function shrinkPng(png: Buffer, maxWidth: number): Buffer {
	if (png.toString("ascii", 12, 16) !== "IHDR") return png;
	const width = png.readUInt32BE(16);
	const height = png.readUInt32BE(20);
	if (width <= maxWidth || png[24] !== 8 || png[25] !== 2 || png[28] !== 0) return png;
	const idat: Buffer[] = [];
	for (let pos = 8; pos < png.length; ) {
		const len = png.readUInt32BE(pos);
		if (png.toString("ascii", pos + 4, pos + 8) === "IDAT") idat.push(png.subarray(pos + 8, pos + 8 + len));
		pos += 12 + len;
	}
	const raw = inflateSync(Buffer.concat(idat));
	const stride = width * 3 + 1;
	for (let y = 0; y < height; y++) if (raw[y * stride] !== 0) return png;
	const rgb = Buffer.alloc(width * height * 3);
	for (let y = 0; y < height; y++) raw.copy(rgb, y * width * 3, y * stride + 1, (y + 1) * stride);
	const small = shrinkRgb(rgb, width, height, maxWidth);
	return encodePng(small.rgb, small.width, small.height);
}

/** One env frame (HxWxC) as a PNG at most `maxWidth` wide. */
export function framePng(frame: NdArray, maxWidth: number): Buffer {
	const img = rgbOf(frame);
	const small = shrinkRgb(img.rgb, img.width, img.height, maxWidth);
	return encodePng(small.rgb, small.width, small.height);
}

function createHub(server: Server, url: string, page: string, liveFps: number) {
	const clients = new Set<ServerResponse>();
	/** The latest env frame (../video.ts FRAME_EVENT), and the live-stream clients with the frame each has. */
	let live: { frame: NdArray; seq: number } | undefined;
	const watchers = new Set<{ res: ServerResponse; width: number; seq: number }>();
	const liveCache = new Map<number, Buffer>();
	let pump: ReturnType<typeof setInterval> | undefined;
	const frameCache = new Map<string, Buffer>();
	let pi: ExtensionAPI | undefined;
	let ctx: ExtensionContext | undefined;
	let nextId = 0;
	let items: Item[] = [];
	let steps: Step[] = [];
	let images: Image[][] = [];
	let episode: Episode | undefined;
	let streaming = new Map<number, Item>();
	/** GUMI teleop / recording / takeover (../gumi) of this runtime, and its latest state. */
	let teleop: Gumi | undefined;
	let teleopState: GumiState | undefined;
	const callArgs = new Map<string, unknown>();
	const callStart = new Map<string, number>();
	/** Messages sent while the agent ran (pi's steering queue), newest last. */
	let queued: Queued[] = [];
	let nextQueued = 0;
	/** The robot's tools for manual calls (../robot.ts PRIMITIVES_EVENT), and the manual call running now. */
	let prims: PrimitivesHandle | undefined;
	let manual: AbortController | undefined;
	let llmChecking = false;
	/** Episode video directory of every session this process served (session id -> dir), current last. */
	const videoDirs = new Map<string, string>();

	const send = (op: Record<string, unknown>) => {
		const line = `data: ${JSON.stringify(op)}\n\n`;
		for (const res of clients) res.write(line);
	};
	const snapshot = () => ({ op: "reset", episode, items, steps, gumi: teleopState, queued });
	const sendQueued = () => send({ op: "queued", queued });
	const touch = () => send({ op: "episode", episode });
	const add = (item: Omit<Item, "id">) => {
		const full = { id: nextId++, ...item };
		items.push(full);
		send({ op: "item", item: full });
		return full;
	};
	const replace = (item: Item) => send({ op: "item", item });

	function partItem(part: Extract<Message, { role: "assistant" }>["content"][number]): Omit<Item, "id"> {
		if (part.type === "text") return { kind: "text", text: part.text };
		if (part.type === "thinking") return { kind: "thinking", text: part.redacted ? "[redacted]" : part.thinking };
		callArgs.set(part.id, part.arguments);
		return { kind: "tool_call", text: "", name: part.name, callId: part.id, args: part.arguments };
	}

	function toolResult(m: Extract<Message, { role: "toolResult" }>) {
		if (!episode) return;
		const text = textOf(m.content);
		const frames = m.content.filter((c): c is Image => c.type === "image");
		let parsed: unknown = text;
		try {
			parsed = JSON.parse(text);
		} catch {}
		const obs = parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : undefined;
		const label = (v: unknown) =>
			typeof v === "string" ? v.split(" ")[0] : String((v as { camera?: unknown })?.camera ?? "image");
		const labels = Array.isArray(obs?.images) ? obs.images.map(label) : [];
		const started = callStart.get(m.toolCallId);
		const step: Step = {
			n: steps.length,
			name: m.toolName,
			args: callArgs.get(m.toolCallId) ?? null,
			result: obs && "result" in obs ? obs.result : parsed,
			isError: m.isError,
			envStep: episode.envStep,
			solved: episode.solved,
			ms: started === undefined ? null : Date.now() - started,
			frames: frames.map((_, k) => labels[k] ?? (frames.length === 1 ? "image" : `image ${k + 1}`)),
		};
		steps.push(step);
		images.push(frames);
		episode.usage.tools++;
		send({ op: "step", step });
		add({
			kind: "tool_result",
			text: clip(text),
			name: m.toolName,
			callId: m.toolCallId,
			isError: m.isError,
			step: step.n,
		});
		touch();
	}

	/** An operator's action (a GUMI step, a manual primitive call): a timeline row with the frames it returned. */
	function operatorStep(
		name: string,
		label: string,
		args: unknown,
		result: AgentToolResult<unknown>,
		isError: boolean,
		ms: number | null,
	) {
		if (!episode) return;
		const frames = result.content.filter((c): c is Image => c.type === "image");
		const text = textOf(result.content);
		const labels = observation(result)?.labels ?? [];
		const step: Step = {
			n: steps.length,
			name,
			args,
			result: clip(text, 400),
			isError,
			envStep: episode.envStep,
			solved: episode.solved,
			ms,
			frames: frames.map((_, k) => labels[k]?.split(" ")[0] || (frames.length === 1 ? "image" : `image ${k + 1}`)),
		};
		steps.push(step);
		images.push(frames);
		send({ op: "step", step });
		add({ kind: "meta", text: `${name} ${label}${isError ? ` failed: ${clip(text, 200)}` : ""}`, step: step.n });
	}

	/** Why the operator may not call a robot tool by hand now, else undefined. */
	function notOperatorMode(): string | undefined {
		if (!ctx) return "session is switching; retry in a moment";
		if (!prims) return "no robot tools (the robot is not up, or it is not a pi-embodied robot)";
		if (manual) return "a manual call is running";
		if (teleop?.state().busy) return "an operator batch (GUMI) is running";
		if (!ctx.isIdle() && teleop?.takeover.mode !== "human")
			return "the agent is running: interrupt it, or take over (GUMI) first";
		return undefined;
	}

	/** The robot's latest status (STATUS_EVENT), kept across attaches within one runtime. */
	function applyStatus(s: RobotStatus) {
		if (!episode) return;
		const { step, solved, ...rest } = s;
		Object.assign(episode, rest, { envStep: step ?? episode.envStep, solved: solved ?? episode.solved });
		touch();
	}

	/** One word per task field, as /robot-task takes them. */
	const validTask = (values: string[]) =>
		values.length === (episode?.fields.length ?? -1) && values.every((v) => /^[\w.:-]+$/.test(v));

	/** Send each live client the latest frame, once; a client whose socket is still draining skips it. */
	function sendLive() {
		if (!live) return;
		const { frame, seq } = live;
		for (const w of watchers) {
			if (w.seq === seq || w.res.writableNeedDrain) continue;
			let png = liveCache.get(w.width);
			if (!png) {
				png = framePng(frame, w.width);
				liveCache.set(w.width, png);
			}
			w.res.write(`--frame\r\nContent-Type: image/png\r\nContent-Length: ${png.length}\r\n\r\n`);
			w.res.write(png);
			w.res.write("\r\n");
			w.seq = seq;
		}
	}

	const hub = {
		url,
		/** An env frame the robot recorded; streamed to live clients at the next tick. */
		frame(frame: NdArray) {
			live = { frame, seq: (live?.seq ?? 0) + 1 };
			liveCache.clear();
		},
		/** Bind to a fresh runtime and rebuild the page state from its session branch. */
		attach(nextPi: ExtensionAPI, nextCtx: ExtensionContext, status: RobotStatus | undefined, g: Gumi) {
			pi = nextPi;
			ctx = nextCtx;
			teleop = g;
			teleopState = g.state();
			items = [];
			steps = [];
			images = [];
			streaming = new Map();
			callArgs.clear();
			callStart.clear();
			// pi's queue belongs to the session that was left; its messages are not this session's.
			queued = [];
			frameCache.clear();
			live = undefined;
			liveCache.clear();
			episode = {
				...taskEntry(nextCtx),
				// Frame URLs (/frame/<gen>/...) are cached by the browser: the first gen is the process start
				// time, so another pi process on the same port never reuses an earlier episode's URLs.
				gen: (episode?.gen ?? Date.now()) + 1,
				ready: false,
				ended: false,
				claimed: null,
				summary: null,
				attached: true,
				running: false,
				envStep: null,
				solved: false,
				usage: { input: 0, output: 0, tools: 0 },
				model: modelLabel(nextPi, nextCtx),
			};
			const e = episode;
			const task = Object.entries(e.task).map(([k, v]) => `${k} ${v}`);
			items.push({ id: nextId++, kind: "meta", text: [e.robot, ...task].join(" · ") });
			for (const entry of nextCtx.sessionManager.getBranch()) {
				if (entry.type === "message") hub.messageEnd(entry.message);
				else if (entry.type === "custom" && entry.customType === RESULT_ENTRY)
					items.push({ id: nextId++, kind: "meta", text: `${RESULT_ENTRY} ${JSON.stringify(entry.data)}` });
			}
			if (status) applyStatus(status);
			send(snapshot());
		},
		detach() {
			pi = undefined;
			ctx = undefined;
			teleop = undefined;
			if (episode) Object.assign(episode, { attached: false, running: false });
			touch();
		},
		close() {
			hub.detach();
			for (const res of clients) res.end();
			clients.clear();
			for (const w of watchers) w.res.end();
			watchers.clear();
			clearInterval(pump);
			server.closeAllConnections();
			server.close();
			delete (globalThis as Record<symbol, unknown>)[HUB];
		},
		status: applyStatus,
		/** The robot's tools for manual calls, published at each session start. */
		primitives(h: PrimitivesHandle | undefined) {
			prims = h;
		},
		/** This session's episode video directory (../video.ts VIDEO_DIR_EVENT). */
		videoDir(dir: string) {
			const id = ctx?.sessionManager.getSessionId();
			if (!id) return;
			videoDirs.delete(id);
			videoDirs.set(id, dir);
		},
		gumiState(s: GumiState) {
			teleopState = s;
			send({ op: "gumi", gumi: s });
		},
		/** An operator step (../gumi): a timeline row with the frames it returned, like a tool result. */
		teleopStep(label: string, unit: UnitStep, result: AgentToolResult<unknown>, isError: boolean) {
			operatorStep("operator", label, unit, result, isError, null);
		},
		setRunning(running: boolean) {
			if (!episode) return;
			episode.running = running;
			if (pi && ctx) episode.model = modelLabel(pi, ctx);
			touch();
		},
		agentEnd() {
			const last = ctx?.sessionManager.getBranch().at(-1);
			if (last?.type === "custom" && last.customType === RESULT_ENTRY)
				add({ kind: "meta", text: `${RESULT_ENTRY} ${JSON.stringify(last.data)}` });
			hub.setRunning(false);
		},
		toolStart(callId: string) {
			callStart.set(callId, Date.now());
		},
		messageStart(m: Message) {
			if (m.role === "assistant") streaming = new Map();
			if (m.role !== "user") return;
			// The agent took a queued message: it can no longer be withdrawn.
			const text = textOf(m.content);
			const q = queued.find((x) => x.status === "queued" && x.text === text);
			if (!q) return;
			q.status = "delivered";
			sendQueued();
		},
		messageUpdate(ev: StreamEvent) {
			if (ev.type === "thinking_start" || ev.type === "text_start") {
				streaming.set(ev.contentIndex, add({ kind: ev.type === "text_start" ? "text" : "thinking", text: "" }));
			} else if (ev.type === "thinking_delta" || ev.type === "text_delta") {
				const item = streaming.get(ev.contentIndex);
				if (!item) return;
				item.text += ev.delta;
				send({ op: "append", id: item.id, text: ev.delta });
			} else if (ev.type === "toolcall_end") {
				streaming.set(ev.contentIndex, add(partItem(ev.toolCall)));
			}
		},
		/** Final form of a message, live or replayed from the branch. */
		messageEnd(m: Message) {
			if (m.role === "user") add({ kind: "user", text: textOf(m.content) });
			if (m.role === "toolResult") toolResult(m);
			if (m.role !== "assistant") return;
			m.content.forEach((part, i) => {
				const item = streaming.get(i);
				if (item) replace(Object.assign(item, partItem(part)));
				else add(partItem(part));
			});
			streaming = new Map();
			if (m.stopReason === "error" || m.stopReason === "aborted")
				add({ kind: "meta", text: `${m.stopReason}${m.errorMessage ? `: ${m.errorMessage}` : ""}` });
			if (episode) {
				episode.usage.input += m.usage.input + m.usage.cacheRead + m.usage.cacheWrite;
				episode.usage.output += m.usage.output;
				touch();
			}
		},
		async handle(req: IncomingMessage, res: ServerResponse) {
			const url = new URL(req.url ?? "/", "http://dashboard");
			const reply = (status: number, body: unknown) => {
				res.writeHead(status, { "Content-Type": "application/json" });
				res.end(JSON.stringify(body));
			};
			if (req.method === "GET" && url.pathname === "/") {
				res.writeHead(200, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" });
				res.end(page);
				return;
			}
			if (req.method === "GET" && url.pathname === "/events") {
				res.writeHead(200, {
					"Content-Type": "text/event-stream",
					"Cache-Control": "no-store",
					Connection: "keep-alive",
				});
				res.write(`data: ${JSON.stringify(snapshot())}\n\n`);
				clients.add(res);
				req.on("close", () => clients.delete(res));
				return;
			}
			if (req.method === "GET" && url.pathname === "/live") {
				res.writeHead(200, {
					"Content-Type": "multipart/x-mixed-replace; boundary=frame",
					"Cache-Control": "no-store",
					Connection: "keep-alive",
				});
				const width = Math.min(1024, Math.max(64, Number(url.searchParams.get("w")) || 480));
				const w = { res, width, seq: 0 };
				watchers.add(w);
				pump ??= setInterval(sendLive, 1000 / liveFps);
				sendLive();
				req.on("close", () => {
					watchers.delete(w);
					if (watchers.size) return;
					clearInterval(pump);
					pump = undefined;
				});
				return;
			}
			if (req.method === "GET" && url.pathname === "/primitives") {
				const why = notOperatorMode();
				return reply(200, { available: !why, reason: why ?? null, tools: prims?.tools() ?? [] });
			}
			if (req.method === "GET" && url.pathname === "/downloads") {
				const current = ctx?.sessionManager.getSessionId();
				const sessions = [...videoDirs].reverse().map(([id, dir]) => ({
					id,
					current: id === current,
					videos: existsSync(dir)
						? readdirSync(dir)
								.filter((f) => VIDEO_FILE.test(f))
								.sort()
						: [],
				}));
				return reply(200, { session: current ?? null, sessions });
			}
			if (req.method === "GET" && url.pathname === "/download/session") {
				// pi's own /export (JSONL or HTML) of the current session branch, into a temp file sent once.
				const format = url.searchParams.get("format") === "html" ? "html" : "jsonl";
				if (!ctx) return reply(409, { error: "session is switching; retry in a moment" });
				const dir = mkdtempSync(join(tmpdir(), "pi-dashboard-export-"));
				const name = `session-${ctx.sessionManager.getSessionId()}.${format}`;
				try {
					const path = await ctx.exportSession(format, join(dir, name));
					res.writeHead(200, {
						"Content-Type": format === "html" ? "text/html; charset=utf-8" : "application/x-ndjson",
						"Content-Disposition": `attachment; filename="${name}"`,
						"Cache-Control": "no-store",
					});
					res.end(readFileSync(path));
				} finally {
					rmSync(dir, { recursive: true, force: true });
				}
				return;
			}
			const video = url.pathname.match(/^\/download\/video\/([\w-]+)\/([^/]+)$/);
			if (req.method === "GET" && video) {
				const dir = videoDirs.get(video[1]);
				const file = decodeURIComponent(video[2]);
				if (!dir || !VIDEO_FILE.test(file) || basename(file) !== file || !existsSync(join(dir, file)))
					return reply(404, { error: "no such video" });
				res.writeHead(200, {
					"Content-Type": "video/mp4",
					"Content-Disposition": `attachment; filename="${video[1]}-${file}"`,
					"Cache-Control": "no-store",
				});
				createReadStream(join(dir, file)).pipe(res);
				return;
			}
			const frame = url.pathname.match(/^\/frame\/(\d+)\/(\d+)\/(\d+)$/);
			if (req.method === "GET" && frame) {
				const [gen, n, k] = frame.slice(1).map(Number);
				const img = gen === episode?.gen ? images[n]?.[k] : undefined;
				if (!img) return reply(404, { error: "no such frame" });
				const width = Number(url.searchParams.get("w") ?? 0);
				const key = `${gen}/${n}/${k}/${width}`;
				let body = frameCache.get(key);
				if (!body) {
					body = Buffer.from(img.data, "base64");
					if (width > 0 && img.mimeType === "image/png") body = shrinkPng(body, width);
					frameCache.set(key, body);
					if (frameCache.size > 32) frameCache.delete(frameCache.keys().next().value as string);
				}
				res.writeHead(200, { "Content-Type": img.mimeType, "Cache-Control": "private, max-age=86400" });
				res.end(body);
				return;
			}
			if (req.method !== "POST") return reply(404, { error: "not found" });
			// JSON-only POSTs: a cross-origin page cannot send them without a CORS preflight we never answer.
			if (!req.headers["content-type"]?.startsWith("application/json"))
				return reply(415, { error: "expected application/json" });
			let raw = "";
			for await (const chunk of req) {
				raw += chunk;
				if (raw.length > 65_536) return reply(413, { error: "body too large" });
			}
			let body: Record<string, unknown>;
			try {
				body = JSON.parse(raw || "{}");
			} catch {
				return reply(400, { error: "invalid JSON" });
			}
			if (!pi || !ctx) return reply(409, { error: "session is switching; retry in a moment" });
			const usage = `/robot-task ${(episode?.fields ?? []).map((k) => `<${k}>`).join(" ")}`;
			if (url.pathname === "/task") {
				const values = Array.isArray(body.values) ? body.values.map((v) => String(v).trim()) : [];
				if (!validTask(values)) return reply(422, { error: `usage: ${usage}` });
				pi.sendUserMessage(`/robot-task ${values.join(" ")}`, { expandPromptTemplates: true });
				return reply(202, { ok: true });
			}
			if (url.pathname === "/message") {
				const text = typeof body.text === "string" ? body.text.trim() : "";
				if (!text) return reply(422, { error: "empty message" });
				if (text.startsWith("/")) {
					if (!text.startsWith("/robot-task ") || !validTask(text.slice(12).trim().split(/\s+/)))
						return reply(422, { error: `the only command here is ${usage}` });
					pi.sendUserMessage(text, { expandPromptTemplates: true });
					return reply(202, { ok: true, steered: false });
				}
				if (ctx.isIdle()) {
					pi.sendUserMessage(text);
					return reply(202, { ok: true, steered: false });
				}
				// While the agent runs: pi's steering queue, withdrawable until the agent takes it.
				pi.sendUserMessage(text, { deliverAs: "steer" });
				const q: Queued = { id: nextQueued++, text, status: "queued" };
				queued.push(q);
				sendQueued();
				return reply(202, { ok: true, steered: true, id: q.id });
			}
			if (url.pathname === "/message/withdraw") {
				const q = queued.find((x) => x.id === Number(body.id));
				if (!q) return reply(404, { error: "no such message" });
				if (q.status !== "queued") return reply(409, { error: `the message was already ${q.status}` });
				if (!ctx.withdrawQueuedMessage(q.text))
					return reply(409, { error: "the message is no longer in pi's queue (the agent is taking it)" });
				q.status = "withdrawn";
				sendQueued();
				return reply(200, { ok: true, text: q.text });
			}
			if (url.pathname === "/primitive") {
				const name = typeof body.name === "string" ? body.name : "";
				const args = body.arguments ?? {};
				if (!name) return reply(422, { error: "name must be a non-empty string" });
				if (typeof args !== "object" || Array.isArray(args) || args === null)
					return reply(422, { error: "arguments must be a JSON object" });
				const why = notOperatorMode();
				if (why) return reply(409, { error: why });
				const h = prims as PrimitivesHandle;
				if (!h.tools().some((t) => t.name === name)) return reply(404, { error: `no active robot tool ${name}` });
				let call: ReturnType<PrimitivesHandle["prepare"]>;
				try {
					call = h.prepare(name, args);
				} catch (err) {
					return reply(422, { error: (err as Error).message });
				}
				const controller = new AbortController();
				manual = controller;
				// The agent's robot calls wait while it runs (../gumi's takeover gate), and the video labels its frames.
				teleop?.takeover.begin();
				pi.events.emit(NOTE_EVENT, { actor: "human", action: name } satisfies VideoNote);
				const t0 = Date.now();
				let result: AgentToolResult<unknown>;
				let isError = false;
				try {
					result = await call(controller.signal);
				} catch (err) {
					isError = true;
					result = { content: [{ type: "text", text: (err as Error).message }], details: {} };
				} finally {
					manual = undefined;
					pi.events.emit(NOTE_EVENT, { actor: null } satisfies VideoNote);
					teleop?.takeover.end();
				}
				operatorStep(`operator:${name}`, JSON.stringify(args), args, result, isError, Date.now() - t0);
				return reply(isError ? 422 : 200, {
					ok: !isError,
					...(isError ? { error: clip(textOf(result.content), 500) } : {}),
					text: clip(textOf(result.content), 4000),
				});
			}
			if (url.pathname === "/llm-check") {
				if (!ctx.model) return reply(409, { error: "no model selected" });
				// Single flight: a diagnostic, not a load generator.
				if (llmChecking) return reply(409, { error: "a check is already running" });
				llmChecking = true;
				try {
					return reply(200, await llmCheck(ctx.modelRegistry, ctx.model));
				} finally {
					llmChecking = false;
				}
			}
			// GUMI (../gumi): teleop steps, the rollout recorder, and takeover from the agent.
			if (url.pathname.startsWith("/gumi/") && teleop) {
				const g = teleop;
				try {
					if (url.pathname === "/gumi/step") return reply(200, await g.step(body));
					if (url.pathname === "/gumi/record")
						return reply(200, g.record(String(body.action ?? ""), body.success));
					if (url.pathname === "/gumi/control") return reply(200, g.control(String(body.action ?? "")));
				} catch (err) {
					const status = (err as { status?: number }).status;
					if (!status) throw err;
					return reply(status, { error: (err as Error).message, state: g.state() });
				}
			}
			if (url.pathname === "/interrupt") {
				// Also an operator's GUMI batch, which runs while the agent is idle too.
				const stopped = (teleop?.stop() ?? false) || manual !== undefined;
				manual?.abort();
				const idle = ctx.isIdle();
				if (!idle) ctx.abort();
				return reply(idle && !stopped ? 200 : 202, { ok: true, interrupted: !idle || stopped });
			}
			return reply(404, { error: "not found" });
		},
	};
	setInterval(() => {
		for (const res of clients) res.write(": ping\n\n");
	}, 15_000).unref();
	return hub;
}

function startHub(host: string, port: number, language: string, liveFps: number): Promise<Hub> {
	const lang = language === "zh-cn" ? "zh-cn" : "en";
	const page = readFileSync(new URL("./page.html", import.meta.url), "utf8").replace("__LANG__", lang);
	return new Promise((resolve, reject) => {
		let hub: Hub | undefined;
		const server = createServer((req, res) => {
			hub?.handle(req, res).catch((err) => {
				if (!res.headersSent) res.writeHead(500, { "Content-Type": "application/json" });
				res.end(JSON.stringify({ error: String(err) }));
			});
		});
		server.once("error", reject);
		server.listen(port, host, () => {
			const { port: bound } = server.address() as { port: number };
			const shown = host === "0.0.0.0" || host === "::" ? hostname() : host.includes(":") ? `[${host}]` : host;
			hub = createHub(server, `http://${shown}:${bound}/`, page, liveFps);
			server.unref();
			resolve(hub);
		});
	});
}

export default function dashboard(pi: ExtensionAPI) {
	pi.registerFlag("dashboard", { type: "boolean", default: false, description: "Serve the live dashboard" });
	pi.registerFlag("dashboard-host", { type: "string", default: "127.0.0.1", description: "Dashboard bind address" });
	pi.registerFlag("dashboard-port", { type: "string", default: "0", description: "Dashboard port (0 = any free)" });
	pi.registerFlag("dashboard-language", { type: "string", default: "en", description: "Dashboard UI: en | zh-cn" });
	pi.registerFlag("dashboard-live-fps", {
		type: "string",
		default: "5",
		description: "Most frames per second of the dashboard's live camera stream",
	});
	let hub: Hub | undefined;
	let status: RobotStatus | undefined;
	const on = <T>(fn: (h: Hub) => T) => (hub ? fn(hub) : undefined);
	const teleop = gumi(pi, {
		onState: (s) => on((h) => h.gumiState(s)),
		onStep: (label, step, result, isError) => on((h) => h.teleopStep(label, step, result, isError)),
		// The operator unit the episode video labels the next frames with (../video.ts).
		onAction: (label) =>
			pi.events.emit(
				NOTE_EVENT,
				(label === null ? { actor: null } : { actor: "human", action: label }) satisfies VideoNote,
			),
	});
	// The robot in this runtime publishes its status here; it may do so before the hub is attached.
	pi.events.on(STATUS_EVENT, (data) => {
		status = data as RobotStatus;
		on((h) => h.status(status as RobotStatus));
	});
	pi.events.on(FRAME_EVENT, (data) => on((h) => h.frame(data as NdArray)));
	// The robot's tools for manual calls and its video directory, published at its session start (before or after ours).
	let prims: PrimitivesHandle | undefined;
	let videoDir: string | undefined;
	pi.events.on(PRIMITIVES_EVENT, (data) => {
		prims = data as PrimitivesHandle;
		on((h) => h.primitives(prims));
	});
	pi.events.on(VIDEO_DIR_EVENT, (data) => {
		videoDir = String(data);
		on((h) => h.videoDir(videoDir as string));
	});

	pi.on("session_start", async (_event, ctx) => {
		if (pi.getFlag("dashboard") !== true) return;
		const slot = globalThis as Record<symbol, Promise<Hub> | undefined>;
		const first = !slot[HUB];
		slot[HUB] ??= startHub(
			String(pi.getFlag("dashboard-host")),
			Number(pi.getFlag("dashboard-port")),
			String(pi.getFlag("dashboard-language")),
			Math.min(30, Math.max(0.2, Number(pi.getFlag("dashboard-live-fps")) || 5)),
		);
		hub = await slot[HUB];
		hub.attach(pi, ctx, status, teleop);
		hub.primitives(prims);
		if (videoDir) hub.videoDir(videoDir);
		if (first) {
			if (ctx.hasUI) ctx.ui.notify(`Dashboard: ${hub.url}`, "info");
			else console.error(`[dashboard] ${hub.url}`);
		}
	});
	pi.on("session_shutdown", (event) => {
		on((h) => (event.reason === "quit" ? h.close() : h.detach()));
		hub = undefined;
	});
	pi.on("agent_start", () => on((h) => h.setRunning(true)));
	pi.on("agent_end", () => on((h) => h.agentEnd()));
	pi.on("message_start", (event) => on((h) => h.messageStart(event.message)));
	pi.on("message_update", (event) => on((h) => h.messageUpdate(event.assistantMessageEvent)));
	pi.on("message_end", (event) => on((h) => h.messageEnd(event.message)));
	pi.on("tool_execution_start", (event) => on((h) => h.toolStart(event.toolCallId)));
}
