/**
 * Flash: replay a recorded plan with live grounding and no LLM, for any robot with a Flash hook.
 *
 *   pi -p -e src/<robot> --model flash/replay <the robot's task and Flash flags> "Solve the task."
 *
 * Flash is the planner, and in pi the planner is the model: this module registers a `flash/replay`
 * provider whose every turn is the plan's next tool call (with zero usage; an abort ends the
 * replay). The robot's tools execute it, so the session, `finish`, and the `robot_result` row are
 * exactly those of an LLM run. It is mounted by `defineRobot` for a robot whose spec has `flash`.
 *
 * What a plan holds, where it lives, and how it meets the live scene are the robot's (its
 * `FlashHook`). Flash loads the robot's program, lets the robot re-localize it (`start`), then walks
 * the plan in order: the robot rewrites each planned call against what it localized (or skips it,
 * or stops the replay), the call runs, and the robot sees its result (`after`). A robot that names
 * its picks gets grasp retries: a pick the robot judges failed is retried in place after replaying
 * its approach (the motion since the last pick or release), and a pick that never takes hold skips
 * its carry up to and including the next release.
 *
 * Verbatim replay (../replay, `replay/session`) is Flash without re-localization: a recording's
 * calls sent unchanged. The two are separate providers for now.
 */

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

type Json = Record<string, unknown>;
export type FlashCall = { name: string; arguments: Json };
/** One tool result: its JSON (the text result parsed), its images, and its error text when it failed. */
export type FlashReply = { json: Json; images: string[]; error?: string };
/** One planned call; a robot's plans may carry more per entry (e.g. the anchor a waypoint was written against). */
export type FlashEntry = { action: string; arguments: Json };
export type FlashProgram<E extends FlashEntry = FlashEntry> = { name: string; plan: E[] };

/** The robot as the replay sees it: tool calls go out as model turns and resolve with their results. */
export type FlashRobot = {
	/** Run `calls` as one turn; their replies, in order. */
	act: (calls: FlashCall[]) => Promise<FlashReply[]>;
	/** Run one motion call; its reply becomes the latest observation. A failed call ends the replay. */
	move: (call: FlashCall) => Promise<FlashReply>;
	/** The latest motion reply (empty before the first). */
	latest: () => FlashReply;
	/** A line of the next turn's text. */
	note: (line: string) => void;
};

/** How a robot's picks are judged and retried. */
export type FlashPicks = {
	/** A grasp that can be judged and retried. */
	isPick: (name: string) => boolean;
	succeeded: (reply: FlashReply) => boolean;
	attempts: number;
	/** Calls that make up a pick's approach, replayed before a retry (the last `keep`). */
	approach: readonly string[];
	keep: number;
	/** Calls besides picks after which the approach starts over. */
	boundary: readonly string[];
	/** The call that ends a failed pick's carry; it is skipped with the carry. */
	release: string;
};

export type Rewritten = FlashCall | FlashCall[] | "skip" | "stop";

/** A replay underway: how the plan meets the live scene. */
export type FlashReplay<E extends FlashEntry = FlashEntry> = {
	/** Anchors located on the live scene, for the summary. */
	localized: number;
	/**
	 * The call to send for `entry`, with its arguments rewritten (several when one planned move is split
	 * into the robot's per-call steps); "skip" to leave it out; "stop" to end the replay.
	 */
	rewrite(entry: E): Rewritten | Promise<Rewritten>;
	/** Called after each sent call (after its retries). */
	after?: (call: FlashCall, reply: FlashReply) => void | Promise<void>;
	/** Retry picks that did not take hold; without it every call runs once. */
	picks?: FlashPicks;
};

/**
 * A robot's Flash: its programs, and how it replays one. `start` and `rewrite` are methods, so a hook
 * over the robot's own program type is a `FlashHook` (what RobotSpec takes).
 */
export type FlashHook<P extends FlashProgram = FlashProgram> = {
	/** This episode's program; `cwd` resolves relative plan directories. Throws when there is none. */
	load: (cwd: string) => P | Promise<P>;
	/** Re-localize the program on the live scene (the robot may move and observe) and return the replay. */
	start(program: P, robot: FlashRobot): Promise<FlashReplay<P["plan"][number]>>;
	/** The episode is over: no further calls are sent. */
	over: (latest: FlashReply) => boolean;
	solved: (latest: FlashReply) => boolean;
	/** A tool's text result that is not JSON but still reports the episode (e.g. it has ended), as JSON. */
	textResult?: (text: string) => Json | undefined;
};

/** Walk `program`: rewrite each call through the robot, run it, retry picks. */
export async function runFlash<P extends FlashProgram>(hook: FlashHook<P>, program: P, robot: FlashRobot) {
	const replay = await hook.start(program, robot);
	const { picks } = replay;
	const over = () => hook.over(robot.latest());
	let recent: FlashCall[] = [];
	let skipCarry = false;
	for (const entry of program.plan) {
		if (over()) break;
		const name = entry.action;
		if (skipCarry && picks) {
			// A pick that never took hold must not fall through into its carry.
			if (name === picks.release) {
				skipCarry = false;
				recent = [];
				continue;
			}
			if (!picks.isPick(name)) continue;
			skipCarry = false;
		}
		const rewritten = await replay.rewrite(entry);
		if (rewritten === "stop") break;
		if (rewritten === "skip") continue;
		for (const call of Array.isArray(rewritten) ? rewritten : [rewritten]) {
			if (over()) break;
			let reply = await robot.move(call);
			if (picks?.isPick(call.name) && !picks.succeeded(reply)) {
				for (let attempt = 1; attempt < picks.attempts && !over(); attempt++) {
					for (const again of recent) await robot.move({ name: again.name, arguments: { ...again.arguments } });
					reply = await robot.move({ name: call.name, arguments: { ...call.arguments } });
					if (picks.succeeded(reply)) break;
				}
				if (!picks.succeeded(reply)) {
					robot.note("pick unconfirmed, skipping its carry");
					skipCarry = true;
				}
			}
			await replay.after?.(call, reply);
			if (!picks) continue;
			if (picks.isPick(call.name) || picks.boundary.includes(call.name)) recent = [];
			else if (picks.approach.includes(call.name)) recent = [...recent, call].slice(-picks.keep);
		}
	}
	return { done: hook.solved(robot.latest()), anchors: replay.localized, plan: program.plan.length };
}

/** Parse the tool results for `ids` out of the transcript the model turn received. */
function repliesFor(messages: Message[], ids: string[], textResult?: (text: string) => Json | undefined): FlashReply[] {
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
			const reported = textResult?.(text);
			return reported ? { json: reported, images } : { json: {}, images, error: text };
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

/** Register the `flash/replay` model, replaying the programs of the robot's `hook`. */
export function flash<P extends FlashProgram>(pi: ExtensionAPI, hook: FlashHook<P>) {
	type Turn = { text: string; calls: ToolCall[] };
	let toModel: { resolve: (turn: Turn) => void; reject: (err: Error) => void } | undefined;
	let toReplay: { resolve: (messages: Message[]) => void; reject: (err: Error) => void } | undefined;
	let pending: string[] = [];
	let notes: string[] = [];
	let latest: FlashReply = { json: {}, images: [] };
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
	const toolCall = (c: FlashCall): ToolCall => ({
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

	async function act(batch: FlashCall[]): Promise<FlashReply[]> {
		if (stopped) throw stopped;
		const toolCalls = batch.map(toolCall);
		pending = toolCalls.map((c) => c.id);
		const results = new Promise<Message[]>((resolve, reject) => {
			toReplay = { resolve, reject };
		});
		say({ text: flush(), calls: toolCalls });
		return repliesFor(await results, pending, hook.textResult);
	}

	const robot: FlashRobot = {
		act,
		async move(call) {
			const [reply] = await act([call]);
			if (reply.error !== undefined) throw new Error(`${call.name} failed: ${reply.error}`);
			latest = reply;
			return reply;
		},
		latest: () => latest,
		note: (line) => notes.push(line),
	};

	async function run() {
		const t0 = Date.now();
		let status: "success" | "failure" = "failure";
		let summary: string;
		try {
			const program = await hook.load(cwd);
			notes.push(`replaying the ${program.name} program`);
			const out = await runFlash(hook, program, robot);
			status = out.done ? "success" : "failure";
			summary =
				`replayed the ${program.name} program: ${out.plan} actions, ${out.anchors} anchors re-localized, ` +
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
		latest = { json: {}, images: [] };
	});
}
