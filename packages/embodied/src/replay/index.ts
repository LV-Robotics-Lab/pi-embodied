/**
 * Verbatim replay of a recorded pi-embodied session, for any robot (`--replay`).
 *
 *   pi -p -e packages/embodied/src/<robot> -e packages/embodied/src/replay \
 *     [-e packages/embodied/src/dashboard --dashboard] --model replay/session \
 *     --replay <session.jsonl | session dir> <the robot's usual task flags> "Replay the session."
 *
 * The replay is the planner, and in pi the planner is the model: this extension registers a
 * `replay/session` provider whose every turn is the recorded session's next assistant turn: its
 * text and its tool calls with their recorded arguments, all of a turn's calls together, in order
 * (with zero usage; an abort ends the replay). The robot's own tools execute them, so the session,
 * `finish` and the `robot_result` row are exactly those of a real run. After the last recorded turn
 * the replay ends; a recording without `finish` ends with a text turn, and the robot reports the
 * episode at shutdown.
 *
 * The calls are verbatim: results of this run never change them. Calls that never ran in the
 * recording are skipped with a note: those the robot base, the operator or ../gumi blocked (refused,
 * or dropped as stale after an operator takeover), and every call of a reply that did not end
 * normally (stopReason other than toolUse/stop: pi runs none of them; a reply cut at the output token
 * limit may carry truncated arguments); replayed, they would move the robot where the recorded run did
 * not. Calls to pi's workspace-writing tools (write, edit, bash) are skipped too: replayed, they would
 * rewrite the recorded run's files. A replayed call whose outcome diverges from the recording's ends
 * the replay at that call, even inside a multi-call turn (the rest of the turn is blocked unsent): one
 * that errors where the recorded one did not (the robot erred, broke, or stopped answering), or that
 * succeeds where the recorded one failed (the robot is now where the recording never went). An error
 * the recording also had is noted and the replay goes on.
 *
 * The task comes from the robot's usual flags; a warning goes to stderr when the recording's
 * `robot_task` entry differs from this run's.
 *
 * This is Flash (../flash, `flash/replay`) without re-localization: Flash also sends recorded calls,
 * but first lets the robot's hook rewrite each one against the live scene and retries failed picks.
 * The two stay separate providers for now.
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import {
	type Api,
	type AssistantMessage,
	createAssistantMessageEventStream,
	createProvider,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type ToolCall,
	type ToolResultMessage,
	type TranscriptContext,
	type Usage,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { TASK_ENTRY } from "../robot.ts";

type Json = Record<string, unknown>;
/** A recorded call; `error` is its recorded error result, if it had one. */
export type Call = { name: string; arguments: Json; error?: string };
/** `blocked`: the turn's calls that never ran in the recording, with the reason they were blocked. */
export type Turn = { text: string; thinking: string; calls: Call[]; blocked: { name: string; reason: string }[] };
export type Recording = { path: string; turns: Turn[]; task?: Json; finished: boolean };

/** pi's tools that change the workstation, not the robot. */
const SKIP = new Set(["write", "edit", "bash"]);
/**
 * Error results of calls that never executed: pi's own (a block without a reason, an abort before the
 * call ran, an unknown tool) and the block reasons of the robot base (../robot.ts), the operator
 * (../operator.ts), ../gumi's stale drop, the units verifier and the memory guard.
 */
const NOT_RUN = [
	/^Tool execution was blocked$/,
	/^Operation aborted$/,
	/^Tool \S+ not found$/,
	/^Tool call "\S+" was not executed: the response hit the output token limit/,
	/^\S+ is not available\.$/,
	/^The robot failed: [\s\S]*The episode is over\.$/,
	/^The episode is finished\.$/,
	/^Planner \w+ budget exhausted; the episode is over\.$/,
	/^operator (submitted a terminal verdict|aborted this run)/,
	/^refused; request_scene_reset/,
	/^finish refused/,
	/was not executed; decide again from the current observation\.$/,
	/^file access (is disabled|denied)/,
];
const resultText = (m: { content: { type: string; text?: string }[] }) =>
	m.content.flatMap((c) => (c.type === "text" && c.text ? [c.text] : [])).join(" ");
const brief = (s: string) => s.replace(/\s+/g, " ").slice(0, 300);

/** The custom entry recording which session this run replays. */
export const REPLAY_ENTRY = "replay_source";

const isSession = (path: string) => {
	try {
		const first = readFileSync(path, "utf8").split("\n", 1)[0];
		return (JSON.parse(first) as Json).type === "session";
	} catch {
		return false;
	}
};

/** The session file at `path`, or the newest session file in the directory `path`. */
export function sessionFile(path: string): string {
	if (!statSync(path).isDirectory()) return path;
	const files = readdirSync(path)
		.filter((f) => f.endsWith(".jsonl"))
		.map((f) => join(path, f))
		.filter(isSession)
		.sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs);
	if (!files.length) throw new Error(`no pi session file in ${path}`);
	return files[0];
}

/** The recorded assistant turns with tool calls, and the task, on the branch ending at the session's last entry. */
export function loadRecording(path: string): Recording {
	const file = sessionFile(resolve(path));
	const entries = readFileSync(file, "utf8")
		.split("\n")
		.filter((l) => l.trim())
		.map((l) => JSON.parse(l) as Json)
		.filter((e) => typeof e.id === "string");
	const byId = new Map(entries.map((e) => [e.id as string, e]));
	const branch: Json[] = [];
	for (let e = entries.at(-1); e; e = typeof e.parentId === "string" ? byId.get(e.parentId) : undefined)
		branch.unshift(e);
	const results = new Map<string, ToolResultMessage>();
	for (const e of branch) {
		const m = e.type === "message" ? (e.message as Message) : undefined;
		if (m?.role === "toolResult") results.set(m.toolCallId, m);
	}
	const turns: Turn[] = [];
	let task: Json | undefined;
	for (const e of branch) {
		if (e.type === "custom" && e.customType === TASK_ENTRY) task = e.data as Json;
		const m = e.type === "message" ? (e.message as AssistantMessage) : undefined;
		if (m?.role !== "assistant") continue;
		// pi runs the calls of a reply that ended normally only: an errored or aborted reply's calls never
		// ran, and a reply cut at the output token limit ("length") has calls whose arguments may be truncated.
		const abnormal = m.stopReason === "toolUse" || m.stopReason === "stop" ? undefined : m.stopReason;
		const calls: Call[] = [];
		const blocked: Turn["blocked"] = [];
		for (const c of m.content) {
			if (c.type !== "toolCall") continue;
			const r = results.get(c.id);
			const error = r?.isError ? resultText(r) : undefined;
			if (abnormal !== undefined)
				blocked.push({
					name: c.name,
					reason: `the reply ended with stopReason "${abnormal}"${abnormal === "length" ? ", so its arguments may be truncated" : ""}`,
				});
			else if (error !== undefined && NOT_RUN.some((p) => p.test(error.trim())))
				blocked.push({ name: c.name, reason: brief(error) });
			else calls.push({ name: c.name, arguments: c.arguments, ...(error !== undefined ? { error } : {}) });
		}
		if (!calls.length && !blocked.length) continue;
		const text = m.content.flatMap((c) => (c.type === "text" ? [c.text] : [])).join("\n");
		const thinking = m.content.flatMap((c) => (c.type === "thinking" ? [c.thinking] : [])).join("\n");
		turns.push({ text, thinking, calls, blocked });
	}
	const finished = turns.some((t) => t.calls.some((c) => c.name === "finish"));
	return { path: file, turns, task, finished };
}

/** Fields where this run's task differs from the recorded one, as `key: recorded -> now`. */
export function taskDiff(recorded: Json, now: Json | undefined): string[] {
	const keys = [...new Set([...Object.keys(recorded), ...Object.keys(now ?? {})])];
	return keys.flatMap((k) => {
		const a = recorded[k] === undefined ? "" : String(recorded[k]);
		const b = now?.[k] === undefined ? "" : String(now[k]);
		return a === b ? [] : [`${k}: ${a || "(none)"} -> ${b || "(none)"}`];
	});
}

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
/** A replay runs no model, so every turn reports zero usage. */
const USAGE: Usage = { ...ZERO_COST, totalTokens: 0, cost: { ...ZERO_COST, total: 0 } };
const MODEL: Model<"replay"> = {
	id: "session",
	name: "Session replay",
	api: "replay",
	provider: "replay",
	baseUrl: "",
	reasoning: false,
	input: ["text", "image"],
	cost: ZERO_COST,
	contextWindow: 100_000_000,
	maxTokens: 16_384,
};

export default function replay(pi: ExtensionAPI) {
	pi.registerFlag("replay", {
		type: "string",
		description: "Recorded pi session (.jsonl, or a session dir: its newest session) for --model replay/session",
	});

	let recording: Recording | undefined;
	let loadError: string | undefined;
	let cursor = 0;
	let started = false;
	let over = false;
	let hasUI = false;
	let emitted: { id: string; name: string; recorded?: string }[] = [];
	/** Why the replay stopped inside the current batch: its first call whose outcome diverged from the recording. */
	let halt: string | undefined;
	let ids = 0;

	const warn = (ctx: ExtensionContext | undefined, s: string) => {
		if (ctx?.hasUI) ctx.ui.notify(s, "warning");
		else console.error(`[replay] ${s}`);
	};

	pi.on("session_start", (_event, ctx) => {
		hasUI = ctx.hasUI;
		cursor = ids = 0;
		started = over = false;
		emitted = [];
		recording = loadError = halt = undefined;
		const path = pi.getFlag("replay");
		if (!path) return;
		try {
			recording = loadRecording(resolve(ctx.cwd, String(path)));
		} catch (err) {
			loadError = `cannot read --replay ${path}: ${err instanceof Error ? err.message : String(err)}`;
			warn(ctx, loadError);
		}
	});

	pi.on("before_agent_start", (_event, ctx) => {
		if (!pi.getFlag("replay") || started) return;
		if (ctx.model?.provider !== "replay") warn(ctx, `--replay is set but the model is not replay/session`);
		const recorded = recording?.task;
		if (!recorded) return;
		const now = ctx.sessionManager
			.getBranch()
			.filter(
				(e) => e.type === "custom" && e.customType === TASK_ENTRY && (e.data as Json)?.robot === recorded.robot,
			)
			.pop();
		const diff = taskDiff(recorded, now?.type === "custom" ? (now.data as Json) : undefined);
		if (diff.length) warn(ctx, `this run's task differs from the recorded robot_task: ${diff.join("; ")}`);
	});

	// Whatever ended the run (finish, a terminating block, an abort), the replay does not resume in this session.
	pi.on("agent_end", () => {
		if (started) over = true;
	});

	/** How this run's result of an emitted call diverged from the recording's, if it did. */
	const divergence = (e: { name: string; recorded?: string }, isError: boolean, text: string) =>
		isError && e.recorded === undefined
			? `${e.name} failed: ${brief(text)}; it succeeded in the recording`
			: !isError && e.recorded !== undefined
				? `${e.name} succeeded; it failed in the recording (${brief(e.recorded)})`
				: undefined;

	// A multi-call turn stops at its first diverging call: the rest of the batch is blocked unsent.
	// Robot tools and finish are sequential, so pi runs such a batch one call at a time and this hook
	// sees each result before the next call is prepared.
	pi.on("tool_result", (event) => {
		const e = emitted.find((x) => x.id === event.toolCallId);
		if (!e || halt !== undefined) return;
		halt = divergence(e, event.isError, resultText(event));
	});
	pi.on("tool_call", (event) => {
		if (halt === undefined || !emitted.some((x) => x.id === event.toolCallId)) return;
		return { block: true, reason: `replay stopped (${halt}); this recorded call was not sent` };
	});

	/**
	 * Errors among the results of the last replayed calls, as notes; `stop` when an outcome diverged
	 * from the recording's in either direction (a call failed where the recorded one succeeded: the robot
	 * erred or broke; or succeeded where it failed: the robot is now where the recording never went).
	 */
	function failures(messages: Message[]): { notes: string[]; stop: boolean } {
		const results = new Map<string, ToolResultMessage>();
		for (const m of messages) if (m.role === "toolResult") results.set(m.toolCallId, m);
		let stop = false;
		const notes = emitted.flatMap((e) => {
			const m = results.get(e.id);
			if (stop) return [`${e.name} not sent`];
			if (!m) return [`${e.name}: no result; continuing`];
			const diverged = divergence(e, m.isError, resultText(m));
			if (diverged !== undefined) {
				stop = true;
				return [diverged];
			}
			return m.isError ? [`${e.name} failed, as in the recording: ${brief(resultText(m))}; continuing`] : [];
		});
		return { notes, stop };
	}

	/** The next turn: the recording's next tool-call turn, or a closing text turn. */
	function next(messages: Message[]): { notes: string[]; text: string; thinking: string; calls: ToolCall[] } {
		const failed = started ? failures(messages) : { notes: [], stop: false };
		const notes = failed.notes;
		emitted = [];
		halt = undefined;
		const end = (text: string) => ({
			notes,
			text: [...notes.map((n) => `replay: ${n}`), text].join("\n"),
			thinking: "",
			calls: [],
		});
		if (over) return end("The replay already ran in this session.");
		if (!started) {
			started = true;
			if (recording) {
				pi.appendEntry(REPLAY_ENTRY, { session: recording.path, turns: recording.turns.length });
				notes.push(`${recording.turns.length} recorded turns from ${recording.path}`);
			}
		}
		if (!recording) {
			over = true;
			return end(loadError ?? "no --replay session given; nothing to replay.");
		}
		if (failed.stop) {
			over = true;
			return end(
				`Replay stopped: a call's outcome diverged from the recording's; no further recorded motion is sent (${cursor} of ${recording.turns.length} turns replayed).`,
			);
		}
		while (cursor < recording.turns.length) {
			const turn = recording.turns[cursor++];
			for (const b of turn.blocked) notes.push(`skipped ${b.name} (it never ran in the recording: ${b.reason})`);
			for (const c of turn.calls)
				if (SKIP.has(c.name)) notes.push(`skipped ${c.name} (it would rewrite the recorded run's files)`);
			const calls = turn.calls
				.filter((c) => !SKIP.has(c.name))
				.map(
					(c): ToolCall => ({
						type: "toolCall",
						id: `replay_${++ids}`,
						name: c.name,
						arguments: structuredClone(c.arguments) as ToolCall["arguments"],
					}),
				);
			if (!calls.length) continue;
			const kept = turn.calls.filter((c) => !SKIP.has(c.name));
			emitted = calls.map((c, i) => ({ id: c.id, name: c.name, recorded: kept[i].error }));
			const text = [...notes.map((n) => `replay: ${n}`), turn.text].filter(Boolean).join("\n");
			return { notes, text, thinking: turn.thinking, calls };
		}
		over = true;
		return end(
			recording.finished
				? "Replay complete: all recorded turns were replayed."
				: "Replay complete: the recording has no finish; the episode is reported at shutdown.",
		);
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
		if (options?.signal?.aborted) {
			if (started) over = true;
			message.stopReason = "aborted";
			message.errorMessage = "Replay aborted";
			stream.push({ type: "error", reason: "aborted", error: message });
			stream.end();
			return stream;
		}
		const turn = next(context.messages as Message[]);
		if (!hasUI) for (const n of turn.notes) console.error(`[replay] ${n}`);
		stream.push({ type: "start", partial: message });
		if (turn.thinking) {
			const contentIndex = message.content.push({ type: "thinking", thinking: turn.thinking }) - 1;
			stream.push({ type: "thinking_start", contentIndex, partial: message });
			stream.push({ type: "thinking_delta", contentIndex, delta: turn.thinking, partial: message });
			stream.push({ type: "thinking_end", contentIndex, content: turn.thinking, partial: message });
		}
		if (turn.text) {
			const contentIndex = message.content.push({ type: "text", text: turn.text }) - 1;
			stream.push({ type: "text_start", contentIndex, partial: message });
			stream.push({ type: "text_delta", contentIndex, delta: turn.text, partial: message });
			stream.push({ type: "text_end", contentIndex, content: turn.text, partial: message });
		}
		for (const call of turn.calls) {
			const contentIndex = message.content.push(call) - 1;
			stream.push({ type: "toolcall_start", contentIndex, partial: message });
			stream.push({ type: "toolcall_end", contentIndex, toolCall: call, partial: message });
		}
		message.stopReason = turn.calls.length ? "toolUse" : "stop";
		stream.push({ type: "done", reason: message.stopReason, message });
		stream.end();
		return stream;
	}

	pi.registerProvider(
		createProvider({
			id: "replay",
			name: "Replay",
			auth: { apiKey: { name: "Replay", resolve: async () => ({ auth: {} }) } },
			models: [MODEL],
			api: { stream: streamSimple, streamSimple },
		}),
	);
}
