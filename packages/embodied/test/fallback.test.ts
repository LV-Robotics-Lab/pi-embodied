import assert from "node:assert/strict";
import { test } from "node:test";
import {
	type AssistantMessage,
	type AssistantMessageEvent,
	createAssistantMessageEventStream,
	type Model,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { FALLBACK_ENTRY, fallback, fallbackArgs } from "../src/fallback.ts";
import { defineRobot, RESULT_ENTRY } from "../src/robot.ts";

type Handler = (event: any, ctx: any) => unknown;
/** One scripted delegate turn: a reply, an error message, an error after some text, or silence. */
type Script = "ok" | { error: string; cost?: number } | { midway: string } | "hang";

const PRIMARY = "relay/gpt-6-astra";
const BACKUP = "selfhost/muse-glimmer-30b";
const ARGV = ["node", "pi", "--model", `fallback/${PRIMARY}`, "--fallback-model", BACKUP];

const model = (ref: string): Model<any> => {
	const [provider, id] = [ref.slice(0, ref.indexOf("/")), ref.slice(ref.indexOf("/") + 1)];
	return {
		id,
		provider,
		api: "openai-completions",
		name: id,
		baseUrl: "",
		reasoning: true,
		input: ["text"],
		cost: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0 },
		// The primary's window is far larger than the backup's (relay GPT vs a 64k self-hosted model).
		contextWindow: ref === PRIMARY ? 400_000 : 65_536,
		maxTokens: ref === PRIMARY ? 128_000 : 16_384,
	};
};

/** A stub pi and model registry: each delegate's turns follow its script; `calls` records who was asked with what. */
function fakePi(flagValues: Record<string, unknown> = {}) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const providers: any[] = [];
	const entries: { type: string; data: any }[] = [];
	const stderr: string[] = [];
	const tools = new Map<string, any>();
	const scripts: Record<string, Script[]> = { [PRIMARY]: [], [BACKUP]: [] };
	const calls: { model: string; options: any; signal: AbortSignal }[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerProvider: (p: any) => providers.push(p),
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		setActiveTools: () => {},
		getActiveTools: () => [],
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: { emit: (channel: string, data: any) => events.push({ channel, data }), on: () => () => {} },
	} as unknown as ExtensionAPI;
	const events: { channel: string; data: any }[] = [];
	const registry = {
		find: (provider: string, id: string) => {
			const ref = `${provider}/${id}`;
			return ref in scripts ? model(ref) : undefined;
		},
		streamSimple(m: Model<any>, _context: unknown, options: any) {
			const ref = `${m.provider}/${m.id}`;
			calls.push({ model: ref, options, signal: options.signal });
			const script = scripts[ref].shift() ?? "ok";
			const stream = createAssistantMessageEventStream();
			const message: AssistantMessage = {
				role: "assistant",
				content: [],
				api: m.api,
				provider: m.provider,
				model: m.id,
				usage: {
					input: 10,
					output: 5,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 15,
					cost: { input: 0.1, output: 0.05, cacheRead: 0, cacheWrite: 0, total: 0.15 },
				},
				stopReason: "stop",
				timestamp: 1,
			};
			const fail = (text: string) => {
				message.stopReason = "error";
				message.errorMessage = text;
				stream.push({ type: "error", reason: "error", error: message });
				stream.end();
			};
			options.signal.addEventListener("abort", () => {
				message.stopReason = "aborted";
				message.errorMessage = "aborted";
				stream.push({ type: "error", reason: "aborted", error: message });
				stream.end();
			});
			if (script === "hang") return stream;
			stream.push({ type: "start", partial: message });
			if (typeof script === "object" && "error" in script) {
				// A delegate that billed something before failing reports it on the error message.
				if (script.cost !== undefined)
					message.usage = {
						...message.usage,
						totalTokens: 30,
						cost: { ...message.usage.cost, total: script.cost },
					};
				else message.usage = { ...message.usage, totalTokens: 0, cost: { ...message.usage.cost, total: 0 } };
				fail(script.error);
				return stream;
			}
			const text = `from ${ref}`;
			message.content.push({ type: "text", text });
			stream.push({ type: "text_start", contentIndex: 0, partial: message });
			stream.push({ type: "text_delta", contentIndex: 0, delta: text, partial: message });
			if (typeof script === "object") {
				setTimeout(() => fail(script.midway), 5);
				return stream;
			}
			stream.push({ type: "text_end", contentIndex: 0, content: text, partial: message });
			stream.push({ type: "done", reason: "stop", message });
			stream.end();
			return stream;
		},
	};
	const ctx = {
		hasUI: false,
		cwd: "/",
		ui: { notify: () => {} },
		// The session's model object: pi compacts against its contextWindow, so the module sizes it at session start.
		model: { provider: "fallback", id: PRIMARY, contextWindow: 128_000, maxTokens: 16_384 } as Record<
			string,
			unknown
		>,
		modelRegistry: registry,
		sessionManager: { getBranch: () => [] },
	};
	const emit = async (name: string, event: Record<string, unknown> = {}) => {
		for (const fn of handlers.get(name) ?? []) await fn({ type: name, ...event }, ctx);
	};
	const log = console.error;
	console.error = (line: string) => stderr.push(line);
	const restore = () => {
		console.error = log;
	};
	/** One turn of the registered `fallback` model: its events and final message. */
	const turn = async (signal?: AbortSignal) => {
		const provider = providers.find((p) => p.id === "fallback");
		const events: AssistantMessageEvent[] = [];
		const stream = provider.streamSimple(provider.getModels()[0], { messages: [] }, { signal, reasoning: "low" });
		for await (const ev of stream) events.push(ev);
		return { events, message: await stream.result() };
	};
	const switches = () => entries.filter((e) => e.type === FALLBACK_ENTRY).map((e) => e.data);
	return { pi, ctx, emit, turn, providers, entries, events, switches, stderr, scripts, calls, tools, restore };
}

test("fallbackArgs reads --model fallback/<primary> and --fallback-model in both flag forms", () => {
	assert.deepEqual(fallbackArgs(ARGV), { primary: PRIMARY, fallback: BACKUP });
	assert.deepEqual(fallbackArgs(["--model=fallback/a/b", `--fallback-model=${BACKUP}`]), {
		primary: "a/b",
		fallback: BACKUP,
	});
	assert.deepEqual(fallbackArgs(["--model", PRIMARY, "--fallback-model", BACKUP]), {
		primary: undefined,
		fallback: BACKUP,
	});
	assert.deepEqual(fallbackArgs(["--model", PRIMARY]), { primary: undefined, fallback: undefined });
});

test("without --fallback-model nothing is registered", (t) => {
	const f = fakePi();
	t.after(f.restore);
	assert.equal(fallback(f.pi, ["node", "pi", "--model", PRIMARY]), undefined);
	assert.deepEqual(f.providers, []);
	assert.equal(f.pi.getFlag("fallback-after"), undefined);
});

test("the primary is retried until --fallback-after, then the backup plans; the entry, usage and model id follow", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	const fb = fallback(f.pi, ARGV)!;
	await f.emit("session_start");
	const [m] = f.providers[0].getModels();
	assert.equal(`${m.provider}/${m.id}`, `fallback/${PRIMARY}`);
	f.scripts[PRIMARY].push(
		{ error: "Request failed (503): upstream unavailable" },
		{ error: "Request failed (400): model not supported by this channel" },
	);
	const first = await f.turn();
	assert.equal(first.message.stopReason, "stop");
	assert.equal(`${first.message.provider}/${first.message.model}`, BACKUP, "the reply is the backup's");
	assert.equal(first.message.usage.cost.total, 0.15, "usage and cost pass through");
	const provider = (e: AssistantMessageEvent) =>
		(e.type === "done" ? e.message : e.type === "error" ? e.error : e.partial).provider;
	assert.ok(
		first.events.every((e) => provider(e) === "selfhost"),
		"no primary events leak",
	);
	assert.deepEqual(
		f.calls.map((c) => c.model),
		[PRIMARY, PRIMARY, BACKUP],
	);
	assert.equal(f.calls[0].options.reasoning, "low", "stream options pass through");
	assert.equal(f.calls[0].options.apiKey, undefined, "the wrapper's (empty) auth is not forwarded");
	assert.deepEqual(f.switches(), [
		{
			from: PRIMARY,
			to: BACKUP,
			reason: "2 consecutive failures; last: Request failed (400): model not supported by this channel",
			turn: 1,
			failed: { attempts: 2, cost_usd: 0, tokens: 0 },
		},
	]);
	// The rest of the episode is planned by the backup without asking the primary.
	const second = await f.turn();
	assert.equal(second.message.model, "muse-glimmer-30b");
	assert.deepEqual(
		f.calls.map((c) => c.model),
		[PRIMARY, PRIMARY, BACKUP, BACKUP],
	);
	assert.deepEqual(fb.result(), { planner_models: { primary: 0, fallback: 2 } });
	assert.equal(f.stderr.length, 2, "one line for the retried failure, one for the switch");
	// A new session starts on the primary again.
	await f.emit("session_start");
	assert.deepEqual(fb.result(), { planner_models: { primary: 0, fallback: 0 } });
	await f.turn();
	assert.equal(f.calls.at(-1)?.model, PRIMARY);
});

test("a single failure is retried on the primary; a success resets the count and writes no entry", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	const fb = fallback(f.pi, ARGV)!;
	await f.emit("session_start");
	f.scripts[PRIMARY].push({ error: "fetch failed" });
	const r = await f.turn();
	assert.equal(r.message.model, "gpt-6-astra");
	f.scripts[PRIMARY].push({ error: "fetch failed" });
	assert.equal((await f.turn()).message.model, "gpt-6-astra", "the count restarted after the success");
	assert.deepEqual(f.switches(), []);
	assert.deepEqual(fb.result(), { planner_models: { primary: 2, fallback: 0 } });
});

test("an error that matches neither pi's retryable errors nor --fallback-on ends the turn as is", async (t) => {
	const f = fakePi({ "fallback-after": "1" });
	t.after(f.restore);
	fallback(f.pi, ARGV);
	await f.emit("session_start");
	f.scripts[PRIMARY].push({ error: "Request failed (400): invalid tool schema" });
	const r = await f.turn();
	assert.equal(r.message.stopReason, "error");
	assert.equal(r.message.model, "gpt-6-astra");
	assert.deepEqual(f.switches(), []);
	f.scripts[PRIMARY].push({ error: "Request failed (400): invalid tool schema" });
	assert.equal((await f.turn()).message.model, "gpt-6-astra", "still on the primary");
});

test("--fallback-on widens the matcher", async (t) => {
	const f = fakePi({ "fallback-after": "1", "fallback-on": "invalid tool" });
	t.after(f.restore);
	fallback(f.pi, ARGV);
	await f.emit("session_start");
	f.scripts[PRIMARY].push({ error: "Request failed (400): invalid tool schema" });
	assert.equal((await f.turn()).message.model, "muse-glimmer-30b");
});

test("a failure midway through the primary's reply is redone on the backup within the same turn", async (t) => {
	const f = fakePi({ "fallback-after": "1" });
	t.after(f.restore);
	const fb = fallback(f.pi, ARGV)!;
	await f.emit("session_start");
	f.scripts[PRIMARY].push({ midway: "Request failed (502): bad gateway" });
	const r = await f.turn();
	assert.equal(r.message.stopReason, "stop");
	assert.equal(r.message.model, "muse-glimmer-30b");
	assert.equal(r.events[0].type, "start");
	assert.equal((r.events[0] as any).partial.model, "muse-glimmer-30b", "the primary's partial text never reached pi");
	assert.deepEqual(r.message.content, [{ type: "text", text: `from ${BACKUP}` }]);
	assert.deepEqual(fb.result(), { planner_models: { primary: 0, fallback: 1 } });
	assert.equal(f.switches().length, 1);
});

test("an abort propagates to the delegate and is never a failure", async (t) => {
	const f = fakePi({ "fallback-after": "1" });
	t.after(f.restore);
	fallback(f.pi, ARGV);
	await f.emit("session_start");
	f.scripts[PRIMARY].push("hang");
	const ac = new AbortController();
	setTimeout(() => ac.abort(), 5);
	const r = await f.turn(ac.signal);
	assert.equal(r.message.stopReason, "aborted");
	assert.equal(r.message.model, "gpt-6-astra");
	assert.ok(f.calls[0].signal.aborted, "the delegate's request was aborted");
	assert.deepEqual(f.switches(), []);
	assert.deepEqual(
		f.calls.map((c) => c.model),
		[PRIMARY],
	);
});

test("a turn that streams nothing for --fallback-timeout is aborted and counts as a failure", async (t) => {
	const f = fakePi({ "fallback-after": "1", "fallback-timeout": "0.02" });
	t.after(f.restore);
	fallback(f.pi, ARGV);
	await f.emit("session_start");
	f.scripts[PRIMARY].push("hang");
	const r = await f.turn();
	assert.equal(r.message.model, "muse-glimmer-30b");
	assert.ok(f.calls[0].signal.aborted, "the hung request was aborted");
	assert.match(f.switches()[0].reason, /no reply from relay\/gpt-6-astra for 0.02 s/);
});

test("the backup failing ends the turn with its error; no further switch", async (t) => {
	const f = fakePi({ "fallback-after": "1" });
	t.after(f.restore);
	fallback(f.pi, ARGV);
	await f.emit("session_start");
	f.scripts[PRIMARY].push({ error: "Request failed (503): down" });
	f.scripts[BACKUP].push({ error: "Request failed (503): also down" });
	const r = await f.turn();
	assert.equal(r.message.stopReason, "error");
	assert.equal(r.message.model, "muse-glimmer-30b");
	assert.equal(r.message.errorMessage, "Request failed (503): also down");
	assert.equal(f.switches().length, 1);
	assert.deepEqual(
		f.calls.map((c) => c.model),
		[PRIMARY, BACKUP],
	);
});

test("--fallback-retry-primary tries the primary again after M backup turns", async (t) => {
	const f = fakePi({ "fallback-after": "1", "fallback-retry-primary": "2" });
	t.after(f.restore);
	const fb = fallback(f.pi, ARGV)!;
	await f.emit("session_start");
	f.scripts[PRIMARY].push({ error: "Request failed (503): down" });
	await f.turn();
	await f.turn();
	assert.deepEqual(
		f.calls.map((c) => c.model),
		[PRIMARY, BACKUP, BACKUP],
	);
	// The probe fails: straight back to the backup, within the turn.
	f.scripts[PRIMARY].push({ error: "Request failed (503): still down" });
	assert.equal((await f.turn()).message.model, "muse-glimmer-30b");
	assert.deepEqual(
		f.calls.slice(3).map((c) => c.model),
		[PRIMARY, BACKUP],
	);
	await f.turn();
	// The probe succeeds: the primary plans again.
	assert.equal((await f.turn()).message.model, "gpt-6-astra");
	assert.deepEqual(
		f.switches().map((s) => [s.from === PRIMARY ? "p" : "f", s.turn]),
		[
			["p", 1],
			["f", 3],
			["p", 3],
			["f", 5],
		],
	);
	assert.deepEqual(fb.result(), { planner_models: { primary: 1, fallback: 4 } });
});

test("session start sizes the session's model to the smaller delegate's window and output cap", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	fallback(f.pi, ARGV);
	await f.emit("session_start");
	assert.equal(f.ctx.model.contextWindow, 65_536, "the backup's window, not the primary's 400k");
	assert.equal(f.ctx.model.maxTokens, 16_384);
	// Another model selected: the session's object is left alone.
	f.ctx.model = { provider: "relay", id: "gpt-6-astra", contextWindow: 400_000, maxTokens: 128_000 };
	await f.emit("session_start");
	assert.equal(f.ctx.model.contextWindow, 400_000);
});

test("what failed attempts consumed goes to the budget and into the switch entry", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	fallback(f.pi, ARGV);
	await f.emit("session_start");
	f.scripts[PRIMARY].push(
		{ error: "Request failed (503): down", cost: 0.02 },
		{ error: "Request failed (502): down", cost: 0.03 },
	);
	const r = await f.turn();
	assert.equal(r.message.model, "muse-glimmer-30b");
	assert.deepEqual(
		f.events.map((e) => [e.channel, e.data]),
		[
			["pi-embodied:vlm-cost", 0.02],
			["pi-embodied:vlm-cost", 0.03],
		],
		"each swallowed attempt's cost is reported the way side VLM calls are",
	);
	assert.deepEqual(f.switches()[0].failed, { attempts: 2, cost_usd: 0.05, tokens: 60 });
	// A failure that reported no usage costs nothing and emits nothing.
	await f.emit("session_start");
	f.scripts[PRIMARY].push({ error: "Request failed (503): down" }, { error: "Request failed (503): down" });
	await f.turn();
	assert.equal(f.events.length, 2);
	assert.deepEqual(f.switches()[1].failed, { attempts: 2, cost_usd: 0, tokens: 0 });
});

test("an unknown delegate is an error, not a crash", async (t) => {
	const argv = ["node", "pi", "--model", "fallback/nowhere/x", "--fallback-model", BACKUP];
	const f = fakePi();
	t.after(f.restore);
	fallback(f.pi, argv);
	await f.emit("session_start");
	const r = await f.turn();
	assert.equal(r.message.stopReason, "error");
	assert.match(r.message.errorMessage ?? "", /unknown model nowhere\/x/);
});

test("--fallback-model without --model fallback/... warns once per session", async (t) => {
	const argv = ["node", "pi", "--model", PRIMARY, "--fallback-model", BACKUP];
	const f = fakePi();
	t.after(f.restore);
	fallback(f.pi, argv);
	assert.deepEqual(f.providers[0].getModels(), [], "no primary: no model to register");
	f.ctx.model = { provider: "relay" };
	await f.emit("session_start");
	await f.emit("before_agent_start");
	await f.emit("before_agent_start");
	assert.deepEqual(f.stderr, [
		"[fallback] --fallback-model is set but the model is not fallback/<primary>; nothing fails over",
	]);
});

test("defineRobot mounts it from argv and the result row carries planner_models", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	const argv = process.argv;
	process.argv = [...ARGV];
	t.after(() => {
		process.argv = argv;
	});
	defineRobot(f.pi, {
		name: "toy",
		task: [],
		keepImages: 1,
		start: async () => ["finish"],
		result: () => ({}),
		finish: {
			description: "finish",
			parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
			result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
		},
	});
	assert.equal(f.providers.length, 1, "the fallback provider is registered by the robot");
	await f.emit("session_start");
	await f.emit("agent_start");
	f.scripts[PRIMARY].push({ error: "Request failed (503): down" }, { error: "Request failed (503): down" });
	await f.turn();
	await f.turn();
	await f.tools.get("finish").execute("1", { status: "success", summary: "ok" });
	await f.emit("agent_end");
	const [result] = f.entries.filter((e) => e.type === RESULT_ENTRY).map((e) => e.data);
	assert.deepEqual(result.planner_models, { primary: 0, fallback: 2 });
	assert.equal(f.switches().length, 1);
});

test("off: defineRobot without --fallback-model registers no provider and no planner_models", async (t) => {
	const f = fakePi();
	t.after(f.restore);
	const argv = process.argv;
	process.argv = ["node", "pi", "--model", PRIMARY];
	t.after(() => {
		process.argv = argv;
	});
	defineRobot(f.pi, {
		name: "toy",
		task: [],
		keepImages: 1,
		start: async () => ["finish"],
		result: () => ({}),
		finish: {
			description: "finish",
			parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
			result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
		},
	});
	assert.deepEqual(f.providers, []);
	await f.emit("session_start");
	await f.emit("agent_start");
	await f.tools.get("finish").execute("1", { status: "success", summary: "ok" });
	await f.emit("agent_end");
	const [result] = f.entries.filter((e) => e.type === RESULT_ENTRY).map((e) => e.data);
	assert.equal("planner_models" in result, false);
});
