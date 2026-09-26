import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { type AssistantMessage, createAssistantMessageEventStream, type Model } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { API_GATE_EVENT } from "../src/api-gate.ts";
import { candidates, ENSEMBLE_ENTRY, ensemble, ensembleArgs } from "../src/ensemble.ts";
import { defineRobot, RESULT_ENTRY } from "../src/robot.ts";

type Handler = (event: any, ctx: any) => unknown;
/** One scripted delegate reply: a tool call (and text), an error, or silence until aborted. */
type Script =
	| { call?: string; text?: string; cost?: number; delay?: number }
	| { error: string; cost?: number }
	| "hang";

const BASE = "selfhost/muse-glimmer-30b";
const OTHER = "relay/gpt-6-astra";
const ARGV = ["node", "pi", "--model", `ensemble/${BASE}`];

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
		contextWindow: ref === OTHER ? 400_000 : 65_536,
		maxTokens: ref === OTHER ? 128_000 : 16_384,
	};
};

/** A stub pi and registry. Replies are scripted per call key `<ref>@<temperature>` (or `<ref>` without one); `synth` scripts the call whose last message is the ensemble note. */
function fakePi(flagValues: Record<string, unknown> = {}) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const providers: any[] = [];
	const entries: { type: string; data: any }[] = [];
	const tools = new Map<string, any>();
	const listeners = new Map<string, ((d: unknown) => void)[]>();
	const scripts: Record<string, Script> = {};
	const calls: { key: string; messages: any[]; options: any }[] = [];
	let running = 0;
	let peak = 0;
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
		events: {
			emit: (channel: string, data: unknown) => {
				for (const fn of listeners.get(channel) ?? []) fn(data);
			},
			on: (channel: string, fn: (d: unknown) => void) => {
				listeners.set(channel, [...(listeners.get(channel) ?? []), fn]);
				return () => {};
			},
		},
	} as unknown as ExtensionAPI;
	const registry = {
		find: (provider: string, id: string) => {
			const ref = `${provider}/${id}`;
			return ref === BASE || ref === OTHER ? model(ref) : undefined;
		},
		streamSimple(m: Model<any>, context: { messages: any[] }, options: any) {
			const ref = `${m.provider}/${m.id}`;
			const last = context.messages.at(-1);
			const synth = last?.role === "user" && JSON.stringify(last.content).includes("[ensemble]");
			const key = synth ? "synth" : options.temperature === undefined ? ref : `${ref}@${options.temperature}`;
			calls.push({ key, messages: context.messages, options });
			const script: Script = scripts[key] ?? { call: `act_${key}` };
			const stream = createAssistantMessageEventStream();
			const message: AssistantMessage = {
				role: "assistant",
				content: [],
				api: m.api,
				provider: m.provider,
				model: m.id,
				usage: {
					input: 100,
					output: 10,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 110,
					cost: { input: 0.01, output: 0.01, cacheRead: 0, cacheWrite: 0, total: 0.02 },
				},
				stopReason: "toolUse",
				timestamp: 1,
			};
			if (typeof script === "object" && script.cost !== undefined)
				message.usage.cost = { input: script.cost, output: 0, cacheRead: 0, cacheWrite: 0, total: script.cost };
			running++;
			peak = Math.max(peak, running);
			let settled = false;
			const settle = (fn: () => void) => {
				if (settled) return;
				settled = true;
				running--;
				fn();
				stream.end();
			};
			options.signal?.addEventListener("abort", () =>
				settle(() => {
					message.stopReason = "aborted";
					message.errorMessage = "aborted";
					stream.push({ type: "error", reason: "aborted", error: message });
				}),
			);
			if (script === "hang") return stream;
			setTimeout(
				() =>
					settle(() => {
						stream.push({ type: "start", partial: message });
						if ("error" in script) {
							message.stopReason = "error";
							message.errorMessage = script.error;
							if (script.cost === undefined)
								message.usage.cost = { ...message.usage.cost, total: 0, input: 0, output: 0 };
							stream.push({ type: "error", reason: "error", error: message });
							return;
						}
						if (script.text) message.content.push({ type: "text", text: script.text });
						message.content.push({
							type: "toolCall",
							id: `id_${key}`,
							name: script.call ?? "act",
							arguments: { key },
						});
						stream.push({ type: "done", reason: "toolUse", message });
					}),
				typeof script === "object" && "delay" in script ? (script.delay ?? 5) : 5,
			);
			return stream;
		},
	};
	const ctx = {
		hasUI: false,
		cwd: "/",
		ui: { notify: () => {} },
		model: { provider: "ensemble", id: BASE, contextWindow: 128_000, maxTokens: 16_384 } as Record<string, unknown>,
		modelRegistry: registry,
		sessionManager: { getBranch: () => [] },
	};
	const emit = async (name: string, event: Record<string, unknown> = {}) => {
		for (const fn of handlers.get(name) ?? []) await fn({ type: name, ...event }, ctx);
	};
	const turn = async (signal?: AbortSignal) => {
		const provider = providers.find((p) => p.id === "ensemble");
		const context = {
			messages: [{ role: "user", content: [{ type: "text", text: "Solve the task." }], timestamp: 1 }],
		};
		const stream = provider.streamSimple(provider.getModels()[0], context, { signal, reasoning: "low" });
		for await (const _ of stream);
		const message: AssistantMessage = await stream.result();
		// What robot.ts does at message_end.
		for (const fn of handlers.get("message_end") ?? []) await fn({ type: "message_end", message }, ctx);
		return message;
	};
	const turns = () => entries.filter((e) => e.type === ENSEMBLE_ENTRY).map((e) => e.data);
	return { pi, ctx, emit, turn, providers, entries, turns, scripts, calls, tools, peak: () => peak };
}

test("ensembleArgs and candidates: temperatures, models, both", () => {
	assert.deepEqual(ensembleArgs(ARGV), { base: BASE });
	assert.deepEqual(ensembleArgs([`--model=ensemble/${OTHER}`]), { base: OTHER });
	assert.deepEqual(ensembleArgs(["--model", BASE]), { base: undefined });
	assert.deepEqual(candidates(BASE), [
		{ model: BASE, temperature: 0.3 },
		{ model: BASE, temperature: 0.7 },
		{ model: BASE, temperature: 1 },
	]);
	assert.deepEqual(candidates(BASE, `${OTHER}, ${BASE}`), [{ model: OTHER }, { model: BASE }]);
	assert.deepEqual(candidates(BASE, OTHER, "0,0.5"), [
		{ model: OTHER, temperature: 0 },
		{ model: OTHER, temperature: 0.5 },
	]);
	assert.throws(() => candidates(BASE, undefined, "0.2,hot"), /bad --ensemble-temps/);
});

test("off: without --model ensemble/... nothing is registered", () => {
	const f = fakePi();
	assert.equal(ensemble(f.pi, ["node", "pi", "--model", BASE]), undefined);
	assert.deepEqual(f.providers, []);
	assert.equal(f.pi.getFlag("ensemble-temps"), undefined);
});

test("candidates run in parallel at each temperature; the synthesis sees them all and its tool call is the turn", async () => {
	const f = fakePi();
	const en = ensemble(f.pi, ARGV)!;
	await f.emit("session_start");
	f.scripts[`${BASE}@0.3`] = { text: "go left", call: "act" };
	f.scripts[`${BASE}@0.7`] = { text: "go right", call: "act", delay: 20 };
	f.scripts.synth = { text: "left is safer", call: "move_to" };
	const m = await f.turn();
	assert.equal(m.stopReason, "toolUse");
	assert.deepEqual(
		m.content.filter((c) => c.type === "toolCall").map((c) => c.name),
		["move_to"],
	);
	assert.deepEqual(
		f.calls.map((c) => c.key),
		[`${BASE}@0.3`, `${BASE}@0.7`, `${BASE}@1`, "synth"],
	);
	assert.equal(f.peak(), 3, "the three candidates were in flight at once");
	assert.equal(f.calls[0].options.reasoning, "low", "stream options pass through");
	assert.equal(f.calls[3].options.temperature, undefined, "the synthesis runs at the model's own temperature");
	const note = JSON.stringify(f.calls[3].messages.at(-1).content);
	for (const s of ["go left", "go right", `model=\\"${BASE}@0.7\\"`, '- act {\\"key\\"'])
		assert.ok(note.includes(s), `the note shows ${s}`);
	assert.equal(f.calls[3].messages[0].content[0].text, "Solve the task.", "the transcript precedes the note");
	// Cost: every call's, once, on the turn's message; tokens stay the synthesis call's.
	assert.equal(Number(m.usage.cost.total.toFixed(6)), 0.08);
	assert.equal(m.usage.totalTokens, 110);
	const [entry] = f.turns();
	assert.equal(entry.turn, 1);
	assert.equal(entry.cost_usd, 0.08);
	assert.deepEqual(
		entry.candidates.map((c: any) => [c.temperature, c.ok, c.text, c.tool_calls[0].name]),
		[
			[0.3, true, "go left", "act"],
			[0.7, true, "go right", "act"],
			[1, true, "", `act_${BASE}@1`],
		],
	);
	assert.deepEqual(entry.synth.tool_calls, ["move_to"]);
	assert.deepEqual(en.result().ensemble, {
		candidates: [`${BASE}@0.3`, `${BASE}@0.7`, `${BASE}@1`],
		synth: BASE,
		turns: 1,
		candidate_calls: 3,
		candidate_failures: 0,
		synth_failures: 0,
		cost_usd: 0.08,
	});
});

test("--ensemble-models and --ensemble-synth-model; the session model is sized to the smallest delegate", async () => {
	const f = fakePi({ "ensemble-models": `${OTHER},${BASE}`, "ensemble-synth-model": OTHER });
	f.ctx.model.contextWindow = 1_000_000;
	ensemble(f.pi, ARGV);
	await f.emit("session_start");
	assert.equal(f.ctx.model.contextWindow, 65_536);
	const m = await f.turn();
	assert.equal(m.model, "gpt-6-astra", "the reply is the synthesis model's");
	assert.deepEqual(
		f.calls.map((c) => c.key),
		[OTHER, BASE, "synth"],
	);
});

test("a failing candidate is left out of the synthesis; its reported cost still counts", async () => {
	const f = fakePi();
	const en = ensemble(f.pi, ARGV)!;
	await f.emit("session_start");
	f.scripts[`${BASE}@0.7`] = { error: "Request failed (503): down", cost: 0.05 };
	const m = await f.turn();
	assert.equal(m.stopReason, "toolUse");
	const note = JSON.stringify(f.calls.at(-1)?.messages.at(-1).content);
	assert.ok(note.includes("2 candidate next steps"));
	assert.ok(!note.includes("@0.7"));
	assert.equal(Number(m.usage.cost.total.toFixed(6)), 0.11);
	assert.deepEqual(f.turns()[0].candidates[1], {
		model: BASE,
		temperature: 0.7,
		ok: false,
		error: "Request failed (503): down",
		text: "",
		tool_calls: [],
		cost_usd: 0.05,
		tokens: 110,
		ms: f.turns()[0].candidates[1].ms,
	});
	assert.equal(en.result().ensemble.candidate_failures, 1);
});

test("all candidates failing ends the turn with the first error (pi retries as usual) and no synthesis", async () => {
	const f = fakePi({ "ensemble-temps": "0.1,0.9" });
	ensemble(f.pi, ARGV);
	await f.emit("session_start");
	f.scripts[`${BASE}@0.1`] = { error: "Request failed (503): down", cost: 0.01 };
	f.scripts[`${BASE}@0.9`] = { error: "Request failed (502): gateway" };
	const m = await f.turn();
	assert.equal(m.stopReason, "error");
	assert.equal(m.errorMessage, "Request failed (503): down (ensemble: all 2 candidates failed)");
	assert.equal(m.usage.cost.total, 0.01);
	assert.ok(!f.calls.some((c) => c.key === "synth"));
	assert.equal(f.turns()[0].synth, null);
});

test("a failing synthesis ends the turn with its error, carrying the candidates' cost", async () => {
	const f = fakePi();
	const en = ensemble(f.pi, ARGV)!;
	await f.emit("session_start");
	f.scripts.synth = { error: "Request failed (400): bad request" };
	const m = await f.turn();
	assert.equal(m.stopReason, "error");
	assert.equal(m.errorMessage, "Request failed (400): bad request");
	assert.equal(Number(m.usage.cost.total.toFixed(6)), 0.06);
	assert.equal(en.result().ensemble.synth_failures, 1);
});

test("an abort reaches every candidate and ends the turn as aborted", async () => {
	const f = fakePi();
	ensemble(f.pi, ARGV);
	await f.emit("session_start");
	for (const t of [0.3, 0.7, 1]) f.scripts[`${BASE}@${t}`] = "hang";
	const ac = new AbortController();
	setTimeout(() => ac.abort(), 10);
	const m = await f.turn(ac.signal);
	assert.equal(m.stopReason, "aborted");
	assert.ok(f.calls.every((c) => c.options.signal.aborted));
	assert.ok(!f.calls.some((c) => c.key === "synth"));
	assert.equal(f.turns().length, 1);
});

test("--max-api-concurrency: the turn's own slot plus the free ones; n = 1 runs the candidates one after another", async () => {
	for (const [n, want] of [
		[1, 1],
		[2, 2],
	] as const) {
		const f = fakePi();
		ensemble(f.pi, ARGV);
		const dir = mkdtempSync(join(tmpdir(), "ens-gate-"));
		// The gate holds slot-0 for the turn (api-gate's `context` hook).
		writeFileSync(join(dir, "slot-0"), String(process.pid));
		f.pi.events.emit(API_GATE_EVENT, { dir, n });
		await f.emit("session_start");
		const m = await f.turn();
		assert.equal(m.stopReason, "toolUse");
		assert.equal(f.peak(), want, `n=${n}`);
		assert.equal(f.calls.length, 4);
	}
});

test("defineRobot mounts it from argv; --max-cost sees every call and the result row carries ensemble", async (t) => {
	const f = fakePi({ "max-cost": "10" });
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
	assert.equal(f.providers.length, 1);
	await f.emit("session_start");
	await f.emit("agent_start");
	f.scripts[`${BASE}@1`] = { error: "Request failed (503): down", cost: 0.5 };
	await f.turn();
	await f.turn();
	await f.tools.get("finish").execute("1", { status: "success", summary: "ok" });
	await f.emit("agent_end");
	const [result] = f.entries.filter((e) => e.type === RESULT_ENTRY).map((e) => e.data);
	// Two turns of 0.02 + 0.02 + 0.5 (the failed candidate's) + 0.02 (synthesis).
	assert.equal(result.cost_usd, 1.12);
	assert.equal(result.ensemble.cost_usd, 1.12);
	assert.equal(result.ensemble.turns, 2);
	assert.equal(result.ensemble.candidate_failures, 2);
});

test("off: defineRobot without ensemble/... registers no provider and no ensemble field", async (t) => {
	const f = fakePi();
	const argv = process.argv;
	process.argv = ["node", "pi", "--model", BASE];
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
	assert.equal("ensemble" in result, false);
});
