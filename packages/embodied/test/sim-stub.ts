/**
 * Test helpers for the simulated robots: a stub pi whose session starts without a UI, and a fake env
 * server speaking ../src/rpc.ts's wire protocol with a per-method handler.
 */

import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi (flags from `values`, else their defaults; tools; handlers in registration order) and a context without a UI. */
export function stubPi(values: Record<string, unknown> = {}) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	const branch: any[] = [];
	const dir = mkdtempSync(join(tmpdir(), "pi-sim-stub-"));
	let active: string[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		registerProvider: () => {},
		registerMessageRenderer: () => {},
		registerShortcut: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		getThinkingLevel: () => "off",
		appendEntry: (type: string, data: unknown) => entries.push({ type, data }),
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const ctx = {
		hasUI: false,
		cwd: dir,
		ui: { notify: () => {} },
		shutdown: () => {},
		abort: () => {},
		sessionManager: {
			getBranch: () => branch,
			getEntries: () => branch,
			getSessionDir: () => dir,
			getSessionFile: () => undefined,
			getSessionId: () => "sess",
		},
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) result = (await fn({ type: name, ...event }, ctx)) ?? result;
		return result;
	}
	const run = async (name: string, params: unknown) =>
		(await tools.get(name).execute("id", params, undefined, undefined, ctx)) as any;
	/** Record a tool result on the branch, as pi does after execution. */
	const record = (toolName: string, details: Record<string, unknown>, isError = false) =>
		branch.push({ type: "message", message: { role: "toolResult", toolName, toolCallId: "id", details, isError } });
	return { pi, flags, tools, entries, emit, run, record, dir, active: () => active };
}

/**
 * Exploration on a simulated robot: `reset` is refused until the attempt is archived, then restarts the
 * episode through env.reset (`resets` counts them) with a fresh observation; the exploration prompt names
 * the cell; an evaluation prompt carries the memory section instead.
 */
export async function checkSimExplore(o: {
	load: (pi: ExtensionAPI) => unknown;
	values: Record<string, unknown>;
	tag: string;
	resets: () => number;
	observe: string;
}) {
	const s = stubPi({ ...o.values, explore: true, "output-dir": "run", "memory-dir": "memory" });
	o.load(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.ok(s.flags.explore === true && "explore-sessions" in s.flags);
	assert.ok(s.active().includes("finish"), "the robot started");
	await assert.rejects(s.run("reset", { reason: "slipped" }), /Close out attempt 1 first/);
	const before = o.resets();
	const attempts = join(s.dir, "run", "attempts", o.tag);
	mkdirSync(attempts, { recursive: true });
	writeFileSync(join(attempts, "attempt_1_failed.json"), "{}");
	const r = await s.run("reset", { reason: "slipped" });
	assert.equal(o.resets(), before + 1, "reset restarts the episode on the env server");
	assert.equal(r.details.result?.attempt ?? r.details.attempt, 2);
	assert.equal(r.details.terminated ?? r.details.result?.terminated, false);
	s.pi.setActiveTools([o.observe]);
	const explored = (await s.emit("before_agent_start", { systemPrompt: "base" })).systemPrompt as string;
	assert.ok(s.active().includes("reset"));
	assert.match(explored, new RegExp(`MULTI-ATTEMPT EXPLORE mode[\\s\\S]*cell \`${o.tag}\``));
	assert.doesNotMatch(explored, /\{\{\w+\}\}/);

	const e = stubPi({ ...o.values, "memory-dir": "memory" });
	o.load(e.pi);
	mkdirSync(join(e.dir, "memory"));
	await e.emit("session_start");
	process.exitCode = undefined;
	assert.ok(e.active().includes("read"), `memory's read tool is active: ${e.active()}`);
	assert.ok(!e.active().includes("reset"));
	const evaluated = (await e.emit("before_agent_start", { systemPrompt: "base" })).systemPrompt as string;
	assert.match(evaluated, /# Memory\nNotes from earlier explored episodes/);
	assert.doesNotMatch(evaluated, /MULTI-ATTEMPT|\{\{\w+\}\}/);
}

/** A numpy array on the wire. */
export const nd = (dtype: string, shape: number[], data: Buffer) => ({
	__ndarray__: data.toString("base64"),
	dtype,
	shape,
});
export const f32 = (v: number[]) => nd("float32", [v.length], Buffer.from(Float32Array.from(v).buffer));
export const rgb = (h = 2, w = 2) => nd("uint8", [h, w, 3], Buffer.alloc(h * w * 3));

export type Call = { method: string; args: unknown[]; kwargs: Record<string, any> };

/** A fake env server: `answer` returns each method's result (undefined: `{status: "ok"}`); code.api serves an empty registry. */
export async function fakeEnv(answer: (c: Call) => unknown) {
	const calls: Call[] = [];
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, args = [], kwargs = {} } = JSON.parse(body);
			const c = { method, args, kwargs };
			calls.push(c);
			try {
				let result = method === "code.api" ? { tier: null, primitives: [], digest: "d" } : answer(c);
				if (result === undefined) result = { status: "ok" };
				res.end(JSON.stringify({ ok: true, result }));
			} catch (err) {
				res.end(JSON.stringify({ ok: false, error: String(err instanceof Error ? err.message : err) }));
			}
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
	const close = () => {
		server.closeAllConnections();
		server.close();
	};
	return { url, calls, close };
}
