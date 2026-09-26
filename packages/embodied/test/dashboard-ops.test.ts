/**
 * The dashboard's operator tools against a fake robot (a real defineRobot on a stub pi): withdrawing a
 * queued message, manual robot-tool calls through the robot's gates, the LLM check, and downloads.
 */
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { request } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import dashboard, { checkAccess, dashboardAccess, MANUAL_ENTRY } from "../src/dashboard/index.ts";
import { defineRobot } from "../src/robot.ts";
import { NOTE_EVENT, VIDEO_DIR_EVENT, type VideoNote } from "../src/video.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi with a toy robot (one `move` tool) and the dashboard, run like pi's runner. */
function rig(o: { idle?: boolean; flags?: Record<string, unknown> } = {}) {
	const handlers = new Map<string, Handler[]>();
	const listeners = new Map<string, ((data: unknown) => void)[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const notes: string[] = [];
	const sent: { text: unknown; options: unknown }[] = [];
	/** pi's steering queue, as the stub's withdrawQueuedMessage sees it. */
	const queue: string[] = [];
	const moves: number[] = [];
	/** Each move's start and end, in order ("start 0.02", "end 0.02"). */
	const trace: string[] = [];
	/** Set to hold every move until `release()`. */
	let hold: Promise<void> | undefined;
	let release = () => {};
	const branch: any[] = [];
	const exports: string[] = [];
	let active: string[] = [];
	let idle = o.idle ?? true;
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, f: { default?: unknown }) => {
			flags[name] = o.flags && name in o.flags ? o.flags[name] : f.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: (customType: string, data: unknown) => branch.push({ type: "custom", customType, data }),
		sendMessage: (
			m: { customType: string; content: unknown; display: boolean; details: unknown },
			options: unknown,
		) => branch.push({ type: "custom_message", ...m, options }),
		getThinkingLevel: () => "off",
		sendUserMessage: (text: unknown, options: any) => {
			sent.push({ text, options });
			if (options?.deliverAs === "steer") queue.push(String(text));
		},
		events: {
			emit: (channel: string, data: unknown) => {
				for (const fn of listeners.get(channel) ?? []) fn(data);
			},
			on: (channel: string, fn: (data: unknown) => void) => {
				listeners.set(channel, [...(listeners.get(channel) ?? []), fn]);
				return () => {};
			},
		},
	} as unknown as ExtensionAPI;
	const model = { provider: "faux", id: "planner", reasoning: false };
	let reply: { stopReason: string; content: unknown[]; errorMessage?: string } = {
		stopReason: "length",
		content: [{ type: "text", text: "OK" }],
	};
	const requests: any[] = [];
	const ctx = {
		hasUI: true,
		ui: { notify: (m: string) => notes.push(m) },
		sessionManager: { getBranch: () => branch, getSessionId: () => "sess-1", getSessionFile: () => undefined },
		isIdle: () => idle,
		hasPendingMessages: () => queue.length > 0,
		signal: undefined,
		abort: () => {},
		shutdown: () => {},
		model,
		modelRegistry: {
			streamSimple: (m: unknown, context: unknown, options: unknown) => {
				requests.push({ m, context, options });
				return { result: async () => reply };
			},
		},
		withdrawQueuedMessage: (text: string) => {
			const i = queue.indexOf(text);
			if (i < 0) return false;
			queue.splice(i, 1);
			return true;
		},
		exportSession: async (format: string, path: string) => {
			exports.push(format);
			writeFileSync(path, format === "html" ? "<html>session</html>" : '{"type":"session"}\n');
			return path;
		},
	};
	const robot = defineRobot(pi, {
		name: "toy",
		task: [],
		keepImages: 1,
		budget: { turns: 0, seconds: 0 },
		start: async () => ["move", "finish"],
		result: () => ({}),
		finish: {
			description: "finish",
			parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
			result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
		},
	});
	robot.tool(
		"move",
		"Move by dx metres",
		Type.Object({ dx: Type.Number({ minimum: -0.05, maximum: 0.05, description: "metres" }) }),
		async ({ dx }) => {
			trace.push(`start ${dx}`);
			await hold;
			trace.push(`end ${dx}`);
			moves.push(dx);
			return { content: [{ type: "text", text: JSON.stringify({ moved: dx }) }], details: {} };
		},
	);
	dashboard(pi);
	flags.dashboard = true;
	const emit = async (name: string, event: Record<string, unknown> = {}) => {
		for (const fn of handlers.get(name) ?? []) await fn({ type: name, ...event }, ctx);
	};
	return {
		pi,
		emit,
		tools,
		trace,
		branch,
		ctx,
		hold: () => {
			hold = new Promise<void>((r) => {
				release = r;
			});
		},
		release: () => {
			hold = undefined;
			release();
		},
		sent,
		queue,
		moves,
		exports,
		requests,
		setIdle: (v: boolean) => {
			idle = v;
		},
		setReply: (r: typeof reply) => {
			reply = r;
		},
		async start() {
			await emit("session_start");
			return (notes.find((n) => n.startsWith("Dashboard: ")) as string).slice(11);
		},
		quit: () => emit("session_shutdown", { reason: "quit" }),
	};
}

function call(url: string, method = "GET", body?: unknown) {
	return new Promise<{ status: number; type: string; headers: Record<string, unknown>; text: string; json: any }>(
		(resolve, reject) => {
			const req = request(url, { method, headers: { "Content-Type": "application/json" } }, (res) => {
				let raw = "";
				res.on("data", (c) => {
					raw += c;
				});
				res.on("end", () => {
					let json: any;
					try {
						json = JSON.parse(raw);
					} catch {}
					resolve({
						status: res.statusCode ?? 0,
						type: String(res.headers["content-type"]),
						headers: res.headers,
						text: raw,
						json,
					});
				});
			});
			req.on("error", reject);
			req.end(body === undefined ? undefined : JSON.stringify(body));
		},
	);
}
const post = (url: string, body: unknown) => call(url, "POST", body);

/** The dashboard's current snapshot (the first SSE event). */
function snapshot(url: string): Promise<any> {
	return new Promise((resolve, reject) => {
		const req = request(`${url}events`, (res) => {
			let raw = "";
			res.on("data", (c) => {
				raw += c;
				if (!raw.includes("\n\n")) return;
				req.destroy();
				resolve(JSON.parse(raw.split("\n\n")[0].slice(6)));
			});
		});
		req.on("error", reject);
		req.end();
	});
}

test("a message sent while the agent runs goes to pi's queue and can be withdrawn until the agent takes it", async () => {
	const r = rig({ idle: false });
	const url = await r.start();
	try {
		const a = await post(`${url}message`, { text: "go left instead" });
		assert.equal(a.status, 202);
		assert.equal(a.json.steered, true);
		assert.deepEqual(r.sent.at(-1), { text: "go left instead", options: { deliverAs: "steer" } });
		assert.deepEqual(r.queue, ["go left instead"]);
		assert.equal((await snapshot(url)).queued[0].status, "queued");

		const w = await post(`${url}message/withdraw`, { id: a.json.id });
		assert.equal(w.status, 200);
		assert.equal(w.json.text, "go left instead");
		assert.deepEqual(r.queue, []);
		assert.equal((await post(`${url}message/withdraw`, { id: a.json.id })).status, 409);
		assert.equal((await snapshot(url)).queued[0].status, "withdrawn");

		// Once the agent takes it (its user message starts), it is delivered and stays sent.
		const b = await post(`${url}message`, { text: "stop" });
		r.queue.splice(0);
		await r.emit("message_start", { message: { role: "user", content: [{ type: "text", text: "stop" }] } });
		const late = await post(`${url}message/withdraw`, { id: b.json.id });
		assert.equal(late.status, 409);
		assert.match(late.json.error, /already delivered/);
		assert.equal((await post(`${url}message/withdraw`, { id: 99 })).status, 404);

		// After an Interrupt pi holds nothing any more: at the run's end the list says so.
		const d = await post(`${url}message`, { text: "after the stop" });
		await r.emit("agent_end");
		assert.equal((await snapshot(url)).queued.at(-1).status, "queued");
		r.queue.splice(0);
		await r.emit("agent_end");
		assert.equal((await snapshot(url)).queued.at(-1).status, "dropped");
		assert.match((await post(`${url}message/withdraw`, { id: d.json.id })).json.error, /already dropped/);

		// Idle: a plain prompt, nothing to withdraw.
		r.setIdle(true);
		const c = await post(`${url}message`, { text: "hello" });
		assert.equal(c.json.steered, false);
		assert.equal(c.json.id, undefined);
		assert.deepEqual(r.sent.at(-1), { text: "hello", options: undefined });
	} finally {
		await r.quit();
	}
});

test("a manual robot-tool call runs through the robot's own path and gates, in operator mode only", async () => {
	const r = rig();
	const url = await r.start();
	const labels: VideoNote[] = [];
	r.pi.events.on(NOTE_EVENT, (n) => labels.push(n as VideoNote));
	try {
		const list = await call(`${url}primitives`);
		assert.equal(list.json.available, true);
		assert.deepEqual(
			list.json.tools.map((t: any) => t.name),
			["move"],
		);
		assert.equal(list.json.tools[0].parameters.properties.dx.maximum, 0.05);

		const ok = await post(`${url}primitive`, { name: "move", arguments: { dx: 0.02 } });
		assert.equal(ok.status, 200, ok.text);
		assert.deepEqual(r.moves, [0.02]);
		assert.deepEqual(labels, [{ actor: "human", action: "move" }, { actor: null }]);
		const snap = await snapshot(url);
		assert.equal(snap.steps.at(-1).name, "operator:move");
		assert.deepEqual(snap.steps.at(-1).args, { dx: 0.02 });

		// The schema is the tool's: out of range and a wrong type are refused before anything moves.
		const far = await post(`${url}primitive`, { name: "move", arguments: { dx: 0.5 } });
		assert.equal(far.status, 422);
		assert.match(far.json.error, /dx/);
		assert.equal((await post(`${url}primitive`, { name: "move", arguments: [] })).status, 422);
		assert.equal((await post(`${url}primitive`, { name: "finish", arguments: {} })).status, 404);
		assert.equal((await post(`${url}primitive`, { name: "nope", arguments: {} })).status, 404);
		assert.deepEqual(r.moves, [0.02]);

		// The agent running: not operator mode.
		r.setIdle(false);
		const busy = await post(`${url}primitive`, { name: "move", arguments: { dx: 0.01 } });
		assert.equal(busy.status, 409);
		assert.match(busy.json.error, /agent is running/);
		assert.equal((await call(`${url}primitives`)).json.available, false);
		r.setIdle(true);

		// The robot's gates: after `finish` the episode is over and the call is refused.
		await r.tools.get("finish").execute("f", { status: "success", summary: "" });
		const over = await post(`${url}primitive`, { name: "move", arguments: { dx: 0.01 } });
		assert.equal(over.status, 422);
		assert.match(over.json.error, /episode is finished/);
		assert.deepEqual(r.moves, [0.02]);
	} finally {
		await r.quit();
	}
});

test("the LLM check sends the session's model one real 1-token completion through pi's model registry", async () => {
	const r = rig();
	const url = await r.start();
	try {
		const ok = await post(`${url}llm-check`, {});
		assert.equal(ok.status, 200);
		assert.equal(ok.json.ok, true);
		assert.equal(ok.json.model, "faux/planner");
		assert.match(ok.json.detail, /answered in \d+ ms \("OK"\)/);
		assert.equal(r.requests[0].options.maxTokens, 1);
		assert.equal(r.requests[0].context.messages[0].role, "user");

		r.setReply({ stopReason: "error", content: [], errorMessage: "401 invalid key" });
		const bad = await post(`${url}llm-check`, {});
		assert.equal(bad.json.ok, false);
		assert.equal(bad.json.detail, "401 invalid key");
	} finally {
		await r.quit();
	}
});

test("downloads: the session is pi's /export (JSONL or HTML); the episode videos are listed and served", async () => {
	const r = rig();
	const url = await r.start();
	const dir = mkdtempSync(join(tmpdir(), "dash-video-"));
	mkdirSync(join(dir, "sub"));
	writeFileSync(join(dir, "episode.mp4"), "mp4-bytes");
	writeFileSync(join(dir, "notes.txt"), "x");
	r.pi.events.emit(VIDEO_DIR_EVENT, dir);
	try {
		const jsonl = await call(`${url}download/session`);
		assert.equal(jsonl.status, 200);
		assert.equal(jsonl.text, '{"type":"session"}\n');
		assert.match(String(jsonl.headers["content-disposition"]), /session-sess-1\.jsonl/);
		const html = await call(`${url}download/session?format=html`);
		assert.equal(html.text, "<html>session</html>");
		assert.deepEqual(r.exports, ["jsonl", "html"]);

		const list = await call(`${url}downloads`);
		assert.deepEqual(list.json.sessions, [{ id: "sess-1", current: true, videos: ["episode.mp4"] }]);
		const video = await call(`${url}download/video/sess-1/episode.mp4`);
		assert.equal(video.status, 200);
		assert.equal(video.type, "video/mp4");
		assert.equal(video.text, "mp4-bytes");
		assert.equal((await call(`${url}download/video/sess-1/notes.txt`)).status, 404);
		assert.equal((await call(`${url}download/video/sess-1/..%2Fepisode.mp4`)).status, 404);
		assert.equal((await call(`${url}download/video/other/episode.mp4`)).status, 404);
	} finally {
		await r.quit();
	}
});

test("a manual call is a session entry on the timeline, and never overlaps an agent robot call", async () => {
	const r = rig();
	const url = await r.start();
	try {
		const ok = await post(`${url}primitive`, { name: "move", arguments: { dx: 0.01 } });
		assert.equal(ok.status, 200, ok.text);
		const entry = r.branch.find((e) => e.customType === MANUAL_ENTRY);
		// A session message the agent reads (appended at once while it is idle).
		assert.equal(entry.type, "custom_message");
		assert.equal(entry.options, undefined);
		assert.match(entry.content, /^The operator called the robot tool move \{"dx":0.01\} by hand \(ok\)/);
		assert.deepEqual(
			{ ...entry.details, ms: 0, timestamp: 0 },
			{
				source: "operator",
				tool: "move",
				args: { dx: 0.01 },
				ok: true,
				result: '{"moved":0.01}',
				ms: 0,
				timestamp: 0,
			},
		);
		// A reload / resume rebuilds the timeline from the session, the manual call included.
		await r.emit("session_start");
		const rebuilt = (await snapshot(url)).steps.filter((s: any) => s.name === "operator:move");
		assert.equal(rebuilt.length, 1);
		assert.deepEqual(rebuilt[0].args, { dx: 0.01 });

		// The operator's call holds the robot: an agent call that arrives meanwhile waits for it.
		r.trace.length = 0;
		r.hold();
		const manual = post(`${url}primitive`, { name: "move", arguments: { dx: 0.02 } });
		while (!r.trace.length) await new Promise((res) => setTimeout(res, 1));
		// Nothing may start the agent meanwhile.
		const refused = await post(`${url}message`, { text: "go on" });
		assert.equal(refused.status, 409);
		assert.match(refused.json.error, /manual robot call is running/);
		const agent = r.tools.get("move").execute("call-1", { dx: 0.03 }, undefined, undefined, r.ctx);
		await new Promise((res) => setTimeout(res, 20));
		assert.deepEqual(r.trace, ["start 0.02"]);
		r.release();
		assert.equal((await manual).status, 200);
		await agent;
		assert.deepEqual(r.trace, ["start 0.02", "end 0.02", "start 0.03", "end 0.03"]);

		// And the other way round: a manual call waits for the agent's; an abort gives up the wait.
		r.trace.length = 0;
		r.hold();
		const first = r.tools.get("move").execute("call-2", { dx: 0.04 }, undefined, undefined, r.ctx);
		while (!r.trace.length) await new Promise((res) => setTimeout(res, 1));
		const late = post(`${url}primitive`, { name: "move", arguments: { dx: 0.05 } });
		const stop = new AbortController();
		const waiting = r.tools.get("move").execute("call-3", { dx: 0.01 }, stop.signal, undefined, r.ctx);
		stop.abort();
		await assert.rejects(waiting, /aborted while another robot call ran/);
		await new Promise((res) => setTimeout(res, 20));
		assert.deepEqual(r.trace, ["start 0.04"]);
		r.release();
		await first;
		assert.equal((await late).status, 200);
		assert.deepEqual(r.trace, ["start 0.04", "end 0.04", "start 0.05", "end 0.05"]);
	} finally {
		await r.quit();
	}
});

test("the dashboard checks the Host header, and off loopback every request needs its token", async () => {
	const local = dashboardAccess("127.0.0.1", "");
	const at = (host: string, extra: Record<string, string> = {}, path = "/") =>
		checkAccess(local, { host, ...extra }, new URL(path, "http://dashboard"));
	assert.equal(local.token, undefined);
	assert.deepEqual(at("127.0.0.1:8770"), {});
	assert.deepEqual(at("localhost:18770"), {});
	assert.deepEqual(at("[::1]:8770"), {});
	// A page on another domain rebinding its DNS to 127.0.0.1 still sends its own name.
	assert.equal(at("evil.example:8770").refused?.[0], 421);
	assert.equal(at("10.0.0.5:8770").refused?.[0], 421);

	const lan = dashboardAccess("0.0.0.0", "", ["rig.tailnet.ts.net"]);
	assert.match(String(lan.token), /^[0-9a-f]{32}$/);
	const t = lan.token as string;
	const req = (host: string, headers: Record<string, string> = {}, path = "/") =>
		checkAccess(lan, { host, ...headers }, new URL(path, "http://dashboard"));
	assert.equal(req("evil.example").refused?.[0], 421);
	assert.equal(req("10.0.0.5:8770").refused?.[0], 401);
	assert.equal(req("rig.tailnet.ts.net:8770").refused?.[0], 401);
	assert.match(
		String(req("10.0.0.5:8770", {}, `/?token=${t}`).setCookie),
		new RegExp(`^pi_dashboard_token=${t}; HttpOnly; SameSite=Strict`),
	);
	assert.deepEqual(req("10.0.0.5", { cookie: `a=1; pi_dashboard_token=${t}` }), {});
	assert.deepEqual(req("10.0.0.5", { authorization: `Bearer ${t}` }), {});
	assert.equal(req("10.0.0.5", { cookie: "pi_dashboard_token=wrong" }).refused?.[0], 401);
	assert.equal(req("10.0.0.5", {}, "/?token=wrong").refused?.[0], 401);

	// End to end, bound off loopback with a given token: the printed URL carries it.
	const r = rig({ flags: { "dashboard-host": "0.0.0.0", "dashboard-token": "t0k3n" } });
	const url = await r.start();
	try {
		assert.match(url, /\/\?token=t0k3n$/);
		const base = `http://127.0.0.1:${new URL(url).port}/`;
		assert.equal((await call(`${base}primitives`)).status, 401);
		const first = await call(`${base}primitives?token=t0k3n`);
		assert.equal(first.status, 200);
		assert.match(String(first.headers["set-cookie"]), /pi_dashboard_token=t0k3n/);
		assert.equal((await post(`${base}message`, { text: "x" })).status, 401);
	} finally {
		await r.quit();
	}
});
