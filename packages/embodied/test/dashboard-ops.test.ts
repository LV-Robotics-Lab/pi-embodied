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
import dashboard from "../src/dashboard/index.ts";
import { defineRobot } from "../src/robot.ts";
import { NOTE_EVENT, VIDEO_DIR_EVENT, type VideoNote } from "../src/video.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi with a toy robot (one `move` tool) and the dashboard, run like pi's runner. */
function rig(o: { idle?: boolean } = {}) {
	const handlers = new Map<string, Handler[]>();
	const listeners = new Map<string, ((data: unknown) => void)[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const notes: string[] = [];
	const sent: { text: unknown; options: unknown }[] = [];
	/** pi's steering queue, as the stub's withdrawQueuedMessage sees it. */
	const queue: string[] = [];
	const moves: number[] = [];
	const exports: string[] = [];
	let active: string[] = [];
	let idle = o.idle ?? true;
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, f: { default?: unknown }) => {
			flags[name] = f.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: () => {},
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
		sessionManager: { getBranch: () => [], getSessionId: () => "sess-1", getSessionFile: () => undefined },
		isIdle: () => idle,
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
