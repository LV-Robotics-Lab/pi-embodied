import assert from "node:assert/strict";
import { request } from "node:http";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import dashboard from "../src/dashboard/index.ts";
import { NdArray } from "../src/rpc.ts";
import { FRAME_EVENT } from "../src/video.ts";

type Handler = (event: any, ctx: any) => unknown;

/** One pi runtime with the dashboard on a free port; `start()` returns its URL. */
function fakePi(flagValues: Record<string, unknown> = {}) {
	const handlers = new Map<string, Handler[]>();
	const listeners = new Map<string, ((data: unknown) => void)[]>();
	const flags: Record<string, unknown> = {};
	const notes: string[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		getThinkingLevel: () => "low",
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
	const ctx = {
		hasUI: true,
		ui: { notify: (m: string) => notes.push(m) },
		sessionManager: { getBranch: () => [] },
		isIdle: () => true,
	};
	const emit = async (name: string, event: Record<string, unknown> = {}) => {
		for (const fn of handlers.get(name) ?? []) await fn(event, ctx);
	};
	dashboard(pi);
	flags.dashboard = true;
	return {
		pi,
		emit,
		async start() {
			await emit("session_start");
			return (notes.find((n) => n.startsWith("Dashboard: ")) as string).slice(11);
		},
		quit: () => emit("session_shutdown", { reason: "quit" }),
	};
}

/** A streaming response (SSE, multipart) read until `until` holds, then dropped. */
function head(url: string, until: (body: Buffer) => boolean): Promise<{ type: string; body: Buffer }> {
	return new Promise((resolve, reject) => {
		const req = request(url, (res) => {
			let body = Buffer.alloc(0);
			res.on("data", (c: Buffer) => {
				body = Buffer.concat([body, c]);
				if (!until(body)) return;
				req.destroy();
				resolve({ type: String(res.headers["content-type"]), body });
			});
		});
		req.on("error", reject);
		req.end();
	});
}

const snapshotGen = async (url: string) => {
	const { body } = await head(`${url}events`, (b) => b.includes("\n\n"));
	return JSON.parse(body.toString().split("\n\n")[0].slice(6)).episode.gen as number;
};

test("frame URLs differ between two pi processes on the same port (the browser caches them)", async () => {
	const a = fakePi();
	const urlA = await a.start();
	const genA = await snapshotGen(urlA);
	await a.quit();
	await new Promise((r) => setTimeout(r, 5));
	const port = new URL(urlA).port;
	const b = fakePi({ "dashboard-port": port });
	const urlB = await b.start();
	assert.equal(urlB, urlA);
	const genB = await snapshotGen(urlB);
	await b.quit();
	assert.notEqual(genA, genB);
	assert.ok(genA > 1_000_000 && genB > genA);
});

test("GET /live streams the latest env frame as multipart PNG, downscaled to ?w=", async () => {
	const p = fakePi({ "dashboard-live-fps": "20" });
	const url = await p.start();
	try {
		const [h, w] = [60, 80];
		const rgb = Buffer.alloc(h * w * 3);
		for (let i = 0; i < h * w; i++) rgb[i * 3] = 255;
		p.pi.events.emit(FRAME_EVENT, new NdArray("uint8", [h, w, 3], rgb));
		const { type, body } = await head(`${url}live?w=64`, (b) => b.includes("IEND"));
		assert.equal(type, "multipart/x-mixed-replace; boundary=frame");
		const text = body.toString("latin1");
		assert.match(text, /^--frame\r\nContent-Type: image\/png\r\nContent-Length: \d+\r\n\r\n\x89PNG/);
		const png = body.subarray(body.indexOf(Buffer.from([0x89, 0x50, 0x4e, 0x47])));
		// 80 px wide at w=64: a 2x box filter.
		assert.deepEqual([png.readUInt32BE(16), png.readUInt32BE(20)], [40, 30]);
	} finally {
		await p.quit();
	}
});
