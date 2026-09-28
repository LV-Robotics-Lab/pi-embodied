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

/** A fake env server: `answer` returns each method's result (undefined: `{status: "ok"}`); code.api serves an empty API. */
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
				let result = answer(c);
				// code.api: a manifest digest no pi agrees with (code mode refuses it) and nothing available.
				if (result === undefined && method === "code.api")
					result = { tier: null, manifest_digest: "fake", available: [], digest: "d" };
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

/** Answers for the env server's perception primitives: SAM3 ids and UniDepth (merge into a fake server's answers). */
export function perceptionAnswers(c: Call): unknown {
	if (c.method === "molmo.ground") return { point_xy: [1.2, 0.6], answer: "<point>" };
	if (c.method === "molmo.ground_set")
		return { points: [{ image_index: 1, pixel_x: 0, pixel_y: 1 }], answer: "<points>" };
	if (c.method === "env.detect")
		return {
			found: true,
			observation: 1,
			ids: ["d1"],
			invalidated: [],
			detections: [
				{
					id: "d1",
					score: 0.9,
					box: [0, 0, 1, 1],
					area_px: 4,
					centroid_rc: [1, 1],
					depth_m: 0.5,
					mask_png_base64: "x",
				},
			],
		};
	if (c.method === "env.select_detection")
		return {
			ok: true,
			ids: ["d1"],
			selected: "d1",
			rejected: [],
			detection: { id: "d1", camera: c.kwargs.camera, centroid_rc: [1, 1] },
		};
	if (c.method === "env.reject_detection") return { ok: true, ids: ["d1"], selected: null, rejected: ["d1"] };
	if (c.method === "env.enhance_depth")
		return { ok: true, observation: 1, report: { mode: "mono_only" }, estimate: {} };
	return undefined;
}

/** The env server's meta with its perception capabilities. */
export const withPerception = (meta: Record<string, unknown>) => ({
	...meta,
	capabilities: { ...((meta.capabilities as object) ?? {}), perception: { segment: true, enhance_depth: true } },
});

/**
 * --detections / --unidepth on a simulated robot whose fake env server reports perception: the four tools
 * are active only with the flags, detect reaches env.detect on `camera` and the prompt describes them.
 */
export async function checkDetections(o: {
	load: (pi: ExtensionAPI) => unknown;
	values: Record<string, unknown>;
	calls: Call[];
	camera: string;
}) {
	const off = stubPi(o.values);
	o.load(off.pi);
	await off.emit("session_start");
	process.exitCode = undefined;
	assert.ok(off.active().includes("finish"), "the robot started");
	assert.ok(!off.active().some((t) => ["detect", "select_detection", "enhance_depth"].includes(t)), "off by default");
	const s = stubPi({ ...o.values, detections: true, unidepth: "http://127.0.0.1:1" });
	o.load(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	for (const name of ["detect", "select_detection", "reject_detection", "enhance_depth"])
		assert.ok(s.active().includes(name), name);
	const r = await s.run("detect", { prompt: "cube", camera: o.camera });
	assert.deepEqual(o.calls.filter((c) => c.method === "env.detect").at(-1)?.kwargs, {
		camera: o.camera,
		prompt: "cube",
		min_score: 0.2,
		all: false,
	});
	assert.deepEqual(r.details.ids, ["d1"]);
	assert.equal(r.details.detections[0].mask_png_base64, undefined);
	assert.equal((await s.run("select_detection", { id: "d1" })).details.selected, "d1");
	assert.deepEqual((await s.run("reject_detection", { id: "d1" })).details.rejected, ["d1"]);
	assert.equal((await s.run("enhance_depth", { camera: o.camera })).details.report.mode, "mono_only");
	s.pi.setActiveTools(s.active());
	const prompt = (await s.emit("before_agent_start", { systemPrompt: "" }))?.systemPrompt as string | undefined;
	if (prompt !== undefined) assert.match(prompt, /`detect` gives SAM3 masks[\s\S]*`enhance_depth` fuses/);
	return s;
}

/**
 * --point on a robot whose fake env server also answers Molmo (it is the --molmo server here): point is
 * active only with the flag, one camera asks molmo.ground, several ask molmo.ground_set and each point names its camera.
 */
export async function checkPoint(o: {
	load: (pi: ExtensionAPI) => unknown;
	values: Record<string, unknown>;
	url: string;
	calls: Call[];
	cameras: [string, string];
	started?: (s: ReturnType<typeof stubPi>) => Promise<void> | void;
}) {
	const off = stubPi({ ...o.values, molmo: o.url });
	o.load(off.pi);
	await off.emit("session_start");
	process.exitCode = undefined;
	assert.ok(off.active().includes("finish"), "the robot started");
	assert.ok(!off.active().includes("point"), "off by default");
	const s = stubPi({ ...o.values, molmo: o.url, point: true });
	o.load(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	await o.started?.(s);
	assert.ok(s.active().includes("point"));
	const one = await s.run("point", { query: "the cube", camera: o.cameras[0] });
	assert.equal(one.details.found, true);
	assert.equal(one.details.camera, o.cameras[0]);
	assert.deepEqual(one.details.pixel, [1, 1]);
	assert.equal(o.calls.filter((c) => c.method === "molmo.ground").at(-1)?.kwargs.query, "the cube");
	assert.equal(one.content.filter((c: { type: string }) => c.type === "image").length, 1);
	const set = await s.run("point", { query: "the cube in Image 2", cameras: o.cameras });
	const call = o.calls.filter((c) => c.method === "molmo.ground_set").at(-1);
	assert.equal(call?.kwargs.images_base64.length, 2);
	assert.equal(set.details.points[0].camera, o.cameras[1]);
	assert.deepEqual(set.details.points[0].pixel, [1, 0]);
	s.pi.setActiveTools(s.active());
	const prompt = (await s.emit("before_agent_start", { systemPrompt: "" }))?.systemPrompt as string | undefined;
	if (prompt !== undefined) assert.match(prompt, /`point` \(Molmo\) finds/);
	return { s, one };
}
