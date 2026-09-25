/**
 * Helpers shared by robots that run RPent services: the episode lifecycle, starting or
 * attaching to an RPent RPC server, reading RPent's own Python definitions, numpy payloads,
 * camera frames, rigid transforms, tool results and camera-frame pruning.
 */

import { type ChildProcess, execFile, spawn } from "node:child_process";
import { closeSync, openSync } from "node:fs";
import { createServer } from "node:net";
import { promisify } from "node:util";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { NdArray, RpcClient } from "./rpc.ts";

export type Json = Record<string, any>;
export type Mat = number[][];
export type Rgb = { width: number; height: number; rgb: Buffer };
export type Grid = { height: number; width: number; data: Float32Array };
/** An RPent checkout and the Python that has its dependencies. */
export type Rpent = { root: string; python: string; env?: Record<string, string> };

export const round = (v: number, d = 5) => Number(Number(v).toFixed(d));
export const roundAll = (v: number[], d = 5) => v.map((x) => round(x, d));
export const median = (v: number[]) => {
	const s = [...v].sort((a, b) => a - b);
	return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};
export const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

// ---------------------------------------------------------------------------
// episode lifecycle

/**
 * A robot's episode lifecycle, failing closed. Call it before registering any other handler.
 * - Every session starts with no active tools, and every tool call and prompt is refused until
 *   `start` succeeds. A failed start is shown in the UI, or without one printed to stderr as
 *   `[name] unavailable: ...` with exit code 1.
 * - pi stops early only when every result in a batch terminates: robot tools return
 *   `terminating(result)` so a batch that calls `finish` ends with it, `finish` calls `end()`,
 *   and nothing but `finish` runs after the episode ended.
 * - `result()` becomes the session's single `<name>_result` entry (and, without a UI, a
 *   `[name] {...}` stderr line): at the first agent_end after the episode ended, otherwise at
 *   session_shutdown once an agent has run. A robot that never started writes no result.
 */
export function episode(pi: ExtensionAPI, name: string, result: (ended: "agent_end" | "shutdown") => Json) {
	let ready = false;
	let ran = false;
	let ended = false;
	let reported = false;
	let finishing = false;

	function report(hasUI: boolean, when: "agent_end" | "shutdown") {
		if (!ready || !ran || reported) return;
		reported = true;
		const r = result(when);
		try {
			pi.appendEntry(`${name}_result`, r);
		} catch {}
		if (!hasUI) console.error(`[${name}] ${JSON.stringify(r)}`);
	}

	pi.on("session_start", () => {
		ready = ran = ended = reported = finishing = false;
		pi.setActiveTools([]);
	});
	pi.on("input", (_event, ctx) => {
		if (ready) return undefined;
		if (ctx.hasUI) ctx.ui.notify(`${name} is not available; see the startup error.`, "error");
		return { action: "handled" as const };
	});
	pi.on("agent_start", () => {
		ran = true;
	});
	pi.on("message_end", (event) => {
		const m = event.message;
		if (m.role === "assistant") finishing = m.content.some((c) => c.type === "toolCall" && c.name === "finish");
	});
	pi.on("tool_call", (event) => {
		if (!ready) return { block: true, reason: `${name} is not available.`, terminate: true };
		if (ended && event.toolName !== "finish")
			return { block: true, reason: "The episode is finished.", terminate: true };
		return undefined;
	});
	pi.on("agent_end", (_event, ctx) => {
		if (ended) report(ctx.hasUI, "agent_end");
	});
	pi.on("session_shutdown", (_event, ctx) => report(ctx.hasUI, "shutdown"));

	return {
		ready: () => ready,
		/** Run the robot's startup; on failure the session keeps no tools. */
		async start(ctx: ExtensionContext, run: () => Promise<void>) {
			try {
				await run();
				ready = true;
			} catch (err) {
				pi.setActiveTools([]);
				const text = `unavailable: ${message(err)}`;
				if (ctx.hasUI) ctx.ui.notify(`${name} ${text}`, "error");
				else {
					console.error(`[${name}] ${text}`);
					process.exitCode = 1;
					ctx.shutdown();
				}
			}
		},
		/** The episode is over (`finish`, or a spent budget): only `finish` may still run. */
		end() {
			ended = true;
		},
		/** A tool result that ends its batch when the batch also calls `finish`. */
		terminating: <T extends object>(r: T) => ({ ...r, terminate: finishing }),
	};
}

// ---------------------------------------------------------------------------
// RPent processes

function rpentEnv(r: Rpent): NodeJS.ProcessEnv {
	return { ...process.env, PYTHONPATH: [r.root, process.env.PYTHONPATH].filter(Boolean).join(":"), ...r.env };
}

export function freePort(): Promise<number> {
	return new Promise((resolve, reject) => {
		const srv = createServer();
		srv.listen(0, "127.0.0.1", () => {
			const { port } = srv.address() as { port: number };
			srv.close(() => resolve(port));
		});
		srv.on("error", reject);
	});
}

/** Run `python -c code ...args` in the RPent checkout; the last stdout line is JSON. */
export async function rpentJson<T>(r: Rpent, code: string, args: string[]): Promise<T> {
	const { stdout } = await promisify(execFile)(r.python, ["-c", code, ...args], {
		cwd: r.root,
		env: rpentEnv(r),
		maxBuffer: 64 << 20,
	});
	return JSON.parse(stdout.trim().split("\n").pop() ?? "") as T;
}

/** Start `python -m module` as an RPent HTTP RPC server on a free port and wait for healthz. */
export async function startServer(
	r: Rpent,
	module: string,
	args: string[],
	log: string,
	readyMs = 300_000,
): Promise<{ rpc: RpcClient; proc: ChildProcess }> {
	const port = await freePort();
	const fd = openSync(log, "a");
	const argv = ["-m", module, "--transport", "http", "--host", "127.0.0.1", "--port", String(port)];
	// --parent-watch: the server exits when this stdin pipe closes, i.e. when pi dies.
	const proc = spawn(r.python, [...argv, ...args, "--parent-watch"], {
		cwd: r.root,
		env: rpentEnv(r),
		stdio: ["pipe", fd, fd],
	});
	closeSync(fd);
	const rpc = new RpcClient(`http://127.0.0.1:${port}`);
	const exited = new Promise<never>((_, reject) => {
		proc.once("exit", (code) => reject(new Error(`${module} exited (${code}); see ${log}`)));
		proc.once("error", reject);
	});
	exited.catch(() => {});
	try {
		await Promise.race([rpc.ready(readyMs), exited]);
	} catch (err) {
		proc.kill();
		throw err;
	}
	return { rpc, proc };
}

/** Attach to a running RPent service and wait for healthz. */
export async function attach(endpoint: string, readyMs = 300_000): Promise<RpcClient> {
	const rpc = new RpcClient(endpoint);
	await rpc.ready(readyMs);
	return rpc;
}

/**
 * Refuse a real-robot `move_delta` larger than one call may move, in meters. RPent clips each
 * servo step on the server (0.02 m) but bounds no call's total; its tasks document the per-call
 * limit only in the prompt ("Keep translation commands at or below 0.02 m per call"). The
 * tighter of that and `cap` applies.
 */
export function checkMove(delta: number[], cap: number, constraints: string[] = []) {
	const documented = constraints.map((c) => /translation commands at or below ([\d.]+) m per call/i.exec(c)?.[1]);
	const limit = Math.min(cap, ...documented.filter((v) => v !== undefined).map(Number));
	const norm = Math.hypot(...delta);
	if (!(norm <= limit))
		throw new Error(
			`delta_xyz moves ${round(norm, 4)} m; the limit is ${limit} m per call. Split the motion into smaller calls.`,
		);
}

// ---------------------------------------------------------------------------
// numpy payloads

/** Elements of an array of any numeric or bool dtype, in C order. */
export function numbers(a: NdArray): number[] {
	const b = new Uint8Array(a.data).buffer;
	switch (a.dtype) {
		case "float32":
			return Array.from(new Float32Array(b));
		case "float64":
			return Array.from(new Float64Array(b));
		case "int8":
			return Array.from(new Int8Array(b));
		case "int16":
			return Array.from(new Int16Array(b));
		case "int32":
			return Array.from(new Int32Array(b));
		case "int64":
			return Array.from(new BigInt64Array(b), Number);
		case "uint16":
			return Array.from(new Uint16Array(b));
		case "uint32":
			return Array.from(new Uint32Array(b));
		case "uint64":
			return Array.from(new BigUint64Array(b), Number);
		case "uint8":
		case "bool":
			return Array.from(new Uint8Array(b));
		default:
			throw new Error(`unsupported dtype ${a.dtype}`);
	}
}

/** A list, NdArray or scalar as plain numbers (for poses and state vectors). */
export function vec(v: unknown): number[] {
	if (v instanceof NdArray) return numbers(v);
	if (Array.isArray(v)) return v.map(Number);
	return v === null || v === undefined ? [] : [Number(v)];
}

export const size = (a: NdArray) => a.shape.reduce((x, y) => x * y, 1);
export const f32 = (a: NdArray) => (a.dtype === "float32" ? a : NdArray.f32(numbers(a), a.shape));
export const u8 = (a: NdArray) =>
	a.dtype === "uint8" ? a : new NdArray("uint8", a.shape, Buffer.from(Uint8Array.from(numbers(a))));

/** Element `i` along the first axis. */
export function sub(a: NdArray, i: number): NdArray {
	const n = a.data.length / a.shape[0];
	return new NdArray(a.dtype, a.shape.slice(1), a.data.subarray(i * n, (i + 1) * n));
}

function nest(flat: unknown[], shape: number[]): unknown {
	if (shape.length === 0) return flat[0];
	if (shape.length === 1) return flat;
	const n = flat.length / shape[0];
	return Array.from({ length: shape[0] }, (_, i) => nest(flat.slice(i * n, (i + 1) * n), shape.slice(1)));
}

/** JSON-safe copy: small arrays become nested lists (numpy `tolist`), large ones a shape stub. */
export function plain(v: unknown): unknown {
	if (v instanceof NdArray) {
		if (size(v) > 256) return { ndarray: v.dtype, shape: v.shape };
		const flat = v.dtype === "bool" ? numbers(v).map(Boolean) : numbers(v);
		return nest(flat, v.shape);
	}
	if (Buffer.isBuffer(v)) return `<${v.length} bytes>`;
	if (Array.isArray(v)) return v.map(plain);
	if (v && typeof v === "object") return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, plain(x)]));
	return v;
}

// ---------------------------------------------------------------------------
// camera frames

/** An `[H, W, C>=3]` image as 8-bit RGB. */
export function rgbOf(a: NdArray): Rgb {
	if (a.shape.length !== 3 || a.shape[2] < 3) throw new Error(`expected an [H,W,3] image, got [${a.shape}]`);
	const [height, width, c] = a.shape;
	const src = u8(a).data;
	if (c === 3) return { width, height, rgb: Buffer.from(src) };
	const rgb = Buffer.alloc(width * height * 3);
	for (let i = 0; i < width * height; i++) src.copy(rgb, i * 3, i * c, i * c + 3);
	return { width, height, rgb };
}

/** A depth map with singleton axes squeezed, as float32 meters. */
export function gridOf(a: NdArray): Grid {
	const shape = a.shape.filter((d) => d !== 1);
	if (shape.length !== 2) throw new Error(`expected 2D depth, got [${a.shape}]`);
	return { height: shape[0], width: shape[1], data: Float32Array.from(numbers(a)) };
}

/** Copy of `img` with a crosshair and ring at pixel (row, col). */
export function mark(img: Rgb, row: number, col: number, color: [number, number, number]): Buffer {
	const out = Buffer.from(img.rgb);
	const { width: w, height: h } = img;
	const x = Math.max(0, Math.min(w - 1, col));
	const y = Math.max(0, Math.min(h - 1, row));
	const put = (px: number, py: number) => {
		if (px < 0 || py < 0 || px >= w || py >= h) return;
		out[(py * w + px) * 3] = color[0];
		out[(py * w + px) * 3 + 1] = color[1];
		out[(py * w + px) * 3 + 2] = color[2];
	};
	for (let d = -15; d <= 15; d++)
		for (let t = 0; t < 2; t++) {
			put(x + d, y + t);
			put(x + t, y + d);
		}
	for (let dy = -13; dy <= 13; dy++)
		for (let dx = -13; dx <= 13; dx++) {
			const r = Math.hypot(dx, dy);
			if (r >= 11 && r <= 13) put(x + dx, y + dy);
		}
	return out;
}

// ---------------------------------------------------------------------------
// rigid transforms

/** Rotation matrix of an xyzw quaternion (normalized first). */
export function quatMat(q: number[]): Mat {
	const n = Math.hypot(q[0], q[1], q[2], q[3]);
	if (!(n > 0)) throw new Error("zero-norm quaternion");
	const [x, y, z, w] = q.map((v) => v / n);
	return [
		[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
		[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
		[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
	];
}

/** Homogeneous transform of an `[x, y, z, qx, qy, qz, qw]` pose. */
export function pose7(p: number[]): Mat {
	if (p.length !== 7) throw new Error(`expected tcp_pose shape (7,), got (${p.length},)`);
	const r = quatMat(p.slice(3));
	return [
		[...r[0], p[0]],
		[...r[1], p[1]],
		[...r[2], p[2]],
		[0, 0, 0, 1],
	];
}

/** First three rows of `t @ [p, 1]` (or `t @ p` for a 3x3 `t`). */
export const apply = (t: Mat, p: number[]): number[] =>
	[0, 1, 2].map((i) => t[i][0] * p[0] + t[i][1] * p[1] + t[i][2] * p[2] + (t[i][3] ?? 0));

export function inv3(m: Mat): Mat {
	const [[a, b, c], [d, e, f], [g, h, i]] = m;
	const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
	if (!det) throw new Error("singular intrinsic_K");
	return [
		[(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
		[(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
		[(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
	];
}

// ---------------------------------------------------------------------------
// pi plumbing

/** Tool result: JSON text (capped at 60 kB like RPent) followed by PNG images. */
export function toolResult(out: Record<string, unknown>, pngs: Buffer[] = []) {
	const details = plain(out) as Json;
	let text = JSON.stringify(details, null, 2) ?? "null";
	if (Buffer.byteLength(text) > 60_000) text = `${Buffer.from(text).subarray(0, 60_000).toString()}\n[truncated]`;
	return {
		content: [
			{ type: "text" as const, text },
			...pngs.map((png) => ({ type: "image" as const, data: png.toString("base64"), mimeType: "image/png" })),
		],
		details,
	};
}

/** Replace all but the newest `keep()` camera frames in tool results with a text stub. */
export function pruneImages(pi: ExtensionAPI, keep: () => number) {
	pi.on("context", (event) => {
		let left = keep();
		let pruned = false;
		const messages = [...event.messages].reverse().map((m) => {
			if (m.role !== "toolResult") return m;
			const content = m.content.map((part) => {
				if (part.type !== "image" || left-- > 0) return part;
				pruned = true;
				return { type: "text" as const, text: "[older camera frame omitted]" };
			});
			return { ...m, content };
		});
		return pruned ? { messages: messages.reverse() } : undefined;
	});
}
