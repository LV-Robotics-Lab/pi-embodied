/**
 * Robot service RPC: `POST <endpoint>/call` with `{method, args, kwargs}`, answered by
 * `{ok, result}` or `{ok: false, error, traceback}`. numpy arrays travel as
 * `{__ndarray__: <base64 C-order bytes>, dtype, shape}`, scalars as `{__npscalar__}`.
 */

import { Agent as HttpAgent, request as httpRequest } from "node:http";
import { Agent as HttpsAgent, request as httpsRequest } from "node:https";

export class NdArray {
	dtype: string;
	shape: number[];
	data: Buffer;

	constructor(dtype: string, shape: number[], data: Buffer) {
		this.dtype = dtype.replace(/^[<>|=]/, "");
		this.shape = shape;
		this.data = data;
	}

	static f32(values: number[], shape = [values.length]): NdArray {
		return new NdArray("float32", shape, Buffer.from(Float32Array.from(values).buffer));
	}

	/** Same bytes with a leading batch dimension. */
	batched(): NdArray {
		return new NdArray(this.dtype, [1, ...this.shape], this.data);
	}

	toArray(): number[] {
		const { buffer, byteOffset, byteLength } = this.data;
		const bytes = buffer.slice(byteOffset, byteOffset + byteLength);
		switch (this.dtype) {
			case "float32":
				return Array.from(new Float32Array(bytes));
			case "float64":
				return Array.from(new Float64Array(bytes));
			case "int32":
				return Array.from(new Int32Array(bytes));
			case "int64":
				return Array.from(new BigInt64Array(bytes), Number);
			default:
				return Array.from(new Uint8Array(bytes));
		}
	}
}

function encode(value: unknown): unknown {
	if (value instanceof NdArray)
		return { __ndarray__: value.data.toString("base64"), dtype: value.dtype, shape: value.shape };
	if (Array.isArray(value)) return value.map(encode);
	if (value && typeof value === "object")
		return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, encode(v)]));
	return value;
}

function decode(value: unknown): unknown {
	if (Array.isArray(value)) return value.map(decode);
	if (value && typeof value === "object") {
		const v = value as Record<string, unknown>;
		if (typeof v.__ndarray__ === "string") {
			return new NdArray(String(v.dtype), v.shape as number[], Buffer.from(v.__ndarray__, "base64"));
		}
		if ("__npscalar__" in v) return v.__npscalar__;
		return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, decode(x)]));
	}
	return value;
}

const NONFINITE: Record<string, number> = { NaN: Number.NaN, Infinity: Infinity, "-Infinity": -Infinity };
const NONFINITE_TAG = "\u0000pi-nonfinite:";

/**
 * JSON.parse that also accepts the `NaN`, `Infinity` and `-Infinity` tokens Python's `json.dumps`
 * emits for non-finite floats (standard JSON has none, so JSON.parse rejects the whole reply).
 * Tokens inside strings are left alone.
 */
export function parseJson(text: string): unknown {
	try {
		return JSON.parse(text);
	} catch (err) {
		if (!/NaN|Infinity/.test(text)) throw err;
	}
	const tagged = text.replace(/"(?:[^"\\]|\\.)*"|-?Infinity|NaN/g, (m) =>
		m.startsWith('"') ? m : JSON.stringify(NONFINITE_TAG + m),
	);
	return JSON.parse(tagged, (_k, v) =>
		typeof v === "string" && v.startsWith(NONFINITE_TAG) ? NONFINITE[v.slice(NONFINITE_TAG.length)] : v,
	);
}

/**
 * Calls to one endpoint run one at a time: a robot env server may accept concurrent calls
 * (read-only calls may run in parallel) while its worker pipe does not. A call is
 * sent only once the server has answered the previous one. An abort or timeout releases the
 * caller, never the endpoint: the server is still executing that call, so later calls keep
 * waiting for its answer, each within its own timeout (rather than failing at once, since the
 * answer usually comes moments later). An abort also asks the server to `stop` that call.
 */
const busy = new Map<string, { method: string; answered: Promise<unknown> }>();

/**
 * Robot services are local or on the tailnet: never go through HTTP_PROXY. fetch and, with
 * NODE_USE_ENV_PROXY, Node's global agents would; explicit agents do not.
 */
const direct = { http: new HttpAgent({ keepAlive: true }), https: new HttpsAgent({ keepAlive: true }) };

/** The service is gone: unreachable, dropped the connection, or stopped answering (see `call`). */
export class RpcUnavailable extends Error {}

/** Endpoints whose server stopped answering, with why; every later call to them fails at once. */
const dead = new Map<string, string>();

/** Give every endpoint marked unresponsive another chance (a new episode: services may have been restarted). */
export function forgetUnresponsive() {
	dead.clear();
}

/** One request and its full answer. `signal` only abandons a server that stopped answering. */
function post(url: string, body: string, signal?: AbortSignal): Promise<string> {
	return new Promise((resolve, reject) => {
		const target = new URL(url);
		const https = target.protocol === "https:";
		const send = https ? httpsRequest : httpRequest;
		const req = send(
			target,
			{
				method: "POST",
				agent: https ? direct.https : direct.http,
				signal,
				headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) },
			},
			(res) => {
				const chunks: Buffer[] = [];
				res.on("data", (c: Buffer) => chunks.push(c));
				res.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
				res.on("error", (err) => reject(new RpcUnavailable(`${url}: ${err.message}`)));
			},
		);
		req.on("error", (err) => reject(new RpcUnavailable(`${url}: ${err.message}`)));
		req.end(body);
	});
}

export class RpcClient {
	url: string;
	/** Server-side session id, for servers that scope state per client (RoboCasa's VLA). */
	session: string | null = null;

	/** How long a sent call that timed out or was aborted may stay unanswered before the endpoint is given up on. */
	graceMs: number | undefined;

	constructor(endpoint: string, o: { graceMs?: number } = {}) {
		this.graceMs = o.graceMs;
		const base = endpoint.includes("://") ? endpoint : `http://${endpoint}`;
		this.url = `${base.replace(/\/$/, "")}/call`;
	}

	/**
	 * Call `method`. `timeoutMs` (counted from this call, queueing included) and `signal` end the
	 * wait; a call that has not been sent by then is never sent, and one already sent keeps the
	 * endpoint until the server answers it. A server that still has not answered `graceMs` later
	 * (default: `timeoutMs`, at least 60 s) is given up on: the endpoint is marked unresponsive
	 * (until `forgetUnresponsive`) and every later call fails with `RpcUnavailable`, as do transport
	 * errors. A timeout itself is an ordinary error.
	 */
	async call<T = unknown>(
		method: string,
		kwargs: Record<string, unknown> = {},
		timeoutMs = 120_000,
		args: unknown[] = [],
		signal?: AbortSignal,
	): Promise<T> {
		if (signal?.aborted) throw new Error(`${method}: aborted`);
		const down = dead.get(this.url);
		if (down) throw new RpcUnavailable(`${method}: ${this.url} stopped answering (${down})`);
		const body = JSON.stringify({ method, args: encode(args), kwargs: encode(kwargs), session_id: this.session });
		const previous = busy.get(this.url);
		let sent = false;
		let gaveUp = false;
		const abandon = new AbortController();
		let grace: ReturnType<typeof setTimeout> | undefined;
		const answered = (async () => {
			await previous?.answered.catch(() => {});
			if (gaveUp) return undefined;
			const down = dead.get(this.url);
			if (down) throw new RpcUnavailable(`${method}: ${this.url} stopped answering (${down})`);
			sent = true;
			return post(this.url, body, abandon.signal);
		})();
		const slot = { method, answered };
		busy.set(this.url, slot);
		answered
			.catch(() => {})
			.finally(() => {
				clearTimeout(grace);
				if (busy.get(this.url) === slot) busy.delete(this.url);
			});

		let timer: ReturnType<typeof setTimeout> | undefined;
		let onAbort: (() => void) | undefined;
		const released = new Promise<never>((_, reject) => {
			const release = (why: string) => {
				gaveUp = true;
				if (sent) {
					const graceMs = this.graceMs ?? Math.max(timeoutMs, 60_000);
					grace = setTimeout(() => {
						dead.set(this.url, `${method} unanswered ${(timeoutMs + graceMs) / 1000} s after it was sent`);
						abandon.abort();
					}, graceMs);
					grace.unref();
				}
				reject(
					new Error(
						sent
							? `${method}: ${why}; the server is still running it`
							: `${method}: ${why} waiting for ${previous?.method ?? "the server"}, which the server is still running`,
					),
				);
			};
			timer = setTimeout(() => release(`timed out after ${timeoutMs} ms`), timeoutMs);
			onAbort = () => {
				if (sent) void this.interrupt();
				release("aborted");
			};
			signal?.addEventListener("abort", onAbort, { once: true });
		});
		let text: string | undefined;
		try {
			text = await Promise.race([answered, released]);
		} finally {
			clearTimeout(timer);
			if (onAbort) signal?.removeEventListener("abort", onAbort);
		}
		const reply = parseJson(text ?? "") as { ok: boolean; result?: unknown; error?: string };
		if (!reply.ok) throw new Error(`${method}: ${reply.error}`);
		return decode(reply.result) as T;
	}

	/**
	 * Ask the server to interrupt the call it is running, bypassing the queue: the `stop` method of
	 * servers that have one. Best effort: a server without it answers with an error, which is ignored.
	 */
	async interrupt(timeoutMs = 5_000): Promise<void> {
		const body = JSON.stringify({ method: "stop", args: [], kwargs: {}, session_id: this.session });
		await Promise.race([post(this.url, body), new Promise((r) => setTimeout(r, timeoutMs).unref())]).catch(() => {});
	}

	async ready(timeoutMs = 300_000): Promise<void> {
		const deadline = Date.now() + timeoutMs;
		for (;;) {
			try {
				await this.call("healthz", {}, 3_000);
				return;
			} catch (err) {
				if (Date.now() > deadline) throw new Error(`${this.url} not ready: ${err}`);
				await new Promise((r) => setTimeout(r, 1_000));
			}
		}
	}
}
