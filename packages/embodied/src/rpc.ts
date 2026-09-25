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

/**
 * Calls to one endpoint run one at a time. A robot env server may accept concurrent
 * calls (RPent's facades allow parallel read-only calls) while its worker pipe does not.
 */
const queues = new Map<string, Promise<unknown>>();

/**
 * Robot services are local or on the tailnet: never go through HTTP_PROXY. fetch and, with
 * NODE_USE_ENV_PROXY, Node's global agents would; explicit agents do not.
 */
const direct = { http: new HttpAgent({ keepAlive: true }), https: new HttpsAgent({ keepAlive: true }) };

function post(url: string, body: string, timeoutMs: number, signal?: AbortSignal): Promise<string> {
	return new Promise((resolve, reject) => {
		if (signal?.aborted) return reject(new Error("aborted"));
		const target = new URL(url);
		const https = target.protocol === "https:";
		const send = https ? httpsRequest : httpRequest;
		const req = send(
			target,
			{
				method: "POST",
				agent: https ? direct.https : direct.http,
				headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) },
			},
			(res) => {
				const chunks: Buffer[] = [];
				res.on("data", (c: Buffer) => chunks.push(c));
				res.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
				res.on("error", reject);
			},
		);
		const onAbort = () => req.destroy(new Error("aborted"));
		signal?.addEventListener("abort", onAbort, { once: true });
		req.setTimeout(timeoutMs, () => req.destroy(new Error(`timed out after ${timeoutMs} ms`)));
		req.on("error", reject);
		req.on("close", () => signal?.removeEventListener("abort", onAbort));
		req.end(body);
	});
}

export class RpcClient {
	url: string;
	/** Server-side session id, for servers that scope state per client (RoboCasa's VLA). */
	session: string | null = null;

	constructor(endpoint: string) {
		const base = endpoint.includes("://") ? endpoint : `http://${endpoint}`;
		this.url = `${base.replace(/\/$/, "")}/call`;
	}

	async call<T = unknown>(
		method: string,
		kwargs: Record<string, unknown> = {},
		timeoutMs = 120_000,
		args: unknown[] = [],
		signal?: AbortSignal,
	): Promise<T> {
		const body = JSON.stringify({ method, args: encode(args), kwargs: encode(kwargs), session_id: this.session });
		const previous = queues.get(this.url) ?? Promise.resolve();
		const current = previous.catch(() => {}).then(() => post(this.url, body, timeoutMs, signal));
		queues.set(this.url, current);
		let text: string;
		try {
			text = await current;
		} finally {
			if (queues.get(this.url) === current) queues.delete(this.url);
		}
		const reply = JSON.parse(text) as { ok: boolean; result?: unknown; error?: string };
		if (!reply.ok) throw new Error(`${method}: ${reply.error}`);
		return decode(reply.result) as T;
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
