/**
 * Client for the robot service RPC used by env / VLA / perception servers.
 *
 * Every call is `POST <endpoint>/call` with `{method, args, kwargs}`; the server
 * always answers 200 with `{ok: true, result}` or `{ok: false, error, traceback}`.
 * numpy arrays travel as `{__ndarray__: <base64 raw C-order bytes>, dtype, shape}`
 * and numpy scalars as `{__npscalar__, dtype}`. Python servers written against
 * this contract (including existing LIBERO / pi0.5 / SAM3 servers) attach unchanged.
 */

export type DType = "uint8" | "int32" | "int64" | "float32" | "float64" | "bool";

export class NdArray {
	readonly dtype: DType;
	readonly shape: number[];
	readonly data: Buffer;

	constructor(dtype: DType, shape: number[], data: Buffer) {
		this.dtype = dtype;
		this.shape = shape;
		this.data = data;
	}

	static fromNumbers(values: number[] | number[][], dtype: "float32" | "float64" = "float32"): NdArray {
		const rows = Array.isArray(values[0]) ? (values as number[][]) : [values as number[]];
		const flat = rows.flat();
		const shape = Array.isArray(values[0]) ? [rows.length, rows[0].length] : [flat.length];
		const typed = dtype === "float32" ? Float32Array.from(flat) : Float64Array.from(flat);
		return new NdArray(dtype, shape, Buffer.from(typed.buffer));
	}

	get size(): number {
		return this.shape.reduce((a, b) => a * b, 1);
	}

	/** Flat numeric view. Assumes little-endian payloads, which is what numpy produces on x86 and ARM. */
	toArray(): number[] {
		const { buffer, byteOffset, byteLength } = this.data;
		const copy = buffer.slice(byteOffset, byteOffset + byteLength);
		switch (this.dtype) {
			case "uint8":
				return Array.from(new Uint8Array(copy));
			case "bool":
				return Array.from(new Uint8Array(copy));
			case "int32":
				return Array.from(new Int32Array(copy));
			case "int64":
				return Array.from(new BigInt64Array(copy), Number);
			case "float32":
				return Array.from(new Float32Array(copy));
			case "float64":
				return Array.from(new Float64Array(copy));
		}
	}
}

function encode(value: unknown): unknown {
	if (value instanceof NdArray) {
		return { __ndarray__: value.data.toString("base64"), dtype: value.dtype, shape: value.shape };
	}
	if (Array.isArray(value)) return value.map(encode);
	if (value && typeof value === "object") {
		return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, encode(v)]));
	}
	return value;
}

function decode(value: unknown): unknown {
	if (Array.isArray(value)) return value.map(decode);
	if (value && typeof value === "object") {
		const obj = value as Record<string, unknown>;
		if (typeof obj.__ndarray__ === "string") {
			const dtype = String(obj.dtype).replace(/^[<>|=]/, "") as DType;
			return new NdArray(dtype, obj.shape as number[], Buffer.from(obj.__ndarray__, "base64"));
		}
		if ("__npscalar__" in obj) return obj.__npscalar__;
		return Object.fromEntries(Object.entries(obj).map(([k, v]) => [k, decode(v)]));
	}
	return value;
}

export class RpcError extends Error {
	readonly method: string;
	readonly remoteTraceback?: string;

	constructor(method: string, message: string, remoteTraceback?: string) {
		super(`${method}: ${message}`);
		this.method = method;
		this.remoteTraceback = remoteTraceback;
	}
}

export class RpcClient {
	readonly url: string;

	constructor(endpoint: string) {
		const base = endpoint.includes("://") ? endpoint : `http://${endpoint}`;
		if (!base.startsWith("http://") && !base.startsWith("https://")) {
			throw new Error(`Only http(s) RPC endpoints are supported, got ${endpoint}`);
		}
		this.url = `${base.replace(/\/$/, "")}/call`;
	}

	async call<T = unknown>(
		method: string,
		kwargs: Record<string, unknown> = {},
		options: { timeoutMs?: number; args?: unknown[]; signal?: AbortSignal } = {},
	): Promise<T> {
		const timeout = AbortSignal.timeout(options.timeoutMs ?? 30_000);
		const signal = options.signal ? AbortSignal.any([options.signal, timeout]) : timeout;
		const res = await fetch(this.url, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ method, args: encode(options.args ?? []), kwargs: encode(kwargs), session_id: null }),
			signal,
		});
		if (!res.ok) throw new RpcError(method, `HTTP ${res.status}`);
		const body = (await res.json()) as { ok: boolean; result?: unknown; error?: string; traceback?: string };
		if (!body.ok) throw new RpcError(method, body.error ?? "unknown error", body.traceback);
		return decode(body.result) as T;
	}

	async waitForReady(timeoutMs = 60_000): Promise<void> {
		const deadline = Date.now() + timeoutMs;
		let last: unknown;
		while (Date.now() < deadline) {
			try {
				await this.call("healthz", {}, { timeoutMs: 3_000 });
				return;
			} catch (err) {
				last = err;
				await new Promise((r) => setTimeout(r, 500));
			}
		}
		throw new Error(`RPC server at ${this.url} not ready: ${String(last)}`);
	}
}
