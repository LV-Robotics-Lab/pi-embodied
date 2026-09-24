/**
 * Robot service RPC: `POST <endpoint>/call` with `{method, args, kwargs}`, answered by
 * `{ok, result}` or `{ok: false, error, traceback}`. numpy arrays travel as
 * `{__ndarray__: <base64 C-order bytes>, dtype, shape}`, scalars as `{__npscalar__}`.
 */

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
	): Promise<T> {
		const res = await fetch(this.url, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ method, args: encode(args), kwargs: encode(kwargs), session_id: this.session }),
			signal: AbortSignal.timeout(timeoutMs),
		});
		const body = (await res.json()) as { ok: boolean; result?: unknown; error?: string };
		if (!body.ok) throw new Error(`${method}: ${body.error}`);
		return decode(body.result) as T;
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
