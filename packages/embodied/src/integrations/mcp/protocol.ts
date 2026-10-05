/**
 * A minimal MCP server over pi's MCP package (@earendil-works/pi-mcp, a client library: its JSON-RPC
 * types, protocol versions and `McpTransport` are reused here; the server side it does not have is
 * these two classes). `McpToolServer` answers `initialize`, `ping`, `tools/list` and `tools/call`
 * for one `ToolProvider` and honours `notifications/cancelled`; `StdioServerTransport` is the
 * newline-delimited JSON-RPC stream on this process's stdin/stdout (what `StdioTransport` speaks
 * from the client side). Nothing else may write to stdout while it runs: logs go to stderr.
 */

import {
	type CallToolResult,
	type InitializeResult,
	isJsonRpcNotification,
	isJsonRpcRequest,
	JSON_RPC_ERROR_CODES,
	type JsonRpcId,
	type JsonRpcMessage,
	type JsonRpcRequest,
	LATEST_PROTOCOL_VERSION,
	McpError,
	type McpTransport,
	type McpTransportCloseListener,
	type McpTransportErrorListener,
	type McpTransportMessageListener,
	parseJsonRpcMessage,
	SUPPORTED_PROTOCOL_VERSIONS,
	type Tool,
} from "@earendil-works/pi-mcp";

/** What a server exposes: its identity, the tools, and how a call runs. */
export type ToolProvider = {
	info: { name: string; version: string; title?: string };
	instructions?: string;
	list(): Promise<Tool[]> | Tool[];
	/** Run a tool; an unknown tool throws `McpError` (a protocol error), a failed run returns `isError`. */
	call(name: string, args: Record<string, unknown>, signal: AbortSignal): Promise<CallToolResult>;
};

/** The protocol version answered: the client's when this server knows it, else the latest. */
export const negotiate = (asked: unknown) =>
	typeof asked === "string" && (SUPPORTED_PROTOCOL_VERSIONS as readonly string[]).includes(asked)
		? asked
		: LATEST_PROTOCOL_VERSION;

export class McpToolServer {
	private readonly pending = new Map<JsonRpcId, AbortController>();
	private closed = false;
	private readonly provider: ToolProvider;
	private readonly transport: McpTransport;

	constructor(provider: ToolProvider, transport: McpTransport) {
		this.provider = provider;
		this.transport = transport;
	}

	async start(): Promise<void> {
		this.transport.onMessage((m) => void this.handle(m));
		this.transport.onClose(() => {
			this.closed = true;
			for (const c of this.pending.values()) c.abort();
			this.pending.clear();
		});
		await this.transport.start();
	}

	async close(): Promise<void> {
		await this.transport.close();
	}

	private async handle(message: JsonRpcMessage): Promise<void> {
		if (isJsonRpcNotification(message)) {
			if (message.method === "notifications/cancelled") {
				const id = (message.params as { requestId?: JsonRpcId } | undefined)?.requestId;
				if (id !== undefined) this.pending.get(id)?.abort();
			}
			return;
		}
		if (!isJsonRpcRequest(message)) return;
		const controller = new AbortController();
		this.pending.set(message.id, controller);
		try {
			const result = await this.dispatch(message, controller.signal);
			await this.reply({ jsonrpc: "2.0", id: message.id, result });
		} catch (err) {
			const e =
				err instanceof McpError
					? { code: err.code, message: err.message, ...(err.data === undefined ? {} : { data: err.data }) }
					: {
							code: JSON_RPC_ERROR_CODES.internalError,
							message: err instanceof Error ? err.message : String(err),
						};
			await this.reply({ jsonrpc: "2.0", id: message.id, error: e });
		} finally {
			this.pending.delete(message.id);
		}
	}

	private async reply(message: JsonRpcMessage) {
		if (this.closed) return;
		await this.transport.send(message).catch(() => {});
	}

	private async dispatch(req: JsonRpcRequest, signal: AbortSignal): Promise<unknown> {
		const params = (req.params ?? {}) as Record<string, unknown>;
		switch (req.method) {
			case "initialize": {
				const result: InitializeResult = {
					protocolVersion: negotiate(params.protocolVersion),
					capabilities: { tools: {} },
					serverInfo: this.provider.info,
					...(this.provider.instructions ? { instructions: this.provider.instructions } : {}),
				};
				return result;
			}
			case "ping":
				return {};
			case "tools/list":
				return { tools: await this.provider.list() };
			case "tools/call": {
				const name = params.name;
				if (typeof name !== "string")
					throw new McpError(JSON_RPC_ERROR_CODES.invalidParams, "tools/call: params.name must be a string");
				const args = params.arguments;
				if (args !== undefined && (typeof args !== "object" || args === null || Array.isArray(args)))
					throw new McpError(JSON_RPC_ERROR_CODES.invalidParams, "tools/call: params.arguments must be an object");
				return this.provider.call(name, (args ?? {}) as Record<string, unknown>, signal);
			}
			default:
				throw new McpError(JSON_RPC_ERROR_CODES.methodNotFound, `Method not found: ${req.method}`);
		}
	}
}

/** Listener bookkeeping (pi-mcp's `TransportEvents` is not exported; this is the same contract). */
class Listeners {
	message = new Set<McpTransportMessageListener>();
	error = new Set<McpTransportErrorListener>();
	close = new Set<McpTransportCloseListener>();
	closed = false;
	emitClose() {
		if (this.closed) return;
		this.closed = true;
		for (const l of this.close) l();
	}
}

/**
 * The server end of MCP's stdio transport: one JSON-RPC message per line on `input` (default
 * `process.stdin`), replies on `output` (default `process.stdout`). EOF on `input` closes it.
 */
export class StdioServerTransport implements McpTransport {
	private readonly l = new Listeners();
	private buffer = "";
	private readonly input: NodeJS.ReadableStream;
	private readonly output: NodeJS.WritableStream;

	constructor(o: { input?: NodeJS.ReadableStream; output?: NodeJS.WritableStream } = {}) {
		this.input = o.input ?? process.stdin;
		this.output = o.output ?? process.stdout;
	}

	async start(): Promise<void> {
		this.input.setEncoding?.("utf8");
		this.input.on("data", (chunk: string | Buffer) => this.feed(chunk.toString()));
		this.input.on("end", () => this.l.emitClose());
		this.input.on("error", (err: Error) => {
			for (const fn of this.l.error) fn(err);
		});
		this.input.resume?.();
	}

	private feed(text: string) {
		this.buffer += text;
		for (;;) {
			const at = this.buffer.indexOf("\n");
			if (at < 0) return;
			const line = this.buffer.slice(0, at).trim();
			this.buffer = this.buffer.slice(at + 1);
			if (!line) continue;
			let message: JsonRpcMessage;
			try {
				message = parseJsonRpcMessage(JSON.parse(line));
			} catch (err) {
				for (const fn of this.l.error) fn(err instanceof Error ? err : new Error(String(err)));
				continue;
			}
			for (const fn of this.l.message) fn(message);
		}
	}

	async send(message: JsonRpcMessage): Promise<void> {
		if (this.l.closed) throw new Error("stdio transport closed");
		await new Promise<void>((resolve, reject) =>
			this.output.write(`${JSON.stringify(message)}\n`, (err?: Error | null) => (err ? reject(err) : resolve())),
		);
	}

	async close(): Promise<void> {
		this.l.emitClose();
		this.input.pause?.();
	}

	onMessage(listener: McpTransportMessageListener): () => void {
		this.l.message.add(listener);
		return () => this.l.message.delete(listener);
	}

	onError(listener: McpTransportErrorListener): () => void {
		this.l.error.add(listener);
		return () => this.l.error.delete(listener);
	}

	onClose(listener: McpTransportCloseListener): () => void {
		this.l.close.add(listener);
		return () => this.l.close.delete(listener);
	}
}
