import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import libero from "../src/robots/libero/index.ts";

test("LIBERO --env-url URL#token=HEX attaches with the token, like every robot's attach()", async () => {
	const seen: { method: string; token?: string }[] = [];
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const call = JSON.parse(body) as { method: string; token?: string };
			seen.push(call);
			const ok = call.token === "beef";
			res.writeHead(ok ? 200 : 403, { "Content-Type": "application/json" });
			res.end(JSON.stringify(ok ? { ok: true, result: {} } : { ok: false, error: "bad token" }));
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
	const handlers = new Map<string, ((e: unknown, c: unknown) => unknown)[]>();
	const flags: Record<string, unknown> = {};
	const pi = {
		on: (n: string, fn: (e: unknown, c: unknown) => unknown) => handlers.set(n, [...(handlers.get(n) ?? []), fn]),
		registerFlag: (n: string, o: { default?: unknown }) => {
			flags[n] = o.default;
		},
		getFlag: (n: string) => flags[n],
		registerTool: () => {},
		registerCommand: () => {},
		registerProvider: () => {},
		registerMessageRenderer: () => {},
		setActiveTools: () => {},
		getActiveTools: () => [],
		appendEntry: () => {},
		sendMessage: () => {},
		sendUserMessage: () => {},
		getThinkingLevel: () => "off",
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	libero(pi);
	flags["env-url"] = `${url}#token=beef`;
	const ctx = {
		hasUI: true,
		cwd: "/tmp",
		ui: { notify: () => {}, setWidget: () => {}, setStatus: () => {} },
		shutdown: () => {},
		sessionManager: {
			getBranch: () => [],
			getSessionId: () => "s",
			getSessionFile: () => undefined,
			getSessionDir: () => "/tmp",
		},
		modelRegistry: {},
	};
	try {
		for (const fn of handlers.get("session_start") ?? [])
			await Promise.resolve(fn({ type: "session_start" }, ctx)).catch(() => {});
		assert.ok(seen.length > 0, "LIBERO called its env server");
		assert.equal(seen[0].method, "healthz");
		assert.ok(
			seen.every((c) => c.token === "beef"),
			JSON.stringify(seen.map((c) => [c.method, c.token])),
		);
	} finally {
		server.closeAllConnections();
		server.close();
	}
});
