import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import { forgetUnresponsive, parseJson, RpcClient, RpcUnavailable } from "../src/rpc.ts";

/**
 * A fake robot server: `slow` answers after 300 ms; it records every call's start and end, and the
 * most calls it ran at once (the `stop` interrupt is meant to bypass the queue and is not counted).
 */
async function fakeServer() {
	const log: { method: string; start: number; end: number }[] = [];
	let active = 0;
	let maxActive = 0;
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", async () => {
			const { method } = JSON.parse(body) as { method: string };
			const entry = { method, start: Date.now(), end: 0 };
			const counted = method !== "stop";
			if (counted) maxActive = Math.max(maxActive, ++active);
			if (method === "slow") await new Promise((r) => setTimeout(r, 300));
			if (method === "hang") return;
			if (counted) active--;
			entry.end = Date.now();
			log.push(entry);
			res.end(JSON.stringify({ ok: true, result: method }));
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
	const close = () => {
		server.closeAllConnections();
		server.close();
	};
	return { url, log, maxActive: () => maxActive, close };
}

for (const how of ["abort", "timeout"] as const) {
	test(`an ${how} releases the caller but not the endpoint`, async (t) => {
		const srv = await fakeServer();
		t.after(srv.close);
		const rpc = new RpcClient(srv.url);
		const controller = new AbortController();
		const slow = rpc.call("slow", {}, how === "timeout" ? 50 : 10_000, [], controller.signal);
		if (how === "abort") setTimeout(() => controller.abort(), 50);
		await assert.rejects(slow, how === "abort" ? /slow: aborted; the server is still running it/ : /slow: timed out/);
		assert.equal(await rpc.call("fast"), "fast");
		const calls = srv.log.filter((e) => e.method !== "stop");
		assert.deepEqual(
			calls.map((e) => e.method),
			["slow", "fast"],
		);
		assert.ok(calls[1].start >= calls[0].end, "the second call started while the first was in flight");
		assert.equal(srv.maxActive(), 1);
		// Only an abort asks the server to stop the running call.
		assert.equal(
			srv.log.some((e) => e.method === "stop"),
			how === "abort",
		);
	});
}

test("a call that gives up before its turn is never sent", async (t) => {
	const srv = await fakeServer();
	t.after(srv.close);
	const rpc = new RpcClient(srv.url);
	const slow = rpc.call("slow");
	await assert.rejects(rpc.call("queued", {}, 50), /queued: timed out after 50 ms waiting for slow/);
	const controller = new AbortController();
	controller.abort();
	await assert.rejects(rpc.call("aborted", {}, 10_000, [], controller.signal), /aborted/);
	await slow;
	assert.equal(await rpc.call("fast"), "fast");
	assert.deepEqual(
		srv.log.map((e) => e.method),
		["slow", "fast"],
	);
});

test("a server that never answers is given up on, and later calls fail at once", async (t) => {
	const srv = await fakeServer();
	t.after(srv.close);
	const rpc = new RpcClient(srv.url, { graceMs: 50 });
	// The timeout itself is an ordinary error: a slow call is the model's to handle.
	await assert.rejects(
		rpc.call("hang", {}, 50),
		(e: Error) => !(e instanceof RpcUnavailable) && /hang: timed out/.test(e.message),
	);
	// Queued behind the hung call until the grace period marks the endpoint unresponsive.
	await assert.rejects(
		rpc.call("fast", {}, 1_000),
		(e) => e instanceof RpcUnavailable && /stopped answering/.test(e.message),
	);
	const t0 = Date.now();
	await assert.rejects(rpc.call("fast"), /stopped answering \(hang unanswered 0\.1 s after it was sent\)/);
	assert.ok(Date.now() - t0 < 50, "a call to an unresponsive endpoint fails at once");
	assert.deepEqual(
		srv.log.map((e) => e.method),
		[],
	);
	// A new episode gives it another chance.
	forgetUnresponsive();
	assert.equal(await rpc.call("fast"), "fast");
});

test("a refused connection is RpcUnavailable", async () => {
	const srv = await fakeServer();
	srv.close();
	await assert.rejects(new RpcClient(srv.url).call("fast", {}, 1_000), RpcUnavailable);
});

test("NaN and Infinity from Python's json.dumps parse as numbers, not errors", async (t) => {
	// json.dumps({"ok": True, "result": {...}}) with non-finite floats, a string that mentions them, and an escaped quote.
	const body =
		'{"ok": true, "result": {"q": [NaN, Infinity, -Infinity, 1.5], "note": "NaN \\" -Infinity", "e": 1e-3}}';
	const server = createServer((req, res) => {
		req.resume();
		req.on("end", () => res.end(body));
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	t.after(() => server.close());
	const result = await new RpcClient(`http://127.0.0.1:${(server.address() as AddressInfo).port}`).call<{
		q: number[];
		note: string;
		e: number;
	}>("state");
	assert.deepEqual(result.q, [Number.NaN, Infinity, -Infinity, 1.5]);
	assert.equal(result.note, 'NaN " -Infinity');
	assert.equal(result.e, 0.001);
	assert.throws(() => parseJson("{bad NaN"), SyntaxError);
});

test("the server's token rides on every call, and onSent fires only when a call leaves the queue", async (t) => {
	const bodies: Record<string, unknown>[] = [];
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", async () => {
			const parsed = JSON.parse(body) as Record<string, unknown>;
			bodies.push(parsed);
			if (parsed.method === "slow") await new Promise((r) => setTimeout(r, 100));
			res.end(JSON.stringify({ ok: true, result: parsed.method }));
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	t.after(() => {
		server.closeAllConnections();
		server.close();
	});
	const rpc = new RpcClient(`http://127.0.0.1:${(server.address() as AddressInfo).port}`);
	rpc.token = "ab12";
	const sent: string[] = [];
	const slow = rpc.call("slow", {}, 5_000, [], undefined, () => sent.push("slow"));
	const next = rpc.call("next", {}, 5_000, [], undefined, () => sent.push("next"));
	await new Promise((r) => setTimeout(r, 30));
	assert.deepEqual(sent, ["slow"], "next waits in the queue");
	assert.equal(await slow, "slow");
	assert.equal(await next, "next");
	assert.deepEqual(sent, ["slow", "next"]);
	assert.deepEqual(
		bodies.map((b) => b.token),
		["ab12", "ab12"],
	);
});
