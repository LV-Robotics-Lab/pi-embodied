import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { syncMemory } from "../src/memory/sync.ts";

const blob = (text: string) =>
	createHash("sha1")
		.update(`blob ${Buffer.byteLength(text)}\0`)
		.update(text)
		.digest("hex");

/** A stand-in for the HF API: one revision holding `toy/MEMORY.md` and `toy/task_only/a.json`. */
async function fakeHub() {
	const remote: Record<string, string> = { "toy/MEMORY.md": "# memory\n", "toy/task_only/a.json": "{}\n" };
	const server = createServer((req, res) => {
		if (req.url?.startsWith("/api/datasets/")) {
			const siblings = Object.entries(remote).map(([rfilename, text]) => ({ rfilename, blobId: blob(text) }));
			res.end(JSON.stringify({ sha: "abc123", siblings }));
			return;
		}
		const name = decodeURIComponent(req.url?.split("/resolve/abc123/")[1] ?? "");
		if (name in remote) res.end(remote[name]);
		else res.writeHead(404).end();
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	return { url: `http://127.0.0.1:${(server.address() as AddressInfo).port}`, close: () => server.close() };
}

for (const pinned of [true, false]) {
	test(`a ${pinned ? "pinned" : "unpinned"} sync ${pinned ? "removes" : "keeps"} files the revision lacks`, async (t) => {
		const hub = await fakeHub();
		const env = { ...process.env };
		t.after(() => {
			hub.close();
			process.env = env;
		});
		process.env.HF_ENDPOINT = hub.url;
		delete process.env.HF_HUB_OFFLINE;
		if (pinned) process.env.PI_EMBODIED_MEMORY_REVISION = "abc123";
		else delete process.env.PI_EMBODIED_MEMORY_REVISION;
		const root = join(mkdtempSync(join(tmpdir(), "memsync-")), "toy");
		mkdirSync(join(root, "results"), { recursive: true });
		mkdirSync(join(root, "_internal", "inbox"), { recursive: true });
		writeFileSync(join(root, "results", "old.json"), "stale");
		writeFileSync(join(root, "_internal", "inbox", "notes.md"), "local");
		const logs: string[] = [];
		await syncMemory(root, (m) => logs.push(m), "RLinf/test");
		assert.deepEqual(logs, []);
		assert.equal(readFileSync(join(root, "MEMORY.md"), "utf8"), "# memory\n");
		assert.equal(existsSync(join(root, "task_only", "a.json")), true);
		assert.equal(existsSync(join(root, "results", "old.json")), !pinned);
		assert.equal(existsSync(join(root, "_internal", "inbox", "notes.md")), true);
	});
}
