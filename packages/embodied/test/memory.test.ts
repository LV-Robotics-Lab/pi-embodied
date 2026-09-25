import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI, SessionEntry } from "@earendil-works/pi-coding-agent";
import { mergeMemory, renderLeaf, splitFrontmatter, validateMemory } from "../src/memory/corpus.ts";
import { canonicalPath, denied, memory, recipe } from "../src/memory/index.ts";

const leaf = (meta: string, body: string) => `---\n${meta}\n---\n${body}`;
const globalDraft = (cells: string, body: string) =>
	leaf(
		`id: lift-first\nscope: global\nkind: primitive\ntitle: Lift before carrying\napplies_when: "after pi0_pick: carry"\nconfidence: single-shot\nevidence:\n  cells: [${cells}]\n  attempts: 2`,
		body,
	);

test("frontmatter round-trips through renderLeaf", () => {
	const meta = { id: "x", title: "a: b", n: 3, yes: "yes", evidence: { cells: ["10_t2_s0", "on"], attempts: 1 } };
	const text = renderLeaf(meta, "\n# Body\n");
	assert.deepEqual(splitFrontmatter(text), { meta, body: "\n# Body\n" });
});

test("merge publishes drafts, merges evidence, archives conflicts and task pairs", async () => {
	const root = mkdtempSync(join(tmpdir(), "memory-"));
	const run = mkdtempSync(join(tmpdir(), "run-"));
	const inbox = (cell: string) => join(root, "_internal", "inbox", cell);

	mkdirSync(join(inbox("10_t2_s0"), "wip"), { recursive: true });
	writeFileSync(
		join(inbox("10_t2_s0"), "new_global_primitive_lift-first_draft.md"),
		globalDraft("10_t2_s0", "\nLift.\n"),
	);
	writeFileSync(
		join(inbox("10_t2_s0"), "suite_10_t2_s0_draft.md"),
		leaf(
			"scope: suite\nsuite: libero10\nregime: task\ntask_id: 2\ntask_language: put the bowl on the plate\nconfidence: single-shot\nevidence:\n  cells: [10_t2_s0]",
			"\n## Applicable pattern\n",
		),
	);
	writeFileSync(join(inbox("10_t2_s0"), "broken.md"), "no frontmatter");
	writeFileSync(join(run, "10_t2_s0.json"), "{}");
	writeFileSync(join(run, "10_t2_s0_recipe.jsonl"), "{}\n");

	const first = await mergeMemory(root, "10_t2_s0", run, true);
	assert.deepEqual([first.global, first.suite, first.task, first.evidence, first.conflicts], [1, 1, 1, 0, 0]);
	assert.match(first.skipped[0], /^broken\.md: missing YAML frontmatter/);
	assert.ok(existsSync(join(root, "global", "lift-first.md")));
	assert.ok(existsSync(join(root, "suite", "suite_libero10_task_t2.md")));
	assert.ok(existsSync(join(root, "_internal", "merged", "10_t2_s0", "wip")));
	assert.ok(!existsSync(inbox("10_t2_s0")));
	assert.ok(existsSync(join(root, "task_only", "10_t2_s0_recipe.jsonl")));
	assert.match(
		readFileSync(join(root, "MEMORY.md"), "utf8"),
		/- \[Lift before carrying\]\(global\/lift-first\.md\) — after pi0_pick: carry/,
	);
	assert.deepEqual(validateMemory(root), []);

	mkdirSync(inbox("goal_t1_s0"), { recursive: true });
	writeFileSync(
		join(inbox("goal_t1_s0"), "new_global_primitive_lift-first.md"),
		globalDraft("goal_t1_s0", "\nOther.\n"),
	);
	const second = await mergeMemory(root, "goal_t1_s0", run, false);
	assert.deepEqual([second.global, second.evidence, second.conflicts, second.task], [0, 1, 1, 0]);
	const merged = splitFrontmatter(readFileSync(join(root, "global", "lift-first.md"), "utf8"));
	assert.deepEqual(merged.meta.evidence, { cells: ["10_t2_s0", "goal_t1_s0"], attempts: 4 });
	assert.equal(merged.meta.confidence, "probable");
	assert.equal(merged.body, "\n\nLift.\n");
	assert.ok(existsSync(join(root, "_internal", "conflicts", "lift-first__from_goal_t1_s0.md")));
});

test("guard enforces the memory boundary", () => {
	const g = {
		root: "/m/libero",
		home: "/m",
		output: "/tmp/run",
		tag: "10_t2_s0",
		inbox: "10_t2_s0",
		attempts: "/tmp/run/attempts",
	};
	const ro = { ...g, inbox: undefined, attempts: undefined };
	assert.equal(denied("/m/libero/MEMORY.md", "read", ro), undefined);
	assert.equal(denied("/m/libero/global/a.md", "read", ro), undefined);
	assert.equal(denied("/m/libero", "read", ro), undefined);
	assert.match(denied("/m/libero", "search", ro) ?? "", /memory root/);
	assert.match(denied("/m/libero/_internal/inbox/10_t2_s0/x.md", "read", ro) ?? "", /reading/);
	assert.match(denied("/m/libero/global/a.md", "write", ro) ?? "", /writing/);
	assert.match(denied("/m/robotwin/MEMORY.md", "read", ro) ?? "", /another robot/);
	assert.equal(denied("/tmp/run/10_t2_s0.json", "write", ro), undefined);
	assert.equal(denied("/tmp/run/10_t2_s0.json", "read", ro), undefined);
	assert.equal(denied("/tmp/run", "read", ro), undefined);
	assert.match(denied("/tmp/run", "search", ro) ?? "", /only the memory/);
	assert.match(denied("/tmp/run/10_t2_s01.json", "write", ro) ?? "", /only the memory/);
	assert.match(denied("/tmp/run/sub/10_t2_s0.json", "write", ro) ?? "", /only the memory/);
	assert.match(denied("/tmp/run/other.jsonl", "read", ro) ?? "", /only the memory/);
	assert.match(denied("/etc/passwd", "read", ro) ?? "", /only the memory/);
	assert.match(denied("/", "search", ro) ?? "", /only the memory/);
	assert.equal(denied("/m/libero/_internal/inbox/10_t2_s0/wip/notes.md", "write", g), undefined);
	assert.equal(denied("/m/libero/_internal/inbox/10_t2_s0/wip/notes.md", "read", g), undefined);
	assert.match(denied("/m/libero/_internal/inbox/other/x.md", "write", g) ?? "", /writing/);
	assert.equal(denied("/tmp/run/attempts/attempt_1_failed.json", "write", g), undefined);
	assert.equal(denied("/tmp/run/attempts", "search", g), undefined);
	assert.match(denied("/tmp/run/attempts/attempt_1_failed.json", "write", ro) ?? "", /only the memory/);
	assert.match(denied("/tmp/run/attempts2/x.json", "write", g) ?? "", /only the memory/);
});

test("recipe keeps successful primitives and segments after the last reset", () => {
	let id = 0;
	const entries: unknown[] = [];
	const call = (name: string, args: Record<string, unknown>, details: unknown, isError = false) => {
		const toolCallId = `c${id++}`;
		const base = { id: toolCallId, parentId: null, timestamp: "" };
		entries.push(
			{
				...base,
				type: "message",
				message: { role: "assistant", content: [{ type: "toolCall", id: toolCallId, name, arguments: args }] },
			},
			{
				...base,
				type: "message",
				message: { role: "toolResult", toolCallId, toolName: name, content: [], details, isError, timestamp: 0 },
			},
		);
	};
	const moved = { result: {}, terminated: false };
	call("move_to", { xyz: [0, 0, 1] }, moved);
	call("reset", {}, moved);
	call("view_env_state", {}, moved);
	call("segment", { prompt: " the black bowl " }, { found: true, camera: "agentview" });
	call("segment", { point: [1, 2] }, { found: false, error: "no mask" });
	call("rotate_wrist", {}, undefined, true);
	call("move_to", { xyz: [0.1, 0, 1] }, { result: { name: "move_to" }, terminated: false });
	call("release", {}, { result: {}, terminated: true });
	const primitives = new Set(["move_to", "release", "rotate_wrist", "reset"]);
	assert.deepEqual(recipe(entries as SessionEntry[], primitives), [
		{ action: "segment", prompt: "the black bowl", camera: "agentview" },
		{ action: "move_to", xyz: [0.1, 0, 1] },
		{ action: "release" },
	]);
	assert.equal(recipe(entries.slice(0, 4) as SessionEntry[], primitives), undefined);
});

/** memory() wired to a stub pi: returns the tool_call hook after session_start (unless `start` is false). */
async function guarded(opts: { cell?: string; output?: string; start?: boolean; explore?: boolean } = {}) {
	const base = realpathSync(mkdtempSync(join(tmpdir(), "guard-")));
	const home = join(base, "memory");
	const run = join(base, "run");
	mkdirSync(join(home, "libero", "global"), { recursive: true });
	mkdirSync(join(home, "robotwin"), { recursive: true });
	mkdirSync(run);
	writeFileSync(join(home, "libero", "MEMORY.md"), "# memory\n");
	writeFileSync(join(base, "secret.txt"), "secret\n");
	const hooks = new Map<string, (event: unknown, ctx: unknown) => unknown>();
	const flags: Record<string, unknown> = { "output-dir": opts.output ?? run };
	const pi = {
		on: (name: string, fn: (event: unknown, ctx: unknown) => unknown) => hooks.set(name, fn),
		registerFlag: () => {},
		registerCommand: () => {},
		getFlag: (name: string) => flags[name],
	} as unknown as ExtensionAPI;
	const cell = opts.cell;
	memory(pi, {
		home: () => home,
		cell: () => (cell === undefined ? undefined : { tag: cell, reference: cell }),
		explore: () => opts.explore ?? true,
	});
	const ctx = { cwd: run, hasUI: false, sessionManager: { getSessionDir: () => "", getBranch: () => [] } };
	if (opts.start !== false) await hooks.get("session_start")?.({}, ctx);
	const call = (toolName: string, path?: string) =>
		hooks.get("tool_call")?.({ type: "tool_call", toolName, toolCallId: "t", input: { path } }, ctx) as
			| { block: true; reason: string }
			| undefined;
	return { base, home, run, call };
}

test("guard denies file tools outside memory and the cell's output files", async () => {
	const { base, home, run, call } = await guarded({ cell: "10_t2_s0" });
	for (const tool of ["read", "ls", "grep", "find", "write", "edit"]) {
		assert.ok(call(tool, join(base, "secret.txt"))?.block, `${tool} absolute`);
		assert.ok(call(tool, "../secret.txt")?.block, `${tool} ..`);
		assert.ok(call(tool, join(home, "libero", "..", "..", "secret.txt"))?.block, `${tool} memory/..`);
		assert.ok(call(tool, join(home, "robotwin"))?.block, `${tool} other robot`);
		assert.ok(call(tool, "/")?.block, `${tool} /`);
	}
	assert.ok(call("grep")?.block, "default path is cwd = output dir root");
	assert.ok(call("write", "notes.md")?.block, "relative write in cwd");

	// Symlink escapes: an existing link, a link directory, and a dangling link whose write lands outside.
	symlinkSync(join(base, "secret.txt"), join(run, "10_t2_s0.json"));
	symlinkSync(base, join(home, "libero", "global", "up"));
	symlinkSync(join(base, "planted.txt"), join(run, "10_t2_s0_audit.json"));
	assert.equal(canonicalPath(join(run, "10_t2_s0_audit.json"), run), join(base, "planted.txt"));
	assert.match(call("read", "10_t2_s0.json")?.reason ?? "", /only the memory/);
	assert.ok(call("read", join(home, "libero", "global", "up", "secret.txt"))?.block);
	assert.ok(call("write", join(run, "10_t2_s0_audit.json"))?.block);
	const loop = join(run, "10_t2_s0.loop");
	symlinkSync(loop, loop);
	assert.match(call("write", loop)?.reason ?? "", /symbolic links/);
});

test("guard allows the memory, the exploration inbox and the audit write", async () => {
	const { home, run, call } = await guarded({ cell: "10_t2_s0" });
	assert.equal(call("write", join(run, "10_t2_s0.json")), undefined);
	assert.equal(call("write", "10_t2_s0.json"), undefined);
	assert.equal(call("read", "10_t2_s0.json"), undefined);
	assert.equal(call("ls"), undefined);
	assert.equal(call("read", join(home, "libero", "MEMORY.md")), undefined);
	assert.equal(call("grep", join(home, "libero", "global")), undefined);
	assert.equal(call("write", join(home, "libero", "_internal", "inbox", "10_t2_s0", "x.md")), undefined);
	assert.ok(call("write", join(home, "libero", "_internal", "inbox", "10_t2_s1", "x.md"))?.block);
	assert.ok(call("write", join(home, "libero", "global", "x.md"))?.block);
	assert.equal(call("bash"), undefined, "tools other than the file tools are not the guard's business");
});

test("guard fails closed without a cell, output dir or session_start", async () => {
	for (const setup of [{ start: false, cell: "10_t2_s0" }, { output: "", cell: "10_t2_s0" }, { cell: undefined }]) {
		const { home, run, call } = await guarded(setup);
		for (const [tool, path] of [
			["read", join(home, "libero", "MEMORY.md")],
			["write", join(run, "10_t2_s0.json")],
			["ls", run],
			["find", home],
		] as const)
			assert.match(call(tool, path)?.reason ?? "", /file access is disabled/, `${JSON.stringify(setup)} ${tool}`);
	}
	await assert.rejects(guarded({ cell: "../x" }), /invalid memory cell tag/);
});
