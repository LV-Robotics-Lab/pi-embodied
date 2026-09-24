import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { SessionEntry } from "@earendil-works/pi-coding-agent";
import { mergeMemory, renderLeaf, splitFrontmatter, validateMemory } from "../src/memory/corpus.ts";
import { denied, recipe } from "../src/memory/index.ts";

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

test("guard mirrors RPent's memory boundary", () => {
	const g = { root: "/m/libero", home: "/m", inbox: "10_t2_s0" };
	const ro = { ...g, inbox: undefined };
	assert.equal(denied("/m/libero/MEMORY.md", "read", ro), undefined);
	assert.equal(denied("/m/libero/global/a.md", "read", ro), undefined);
	assert.equal(denied("/m/libero", "read", ro), undefined);
	assert.match(denied("/m/libero", "search", ro) ?? "", /memory root/);
	assert.match(denied("/m/libero/_internal/inbox/10_t2_s0/x.md", "read", ro) ?? "", /reading/);
	assert.match(denied("/m/libero/global/a.md", "write", ro) ?? "", /writing/);
	assert.match(denied("/m/robotwin/MEMORY.md", "read", ro) ?? "", /another robot/);
	assert.equal(denied("/tmp/run/10_t2_s0.json", "write", ro), undefined);
	assert.equal(denied("/m/libero/_internal/inbox/10_t2_s0/wip/notes.md", "write", g), undefined);
	assert.equal(denied("/m/libero/_internal/inbox/10_t2_s0/wip/notes.md", "read", g), undefined);
	assert.match(denied("/m/libero/_internal/inbox/other/x.md", "write", g) ?? "", /writing/);
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
