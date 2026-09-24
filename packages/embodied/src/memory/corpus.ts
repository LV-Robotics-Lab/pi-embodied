/**
 * RPent's memory corpus, the same on disk as RPent so the published RLinf/RPent-memory works as is:
 *
 *   <root>/MEMORY.md                                   index of global/ and suite/ (from frontmatter)
 *   <root>/global/<id>.md, <root>/suite/<id>.md        published leaves with YAML frontmatter
 *   <root>/task_only/<tag>.json + <tag>_recipe.jsonl   solved audit + recipe pairs
 *   <root>/_internal/{inbox/<tag>,merged/<tag>,conflicts}   exploration drafts, never read in evaluation
 *
 * Mirrors rpent/memory/manager.py: schema, merge, validate and index.
 */

import {
	cpSync,
	existsSync,
	mkdirSync,
	readdirSync,
	readFileSync,
	renameSync,
	rmSync,
	statSync,
	writeFileSync,
} from "node:fs";
import { basename, dirname, join } from "node:path";
import { parseFrontmatter } from "@earendil-works/pi-coding-agent";

export const SCOPES = ["global", "suite"];
const KINDS = ["failure", "infra", "perception", "primitive", "strategy"];
const CONFIDENCE = ["single-shot", "probable", "verified"];
const PREFIXES = ["new_global_", "new_suite_", "new_", "suite_", ...KINDS.map((k) => `${k}_`)];

type Meta = Record<string, unknown>;
export type MergeResult = {
	cell: string;
	global: number;
	suite: number;
	task: number;
	evidence: number;
	conflicts: number;
	skipped: string[];
};

export const str = (v: unknown) => (typeof v === "string" ? v : "");
const isMap = (v: unknown): v is Meta => typeof v === "object" && v !== null && !Array.isArray(v);
const list = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const oneOf = (v: unknown, xs: string[]) => typeof v === "string" && xs.includes(v);
const py = (xs: string[]) => `[${xs.map((x) => `'${x}'`).join(", ")}]`;
const repr = (v: unknown) => (typeof v === "string" ? `'${v}'` : v == null ? "None" : JSON.stringify(v));
export const message = (e: unknown) => (e instanceof Error ? e.message : String(e));
const readText = (path: string) => readFileSync(path, "utf8").replace(/\r\n?/g, "\n");
// ---------------------------------------------------------------------------
// Leaves: frontmatter, schema, canonical id, rendering
// ---------------------------------------------------------------------------

export function splitFrontmatter(text: string): { meta: Meta; body: string } {
	if (!text.startsWith("---")) throw new Error("missing YAML frontmatter");
	const end = text.indexOf("\n---", 3);
	if (end < 0) throw new Error("unterminated YAML frontmatter");
	let meta: unknown = null;
	try {
		if (text.slice(3, end).trim()) meta = parseFrontmatter(text.slice(0, end + 4)).frontmatter;
	} catch (e) {
		throw new Error(`invalid YAML frontmatter: ${message(e)}`);
	}
	if (!isMap(meta)) throw new Error("frontmatter must be a mapping");
	return { meta, body: text.slice(end + 4) };
}

export function checkLeaf(m: Meta): void {
	if (!oneOf(m.scope, SCOPES)) throw new Error(`scope must be one of ${py(SCOPES)}, got ${repr(m.scope)}`);
	if (m.scope === "suite") {
		for (const f of ["suite", "regime", "task_id", "task_language"])
			if (m[f] == null || m[f] === "") throw new Error(`suite memory requires '${f}'`);
	} else {
		if (!oneOf(m.kind, KINDS)) throw new Error(`kind must be one of ${py(KINDS)}`);
		for (const f of ["title", "applies_when"])
			if (!String(m[f] ?? "").trim()) throw new Error(`global memory requires '${f}'`);
	}
	if (!oneOf(m.confidence, CONFIDENCE)) throw new Error(`confidence must be one of ${py(CONFIDENCE)}`);
	const evidence = m.evidence || {};
	if (!isMap(evidence)) throw new Error("evidence must be a mapping");
	if (!list(evidence.cells).length) throw new Error("evidence.cells must be a non-empty list");
}

export function canonicalId(file: string, m: Meta): string {
	if (m.scope === "suite") return `suite_${m.suite}_${m.regime}_t${m.task_id}`;
	let stem = basename(file)
		.replace(/\.[^.]*$/, "")
		.replace(/_draft$/, "");
	const prefix = PREFIXES.find((p) => stem.startsWith(p));
	if (prefix) stem = stem.slice(prefix.length);
	return String(m.id || stem).trim();
}

/** YAML that both PyYAML and `yaml` parse: block keys, flow values, JSON-quoted strings. */
function scalar(v: unknown): string {
	if (typeof v === "string")
		return /^[A-Za-z_][\w./-]*$/.test(v) && !/^(true|false|yes|no|on|off|null|y|n)$/i.test(v) ? v : JSON.stringify(v);
	if (Array.isArray(v)) return `[${v.map(scalar).join(", ")}]`;
	if (isMap(v))
		return `{${Object.entries(v)
			.map(([k, x]) => `${scalar(k)}: ${scalar(x)}`)
			.join(", ")}}`;
	return v == null ? "null" : String(v);
}

export function renderLeaf(m: Meta, body: string): string {
	const yaml = Object.entries(m).map(([k, v]) =>
		isMap(v) && Object.keys(v).length
			? `${scalar(k)}:\n${Object.entries(v)
					.map(([kk, x]) => `  ${scalar(kk)}: ${scalar(x)}\n`)
					.join("")}`
			: `${scalar(k)}: ${scalar(v)}\n`,
	);
	return `---\n${yaml.join("")}---${body}`;
}

const order = (a: unknown, b: unknown) =>
	typeof a === "number" && typeof b === "number" ? a - b : String(a) < String(b) ? -1 : String(a) > String(b) ? 1 : 0;
const union = (a: unknown, b: unknown) =>
	[...new Map([...list(a), ...list(b)].map((x) => [JSON.stringify(x), x])).values()].sort(order);
const int = (v: unknown) => Math.trunc(Number(v || 0)) || 0;
/** Attempt counts add up; the published corpus also has per-cell lists, which concatenate. */
const attempts = (a: unknown, b: unknown) =>
	Array.isArray(a) || Array.isArray(b)
		? [a, b].flatMap((v) => (Array.isArray(v) ? v : v == null ? [] : [v]))
		: int(a) + int(b);

/** Union the evidence of two versions of a leaf and recompute its confidence. */
export function mergeEvidence(old: Meta, draft: Meta): Meta {
	const o = isMap(old.evidence) ? old.evidence : {};
	const n = isMap(draft.evidence) ? draft.evidence : {};
	const cells = union(o.cells, n.cells);
	const evidence: Meta = { ...o, cells, attempts: attempts(o.attempts, n.attempts) };
	for (const k of ["solved_seeds", "failed_seeds", "contradicted_by"])
		if (k in o || k in n) evidence[k] = union(o[k], n[k]);
	const tasks = new Set(cells.map((c) => String(c).replace(/_s(?:(?!_s).)*$/, "")));
	const confidence =
		cells.length >= 3 && tasks.size >= 2 ? "verified" : cells.length >= 2 ? "probable" : "single-shot";
	return { ...old, evidence, confidence };
}

// ---------------------------------------------------------------------------
// Corpus operations (rpent-memory merge / validate / build-index / sync)
// ---------------------------------------------------------------------------

function mdFiles(dir: string): string[] {
	try {
		return readdirSync(dir, { withFileTypes: true })
			.filter((e) => !e.isDirectory() && e.name.endsWith(".md"))
			.map((e) => e.name)
			.sort();
	} catch {
		return [];
	}
}

export function hasFiles(dir: string): boolean {
	try {
		return readdirSync(dir, { recursive: true, withFileTypes: true }).some((e) => e.isFile());
	} catch {
		return false;
	}
}

/** Directory lock: merges from parallel cells serialize; a lock older than 5 min is stale. */
async function locked<T>(dir: string, fn: () => T): Promise<T> {
	const lock = join(dir, "merge.lock.d");
	for (;;) {
		try {
			mkdirSync(lock);
			break;
		} catch (e) {
			if ((e as NodeJS.ErrnoException).code !== "EEXIST") throw e;
			try {
				if (Date.now() - statSync(lock).mtimeMs > 300_000) rmSync(lock, { recursive: true, force: true });
			} catch {
				// another process removed it first
			}
			await new Promise((r) => setTimeout(r, 200));
		}
	}
	try {
		return fn();
	} finally {
		rmSync(lock, { recursive: true, force: true });
	}
}

/**
 * Publish one cell: inbox drafts merge into global/ and suite/ (differing prose is archived
 * under _internal/conflicts), a solved audit/recipe pair is copied to task_only/, and
 * MEMORY.md is rebuilt.
 */
export function mergeMemory(root: string, cell: string, runDir: string, solved: boolean): Promise<MergeResult> {
	const internal = join(root, "_internal");
	const inbox = join(internal, "inbox", cell);
	const conflicts = join(internal, "conflicts");
	const merged = join(internal, "merged");
	const tiers: Record<string, string> = {
		global: join(root, "global"),
		suite: join(root, "suite"),
		task: join(root, "task_only"),
	};
	for (const d of [...Object.values(tiers), conflicts, merged, dirname(inbox)]) mkdirSync(d, { recursive: true });
	const result: MergeResult = { cell, global: 0, suite: 0, task: 0, evidence: 0, conflicts: 0, skipped: [] };
	const conflict = (id: string, meta: Meta, body: string) => {
		writeFileSync(join(conflicts, `${id}__from_${cell}.md`), renderLeaf(meta, body));
		result.conflicts++;
	};
	return locked(internal, () => {
		let published = false;
		for (const name of mdFiles(inbox)) {
			let draft: { meta: Meta; body: string };
			try {
				draft = splitFrontmatter(readText(join(inbox, name)));
				checkLeaf(draft.meta);
			} catch (e) {
				result.skipped.push(`${name}: ${message(e)}`);
				continue;
			}
			const { meta, body } = draft;
			const scope = meta.scope as "global" | "suite";
			const id = canonicalId(name, meta);
			if (!id || id.includes("/") || id.includes("\\") || id.startsWith(".")) {
				result.skipped.push(`${name}: id ${repr(id)} is not a file name`);
				continue;
			}
			meta.id = id;
			const dest = join(tiers[scope], `${id}.md`);
			published = true;
			if (!existsSync(dest)) {
				writeFileSync(dest, renderLeaf(meta, body));
				result[scope]++;
				continue;
			}
			let old: { meta: Meta; body: string };
			try {
				old = splitFrontmatter(readText(dest));
			} catch {
				conflict(id, meta, body);
				result.skipped.push(`${name}: existing non-mergeable note ${id}.md left untouched`);
				continue;
			}
			if (list(isMap(old.meta.evidence) ? old.meta.evidence.cells : undefined).includes(cell)) continue;
			writeFileSync(dest, renderLeaf(mergeEvidence(old.meta, meta), old.body));
			result.evidence++;
			if (body.trim() !== old.body.trim()) conflict(id, meta, body);
		}
		if (published) {
			rmSync(join(merged, cell), { recursive: true, force: true });
			renameSync(inbox, join(merged, cell));
		}
		const [audit, recipe] = [`${cell}.json`, `${cell}_recipe.jsonl`];
		if (solved && existsSync(join(runDir, audit)) && existsSync(join(runDir, recipe))) {
			const [a, r] = [existsSync(join(tiers.task, audit)), existsSync(join(tiers.task, recipe))];
			if (!a && !r) {
				for (const f of [audit, recipe]) cpSync(join(runDir, f), join(tiers.task, f), { preserveTimestamps: true });
				result.task = 1;
			} else if (a !== r) result.skipped.push("incomplete existing task audit/recipe pair");
		}
		rebuildIndex(root);
		return result;
	});
}

function leaves(root: string, scope: string): { name: string; meta: Meta }[] {
	return mdFiles(join(root, scope)).flatMap((name) => {
		try {
			return [{ name, meta: splitFrontmatter(readText(join(root, scope, name))).meta }];
		} catch {
			return [];
		}
	});
}

/** Regenerate MEMORY.md from global/ and suite/ frontmatter; nothing is written without leaves. */
export function rebuildIndex(root: string): string | undefined {
	const groups = { global: leaves(root, "global"), suite: leaves(root, "suite") };
	if (!groups.global.length && !groups.suite.length) return undefined;
	const lines = ["# Layered memory index", "", "Generated from memory leaf frontmatter."];
	for (const [scope, title] of [
		["global", "Global"],
		["suite", "Suite"],
	] as const) {
		lines.push("", `## ${title}`, "");
		if (!groups[scope].length) lines.push("_(none)_");
		for (const { name, meta } of groups[scope]) {
			const applies = meta.applies_when || "";
			lines.push(`- [${meta.title || meta.id || name}](${scope}/${name})${applies ? ` — ${applies}` : ""}`);
		}
	}
	writeFileSync(join(root, "MEMORY.md"), `${lines.join("\n")}\n`);
	return join(root, "MEMORY.md");
}

/** Schema problems of frontmatter-bearing global/ and suite/ leaves (plain Markdown passes). */
export function validateMemory(root: string): string[] {
	const problems: string[] = [];
	const ids = new Map<string, string>();
	for (const scope of SCOPES) {
		for (const name of mdFiles(join(root, scope))) {
			const rel = `${scope}/${name}`;
			const text = readText(join(root, rel));
			if (!text.startsWith("---")) continue;
			let meta: Meta;
			try {
				meta = splitFrontmatter(text).meta;
				checkLeaf(meta);
			} catch (e) {
				problems.push(`${rel}: ${message(e)}`);
				continue;
			}
			const id = String(meta.id || "");
			if (id !== name.slice(0, -3)) problems.push(`${rel}: id ${repr(id)} does not match filename`);
			if (ids.has(id)) problems.push(`${rel}: duplicate id also in ${ids.get(id)}`);
			ids.set(id, rel);
		}
	}
	return problems;
}
