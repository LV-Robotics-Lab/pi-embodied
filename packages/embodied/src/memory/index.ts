/**
 * Layered memory as a pi extension helper. A robot calls `memory(pi, { cell, primitives, ... })`:
 * the agent reads memory with pi's built-in read/ls/grep/find and writes with write, a tool_call guard
 * enforces the memory access boundary, the published corpus is synced from Hugging Face, and the
 * solved recipe is rebuilt from the session branch. Loaded on its own it only adds the command:
 *
 *   pi -e packages/embodied/src/memory -p "/memory validate" --memory-dir memory/libero
 */

import { existsSync, mkdirSync, readlinkSync, realpathSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI, ExtensionContext, SessionEntry } from "@earendil-works/pi-coding-agent";
import { hasFiles, mergeMemory, rebuildIndex, str, validateMemory } from "./corpus.ts";
import { syncMemory } from "./sync.ts";

const READABLE = ["global", "suite", "task_only", "results"];
const ACCESS: Record<string, Access> = {
	read: "read",
	ls: "read",
	grep: "search",
	find: "search",
	write: "write",
	edit: "write",
};

type Access = "read" | "search" | "write";
type Details = { terminated?: unknown; error?: unknown; result?: { error?: unknown }; camera?: unknown } | undefined;
/** Canonical roots: the memory corpus, every robot's memory home, the run's output dir and the cell tag, plus read-only dirs. */
export type Guard = { root: string; home: string; output: string; tag: string; inbox?: string; readable?: string[] };
export type MemoryOptions = {
	/** Corpus name under the memory home, also the Hugging Face subdirectory (default "libero"). */
	robot?: string;
	/** Directory holding every robot's corpus (default $PI_EMBODIED_MEMORY, else ~/.pi/embodied/memory). */
	home?: () => string;
	/** The current cell; without it there is no guard, sync, recipe or prompt (maintenance only). */
	cell?: () => { tag: string; reference: string } | undefined;
	/** State-advancing tools that belong in a recipe (plus successful `segment` calls). */
	primitives?: readonly string[];
	/** Exploration run: local profile, the cell's inbox becomes writable. */
	explore?: () => boolean;
	/** Directories the agent may also read and search, e.g. the robot's saved state images. */
	readable?: () => string[];
};

const defaultHome = () => process.env.PI_EMBODIED_MEMORY || join(homedir(), ".pi", "embodied", "memory");

// ---------------------------------------------------------------------------
// Access boundary and recipes (write_recipe_from_states)
// ---------------------------------------------------------------------------

/**
 * Canonical path: pi's tool path normalization, then symlinks resolved on the existing prefix,
 * dangling ones included (a write through them lands on their target). Throws on a symlink loop.
 */
export function canonicalPath(input: string | undefined, cwd: string): string {
	let p = (input || ".").replace(/[\u00A0\u2000-\u200A\u202F\u205F\u3000]/g, " ");
	if (p.startsWith("@")) p = p.slice(1);
	if (p === "~" || p.startsWith("~/")) p = join(homedir(), p.slice(1));
	if (p.startsWith("file://")) p = fileURLToPath(p);
	let head = resolve(cwd, p);
	const tail: string[] = [];
	for (let hops = 0; hops <= 40; ) {
		try {
			return join(realpathSync(head), ...tail);
		} catch {}
		let link: string | undefined;
		try {
			link = readlinkSync(head);
		} catch {}
		if (link !== undefined) {
			head = resolve(dirname(head), link);
			hops++;
		} else if (dirname(head) === head) return resolve(cwd, p);
		else {
			tail.unshift(basename(head));
			head = dirname(head);
		}
	}
	throw new Error(`too many symbolic links: ${input}`);
}

/**
 * Why `path` (canonical) may not be accessed, or undefined. Deny by default: only the published memory
 * (read), the cell's inbox (exploration) and the cell's files in the output dir (`<tag>*`) are reachable.
 */
export function denied(path: string, access: Access, g: Guard): string | undefined {
	const inside = (base: string) => path === base || path.startsWith(base + sep);
	if (access !== "write" && g.readable?.some(inside)) return undefined;
	if (!inside(g.root)) {
		if (inside(g.home)) return `access to another robot's memory is denied: ${path}`;
		if (path === g.output && access === "read") return undefined;
		const name = basename(path);
		const own = name === g.tag || name.startsWith(`${g.tag}.`) || name.startsWith(`${g.tag}_`);
		if (dirname(path) === g.output && own) return undefined;
		return `only the memory and this cell's ${g.tag}.* / ${g.tag}_* files in ${g.output} are accessible: ${path}`;
	}
	const parts = relative(g.root, path).split(sep).filter(Boolean);
	const ownInbox = g.inbox !== undefined && parts[0] === "_internal" && parts[1] === "inbox" && parts[2] === g.inbox;
	if (access === "write") return ownInbox ? undefined : `writing to memory is denied in this mode: ${path}`;
	if (!parts.length)
		return access === "read"
			? undefined
			: `search global/, suite/ or task_only/ rather than the memory root: ${path}`;
	if (READABLE.includes(parts[0]) || (parts.length === 1 && parts[0].endsWith(".md")) || ownInbox) return undefined;
	return `reading this memory path is denied: ${path}`;
}

type Call = { name: string; args: Record<string, unknown>; details: Details; isError: boolean };

function toolCalls(entries: SessionEntry[]): Call[] {
	const args = new Map<string, Record<string, unknown>>();
	const calls: Call[] = [];
	for (const entry of entries) {
		if (entry.type !== "message") continue;
		const m = entry.message;
		if (m.role === "assistant")
			for (const part of m.content) if (part.type === "toolCall") args.set(part.id, part.arguments);
		if (m.role === "toolResult")
			calls.push({
				name: m.toolName,
				args: args.get(m.toolCallId) ?? {},
				details: m.details as unknown as Details,
				isError: m.isError,
			});
	}
	return calls;
}

/**
 * The command sequence after the last successful `reset` if it reached `terminated`, else undefined.
 * Contract: an executed primitive's result carries `details.terminated` (boolean); a failed call is
 * `isError` or has `details.error` / `details.result.error` and is dropped. Lines are
 * `{"action": name, ...args}`, plus successful `segment` calls as `{action, prompt | point, camera}`.
 */
export function recipe(
	entries: SessionEntry[],
	primitives: ReadonlySet<string>,
): Record<string, unknown>[] | undefined {
	const calls = toolCalls(entries);
	const ok = (c: Call) => !c.isError && !c.details?.error && !c.details?.result?.error;
	let start = 0;
	calls.forEach((c, i) => {
		if (c.name === "reset" && ok(c)) start = i + 1;
	});
	const episode = calls.slice(start);
	if (!episode.some((c) => c.details?.terminated === true)) return undefined;
	return episode
		.filter(
			(c) =>
				ok(c) && (c.name === "segment" || (primitives.has(c.name) && typeof c.details?.terminated === "boolean")),
		)
		.map((c) => {
			if (c.name !== "segment") return { action: c.name, ...c.args };
			const camera = c.details?.camera ?? c.args.camera ?? "agentview";
			const prompt = str(c.args.prompt).trim();
			return prompt ? { action: "segment", prompt, camera } : { action: "segment", point: c.args.point, camera };
		});
}

// ---------------------------------------------------------------------------
// Extension wiring
// ---------------------------------------------------------------------------

function say(ctx: ExtensionContext, text: string, level: "info" | "warning" | "error" = "info") {
	if (ctx.hasUI) ctx.ui.notify(text, level);
	else console.error(`[memory] ${text}`);
}

export function memory(pi: ExtensionAPI, opts: MemoryOptions = {}) {
	pi.registerFlag("memory-profile", { type: "string", description: "hf (evaluation default) | local" });
	pi.registerFlag("memory-dir", { type: "string", description: "Local memory root (local profile or exploration)" });
	pi.registerFlag("output-dir", { type: "string", description: "Audit and recipe directory (default: session dir)" });
	pi.registerFlag("auto-merge-memory", {
		type: "boolean",
		default: false,
		description: "Merge the inbox after exploration",
	});
	let root = "";
	let home = "";
	let outputDir = "";
	let profile: "hf" | "local" = "hf";
	let cell: { tag: string; reference: string } | undefined;
	let guard: Guard | undefined;

	function locate(ctx: ExtensionContext) {
		const dir = str(pi.getFlag("memory-dir"));
		const out = str(pi.getFlag("output-dir")) || ctx.sessionManager.getSessionDir();
		home = canonicalPath(opts.home?.() || defaultHome(), ctx.cwd);
		root = canonicalPath(dir || join(home, opts.robot ?? "libero"), ctx.cwd);
		outputDir = out ? canonicalPath(out, ctx.cwd) : "";
	}

	pi.on("session_start", async (_event, ctx) => {
		guard = undefined;
		locate(ctx);
		cell = opts.cell?.();
		if (!cell) return;
		if (!/^[\w.-]+$/.test(cell.tag) || /^\.+$/.test(cell.tag))
			throw new Error(`invalid memory cell tag: ${cell.tag}`);
		const explore = opts.explore?.() ?? false;
		const requested = str(pi.getFlag("memory-profile")) || (explore ? "local" : "hf");
		if (requested !== "hf" && requested !== "local") throw new Error(`unknown --memory-profile ${requested}`);
		if (explore && requested === "hf") throw new Error("--explore cannot be used with --memory-profile hf");
		profile = requested;
		if (profile === "hf" && pi.getFlag("memory-dir"))
			throw new Error("--memory-dir requires --memory-profile local or --explore");
		guard = { root, home, output: outputDir, tag: cell.tag, inbox: explore ? cell.tag : undefined };
		if (explore) return;
		if (profile === "hf") await syncMemory(root, (m) => say(ctx, m, "warning"));
		else if (
			!existsSync(join(root, "MEMORY.md")) &&
			!["global", "suite", "task_only"].some((s) => hasFiles(join(root, s)))
		)
			throw new Error(`local memory corpus not found at ${root}; run exploration first or use --memory-profile hf`);
	});

	// A robot's file tools fail closed: until session_start has established every root, nothing is reachable.
	pi.on("tool_call", (event, ctx) => {
		const access = ACCESS[event.toolName];
		if (!access || !opts.cell) return undefined;
		const g = guard;
		if (!g || ![g.root, g.home, g.output].every(isAbsolute) || !g.tag)
			return { block: true, reason: "file access is disabled: the memory guard has no memory, output dir or cell" };
		try {
			const readable = (opts.readable?.() ?? []).filter(Boolean).map((d) => canonicalPath(d, ctx.cwd));
			const reason = denied(
				canonicalPath(str((event.input as unknown as { path?: unknown }).path), ctx.cwd),
				access,
				{ ...g, readable },
			);
			return reason ? { block: true, reason } : undefined;
		} catch (e) {
			return { block: true, reason: `file access denied: ${(e as Error).message}` };
		}
	});

	// Once the run has settled (continuations such as DISTIL included): export the recipe, merge drafts.
	pi.on("agent_settled", async (_event, ctx) => {
		if (!cell || !outputDir) return;
		const branch = ctx.sessionManager.getBranch();
		const commands = recipe(branch, new Set(opts.primitives ?? []));
		const path = join(outputDir, `${cell.tag}_recipe.jsonl`);
		if (commands) {
			mkdirSync(outputDir, { recursive: true });
			writeFileSync(path, commands.map((c) => `${JSON.stringify(c)}\n`).join(""));
		}
		say(ctx, commands ? `recipe: ${path}` : "recipe: not written (cell unsolved)");
		let failed = false;
		for (const e of branch)
			if (e.type === "message" && e.message.role === "assistant") failed = e.message.stopReason === "error";
		if (guard?.inbox && pi.getFlag("auto-merge-memory") === true && !failed)
			say(ctx, `memory merged: ${JSON.stringify(await mergeMemory(root, cell.tag, outputDir, !!commands))}`);
	});

	const usage = "usage: /memory sync | validate | index | merge <cell> [output-dir] [--solved]";
	pi.registerCommand("memory", {
		description: "Memory corpus maintenance: sync | validate | index | merge <cell> [output-dir] [--solved]",
		handler: async (args, ctx) => {
			if (!root) locate(ctx);
			const [sub, ...rest] = args.trim().split(/\s+/);
			if (sub === "sync") {
				await syncMemory(root, (m) => say(ctx, m, "warning"));
				say(ctx, `memory: ${root}`);
			} else if (sub === "validate") {
				const problems = validateMemory(root);
				if (problems.length) process.exitCode = 1;
				say(
					ctx,
					problems.length ? problems.join("\n") : "local memory is valid",
					problems.length ? "error" : "info",
				);
			} else if (sub === "index") {
				say(ctx, rebuildIndex(root) ?? "no indexable memory leaves; MEMORY.md not regenerated");
			} else if (sub === "merge" && rest[0] && rest[0] !== "--solved") {
				const [tag, dir] = rest.filter((a) => a !== "--solved");
				const result = await mergeMemory(
					root,
					tag,
					dir ? canonicalPath(dir, ctx.cwd) : outputDir,
					rest.includes("--solved"),
				);
				say(ctx, JSON.stringify(result, null, 2));
			} else say(ctx, usage, "error");
		},
	});

	return {
		/** Built-in tools the agent uses on memory; add them to setActiveTools. */
		tools: ["read", "ls", "grep", "find", "write"],
		get profile() {
			return profile;
		},
		/** Fill {{memory_dir}}, {{memory_inbox}}, {{memory_profile}}, {{output_dir}}, {{recipe_tag}}, {{reference_tag}} and `extra`. */
		render(text: string, extra: Record<string, string | number> = {}): string {
			const vars: Record<string, string | number> = {
				memory_dir: root,
				memory_inbox: join(root, "_internal", "inbox", cell?.tag ?? ""),
				memory_profile: profile,
				output_dir: outputDir,
				recipe_tag: cell?.tag ?? "",
				reference_tag: cell?.reference ?? "",
				...extra,
			};
			return text.replace(/\{\{(\w+)\}\}/g, (m, k: string) => (k in vars ? String(vars[k]) : m));
		},
	};
}

export default function (pi: ExtensionAPI) {
	memory(pi);
}
