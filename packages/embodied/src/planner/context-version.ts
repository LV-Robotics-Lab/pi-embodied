/**
 * The context version record (OpenETA's openeta-for-codex episode metadata): what the planner's
 * context was built from, so two results can be told apart when a prompt, template or memory file
 * changed between them. ../robot.ts writes it into every `robot_result` (as `context_version`) and
 * as a `context_version` session entry; eval scripts copy it into result.json as data, never into
 * the configuration key.
 *
 *   system_prompt_sha256  SHA-256 of the system prompt pi sent (after toolSections and every
 *                         extension's rewrite), at the episode's first agent start; a later start
 *                         with another prompt (an exploration continuation) adds `system_prompts_sha256`
 *   templates             { "<path under src/>": sha256 } of every template (SYSTEM.md, explore.md,
 *                         distil.md, memory-*.md, the code/units prompts, closed-loop.md) the loaded
 *                         extensions read through `template`, less the ones this episode's mode does not use
 *   memory_files          { "<path under the memory corpus>": sha256 } of the memory files the agent read
 *   code_api_digest       the primitive registry's digest (../primitives/registry.ts), or null
 *   git_commit            `git rev-parse HEAD` of the package's checkout, "unknown" outside one
 *   git_dirty             whether that checkout has uncommitted changes to tracked files (null outside one)
 *   planner_models        the "provider/id" of every model that answered the planner, in order of first use
 */

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { basename, relative } from "node:path";
import { fileURLToPath } from "node:url";

/** Session entry with the record (see above). */
export const CONTEXT_VERSION_ENTRY = "context_version";
const SRC = fileURLToPath(new URL("..", import.meta.url));

export const sha256 = (s: string | Buffer) => createHash("sha256").update(s).digest("hex");

/** Every template read through `template`, by path under src/. */
const templates = new Map<string, string>();

/** Read a prompt template (a URL next to the module, as `new URL("../SYSTEM.md", import.meta.url)`) and register its digest. */
export function template(url: URL): string {
	const path = fileURLToPath(url);
	const text = readFileSync(path, "utf8");
	templates.set(relative(SRC, path).split("\\").join("/"), sha256(text));
	return text;
}

/** What the episode uses, for picking its templates out of the registry. */
export type TemplateUse = { explore: boolean; memoryProfile?: string; code: boolean; units: boolean };

/**
 * The registered templates this episode uses: explore.md and distil.md only when exploring, a
 * memory-<profile>.md only for that profile (memory.md with any), modes/code/SYSTEM.md and modes/units/SYSTEM.md
 * only in their mode; every other template (a robot's SYSTEM*.md, closed-loop.md) always.
 */
export function usedTemplates(use: TemplateUse): Record<string, string> {
	const out: Record<string, string> = {};
	for (const [path, digest] of [...templates].sort(([a], [b]) => a.localeCompare(b))) {
		const name = basename(path);
		if ((name === "explore.md" || name === "distil.md") && !use.explore) continue;
		const mem = /^memory(?:-(\w+))?\.md$/.exec(name);
		if (mem && (use.memoryProfile === undefined || (mem[1] !== undefined && mem[1] !== use.memoryProfile))) continue;
		if (path === "modes/code/SYSTEM.md" && !use.code) continue;
		if (path === "modes/units/SYSTEM.md" && !use.units) continue;
		out[path] = digest;
	}
	return out;
}

let commit: string | undefined;
/** `git rev-parse HEAD` of the checkout holding this package (resolved once), "unknown" when it is not one. */
export function gitCommit(): string {
	if (commit === undefined) {
		try {
			commit = execFileSync("git", ["-C", SRC, "rev-parse", "HEAD"], {
				encoding: "utf8",
				timeout: 5_000,
				stdio: ["ignore", "pipe", "ignore"],
			}).trim();
			if (!/^[0-9a-f]{40,64}$/.test(commit)) commit = "unknown";
		} catch {
			commit = "unknown";
		}
	}
	return commit;
}

let dirty: boolean | null | undefined;
/** Whether the checkout holding this package has uncommitted changes to tracked files (resolved once); null outside one. */
export function gitDirty(): boolean | null {
	if (dirty === undefined) {
		try {
			const out = execFileSync("git", ["-C", SRC, "status", "--porcelain", "--untracked-files=no"], {
				encoding: "utf8",
				timeout: 5_000,
				stdio: ["ignore", "pipe", "ignore"],
			});
			dirty = out.trim().length > 0;
		} catch {
			dirty = null;
		}
	}
	return dirty;
}
