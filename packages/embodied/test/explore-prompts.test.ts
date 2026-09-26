import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { test } from "node:test";

/** What memory.ts `render` and explore.ts fill in every exploration prompt. */
const FILLED = new Set([
	"attempt_budget",
	"memory_dir",
	"memory_inbox",
	"memory_profile",
	"output_dir",
	"recipe_tag",
	"reference_tag",
	"session_max",
	"session_number",
]);
const src = new URL("../src/", import.meta.url);

test("every robot fills the placeholders of its exploration prompt that the shared modules do not", () => {
	const robots = readdirSync(src).filter((d) => {
		try {
			return readdirSync(new URL(`${d}/`, src)).includes("explore.md");
		} catch {
			return false;
		}
	});
	assert.ok(robots.length >= 8, robots.join(", "));
	for (const robot of robots) {
		const prompt = readFileSync(new URL(`${robot}/explore.md`, src), "utf8");
		const code = readFileSync(new URL(`${robot}/index.ts`, src), "utf8");
		// A robot replaces `{{name}}` literally, or names it in a /\{\{(a|b)\}\}/ alternation.
		const alternations = [...code.matchAll(/\\\{\\\{\(([\w|]+)\)\\\}\\\}/g)].flatMap((m) => m[1].split("|"));
		for (const [, name] of prompt.matchAll(/\{\{(\w+)\}\}/g)) {
			if (FILLED.has(name)) continue;
			assert.ok(
				code.includes(`"{{${name}}}"`) || alternations.includes(name),
				`${robot}/explore.md: {{${name}}} is never filled`,
			);
		}
	}
});
