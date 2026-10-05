#!/usr/bin/env node
/**
 * One source for what the host plugins share: `shared/skills/` and `shared/scripts/` are copied into
 * `claude-code/` (a host's installer copies a plugin directory and does not follow symlinks, so the
 * plugin carries its own copy). Run after editing anything under `shared/`; `--check` reports drift
 * and exits 1 (test/integrations.test.ts runs it).
 *
 *   node packages/embodied/integrations/sync.mjs [--check]
 */

import { chmodSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const shared = join(here, "shared");
const hosts = ["claude-code"];
const manifests = join(here, "..", "src", "primitives", "manifests");
const check = process.argv.includes("--check");
const drift = [];

/** Every file under `dir`, relative to it. */
function files(dir) {
	const out = [];
	const walk = (d) => {
		for (const e of readdirSync(d, { withFileTypes: true })) {
			const p = join(d, e.name);
			if (e.isDirectory()) walk(p);
			else out.push(relative(dir, p));
		}
	};
	if (existsSync(dir)) walk(dir);
	return out.sort();
}

/** Make `target` hold `wanted` (path -> contents), and with `prune` nothing else; scripts are executable. */
function sync(target, wanted, label, prune = true) {
	const have = new Set(prune ? files(target) : []);
	for (const [rel, body] of wanted) {
		const path = join(target, rel);
		const same = existsSync(path) && readFileSync(path).equals(body);
		const exec = rel.endsWith(".sh");
		const mode = existsSync(path) ? statSync(path).mode & 0o111 : 0;
		if (same && (!exec || mode)) {
			have.delete(rel);
			continue;
		}
		drift.push(`${label}/${rel} ${existsSync(path) ? "differs from" : "missing from"} its source`);
		if (!check) {
			mkdirSync(dirname(path), { recursive: true });
			writeFileSync(path, body);
			if (exec) chmodSync(path, 0o755);
		}
		have.delete(rel);
	}
	for (const rel of have) {
		drift.push(`${label}/${rel} has no source`);
		if (!check) rmSync(join(target, rel));
	}
}

for (const host of hosts) {
	const wanted = new Map();
	for (const rel of files(join(shared, "skills"))) wanted.set(join("skills", rel), readFileSync(join(shared, "skills", rel)));
	for (const rel of files(join(shared, "scripts"))) wanted.set(join("scripts", rel), readFileSync(join(shared, "scripts", rel)));
	sync(join(here, host, "skills"), new Map([...wanted].filter(([k]) => k.startsWith("skills/")).map(([k, v]) => [relative("skills", k), v])), `${host}/skills`);
	sync(join(here, host, "scripts"), new Map([...wanted].filter(([k]) => k.startsWith("scripts/")).map(([k, v]) => [relative("scripts", k), v])), `${host}/scripts`);
}

if (drift.length) {
	for (const d of drift) console.error(`${check ? "drift" : "synced"}: ${d}`);
	if (check) process.exit(1);
}
