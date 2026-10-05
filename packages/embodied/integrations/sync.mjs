#!/usr/bin/env node
/**
 * One source for what the Codex and Claude Code plugins share: `shared/skills/` and `shared/scripts/`
 * are copied into `claude-code/` and `codex/` (the hosts' installers copy a plugin directory and do
 * not follow symlinks, so each plugin carries its own copy), and `codex/.mcp.json` is generated from
 * the primitive manifests: every env tool that moves the robot (`mutating`) and the server's
 * built-in motion `reset` (../src/integrations/mcp/tools.ts BUILTIN_MOTIONS) get Codex's
 * `approval_mode: "prompt"`, the rest `approve`. Run after editing anything under `shared/` or a
 * manifest; `--check` reports drift and exits 1 (test/integrations.test.ts runs it).
 *
 *   node packages/embodied/integrations/sync.mjs [--check]
 */

import { chmodSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const shared = join(here, "shared");
const hosts = ["claude-code", "codex"];
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

/** The manifests' env tools that move the robot, by name (common entries resolved by the TS loader; here the raw files suffice: `use` entries name the shared entry). */
function motionTools() {
	const names = new Set();
	const common = new Map();
	for (const f of readdirSync(join(manifests, "common")))
		for (const e of JSON.parse(readFileSync(join(manifests, "common", f), "utf8")).primitives)
			common.set(`${f.slice(0, -5)}/${e.name}`, e);
	for (const f of readdirSync(manifests)) {
		if (!f.endsWith(".json")) continue;
		for (const raw of JSON.parse(readFileSync(join(manifests, f), "utf8")).primitives) {
			const e = raw.use ? { ...common.get(raw.use), ...raw } : raw;
			if (e.side === "env" && e.doc?.tool && !e.module && e.mutating) names.add(e.name);
		}
	}
	return [...names].sort();
}

/** Codex's compatibility .mcp.json: the launcher, the environment it may read, per-tool approval. */
export function codexMcpConfig() {
	return {
		mcpServers: {
			"pi-embodied": {
				command: "./scripts/pi-embodied-mcp.sh",
				args: [],
				cwd: ".",
				env_vars: [
					"PATH",
					"HOME",
					"PI_EMBODIED_ROOT",
					"PI_EMBODIED_ROBOT",
					"PI_EMBODIED_ENV_URL",
					"PI_EMBODIED_SERVE_ARGS",
					"PI_EMBODIED_DEPLOYMENT",
					"PI_EMBODIED_TIER",
					"PI_EMBODIED_PRIVILEGED",
					"PI_EMBODIED_CAPABILITIES",
					"PI_EMBODIED_VARS",
					"PI_EMBODIED_TIMEOUT_MS",
					"PI_EMBODIED_NO_RESET",
					"PI_EMBODIED_CONFIG",
					"PI_EMBODIED_SERVICES",
					"PI_EMBODIED_PYTHON",
					"PI_CODING_AGENT_DIR",
				],
				startup_timeout_sec: 600,
				default_tools_approval_mode: "approve",
				tools: Object.fromEntries(
					[...motionTools(), "reset", "stop", "resume"].map((name) => [name, { approval_mode: "prompt" }]),
				),
			},
		},
	};
}

for (const host of hosts) {
	const wanted = new Map();
	for (const rel of files(join(shared, "skills"))) wanted.set(join("skills", rel), readFileSync(join(shared, "skills", rel)));
	for (const rel of files(join(shared, "scripts"))) wanted.set(join("scripts", rel), readFileSync(join(shared, "scripts", rel)));
	sync(join(here, host, "skills"), new Map([...wanted].filter(([k]) => k.startsWith("skills/")).map(([k, v]) => [relative("skills", k), v])), `${host}/skills`);
	sync(join(here, host, "scripts"), new Map([...wanted].filter(([k]) => k.startsWith("scripts/")).map(([k, v]) => [relative("scripts", k), v])), `${host}/scripts`);
}
sync(join(here, "codex"), new Map([[".mcp.json", Buffer.from(`${JSON.stringify(codexMcpConfig(), null, "\t")}\n`)]]), "codex", false);

if (drift.length) {
	for (const d of drift) console.error(`${check ? "drift" : "synced"}: ${d}`);
	if (check) process.exit(1);
}
