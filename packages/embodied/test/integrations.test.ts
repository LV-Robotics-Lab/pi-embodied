/**
 * The Claude Code plugin (../integrations): one shared source for skills and scripts, copies that do
 * not drift (sync.mjs --check), manifests that parse and point at files that exist, and a hook that
 * gates exactly our MCP server's tools.
 */

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { DEFAULT_SERVER } from "../src/integrations/mcp/hook.ts";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const INT = fileURLToPath(new URL("../integrations/", import.meta.url));
const HOSTS = ["claude-code"] as const;
/** The host's plugin-root variable as the plugin files spell it (a string, not a template). */
const PLUGIN_ROOT = ["$", "{CLAUDE_PLUGIN_ROOT}"].join("");
const json = (path: string) => JSON.parse(readFileSync(path, "utf8"));
const frontmatter = (path: string) => {
	const m = /^---\n([\s\S]*?)\n---\n/.exec(readFileSync(path, "utf8"));
	assert.ok(m, `${path} has frontmatter`);
	return Object.fromEntries(
		m[1].split("\n").map((l) => [l.slice(0, l.indexOf(":")).trim(), l.slice(l.indexOf(":") + 1).trim()] as const),
	);
};

test("the plugin copies of skills and scripts match the shared source (sync.mjs --check)", () => {
	execFileSync(process.execPath, [`${INT}sync.mjs`, "--check"], { stdio: ["ignore", "pipe", "pipe"] });
	const skills = readdirSync(`${INT}shared/skills`).sort();
	assert.deepEqual(skills, ["robot-observe-act", "robot-results-evidence", "robot-safety-approval"]);
	for (const host of HOSTS) {
		assert.deepEqual(readdirSync(`${INT}${host}/skills`).sort(), skills, host);
		for (const s of skills) {
			const fm = frontmatter(`${INT}${host}/skills/${s}/SKILL.md`);
			assert.equal(fm.name, s, `${host}/${s}: frontmatter name is the directory`);
			assert.ok(fm.description && fm.description.length > 40, `${host}/${s}: a description`);
		}
		for (const script of ["pi-embodied-mcp.sh", "approve-motion.sh"]) {
			const path = `${INT}${host}/scripts/${script}`;
			assert.ok(existsSync(path), path);
			assert.ok(statSync(path).mode & 0o111, `${path} is executable`);
			assert.match(readFileSync(path, "utf8"), /src\/integrations\/mcp\/(server|hook)\.ts/);
		}
	}
});

test("Claude Code plugin: manifest, MCP server, hook and command agree on the server name", () => {
	const dir = `${INT}claude-code/`;
	const manifest = json(`${dir}.claude-plugin/plugin.json`);
	assert.equal(manifest.name, "pi-embodied");
	assert.ok(manifest.version && manifest.description && manifest.author?.name);
	assert.equal(manifest.userConfig.repo.type, "directory");
	assert.equal(manifest.userConfig.robot.required, true);
	const mcp = json(`${dir}.mcp.json`);
	const server = mcp.mcpServers[DEFAULT_SERVER];
	assert.ok(server, `server key ${DEFAULT_SERVER}`);
	assert.ok(server.args.some((a: string) => a.includes(`${PLUGIN_ROOT}/scripts/pi-embodied-mcp.sh`)));
	for (const key of Object.keys(manifest.userConfig))
		assert.ok(Object.values(server.env).includes(`\${user_config.${key}}`), `option ${key} reaches the server`);
	const hooks = json(`${dir}hooks/hooks.json`).hooks.PreToolUse;
	assert.equal(hooks.length, 1);
	assert.equal(hooks[0].matcher, `mcp__${DEFAULT_SERVER}__.*`);
	const hook = hooks[0].hooks[0];
	assert.equal(hook.type, "command");
	assert.deepEqual(hook.args.slice(1), ["--decision", "ask"]);
	assert.ok(hook.args[0].includes(`${PLUGIN_ROOT}/scripts/approve-motion.sh`));
	const cmd = frontmatter(`${dir}commands/robot-status.md`);
	assert.ok(cmd.description);
	assert.match(cmd["allowed-tools"] ?? "", new RegExp(`mcp__${DEFAULT_SERVER}__robot_status`));
	assert.ok(!existsSync(`${dir}CLAUDE.md`), "no CLAUDE.md at the plugin root (validate warns)");
});

test("the marketplace at the repository root points at the plugin directory", () => {
	const claude = json(`${ROOT}.claude-plugin/marketplace.json`);
	assert.equal(claude.name, "pi-embodied");
	assert.ok(claude.owner?.name && claude.description);
	assert.equal(claude.plugins.length, 1);
	assert.equal(claude.plugins[0].name, "pi-embodied");
	assert.match(claude.plugins[0].source, /^\.\//);
	assert.ok(existsSync(`${ROOT}${claude.plugins[0].source}/.claude-plugin/plugin.json`));
	assert.equal(claude.plugins[0].version, json(`${INT}claude-code/.claude-plugin/plugin.json`).version);
});
