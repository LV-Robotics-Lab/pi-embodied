/**
 * The README's robot table lists, per robot, the shared modules it mounts (the column's text before
 * the first `;`). A mounted module registers its flags, so each robot is loaded in a stub pi and the
 * modules its flags show must be exactly the row's: the table cannot drift from the code again.
 */

import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

/** Each README module name and the flag its module registers. */
const MODULES: Record<string, string> = {
	memory: "memory-profile",
	explore: "explore",
	video: "video-dir",
	units: "units",
	VDM: "vdm",
	code: "code",
	"`--privileged`": "privileged",
	flywheel: "collect-flywheel-data",
	operator: "operator",
	Flash: "flash-plans",
	XPolicyLab: "xpolicy",
};

const ROBOTS = new URL("../src/robots/", import.meta.url);

async function flags(robot: string): Promise<Set<string>> {
	const mod = await import(new URL(`${robot}/index.ts`, ROBOTS).href);
	const values: Record<string, unknown> = {};
	const api: Record<string, unknown> = {
		registerFlag: (name: string, o: { default?: unknown }) => {
			values[name] = o.default;
		},
		getFlag: (name: string) => values[name],
		getActiveTools: () => [],
		events: { emit: () => {}, on: () => () => {} },
	};
	mod.default(new Proxy(api, { get: (t, k: string) => t[k] ?? (() => {}) }) as unknown as ExtensionAPI);
	return new Set(Object.keys(values));
}

test("the README's shared-modules column is what each robot mounts", async () => {
	const readme = readFileSync(new URL("../README.md", import.meta.url), "utf8");
	const rows = new Map<string, string[]>();
	for (const m of readme.matchAll(/^\| .+? \| `src\/robots\/(\w+)` \| .+ \| (.+) \|$/gm))
		rows.set(
			m[1],
			m[2]
				.split(";")[0]
				.split(",")
				.map((s) => s.trim()),
		);
	const robots = readdirSync(ROBOTS).filter((r) => existsSync(new URL(`${r}/index.ts`, ROBOTS)));
	assert.deepEqual([...rows.keys()].sort(), robots.sort(), "every robot has a row, and only robots");
	for (const robot of robots) {
		const listed = rows.get(robot) ?? [];
		const unknown = listed.filter((m) => !(m in MODULES));
		assert.deepEqual(unknown, [], `${robot}: unknown module names in the README row`);
		const registered = await flags(robot);
		const mounted = Object.keys(MODULES).filter((m) => registered.has(MODULES[m]));
		assert.deepEqual(listed, mounted, `${robot}: README row vs mounted modules`);
	}
});
