/**
 * Every primitive manifest yields a valid MCP tool list in every tier (../src/integrations/mcp/tools.ts):
 * unique names, object schemas, descriptions, only env entries without a module, privileged variants
 * only under the flag, and no collision with the built-in tools. With the robots' list variables
 * given nothing is left out; without them only the enums that need one are.
 */

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import { test } from "node:test";
import {
	capabilitiesFrom,
	MCP_TIERS,
	manifestTools,
	REAL_ROBOTS,
	selectEntries,
} from "../src/integrations/mcp/tools.ts";
import { loadManifest } from "../src/primitives/manifest.ts";
import { codePrimitives } from "../src/primitives/registry.ts";

const DIR = new URL("../src/primitives/manifests/", import.meta.url);
const ROBOTS = readdirSync(DIR)
	.filter((f) => f.endsWith(".json"))
	.map((f) => f.slice(0, -5))
	.sort();
const BUILTINS = ["observe", "finish", "stop", "resume", "robot_status"];
const VARS = {
	arms: ["left", "right"],
	cameras: ["agentview", "wrist"],
	max_move: "0.05",
	max_rotate: "0.5",
	max_yaw: "1.57",
	image_size: "256",
	state_fields: "eef_pos",
	move_subject: "the gripper",
	move_frame: "world",
	the: "the",
};
const everything = () => true;

test("every manifest: a valid tool list per tier, privileged on and off", () => {
	assert.ok(ROBOTS.length >= 10);
	for (const robot of ROBOTS) {
		const m = loadManifest(robot);
		for (const tier of [undefined, ...MCP_TIERS])
			for (const privileged of [false, true]) {
				const where = `${robot} tier=${tier ?? "all"} privileged=${privileged}`;
				const { tools, leftOut } = manifestTools(m, { tier, privileged }, everything, VARS);
				assert.deepEqual(leftOut, [], `${where}: nothing left out with the variables given`);
				const names = tools.map((t) => t.tool.name);
				assert.equal(new Set(names).size, names.length, `${where}: unique names`);
				for (const { tool, entry } of tools) {
					assert.match(tool.name, /^[A-Za-z_]\w*$/, where);
					assert.ok(!BUILTINS.includes(tool.name), `${where}: ${tool.name} collides with a built-in`);
					assert.ok(tool.description && tool.description.length > 0, `${where}: ${tool.name} has a description`);
					assert.equal(tool.inputSchema.type, "object", `${where}: ${tool.name} schema is an object`);
					assert.equal(typeof tool.inputSchema.properties, "object", `${where}: ${tool.name} has properties`);
					assert.equal(entry.side, "env", `${where}: ${tool.name} is an env entry`);
					assert.equal(entry.module, undefined, `${where}: ${tool.name} is not module-owned`);
					assert.equal(typeof entry.method, "string", `${where}: ${tool.name} has a method`);
					if (tier)
						assert.ok(
							entry.tier === tier || (privileged && entry.tier === "privileged"),
							`${where}: ${tool.name} is tier ${entry.tier}`,
						);
					if (!privileged)
						assert.notEqual(entry.tier, "privileged", `${where}: ${tool.name} is privileged without the flag`);
					assert.equal(
						tool.annotations?.destructiveHint,
						Boolean(entry.mutating),
						`${where}: ${tool.name} destructiveHint follows mutating`,
					);
				}
				// The tier filter is a subset of the unfiltered list.
				if (tier) {
					const all = manifestTools(m, { tier: undefined, privileged }, everything, VARS).tools.map(
						(t) => t.tool.name,
					);
					for (const n of names) assert.ok(all.includes(n), `${where}: ${n} is also served without a tier`);
				}
			}
		// Without the list variables, exactly the entries whose enum names one are left out, and named.
		const bare = manifestTools(m, { privileged: false }, everything);
		for (const l of bare.leftOut)
			assert.match(l.reason, /no list variable/, `${robot}: ${l.name} left out for ${l.reason}`);
		const needs = selectEntries(m, { privileged: false }).filter((e) =>
			Object.values(e.params ?? {}).some((p) => p.type === "enum" && typeof p.values === "string"),
		);
		assert.deepEqual(
			bare.leftOut.map((l) => l.name).sort(),
			needs.map((e) => e.name).sort(),
			`${robot}: left out = enum variables`,
		);
	}
});

test("privileged variants replace the plain entry of the same name only under the flag", () => {
	for (const robot of ROBOTS) {
		const m = loadManifest(robot);
		const priv = selectEntries(m, { privileged: true });
		const plain = selectEntries(m, { privileged: false });
		for (const e of plain) assert.notEqual(e.tier, "privileged");
		for (const e of m.primitives.filter(
			(p) => p.side === "env" && p.doc.tool && !p.module && p.tier === "privileged",
		))
			assert.equal(
				priv.find((p) => p.name === e.name)?.method,
				e.method,
				`${robot}: ${e.name}'s privileged method under the flag`,
			);
	}
});

test("requires: met through code.api's available list, --capabilities or --privileged; otherwise left out", () => {
	for (const robot of ROBOTS) {
		const m = loadManifest(robot);
		const sam = codePrimitives(m, undefined, (c) => c === "sam3").map((p) => p.name);
		const has = capabilitiesFrom(m, sam, [], false);
		const requiresSam = m.primitives.some((e) => e.side !== "ts" && e.doc.code && e.requires?.includes("sam3"));
		assert.equal(has("sam3"), requiresSam, `${robot}: sam3 inferred iff a code primitive requires it`);
		assert.equal(has("grasp"), false, robot);
		assert.equal(has("privileged"), false, robot);
		assert.equal(capabilitiesFrom(m, undefined, ["grasp"], true)("grasp"), true);
		assert.equal(capabilitiesFrom(m, undefined, [], true)("privileged"), true);
		const none = manifestTools(m, { privileged: false }, capabilitiesFrom(m, undefined, [], false), VARS);
		for (const { entry } of none.tools)
			assert.deepEqual(entry.requires ?? [], [], `${robot}: ${entry.name} served without its requirements`);
	}
});

test("the real robots in REAL_ROBOTS have manifests", () => {
	for (const r of REAL_ROBOTS) assert.ok(ROBOTS.includes(r), r);
});
