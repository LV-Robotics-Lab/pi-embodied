/**
 * Every primitive manifest yields a valid MCP tool list in every tier (../src/integrations/mcp/tools.ts):
 * unique names, object schemas, descriptions, only env entries without a module, privileged variants
 * only under the flag, and no collision with the built-in tools. With the robots' list variables
 * given nothing is left out; without them only the enums that need one are. Every served tool's
 * adapted call (../src/primitives/arguments.ts) matches the method's signature the manifest declares
 * in code mode, which the env server pins to the facade at its start (components/manifest.py).
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
import { pointArgs, rpcArguments, rpcParams } from "../src/primitives/arguments.ts";
import { loadManifest, type ManifestParam } from "../src/primitives/manifest.ts";
import { codePrimitives } from "../src/primitives/registry.ts";

const DIR = new URL("../src/primitives/manifests/", import.meta.url);
const ROBOTS = readdirSync(DIR)
	.filter((f) => f.endsWith(".json"))
	.map((f) => f.slice(0, -5))
	.sort();
const BUILTINS = ["observe", "reset", "finish", "stop", "resume", "robot_status"];
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

type JsonSchema = {
	type?: string;
	enum?: string[];
	items?: JsonSchema;
	minItems?: number;
	properties?: Record<string, JsonSchema>;
	required?: string[];
};

/** A value that satisfies `s` (the first enum value, small numbers, minItems items). */
function sample(s: JsonSchema): unknown {
	if (s.enum) return s.enum[0];
	switch (s.type) {
		case "number":
			return 0.01;
		case "integer":
			return 1;
		case "boolean":
			return false;
		case "string":
			return "x";
		case "array":
			return Array.from({ length: s.minItems ?? 1 }, () => sample(s.items ?? { type: "number" }));
		case "object":
			return Object.fromEntries(Object.entries(s.properties ?? {}).map(([k, p]) => [k, sample(p)]));
		default:
			throw new Error(`no sample for ${JSON.stringify(s)}`);
	}
}

const inCode = (p: ManifestParam) => !p.modes || p.modes.includes("code");

test("every served tool's adapted call matches the method's declared signature (code-mode params)", () => {
	let adapted = 0;
	for (const robot of ROBOTS) {
		const m = loadManifest(robot);
		for (const privileged of [false, true]) {
			const { tools } = manifestTools(m, { privileged }, everything, VARS);
			for (const { tool, entry } of tools) {
				const where = `${robot} ${tool.name}`;
				const params = entry.params ?? {};
				const method = Object.entries(params).filter(([, p]) => inCode(p));
				const schema = tool.inputSchema as JsonSchema;
				// The schema offers no parameter the method does not take, unless it maps onto ones it does.
				for (const k of Object.keys(schema.properties ?? {}))
					assert.ok(
						inCode(params[k]) || (k === "point" && inCode(params.row) && inCode(params.col)),
						`${where}: ${k} is offered but the method does not take it`,
					);
				// Every parameter given: the kwargs are exactly method parameters, and the required ones are there.
				const args = sample(schema) as Record<string, unknown>;
				const kwargs = rpcArguments(entry, args);
				for (const k of Object.keys(kwargs))
					assert.ok(
						method.some(([name]) => name === k),
						`${where}: kwarg ${k} is not a parameter of ${entry.method} (${method.map(([n]) => n).join(", ")})`,
					);
				for (const [k, p] of method)
					if (p.required)
						assert.ok(k in kwargs, `${where}: ${entry.method} requires ${k}, the adapted call lacks it`);
				// Only the required ones given: the same, with nothing extra.
				const least = Object.fromEntries(
					Object.entries(schema.properties ?? {})
						.filter(([k]) => schema.required?.includes(k))
						.map(([k, p]) => [k, sample(p)]),
				);
				for (const [k, p] of method)
					if (p.required)
						assert.ok(k in rpcArguments(entry, least), `${where}: ${k} from the required tool params`);
				if (Object.keys(schema.properties ?? {}).some((k) => !inCode(params[k]))) adapted++;
			}
		}
	}
	assert.ok(adapted >= 2, `align_wrist's point is adapted on libero and franka (${adapted})`);
});

test("rpcParams and rpcArguments: point maps onto row and col; pi-side state is neither offered nor accepted", () => {
	const libero = loadManifest("libero");
	const align = libero.primitives.find((e) => e.name === "align_wrist");
	assert.ok(align);
	assert.deepEqual(Object.keys(rpcParams(align)), ["point", "max_correction_m", "execute"]);
	assert.deepEqual(rpcArguments(align, { point: [10, 20], max_correction_m: 0.02, execute: undefined }), {
		row: 10,
		col: 20,
		max_correction_m: 0.02,
	});
	assert.deepEqual(pointArgs([3.4, 7.6]), { row: 3, col: 8 });
	assert.throws(() => pointArgs([1]), /\[row, col\]/);
	const segment = libero.primitives.find((e) => e.name === "segment");
	assert.ok(segment);
	assert.ok(!("step" in rpcParams(segment)), "step is pi's recorded state");
	assert.ok("point" in rpcParams(segment), "segment's point is the method's own");
	assert.deepEqual(rpcArguments(segment, { prompt: "bowl", point: [1, 2] }), { prompt: "bowl", point: [1, 2] });
	assert.throws(() => rpcArguments(segment, { prompt: "bowl", step: 0 }), /step is pi's/);
	const back = libero.primitives.find((e) => e.name === "back_project");
	assert.ok(back);
	assert.ok(!("resolution" in rpcParams(back)) && !("step" in rpcParams(back)));
	assert.deepEqual(rpcArguments(back, { row: 1, col: 2, camera: "wrist" }), { row: 1, col: 2, camera: "wrist" });
});

test("the real robots in REAL_ROBOTS have manifests", () => {
	for (const r of REAL_ROBOTS) assert.ok(ROBOTS.includes(r), r);
});
