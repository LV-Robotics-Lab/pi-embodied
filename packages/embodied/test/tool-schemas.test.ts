import assert from "node:assert/strict";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import behavior from "../src/behavior/index.ts";
import dualFranka, { DUAL_FRANKA_UNITS } from "../src/dual_franka/index.ts";
import franka from "../src/franka/index.ts";
import libero from "../src/libero/index.ts";
import maniskill from "../src/maniskill/index.ts";
import piperDual from "../src/piper/dual.ts";
import piper from "../src/piper/index.ts";
import robocasa from "../src/robocasa/index.ts";
import robolab from "../src/robolab/index.ts";
import robotwin from "../src/robotwin/index.ts";
import { units } from "../src/units/index.ts";

/**
 * Snapshot of what every robot registers with pi at default flags: each tool's name, description
 * and JSON parameter schema, in registration order. This is the model-facing contract; a refactor
 * that moves a tool into ../src/primitives must leave it byte-identical. To accept an intended
 * change: UPDATE_TOOL_SCHEMAS=1 node --test --experimental-strip-types test/tool-schemas.test.ts
 */
const FIXTURE = fileURLToPath(new URL("./fixtures/tool-schemas.json", import.meta.url));
const ROBOTS: Record<string, (pi: ExtensionAPI) => unknown> = {
	behavior,
	dual_franka: dualFranka,
	franka,
	libero,
	maniskill,
	piper,
	piper_dual: piperDual,
	robocasa,
	robolab,
	robotwin,
	// Dual Franka whose config streams an inline wrist camera (the default D455 alone has none, so the
	// load-time `dual_franka` entry shows act without the wrist plugins): its units as mounted then.
	dual_franka_wrist_camera: (pi) =>
		units(
			pi,
			{ ...DUAL_FRANKA_UNITS, wrist: true, apply: async () => ({ content: [], details: {} }) },
			(name, description, parameters) =>
				pi.registerTool({
					name,
					description,
					parameters,
					execute: async () => ({ content: [], details: {} }),
				} as never),
		),
};
type Tool = { name: string; description: string; parameters: unknown };

/** Load a robot into a stub pi (flags at their defaults) and return its tools as pi would serialize them. */
function registered(load: (pi: ExtensionAPI) => unknown): Tool[] {
	const flags: Record<string, unknown> = {};
	const tools: Tool[] = [];
	const pi = {
		on: () => {},
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: Tool) => tools.push({ name: t.name, description: t.description, parameters: t.parameters }),
		registerCommand: () => {},
		registerProvider: () => {},
		registerMessageRenderer: () => {},
		registerShortcut: () => {},
		setActiveTools: () => {},
		getActiveTools: () => [],
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	load(pi);
	// The JSON round trip drops typebox's symbol keys, as pi's request to the model does.
	return JSON.parse(JSON.stringify(tools));
}

if (process.env.UPDATE_TOOL_SCHEMAS) {
	test("update the tool schema snapshot", () => {
		const robots = Object.fromEntries(Object.entries(ROBOTS).map(([name, load]) => [name, registered(load)]));
		mkdirSync(new URL("./fixtures/", import.meta.url), { recursive: true });
		writeFileSync(FIXTURE, `${JSON.stringify({ robots }, null, "\t")}\n`);
	});
} else {
	const fixture = JSON.parse(readFileSync(FIXTURE, "utf8")) as { robots: Record<string, Tool[]> };
	test("the snapshot covers every robot", () => {
		assert.deepEqual(Object.keys(fixture.robots).sort(), Object.keys(ROBOTS).sort());
	});
	for (const [name, load] of Object.entries(ROBOTS))
		test(`${name}: tool names, descriptions and schemas are unchanged`, () => {
			const live = registered(load);
			assert.deepEqual(
				live.map((t) => t.name),
				fixture.robots[name].map((t) => t.name),
				`${name} registers other tools than the snapshot; UPDATE_TOOL_SCHEMAS=1 accepts an intended change`,
			);
			for (const [i, tool] of live.entries())
				assert.deepEqual(
					tool,
					fixture.robots[name][i],
					`${name}.${tool.name} differs from the snapshot; UPDATE_TOOL_SCHEMAS=1 accepts an intended change`,
				);
		});
}

if (!process.env.UPDATE_TOOL_SCHEMAS)
	test("dual_franka_wrist_camera is dual Franka's act with a wrist view: the same act plus target_in_wrist and plan", () => {
		const act = (name: string) => registered(ROBOTS[name]).find((t) => t.name === "act") as any;
		const plain = act("dual_franka");
		const wrist = act("dual_franka_wrist_camera");
		assert.equal(wrist.description, plain.description);
		const { target_in_wrist, plan, ...rest } = wrist.parameters.properties;
		assert.ok(target_in_wrist && plan);
		assert.deepEqual(rest, plain.parameters.properties);
	});
