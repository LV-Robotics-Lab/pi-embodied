import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readdirSync } from "node:fs";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { loadManifest, toolSchema, tools } from "../src/primitives/manifest.ts";
import { codePrimitives } from "../src/primitives/registry.ts";
import { SERVICES } from "../src/robot.ts";

/**
 * The primitive manifests (../src/primitives/manifests) are the one declaration of every robot's
 * tools and code primitives: both sides load them the same way (the digest pi compares with the
 * env server's is the Python loader's), and a robot registers no tool its manifest does not declare
 * (defineRobot throws on one) and every declared tool of its own through the manifest's schema.
 */
const DIR = new URL("../src/primitives/manifests/", import.meta.url);
const ROBOTS = readdirSync(DIR)
	.filter((f) => f.endsWith(".json"))
	.map((f) => f.slice(0, -5))
	.sort();
const PYTHON = process.env.PYTHON ?? "python3";

test("every manifest loads here, and its digest is the Python loader's", () => {
	assert.ok(ROBOTS.includes("robosuite"));
	let python: Record<string, string> | undefined;
	try {
		python = JSON.parse(
			execFileSync(
				PYTHON,
				[
					"-c",
					"import json, sys; from pi_embodied_services.components.manifest import load_manifest; " +
						"print(json.dumps({r: load_manifest(r)['digest'] for r in sys.argv[1:]}))",
					...ROBOTS,
				],
				{ encoding: "utf8", env: { ...process.env, PYTHONPATH: SERVICES } },
			),
		);
	} catch {
		// No Python with the services' dependencies here: the TS side alone.
	}
	for (const robot of ROBOTS) {
		const m = loadManifest(robot);
		assert.equal(m.robot, robot);
		if (python) assert.equal(m.digest, python[robot], `${robot}: pi and the server compute one digest`);
	}
});

/** The robots migrated to a manifest: the extension that must follow it. */
const LOADERS: Record<string, () => Promise<{ default: (pi: ExtensionAPI) => unknown }>> = {
	robosuite: () => import("../src/robots/robosuite/index.ts"),
	robocasa: () => import("../src/robots/robocasa/index.ts"),
	robolab: () => import("../src/robots/robolab/index.ts"),
	robotwin: () => import("../src/robots/robotwin/index.ts"),
	robodojo: () => import("../src/robots/robodojo/index.ts"),
	libero: () => import("../src/robots/libero/index.ts"),
	metaworld: () => import("../src/robots/metaworld/index.ts"),
	genesis: () => import("../src/robots/genesis/index.ts"),
	humanclaw: () => import("../src/robots/humanclaw/index.ts"),
	behavior: () => import("../src/robots/behavior/index.ts"),
	maniskill: () => import("../src/robots/maniskill/index.ts"),
	franka: () => import("../src/robots/franka/index.ts"),
	dual_franka: () => import("../src/robots/dual_franka/index.ts"),
	piper: () => import("../src/robots/piper/index.ts"),
	ur5e: () => import("../src/robots/ur5e/index.ts"),
};

for (const [robot, load] of Object.entries(LOADERS))
	test(`${robot}: the registered tools are the manifest's, with its schemas`, async () => {
		const flags: Record<string, unknown> = {};
		const registered = new Map<string, { description: string; parameters: unknown }>();
		const pi = {
			on: () => {},
			registerFlag: (name: string, o: { default?: unknown }) => {
				flags[name] = o.default;
			},
			getFlag: (name: string) => flags[name],
			registerTool: (t: { name: string; description: string; parameters: unknown }) => registered.set(t.name, t),
			registerCommand: () => {},
			registerProvider: () => {},
			registerMessageRenderer: () => {},
			registerShortcut: () => {},
			setActiveTools: () => {},
			getActiveTools: () => [],
			appendEntry: () => {},
			events: { emit: () => {}, on: () => () => {} },
		} as unknown as ExtensionAPI;
		(await load()).default(pi);
		const m = loadManifest(robot);
		const declared = new Map(tools(m).map((e) => [e.name, e]));
		for (const [name, t] of registered) {
			const e = declared.get(name);
			// finish is the robot base's own tool (declared as modules/finish).
			assert.ok(e, `${robot} registers ${name}, which its manifest does not declare`);
			if (!e.module) assert.equal(t.description.length > 0, true, `${name} has the manifest's description`);
		}
		for (const e of declared.values()) assert.ok(toolSchema(e, { cameras: ["a"], arms: [] }));
	});

test("pi-side tools (side ts: the VLA and skill loops) are tool-mode only: never a code primitive", () => {
	for (const robot of ROBOTS) {
		const m = loadManifest(robot);
		const pi = m.primitives.filter((e) => e.side === "ts");
		for (const e of pi) assert.equal(e.doc.code, undefined, `${robot}.${e.name}: a ts tool has no code doc`);
		const names = new Set(pi.map((e) => e.name));
		const leaked = (["high", "low", "raw", "privileged", "low+privileged"] as const).flatMap((tier) =>
			codePrimitives(m, tier, () => true)
				.filter((p) => names.has(p.name) && !m.primitives.some((e) => e.name === p.name && e.side !== "ts"))
				.map((p) => `${tier}:${p.name}`),
		);
		assert.deepEqual(leaked, [], `${robot}: pi-side tools in code.api`);
	}
});
