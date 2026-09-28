/**
 * The model reaches a robot two ways: pi tools (TS, registered by the robot) and code-mode programs
 * (`run_code`) that call the env server's primitive registry (`code.api`, each robot's Python
 * `primitives.py`). Where a primitive and a tool of the same robot share a name, a model that uses
 * both must meet the same call: every parameter the primitive declares is also a parameter of the
 * tool (the tool may add TS-side options). This test loads every robot's tools in a stub pi and every
 * registry in Python (they import only components/code_api.py) and checks that, plus that every robot
 * that records a `codeApi` has a registry. Existing, reviewed differences are listed in `KNOWN` with
 * the reason; a new difference fails.
 */

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { SERVICES } from "../src/robot.ts";
import behavior from "../src/robots/behavior/index.ts";
import dualFranka from "../src/robots/dual_franka/index.ts";
import franka from "../src/robots/franka/index.ts";
import genesis from "../src/robots/genesis/index.ts";
import libero from "../src/robots/libero/index.ts";
import maniskill from "../src/robots/maniskill/index.ts";
import metaworld from "../src/robots/metaworld/index.ts";
import piper from "../src/robots/piper/index.ts";
import robocasa from "../src/robots/robocasa/index.ts";
import robodojo from "../src/robots/robodojo/index.ts";
import robolab from "../src/robots/robolab/index.ts";
import robosuite from "../src/robots/robosuite/index.ts";
import robotwin from "../src/robots/robotwin/index.ts";
import ur5e from "../src/robots/ur5e/index.ts";

const PYTHON = process.env.PYTHON ?? "python3";

/** Each robot: its extension, and the registry tuples its env server serves (module under robots/, attributes). */
const ROBOTS: Record<string, { load: (pi: ExtensionAPI) => unknown; module: string; attrs: string[] }> = {
	behavior: { load: behavior, module: "behavior", attrs: ["BEHAVIOR_PRIMITIVES"] },
	dual_franka: { load: dualFranka, module: "franka", attrs: ["DUAL_FRANKA_PRIMITIVES"] },
	franka: {
		load: franka,
		module: "franka",
		attrs: ["FRANKA_PRIMITIVES", "SEGMENT_PRIMITIVES", "ENHANCE_DEPTH_PRIMITIVES"],
	},
	genesis: { load: genesis, module: "genesis", attrs: ["GENESIS_PRIMITIVES"] },
	libero: { load: libero, module: "libero", attrs: ["LIBERO_PRIMITIVES", "CODE_PRIMITIVES"] },
	maniskill: { load: maniskill, module: "maniskill", attrs: ["MANISKILL_PRIMITIVES"] },
	metaworld: { load: metaworld, module: "metaworld", attrs: ["METAWORLD_PRIMITIVES"] },
	piper: { load: piper, module: "piper", attrs: ["PIPER_PRIMITIVES"] },
	robocasa: { load: robocasa, module: "robocasa", attrs: ["ROBOCASA_PRIMITIVES"] },
	robodojo: { load: robodojo, module: "robodojo", attrs: ["ROBODOJO_PRIMITIVES"] },
	robolab: { load: robolab, module: "robolab", attrs: ["ROBOLAB_PRIMITIVES"] },
	robosuite: { load: robosuite, module: "robosuite", attrs: ["ROBOSUITE_PRIMITIVES"] },
	robotwin: { load: robotwin, module: "robotwin", attrs: ["ROBOTWIN_PRIMITIVES"] },
	ur5e: { load: ur5e, module: "ur5e", attrs: ["UR5E_PRIMITIVES"] },
};

/**
 * Reviewed differences, `<robot>.<name>`: the primitive parameters the same-named tool does not
 * take, and why. Remove an entry when the two are aligned; never add one without a reason.
 */
const KNOWN: Record<string, { params: string[]; why: string }> = {
	"libero.preview_reach": {
		params: ["pos"],
		why: "the tool names the target xyz like move_to; the code primitive keeps the facade's argument name",
	},
	"robosuite.preview_reach": {
		params: ["pos"],
		why: "the tool names the target xyz like move_to; the code primitive keeps the facade's argument name",
	},
	"libero.set_gripper": {
		params: ["close"],
		why: "the tool takes the units-style gripper command (+1 / -1); the code primitive a boolean",
	},
	"genesis.back_project": {
		params: ["camera_name", "pixels"],
		why: "the primitive is the batch server call (many pixels); the tool back-projects one pixel",
	},
	"franka.segment": {
		params: ["text_prompt"],
		why: "not aligned yet: the tool says prompt, the primitive the server's text_prompt",
	},
	"robosuite.move_to": {
		params: ["quat_xyzw", "step_m", "step_rad", "target_xyz", "tol_m", "tol_rad"],
		why: "not aligned yet: the primitive exposes the server's raw servo arguments, the tool its own names (xyz, rotvec, tol)",
	},
	"robosuite.move_delta": {
		params: ["max_steps", "quat_xyzw", "rotvec", "step_m", "step_rad", "tol_m", "tol_rad"],
		why: "not aligned yet: the primitive exposes the server's servo arguments the tool leaves at their defaults",
	},
};
/** Transport plumbing, not a parameter the model chooses: a tool sets it itself. */
const PLUMBING = new Set(["return_frames"]);

type Primitive = { method: string; params: string[] };

/** Every robot's registry as Python declares it: {robot: {primitive: {method, params}}}. */
function registries(): Record<string, Record<string, Primitive>> {
	const table = Object.fromEntries(Object.entries(ROBOTS).map(([r, o]) => [r, [o.module, o.attrs]]));
	const code = `
import importlib, json, sys
out = {}
for robot, (module, attrs) in json.loads(sys.argv[1]).items():
    m = importlib.import_module(f"pi_embodied_services.robots.{module}.primitives")
    out[robot] = {p.name: {"method": p.method, "params": sorted(p.params)} for a in attrs for p in getattr(m, a)}
print(json.dumps(out))
`;
	const stdout = execFileSync(PYTHON, ["-c", code, JSON.stringify(table)], {
		cwd: SERVICES,
		env: { ...process.env, PYTHONPATH: SERVICES },
		encoding: "utf8",
	});
	return JSON.parse(stdout);
}

/** The tools a robot registers at default flags: {name: parameter names}. */
function tools(load: (pi: ExtensionAPI) => unknown): Record<string, string[]> {
	const flags: Record<string, unknown> = {};
	const out: Record<string, string[]> = {};
	const api: Record<string, unknown> = {
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: { name: string; parameters: { properties?: Record<string, unknown> } }) => {
			out[t.name] = Object.keys(t.parameters.properties ?? {}).sort();
		},
		getActiveTools: () => [],
		events: { emit: () => {}, on: () => () => {} },
	};
	load(new Proxy(api, { get: (t, k: string) => t[k] ?? (() => {}) }) as unknown as ExtensionAPI);
	return out;
}

const hasPython = (() => {
	try {
		execFileSync(PYTHON, ["--version"], { stdio: "ignore" });
		return true;
	} catch {
		return false;
	}
})();

test(
	"a primitive and the same-named tool of a robot take the same parameters",
	{ skip: !hasPython && `no ${PYTHON}` },
	() => {
		const py = registries();
		const drift: string[] = [];
		const used = new Set<string>();
		for (const [robot, o] of Object.entries(ROBOTS)) {
			assert.ok(Object.keys(py[robot]).length > 0, `${robot}: empty registry`);
			const ts = tools(o.load);
			for (const [name, p] of Object.entries(py[robot])) {
				if (!(name in ts)) continue;
				const missing = p.params.filter((k) => !PLUMBING.has(k) && !ts[name].includes(k));
				const key = `${robot}.${name}`;
				const known = KNOWN[key];
				if (known) {
					used.add(key);
					assert.deepEqual(missing, known.params, `${key}: the known difference changed`);
				} else if (missing.length)
					drift.push(`${key}: primitive params ${missing.join(", ")} not in the tool's (${ts[name].join(", ")})`);
			}
		}
		assert.deepEqual(drift, [], "tool and code.api primitive disagree");
		assert.deepEqual(
			Object.keys(KNOWN).filter((k) => !used.has(k)),
			[],
			"a KNOWN entry no longer applies: remove it",
		);
	},
);

test("every robot that records a code.api has a registry here, and every registry has a robot", () => {
	const src = new URL("../src/robots/", import.meta.url);
	for (const robot of Object.keys(ROBOTS)) {
		const text = readFileSync(new URL(`${robot}/index.ts`, src), "utf8");
		assert.match(text, /codeApi:/, `${robot} declares no codeApi but has a registry`);
	}
	const declared = execFileSync("grep", ["-l", "codeApi:", "-r", src.pathname], { encoding: "utf8" })
		.trim()
		.split("\n")
		.map((p) => p.slice(src.pathname.length).split("/")[0])
		.filter((d) => d !== "robot.ts");
	assert.deepEqual([...new Set(declared)].sort(), Object.keys(ROBOTS).sort());
});
