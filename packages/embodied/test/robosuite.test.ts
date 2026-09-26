import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import robosuite, {
	ARMS,
	arms,
	EMPTY_WIDTH_M,
	hasGripper,
	MAX_MOVE_M,
	NO_GRIPPER,
	STEP_M,
	TASKS,
	TWO_ARM,
	VECTORS,
	VIEWS,
	YAW_STEP_RAD,
} from "../src/robosuite/index.ts";
import { ground, MOVE_UNITS } from "../src/units/index.ts";

type Handler = (event: any, ctx: any) => unknown;
type Tool = { name: string; description: string; parameters: any; execute: (...a: any[]) => Promise<any> };

/** A stub pi recording the flags and tools the robot registers (flags at their defaults, or `values`). */
function stubPi(values: Record<string, unknown> = {}) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, { default?: unknown; description?: string }> = {};
	const tools = new Map<string, Tool>();
	let active: string[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown; description?: string }) => {
			flags[name] = o;
		},
		getFlag: (name: string) => (name in values ? values[name] : flags[name]?.default),
		registerTool: (t: Tool) => tools.set(t.name, t),
		registerCommand: () => {},
		registerProvider: () => {},
		registerMessageRenderer: () => {},
		registerShortcut: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	return { pi, flags, tools, active: () => active };
}

const close = (a: number[], b: number[]) => a.every((x, k) => Math.abs(x - b[k]) < 1e-9);

test("the seven CaP-X tasks, their arms and grippers", () => {
	assert.deepEqual(
		[...TASKS],
		["Lift", "Stack", "Restack", "Wipe", "NutAssemblySquare", "TwoArmLift", "TwoArmHandover"],
	);
	// The same list as the services' tasks.py TASKS.
	const py = readFileSync(
		new URL("../../../services/pi_embodied_services/robots/robosuite/tasks.py", import.meta.url),
		"utf8",
	);
	for (const t of TASKS) assert.match(py, new RegExp(`^    "${t}": Task\\(`, "m"), t);
	assert.deepEqual([...TWO_ARM], ["TwoArmLift", "TwoArmHandover"]);
	assert.deepEqual([...NO_GRIPPER], ["Wipe"]);
	assert.deepEqual([...arms("Lift")], ["robot0"]);
	assert.deepEqual([...arms("TwoArmLift")], [...ARMS]);
	assert.equal(hasGripper("Wipe"), false);
	assert.equal(hasGripper("Stack"), true);
});

test("flags: --task lists the tasks, --seed, --max-move, the service URLs, and no privileged tool at load", () => {
	const f = stubPi();
	robosuite(f.pi);
	assert.equal(f.flags.task.default, "Lift");
	for (const t of TASKS) assert.match(String(f.flags.task.description), new RegExp(t));
	assert.equal(f.flags.seed.default, "0");
	assert.equal(f.flags["max-move"].default, String(MAX_MOVE_M));
	assert.equal(f.flags.sam3.default, "http://127.0.0.1:18300");
	assert.ok("ik" in f.flags && "graspnet" in f.flags && "cuda-device" in f.flags && "privileged" in f.flags);
	assert.ok(!f.tools.has("ground_truth_poses"), "--privileged off registers nothing");
	// Units and VDM are mounted.
	assert.ok("units" in f.flags && "vdm" in f.flags);
});

test("tool schemas: the perception and motion tools, `arm` on every motion tool, gripper commands", () => {
	const f = stubPi();
	robosuite(f.pi);
	for (const name of [
		"view_env_state",
		"view_camera_meta",
		"segment",
		"back_project",
		"move_to",
		"move_delta",
		"gripper",
		"finish",
	])
		assert.ok(f.tools.has(name), name);
	const props = (name: string) => f.tools.get(name)!.parameters.properties;
	for (const name of ["move_to", "move_delta", "gripper"]) {
		assert.deepEqual(props(name).arm.enum, [...ARMS], `${name}.arm`);
		assert.ok(!f.tools.get(name)!.parameters.required?.includes("arm"), `${name}.arm is optional (one-arm tasks)`);
	}
	assert.deepEqual(props("move_to").xyz.minItems, 3);
	assert.deepEqual(props("move_to").gripper.enum, ["open", "close"]);
	assert.deepEqual(props("move_delta").delta_xyz.maxItems, 3);
	assert.deepEqual(props("gripper").command.enum, ["open", "close"]);
	assert.deepEqual(props("gripper").parameters?.required, undefined);
	assert.deepEqual(f.tools.get("gripper")!.parameters.required, ["command"]);
	assert.deepEqual(props("segment").camera.enum, ["agentview", "wrist"]);
	assert.deepEqual(props("back_project").row_range.minItems, 2);
	assert.deepEqual(f.tools.get("finish")!.parameters.properties.status.enum, ["success", "failure"]);
	assert.match(f.tools.get("move_to")!.description, new RegExp(`${MAX_MOVE_M} m`));
});

test("units grounding: each MV_* is a 2 cm world-frame step, ROTATE_* a 0.15 rad yaw, GRASP closes", () => {
	for (const unit of MOVE_UNITS) {
		const move = ground({ vectors: VECTORS, stepM: STEP_M, yawStepRad: YAW_STEP_RAD }, unit)!;
		assert.ok(
			close(
				move.delta,
				VECTORS[unit].map((x) => x * 0.02),
			),
			unit,
		);
		assert.equal(Math.hypot(...move.delta), STEP_M);
	}
	// robosuite's world frame: +x away from robot0, robot-left is -y (as LIBERO, also robosuite), +z up.
	assert.deepEqual(VECTORS.MV_FWD, [1, 0, 0]);
	assert.deepEqual(VECTORS.MV_LEFT, [0, -1, 0]);
	assert.deepEqual(VECTORS.MV_UP, [0, 0, 1]);
	const cw = ground({ vectors: VECTORS, stepM: STEP_M, yawStepRad: YAW_STEP_RAD }, "ROTATE_CW")!;
	assert.equal(cw.yaw, YAW_STEP_RAD);
	assert.equal(ground({ vectors: VECTORS, stepM: STEP_M }, "GRASP")!.gripper, "close");
	assert.ok(EMPTY_WIDTH_M < 0.01);
});

test("SYSTEM.md wraps the optional tools in [tool:...] blocks and carries the fill-ins", () => {
	const text = readFileSync(new URL("../src/robosuite/SYSTEM.md", import.meta.url), "utf8");
	for (const key of ["{{task_language}}", "{{arms}}", "{{table_z}}", "{{max_move}}"])
		assert.ok(text.includes(key), key);
	for (const tool of ["gripper", "segment", "back_project", "view_camera_meta"])
		assert.ok(text.includes(`[tool:${tool}]`) || text.includes(`[tool:segment|back_project]`), tool);
});

test("the frame text matches robosuite's cameras and the opposed two-arm layout", () => {
	// robot0_robotview (the Panda's robot.xml) and CaP-X's overhead agentview share one orientation
	// (quat [0.653, 0.271, 0.271, 0.653]): they look back at the robot(s) from beyond the far table
	// edge, so world +x (MV_FWD) runs toward the image bottom and +y (MV_RIGHT) toward the image right.
	assert.deepEqual(
		[VECTORS.MV_FWD, VECTORS.MV_RIGHT],
		[
			[1, 0, 0],
			[0, 1, 0],
		],
	);
	assert.match(VIEWS, /MV_FWD moves the gripper toward the image BOTTOM/);
	assert.match(VIEWS, /MV_BACK toward the image TOP/);
	assert.match(VIEWS, /robot0's base is at the image top/);
	// TwoArm "opposed": robosuite turns robot0 by +90 deg (-y, facing +y) and robot1 by -90 deg.
	assert.match(VIEWS, /robot0 stands at the image LEFT and robot1 at the image RIGHT/);
	assert.match(VIEWS, /for robot0 MV_RIGHT moves away from its base/);
	assert.doesNotMatch(VIEWS, /image bottom and robot1 at the top/);
	const text = readFileSync(new URL("../src/robosuite/SYSTEM.md", import.meta.url), "utf8");
	assert.match(text, /robot0 stands at -y facing \+y and robot1 at \+y facing -y/);
	assert.match(text, /\+x runs toward the image bottom/);
	assert.doesNotMatch(text, /\+x points away from robot0 across the table, \+y to robot0's left/);
});
