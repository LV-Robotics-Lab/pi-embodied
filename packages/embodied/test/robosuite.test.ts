import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { ground, MOVE_UNITS } from "../src/modes/units/index.ts";
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
} from "../src/robots/robosuite/index.ts";
import { codeApiReply } from "./helpers/code-api.ts";
import { deployFlags } from "./helpers/deployment.ts";
import {
	type Call,
	checkPoint,
	checkSimExplore,
	f32,
	fakeEnv,
	perceptionAnswers,
	rgb,
	stubPi as simPi,
} from "./sim-stub.ts";

type Handler = (event: any, ctx: any) => unknown;
type Tool = { name: string; description: string; parameters: any; execute: (...a: any[]) => Promise<any> };

/** A stub pi recording the flags and tools the robot registers (flags at their defaults, or `values`). */
function stubPi(values: Record<string, unknown> = {}) {
	values = deployFlags(values);
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, { default?: unknown; description?: string }> = {};
	const tools = new Map<string, Tool>();
	const entries: { type: string; data: any }[] = [];
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
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const sessionDir = mkdtempSync(join(tmpdir(), "robosuite-"));
	const ctx = {
		hasUI: false,
		cwd: tmpdir(),
		ui: { notify: () => {} },
		shutdown: () => {},
		abort: () => {},
		sessionManager: {
			getBranch: () => [],
			getEntries: () => [],
			getSessionDir: () => sessionDir,
			getSessionFile: () => undefined,
			getSessionId: () => "sess",
		},
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) {
			const r: any = await fn({ type: name, ...event }, ctx);
			if (r !== undefined) result = r;
		}
		return result;
	}
	const run = async (name: string, params: unknown) =>
		(await tools.get(name)!.execute("id", params, undefined, undefined, ctx)) as any;
	return { pi, flags, tools, entries, emit, run, sessionDir, active: () => active };
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

test("flags: --task lists the tasks, --seed, --max-move, the service switches, and no privileged tool at load", () => {
	const f = stubPi();
	robosuite(f.pi);
	assert.equal(f.flags.task.default, "Lift");
	for (const t of TASKS) assert.match(String(f.flags.task.description), new RegExp(t));
	assert.equal(f.flags.seed.default, "0");
	assert.equal(f.flags["max-move"].default, String(MAX_MOVE_M));
	assert.ok(!("sam3" in f.flags) && !("cuda-device" in f.flags), "where services run is deployment config");
	assert.ok("ik" in f.flags && "grasp" in f.flags && "privileged" in f.flags);
	assert.ok(!f.tools.has("ground_truth_poses"), "--privileged off registers nothing");
	// Units and VDM are mounted.
	assert.ok("units" in f.flags && "vdm" in f.flags);
});

test("tool schemas: the perception and motion tools, `arm` on every motion tool, gripper commands", () => {
	const f = stubPi();
	robosuite(f.pi);
	for (const name of [
		"view_env_state",
		"get_camera_meta",
		"segment",
		"back_project",
		"move_to",
		"move_delta",
		"set_gripper",
		"finish",
	])
		assert.ok(f.tools.has(name), name);
	const props = (name: string) => f.tools.get(name)!.parameters.properties;
	// The schemas are the manifest's (manifests/robosuite.json); one arm: no `arm` (a two-arm session adds it).
	for (const name of ["move_to", "move_delta", "set_gripper"]) assert.equal(props(name).arm, undefined, name);
	assert.deepEqual(props("move_to").xyz.minItems, 3);
	assert.deepEqual(props("move_to").gripper.enum, ["open", "close"]);
	assert.equal(props("move_to").step_m, undefined, "a program-only servo argument");
	assert.deepEqual(props("move_delta").delta_xyz.maxItems, 3);
	assert.equal(props("set_gripper").close.type, "boolean");
	assert.deepEqual(f.tools.get("set_gripper")!.parameters.required, ["close"]);
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
	const text = readFileSync(new URL("../src/robots/robosuite/SYSTEM.md", import.meta.url), "utf8");
	for (const key of ["{{task_language}}", "{{arms}}", "{{table_z}}", "{{max_move}}"])
		assert.ok(text.includes(key), key);
	for (const tool of ["set_gripper", "segment", "back_project", "get_camera_meta"])
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
	const text = readFileSync(new URL("../src/robots/robosuite/SYSTEM.md", import.meta.url), "utf8");
	assert.match(text, /robot0 stands at -y facing \+y and robot1 at \+y facing -y/);
	assert.match(text, /\+x runs toward the image bottom/);
	assert.doesNotMatch(text, /\+x points away from robot0 across the table, \+y to robot0's left/);
});

test("grasp tools: plan_grasp, plan_place and check_attached are registered over the env server's planner", () => {
	const f = stubPi();
	robosuite(f.pi);
	for (const name of ["grasp", "place", "aux-model"]) assert.ok(name in f.flags, name);
	for (const name of ["plan_grasp", "plan_place", "check_attached"]) assert.ok(f.tools.has(name), name);
	const props = (name: string) => f.tools.get(name)!.parameters.properties;
	assert.deepEqual(props("plan_grasp").camera.enum, ["agentview", "wrist"]);
	assert.equal(props("plan_grasp").arm, undefined, "one arm at load");
	assert.equal(props("check_attached").arm, undefined);
	const text = readFileSync(new URL("../src/robots/robosuite/SYSTEM.md", import.meta.url), "utf8");
	assert.ok(text.includes("[tool:plan_grasp]") && text.includes("[tool:check_attached]"));
});

/** A fake robosuite env server running `task`. */
export async function fakeRobosuite(task = "Lift", answer: (c: Call) => unknown = () => undefined) {
	const two = TWO_ARM.includes(task as never);
	const obs = () => ({
		agentview: rgb(),
		wrist: rgb(),
		robot0_eef_pos: f32([0, 0, 1]),
		robot0_eef_quat: f32([1, 0, 0, 0]),
		robot0_gripper_width: 0.08,
		robot0_gripper_command: "open",
		...(two
			? {
					robot1_eef_pos: f32([0, 0.5, 1]),
					robot1_eef_quat: f32([1, 0, 0, 0]),
					robot1_gripper_width: 0.08,
					robot1_gripper_command: "open",
				}
			: {}),
		success: false,
		success_step: null,
		env_steps: 0,
	});
	return fakeEnv((c) => {
		const own = answer(c);
		if (own !== undefined) return own;
		if (c.method === "env.get_env_meta")
			return {
				task,
				seed: 0,
				arms: two ? ["robot0", "robot1"] : ["robot0"],
				gripper: task !== "Wipe",
				language: "lift the cube",
				box: [],
				z_floor: 0.8,
				table_z: 0.8,
				max_move_m: 0.3,
			};
		if (c.method === "env.reset") return [obs(), {}];
		if (c.method === "env.get_task_language") return "lift the cube";
		if (c.method.startsWith("env.move") || c.method === "env.set_gripper") return { obs: obs(), info: { ok: true } };
		return undefined;
	});
}

test("--contact-graspnet activates plan_grasp / plan_place / check_attached; plan_grasp reaches env.plan_grasp with the arm", async (t) => {
	const env = await fakeRobosuite("Lift", (c) =>
		c.method === "env.plan_grasp" ? { active: "g1", candidates: [{ id: "g1" }], expired_ids: [] } : undefined,
	);
	t.after(env.close);
	const s = simPi({ "env-url": env.url, task: "Lift", "contact-graspnet": "http://127.0.0.1:1" });
	robosuite(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	for (const name of ["plan_grasp", "check_attached"]) assert.ok(s.active().includes(name), name);
	assert.ok(!s.active().includes("plan_place"), "plan_place requires AnyPlace (--anyplace)");
	const r = await s.run("plan_grasp", { object: "red cube", camera: "wrist" });
	assert.equal(r.details.active, "g1");
	const call = env.calls.find((c) => c.method === "env.plan_grasp")!;
	assert.deepEqual(call.kwargs, { object: "red cube", camera: "wrist", arm: "robot0" });
	const prompt = (await s.emit("before_agent_start")).systemPrompt as string;
	assert.match(prompt, /# Planned grasps/);

	// Without a grasp backend nothing is active and the prompt does not describe it.
	const off = simPi({ "env-url": env.url, task: "Lift" });
	robosuite(off.pi);
	await off.emit("session_start");
	process.exitCode = undefined;
	assert.ok(off.active().includes("move_to"));
	assert.ok(!off.active().includes("plan_grasp"));
	assert.doesNotMatch((await off.emit("before_agent_start")).systemPrompt as string, /plan_grasp/);
});

test("Wipe's sponge has no fingers: no grasp tools even with a backend", async (t) => {
	const env = await fakeRobosuite("Wipe");
	t.after(env.close);
	const s = simPi({ "env-url": env.url, task: "Wipe", "contact-graspnet": "http://127.0.0.1:1" });
	robosuite(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.ok(s.active().includes("move_to"));
	assert.ok(!s.active().includes("plan_grasp") && !s.active().includes("set_gripper"));
});

test("--ik activates preview_reach, which asks env.preview_reach for the named arm without moving", async (t) => {
	const env = await fakeRobosuite("TwoArmLift", (c) =>
		c.method === "env.preview_reach" ? { status: "unreachable", reachable: false, message: "far" } : undefined,
	);
	t.after(env.close);
	const s = simPi({ "env-url": env.url, task: "TwoArmLift", ik: "http://127.0.0.1:1" });
	robosuite(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.ok(s.active().includes("preview_reach"));
	const r = await s.run("preview_reach", { xyz: [0.1, 0.2, 0.9], arm: "robot1" });
	assert.equal(r.details.status, "unreachable");
	const call = env.calls.find((c) => c.method === "env.preview_reach")!;
	assert.deepEqual(call.kwargs, { xyz: [0.1, 0.2, 0.9], arm: "robot1" });
	assert.ok(!env.calls.some((c) => c.method.startsWith("env.move")));
	assert.match((await s.emit("before_agent_start")).systemPrompt as string, /`preview_reach` tells/);
	// Two arms: the arm is required.
	await assert.rejects(() => s.run("preview_reach", { xyz: [0, 0, 1] }), /pass arm/);

	const off = simPi({ "env-url": env.url, task: "TwoArmLift" });
	robosuite(off.pi);
	await off.emit("session_start");
	process.exitCode = undefined;
	assert.ok(!off.active().includes("preview_reach"));
	assert.doesNotMatch((await off.emit("before_agent_start")).systemPrompt as string, /preview_reach/);
});

test("memory and exploration: reset restarts the seeded scene, the cell is robosuite_<task>_s<seed>", async (t) => {
	const env = await fakeRobosuite("Stack");
	t.after(env.close);
	await checkSimExplore({
		load: robosuite,
		values: { "env-url": env.url, task: "Stack", seed: "0" },
		tag: "robosuite_Stack_s0",
		resets: () => env.calls.filter((c) => c.method === "env.reset").length,
		observe: "view_env_state",
		distil: true,
	});
});

test("--detections activates detect / select_detection / reject_detection over env.detect; --unidepth adds enhance_depth", async (t) => {
	const env = await fakeRobosuite("Lift", (c) => {
		if (c.method === "env.get_env_meta")
			return {
				task: "Lift",
				seed: 0,
				arms: ["robot0"],
				gripper: true,
				language: "lift",
				box: [],
				z_floor: 0.8,
				table_z: 0.8,
				max_move_m: 0.3,
				capabilities: { perception: { segment: true, enhance_depth: true } },
			};
		if (c.method === "env.detect")
			return {
				found: true,
				observation: 3,
				ids: ["d1"],
				invalidated: ["d0"],
				detections: [
					{
						id: "d1",
						score: 0.91234,
						box: [1, 2, 3, 4],
						area_px: 40,
						centroid_rc: [0, 1],
						depth_m: 0.5,
						mask_png_base64: "x",
					},
				],
				overlay: rgb(),
			};
		if (c.method === "env.select_detection")
			return { ok: false, error: "d1 is stale", ids: [], selected: null, rejected: [] };
		if (c.method === "env.enhance_depth")
			return { ok: true, observation: 3, report: { mode: "filled" }, estimate: {}, depth: f32([1]) };
		if (c.method === "env.render_camera") return [rgb(512, 512), f32(new Array(512 * 512).fill(1))];
		if (c.method === "env.get_camera_meta")
			return {
				intrinsic_K: [
					[1, 0, 0],
					[0, 1, 0],
					[0, 0, 1],
				],
				extrinsic_cam2world: [
					[1, 0, 0, 0],
					[0, 1, 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1],
				],
			};
		return undefined;
	});
	t.after(env.close);
	const off = simPi({ "env-url": env.url, task: "Lift" });
	robosuite(off.pi);
	await off.emit("session_start");
	process.exitCode = undefined;
	assert.ok(!off.active().includes("detect") && !off.active().includes("enhance_depth"), "off by default");

	// The fake env server answers healthz: it stands in for the SAM3 and UniDepth servers the start probes.
	const s = simPi({ "env-url": env.url, task: "Lift", detections: true, sam3: env.url, unidepth: env.url });
	robosuite(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	for (const name of ["detect", "select_detection", "reject_detection", "enhance_depth"])
		assert.ok(s.active().includes(name), name);
	const r = await s.run("detect", { prompt: "red cube", camera: "wrist", all: true });
	assert.deepEqual(env.calls.find((c) => c.method === "env.detect")!.kwargs, {
		camera: "wrist",
		prompt: "red cube",
		min_score: 0.2,
		all: true,
	});
	assert.equal(r.details.detections[0].score, 0.912);
	assert.deepEqual(r.details.detections[0].centroid_pixel, [0, 1]);
	assert.deepEqual(r.details.detections[0].centroid_world_xyz, [1, 0, 1]);
	assert.equal(r.details.detections[0].mask_png_base64, undefined);
	assert.equal(r.content.filter((c: { type: string }) => c.type === "image").length, 1, "the overlay");
	assert.deepEqual(s.entries.filter((e) => e.type === "detections_expired").at(-1)?.data.ids, ["d0"]);
	const stale = await s.run("select_detection", { id: "d1" });
	assert.equal(stale.details.error, "d1 is stale");
	assert.equal(s.entries.filter((e) => e.type === "detections_expired").at(-1)?.data.error, "d1 is stale");
	const d = await s.run("enhance_depth", {});
	assert.deepEqual(env.calls.find((c) => c.method === "env.enhance_depth")!.kwargs, { camera: "agentview" });
	assert.equal(d.details.report.mode, "filled");
	assert.match(
		(await s.emit("before_agent_start")).systemPrompt as string,
		/`detect` gives SAM3 masks[\s\S]*`enhance_depth` fuses/,
	);
});

test("--point: Molmo on the current images; the pixel's world xyz through the world map", async (t) => {
	const env = await fakeRobosuite("Lift", (c) => {
		if (c.method === "env.render_camera") return [rgb(512, 512), f32(new Array(512 * 512).fill(1))];
		if (c.method === "env.get_camera_meta")
			return {
				intrinsic_K: [
					[1, 0, 0],
					[0, 1, 0],
					[0, 0, 1],
				],
				extrinsic_cam2world: [
					[1, 0, 0, 0],
					[0, 1, 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1],
				],
			};
		return perceptionAnswers(c);
	});
	t.after(env.close);
	const { one } = await checkPoint({
		load: robosuite,
		values: { "env-url": env.url, task: "Lift" },
		url: env.url,
		calls: env.calls,
		cameras: ["agentview", "wrist"],
	});
	assert.deepEqual(one.details.world_xyz, [1, 1, 1]);
});

/** A fake robosuite env server (`--env`): Lift seed 0, and a `code.run` that reports `run`'s fields. */
async function fakeCodeEnv(run: Record<string, unknown>, task = "Lift") {
	const calls: { method: string; kwargs: Record<string, unknown> }[] = [];
	const nd = (dtype: string, shape: number[], data: Buffer) => ({
		__ndarray__: data.toString("base64"),
		dtype,
		shape,
	});
	const f32 = (v: number[]) => nd("float32", [v.length], Buffer.from(Float32Array.from(v).buffer));
	const obs = (success: boolean, steps: number) => ({
		agentview: nd("uint8", [2, 2, 3], Buffer.alloc(12)),
		wrist: nd("uint8", [2, 2, 3], Buffer.alloc(12)),
		robot0_eef_pos: f32([0, 0, 1]),
		robot0_eef_quat: f32([0, 0, 0, 1]),
		robot0_gripper_width: 0.08,
		robot0_gripper_command: "open",
		success,
		success_step: success ? steps : null,
		env_steps: steps,
	});
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, kwargs = {} } = JSON.parse(body);
			calls.push({ method, kwargs });
			let result: unknown = { ok: true };
			if (method === "code.api")
				result = codeApiReply("robosuite", kwargs.tier, (c) => c === "sam3" || c === "fingers");
			else if (method === "env.get_env_meta") result = { task, seed: 0, table_z: 0.8 };
			else if (method === "env.reset") result = [obs(false, 0), {}];
			else if (method === "env.get_task_language") result = "lift the red cube";
			else if (method === "code.run")
				result = {
					status: "ran",
					stdout: "",
					stderr: "",
					traceback: null,
					error: null,
					result: null,
					calls: [],
					n_calls: 2,
					move_m: 0.1,
					ms: 5,
					steps: 40,
					obs: obs(true, 37),
					frames: [nd("uint8", [2, 2, 3], Buffer.alloc(12))],
					...run,
				};
			res.end(JSON.stringify({ ok: true, result }));
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
	return {
		url,
		calls,
		close: () => {
			server.closeAllConnections();
			server.close();
		},
	};
}

test("--code=true: run_code runs on the env server and its result becomes the observation and the success", async (t) => {
	const env = await fakeCodeEnv({});
	t.after(env.close);
	const s = stubPi({ "env-url": env.url, task: "Lift", code: "true", "code-api": "low-noexamples" });
	robosuite(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(s.active(), ["run_code", "read", "ls", "grep", "find", "write", "finish"]);
	assert.deepEqual(
		env.calls.filter((c) => c.method === "code.api").map((c) => c.kwargs.tier),
		[undefined, "low-noexamples"],
		"the episode's registry, then code mode's S4 tier",
	);
	await s.emit("agent_start");
	const r = await s.run("run_code", { code: "move_delta([0, 0, 0.05])" });
	const run = env.calls.find((c) => c.method === "code.run")!;
	assert.equal(run.kwargs.tier, "low-noexamples");
	assert.equal(r.details.status, "ran");
	assert.equal(r.details.success, true);
	assert.equal(r.details.step, 37, "the server's step count, absorbed from the run's obs");
	assert.deepEqual(
		r.content.map((c: { type: string }) => c.type),
		["text", "text", "image", "image"],
	);
	await s.run("finish", { status: "success", summary: "lifted" });
	await s.emit("agent_end", { messages: [] });
	const result = s.entries.find((e) => e.type === "robot_result")?.data;
	assert.equal(result.success, true);
	assert.equal(result.success_step, 37);
	assert.equal(result.code, "true");
	assert.equal(result.code_api, "low-noexamples");
});

test("--code-oracle runs the ported CaP-X program once, without the model, and records it", async (t) => {
	const env = await fakeCodeEnv({});
	t.after(env.close);
	const s = stubPi({
		"env-url": env.url,
		task: "Lift",
		code: "true",
		privileged: true,
		"code-oracle": "lift_privileged",
	});
	robosuite(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(await s.emit("input", { text: "Solve the task." }), { action: "handled" });
	const run = env.calls.find((c) => c.method === "code.run")!;
	assert.equal(run.kwargs.tier, "privileged");
	assert.doesNotMatch(String(run.kwargs.code), /def goto_pose\(/, "CaP-X's functions are the server's high tier");
	assert.match(String(run.kwargs.code), /sample_grasp_pose\("red cube"\)/, "CaP-X's program");
	// A second prompt does not run it again.
	assert.deepEqual(await s.emit("input", { text: "again" }), { action: "handled" });
	assert.equal(env.calls.filter((c) => c.method === "code.run").length, 1);
	assert.equal(s.entries.find((e) => e.type === "code_oracle")?.data.file, "lift_privileged.py");
	// No session file is written without an assistant message: the report is kept in the session dir.
	const kept = JSON.parse(readFileSync(join(s.sessionDir, "code_oracle.json"), "utf8"));
	assert.equal(kept.file, "lift_privileged.py");
	assert.equal(kept.run.status, "ran");
	await s.emit("session_shutdown");
	const result = s.entries.find((e) => e.type === "robot_result")?.data;
	assert.equal(result.code_oracle, "lift_privileged.py");
	assert.match(result.code_oracle_sha256, /^[0-9a-f]{64}$/);
	assert.equal(result.success, true);
	assert.equal(result.turns, 0);
});

test("--code-oracle refuses a tier or task the program was not written for", async (t) => {
	const env = await fakeCodeEnv({});
	t.after(env.close);
	for (const [flags, why] of [
		[{ task: "Lift", "code-oracle": "lift_privileged" }, /written for the privileged tier/],
		[{ task: "Stack", privileged: true, "code-oracle": "lift_privileged" }, /written for task Lift/],
		[{ task: "Lift", privileged: true, "code-oracle": "no_such_oracle" }, /no oracle no_such_oracle/],
	] as const) {
		const s = stubPi({ "env-url": env.url, code: "true", ...flags });
		robosuite(s.pi);
		const errors: string[] = [];
		const log = console.error;
		console.error = (m: string) => errors.push(m);
		try {
			await s.emit("session_start");
		} finally {
			console.error = log;
			process.exitCode = undefined;
		}
		assert.match(errors.join("\n"), why);
		assert.deepEqual(s.active(), []);
	}
});

test("a two-arm task starts: --task is read at session start, after pi has set the flags", async (t) => {
	const env = await fakeCodeEnv({}, "TwoArmLift");
	t.after(env.close);
	// pi sets CLI flag values only after every extension has loaded: at load getFlag answers the defaults.
	const late: Record<string, unknown> = {};
	const s = stubPi(late);
	robosuite(s.pi);
	Object.assign(late, { "env-url": env.url, task: "TwoArmLift", units: "true" });
	const errors: string[] = [];
	const log = console.error;
	console.error = (m: string) => errors.push(m);
	try {
		await s.emit("session_start");
	} finally {
		console.error = log;
		process.exitCode = undefined;
	}
	assert.deepEqual(errors, [], "not refused as a one-arm session");
	assert.ok(s.active().includes("act"));
	assert.deepEqual(s.tools.get("act")!.parameters.properties.arm.enum, [...ARMS], "act names the arm on two arms");
	// A one-arm task on the same load keeps `act` without `arm`.
	const one = await fakeCodeEnv({}, "Lift");
	t.after(one.close);
	Object.assign(late, { "env-url": one.url, task: "Lift" });
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.equal(s.tools.get("act")!.parameters.properties.arm, undefined);
});
