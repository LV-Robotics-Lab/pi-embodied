import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import maniskill, {
	calibrate,
	ENV_IDS,
	GAIN,
	GRIPPER_STEPS,
	grasped,
	hasWrist,
	type ManiskillRobot,
	phases,
	ROBOT_IDS,
	ROBOTS,
	robotFor,
	SERVO,
	STEP_M,
	sideBySide,
	VECTORS,
	VIEW_SETUP,
	VIEWS,
	waypoints,
} from "../src/maniskill/index.ts";
import { NdArray } from "../src/rpc.ts";
import { ground, MOVE_UNITS } from "../src/units/index.ts";

const close = (a: number[], b: number[]) => a.every((x, k) => Math.abs(x - b[k]) < 1e-9);

test("each MV_* unit is one ~2 cm decision along the yaml's base-frame vector", () => {
	for (const unit of MOVE_UNITS) {
		const move = ground({ vectors: VECTORS, stepM: STEP_M }, unit)!;
		const w = waypoints([0.1, 0.2, 0.3], move.delta);
		assert.equal(w.length, 1);
		assert.ok(
			close(
				w[0],
				[0.1, 0.2, 0.3].map((p, k) => p + VECTORS[unit][k] * 0.02),
			),
			`${unit} ${w[0]}`,
		);
	}
	assert.deepEqual(VECTORS.MV_LEFT, [0, -1, 0]); // configs/robot_maniskill.yaml: robot-left = -Y
});

test("a long move splits into evenly spaced ~2 cm waypoints; a gripper command or STOP holds in place", () => {
	const w = waypoints([0, 0, 0.2], [0.05, 0, -0.03]);
	assert.equal(w.length, 3); // |delta| 5.8 cm -> 3 decisions
	assert.ok(close(w[2], [0.05, 0, 0.17]));
	assert.ok(close(w[0], [0.05 / 3, 0, 0.19]));
	assert.deepEqual(waypoints([0, 0, 0.2], [0, 0, 0]), [[0, 0, 0.2]]);
});

test("the episode video frame puts the agentview and the wrist view side by side", () => {
	const a = new NdArray("uint8", [2, 1, 3], Buffer.from([1, 1, 1, 2, 2, 2]));
	const b = new NdArray("uint8", [2, 2, 3], Buffer.from([5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]));
	const f = sideBySide(a, b);
	assert.deepEqual(f.shape, [2, 3, 3]);
	assert.deepEqual([...f.data], [1, 1, 1, 5, 5, 5, 6, 6, 6, 2, 2, 2, 7, 7, 7, 8, 8, 8]);
});

test("the grasp flag is read from whichever is_*grasped key the scene reports", () => {
	assert.equal(grasped({ is_grasped: true }), true);
	assert.equal(grasped({ is_cubeA_grasped: 1, success: false }), true);
	assert.equal(grasped({ is_cubeA_grasped: false, is_obj_placed: true }), false);
});

test("move_delta with a gripper change: the fingers settle first, holding still, then the arm moves", () => {
	const start = [0.1, 0, 0.1];
	const p = phases(start, [0, 0, 0.04], true);
	// A GRIPPER_STEPS hold at the start, then the two ~2 cm waypoints of the move with the new command.
	assert.deepEqual(p[0], { target: start, minSteps: GRIPPER_STEPS, maxSteps: GRIPPER_STEPS });
	assert.equal(p.length, 3);
	assert.ok(close(p[2].target, [0.1, 0, 0.14]));
	assert.deepEqual([p[1].minSteps, p[1].maxSteps], [SERVO.minSteps, SERVO.maxSteps]);
	// A pure gripper toggle is the hold alone; a move without a change has no hold; STOP holds one decision.
	assert.deepEqual(phases(start, [0, 0, 0], true), [p[0]]);
	assert.equal(phases(start, [0, 0, 0.04], false).length, 2);
	assert.deepEqual(phases(start, [0, 0, 0], false), [
		{ target: start, minSteps: SERVO.minSteps, maxSteps: SERVO.minSteps },
	]);
});

test("probe-axes: a short unit is scaled up to stepM, a mirrored axis is refused", () => {
	const probes = MOVE_UNITS.map((unit) => ({
		unit,
		n: 4,
		moved: VECTORS[unit].map((x) => x * 0.018 * 4) as [number, number, number],
	}));
	const c = calibrate(VECTORS, STEP_M, probes);
	for (const unit of MOVE_UNITS) {
		const move = ground({ vectors: c.vectors, stepM: STEP_M }, unit)!;
		assert.ok(
			close(
				move.delta,
				VECTORS[unit].map((x) => (x * STEP_M * STEP_M) / 0.018),
			),
		);
	}
	assert.equal((c.units.MV_FWD as { per_unit_m: number }).per_unit_m, 0.018);
	const mirrored = probes.map((p) =>
		p.unit === "MV_LEFT" ? { ...p, moved: [0, 0.08, 0] as [number, number, number] } : p,
	);
	assert.throws(() => calibrate(VECTORS, STEP_M, mirrored), /MV_LEFT/);
});

test("--env-id: the RLinf rigs first (BlockPAP-v1 the default), the eight original ids unchanged, then OpenETA's tasks", () => {
	assert.deepEqual(ENV_IDS.slice(0, 8), [
		"BlockPAP-v1",
		"BlockStack-v1",
		"PickCube-v1",
		"StackCube-v1",
		"PushCube-v1",
		"PullCube-v1",
		"PokeCube-v1",
		"LiftPegUpright-v1",
	]);
	assert.deepEqual(ENV_IDS.slice(8), [
		"PlaceSphere-v1",
		"StackPyramid-v1",
		"PullCubeTool-v1",
		"PegInsertionSide-v1",
		"PlugCharger-v1",
		"PickSingleYCB-v1",
	]);
	// The same list as the env server's ENV_IDS (its INSTRUCTIONS keys after the rigs).
	const py = readFileSync(
		new URL("../../../services/pi_embodied_services/robots/maniskill/env_server.py", import.meta.url),
		"utf8",
	);
	const table = py.slice(py.indexOf("INSTRUCTIONS = {"), py.indexOf("ENV_IDS = "));
	const ids = [...table.matchAll(/^ {4}"([A-Za-z0-9-]+)": "/gm)].map((m) => m[1]);
	assert.deepEqual(ids, ENV_IDS.slice(2));
	assert.equal(new Set(ENV_IDS).size, ENV_IDS.length);
});

const SERVER = readFileSync(
	new URL("../../../services/pi_embodied_services/robots/maniskill/env_server.py", import.meta.url),
	"utf8",
);
/** The env server's ROBOTS rows: id -> its source block. */
function serverRobots(): Record<string, string> {
	const table = SERVER.slice(
		SERVER.indexOf("ROBOTS: dict[str, RobotSpec] = {"),
		SERVER.indexOf("\ndef add_wrist_camera"),
	);
	const rows = [...table.matchAll(/^ {4}"([a-z0-9_]+)": RobotSpec\(/gm)];
	return Object.fromEntries(rows.map((m, i) => [m[1], table.slice(m.index, rows[i + 1]?.index ?? table.length)]));
}

test("--robot: the same arms as the env server's ROBOTS, each with the env ids, wrist camera and view transform it measured", () => {
	const rows = serverRobots();
	assert.deepEqual(Object.keys(rows), ROBOT_IDS);
	assert.deepEqual(ROBOT_IDS, ["panda", "xarm6_robotiq", "widowxai"]);
	for (const id of ROBOT_IDS) {
		const r: ManiskillRobot = ROBOTS[id];
		const row = rows[id];
		const envs = row.includes("envs=_ALL_STOCK")
			? ENV_IDS.slice(2)
			: [.../envs=\(([^)]*)\)/.exec(row)![1].matchAll(/"([A-Za-z0-9-]+)"/g)].map((m) => m[1]);
		assert.deepEqual([...r.envs], envs, id);
		// No wrist camera on the server (wrist=None) is `wrist_mount: "none"` here; otherwise mount and transform agree.
		if (row.includes("wrist=None")) assert.equal(hasWrist(r), false, id);
		else {
			assert.ok(hasWrist(r), id);
			const mount = /"mount": "([a-z_]+)"/.exec(row)![1];
			const rotation = Number(/"rotation": (\d+)/.exec(row)![1]);
			const flip = /"flip": "([a-z]+)"/.exec(row)![1];
			assert.deepEqual(
				[r.setup.wrist_mount, r.setup.wrist_rotation, r.setup.wrist_flip],
				[mount, rotation, flip],
				id,
			);
		}
		assert.equal(r.setup.agentview, "oblique");
	}
	// The default keeps every Panda constant the robot had before --robot.
	const panda = ROBOTS.panda;
	assert.equal(panda.vectors, VECTORS);
	assert.deepEqual([panda.stepM, panda.gain, panda.gripperSteps], [STEP_M, GAIN, GRIPPER_STEPS]);
	assert.equal(panda.setup, VIEW_SETUP);
	assert.equal(panda.views, VIEWS);
	assert.equal(panda.arm, "Franka Panda arm");
});

test("--robot is checked against the env id: a rig runs its own Panda, a stock scene must be one the arm was measured on", () => {
	for (const envId of ENV_IDS) assert.equal(robotFor("panda", envId), ROBOTS.panda);
	assert.equal(robotFor("xarm6_robotiq", "StackCube-v1"), ROBOTS.xarm6_robotiq);
	assert.throws(() => robotFor("xarm6_robotiq", "BlockPAP-v1"), /real2sim rig .*--robot panda only/);
	assert.throws(() => robotFor("widowxai", "BlockStack-v1"), /--robot panda only/);
	// The goal beyond the xArm6's reach; the WidowX AI's reach ends before the stock scenes' objects.
	assert.throws(() => robotFor("xarm6_robotiq", "PushCube-v1"), /runs PickCube-v1, .*not PushCube-v1/);
	assert.throws(() => robotFor("widowxai", "StackCube-v1"), /runs PickCube-v1, not StackCube-v1/);
	assert.throws(() => robotFor("ur5", "PickCube-v1"), /unknown --robot ur5; one of panda, xarm6_robotiq, widowxai/);
	assert.throws(() => robotFor("toString", "PickCube-v1"), /unknown --robot/);
});

test("--robot units: every arm's MV_* is one ~2 cm decision along its measured base-frame axis; its gripper hold is its own", () => {
	for (const id of ROBOT_IDS) {
		const r = ROBOTS[id];
		for (const unit of MOVE_UNITS) {
			const move = ground({ vectors: r.vectors, stepM: r.stepM }, unit)!;
			// Measured on the box: 19.7-20.0 mm per unit along the commanded axis (cos 1.000) for all three.
			assert.ok(Math.abs(Math.hypot(...move.delta) - 0.02) < 1e-9, `${id} ${unit}`);
			assert.deepEqual(
				move.delta.map((v) => Math.sign(v)),
				VECTORS[unit].map((v) => Math.sign(v)),
				`${id} ${unit}`,
			);
			assert.equal(waypoints([0, 0, 0.2], move.delta, r.stepM).length, 1);
		}
		const p = phases([0, 0, 0.2], [0, 0, 0.04], true, r.gripperSteps, r.stepM);
		assert.deepEqual([p[0].minSteps, p[0].maxSteps], [r.gripperSteps, r.gripperSteps]);
		assert.equal(p.length, 3);
	}
	// The WidowX AI's carriages need 10 control steps to close on nothing (4 mm left after 6).
	assert.equal(ROBOTS.widowxai.gripperSteps, 10);
	assert.equal(ROBOTS.xarm6_robotiq.gripperSteps, GRIPPER_STEPS);
});

type Handler = (event: any, ctx: any) => unknown;
/** A stub pi (flags, tools, handlers in registration order) and a context without a UI. */
function stubPi(values: Record<string, unknown> = {}) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	let active: string[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		registerProvider: () => {},
		registerMessageRenderer: () => {},
		registerShortcut: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: (type: string, data: unknown) => entries.push({ type, data }),
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const ctx = {
		hasUI: false,
		cwd: tmpdir(),
		ui: { notify: () => {} },
		shutdown: () => {},
		abort: () => {},
		sessionManager: {
			getBranch: () => [],
			getEntries: () => [],
			getSessionDir: () => tmpdir(),
			getSessionFile: () => undefined,
			getSessionId: () => "sess",
		},
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) result = (await fn({ type: name, ...event }, ctx)) ?? result;
		return result;
	}
	const run = async (name: string, params: unknown) =>
		(await tools.get(name).execute("id", params, undefined, undefined, ctx)) as any;
	return { pi, flags, tools, entries, emit, run, active: () => active };
}

/** A fake ManiSkill env server (the wire protocol of ../src/rpc.ts) running `robot`, with or without a wrist view. */
async function fakeEnv(robot: string | undefined, setup: Record<string, unknown>, wrist: boolean) {
	const calls: { method: string; args: unknown[]; kwargs: Record<string, unknown> }[] = [];
	const nd = (dtype: string, shape: number[], data: Buffer) => ({
		__ndarray__: data.toString("base64"),
		dtype,
		shape,
	});
	const f32 = (v: number[]) => nd("float32", [v.length], Buffer.from(Float32Array.from(v).buffer));
	let tcp = [-0.4, 0, 0.08];
	const obs = () => ({
		agentview: nd("uint8", [2, 2, 3], Buffer.alloc(12)),
		...(wrist ? { wrist: nd("uint8", [2, 2, 3], Buffer.alloc(12)) } : {}),
		tcp_pos: f32(tcp),
		tcp_quat_wxyz: f32([0.7, 0, 0.7, 0]),
		gripper_width: 0.08,
		qpos: f32([0, 0]),
	});
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, args = [], kwargs = {} } = JSON.parse(body);
			calls.push({ method, args, kwargs });
			let result: unknown = { status: "ok" };
			if (method === "code.api") result = { tier: null, primitives: [], digest: "d" };
			else if (method === "env.get_env_meta")
				result = {
					env_id: "PickCube-v1",
					seed: 0,
					scene: null,
					table_z: 0,
					view_size: 256,
					...setup,
					...(robot ? { robot } : {}),
				};
			else if (method === "env.reset") result = [obs(), {}];
			else if (method === "env.get_task_language") result = "pick up the red cube";
			else if (method === "env.servo") {
				tcp = args[0] as number[];
				result = [[obs()], { is_grasped: false }];
			}
			res.end(JSON.stringify({ ok: true, result }));
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
	const close = () => {
		server.closeAllConnections();
		server.close();
	};
	return { url, calls, close };
}

test("--robot widowxai: a wrist-less arm starts on its own server, observes the agentview alone and says so in the prompt", async (t) => {
	const env = await fakeEnv("widowxai", ROBOTS.widowxai.setup, false);
	t.after(env.close);
	const s = stubPi({ env: env.url, "env-id": "PickCube-v1", robot: "widowxai" });
	maniskill(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	// The robot's tools, then memory's read-only file tools.
	assert.deepEqual(s.active().slice(0, 3), ["view_env_state", "move_delta", "finish"]);
	const r = await s.run("view_env_state", {});
	assert.deepEqual(
		r.content.map((c: { type: string }) => c.type),
		["text", "image"],
	);
	assert.deepEqual(r.details.images, ["agentview 2x2"]);
	const prompt = (await s.emit("before_agent_start")).systemPrompt as string;
	assert.match(prompt, /^You control a Trossen WidowX AI arm in the ManiSkill simulator/);
	assert.match(prompt, /no wrist camera/);
	assert.match(prompt, /in the image before each move/);
	assert.doesNotMatch(prompt, /both images|wrist image|\{\{/);
	// A gripper change holds the robot's 10 steps, then each 2 cm waypoint is one servo call.
	await s.run("move_delta", { delta_xyz: [0, 0, -0.04], gripper: "close" });
	const servos = env.calls.filter((c) => c.method === "env.servo");
	assert.deepEqual(
		servos.map((c) => [c.kwargs.min_steps, c.kwargs.max_steps, c.args[1]]),
		[
			[10, 10, -1],
			[SERVO.minSteps, SERVO.maxSteps, -1],
			[SERVO.minSteps, SERVO.maxSteps, -1],
		],
	);
	// The result names the arm; `robot` stays the pi robot.
	await s.emit("agent_start");
	await s.run("finish", { status: "failure", summary: "stop" });
	await s.emit("agent_end", { messages: [] });
	const result = s.entries.find((e) => e.type === "robot_result")?.data;
	assert.equal(result?.robot, "maniskill");
	assert.equal(result?.maniskill_robot, "widowxai");
});

test("the Panda's prompt is unchanged by --robot, and a server running another arm than --robot is refused", async (t) => {
	const panda = await fakeEnv(undefined, VIEW_SETUP, true);
	t.after(panda.close);
	const s = stubPi({ env: panda.url, "env-id": "PickCube-v1" });
	maniskill(s.pi);
	assert.equal(s.flags.robot, "panda");
	await s.emit("session_start");
	process.exitCode = undefined;
	// The robot's tools, then memory's read-only file tools.
	assert.deepEqual(s.active().slice(0, 3), ["view_env_state", "move_delta", "finish"]);
	const prompt = (await s.emit("before_agent_start")).systemPrompt as string;
	const system = readFileSync(new URL("../src/maniskill/SYSTEM.md", import.meta.url), "utf8");
	assert.match(system, /\{\{arm\}\}/);
	assert.match(prompt, /^You control a Franka Panda arm in the ManiSkill simulator/);
	assert.match(prompt, /relative to the gripper in both images before each move\./);
	assert.match(prompt, /check `is_grasped` and the wrist image before carrying\./);
	assert.equal((await s.run("view_env_state", {})).details.images.length, 2);

	const xarm = await fakeEnv("widowxai", ROBOTS.xarm6_robotiq.setup, true);
	t.after(xarm.close);
	const x = stubPi({ env: xarm.url, "env-id": "PickCube-v1", robot: "xarm6_robotiq" });
	maniskill(x.pi);
	await x.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(x.active(), []);
	assert.ok(!xarm.calls.some((c) => c.method === "env.reset"), "the robot never reset");
});

test("--units on --robot widowxai: no wrist view, so fine steps, no target_in_wrist and no wrist rules; the Panda keeps them", async (t) => {
	const plugins = "recovery,auto_release,proprioception,variable_step,action_chunk,rotation,plan,mem_text";
	const run = async (robot: string | undefined, setup: Record<string, unknown>, wrist: boolean) => {
		const env = await fakeEnv(robot, setup, wrist);
		t.after(env.close);
		const s = stubPi({
			env: env.url,
			"env-id": "PickCube-v1",
			...(robot ? { robot } : {}),
			units: "true",
			// The robot's default (auto) on the WidowX AI: naming a wrist-view plugin there refuses to start.
			"units-plugins": wrist ? plugins : "auto",
		});
		maniskill(s.pi);
		const schema = s.tools.get("act").parameters.properties;
		await s.emit("session_start");
		process.exitCode = undefined;
		const prompt = (await s.emit("before_agent_start")).systemPrompt as string;
		// 8 cm above the table: coarse on the Panda (the target not in the wrist view), fine without a wrist view.
		const before = env.calls.filter((c) => c.method === "env.servo").length;
		const r = await s.run("act", { unit: "MV_LEFT", target_in_wrist: false });
		const servos = env.calls.filter((c) => c.method === "env.servo").length - before;
		await s.emit("agent_start");
		await s.run("finish", { status: "failure", summary: "stop" });
		await s.emit("agent_end", { messages: [] });
		const result = s.entries.find((e) => e.type === "robot_result")?.data;
		return { schema, prompt, head: r.content[0].text as string, servos, result, active: s.active() };
	};
	const w = await run("widowxai", ROBOTS.widowxai.setup, false);
	assert.deepEqual(w.active, ["act", "plan", "finish"]);
	assert.ok(!("target_in_wrist" in w.schema) && !("plan" in w.schema), "act at load already follows --robot");
	assert.equal(w.servos, 1, "one 2 cm waypoint: the fine step");
	assert.match(w.head, /target_in_wrist ignored: this robot has no wrist view/);
	assert.doesNotMatch(w.prompt, /WRIST CHECK|target_in_wrist|ACTION PLAN|coarse/);
	assert.match(w.prompt, /There is no wrist view: the third-person view is the only guide/);
	assert.deepEqual(w.result.units_plugins, ["recovery", "auto_release", "proprioception", "plan", "mem_text"]);
	assert.equal(w.result.units_wrist_view, false);

	const p = await run(undefined, VIEW_SETUP, true);
	assert.ok("target_in_wrist" in p.schema && "plan" in p.schema);
	assert.equal(p.servos, 2, "the 4 cm coarse step as two 2 cm waypoints");
	assert.match(p.prompt, /WRIST CHECK/);
	assert.deepEqual(p.result.units_plugins, [
		"recovery",
		"auto_release",
		"proprioception",
		"variable_step",
		"action_chunk",
		"plan",
		"mem_text",
	]);
	assert.equal(p.result.units_wrist_view, true);
});
