import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { NdArray } from "../src/infra/rpc.ts";
import { ground, MOVE_UNITS } from "../src/modes/units/index.ts";
import { loadManifest, toolSchema } from "../src/primitives/manifest.ts";
import maniskill, {
	calibrate,
	ENV_IDS,
	grasped,
	hasGripper,
	hasWrist,
	IK_ROBOTS,
	type ManiskillRobot,
	OTHER_ROBOT_ENVS,
	ROBOT_IDS,
	ROBOTS,
	robotFor,
	STEP_M,
	sideBySide,
	VECTORS,
	VIEW_SETUP,
	VIEWS,
} from "../src/robots/maniskill/index.ts";
import { codeApiReply } from "./helpers/code-api.ts";
import { deployFlags } from "./helpers/deployment.ts";

const close = (a: number[], b: number[]) => a.every((x, k) => Math.abs(x - b[k]) < 1e-9);

test("each MV_* unit is one ~2 cm decision along the yaml's base-frame vector", () => {
	for (const unit of MOVE_UNITS) {
		const move = ground({ vectors: VECTORS, stepM: STEP_M }, unit)!;
		// The env server splits a move into ceil(|delta| / 2 cm) waypoints: one per unit (services test_maniskill_move_delta.py).
		assert.ok(
			close(
				move.delta,
				VECTORS[unit].map((v) => v * 0.02),
			),
			`${unit} ${move.delta}`,
		);
	}
	assert.deepEqual(VECTORS.MV_LEFT, [0, -1, 0]); // configs/robot_maniskill.yaml: robot-left = -Y
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
		"FMBAssembly1Easy-v1",
		"PickCubeWidowXAI-v1",
		"PushT-v1",
		"DrawTriangle-v1",
		"DrawSVG-v1",
		"TwoRobotPickCube-v1",
		"TwoRobotStackCube-v1",
		"PutCarrotOnPlateInScene-v1",
		"PutEggplantInBasketScene-v1",
		"StackGreenCubeOnYellowCubeBakedTexInScene-v1",
		"PutSpoonOnTableClothInScene-v1",
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
	assert.deepEqual(ROBOT_IDS, ["panda", "xarm6_robotiq", "widowxai", "panda_stick", "panda_pair", "widowx250s"]);
	for (const id of ROBOT_IDS) {
		const r: ManiskillRobot = ROBOTS[id];
		const row = rows[id];
		const bridge = SERVER.slice(SERVER.indexOf("BRIDGE_ENVS = ("), SERVER.indexOf("#: Env ids whose registration"));
		const envs = row.includes("envs=_ALL_STOCK")
			? ENV_IDS.slice(2).filter((e) => !OTHER_ROBOT_ENVS[e])
			: [
					...(row.includes("envs=BRIDGE_ENVS") ? bridge : /envs=\(([^)]*)\)/.exec(row)![1]).matchAll(
						/"([A-Za-z0-9-]+)"/g,
					),
				].map((m) => m[1]);
		assert.deepEqual([...r.envs], envs, id);
		// No wrist camera on the server (wrist=None) is `wrist_mount: "none"` here; otherwise mount and transform agree.
		// No gripper on the server (gripper=None) is `gripper: false` here.
		assert.equal(hasGripper(r), !row.includes("gripper=None"), id);
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
		// The scene's own camera (agentview="...") or the shared oblique one.
		assert.equal(r.setup.agentview, /agentview="([a-z0-9_]+)"/.exec(row)?.[1] ?? "oblique", id);
	}
	// The default keeps every Panda constant the robot had before --robot.
	const panda = ROBOTS.panda;
	assert.equal(panda.vectors, VECTORS);
	assert.equal(panda.stepM, STEP_M);
	// The waypoint split and the per-call cap are the server's: its STEP_M and MAX_MOVE_M are pi's.
	assert.match(SERVER, /^STEP_M = 0\.02$/m);
	assert.match(SERVER, /^MAX_MOVE_M = 0\.2$/m);
	assert.equal(panda.setup, VIEW_SETUP);
	assert.equal(panda.views, VIEWS);
	assert.equal(panda.arm, "Franka Panda arm");
});

test("--robot is checked against the env id: a rig runs its own Panda, a stock scene must be one the arm was measured on", () => {
	for (const envId of ENV_IDS) if (!OTHER_ROBOT_ENVS[envId]) assert.equal(robotFor("panda", envId), ROBOTS.panda);
	// A task built for another robot names it; that robot runs it.
	for (const [envId, owner] of Object.entries(OTHER_ROBOT_ENVS)) {
		assert.throws(() => robotFor("panda", envId), new RegExp(`${envId} runs on --robot ${owner}`));
		assert.equal(robotFor(owner!, envId), ROBOTS[owner as keyof typeof ROBOTS]);
	}
	// The server's own table of them is the same.
	const other = SERVER.slice(SERVER.indexOf("_OTHER_ROBOT = {"), SERVER.indexOf("#: The stock env ids the Panda"));
	assert.deepEqual(
		Object.fromEntries([...other.matchAll(/"([A-Za-z0-9-]+)": "([a-z0-9_]+)"/g)].map((m) => [m[1], m[2]])),
		OTHER_ROBOT_ENVS,
	);
	assert.equal(robotFor("xarm6_robotiq", "StackCube-v1"), ROBOTS.xarm6_robotiq);
	assert.throws(() => robotFor("xarm6_robotiq", "BlockPAP-v1"), /real2sim rig .*--robot panda only/);
	assert.throws(() => robotFor("widowxai", "BlockStack-v1"), /--robot panda only/);
	// The goal beyond the xArm6's reach; the WidowX AI's reach ends before the stock scenes' objects.
	assert.throws(() => robotFor("xarm6_robotiq", "PushCube-v1"), /runs PickCube-v1, .*not PushCube-v1/);
	// PullCubeTool's "within 0.6 m of the base" holds at reset on ~8 % of seeds with the nearer xArm6 base.
	assert.throws(() => robotFor("xarm6_robotiq", "PullCubeTool-v1"), /not PullCubeTool-v1/);
	assert.throws(() => robotFor("widowxai", "StackCube-v1"), /runs PickCube-v1, PickCubeWidowXAI-v1, not StackCube-v1/);
	assert.throws(
		() => robotFor("ur5", "PickCube-v1"),
		/unknown --robot ur5; one of panda, xarm6_robotiq, widowxai, panda_stick, panda_pair, widowx250s/,
	);
	assert.throws(() => robotFor("toString", "PickCube-v1"), /unknown --robot/);
});

test("--robot units: every arm's MV_* is one ~2 cm decision along its measured base-frame axis", () => {
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
		}
	}
	// Each arm's gripper hold (the WidowX AI's 10 steps, ...) is the env server's RobotSpec.gripper_steps.
	assert.match(serverRobots().widowxai, /gripper_steps=10,/);
});

type Handler = (event: any, ctx: any) => unknown;
/** A stub pi (flags, tools, handlers in registration order) and a context without a UI. */
function stubPi(values: Record<string, unknown> = {}) {
	values = deployFlags(values);
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
async function fakeEnv(
	robot: string | undefined,
	setup: Record<string, unknown>,
	wrist: boolean,
	envId = "PickCube-v1",
	arms?: string[],
) {
	const calls: { method: string; args: unknown[]; kwargs: Record<string, unknown> }[] = [];
	const nd = (dtype: string, shape: number[], data: Buffer) => ({
		__ndarray__: data.toString("base64"),
		dtype,
		shape,
	});
	const f32 = (v: number[]) => nd("float32", [v.length], Buffer.from(Float32Array.from(v).buffer));
	let tcp = [-0.4, 0, 0.08];
	const armTcp: Record<string, number[]> = Object.fromEntries(
		(arms ?? []).map((a, i) => [a, [0, i ? 0.12 : -0.12, 0.18]]),
	);
	const one = (p: number[]) => ({
		tcp_pos: f32(p),
		tcp_quat_wxyz: f32([0.7, 0, 0.7, 0]),
		gripper_width: 0.08,
		qpos: f32([0, 0]),
	});
	const obs = () => ({
		agentview: nd("uint8", [2, 2, 3], Buffer.alloc(12)),
		...(wrist ? { wrist: nd("uint8", [2, 2, 3], Buffer.alloc(12)) } : {}),
		...(arms ? { arms: Object.fromEntries(arms.map((a) => [a, one(armTcp[a])])) } : one(tcp)),
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
			if (method === "code.api") result = codeApiReply("maniskill", kwargs.tier as string | undefined);
			else if (method === "env.get_env_meta")
				result = {
					env_id: envId,
					seed: 0,
					scene: null,
					table_z: 0,
					view_size: 256,
					...setup,
					...(robot ? { robot } : {}),
				};
			else if (method === "env.reset") result = [obs(), {}];
			else if (method === "env.get_task_language") result = "pick up the red cube";
			else if (method === "env.move_delta") {
				// The server's refusals (env_server.py move_delta); the waypoint legs themselves are tested in Python.
				const d = kwargs.delta_xyz as number[];
				if (robot === "panda_stick" && kwargs.gripper) {
					res.end(JSON.stringify({ ok: false, error: "ValueError: this robot holds a stick and has no gripper" }));
					return;
				}
				const a = kwargs.arm as string | undefined;
				if (arms && !a) {
					res.end(
						JSON.stringify({
							ok: false,
							error: `ValueError: this robot has two arms; arm must be one of ${arms.join(", ")}`,
						}),
					);
					return;
				}
				const from = a ? armTcp[a] : tcp;
				const to = from.map((v, k) => v + d[k]);
				if (a) armTcp[a] = to;
				else tcp = to;
				const limited = Boolean(process.env.FAKE_STEP_LIMIT);
				result = {
					result: {
						...(a ? { arm: a } : {}),
						commanded_m: d,
						moved_m: d,
						gripper: kwargs.gripper ?? "open",
						env_steps: 2,
						...(limited
							? { step_limit: "the episode's step limit is reached; no further motion runs: call finish" }
							: {}),
					},
					frames: [
						{ ...obs(), action: f32([0, 0, 0, 1]), success: false },
						{ ...obs(), action: f32([0, 0, 0, 1]), success: false },
					],
					obs: obs(),
					info: { is_grasped: false },
				};
			} else if (method === "code.run") {
				tcp = [-0.4, 0, 0.3];
				result = {
					status: "ran",
					stdout: "",
					stderr: "",
					traceback: null,
					error: null,
					result: null,
					calls: [],
					n_calls: 3,
					move_m: 0.2,
					ms: 5,
					steps: 12,
					success: true,
					obs: obs(),
					info: { success: true, is_grasped: true },
					gripper: -1,
					frames: [nd("uint8", [2, wrist ? 4 : 2, 3], Buffer.alloc(wrist ? 24 : 12))],
				};
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
	// One env.move_delta with the tool's parameters (the server holds the robot's 10 gripper steps, then servos
	// each 2 cm waypoint); every control step it returns goes to the video.
	const r0 = await s.run("move_delta", { delta_xyz: [0, 0, -0.04], gripper: "close" });
	const moves = env.calls.filter((c) => c.method === "env.move_delta");
	assert.deepEqual(
		moves.map((c) => c.kwargs),
		[{ delta_xyz: [0, 0, -0.04], gripper: "close" }],
	);
	assert.ok(!env.calls.some((c) => c.method === "env.servo"), "no servo loop in TypeScript");
	assert.equal(r0.details.step, 2);
	assert.equal(r0.details.state.gripper_command, "close");
	assert.deepEqual(r0.details.result.moved_m, [0, 0, -0.04]);
	// The result names the arm; `robot` stays the pi robot.
	await s.emit("agent_start");
	await s.run("finish", { status: "failure", summary: "stop" });
	await s.emit("agent_end", { messages: [] });
	const result = s.entries.find((e) => e.type === "robot_result")?.data;
	assert.equal(result?.robot, "maniskill");
	assert.equal(result?.maniskill_robot, "widowxai");
});

test("--robot panda_stick: no gripper, so move_delta refuses `gripper`, the state has no gripper fields, the prompt says stick", async (t) => {
	const env = await fakeEnv("panda_stick", ROBOTS.panda_stick.setup, false, "PushT-v1");
	t.after(env.close);
	const s = stubPi({ env: env.url, "env-id": "PushT-v1", robot: "panda_stick" });
	maniskill(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	const prompt = (await s.emit("before_agent_start")).systemPrompt as string;
	assert.match(prompt, /^You control a Franka Panda arm holding a stick \(no gripper\) in the ManiSkill simulator/);
	assert.match(prompt, /`tcp_pos` is the stick's tip\. There is no gripper: never pass `gripper`\./);
	assert.match(prompt, /PushT's T block is about 4 cm tall/);
	assert.doesNotMatch(prompt, /\[\/?(gripper|stick)\]|\{\{|fingertips straddle|`gripper: "close"`/);
	await assert.rejects(s.run("move_delta", { delta_xyz: [0, 0, -0.02], gripper: "close" }), /has no gripper/);
	const r = await s.run("move_delta", { delta_xyz: [0, 0, -0.04] });
	assert.deepEqual(Object.keys(r.details.state), ["tcp_pos"]);
	assert.deepEqual(
		env.calls.filter((c) => c.method === "env.move_delta").map((c) => c.kwargs),
		[{ delta_xyz: [0, 0, -0.02], gripper: "close" }, { delta_xyz: [0, 0, -0.04] }],
	);
	// The Panda cannot run it.
	assert.throws(() => robotFor("panda", "DrawTriangle-v1"), /runs on --robot panda_stick/);
});

test("--units on --robot panda_stick: act has no GRASP / RELEASE and the units state no gripper", async (t) => {
	const env = await fakeEnv("panda_stick", ROBOTS.panda_stick.setup, false, "DrawTriangle-v1");
	t.after(env.close);
	const s = stubPi({ env: env.url, "env-id": "DrawTriangle-v1", robot: "panda_stick", units: "true" });
	maniskill(s.pi);
	const units = s.tools.get("act").parameters.properties.unit.enum as string[];
	assert.ok(!units.includes("GRASP") && !units.includes("RELEASE") && units.includes("MV_DOWN"));
	await s.emit("session_start");
	process.exitCode = undefined;
	const prompt = (await s.emit("before_agent_start")).systemPrompt as string;
	assert.match(prompt, /This robot has NO gripper/);
	assert.doesNotMatch(prompt, /GRIPPER:/);
	const r = await s.run("act", { unit: "MV_DOWN" });
	assert.doesNotMatch(r.content[0].text as string, /gripper_width|is_grasped/);
});

test("--robot panda_pair: move_delta takes `arm`, moves that arm alone in the world frame, and the state is per arm", async (t) => {
	const env = await fakeEnv("panda_pair", ROBOTS.panda_pair.setup, false, "TwoRobotStackCube-v1", ["left", "right"]);
	t.after(env.close);
	const s = stubPi({ env: env.url, "env-id": "TwoRobotStackCube-v1", robot: "panda_pair" });
	maniskill(s.pi);
	const tool = s.tools.get("move_delta");
	assert.deepEqual(tool.parameters.properties.arm.enum, ["left", "right"]);
	assert.match(tool.description, /ONE arm's gripper .*world-frame/);
	await s.emit("session_start");
	process.exitCode = undefined;
	const prompt = (await s.emit("before_agent_start")).systemPrompt as string;
	assert.match(prompt, /^You control a pair of Franka Panda arms \(left and right\)/);
	assert.match(prompt, /There are two arms, `left` and `right`/);
	assert.doesNotMatch(prompt, /\[\/?\w+\]|base-frame|\{\{/);
	await assert.rejects(s.run("move_delta", { delta_xyz: [0, 0, -0.02] }), /arm must be one of left, right/);
	const r = await s.run("move_delta", { delta_xyz: [0, 0.02, -0.02], gripper: "close", arm: "right" });
	const move = env.calls.filter((c) => c.method === "env.move_delta").at(-1);
	assert.deepEqual(move?.kwargs, { delta_xyz: [0, 0.02, -0.02], gripper: "close", arm: "right" });
	assert.equal(r.details.result.arm, "right");
	assert.deepEqual(r.details.state.arms.right.tcp_pos, [0, 0.14, 0.16]);
	assert.equal(r.details.state.arms.right.gripper_command, "close");
	assert.equal(r.details.state.arms.left.gripper_command, "open");
	assert.deepEqual(r.details.state.arms.left.tcp_pos, [0, -0.12, 0.18]);
	// One-arm robots take no arm.
	assert.equal((ROBOTS.panda as ManiskillRobot).arms, undefined);
});

test("--units on --robot panda_pair: act takes `arm` and drives that arm", async (t) => {
	const env = await fakeEnv("panda_pair", ROBOTS.panda_pair.setup, false, "TwoRobotPickCube-v1", ["left", "right"]);
	t.after(env.close);
	const s = stubPi({ env: env.url, "env-id": "TwoRobotPickCube-v1", robot: "panda_pair", units: "true" });
	maniskill(s.pi);
	assert.ok("arm" in s.tools.get("act").parameters.properties);
	await s.emit("session_start");
	process.exitCode = undefined;
	await s.run("act", { unit: "MV_RIGHT", arm: "left" });
	const move = env.calls.filter((c) => c.method === "env.move_delta").at(-1);
	assert.equal(move?.kwargs.arm, "left");
	assert.deepEqual(
		(move?.kwargs.delta_xyz as number[]).map((v) => Number(v.toFixed(4))),
		[0, 0.02, 0],
	);
});

test("--robot widowx250s: the bridge twin's own camera and a world-frame move, stated in the prompt and move_delta", async (t) => {
	const env = await fakeEnv("widowx250s", ROBOTS.widowx250s.setup, false, "PutCarrotOnPlateInScene-v1");
	t.after(env.close);
	const s = stubPi({ env: env.url, "env-id": "PutCarrotOnPlateInScene-v1", robot: "widowx250s" });
	maniskill(s.pi);
	assert.match(s.tools.get("move_delta").description, /world-frame \[dx, dy, dz\] in metres: \+x toward the camera/);
	await s.emit("session_start");
	process.exitCode = undefined;
	const prompt = (await s.emit("before_agent_start")).systemPrompt as string;
	assert.match(prompt, /^You control a WidowX 250 S arm \(a BridgeData V2 real-to-sim scene\)/);
	assert.match(prompt, /`move_delta` takes a world-frame `\[dx, dy, dz\]` in metres: \+x toward the camera/);
	assert.match(prompt, /The table top is at about z = 0\.87/);
	assert.doesNotMatch(prompt, /\{\{|away from the robot base/);
	assert.throws(() => robotFor("panda", "PutSpoonOnTableClothInScene-v1"), /runs on --robot widowx250s/);
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
	const system = readFileSync(new URL("../src/robots/maniskill/SYSTEM.md", import.meta.url), "utf8");
	assert.match(system, /\{\{arm\}\}/);
	assert.match(prompt, /^You control a Franka Panda arm in the ManiSkill simulator/);
	assert.match(prompt, /relative to the gripper in both images before each move\./);
	assert.match(prompt, /check `is_grasped` and the wrist image before carrying\./);
	assert.equal((await s.run("view_env_state", {})).details.images.length, 2);
	// The manifest's move_delta: one arm takes no `arm`, the base frame and the cap are stated.
	const md = s.tools.get("move_delta");
	assert.deepEqual(Object.keys(md.parameters.properties), ["delta_xyz", "gripper"]);
	assert.match(
		md.description,
		/^Translate the gripper by a base-frame \[dx, dy, dz\] in metres \(\+x away from the base, .*\(at most 0\.2 m per call\)/,
	);
	assert.deepEqual(
		(
			toolSchema(loadManifest("maniskill").primitives.find((e) => e.name === "move_delta")!, {
				arms: ["left", "right"],
			}).properties.arm as { enum?: string[] }
		).enum,
		["left", "right"],
	);

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
		const r = await s.run("act", { unit: "MV_LEFT", target_in_wrist: false });
		const move = env.calls.filter((c) => c.method === "env.move_delta").at(-1);
		const servos = Math.round(Math.hypot(...(move?.kwargs.delta_xyz as number[])) / STEP_M);
		await s.emit("agent_start");
		await s.run("finish", { status: "failure", summary: "stop" });
		await s.emit("agent_end", { messages: [] });
		const result = s.entries.find((e) => e.type === "robot_result")?.data;
		return { schema, prompt, head: r.content[0].text as string, servos, result, active: s.active() };
	};
	const w = await run("widowxai", ROBOTS.widowxai.setup, false);
	assert.deepEqual(w.active, ["act", "plan", "read", "ls", "grep", "find", "write", "finish"]);
	assert.ok(!("target_in_wrist" in w.schema) && !("plan" in w.schema), "act at load already follows --robot");
	assert.equal(w.servos, 1, "one 2 cm unit: the fine step");
	assert.match(w.head, /target_in_wrist ignored: this robot has no wrist view/);
	assert.doesNotMatch(w.prompt, /WRIST CHECK|target_in_wrist|ACTION PLAN|coarse/);
	assert.match(w.prompt, /There is no wrist view: the third-person view is the only guide/);
	assert.deepEqual(w.result.units_plugins, ["recovery", "auto_release", "proprioception", "plan", "mem_text"]);
	assert.equal(w.result.units_wrist_view, false);

	const p = await run(undefined, VIEW_SETUP, true);
	assert.ok("target_in_wrist" in p.schema && "plan" in p.schema);
	assert.equal(p.servos, 2, "the 4 cm coarse step (two 2 cm waypoints on the server)");
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

test("--code=true: run_code runs on the env server and its result becomes the observation, the grasp and the success", async (t) => {
	const env = await fakeEnv(undefined, VIEW_SETUP, true);
	t.after(env.close);
	const s = stubPi({ env: env.url, "env-id": "PickCube-v1", code: "true", "code-api": "low-noexamples" });
	maniskill(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(s.active().slice(0, 7), ["run_code", "read", "ls", "grep", "find", "write", "finish"]);
	assert.deepEqual(
		env.calls.filter((c) => c.method === "code.api").map((c) => c.kwargs.tier),
		[undefined, "low-noexamples"],
	);
	await s.emit("agent_start");
	const r = await s.run("run_code", { code: "chunk_step([[0, 0, 1, -1]] * 2)" });
	const run = env.calls.find((c) => c.method === "code.run")!;
	assert.equal(run.kwargs.tier, "low-noexamples");
	assert.equal(r.details.status, "ran");
	assert.equal(r.details.success, true);
	assert.equal(r.details.step, 12);
	assert.deepEqual(r.details.state.tcp_pos, [-0.4, 0, 0.3]);
	assert.equal(r.details.state.gripper_command, "close");
	assert.equal(r.details.state.is_grasped, true);
	assert.match((await s.run("run_code", { code: "state()" })).content[0].text, /already solved/);
	await s.run("finish", { status: "success", summary: "picked" });
	await s.emit("agent_end", { messages: [] });
	const result = s.entries.find((e) => e.type === "robot_result")?.data;
	assert.equal(result.success, true);
	assert.equal(result.ever_grasped, true);
	assert.equal(result.env_steps, 12);
	assert.equal(result.code, "true");
	assert.equal(result.code_api, "low-noexamples");
});

test("--ik: preview_reach passes the manifest's xyz / quat_xyzw straight to the env server (Panda and xArm6 only)", async (t) => {
	const env = await fakeEnv(undefined, VIEW_SETUP, true);
	t.after(env.close);
	const s = stubPi({ env: env.url, "env-id": "PickCube-v1", ik: "http://127.0.0.1:1" });
	maniskill(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.ok(s.active().includes("preview_reach"));
	await s.run("preview_reach", { xyz: [0, 0.1, 0.05] });
	assert.deepEqual(env.calls.find((c) => c.method === "env.preview_reach")?.kwargs, { xyz: [0, 0.1, 0.05] });
	assert.deepEqual(IK_ROBOTS, ["panda", "xarm6_robotiq"]);
});

test("a drawing scene's spent step budget ends a move and says to finish", async (t) => {
	const env = await fakeEnv("panda_stick", ROBOTS.panda_stick.setup, false, "DrawTriangle-v1");
	t.after(env.close);
	const s = stubPi({ env: env.url, "env-id": "DrawTriangle-v1", robot: "panda_stick" });
	maniskill(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	process.env.FAKE_STEP_LIMIT = "1";
	t.after(() => delete process.env.FAKE_STEP_LIMIT);
	const r = await s.run("move_delta", { delta_xyz: [0.06, 0, 0] });
	// The server stops at the spent budget and says so; pi shows it.
	assert.equal(env.calls.filter((c) => c.method === "env.move_delta").length, 1);
	assert.match(r.details.result.step_limit, /step limit is reached.*call finish/);
});
