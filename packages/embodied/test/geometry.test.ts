import assert from "node:assert/strict";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import franka from "../src/franka/index.ts";
import libero from "../src/libero/index.ts";
import {
	eulerXyz,
	GEOMETRY_TOOLS,
	type GripPlan,
	geometryArgs,
	geometryDefs,
	geometryTools,
	planRotation,
	quatMatrix,
	rotvec,
	runGripPlan,
	type ServoIo,
	servoGrip,
} from "../src/primitives/geometry.ts";
import robosuite from "../src/robosuite/index.ts";

type Json = Record<string, any>;

/** A stub pi with flags (overridable) and a tool registry. */
function fakePi(flagValues: Record<string, unknown> = {}) {
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const api: Record<string, unknown> = {
		on: () => {},
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		setActiveTools: () => {},
		getActiveTools: () => [],
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	};
	const pi = new Proxy(api, { get: (t, k: string) => t[k] ?? (() => {}) }) as unknown as ExtensionAPI;
	return { pi, flags, tools };
}

/** A fake env: answers `env.*` from `answers` (a value, an Error, or a function of the kwargs). */
function fakeEnv(answers: Record<string, Json | Error | ((kw: Json) => Json)>) {
	const calls: { method: string; kwargs: Json }[] = [];
	const call = async (method: string, kwargs: Json) => {
		calls.push({ method, kwargs });
		const a = answers[method];
		if (a === undefined) throw new Error(`unexpected ${method}`);
		if (a instanceof Error) throw a;
		return typeof a === "function" ? a(kwargs) : a;
	};
	return { call, calls };
}

const PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
const text = (r: any) => JSON.parse(r.content[0].text);
const images = (r: any) => r.content.filter((c: any) => c.type === "image").length;

const noRig = {
	call: async () => ({}),
	cameras: ["agentview", "wrist"],
	execute: async () => ({ content: [], details: {} }),
};

test("--geometry is off by default; on, the tools mount once, at the first start", () => {
	const off = fakePi();
	const none: string[] = [];
	const activateOff = geometryTools(off.pi, noRig, (d) => none.push(d.name));
	assert.equal(off.flags.geometry, false);
	assert.deepEqual(geometryArgs(off.pi), []);
	assert.deepEqual(activateOff(), []);
	assert.deepEqual(none, [], "nothing is registered with the flag off");
	const on = fakePi({ geometry: true });
	const mounted: string[] = [];
	const activate = geometryTools(on.pi, noRig, (d) => mounted.push(d.name));
	assert.deepEqual(geometryArgs(on.pi), ["--geometry"]);
	assert.deepEqual(mounted, [], "not at load");
	assert.deepEqual(activate(), [...GEOMETRY_TOOLS]);
	assert.deepEqual(activate(), [...GEOMETRY_TOOLS]);
	assert.deepEqual(mounted, [...GEOMETRY_TOOLS], "mounted once");
	// The robots register the flag, and no tool at load.
	for (const load of [libero, franka, robosuite]) {
		const g = fakePi({ geometry: true });
		load(g.pi);
		assert.ok("geometry" in g.flags);
		for (const t of GEOMETRY_TOOLS) assert.ok(!g.tools.has(t), `${t} is not registered at load`);
	}
});

test("the tools' schemas: the robot's cameras as views, the move_grip target fields", () => {
	for (const cams of [
		["agentview", "wrist"],
		["third_person", "wrist"],
	]) {
		const [view, mark, move] = geometryDefs({ ...noRig, cameras: cams }).map((d) => d.parameters as any);
		assert.deepEqual(view.properties.views.items.enum, [
			"pointcloud_top",
			"pointcloud_front",
			"pointcloud_side",
			...cams,
		]);
		assert.deepEqual(mark.required, ["point_id", "view", "x", "y"]);
		for (const k of ["xyz", "point_id", "delta_mm", "delta_frame", "approach", "jaw", "gripper", "preview"])
			assert.ok(k in move.properties, `move_grip takes ${k}`);
		assert.deepEqual(move.properties.gripper.enum, ["open", "close"]);
		assert.deepEqual(move.properties.delta_frame.enum, ["world", "grip_site"]);
		assert.equal(move.required, undefined, "every field is optional");
	}
});

test("view_points and mark_point relay the env call and attach its images; errors are results", async () => {
	const env = fakeEnv({
		"env.point_views": { views: [{ view: "pointcloud_top" }], images: [PNG, PNG], grip_xyz_m: [0, 0, 0.3] },
		"env.mark_point": new Error("env.mark_point: view 'pointcloud_front' was not rendered for the current state"),
	});
	const [view, mark] = geometryDefs({
		call: env.call,
		cameras: ["agentview"],
		execute: async () => assert.fail("no motion"),
	});
	const r = await view.run({ views: ["pointcloud_top", "agentview"] }, undefined);
	assert.deepEqual(env.calls[0], { method: "env.point_views", kwargs: { views: ["pointcloud_top", "agentview"] } });
	assert.equal(images(r), 2);
	assert.ok(!("images" in text(r)), "the base64 images are not in the text");
	await view.run({}, undefined);
	assert.deepEqual(env.calls[1].kwargs, {});
	const m = await mark.run({ point_id: "P1", view: "pointcloud_front", x: 3, y: 4 }, undefined);
	assert.match(text(m).error, /not rendered for the current state/);
	assert.deepEqual(env.calls[2].kwargs, { point_id: "P1", view: "pointcloud_front", x: 3, y: 4 });
});

const plan = (over: Partial<GripPlan> = {}): GripPlan => ({
	status: "execute",
	motion: true,
	gripper: null,
	target: {
		grip_xyz_m: [0.1, 0, 0.2],
		approach_world: [0, 0, -1],
		jaw_world: [1, 0, 0],
		tool_quat_xyzw: [1, 0, 0, 0],
	},
	current: { grip_xyz_m: [0, 0, 0.3], approach_world: [0, 0, -1], jaw_world: [1, 0, 0], tool_quat_xyzw: [1, 0, 0, 0] },
	delta_mm: [100, 0, -100],
	rotation_deg: 0,
	...over,
});

test("move_grip previews without moving, executes a plan through the robot, and honours a refusal", async () => {
	const executed: GripPlan[] = [];
	let answer: Json = { ...plan({ status: "preview", gripper: "close" }), preview_id: "pv1234", images: [PNG, PNG] };
	const env = fakeEnv({ "env.grip_target": () => answer });
	let refusal: string | undefined;
	const [, , move] = geometryDefs({
		call: env.call,
		cameras: ["agentview"],
		refuse: () => refusal,
		execute: async (p) => {
			executed.push(p);
			return { content: [{ type: "text", text: "{}" }], details: {} };
		},
	});
	const r = await move.run({ delta_mm: [0, 0, -50], gripper: "close" }, undefined);
	assert.equal(text(r).motion_status, "previewed");
	assert.equal(text(r).preview_id, "pv1234");
	assert.equal(images(r), 2);
	assert.equal(executed.length, 0);
	assert.deepEqual(env.calls[0], { method: "env.grip_target", kwargs: { delta_mm: [0, 0, -50], gripper: "close" } });
	answer = plan({ gripper: "close", preview_id: "pv1234" });
	await move.run({ execute_preview_id: "pv1234" }, undefined);
	assert.equal(executed.length, 1);
	assert.equal(executed[0].preview_id, "pv1234");
	refusal = "Episode already ended (terminated=true, truncated=false).";
	const refused = await move.run({ xyz: [0, 0, 0.2] }, undefined);
	assert.match(text(refused).error, /already ended/);
	assert.equal(env.calls.length, 2, "a refused move is not even planned");
});

/** A simulated arm: action[:3] * 0.05 m and a world rotation vector action[3:6] * 0.1 rad per step. */
function arm(start: { pos: number[]; R: number[][] }, floorZ = -1) {
	const s = { pos: [...start.pos], R: start.R.map((r) => [...r]), width: 0.08, steps: 0, actions: [] as number[][] };
	const expm = (v: number[]) => {
		const t = Math.hypot(...v);
		if (t < 1e-12)
			return [
				[1, 0, 0],
				[0, 1, 0],
				[0, 0, 1],
			];
		const [x, y, z] = v.map((c) => c / t);
		const [c, sn, C] = [Math.cos(t), Math.sin(t), 1 - Math.cos(t)];
		return [
			[c + x * x * C, x * y * C - z * sn, x * z * C + y * sn],
			[y * x * C + z * sn, c + y * y * C, y * z * C - x * sn],
			[z * x * C - y * sn, z * y * C + x * sn, c + z * z * C],
		];
	};
	const mul = (a: number[][], b: number[][]) =>
		a.map((row) => [0, 1, 2].map((j) => row.reduce((acc, v, k) => acc + v * b[k][j], 0)));
	const quat = (R: number[][]) => {
		const w = Math.sqrt(Math.max(0, 1 + R[0][0] + R[1][1] + R[2][2])) / 2;
		if (w > 1e-6)
			return [(R[2][1] - R[1][2]) / (4 * w), (R[0][2] - R[2][0]) / (4 * w), (R[1][0] - R[0][1]) / (4 * w), w];
		const x = Math.sqrt(Math.max(0, 1 + R[0][0] - R[1][1] - R[2][2])) / 2;
		return [x, (R[0][1] + R[1][0]) / (4 * x), (R[0][2] + R[2][0]) / (4 * x), (R[2][1] - R[1][2]) / (4 * x)];
	};
	const io: ServoIo = {
		pose: async () => ({ pos: [...s.pos], quat: quat(s.R) }),
		step: async (a) => {
			s.actions.push(a);
			s.steps++;
			s.pos = s.pos.map((v, i) => v + a[i] * 0.05);
			s.pos[2] = Math.max(s.pos[2], floorZ);
			s.R = mul(expm(a.slice(3, 6).map((v) => v * 0.1)), s.R);
		},
		hold: () => -1,
		actuate: async (g) => {
			s.width = g > 0 ? 0.02 : 0.08;
			return 4;
		},
		ended: () => false,
	};
	return { s, io, quat };
}

const DOWN = [
	[1, 0, 0],
	[0, -1, 0],
	[0, 0, -1],
];

test("the servo turns and moves the tool frame together and stops within tolerance", async () => {
	const { s, io, quat } = arm({ pos: [0, 0, 0.3], R: DOWN });
	// Tilt the approach 30 degrees toward +x and yaw a quarter turn, 10 cm away.
	const goalR = [
		[0, 1, 0],
		[1, 0, 0],
		[0, 0, -1],
	].map((r) => [...r]);
	const tilt = (a: number) => [
		[Math.cos(a), 0, Math.sin(a)],
		[0, 1, 0],
		[-Math.sin(a), 0, Math.cos(a)],
	];
	const G = tilt(Math.PI / 6).map((row) => [0, 1, 2].map((j) => row.reduce((acc, v, k) => acc + v * goalR[k][j], 0)));
	const r = await servoGrip({ position: [0.1, 0.05, 0.2], quat: quat(G) }, io);
	assert.ok(r.reached, JSON.stringify(r));
	assert.ok(Math.hypot(...s.pos.map((v, i) => v - [0.1, 0.05, 0.2][i])) < 0.005);
	const err = rotvec(G.map((row) => [0, 1, 2].map((j) => row.reduce((acc, v, k) => acc + v * s.R[j][k], 0))));
	assert.ok(Math.hypot(...err) < 0.05);
	assert.ok(
		s.actions.every((a) => a.every((v) => Math.abs(v) <= 1)),
		"actions stay in [-1, 1]",
	);
	assert.ok(
		s.actions.every((a) => a[6] === -1),
		"the held gripper command",
	);
});

test("a plan that does not reach skips its gripper and reports the env's residual and contact image", async () => {
	const { s, io } = arm({ pos: [0, 0, 0.3], R: DOWN }, 0.1);
	const env = fakeEnv({
		"env.grip_state": (kw) => ({
			grip_xyz_m: s.pos,
			remaining_delta_mm: [0, 0, -100],
			motion_status: "not_reached",
			contacts: [{ xyz_m: [0, 0, 0.1], robot_geom: "gripper0_finger1", other_geom: "table" }],
			views: ["agentview_contacts"],
			images: [PNG],
			seen: kw,
		}),
	});
	const blocked = plan({
		gripper: "close",
		preview_id: "pv9",
		target: {
			grip_xyz_m: [0, 0, 0.0],
			approach_world: [0, 0, -1],
			jaw_world: [1, 0, 0],
			tool_quat_xyzw: [1, 0, 0, 0],
		},
	});
	const { result, pngs } = await runGripPlan(blocked, io, env.call, 40);
	assert.equal(result.motion_status, "not_reached");
	assert.equal(result.gripper, undefined);
	assert.match(result.gripper_skipped, /did not reach/);
	assert.equal(s.width, 0.08, "the fingers did not close");
	assert.equal(result.preview_id, "pv9");
	assert.equal(result.contacts[0].other_geom, "table");
	assert.equal(pngs.length, 1);
	assert.ok(!("images" in result));
	assert.deepEqual(env.calls[0].kwargs, { target_xyz: [0, 0, 0], target_approach: [0, 0, -1], target_jaw: [1, 0, 0] });
	// A gripper-only plan closes in place and asks grip_state without a target.
	const only = await runGripPlan(plan({ motion: false, gripper: "close" }), io, env.call);
	assert.equal(only.result.motion_status, "not_requested");
	assert.equal(only.result.gripper, "close");
	assert.equal(s.width, 0.02);
	assert.deepEqual(env.calls[1].kwargs, {});
});

test("a real arm's rotate_delta angles invert its extrinsic xyz Euler rotation", () => {
	const R = (a: number, b: number, c: number) => {
		const [ca, sa, cb, sb, cc, sc] = [Math.cos(a), Math.sin(a), Math.cos(b), Math.sin(b), Math.cos(c), Math.sin(c)];
		// Rz(c) Ry(b) Rx(a)
		return [
			[cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa],
			[sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa],
			[-sb, cb * sa, cb * ca],
		];
	};
	for (const angles of [
		[0.1, -0.2, 0.3],
		[-0.4, 0.05, -0.25],
	]) {
		const got = eulerXyz(R(angles[0], angles[1], angles[2]));
		for (const [i, v] of got.entries()) assert.ok(Math.abs(v - angles[i]) < 1e-9, `${got} vs ${angles}`);
	}
	// The plan's rotation is target * current^T in the base frame.
	const quarter = [0, 0, Math.SQRT1_2, Math.SQRT1_2]; // +90 deg about z
	const rot = planRotation(
		plan({
			current: { ...plan().current, tool_quat_xyzw: [0, 0, 0, 1] },
			target: { ...plan().target, tool_quat_xyzw: quarter },
		}),
	);
	const e = eulerXyz(rot);
	assert.ok(Math.abs(e[2] - Math.PI / 2) < 1e-9 && Math.abs(e[0]) < 1e-9 && Math.abs(e[1]) < 1e-9);
	assert.deepEqual(
		quatMatrix([0, 0, 0, 1]).map((r) => r.map((v) => Math.round(v))),
		[
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1],
		],
	);
});
