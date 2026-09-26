import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, realpathSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import ur5e, { cameraMount, hasWristCamera, UR5E_UNITS } from "../src/ur5e/index.ts";

type Handler = (event: any, ctx: any) => unknown;

/** The values of an enum schema (StringEnum or a union of literals). */
const enumOf = (schema: any): string[] => schema.enum ?? schema.anyOf.map((u: any) => u.const);

/** A stub pi (as in piper.test.ts); `confirm` answers from `confirms`. */
function fakePi(flagValues: Record<string, unknown> = {}, hasUI = true) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const notes: string[] = [];
	const confirms: boolean[] = [];
	const dialogs: string[] = [];
	const dialogBodies: string[] = [];
	let active: string[] = ["stale"];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const dir = realpathSync(mkdtempSync(join(tmpdir(), "ur5e-")));
	let shutdown = false;
	const ctx = {
		hasUI,
		cwd: dir,
		ui: {
			notify: (m: string) => notes.push(m),
			setWidget: () => {},
			input: async () => "",
			select: async () => undefined,
			confirm: async (title: string, body?: string) => {
				dialogs.push(title);
				dialogBodies.push(body ?? "");
				return confirms.shift() ?? false;
			},
		},
		shutdown: () => {
			shutdown = true;
		},
		sessionManager: {
			getBranch: () => [],
			getSessionDir: () => dir,
			getSessionFile: () => undefined,
			getSessionId: () => "s",
		},
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) {
			const r: any = await fn({ type: name, ...event }, ctx);
			if (r !== undefined) result = r;
			if (r && name === "tool_call" && r.block) return r;
		}
		return result;
	}
	const run = (name: string, params: Record<string, unknown> = {}) =>
		tools.get(name).execute("id", params, undefined, undefined, ctx);
	return {
		pi,
		emit,
		run,
		tools,
		notes,
		confirms,
		dialogs,
		dialogBodies,
		dir,
		active: () => active,
		shut: () => shutdown,
	};
}

async function start(f: ReturnType<typeof fakePi>) {
	const errors: string[] = [];
	const log = console.error;
	console.error = (...a: unknown[]) => errors.push(a.join(" "));
	try {
		await f.emit("session_start");
	} finally {
		console.error = log;
	}
	const code = process.exitCode;
	process.exitCode = 0;
	return { errors, code };
}

const ARM = "2023300001";

/**
 * A mocked UR5e env server transport (the JSON RPC of ../src/rpc.ts): canned state, a 4x4 wrist
 * RGB-D frame and a 4x4 RGB-only front frame, an identity wrist calibration; records every call.
 */
async function mockServer(
	o: {
		armId?: string | null;
		robot?: string;
		beginPose?: boolean;
		/** The camera metadata per camera (default: an eye-in-hand wrist and an uncalibrated front camera). */
		cameraMeta?: Record<string, Record<string, unknown>>;
	} = {},
) {
	const calls: { method: string; kwargs: Record<string, any> }[] = [];
	const img = { __ndarray__: Buffer.alloc(4 * 4 * 3, 90).toString("base64"), dtype: "uint8", shape: [4, 4, 3] };
	const depth = {
		__ndarray__: Buffer.from(Float32Array.from(Array(16).fill(0.5)).buffer).toString("base64"),
		dtype: "float32",
		shape: [4, 4],
	};
	const tcp = [0.4, 0.1, 0.3, 1, 0, 0, 0]; // pointing down: 180 deg about x
	const state = () => ({
		raw_base_state: {
			tcp_pose: tcp,
			joints: [1.57, -1.57, 1.57, -1.57, -1.57, 0],
			setpoint_pose: tcp,
			gripper_position: [0.085],
			gripper_open: true,
			gripper_grasped: false,
			z_floor_m: 0.14,
		},
		backend: "ur_rtde",
		arm_id: o.armId === undefined ? ARM : o.armId,
	});
	const K = [
		[2, 0, 2],
		[0, 2, 2],
		[0, 0, 1],
	];
	const I4 = [
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1],
	];
	const answer = (method: string, kwargs: Record<string, any>): unknown => {
		switch (method) {
			case "healthz":
				return { status: "ok" };
			case "code.api":
				return { tier: null, primitives: [], digest: "0".repeat(64) };
			case "env.get_env_meta":
				return {
					ok: true,
					robot: o.robot ?? "ur5e",
					backend: "ur_rtde",
					arm_id: o.armId === undefined ? ARM : o.armId,
					cameras: ["front", "wrist"],
					main_camera: "wrist",
					gripper: "robotiq",
					has_begin_pose: o.beginPose ?? true,
					limits: {
						max_move_m: 0.08,
						max_rotate_rad: 0.2,
						z_floor_m: 0.14,
						empty_width_m: 0.011,
						reset_lift_m: 0.05,
					},
					tasks: { block_bowl: { instruction: "put the block in the bowl" } },
				};
			case "env.reset":
				return { ok: true, gripper: { ok: true }, move: { ok: true }, robot_state: state() };
			case "env.get_observation":
				return { images: { wrist: img, front: img }, depths: { wrist: depth }, timestamps: { wrist: 1, front: 1 } };
			case "env.get_robot_state":
				return state();
			case "env.get_camera_meta":
				if (o.cameraMeta)
					return { cameras: o.cameraMeta, observation_camera_map: { main: "wrist", extra_0: "front" } };
				return {
					cameras: {
						wrist: {
							name: "wrist",
							has_depth: true,
							intrinsic_K: K,
							extrinsic: { frame: "tcp", matrix: I4, path: "w.yaml", arm_id: ARM },
						},
						front: { name: "front", has_depth: false, intrinsic_K: null, extrinsic: null },
					},
					observation_camera_map: { main: "wrist", extra_0: "front" },
				};
			case "env.move_delta":
			case "env.move_pose":
			case "env.rotate_delta":
				return { ok: true, final_tcp_pose: tcp, states: null };
			case "env.set_gripper":
				return kwargs.open
					? { ok: true, gripper_width_m: 0.085 }
					: { ok: false, gripper_jammed: true, note: "gripper jammed: the fingers stayed" };
			default:
				throw new Error(`unexpected ${method}`);
		}
	};
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, kwargs } = JSON.parse(body) as { method: string; kwargs: Record<string, any> };
			if (method !== "healthz") calls.push({ method, kwargs });
			try {
				res.end(JSON.stringify({ ok: true, result: answer(method, kwargs) }));
			} catch (err) {
				res.end(JSON.stringify({ ok: false, error: String(err) }));
			}
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
	const close = () => {
		server.closeAllConnections();
		server.close();
	};
	return { url, calls, close, methods: () => calls.map((c) => c.method) };
}

async function started(
	flags: Record<string, unknown> = {},
	server: Parameters<typeof mockServer>[0] = {},
	confirms = [true],
) {
	const m = await mockServer(server);
	const f = fakePi({ operator: true, "arm-id": ARM, task: "block_bowl", "robot-env": m.url, ...flags });
	ur5e(f.pi);
	f.confirms.push(...confirms);
	const s = await start(f);
	return { f, m, s };
}

test("ur5e refuses to start without an operator UI and stays tool-less", async () => {
	const f = fakePi({ operator: true, "arm-id": ARM }, false);
	ur5e(f.pi);
	const { errors, code } = await start(f);
	assert.equal(code, 1);
	assert.match(errors.join("\n"), /\[ur5e\] unavailable: ur5e drives a real robot: run pi interactively/);
	assert.deepEqual(f.active(), []);
	assert.ok(f.shut());
	const blocked = await f.emit("tool_call", { toolName: "move_delta" });
	assert.equal(blocked.block, true);
	assert.match(blocked.reason, /ur5e is not available/);
});

test("ur5e refuses to start without --operator or --arm-id, before touching the robot", async () => {
	const f = fakePi({ python: "/nonexistent/python", "arm-id": ARM });
	ur5e(f.pi);
	await start(f);
	assert.match(f.notes.join("\n"), /ur5e unavailable: .*start pi with --operator/);
	assert.deepEqual(f.active(), []);
	const g = fakePi({ python: "/nonexistent/python", operator: true });
	ur5e(g.pi);
	await start(g);
	assert.match(g.notes.join("\n"), /ur5e unavailable: ur5e is bound to one arm: start pi with --arm-id/);
	assert.deepEqual(g.active(), []);
	const h = fakePi({ python: "/nonexistent/python", operator: true, "arm-id": ARM, "max-move": "abc" });
	ur5e(h.pi);
	await start(h);
	assert.match(h.notes.join("\n"), /--max-move must be a positive number/);
});

test("--arm-id must be the arm the env server is bound to; nothing moves otherwise", async () => {
	const other = await started({ "arm-id": "1999" });
	try {
		assert.match(
			other.f.notes.join("\n"),
			/--arm-id 1999 is not the arm the env server drives \(its config is bound to 2023300001\)/,
		);
		assert.deepEqual(other.f.active(), []);
		assert.ok(!other.m.methods().includes("env.reset"), "nothing moved");
	} finally {
		other.m.close();
	}
	const unbound = await started({}, { armId: null });
	try {
		assert.match(unbound.f.notes.join("\n"), /bound its config to no arm .*--arm-id cannot be verified/);
		assert.ok(!unbound.m.methods().includes("env.reset"));
	} finally {
		unbound.m.close();
	}
	const wrong = await started({}, { robot: "piper" });
	try {
		assert.match(wrong.f.notes.join("\n"), /serves piper, not a ur5e env server/);
	} finally {
		wrong.m.close();
	}
	const noPose = await started({}, { beginPose: false });
	try {
		assert.match(noPose.f.notes.join("\n"), /begin_joints is not set/);
		assert.ok(!noPose.m.methods().includes("env.reset"));
	} finally {
		noPose.m.close();
	}
	const declined = await started({}, {}, [false]);
	try {
		assert.deepEqual(declined.f.dialogs, [`Move the UR5e arm ${ARM}?`]);
		// The operator is told everything the reset does: release, lift, joint move.
		const body = declined.f.dialogBodies[0];
		assert.match(body, /released where it is/);
		assert.match(body, /lift 50 mm straight up/);
		assert.match(body, /moveJ to its begin pose/);
		assert.match(declined.f.notes.join("\n"), /operator declined the reset/);
		assert.ok(!declined.m.methods().includes("env.reset"));
	} finally {
		declined.m.close();
	}
});

test("motion beyond the per-call limits is refused before it reaches the robot", async () => {
	const f = fakePi({ operator: true, "arm-id": ARM });
	ur5e(f.pi);
	const far = await f.run("move_delta", { delta_xyz: [0.06, 0.06, 0] });
	assert.match(far.details.error, /moves 0\.0849 m; the limit is 0\.08 m per call/);
	const turn = await f.run("rotate_delta", { delta_rpy: [0, 0, 0.3] });
	assert.match(turn.details.error, /the rotation is 0\.3 rad; the limit is 0\.2 rad per call/);
	const nan = await f.run("move_delta", { delta_xyz: [Number.NaN, 0, 0] });
	assert.match(nan.details.error, /finite/);
	// Within the limits the call proceeds to the (absent) robot.
	const ok = await f.run("move_delta", { delta_xyz: [0.02, 0, 0] });
	assert.match(ok.details.error, /ur5e is not initialized/);
	const tight = fakePi({ operator: true, "arm-id": ARM, "max-move": "0.01" });
	ur5e(tight.pi);
	const unit = await tight.run("move_delta", { delta_xyz: [0, 0, -0.02] });
	assert.match(unit.details.error, /the limit is 0\.01 m per call/);
});

test("tool schemas: gripper action enum, move_pose rotvec or rpy, camera flags", () => {
	const f = fakePi({ operator: true, "arm-id": ARM });
	ur5e(f.pi);
	assert.deepEqual(enumOf(f.tools.get("gripper").parameters.properties.action), ["open", "close"]);
	const pose = f.tools.get("move_pose").parameters;
	assert.deepEqual(pose.required, ["xyz"]);
	assert.deepEqual(Object.keys(pose.properties).sort(), ["rotvec", "rpy", "xyz"]);
	assert.ok(f.tools.has("back_project") && f.tools.has("view_camera_meta"));
	assert.ok(f.tools.has("segment"), "registered, activated only with --robot-sam3");
	assert.equal(UR5E_UNITS.stepM, 0.02);
	assert.equal(UR5E_UNITS.yawStepRad, 0.15);
	assert.deepEqual(UR5E_UNITS.vectors.MV_LEFT, [0, 1, 0]);
});

test("a started ur5e resets once, records steps and back-projects through depth and the wrist calibration", async () => {
	const { f, m } = await started();
	try {
		assert.deepEqual(m.methods().slice(0, 2), ["env.get_env_meta", "env.reset"]);
		assert.deepEqual(f.active(), [
			"view_env_state",
			"view_camera_meta",
			"back_project",
			"move_delta",
			"move_pose",
			"rotate_delta",
			"gripper",
			"finish",
			"request_operator_verdict",
			"request_scene_reset",
		]);
		assert.match(f.notes.join("\n"), /UR5e 2023300001 ready: block_bowl; cameras wrist, front/);
		const s0 = await f.run("view_env_state", {});
		assert.equal(s0.details.step_idx, 0);
		assert.equal(s0.content.filter((c: any) => c.type === "image").length, 2, "wrist then front");
		assert.ok(s0.details.artifacts.includes("wrist_depth.f32") && !s0.details.artifacts.includes("front_depth.f32"));

		// Pixel (2, 3) at 0.5 m through K = [[2,0,2],[0,2,2]]: camera point (0.25, 0, 0.5); identity T_tcp_cam;
		// the TCP points down at (0.4, 0.1, 0.3): base point (0.65, 0.1, -0.2).
		const bp = await f.run("back_project", { row: 2, col: 3 });
		assert.deepEqual(bp.details.point_camera, [0.25, 0, 0.5]);
		assert.deepEqual(bp.details.point_base, [0.65, 0.1, -0.2]);
		assert.equal(bp.details.target_frame, "tcp");
		assert.ok(readFileSync(bp.details.selected_pixel_overlay).length > 0);
		const rgbOnly = await f.run("back_project", { row: 1, col: 1, camera: "front" });
		assert.match(rgbOnly.details.error, /camera front has no depth \(RGB-only source\)/);
		const unknown = await f.run("back_project", { row: 1, col: 1, camera: "side" });
		assert.match(unknown.details.error, /unknown camera 'side'/);

		const before = m.calls.length;
		const mv = await f.run("move_delta", { delta_xyz: [0.02, 0, 0] });
		assert.equal(mv.details.step_idx, 1);
		assert.deepEqual(m.calls[before], { method: "env.move_delta", kwargs: { delta_xyz: [0.02, 0, 0] } });
		const pose = await f.run("move_pose", { xyz: [0.42, 0.1, 0.3], rpy: [0, 0, 0.1] });
		assert.deepEqual(m.calls.find((c) => c.method === "env.move_pose")?.kwargs, {
			xyz: [0.42, 0.1, 0.3],
			rpy: [0, 0, 0.1],
		});
		assert.equal(pose.details.step_idx, 2);
		const farPose = await f.run("move_pose", { xyz: [0.6, 0.1, 0.3] });
		assert.match(farPose.details.error, /the limit is 0\.08 m per call/);
		const both = await f.run("move_pose", { xyz: [0.41, 0.1, 0.3], rpy: [0, 0, 0], rotvec: [0, 0, 0] });
		assert.match(both.details.error, /rotvec or rpy, not both/);
		assert.equal(m.calls.filter((c) => c.method === "env.move_pose").length, 1, "refusals send nothing");
		const grip = await f.run("gripper", { action: "close" });
		assert.equal(grip.details.gripper_jammed, true);
		assert.match(grip.details.gripper_note, /gripper jammed/);
		assert.deepEqual(m.calls.find((c) => c.method === "env.set_gripper")?.kwargs, { open: false });
	} finally {
		m.close();
	}
});

test("a camera's mount: its config, else its calibration's frame; all fixed means no wrist view", async () => {
	assert.equal(cameraMount({ mount: "wrist" }), "wrist");
	assert.equal(cameraMount({ mount: "fixed", extrinsic: { frame: "tcp" } }), "fixed", "the config's mount wins");
	assert.equal(cameraMount({ mount: null, extrinsic: { frame: "base" } }), "fixed");
	assert.equal(cameraMount({ extrinsic: { frame: "tcp" } }), "wrist");
	assert.equal(cameraMount({ extrinsic: null }), null);
	assert.equal(cameraMount(undefined), null);
	assert.equal(hasWristCamera(["fixed", "fixed"]), false);
	assert.equal(hasWristCamera(["fixed", "wrist"]), true);
	assert.equal(hasWristCamera(["fixed", null]), true, "an unknown mount may be a wrist camera");
	assert.equal(hasWristCamera([]), true, "cameras not known yet");

	// Two fixed cameras (no `mount` set, calibrated to the base): act without target_in_wrist.
	const base = {
		has_depth: false,
		intrinsic_K: null,
		extrinsic: { frame: "base", matrix: [], path: "c.yaml", arm_id: ARM },
	};
	const fixed = await started(
		{ units: true, "units-plugins": "auto" },
		{ cameraMeta: { wrist: { name: "wrist", ...base }, front: { name: "front", ...base } } },
	);
	try {
		assert.ok(fixed.f.active().includes("act"));
		assert.ok(!("target_in_wrist" in fixed.f.tools.get("act").parameters.properties));
	} finally {
		fixed.m.close();
	}
	// The default rig: an eye-in-hand camera keeps it.
	const rig = await started({ units: true, "units-plugins": "auto" });
	try {
		assert.ok("target_in_wrist" in rig.f.tools.get("act").parameters.properties);
	} finally {
		rig.m.close();
	}
});
