/**
 * Flywheel recording on the simulators whose env servers run the motion (Metaworld, Genesis,
 * Robosuite, ManiSkill): against fake env servers, one transition per control step the server
 * reports, with the robot's arrays, raw path, episode metadata and export space.
 */
import assert from "node:assert/strict";
import { mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { inflateRawSync } from "node:zlib";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import genesis from "../src/robots/genesis/index.ts";
import maniskill, { FLYWHEEL_ACTION, ROBOT_IDS, ROBOTS, VIEW_SETUP } from "../src/robots/maniskill/index.ts";
import metaworld from "../src/robots/metaworld/index.ts";
import robosuite from "../src/robots/robosuite/index.ts";
import { deployFlags } from "./helpers/deployment.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi (flags, tools, handlers, commands, exec) and a context without a UI. */
function stubPi(values: Record<string, unknown>) {
	values = deployFlags(values);
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const commands = new Map<string, any>();
	const exec: string[][] = [];
	let active: string[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: (name: string, c: any) => commands.set(name, c),
		registerProvider: () => {},
		registerMessageRenderer: () => {},
		registerShortcut: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: () => {},
		exec: async (cmd: string, args: string[]) => {
			exec.push([cmd, ...args]);
			return { code: 0, stdout: "{}", stderr: "" };
		},
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
		for (const fn of handlers.get(name) ?? []) await fn({ type: name, ...event }, ctx);
	}
	const run = async (name: string, params: unknown) =>
		(await tools.get(name).execute("id", params, undefined, undefined, ctx)) as any;
	const exportArgs = async () => {
		await commands.get("flywheel-export").handler("", ctx);
		process.exitCode = undefined;
		return exec[exec.length - 1];
	};
	return { pi, run, emit, exportArgs, active: () => active };
}

const nd = (dtype: string, shape: number[], data: Buffer) => ({ __ndarray__: data.toString("base64"), dtype, shape });
const f32 = (v: number[]) => nd("float32", [v.length], Buffer.from(Float32Array.from(v).buffer));
const image = (size: number, v = 0) => nd("uint8", [size, size, 3], Buffer.alloc(size * size * 3, v));

/** A fake env server (the wire protocol of ../src/rpc.ts): `answer` gives a method's result, else `{status: "ok"}`. */
async function fakeEnv(answer: (method: string, args: any[], kwargs: Record<string, any>) => unknown) {
	const calls: { method: string; args: any[]; kwargs: Record<string, any> }[] = [];
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, args = [], kwargs = {} } = JSON.parse(body);
			calls.push({ method, args, kwargs });
			let result =
				method === "code.api"
					? { tier: null, manifest_digest: "fake", available: [], digest: "d" }
					: answer(method, args, kwargs);
			result ??= { status: "ok" };
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

/** The `.npy` header text of each member of a deflated `.npz` (zip local headers, in order). */
function npzHeaders(zip: Buffer): Record<string, string> {
	const out: Record<string, string> = {};
	for (let o = 0; zip.readUInt32LE(o) === 0x04034b50; ) {
		const size = zip.readUInt32LE(o + 18);
		const nameLength = zip.readUInt16LE(o + 26);
		const extra = zip.readUInt16LE(o + 28);
		const name = zip.subarray(o + 30, o + 30 + nameLength).toString();
		const body = zip.subarray(o + 30 + nameLength + extra, o + 30 + nameLength + extra + size);
		out[name] = inflateRawSync(body).subarray(0, 128).toString("latin1");
		o += 30 + nameLength + extra + size;
	}
	return out;
}

/** The one episode written under `dir`: its episode.json and transitions.npz headers. */
function episode(dir: string) {
	const [id] = readdirSync(dir);
	return {
		meta: JSON.parse(readFileSync(join(dir, id, "episode.json"), "utf8")),
		arrays: npzHeaders(readFileSync(join(dir, id, "transitions.npz"))),
	};
}
const shape = (header: string) => header.match(/'shape': \(([^)]*)\)/)?.[1];

test("Metaworld: every control step of a motion is one transition, with the env action and success", async (t) => {
	const root = mkdtempSync(join(tmpdir(), "fly-metaworld-"));
	const obs = (z: number) => ({
		agentview: image(256),
		wrist: image(256),
		tcp_pos: f32([0, 0.6, z]),
		gripper_width: 0.09,
		obs: f32(new Array(39).fill(0)),
	});
	const env = await fakeEnv((method, args, kwargs) => {
		if (method === "env.get_env_meta")
			return { task: "reach-v3", seed: 3, agentview: "corner4", wrist: "gripperPOV", view_size: 256 };
		if (method === "env.reset") return [obs(0.2), {}];
		if (method === "env.get_task_language") return "reach the ball";
		if (method === "env.move_delta") {
			const n = (kwargs.delta_xyz ?? args[0])[2] < 0 ? 3 : 2;
			const frames = Array.from({ length: n }, (_, k) => ({
				...obs(0.2 - 0.01 * (k + 1)),
				action: f32([0, 0, -1, -1]),
				success: n === 3 && k === 2,
			}));
			return {
				ok: true,
				final_tcp_pos: [0, 0.6, 0.17],
				final_error_m: 0,
				moved_m: [0, 0, -0.03],
				gripper: "open",
				gripper_width: 0.09,
				steps_used: n,
				frames,
				info: { success: n === 3 },
			};
		}
	});
	t.after(env.close);
	const s = stubPi({
		"env-url": env.url,
		task: "reach-v3",
		seed: "3",
		"collect-flywheel-data": true,
		"flywheel-root": root,
	});
	metaworld(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.ok(s.active().includes("move_delta"));
	await s.run("move_delta", { delta_xyz: [0, 0.02, 0] });
	await s.run("move_delta", { delta_xyz: [0, 0, -0.03] });
	await s.emit("session_shutdown");
	const { meta, arrays } = episode(join(root, "raw", "metaworld", "reach-v3", "seed_003"));
	assert.deepEqual([meta.step_count, meta.training_step_count, meta.is_success], [5, 5, true]);
	assert.deepEqual([meta.task, meta.seed, meta.task_language], ["reach-v3", 3, "reach the ball"]);
	assert.equal(shape(arrays["actions.npy"]), "5, 4");
	assert.equal(shape(arrays["states.npy"]), "6, 4");
	assert.equal(shape(arrays["agentview_images.npy"]), "6, 256, 256, 3");
	assert.equal(shape(arrays["wrist_images.npy"]), "6, 256, 256, 3");
	const argv = await s.exportArgs();
	assert.deepEqual(argv.slice(argv.indexOf("--robot")), ["--robot", "metaworld", "--select", "reach-v3"]);
});

test("Genesis: a motion returns its control steps only while recording, one transition each", async (t) => {
	const root = mkdtempSync(join(tmpdir(), "fly-genesis-"));
	const obs = (success = false) => ({
		agentview: image(4),
		wrist: image(4),
		tcp_pos: f32([0.5, 0, 0.3]),
		tcp_quat_wxyz: f32([0, 1, 0, 0]),
		gripper_width: 0.08,
		gripper_command: "open",
		qpos: f32(new Array(9).fill(0)),
		success,
		is_grasped: false,
		lift_m: 0,
		env_steps: 0,
	});
	const steps = (n: number) => Array.from({ length: n }, () => ({ ...obs(), action: f32([0.01, 0, 0, 1]) }));
	const env = await fakeEnv((method, _args, kwargs) => {
		if (method === "env.get_env_meta") return { task: "cube_pick", seed: 0, instruction: "lift the cube" };
		if (method === "env.reset") return [obs(), {}];
		if (method === "env.move_delta")
			return {
				...obs(),
				commanded_m: [0.02, 0, 0],
				moved_m: [0.02, 0, 0],
				decisions: 1,
				control_steps: 4,
				frames: [],
				...(kwargs.record ? { steps: steps(4) } : {}),
			};
		if (method === "env.set_gripper")
			return { ...obs(), control_steps: 3, frames: [], ...(kwargs.record ? { steps: steps(3) } : {}) };
	});
	t.after(env.close);
	for (const collect of [false, true]) {
		const s = stubPi({ "env-url": env.url, "collect-flywheel-data": collect, "flywheel-root": root });
		genesis(s.pi);
		await s.emit("session_start");
		process.exitCode = undefined;
		const before = env.calls.length;
		await s.run("move_delta", { delta_xyz: [0.02, 0, 0] });
		await s.run("set_gripper", { close: true });
		const motions = env.calls.slice(before).filter((c) => c.method !== "code.api");
		assert.deepEqual(
			motions.map((c) => c.kwargs.record),
			collect ? [true, true] : [undefined, undefined],
		);
		await s.emit("session_shutdown");
	}
	const { meta, arrays } = episode(join(root, "raw", "genesis", "cube_pick", "seed_000"));
	assert.deepEqual([meta.step_count, meta.is_success, meta.task_language], [7, false, "lift the cube"]);
	assert.equal(shape(arrays["actions.npy"]), "7, 4");
	assert.equal(shape(arrays["states.npy"]), "8, 8");
	assert.equal(shape(arrays["agentview_images.npy"]), "8, 4, 4, 3");
});

test("Robosuite: the task's arms set the widths and the export space; the reset is rendered at the recorded size", async (t) => {
	const root = mkdtempSync(join(tmpdir(), "fly-robosuite-"));
	const obs = (size: number, success = false) => ({
		agentview: image(size),
		wrist: image(size),
		...Object.fromEntries(
			["robot0", "robot1"].flatMap((a) => [
				[`${a}_eef_pos`, f32([0, 0, 1])],
				[`${a}_eef_quat`, f32([0, 0, 0, 1])],
				[`${a}_gripper_qpos`, f32([0.04, -0.04])],
				[`${a}_gripper_width`, 0.08],
				[`${a}_gripper_command`, "open"],
			]),
		),
		success,
		success_step: null,
		truncated: false,
		env_steps: 0,
	});
	const env = await fakeEnv((method, _args, kwargs) => {
		if (method === "env.get_env_meta") return { task: "TwoArmLift", seed: 1, table_z: 0.8, max_move_m: 0.3 };
		if (method === "env.reset") return [obs(8), {}];
		if (method === "env.get_task_language") return "lift the pot";
		if (method === "env.render_camera") return image(kwargs.height);
		if (method === "env.move_delta" || method === "env.set_gripper")
			return {
				obs: obs(8),
				info: {
					ok: true,
					steps_used: 2,
					frames: [],
					...(kwargs.record
						? { steps: [0, 1].map((k) => ({ ...obs(256, k === 1), action: f32(new Array(14).fill(0.5)) })) }
						: {}),
				},
			};
	});
	t.after(env.close);
	const s = stubPi({
		"env-url": env.url,
		task: "TwoArmLift",
		seed: "1",
		"collect-flywheel-data": true,
		"flywheel-root": root,
	});
	robosuite(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	await s.run("move_delta", { delta_xyz: [0, 0, 0.05], arm: "robot1" });
	await s.run("set_gripper", { close: true, arm: "robot0" });
	await s.emit("session_shutdown");
	assert.deepEqual(
		env.calls.filter((c) => c.method === "env.render_camera").map((c) => [c.kwargs.camera_name, c.kwargs.height]),
		[
			["agentview", 256],
			["wrist", 256],
		],
	);
	const { meta, arrays } = episode(join(root, "raw", "robosuite", "TwoArmLift", "seed_001"));
	assert.deepEqual([meta.step_count, meta.training_step_count, meta.space], [4, 2, "two_arm"]);
	assert.equal(shape(arrays["actions.npy"]), "4, 14");
	assert.equal(shape(arrays["states.npy"]), "5, 18");
	assert.equal(shape(arrays["wrist_images.npy"]), "5, 256, 256, 3");
	const argv = await s.exportArgs();
	assert.deepEqual(argv.slice(argv.indexOf("--robot")), [
		"--robot",
		"robosuite",
		"--select",
		"TwoArmLift",
		"--space",
		"two_arm",
	]);
});

test("ManiSkill: each control step env.move_delta returns is a transition; the arm is the raw path's first part and the export space, and a wrist-less arm records the agentview alone", async (t) => {
	const root = mkdtempSync(join(tmpdir(), "fly-maniskill-"));
	for (const [arm, wrist] of [
		["widowxai", false],
		["panda", true],
	] as const) {
		const obs = (z: number) => ({
			agentview: image(4),
			...(wrist ? { wrist: image(4) } : {}),
			tcp_pos: f32([-0.4, 0, z]),
			tcp_quat_wxyz: f32([0, 1, 0, 0]),
			gripper_width: 0.08,
			qpos: f32([0, 0]),
		});
		const env = await fakeEnv((method, _args, kwargs) => {
			if (method === "env.get_env_meta")
				return {
					env_id: "PickCube-v1",
					seed: 2,
					scene: null,
					table_z: 0,
					view_size: 256,
					...(wrist ? VIEW_SETUP : ROBOTS.widowxai.setup),
					...(arm === "panda" ? {} : { robot: arm }),
				};
			if (method === "env.reset") return [obs(0.08), {}];
			if (method === "env.get_task_language") return "pick up the cube";
			if (method === "env.move_delta") {
				// The server's two 2 cm legs of 2 control steps each (a leg that ran no step has no action).
				const frames = [
					...Array.from({ length: 4 }, (_, i) => ({
						...obs(0.08 + kwargs.delta_xyz[2] * ((i + 1) / 4)),
						action: f32([0, 0, -0.5, 1]),
						success: false,
					})),
					obs(0.04),
				];
				return {
					result: { commanded_m: kwargs.delta_xyz, moved_m: kwargs.delta_xyz, gripper: "open", env_steps: 4 },
					frames,
					obs: frames[4],
					info: { is_grasped: false },
				};
			}
		});
		t.after(env.close);
		const s = stubPi({
			"env-url": env.url,
			task: "PickCube-v1",
			seed: "2",
			arm: arm,
			"collect-flywheel-data": true,
			"flywheel-root": root,
		});
		maniskill(s.pi);
		await s.emit("session_start");
		process.exitCode = undefined;
		await s.run("move_delta", { delta_xyz: [0, 0, -0.04] });
		const servoSteps = 4;
		await s.emit("session_shutdown");
		const { meta, arrays } = episode(join(root, "raw", "maniskill", arm, "PickCube-v1", "default", "seed_002"));
		assert.equal(meta.step_count, servoSteps);
		assert.deepEqual([meta.maniskill_robot, meta.env_id, meta.scene], [arm, "PickCube-v1", ""]);
		assert.equal(shape(arrays["actions.npy"]), `${servoSteps}, 4`);
		assert.equal(shape(arrays["states.npy"]), `${servoSteps + 1}, 8`);
		assert.equal("wrist_images.npy" in arrays, wrist);
		const argv = await s.exportArgs();
		assert.deepEqual(argv.slice(argv.indexOf("--select")), [
			"--select",
			`${arm}/PickCube-v1/default`,
			"--space",
			arm,
		]);
	}
});

test("ManiSkill: every --robot has a Flywheel action width", () => {
	assert.deepEqual(Object.keys(FLYWHEEL_ACTION).sort(), [...ROBOT_IDS].sort());
});
