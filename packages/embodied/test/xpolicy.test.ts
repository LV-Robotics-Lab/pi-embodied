import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { NdArray } from "../src/infra/rpc.ts";
import { loadManifest } from "../src/primitives/manifest.ts";
import {
	gripperCommand,
	parseAction,
	poseDelta,
	toWxyz,
	toXyzw,
	wireObs,
	XPOLICY_ENTRY,
	type XPolicyAction,
	type XPolicySpec,
} from "../src/primitives/xpolicy.ts";
import { defineRobot, RESULT_ENTRY, type RobotSpec } from "../src/robot.ts";
import { deployFlags } from "./helpers/deployment.ts";

const DUAL = { arm_dim: [6, 6], ee_dim: [1, 1] };
const SINGLE = { arm_dim: [7], ee_dim: [1] };
const close = (a: number[], b: number[]) => a.every((v, i) => Math.abs(v - b[i]) < 1e-9);

test("parseAction reads action dicts per arm, with action_type overriding the flag", () => {
	const a = parseAction(
		{
			left_arm_joint_state: [0, 1, 2, 3, 4, 5],
			left_ee_joint_state: 0.5,
			right_arm_joint_state: [6, 7, 8, 9, 10, 11],
			right_ee_joint_state: [1],
		},
		DUAL,
		"joint",
	);
	assert.deepEqual(a, {
		type: "joint",
		arms: {
			left_: { joints: [0, 1, 2, 3, 4, 5], ee: [0.5] },
			right_: { joints: [6, 7, 8, 9, 10, 11], ee: [1] },
		},
	});
	// One arm only: the other keeps its command (the robot fills it in).
	const ee = parseAction({ action_type: "endpose", right_ee_pose: [1, 2, 3, 1, 0, 0, 0] }, DUAL, "joint");
	assert.deepEqual(ee, { type: "ee", arms: { right_: { pose: [1, 2, 3, 1, 0, 0, 0] } } });
	// Single-arm robots use unprefixed keys.
	assert.deepEqual(parseAction({ ee_pose: [0, 0, 0, 1, 0, 0, 0], ee_joint_state: [0.02] }, SINGLE, "ee").arms, {
		"": { pose: [0, 0, 0, 1, 0, 0, 0], ee: [0.02] },
	});
});

test("parseAction splits flat vectors in the packed [arm, ee] per-arm order", () => {
	const v = Array.from({ length: 14 }, (_, i) => i);
	assert.deepEqual(parseAction(v, DUAL, "joint").arms, {
		left_: { joints: [0, 1, 2, 3, 4, 5], ee: [6] },
		right_: { joints: [7, 8, 9, 10, 11, 12], ee: [13] },
	});
	assert.deepEqual(
		parseAction(
			Array.from({ length: 8 }, (_, i) => i),
			SINGLE,
			"ee",
		).arms,
		{
			"": { pose: [0, 1, 2, 3, 4, 5, 6], ee: [7] },
		},
	);
	assert.throws(() => parseAction(v.slice(1), DUAL, "joint"), /must hold 14 finite values/);
});

test("parseAction refuses wrong dimensions, non-finite values and actions without their main key", () => {
	assert.throws(() => parseAction({ left_arm_joint_state: [1, 2, 3] }, DUAL, "joint"), /must hold 6 finite values/);
	assert.throws(
		() => parseAction({ left_arm_joint_state: [0, 0, 0, 0, 0, Number.NaN] }, DUAL, "joint"),
		/finite values/,
	);
	assert.throws(
		() => parseAction({ left_ee_joint_state: [1] }, DUAL, "joint"),
		/needs left_arm_joint_state or right_arm_joint_state/,
	);
	assert.throws(() => parseAction({ arm_joint_state: [0, 0, 0, 0, 0, 0, 0] }, SINGLE, "ee"), /needs ee_pose/);
	assert.throws(() => parseAction({ action_type: "torque" }, SINGLE, "ee"), /unknown XPolicyLab action_type/);
});

test("wireObs builds the v1.0 observation and checks the state against the robot's dims", () => {
	const rgba = new NdArray("uint8", [2, 2, 4], Buffer.alloc(16, 7));
	const obs = wireObs(
		{
			instruction: "beat the block",
			vision: {
				cam_head: {
					color: rgba,
					intrinsic_matrix: [
						[1, 0, 0],
						[0, 1, 0],
						[0, 0, 1],
					],
				},
			},
			state: { left_arm_joint_state: [0, 0, 0, 0, 0, 0], left_ee_joint_state: [1] },
			info: { frequency: 30 },
		},
		DUAL,
	);
	assert.equal(obs.data_format_version, "v1.0");
	assert.deepEqual(obs.instructions, ["beat the block"]);
	assert.deepEqual(obs.additional_info, { frequency: 30 });
	const cam = obs.vision.cam_head;
	assert.deepEqual(cam.color.shape, [2, 2, 3], "RGBA is cut to RGB");
	assert.deepEqual(cam.shape, [2, 2]);
	assert.deepEqual(cam.intrinsic_matrix.shape, [3, 3]);
	assert.equal(cam.intrinsic_matrix.dtype, "float32");
	assert.equal(obs.state.left_arm_joint_state.dtype, "float32");
	assert.throws(
		() => wireObs({ instruction: "", vision: {}, state: { right_arm_joint_state: [0, 0] } }, DUAL),
		/right_arm_joint_state must hold 6/,
	);
	assert.throws(() => wireObs({ instruction: "", vision: {}, state: { ee_pose: [0, 0, 0] } }, SINGLE), /ee_pose/);
});

test("poseDelta, quaternion order and gripperCommand", () => {
	const xyzw = [1, 2, 3, 0.1, 0.2, 0.3, 0.9];
	assert.deepEqual(toXyzw(toWxyz(xyzw)), xyzw);
	const h = Math.SQRT1_2;
	// A quarter turn about z from the identity, and a 1 cm move.
	const d = poseDelta([0, 0, 0, 1, 0, 0, 0], [0.01, 0, 0, h, 0, 0, h]);
	assert.ok(close(d.delta, [0.01, 0, 0]));
	assert.ok(close(d.rpy, [0, 0, Math.PI / 2]));
	// Relative to a rotated start the delta is the difference.
	const e = poseDelta([0, 0, 0, h, 0, 0, h], [0, 0, 0, 1, 0, 0, 0]);
	assert.ok(close(e.rpy, [0, 0, -Math.PI / 2]));
	assert.equal(gripperCommand(0.01, 0.08, false), "close");
	assert.equal(gripperCommand(0.01, 0.08, true), null);
	assert.equal(gripperCommand(0.07, 0.08, true), "open");
	assert.equal(gripperCommand(0.07, 0.08, false), null);
});

// ---------------------------------------------------------------------------
// the module on a toy robot, against a fake bridge

type Call = { method: string; kwargs: Record<string, any> };

/** A fake xpolicy bridge (`POST /call`): `chunks` are get_action's replies in order. */
async function fakeBridge(chunks: unknown[][], failOn?: string) {
	const calls: Call[] = [];
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, kwargs } = JSON.parse(body) as Call;
			if (method !== "healthz") calls.push({ method, kwargs });
			const reply = (result: unknown) => res.end(JSON.stringify({ ok: true, result }));
			if (method === failOn) return res.end(JSON.stringify({ ok: false, error: `${method}: WsError (timeout): x` }));
			if (method === "healthz") return reply({ status: "ok" });
			if (method === "xpolicy.action_dims") return reply({ robot: "toy", ...DUAL });
			if (method === "xpolicy.connect")
				return reply({ server_instance_id: "srv-1", xpolicylab_rev: "d6332bf10b15", ms: 1 });
			if (method === "xpolicy.get_action") return reply({ actions: chunks.shift() ?? [], ms: 5 });
			reply({ result: null, ms: 1 });
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

function fakePi(flagValues: Record<string, unknown>) {
	flagValues = deployFlags(flagValues);
	const handlers = new Map<string, ((event: any, ctx: any) => unknown)[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	let active: string[] = [];
	const bus = new EventEmitter();
	const pi = {
		on: (name: string, fn: (event: any, ctx: any) => unknown) =>
			handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		setActiveTools: (t: string[]) => {
			active = t;
		},
		getActiveTools: () => active,
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		getThinkingLevel: () => "low",
		events: {
			emit: (channel: string, data: unknown) => bus.emit(channel, data),
			on: (channel: string, fn: (data: unknown) => void) => {
				bus.on(channel, fn);
				return () => bus.off(channel, fn);
			},
		},
	} as unknown as ExtensionAPI;
	const ctx = {
		hasUI: true,
		ui: { notify: () => {}, setWidget: () => {} },
		shutdown: () => {},
		sessionManager: { getBranch: () => [], getSessionDir: () => "/tmp" },
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) {
			const r: any = await fn({ type: name, ...event }, ctx);
			if (r !== undefined) result = r;
		}
		return result;
	}
	return { pi, emit, tools, entries, active: () => active };
}

const finish: RobotSpec["finish"] = {
	description: "finish",
	parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
	result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
};

/** A two-armed toy: `overAfter` actions end the episode; it logs every observation and action. */
async function toy(
	flags: Record<string, unknown>,
	o: { overAfter?: number; actions?: XPolicySpec["actions"] } = {},
	extra: Partial<RobotSpec> = {},
) {
	const f = fakePi(flags);
	const log: string[] = [];
	const acted: XPolicyAction[] = [];
	let observations = 0;
	const spec: XPolicySpec = {
		envCfgType: "aloha_agilex",
		actions: o.actions ?? ["joint"],
		observe: async () => {
			log.push(`obs${observations++}`);
			return {
				instruction: "beat the block with the hammer",
				vision: { cam_head: { color: new NdArray("uint8", [1, 1, 3], Buffer.alloc(3)) } },
				state: { left_arm_joint_state: [0, 0, 0, 0, 0, 0], left_ee_joint_state: [1] },
			};
		},
		act: async (a) => {
			log.push("act");
			acted.push(a);
		},
		over: () => o.overAfter !== undefined && acted.length >= o.overAfter,
		caseMeta: () => ({ task_name: "beat_block_hammer", seed: 7 }),
		trialResult: () => ({ success: acted.length > 0 }),
		present: async (run) => ({ content: [{ type: "text", text: JSON.stringify(run) }], details: run }),
	};
	defineRobot(f.pi, {
		name: "toy",
		task: [],
		keepImages: 4,
		start: async () => ["finish"],
		result: () => ({}),
		finish,
		xpolicy: spec,
		...extra,
	});
	await f.emit("session_start");
	await f.emit("agent_start");
	return { ...f, log, acted };
}

const joint = (v: number) => ({
	left_arm_joint_state: [v, 0, 0, 0, 0, 0],
	left_ee_joint_state: [1],
	right_arm_joint_state: [0, 0, 0, 0, 0, v],
	right_ee_joint_state: [0],
});

test("--xpolicy-precision is validated and needs --xpolicy; undeclared it is null", async (t) => {
	// Audit 92245e3 CM-6: the precision was XPOLICY_PRECISION from pi's shell, free text, unkeyed.
	const bridge = await fakeBridge([]);
	t.after(bridge.close);
	const bad = await toy({ xpolicy: "ws://policy:19000", "xpolicy-bridge": bridge.url, "xpolicy-precision": "fp8" });
	assert.deepEqual(bad.active(), [], "the robot did not start");
	assert.equal(bridge.calls.filter((c) => c.method === "xpolicy.connect").length, 0);
	const alone = await toy({ "xpolicy-precision": "bf16" });
	assert.deepEqual(alone.active(), [], "a declared precision without --xpolicy is a misconfiguration");
	const plain = await toy({ xpolicy: "ws://policy:19000", "xpolicy-bridge": bridge.url });
	await plain.tools.get("finish").execute("id", { status: "failure", summary: "" });
	await plain.emit("agent_end");
	assert.equal(plain.entries.find((e) => e.type === RESULT_ENTRY)?.data.xpolicy_precision, null);
});

test("without --xpolicy nothing is registered, connected or reported", async () => {
	const f = await toy({});
	assert.equal(f.tools.has("xpolicy_act"), false);
	assert.deepEqual(f.active(), ["finish"]);
	await f.tools.get("finish").execute("id", { status: "failure", summary: "" });
	await f.emit("agent_end");
	const r = f.entries.find((e) => e.type === RESULT_ENTRY)?.data;
	assert.equal("xpolicy" in r, false);
});

test("xpolicy_act runs XPolicyLab's deploy loop: case, reset, then update_obs / get_action / every action", async (t) => {
	const bridge = await fakeBridge([
		[joint(1), joint(2), joint(3)],
		[joint(4), joint(5)],
	]);
	t.after(bridge.close);
	const f = await toy({ xpolicy: "ws://policy:19000", "xpolicy-bridge": bridge.url, "xpolicy-precision": "bf16" });
	assert.deepEqual(f.active(), ["finish", "xpolicy_act"]);
	const connect = bridge.calls.find((c) => c.method === "xpolicy.connect")?.kwargs;
	assert.equal(connect?.url, "ws://policy:19000");
	assert.match(String(connect?.trial_id), /^toy_/);
	assert.equal(connect?.encode_images, false);
	assert.equal(connect?.request_timeout_s, 180);

	const out = await f.tools.get("xpolicy_act").execute("id", { chunks: 2 }, undefined, undefined, {});
	assert.deepEqual(
		bridge.calls.map((c) => c.method),
		[
			"xpolicy.action_dims",
			"xpolicy.connect",
			"xpolicy.prepare_case",
			"xpolicy.reset",
			"xpolicy.update_obs",
			"xpolicy.get_action",
			"xpolicy.update_obs",
			"xpolicy.update_obs",
			"xpolicy.update_obs",
			"xpolicy.get_action",
			"xpolicy.update_obs",
		],
	);
	assert.deepEqual(bridge.calls[2].kwargs.case_meta, {
		task_name: "beat_block_hammer",
		seed: 7,
		action_type: "joint",
	});
	// update_obs between two actions of a chunk, never after its last one (a new chunk observes afresh).
	assert.deepEqual(f.log, ["obs0", "act", "obs1", "act", "obs2", "act", "obs3", "act", "obs4", "act"]);
	assert.deepEqual(
		f.acted.map((a) => a.arms.left_.joints?.[0]),
		[1, 2, 3, 4, 5],
	);
	const obs = bridge.calls[4].kwargs.obs;
	assert.equal(obs.instruction, "beat the block with the hammer");
	assert.equal(obs.state.left_arm_joint_state.dtype, "float32");
	assert.equal(obs.vision.cam_head.color.__ndarray__, "AAAA");
	assert.deepEqual(out.details.xpolicy, {
		chunks: 2,
		chunk_sizes: [3, 2],
		executed_actions: 5,
		action_type: "joint",
		policy_ms: 15,
		stop_reason: "completed",
	});

	// A second call does not reset the policy again; an empty chunk is an error.
	await assert.rejects(f.tools.get("xpolicy_act").execute("id", {}, undefined, undefined, {}), /empty action chunk/);
	assert.equal(bridge.calls.filter((c) => c.method === "xpolicy.reset").length, 1);

	await f.tools.get("finish").execute("id", { status: "success", summary: "" });
	await f.emit("agent_end");
	const r = f.entries.find((e) => e.type === RESULT_ENTRY)?.data;
	assert.equal(r.xpolicy, "ws://policy:19000");
	assert.equal(r.xpolicy_server_instance_id, "srv-1");
	assert.equal(
		r.xpolicy_precision,
		"bf16",
		"the operator's declaration (--xpolicy-precision), not an env var of pi's shell",
	);
	assert.equal(r.xpolicy_chunks, 2, "the empty chunk of the second call raised before it counted");
	assert.equal(r.xpolicy_actions, 5);
	assert.deepEqual(
		f.entries.filter((e) => e.type === XPOLICY_ENTRY).map((e) => e.data.kind),
		["connect", "act"],
	);
	// The next session ends the trial and closes the client.
	await f.emit("session_start");
	const tail = bridge.calls.map((c) => c.method).slice(-4);
	assert.deepEqual(tail, ["xpolicy.trial_end", "xpolicy.close", "xpolicy.action_dims", "xpolicy.connect"]);
	assert.deepEqual(bridge.calls.find((c) => c.method === "xpolicy.trial_end")?.kwargs.result, { success: true });
});

test("the episode ending stops the chunk, and a scene reset prepares and resets the policy again", async (t) => {
	const bridge = await fakeBridge([[joint(1), joint(2), joint(3)], [joint(4)]]);
	t.after(bridge.close);
	const f = await toy({ xpolicy: "ws://policy:19000", "xpolicy-bridge": bridge.url }, { overAfter: 2 });
	const out = await f.tools.get("xpolicy_act").execute("id", { chunks: 5 }, undefined, undefined, {});
	assert.equal(out.details.xpolicy.executed_actions, 2);
	assert.equal(out.details.xpolicy.stop_reason, "episode_over");
	assert.deepEqual(f.log, ["obs0", "act", "obs1", "act"]);
	await f.emit("tool_result", { toolName: "reset", isError: false, content: [] });
	await f.tools.get("xpolicy_act").execute("id", {}, undefined, undefined, {});
	assert.equal(bridge.calls.filter((c) => c.method === "xpolicy.reset").length, 2);
	assert.equal(bridge.calls.filter((c) => c.method === "xpolicy.prepare_case").length, 2);
});

test("a chunk with an invalid action runs none of it", async (t) => {
	const bridge = await fakeBridge([[joint(1), { left_arm_joint_state: [1, 2] }]]);
	t.after(bridge.close);
	const f = await toy({ xpolicy: "ws://policy:19000", "xpolicy-bridge": bridge.url });
	await assert.rejects(
		f.tools.get("xpolicy_act").execute("id", {}, undefined, undefined, {}),
		/action 1 of the chunk: left_arm_joint_state must hold 6 finite values/,
	);
	assert.deepEqual(f.acted, []);
	// An ee action on a joint-only robot is refused the same way.
	const ee = await fakeBridge([[{ action_type: "ee", left_ee_pose: [0, 0, 0, 1, 0, 0, 0] }]]);
	t.after(ee.close);
	const g = await toy({ xpolicy: "ws://policy:19000", "xpolicy-bridge": ee.url });
	await assert.rejects(
		g.tools.get("xpolicy_act").execute("id", {}, undefined, undefined, {}),
		/executes joint actions, not ee/,
	);
});

test("a bridge error (a timed-out call) surfaces as the tool's error", async (t) => {
	const bridge = await fakeBridge([[joint(1)]], "xpolicy.get_action");
	t.after(bridge.close);
	const f = await toy({ xpolicy: "ws://policy:19000", "xpolicy-bridge": bridge.url });
	await assert.rejects(f.tools.get("xpolicy_act").execute("id", {}, undefined, undefined, {}), /WsError \(timeout\)/);
});

test("--xpolicy-action the robot cannot execute fails the start closed", async (t) => {
	const bridge = await fakeBridge([]);
	t.after(bridge.close);
	const f = await toy({ xpolicy: "ws://policy:19000", "xpolicy-bridge": bridge.url, "xpolicy-action": "ee" });
	assert.deepEqual(f.active(), []);
	assert.equal(
		bridge.calls.some((c) => c.method === "xpolicy.connect"),
		false,
	);
});

test("the manifests declare xpolicy_act on the dual rigs only, behind --xpolicy", () => {
	const entry = (robot: string) => loadManifest(robot).primitives.find((e) => e.name === "xpolicy_act");
	assert.deepEqual(entry("dual_franka")?.requires, ["xpolicy"]);
	assert.deepEqual(entry("piper")?.requires, ["dual", "xpolicy"], "the single-arm Piper has no XPolicyLab env_cfg");
	assert.equal(entry("franka"), undefined, "nor has the single-arm Franka");
	assert.equal(entry("dual_franka")?.module, "xpolicy");
});

test("xpolicy_act is activated as far as the robot's manifest entry allows", async (t) => {
	const bridge = await fakeBridge([]);
	t.after(bridge.close);
	const flags = { xpolicy: "ws://policy:19000", "xpolicy-bridge": bridge.url };
	const on = await toy(flags, {}, { manifest: "dual_franka", capabilities: (c) => c === "xpolicy" });
	assert.ok(on.active().includes("xpolicy_act"));
	const off = await toy(flags, {}, { manifest: "dual_franka", capabilities: () => false });
	assert.equal(off.active().includes("xpolicy_act"), false, "an unmet `requires` keeps it off");
});
