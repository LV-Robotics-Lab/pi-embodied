/**
 * XPolicyLab policies (github.com/XPolicyLab/XPolicyLab @d6332bf) as one robot tool, `xpolicy_act`.
 *
 *   pi -e packages/embodied/src/robotwin --xpolicy ws://127.0.0.1:19000 [--xpolicy-action joint|ee] ...
 *
 * XPolicyLab is the layer between policies and evaluation environments: every policy is an adapter
 * (`policy/<name>/model.py`: update_obs / get_action / reset) served in its own env by its
 * `setup_eval_policy_server.sh`, over websocket + msgpack. pi-embodied is only the environment
 * client: a robot opts in with `xpolicy` in its defineRobot spec (how it builds XPolicyLab's
 * observation and executes one action of a chunk); ../robot.ts mounts this module. The protocol
 * itself (HELLO handshake, request ids reused across a reconnect, ServerRestartedError, the numpy
 * msgpack extension) lives in the Python bridge (services pi_embodied_services/components/
 * xpolicy_bridge.py), which holds XPolicyLab's own `WsModelClient`; this module calls it over the
 * usual RPC (`xpolicy.*`). It starts the bridge with --xpolicy-python in the services dir, or
 * attaches to a running one (--xpolicy-bridge).
 *
 * With `--xpolicy <ws url>` the session start connects (a new trial per episode) and fails closed
 * when the server is unreachable or the robot cannot run --xpolicy-action. Dimensions come from the
 * services' env_cfg (`xpolicy.action_dims`, XPolicyLab's get_robot_action_dim_info against
 * components/xpolicy_env_cfg): single-arm robots use unprefixed state/action keys, two-armed ones
 * `left_` / `right_`. `xpolicy_act` (registered at the first such start) follows XPolicyLab's deploy loop (policy/<name>/deploy.py):
 * before the episode's first chunk (and after a scene reset) prepare_case (the robot's case meta)
 * and reset; per chunk update_obs + get_action, then every action of the chunk, with an update_obs
 * of the fresh observation between two actions; it stops at the end of the episode. An action is a
 * dict (`<arm>_arm_joint_state`, `<arm>_ee_pose` [x, y, z, qw, qx, qy, qz], `<arm>_ee_joint_state`,
 * optional `action_type`) or a flat vector in the packed [arm, ee] per-arm order; a whole chunk is
 * validated before its first action runs. A `timeout` or ServerRestartedError ends the trial (every
 * later call fails, as XPolicyLab prescribes). Each connect and each call is an `xpolicy` session
 * entry; the robot result carries the xpolicy_* fields. Off (the default), only the flags exist.
 */

import { type ChildProcess, spawn } from "node:child_process";
import { closeSync, openSync, writeSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";
import { frameOf, type Json, message, quatMat, SERVICES, servicesEnv, shutdown } from "./robot.ts";
import { NdArray, RpcClient } from "./rpc.ts";

/** Session entry per connect and per `xpolicy_act`. */
export const XPOLICY_ENTRY = "xpolicy";
export type XPolicyActionType = "joint" | "ee";
/** `get_robot_action_dim_info(env_cfg_type)`: one entry per arm. */
export type XPolicyDims = { robot?: string; arm_dim: number[]; ee_dim: number[] };
export type XPolicyCamera = {
	/** [H, W, C>=3] RGB (sent as uint8 [H, W, 3]). */
	color: NdArray;
	depth?: NdArray;
	intrinsic_matrix?: NdArray | number[][];
	extrinsics_matrix?: NdArray | number[][];
};
/** One observation in XPolicyLab's "Observation Data Format" (state keys prefixed per arm). */
export type XPolicyObs = {
	instruction: string;
	/** By XPolicyLab camera name: cam_head, cam_left_wrist, cam_right_wrist, cam_wrist, cam_third_view. */
	vision: Record<string, XPolicyCamera>;
	/** `<arm>_arm_joint_state`, `<arm>_ee_joint_state`, `<arm>_ee_pose`, `<arm>_tcp_pose`, ... */
	state: Record<string, number[]>;
	/** `additional_info` (e.g. frequency). */
	info?: Json;
};
/** One arm's part of an action: absent parts keep that arm's current command. */
export type XPolicyArm = { joints?: number[]; pose?: number[]; ee?: number[] };
/** One action of a chunk, by arm prefix ("" on a single arm, "left_" / "right_" on two). */
export type XPolicyAction = { type: XPolicyActionType; arms: Record<string, XPolicyArm> };
export type XPolicySpec = {
	/** env_cfg type: the robot's row in components/xpolicy_env_cfg (aloha_agilex, piper, franka). */
	envCfgType: string;
	/** Action types this robot executes. */
	actions: readonly XPolicyActionType[];
	observe: (signal?: AbortSignal) => Promise<XPolicyObs>;
	/** Execute one action of a chunk. */
	act: (action: XPolicyAction, signal?: AbortSignal) => Promise<void>;
	/** The episode is over (success, budget): no further action runs. */
	over: () => boolean;
	/** The tool result after a call: the robot's new state, with `run` (what the call did) in it. */
	present: (run: Json) => Promise<AgentToolResult<unknown>>;
	/** prepare_case's case meta (RoboTwin: task_name, seed, instruction); action_type is added. */
	caseMeta?: () => Json;
	/** trial_end's result at the end of the episode (e.g. `{success}`). */
	trialResult?: () => Json;
};
/** Register a sequential robot tool (the base's `tool`, ../robot.ts). */
export type RobotTool = <P extends TSchema>(
	name: string,
	description: string,
	parameters: P,
	run: (
		params: Static<P>,
		signal: AbortSignal | undefined,
		ctx: ExtensionContext,
	) => Promise<AgentToolResult<unknown>>,
) => void;

/** Tools that restore the scene: the policy is prepared and reset again before its next chunk. */
const RESETS = ["reset", "request_scene_reset"];

export function actionType(value: string): XPolicyActionType {
	const t = value.trim().toLowerCase();
	if (t === "joint" || t === "qpos") return "joint";
	if (t === "ee" || t === "endpose") return "ee";
	throw new Error(`unknown XPolicyLab action_type "${value}" (joint or ee)`);
}

/** State/action key prefixes: "" for one arm, "left_" and "right_" for two. */
export function armPrefixes(dims: XPolicyDims): string[] {
	if (dims.arm_dim.length !== dims.ee_dim.length) throw new Error(`arm_dim and ee_dim differ in length`);
	if (dims.arm_dim.length === 1) return [""];
	if (dims.arm_dim.length === 2) return ["left_", "right_"];
	throw new Error(`unsupported arm count ${dims.arm_dim.length}`);
}

const sum = (v: number[]) => v.reduce((a, b) => a + b, 0);
const finite = (v: number[]) => v.every(Number.isFinite);

/** One action of a `get_action` chunk (a dict, or a flat vector in the packed per-arm [arm, ee] order), validated. */
export function parseAction(raw: unknown, dims: XPolicyDims, fallback: XPolicyActionType): XPolicyAction {
	const prefixes = armPrefixes(dims);
	if (Array.isArray(raw)) {
		const v = raw.map(Number);
		const sizes = prefixes.flatMap((_, i) => [fallback === "joint" ? dims.arm_dim[i] : 7, dims.ee_dim[i]]);
		if (v.length !== sum(sizes) || !finite(v))
			throw new Error(`a flat ${fallback} action must hold ${sum(sizes)} finite values [${sizes}], got ${v.length}`);
		const arms: Record<string, XPolicyArm> = {};
		prefixes.forEach((p, i) => {
			const at = sum(sizes.slice(0, 2 * i));
			const main = v.slice(at, at + sizes[2 * i]);
			const ee = v.slice(at + sizes[2 * i], at + sizes[2 * i] + sizes[2 * i + 1]);
			arms[p] = fallback === "joint" ? { joints: main, ee } : { pose: main, ee };
		});
		return { type: fallback, arms };
	}
	if (!raw || typeof raw !== "object") throw new Error(`an action must be a dict or a vector, got ${typeof raw}`);
	const d = raw as Record<string, unknown>;
	const type = d.action_type === undefined ? fallback : actionType(String(d.action_type));
	const get = (key: string, n: number) => {
		if (!(key in d)) return undefined;
		const v = [d[key]].flat(Number.POSITIVE_INFINITY).map(Number);
		if (v.length !== n || !finite(v))
			throw new Error(`${key} must hold ${n} finite values, got ${JSON.stringify(d[key])}`);
		return v;
	};
	const arms: Record<string, XPolicyArm> = {};
	prefixes.forEach((p, i) => {
		const parts = {
			joints: get(`${p}arm_joint_state`, dims.arm_dim[i]),
			pose: get(`${p}ee_pose`, 7),
			ee: get(`${p}ee_joint_state`, dims.ee_dim[i]),
		};
		const arm = Object.fromEntries(Object.entries(parts).filter(([, v]) => v !== undefined)) as XPolicyArm;
		if (Object.keys(arm).length) arms[p] = arm;
	});
	const main = type === "joint" ? "joints" : "pose";
	if (!Object.values(arms).some((a) => a[main]))
		throw new Error(
			`a ${type} action needs ${prefixes.map((p) => `${p}${type === "joint" ? "arm_joint_state" : "ee_pose"}`).join(" or ")}; got keys ${Object.keys(d).join(", ")}`,
		);
	return { type, arms };
}

const f32 = (m: NdArray | number[][] | number[], shape: number[]) =>
	m instanceof NdArray ? NdArray.f32(m.toArray(), m.shape) : NdArray.f32(m.flat(), shape);

/** The wire observation (XPolicyLab "Observation Data Format" v1.0), its state checked against `dims`. */
export function wireObs(obs: XPolicyObs, dims: XPolicyDims): Json {
	armPrefixes(dims).forEach((p, i) => {
		const check = (key: string, n: number) => {
			const v = obs.state[key];
			if (v !== undefined && (v.length !== n || !finite(v)))
				throw new Error(`observation state ${key} must hold ${n} finite values, got ${v.length}`);
		};
		check(`${p}arm_joint_state`, dims.arm_dim[i]);
		check(`${p}ee_joint_state`, dims.ee_dim[i]);
		for (const k of ["ee_pose", "tcp_pose", "delta_ee_pose"]) check(`${p}${k}`, 7);
	});
	const vision: Json = {};
	for (const [name, cam] of Object.entries(obs.vision)) {
		const color = frameOf(cam.color);
		vision[name] = {
			color,
			shape: color.shape.slice(0, 2),
			...(cam.depth ? { depth: cam.depth } : {}),
			...(cam.intrinsic_matrix ? { intrinsic_matrix: f32(cam.intrinsic_matrix, [3, 3]) } : {}),
			...(cam.extrinsics_matrix ? { extrinsics_matrix: f32(cam.extrinsics_matrix, [4, 4]) } : {}),
		};
	}
	return {
		data_format_version: "v1.0",
		instruction: obs.instruction,
		instructions: obs.instruction ? [obs.instruction] : [],
		env_idx: 0,
		vision,
		state: Object.fromEntries(Object.entries(obs.state).map(([k, v]) => [k, NdArray.f32(v)])),
		...(obs.info ? { additional_info: obs.info } : {}),
	};
}

// ---------------------------------------------------------------------------
// helpers for robots that execute ee targets as bounded relative moves

/** [x, y, z, qx, qy, qz, qw] <-> XPolicyLab's [x, y, z, qw, qx, qy, qz]. */
export const toWxyz = (p: number[]) => [p[0], p[1], p[2], p[6], p[3], p[4], p[5]];
export const toXyzw = (p: number[]) => [p[0], p[1], p[2], p[4], p[5], p[6], p[3]];

/**
 * The base-frame move from `current` to `target` (both [x, y, z, qw, qx, qy, qz]): the translation,
 * and the rotation R_target R_current^T as extrinsic xyz euler angles (scipy's "xyz"), in rad.
 */
export function poseDelta(current: number[], target: number[]): { delta: number[]; rpy: number[] } {
	const c = quatMat(toXyzw(current).slice(3));
	const t = quatMat(toXyzw(target).slice(3));
	const r = [0, 1, 2].map((i) => [0, 1, 2].map((j) => t[i][0] * c[j][0] + t[i][1] * c[j][1] + t[i][2] * c[j][2]));
	return {
		delta: [0, 1, 2].map((i) => target[i] - current[i]),
		rpy: [Math.atan2(r[2][1], r[2][2]), Math.asin(Math.max(-1, Math.min(1, -r[2][0]))), Math.atan2(r[1][0], r[0][0])],
	};
}

/**
 * An `ee_joint_state` gripper target on a gripper that only opens and closes: below half of `widest`
 * (the widest opening observed) closes, otherwise opens; null when the gripper is already there.
 */
export function gripperCommand(target: number, widest: number, closed: boolean): "open" | "close" | null {
	const close = target < widest / 2;
	return close === closed ? null : close ? "close" : "open";
}

// ---------------------------------------------------------------------------
// the bridge process

type Bridge = { rpc: RpcClient; proc?: ChildProcess; log?: string };

/**
 * Start the bridge in the services dir; it binds port 0 and prints the port. It stays up across
 * sessions (each episode connects a new trial); pi stops it at shutdown, and it exits with pi
 * (--parent-watch) should pi die first.
 */
async function spawnBridge(python: string, services: string, root: string, readyMs: number): Promise<Bridge> {
	const argv = [
		...["-m", "pi_embodied_services.components.xpolicy_bridge"],
		...(root ? ["--xpolicylab-root", root] : []),
		...["--transport", "http", "--host", "127.0.0.1", "--port", "0", "--parent-watch"],
	];
	const proc = spawn(python, argv, {
		cwd: services,
		env: servicesEnv({ root: services, python }),
		stdio: ["pipe", "pipe", "pipe"],
	});
	const log = join(tmpdir(), `pi-embodied-xpolicy-bridge-${process.pid}-${Date.now()}.log`);
	const fd = openSync(log, "a");
	proc.once("close", () => closeSync(fd));
	let seen: string | undefined = "";
	const port = await new Promise<number>((resolve, reject) => {
		const sink = (chunk: Buffer) => {
			writeSync(fd, chunk);
			if (seen === undefined) return;
			seen += chunk.toString();
			const m = /RPC server listening on http:\/\/[^\s:]+:(\d+)/.exec(seen);
			if (!m) return;
			seen = undefined;
			resolve(Number(m[1]));
		};
		proc.stdout?.on("data", sink);
		proc.stderr?.on("data", sink);
		proc.once("exit", (code) => reject(new Error(`xpolicy bridge exited (${code}); see ${log}`)));
		proc.once("error", (err) => reject(new Error(`xpolicy bridge failed to start: ${err.message}`)));
		setTimeout(() => reject(new Error(`xpolicy bridge bound no port in ${readyMs} ms; see ${log}`)), readyMs).unref();
	});
	const rpc = new RpcClient(`http://127.0.0.1:${port}`);
	await rpc.ready(readyMs);
	return { rpc, proc, log };
}

// ---------------------------------------------------------------------------
// the module

type Session = {
	url: string;
	dims: XPolicyDims;
	type: XPolicyActionType;
	info: Json;
	/** prepare_case + reset before the next chunk (a new episode or a restored scene). */
	fresh: boolean;
	acted: boolean;
	chunks: number;
	actions: number;
};

const DESCRIPTION =
	"Run the XPolicyLab policy served at --xpolicy on the native task instruction, for `chunks` action chunks (default 1): each chunk sends the current observation (update_obs), gets an action chunk (get_action) and executes every action of it, sending the fresh observation between two actions; it stops early when the episode ends. Returns the new state.";

/**
 * Register the XPolicyLab flags and `xpolicy_act` (active only with --xpolicy). `task()` names the
 * episode (the trial id); `name` is the robot's.
 */
export function xpolicy(
	pi: ExtensionAPI,
	spec: XPolicySpec | (() => XPolicySpec | undefined),
	tool: RobotTool,
	name: string,
	task: () => Record<string, string>,
) {
	pi.registerFlag("xpolicy", {
		type: "string",
		default: "",
		description: "XPolicyLab policy server (ws://host:port): adds xpolicy_act (empty = off)",
	});
	pi.registerFlag("xpolicy-action", {
		type: "string",
		default: "joint",
		description: "XPolicyLab action_type the policy emits: joint | ee",
	});
	pi.registerFlag("xpolicy-bridge", {
		type: "string",
		default: "",
		description: "Attach to a running XPolicyLab bridge (http endpoint) instead of starting one",
	});
	pi.registerFlag("xpolicylab", {
		type: "string",
		default: process.env.XPOLICYLAB_ROOT ?? "",
		description: "XPolicyLab checkout whose websocket client the bridge uses (default XPOLICYLAB_ROOT)",
	});
	pi.registerFlag("xpolicy-python", {
		type: "string",
		default: process.env.PI_EMBODIED_XPOLICY_PYTHON ?? "python",
		description: "Python with the services' [xpolicy] extra, for the bridge",
	});
	pi.registerFlag("xpolicy-encode-images", {
		type: "boolean",
		default: false,
		description: "Send camera colors JPEG-encoded (XPolicyLab's encode_image_bit) instead of raw RGB",
	});
	pi.registerFlag("xpolicy-timeout", {
		type: "string",
		default: "180",
		description: "Seconds one policy call may take (XPolicyLab request_timeout_s); a timeout ends the trial",
	});
	pi.registerFlag("xpolicy-connect-timeout", {
		type: "string",
		default: "900",
		description: "Seconds the connect may wait for a policy server that is still loading (max_connect_seconds)",
	});
	const current = () => (typeof spec === "function" ? spec() : spec);
	const flag = (key: string) => String(pi.getFlag(key) ?? "").trim();
	const on = () => flag("xpolicy") !== "";
	const callMs = () => Number(flag("xpolicy-timeout")) * 1000 + 30_000;
	let bridge: Bridge | undefined;
	let stopping = false;
	let session: Session | undefined;

	async function ensureBridge(): Promise<RpcClient> {
		const endpoint = flag("xpolicy-bridge");
		if (endpoint) {
			if (bridge?.rpc.url !== new RpcClient(endpoint).url) {
				bridge = { rpc: new RpcClient(endpoint) };
				await bridge.rpc.ready(60_000);
			}
			return bridge.rpc;
		}
		if (bridge?.proc && bridge.proc.exitCode === null && bridge.proc.signalCode === null) return bridge.rpc;
		const services = String(pi.getFlag("services") || SERVICES);
		bridge = await spawnBridge(flag("xpolicy-python") || "python", services, flag("xpolicylab"), 120_000);
		// Registered now, after the base's handlers: the episode's trial ends (stop) before the bridge goes.
		// A child with open pipes keeps pi's event loop alive, so pi would never exit without this.
		if (!stopping) {
			stopping = true;
			pi.on("session_shutdown", async () => {
				const b = bridge;
				bridge = undefined;
				if (b?.proc) await shutdown(b.proc, b.rpc);
			});
		}
		return bridge.rpc;
	}

	pi.on("tool_result", (event) => {
		if (session && RESETS.includes(event.toolName) && !event.isError) session.fresh = true;
	});

	/** Registered at the first start with --xpolicy (off registers nothing, like --privileged's ground_truth_poses). */
	let registered = false;
	const register = () =>
		tool(
			"xpolicy_act",
			DESCRIPTION,
			Type.Object({ chunks: Type.Optional(Type.Integer({ minimum: 1, maximum: 200, description: "Default 1" })) }),
			async ({ chunks = 1 }, signal) => {
				const s = current();
				const rpc = bridge?.rpc;
				if (!session || !s || !rpc) throw new Error("xpolicy_act needs --xpolicy <ws url> (see the session start)");
				const live = session;
				const call = <T = Json>(method: string, kwargs: Json) => rpc.call<T>(method, kwargs, callMs(), [], signal);
				let policyMs = 0;
				const update = async () => {
					const r = await call<{ ms: number }>("xpolicy.update_obs", {
						obs: wireObs(await s.observe(signal), live.dims),
					});
					policyMs += r.ms;
				};
				if (live.fresh) {
					const meta = s.caseMeta?.();
					if (meta) await call("xpolicy.prepare_case", { case_meta: { ...meta, action_type: live.type } });
					await call("xpolicy.reset", {});
					live.fresh = false;
				}
				live.acted = true;
				const sizes: number[] = [];
				let executed = 0;
				let stop = "completed";
				for (let c = 0; c < chunks; c++) {
					if (s.over()) {
						stop = "episode_over";
						break;
					}
					await update();
					const got = await call<{ actions: unknown[]; ms: number }>("xpolicy.get_action", {});
					policyMs += got.ms;
					if (!Array.isArray(got.actions) || !got.actions.length)
						throw new Error("the policy returned an empty action chunk");
					// The whole chunk is checked before its first action moves the robot.
					const actions = got.actions.map((a, i) => {
						try {
							const parsed = parseAction(a, live.dims, live.type);
							if (!s.actions.includes(parsed.type))
								throw new Error(`this robot executes ${s.actions.join(" or ")} actions, not ${parsed.type}`);
							return parsed;
						} catch (err) {
							throw new Error(`action ${i} of the chunk: ${message(err)}`);
						}
					});
					sizes.push(actions.length);
					for (let j = 0; j < actions.length; j++) {
						if (signal?.aborted) throw new Error("xpolicy_act interrupted");
						await s.act(actions[j], signal);
						executed++;
						if (s.over()) break;
						if (j < actions.length - 1) await update();
					}
					live.chunks++;
					if (s.over()) {
						stop = "episode_over";
						break;
					}
				}
				live.actions += executed;
				const run = {
					chunks: sizes.length,
					chunk_sizes: sizes,
					executed_actions: executed,
					action_type: live.type,
					policy_ms: Math.round(policyMs),
					stop_reason: stop,
				};
				pi.appendEntry(XPOLICY_ENTRY, { kind: "act", requested_chunks: chunks, ...run });
				return s.present({ xpolicy: run });
			},
		);

	return {
		/** Connect for this episode (with --xpolicy) and name the tool to activate. Throws when it cannot run. */
		async start(): Promise<string[]> {
			session = undefined;
			if (!on()) return [];
			const s = current();
			if (!s) throw new Error("--xpolicy: this robot has no XPolicyLab observation");
			const type = actionType(flag("xpolicy-action"));
			if (!s.actions.includes(type))
				throw new Error(`--xpolicy-action ${type}: this robot executes ${s.actions.join(" or ")} actions only`);
			if (!registered) {
				registered = true;
				register();
			}
			const rpc = await ensureBridge();
			const dims = await rpc.call<XPolicyDims>("xpolicy.action_dims", { env_cfg_type: s.envCfgType }, 30_000);
			armPrefixes(dims);
			const t = Object.values(task()).join("_");
			const connectS = Number(flag("xpolicy-connect-timeout"));
			const info = await rpc.call<Json>(
				"xpolicy.connect",
				{
					url: flag("xpolicy"),
					trial_id: `${name}_${t}_${Date.now()}`,
					evaluation_id: "pi-embodied",
					action_case_id: `${name}_${t}`,
					encode_images: pi.getFlag("xpolicy-encode-images") === true,
					request_timeout_s: Number(flag("xpolicy-timeout")),
					max_connect_seconds: connectS,
				},
				(connectS + 60) * 1000,
			);
			session = { url: flag("xpolicy"), dims, type, info, fresh: true, acted: false, chunks: 0, actions: 0 };
			pi.appendEntry(XPOLICY_ENTRY, {
				kind: "connect",
				env_cfg_type: s.envCfgType,
				action_type: type,
				dims,
				...info,
				...(bridge?.log ? { bridge_log: bridge.log } : {}),
			});
			return ["xpolicy_act"];
		},
		/** End the trial (trial_end after an episode that acted) and close the policy client; the bridge stays up. */
		async stop() {
			const live = session;
			session = undefined;
			const rpc = bridge?.rpc;
			if (!live || !rpc) return;
			if (live.acted)
				await rpc.call("xpolicy.trial_end", { result: current()?.trialResult?.() ?? {} }, 30_000).catch(() => {});
			await rpc.call("xpolicy.close", {}, 30_000).catch(() => {});
		},
		/** The robot result's xpolicy fields (none without --xpolicy). */
		result: (): Json =>
			session
				? {
						xpolicy: session.url,
						xpolicy_action: session.type,
						xpolicy_server_instance_id: session.info.server_instance_id ?? null,
						xpolicylab_rev: session.info.xpolicylab_rev ?? null,
						xpolicy_chunks: session.chunks,
						xpolicy_actions: session.actions,
					}
				: {},
	};
}
