/**
 * RoboTwin robot for pi.
 *
 *   pi -e packages/embodied/src/robotwin --task-name beat_block_hammer --seed 100000
 *
 * Starts one RoboTwin env server per session (the RLinf RoboTwin facade) and
 * attaches to a running LingBot-VLA WebSocket server (see serve.sh). Tools follow
 * the RoboTwin primitives for the dual-arm aloha-agilex: `move_to` plans with
 * the env's curobo planner (`env.plan_arm_path`) and executes qpos waypoints,
 * `lingbot_act` runs eef16 chunks. Every action returns a new numbered state with
 * the head and both wrist images; success is RoboTwin's own `eval_success`,
 * recorded in the session's `robot_result` entry.
 */

import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";
import { recipeFlash } from "../flash/recipe.ts";
import type { FlywheelObs, FlywheelSpec } from "../flywheel.ts";
import { encodePng } from "../png.ts";
import { attach, defineRobot, median, SERVICES, u8 } from "../robot.ts";
import { NdArray, type RpcClient } from "../rpc.ts";
import type { Move } from "../units/index.ts";
import { vlaSeeds } from "../vla-seed.ts";

const read = (name: string) => readFileSync(new URL(name, import.meta.url), "utf8");
const SYSTEM = read("./SYSTEM.md");
const MEMORY = read("./memory.md");
const EXPLORE = read("./explore.md");
const VIEWS = ["head", "left_wrist", "right_wrist"] as const;
type View = (typeof VIEWS)[number];
type Arm = "left" | "right";
/**
 * What LingBot reads and emits, eef16 (services robots/robotwin/flywheel.py); the camera sizes are the
 * config's. The joint state rides along for the joint-space (XPolicyLab) export: `joint_states` the
 * measured joints, `joint_targets` the commanded ones, both [left joints6, left gripper, right joints6,
 * right gripper].
 */
const FLYWHEEL: FlywheelSpec = {
	robot: "robotwin",
	images: { head_images: null, left_wrist_images: null, right_wrist_images: null },
	state: 16,
	action: 16,
	vectors: { joint_states: 14, joint_targets: 14 },
};
/** The server's policy frame (env.policy_frame, chunk_step's policy_frames). */
type PolicyFrame = {
	head: NdArray;
	left_wrist: NdArray;
	right_wrist: NdArray;
	state: NdArray;
	qpos: NdArray;
	qpos_target: NdArray;
};
const flyObs = (f: PolicyFrame): FlywheelObs => ({
	images: { head_images: u8(f.head), left_wrist_images: u8(f.left_wrist), right_wrist_images: u8(f.right_wrist) },
	state: f.state.toArray(),
	vectors: { joint_states: f.qpos.toArray(), joint_targets: f.qpos_target.toArray() },
});
/**
 * How the action units look (--units): RoboTwin's world frame, in which the robot faces +y with its
 * right arm on +x. Derived from the head camera's mounting, not yet calibrated in the simulator.
 */
const UNITS_VIEWS = `Each result shows the head view, then the left wrist view, then the right wrist view; every unit names the arm it moves. MV_FWD moves that gripper away from the robot (toward the far side of the table), MV_BACK toward the robot, MV_LEFT / MV_RIGHT toward the robot's left / right, MV_UP / MV_DOWN up and down. ROTATE_CW / ROTATE_CCW turn the gripper about the vertical.
- Head view (first image): the robot's head camera, above and behind the arms, looking forward at the table: MV_FWD moves the gripper toward the image top, MV_BACK toward the bottom, MV_LEFT / MV_RIGHT toward the image left / right. The left arm is on the image left.
- Wrist views: each moves with its gripper. These directions come from the robot's geometry and are not calibrated: after the first move, check where the gripper went in the head view and trust what you see.`;
type Status = { eval_success: boolean; take_action_cnt: number; step_lim: number | null; actual_seed: number };
type Info = {
	robot_state: Record<string, unknown>;
	episode_status: Status;
	executed_actions?: number;
	instruction?: string;
	instruction_source?: string;
};
type StepReturn = [unknown, unknown, unknown, unknown, Info];
/** chunk_step's observation with return_all_frames: the head frame after every native action. */
type Frames = { frames?: NdArray[]; policy_frames?: PolicyFrame[] };
type CameraMeta = { intrinsic_K: NdArray; cam2world_gl: NdArray; width: number; height: number };
type WorldMap = { height: number; width: number; xyz: Float32Array };
type Snapshot = { payload: Record<string, unknown>; images: Buffer[]; world: Record<View, WorldMap> };

const READ_MS = 120_000;
const MUTATE_MS = 600_000;
const USE_LENGTH = 50; // LingBot RoboTwin EEF checkpoint contract
const LINGBOT_CONTRACT = {
	runtime: "lingbotvla",
	policy_name: "robotwin_eef",
	state_layout: "eef16",
	action_layout: "eef16",
	use_length: USE_LENGTH,
};

const round = (v: number, d = 4) => Number(v.toFixed(d));
/** numpy's `linspace(0, n - 1, k).astype(int)`. */
const linspace = (n: number, k: number) =>
	Array.from({ length: k }, (_, i) => (k === 1 ? 0 : Math.floor((i * (n - 1)) / (k - 1))));
const f64 = (values: number[]) =>
	new NdArray("float64", [values.length], Buffer.from(Float64Array.from(values).buffer));
const plain = (v: unknown): unknown =>
	v instanceof NdArray ? v.toArray().map((x) => round(x)) : typeof v === "number" ? round(v) : v;

/** Hamilton product of wxyz quaternions. */
function qmult([w1, x1, y1, z1]: number[], [w2, x2, y2, z2]: number[]): number[] {
	return [
		w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
		w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
		w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
		w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
	];
}

/** World xyz per pixel from metric depth (NaN = no hit) and the OpenGL cam-to-world pose. */
function worldMap(depth: NdArray, meta: CameraMeta): WorldMap {
	const [height, width] = depth.shape;
	if (meta.height !== height || meta.width !== width)
		throw new Error(`depth ${height}x${width} does not match camera meta ${meta.height}x${meta.width}`);
	const d = depth.toArray();
	const [fx, , cx, , fy, cy] = meta.intrinsic_K.toArray();
	const m = meta.cam2world_gl.toArray();
	const xyz = new Float32Array(height * width * 3);
	for (let r = 0; r < height; r++) {
		for (let c = 0; c < width; c++) {
			const i = r * width + c;
			const z = d[i];
			const p = [((c - cx) * z) / fx, (-(r - cy) * z) / fy, -z];
			for (let k = 0; k < 3; k++)
				xyz[i * 3 + k] = m[k * 4] * p[0] + m[k * 4 + 1] * p[1] + m[k * 4 + 2] * p[2] + m[k * 4 + 3];
		}
	}
	return { height, width, xyz };
}

// --- msgpack with openpi's numpy extension (what LingBot's WebSocket server speaks) ---

const DESCR: Record<string, string> = {
	uint8: "|u1",
	bool: "|b1",
	float32: "<f4",
	float64: "<f8",
	int32: "<i4",
	int64: "<i8",
};
const DTYPE: Record<string, string> = {
	u1: "uint8",
	b1: "bool",
	f4: "float32",
	f8: "float64",
	i4: "int32",
	i8: "int64",
};

function pack(value: unknown, out: Buffer[] = []): Buffer[] {
	const head = (bytes: number[]) => out.push(Buffer.from(bytes));
	const len = (
		n: number,
		small: number,
		tiny: number | undefined,
		c8: number | undefined,
		c16: number,
		c32: number,
	) => {
		if (tiny !== undefined && n < small) head([tiny | n]);
		else if (c8 !== undefined && n < 0x100) head([c8, n]);
		else if (n < 0x10000) head([c16, n >> 8, n & 0xff]);
		else {
			const b = Buffer.alloc(5);
			b[0] = c32;
			b.writeUInt32BE(n, 1);
			out.push(b);
		}
	};
	const bin = (b: Buffer) => {
		len(b.length, 0, undefined, 0xc4, 0xc5, 0xc6);
		out.push(b);
	};
	const str = (s: string) => {
		const b = Buffer.from(s, "utf8");
		len(b.length, 32, 0xa0, 0xd9, 0xda, 0xdb);
		out.push(b);
	};
	if (value === null || value === undefined) head([0xc0]);
	else if (typeof value === "boolean") head([value ? 0xc3 : 0xc2]);
	else if (typeof value === "string") str(value);
	else if (typeof value === "number") {
		if (Number.isInteger(value) && value >= 0 && value < 0x80) head([value]);
		else if (Number.isInteger(value) && value >= -0x80000000 && value <= 0x7fffffff) {
			const b = Buffer.alloc(5);
			b[0] = 0xd2;
			b.writeInt32BE(value, 1);
			out.push(b);
		} else {
			const b = Buffer.alloc(9);
			b[0] = 0xcb;
			b.writeDoubleBE(value, 1);
			out.push(b);
		}
	} else if (value instanceof NdArray) {
		head([0x84]);
		bin(Buffer.from("__ndarray__"));
		head([0xc3]);
		bin(Buffer.from("data"));
		bin(value.data);
		bin(Buffer.from("dtype"));
		str(DESCR[value.dtype] ?? value.dtype);
		bin(Buffer.from("shape"));
		pack(value.shape, out);
	} else if (Buffer.isBuffer(value)) bin(value);
	else if (Array.isArray(value)) {
		len(value.length, 16, 0x90, undefined, 0xdc, 0xdd);
		for (const v of value) pack(v, out);
	} else if (typeof value === "object") {
		const entries = Object.entries(value);
		len(entries.length, 16, 0x80, undefined, 0xde, 0xdf);
		for (const [k, v] of entries) {
			str(k);
			pack(v, out);
		}
	} else throw new Error(`cannot msgpack ${typeof value}`);
	return out;
}

function unpack(buf: Buffer): unknown {
	let pos = 0;
	const uint = (n: 1 | 2 | 4) => {
		const v = n === 1 ? buf.readUInt8(pos) : n === 2 ? buf.readUInt16BE(pos) : buf.readUInt32BE(pos);
		pos += n;
		return v;
	};
	const take = <T>(n: number, f: (at: number) => T) => {
		const v = f(pos);
		pos += n;
		return v;
	};
	const bytes = (n: number) => take(n, (at) => Buffer.from(buf.subarray(at, at + n)));
	const text = (n: number) => take(n, (at) => buf.toString("utf8", at, at + n));
	const list = (n: number) => Array.from({ length: n }, () => read());
	const map = (n: number) => {
		const o: Record<string, unknown> = {};
		for (let i = 0; i < n; i++) {
			const k = read();
			o[Buffer.isBuffer(k) ? k.toString("utf8") : String(k)] = read();
		}
		if (o.__ndarray__) {
			const descr = String(o.dtype);
			return new NdArray(DTYPE[descr.replace(/^[<>|=]/, "")] ?? descr, o.shape as number[], o.data as Buffer);
		}
		return o.__npgeneric__ ? o.data : o;
	};
	function read(): unknown {
		const t = buf[pos++];
		if (t < 0x80) return t;
		if (t < 0x90) return map(t & 0x0f);
		if (t < 0xa0) return list(t & 0x0f);
		if (t < 0xc0) return text(t & 0x1f);
		if (t >= 0xe0) return t - 0x100;
		switch (t) {
			case 0xc0:
				return null;
			case 0xc2:
				return false;
			case 0xc3:
				return true;
			case 0xc4:
				return bytes(uint(1));
			case 0xc5:
				return bytes(uint(2));
			case 0xc6:
				return bytes(uint(4));
			case 0xca:
				return take(4, (at) => buf.readFloatBE(at));
			case 0xcb:
				return take(8, (at) => buf.readDoubleBE(at));
			case 0xcc:
				return uint(1);
			case 0xcd:
				return uint(2);
			case 0xce:
				return uint(4);
			case 0xcf:
				return take(8, (at) => Number(buf.readBigUInt64BE(at)));
			case 0xd0:
				return take(1, (at) => buf.readInt8(at));
			case 0xd1:
				return take(2, (at) => buf.readInt16BE(at));
			case 0xd2:
				return take(4, (at) => buf.readInt32BE(at));
			case 0xd3:
				return take(8, (at) => Number(buf.readBigInt64BE(at)));
			case 0xd9:
				return text(uint(1));
			case 0xda:
				return text(uint(2));
			case 0xdb:
				return text(uint(4));
			case 0xdc:
				return list(uint(2));
			case 0xdd:
				return list(uint(4));
			case 0xde:
				return map(uint(2));
			case 0xdf:
				return map(uint(4));
			default:
				throw new Error(`unsupported msgpack type 0x${t.toString(16)}`);
		}
	}
	return read();
}

/** LingBot-VLA policy client: WebSocket, msgpack frames, server metadata first, then one reply per request. */
class LingBot {
	ws: WebSocket;
	metadata: Record<string, unknown> = {};
	failed: Error | undefined;
	inbox: (Buffer | string)[] = [];
	waiters: { resolve: (v: Buffer | string) => void; reject: (e: Error) => void }[] = [];

	constructor(url: string) {
		this.ws = new WebSocket(url);
		this.ws.binaryType = "arraybuffer";
		this.ws.addEventListener("message", (e) => {
			const data = typeof e.data === "string" ? e.data : Buffer.from(e.data as ArrayBuffer);
			const waiter = this.waiters.shift();
			if (waiter) waiter.resolve(data);
			else this.inbox.push(data);
		});
		const fail = (why: string) => {
			this.failed ??= new Error(why);
			for (const w of this.waiters.splice(0)) w.reject(this.failed);
		};
		this.ws.addEventListener("close", (e) => fail(`LingBot socket closed (${e.code} ${e.reason})`));
		this.ws.addEventListener("error", () => fail(`LingBot socket error at ${url}`));
	}

	static async connect(url: string): Promise<LingBot> {
		const bot = new LingBot(url);
		bot.metadata = unpack((await bot.recv(60_000)) as Buffer) as Record<string, unknown>;
		return bot;
	}

	recv(timeoutMs: number): Promise<Buffer | string> {
		const queued = this.inbox.shift();
		if (queued !== undefined) return Promise.resolve(queued);
		if (this.failed) return Promise.reject(this.failed);
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.failed ??= new Error(`LingBot reply timed out after ${timeoutMs} ms`);
				this.ws.close();
				reject(this.failed);
			}, timeoutMs);
			this.waiters.push({
				resolve: (v) => {
					clearTimeout(timer);
					resolve(v);
				},
				reject: (e) => {
					clearTimeout(timer);
					reject(e);
				},
			});
		});
	}

	async infer(obs: Record<string, unknown>): Promise<Record<string, unknown>> {
		if (this.failed) throw this.failed;
		this.ws.send(Buffer.concat(pack(obs)));
		const reply = await this.recv(READ_MS);
		if (typeof reply === "string") throw new Error(`LingBot server error:\n${reply}`);
		return unpack(reply) as Record<string, unknown>;
	}
}

export default function robotwin(pi: ExtensionAPI) {
	const flag = (name: string, fallback: string) => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("task-name", { type: "string", default: "beat_block_hammer", description: "RoboTwin task" });
	pi.registerFlag("task-config", {
		type: "string",
		default: "demo_randomized",
		description: "demo_randomized | demo_clean",
	});
	pi.registerFlag("seed", { type: "string", default: "100000", description: "Exact RoboTwin scene seed" });
	pi.registerFlag("max-episode-steps", { type: "string", default: "10000", description: "Native action budget" });
	pi.registerFlag("assets", {
		type: "string",
		default: process.env.ROBOTWIN_ASSETS_PATH ?? "",
		description: "RoboTwin asset snapshot",
	});
	pi.registerFlag("lingbot", { type: "string", default: "ws://127.0.0.1:18400", description: "LingBot-VLA server" });
	pi.registerFlag("env", { type: "string", description: "Attach to a running env server instead of starting one" });
	const seeds = vlaSeeds(pi, () => ["robotwin", robot.task]);
	pi.registerFlag("services", {
		type: "string",
		default: process.env.PI_EMBODIED_SERVICES ?? SERVICES,
		description: "pi-embodied services dir",
	});
	pi.registerFlag("python", {
		type: "string",
		default: process.env.PI_EMBODIED_PYTHON ?? "python",
		description: "Python for the env server",
	});

	let env: RpcClient;
	let lingbot: LingBot | undefined;
	let info: Info;
	let language = "";
	let snapshots: Snapshot[] = [];
	let policyActions = 0;
	let nativeActions = 0;
	const cell = () => ({ task: robot.task["task-name"], config: robot.task["task-config"], seed: robot.task.seed });
	const tag = (seed: string) => `robotwin_${cell().task}_${cell().config}_s${seed}`;
	/** Local corpora (and exploration) key the seed-0 reference like the cell; the published HF corpus by task. */
	const local = () => pi.getFlag("memory-profile") === "local" || pi.getFlag("explore") === true;
	/** --units: one arm's move as move_to / rotate_wrist / set_gripper make it, then a new recorded state. */
	async function unitMove(m: Move) {
		const arm = m.arm as Arm;
		if (success() || exhausted())
			return {
				content: [
					{
						type: "text" as const,
						text: `Episode is terminal (eval_success=${success()}, budget_exhausted=${exhausted()}); call finish.`,
					},
				],
				details: {},
			};
		const result: Record<string, unknown> = {};
		if (m.gripper) result.gripper = await setGripper(arm, m.gripper === "close" ? 0 : 1, 10);
		const current = pose(arm);
		const half = m.yaw / 2;
		const quat = m.yaw ? qmult([Math.cos(half), 0, 0, Math.sin(half)], current.slice(3)) : undefined;
		if (m.yaw || m.delta.some((v) => v !== 0))
			result.move = await moveTo(
				arm,
				current.slice(0, 3).map((v, k) => v + m.delta[k]),
				quat,
				undefined,
				25,
			);
		return present(await capture({ action: "act", arm, delta: m.delta, yaw: m.yaw, gripper: m.gripper }, result));
	}

	const robot = defineRobot(pi, {
		name: "robotwin",
		task: ["task-name", "task-config", "seed"],
		// The env server's primitive registry (code.api), recorded per episode.
		codeApi: () => env,
		keepImages: 6,
		imageStub: "[older camera frame omitted; view_env_state(step) re-reads it]",
		budget: { turns: 100, seconds: 4800 },
		// Observations carry the head, left wrist and right wrist images.
		vdm: { views: VIEWS.length, wrist: [1, 2] },
		flywheel: { spec: FLYWHEEL, select: () => `${cell().config}/${cell().task}` },
		flash: recipeFlash(pi, {
			// This cell's program, else the task's seed-0 reference (local and HF memory name it differently).
			names: () => [tag(cell().seed), tag("0"), `${cell().task}_s0`],
			memory: () => robot.mem?.render("{{memory_dir}}") ?? "",
			observe: "view_env_state",
			targets: { move_to: "xyz" },
			// Molmo points in the head view; the same step's metric depth gives the world point.
			backProject: async (fr, [col, row]) => {
				const [r] = await fr.act([
					{ name: "sample_world_xyz", arguments: { view: "head", pixels: [[Math.round(row), Math.round(col)]] } },
				]);
				const xyz = (r.json.samples as { xyz?: number[] }[] | undefined)?.[0]?.xyz;
				return r.error === undefined && Array.isArray(xyz) ? xyz : undefined;
			},
			over: (latest) => latest.json.eval_success === true || latest.json.budget_exhausted === true,
			solved: (latest) => latest.json.eval_success === true,
			// Motion tools answer "Episode is terminal (eval_success=.., budget_exhausted=..)" once it is over.
			textResult: (text) =>
				/^Episode is terminal/.test(text)
					? { eval_success: /eval_success=true/.test(text), budget_exhausted: /budget_exhausted=true/.test(text) }
					: undefined,
		}),
		units: {
			// RoboTwin's world frame (the robot faces +y, right arm on +x): the head view's directions.
			vectors: {
				MV_FWD: [0, 1, 0],
				MV_BACK: [0, -1, 0],
				MV_LEFT: [-1, 0, 0],
				MV_RIGHT: [1, 0, 0],
				MV_UP: [0, 0, 1],
				MV_DOWN: [0, 0, -1],
			},
			stepM: 0.02,
			yawStepRad: 0.15,
			arms: ["left", "right"],
			instruction: () => language,
			views: UNITS_VIEWS,
			// left_wrist and right_wrist, the second and third images.
			wrist: true,
			apply: (m) => unitMove(m),
			// RoboTwin's gripper is normalized (0 closed .. 1 open), not a width in metres: no empty-grasp check.
			state: async (arm) => ({
				eef_xyz: pose(arm as Arm)
					.slice(0, 3)
					.map((v) => round(v)),
				gripper_opening: grip(arm as Arm),
			}),
			plugins: ["proprioception", "variable_step", "action_chunk", "rotation", "plan", "mem_text"],
		},
		memory: {
			cell: () => ({ tag: tag(cell().seed), reference: local() ? tag("0") : `${cell().task}_s0` }),
			primitives: ["lingbot_act", "move_to", "rotate_wrist", "set_gripper", "release"],
		},
		video: true,
		groundTruth: (names) => env.call("env.ground_truth_poses", { names: names ?? null }, READ_MS, [], robot.signal),
		explore: {
			// The exploration `reset` tool is not a robot.tool, so robot.signal is unset here; use its own signal.
			reset: async (result, _ctx, signal) => {
				const [, reset] = await env.call<[unknown, Info]>("env.reset", {}, MUTATE_MS, [], signal);
				info = reset;
				// Action counts are per attempt, like the env's take_action_cnt that the reset zeroed.
				policyActions = nativeActions = 0;
				seeds.reset();
				language = reset.instruction ?? (await env.call<string>("env.get_task_language"));
				await startFlywheel();
				return present(await capture({ action: "reset" }, { ...result, success: true, instruction: language }));
			},
			prompt: () =>
				EXPLORE.replaceAll("{{task_name}}", cell().task)
					.replaceAll("{{task_config}}", cell().config)
					.replaceAll("{{seed}}", cell().seed),
			rewrite: [
				[
					/Satisfy the complete task in one no-restart episode\. Prefer one accurate sequence over broad exploration, and protect every achieved subgoal\./,
					"This is an exploration run: `reset` starts a fresh attempt (see Exploration). Within an attempt, prefer one accurate sequence and protect every achieved subgoal.",
				],
			],
		},
		start: startEpisode,
		stop: () => {
			lingbot?.ws.close();
			lingbot = undefined;
		},
		prompt: () => {
			const mem = robot.mem!;
			const vars: Record<string, string> = {
				task_language: language,
				task_name: cell().task,
				task_config: cell().config,
				seed: cell().seed,
				memory: pi.getFlag("explore") === true ? "" : mem.render(MEMORY).trim(),
			};
			return SYSTEM.replace(/\{\{(\w+)\}\}/g, (_, k: string) => vars[k] ?? "");
		},
		result: () => ({
			task_name: cell().task,
			task_config: cell().config,
			seed: Number(cell().seed),
			task_language: language,
			success: success(),
			budget_exhausted: exhausted(),
			take_action_cnt: status().take_action_cnt,
			policy_actions: policyActions,
			native_actions: nativeActions,
			states: snapshots.length,
		}),
		status: () => ({ language, step: snapshots.length - 1, solved: success() }),
		finish: {
			description:
				"Stop the run after a fresh status check. RoboTwin's eval_success is authoritative; requesting success cannot override it.",
			parameters: Type.Object({ status: Type.String({ description: "success | failure" }), summary: Type.String() }),
			result: (params) => {
				const requested = params.status.toLowerCase() === "success";
				const result = {
					status: success() ? "success" : requested ? "failure" : params.status,
					summary: params.summary,
					requested_success: requested,
					success: success(),
					episode_status: status(),
				};
				return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
			},
		},
	});
	const fly = robot.fly!;
	/** A Flywheel episode from the reset scene (--collect-flywheel-data): raw/robotwin/<config>/<task>/seed_NNN. */
	async function startFlywheel() {
		if (pi.getFlag("collect-flywheel-data") !== true) return;
		const f = await env.call<PolicyFrame>("env.policy_frame", {}, READ_MS);
		const { task, config, seed } = cell();
		fly.reset(flyObs(f), {
			path: [config, task, `seed_${seed.padStart(3, "0")}`],
			metadata: { task_config: config, task_name: task, seed: Number(seed), task_language: language },
		});
	}

	const status = () => info.episode_status;
	const success = () => status().eval_success === true;
	const exhausted = () => status().step_lim !== null && status().take_action_cnt >= (status().step_lim as number);
	const pose = (arm: Arm) => (info.robot_state[`${arm}_eef_pose`] as NdArray).toArray();
	const grip = (arm: Arm) => Number(info.robot_state[`${arm}_gripper`]);

	async function vla(): Promise<LingBot> {
		if (lingbot && !lingbot.failed) return lingbot;
		lingbot?.ws.close();
		lingbot = await LingBot.connect(flag("lingbot", ""));
		for (const [k, v] of Object.entries(LINGBOT_CONTRACT))
			if (lingbot.metadata[k] !== v)
				throw new Error(`LingBot metadata ${k}=${JSON.stringify(lingbot.metadata[k])}, expected ${v}`);
		return lingbot;
	}

	function completion(requested: number, executed: number) {
		const completed = executed === requested;
		return {
			completed,
			requested_steps: requested,
			executed_steps: executed,
			stop_reason: success()
				? "native_success"
				: exhausted()
					? "budget_exhausted"
					: completed
						? "completed"
						: "runtime_failure",
		};
	}

	/** One qpos14 env step per update, each composed from the latest commanded qpos. */
	async function applyQpos(updates: { arm: Arm; arm_qpos?: number[]; gripper?: number }[]) {
		let executed = 0;
		for (const u of updates) {
			const action = (info.robot_state.qpos_target14 as NdArray).toArray();
			const offset = u.arm === "left" ? 0 : 7;
			if (u.arm_qpos) action.splice(offset, 6, ...u.arm_qpos);
			if (u.gripper !== undefined) action[offset + 6] = u.gripper;
			const ret = await env.call<StepReturn>(
				"env.step",
				{ action_type: "qpos" },
				MUTATE_MS,
				[f64(action)],
				robot.signal,
			);
			info = ret[4];
			const main = (ret[0] as { main_images?: unknown } | null)?.main_images;
			if (main instanceof NdArray) robot.video.frame(u8(main));
			executed += ret[4].executed_actions ?? 0;
			// Flywheel: a scripted qpos step, recorded in the policy's eef16 space as the pose it reached.
			if (fly.recording) {
				const f = await env.call<PolicyFrame>("env.policy_frame", {}, READ_MS);
				fly.transition(f.state.toArray(), flyObs(f), success() ? 1 : 0, success(), exhausted());
			}
			if (success() || exhausted()) break;
		}
		nativeActions += executed;
		return { action_type: "qpos", requested_actions: updates.length, executed_actions: executed };
	}

	async function moveTo(
		arm: Arm,
		xyz: number[],
		quat: number[] | undefined,
		gripper: number | undefined,
		substeps: number,
	) {
		if (substeps < 0) throw new Error("substeps must be non-negative");
		const target = [...xyz, ...(quat ?? pose(arm).slice(3))];
		const planned = await env.call<{ status: string; position: NdArray | null }>(
			"env.plan_arm_path",
			{ arm, target_pose: target },
			READ_MS,
			[],
			robot.signal,
		);
		if (planned.status !== "Success" || !planned.position)
			return {
				completed: false,
				requested_steps: 0,
				executed_steps: 0,
				stop_reason: "plan_failed",
				success: false,
				plan_status: planned.status,
				hint: "target may be unreachable or in collision",
			};
		const [n, dof] = planned.position.shape;
		const flat = planned.position.toArray();
		let path = Array.from({ length: n }, (_, i) => flat.slice(i * dof, (i + 1) * dof));
		if (substeps === 1) path = path.slice(-1);
		else if (substeps >= 2 && path.length > substeps) path = linspace(n, substeps).map((i) => path[i]);
		const execution = await applyQpos(path.map((arm_qpos) => ({ arm, arm_qpos, gripper })));
		const final = pose(arm);
		return {
			...execution,
			...completion(path.length, execution.executed_actions),
			success: true,
			plan_status: planned.status,
			waypoints: path.length,
			final_eef_xyz: final.slice(0, 3).map((v) => round(v)),
			final_dist_m: round(Math.hypot(...xyz.map((v, i) => v - final[i]))),
		};
	}

	async function setGripper(arm: Arm, val: number, steps: number) {
		if (steps < 1) throw new Error("steps must be at least 1");
		const current = grip(arm);
		const updates = Array.from({ length: steps }, (_, i) => ({
			arm,
			gripper: current + ((val - current) * (i + 1)) / steps,
		}));
		const execution = await applyQpos(updates);
		return {
			...execution,
			...completion(updates.length, execution.executed_actions),
			success: true,
			gripper_val: round(grip(arm)),
		};
	}

	/** Record a new numbered state: images, same-step world maps, robot state and native status. */
	async function capture(command: Record<string, unknown>, result: Record<string, unknown>): Promise<Snapshot> {
		const images: Buffer[] = [];
		const world = {} as Record<View, WorldMap>;
		const specs: Record<string, number[]> = {};
		for (const view of VIEWS) {
			const [rgb, depth] = await env.call<[NdArray, NdArray]>(
				"env.render_camera",
				{ camera_name: view, depth: true },
				READ_MS,
			);
			const meta = await env.call<CameraMeta>("env.get_camera_meta", { camera_name: view }, READ_MS);
			const [h, w, ch] = rgb.shape;
			let pixels = rgb.data;
			if (ch !== 3) {
				pixels = Buffer.alloc(h * w * 3);
				for (let i = 0; i < h * w; i++) rgb.data.copy(pixels, i * 3, i * ch, i * ch + 3);
			}
			images.push(encodePng(pixels, w, h));
			world[view] = worldMap(depth, meta);
			specs[view] = [h, w];
		}
		const step = snapshots.length;
		const limit = status().step_lim;
		const payload = {
			step,
			command,
			result,
			eval_success: success(),
			budget_exhausted: exhausted(),
			episode_status: {
				...status(),
				remaining_steps: limit === null ? null : limit - status().take_action_cnt,
				policy_actions: policyActions,
				native_actions: nativeActions,
			},
			task_language: language,
			robot_state: Object.fromEntries(Object.entries(info.robot_state).map(([k, v]) => [k, plain(v)])),
			view_specs: specs,
			images: VIEWS.map((v) => `${v} ${specs[v][1]}x${specs[v][0]} (pixel [row, col])`),
		};
		const snap = { payload, images, world };
		snapshots.push(snap);
		return snap;
	}

	const present = (snap: Snapshot) => ({
		content: [
			{ type: "text" as const, text: JSON.stringify(snap.payload) },
			...snap.images.map((png) => ({ type: "image" as const, data: png.toString("base64"), mimeType: "image/png" })),
		],
		details: {
			step: snap.payload.step,
			result: snap.payload.result,
			eval_success: snap.payload.eval_success,
			// Memory recipes and exploration read the env's success as `terminated`.
			terminated: snap.payload.eval_success,
		},
	});

	/** Look up a recorded state; -1 (default) is the latest, other negatives count from the end. */
	function recorded(view: View, step: number | undefined) {
		const i = step === undefined ? snapshots.length - 1 : step < 0 ? snapshots.length + step : step;
		const snap = snapshots[i];
		if (!snap)
			return { error: { success: false, error: { code: "state_not_found", message: "No such state.", step } } };
		return { step: i, map: snap.world[view] };
	}

	/**
	 * Register a tool. `act` tools change the episode and are refused once it ended; `act`
	 * and `observe` tools return a new recorded state; `read` tools return their result.
	 */
	function tool<P extends TSchema>(
		name: string,
		description: string,
		parameters: P,
		run: (p: Static<P>) => Promise<Record<string, unknown>>,
		kind: "act" | "observe" | "read" = "act",
	) {
		robot.tool(name, description, parameters, async (params) => {
			if (kind === "act" && (success() || exhausted())) {
				return {
					content: [
						{
							type: "text" as const,
							text: `Episode is terminal (eval_success=${success()}, budget_exhausted=${exhausted()}); call finish.`,
						},
					],
					details: {},
				};
			}
			const result = await run(params);
			if (kind === "read")
				return { content: [{ type: "text" as const, text: JSON.stringify(result) }], details: result };
			return present(await capture({ action: name, ...params }, result));
		});
	}

	const arm = StringEnum(["left", "right"] as const);
	const view = StringEnum(["head", "left_wrist", "right_wrist"] as const, {
		description: "View whose RGB supplied the pixels; also the pixel coordinate space",
	});
	const step = Type.Optional(Type.Integer({ description: "Recorded state; -1 (default) = latest, 0 = initial" }));
	const gripper = Type.Optional(
		Type.Number({ description: "Gripper command held along the path, 0 closed .. 1 open; omit to keep it" }),
	);
	const substeps = Type.Optional(
		Type.Integer({ minimum: 0, description: "Planner waypoints executed, evenly subsampled (default 25; 0 = all)" }),
	);

	robot.tool(
		"view_env_state",
		"Read one recorded state (step -1 = latest, 0 = initial): command, result, native episode status, robot state, and the head, left wrist and right wrist RGB images.",
		Type.Object({ step }),
		async (params) => {
			const i = params.step === undefined ? -1 : params.step;
			const snap = snapshots[i < 0 ? snapshots.length + i : i];
			if (!snap) return { content: [{ type: "text" as const, text: `state step ${i} not available` }], details: {} };
			return present(snap);
		},
	);

	tool(
		"render",
		"Capture a fresh synchronized observation as a new state without moving.",
		Type.Object({}),
		async () => ({
			success: true,
		}),
		"observe",
	);

	tool(
		"sample_world_xyz",
		"Median world xyz (m) around [row, col] pixels of a recorded state's view, from same-step metric depth. Read-only.",
		Type.Object({
			view,
			pixels: Type.Array(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 }), { minItems: 1, maxItems: 256 }),
			step,
			neighborhood: Type.Optional(
				Type.Integer({ minimum: 0, maximum: 32, description: "Radius in pixels (default 1)" }),
			),
		}),
		async ({ view: v, pixels, step: s, neighborhood = 1 }) => {
			const found = recorded(v, s);
			if (found.error) return found.error;
			const { height, width, xyz } = found.map as WorldMap;
			const samples = [];
			for (const [row, col] of pixels as number[][]) {
				if (row < 0 || row >= height || col < 0 || col >= width)
					return {
						success: false,
						error: {
							code: "pixel_out_of_bounds",
							message: "Pixel outside this view. Use the view whose RGB supplied the pixel.",
							pixel: [row, col],
							shape: [height, width],
						},
					};
				const axes: number[][] = [[], [], []];
				let validPoints = 0;
				for (let r = Math.max(0, row - neighborhood); r < Math.min(height, row + neighborhood + 1); r++) {
					for (let c = Math.max(0, col - neighborhood); c < Math.min(width, col + neighborhood + 1); c++) {
						const p = [0, 1, 2].map((k) => xyz[(r * width + c) * 3 + k]);
						for (let k = 0; k < 3; k++) if (Number.isFinite(p[k])) axes[k].push(p[k]);
						if (p.every(Number.isFinite)) validPoints++;
					}
				}
				if (axes.some((a) => a.length === 0))
					return {
						success: false,
						error: {
							code: "no_valid_world_points",
							message: "The pixel neighborhood has no finite xyz.",
							pixel: [row, col],
							neighborhood,
						},
					};
				samples.push({
					pixel: [row, col],
					xyz: axes.map((a) => round(median(a))),
					valid_points: validPoints,
					valid_coordinates: axes.map((a) => a.length),
				});
			}
			return {
				success: true,
				step_idx: found.step,
				view: v,
				image_shape: [height, width],
				frame: "world",
				unit: "metre",
				samples,
			};
		},
		"read",
	);

	tool(
		"query_world_map",
		"World-xyz samples and min/max/median stats over a half-open [row_start, col_start, row_end, col_end] box of a recorded state's view. Read-only.",
		Type.Object({
			view,
			bbox: Type.Array(Type.Integer(), { minItems: 4, maxItems: 4 }),
			step,
			max_points: Type.Optional(Type.Integer({ minimum: 1, maximum: 4096, description: "Default 256" })),
		}),
		async ({ view: v, bbox, step: s, max_points = 256 }) => {
			const found = recorded(v, s);
			if (found.error) return found.error;
			const { height, width, xyz } = found.map as WorldMap;
			const [r0, c0, r1, c1] = bbox as number[];
			if (!(r0 >= 0 && r0 < r1 && r1 <= height && c0 >= 0 && c0 < c1 && c1 <= width))
				return {
					success: false,
					error: {
						code: "bbox_out_of_bounds",
						message: "bbox must be a non-empty half-open region inside this view.",
						bbox,
						valid_bbox: [0, 0, height, width],
					},
				};
			const pts: { pixel: number[]; xyz: number[] }[] = [];
			for (let r = r0; r < r1; r++) {
				for (let c = c0; c < c1; c++) {
					const p = [0, 1, 2].map((k) => xyz[(r * width + c) * 3 + k]);
					if (p.every(Number.isFinite)) pts.push({ pixel: [r, c], xyz: p });
				}
			}
			if (!pts.length)
				return {
					success: false,
					error: { code: "no_valid_world_points", message: "No finite world xyz in the box.", bbox },
				};
			const axis = (k: number) => pts.map((p) => p.xyz[k]);
			const picked = pts.length > max_points ? linspace(pts.length, max_points).map((i) => pts[i]) : pts;
			return {
				success: true,
				step_idx: found.step,
				view: v,
				image_shape: [height, width],
				bbox: [r0, c0, r1, c1],
				frame: "world",
				unit: "metre",
				valid_points: pts.length,
				returned_points: picked.length,
				xyz_min: [0, 1, 2].map((k) => round(Math.min(...axis(k)))),
				xyz_max: [0, 1, 2].map((k) => round(Math.max(...axis(k)))),
				xyz_median: [0, 1, 2].map((k) => round(median(axis(k)))),
				points: picked.map((p) => ({ pixel: p.pixel, xyz: p.xyz.map((x) => round(x)) })),
			};
		},
		"read",
	);

	tool(
		"lingbot_act",
		"Run LingBot-VLA eef16 action chunks (50 native actions each) on the native task instruction. The optional prompt is recorded but never sent to the policy.",
		Type.Object({
			chunks: Type.Optional(Type.Integer({ minimum: 1, description: "Default 4" })),
			use_length: Type.Optional(
				Type.Integer({ minimum: USE_LENGTH, maximum: USE_LENGTH, description: "Must be 50" }),
			),
			prompt: Type.Optional(Type.String()),
		}),
		async ({ chunks = 4, use_length = USE_LENGTH, prompt }) => {
			if (use_length !== USE_LENGTH) throw new Error(`RoboTwin LingBot requires use_length=${USE_LENGTH}`);
			const policy = await vla();
			let executed = 0;
			let nativePrompt: string | null = null;
			const vla_seeds: (number | null)[] = [];
			for (let i = 0; i < chunks && !success() && !exhausted(); i++) {
				if (robot.signal?.aborted) throw new Error("interrupted");
				nativePrompt = await env.call<string>("env.get_task_language", {}, READ_MS);
				const views: NdArray[] = [];
				for (const v of VIEWS)
					views.push(await env.call<NdArray>("env.render_camera", { camera_name: v, depth: false }, READ_MS));
				const seed = seeds.next();
				vla_seeds.push(seed ?? null);
				const out = await policy.infer({
					...(seed === undefined ? {} : { seed }),
					"observation.images.cam_high": views[0],
					"observation.images.cam_left_wrist": views[1],
					"observation.images.cam_right_wrist": views[2],
					"observation.state": NdArray.f32([...pose("left"), grip("left"), ...pose("right"), grip("right")]),
					task: nativePrompt,
				});
				const actions = out.action;
				if (!(actions instanceof NdArray) || actions.shape.length !== 2 || actions.shape[1] !== 16)
					throw new Error(
						`LingBot returned ${actions instanceof NdArray ? actions.shape : typeof actions}; expected [chunk, 16]`,
					);
				const rows = Math.min(USE_LENGTH, actions.shape[0]);
				const bytes = actions.data.length / actions.shape[0];
				const chunk = new NdArray(actions.dtype, [rows, 16], actions.data.subarray(0, rows * bytes));
				const recording = fly.recording;
				const vlaId = fly.proposal(nativePrompt, chunk);
				// Record the head frame after every native action for the episode video (and, for the
				// Flywheel, what the policy reads after it).
				const ret = await env.call<StepReturn>(
					"env.chunk_step",
					{ action_type: "ee", return_all_frames: true, return_policy_frames: recording },
					MUTATE_MS,
					[chunk],
					robot.signal,
				);
				info = ret[4];
				const returned = ret[0] as Frames | null;
				for (const frame of returned?.frames ?? []) robot.video.frame(u8(frame));
				if (recording) {
					const a = chunk.toArray();
					const [r, te, tr] = [ret[1], ret[2], ret[3]].map((v) => (v as NdArray).toArray());
					(returned?.policy_frames ?? []).forEach((f, i) => {
						fly.transition(
							a.slice(i * 16, (i + 1) * 16),
							flyObs(f),
							r[i],
							Boolean(te[i]),
							Boolean(tr[i]),
							vlaId,
							i,
						);
					});
				}
				const count = ret[4].executed_actions ?? 0;
				executed += count;
				policyActions += count;
				nativeActions += count;
			}
			return {
				...completion(chunks * USE_LENGTH, executed),
				success: true,
				prompt: nativePrompt,
				agent_prompt_ignored: prompt !== undefined,
				ignored_agent_prompt: prompt ?? null,
				vla_seeds,
			};
		},
	);

	tool(
		"move_to",
		"Plan (curobo) and move one arm's EEF to a world-frame xyz and wxyz orientation (default: keep current). The planned qpos waypoints are executed from fresh state. EEF is not TCP: do not send a raw object surface point.",
		Type.Object({
			arm,
			xyz: Type.Array(Type.Number(), { minItems: 3, maxItems: 3, description: "World [x, y, z], m" }),
			quat: Type.Optional(Type.Array(Type.Number(), { minItems: 4, maxItems: 4, description: "[qw, qx, qy, qz]" })),
			gripper,
			substeps,
		}),
		async (p) => moveTo(p.arm, p.xyz, p.quat, p.gripper, p.substeps ?? 25),
	);

	tool(
		"rotate_wrist",
		"Rotate one EEF about world z by a relative angle in degrees (EEF xyz fixed; the TCP and a held object sweep an arc).",
		Type.Object({ arm, delta_yaw_deg: Type.Number(), gripper, substeps }),
		async (p) => {
			const current = pose(p.arm);
			const half = (p.delta_yaw_deg * Math.PI) / 360;
			const quat = qmult([Math.cos(half), 0, 0, Math.sin(half)], current.slice(3));
			return {
				...(await moveTo(p.arm, current.slice(0, 3), quat, p.gripper, p.substeps ?? 25)),
				requested_delta_yaw_deg: p.delta_yaw_deg,
			};
		},
	);

	tool(
		"set_gripper",
		"Linearly move one normalized gripper (0 closed, 1 open) to val over `steps` native actions (default 10).",
		Type.Object({
			arm,
			val: Type.Number({ minimum: 0, maximum: 1 }),
			steps: Type.Optional(Type.Integer({ minimum: 1 })),
		}),
		async (p) => setGripper(p.arm, p.val, p.steps ?? 10),
	);

	tool(
		"release",
		"Open one gripper to val (default 1.0) over `steps` native actions (default 10).",
		Type.Object({ arm, val: Type.Optional(Type.Number()), steps: Type.Optional(Type.Integer({ minimum: 1 })) }),
		async (p) => setGripper(p.arm, p.val ?? 1, p.steps ?? 10),
	);

	async function startEpisode() {
		snapshots = [];
		policyActions = nativeActions = 0;
		seeds.reset();
		const { task, config, seed } = cell();
		const endpoint = pi.getFlag("env") as string | undefined;
		if (endpoint) env = await attach(endpoint, 900_000);
		else {
			const services = flag("services", SERVICES);
			const assets = flag("assets", "");
			if (!assets) throw new Error("set --assets (or ROBOTWIN_ASSETS_PATH) to the RoboTwin asset snapshot");
			env = await robot.serve({
				python: flag("python", "python"),
				args: [
					...["-m", "pi_embodied_services.robots.robotwin.env_server"],
					...["--task-name", task, "--task-config", config, "--seed", seed],
					...["--max-episode-steps", flag("max-episode-steps", "10000"), "--assets-path", assets],
				],
				cwd: services,
				env: { ...process.env, PYTHONPATH: services, ROBOTWIN_ASSETS_PATH: assets },
				log: (port) => join(tmpdir(), `pi-embodied-robotwin-${task}-s${seed}-${port}.log`),
				readyMs: 900_000,
			});
		}
		const meta = await env.call<{ task_name: string; task_config: string; seed: number }>("env.get_env_meta");
		if (meta.task_name !== task || meta.task_config !== config || meta.seed !== Number(seed))
			throw new Error(
				`env server runs ${meta.task_name}/${meta.task_config}/${meta.seed}, not ${task}/${config}/${seed}`,
			);
		const [, reset] = await env.call<[unknown, Info]>("env.reset", {}, MUTATE_MS);
		info = reset;
		if (status().actual_seed !== Number(seed))
			throw new Error(`reset used seed ${status().actual_seed}, not ${seed}`);
		language = reset.instruction ?? (await env.call<string>("env.get_task_language"));
		await vla();
		await capture(
			{ action: "reset" },
			{ success: true, instruction: language, instruction_source: reset.instruction_source ?? null },
		);
		await startFlywheel();
		return [
			"view_env_state",
			"render",
			"sample_world_xyz",
			"query_world_map",
			"lingbot_act",
			"move_to",
			"rotate_wrist",
			"set_gripper",
			"release",
			"finish",
		];
	}
}
