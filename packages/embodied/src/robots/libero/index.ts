/**
 * LIBERO robot for pi.
 *
 *   pi -e packages/embodied/src/robots/libero --suite libero_10 --task 2 --seed 0
 *
 * Starts one LIBERO env server per session and attaches to running Pi0.5 VLA and
 * SAM3 servers (see serve.sh). Tools are the LIBERO primitives. Every motion
 * tool returns the new state with agentview and wrist images; success is LIBERO's
 * own `terminated` flag, recorded in the session's `robot_result` entry.
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync, gzipSync } from "node:zlib";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";
import { type FlywheelObs, type FlywheelSpec, flywheelSuite } from "../../capabilities/flywheel.ts";
import { MOLMO, pi05, SAM3 } from "../../infra/model-services.ts";
import { decodePng, decodePngChannel, encodePng } from "../../infra/png.ts";
import { NdArray, RpcClient } from "../../infra/rpc.ts";
import { finishMove, type Move, type UnitsSpec } from "../../modes/units/index.ts";
import { template } from "../../planner/context-version.ts";
import { vlaSeeds } from "../../planner/vla-seed.ts";
import { graspAdvisorTool } from "../../primitives/advisor.ts";
import {
	detectionActive,
	detectionArgs,
	detectionTools,
	type PerceptionCaps,
	registerDetectionFlags,
} from "../../primitives/detections.ts";
import { geometryArgs, geometryTools, runGripPlan } from "../../primitives/geometry.ts";
import {
	DETECTIONS_EXPIRED_ENTRY,
	graspActive,
	graspArgs,
	graspTools,
	isStale,
	mountGraspTool,
	registerGraspFlags,
} from "../../primitives/grasp.ts";
import {
	ikArgs,
	type MotionCheck,
	type MotionPlan,
	planRefusal,
	type Reach,
	reachRefusal,
	registerIkFlag,
} from "../../primitives/ik.ts";
import { pointActive, pointTool, registerPointFlags } from "../../primitives/pointing.ts";
import {
	PICK_PARAMETERS,
	type PickParams,
	pickDescription,
	pickTracker,
	suiteMismatch,
	VLA_ADAPTERS,
	vlaIdentity,
	vlaInfo,
} from "../../primitives/vla-adapters.ts";
import { waypointsTool } from "../../primitives/waypoints.ts";
import { alignWristTool, projectPoints } from "../../primitives/wrist.ts";
import { attach, defineRobot, mark, median, message, rgbOf, SERVICES } from "../../robot.ts";
import { liberoFlash } from "./flash.ts";

const read = (name: string) => template(new URL(name, import.meta.url));
/**
 * `--libero-prompt rpent` (default): RPent's LIBERO evaluate and explore prompts (robots/libero/prompts,
 * eecf206) with pi's tool names, and its three guides, which the agent reads with `read` (PROMPT_PORT.md).
 */
const RPENT = { system: read("./SYSTEM.md"), explore: read("./explore.md"), distil: read("./distil.md") };
/** `--libero-prompt compact`: the short prompt pi-embodied used before, kept for comparison. */
const COMPACT = {
	system: read("./compact/SYSTEM.md"),
	explore: read("./compact/explore.md"),
	distil: read("./compact/distil.md"),
	memory: { hf: read("./compact/memory-hf.md"), local: read("./compact/memory-local.md") },
};
export const LIBERO_PROMPTS = ["rpent", "compact"] as const;
/** RPent's guides, read-only to the agent (the memory guard's `readable`). */
export const GUIDES = fileURLToPath(new URL("./guides", import.meta.url));

/**
 * Render an RPent-port template for a memory profile: explore.md's `[include:x]` lines take SYSTEM.md's
 * `[part:x]` sections (as RPent's explore.py imports evaluate.py's), `[memory:hf|local]` blocks follow the
 * profile, and `#.` workflow steps are numbered in order. `[tool:x]` blocks are the robot base's.
 */
export function renderRpent(text: string, profile: string, parts = RPENT.system): string {
	const shared = new Map(
		[...parts.matchAll(/^\[part:([\w-]+)\]\n([\s\S]*?)\n\[\/part:\1\]$/gm)].map((m) => [m[1], m[2]]),
	);
	let n = 0;
	return text
		.replace(/^\[include:([\w-]+)\]$/gm, (_, name: string) => {
			const part = shared.get(name);
			if (part === undefined) throw new Error(`[include:${name}] names no [part:${name}] in SYSTEM.md`);
			return part;
		})
		.replace(/^\[\/?part:[\w-]+\]\n/gm, "")
		.replace(/^\[memory:(hf|local)\]\n([\s\S]*?)^\[\/memory:\1\]\n/gm, (_, p: string, body: string) =>
			p === profile ? body : "",
		)
		.replace(/^#\. /gm, () => `${++n}. `)
		.replace(/\n{3,}/g, "\n\n");
}
/** The compact prompt's single-episode lines, which its exploration replaces rather than contradicts. */
const REWRITE: [RegExp, string][] = [
	[
		/^This is a single episode\..*$/m,
		"This is an exploration run: `reset` starts a fresh episode (see Exploration). The task is done when a tool result shows `terminated: true`; that flag is the only success signal.",
	],
	[
		/^11\. .*$/m,
		"11. Keep reasoning to one or two sentences before each tool call. When an episode is unrecoverable, close it out and `reset`; when `terminated` is true, run DISTIL, then `finish` (see Exploration).",
	],
];
const PRIMITIVES = [
	"move_to",
	"pi0_pick",
	"pi0_doubled",
	"release",
	"set_gripper",
	"rotate_wrist",
	"rotate_pitch",
	"move_pose",
];
/**
 * The state-advancing tools a solved episode's recipe records (../memory): the always-on primitives
 * and the planned-grasp executors, which are active with a grasp backend.
 */
export const RECIPE_PRIMITIVES = [...PRIMITIVES, "execute_grasp", "execute_place"];
/** `preview_reach` is served only with --ik (startEpisode filters it out otherwise). */
const TOOLS = [
	...PRIMITIVES,
	"view_env_state",
	"view_camera_meta",
	"segment",
	"back_project",
	"preview_reach",
	"finish",
];
const CAMERAS = { agentview: "agentview", wrist: "robot0_eye_in_hand" } as const;
/** Units mode (../units): how the two images look, and which way each unit moves in them. */
const VIEWS = `Each result shows the agentview, then the wrist view (verified in LIBERO: MV_FWD is world +x, MV_LEFT is -y).
- Agentview (first image) faces the robot, whose base is at the image top: MV_LEFT / MV_RIGHT move the gripper toward the image left / right, MV_FWD toward the image bottom (toward the camera), MV_BACK toward the image top.
- Wrist view (second image) looks straight down from the gripper; the two fingers are at the image bottom corners and the grasp point is between them, at the horizontal center just above the fingers. It is turned half around relative to the agentview: MV_FWD moves toward the wrist image TOP, MV_BACK toward its bottom, MV_LEFT toward its RIGHT and MV_RIGHT toward its LEFT. So a target above the grasp point in the wrist image needs MV_FWD, one to its right needs MV_LEFT.`;
type Camera = keyof typeof CAMERAS;
type Json = Record<string, any>;
/** One env step of a server motion (`tool_call`, env_server.py `_for_tools`). */
type Transition = { action: NdArray; obs: Obs; reward: unknown; terminated: boolean; truncated: boolean };
type Obs = { main_images: NdArray; wrist_images?: NdArray | null; states: NdArray };
/** The Pi0.5 policy input and output LIBERO records (services robots/libero/flywheel.py). */
const FLYWHEEL: FlywheelSpec = {
	robot: "libero",
	images: { main_images: [256, 256, 3], wrist_images: [256, 256, 3] },
	state: 8,
	action: 7,
};
const flyObs = (o: Obs): FlywheelObs => ({
	images: { main_images: o.main_images, wrist_images: o.wrist_images },
	state: o.states.toArray(),
});
type StepReturn = [Obs, unknown, boolean | NdArray, boolean | NdArray, unknown];
type ChunkReturn = [Obs[], NdArray, NdArray, NdArray, unknown];
type CameraMeta = { intrinsic_K: number[][]; extrinsic_cam2world: number[][]; depth_near?: number; depth_far?: number };
type WorldMap = { envStep: number; size: number; rgb: Buffer; xyz: Float32Array };
/** One camera's view of a state: image rows top first, metric depth in the same order, calibration. */
type Shot = { rgb: Buffer; depth: Float32Array; meta: CameraMeta };
/**
 * A state record (RPent's `step`): the state the model was shown, persisted in `dir` with both cameras'
 * 1024 images, depth and calibration, so tools can look back at it (`step`: 0 = initial, -1 = latest).
 */
type StateRecord = { envStep: number; dir: string };
/** Persisted depth is uint16 in units of 0.1 mm (0xffff: none); a pixel's world point moves < 0.1 mm. */
const DEPTH_UNIT_M = 1e-4;
const CAMERA_NAMES = ["agentview", "wrist"] as const;

/** Metric depth (row 0 = image top) from LIBERO's normalized z-buffer (row 0 = image bottom). */
export function metricDepth(raw: number[], meta: CameraMeta, size: number): Float32Array {
	const { depth_near: near, depth_far: far } = meta;
	const out = new Float32Array(size * size);
	for (let r = 0; r < size; r++) {
		const src = (size - 1 - r) * size;
		for (let c = 0; c < size; c++) {
			const z = raw[src + c];
			out[r * size + c] = near !== undefined && far !== undefined ? near / (1 - z * (1 - near / far)) : z;
		}
	}
	return out;
}

/** Per-pixel world xyz from metric depth and calibration (the world map). */
export function worldXyz(depth: Float32Array, meta: CameraMeta, size: number): Float32Array {
	const [[fx, , cx], [, fy, cy]] = meta.intrinsic_K;
	const e = meta.extrinsic_cam2world;
	const xyz = new Float32Array(size * size * 3);
	for (let r = 0; r < size; r++)
		for (let c = 0; c < size; c++) {
			const z = depth[r * size + c];
			const x = ((c - cx) * z) / fx;
			const y = ((r - cy) * z) / fy;
			const i = (r * size + c) * 3;
			for (let k = 0; k < 3; k++) xyz[i + k] = e[k][0] * x + e[k][1] * y + e[k][2] * z + e[k][3];
		}
	return xyz;
}

export const packDepth = (depth: Float32Array) =>
	gzipSync(
		Buffer.from(
			Uint16Array.from(depth, (z) =>
				Number.isFinite(z) && z >= 0 ? Math.min(0xfffe, Math.round(z / DEPTH_UNIT_M)) : 0xffff,
			).buffer,
		),
		{ level: 1 },
	);
export function unpackDepth(gz: Buffer): Float32Array {
	const raw = gunzipSync(gz);
	const u16 = new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2);
	return Float32Array.from(u16, (v) => (v === 0xffff ? Number.NaN : v * DEPTH_UNIT_M));
}

const done = (v: boolean | NdArray) => (v instanceof NdArray ? v.toArray().some(Boolean) : Boolean(v));
/**
 * The env step at which LIBERO first reported success, latched: `previous` once set, else the
 * first success in this step or chunk (`terminated`, per step) counted from `stepsBefore`.
 */
export function latchSuccess(previous: number | undefined, terminated: boolean | NdArray, stepsBefore: number) {
	if (previous !== undefined) return previous;
	const i = (terminated instanceof NdArray ? terminated.toArray() : [terminated]).findIndex(Boolean);
	return i < 0 ? undefined : stepsBefore + i + 1;
}
/**
 * The memory cell of a LIBERO episode: `<suite>_t<task>_s<seed>` (`libero_` dropped), e.g. `spatial_t0_s0`.
 * LIBERO-plus task indices (thousands per suite) name other tasks than standard/pro ones, so plus cells
 * are `<suite>_plus_t<task>_s<seed>` and never read or write a standard/pro cell's audit, recipe or
 * inbox. Standard and pro share cells: their task sets are identical.
 */
export const memoryTag = (suite: string, task: string, seed: string, liberoType: string) =>
	`${suite.replace(/^libero_/, "")}${liberoType === "plus" ? "_plus" : ""}_t${task}_s${seed}`;
const round = (v: number, d = 4) => Number(v.toFixed(d));
/** The longest horizontal move one move_to (or an executor's first leg) makes (env_server MAX_XY_MOVE_M). */
export const MAX_XY_MOVE_M = 0.3;
/** Why a move from `from` to `to` is refused for its xy length, or undefined. */
export function xyRefusal(from: number[], to: number[], what: string): string | undefined {
	const xy = Math.hypot(to[0] - from[0], to[1] - from[1]);
	return xy > MAX_XY_MOVE_M + 1e-9
		? `${what} would move ${xy.toFixed(3)} m in xy, more than ${MAX_XY_MOVE_M} m: split it into waypoints at carry height`
		: undefined;
}
const clip = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Rows of an HxWxC byte image in reverse order (LIBERO renders upside down). */
function flipRows(data: Buffer, height: number, rowBytes: number): Buffer {
	const out = Buffer.alloc(data.length);
	for (let y = 0; y < height; y++) data.copy(out, (height - 1 - y) * rowBytes, y * rowBytes, (y + 1) * rowBytes);
	return out;
}

/** Rotation matrix of an xyzw quaternion. */
function rotation([x, y, z, w]: number[]): number[][] {
	return [
		[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
		[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
		[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
	];
}
/**
 * World-frame rotation vector from the current EEF orientation (xyzw) to the target, or to the target
 * turned half about its approach (the EEF's +z) when that is shorter: the fingers are symmetric.
 */
export function orientationError(current: number[], target: number[]): number[] {
	const Rc = transpose(rotation(current));
	const Rt = rotation(target);
	const flipped = Rt.map((r) => [-r[0], -r[1], r[2]]);
	const [a, b] = [matrixToRotvec(matmul(Rt, Rc)), matrixToRotvec(matmul(flipped, Rc))];
	return Math.hypot(...b) < Math.hypot(...a) ? b : a;
}

/** A claimed grasp or place path (`env.claim_waypoints`, services utils/grasp.py). */
export type Claim = {
	id: string;
	waypoints: Record<string, number[]>;
	steps: { to?: string; gripper: number }[];
	eef_quat_xyzw: number[];
};

/**
 * Run a claimed path leg by leg: servo to each waypoint with the claim's full orientation, or drive the
 * gripper in place. Stops at the episode's end, at a leg the servo refused (--ik: no collision-free
 * path, or predicted contact; `stalled` with its reason as `error`) or at the first leg that ends more
 * than 3 cm short (`stalled`, with the `error` the model reads).
 */
export async function runClaim(
	c: Claim,
	io: {
		servo: (
			target: number[],
			quat: number[],
			g: number,
		) => Promise<{ steps: number; final_dist_m: number; refused?: string }>;
		actuate: (g: number) => Promise<number>;
		width: () => number;
		ended: () => boolean;
		/** After the hand opened: the env server ends the held grasp once the fingers hold nothing. */
		released?: () => Promise<unknown>;
		/** After the fingers closed: the env server measures the grasp height above its support. */
		closed?: () => Promise<unknown>;
	},
): Promise<{ legs: Record<string, unknown>[]; steps_used: number; stalled?: string; error?: string }> {
	const legs: Record<string, unknown>[] = [];
	let steps = 0;
	for (const leg of c.steps) {
		if (io.ended()) break;
		if (leg.to === undefined) {
			steps += await io.actuate(leg.gripper);
			legs.push({ gripper: leg.gripper, gripper_width: round(io.width()) });
			if (leg.gripper > 0) await io.closed?.();
			if (leg.gripper < 0) await io.released?.();
			continue;
		}
		const r = await io.servo(c.waypoints[leg.to], c.eef_quat_xyzw, leg.gripper);
		steps += r.steps;
		legs.push({ to: leg.to, final_dist_m: r.final_dist_m });
		if (r.refused) return { legs, steps_used: steps, stalled: leg.to, error: r.refused };
		// Short of the waypoint because the episode ended is not a stall.
		if (r.final_dist_m > 0.03 && !io.ended())
			return {
				legs,
				steps_used: steps,
				stalled: leg.to,
				error: `stalled ${r.final_dist_m} m short of ${leg.to}: unreachable or blocked; plan again from the new observation`,
			};
	}
	return { legs, steps_used: steps };
}

/** World yaw of an xyzw quaternion: the right-hand angle about base +z (unitStep's `move.yaw` servo). */
export const yawOf = (q: number[]) => {
	const r = rotation(q);
	return Math.atan2(r[1][0], r[0][0]);
};
/**
 * The turn units on LIBERO (base +z up). ROTATE_CW is +yawStepRad about base +z, i.e. counter-clockwise
 * seen from above, as Show-Harness executes it (configs/primitives_franka.yaml `ROTATE_CW: 1.0`, "Signs
 * calibrated so the token matches the turn seen in the WRIST view"; interpreters/real_atomic_controller.py
 * `_target_euler[2] += yaw  # yaw about base +Z`). RT_* (--units-rt): 10 deg about a world axis through
 * the TCP, as the aaroncaozj LIBERO adapters' model card says; the card gives no sign and their dataset's
 * labelled states are gated, so the signs here are unverified: RT_ROLL_LEFT tilts the gripper's top
 * toward MV_LEFT (world -y), RT_PITCH_FWD toward MV_FWD (+x), RT_YAW_CCW is counter-clockwise seen from
 * above (+z). Under these, ROTATE_CW and RT_YAW_CCW are the same physical turn.
 */
export const LIBERO_TURNS = {
	yawStepRad: 0.15,
	rt: { stepRad: Math.PI / 18, axes: { roll: [1, 0, 0], pitch: [0, 1, 0], yaw: [0, 0, 1] } },
} satisfies Pick<UnitsSpec, "yawStepRad" | "rt">;
type Mat3 = number[][];
const matmul = (a: Mat3, b: Mat3) => a.map((row) => [0, 1, 2].map((j) => row.reduce((s, v, k) => s + v * b[k][j], 0)));
const transpose = (a: Mat3) => [0, 1, 2].map((i) => [0, 1, 2].map((j) => a[j][i]));
/** Rodrigues: the rotation matrix of a rotation vector (axis x angle). */
export function rotvecToMatrix(v: number[]): Mat3 {
	const t = Math.hypot(...v);
	if (t < 1e-12) return [1, 0, 0].map((_, i) => [0, 1, 2].map((j) => (i === j ? 1 : 0)));
	const [x, y, z] = v.map((c) => c / t);
	const [c, s, C] = [Math.cos(t), Math.sin(t), 1 - Math.cos(t)];
	return [
		[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
		[y * x * C + z * s, c + y * y * C, y * z * C - x * s],
		[z * x * C - y * s, z * y * C + x * s, c + z * z * C],
	];
}
/** The rotation vector of a rotation matrix (angle in [0, pi]). */
export function matrixToRotvec(r: Mat3): number[] {
	const t = Math.acos(clip((r[0][0] + r[1][1] + r[2][2] - 1) / 2, -1, 1));
	const w = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
	if (t < 1e-9) return [0, 0, 0];
	if (Math.PI - t < 1e-6) {
		// Half turn: the axis is the column of R + I with the largest norm.
		const cols = [0, 1, 2].map((j) => [0, 1, 2].map((i) => r[i][j] + (i === j ? 1 : 0)));
		const norms = cols.map((col) => Math.hypot(...col));
		const a = cols[norms.indexOf(Math.max(...norms))];
		const n = Math.hypot(...a);
		return a.map((v) => (v / n) * t);
	}
	return w.map((v) => (v * t) / (2 * Math.sin(t)));
}

export default function libero(pi: ExtensionAPI) {
	const flag = (name: string, fallback: string) => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("suite", { type: "string", default: "libero_10", description: "LIBERO suite, e.g. libero_10" });
	pi.registerFlag("task", { type: "string", default: "0", description: "Task index within the suite" });
	pi.registerFlag("seed", { type: "string", default: "0", description: "Initial-state seed" });
	pi.registerFlag("libero-type", { type: "string", default: "pro", description: "standard | pro | plus" });
	pi.registerFlag("libero-prompt", {
		type: "string",
		default: "rpent",
		description: "System prompt: rpent (RPent's full evaluate/explore prompts and guides, default) | compact",
	});
	// Each state record is about 2.7 MB (two 1024 images, depth, calibration).
	pi.registerFlag("step-history", {
		type: "string",
		default: "clean",
		description:
			"State records for `step` look-back: clean (default; kept during the session, removed at its end except segments/) | keep | off (no look-back)",
	});
	pi.registerFlag("vla", { type: "string", default: "http://127.0.0.1:18200", description: "Pi0.5 VLA server" });
	// The third-party VLAs (../vla-adapters.ts): `--openvla <url>` mounts `openvla_act`, and so on; unset mounts nothing.
	for (const a of VLA_ADAPTERS)
		pi.registerFlag(a.flag, { type: "string", description: `${a.model} server; mounts ${a.tool}` });
	const seeds = vlaSeeds(pi, () => ["libero", robot.task]);
	pi.registerFlag("sam3", { type: "string", default: "http://127.0.0.1:18300", description: "SAM3 server" });
	registerIkFlag(pi);
	pi.registerFlag("env", { type: "string", description: "Attach to a running env server instead of starting one" });
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
	pi.registerFlag("cuda-device", { type: "string", description: "GPU ordinal for MuJoCo EGL rendering and torch" });
	// 4 mm stops a 2 cm unit about 3.6 mm short (measured: 16.4 mm in 5 steps); the aaroncaozj adapters
	// were labelled with full 2 cm steps.
	pi.registerFlag("unit-tol", {
		type: "string",
		default: "0.004",
		description: "Units mode: an MV_* servo stops within this distance of its target, m",
	});
	// --contact-graspnet/--graspgenx/--anyplace/--anygrasp/--graspnet1b: plan_grasp, plan_place, check_attached (../primitives/grasp.ts).
	registerGraspFlags(pi);
	// --point: Molmo's point as molmo_point (the units' point plugin owns `point`); LIBERO's Flash registers --molmo.
	registerPointFlags(pi);
	// --detections / --unidepth: detect, select_detection, reject_detection, enhance_depth (../primitives/detections.ts).
	registerDetectionFlags(pi);

	let env: RpcClient;
	let vla: RpcClient;
	/** The mounted third-party VLA clients by tool name, and which VLA each grasp tool used this episode (healthz name, model@revision). */
	const adapters = new Map<string, RpcClient>();
	const vlaUsed: Record<string, string> = {};
	let sam3: RpcClient;
	let obs: Obs;
	let terminated = false;
	/** The env step of the first success: the outcome, whatever happens after it (a release, a knock-over). */
	let successStep: number | undefined;
	let truncated = false;
	let envStep = 0;
	let language = "";
	/** The gripper command units mode holds between units: -1 open, +1 closed. */
	let grip = -1;
	/** The table (or floor) height in front of the robot, for units' proprioception and variable_step. */
	let tableZ: number | undefined;
	const worldMaps = new Map<string, WorldMap>();
	/** The episode's state records, the history directory, and segment readings so far. */
	const records: StateRecord[] = [];
	let historyDir = "";
	/** The history is in a temp dir (no output dir): removed whole at the session's end. */
	let tempHistory = false;
	const history = () => flag("step-history", "clean");
	/**
	 * At the session's end: --step-history clean removes the step records and keeps segments/ (the flash
	 * anchors, a few KB); a temp-dir history goes entirely, whatever the mode.
	 */
	pi.on("session_shutdown", () => {
		if (!historyDir || !existsSync(historyDir)) return;
		if (tempHistory) rmSync(join(historyDir, ".."), { recursive: true, force: true });
		else if (history() === "clean")
			for (const d of readdirSync(historyDir))
				if (d.startsWith("step_")) rmSync(join(historyDir, d), { recursive: true, force: true });
	});
	let segments = 0;
	/** Both cameras at the current env step (taken by `snapshot`), and the one past world map in use. */
	let shots: { envStep: number; cams: Record<Camera, Shot> } | undefined;
	let pastMap: { key: string; map: WorldMap } | undefined;

	const tag = () => memoryTag(robot.task.suite, robot.task.task, robot.task.seed, flag("libero-type", "pro"));
	const variant = () => flag("libero-prompt", "rpent");
	const exploring = () => pi.getFlag("explore") === true;
	/** The cell's own template variables (memory's are filled by `mem.render`). */
	const cellVars = (text: string) => {
		const vars: Record<string, string> = { ...robot.task, guides_dir: GUIDES, task_language: language };
		return text.replace(/\{\{(suite|task|seed|guides_dir|task_language)\}\}/g, (_, k: string) => vars[k]);
	};
	const robot = defineRobot(pi, {
		name: "libero",
		// Tools and code primitives: ../../primitives/manifests/libero.json (the env server reads it too).
		manifest: "libero",
		vars: () => ({ cameras: ["agentview", "wrist"], arms: [] }),
		// As the env server's `_has`: the flags pi starts it with.
		capabilities: (c) =>
			({
				sam3: Boolean(flag("sam3", "")),
				// The env server installs env.detect & co. whenever it has a SAM3 server.
				detections: Boolean(flag("sam3", "")),
				ik: Boolean(flag("ik", "")),
				motion: Boolean(flag("ik", "")),
				grasp: graspActive(pi).length > 0,
				place: Boolean(flag("anyplace", "")),
				geometry: pi.getFlag("geometry") === true,
				unidepth: Boolean(String(pi.getFlag("unidepth") ?? "").trim()),
			})[c] ?? false,
		services: { models: [pi05("libero"), SAM3, MOLMO] },
		task: ["suite", "task", "seed"],
		// The env server's primitive registry (code.api), recorded per episode.
		codeApi: () => env,
		keepImages: 4,
		memory: {
			cell: () => ({ tag: tag(), reference: tag().replace(/_s\d+$/, "_s0") }),
			primitives: RECIPE_PRIMITIVES,
			readable: () => (variant() === "rpent" ? [GUIDES] : []),
		},
		video: true,
		flywheel: {
			spec: FLYWHEEL,
			select: () =>
				`${flywheelSuite(robot.task.suite, flag("libero-type", "pro"))}/task_${robot.task.task.padStart(2, "0")}`,
		},
		groundTruth: (names) => call(env, "env.ground_truth_poses", { names: names ?? null }),
		// Observations carry the agentview, then the wrist view.
		vdm: { views: 2, wrist: 1, observe: ["view_env_state"] },
		// --viser: the env server's views (agentview and wrist, with depth and calibration) in 3D.
		viser: { source: "libero", env: () => env },
		flash: liberoFlash(pi, () => ({
			suite: robot.task.suite,
			task: robot.task.task,
			liberoType: flag("libero-type", "pro"),
		})),
		operator: {
			step: () => envStep,
			// In simulation the operator's scene restore is the env's own reset to the episode's initial state.
			reset: async () => {
				await resetEpisode();
				fly.reset(flyObs(obs), flyMeta());
				return { step: envStep, terminated, truncated };
			},
		},
		explore: {
			// The exploration `reset` tool is not a robot.tool, so robot.signal is unset here; use its own signal.
			reset: async (result, _ctx, signal) => {
				await resetEpisode(signal);
				fly.reset(flyObs(obs), flyMeta());
				return observe(result);
			},
			// RPent's explore prompt is the whole system prompt (the robot's own is empty then); the compact one is appended.
			prompt: () => (variant() === "compact" ? COMPACT.explore : cellVars(renderRpent(RPENT.explore, "local"))),
			distil: () => (variant() === "compact" ? COMPACT.distil : RPENT.distil),
			rewrite: REWRITE,
		},
		start: startEpisode,
		prompt: () => {
			if (variant() === "rpent")
				return exploring() ? "" : mem.render(cellVars(renderRpent(RPENT.system, mem.profile)));
			const system = COMPACT.system.replaceAll("{{task_language}}", language);
			// Exploration appends its own memory instructions.
			if (exploring()) return system;
			return `${system}\n\n${mem.render(COMPACT.memory[mem.profile], { task: robot.task.task })}`;
		},
		result: () => ({
			suite: robot.task.suite,
			task: Number(robot.task.task),
			seed: Number(robot.task.seed),
			terminated,
			success_step: successStep ?? null,
			truncated,
			env_steps: envStep,
			vla: vlaUsed,
			libero_prompt: variant(),
		}),
		status: () => ({ language, step: envStep, solved: terminated }),
		// Code mode (../code): the env server runs the program against its manifest's primitives
		// (manifests/libero.json, env_server.py); the result carries the steps it took,
		// LIBERO's flags and the frames.
		code: {
			rpc: () => env,
			instruction: () => language,
			refuse: () => {
				op.check();
				return terminated || truncated
					? `Episode already ended (terminated=${terminated}, truncated=${truncated}).`
					: undefined;
			},
			observe: async (r) => {
				const before = envStep;
				const steps = Number(r.steps) || 0;
				envStep += steps;
				const first = r.success_step;
				if (typeof first === "number" && successStep === undefined) successStep = before + first;
				terminated = successStep !== undefined;
				truncated ||= r.truncated === true;
				if (r.obs) obs = r.obs as Obs;
				for (const f of (r.frames as NdArray[] | undefined) ?? []) video.frame(f);
				worldMaps.clear();
				return observe({ name: "run_code", status: r.status, env_steps: steps });
			},
		},
		units: {
			vectors: {
				MV_FWD: [1, 0, 0],
				MV_BACK: [-1, 0, 0],
				MV_LEFT: [0, -1, 0],
				MV_RIGHT: [0, 1, 0],
				MV_UP: [0, 0, 1],
				MV_DOWN: [0, 0, -1],
			},
			stepM: 0.02,
			...LIBERO_TURNS,
			apply: (move) => unitStep(move),
			state: async () => ({
				eef_xyz: eef().map((v) => round(v)),
				gripper_width: round(gripper()),
				...(tableZ === undefined ? {} : { table_z: tableZ }),
			}),
			instruction: () => language,
			views: VIEWS,
			// robot0_eye_in_hand, the second image.
			wrist: true,
			// Measured: fingers closed on nothing read <= 0.003 (sum of both finger joints); a held can reads ~0.06.
			emptyWidthM: 0.004,
			point: { cameras: ["agentview", "wrist"], locate: (camera, points) => locate(camera as Camera, points) },
		},
		finish: {
			description:
				"End the episode after checking the latest state. Success is LIBERO's terminated flag, not this call.",
			parameters: Type.Object({
				status: StringEnum(["success", "failure"] as const),
				summary: Type.String(),
			}),
			result: (params) => ({
				content: [{ type: "text", text: `Episode finished (terminated=${terminated}).` }],
				details: params,
			}),
		},
	});
	const { video, op } = robot;
	const mem = robot.mem!;
	const fly = robot.fly!;

	/** Every robot RPC carries the running tool's abort signal, so an abort stops motion between calls. */
	const call = <T = unknown>(
		client: RpcClient,
		method: string,
		kwargs: Record<string, unknown> = {},
		timeoutMs = 120_000,
		args: unknown[] = [],
	) => client.call<T>(method, kwargs, timeoutMs, args, robot.signal);

	const states = () => obs.states.toArray();
	const eef = () => states().slice(0, 3);
	const gripper = () => Math.abs(states()[6]) + Math.abs(states()[7]);
	const quat = async () => (await call<Record<string, NdArray>>(env, "env.raw_obs")).robot0_eef_quat.toArray();

	function absorb(ret: StepReturn, steps: number) {
		obs = ret[0];
		successStep = latchSuccess(successStep, ret[2], envStep);
		terminated = successStep !== undefined;
		truncated ||= done(ret[3]);
		envStep += steps;
	}

	const scalar = (v: unknown) => (v instanceof NdArray ? v.toArray()[0] : Number(v));

	/** Every env transition goes to the episode video and the flywheel recorder. */
	function record(action: number[], o: Obs, reward: number, term: boolean, trunc: boolean, vlaId = -1, index = -1) {
		video.frame(o.main_images);
		fly.transition(action, flyObs(o), reward, term, trunc, vlaId, index);
	}

	async function step(action: number[]) {
		op.check();
		const ret = await call<StepReturn>(env, "env.step", {}, 60_000, [NdArray.f32(action)]);
		record(action, ret[0], scalar(ret[1]), done(ret[2]), done(ret[3]));
		absorb(ret, 1);
	}

	/**
	 * A motion the env server runs (the method a program's primitive calls too): its limits, stop
	 * handling and success latch are the server's. `tool_call` returns every env step it took, which
	 * goes to the episode video and the Flywheel recorder as pi's own steps did, and a refusal
	 * (unmoved) as `refused`.
	 */
	async function serverMotion(method: string, params: Record<string, unknown>) {
		op.check();
		const { transitions, ...out } = await call<Json & { transitions?: Transition[] }>(
			env,
			method,
			{ ...params, tool_call: true },
			600_000,
		);
		for (const t of transitions ?? []) {
			const action = t.action.toArray();
			record(action, t.obs, scalar(t.reward), Boolean(t.terminated), Boolean(t.truncated));
			absorb([t.obs, t.reward, Boolean(t.terminated), Boolean(t.truncated), {}], 1);
		}
		return out;
	}
	/** The pose the tools report after a server motion (their result fields before the move to the server). */
	const final = (r: Json) => ({ final_eef_pos: r.eef_pos, ...(r.refused ? { refused: r.refused } : {}) });

	/**
	 * One VLA forward pass (Pi0.5 by default, or a mounted adapter) with `prompt` as the instruction,
	 * executed as one action chunk; returns its seed. The first call of a tool records which VLA answered,
	 * and refuses an adapter whose checkpoint is another suite's fine-tune (it stays unrecorded, so
	 * every call refuses until the right server is up).
	 */
	async function vlaChunk(prompt: string, client = vla, toolName = "pi0") {
		if (!(toolName in vlaUsed)) {
			const info = await vlaInfo(client).catch(() => undefined);
			const why = info && suiteMismatch(info, robot.task.suite);
			if (why) throw new Error(why);
			vlaUsed[toolName] = info ? vlaIdentity(info) : "unknown";
		}
		const wire = {
			main_images: obs.main_images.batched(),
			wrist_images: obs.wrist_images ? obs.wrist_images.batched() : null,
			extra_view_images: null,
			states: NdArray.f32(states(), [1, states().length]),
			task_descriptions: [prompt],
		};
		op.check();
		const seed = seeds.next();
		const options = seed === undefined ? { mode: "eval" } : { mode: "eval", seed };
		const actions = await call<NdArray>(client, "vla.predict", {}, 120_000, [wire, options]);
		const chunk = new NdArray(actions.dtype, actions.shape.slice(1), actions.data);
		const vlaId = fly.proposal(prompt, chunk);
		op.check();
		const [frames, rew, term, trunc, info] = await call<ChunkReturn>(
			env,
			"env.chunk_step",
			{ return_all_frames: true },
			120_000,
			[chunk],
		);
		const [a, r, te, tr] = [chunk.toArray(), rew.toArray(), term.toArray(), trunc.toArray()];
		const width = chunk.shape[1];
		frames.forEach((o, i) => {
			record(a.slice(i * width, (i + 1) * width), o, r[i], Boolean(te[i]), Boolean(tr[i]), vlaId, i);
		});
		absorb([frames[frames.length - 1], rew, term, trunc, info], chunk.shape[0]);
		return seed;
	}

	/** raw/libero/<suite>/task_NN/seed_NNN, as the services' LIBERO spec reads it. */
	const flyMeta = () => {
		const suite = flywheelSuite(robot.task.suite, flag("libero-type", "pro"));
		const [task, seed] = [Number(robot.task.task), Number(robot.task.seed)];
		return {
			path: [suite, `task_${String(task).padStart(2, "0")}`, `seed_${String(seed).padStart(3, "0")}`],
			metadata: { suite, task_id: task, seed, task_language: language },
		};
	};

	/** Restore the episode's initial scene (session start, exploration `reset`); `signal` aborts the env reset. */
	async function resetEpisode(signal = robot.signal) {
		worldMaps.clear();
		startHistory();
		terminated = truncated = false;
		successStep = undefined;
		grip = -1;
		envStep = 0;
		seeds.reset();
		[obs] = await env.call<[Obs, unknown]>("env.reset", {}, 300_000, [], signal);
		tableZ = await surfaceZ().catch(() => undefined);
	}

	/**
	 * A fresh state history for a new episode in `<output dir>/<cell>_steps` (a temp dir without one).
	 * An earlier episode's history (before an exploration `reset`) moves to `<cell>_steps.<n>`, so the
	 * directory and its `segments/segment_NN.json` always describe the episode the recipe is exported
	 * from (flash-generate.ts reads them as anchors).
	 */
	function startHistory() {
		records.length = 0;
		segments = 0;
		shots = undefined;
		pastMap = undefined;
		if (history() === "off") return;
		const out = mem.render("{{output_dir}}");
		tempHistory = !out;
		historyDir = join(out || join(tmpdir(), `pi-embodied-libero-${process.pid}`), `${tag()}_steps`);
		// Only --step-history keep keeps an earlier episode's records; clean drops them at once.
		if (existsSync(historyDir) && history() !== "keep") rmSync(historyDir, { recursive: true, force: true });
		if (existsSync(historyDir)) {
			let n = 1;
			while (existsSync(`${historyDir}.${n}`)) n++;
			renameSync(historyDir, `${historyDir}.${n}`);
		}
		mkdirSync(join(historyDir, "segments"), { recursive: true });
	}

	async function render(camera: Camera, size: number, depth: boolean) {
		// The env returns [rgb, depth] with depth, and the bare rgb array without it.
		const out = await call<NdArray | [NdArray, NdArray]>(env, "env.render_camera", {
			camera_name: CAMERAS[camera],
			height: size,
			width: size,
			depth,
		});
		const [rgb, d] = out instanceof NdArray ? [out, null] : out;
		return { rgb: flipRows(rgb.data, size, size * 3), depth: d };
	}

	/** Image, metric depth and calibration of one camera now. */
	async function shoot(camera: Camera, size: number): Promise<Shot> {
		const { rgb, depth } = await render(camera, size, true);
		const meta = await call<CameraMeta>(env, "env.get_camera_meta", {
			camera_name: CAMERAS[camera],
			height: size,
			width: size,
		});
		return { rgb, depth: metricDepth((depth as NdArray).toArray(), meta, size), meta };
	}

	/**
	 * The index of the state record `step` names (default latest; negative from the end), or undefined for
	 * the current state when that is the latest record (tools then read the live env). Refuses an unrecorded step.
	 */
	function past(step: number | undefined): number | undefined {
		if (step === undefined || step === null) return undefined;
		const i = step < 0 ? records.length + step : step;
		if (!records[i]) throw new Error(`step ${step} is not recorded (have 0..${records.length - 1})`);
		if (i === records.length - 1 && records[i].envStep === envStep) return undefined;
		if (history() === "off") throw new Error("earlier steps are not kept (--step-history off); use the latest state");
		return i;
	}

	/** Per-pixel world xyz of the current step (or of state record `step`), from metric depth + calibration. */
	async function worldMap(camera: Camera, size: number, step?: number): Promise<WorldMap> {
		const i = past(step);
		if (i !== undefined) {
			if (size !== 1024) throw new Error(`step ${step} keeps only the 1024 (high) world map`);
			const key = `${camera}@${i}`;
			if (pastMap?.key !== key) {
				const dir = records[i].dir;
				const png = decodePng(readFileSync(join(dir, `${camera}_high.png`)));
				const meta = JSON.parse(readFileSync(join(dir, `${camera}_meta.json`), "utf8")) as CameraMeta;
				const depth = unpackDepth(readFileSync(join(dir, `${camera}_depth_high.u16.gz`)));
				const map = {
					envStep: records[i].envStep,
					size,
					rgb: Buffer.from(png.data),
					xyz: worldXyz(depth, meta, size),
				};
				pastMap = { key, map };
			}
			return pastMap.map;
		}
		const cacheKey = `${camera}:${size}`;
		const cached = worldMaps.get(cacheKey);
		if (cached?.envStep === envStep) return cached;
		const shot = size === 1024 && shots?.envStep === envStep ? shots.cams[camera] : await shoot(camera, size);
		const map = { envStep, size, rgb: shot.rgb, xyz: worldXyz(shot.depth, shot.meta, size) };
		worldMaps.set(cacheKey, map);
		return map;
	}

	/**
	 * The current state as the model sees it, recorded as a new state record when the env moved since the
	 * last one: raw proprioception, both cameras' 1024 images (with depth and calibration, persisted).
	 */
	async function snapshot(result: Record<string, unknown>) {
		const raw = await call<Record<string, NdArray>>(env, "env.raw_obs");
		// One call at a time: concurrent calls interleave on the env server's worker pipe.
		if (shots?.envStep !== envStep)
			shots = { envStep, cams: { agentview: await shoot("agentview", 1024), wrist: await shoot("wrist", 1024) } };
		const cams = shots.cams;
		const state = {
			robot0_eef_pos: raw.robot0_eef_pos.toArray().map((v) => round(v)),
			robot0_eef_quat: raw.robot0_eef_quat.toArray().map((v) => round(v)),
			robot0_gripper_qpos: raw.robot0_gripper_qpos.toArray().map((v) => round(v)),
			object_names: Object.keys(raw)
				.filter((k) => k.endsWith("_pos") && !k.includes("robot0") && !k.includes("to_robot"))
				.map((k) => k.slice(0, -4))
				.sort(),
		};
		const fresh = records.at(-1)?.envStep !== envStep;
		const index = fresh ? records.length : records.length - 1;
		const body = {
			result,
			state_step: index,
			step: envStep,
			terminated,
			truncated,
			task_language: language,
			state,
			images: ["agentview_high 1024x1024", "wrist_high 1024x1024"],
		};
		const pngs = CAMERA_NAMES.map((c) => encodePng(cams[c].rgb, 1024, 1024));
		if (fresh && history() === "off") records.push({ envStep, dir: "" });
		else if (fresh) {
			const dir = join(historyDir, `step_${String(index).padStart(3, "0")}`);
			mkdirSync(dir, { recursive: true });
			writeFileSync(join(dir, "state.json"), `${JSON.stringify(body)}\n`);
			CAMERA_NAMES.forEach((c, k) => {
				writeFileSync(join(dir, `${c}_high.png`), pngs[k]);
				writeFileSync(join(dir, `${c}_depth_high.u16.gz`), packDepth(cams[c].depth));
				writeFileSync(join(dir, `${c}_meta.json`), `${JSON.stringify(cams[c].meta)}\n`);
			});
			records.push({ envStep, dir });
		}
		return { body, pngs };
	}

	const shown = (body: Record<string, unknown>, pngs: Buffer[], details: Record<string, unknown>) => ({
		content: [
			{ type: "text" as const, text: JSON.stringify(body) },
			...pngs.map((png) => ({ type: "image" as const, data: png.toString("base64"), mimeType: "image/png" })),
		],
		details,
	});

	async function observe(result: Record<string, unknown>) {
		const { body, pngs } = await snapshot(result);
		return shown(body, pngs, { result, terminated, truncated });
	}

	/** Write `segment_NN.json` (the anchors flash-generate.ts reads) and its overlay next to its state record. */
	function saveSegment(index: number, reading: Record<string, unknown>, overlay: Buffer | undefined) {
		if (history() === "off") return {};
		const n = String(++segments).padStart(2, "0");
		const name = `segment_${n}.json`;
		writeFileSync(
			join(historyDir, "segments", name),
			`${JSON.stringify({ segment_index: segments, step: index, ...reading })}\n`,
		);
		if (!overlay) return { segment_artifact: name };
		writeFileSync(join(records[index].dir, `segment_overlay_${n}.png`), overlay);
		return { segment_artifact: name, overlay_artifact: `segment_overlay_${n}.png` };
	}

	/** Register a tool; motion tools return a fresh observation, read-only tools return their result. */
	function tool<P extends TSchema>(
		name: string,
		description: string,
		parameters: P,
		run: (p: Static<P>) => Promise<Record<string, unknown>>,
		motion = true,
	) {
		robot.tool(name, description, parameters, async (params) => {
			if (motion && (terminated || truncated)) {
				return {
					content: [
						{
							type: "text",
							text: `Episode already ended (terminated=${terminated}, truncated=${truncated}).`,
						},
					],
					details: { terminated, truncated },
				};
			}
			const result = await run(params);
			if (motion) return observe(result);
			const { _image, ...rest } = result as { _image?: Buffer };
			const content = [{ type: "text" as const, text: JSON.stringify(rest) }];
			if (!_image) return { content, details: rest };
			return {
				content: [...content, { type: "image" as const, data: _image.toString("base64"), mimeType: "image/png" }],
				details: rest,
			};
		});
	}

	/**
	 * Servo xyz and the full orientation (roll, pitch, yaw; any tilt direction) toward a target each
	 * step: the OSC's rotation delta is the world-frame rotation vector to it (orientationError).
	 */
	async function servoOrientation(
		target: number[],
		quatTarget: number[],
		g: number,
		max_steps = 150,
		{ tol = 0.012, ori_tol = 0.05 } = {},
	) {
		let steps = 0;
		for (; steps < max_steps && !terminated && !truncated; steps++) {
			const diff = target.map((v: number, i: number) => v - eef()[i]);
			const err = orientationError(await quat(), quatTarget);
			const angle = Math.hypot(...err);
			if (Math.hypot(...diff) < tol && angle < ori_tol) break;
			const scale = Math.min(1, 0.08 / Math.max(angle, 1e-9));
			await step([
				...diff.map((d: number) => clip(clip(d, -0.02, 0.02) / 0.05, -1, 1)),
				...err.map((e) => clip((e * scale) / 0.1, -1, 1)),
				g,
			]);
		}
		return { steps, final_dist_m: round(Math.hypot(...target.map((v: number, i: number) => v - eef()[i]))) };
	}

	/** Drive the gripper until the fingers stop (on an object, or fully open / closed), at most 15 steps. */
	async function actuate(g: number) {
		let steps = 0;
		for (let prev = Number.NaN; steps < 15 && !terminated && !truncated; ) {
			await step([0, 0, 0, 0, 0, 0, g]);
			steps++;
			if (steps > 3 && Math.abs(gripper() - prev) < 5e-4) break;
			prev = gripper();
		}
		return steps;
	}

	/**
	 * servoOrientation along a collision-free path (--ik): `env.plan_motion` to the target with the
	 * target orientation, `env.check_motion` before each of its segments, the last segment at
	 * servoOrientation's own tolerances; `refused` when no path exists or contact is predicted. An
	 * unknown plan (no ik service answered) servos straight, as without --ik.
	 */
	async function plannedServo(
		target: number[],
		quatTarget: number[],
		g: number,
		max_steps = 150,
	): Promise<{ steps: number; final_dist_m: number; refused?: string }> {
		const dist = () => round(Math.hypot(...target.map((v: number, i: number) => v - eef()[i])));
		const plan = await call<MotionPlan>(env, "env.plan_motion", { pos: target, quat_xyzw: quatTarget });
		const refusal = planRefusal(plan);
		if (refusal) return { steps: 0, final_dist_m: dist(), refused: refusal };
		if (plan.status !== "planned") return servoOrientation(target, quatTarget, g, max_steps);
		let steps = 0;
		for (let i = 0; i < plan.waypoints.length; i++) {
			const check = await call<MotionCheck>(env, "env.check_motion", { segment: i });
			if (check.status === "contact")
				return { steps, final_dist_m: dist(), refused: `stopped: collision check: ${check.message}` };
			const last = i === plan.waypoints.length - 1;
			const r = await servoOrientation(
				last ? target : plan.waypoints[i].slice(0, 3),
				quatTarget,
				g,
				max_steps - steps,
				last ? {} : { tol: 0.02, ori_tol: Math.PI },
			);
			steps += r.steps;
			if (steps >= max_steps || terminated || truncated) break;
		}
		return { steps, final_dist_m: dist() };
	}

	/**
	 * Execute a planned grasp or place id (../primitives/grasp.ts) from one resolution: the env server
	 * resolves the candidate's whole path at once (`env.claim_waypoints`: pre-grasp, grasp, lift, or
	 * pre-place, place, retreat) and the legs run on those coordinates, since their own steps expire the
	 * id. A stale id is refused unmoved and recorded as `detections_expired`. With --ik the grasp pose
	 * must be reachable and the first leg must have a collision-free path before the claim (refused
	 * unmoved otherwise), and every leg then follows its own planned path (`plannedServo`).
	 */
	async function executePlanned(
		name: string,
		id: string,
		kind: "grasp" | "placement",
		kwargs: Record<string, number>,
	) {
		try {
			const resolved = await call<{ kind: string }>(env, "env.resolve_grasp", { grasp_id: id });
			if (resolved.kind !== kind)
				return {
					name,
					error: `${id} is a ${resolved.kind} id; use ${kind === "grasp" ? "execute_place" : "execute_grasp"}`,
				};
			const pre = await call<{ eef_position: number[]; eef_quat_xyzw: number[] }>(env, "env.resolve_grasp", {
				grasp_id: id,
				standoff: kwargs.standoff,
			});
			const far = xyRefusal(eef(), pre.eef_position, `${name}'s first leg`);
			if (far) return { name, id, refused: far, steps_used: 0 };
			const planned = Boolean(flag("ik", ""));
			if (planned) {
				const grasp = await call<{ eef_position: number[] }>(env, "env.resolve_grasp", {
					grasp_id: id,
					standoff: 0,
				});
				const unreachable = reachRefusal(await call<Reach>(env, "env.preview_reach", { xyz: grasp.eef_position }));
				if (unreachable) return { name, id, refused: unreachable, steps_used: 0 };
				const blocked = planRefusal(
					await call<MotionPlan>(env, "env.plan_motion", { pos: pre.eef_position, quat_xyzw: pre.eef_quat_xyzw }),
				);
				if (blocked) return { name, id, refused: blocked, steps_used: 0 };
			}
			const c = await call<Claim>(env, "env.claim_waypoints", {
				grasp_id: id,
				standoff: kwargs.standoff,
				lift: kwargs.lift,
			});
			const run = await runClaim(c, {
				servo: (target, q, g) =>
					planned
						? plannedServo(target, q, g, kwargs.max_steps)
						: servoOrientation(target, q, g, kwargs.max_steps),
				actuate,
				width: gripper,
				ended: () => terminated || truncated,
				released: () => call(env, "env.release_held", {}),
				closed: () => call(env, "env.note_grasp_closed", {}),
			});
			return { name, id, ...run, final_eef_pos: eef().map((v) => round(v)), gripper_width: round(gripper()) };
		} catch (err) {
			if (isStale(err)) pi.appendEntry(DETECTIONS_EXPIRED_ENTRY, { tool: name, ids: [id], error: message(err) });
			return { name, error: message(err) };
		}
	}

	const stepParam = Type.Optional(
		Type.Integer({ description: "State record (a result's state_step): 0 = initial, -1 = latest (default)" }),
	);
	robot.tool(
		"view_env_state",
		"A state with agentview (global layout) and wrist (close range) 1024x1024 images: the current one, or the earlier state record `step` (0 = initial). Pixel (row, col) in these images feed back_project with the same step.",
		Type.Object({ step: stepParam }),
		async ({ step: at }) => {
			if (terminated || truncated)
				return {
					content: [
						{ type: "text", text: `Episode already ended (terminated=${terminated}, truncated=${truncated}).` },
					],
					details: { terminated, truncated },
				};
			if (!records.length) await snapshot({});
			const i = past(at);
			if (i === undefined) return observe({});
			const dir = records[i].dir;
			const body = JSON.parse(readFileSync(join(dir, "state.json"), "utf8")) as Record<string, unknown>;
			const pngs = CAMERA_NAMES.map((c) => readFileSync(join(dir, `${c}_high.png`)));
			const latest = { latest_state_step: records.length - 1, latest_step: envStep };
			return shown({ ...body, ...latest }, pngs, { viewed_state_step: i, ...latest, terminated, truncated });
		},
	);

	// The motion tools run on the env server (its env.* methods, the ones a program's primitives call),
	// with the manifest's parameters and defaults (manifests/libero.json).
	tool("move_to", "", Type.Object({}), async (params: Json) => {
		const r = await serverMotion("env.move_to", params);
		return {
			name: "move_to",
			...final(r),
			...(r.refused
				? { steps_used: 0 }
				: {
						final_dist_m: r.final_dist_m,
						steps_used: r.steps_used,
						...(r.planned ? { planned_segments: r.planned.segments } : {}),
						...(r.stopped ? { stopped: `collision check: ${r.contact}` } : {}),
					}),
		};
	});

	tool(
		"preview_reach",
		"",
		Type.Object({}),
		async (params: Json) => call<Reach>(env, "env.preview_reach", params),
		false,
	);

	/** A closed-loop grasp with `client`'s VLA: chunks until the pick heuristics (../vla-adapters.ts) say lifted, the episode ends, or the budget runs out. */
	async function pick(client: RpcClient, toolName: string, p: PickParams) {
		const track = pickTracker(eef()[2], gripper(), p);
		const max_chunks = p.max_chunks ?? 24;
		let success = false;
		let chunks = 0;
		const vla_seeds: (number | null)[] = [];
		while (chunks < max_chunks) {
			vla_seeds.push((await vlaChunk(p.prompt, client, toolName)) ?? null);
			chunks++;
			if (track.update(eef()[2], gripper())) {
				success = true;
				break;
			}
			if (terminated || truncated) {
				success = terminated;
				break;
			}
		}
		const motion = track.summary();
		return {
			name: "pick",
			instruction: p.prompt,
			success,
			chunks_used: chunks,
			peak_lift_m: round(motion.peak_lift_m),
			descent_m: round(motion.descent_m),
			min_gripper_opening: round(motion.min_gripper_opening),
			final_gripper_opening: round(gripper()),
			vla_seeds,
		};
	}

	tool(
		"pi0_pick",
		"Pi0.5 closed-loop grasp. Use it only for the grasp; you do every move_to and release. Success needs a descent then a lift with the gripper partly closed; it is a hint, confirm from gripper opening and the wrist image.",
		PICK_PARAMETERS,
		(p) => pick(vla, "pi0", p),
	);

	// The same grasp tool per third-party VLA (`--openvla <url>` etc.): registered always, because pi sets
	// the command-line flag values only after the extensions have loaded (a getFlag here reads the
	// default), and made active at start only when its flag names a server; the client is made there too.
	for (const a of VLA_ADAPTERS)
		tool(a.tool, pickDescription(a.model), PICK_PARAMETERS, (p) => pick(adapters.get(a.tool)!, a.tool, p));

	tool("pi0_doubled", "", Type.Object({}), async ({ prompt, max_chunks = 20 }: Json) => {
		let chunks = 0;
		const vla_seeds: (number | null)[] = [];
		while (chunks < max_chunks && !terminated && !truncated) {
			vla_seeds.push((await vlaChunk(prompt)) ?? null);
			chunks++;
		}
		return { name: "pi0_doubled", instruction: prompt, success: terminated, chunks_used: chunks, vla_seeds };
	});

	tool("release", "", Type.Object({}), async (params: Json) => {
		const r = await serverMotion("env.release", params);
		return {
			name: "release",
			steps_used: r.steps_used,
			start_gripper_opening: r.start_gripper_opening,
			final_gripper_opening: r.final_gripper_opening,
		};
	});

	tool("set_gripper", "", Type.Object({}), async (params: Json) => {
		const r = await serverMotion("env.set_gripper", params);
		return { name: "set_gripper", gripper: r.gripper, steps: r.steps_used };
	});

	for (const kind of ["wrist", "pitch"] as const) {
		const name = `rotate_${kind}`;
		const angle = kind === "wrist" ? "yaw" : "pitch";
		tool(name, "", Type.Object({}), async (params: Json) => {
			const r = await serverMotion(`env.${name}`, params);
			return {
				name,
				[`start_${angle}`]: r[`start_${angle}`],
				[`target_${angle}`]: r[`target_${angle}`],
				[`final_${angle}`]: r[`final_${angle}`],
				final_err: r.final_err,
				steps_used: r.steps_used,
			};
		});
	}

	tool("move_pose", "", Type.Object({}), async (params: Json) => {
		const r = await serverMotion("env.move_pose", params);
		return {
			name: "move_pose",
			...final(r),
			final_dist_m: r.final_dist_m,
			final_pitch: r.final_pitch,
			steps_used: r.steps_used,
		};
	});

	// CaP-X's high tier (manifest tier high, active with --api high): the semantic functions run on the
	// server; --privileged answers get_object_pose and sample_grasp_pose from the simulator.
	const privileged = () => pi.getFlag("privileged") === true;
	tool(
		"get_object_pose",
		"",
		Type.Object({}),
		async (params: Json) => ({
			pose: await call(env, privileged() ? "env.get_object_pose_privileged" : "env.get_object_pose", params),
		}),
		false,
	);
	tool(
		"sample_grasp_pose",
		"",
		Type.Object({}),
		async (params: Json) => ({
			grasp: await call(
				env,
				privileged() ? "env.sample_grasp_pose_privileged" : "env.sample_grasp_pose",
				params,
				600_000,
			),
		}),
		false,
	);
	for (const name of ["goto_pose", "home_pose", "open_gripper", "close_gripper"] as const)
		tool(name, "", Type.Object({}), async (params: Json) => ({
			...(await serverMotion(`env.${name}`, params)),
			name,
		}));

	tool(
		"view_camera_meta",
		"",
		Type.Object({}),
		async ({ camera: c = "agentview", resolution = "high", step: at }: Json) => {
			const size = resolution === "high" ? 1024 : 256;
			const i = past(at);
			if (i !== undefined) {
				if (size !== 1024) return { error: `step ${at} keeps only the high-resolution calibration` };
				const meta = JSON.parse(readFileSync(join(records[i].dir, `${c}_meta.json`), "utf8"));
				return { camera: c, resolution, step: i, meta };
			}
			return {
				camera: c,
				resolution,
				meta: await call(env, "env.get_camera_meta", {
					camera_name: CAMERAS[c as Camera],
					height: size,
					width: size,
				}),
			};
		},
		false,
	);

	// segment and back_project read pi's state history (the 1024 images of earlier state records), so
	// the tools run here; a program's primitives of the same names are the server's (512 images).
	tool(
		"segment",
		"",
		Type.Object({}),
		async ({ prompt, point, camera: c = "agentview", min_score = 0.2, step: at }: Json) => {
			// Models often fill both optional fields; a non-empty prompt wins.
			const text = prompt?.trim();
			if (!text && !point) return { error: "give a text prompt or a point [row, col]" };
			if (records.at(-1)?.envStep !== envStep) await snapshot({});
			const index = past(at) ?? records.length - 1;
			const map = await worldMap(c, 1024, at);
			const query = text ? { mode: "text", prompt: text } : { mode: "point", point };
			const png = encodePng(map.rgb, 1024, 1024);
			const res = await call<{
				found: boolean;
				score?: number;
				box?: number[];
				mask_png_base64?: string;
				reason?: string;
			}>(sam3, "sam3.segment", {
				image_base64: png.toString("base64"),
				...(text ? { text_prompt: text } : { point }),
				min_score,
			});
			if (!res.found || !res.mask_png_base64) {
				const error = res.reason ?? "no mask";
				const saved = saveSegment(index, { ...query, camera: c, found: false, error }, undefined);
				return {
					found: false,
					step: index,
					error,
					fallback: "Pick pixels in the image and use back_project.",
					...saved,
				};
			}
			const mask = decodePngChannel(Buffer.from(res.mask_png_base64, "base64"));
			if (mask.width !== 1024 || mask.height !== 1024)
				return { found: true, error: `mask ${mask.width}x${mask.height} does not match the 1024 world map` };
			const xs: number[] = [];
			const ys: number[] = [];
			const pts: number[][] = [];
			const overlay = Buffer.from(map.rgb);
			for (let i = 0; i < mask.data.length; i++) {
				if (mask.data[i] < 128) continue;
				ys.push(Math.floor(i / 1024));
				xs.push(i % 1024);
				overlay[i * 3] = Math.round(0.55 * overlay[i * 3] + 0.45 * 255);
				overlay[i * 3 + 1] = Math.round(0.55 * overlay[i * 3 + 1]);
				overlay[i * 3 + 2] = Math.round(0.55 * overlay[i * 3 + 2]);
				const p = [map.xyz[i * 3], map.xyz[i * 3 + 1], map.xyz[i * 3 + 2]];
				if (p.every(Number.isFinite) && Math.abs(p[0]) + Math.abs(p[1]) + Math.abs(p[2]) > 1e-6) pts.push(p);
			}
			const out: Record<string, unknown> = {
				found: true,
				step: index,
				camera: c,
				score: res.score === undefined ? null : round(res.score, 3),
				box: res.box,
				n_pixels: xs.length,
				n_valid: pts.length,
				centroid_pixel: [Math.round(median(xs)), Math.round(median(ys))],
			};
			out.world_xyz = pts.length < 10 ? null : [0, 1, 2].map((k) => round(median(pts.map((p) => p[k]))));
			if (pts.length < 10) out.world_error = `too few valid depth pixels (${pts.length})`;
			const image = encodePng(overlay, 1024, 1024);
			const { step: _, ...reading } = out;
			Object.assign(out, saveSegment(index, { ...query, ...reading }, image));
			out._image = image;
			return out;
		},
		false,
	);

	tool(
		"back_project",
		"",
		Type.Object({}),
		async ({
			row,
			col,
			camera: c = "agentview",
			resolution = "high",
			row_range,
			col_range,
			z_min,
			z_max,
			step: s,
		}: Json) => {
			const size = resolution === "high" ? 1024 : 256;
			const map = await worldMap(c, size, s);
			const shownStep = past(s) ?? (records.at(-1)?.envStep === envStep ? records.length - 1 : undefined);
			const stepOf = shownStep === undefined ? {} : { step: shownStep };
			const at = (r: number, cc: number) => {
				const i = (r * size + cc) * 3;
				return [map.xyz[i], map.xyz[i + 1], map.xyz[i + 2]];
			};
			const valid = (p: number[]) =>
				p.every(Number.isFinite) && Math.abs(p[0]) + Math.abs(p[1]) + Math.abs(p[2]) > 1e-6;
			// An empty window ([0, 0]) is a placeholder, not a region.
			const span = (r?: number[]) => (r && Math.max(...r) > Math.min(...r) ? r : undefined);
			const rows = span(row_range);
			const cols = span(col_range);
			if (rows || cols) {
				if (!rows || !cols) return { error: "region mode needs both row_range and col_range" };
				const [r0, r1] = [clip(Math.min(...rows), 0, size), clip(Math.max(...rows), 0, size)];
				const [c0, c1] = [clip(Math.min(...cols), 0, size), clip(Math.max(...cols), 0, size)];
				let pts: number[][] = [];
				for (let r = r0; r < r1; r++) for (let cc = c0; cc < c1; cc++) if (valid(at(r, cc))) pts.push(at(r, cc));
				if (z_min !== undefined) pts = pts.filter((p) => p[2] >= z_min);
				if (z_max !== undefined) pts = pts.filter((p) => p[2] <= z_max);
				if (pts.length < 8)
					return { error: `too few valid pixels in region (${pts.length}); widen the window or the z band` };
				const axis = (k: number) => pts.map((p) => p[k]);
				return {
					camera: c,
					resolution,
					...stepOf,
					mode: "region",
					center_xyz: [
						round((Math.min(...axis(0)) + Math.max(...axis(0))) / 2),
						round((Math.min(...axis(1)) + Math.max(...axis(1))) / 2),
						round(median(axis(2))),
					],
					median_xyz: [0, 1, 2].map((k) => round(median(axis(k)))),
					n_valid: pts.length,
				};
			}
			if (row === undefined || col === undefined) return { error: "give row and col, or row_range and col_range" };
			if (row < 0 || row >= size || col < 0 || col >= size)
				return { error: `pixel (${row},${col}) out of bounds for ${size}x${size}` };
			const p = at(row, col);
			if (!valid(p)) return { error: `invalid world xyz at (${row},${col}); pick another pixel` };
			return { camera: c, resolution, ...stepOf, pixel: [row, col], world_xyz: p.map((v) => round(v)) };
		},
		false,
	);

	// Molmo pointing on the latest 256 px images (active with --point).
	mountGraspTool(
		robot.tool,
		pointTool(pi, {
			name: "molmo_point",
			cameras: ["agentview", "wrist"],
			frame: async (c) => {
				const a = c === "wrist" ? obs.wrist_images : obs.main_images;
				if (!a) throw new Error(`no ${c} image in the latest observation`);
				return rgbOf(a);
			},
			signal: () => robot.signal,
		}),
	);

	// plan_grasp / plan_place / check_attached over the env server's grasp primitives (active with a backend flag).
	for (const d of graspTools(pi, {
		call: (method, kwargs, timeoutMs) => call(env, method, kwargs, timeoutMs),
		cameras: ["agentview", "wrist"],
		task: () => language,
		executes: true,
	}))
		mountGraspTool(robot.tool, d);

	// SAM3 masks with ids and UniDepth over the env server's perception (active with --detections / --unidepth).
	for (const d of detectionTools(pi, {
		call: (method, kwargs, timeoutMs) => call(env, method, kwargs, timeoutMs),
		cameras: ["agentview", "wrist"],
	}))
		mountGraspTool(robot.tool, d);

	// --geometry: view_points, mark_point, move_grip (../primitives/geometry.ts). The env server resolves
	// move_grip's target; its motion steps here, like every tool's, so the video, the recorder and the
	// success latch see it.
	const geometry = geometryTools(
		pi,
		{
			call: (method, kwargs, timeoutMs) => call(env, method, kwargs, timeoutMs),
			cameras: ["agentview", "wrist"],
			refuse: () =>
				terminated || truncated
					? `Episode already ended (terminated=${terminated}, truncated=${truncated}).`
					: undefined,
			execute: async (plan) => {
				// The same xy bound as move_to: a longer grip move is refused unmoved.
				const far = plan.motion
					? xyRefusal(plan.current.grip_xyz_m, plan.target.grip_xyz_m, "move_grip")
					: undefined;
				if (far) return observe({ name: "move_grip", refused: far, steps_used: 0 });
				const { result, pngs } = await runGripPlan(
					plan,
					{
						pose: async () => ({ pos: eef(), quat: await quat() }),
						step,
						// Hold the fingers as they are: closed (on something or nothing) unless fully open.
						hold: () => (gripper() < 0.07 ? 1 : -1),
						actuate,
						ended: () => terminated || truncated,
					},
					(method, kwargs) => call(env, method, kwargs),
				);
				const shown = await observe(result);
				const extra = pngs.map((png) => ({
					type: "image" as const,
					data: png.toString("base64"),
					mimeType: "image/png",
				}));
				return { ...shown, content: [...shown.content, ...extra] };
			},
		},
		(d) => robot.tool(d.name, d.description, d.parameters, d.run),
	);

	// A planned grasp or place runs as one tool from one resolution of its id (executePlanned).
	tool("execute_grasp", "", Type.Object({}), async ({ grasp_id, standoff = 0.1, lift = 0.1, max_steps = 150 }: Json) =>
		executePlanned("execute_grasp", grasp_id, "grasp", { standoff, lift, max_steps }),
	);

	tool("execute_place", "", Type.Object({}), async ({ place_id, standoff = 0.1, max_steps = 150 }: Json) =>
		executePlanned("execute_place", place_id, "placement", { standoff, lift: 0, max_steps }),
	);

	// OpenETA extras, each registered only with its flag: follow_waypoints (--waypoints), align_wrist
	// (--align-wrist), suggest_grasp (--grasp-advisor) over ../primitives/{waypoints,wrist,advisor}.ts.
	const cameraMeta = (c: Camera) =>
		call<CameraMeta>(env, "env.get_camera_meta", { camera_name: CAMERAS[c], height: 1024, width: 1024 });
	const extras = [
		waypointsTool(
			pi,
			{
				current: eef,
				maxSegment: () => 0.3,
				maxPath: () => 0.6,
				segment: async (_from, to, g) => {
					const dist = () => Math.hypot(...to.map((v, i) => v - eef()[i]));
					let steps = 0;
					for (; steps < 80 && !terminated && !truncated && dist() >= 0.012; steps++)
						await step([...to.map((v, i) => clip(clip(v - eef()[i], -0.025, 0.025) / 0.05, -1, 1)), 0, 0, 0, g]);
					return {
						reached: dist() < 0.012,
						ended: terminated || truncated,
						final_dist_m: round(dist()),
						steps_used: steps,
					};
				},
			},
			(d) => tool(d.name, d.description, d.parameters, (p) => d.run(p, robot.signal)),
		),
		alignWristTool(
			pi,
			{
				moveWith: "move_to xyz",
				gripper: eef,
				view: async (row, col) => {
					const map = await worldMap("wrist", 1024);
					const i = (clip(row, 0, 1023) * 1024 + clip(col, 0, 1023)) * 3;
					const target = [map.xyz[i], map.xyz[i + 1], map.xyz[i + 2]];
					if (!target.every(Number.isFinite) || !target.some((v) => Math.abs(v) > 1e-6))
						throw new Error(`no valid depth at wrist pixel (${row},${col}); pick another pixel`);
					const meta = await cameraMeta("wrist");
					const image = { width: 1024, height: 1024, rgb: map.rgb };
					return { K: meta.intrinsic_K, cam2world: meta.extrinsic_cam2world, target, image };
				},
			},
			(d) =>
				tool(
					d.name,
					d.description,
					d.parameters,
					async (p) => {
						const { _pngs, ...rest } = await d.run(p, robot.signal);
						return { ...rest, ...(_pngs?.[0] ? { _image: _pngs[0] } : {}) };
					},
					false,
				),
		),
	];
	const advisor = graspAdvisorTool(
		pi,
		{
			image: async (c) => ({ width: 1024, height: 1024, rgb: (await render(c as Camera, 1024, false)).rgb }),
			project: async (c, points) => {
				const meta = await cameraMeta(c as Camera);
				return projectPoints(meta.intrinsic_K, meta.extrinsic_cam2world, points);
			},
			stamp: () => envStep,
			task: () => language,
		},
		(d) => mountGraspTool(robot.tool, d),
	);

	/**
	 * One action unit (../units): drive the gripper, servo the EEF to its current position plus
	 * `delta` (holding the gripper command), turn the wrist by `yaw` or an RT_* `rot`, or hold one step (STOP).
	 */
	async function unitStep(move: Move) {
		// After success only the finish sequence moves (opening, lifting straight up); success stays latched.
		const finishing = finishMove(move);
		if (truncated || (terminated && !finishing))
			return {
				content: [{ type: "text" as const, text: `Episode already ended (terminated=${terminated}).` }],
				details: { terminated, truncated },
			};
		let steps = 0;
		const live = () => !truncated && (!terminated || finishing);
		if (move.gripper) {
			grip = move.gripper === "close" ? 1 : -1;
			// Until the fingers stop moving (they stop on a grasped object), at most 15 steps.
			for (let prev = Number.NaN; steps < 15 && live(); ) {
				await step([0, 0, 0, 0, 0, 0, grip]);
				steps++;
				if (steps > 3 && Math.abs(gripper() - prev) < 5e-4) break;
				prev = gripper();
			}
		}
		if (Math.hypot(...move.delta) > 0) {
			const target = eef().map((v, i) => v + move.delta[i]);
			const tol = Number(flag("unit-tol", "0.004")) || 0.004;
			for (let k = 0; k < 25 && live(); k++) {
				const diff = target.map((v, i) => v - eef()[i]);
				if (Math.hypot(...diff) < tol) break;
				await step([...diff.map((d) => clip(clip(d, -0.025, 0.025) / 0.05, -1, 1)), 0, 0, 0, grip]);
				steps++;
			}
		}
		if (move.yaw)
			steps += Number(
				(
					await serverMotion("env.rotate_wrist", {
						delta_yaw: move.yaw,
						gripper: grip,
						max_steps: 25,
						tol: 0.02,
						step_clip: 0.1,
					})
				).steps_used,
			);
		if (move.rot && Math.hypot(...move.rot) > 0) {
			// An RT_* turn: servo to rot * R0 (a world-frame rotation, the OSC delta convention) while
			// holding the TCP position.
			const hold = eef();
			const goal = matmul(rotvecToMatrix(move.rot), rotation(await quat()));
			for (let k = 0; k < 25 && live(); k++) {
				const err = matrixToRotvec(matmul(goal, transpose(rotation(await quat()))));
				const diff = hold.map((v, i) => v - eef()[i]);
				const size = Math.hypot(...err);
				if (size < 0.02 && Math.hypot(...diff) < 0.004) break;
				const scale = size > 0.1 ? 0.1 / size : 1;
				await step([
					...diff.map((d) => clip(clip(d, -0.025, 0.025) / 0.05, -1, 1)),
					...err.map((e) => clip((e * scale) / 0.1, -1, 1)),
					grip,
				]);
				steps++;
			}
		}
		if (!move.gripper && !Math.hypot(...move.delta) && !move.yaw && !move.rot?.some(Boolean)) {
			await step([0, 0, 0, 0, 0, 0, grip]);
			steps++;
		}
		return observe({
			name: "act",
			steps_used: steps,
			eef_pos: eef().map((v) => round(v)),
			gripper: round(gripper()),
		});
	}

	/** Median world z of the agentview's bottom-center strip: the table or floor surface nearest the camera. */
	async function surfaceZ() {
		const map = await worldMap("agentview", 256);
		const z: number[] = [];
		for (let r = 218; r < 256; r++)
			for (let c = 77; c < 179; c++) {
				const v = map.xyz[(r * 256 + c) * 3 + 2];
				if (Number.isFinite(v)) z.push(v);
			}
		return z.length ? round(median(z)) : undefined;
	}

	/** Affordance points (../units `point`): the marked 1024 image and world xyz (median of a 7x7 window). */
	async function locate(camera: Camera, points: [number, number][]) {
		const map = await worldMap(camera, 1024);
		let rgb: Buffer = map.rgb;
		const xyz = points.map(([fy, fx]) => {
			const row = clip(Math.round(fy * 1023), 0, 1023);
			const col = clip(Math.round(fx * 1023), 0, 1023);
			rgb = mark({ width: 1024, height: 1024, rgb }, row, col, [255, 32, 32]);
			const pts: number[][] = [];
			for (let r = Math.max(0, row - 3); r <= Math.min(1023, row + 3); r++)
				for (let c = Math.max(0, col - 3); c <= Math.min(1023, col + 3); c++) {
					const i = (r * 1024 + c) * 3;
					const p = [map.xyz[i], map.xyz[i + 1], map.xyz[i + 2]];
					if (p.every(Number.isFinite) && Math.abs(p[0]) + Math.abs(p[1]) + Math.abs(p[2]) > 1e-6) pts.push(p);
				}
			return pts.length < 5 ? null : [0, 1, 2].map((k) => round(median(pts.map((p) => p[k]))));
		});
		return { image: encodePng(rgb, 1024, 1024), xyz };
	}

	/** Start (or attach to) the env server and restore the initial scene; returns the tools to activate. */
	async function startEpisode() {
		const { suite, task, seed } = robot.task;
		if (!["clean", "keep", "off"].includes(history()))
			throw new Error(`--step-history must be clean, keep or off, not ${history()}`);
		if (!(LIBERO_PROMPTS as readonly string[]).includes(variant()))
			throw new Error(`--libero-prompt must be one of ${LIBERO_PROMPTS.join(", ")}, not ${variant()}`);
		vla = new RpcClient(flag("vla", ""));
		for (const a of VLA_ADAPTERS) if (flag(a.flag, "")) adapters.set(a.tool, new RpcClient(flag(a.flag, "")));
		for (const k of Object.keys(vlaUsed)) delete vlaUsed[k];
		sam3 = new RpcClient(flag("sam3", ""));
		const endpoint = pi.getFlag("env") as string | undefined;
		if (endpoint) {
			// `URL#token=HEX` for a server that requires its RPC token, as every robot attaches.
			env = await attach(endpoint);
		} else {
			const services = flag("services", SERVICES);
			const cuda = pi.getFlag("cuda-device") as string | undefined;
			env = await robot.serve({
				python: flag("python", "python"),
				args: [
					...["-m", "pi_embodied_services.robots.libero.env_server"],
					...["--suite", suite, "--task", task, "--seed", seed],
					// Code mode's `segment` primitive asks the same SAM3 server as the `segment` tool.
					...(flag("sam3", "") ? ["--sam3", flag("sam3", "")] : []),
					...ikArgs(flag("ik", "")),
					...(cuda ? ["--cuda-device", cuda] : []),
					...graspArgs(pi),
					// --sam3 goes to the server already (above).
					...detectionArgs(pi, ""),
					...geometryArgs(pi),
				],
				cwd: services,
				env: {
					...process.env,
					PYTHONPATH: services,
					LIBERO_TYPE: flag("libero-type", "pro"),
					MUJOCO_GL: "egl",
					ROBOT_PLATFORM: "LIBERO",
				},
				log: (port) => join(tmpdir(), `pi-embodied-env-${suite}-t${task}-s${seed}-${port}.log`),
			});
		}
		await resetEpisode();
		language = await call<string>(env, "env.get_task_language");
		fly.reset(flyObs(obs), flyMeta());
		const tools = flag("ik", "") ? TOOLS : TOOLS.filter((name) => name !== "preview_reach");
		const grasp = graspActive(pi);
		const extra = [...extras.flatMap((on) => on()), ...advisor(grasp.length > 0)];
		const perception = (await call<{ capabilities?: { perception?: PerceptionCaps } }>(env, "env.get_env_meta"))
			.capabilities?.perception;
		return [
			...tools,
			...grasp,
			...(grasp.length ? ["execute_grasp", "execute_place"] : []),
			...detectionActive(pi, perception),
			...pointActive(pi, "molmo_point"),
			...geometry(),
			...adapters.keys(),
			...extra,
		];
	}
}
