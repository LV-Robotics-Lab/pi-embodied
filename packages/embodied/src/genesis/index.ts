/**
 * Genesis robot for pi: a Franka Panda in the Genesis simulator on a translation-only IK controller.
 *
 *   pi -e packages/embodied/src/genesis --seed 0                      (cube_pick, the only task so far)
 *   pi -e packages/embodied/src/genesis --units --task cube_pick --seed 3
 *   pi -e packages/embodied/src/genesis --privileged --seed 0          (adds ground_truth_poses)
 *
 * Starts one Genesis env server per session (services/.../robots/genesis/env_server.py, the `genesis`
 * venv; rendering needs a GPU). The server owns the motion: `move_delta` and the units hook `apply`
 * run a base-frame delta as ~2 cm IK decisions with the reset orientation held, inside a workspace
 * box, above a Z floor and within a per-call cap, all checked before anything moves; `gripper` opens
 * or closes and holds. Every result carries the front and wrist images and the state; success is the
 * task's own predicate (cube_pick: the cube lifted 8 cm off the table), recorded in `robot_result`.
 * `segment` (SAM3, `--sam3`) and `back_project` give world coordinates from the current image
 * through the server's depth.
 *
 * OpenETA (github.com/OpenETA at 7d4a0a1) sim/envs/genesis: its Franka scene and cube_pick task,
 * ported as a pi robot with the pi-embodied motion limits and cameras.
 */

import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { decodePngChannel, encodePng } from "../png.ts";
import { attach, defineRobot, median, SERVICES } from "../robot.ts";
import { type NdArray, RpcClient } from "../rpc.ts";
import type { MoveUnit, Vec3 } from "../units/index.ts";

const SYSTEM = readFileSync(new URL("./SYSTEM.md", import.meta.url), "utf8");

/** The env server's tasks (its TASKS table); `--task` takes one of them. */
export const TASKS = ["cube_pick"] as const;
export type Task = (typeof TASKS)[number];
export const CAMERAS = ["agentview", "wrist"] as const;
/** Base-frame unit vectors: +x away from the base, -y = MV_LEFT (the robot's right), +z up. */
export const VECTORS: Record<MoveUnit, Vec3> = {
	MV_FWD: [1, 0, 0],
	MV_BACK: [-1, 0, 0],
	MV_LEFT: [0, -1, 0],
	MV_RIGHT: [0, 1, 0],
	MV_UP: [0, 0, 1],
	MV_DOWN: [0, 0, -1],
};
/** Physical metres per decision (the 2 cm convention; the server splits a move into them). */
export const STEP_M = 0.02;
/** Largest translation one call may command, m (the server refuses more; env_server MAX_MOVE_M). */
export const MAX_MOVE_M = 0.2;
/** A closed gripper at or below this width holds nothing, m (env_server EMPTY_WIDTH_M). */
export const EMPTY_WIDTH_M = 0.005;
/** Mask pixels back-projected per segment call (a uniform subsample of the mask). */
export const SEGMENT_SAMPLES = 400;

/**
 * How the views look (env_server AGENTVIEW / WRIST_OFFSET): the front camera stands in front of the
 * table facing the robot, so the base is at the top of the image and image right is +y; the wrist
 * camera looks along the gripper's approach direction with the fingertips at the top edge.
 */
export const VIEWS = `Each result shows the front view, then the wrist view (both 256x256). In BOTH views MV_LEFT / MV_RIGHT move the gripper toward the image left / right, MV_FWD toward the image bottom, MV_BACK toward the image top.
- Front view: a fixed camera in front of the table facing the robot, slightly from the robot's left; the robot base is at the top and the table edge nearest the camera at the bottom; MV_FWD brings the gripper toward the camera (down in the image, and larger).
- Wrist view: it looks straight down past the gripper; the two fingertips stay fixed at the top edge, and the point under the gripper is horizontally centred, about a third of the way down. A target right of that point needs MV_RIGHT, left of it MV_LEFT, below it MV_FWD, above it MV_BACK; a target on it is under the gripper: MV_DOWN. The camera moves with the gripper, so after MV_RIGHT the scene shifts left.`;

type Obs = {
	agentview: NdArray;
	wrist: NdArray;
	tcp_pos: NdArray;
	tcp_quat_wxyz: NdArray;
	gripper_width: number;
	gripper_command: "open" | "close";
	qpos: NdArray;
	success: boolean;
	is_grasped: boolean;
	lift_m: number;
	env_steps: number;
};
type Moved = Obs & {
	commanded_m: number[];
	moved_m: number[];
	decisions: number;
	control_steps: number;
	frames?: NdArray[];
	cancelled?: boolean;
};
type Gripped = Obs & { control_steps: number; frames?: NdArray[]; grasp_empty?: boolean; cancelled?: boolean };
type Meta = {
	task: string;
	seed: number;
	instruction: string;
	workspace: { min: number[]; max: number[] };
	z_floor_m: number;
	max_move_m: number;
	lift_m: number;
};
type CameraMeta = { intrinsic_K: number[][]; extrinsic_cam2world: number[][]; width: number; height: number };

const round = (v: number, d = 4) => Number(v.toFixed(d));

/**
 * The (row, col) pixels of a SAM3 mask (a decoded PNG channel, >= 128 = in), at most `limit` of
 * them spread evenly over the mask, and the mask's pixel count and median pixel.
 */
export function maskPixels(mask: { width: number; height: number; data: Uint8Array }, limit = SEGMENT_SAMPLES) {
	const rows: number[] = [];
	const cols: number[] = [];
	for (let i = 0; i < mask.data.length; i++) {
		if (mask.data[i] < 128) continue;
		rows.push(Math.floor(i / mask.width));
		cols.push(i % mask.width);
	}
	const stride = Math.max(1, Math.ceil(rows.length / limit));
	const pixels: [number, number][] = [];
	for (let i = 0; i < rows.length; i += stride) pixels.push([rows[i], cols[i]]);
	return {
		n: rows.length,
		centroid: rows.length ? [Math.round(median(rows)), Math.round(median(cols))] : null,
		pixels,
	};
}

/** The per-axis median of the valid back-projected points, or null with fewer than `min` of them. */
export function medianPoint(points: (number[] | null)[], min = 10): number[] | null {
	const pts = points.filter((p): p is number[] => Array.isArray(p) && p.every(Number.isFinite));
	if (pts.length < min) return null;
	return [0, 1, 2].map((k) => round(median(pts.map((p) => p[k]))));
}

export default function genesis(pi: ExtensionAPI) {
	const flag = (name: string, fallback: string) => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("task", { type: "string", default: "cube_pick", description: `Genesis task: ${TASKS.join(", ")}` });
	pi.registerFlag("seed", { type: "string", default: "0", description: "Reset seed (the object layout)" });
	pi.registerFlag("backend", {
		type: "string",
		default: "gpu",
		description: "Genesis compute backend for the env server: gpu (default), cuda or cpu",
	});
	pi.registerFlag("env", { type: "string", description: "Attach to a running env server instead of starting one" });
	pi.registerFlag("sam3", { type: "string", default: "http://127.0.0.1:18300", description: "SAM3 server (segment)" });
	pi.registerFlag("services", {
		type: "string",
		default: process.env.PI_EMBODIED_SERVICES ?? SERVICES,
		description: "pi-embodied services dir",
	});
	pi.registerFlag("python", {
		type: "string",
		default: process.env.PI_EMBODIED_PYTHON ?? "python",
		description: "Python for the env server (the genesis venv)",
	});

	let env: RpcClient;
	let sam3: RpcClient;
	let obs: Obs;
	let meta: Meta;

	const robot = defineRobot(pi, {
		name: "genesis",
		task: ["task", "seed"],
		keepImages: 4,
		video: true,
		groundTruth: (names) => call("env.ground_truth_poses", { names: names ?? null }),
		start: startEpisode,
		prompt: () => SYSTEM.replaceAll("{{task_language}}", meta.instruction),
		result: () => ({
			task: robot.task.task,
			seed: Number(robot.task.seed),
			success: obs?.success ?? false,
			ever_grasped: everGrasped,
			env_steps: obs?.env_steps ?? 0,
		}),
		status: () => ({ language: meta.instruction, step: obs.env_steps, solved: obs.success }),
		finish: {
			description:
				"End the episode after checking the latest state. Success is the task's own predicate, not this call.",
			parameters: Type.Object({
				status: StringEnum(["success", "failure"] as const),
				summary: Type.String(),
			}),
			result: (params) => ({
				content: [{ type: "text", text: `Episode finished (success=${obs.success}).` }],
				details: params,
			}),
		},
		units: {
			vectors: VECTORS,
			stepM: STEP_M,
			instruction: () => meta.instruction,
			views: VIEWS,
			emptyWidthM: EMPTY_WIDTH_M,
			maxMoveM: () => MAX_MOVE_M,
			apply: async (m, signal) => {
				if (m.yaw || m.rot) throw new Error("this robot has no rotation (the orientation is held)");
				return observe(await move(m.delta, m.gripper, signal));
			},
			state: async () => ({
				eef_xyz: obs.tcp_pos.toArray().map((v) => round(v)),
				gripper_width: obs.gripper_width,
				table_z: 0,
				is_grasped: obs.is_grasped,
			}),
		},
	});
	const { video } = robot;
	let everGrasped = false;

	const call = <T = unknown>(
		method: string,
		kwargs: Record<string, unknown> = {},
		args: unknown[] = [],
		signal = robot.signal,
	) => env.call<T>(method, kwargs, 300_000, args, signal);

	function absorb(o: Obs) {
		obs = o;
		everGrasped ||= o.is_grasped;
	}

	/** One base-frame move (m) with an optional gripper command first; the server checks the limits. */
	async function move(delta: Vec3, gripper: "open" | "close" | null, signal: AbortSignal | undefined) {
		const r = await call<Moved>("env.move_delta", { gripper, return_frames: true }, [delta], signal);
		for (const f of r.frames ?? []) video.frame(f);
		const { frames: _frames, commanded_m, moved_m, decisions, control_steps, cancelled, ...o } = r;
		absorb(o);
		return { commanded_m, moved_m, decisions, control_steps, ...(cancelled ? { cancelled } : {}) };
	}

	/** The result with the new state, then the front and wrist images. */
	function observe(result: Record<string, unknown>) {
		const images = [obs.agentview, obs.wrist].map((a) => encodePng(a.data, a.shape[1], a.shape[0]));
		const details = {
			result,
			step: obs.env_steps,
			success: obs.success,
			terminated: obs.success,
			task_language: meta.instruction,
			state: {
				tcp_pos: obs.tcp_pos.toArray().map((v) => round(v)),
				gripper_width: round(obs.gripper_width),
				gripper_command: obs.gripper_command,
				is_grasped: obs.is_grasped,
			},
			images: [
				`front ${obs.agentview.shape[1]}x${obs.agentview.shape[0]}`,
				`wrist ${obs.wrist.shape[1]}x${obs.wrist.shape[0]}`,
			],
		};
		return {
			content: [
				{ type: "text" as const, text: JSON.stringify(details) },
				...images.map((png) => ({ type: "image" as const, data: png.toString("base64"), mimeType: "image/png" })),
			],
			details,
		};
	}

	const camera = Type.Optional(StringEnum(CAMERAS, { description: "agentview (front, default) or wrist" }));

	robot.tool(
		"view_env_state",
		"Current state with the front (third-person) and wrist images.",
		Type.Object({}),
		async () => observe({}),
	);

	robot.tool(
		"view_camera_meta",
		"Camera calibration of the current images: intrinsic K (3x3), camera-to-world extrinsic (4x4, OpenCV frame) and the image size.",
		Type.Object({ camera }),
		async ({ camera: c = "agentview" }) => {
			const m = await call<CameraMeta>("env.get_camera_meta", { camera_name: c });
			return { content: [{ type: "text" as const, text: JSON.stringify({ camera: c, ...m }) }], details: m };
		},
	);

	robot.tool(
		"back_project",
		"World xyz (m, base frame) of a pixel (row, col; row 0 = top) of the current camera image, from the simulator's depth.",
		Type.Object({ row: Type.Integer(), col: Type.Integer(), camera }),
		async ({ row, col, camera: c = "agentview" }) => {
			const [p] = await call<(number[] | null)[]>("env.back_project", { camera_name: c, pixels: [[row, col]] });
			const details = p
				? { camera: c, pixel: [row, col], world_xyz: p }
				: { camera: c, pixel: [row, col], error: "no depth at that pixel (background or out of the image)" };
			return { content: [{ type: "text" as const, text: JSON.stringify(details) }], details };
		},
	);

	robot.tool(
		"segment",
		"SAM3 segmentation of the current camera image by a text prompt or a positive point [row, col] (give exactly one). The mask's pixels are back-projected through the depth; world_xyz is their median. Returns an overlay image.",
		Type.Object({
			prompt: Type.Optional(Type.String()),
			point: Type.Optional(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 })),
			camera,
			min_score: Type.Optional(Type.Number({ description: "Default 0.2" })),
		}),
		async ({ prompt, point, camera: c = "agentview", min_score = 0.2 }) => {
			const text = prompt?.trim();
			const fail = (error: string) => ({
				content: [{ type: "text" as const, text: JSON.stringify({ error }) }],
				details: { error },
			});
			if (!text && !point) return fail("give a text prompt or a point [row, col]");
			const image = c === "wrist" ? obs.wrist : obs.agentview;
			const [h, w] = image.shape;
			const png = encodePng(image.data, w, h);
			const res = await sam3.call<{
				found: boolean;
				score?: number;
				box?: number[];
				mask_png_base64?: string;
				reason?: string;
			}>(
				"sam3.segment",
				{ image_base64: png.toString("base64"), ...(text ? { text_prompt: text } : { point }), min_score },
				120_000,
			);
			if (!res.found || !res.mask_png_base64)
				return fail(`${res.reason ?? "no mask"}; pick a pixel and use back_project`);
			const mask = decodePngChannel(Buffer.from(res.mask_png_base64, "base64"));
			if (mask.width !== w || mask.height !== h)
				return fail(`mask ${mask.width}x${mask.height} does not match the ${w}x${h} image`);
			const { n, centroid, pixels } = maskPixels(mask);
			const points = pixels.length
				? await call<(number[] | null)[]>("env.back_project", { camera_name: c, pixels })
				: [];
			const overlay = Buffer.from(image.data);
			for (let i = 0; i < mask.data.length; i++) {
				if (mask.data[i] < 128) continue;
				overlay[i * 3] = Math.round(0.55 * overlay[i * 3] + 0.45 * 255);
				overlay[i * 3 + 1] = Math.round(0.55 * overlay[i * 3 + 1]);
				overlay[i * 3 + 2] = Math.round(0.55 * overlay[i * 3 + 2]);
			}
			const world = medianPoint(points);
			const details = {
				found: true,
				camera: c,
				score: res.score === undefined ? null : round(res.score, 3),
				box: res.box,
				n_pixels: n,
				centroid_pixel: centroid,
				world_xyz: world,
				...(world ? {} : { world_error: "too few pixels with depth" }),
			};
			return {
				content: [
					{ type: "text" as const, text: JSON.stringify(details) },
					{ type: "image" as const, data: encodePng(overlay, w, h).toString("base64"), mimeType: "image/png" },
				],
				details,
			};
		},
	);

	robot.tool(
		"move_delta",
		`Translate the gripper by a base-frame [dx, dy, dz] in metres (+x away from the base, +y toward the robot's left, +z up; at most ${MAX_MOVE_M} m per call, inside the workspace box and above the table), optionally opening or closing the gripper first (the arm holds still until the fingers settle, then moves). The orientation is held. Returns the new state and images.`,
		Type.Object({
			delta_xyz: Type.Array(Type.Number(), { minItems: 3, maxItems: 3 }),
			gripper: Type.Optional(StringEnum(["open", "close"] as const)),
		}),
		async ({ delta_xyz, gripper }, signal) => {
			if (obs.success) return observe({ error: "the task is already solved; call finish" });
			return observe(await move(delta_xyz as Vec3, gripper ?? null, signal));
		},
	);

	robot.tool(
		"gripper",
		"Open or close the gripper and hold it (the command persists). A close that ends nearly shut holds nothing (`grasp_empty`). Returns the new state and images.",
		Type.Object({ action: StringEnum(["open", "close"] as const) }),
		async ({ action }, signal) => {
			const r = await call<Gripped>("env.set_gripper", { open: action === "open", return_frames: true }, [], signal);
			for (const f of r.frames ?? []) video.frame(f);
			const { frames: _frames, control_steps, grasp_empty, cancelled, ...o } = r;
			absorb(o);
			return observe({
				gripper: action,
				control_steps,
				...(grasp_empty ? { grasp_empty } : {}),
				...(cancelled ? { cancelled } : {}),
			});
		},
	);

	async function startEpisode() {
		const { task, seed } = robot.task;
		if (!(TASKS as readonly string[]).includes(task))
			throw new Error(`unknown --task ${task}; one of ${TASKS.join(", ")}`);
		const endpoint = pi.getFlag("env") as string | undefined;
		if (endpoint) env = await attach(endpoint);
		else {
			const services = flag("services", SERVICES);
			env = await robot.serve({
				python: flag("python", "python"),
				args: [
					...["-m", "pi_embodied_services.robots.genesis.env_server"],
					...["--task", task, "--seed", seed, "--backend", flag("backend", "gpu")],
				],
				cwd: services,
				env: { ...process.env, PYTHONPATH: services },
				log: (port) => join(tmpdir(), `pi-embodied-genesis-${task}-s${seed}-${port}.log`),
				// Genesis compiles its kernels on the first build (minutes on a cold cache).
				readyMs: 1_200_000,
			});
		}
		sam3 = new RpcClient(flag("sam3", ""));
		meta = await env.call<Meta>("env.get_env_meta");
		if (meta.task !== task || meta.seed !== Number(seed))
			throw new Error(`env server runs ${meta.task} seed ${meta.seed}, not ${task} seed ${seed}`);
		everGrasped = false;
		const [o] = await env.call<[Obs, unknown]>("env.reset", {}, 300_000);
		absorb(o);
		return ["view_env_state", "view_camera_meta", "segment", "back_project", "move_delta", "gripper", "finish"];
	}
}
