/**
 * Metaworld robot for pi: a Sawyer in Metaworld's MT50 / ML45 tabletop tasks (metaworld 3.1.1).
 *
 *   pi -e packages/embodied/src/metaworld --task reach-v3 --seed 0
 *   pi -e packages/embodied/src/metaworld --task pick-place-v3 --seed 3 --units=true
 *   pi -e packages/embodied/src/metaworld --task button-press-v3 --seed 0 --privileged
 *
 * Starts one Metaworld env server per session (services/.../robots/metaworld/env_server.py, the
 * `metaworld` venv; MuJoCo renders through EGL, or on the CPU with MUJOCO_GL=osmesa). The action is a world-frame hand translation plus
 * the gripper effort, so `move_delta` and the units hook `apply` share one server-side closed-loop
 * motion (`env.move_delta`: at most 0.2 m per call, inside the task's workspace box). Every result
 * carries the agentview (Metaworld's `corner4`) and wrist (`gripperPOV`) images and the state;
 * success is the env's own `info["success"]`, latched and recorded in `robot_result`.
 */

import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { sideBySide } from "../maniskill/index.ts";
import { decodePngChannel, encodePng } from "../png.ts";
import { attach, defineRobot, type Mat, median, round, SERVICES } from "../robot.ts";
import { type NdArray, RpcClient } from "../rpc.ts";
import type { MoveUnit, Vec3 } from "../units/index.ts";

const SYSTEM = readFileSync(new URL("./SYSTEM.md", import.meta.url), "utf8");

/** Metaworld's MT50 task names (`metaworld.env_dict.ALL_V3_ENVIRONMENTS`), the server's table. */
export const TASKS = [
	"assembly-v3",
	"basketball-v3",
	"bin-picking-v3",
	"box-close-v3",
	"button-press-topdown-v3",
	"button-press-topdown-wall-v3",
	"button-press-v3",
	"button-press-wall-v3",
	"coffee-button-v3",
	"coffee-pull-v3",
	"coffee-push-v3",
	"dial-turn-v3",
	"disassemble-v3",
	"door-close-v3",
	"door-lock-v3",
	"door-open-v3",
	"door-unlock-v3",
	"hand-insert-v3",
	"drawer-close-v3",
	"drawer-open-v3",
	"faucet-open-v3",
	"faucet-close-v3",
	"hammer-v3",
	"handle-press-side-v3",
	"handle-press-v3",
	"handle-pull-side-v3",
	"handle-pull-v3",
	"lever-pull-v3",
	"pick-place-wall-v3",
	"pick-out-of-hole-v3",
	"pick-place-v3",
	"plate-slide-v3",
	"plate-slide-side-v3",
	"plate-slide-back-v3",
	"plate-slide-back-side-v3",
	"peg-insert-side-v3",
	"peg-unplug-side-v3",
	"soccer-v3",
	"stick-push-v3",
	"stick-pull-v3",
	"push-v3",
	"push-wall-v3",
	"push-back-v3",
	"reach-v3",
	"reach-wall-v3",
	"shelf-place-v3",
	"sweep-into-v3",
	"sweep-v3",
	"window-open-v3",
	"window-close-v3",
] as const;
export type Task = (typeof TASKS)[number];

/**
 * World frame of Metaworld's Sawyer: it stands at the origin facing +y across the table, so +y is
 * away from the base, +x is the robot's right, +z up.
 */
export const VECTORS: Record<MoveUnit, Vec3> = {
	MV_FWD: [0, 1, 0],
	MV_BACK: [0, -1, 0],
	MV_LEFT: [-1, 0, 0],
	MV_RIGHT: [1, 0, 0],
	MV_UP: [0, 0, 1],
	MV_DOWN: [0, 0, -1],
};
/** Metres per MV_* decision (Show-Harness's 2 cm). */
export const STEP_M = 0.02;
/** Largest translation one call may command, m (the server refuses more). */
export const MAX_MOVE_M = 0.2;
/** Closed-and-empty gripper width, m: the pad distance settles at 0.023 on nothing (0.095 open). */
export const EMPTY_WIDTH_M = 0.025;
/** The table top, world z (Metaworld's tabletop geom; objects rest just above it). */
export const TABLE_Z = 0;
/** Square view size the server renders, px. */
export const VIEW_SIZE = 256;
/** The camera setup VIEWS describes; a server rendering anything else is refused. */
export const VIEW_SETUP = { agentview: "corner4", wrist: "gripperPOV", view_size: VIEW_SIZE };

/**
 * How the views look, measured on pick-place-v3 seed 0 by projecting the TCP and the puck through
 * the calibration while stepping each axis 6 cm: the agentview (`corner4`, turned 180 deg) stands
 * at the robot's right, above the table, looking across it: the robot base is at the image left,
 * +y (MV_FWD) runs toward the image right, +x (MV_RIGHT) toward the camera (image bottom), +z up.
 * The wrist camera (`gripperPOV`) rides on the hand: the two finger pads are fixed at the left,
 * one near the top and one near the bottom, and the grasp point is between them at mid-height,
 * about a quarter of the width from the left edge; MV_FWD shifts the scene down, MV_RIGHT left.
 */
export const VIEWS = `Each result shows the third-person view, then the wrist view (both ${VIEW_SIZE}x${VIEW_SIZE}).
- Third-person view: a fixed camera at the robot's right, looking across the table with the robot base at the image LEFT: MV_FWD moves the gripper toward the image RIGHT, MV_BACK toward the LEFT (toward the robot), MV_RIGHT toward the image bottom (toward the camera), MV_LEFT toward the image top, MV_UP up.
- Wrist view: it rides on the hand looking past the fingers toward the table; the two finger pads stay fixed at the left edge (one near the top, one near the bottom) and the grasp point is between them, at mid-height, about a quarter of the width from the left. A target above the grasp point in the image needs MV_FWD, below it MV_BACK, to its right MV_RIGHT, to its left MV_LEFT; a target between the pads is under the gripper: MV_DOWN. The camera moves with the gripper, so after MV_FWD the scene shifts down and after MV_RIGHT it shifts left.`;

type Obs = {
	agentview: NdArray;
	wrist: NdArray;
	tcp_pos: NdArray;
	gripper_width: number;
	obs: NdArray;
};
type Info = Record<string, unknown>;
type MoveResult = {
	ok: boolean;
	final_tcp_pos: number[];
	final_error_m: number;
	moved_m: number[];
	gripper: string;
	gripper_width: number;
	steps_used: number;
	frames: Obs[];
	info: Info;
	cancelled?: boolean;
};
type Meta = { task: string; seed: number; metaworld: string; workspace?: { min: number[]; max: number[] } } & Partial<
	typeof VIEW_SETUP
>;
type CameraMeta = { intrinsic_K: Mat; extrinsic_cam2world: Mat; height: number; width: number };
type WorldMap = { envStep: number; size: number; rgb: Buffer; xyz: Float32Array };
type Camera = "agentview" | "wrist";
const CAMERAS: readonly Camera[] = ["agentview", "wrist"];

/** A pixel's world point is valid when finite and not the zero the renderer gives the sky. */
const valid = (p: number[]) => p.every(Number.isFinite) && Math.abs(p[0]) + Math.abs(p[1]) + Math.abs(p[2]) > 1e-6;
const clip = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/**
 * Back-project a metric depth map (rows top-first, OpenCV intrinsics) through a camera-to-world
 * transform: the world xyz of every pixel, [row-major, 3 per pixel].
 */
export function backProject(depth: number[], size: number, k: Mat, c2w: Mat): Float32Array {
	const [[fx, , cx], [, fy, cy]] = k;
	const xyz = new Float32Array(size * size * 3);
	for (let r = 0; r < size; r++)
		for (let c = 0; c < size; c++) {
			const z = depth[r * size + c];
			const x = ((c - cx) * z) / fx;
			const y = ((r - cy) * z) / fy;
			const i = (r * size + c) * 3;
			for (let d = 0; d < 3; d++) xyz[i + d] = c2w[d][0] * x + c2w[d][1] * y + c2w[d][2] * z + c2w[d][3];
		}
	return xyz;
}

export default function metaworld(pi: ExtensionAPI) {
	const flag = (name: string, fallback: string) => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("task", {
		type: "string",
		default: "reach-v3",
		description: `Metaworld MT50 task (${TASKS.join(", ")})`,
	});
	pi.registerFlag("seed", { type: "string", default: "0", description: "Reset seed (the object layout)" });
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
		description: "Python for the env server (the metaworld venv)",
	});

	let env: RpcClient;
	let sam3: RpcClient;
	let obs: Obs;
	let info: Info = {};
	let success = false;
	let envStep = 0;
	let gripper: "open" | "close" = "open";
	let language = "";
	let workspace: Meta["workspace"];
	const worldMaps = new Map<string, WorldMap>();

	const robot = defineRobot(pi, {
		name: "metaworld",
		task: ["task", "seed"],
		keepImages: 4,
		video: true,
		vdm: { views: 2, wrist: 1 },
		groundTruth: (names) => call("env.ground_truth_poses", { names: names ?? null }),
		start: startEpisode,
		prompt: () => SYSTEM.replaceAll("{{task_language}}", language).replaceAll("{{table_z}}", String(TABLE_Z)),
		result: () => ({
			task: robot.task.task,
			seed: Number(robot.task.seed),
			success,
			env_steps: envStep,
		}),
		status: () => ({ language, step: envStep, solved: success }),
		// The env server's primitive registry (code.api), recorded per episode.
		codeApi: () => env,
		finish: {
			description:
				"End the episode after checking the latest state. Success is Metaworld's own success flag, not this call.",
			parameters: Type.Object({
				status: StringEnum(["success", "failure"] as const),
				summary: Type.String(),
			}),
			result: (params) => ({
				content: [{ type: "text", text: `Episode finished (success=${success}).` }],
				details: params,
			}),
		},
		units: {
			vectors: VECTORS,
			stepM: STEP_M,
			instruction: () => language,
			views: VIEWS,
			// gripperPOV, the second image (VIEW_SETUP).
			wrist: true,
			emptyWidthM: EMPTY_WIDTH_M,
			maxMoveM: () => MAX_MOVE_M,
			apply: async (m, signal) => {
				if (m.yaw || m.rot?.some(Boolean))
					throw new Error("this robot has no rotation (the hand orientation is fixed)");
				return observe(await move(m.delta, m.gripper, signal));
			},
			state: async () => ({
				eef_xyz: obs.tcp_pos.toArray().map((v) => round(v, 4)),
				gripper_width: round(obs.gripper_width, 4),
				table_z: TABLE_Z,
			}),
		},
	});
	const { video } = robot;

	const call = <T = unknown>(
		method: string,
		kwargs: Record<string, unknown> = {},
		args: unknown[] = [],
		signal = robot.signal,
	) => env.call<T>(method, kwargs, 120_000, args, signal);

	function absorb(o: Obs, i: Info) {
		obs = o;
		info = i;
		success ||= Boolean(i.success);
	}

	/** One server-side closed-loop move with an optional gripper command; every control step goes to the video. */
	async function move(delta: Vec3, grip: "open" | "close" | null, signal: AbortSignal | undefined) {
		const norm = Math.hypot(...delta);
		if (!(norm <= MAX_MOVE_M))
			throw new Error(`delta moves ${round(norm, 4)} m; the limit is ${MAX_MOVE_M} m per call. Split the motion.`);
		const r = await call<MoveResult>(
			"env.move_delta",
			grip ? { gripper: grip } : {},
			[delta.map((v) => round(v, 6))],
			signal,
		);
		for (const f of r.frames) video.frame(sideBySide(f.agentview, f.wrist));
		envStep += r.frames.length;
		worldMaps.clear();
		gripper = r.gripper === "close" ? "close" : "open";
		absorb(r.frames[r.frames.length - 1], r.info);
		return {
			commanded_m: delta.map((v) => round(v, 4)),
			moved_m: r.moved_m.map((v) => round(v, 4)),
			final_error_m: round(r.final_error_m, 4),
			gripper,
			env_steps: r.steps_used,
			...(r.cancelled ? { cancelled: true } : {}),
		};
	}

	/** The motion result with the new state, then the agentview and wrist images. */
	function observe(result: Record<string, unknown>) {
		const images = [obs.agentview, obs.wrist].map((a) => encodePng(a.data, a.shape[1], a.shape[0]));
		const details = {
			result,
			step: envStep,
			success,
			terminated: success,
			task_language: language,
			state: {
				tcp_pos: obs.tcp_pos.toArray().map((v) => round(v, 4)),
				gripper_width: round(obs.gripper_width, 4),
				gripper_command: gripper,
				grasp_success: Boolean(info.grasp_success),
				...(workspace ? { workspace } : {}),
			},
			images: [
				`agentview ${obs.agentview.shape[1]}x${obs.agentview.shape[0]}`,
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

	/** The current camera image with the world xyz of every pixel (depth back-projected), cached per env step. */
	async function worldMap(camera: Camera, size: number): Promise<WorldMap> {
		const key = `${camera}:${size}`;
		const cached = worldMaps.get(key);
		if (cached?.envStep === envStep) return cached;
		const [rgb, depth] = await call<[NdArray, NdArray]>("env.render_camera", {
			camera_name: camera,
			height: size,
			width: size,
			depth: true,
		});
		const meta = await call<CameraMeta>("env.get_camera_meta", { camera_name: camera, height: size, width: size });
		const map = {
			envStep,
			size,
			rgb: Buffer.from(rgb.data),
			xyz: backProject(depth.toArray(), size, meta.intrinsic_K, meta.extrinsic_cam2world),
		};
		worldMaps.set(key, map);
		return map;
	}

	const camera = Type.Optional(StringEnum(CAMERAS, { description: "Default agentview" }));
	const resolution = Type.Optional(
		StringEnum(["low", "high"] as const, {
			description: `low = ${VIEW_SIZE} (the images shown, default), high = 1024`,
		}),
	);
	const sizeOf = (r: "low" | "high") => (r === "high" ? 1024 : VIEW_SIZE);
	const text = (out: Record<string, unknown>, image?: Buffer) => ({
		content: [
			{ type: "text" as const, text: JSON.stringify(out) },
			...(image ? [{ type: "image" as const, data: image.toString("base64"), mimeType: "image/png" }] : []),
		],
		details: out,
	});

	robot.tool(
		"view_env_state",
		"Current state with the agentview (third-person) and wrist images.",
		Type.Object({}),
		async () => observe({}),
	);

	robot.tool(
		"view_camera_meta",
		"Camera calibration (OpenCV intrinsic K and camera-to-world extrinsic) of the agentview or wrist camera at the given resolution.",
		Type.Object({ camera, resolution }),
		async ({ camera: c = "agentview", resolution: r = "low" }) => {
			const size = sizeOf(r);
			const meta = await call<CameraMeta>("env.get_camera_meta", { camera_name: c, height: size, width: size });
			return text({ camera: c, resolution: r, meta });
		},
	);

	robot.tool(
		"segment",
		`SAM3 segmentation of the current camera image (${VIEW_SIZE}x${VIEW_SIZE} as shown, or 1024 with resolution high). Give exactly one of a text prompt or a positive point [row, col]. The top mask is projected through the depth map; world_xyz is the median over its pixels. Returns an overlay image.`,
		Type.Object({
			prompt: Type.Optional(Type.String()),
			point: Type.Optional(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 })),
			camera,
			resolution,
			min_score: Type.Optional(Type.Number({ description: "Default 0.2" })),
		}),
		async ({ prompt, point, camera: c = "agentview", resolution: r = "low", min_score = 0.2 }) => {
			const query = prompt?.trim();
			if (!query && !point) return text({ error: "give a text prompt or a point [row, col]" });
			const size = sizeOf(r);
			const map = await worldMap(c, size);
			const res = await sam3.call<{
				found: boolean;
				score?: number;
				box?: number[];
				mask_png_base64?: string;
				reason?: string;
			}>("sam3.segment", {
				image_base64: encodePng(map.rgb, size, size).toString("base64"),
				...(query ? { text_prompt: query } : { point }),
				min_score,
			});
			if (!res.found || !res.mask_png_base64)
				return text({
					found: false,
					error: res.reason ?? "no mask",
					fallback: "Pick pixels in the image and use back_project.",
				});
			const mask = decodePngChannel(Buffer.from(res.mask_png_base64, "base64"));
			if (mask.width !== size || mask.height !== size)
				return text({ found: true, error: `mask ${mask.width}x${mask.height} does not match the ${size} image` });
			const xs: number[] = [];
			const ys: number[] = [];
			const pts: number[][] = [];
			const overlay = Buffer.from(map.rgb);
			for (let i = 0; i < mask.data.length; i++) {
				if (mask.data[i] < 128) continue;
				ys.push(Math.floor(i / size));
				xs.push(i % size);
				overlay[i * 3] = Math.round(0.55 * overlay[i * 3] + 0.45 * 255);
				overlay[i * 3 + 1] = Math.round(0.55 * overlay[i * 3 + 1]);
				overlay[i * 3 + 2] = Math.round(0.55 * overlay[i * 3 + 2]);
				const p = [map.xyz[i * 3], map.xyz[i * 3 + 1], map.xyz[i * 3 + 2]];
				if (valid(p)) pts.push(p);
			}
			return text(
				{
					found: true,
					camera: c,
					resolution: r,
					score: res.score === undefined ? null : round(res.score, 3),
					box: res.box,
					n_pixels: xs.length,
					n_valid: pts.length,
					centroid_pixel: [Math.round(median(ys)), Math.round(median(xs))],
					world_xyz: pts.length < 10 ? null : [0, 1, 2].map((k) => round(median(pts.map((p) => p[k])), 4)),
					...(pts.length < 10 ? { world_error: `too few valid depth pixels (${pts.length})` } : {}),
				},
				encodePng(overlay, size, size),
			);
		},
	);

	robot.tool(
		"back_project",
		`World xyz of a pixel (row, col; row 0 = top) in the current camera image (${VIEW_SIZE}x${VIEW_SIZE} as shown, or 1024 with resolution high), from the depth map. Region mode: row_range + col_range (+ optional z_min/z_max) returns the midpoint of world xy over that window, e.g. a container's interior centre.`,
		Type.Object({
			row: Type.Optional(Type.Integer()),
			col: Type.Optional(Type.Integer()),
			camera,
			resolution,
			row_range: Type.Optional(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 })),
			col_range: Type.Optional(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 })),
			z_min: Type.Optional(Type.Number()),
			z_max: Type.Optional(Type.Number()),
		}),
		async ({ row, col, camera: c = "agentview", resolution: r = "low", row_range, col_range, z_min, z_max }) => {
			const size = sizeOf(r);
			const map = await worldMap(c, size);
			const at = (rr: number, cc: number) => {
				const i = (rr * size + cc) * 3;
				return [map.xyz[i], map.xyz[i + 1], map.xyz[i + 2]];
			};
			// An empty window ([0, 0]) is a placeholder, not a region.
			const span = (v?: number[]) => (v && Math.max(...v) > Math.min(...v) ? v : undefined);
			const rows = span(row_range);
			const cols = span(col_range);
			if (rows || cols) {
				if (!rows || !cols) return text({ error: "region mode needs both row_range and col_range" });
				const [r0, r1] = [clip(Math.min(...rows), 0, size), clip(Math.max(...rows), 0, size)];
				const [c0, c1] = [clip(Math.min(...cols), 0, size), clip(Math.max(...cols), 0, size)];
				let pts: number[][] = [];
				for (let rr = r0; rr < r1; rr++)
					for (let cc = c0; cc < c1; cc++) if (valid(at(rr, cc))) pts.push(at(rr, cc));
				if (z_min !== undefined) pts = pts.filter((p) => p[2] >= z_min);
				if (z_max !== undefined) pts = pts.filter((p) => p[2] <= z_max);
				if (pts.length < 8)
					return text({ error: `too few valid pixels in region (${pts.length}); widen the window or the z band` });
				const axis = (k: number) => pts.map((p) => p[k]);
				return text({
					camera: c,
					resolution: r,
					mode: "region",
					center_xyz: [
						round((Math.min(...axis(0)) + Math.max(...axis(0))) / 2, 4),
						round((Math.min(...axis(1)) + Math.max(...axis(1))) / 2, 4),
						round(median(axis(2)), 4),
					],
					median_xyz: [0, 1, 2].map((k) => round(median(axis(k)), 4)),
					n_valid: pts.length,
				});
			}
			if (row === undefined || col === undefined)
				return text({ error: "give row and col, or row_range and col_range" });
			if (row < 0 || row >= size || col < 0 || col >= size)
				return text({ error: `pixel (${row},${col}) out of bounds for ${size}x${size}` });
			const p = at(row, col);
			if (!valid(p)) return text({ error: `invalid world xyz at (${row},${col}); pick another pixel` });
			return text({ camera: c, resolution: r, pixel: [row, col], world_xyz: p.map((v) => round(v, 4)) });
		},
	);

	const xyz = Type.Array(Type.Number(), { minItems: 3, maxItems: 3 });
	robot.tool(
		"move_delta",
		`Translate the gripper by a world-frame [dx, dy, dz] in metres (+y away from the robot, +x toward the robot's right, +z up; at most ${MAX_MOVE_M} m per call, inside the workspace box), optionally opening or closing the gripper first (the arm holds still until the fingers settle, then moves). The orientation is fixed. Returns the new state and images.`,
		Type.Object({
			delta_xyz: xyz,
			gripper: Type.Optional(StringEnum(["open", "close"] as const)),
		}),
		async ({ delta_xyz, gripper: g }, signal) => {
			if (success) return observe({ error: "the task is already solved; call finish" });
			return observe(await move(delta_xyz as Vec3, g ?? null, signal));
		},
	);

	robot.tool(
		"gripper",
		"Open or close the gripper in place and hold until the fingers settle. The command persists across moves. Returns the new state and images.",
		Type.Object({ action: StringEnum(["open", "close"] as const) }),
		async ({ action }, signal) => {
			if (success) return observe({ error: "the task is already solved; call finish" });
			return observe(await move([0, 0, 0], action, signal));
		},
	);

	async function startEpisode() {
		const { task, seed } = robot.task;
		if (!(TASKS as readonly string[]).includes(task))
			throw new Error(`unknown Metaworld task "${task}"; the ${TASKS.length} tasks are ${TASKS.join(", ")}`);
		sam3 = new RpcClient(flag("sam3", ""));
		const endpoint = pi.getFlag("env") as string | undefined;
		if (endpoint) env = await attach(endpoint);
		else {
			const services = flag("services", SERVICES);
			env = await robot.serve({
				python: flag("python", "python"),
				args: ["-m", "pi_embodied_services.robots.metaworld.env_server", "--task", task, "--seed", seed],
				cwd: services,
				// EGL unless the caller picks MUJOCO_GL=osmesa (CPU rendering).
				env: { ...process.env, PYTHONPATH: services, MUJOCO_GL: process.env.MUJOCO_GL ?? "egl" },
				log: (port) => join(tmpdir(), `pi-embodied-metaworld-${task}-s${seed}-${port}.log`),
				readyMs: 300_000,
			});
		}
		const meta = await env.call<Meta>("env.get_env_meta");
		if (meta.task !== task || meta.seed !== Number(seed))
			throw new Error(`env server runs ${meta.task} seed ${meta.seed}, not ${task} seed ${seed}`);
		const setup = Object.entries(VIEW_SETUP).filter(([k, v]) => meta[k as keyof typeof VIEW_SETUP] !== v);
		if (setup.length)
			throw new Error(
				`env server cameras (${setup.map(([k]) => `${k}=${meta[k as keyof typeof VIEW_SETUP]}`).join(", ")}) differ from the ones the prompt describes (${JSON.stringify(VIEW_SETUP)}); update the services dir`,
			);
		success = false;
		envStep = 0;
		gripper = "open";
		worldMaps.clear();
		const [o, i] = await env.call<[Obs, Info]>("env.reset", {}, 300_000);
		absorb(o, i);
		workspace = (await env.call<Meta>("env.get_env_meta")).workspace;
		language = await env.call<string>("env.get_task_language");
		return ["view_env_state", "view_camera_meta", "segment", "back_project", "move_delta", "gripper", "finish"];
	}
}
