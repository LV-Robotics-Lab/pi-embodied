/**
 * BEHAVIOR-1K robot for pi: an R1Pro (holonomic base, torso, two arms with parallel grippers, a ZED
 * head camera and a RealSense on each wrist) on one 2025-challenge task in OmniGibson (Isaac Sim).
 *
 *   pi -e packages/embodied/src/behavior --task turning_on_radio --seed 0 --gpu-id 1
 *   pi -e packages/embodied/src/behavior --task picking_up_trash --seed 2 --privileged
 *
 * Starts one BEHAVIOR env server per session (services/.../robots/behavior/env_server.py in the
 * `behavior` venv, see robots/behavior/install.sh; OmniGibson loads a whole house, minutes). The
 * tools are OmniGibson's semantic primitives as CaP-X's R1ProControlApi exposed them
 * (navigate_to_pose, move_hand, grasp_object, open/close_gripper, get_robot_position), the
 * perception tools (segment via SAM3, point via Molmo, back_project through the cameras' metric
 * depth) and view_env_state; every motion result carries the head and both wrist images. Success is
 * the BDDL task's `success`, and `q_score` (BEHAVIOR's partial credit) goes into `robot_result`;
 * what only the simulator knows (the object OmniGibson's grasping holds) reaches the planner only
 * under --privileged, and CaP-X's "picked" judgement is recorded as a reference field, never as
 * success. --seed is the task's pre-sampled instance id.
 *
 * Copyright 2026 The CaP-X Authors (github.com/capgym/cap-x @53e9966). Licensed under the Apache
 * License, Version 2.0. Modified by pi-embodied: capx/integrations/r1pro/control.py's primitive set
 * ported as pi tools; the reward of capx/envs/simulators/r1pro_b1k.py is not.
 */

import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { decodePngChannel, encodePng } from "../png.ts";
import { attach, defineRobot, type Mat, mark, median, round, SERVICES, toolResult } from "../robot.ts";
import { type NdArray, RpcClient } from "../rpc.ts";

const SYSTEM = readFileSync(new URL("./SYSTEM.md", import.meta.url), "utf8");

/** The 2025 challenge tasks, in the env server's order (services/.../robots/behavior/tasks.py): CaP-X's two first. */
export const TASKS = [
	"turning_on_radio",
	"picking_up_trash",
	"putting_away_Halloween_decorations",
	"cleaning_up_plates_and_food",
	"can_meat",
	"setting_mousetraps",
	"hiding_Easter_eggs",
	"picking_up_toys",
	"rearranging_kitchen_furniture",
	"putting_up_Christmas_decorations_inside",
	"set_up_a_coffee_station_in_your_kitchen",
	"putting_dishes_away_after_cleaning",
	"preparing_lunch_box",
	"loading_the_car",
	"carrying_in_groceries",
	"bringing_in_wood",
	"moving_boxes_to_storage",
	"bringing_water",
	"tidying_bedroom",
	"outfit_a_basic_toolbox",
	"sorting_vegetables",
	"collecting_childrens_toys",
	"putting_shoes_on_rack",
	"boxing_books_up_for_storage",
	"storing_food",
	"clearing_food_from_table_into_fridge",
	"assembling_gift_baskets",
	"sorting_household_items",
	"getting_organized_for_work",
	"clean_up_your_desk",
	"setting_the_fire",
	"clean_boxing_gloves",
	"wash_a_baseball_cap",
	"wash_dog_toys",
	"hanging_pictures",
	"attach_a_camera_to_a_tripod",
	"clean_a_patio",
	"clean_a_trumpet",
	"spraying_for_bugs",
	"spraying_fruit_trees",
	"make_microwave_popcorn",
	"cook_cabbage",
	"chop_an_onion",
	"slicing_vegetables",
	"chopping_wood",
	"cook_hot_dogs",
	"cook_bacon",
	"freeze_pies",
	"canning_food",
	"make_pizza",
] as const;
export type Task = (typeof TASKS)[number];
export const TaskSchema = StringEnum(TASKS);

export const CAMERAS = ["head", "left_wrist", "right_wrist"] as const;
export type Camera = (typeof CAMERAS)[number];
export const ARMS = ["left", "right"] as const;
/** Depth beyond this is no hit (OmniGibson's depth_linear on the sky), m. */
export const MAX_DEPTH_M = 20;

type Eef = { pos: NdArray; quat_xyzw: NdArray; gripper_width: number };
type Privileged = { in_hand: Record<(typeof ARMS)[number], string | null>; picked: boolean };
export type Obs = {
	head: NdArray;
	head_depth: NdArray;
	left_wrist: NdArray;
	left_wrist_depth: NdArray;
	right_wrist: NdArray;
	right_wrist_depth: NdArray;
	base_pos: NdArray;
	base_quat_xyzw: NdArray;
	base_yaw: number;
	eef: Record<(typeof ARMS)[number], Eef>;
	/** BDDL success (the task's own termination). */
	success: boolean;
	/** BEHAVIOR's partial credit: newly satisfied goal predicates over all, 1 on success. */
	q_score: number;
	goals: { satisfied: number; total: number };
	terminated: boolean;
	truncated: boolean;
	env_steps: number;
	/** What only the simulator knows; shown to the planner under --privileged only. */
	privileged: Privileged;
};
type Motion = Obs & {
	primitive: string;
	ok: boolean;
	phase?: string;
	steps: number;
	error?: string;
	cancelled?: boolean;
	[k: string]: unknown;
};
type Meta = { task: string; seed: number; instruction: string; image_size: number; grasping_mode: string };
type CameraMeta = { intrinsic_K: Mat; extrinsic_cam2world: Mat; convention: string; width: number; height: number };
type WorldMap = { step: number; width: number; height: number; rgb: Buffer; xyz: Float32Array };

/**
 * World xyz of every pixel of an OmniGibson camera: OpenGL convention (looks along -Z, +Y up), so
 * `x = (c - cx) d / fx, y = -(r - cy) d / fy, z = -d` before the cam-to-world transform. NaN where
 * there is no hit.
 */
export function project(depth: Float32Array, width: number, height: number, meta: CameraMeta): Float32Array {
	const [[fx, , cx], [, fy, cy]] = meta.intrinsic_K;
	const e = meta.extrinsic_cam2world;
	const xyz = new Float32Array(width * height * 3);
	for (let r = 0; r < height; r++)
		for (let c = 0; c < width; c++) {
			const d = depth[r * width + c];
			const i = (r * width + c) * 3;
			if (!(d > 0 && d < MAX_DEPTH_M)) {
				xyz[i] = xyz[i + 1] = xyz[i + 2] = Number.NaN;
				continue;
			}
			const x = ((c - cx) * d) / fx;
			const y = (-(r - cy) * d) / fy;
			const z = -d;
			for (let k = 0; k < 3; k++) xyz[i + k] = e[k][0] * x + e[k][1] * y + e[k][2] * z + e[k][3];
		}
	return xyz;
}

export default function behavior(pi: ExtensionAPI) {
	const flag = (name: string, fallback: string) => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("task", {
		type: "string",
		default: TASKS[0],
		description: `BEHAVIOR-1K challenge activity: ${TASKS.slice(0, 3).join(", ")}, ... (50)`,
	});
	pi.registerFlag("seed", { type: "string", default: "0", description: "The task's pre-sampled instance id" });
	pi.registerFlag("gpu-id", { type: "string", default: "0", description: "GPU for Isaac Sim (OMNIGIBSON_GPU_ID)" });
	pi.registerFlag("image-size", { type: "string", default: "480", description: "Camera frames, px (square)" });
	pi.registerFlag("grasping-mode", {
		type: "string",
		default: "sticky",
		description: "OmniGibson grasping: sticky (CaP-X) or assisted",
	});
	pi.registerFlag("env", { type: "string", description: "Attach to a running env server instead of starting one" });
	pi.registerFlag("sam3", { type: "string", default: "http://127.0.0.1:18300", description: "SAM3 server (segment)" });
	pi.registerFlag("molmo", { type: "string", default: "http://127.0.0.1:18400", description: "Molmo server (point)" });
	pi.registerFlag("services", {
		type: "string",
		default: process.env.PI_EMBODIED_SERVICES ?? SERVICES,
		description: "pi-embodied services dir",
	});
	pi.registerFlag("python", {
		type: "string",
		default: process.env.PI_EMBODIED_PYTHON ?? "python",
		description: "Python for the env server (the behavior venv)",
	});

	let env: RpcClient;
	let sam3: RpcClient;
	let molmo: RpcClient;
	let obs: Obs;
	let meta: Meta;
	const worldMaps = new Map<Camera, WorldMap>();
	const privileged = () => pi.getFlag("privileged") === true;

	const robot = defineRobot(pi, {
		name: "behavior",
		task: ["task", "seed"],
		// The env server's primitive registry (code.api, services robots/behavior/primitives.py), recorded per episode.
		codeApi: () => env,
		keepImages: 6,
		video: true,
		// Observations carry the head image, then the left and right wrist images.
		vdm: { views: 3, wrist: [1, 2] },
		groundTruth: (names) => env.call("env.ground_truth_poses", { names: names ?? null }, 120_000, [], robot.signal),
		start: startEpisode,
		prompt: () => SYSTEM.replaceAll("{{task_language}}", meta.instruction),
		result: () => ({
			task: robot.task.task,
			seed: Number(robot.task.seed),
			instruction: meta?.instruction ?? null,
			success: obs?.success ?? false,
			q_score: obs?.q_score ?? 0,
			goals: obs?.goals ?? null,
			// CaP-X's pick_up_*_reward judgement, for comparison only: never success.
			reference: { picked: obs?.privileged.picked ?? false, in_hand: obs?.privileged.in_hand ?? null },
			truncated: obs?.truncated ?? false,
			env_steps: obs?.env_steps ?? 0,
		}),
		status: () => ({ language: meta.instruction, step: obs.env_steps, solved: obs.success }),
		finish: {
			description:
				"End the episode after checking the latest state. Success is the BDDL task's own predicate, not this call.",
			parameters: Type.Object({
				status: StringEnum(["success", "failure"] as const),
				summary: Type.String(),
			}),
			result: (params) => ({
				content: [{ type: "text", text: `Episode finished (success=${obs.success}, q_score=${obs.q_score}).` }],
				details: params,
			}),
		},
	});
	const { video } = robot;

	/** Run one primitive on the server (minutes: cuRobo plans plus hundreds of control steps); the head frame goes to the video. */
	async function motion(method: string, kwargs: Record<string, unknown>, signal: AbortSignal | undefined) {
		const r = await env.call<Motion>(method, kwargs, 900_000, [], signal ?? robot.signal);
		absorb(r);
		const { primitive, ok, phase, steps, error, cancelled, ...rest } = r;
		const extra = Object.fromEntries(Object.entries(rest).filter(([k]) => !(k in obs)));
		return {
			primitive,
			ok,
			...(phase ? { phase } : {}),
			steps,
			...(error ? { error } : {}),
			...(cancelled ? { cancelled } : {}),
			...extra,
		};
	}

	const OBS_KEYS: Record<keyof Obs, true> = {
		head: true,
		head_depth: true,
		left_wrist: true,
		left_wrist_depth: true,
		right_wrist: true,
		right_wrist_depth: true,
		base_pos: true,
		base_quat_xyzw: true,
		base_yaw: true,
		eef: true,
		success: true,
		q_score: true,
		goals: true,
		terminated: true,
		truncated: true,
		env_steps: true,
		privileged: true,
	};

	function absorb(o: Obs) {
		obs = Object.fromEntries(Object.entries(o).filter(([k]) => k in OBS_KEYS)) as Obs;
		worldMaps.clear();
		video.frame(obs.head);
	}

	const eefState = (e: Eef) => ({
		pos: e.pos.toArray().map((v) => round(v, 4)),
		quat_xyzw: e.quat_xyzw.toArray().map((v) => round(v, 4)),
		gripper_width: round(e.gripper_width, 4),
	});

	/** The result with the new state, then the head, left wrist and right wrist images. */
	function observe(result: Record<string, unknown>) {
		const frames = [obs.head, obs.left_wrist, obs.right_wrist];
		const pngs = frames.map((a) => encodePng(a.data, a.shape[1], a.shape[0]));
		const details = {
			result,
			step: obs.env_steps,
			success: obs.success,
			q_score: obs.q_score,
			goals: obs.goals,
			truncated: obs.truncated,
			task_language: meta.instruction,
			state: {
				base_pos: obs.base_pos.toArray().map((v) => round(v, 4)),
				base_yaw: round(obs.base_yaw, 4),
				eef: { left: eefState(obs.eef.left), right: eefState(obs.eef.right) },
				// Simulator-only: the object OmniGibson's grasping holds (CaP-X's S1 tier).
				...(privileged() ? { in_hand: obs.privileged.in_hand } : {}),
			},
			images: CAMERAS.map((c, i) => `${c} ${frames[i].shape[1]}x${frames[i].shape[0]}`),
		};
		return {
			content: [
				{ type: "text" as const, text: JSON.stringify(details) },
				...pngs.map((png) => ({ type: "image" as const, data: png.toString("base64"), mimeType: "image/png" })),
			],
			details,
		};
	}

	/** World xyz per pixel of a camera's latest frame (cached per env step; the cameras move with the robot). */
	async function worldMap(camera: Camera): Promise<WorldMap> {
		const cached = worldMaps.get(camera);
		if (cached?.step === obs.env_steps) return cached;
		const rgb = obs[camera] as NdArray;
		const depth = obs[`${camera}_depth`] as NdArray;
		const [height, width] = rgb.shape;
		const cm = await env.call<CameraMeta>("env.get_camera_meta", { camera_name: camera }, 60_000, [], robot.signal);
		if (cm.convention !== "opengl") throw new Error(`camera ${camera}: unexpected convention ${cm.convention}`);
		const map = {
			step: obs.env_steps,
			width,
			height,
			rgb: Buffer.from(rgb.data),
			xyz: project(Float32Array.from(depth.toArray()), width, height, cm),
		};
		worldMaps.set(camera, map);
		return map;
	}

	const valid = (p: number[]) => p.every(Number.isFinite);
	/** Median world xyz of the valid pixels in a (2k+1)^2 window around (row, col), or null. */
	function around(map: WorldMap, row: number, col: number, k: number) {
		const pts: number[][] = [];
		for (let r = Math.max(0, row - k); r <= Math.min(map.height - 1, row + k); r++)
			for (let c = Math.max(0, col - k); c <= Math.min(map.width - 1, col + k); c++) {
				const i = (r * map.width + c) * 3;
				const p = [map.xyz[i], map.xyz[i + 1], map.xyz[i + 2]];
				if (valid(p)) pts.push(p);
			}
		return pts.length < 3 ? null : [0, 1, 2].map((j) => round(median(pts.map((p) => p[j])), 4));
	}

	const arm = StringEnum(ARMS, { description: "Which arm" });
	const camera = Type.Optional(StringEnum(CAMERAS, { description: "Default head" }));
	const xyz = Type.Array(Type.Number(), { minItems: 3, maxItems: 3, description: "World [x, y, z], m" });
	const quat = Type.Optional(
		Type.Array(Type.Number(), {
			minItems: 4,
			maxItems: 4,
			description: "World xyzw orientation (default: keep the current one)",
		}),
	);
	const ended = (): ReturnType<typeof observe> | undefined => {
		if (obs.truncated) return observe({ error: "the episode is over (max steps)" });
		if (obs.success) return observe({ error: "the task is already solved; call finish" });
		return undefined;
	};

	robot.tool(
		"view_env_state",
		"Current state with the head (ZED) image and the left and right wrist (RealSense) images. Pixel (row, col) in these images feed back_project.",
		Type.Object({}),
		async () => observe({}),
	);

	robot.tool(
		"get_robot_position",
		"World pose of the base (position, xyzw, yaw in rad) and of both end effectors; no motion.",
		Type.Object({}),
		async () =>
			toolResult(
				(await env.call("env.get_robot_position", {}, 60_000, [], robot.signal)) as Record<string, unknown>,
			),
	);

	robot.tool(
		"navigate_to_pose",
		"Drive the base to world (x, y) facing yaw (rad about +z; 0 = +x). The planner avoids obstacles and refuses goals inside them or farther than 5 m; stand about 0.5 m from a table edge, facing it. Returns ok, the pose reached and images.",
		Type.Object({ x: Type.Number(), y: Type.Number(), yaw: Type.Number() }),
		async ({ x, y, yaw }, signal) => ended() ?? observe(await motion("env.navigate_to_pose", { x, y, yaw }, signal)),
	);

	robot.tool(
		"move_hand",
		"Plan and move one arm's end effector to a world position (and xyzw orientation, default the current one), avoiding obstacles. Reach is about 1.5 m from the base in xy: navigate first. Returns ok, the eef pose reached, distance_left_m and images.",
		Type.Object({ arm, position: xyz, quat_xyzw: quat }),
		async ({ arm: a, position, quat_xyzw }, signal) =>
			ended() ?? observe(await motion("env.move_hand", { arm: a, position, quat_xyzw: quat_xyzw ?? null }, signal)),
	);

	robot.tool(
		"grasp_object",
		"Grasp at a world pose with one arm: open, move to pregrasp_offset_m above it, descend along the approach, close, settle, lift back. Aim at the object's grasp point from segment / point / back_project. Judge the grasp from gripper_width (near 0 = nothing held) and the wrist image.",
		Type.Object({
			arm,
			position: xyz,
			quat_xyzw: quat,
			pregrasp_offset_m: Type.Optional(Type.Number({ description: "Default 0.1 (0.02-0.5)" })),
		}),
		async ({ arm: a, position, quat_xyzw, pregrasp_offset_m }, signal) =>
			ended() ??
			observe(
				await motion(
					"env.grasp_object",
					{ arm: a, position, quat_xyzw: quat_xyzw ?? null, pregrasp_offset_m: pregrasp_offset_m ?? 0.1 },
					signal,
				),
			),
	);

	robot.tool(
		"open_gripper",
		"Open one arm's gripper fully (releases what it holds). Returns gripper_width and images.",
		Type.Object({ arm }),
		async ({ arm: a }, signal) => ended() ?? observe(await motion("env.open_gripper", { arm: a }, signal)),
	);

	robot.tool(
		"close_gripper",
		"Close one arm's gripper fully. Returns gripper_width and images.",
		Type.Object({ arm }),
		async ({ arm: a }, signal) => ended() ?? observe(await motion("env.close_gripper", { arm: a }, signal)),
	);

	robot.tool(
		"segment",
		"SAM3 segmentation of the latest image of a camera. Give exactly one of a text prompt or a positive point [row, col]. The top mask is projected through that camera's depth; world_xyz is the median over mask pixels, top_xyz the median of its highest points (a grasp point). Returns an overlay image.",
		Type.Object({
			prompt: Type.Optional(Type.String()),
			point: Type.Optional(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 })),
			camera,
			min_score: Type.Optional(Type.Number({ description: "Default 0.2" })),
		}),
		async ({ prompt, point, camera: c = "head", min_score = 0.2 }) => {
			const text = prompt?.trim();
			if (!text && !point) return toolResult({ error: "give a text prompt or a point [row, col]" });
			const map = await worldMap(c);
			const png = encodePng(map.rgb, map.width, map.height);
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
				[],
				robot.signal,
			);
			if (!res.found || !res.mask_png_base64)
				return toolResult({
					found: false,
					camera: c,
					error: res.reason ?? "no mask",
					fallback: "Pick pixels in the image and use back_project.",
				});
			const mask = decodePngChannel(Buffer.from(res.mask_png_base64, "base64"));
			if (mask.width !== map.width || mask.height !== map.height)
				return toolResult({
					found: true,
					error: `mask ${mask.width}x${mask.height} does not match the ${map.width}x${map.height} image`,
				});
			const rows: number[] = [];
			const cols: number[] = [];
			const pts: number[][] = [];
			const overlay = Buffer.from(map.rgb);
			for (let i = 0; i < mask.data.length; i++) {
				if (mask.data[i] < 128) continue;
				rows.push(Math.floor(i / map.width));
				cols.push(i % map.width);
				overlay[i * 3] = Math.round(0.55 * overlay[i * 3] + 0.45 * 255);
				overlay[i * 3 + 1] = Math.round(0.55 * overlay[i * 3 + 1]);
				overlay[i * 3 + 2] = Math.round(0.55 * overlay[i * 3 + 2]);
				const p = [map.xyz[i * 3], map.xyz[i * 3 + 1], map.xyz[i * 3 + 2]];
				if (valid(p)) pts.push(p);
			}
			const out: Record<string, unknown> = {
				found: true,
				camera: c,
				score: res.score === undefined ? null : round(res.score, 3),
				box: res.box,
				n_pixels: rows.length,
				n_valid: pts.length,
				centroid_pixel: rows.length ? [Math.round(median(rows)), Math.round(median(cols))] : null,
			};
			if (pts.length < 10) {
				out.world_xyz = null;
				out.world_error = `too few valid depth pixels (${pts.length})`;
			} else {
				out.world_xyz = [0, 1, 2].map((k) => round(median(pts.map((p) => p[k])), 4));
				// The object's top: the highest tenth of its points.
				const top = [...pts].sort((a, b) => b[2] - a[2]).slice(0, Math.max(10, Math.floor(pts.length / 10)));
				out.top_xyz = [0, 1, 2].map((k) => round(median(top.map((p) => p[k])), 4));
			}
			return toolResult(out, [encodePng(overlay, map.width, map.height)]);
		},
	);

	robot.tool(
		"point",
		"Molmo points at what a short noun phrase names in a camera's latest image ('the radio on the table'). Returns the pixel, its world xyz through the depth (median of a 7x7 window) and the image with the point marked.",
		Type.Object({ query: Type.String(), camera }),
		async ({ query, camera: c = "head" }) => {
			const map = await worldMap(c);
			const png = encodePng(map.rgb, map.width, map.height);
			const res = await molmo.call<{ point_xy?: number[] | null; answer?: string; image_size?: number[] }>(
				"molmo.ground",
				{ image_base64: png.toString("base64"), query },
				120_000,
				[],
				robot.signal,
			);
			if (!res.point_xy)
				return toolResult({
					found: false,
					camera: c,
					answer: res.answer ?? null,
					fallback: "Use segment or back_project.",
				});
			const col = Math.max(0, Math.min(map.width - 1, Math.round(res.point_xy[0])));
			const row = Math.max(0, Math.min(map.height - 1, Math.round(res.point_xy[1])));
			const marked = mark({ width: map.width, height: map.height, rgb: map.rgb }, row, col, [255, 32, 32]);
			return toolResult(
				{
					found: true,
					camera: c,
					pixel: [row, col],
					world_xyz: around(map, row, col, 3),
					answer: res.answer ?? null,
				},
				[encodePng(marked, map.width, map.height)],
			);
		},
	);

	robot.tool(
		"back_project",
		"World xyz of a pixel (row, col; row 0 = top) in a camera's latest image, through its metric depth (median of a 3x3 window). Region mode: row_range + col_range returns the median and the xy midpoint over that window.",
		Type.Object({
			row: Type.Optional(Type.Integer()),
			col: Type.Optional(Type.Integer()),
			camera,
			row_range: Type.Optional(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 })),
			col_range: Type.Optional(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 })),
		}),
		async ({ row, col, camera: c = "head", row_range, col_range }) => {
			const map = await worldMap(c);
			const span = (r?: number[]) => (r && Math.max(...r) > Math.min(...r) ? r : undefined);
			const rows = span(row_range);
			const cols = span(col_range);
			if (rows || cols) {
				if (!rows || !cols) return toolResult({ error: "region mode needs both row_range and col_range" });
				const clip = (v: number, hi: number) => Math.max(0, Math.min(hi, v));
				const pts: number[][] = [];
				for (let r = clip(Math.min(...rows), map.height); r < clip(Math.max(...rows), map.height); r++)
					for (let cc = clip(Math.min(...cols), map.width); cc < clip(Math.max(...cols), map.width); cc++) {
						const i = (r * map.width + cc) * 3;
						const p = [map.xyz[i], map.xyz[i + 1], map.xyz[i + 2]];
						if (valid(p)) pts.push(p);
					}
				if (pts.length < 8)
					return toolResult({ error: `too few valid pixels in region (${pts.length}); widen the window` });
				const axis = (k: number) => pts.map((p) => p[k]);
				return toolResult({
					camera: c,
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
				return toolResult({ error: "give row and col, or row_range and col_range" });
			if (row < 0 || row >= map.height || col < 0 || col >= map.width)
				return toolResult({ error: `pixel (${row},${col}) out of bounds for ${map.width}x${map.height}` });
			const p = around(map, row, col, 1);
			if (!p) return toolResult({ error: `no depth at (${row},${col}); pick another pixel` });
			return toolResult({ camera: c, pixel: [row, col], world_xyz: p });
		},
	);

	async function startEpisode() {
		const { task, seed } = robot.task;
		if (!TASKS.includes(task as Task))
			throw new Error(`--task ${task} is not a BEHAVIOR-1K challenge task; one of: ${TASKS.join(", ")}`);
		if (!/^\d+$/.test(seed))
			throw new Error(`--seed must be a task instance id (a non-negative integer), got ${seed}`);
		sam3 = new RpcClient(flag("sam3", ""));
		molmo = new RpcClient(flag("molmo", ""));
		const endpoint = pi.getFlag("env") as string | undefined;
		if (endpoint) env = await attach(endpoint);
		else {
			const services = flag("services", SERVICES);
			env = await robot.serve({
				python: flag("python", "python"),
				args: [
					...["-m", "pi_embodied_services.robots.behavior.env_server"],
					...["--task", task, "--seed", seed, "--gpu-id", flag("gpu-id", "0")],
					...["--image-size", flag("image-size", "480"), "--grasping-mode", flag("grasping-mode", "sticky")],
				],
				cwd: services,
				env: { ...process.env, PYTHONPATH: services, OMNI_KIT_ACCEPT_EULA: "YES" },
				log: (port) => join(tmpdir(), `pi-embodied-behavior-${task}-s${seed}-${port}.log`),
				// Isaac Sim's start and a whole house of assets come before the server binds.
				readyMs: 1_800_000,
			});
		}
		meta = await env.call<Meta>("env.get_env_meta");
		if (meta.task !== task || meta.seed !== Number(seed))
			throw new Error(`env server runs ${meta.task} instance ${meta.seed}, not ${task} instance ${seed}`);
		// The server comes up reset; a new session on an attached server resets it again.
		const [o] = await env.call<[Obs, unknown]>("env.reset", {}, 600_000);
		absorb(o);
		return [
			"view_env_state",
			"get_robot_position",
			"navigate_to_pose",
			"move_hand",
			"grasp_object",
			"open_gripper",
			"close_gripper",
			"segment",
			"point",
			"back_project",
			"finish",
		];
	}
}
