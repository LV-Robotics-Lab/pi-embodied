/**
 * Robosuite robot for pi: CaP-X's seven robosuite 1.5 tasks (Lift, Stack, Restack, Wipe,
 * NutAssemblySquare, TwoArmLift, TwoArmHandover) with Pandas under the OSC_POSE controller.
 *
 *   pi -e packages/embodied/src/robosuite --task Lift --seed 0
 *   pi -e packages/embodied/src/robosuite --task TwoArmLift --seed 3 --units=true
 *   pi -e packages/embodied/src/robosuite --task Stack --seed 0 --privileged    (ground_truth_poses)
 *
 * Starts one env server per session (services/.../robots/robosuite/env_server.py, the `robosuite`
 * venv: robosuite 1.5 conflicts with LIBERO's 1.4) and attaches to a running SAM3 server for
 * `segment`; the server's primitive registry (code.api) is recorded per episode. Motion tools are closed-loop Cartesian servos the server bounds (per-call travel
 * cap, workspace box, z floor); every motion result carries the task camera and the wrist view
 * at 512 px and the arms' state. Success is robosuite's `_check_success` (Restack adds CaP-X's
 * off-table rule), latched at its first step and recorded in `robot_result`. The two-arm tasks
 * take an `arm` on every motion tool (robot0 | robot1), like dual_franka's left | right.
 */

import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";
import { decodePngChannel, encodePng } from "../png.ts";
import { attach, defineRobot, median, SERVICES, toolResult } from "../robot.ts";
import { NdArray, RpcClient } from "../rpc.ts";
import { finishMove, type Move, type MoveUnit, type Vec3 } from "../units/index.ts";

const SYSTEM = readFileSync(new URL("./SYSTEM.md", import.meta.url), "utf8");

/** The seven tasks (services/.../robots/robosuite/tasks.py TASKS), the `--task` values. */
export const TASKS = ["Lift", "Stack", "Restack", "Wipe", "NutAssemblySquare", "TwoArmLift", "TwoArmHandover"] as const;
export type Task = (typeof TASKS)[number];
export const ARMS = ["robot0", "robot1"] as const;
export type Arm = (typeof ARMS)[number];
/** Tasks with two Pandas: their motion tools take `arm`. */
export const TWO_ARM: readonly Task[] = ["TwoArmLift", "TwoArmHandover"];
/** Wipe's sponge gripper has no fingers: no gripper tool, no GRASP / RELEASE. */
export const NO_GRIPPER: readonly Task[] = ["Wipe"];
export const arms = (task: string): readonly Arm[] => (TWO_ARM.includes(task as Task) ? ARMS : ["robot0"]);
export const hasGripper = (task: string) => !NO_GRIPPER.includes(task as Task);

/**
 * World-frame vectors of the MV_* units (robosuite's world frame, +z up), chosen so each unit
 * matches its look in the task camera (VIEWS): MV_FWD (+x) moves toward the image bottom, MV_RIGHT
 * (+y) toward the image right. On the one-arm tasks robot0 stands at -x facing +x, so MV_FWD is its
 * forward and MV_RIGHT its left. On the two-arm "opposed" tasks robosuite turns robot0 by +90 deg
 * (at -y, facing +y) and robot1 by -90 deg (at +y, facing -y), so for robot0 MV_RIGHT is forward
 * (toward robot1) and MV_FWD moves sideways to its right; robot1 is the mirror.
 */
export const VECTORS: Record<MoveUnit, Vec3> = {
	MV_FWD: [1, 0, 0],
	MV_BACK: [-1, 0, 0],
	MV_LEFT: [0, -1, 0],
	MV_RIGHT: [0, 1, 0],
	MV_UP: [0, 0, 1],
	MV_DOWN: [0, 0, -1],
};
/** Metres per MV_* unit (the MVTOKEN 2 cm convention) and radians per ROTATE_* unit. */
export const STEP_M = 0.02;
export const YAW_STEP_RAD = 0.15;
/** Largest translation one call may command, m (the server refuses beyond its own --max-move too). */
export const MAX_MOVE_M = 0.3;
/** Fingers closed on nothing read about this width (sum of the two finger joints), m. */
export const EMPTY_WIDTH_M = 0.004;
export const IMAGE_SIZE = 512;

/**
 * How the views look. robosuite's robot0_robotview (the Panda's robot.xml: pos [1.0, 0, 0.4] from
 * the base, quat [0.653, 0.271, 0.271, 0.653]) stands 1 m in front of robot0, across the table,
 * looking back at it and down: image up is world -x (+z), image right is world +y. CaP-X's
 * overhead agentview of the two-arm tasks (tasks.OVERHEAD_CAMERA) has the same orientation from
 * [1.5, 0, 2.5], so the image axes are the same there; the robots, facing each other along y,
 * sit at the image left (robot0, -y) and right (robot1, +y).
 */
export const VIEWS = `Each result shows the task camera, then the wrist view (both 512x512).
- Task camera (first image): a fixed camera beyond the far edge of the table looking back at the robot(s) from above. In every task MV_FWD moves the gripper toward the image BOTTOM (toward the camera, larger), MV_BACK toward the image TOP, MV_LEFT toward the image left and MV_RIGHT toward the image right. On the one-arm tasks robot0's base is at the image top, so MV_BACK goes toward the base and MV_FWD away from it. On the two-arm tasks the robots face each other across the image: robot0 stands at the image LEFT and robot1 at the image RIGHT, so for robot0 MV_RIGHT moves away from its base (toward robot1) and MV_LEFT back toward it, while MV_FWD / MV_BACK move it sideways; for robot1 MV_LEFT moves away from its base and MV_RIGHT back toward it.
- Wrist view (second image): looks down from the gripper; the fingers are at the image sides and the grasp point is at the centre. Judge fine alignment here, gross layout in the task camera.`;

type Obs = Record<string, unknown> & { agentview: NdArray; wrist: NdArray };
type Motion = { obs: Obs; info: Record<string, unknown> & { frames?: NdArray[]; ok?: boolean; cancelled?: boolean } };
type CameraMeta = { intrinsic_K: number[][]; extrinsic_cam2world: number[][] };
type Meta = {
	task: string;
	seed: number;
	arms: string[];
	gripper: boolean;
	language: string;
	box: number[];
	z_floor: number;
	table_z: number;
	max_move_m: number;
};
type WorldMap = { envStep: number; size: number; rgb: Buffer; xyz: Float32Array };
type Camera = "agentview" | "wrist";

const round = (v: number, d = 4) => Number(v.toFixed(d));
const num = (v: unknown) => (v instanceof NdArray ? v.toArray() : Array.isArray(v) ? v.map(Number) : [Number(v)]);
const clip = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export default function robosuite(pi: ExtensionAPI) {
	const flag = (name: string, fallback: string) => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("task", {
		type: "string",
		default: "Lift",
		description: `Task: ${TASKS.join(" | ")}`,
	});
	pi.registerFlag("seed", { type: "string", default: "0", description: "Reset seed (the object layout)" });
	pi.registerFlag("sam3", {
		type: "string",
		default: "http://127.0.0.1:18300",
		description: "SAM3 server (segment, and code mode's segment primitive)",
	});
	pi.registerFlag("ik", {
		type: "string",
		description: "IK service URL (components/ik_server.py): the env server checks reach before every move",
	});
	pi.registerFlag("graspnet", {
		type: "string",
		default: "",
		description: "Contact-GraspNet server URL: adds the env server's plan_grasp primitives (code.api)",
	});
	pi.registerFlag("max-move", {
		type: "string",
		default: String(MAX_MOVE_M),
		description: "Largest translation one move_to / move_delta may command, m",
	});
	pi.registerFlag("cuda-device", {
		type: "string",
		description: "GPU for the env server's MuJoCo EGL rendering (physical CUDA ordinal)",
	});
	pi.registerFlag("env", { type: "string", description: "Attach to a running env server instead of starting one" });
	pi.registerFlag("services", {
		type: "string",
		default: process.env.PI_EMBODIED_SERVICES ?? SERVICES,
		description: "pi-embodied services dir",
	});
	pi.registerFlag("python", {
		type: "string",
		default: process.env.PI_EMBODIED_PYTHON ?? "python",
		description: "Python for the env server (the robosuite venv)",
	});

	let env: RpcClient;
	let sam3: RpcClient;
	let obs: Obs;
	let meta: Meta;
	let success = false;
	let successStep: number | null = null;
	let envStep = 0;
	let language = "";
	/** The gripper command units mode holds between units, per arm. */
	const grip = new Map<string, "open" | "close">();
	const worldMaps = new Map<string, WorldMap>();

	/** The task the flags name at load: units reads `arms` once, so the arm set is fixed here (startEpisode checks it). */
	const loadedTask = flag("task", "Lift");
	const twoArm = () => TWO_ARM.includes(robot.task.task as Task);
	const robot = defineRobot(pi, {
		name: "robosuite",
		task: ["task", "seed"],
		keepImages: 4,
		video: true,
		groundTruth: (names) => call("env.ground_truth_poses", { names: names ?? null }),
		// Observations carry the task camera, then the wrist view.
		vdm: { views: 2, wrist: 1 },
		start: startEpisode,
		prompt: () =>
			SYSTEM.replaceAll("{{task_language}}", language)
				.replaceAll("{{table_z}}", String(round(meta?.table_z ?? 0.8, 3)))
				.replaceAll(
					"{{arms}}",
					twoArm()
						? "Two Panda arms face each other across the table along y: robot0 stands at -y facing +y, robot1 at +y facing -y. Every motion tool takes `arm`."
						: "One Panda arm (robot0).",
				)
				.replaceAll("{{max_move}}", flag("max-move", String(MAX_MOVE_M))),
		result: () => ({
			task: robot.task.task,
			seed: Number(robot.task.seed),
			success,
			success_step: successStep,
			env_steps: envStep,
		}),
		status: () => ({ language, step: envStep, solved: success }),
		// The env server's primitive registry (code.api, robots/robosuite/primitives.py), recorded per episode.
		codeApi: () => env,
		units: {
			vectors: VECTORS,
			stepM: STEP_M,
			yawStepRad: YAW_STEP_RAD,
			arms: TWO_ARM.includes(loadedTask as Task) ? ARMS : undefined,
			maxMoveM: () => Number(flag("max-move", String(MAX_MOVE_M))),
			apply: (move, signal) => unitStep(move, signal),
			state: async (arm) => {
				const a = arm ?? "robot0";
				return {
					eef_xyz: num(obs[`${a}_eef_pos`]).map((v) => round(v)),
					...(hasGripper(robot.task.task) ? { gripper_width: round(Number(obs[`${a}_gripper_width`])) } : {}),
					table_z: round(meta.table_z, 3),
				};
			},
			instruction: () => language,
			views: VIEWS,
			emptyWidthM: EMPTY_WIDTH_M,
		},
		finish: {
			description:
				"End the episode after checking the latest state. Success is robosuite's own check, not this call.",
			parameters: Type.Object({
				status: StringEnum(["success", "failure"] as const),
				summary: Type.String(),
			}),
			result: (params) => ({
				content: [{ type: "text", text: `Episode finished (success=${success}).` }],
				details: params,
			}),
		},
	});
	const { video } = robot;

	/** Every robot RPC carries the running tool's abort signal, so an abort stops motion between calls. */
	const call = <T = unknown>(
		method: string,
		kwargs: Record<string, unknown> = {},
		args: unknown[] = [],
		signal = robot.signal,
		timeoutMs = 300_000,
	) => env.call<T>(method, kwargs, timeoutMs, args, signal);

	function absorb(o: Obs) {
		obs = o;
		success = Boolean(o.success);
		successStep = o.success_step === null || o.success_step === undefined ? null : Number(o.success_step);
		envStep = Number(o.env_steps);
		worldMaps.clear();
	}

	/** A motion call's result: its video frames go to the episode video, the observation is absorbed. */
	function motion(r: Motion) {
		for (const f of r.info.frames ?? []) video.frame(f);
		const { frames: _frames, ...info } = r.info;
		absorb(r.obs);
		return info;
	}

	/** The arm a tool call means: `arm` on the two-arm tasks (required there), robot0 otherwise. */
	function armOf(arm: string | undefined): string | undefined {
		if (!twoArm()) {
			if (arm !== undefined && arm !== "robot0") throw new Error(`${robot.task.task} has one arm (robot0)`);
			return undefined;
		}
		if (!arm) throw new Error(`${robot.task.task} has two arms: pass arm robot0 or robot1`);
		return arm;
	}

	/** The motion result with the new state, then the task camera and wrist images. */
	function observe(result: Record<string, unknown>) {
		const images = [obs.agentview, obs.wrist].map((a) => encodePng(a.data, a.shape[1], a.shape[0]));
		const state: Record<string, unknown> = {};
		for (const a of arms(robot.task.task)) {
			state[`${a}_eef_pos`] = num(obs[`${a}_eef_pos`]).map((v) => round(v));
			state[`${a}_eef_quat_xyzw`] = num(obs[`${a}_eef_quat`]).map((v) => round(v));
			if (hasGripper(robot.task.task)) {
				state[`${a}_gripper_width`] = round(Number(obs[`${a}_gripper_width`]));
				state[`${a}_gripper_command`] = obs[`${a}_gripper_command`];
			}
		}
		const details = {
			result,
			step: envStep,
			success,
			success_step: successStep,
			task_language: language,
			state,
			images: [`agentview ${IMAGE_SIZE}x${IMAGE_SIZE}`, `wrist ${IMAGE_SIZE}x${IMAGE_SIZE}`],
		};
		return {
			content: [
				{ type: "text" as const, text: JSON.stringify(details) },
				...images.map((png) => ({ type: "image" as const, data: png.toString("base64"), mimeType: "image/png" })),
			],
			details,
		};
	}

	/** Register a tool; motion tools return a fresh observation, read-only tools their result (+ an optional image). */
	function tool<P extends TSchema>(
		name: string,
		description: string,
		parameters: P,
		run: (p: Static<P>, signal: AbortSignal | undefined) => Promise<Record<string, unknown>>,
		kind: "motion" | "read" = "motion",
	) {
		robot.tool(name, description, parameters, async (params, signal) => {
			const result = await run(params, signal);
			if (kind === "motion") return observe(result);
			const { _image, ...rest } = result as { _image?: Buffer };
			return toolResult(rest, _image ? [_image] : []);
		});
	}

	const xyz = Type.Array(Type.Number(), { minItems: 3, maxItems: 3, description: "World-frame [x, y, z] in metres" });
	const arm = Type.Optional(StringEnum(ARMS, { description: "Two-arm tasks: which arm (required there)" }));
	const gripper = Type.Optional(
		StringEnum(["open", "close"] as const, { description: "Set the held gripper command first" }),
	);
	const camera = Type.Optional(StringEnum(["agentview", "wrist"] as const, { description: "Default agentview" }));
	const int = (description: string) => Type.Optional(Type.Integer({ description }));
	const opt = (description: string) => Type.Optional(Type.Number({ description }));

	tool(
		"view_env_state",
		`Current state with the task camera (global layout) and wrist (close range) ${IMAGE_SIZE}x${IMAGE_SIZE} images. Pixels (row, col) in these images feed back_project.`,
		Type.Object({}),
		async () => ({}),
	);

	tool(
		"move_to",
		`Closed-loop servo of the TCP to a world xyz, holding the orientation (or turning by rotvec, a world-frame axis-angle in rad). The server refuses a target more than ${MAX_MOVE_M} m away, outside the table workspace or below the z floor; split long moves. gripper sets the held command first.`,
		Type.Object({
			xyz,
			arm,
			gripper,
			rotvec: Type.Optional(
				Type.Array(Type.Number(), { minItems: 3, maxItems: 3, description: "World-frame turn, axis x angle, rad" }),
			),
			tol: opt("Position tolerance, m (default 0.005)"),
			max_steps: int("Control-step budget (default 100)"),
		}),
		async ({ xyz: target, arm: a, gripper: g, rotvec, tol, max_steps }, signal) => {
			checkMove(target, num(obs[`${armOf(a) ?? "robot0"}_eef_pos`]));
			return {
				name: "move_to",
				...motion(
					await call<Motion>(
						"env.move_to",
						{
							arm: armOf(a) ?? null,
							...(g ? { gripper: g } : {}),
							...(rotvec ? { rotvec } : {}),
							...(tol !== undefined ? { tol_m: tol } : {}),
							...(max_steps !== undefined ? { max_steps } : {}),
						},
						[target],
						signal,
					),
				),
			};
		},
	);

	tool(
		"move_delta",
		`Translate the TCP by a world-frame [dx, dy, dz] in metres (+z up; in the task camera +x runs toward the image bottom and +y toward the image right: on the one-arm tasks +x is away from robot0 and +y to its left, on the two-arm tasks +y runs from robot0 toward robot1; at most ${MAX_MOVE_M} m per call), holding the orientation. gripper sets the held command first.`,
		Type.Object({ delta_xyz: xyz, arm, gripper }),
		async ({ delta_xyz, arm: a, gripper: g }, signal) => {
			checkMove(delta_xyz, [0, 0, 0]);
			return {
				name: "move_delta",
				...motion(
					await call<Motion>(
						"env.move_delta",
						{ arm: armOf(a) ?? null, ...(g ? { gripper: g } : {}) },
						[delta_xyz],
						signal,
					),
				),
			};
		},
	);

	tool(
		"gripper",
		"Hold the arm and open or close its gripper (up to `steps` control steps; the fingers stop on a grasped object). The command stays in force for later moves: carry with close.",
		Type.Object({
			command: StringEnum(["open", "close"] as const),
			arm,
			steps: int("Default 15"),
		}),
		async ({ command, arm: a, steps }, signal) => {
			if (!hasGripper(robot.task.task)) throw new Error(`${robot.task.task}'s wiping gripper has no fingers`);
			return {
				name: "gripper",
				...motion(
					await call<Motion>(
						"env.set_gripper",
						{ arm: armOf(a) ?? null, ...(steps !== undefined ? { steps } : {}) },
						[command],
						signal,
					),
				),
			};
		},
	);

	/** Refuse a translation larger than --max-move before asking the server (which checks its own cap too). */
	function checkMove(target: number[], from: number[]) {
		const cap = Number(flag("max-move", String(MAX_MOVE_M)));
		const norm = Math.hypot(...target.map((v, k) => v - from[k]));
		if (!(norm <= cap))
			throw new Error(`the move travels ${round(norm)} m; the limit is ${cap} m per call. Split the motion.`);
	}

	async function render(camera: Camera, size: number, depth: boolean) {
		const out = await call<NdArray | [NdArray, NdArray]>("env.render_camera", {
			camera_name: camera,
			height: size,
			width: size,
			depth,
		});
		const [rgb, d] = out instanceof NdArray ? [out, null] : out;
		return { rgb: Buffer.from(rgb.data), depth: d };
	}

	/** Per-pixel world xyz for the current step, from metric depth and robosuite's calibration (the world map). */
	async function worldMap(camera: Camera, size: number): Promise<WorldMap> {
		const key = `${camera}:${size}`;
		const cached = worldMaps.get(key);
		if (cached?.envStep === envStep) return cached;
		const { rgb, depth } = await render(camera, size, true);
		const cam = await call<CameraMeta>("env.get_camera_meta", { camera_name: camera, height: size, width: size });
		const raw = (depth as NdArray).toArray();
		const [[fx, , cx], [, fy, cy]] = cam.intrinsic_K;
		const e = cam.extrinsic_cam2world;
		const xyz = new Float32Array(size * size * 3);
		for (let r = 0; r < size; r++)
			for (let c = 0; c < size; c++) {
				const z = raw[r * size + c];
				const x = ((c - cx) * z) / fx;
				const y = ((r - cy) * z) / fy;
				const i = (r * size + c) * 3;
				for (let k = 0; k < 3; k++) xyz[i + k] = e[k][0] * x + e[k][1] * y + e[k][2] * z + e[k][3];
			}
		const map = { envStep, size, rgb, xyz };
		worldMaps.set(key, map);
		return map;
	}
	const valid = (p: number[]) => p.every(Number.isFinite) && Math.abs(p[0]) + Math.abs(p[1]) + Math.abs(p[2]) > 1e-6;

	tool(
		"view_camera_meta",
		"Camera calibration (intrinsic K and cam-to-world extrinsic for the 512x512 image; depth is metric) for the current step.",
		Type.Object({ camera }),
		async ({ camera: c = "agentview" }) => ({
			camera: c,
			meta: await call("env.get_camera_meta", { camera_name: c, height: IMAGE_SIZE, width: IMAGE_SIZE }),
		}),
		"read",
	);

	tool(
		"segment",
		`SAM3 segmentation of the current ${IMAGE_SIZE}x${IMAGE_SIZE} camera image. Give exactly one of a text prompt or a positive point [row, col]. The top mask is projected through the depth world map; world_xyz is the median over mask pixels. Returns an overlay image.`,
		Type.Object({
			prompt: Type.Optional(Type.String()),
			point: Type.Optional(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 })),
			camera,
			min_score: opt("Default 0.2"),
		}),
		async ({ prompt, point, camera: c = "agentview", min_score = 0.2 }) => {
			const text = prompt?.trim();
			if (!text && !point) return { error: "give a text prompt or a point [row, col]" };
			const size = IMAGE_SIZE;
			const map = await worldMap(c, size);
			const png = encodePng(map.rgb, size, size);
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
				return {
					found: false,
					error: res.reason ?? "no mask",
					fallback: "Pick pixels in the image and use back_project.",
				};
			const mask = decodePngChannel(Buffer.from(res.mask_png_base64, "base64"));
			if (mask.width !== size || mask.height !== size)
				return { found: true, error: `mask ${mask.width}x${mask.height} does not match the ${size} world map` };
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
			return {
				found: true,
				camera: c,
				score: res.score === undefined ? null : round(res.score, 3),
				box: res.box,
				n_pixels: xs.length,
				n_valid: pts.length,
				centroid_pixel: [Math.round(median(ys)), Math.round(median(xs))],
				world_xyz: pts.length < 10 ? null : [0, 1, 2].map((k) => round(median(pts.map((p) => p[k])))),
				...(pts.length < 10 ? { world_error: `too few valid depth pixels (${pts.length})` } : {}),
				_image: encodePng(overlay, size, size),
			};
		},
		"read",
	);

	tool(
		"back_project",
		`World xyz of a pixel (row, col; row 0 = top) in the current ${IMAGE_SIZE}x${IMAGE_SIZE} camera image, from the depth world map. Region mode: row_range + col_range (+ optional z_min/z_max) returns the midpoint of world xy over that window and the median z.`,
		Type.Object({
			row: int("Pixel row"),
			col: int("Pixel column"),
			camera,
			row_range: Type.Optional(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 })),
			col_range: Type.Optional(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 })),
			z_min: opt("Region mode: keep pixels with world z >= z_min"),
			z_max: opt("Region mode: keep pixels with world z <= z_max"),
		}),
		async ({ row, col, camera: c = "agentview", row_range, col_range, z_min, z_max }) => {
			const size = IMAGE_SIZE;
			const map = await worldMap(c, size);
			const at = (r: number, cc: number) => {
				const i = (r * size + cc) * 3;
				return [map.xyz[i], map.xyz[i + 1], map.xyz[i + 2]];
			};
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
			return { camera: c, pixel: [row, col], world_xyz: p.map((v) => round(v)) };
		},
		"read",
	);

	/**
	 * One action unit (../units): drive the gripper, servo the TCP by `delta` (holding the gripper
	 * command), turn by `yaw` about world +z or by an RT_* `rot`, or hold one step (STOP).
	 */
	async function unitStep(move: Move, signal: AbortSignal | undefined) {
		const a = twoArm() ? move.arm : undefined;
		if (twoArm() && !a) throw new Error("two arms: act needs `arm`");
		const name = a ?? "robot0";
		// After success only the finish sequence moves (opening, lifting straight up); success stays latched.
		if (success && !finishMove(move))
			return {
				content: [{ type: "text" as const, text: `Episode already ended (success=${success}).` }],
				details: { success },
			};
		let steps = 0;
		let info: Record<string, unknown> = {};
		if (move.gripper) {
			if (!hasGripper(robot.task.task)) throw new Error(`${robot.task.task}'s wiping gripper has no fingers`);
			grip.set(name, move.gripper);
			info = motion(await call<Motion>("env.set_gripper", { arm: a ?? null }, [move.gripper], signal));
			steps += Number(info.steps_used ?? 0);
		}
		const turning = Boolean(move.yaw) || Boolean(move.rot?.some(Boolean));
		if (Math.hypot(...move.delta) > 0 || turning) {
			const rotvec = move.rot?.some(Boolean) ? move.rot : move.yaw ? [0, 0, move.yaw] : undefined;
			info = motion(
				await call<Motion>(
					"env.move_delta",
					{ arm: a ?? null, ...(rotvec ? { rotvec } : {}), tol_m: 0.004, max_steps: 40 },
					[move.delta],
					signal,
				),
			);
			steps += Number(info.steps_used ?? 0);
		}
		if (!move.gripper && !turning && !Math.hypot(...move.delta)) {
			// STOP: hold the setpoint for one step (a zero move).
			info = motion(await call<Motion>("env.move_delta", { arm: a ?? null, max_steps: 1 }, [[0, 0, 0]], signal));
			steps += 1;
		}
		return observe({
			name: "act",
			arm: name,
			steps_used: steps,
			...(info.ok !== undefined ? { ok: info.ok } : {}),
			eef_pos: num(obs[`${name}_eef_pos`]).map((v) => round(v)),
			...(hasGripper(robot.task.task) ? { gripper_width: round(Number(obs[`${name}_gripper_width`])) } : {}),
		});
	}

	/** Start (or attach to) the env server and reset the scene; returns the tools to activate. */
	async function startEpisode() {
		const { task, seed } = robot.task;
		if (!TASKS.includes(task as Task)) throw new Error(`unknown --task ${task}: ${TASKS.join(", ")}`);
		if (TWO_ARM.includes(task as Task) !== TWO_ARM.includes(loadedTask as Task))
			throw new Error(
				`${task} has ${arms(task).length} arm(s) but pi was started for ${loadedTask}; restart with --task ${task}`,
			);
		sam3 = new RpcClient(flag("sam3", ""));
		const endpoint = pi.getFlag("env") as string | undefined;
		if (endpoint) env = await attach(endpoint);
		else {
			const services = flag("services", SERVICES);
			const cuda = pi.getFlag("cuda-device") as string | undefined;
			env = await robot.serve({
				python: flag("python", "python"),
				args: [
					"-m",
					"pi_embodied_services.robots.robosuite.env_server",
					...["--task", task, "--seed", seed, "--max-move", flag("max-move", String(MAX_MOVE_M))],
					...["--sam3", flag("sam3", "")],
					...(cuda ? ["--cuda-device", cuda] : []),
					...(pi.getFlag("ik") ? ["--ik", flag("ik", "")] : []),
					...(flag("graspnet", "") ? ["--graspnet", flag("graspnet", "")] : []),
				],
				cwd: services,
				env: { ...process.env, PYTHONPATH: services, MUJOCO_GL: "egl" },
				log: (port) => join(tmpdir(), `pi-embodied-robosuite-${task}-s${seed}-${port}.log`),
				readyMs: 600_000,
			});
		}
		meta = await env.call<Meta>("env.get_env_meta");
		if (meta.task !== task || meta.seed !== Number(seed))
			throw new Error(`env server runs ${meta.task} seed ${meta.seed}, not ${task} seed ${seed}`);
		grip.clear();
		const [o] = await env.call<[Obs, unknown]>("env.reset", {}, 600_000);
		absorb(o);
		language = await env.call<string>("env.get_task_language");
		return [
			"view_env_state",
			"view_camera_meta",
			"segment",
			"back_project",
			"move_to",
			"move_delta",
			...(hasGripper(task) ? ["gripper"] : []),
			"finish",
		];
	}
}
