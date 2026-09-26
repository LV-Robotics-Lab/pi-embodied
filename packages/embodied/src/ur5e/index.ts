/**
 * One physical Universal Robots UR5e arm for pi.
 *
 *   pi -e packages/embodied/src/ur5e --operator --arm-id 2023300001 --task block_bowl --robot-config my_ur5e.yaml
 *
 * Starts the env server (pi_embodied_services.robots.ur5e.env_server: ur_rtde to the controller, a
 * Robotiq gripper over the URCap socket, RealSense / webcam / RTSP cameras through the services'
 * shared camera layer) or attaches to one with --robot-env. The server owns the safety limits from
 * the robot YAML: per-call translation and rotation refusal, the workspace box and Z floor, the tool
 * tilt limit, `stop` that really stops the running moveL/moveJ (stopL/stopJ), the setpoint cleared
 * after a stop or failure, and the gripper's jammed / empty-grasp detection. The client refuses the
 * same per-call caps (the tighter of --max-move / --max-rotate and the server's) before any call.
 *
 * A real robot needs an operator: pi must have a UI, --operator must be on (the base then asks for a
 * verdict before `finish`), and the operator confirms the reset before any motion. The robot is bound
 * to one arm: --arm-id names the arm the operator intends to drive and must equal the identity the
 * server bound its config (limits, begin pose, camera calibrations) to; a mismatch refuses to start.
 * Motion tools record a state step (robot state, every camera's RGB and, where the camera has it,
 * depth, camera metadata) under --out and return it with the images. back_project reads a pixel's
 * depth through the camera's intrinsics and its hand-eye calibration (robots/ur5e/calibrate.py); an
 * RGB-only camera has no depth to project. segment is active only with --robot-sam3.
 */

import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";
import { decodePngChannel, encodePng } from "../png.ts";
import {
	apply,
	attach,
	checkMove,
	defineRobot,
	frameOf,
	gridOf,
	inv3,
	type Json,
	type Mat,
	mark,
	median,
	message,
	plain,
	pose7,
	rgbOf,
	round,
	roundAll,
	SERVICES,
	type Services,
	servicesEnv,
	toolResult,
	vec,
} from "../robot.ts";
import { NdArray, type RpcClient } from "../rpc.ts";
import type { Move, MoveUnit, Vec3 } from "../units/index.ts";

const SYSTEM = readFileSync(new URL("./SYSTEM.md", import.meta.url), "utf8");

type Task = { instruction: string; success_criteria?: string };
type Meta = {
	robot: string;
	backend: string;
	arm_id: string | null;
	cameras: string[];
	main_camera: string;
	gripper: string;
	has_begin_pose: boolean;
	limits: { max_move_m: number; max_rotate_rad: number; z_floor_m: number | null; empty_width_m: number | null };
	tasks: Record<string, Task>;
};
type CameraMeta = {
	name: string;
	has_depth: boolean;
	intrinsic_K: Mat | null;
	extrinsic: { frame: "tcp" | "base"; matrix: Mat; path: string; arm_id: string | null } | null;
	[k: string]: unknown;
};
type Step = { blob: Json; dir: string; meta: Json | null; images: Record<string, string> };

/**
 * The action-unit grounding for a UR5e on its standard mounting: base +x away from the robot,
 * +y to its left (right-handed, so MV_LEFT is +y), +z up; 2 cm per unit, 0.15 rad per ROTATE_*.
 */
export const UR5E_UNITS = {
	vectors: {
		MV_FWD: [1, 0, 0],
		MV_BACK: [-1, 0, 0],
		MV_LEFT: [0, 1, 0],
		MV_RIGHT: [0, -1, 0],
		MV_UP: [0, 0, 1],
		MV_DOWN: [0, 0, -1],
	} as Record<MoveUnit, Vec3>,
	stepM: 0.02,
	yawStepRad: 0.15,
};

/** The robot's own tools; `segment` is activated only with --robot-sam3. */
const TOOLS = [
	"view_env_state",
	"view_camera_meta",
	"back_project",
	"move_delta",
	"move_pose",
	"rotate_delta",
	"gripper",
	"finish",
];

function vec3(v: unknown, name: string): number[] {
	const a = vec(v);
	if (a.length !== 3 || !a.every(Number.isFinite)) throw new Error(`${name} must contain exactly 3 finite values`);
	return a;
}

export default function ur5e(pi: ExtensionAPI) {
	const flag = (name: string, fallback = "") => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("task", {
		type: "string",
		default: "smoke",
		description: "Task name from the robot YAML's `tasks:`",
	});
	pi.registerFlag("arm-id", {
		type: "string",
		default: "",
		description:
			"The arm this session drives (its controller serial; env_server --print-identity). Required; must match the arm the server's config is bound to",
	});
	pi.registerFlag("robot-config", {
		type: "string",
		description: "Robot YAML (default: services/pi_embodied_services/robots/ur5e/config/example.yaml)",
	});
	pi.registerFlag("robot-env", {
		type: "string",
		description: "Attach to a running UR5e env server instead of starting one",
	});
	pi.registerFlag("robot-cameras", {
		type: "string",
		default: "",
		description:
			"Camera sources for the env server, overriding the YAML's cameras.devices: name=type:source,... (realsense:<serial>, webcam:<index|/dev/videoN>, rtsp://<url>); the first is the main camera",
	});
	pi.registerFlag("robot-sam3", {
		type: "string",
		default: "",
		description: "SAM3 server (attach-only; enables segment)",
	});
	pi.registerFlag("services", {
		type: "string",
		default: process.env.PI_EMBODIED_SERVICES ?? SERVICES,
		description: "pi-embodied services dir",
	});
	pi.registerFlag("python", {
		type: "string",
		default: process.env.PI_EMBODIED_PYTHON ?? "python",
		description: "Python with the services' [ur5e] extra",
	});
	pi.registerFlag("out", {
		type: "string",
		description: "Step artifact directory (default: a new directory under the OS temp dir)",
	});
	pi.registerFlag("max-move", {
		type: "string",
		default: "0.08",
		description: "Largest translation per call, m (the server's limits.max_move_m applies if tighter)",
	});
	pi.registerFlag("max-rotate", {
		type: "string",
		default: "0.2",
		description: "Largest rotation per call, rad (the server's limits.max_rotate_rad applies if tighter)",
	});

	let env: RpcClient | undefined;
	let sam3: RpcClient | undefined;
	let meta: Meta | undefined;
	let task: Task | undefined;
	let out = "";
	const steps: Step[] = [];
	const taskName = () => robot.task.task;
	const armId = () => flag("arm-id").trim();

	const robot = defineRobot(pi, {
		name: "ur5e",
		task: ["task"],
		keepImages: 4,
		operator: { step: () => steps.length, reset: resetArm },
		video: true,
		// One image per camera, main first; the wrist-mounted cameras are the VDM's wrist views.
		vdm: () => {
			const c = cameras();
			if (!c.length) return undefined;
			const wrist = c.flatMap((name, i) => (mountOf(name) === "wrist" ? [i] : []));
			return { views: c.length, ...(wrist.length ? { wrist } : {}) };
		},
		// The env server's primitive registry (services robots/ur5e/primitives.py).
		codeApi: () => env,
		start: startRobot,
		stop: () => {
			env = sam3 = meta = task = undefined;
		},
		prompt: () => {
			if (!task || !meta) return undefined;
			const vars: Record<string, string> = {
				task_name: taskName(),
				instruction: task.instruction,
				success_criteria: task.success_criteria ?? "Judged by the operator.",
				max_move: String(maxMove()),
				max_rotate: String(maxRotate()),
				arm_id: meta.arm_id ?? "unbound",
				cameras: cameras()
					.map((c, i) => `Image ${i + 1}: ${c}${c === meta?.main_camera ? " (main)" : ""}`)
					.join("; "),
			};
			return SYSTEM.replace(/\{\{(\w+)\}\}/g, (m, k: string) => vars[k] ?? m);
		},
		result: () => ({
			task: taskName(),
			arm_id: meta?.arm_id ?? null,
			backend: meta?.backend ?? null,
			steps: steps.length,
			out,
		}),
		status: () => ({ language: task?.instruction, step: steps.length - 1 }),
		units: {
			...UR5E_UNITS,
			maxYawRad: () => maxRotate(),
			maxMoveM: () => maxMove(),
			apply: (move, signal) => unitStep(move, signal),
			state: unitState,
			instruction: () => task?.instruction ?? "",
			get emptyWidthM() {
				return meta?.limits.empty_width_m ?? 0.005;
			},
		},
		finish: {
			description:
				"Call when the task is complete or unrecoverable. Halts the agent loop. With --operator, the operator gives a verdict first.",
			parameters: Type.Object({
				status: Type.String({ description: "Outcome, e.g. 'success', 'failure', or 'stuck'." }),
				summary: Type.String({ description: "Short natural-language summary of the run." }),
			}),
			result: (params) => toolResult({ _finish: true, ...params }),
		},
	});
	const { op } = robot;

	const maxMove = () => Math.min(Number(flag("max-move", "0.08")), meta?.limits.max_move_m ?? Infinity);
	const maxRotate = () => Math.min(Number(flag("max-rotate", "0.2")), meta?.limits.max_rotate_rad ?? Infinity);
	/** Main camera first, then the others in the server's order. */
	const cameras = () => {
		const names = meta?.cameras ?? [];
		const main = meta?.main_camera;
		return main && names.includes(main) ? [main, ...names.filter((n) => n !== main)] : names;
	};
	/** A camera's mount (`wrist` or `fixed`) from the latest recorded camera metadata. */
	const mountOf = (name: string) => steps[steps.length - 1]?.meta?.cameras?.[name]?.mount ?? null;

	function call<T = Json>(method: string, kwargs: Json = {}, timeoutMs = 30_000, signal?: AbortSignal) {
		if (!env) throw new Error("ur5e is not initialized; see the session start error");
		return env.call<T>(method, kwargs, timeoutMs, [], signal);
	}
	const check = (signal?: AbortSignal) => {
		op.check();
		if (signal?.aborted) throw new Error("tool operation interrupted");
	};

	// ---- state steps

	/** Record the current observation as step N (PNG and depth per camera, camera_meta.json, states.jsonl). */
	async function record(command: Json | null, result: Json | null, elapsed: number | null): Promise<Step> {
		const obs = await call<Json>("env.get_observation");
		const state = await call<Json>("env.get_robot_state");
		const camMeta = await call<Json | null>("env.get_camera_meta");
		const idx = steps.length;
		const dir = join(out, `step_${String(idx).padStart(4, "0")}`);
		mkdirSync(dir, { recursive: true });
		const images: Record<string, string> = {};
		const artifacts: string[] = [];
		let framed = false;
		for (const name of cameras()) {
			const rgb = obs.images?.[name];
			if (rgb instanceof NdArray) {
				// The episode video follows the main camera.
				if (!framed) robot.video.frame(frameOf(rgb));
				framed = true;
				const img = rgbOf(rgb);
				writeFileSync(join(dir, `${name}.png`), encodePng(img.rgb, img.width, img.height));
				writeFileSync(join(dir, `${name}.json`), JSON.stringify({ width: img.width, height: img.height }));
				writeFileSync(join(dir, `${name}.rgb`), img.rgb);
				images[name] = join(dir, `${name}.png`);
				artifacts.push(`${name}.png`);
			}
			const depth = obs.depths?.[name];
			if (depth instanceof NdArray) {
				const g = gridOf(depth);
				writeFileSync(join(dir, `${name}_depth.f32`), Buffer.from(g.data.buffer));
				writeFileSync(join(dir, `${name}_depth.json`), JSON.stringify({ height: g.height, width: g.width }));
				artifacts.push(`${name}_depth.f32`);
			}
		}
		if (camMeta) {
			writeFileSync(join(dir, "camera_meta.json"), JSON.stringify(plain(camMeta)));
			artifacts.push("camera_meta.json");
		}
		const blob: Json = { step_idx: idx, state: plain(state), timestamps: plain(obs.timestamps), artifacts, images };
		if (command) blob.command = command;
		if (result) blob.result = plain(result);
		if (elapsed !== null) blob.elapsed_s = elapsed;
		appendFileSync(join(out, "states.jsonl"), `${JSON.stringify(blob)}\n`);
		const step = { blob, dir, meta: camMeta ? (plain(camMeta) as Json) : null, images };
		steps.push(step);
		return step;
	}

	function getStep(step?: number | null): Step {
		const i = step === undefined || step === null ? steps.length - 1 : step < 0 ? steps.length + step : step;
		const s = steps[i];
		if (!s) throw new Error(`step ${step} is not recorded (have 0..${steps.length - 1})`);
		return s;
	}

	/** The step's images, main camera first. */
	const stepImages = (s: Step) =>
		cameras()
			.filter((k) => s.images[k])
			.map((k) => readFileSync(s.images[k]));

	/**
	 * Register a tool. Mutating tools run, then record a fresh state step and return it (errors
	 * included); read-only tools return their result or `{error}`, plus any PNGs in `_pngs`.
	 */
	function tool<P extends TSchema>(
		name: string,
		description: string,
		parameters: P,
		run: (p: Static<P>, signal: AbortSignal | undefined) => Promise<Json>,
		mutating = true,
	) {
		robot.tool(name, description, parameters, (params, signal) =>
			outcome(name, params as Json, () => run(params, signal), mutating),
		);
	}

	async function outcome(name: string, params: Json, run: () => Promise<Json>, mutating: boolean) {
		const started = performance.now();
		let result: Json;
		let failed = false;
		try {
			result = await run();
		} catch (err) {
			result = { error: message(err) };
			failed = true;
		}
		if (!mutating) {
			const { _pngs, ...rest } = result;
			return toolResult(rest, _pngs ?? []);
		}
		if (failed && !env) return toolResult({ ...result, command: { action: name, ...params } });
		const elapsed = round((performance.now() - started) / 1000, 2);
		try {
			const s = await record({ action: name, ...params }, result, elapsed);
			const output: Json = { ...s.blob, agent_elapsed_s: elapsed };
			if (failed) for (const [k, v] of Object.entries(result)) output[k] ??= v;
			// A jammed gripper (the fingers did not move as commanded, nothing grasped) heads the result.
			const jam = [result, result.gripper].find((r) => r?.gripper_jammed === true);
			const pngs = stepImages(s);
			if (jam) return toolResult({ gripper_jammed: true, gripper_note: jam.note, ...output }, pngs);
			return toolResult(output, pngs);
		} catch (err) {
			return toolResult({
				...result,
				state_capture_error: message(err),
				error: result.error ?? `failed to capture state after ${name}: ${message(err)}`,
			});
		}
	}

	// ---- motion

	function checkRotate(norm: number) {
		const limit = maxRotate();
		if (!(norm <= limit))
			throw new Error(
				`the rotation is ${round(norm, 4)} rad; the limit is ${limit} rad per call. Split it into smaller calls.`,
			);
	}

	const motion = (method: string, kwargs: Json, signal?: AbortSignal) => call(method, kwargs, 120_000, signal);

	/** Units mode (../units): one grounded action unit on the move primitives. */
	function unitStep(move: Move, signal: AbortSignal | undefined) {
		return outcome(
			"act",
			{ move },
			async () => {
				check(signal);
				const out: Json = {};
				if (move.gripper) out.gripper = await motion("env.set_gripper", { open: move.gripper === "open" }, signal);
				if (Math.hypot(...move.delta) > 0) {
					checkMove(move.delta, maxMove());
					out.move = await motion("env.move_delta", { delta_xyz: move.delta }, signal);
				}
				if (move.yaw) {
					checkRotate(Math.abs(move.yaw));
					out.rotate = await motion("env.rotate_delta", { delta_rpy: [0, 0, move.yaw] }, signal);
				}
				return out;
			},
			true,
		);
	}

	/** Proprioception for units mode: TCP position and gripper width from the latest state step. */
	async function unitState(): Promise<Json> {
		const base = steps[steps.length - 1]?.blob.state?.raw_base_state ?? {};
		const width = vec(base.gripper_position);
		return {
			eef_xyz: roundAll(vec(base.tcp_pose).slice(0, 3)),
			...(width.length ? { gripper_width: round(width[0]) } : {}),
			gripper_open: base.gripper_open ?? null,
			gripper_grasped: base.gripper_grasped ?? null,
			...(typeof meta?.limits.z_floor_m === "number" ? { table_z: meta.limits.z_floor_m } : {}),
		};
	}

	const stepParam = Type.Optional(Type.Integer({ description: "State step (default -1 = latest)" }));
	const xyz = Type.Array(Type.Number(), { minItems: 3, maxItems: 3 });
	const cameraParam = Type.Optional(Type.String({ description: "Camera name (default: the main camera)" }));

	tool(
		"view_env_state",
		"Read a UR5e state step (TCP pose, joints, gripper, setpoint) and its camera images, main camera first.",
		Type.Object({ step: stepParam }),
		async ({ step = -1 }) => {
			const s = getStep(step);
			return { ...s.blob, _pngs: stepImages(s) };
		},
		false,
	);

	tool(
		"view_camera_meta",
		"Read every camera's intrinsics, depth availability, mount and hand-eye calibration.",
		Type.Object({ step: stepParam }),
		async ({ step = -1 }) => {
			const s = getStep(step);
			if (!s.meta) return { error: "camera metadata is unavailable", step };
			return { step: s.blob.step_idx, camera_meta: s.meta };
		},
		false,
	);

	function cameraOf(s: Step, name: string | undefined): CameraMeta {
		const n = name?.trim() || meta?.main_camera || "";
		const cam = s.meta?.cameras?.[n];
		if (!cam) throw new Error(`unknown camera '${n}' (have ${cameras().join(", ")})`);
		return cam as CameraMeta;
	}

	function tcpPose(s: Step): number[] {
		const pose = s.blob.state?.raw_base_state?.tcp_pose;
		if (!pose) throw new Error("recorded state is missing raw_base_state.tcp_pose");
		return vec(pose);
	}

	/** A pixel's base-frame point through the step's depth, the camera's K and its hand-eye calibration. */
	function project(s: Step, cam: CameraMeta, row: number, col: number): Json {
		if (!cam.has_depth)
			throw new Error(
				`camera ${cam.name} has no depth (RGB-only source); pick a camera with depth or enhance its depth first`,
			);
		const K = cam.intrinsic_K;
		if (!K || K.length !== 3 || !K.flat().every(Number.isFinite))
			throw new Error(`camera ${cam.name} reports no intrinsics (set cameras.devices.${cam.name}.intrinsics)`);
		if (!cam.extrinsic)
			throw new Error(
				`camera ${cam.name} has no hand-eye calibration: run robots/ur5e/calibrate.py (capture, solve, apply) for it`,
			);
		let shape: { height: number; width: number };
		let depth: Float32Array;
		try {
			shape = JSON.parse(readFileSync(join(s.dir, `${cam.name}_depth.json`), "utf8"));
			depth = new Float32Array(new Uint8Array(readFileSync(join(s.dir, `${cam.name}_depth.f32`))).buffer);
		} catch {
			throw new Error(`depth artifact not found for camera ${cam.name} at step ${s.blob.step_idx}`);
		}
		if (row < 0 || row >= shape.height || col < 0 || col >= shape.width)
			throw new Error(`pixel (${row},${col}) out of bounds for ${cam.name} depth ${shape.height}x${shape.width}`);
		const z = depth[row * shape.width + col];
		if (!Number.isFinite(z) || z <= 0 || z > 10)
			throw new Error(`invalid depth ${z.toFixed(4)}m at ${cam.name} pixel (${row},${col})`);
		const pointCamera = apply(inv3(K), [col, row, 1]).map((v) => v * z);
		const pointTarget = apply(cam.extrinsic.matrix, pointCamera);
		const pointBase = cam.extrinsic.frame === "tcp" ? apply(pose7(tcpPose(s)), pointTarget) : pointTarget;
		return {
			camera: cam.name,
			pixel: [row, col],
			depth_m: round(z),
			point_camera: roundAll(pointCamera),
			point_base: roundAll(pointBase),
			target_frame: cam.extrinsic.frame,
		};
	}

	/** Mark the selected pixel on the step's image (`<camera>_selected.png`). */
	function overlay(s: Step, name: string, row: number, col: number): string | undefined {
		try {
			const { width, height } = JSON.parse(readFileSync(join(s.dir, `${name}.json`), "utf8"));
			const rgb = mark({ width, height, rgb: readFileSync(join(s.dir, `${name}.rgb`)) }, row, col, [255, 0, 0]);
			const path = join(s.dir, `${name}_selected.png`);
			writeFileSync(path, encodePng(rgb, width, height));
			return path;
		} catch {
			return undefined;
		}
	}

	tool(
		"back_project",
		"Back-project one camera pixel (row, col; row 0 = top) into UR5e base coordinates through its depth, intrinsics and hand-eye calibration.",
		Type.Object({
			row: Type.Integer({ minimum: 0 }),
			col: Type.Integer({ minimum: 0 }),
			camera: cameraParam,
			step: stepParam,
		}),
		async ({ row, col, camera, step }) => {
			const s = getStep(step);
			const cam = cameraOf(s, camera);
			const p = project(s, cam, row, col);
			const selected = overlay(s, cam.name, row, col);
			return {
				...p,
				world_xyz: p.point_base,
				coordinate_frame: "ur5e_base",
				step: s.blob.step_idx,
				tcp_pose_xyzw: roundAll(tcpPose(s)),
				...(selected ? { selected_pixel_overlay: selected } : {}),
			};
		},
		false,
	);

	tool(
		"segment",
		"SAM3 segmentation of a camera image from the latest (or given) step. Give exactly one of a text prompt or a positive point [row, col]. On a camera with depth the mask's pixels are back-projected and world_xyz is their median base-frame point. Returns an overlay image.",
		Type.Object({
			prompt: Type.Optional(Type.String()),
			point: Type.Optional(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 })),
			camera: cameraParam,
			step: stepParam,
			min_score: Type.Optional(Type.Number({ description: "Default 0.2" })),
		}),
		async ({ prompt, point, camera, step, min_score = 0.2 }) => {
			if (!sam3) throw new Error("segment requires --robot-sam3");
			const text = prompt?.trim();
			if (!text && !point) return { error: "give a text prompt or a point [row, col]" };
			const s = getStep(step);
			const cam = cameraOf(s, camera);
			const path = s.images[cam.name];
			if (!path) throw new Error(`no image recorded for camera ${cam.name} at step ${s.blob.step_idx}`);
			const { width, height } = JSON.parse(readFileSync(join(s.dir, `${cam.name}.json`), "utf8"));
			const rgb = readFileSync(join(s.dir, `${cam.name}.rgb`));
			const res = await sam3.call<{
				found: boolean;
				score?: number;
				box?: number[];
				mask_png_base64?: string;
				reason?: string;
			}>(
				"sam3.segment",
				{
					image_base64: readFileSync(path).toString("base64"),
					...(text ? { text_prompt: text } : { point }),
					min_score,
				},
				120_000,
			);
			if (!res.found || !res.mask_png_base64)
				return {
					found: false,
					error: res.reason ?? "no mask",
					fallback: "Pick pixels in the image and use back_project.",
				};
			const mask = decodePngChannel(Buffer.from(res.mask_png_base64, "base64"));
			if (mask.width !== width || mask.height !== height)
				return {
					found: true,
					error: `mask ${mask.width}x${mask.height} does not match the ${width}x${height} image`,
				};
			const rows: number[] = [];
			const cols: number[] = [];
			const pts: number[][] = [];
			const over = Buffer.from(rgb);
			for (let i = 0; i < mask.data.length; i++) {
				if (mask.data[i] < 128) continue;
				const r = Math.floor(i / width);
				const c = i % width;
				rows.push(r);
				cols.push(c);
				over[i * 3] = Math.round(0.55 * over[i * 3] + 0.45 * 255);
				over[i * 3 + 1] = Math.round(0.55 * over[i * 3 + 1]);
				over[i * 3 + 2] = Math.round(0.55 * over[i * 3 + 2]);
				if (cam.has_depth && cam.extrinsic && cam.intrinsic_K && (i & 7) === 0) {
					try {
						pts.push(project(s, cam, r, c).point_base);
					} catch {}
				}
			}
			const result: Json = {
				found: true,
				camera: cam.name,
				step: s.blob.step_idx,
				score: res.score === undefined ? null : round(res.score, 3),
				box: res.box,
				n_pixels: rows.length,
				centroid_pixel: rows.length ? [Math.round(median(rows)), Math.round(median(cols))] : null,
				world_xyz: pts.length >= 10 ? [0, 1, 2].map((k) => round(median(pts.map((p) => p[k])))) : null,
			};
			if (pts.length < 10)
				result.world_note = cam.has_depth
					? `too few valid depth pixels (${pts.length}); back_project a pixel of the mask instead`
					: `camera ${cam.name} has no depth; back_project needs a camera with depth`;
			result._pngs = [encodePng(over, width, height)];
			return result;
		},
		false,
	);

	tool(
		"move_delta",
		"Move the UR5e TCP by a bounded base-frame xyz delta in meters (x forward, y left, z up); the orientation is held.",
		Type.Object({ delta_xyz: xyz }),
		async ({ delta_xyz }, signal) => {
			check(signal);
			const d = vec3(delta_xyz, "delta_xyz");
			checkMove(d, maxMove());
			return motion("env.move_delta", { delta_xyz: d }, signal);
		},
	);

	tool(
		"move_pose",
		"Move the UR5e TCP to an absolute base-frame pose within the per-call limits: xyz in meters plus an orientation as rotvec (axis-angle, rad) or rpy (extrinsic xyz Euler, rad); omit both to keep the orientation.",
		Type.Object({
			xyz,
			rotvec: Type.Optional(xyz),
			rpy: Type.Optional(xyz),
		}),
		async ({ xyz: p, rotvec, rpy }, signal) => {
			check(signal);
			const target = vec3(p, "xyz");
			if (rotvec !== undefined && rpy !== undefined) throw new Error("give rotvec or rpy, not both");
			const tcp = tcpPose(getStep());
			checkMove(
				target.map((v, i) => v - tcp[i]),
				maxMove(),
			);
			const kwargs: Json = { xyz: target };
			if (rotvec !== undefined) kwargs.rotvec = vec3(rotvec, "rotvec");
			if (rpy !== undefined) kwargs.rpy = vec3(rpy, "rpy");
			return motion("env.move_pose", kwargs, signal);
		},
	);

	tool(
		"rotate_delta",
		"Rotate the UR5e TCP by a bounded base-frame rpy delta in radians (extrinsic xyz: roll about base x, pitch about y, yaw about z).",
		Type.Object({ delta_rpy: xyz }),
		async ({ delta_rpy }, signal) => {
			check(signal);
			const d = vec3(delta_rpy, "delta_rpy");
			checkRotate(Math.hypot(...d));
			return motion("env.rotate_delta", { delta_rpy: d }, signal);
		},
	);

	tool(
		"gripper",
		"Open or close the Robotiq gripper and wait for the fingers to settle. A close that catches nothing reopens and reports grasp_empty; fingers that do not move report gripper_jammed.",
		Type.Object({ action: StringEnum(["open", "close"] as const) }),
		async ({ action }, signal) => {
			check(signal);
			return motion("env.set_gripper", { open: action === "open" }, signal);
		},
	);

	// ---- lifecycle

	/** Operator-confirmed reset (the start dialog, request_scene_reset): open the gripper, moveJ to the begin pose. */
	async function resetArm(): Promise<Json> {
		const r = await call<Json>("env.reset", {}, 120_000);
		if (!r.ok) throw new Error(`reset did not reach the begin pose: ${JSON.stringify(plain(r.move ?? r))}`);
		await record({ action: "reset" }, r, null);
		return { ok: true, step: steps.length - 1 };
	}

	async function startRobot(ctx: ExtensionContext): Promise<string[]> {
		if (!ctx.hasUI)
			throw new Error("ur5e drives a real robot: run pi interactively (or over RPC) so an operator is present");
		if (pi.getFlag("operator") !== true)
			throw new Error("ur5e drives a real robot: start pi with --operator so an operator judges every episode");
		if (!armId())
			throw new Error(
				"ur5e is bound to one arm: start pi with --arm-id <controller serial> (env_server --print-identity), the arm whose config (limits, begin pose, camera calibrations) this session drives",
			);
		for (const name of ["max-move", "max-rotate"])
			if (!(Number(flag(name)) > 0)) throw new Error(`--${name} must be a positive number (got '${flag(name)}')`);
		const r: Services = { root: flag("services"), python: flag("python", "python") };
		const configFlag = flag("robot-config");
		const config = configFlag ? resolve(ctx.cwd, configFlag) : "";
		const stamp = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
		out = resolve(ctx.cwd, flag("out") || join(tmpdir(), "pi-embodied", `ur5e_${taskName()}_${stamp}`));
		mkdirSync(out, { recursive: true });
		steps.length = 0;
		const endpoint = flag("robot-env");
		const camerasFlag = flag("robot-cameras").trim();
		const [rpc, sam] = await Promise.all([
			endpoint
				? attach(endpoint)
				: robot.serve({
						python: r.python,
						args: [
							"-m",
							"pi_embodied_services.robots.ur5e.env_server",
							...(config ? ["--robot-config", config] : []),
							...(camerasFlag ? ["--cameras", camerasFlag] : []),
						],
						cwd: r.root,
						env: servicesEnv(r),
						log: () => join(out, "ur5e_env_server.log"),
						readyMs: 120_000,
					}),
			flag("robot-sam3") ? attach(flag("robot-sam3")) : undefined,
		]);
		const m = await rpc.call<Meta>("env.get_env_meta", {}, 30_000);
		if (m.robot !== "ur5e")
			throw new Error(`--robot-env serves ${m.robot ?? "an unknown robot"}, not a ur5e env server`);
		if (m.arm_id === null || m.arm_id === undefined)
			throw new Error(
				"the env server bound its config to no arm (robot.identity: none), so --arm-id cannot be verified; set robot.identity: serial and calibration.arm_id",
			);
		if (String(m.arm_id) !== armId())
			throw new Error(
				`--arm-id ${armId()} is not the arm the env server drives (its config is bound to ${m.arm_id}); nothing moved`,
			);
		const t = m.tasks?.[taskName()];
		if (!t?.instruction)
			throw new Error(
				`task '${taskName()}' is not in the robot config's tasks (have: ${Object.keys(m.tasks ?? {}).join(", ") || "none"})`,
			);
		if (!m.has_begin_pose)
			throw new Error("calibration.begin_joints is not set in the robot config; reset would fail");
		const go = await ctx.ui.confirm(
			`Move the UR5e arm ${m.arm_id}?`,
			`The arm will open its gripper and moveJ to its begin pose, then the agent drives it for: ${t.instruction}. Clear the workspace and keep the emergency stop in reach.`,
		);
		if (!go) throw new Error("operator declined the reset; the UR5e tools stay disabled");
		env = rpc;
		sam3 = sam;
		meta = m;
		try {
			await resetArm();
		} catch (err) {
			env = sam3 = meta = undefined;
			throw err;
		}
		task = t;
		ctx.ui.notify(
			`UR5e ${m.arm_id} ready: ${taskName()}; cameras ${cameras().join(", ")}; steps under ${out}`,
			"info",
		);
		return sam ? [...TOOLS, "segment"] : TOOLS;
	}
}
