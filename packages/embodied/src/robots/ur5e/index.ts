/**
 * One physical Universal Robots UR5e arm for pi.
 *
 *   pi -e packages/embodied/src/robots/ur5e --operator --arm-id 2023300001 --task block_bowl --robot-config my_ur5e.yaml
 *   pi -e packages/embodied/src/robots/ur5e --operator --arm-id 2023300001 --task block_bowl --code=true --code-real
 *      (run_code: the env server runs with --code; a program meets the tools' server-side limits; every program is confirmed)
 *
 * Starts the env server (pi_embodied_services.robots.ur5e.env_server: ur_rtde to the controller, a
 * Robotiq gripper over the URCap socket, RealSense / webcam / RTSP cameras through the services'
 * shared camera layer) or attaches to one with --robot-env. The server owns the safety limits: pi's
 * --max-move / --max-rotate (passed at spawn; an attached server must enforce them or tighter ones,
 * ../../primitives/motion.ts servedLimits) and the robot YAML's per-call translation and rotation
 * refusal, the workspace box and Z floor, the tool tilt limit, `stop` that really stops the running
 * moveL/moveJ (stopL/stopJ), the setpoint cleared after a stop or failure, and the gripper's jammed /
 * empty-grasp detection. The tools and code primitives are the manifest's
 * (../../primitives/manifests/ur5e.json): move_delta / move_pose / rotate_delta / open_gripper /
 * close_gripper run the server methods of the same names, which a program calls too.
 *
 * A real robot needs an operator: pi must have a UI, --operator must be on (the base then asks for a
 * verdict before `finish`), and the operator confirms the reset before any motion. The robot is bound
 * to one arm: --arm-id names the arm the operator intends to drive and must equal the identity the
 * server bound its config (limits, begin pose, camera calibrations) to; a mismatch refuses to start.
 * Motion tools record a state step (robot state, every camera's RGB and, where the camera has it,
 * depth, camera metadata) under --out and return it with the images. back_project reads a pixel's
 * depth through the camera's intrinsics and its hand-eye calibration (robots/ur5e/calibrate.py); an
 * RGB-only camera has no depth to project. segment is active only with --segment.
 *
 * --explore (../explore.ts, `/explore`) runs operator-judged attempts: `reset` is the operator's scene
 * reset followed by the arm's, a success verdict is the solve, and the motion commands after the last
 * reset are exported as the cell's recipe (../memory, cell `ur5e_<arm-id>_<task>`).
 */

import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { dir, python, service, servicesDir } from "../../infra/config.ts";
import { trackFlags } from "../../infra/params.ts";
import { decodePngChannel, encodePng } from "../../infra/png.ts";
import { NdArray, type RpcClient } from "../../infra/rpc.ts";
import type { Move, MoveUnit, Vec3 } from "../../modes/units/index.ts";
import { template } from "../../planner/context-version.ts";
import {
	detectionActive,
	detectionArgs,
	detectionTools,
	type PerceptionCaps,
	registerDetectionFlags,
} from "../../primitives/detections.ts";
import { mountGraspTool } from "../../primitives/grasp.ts";
import { checkRotate, checkRoute, limitArgs, type MotionLimits, servedLimits } from "../../primitives/motion.ts";
import { pointActive, pointTool, registerPointFlags } from "../../primitives/pointing.ts";
import {
	apply,
	attach,
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
	type Services,
	servicesEnv,
	toolResult,
	vec,
} from "../../robot.ts";

const SYSTEM = template(new URL("./SYSTEM.md", import.meta.url));
const EXPLORE = template(new URL("./explore.md", import.meta.url));
/** The state-advancing tools a memory recipe keeps. */
const MOTION = ["move_delta", "move_pose", "rotate_delta", "open_gripper", "close_gripper", "act"];

type Task = { instruction: string; success_criteria?: string };
type Meta = {
	robot: string;
	backend: string;
	arm_id: string | null;
	cameras: string[];
	main_camera: string;
	gripper: string;
	has_begin_pose: boolean;
	limits: {
		max_move_m: number;
		max_rotate_rad: number;
		z_floor_m: number | null;
		empty_width_m: number | null;
		reset_lift_m?: number;
	};
	tasks: Record<string, Task>;
	/** pi's limits as the server enforces them (services utils/code_real.py). */
	motion_limits?: MotionLimits;
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
 * A camera's mount from its recorded metadata: the config's `mount`, else its calibration's frame
 * (tcp: eye-in-hand, a wrist camera; base: fixed), else null (unknown).
 */
export function cameraMount(cam: Json | null | undefined): "wrist" | "fixed" | null {
	if (cam?.mount === "wrist" || cam?.mount === "fixed") return cam.mount;
	const frame = cam?.extrinsic?.frame;
	return frame === "tcp" ? "wrist" : frame === "base" ? "fixed" : null;
}

/** Whether the cameras include a wrist view: all of them known to be fixed means none; an unknown mount may be one. */
export function hasWristCamera(mounts: readonly ("wrist" | "fixed" | null)[]): boolean {
	return !mounts.length || mounts.some((m) => m !== "fixed");
}

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

/** The robot's own tools; `segment` is activated only with --segment. */
const TOOLS = [
	"view_env_state",
	"view_camera_meta",
	"back_project",
	"move_delta",
	"move_pose",
	"rotate_delta",
	"open_gripper",
	"close_gripper",
	"finish",
];
/** The motion tools: each runs the server method of its name with the manifest's params. */
const MOTION_TOOLS = ["move_delta", "move_pose", "rotate_delta", "open_gripper", "close_gripper"];

export default function ur5e(pi: ExtensionAPI) {
	// Every flag this robot registers is tracked: numbers fail closed, the result records them (../../infra/params.ts).
	trackFlags(pi);
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
	pi.registerFlag("segment", {
		type: "boolean",
		default: false,
		description: "SAM3 on the env server (services.sam3 of the deployment config; enables segment)",
	});
	const sam3Url = () => (pi.getFlag("segment") === true ? service(pi, "sam3") : "");
	// --detections (the env server's SAM3 masks with ids through --segment) / --depth unidepth (enhance_depth:
	// UniDepth depth for an RGB-only camera, which back_project then reads): ../primitives/detections.ts.
	registerDetectionFlags(pi);
	// --point: Molmo's point over services.molmo (../primitives/pointing.ts).
	registerPointFlags(pi);
	pi.registerFlag("max-move", {
		type: "string",
		default: "0.08",
		description:
			"Largest translation per call, m, enforced by the env server (its limits.max_move_m applies if tighter)",
	});
	pi.registerFlag("max-rotate", {
		type: "string",
		default: "0.2",
		description:
			"Largest rotation per call, rad, enforced by the env server (its limits.max_rotate_rad applies if tighter)",
	});

	let env: RpcClient | undefined;
	let sam3: RpcClient | undefined;
	let meta: Meta | undefined;
	/** The limits the env server enforces (at least pi's flags). */
	let enforced: MotionLimits | undefined;
	let task: Task | undefined;
	let out = "";
	const steps: Step[] = [];
	/** Main camera first, then the others in the server's order. */
	const cameras = () => {
		const names = meta?.cameras ?? [];
		const main = meta?.main_camera;
		return main && names.includes(main) ? [main, ...names.filter((n) => n !== main)] : names;
	};
	/**
	 * A camera's mount (`wrist` or `fixed`) from the latest recorded camera metadata; a camera whose config
	 * sets no mount falls back to its calibration's frame (tcp: eye-in-hand, base: fixed), else null.
	 */
	const mountOf = (name: string) => cameraMount(steps[steps.length - 1]?.meta?.cameras?.[name]);
	const taskName = () => robot.task.task;
	const armId = () => flag("arm-id").trim();

	const robot = defineRobot(pi, {
		name: "ur5e",
		// Tools and code primitives: ../../primitives/manifests/ur5e.json (the env server reads it too).
		manifest: "ur5e",
		vars: () => ({ cameras: cameras(), max_move: String(maxMove()), max_rotate: String(maxRotate()) }),
		// Must agree with the env server's _has (code mode refuses otherwise): what the flags ask for and the
		// server serves (its perception capabilities); robot_sam3 is pi's own segment tool.
		capabilities: (c) => {
			const served = (meta as { capabilities?: { perception?: PerceptionCaps } } | undefined)?.capabilities
				?.perception;
			return (
				{
					sam3: pi.getFlag("detections") === true && Boolean(served?.segment),
					unidepth: Boolean(String(pi.getFlag("depth") ?? "").trim()) && Boolean(served?.enhance_depth),
					robot_sam3: Boolean(sam3Url()),
				}[c] ?? false
			);
		},
		task: ["task"],
		keepImages: 4,
		operator: { step: () => steps.length, reset: resetArm },
		// No corpus is published for the UR5e: memory is what exploration writes locally, one cell per arm and
		// task; the guard also opens the step images the results point at.
		memory: {
			cell: () => ({ tag: `ur5e_${armId() || "unbound"}_${taskName()}`, reference: "" }),
			primitives: MOTION,
			readable: () => (out ? [out] : []),
			published: false,
		},
		explore: {
			// The operator restores the scene, then the arm resets (resetArm); a failed or unconfirmed reset throws and starts no attempt.
			reset: async (result, ctx, signal) => {
				const r: Json = await op.sceneReset(ctx, String(result.reason ?? ""), "", signal);
				if (r.error) throw new Error(JSON.stringify(r));
				const s = getStep(-1);
				return toolResult(
					{ ...s.blob, ...result, robot_reset: r.robot_reset, scene_reset_confirmed: true },
					stepImages(s),
				);
			},
			prompt: () => EXPLORE.replaceAll("{{task_id}}", taskName()),
			rewrite: [
				[
					/^5\. When you believe the task is done, ask for the operator's verdict \(request_operator_verdict\), then finish\.$/m,
					"5. This is an exploration run: follow the Exploration workflow below. Success is only the operator's verdict.",
				],
			],
			// Every attempt costs the operator a manual scene reset (RPent's real-robot defaults).
			budget: { sessions: 1, attempts: 3 },
			operatorJudged: true,
		},
		video: true,
		// One image per camera, main first; the wrist-mounted cameras are the VDM's wrist views.
		vdm: () => {
			const c = cameras();
			if (!c.length) return undefined;
			const wrist = c.flatMap((name, i) => (mountOf(name) === "wrist" ? [i] : []));
			return { views: c.length, ...(wrist.length ? { wrist } : {}) };
		},
		// The env server's code.api (from the same manifest), recorded per episode.
		codeApi: () => env,
		// Code mode (../code) on the real arm: --code-real and --operator, every program confirmed. The
		// server (started with --code) runs it through the tools' own env methods, which hold pi's
		// per-call limits for every caller; the run becomes the next state step.
		code: {
			real: true,
			rpc: () => env as RpcClient,
			instruction: () => task?.instruction ?? "",
			observe: async (r) => {
				for (const f of (r.frames as NdArray[] | undefined) ?? []) robot.video.frame(frameOf(f));
				const s = await record({ action: "run_code" }, { status: r.status, motions: r.motions ?? 0 }, null);
				return toolResult({ ...s.blob }, stepImages(s));
			},
		},
		start: startRobot,
		stop: () => {
			env = sam3 = meta = task = enforced = undefined;
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
				return meta?.limits.empty_width_m ?? 0.011;
			},
			// A wrist view unless every camera is fixed (by its mount or its calibration's frame); an unknown one may be a wrist.
			wrist: () => hasWristCamera(cameras().map(mountOf)),
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

	/** pi's limits (its flags), which the env server enforces. */
	const wanted = (): MotionLimits => ({
		max_move_m: Number(flag("max-move", "0.08")),
		max_rotate_rad: Number(flag("max-rotate", "0.2")),
	});
	/** The per-call limits in force (prompt, units): the served ones and the config's caps. */
	const maxMove = () =>
		Math.min(enforced?.max_move_m ?? wanted().max_move_m ?? Infinity, meta?.limits.max_move_m ?? Infinity);
	const maxRotate = () =>
		Math.min(
			enforced?.max_rotate_rad ?? wanted().max_rotate_rad ?? Infinity,
			meta?.limits.max_rotate_rad ?? Infinity,
		);
	/** Code mode is on (--code): the env server serves code.run. */
	const coding = () => (pi.getFlag("code") ?? "false") !== "false";

	function call<T = Json>(method: string, kwargs: Json = {}, timeoutMs = 30_000, signal?: AbortSignal) {
		if (!env) throw new Error("ur5e is not initialized; see the session start error");
		return env.call<T>(method, kwargs, timeoutMs, [], signal);
	}
	const check = (signal?: AbortSignal) => {
		op.check();
		if (signal?.aborted) throw new Error("tool operation interrupted");
		if (pi.getFlag("explore") === true && (op.result() as Json).operator_verdict === "success")
			throw new Error(
				"motion refused: the operator judged this attempt a success. Write the audit and memory drafts, then call finish.",
			);
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
		// terminated: the operator's success verdict, not a step, solves an attempt (the memory recipe keeps these steps).
		const blob: Json = {
			step_idx: idx,
			state: plain(state),
			timestamps: plain(obs.timestamps),
			terminated: false,
			artifacts,
			images,
		};
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
	 * Register a manifest tool (its schema and description are manifests/ur5e.json's). Mutating tools
	 * run, then record a fresh state step and return it (errors included); read-only tools return their
	 * result or `{error}`, plus any PNGs in `_pngs`.
	 */
	function tool(name: string, run: (p: Json, signal: AbortSignal | undefined) => Promise<Json>, mutating = true) {
		robot.tool(name, "", Type.Object({}), (params: Json, signal) =>
			outcome(name, params, () => run(params, signal), mutating),
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

	const motion = (method: string, kwargs: Json, signal?: AbortSignal) => call(method, kwargs, 120_000, signal);

	/** Units mode (../units): one grounded action unit on the move primitives. */
	function unitStep(move: Move, signal: AbortSignal | undefined) {
		return outcome(
			"act",
			{ move },
			async () => {
				check(signal);
				// A unit is up to three server calls: refused whole before the first (the server checks each again).
				checkRoute([0, 0, 0], [move.delta], { max_move_m: maxMove() });
				checkRotate([0, 0, move.yaw ?? 0], maxRotate());
				const out: Json = {};
				if (move.gripper) out.gripper = await motion(`env.${move.gripper}_gripper`, {}, signal);
				if (Math.hypot(...move.delta) > 0)
					out.move = await motion("env.move_delta", { delta_xyz: move.delta }, signal);
				if (move.yaw) out.rotate = await motion("env.rotate_delta", { delta_rpy: [0, 0, move.yaw] }, signal);
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

	const stepOf = (p: Json): number => (typeof p.step === "number" ? p.step : -1);

	tool(
		"view_env_state",
		async (p) => {
			const s = getStep(stepOf(p));
			return { ...s.blob, _pngs: stepImages(s) };
		},
		false,
	);

	tool(
		"view_camera_meta",
		async (p) => {
			const step = stepOf(p);
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
		// An RGB-only camera has depth once enhance_depth stored the estimate for this step.
		if (!cam.has_depth && !existsSync(join(s.dir, `${cam.name}_depth.f32`)))
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
		async ({ row, col, camera, step }) => {
			const s = getStep(step ?? -1);
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

	// Molmo pointing on the current images (active with --point).
	mountGraspTool(
		robot.tool,
		pointTool(pi, {
			cameras: [],
			defaultCamera: () => cameras()[0] ?? "",
			frame: async (c) => {
				const s = getStep(-1);
				const { width, height } = JSON.parse(readFileSync(join(s.dir, `${c}.json`), "utf8"));
				return { width, height, rgb: readFileSync(join(s.dir, `${c}.rgb`)) };
			},
			locate: async (c, row, col) => {
				const s = getStep(-1);
				try {
					const p = project(s, cameraOf(s, c), row, col);
					return { world_xyz: p.point_base, coordinate_frame: "ur5e_base", depth_m: p.depth_m };
				} catch {
					return undefined;
				}
			},
			signal: () => robot.signal,
		}),
	);

	// SAM3 masks with ids and UniDepth over the env server's perception, on the latest step's frames.
	for (const d of detectionTools(pi, {
		call: (method, kwargs, timeoutMs) => call<Json>(method, kwargs, timeoutMs ?? 120_000, robot.signal),
		cameras: [],
		defaultCamera: () => cameras()[0] ?? "",
		// The fused depth belongs to the latest step (the server's current observation): store it there for back_project.
		onDepth: (camera, depth) => {
			const s = getStep(-1);
			const g = gridOf(depth);
			writeFileSync(join(s.dir, `${camera}_depth.f32`), Buffer.from(g.data.buffer));
			writeFileSync(join(s.dir, `${camera}_depth.json`), JSON.stringify({ height: g.height, width: g.width }));
			if (!s.blob.artifacts.includes(`${camera}_depth.f32`)) s.blob.artifacts.push(`${camera}_depth.f32`);
			return { step: s.blob.step_idx, depth_path: join(s.dir, `${camera}_depth.f32`) };
		},
	}))
		mountGraspTool(robot.tool, d);

	tool(
		"segment",
		async ({ prompt, point, camera, step, min_score = 0.2 }) => {
			if (!sam3) throw new Error("segment requires --segment");
			const text = typeof prompt === "string" ? prompt.trim() : "";
			if (!text && !point) return { error: "give a text prompt or a point [row, col]" };
			const s = getStep(step ?? -1);
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

	// The motion tools: the manifest's params straight to the server method of the same name, which holds
	// pi's limits (and the config's) for the tools and a program alike.
	for (const name of MOTION_TOOLS)
		tool(name, async (p, signal) => {
			check(signal);
			return motion(`env.${name}`, p, signal);
		});

	// ---- lifecycle

	/** Operator-confirmed reset (the start dialog, request_scene_reset): open the gripper (releasing a held object), lift, moveJ to the begin pose. */
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
		const limits = wanted();
		const r: Services = { root: servicesDir(pi), python: python(pi, "ur5e") };
		const configFlag = flag("robot-config");
		const config = configFlag ? resolve(ctx.cwd, configFlag) : "";
		const stamp = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
		out = resolve(ctx.cwd, dir(pi, "artifacts") || join(tmpdir(), "pi-embodied", `ur5e_${taskName()}_${stamp}`));
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
							// pi's per-call limits, enforced by the server for tools and programs alike.
							...limitArgs(limits),
							...detectionArgs(pi, { sam3: Boolean(sam3Url()) }),
							...(coding() ? ["--code"] : []),
						],
						cwd: r.root,
						env: servicesEnv(r),
						log: () => join(out, "ur5e_env_server.log"),
						readyMs: 120_000,
					}),
			sam3Url() ? attach(sam3Url()) : undefined,
		]);
		const m = await rpc.call<Meta>("env.get_env_meta", {}, 30_000);
		if (m.robot !== "ur5e")
			throw new Error(`--robot-env serves ${m.robot ?? "an unknown robot"}, not a ur5e env server`);
		// An attached server must enforce pi's limits (or tighter ones); throws otherwise.
		const served = servedLimits(m.motion_limits, limits);
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
		const lift = m.limits?.reset_lift_m;
		const go = await ctx.ui.confirm(
			`Move the UR5e arm ${m.arm_id}?`,
			`The arm will open its gripper (anything it holds is released where it is), lift ${typeof lift === "number" ? `${Math.round(lift * 1000)} mm` : "a few cm"} straight up, then moveJ to its begin pose; then the agent drives it for: ${t.instruction}. Clear the workspace and the path to the begin pose, and keep the emergency stop in reach.`,
		);
		if (!go) throw new Error("operator declined the reset; the UR5e tools stay disabled");
		env = rpc;
		sam3 = sam;
		meta = m;
		enforced = served;
		try {
			await resetArm();
		} catch (err) {
			env = sam3 = meta = enforced = undefined;
			throw err;
		}
		task = t;
		ctx.ui.notify(
			`UR5e ${m.arm_id} ready: ${taskName()}; cameras ${cameras().join(", ")}; steps under ${out}`,
			"info",
		);
		const perception = (m as { capabilities?: { perception?: PerceptionCaps } }).capabilities?.perception;
		return [...TOOLS, ...(sam ? ["segment"] : []), ...detectionActive(pi, perception), ...pointActive(pi)];
	}
}
