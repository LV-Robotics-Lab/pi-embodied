/**
 * One physical Franka arm for pi.
 *
 *   pi -e packages/embodied/src/franka --task 1 --robot-config my_franka.yaml --robot-vla http://VLA_HOST:PORT
 *   pi -e packages/embodied/src/franka --robot-backend polymetis --robot-config my_polymetis.yaml
 *
 * Starts an env server for --robot-backend: rlinf (default; pi_embodied_services.robots.franka.env_server,
 * Ray must already run on the controller node) or polymetis (robots.franka_polymetis.env_server,
 * Show-Harness's Polymetis NUC stack), or attaches to a running one with --robot-env. Both serve the
 * same env.* RPC; the server's env.get_env_meta capabilities hide what it cannot serve (vla_grasp
 * needs has_vla). Only one backend may drive the arm at a time. The VLA is attach-only. The
 * server enforces the workspace limits, per-step clips and servo tolerances from its config;
 * task definitions and easy_handeye calibration come from the same services package. A real robot needs an operator: pi must have a UI, the operator confirms
 * the reset motion, and --operator adds the verdict gate (see ../operator.ts). The
 * single-arm robot has no success signal; the result entry records the agent's claim and
 * the operator's verdict, if any. Mutating tools record a state step (robot state, external
 * and wrist RGB-D, camera metadata) under --out and return it with both images; file tools
 * reach only the memory and that directory.
 */

import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";
import { ikArgs, registerIkFlag } from "../ik.ts";
import { encodePng } from "../png.ts";
import { graspActive, graspArgs, graspTools, registerGraspFlags } from "../primitives/grasp.ts";
import { type MotionRig, moveDelta, rotateDelta, setGripper } from "../primitives/motion.ts";
import { viewCameraMeta, viewEnvState } from "../primitives/perception.ts";
import { type Step as BaseStep, getStep, outcome, type StepsIO, stepParam, type ToolDef } from "../primitives/steps.ts";
import {
	apply,
	attach,
	checkMove,
	defineRobot,
	f32,
	frameOf,
	gridOf,
	inv3,
	type Json,
	type Mat,
	mark,
	message,
	moveLimit,
	numbers,
	plain,
	pose7,
	rgbOf,
	round,
	roundAll,
	SERVICES,
	type Services,
	servicesEnv,
	servicesJson,
	sub,
	toolResult,
	u8,
	vec,
	workspaceLimits,
} from "../robot.ts";
import { NdArray, type RpcClient } from "../rpc.ts";
import type { Move } from "../units/index.ts";

const SYSTEM = readFileSync(new URL("./SYSTEM.md", import.meta.url), "utf8");
const EXPLORE = readFileSync(new URL("./explore.md", import.meta.url), "utf8");
/** The motion tools: the recipe of a solved exploration attempt. */
const MOTION = ["move_delta", "rotate_delta", "open_gripper", "close_gripper", "vla_grasp", "act"];

type Task = { name: string; instruction: string; success_criteria: string; constraints: string[] };
type Calibration = {
	path: string;
	eye_on_hand: boolean;
	base_frame: unknown;
	tracking_base_frame: unknown;
	matrix: Mat;
};
type Setup = {
	task: Task;
	calibration?: { external: Calibration; wrist: Calibration; convention: string };
	calibration_error?: string;
};
/** A recorded step (../primitives/steps.ts) and the saved wrist and external camera images. */
type Step = BaseStep & { images: Record<string, string> };
/** env.get_env_meta().capabilities; servers from before it are the RLinf backend. */
type Caps = {
	backend: string;
	has_vla: boolean;
	max_move_m?: number | null;
	max_rotate_rad?: number | null;
	table_z_m?: number | null;
	/** Server-side perception (--robot-sam3 / --robot-unidepth): which of its primitives exist. */
	perception?: { segment?: boolean; enhance_depth?: boolean };
	[k: string]: unknown;
};
/** A mask of the current observation, as the env server reports it (`env.segment`). */
type Detection = {
	id: string;
	rank: number;
	score: number | null;
	box: number[] | null;
	area_px: number;
	centroid_rc: [number, number] | null;
	depth_m: number | null;
	point_camera: number[] | null;
	mask_png_base64?: string;
	[k: string]: unknown;
};
/** Session entry: detection ids that stopped being valid (a new observation was taken). */
export const DETECTIONS_ENTRY = "robot_detections";
const PERCEPTION_TOOLS: Record<string, keyof NonNullable<Caps["perception"]>> = {
	segment: "segment",
	select_detection: "segment",
	reject_detection: "segment",
	enhance_depth: "enhance_depth",
};
const BACKENDS: Record<string, string> = {
	rlinf: "pi_embodied_services.robots.franka.env_server",
	polymetis: "pi_embodied_services.robots.franka_polymetis.env_server",
};

/** Task and calibration from the services, validated like the services' parse_config. */
const SETUP_PY = `
import dataclasses, json, sys
from pi_embodied_services.robots.franka.runtime_config import set_robot_config_path, validate_calibration_sources
from pi_embodied_services.robots.franka.tasks import get_franka_task
set_robot_config_path(sys.argv[2] or None)
validate_calibration_sources()
out = {"task": dataclasses.asdict(get_franka_task(int(sys.argv[1])))}
try:
    from pi_embodied_services.robots.franka.perception import load_calibration_bundle
    out["calibration"] = load_calibration_bundle()
except Exception as exc:
    out["calibration_error"] = str(exc)
print(json.dumps(out, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))
`;

const TOOLS = [
	"view_env_state",
	"view_camera_meta",
	"view_perception_setup",
	"segment",
	"select_detection",
	"reject_detection",
	"enhance_depth",
	"back_project",
	"back_project_correspondence",
	"move_delta",
	"rotate_delta",
	"open_gripper",
	"close_gripper",
	"vla_grasp",
	"finish",
];
/** Observation camera key -> saved image and depth artifact names. */
const ARTIFACTS = { main: ["wrist", "wrist_depth"], extra_0: ["camera", "camera_depth"] } as const;

/** `[H,W,3]` -> `[1,H,W,3]`, `[N,H,W,3]` -> `[1,N,H,W,3]`. */
function batchViews(v: unknown): NdArray | null {
	if (!(v instanceof NdArray)) return null;
	if (v.shape.length !== 3 && v.shape.length !== 4)
		throw new Error(`expected [H,W,3] or [N,H,W,3] image, got shape [${v.shape}]`);
	return u8(v).batched();
}

function confidence(delta: number): Json {
	const level = delta <= 0.015 ? "high" : delta <= 0.03 ? "medium" : delta <= 0.06 ? "low" : "very_low";
	return {
		score: round(Math.max(0, Math.min(1, 1 - delta / 0.08)), 3),
		level,
		base_point_delta_m: round(delta),
		meaning:
			"Confidence is based on disagreement between independently back-projected wrist and third-person base-frame points.",
	};
}
const unavailable = (reason: string) => ({ score: null, level: "unavailable", reason });
const compact = (r: Json | undefined) =>
	r &&
	Object.fromEntries(
		[
			"camera_alias",
			"camera_key",
			"camera_name",
			"pixel",
			"depth_m",
			"point_base",
			"target_frame",
			"error",
			"error_type",
		]
			.filter((k) => k in r)
			.map((k) => [k, r[k]]),
	);
const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

function warnings(third: Json | undefined, wrist: Json): string[] {
	const out: string[] = [];
	if (third?.error) out.push(`third_person back-projection failed: ${third.error}`);
	if (wrist.error) out.push(`wrist back-projection failed: ${wrist.error}`);
	if (third?.point_base && wrist.point_base) {
		const d = dist(third.point_base, wrist.point_base);
		if (d > 0.06)
			out.push(
				`third_person/wrist base-frame estimates differ by ${d.toFixed(3)}m; treat this correspondence as low confidence`,
			);
	}
	return out;
}

function cameraAlias(camera: string): "wrist" | "third_person" {
	const v = camera.trim().toLowerCase();
	if (v === "wrist" || v === "main") return "wrist";
	if (["third_person", "third-person", "external", "extra_0", "agentview"].includes(v)) return "third_person";
	throw new Error("unsupported camera; use 'wrist' or 'third_person'");
}

export default function franka(pi: ExtensionAPI) {
	const flag = (name: string, fallback = "") => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("task", {
		type: "string",
		default: "0",
		description: "Franka task id (0 smoke test, 1 VLA grasp)",
	});
	pi.registerFlag("robot-config", {
		type: "string",
		description:
			"Robot YAML (default: the backend's config/example.yaml under services/pi_embodied_services/robots/franka[_polymetis]/)",
	});
	pi.registerFlag("robot-backend", {
		type: "string",
		default: "",
		description:
			"Env server to start: rlinf (default) or polymetis; with --robot-env, the attached server's backend must match if set",
	});
	pi.registerFlag("robot-env", {
		type: "string",
		description: "Attach to a running Franka env server instead of starting one",
	});
	pi.registerFlag("robot-vla", {
		type: "string",
		description: "External Franka Pi0.5 VLA server (attach-only; enables vla_grasp)",
	});
	pi.registerFlag("robot-sam3", {
		type: "string",
		description: "SAM3 server for the env server's segment / select_detection / reject_detection (off without it)",
	});
	pi.registerFlag("robot-unidepth", {
		type: "string",
		description: "UniDepth server for the env server's enhance_depth (off without it)",
	});
	pi.registerFlag("services", {
		type: "string",
		default: process.env.PI_EMBODIED_SERVICES ?? SERVICES,
		description: "pi-embodied services dir",
	});
	pi.registerFlag("python", {
		type: "string",
		default: process.env.PI_EMBODIED_PYTHON ?? "python",
		description: "Python with the services' [franka] extra",
	});
	pi.registerFlag("out", {
		type: "string",
		description: "Step artifact directory (default: a new directory under the OS temp dir)",
	});
	pi.registerFlag("max-move", {
		type: "string",
		default: "0.1",
		description: "Largest move_delta per call, m (a task's documented limit applies if tighter)",
	});
	pi.registerFlag("workspace-xy", {
		type: "string",
		default: "0.159,1.159,-0.456,0.544",
		description:
			"TCP x/y box for move_delta and units, m: xmin,xmax,ymin,ymax (default: example.yaml's ee_pose_limit; '' = off)",
	});
	pi.registerFlag("z-floor", {
		type: "string",
		default: "",
		description:
			"Lowest TCP z for move_delta and units, m; also the table height for units' proprioception (required: the robot does not start without it; Show-Harness's empty-table floor is 0.14)",
	});
	pi.registerFlag("max-rotate", {
		type: "string",
		default: "0.5",
		description: "Largest rotate_delta per call, rad (norm of delta_rpy)",
	});
	// --graspnet/--graspgenx/--anyplace/--anygrasp: plan_grasp, plan_place, check_attached (../primitives/grasp.ts).
	registerGraspFlags(pi);
	registerIkFlag(pi);

	let env: RpcClient | undefined;
	let vla: RpcClient | undefined;
	let setup: Setup | undefined;
	let out = "";
	let lastStates: unknown;
	let caps: Caps = { backend: "rlinf", has_vla: true };
	/** env.get_env_meta().smooth.chaining: move_delta takes `continuous` (polymetis smooth + blend). */
	let chaining = false;
	const steps: Step[] = [];
	const task = () => robot.task.task;
	/** The first state step of the current attempt: localization refuses older ones after a scene reset. */
	let attemptStart = 0;
	const exploring = () => pi.getFlag("explore") === true;
	const judgedSuccess = () => (op.result() as Json).operator_verdict === "success";
	const robot = defineRobot(pi, {
		name: "franka",
		task: ["task"],
		keepImages: 4,
		video: true,
		// Observations carry the external camera then the wrist image.
		vdm: { views: 2, wrist: 1 },
		// The evaluation prompt names no memory; exploration writes it. The guard also opens the step artifacts.
		memory: {
			cell: () => ({ tag: `franka_t${task()}`, reference: "" }),
			primitives: MOTION,
			readable: () => [out],
		},
		operator: { step: () => steps.length, reset: resetRobot },
		explore: {
			// The operator restores the scene; a failed or unconfirmed reset throws and starts no attempt.
			reset: async (result, ctx, signal) => {
				const r: Json = await op.sceneReset(ctx, String(result.reason ?? ""), "", signal);
				if (r.error) throw new Error(JSON.stringify(r));
				const { output, pngs } = view(steps[steps.length - 1]);
				return toolResult({ ...output, ...result, robot_reset: r.robot_reset, scene_reset_confirmed: true }, pngs);
			},
			prompt: () =>
				EXPLORE.replace(/\{\{(task_id|task_name|instruction)\}\}/g, (_, k: string) =>
					k === "task_id"
						? task()
						: k === "task_name"
							? (setup?.task.name ?? "")
							: (setup?.task.instruction ?? ""),
				),
			rewrite: [
				[
					/^5\. Finish only when the success evidence is visible and consistent with state\.$/m,
					"5. This is an exploration run: follow the Exploration workflow below. Success is only the operator's verdict.",
				],
			],
			// Every attempt costs the operator a manual scene reset (RPent's real-robot defaults).
			budget: { sessions: 1, attempts: 3 },
			operatorJudged: true,
		},
		// The env server's primitive registry (services robots/franka/primitives.py), both backends.
		codeApi: () => env,
		start: startRobot,
		stop: () => {
			env = vla = undefined;
		},
		prompt: () => {
			if (!setup) return undefined;
			const t = setup.task;
			const vars: Record<string, string> = {
				task_name: t.name,
				instruction: t.instruction,
				success_criteria: t.success_criteria,
				constraints: t.constraints.map((c, i) => `${i + 1}. ${c}`).join("\n"),
			};
			return SYSTEM.replace(/\{\{(\w+)\}\}/g, (m, k: string) => vars[k] ?? m);
		},
		result: () => ({
			task: Number(task()),
			task_name: setup?.task.name ?? null,
			backend: caps.backend,
			steps: steps.length,
			out,
		}),
		status: () => ({ step: steps.length - 1 }),
		units: {
			// Show-Harness configs/primitives_franka.yaml (base frame: MV_LEFT is base -y).
			vectors: {
				MV_FWD: [1, 0, 0],
				MV_BACK: [-1, 0, 0],
				MV_LEFT: [0, -1, 0],
				MV_RIGHT: [0, 1, 0],
				MV_UP: [0, 0, 1],
				MV_DOWN: [0, 0, -1],
			},
			stepM: 0.02,
			yawStepRad: 0.15,
			// A continuous move_delta returns before the arm settles (Polymetis smooth chaining).
			chains: () => chaining,
			maxYawRad: () => maxRotate(),
			maxMoveM: () => moveLimit(maxMove(), setup?.task.constraints),
			apply: (move, signal) => unitStep(move, signal),
			state: unitState,
			instruction: () => setup?.task.instruction ?? "",
			// Show-Harness plugins/recovery: a closed Franka gripper at or below 1 mm holds nothing.
			emptyWidthM: 0.001,
			// The wrist camera is the rig's main camera (observation `main`).
			wrist: true,
		},
		finish: {
			description:
				"Call when the task is complete or unrecoverable. Halts the agent loop. Save any artifacts (recipe, audit) BEFORE calling finish.",
			parameters: Type.Object({
				status: Type.String({ description: "Outcome, e.g. 'success', 'failure', or 'stuck'." }),
				summary: Type.String({ description: "Short natural-language summary of the run." }),
			}),
			result: (params) => toolResult({ _finish: true, ...params }),
		},
	});
	const { op } = robot;

	const call = <T = Json>(method: string, kwargs: Json = {}, timeoutMs = 30_000, signal?: AbortSignal) =>
		(env as RpcClient).call<T>(method, kwargs, timeoutMs, [], signal);
	const remember = (states: unknown) => {
		if (states !== undefined && states !== null) lastStates = states;
	};
	/** The tighter of the flag and the server's own per-call limit (it refuses larger calls). */
	const maxMove = () => Math.min(Number(flag("max-move", "0.1")), caps.max_move_m ?? Number.POSITIVE_INFINITY);
	const maxRotate = () => Math.min(Number(flag("max-rotate", "0.5")), caps.max_rotate_rad ?? Number.POSITIVE_INFINITY);
	const check = (signal?: AbortSignal) => {
		op.check();
		if (signal?.aborted) throw new Error("tool operation interrupted");
		if (exploring() && judgedSuccess())
			throw new Error(
				"motion refused: the operator judged this attempt a success. Write the audit and memory drafts, then call finish.",
			);
	};

	/** The operator's scene reset (after the operator confirmed the scene): the env's reset, then a fresh step. */
	async function resetRobot(): Promise<Json> {
		if (!env || !steps.length) throw new Error("franka is not initialized; see the session start error");
		const r = await call("env.reset", {}, 180_000);
		if (r?.error) throw new Error(`env.reset failed: ${JSON.stringify(plain(r))}`);
		remember(r.states);
		const s = await dumpState({ action: "scene_reset" }, { ok: true }, null);
		attemptStart = s.blob.step_idx;
		return { ok: true, step: s.blob.step_idx };
	}

	/** A step to localize in: none from before the last scene reset. */
	function freshStep(step?: number | null): Step {
		const s = getStep(steps, step);
		if (s.blob.step_idx < attemptStart)
			throw new Error(
				`localization refused: step ${s.blob.step_idx} predates the last scene reset (step ${attemptStart}); use a fresh observation`,
			);
		return s;
	}

	// ---- env client (the server returns camera frames, the client caches states)

	/** Ids of the masks segment returned for the current observation (env-side `DetectionBook`). */
	let liveIds: string[] = [];

	async function observation(): Promise<Json> {
		// The env server drops the detection ids only when the arm moved since they were cut (its
		// state digest); the next perception result reports them (noteInvalidated).
		const obs = await call("env.get_observation");
		if (!("states" in obs) && lastStates !== undefined) obs.states = lastStates;
		remember(obs.states);
		return obs;
	}

	async function robotState(): Promise<Json> {
		const state = await call("env.get_robot_state");
		if (lastStates !== undefined) state.wrapped_state_vector = lastStates;
		return state;
	}

	async function motion(method: string, kwargs: Json, signal?: AbortSignal, timeoutMs = 120_000): Promise<Json> {
		const result = await call(method, kwargs, timeoutMs, signal);
		remember(result.states);
		return result;
	}

	// ---- state steps (robot state dumps)

	async function dumpState(command: Json | null, result: Json | null, elapsed: number | null): Promise<Step> {
		const obs = await observation();
		const state = await robotState();
		const meta = await call<Json | null>("env.get_camera_meta");
		const idx = steps.length;
		const dir = join(out, `step_${String(idx).padStart(4, "0")}`);
		mkdirSync(dir, { recursive: true });
		const artifacts: string[] = [];
		const images: Record<string, string> = {};
		const first = (v: unknown, ndim: number) => (v instanceof NdArray && v.shape.length === ndim ? sub(v, 0) : v);
		const saveImage = (name: string, v: unknown) => {
			if (!(v instanceof NdArray)) return;
			const img = rgbOf(v);
			writeFileSync(join(dir, `${name}.png`), encodePng(img.rgb, img.width, img.height));
			writeFileSync(join(dir, `${name}.json`), JSON.stringify({ width: img.width, height: img.height }));
			writeFileSync(join(dir, `${name}.rgb`), img.rgb);
			images[name] = join(dir, `${name}.png`);
			artifacts.push(`${name}.png`);
		};
		const saveDepth = (name: string, v: unknown) => {
			if (!(v instanceof NdArray)) return;
			const g = gridOf(v);
			writeFileSync(join(dir, `${name}.f32`), Buffer.from(g.data.buffer));
			writeFileSync(join(dir, `${name}.json`), JSON.stringify({ height: g.height, width: g.width }));
			artifacts.push(`${name}.f32`);
		};
		const camera = first(obs.extra_view_images, 4);
		saveImage("wrist", obs.main_images);
		saveImage("camera", camera);
		// The episode video follows the external camera (the wrist one when there is none).
		const shown = camera instanceof NdArray ? camera : obs.main_images;
		if (shown instanceof NdArray) robot.video.frame(frameOf(shown));
		saveDepth("wrist_depth", obs.main_depths);
		saveDepth("camera_depth", first(obs.extra_view_depths, 3));
		if (meta) {
			writeFileSync(join(dir, "camera_meta.json"), JSON.stringify(plain(meta)));
			artifacts.push("camera_meta.json");
		}
		const blob: Json = {
			step_idx: idx,
			state: plain(state),
			terminated: false,
			truncated: false,
			artifacts: artifacts.sort(),
		};
		if (command) blob.command = command;
		if (result) blob.result = plain(result);
		if (elapsed !== null) blob.elapsed_s = elapsed;
		appendFileSync(join(out, "states.jsonl"), `${JSON.stringify(blob)}\n`);
		const step = { blob, dir, meta: meta ? (plain(meta) as Json) : null, images };
		steps.push(step);
		return step;
	}

	/** view_env_state: the step blob plus external then wrist image. */
	function view(s: Step) {
		const output: Json = { ...s.blob };
		const pngs: Buffer[] = [];
		if (s.images.wrist) output.image_wrist_path = s.images.wrist;
		if (s.images.camera) {
			output.image_cam_path = s.images.camera;
			pngs.push(readFileSync(s.images.camera));
		}
		if (s.images.wrist) pngs.push(readFileSync(s.images.wrist));
		return { output, pngs };
	}

	// ---- tools

	/** The recorded-state layer (../primitives/steps.ts): mutating tools record and return a fresh step. */
	const io: StepsIO<Step> = {
		steps,
		ready: () => env !== undefined,
		dump: dumpState,
		view,
		// A jammed gripper (polymetis: the fingers did not move as commanded, nothing grasped) heads the result.
		headline: (result) => {
			const jam = [result, result.gripper].find((r) => r?.gripper_jammed === true);
			return jam && { gripper_jammed: true, gripper_note: jam.note };
		},
	};

	/**
	 * Register a tool. Mutating tools run, then record a fresh state step and return it (errors
	 * included); read-only tools return their result or `{error}`,
	 * plus any PNGs in `_pngs`.
	 */
	function tool<P extends TSchema>(
		name: string,
		description: string,
		parameters: P,
		run: (p: Static<P>, signal: AbortSignal | undefined, ctx: ExtensionContext) => Promise<Json>,
		mutating = true,
	) {
		robot.tool(name, description, parameters, (params, signal, ctx) =>
			outcome(io, name, params as Json, () => run(params, signal, ctx), mutating),
		);
	}
	/** Mount a shared primitive (../primitives) as one of this robot's tools. */
	const mount = <P extends TSchema>(d: ToolDef<P>, mutating = true) =>
		tool(d.name, d.description, d.parameters, d.run, mutating);

	/** Refuse a move whose target leaves the --workspace-xy box or goes below --z-floor (unless it moves back in). */
	function checkWorkspace(delta: number[]) {
		const { box, floor } = workspaceLimits(flag("workspace-xy"), flag("z-floor"));
		const tcp = tcpPose(steps[steps.length - 1]);
		const outside = (p: number[]) =>
			(box ? Math.max(0, box[0] - p[0], p[0] - box[1]) + Math.max(0, box[2] - p[1], p[1] - box[3]) : 0) +
			Math.max(0, floor - p[2]);
		const target = tcp.slice(0, 3).map((v, i) => v + delta[i]);
		if (outside(target) > 1e-6 && outside(target) >= outside(tcp) - 1e-6)
			throw new Error(
				`the move ends at [${roundAll(target, 3)}], outside the workspace (${box ? `x ${box[0]}..${box[1]}, y ${box[2]}..${box[3]}, ` : ""}z >= ${floor} m; --workspace-xy / --z-floor)`,
			);
	}

	/** Units mode (../units): one grounded action unit on the existing move primitives. */
	function unitStep(move: Move, signal: AbortSignal | undefined) {
		return outcome(io, "act", { move }, async () => {
			check(signal);
			const out: Json = {};
			if (move.gripper) out.gripper = await motion("env.set_gripper", { open: move.gripper === "open" }, signal);
			if (Math.hypot(...move.delta) > 0) {
				checkMove(move.delta, maxMove(), setup?.task.constraints);
				checkWorkspace(move.delta);
				out.move = await motion(
					"env.move_delta",
					{ delta_xyz: NdArray.f32(move.delta), ...(chaining && move.continuous ? { continuous: true } : {}) },
					signal,
				);
			}
			if (move.yaw) {
				if (!(Math.abs(move.yaw) <= maxRotate()))
					throw new Error(`yaw ${round(move.yaw, 4)} rad exceeds the limit of ${maxRotate()} rad per call`);
				out.rotate = await motion("env.rotate_delta", { delta_rpy: NdArray.f32([0, 0, move.yaw]) }, signal);
			}
			return out;
		});
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
			gripper_commanded_open: base.gripper_commanded_open ?? null,
			table_z: Number(flag("z-floor")),
		};
	}

	const pixel = (description: string) => Type.Optional(Type.Integer({ minimum: 0, description }));

	mount(viewEnvState(io, "Read a Franka state snapshot and its synchronized RGB images."), false);
	mount(viewCameraMeta(io, "Read camera intrinsics, crop, depth, and calibration metadata."), false);

	function calibration() {
		if (!setup?.calibration) throw new Error(setup?.calibration_error ?? "no hand-eye calibration loaded");
		return setup.calibration;
	}

	tool(
		"view_perception_setup",
		"Read calibrated camera geometry and projection conventions.",
		Type.Object({ step: stepParam }),
		async ({ step = -1 }) => {
			const s = getStep(steps, step);
			if (!s.meta) throw new Error("camera metadata not found in the recorded state");
			const cal = calibration();
			const brief = (c: Calibration) => ({
				path: c.path,
				eye_on_hand: c.eye_on_hand,
				base_frame: c.base_frame,
				tracking_base_frame: c.tracking_base_frame,
				translation: roundAll([c.matrix[0][3], c.matrix[1][3], c.matrix[2][3]], 6),
				matrix_maps: "camera_frame_point -> base_frame_point",
			});
			return {
				camera_meta: s.meta,
				calibration: { external: brief(cal.external), wrist: brief(cal.wrist) },
				convention: cal.convention,
				current_policy:
					"back_project accepts a single pixel from one camera and returns the corresponding point in the robot base frame. The tool owns depth lookup, intrinsics, and hand-eye calibration; callers should not manually compute camera transforms.",
			};
		},
		false,
	);

	function tcpPose(s: Step): number[] {
		const pose = s.blob.state?.raw_base_state?.tcp_pose;
		if (!pose) throw new Error("recorded state is missing raw_base_state.tcp_pose");
		return vec(pose);
	}

	/** Observation key and camera name for a camera alias. */
	function resolveCamera(meta: Json, alias: "wrist" | "third_person"): ["main" | "extra_0", string | undefined] {
		const map = meta.observation_camera_map ?? {};
		const names = Object.keys(meta.cameras ?? {}).sort();
		if (alias === "third_person") return ["extra_0", map.extra_0 ?? names.find((n) => n !== map.main)];
		return ["main", map.main ?? names.find((n) => n.toLowerCase().includes("wrist"))];
	}

	/** Project one pixel through depth, intrinsics and hand-eye calibration. */
	function projectView(s: Step, alias: "wrist" | "third_person", row: number, col: number): Json {
		try {
			if (!s.meta) throw new Error("camera metadata not found in the recorded state");
			const cal = calibration();
			const [key, name] = resolveCamera(s.meta, alias);
			const cam = (name && s.meta.cameras?.[name]) || s.meta.cameras?.[key];
			if (!cam) throw new Error(`camera metadata not found for key='${key}', name='${name}'`);
			const K = cam.intrinsic_K as Mat;
			if (K?.length !== 3 || !K.flat().every(Number.isFinite))
				throw new Error(`intrinsic_K missing for camera ${name}`);
			const depthName = ARTIFACTS[key][1];
			const depthPath = join(s.dir, `${depthName}.f32`);
			let shape: { height: number; width: number };
			let depth: Float32Array;
			try {
				shape = JSON.parse(readFileSync(join(s.dir, `${depthName}.json`), "utf8"));
				depth = new Float32Array(new Uint8Array(readFileSync(depthPath)).buffer);
			} catch {
				throw new Error(`depth artifact not found: ${depthPath}`);
			}
			if (row < 0 || row >= shape.height || col < 0 || col >= shape.width)
				throw new Error(`pixel (${row},${col}) out of bounds for ${key} depth ${shape.height}x${shape.width}`);
			const z = depth[row * shape.width + col];
			if (!Number.isFinite(z) || z <= 0 || z > 10)
				throw new Error(`invalid depth ${z.toFixed(4)}m at ${key} pixel (${row},${col})`);
			const pointCamera = apply(inv3(K), [col, row, 1]).map((v) => v * z);
			const target = alias === "wrist" ? cal.wrist.matrix : cal.external.matrix;
			const pointTarget = apply(target, pointCamera);
			const pointBase = alias === "wrist" ? apply(pose7(tcpPose(s)), pointTarget) : pointTarget;
			return {
				camera_alias: alias,
				camera_key: key,
				camera_name: name,
				pixel: [row, col],
				depth_m: round(z),
				depth_path: depthPath,
				point_camera: roundAll(pointCamera),
				point_target: roundAll(pointTarget),
				point_base: roundAll(pointBase),
				target_frame: alias === "wrist" ? "tcp" : "base",
			};
		} catch (err) {
			return { camera_alias: alias, pixel: [row, col], error: message(err), error_type: "ValueError" };
		}
	}

	/** Mark the selected pixel on the step's image (selected_pixel.png). */
	function overlay(s: Step, key: "main" | "extra_0", row: number, col: number): string | undefined {
		try {
			const name = ARTIFACTS[key][0];
			const { width, height } = JSON.parse(readFileSync(join(s.dir, `${name}.json`), "utf8"));
			const rgb = mark({ width, height, rgb: readFileSync(join(s.dir, `${name}.rgb`)) }, row, col, [255, 0, 0]);
			const path = join(s.dir, "selected_pixel.png");
			writeFileSync(path, encodePng(rgb, width, height));
			if (!s.blob.artifacts.includes("selected_pixel.png")) s.blob.artifacts.push("selected_pixel.png");
			return path;
		} catch {
			return undefined;
		}
	}

	const cameraParam = Type.Optional(StringEnum(["wrist", "third_person"] as const, { description: "Default wrist" }));
	const idParam = Type.String({ description: "A detection id from segment (e.g. d3) of the current observation" });

	/** Base-frame point of a detection's centroid through the latest step's depth, or the reason there is none. */
	function locate(s: Step, alias: "wrist" | "third_person", d: Detection): Json {
		if (!d.centroid_rc) return { point_base: null, point_error: "empty mask" };
		const p = projectView(s, alias, d.centroid_rc[0], d.centroid_rc[1]);
		return p.point_base ? { point_base: p.point_base } : { point_base: null, point_error: p.error };
	}

	/** `invalidated` from a perception result: ids the env server dropped that we had not recorded yet. */
	function noteInvalidated(res: Json) {
		const ids = Array.isArray(res.invalidated)
			? res.invalidated.filter((v: unknown) => liveIds.includes(String(v)))
			: [];
		if (!ids.length) return;
		pi.appendEntry(DETECTIONS_ENTRY, { invalidated: ids, step: steps.length - 1, reason: "env server" });
		liveIds = liveIds.filter((id) => !ids.includes(id));
	}

	tool(
		"segment",
		"SAM3 masks on the latest camera image, each with a short id (d3) drawn on the returned overlay in rank order (red, blue, green, yellow, ...). Give exactly one of a text prompt or a positive point [row, col]. all=true returns every candidate instead of only the best; then pick one with select_detection. Ids die with the observation: segment again after any motion.",
		Type.Object({
			prompt: Type.Optional(Type.String()),
			point: Type.Optional(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 })),
			camera: cameraParam,
			min_score: Type.Optional(Type.Number({ description: "Default 0.2" })),
			all: Type.Optional(Type.Boolean({ description: "Return every mask, not only the best (default false)" })),
		}),
		async ({ prompt, point, camera = "wrist", min_score = 0.2, all = false }) => {
			// Models often fill both optional fields; a non-empty prompt wins.
			const text = prompt?.trim();
			if (!text && !point) return { error: "give a text prompt or a point [row, col]" };
			const alias = cameraAlias(camera);
			const s = getStep(steps, -1);
			const res = await call<Json>(
				"env.segment",
				{ camera: alias, ...(text ? { text_prompt: text } : { point }), min_score, all },
				120_000,
			);
			noteInvalidated(res);
			const detections: Detection[] = res.detections ?? [];
			liveIds = [...liveIds, ...detections.map((d) => d.id)];
			if (!res.found)
				return {
					found: false,
					camera: alias,
					error: res.reason ?? "no mask",
					fallback: "Pick pixels in the image and use back_project.",
				};
			const out: Json = {
				found: true,
				camera: alias,
				step: s.blob.step_idx,
				count: res.count,
				ids: res.ids,
				detections: detections.map((d) => ({
					id: d.id,
					score: d.score === null || d.score === undefined ? null : round(d.score, 3),
					box: d.box,
					area_px: d.area_px,
					centroid_pixel: d.centroid_rc,
					depth_m: d.depth_m,
					...locate(s, alias, d),
				})),
				coordinate_frame: "franka_base",
			};
			if (res.overlay instanceof NdArray) {
				const img = rgbOf(res.overlay);
				const png = encodePng(img.rgb, img.width, img.height);
				const n = String(res.observation).padStart(4, "0");
				writeFileSync(join(s.dir, `${alias}_segment_overlay_${n}.png`), png);
				out.overlay_path = join(s.dir, `${alias}_segment_overlay_${n}.png`);
				out._pngs = [png];
			}
			return out;
		},
		false,
	);

	tool(
		"select_detection",
		"Choose one segment mask (by id) as the target of the current observation, after checking the overlay.",
		Type.Object({ id: idParam }),
		async ({ id }) => {
			const res = await call<Json>("env.select_detection", { id });
			noteInvalidated(res);
			if (!res.ok) return { error: res.error, ids: res.ids, selected: res.selected, rejected: res.rejected };
			const { mask_png_base64: _mask, ...d } = res.detection as Detection;
			return {
				selected: res.selected,
				rejected: res.rejected,
				detection: { ...d, ...locate(getStep(steps, -1), cameraAlias(String(d.camera)), res.detection) },
			};
		},
		false,
	);

	tool(
		"reject_detection",
		"Rule out one segment mask (by id) of the current observation; it stays listed as rejected.",
		Type.Object({ id: idParam }),
		async ({ id }) => {
			const res = await call<Json>("env.reject_detection", { id });
			noteInvalidated(res);
			if (!res.ok) return { error: res.error, ids: res.ids, selected: res.selected, rejected: res.rejected };
			return { rejected: res.rejected, selected: res.selected, ids: res.ids };
		},
		false,
	);

	tool(
		"enhance_depth",
		"Fill the holes of a camera's depth on the latest step with a UniDepth estimate scaled to the sensor (or supply depth where the camera has none). back_project and segment on this step then use the filled depth; the sensor depth is kept as <camera>_depth_sensor.f32.",
		Type.Object({ camera: cameraParam }),
		async ({ camera = "wrist" }) => {
			const alias = cameraAlias(camera);
			const s = getStep(steps, -1);
			const res = await call<Json>("env.enhance_depth", { camera: alias }, 180_000);
			noteInvalidated(res);
			if (!(res.depth instanceof NdArray)) throw new Error("env.enhance_depth returned no depth map");
			const g = gridOf(res.depth);
			const name = ARTIFACTS[alias === "wrist" ? "main" : "extra_0"][1];
			try {
				if (!s.blob.artifacts.includes(`${name}_sensor.f32`)) {
					writeFileSync(join(s.dir, `${name}_sensor.f32`), readFileSync(join(s.dir, `${name}.f32`)));
					s.blob.artifacts.push(`${name}_sensor.f32`);
				}
			} catch {
				// No sensor depth for this camera: the estimate is the only depth there is.
			}
			writeFileSync(join(s.dir, `${name}.f32`), Buffer.from(g.data.buffer));
			writeFileSync(join(s.dir, `${name}.json`), JSON.stringify({ height: g.height, width: g.width }));
			if (!s.blob.artifacts.includes(`${name}.f32`)) s.blob.artifacts.push(`${name}.f32`);
			return {
				camera: alias,
				step: s.blob.step_idx,
				depth_path: join(s.dir, `${name}.f32`),
				report: res.report,
				estimate: res.estimate,
			};
		},
		false,
	);

	tool(
		"back_project",
		"Back-project one wrist or external-camera pixel into Franka base coordinates.",
		Type.Object({
			row: Type.Integer({ minimum: 0 }),
			col: Type.Integer({ minimum: 0 }),
			step: stepParam,
			camera: Type.Optional(StringEnum(["wrist", "third_person"] as const, { description: "Default wrist" })),
			debug: Type.Optional(Type.Boolean({ description: "Default false" })),
		}),
		async ({ row, col, step, camera = "wrist", debug = false }) => {
			const s = freshStep(step);
			if (!s.meta) throw new Error("camera metadata not found in the recorded state");
			calibration();
			const tcp = tcpPose(s);
			const alias = cameraAlias(camera);
			const p = projectView(s, alias, row, col);
			const selected = overlay(s, p.camera_key ?? (alias === "wrist" ? "main" : "extra_0"), row, col);
			if (!p.point_base) {
				const err: Json = {
					error: p.error ?? "back-projection failed",
					error_type: p.error_type ?? "ValueError",
					camera: alias,
					pixel: [row, col],
					step: s.blob.step_idx,
				};
				if (selected) err.selected_pixel_overlay = selected;
				return err;
			}
			const result: Json = {
				camera: alias,
				pixel: [row, col],
				point_base: p.point_base,
				world_xyz: p.point_base,
				coordinate_frame: "franka_base",
				depth_m: p.depth_m,
				step: s.blob.step_idx,
				source: "single_view_rgbd",
				source_artifact: p.depth_path,
				camera_key: p.camera_key,
				camera_name: p.camera_name,
			};
			if (selected) result.selected_pixel_overlay = selected;
			if (debug)
				result.debug = {
					point_camera: p.point_camera,
					point_target: p.point_target,
					target_frame: p.target_frame,
					tcp_pose_xyzw: roundAll(tcp),
					note: "Wrist points are transformed through the current robot TCP pose. Third-person points use the fixed external camera calibration.",
				};
			return result;
		},
		false,
	);

	type Pixels = {
		third_person_row: number | null;
		third_person_col: number | null;
		wrist_row: number;
		wrist_col: number;
	};

	function pair(v: unknown, name: string): [number, number] {
		if (!Array.isArray(v) || v.length !== 2) throw new Error(`${name} must be [row, col]`);
		const [r, c] = v.map(Number);
		if (!Number.isInteger(r) || !Number.isInteger(c)) throw new Error(`${name} must contain integer row/col values`);
		return [r, c];
	}

	function correspondence(s: Step, req: Pixels, debug: boolean): Json {
		const { third_person_row: tr, third_person_col: tc, wrist_row: wr, wrist_col: wc } = req;
		const wrist = projectView(s, "wrist", wr, wc);
		const third = tr !== null && tc !== null ? projectView(s, "third_person", tr, tc) : undefined;
		const pixels = { third_person: tr !== null && tc !== null ? [tr, tc] : null, wrist: [wr, wc] };
		if (!wrist.point_base)
			return {
				error: wrist.error ?? "wrist back-projection failed",
				error_type: wrist.error_type ?? "ValueError",
				source: "multi_view_rgbd",
				step: s.blob.step_idx,
				pixel_correspondence: pixels,
				warnings: warnings(third, wrist),
				diagnostics: {
					fusion_enabled: false,
					confidence: unavailable("wrist projection failed"),
					third_person: compact(third) ?? null,
					wrist: compact(wrist),
				},
			};
		let pointBase: number[] = wrist.point_base;
		let conf: Json = unavailable("third-person correspondence not provided");
		let source = "wrist_only";
		let delta: number | undefined;
		if (third?.point_base) {
			delta = dist(wrist.point_base, third.point_base);
			conf = confidence(delta);
			if (conf.level === "high" || conf.level === "medium") {
				pointBase = roundAll([0, 1, 2].map((k) => (wrist.point_base[k] + third.point_base[k]) * 0.5));
				source = "multi_view_fused";
			} else source = "wrist_with_low_confidence_third_person_check";
		}
		const result: Json = {
			point_base: pointBase,
			source,
			step: s.blob.step_idx,
			pixel_correspondence: pixels,
			confidence: conf,
			tcp_pose_source: `${caps.backend} raw_base_state.tcp_pose`,
			warnings: warnings(third, wrist),
			point_base_wrist: wrist.point_base,
		};
		if (third?.point_base) result.point_base_third_person = third.point_base;
		if (delta !== undefined) result.base_point_delta_m = round(delta);
		if (debug) {
			const t = pose7(tcpPose(s));
			result.tcp_pose_xyzw = roundAll(tcpPose(s));
			result.transforms = {
				T_base_to_tcp: t.map((r) => roundAll(r, 6)),
				note: "Wrist points are transformed through the current robot TCP pose. Third-person points are transformed through the fixed external camera calibration. point_base is fused only when the two base-frame estimates agree closely.",
			};
			result.diagnostics = {
				fusion_enabled: source === "multi_view_fused",
				confidence: conf,
				wrist: compact(wrist),
				third_person: compact(third) ?? null,
				...(delta !== undefined ? { base_point_delta_m: round(delta) } : {}),
			};
		}
		return result;
	}

	tool(
		"back_project_correspondence",
		"Fuse matched wrist and external-camera pixels into a Franka base point.",
		Type.Object({
			third_person_row: pixel("External-camera row"),
			third_person_col: pixel("External-camera column"),
			wrist_row: pixel("Wrist row"),
			wrist_col: pixel("Wrist column"),
			pixels: Type.Optional(
				Type.Array(Type.Object({}, { additionalProperties: true }), {
					description: "Several matches: [{wrist: [row, col], third_person: [row, col]}, ...]",
				}),
			),
			step: stepParam,
			debug: Type.Optional(Type.Boolean({ description: "Default false" })),
		}),
		async ({ third_person_row, third_person_col, wrist_row, wrist_col, pixels, step, debug = false }) => {
			const s = freshStep(step);
			if (!s.meta) throw new Error("camera metadata not found in the recorded state");
			calibration();
			tcpPose(s);
			let requests: Pixels[];
			if (pixels !== undefined) {
				if (!Array.isArray(pixels) || !pixels.length) throw new Error("pixels must be a non-empty list");
				requests = pixels.map((item: Json, i: number) => {
					let third = item.third_person;
					if (third === undefined && (item.third_person_row != null || item.third_person_col != null))
						third = [item.third_person_row, item.third_person_col];
					const [wr, wc] = pair(item.wrist ?? [item.wrist_row, item.wrist_col], `pixels[${i}].wrist`);
					const [tr, tc] = third == null ? [null, null] : pair(third, `pixels[${i}].third_person`);
					return { third_person_row: tr, third_person_col: tc, wrist_row: wr, wrist_col: wc };
				});
			} else {
				const missing = [wrist_row === undefined && "wrist_row", wrist_col === undefined && "wrist_col"].filter(
					Boolean,
				);
				if (missing.length) throw new Error(`single-point call is missing required fields: ${missing.join(", ")}`);
				if ((third_person_row === undefined) !== (third_person_col === undefined))
					throw new Error("third_person_row and third_person_col must be provided together");
				requests = [
					{
						third_person_row: third_person_row ?? null,
						third_person_col: third_person_col ?? null,
						wrist_row: wrist_row as number,
						wrist_col: wrist_col as number,
					},
				];
			}
			const points = requests.map((r) => correspondence(s, r, debug));
			if (pixels === undefined) return points[0];
			const valid = points.filter((p) => p.point_base);
			const reliable = valid.filter((p) => ["high", "medium"].includes(p.confidence?.level));
			const use = reliable.length ? reliable : valid;
			const scores = use.map((p) => p.confidence?.score).filter((v): v is number => typeof v === "number");
			const deltas = use.map((p) => p.base_point_delta_m).filter((v): v is number => typeof v === "number");
			const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
			const axis = (k: number) => use.map((p) => p.point_base[k] as number);
			const sorted = (v: number[]) => [...v].sort((a, b) => a - b);
			const med = (v: number[]) => {
				const x = sorted(v);
				return x.length % 2 ? x[(x.length - 1) / 2] : (x[x.length / 2 - 1] + x[x.length / 2]) / 2;
			};
			return {
				points,
				source: "multi_view_rgbd",
				step: s.blob.step_idx,
				count: points.length,
				valid_count: valid.length,
				reliable_count: reliable.length,
				aggregate: use.length
					? {
							point_base_mean: roundAll([0, 1, 2].map((k) => mean(axis(k)))),
							point_base_median: roundAll([0, 1, 2].map((k) => med(axis(k)))),
							confidence_score_mean: scores.length ? round(mean(scores), 3) : null,
							base_point_delta_mean_m: deltas.length ? round(mean(deltas)) : null,
						}
					: null,
				tcp_pose_source: `${caps.backend} raw_base_state.tcp_pose`,
			};
		},
		false,
	);

	/** The motion primitives (../primitives/motion.ts) on this arm: its env, limits, workspace and operator gate. */
	const rig: MotionRig = {
		check,
		motion,
		maxMove,
		maxRotate,
		constraints: () => setup?.task.constraints,
		workspace: (delta) => checkWorkspace(delta),
	};
	mount(moveDelta(rig, "Move the Franka TCP by a bounded base-frame xyz delta in meters."));
	mount(rotateDelta(rig, "Rotate the Franka TCP by a bounded base-frame rpy delta in radians."));
	mount(setGripper(rig, true, "Open the Franka gripper and wait for the command to settle."));
	mount(setGripper(rig, false, "Close the Franka gripper and wait for the command to settle."));

	tool(
		"vla_grasp",
		"Run bounded real-world VLA action chunks for a local grasp attempt.",
		Type.Object({
			prompt: Type.String(),
			max_chunks: Type.Optional(Type.Integer({ minimum: 1, maximum: 20, description: "Default 4" })),
		}),
		async ({ prompt, max_chunks = 4 }, signal) => {
			if (!caps.has_vla) throw new Error(`the ${caps.backend} backend has no VLA action space`);
			if (!vla) throw new Error("vla_grasp requires --robot-vla");
			if (!prompt.trim()) throw new Error("prompt must be non-empty");
			if (!(max_chunks >= 1 && max_chunks <= 20)) throw new Error("max_chunks must be between 1 and 20");
			const chunks: Json[] = [];
			let obs: Json | undefined;
			for (let i = 0; i < max_chunks; i++) {
				check(signal);
				obs ??= await observation();
				const main = obs.main_images;
				const states = obs.states;
				if (!(main instanceof NdArray) || main.shape.length !== 3)
					throw new Error(`expected [H,W,3] image, got shape [${main?.shape}]`);
				if (!(states instanceof NdArray) || states.shape.length !== 1)
					throw new Error(`states must be single-env shape [state_dim]; got [${states?.shape}]`);
				const wire = {
					main_images: u8(main).batched(),
					wrist_images: null,
					extra_view_images: batchViews(obs.extra_view_images),
					states: f32(states).batched(),
					task_descriptions: [prompt || setup?.task.instruction],
				};
				const actions = await vla.call<NdArray>("vla.predict", {}, 120_000, [wire, { mode: "eval" }], signal);
				const chunk = f32(new NdArray(actions.dtype, actions.shape.slice(1), actions.data));
				if (chunk.shape.length !== 2 || chunk.shape[0] < 1)
					throw new Error(`vla_grasp expected [chunk, action_dim] actions, got [${chunk.shape}]`);
				if (!numbers(chunk).every(Number.isFinite)) throw new Error("vla_grasp received non-finite VLA actions");
				check(signal);
				const result = await call("env.chunk_step", { actions: chunk, return_all_frames: false }, 300_000, signal);
				const next = result.observation;
				remember(Array.isArray(next) ? next.at(-1)?.states : next?.states);
				chunks.push(result);
				if (result.terminated || result.truncated) break;
				obs = next && !Array.isArray(next) && typeof next === "object" ? { ...next } : undefined;
			}
			return {
				ok: true,
				chunks_executed: chunks.length,
				last_chunk: chunks.at(-1) ?? null,
				robot_state: await robotState(),
			};
		},
	);

	// plan_grasp / plan_place / check_attached (../primitives/grasp.ts): the env server plans over its
	// calibrated RGB-D cameras; active with --graspnet/--graspgenx/--anyplace/--anygrasp.
	for (const d of graspTools(pi, {
		call: (method, kwargs, timeoutMs) => call(method, kwargs, timeoutMs ?? 120_000),
		cameras: ["wrist", "third_person"],
		task: () => setup?.task.instruction ?? "",
	}))
		tool(d.name, d.description, d.parameters, d.run, false);

	// ---- lifecycle

	async function startRobot(ctx: ExtensionContext) {
		if (!ctx.hasUI)
			throw new Error("franka drives a real robot: run pi interactively (or over RPC) so an operator is present");
		workspaceLimits(flag("workspace-xy"), flag("z-floor"));
		const r: Services = { root: flag("services"), python: flag("python", "python") };
		const configFlag = pi.getFlag("robot-config");
		const config = typeof configFlag === "string" && configFlag ? resolve(ctx.cwd, configFlag) : "";
		setup = await servicesJson<Setup>(r, SETUP_PY, [task(), config]);
		const stamp = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
		out = resolve(ctx.cwd, flag("out") || join(tmpdir(), "pi-embodied", `franka_t${task()}_${stamp}`));
		mkdirSync(out, { recursive: true });
		const endpoint = flag("robot-env");
		const backend = flag("robot-backend").trim().toLowerCase();
		if (backend && !BACKENDS[backend])
			throw new Error(`--robot-backend must be one of ${Object.keys(BACKENDS).join(", ")}, got "${backend}"`);
		const [envRpc, vlaRpc] = await Promise.all([
			endpoint
				? attach(endpoint)
				: robot.serve({
						python: r.python,
						args: [
							...["-m", BACKENDS[backend || "rlinf"]],
							...["--task-description", setup.task.instruction, ...(config ? ["--robot-config", config] : [])],
							...ikArgs(flag("ik")),
							...(flag("robot-sam3") ? ["--sam3", flag("robot-sam3")] : []),
							...(flag("robot-unidepth") ? ["--unidepth", flag("robot-unidepth")] : []),
							...graspArgs(pi),
						],
						cwd: r.root,
						env: servicesEnv(r),
						log: () => join(out, "franka_env_server.log"),
					}),
			flag("robot-vla") ? attach(flag("robot-vla")) : undefined,
		]);
		const meta = await envRpc.call<Json>("env.get_env_meta", {}, 30_000);
		caps = { backend: "rlinf", has_vla: true, ...(meta.capabilities ?? {}) };
		chaining = meta.smooth?.chaining === true;
		if (backend && caps.backend !== backend)
			throw new Error(`--robot-env serves the ${caps.backend} backend, not --robot-backend ${backend}`);
		if (!caps.has_vla && (vlaRpc || setup.task.name === "vla_grasp"))
			throw new Error(
				`the ${caps.backend} backend has no VLA action space: drop --robot-vla and use a task without vla_grasp`,
			);
		const go = await ctx.ui.confirm(
			`Reset the Franka arm (${caps.backend})?`,
			"The arm will move to its configured reset pose. Clear the workspace and keep the emergency stop in reach.",
		);
		if (!go) throw new Error("operator declined the reset; the Franka tools stay disabled");
		const reset = await envRpc.call<Json>("env.reset", {}, 180_000);
		env = envRpc;
		vla = vlaRpc;
		remember(reset.states);
		attemptStart = (await dumpState(null, null, null)).blob.step_idx;
		ctx.ui.notify(`Franka ready (${caps.backend}): task ${task()} (${setup.task.name}); steps under ${out}`, "info");
		// Perception tools exist only when the env server was started with --sam3 / --unidepth.
		return [
			...TOOLS.filter(
				(name) =>
					(name !== "vla_grasp" || caps.has_vla) &&
					(!(name in PERCEPTION_TOOLS) || caps.perception?.[PERCEPTION_TOOLS[name]] === true),
			),
			...graspActive(pi),
		];
	}
}
