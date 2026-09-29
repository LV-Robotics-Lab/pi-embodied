/**
 * Robosuite robot for pi: CaP-X's seven robosuite 1.5 tasks (Lift, Stack, Restack, Wipe,
 * NutAssemblySquare, TwoArmLift, TwoArmHandover) with Pandas under the OSC_POSE controller.
 *
 *   pi -e packages/embodied/src/robots/robosuite --task Lift --seed 0
 *   pi -e packages/embodied/src/robots/robosuite --task TwoArmLift --seed 3 --units=true
 *   pi -e packages/embodied/src/robots/robosuite --task Stack --seed 0 --privileged    (ground_truth_poses)
 *   pi -e packages/embodied/src/robots/robosuite --task Lift --seed 0 --code=true --code-api=low   (run_code, CaP-X's S3)
 *   pi -e packages/embodied/src/robots/robosuite --task Lift --seed 0 --code=true --privileged \
 *      --code-oracle lift_privileged                          (CaP-X's human oracle, no model: ./oracle)
 *
 * Starts one env server per session (services/.../robots/robosuite/env_server.py, the `robosuite`
 * venv: robosuite 1.5 conflicts with LIBERO's 1.4) and attaches to a running SAM3 server for
 * `segment`; the server's primitive registry (code.api) is recorded per episode. Motion tools are closed-loop Cartesian servos the server bounds (per-call travel
 * cap, workspace box, z floor); every motion result carries the task camera and the wrist view
 * at 512 px and the arms' state. Success is robosuite's `_check_success` (Restack adds CaP-X's
 * off-table rule), latched at its first step and recorded in `robot_result`. The two-arm tasks
 * take an `arm` on every motion tool (robot0 | robot1), like dual_franka's left | right.
 * --collect-flywheel-data records every control step of a motion (services robots/robosuite/flywheel.py).
 */

import { tmpdir } from "node:os";
import { join } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";
import { simDistil } from "../../capabilities/explore.ts";
import type { FlywheelObs, FlywheelSpec } from "../../capabilities/flywheel.ts";
import { MOLMO, SAM3 } from "../../infra/model-services.ts";
import { encodePng } from "../../infra/png.ts";
import { NdArray, type RpcClient } from "../../infra/rpc.ts";
import { finishMove, type Move, type MoveUnit, type Vec3 } from "../../modes/units/index.ts";
import { template } from "../../planner/context-version.ts";
import { detectionActive, detectionArgs, detectionTools, registerDetectionFlags } from "../../primitives/detections.ts";
import { geometryArgs, geometryTools, SERVO, splitImages } from "../../primitives/geometry.ts";
import { graspActive, graspArgs, graspTools, mountGraspTool, registerGraspFlags } from "../../primitives/grasp.ts";
import { ikArgs, type Reach, registerIkFlag } from "../../primitives/ik.ts";
import { pointActive, pointTool, registerPointFlags } from "../../primitives/pointing.ts";
import { attach, defineRobot, type Json, plain, rgbOf, SERVICES, toolResult } from "../../robot.ts";

const SYSTEM = template(new URL("./SYSTEM.md", import.meta.url));
const EXPLORE = template(new URL("./explore.md", import.meta.url));
const MEMORY = template(new URL("./memory.md", import.meta.url));

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
/** A recorded control step (`record`): the cameras at 256 px, the robot state, the composite action. */
type Step = Obs & { action: NdArray; success: boolean };
type Motion = {
	obs: Obs;
	info: Record<string, unknown> & { frames?: NdArray[]; steps?: Step[]; ok?: boolean; cancelled?: boolean };
};
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
	capabilities?: { perception?: { segment?: boolean; enhance_depth?: boolean } };
};
type WorldMap = { envStep: number; size: number; rgb: Buffer; xyz: Float32Array };
type Camera = "agentview" | "wrist";

const round = (v: number, d = 4) => Number(v.toFixed(d));
const num = (v: unknown) => (v instanceof NdArray ? v.toArray() : Array.isArray(v) ? v.map(Number) : [Number(v)]);

/** Size of a recorded step's cameras (env_server RECORD_SIZE). */
export const RECORD_SIZE = 256;
/** The services spec's space of a task (robots/robosuite/flywheel.py SPACES): its arms and gripper. */
export const flywheelSpace = (task: string) =>
	!hasGripper(task) ? "wipe" : arms(task).length > 1 ? "two_arm" : "one_arm";
/** Per arm: eef_pos, eef_quat (xyzw), the finger joints (a task with a gripper). */
export function flyState(task: string, o: Obs): number[] {
	return arms(task).flatMap((a) => [
		...num(o[`${a}_eef_pos`]),
		...num(o[`${a}_eef_quat`]),
		...(hasGripper(task) ? num(o[`${a}_gripper_qpos`]) : []),
	]);
}

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
	// --ik: the env server checks reach before every move, and preview_reach asks it (../ik.ts).
	registerIkFlag(pi);
	// --contact-graspnet / --graspgenx / --anyplace / --anygrasp: plan_grasp, plan_place and check_attached (../primitives/grasp.ts).
	registerGraspFlags(pi);
	// --detections / --unidepth: detect, select_detection, reject_detection, enhance_depth (../primitives/detections.ts).
	registerDetectionFlags(pi);
	// --point: Molmo's point over its --molmo server (../primitives/pointing.ts).
	registerPointFlags(pi, { molmo: true });
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
	let obs: Obs;
	let meta: Meta;
	let success = false;
	let successStep: number | null = null;
	let envStep = 0;
	let language = "";
	/** The gripper command units mode holds between units, per arm. */
	const grip = new Map<string, "open" | "close">();
	const worldMaps = new Map<string, WorldMap>();

	/** The memory cell of this task at `seed`. */
	const tag = (seed: string) => `robosuite_${robot.task.task}_s${seed}`;
	const twoArm = () => TWO_ARM.includes(robot.task.task as Task);
	/** The task's cameras, robot state and composite action; their widths are the task's space. */
	const FLYWHEEL: FlywheelSpec = {
		robot: "robosuite",
		get space() {
			return flywheelSpace(robot.task.task);
		},
		images: { agentview_images: [RECORD_SIZE, RECORD_SIZE, 3], wrist_images: [RECORD_SIZE, RECORD_SIZE, 3] },
		get state() {
			return arms(robot.task.task).length * (hasGripper(robot.task.task) ? 9 : 7);
		},
		get action() {
			return arms(robot.task.task).length * (hasGripper(robot.task.task) ? 7 : 6);
		},
	};
	const flyObs = (o: Obs): FlywheelObs => ({
		images: { agentview_images: o.agentview, wrist_images: o.wrist },
		state: flyState(robot.task.task, o),
	});
	const robot = defineRobot(pi, {
		name: "robosuite",
		// Tools and code primitives: ../../primitives/manifests/robosuite.json (the env server reads it too).
		manifest: "robosuite",
		vars: () => ({
			image_size: IMAGE_SIZE,
			cameras: ["agentview", "wrist"],
			max_move: flag("max-move", String(MAX_MOVE_M)),
			// `arm` exists on the two-arm tasks only (the one-arm tools and primitives leave it out).
			arms: robot?.task.task && twoArm() ? [...ARMS] : [],
		}),
		capabilities: (c) =>
			({
				sam3: Boolean(flag("sam3", "")),
				ik: Boolean(flag("ik", "")),
				grasp: graspActive(pi).length > 0,
				place: Boolean(flag("anyplace", "")),
				geometry: pi.getFlag("geometry") === true && !twoArm(),
				unidepth: Boolean(String(pi.getFlag("unidepth") ?? "").trim()),
				fingers: hasGripper(robot.task.task),
			})[c] ?? false,
		services: { models: [SAM3, MOLMO] },
		task: ["task", "seed"],
		keepImages: 4,
		video: true,
		groundTruth: (names) => call("env.ground_truth_poses", { names: names ?? null }),
		// Observations carry the task camera, then the wrist view.
		vdm: { views: 2, wrist: 1 },
		// No corpus is published for robosuite: memory is what exploration writes locally, one cell per task and seed.
		memory: {
			cell: () => ({ tag: tag(robot.task.seed), reference: tag("0") }),
			primitives: ["move_to", "move_delta", "set_gripper", "act"],
			published: false,
		},
		explore: {
			// The exploration `reset` tool is not a robot.tool, so robot.signal is unset here; use its own signal.
			reset: async (result, _ctx, signal) => {
				grip.clear();
				const [o] = await env.call<[Obs, unknown]>("env.reset", {}, 600_000, [], signal);
				absorb(o);
				await flyReset(o, signal);
				return observe({ ...result, reset: true });
			},
			prompt: () => EXPLORE.replaceAll("{{task}}", robot.task.task).replaceAll("{{seed}}", robot.task.seed),
			// The DISTIL pass once the cell is solved: the suite draft and lessons the memory corpus is merged from.
			distil: () =>
				simDistil({
					suite: "robosuite",
					task: robot.task.task,
					seed: robot.task.seed,
					auditFields: "task, seed, success",
				}),
			rewrite: [
				[
					/This is a single episode\. You may recover within it \(re-position, re-grasp\), but you cannot restart it\./,
					"This is an exploration run: `reset` starts a fresh attempt (see Exploration). Within an attempt, recover in place (re-position, re-grasp).",
				],
			],
		},
		flywheel: { spec: FLYWHEEL, select: () => robot.task.task },
		start: startEpisode,
		prompt: () =>
			SYSTEM.replaceAll("{{task_language}}", language)
				.replaceAll("{{memory}}", pi.getFlag("explore") === true ? "" : robot.mem!.render(MEMORY).trim())
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
		// Code mode (../code): the env server runs the program against that registry; the result
		// carries the control steps, the latched success, the new observation and the video frames.
		code: {
			rpc: () => env,
			instruction: () => language,
			observe: async (r) => {
				for (const f of (r.frames as NdArray[] | undefined) ?? []) video.frame(f);
				if (r.obs) absorb(r.obs as Obs);
				return observe({ name: "run_code", status: r.status, env_steps: Number(r.steps) || 0 });
			},
		},
		units: {
			vectors: VECTORS,
			stepM: STEP_M,
			yawStepRad: YAW_STEP_RAD,
			// Read at session start, after pi has set --task: the two-arm tasks give `act` its `arm`.
			arms: () => (twoArm() ? ARMS : undefined),
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
			// The wrist camera, the second image (one- and two-arm tasks).
			wrist: true,
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
	const fly = robot.fly!;
	/**
	 * Start a Flywheel episode at a reset (raw/robosuite/<task>/seed_NNN, services robots/robosuite/flywheel.py):
	 * the reset rendered at 256 px. Only under --collect-flywheel-data.
	 */
	async function flyReset(o: Obs, signal?: AbortSignal) {
		if (!pi.getFlag("collect-flywheel-data")) return;
		const { task, seed } = robot.task;
		const shot = (camera: string) =>
			call<NdArray>(
				"env.render_camera",
				{ camera_name: camera, height: RECORD_SIZE, width: RECORD_SIZE },
				[],
				signal ?? robot.signal,
			);
		const first = { ...o, agentview: await shot("agentview"), wrist: await shot("wrist") };
		fly.reset(flyObs(first), {
			path: [task, `seed_${seed.padStart(3, "0")}`],
			metadata: { task, seed: Number(seed), space: flywheelSpace(task), task_language: language },
		});
	}

	/** Every robot RPC carries the running tool's abort signal, so an abort stops motion between calls. */
	const call = <T = unknown>(
		method: string,
		kwargs: Record<string, unknown> = {},
		args: unknown[] = [],
		signal = robot.signal,
		timeoutMs = 300_000,
	) => env.call<T>(method, kwargs, timeoutMs, args, signal);
	/** A motion's kwargs: while the Flywheel records, the server returns every control step. */
	const rec = (kwargs: Record<string, unknown>) => (fly.recording ? { ...kwargs, record: true } : kwargs);

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
		for (const s of r.info.steps ?? [])
			fly.transition(s.action.toArray(), flyObs(s), s.success ? 1 : 0, s.success, Boolean(s.truncated));
		const { frames: _frames, steps: _steps, ...info } = r.info;
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
			// Exploration's and the memory recipe's success signal (../explore.ts, ../memory).
			terminated: success,
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

	const arm = Type.Optional(StringEnum(ARMS, { description: "Two-arm tasks: which arm (required there)" }));

	tool(
		"view_env_state",
		`Current state with the task camera (global layout) and wrist (close range) ${IMAGE_SIZE}x${IMAGE_SIZE} images. Pixels (row, col) in these images feed back_project.`,
		Type.Object({}),
		async () => ({}),
	);

	// The motion tools are the server's methods with the manifest's parameters (manifests/robosuite.json).
	tool("move_to", "", Type.Object({}), async (params: Json, signal) => {
		checkMove(params.xyz, num(obs[`${armOf(params.arm) ?? "robot0"}_eef_pos`]));
		return {
			name: "move_to",
			...motion(await call<Motion>("env.move_to", rec({ ...params, arm: armOf(params.arm) ?? null }), [], signal)),
		};
	});

	tool("move_delta", "", Type.Object({}), async (params: Json, signal) => {
		checkMove(params.delta_xyz, [0, 0, 0]);
		return {
			name: "move_delta",
			...motion(
				await call<Motion>("env.move_delta", rec({ ...params, arm: armOf(params.arm) ?? null }), [], signal),
			),
		};
	});

	tool("set_gripper", "", Type.Object({}), async (params: Json, signal) => ({
		name: "set_gripper",
		...motion(await call<Motion>("env.set_gripper", rec({ ...params, arm: armOf(params.arm) ?? null }), [], signal)),
	}));

	// --geometry (one-arm tasks): view_points, mark_point, move_grip (../primitives/geometry.ts). The env
	// server resolves move_grip's target; its move_to (per-call cap, workspace box, z floor) servos the
	// TCP to the target's position and quat_xyzw.
	const geometry = geometryTools(
		pi,
		{
			call: (method, kwargs, timeoutMs) =>
				call<Record<string, unknown>>(method, kwargs, [], robot.signal, timeoutMs),
			cameras: ["agentview", "wrist"],
			execute: async (plan, signal) => {
				const out: Record<string, unknown> = { name: "move_grip", target: plan.target };
				if (plan.preview_id) out.preview_id = plan.preview_id;
				let status = "not_requested";
				if (plan.motion) {
					checkMove(plan.target.grip_xyz_m, num(obs.robot0_eef_pos));
					const info = motion(
						await call<Motion>(
							"env.move_to",
							{
								arm: null,
								quat_xyzw: plan.target.tool_quat_xyzw,
								tol_m: SERVO.tolM,
								tol_rad: SERVO.tolRad,
								max_steps: SERVO.maxSteps,
							},
							[plan.target.grip_xyz_m],
							signal,
						),
					);
					status = info.ok ? "reached" : "not_reached";
					out.steps_used = info.steps_used;
				}
				if (plan.gripper) {
					if (status === "not_reached") out.gripper_skipped = "the motion did not reach its target";
					else {
						motion(await call<Motion>("env.set_gripper", { arm: null }, [plan.gripper], signal));
						out.gripper = plan.gripper;
					}
				}
				const target = plan.motion
					? {
							target_xyz: plan.target.grip_xyz_m,
							target_approach: plan.target.approach_world,
							target_jaw: plan.target.jaw_world,
						}
					: {};
				const { rest, pngs } = splitImages(await call<Record<string, unknown>>("env.grip_state", target));
				delete rest.motion_status;
				const shown = observe({ ...out, motion_status: status, ...rest });
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
		// The server sends K and the extrinsic as numpy arrays: nested lists here.
		const cam = plain(
			await call<CameraMeta>("env.get_camera_meta", { camera_name: camera, height: size, width: size }),
		) as CameraMeta;
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
		"get_camera_meta",
		"",
		Type.Object({}),
		async (params: Json) => ({
			camera: params.camera_name ?? "agentview",
			meta: await call("env.get_camera_meta", {
				camera_name: params.camera_name ?? "agentview",
				height: params.height ?? IMAGE_SIZE,
				width: params.width ?? IMAGE_SIZE,
			}),
		}),
		"read",
	);

	tool(
		"segment",
		"",
		Type.Object({}),
		async (params: Json) => {
			const {
				mask: _mask,
				overlay_png_base64,
				...rest
			} = await call<Json>("env.segment", params, [], robot.signal, 120_000);
			return { ...rest, ...(overlay_png_base64 ? { _image: Buffer.from(overlay_png_base64, "base64") } : {}) };
		},
		"read",
	);

	tool("back_project", "", Type.Object({}), async (params: Json) => call<Json>("env.back_project", params), "read");

	// CaP-X's high tier (manifest tier high; activated by --api high, PARAMS.md 4): the semantic functions run on the
	// server; --privileged answers get_object_pose from the simulator.
	tool(
		"get_object_pose",
		"",
		Type.Object({}),
		async (params: Json) => ({
			pose: await call(
				pi.getFlag("privileged") === true ? "env.get_object_pose_privileged" : "env.get_object_pose",
				params,
				[],
				robot.signal,
				120_000,
			),
		}),
		"read",
	);
	tool(
		"sample_grasp_pose",
		"",
		Type.Object({}),
		async (params: Json) => ({
			grasp: await call(
				pi.getFlag("privileged") === true ? "env.sample_grasp_pose_privileged" : "env.sample_grasp_pose",
				{ ...params, arm: armOf(params.arm) ?? null },
				[],
				robot.signal,
				600_000,
			),
		}),
		"read",
	);
	for (const name of ["goto_pose", "home_pose", "open_gripper", "close_gripper"] as const)
		tool(name, "", Type.Object({}), async (params: Json, signal) => ({
			name,
			...motion(await call<Motion>(`env.${name}`, { ...params, arm: armOf(params.arm) ?? null }, [], signal)),
		}));

	tool(
		"preview_reach",
		"",
		Type.Object({}),
		async (params: Json) => call<Reach>("env.preview_reach", { ...params, arm: armOf(params.arm) ?? null }),
		"read",
	);

	// plan_grasp / plan_place / check_attached over the env server's grasp planner (active with a backend flag).
	for (const d of graspTools(pi, {
		call: (method, kwargs, timeoutMs) => call<Json>(method, kwargs, [], robot.signal, timeoutMs ?? 120_000),
		cameras: ["agentview", "wrist"],
		task: () => language,
		arm: { schema: arm, name: (v) => armOf(v as string | undefined) ?? "robot0" },
	}))
		mountGraspTool(robot.tool, d);

	// Molmo pointing on the current images (active with --point).
	mountGraspTool(
		robot.tool,
		pointTool(pi, {
			cameras: ["agentview", "wrist"],
			frame: async (c) => {
				const a = obs[c as Camera];
				return rgbOf(a);
			},
			locate: async (c, row, col) => {
				const map = await worldMap(c as Camera, IMAGE_SIZE);
				const i = (row * IMAGE_SIZE + col) * 3;
				const p = [map.xyz[i], map.xyz[i + 1], map.xyz[i + 2]];
				return valid(p) ? { world_xyz: p.map((v) => round(v)) } : undefined;
			},
			signal: () => robot.signal,
		}),
	);

	// SAM3 masks with ids and UniDepth over the env server's perception (active with --detections / --unidepth).
	for (const d of detectionTools(pi, {
		call: (method, kwargs, timeoutMs) => call<Json>(method, kwargs, [], robot.signal, timeoutMs ?? 120_000),
		cameras: ["agentview", "wrist"],
		// The server renders the same 512 px views: the centroid's world xyz through this step's world map.
		locate: async (camera, d) => {
			const [row, col] = (d.centroid_rc as number[] | null) ?? [];
			if (row === undefined) return {};
			const map = await worldMap(camera as Camera, IMAGE_SIZE);
			const i = (row * IMAGE_SIZE + col) * 3;
			const p = [map.xyz[i], map.xyz[i + 1], map.xyz[i + 2]];
			return valid(p) ? { centroid_world_xyz: p.map((v) => round(v)) } : {};
		},
	}))
		mountGraspTool(robot.tool, d);

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
			info = motion(await call<Motion>("env.set_gripper", rec({ arm: a ?? null }), [move.gripper], signal));
			steps += Number(info.steps_used ?? 0);
		}
		const turning = Boolean(move.yaw) || Boolean(move.rot?.some(Boolean));
		if (Math.hypot(...move.delta) > 0 || turning) {
			const rotvec = move.rot?.some(Boolean) ? move.rot : move.yaw ? [0, 0, move.yaw] : undefined;
			info = motion(
				await call<Motion>(
					"env.move_delta",
					rec({ arm: a ?? null, ...(rotvec ? { rotvec } : {}), tol_m: 0.004, max_steps: 40 }),
					[move.delta],
					signal,
				),
			);
			steps += Number(info.steps_used ?? 0);
		}
		if (!move.gripper && !turning && !Math.hypot(...move.delta)) {
			// STOP: hold the setpoint for one step (a zero move).
			info = motion(
				await call<Motion>("env.move_delta", rec({ arm: a ?? null, max_steps: 1 }), [[0, 0, 0]], signal),
			);
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
		if (TWO_ARM.includes(task as Task) && pi.getFlag("geometry") === true)
			throw new Error(`--geometry needs a one-arm task; ${task} has two arms`);
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
					...ikArgs(pi.getFlag("ik")),
					...graspArgs(pi),
					// The server takes --sam3 for its own segment already.
					...detectionArgs(pi, ""),
					...(TWO_ARM.includes(task as Task) ? [] : geometryArgs(pi)),
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
		await flyReset(o);
		return [
			"view_env_state",
			"get_camera_meta",
			"segment",
			"back_project",
			"move_to",
			"move_delta",
			...(hasGripper(task) ? ["set_gripper"] : []),
			// preview_reach requires --ik (the manifest drops it without).
			"preview_reach",
			...detectionActive(pi, meta.capabilities?.perception),
			...pointActive(pi),
			// Grasping needs fingers: Wipe's sponge has none.
			...(hasGripper(task) ? graspActive(pi) : []),
			...(TWO_ARM.includes(task as Task) ? [] : geometry()),
			"finish",
		];
	}
}
