/**
 * BEHAVIOR-1K robot for pi: an R1Pro (holonomic base, torso, two arms with parallel grippers, a ZED
 * head camera and a RealSense on each wrist) on one 2025-challenge task in OmniGibson (Isaac Sim).
 *
 *   pi -e packages/embodied/src/robots/behavior --task turning_on_radio --seed 0
 *   pi -e packages/embodied/src/robots/behavior --task picking_up_trash --seed 2 --privileged
 *   pi -e packages/embodied/src/robots/behavior --task turning_on_radio --seed 0 --code=true
 *      (run_code over the server's registry; a primitive takes minutes: --code-timeout defaults to 900 s)
 *
 * Starts one BEHAVIOR env server per session (services/.../robots/behavior/env_server.py in the
 * `behavior` venv, see robots/behavior/install.sh; OmniGibson loads a whole house, minutes). The
 * tools are OmniGibson's semantic primitives as CaP-X's R1ProControlApi exposed them
 * (navigate_to_pose, move_hand, grasp_object, open/close_gripper, get_robot_position), the
 * perception tools (segment via SAM3, point via Molmo, back_project through the cameras' metric
 * depth; the env server runs them) and view_env_state; every motion result carries the head and
 * both wrist images. Schemas and descriptions: ../../primitives/manifests/behavior.json, which the
 * env server's code.api reads too. Success is
 * the BDDL task's `success`, and `q_score` (BEHAVIOR's partial credit) goes into `robot_result`;
 * what only the simulator knows (the object OmniGibson's grasping holds) reaches the planner only
 * under --privileged, and CaP-X's "picked" judgement is recorded as a reference field, never as
 * success. --seed is the task's pre-sampled instance id.
 *
 * Copyright 2026 The CaP-X Authors (github.com/capgym/cap-x @53e9966). Licensed under the Apache
 * License, Version 2.0. Modified by pi-embodied: capx/integrations/r1pro/control.py's primitive set
 * ported as pi tools; the reward of capx/envs/simulators/r1pro_b1k.py is not.
 */

import { tmpdir } from "node:os";
import { join } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { cudaDevice, python, service, servicesDir } from "../../infra/config.ts";
import { MOLMO, SAM3 } from "../../infra/model-services.ts";
import { trackFlags } from "../../infra/params.ts";
import { encodePng } from "../../infra/png.ts";
import type { NdArray, RpcClient } from "../../infra/rpc.ts";
import type { Move, MoveUnit, Vec3 } from "../../modes/units/index.ts";
import { template } from "../../planner/context-version.ts";
import { detectionActive, detectionArgs, detectionTools, registerDetectionFlags } from "../../primitives/detections.ts";
import { mountGraspTool } from "../../primitives/grasp.ts";
import { attach, defineRobot, type Json, round, toolResult } from "../../robot.ts";

const SYSTEM = template(new URL("./SYSTEM.md", import.meta.url));
const EXPLORE = template(new URL("./explore.md", import.meta.url));

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
/** Code mode's default --code-max-move, m: four of navigate_to_pose's 5 m drives. */
export const CODE_MAX_MOVE_M = 20;
/** Code mode: the default --code-timeout, s. */
export const CODE_TIMEOUT_S = 900;

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
type Meta = {
	task: string;
	seed: number;
	instruction: string;
	image_size: number;
	grasping_mode: string;
	capabilities?: { perception?: { segment?: boolean; enhance_depth?: boolean } };
};

/**
 * Units (../units, --units) run on `env.move_hand_delta`: each MV_* is a 2 cm step in the robot BASE
 * frame (+x ahead of the base, +y to its left, +z up), so it looks the same in the head camera wherever
 * the base stands; the server turns it into the world by the base yaw.
 */
export const VECTORS: Record<MoveUnit, Vec3> = {
	MV_FWD: [1, 0, 0],
	MV_BACK: [-1, 0, 0],
	MV_LEFT: [0, 1, 0],
	MV_RIGHT: [0, -1, 0],
	MV_UP: [0, 0, 1],
	MV_DOWN: [0, 0, -1],
};
export const STEP_M = 0.02;
export const YAW_STEP_RAD = 0.15;
/** env_server MAX_HAND_STEP_M / MAX_HAND_YAW_RAD: one move_hand_delta call's limits. */
export const MAX_HAND_STEP_M = 0.1;
export const MAX_HAND_YAW_RAD = 0.3;
export const UNITS_VIEWS = `Each result shows the head camera, then the left wrist view, then the right wrist view; every unit names the arm it moves.
- Head camera (first image): on the robot's head looking ahead and down at the workspace. MV_FWD moves the gripper away from the robot (up the image, smaller), MV_BACK toward it, MV_LEFT toward the image left and MV_RIGHT toward the image right; MV_UP / MV_DOWN raise and lower it.
- Wrist views: close range from each gripper; judge contact there, direction in the head camera.`;

export default function behavior(pi: ExtensionAPI) {
	// Every flag this robot registers is tracked: numbers fail closed, the result records them (../../infra/params.ts).
	trackFlags(pi);
	const flag = (name: string, fallback: string) => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("task", {
		type: "string",
		default: TASKS[0],
		description: `BEHAVIOR-1K challenge activity: ${TASKS.slice(0, 3).join(", ")}, ... (50)`,
	});
	pi.registerFlag("seed", { type: "string", default: "0", description: "The task's pre-sampled instance id" });
	pi.registerFlag("image-size", { type: "string", default: "480", description: "Camera frames, px (square)" });
	pi.registerFlag("grasping-mode", {
		type: "string",
		default: "sticky",
		description: "OmniGibson grasping: sticky (CaP-X) or assisted",
	});
	pi.registerFlag("env-url", {
		type: "string",
		description: "Attach to a running env server instead of starting one",
	});
	// --detections / --depth unidepth: detect, select_detection, reject_detection, enhance_depth (../primitives/detections.ts).
	registerDetectionFlags(pi);

	let env: RpcClient;
	let obs: Obs;
	let meta: Meta;
	const privileged = () => pi.getFlag("privileged") === true;

	/** The memory cell of this task at instance `seed`. */
	const tag = (seed: string) => `behavior_${robot.task.task}_s${seed}`;
	const robot = defineRobot(pi, {
		name: "behavior",
		// Tools and code primitives: ../../primitives/manifests/behavior.json (the env server reads it too).
		manifest: "behavior",
		vars: () => ({ cameras: [...CAMERAS] }),
		// Must agree with the env server's `_has` (it gets --sam3 / --molmo / --unidepth from the deployment's services).
		capabilities: (c) =>
			({
				sam3: Boolean(service(pi, "sam3")),
				molmo: Boolean(service(pi, "molmo")),
				unidepth: Boolean(String(pi.getFlag("depth") ?? "").trim()),
			})[c] ?? false,
		services: { models: [SAM3, MOLMO] },
		task: ["task", "seed"],
		// The env server's code.api (its manifest's digest and what this run has), recorded per episode.
		codeApi: () => env,
		// Code mode (../code): the env server runs the program against that registry; the result
		// carries the control steps, the latched success, the new observation and the head frames.
		code: {
			rpc: () => env,
			instruction: () => meta.instruction,
			// A house: one navigate_to_pose may drive 5 m.
			maxMoveM: CODE_MAX_MOVE_M,
			// One primitive (a navigate, a grasp) takes minutes: programs get 15 minutes by default.
			timeoutS: CODE_TIMEOUT_S,
			// Like the motion tools (`ended`): nothing runs once the episode is over.
			refuse: () =>
				obs?.truncated
					? "the episode is over (max steps)"
					: obs?.success
						? "the task is already solved; call finish"
						: undefined,
			observe: async (r) => {
				for (const f of (r.frames as NdArray[] | undefined) ?? []) video.frame(f);
				if (r.obs) absorb(r.obs as Obs);
				return observe({ name: "run_code", status: r.status, env_steps: Number(r.steps) || 0 });
			},
		},
		keepImages: 6,
		video: true,
		// Observations carry the head image, then the left and right wrist images.
		vdm: { views: 3, wrist: [1, 2] },
		// Show-Harness action units on the server's small base-frame hand step (env.move_hand_delta).
		units: {
			vectors: VECTORS,
			stepM: STEP_M,
			yawStepRad: YAW_STEP_RAD,
			maxYawRad: () => MAX_HAND_YAW_RAD,
			maxMoveM: () => MAX_HAND_STEP_M,
			arms: ARMS,
			apply: (move, signal) => unitStep(move, signal),
			state: async (arm) => {
				const a = (arm ?? "right") as (typeof ARMS)[number];
				const base = obs.base_pos.toArray();
				const e = obs.eef[a].pos.toArray();
				const [c, sn] = [Math.cos(obs.base_yaw), Math.sin(obs.base_yaw)];
				const [dx, dy] = [e[0] - base[0], e[1] - base[1]];
				return {
					eef_xyz: [c * dx + sn * dy, -sn * dx + c * dy, e[2] - base[2]].map((v) => round(v, 4)),
					gripper_width: round(obs.eef[a].gripper_width, 4),
				};
			},
			instruction: () => meta?.instruction ?? "",
			views: UNITS_VIEWS,
			// The wrist views are not calibrated to the units' directions: the wrist-judged plugins stay off.
			wrist: false,
			emptyWidthM: 0.004,
		},
		groundTruth: (names) => env.call("env.ground_truth_poses", { names: names ?? null }, 120_000, [], robot.signal),
		// No corpus is published for BEHAVIOR: memory is what exploration writes locally, one cell per task instance.
		memory: {
			cell: () => ({ tag: tag(robot.task.seed), reference: tag("0") }),
			primitives: ["navigate_to_pose", "move_hand", "grasp_object", "open_gripper", "close_gripper"],
			published: false,
		},
		explore: {
			// The exploration `reset` tool is not a robot.tool, so robot.signal is unset here; use its own signal.
			reset: async (result, _ctx, signal) => {
				const [o] = await env.call<[Obs, unknown]>("env.reset", {}, 600_000, [], signal);
				absorb(o);
				return observe({ ...result, reset: true });
			},
			prompt: () => EXPLORE.replaceAll("{{task}}", robot.task.task).replaceAll("{{seed}}", robot.task.seed),
			rewrite: [
				[
					/This is a single episode with a time limit\. You may recover within it \(re-navigate, re-grasp\), but you cannot restart it\./,
					"This is an exploration run: `reset` starts a fresh attempt (see Exploration). Within an attempt, recover in place (re-navigate, re-grasp).",
				],
			],
			// A reset reloads the whole house (minutes): fewer, longer attempts.
			budget: { sessions: 2, attempts: 3 },
		},
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
			// Exploration's and the memory recipe's success signal (../explore.ts, ../memory).
			terminated: obs.success,
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

	const ended = (): ReturnType<typeof observe> | undefined => {
		if (obs.truncated) return observe({ error: "the episode is over (max steps)" });
		if (obs.success) return observe({ error: "the task is already solved; call finish" });
		return undefined;
	};

	robot.tool("view_env_state", "", Type.Object({}), async () => observe({}));

	// The manifest's tools run the server's methods with its parameters (manifests/behavior.json).
	robot.tool("get_robot_position", "", Type.Object({}), async () =>
		toolResult(await env.call<Json>("env.get_robot_position", {}, 60_000, [], robot.signal)),
	);

	for (const name of ["navigate_to_pose", "move_hand", "grasp_object", "open_gripper", "close_gripper"])
		robot.tool(
			name,
			"",
			Type.Object({}),
			async (params: Json, signal) => ended() ?? observe(await motion(`env.${name}`, params, signal)),
		);

	/** A perception read on the server: its numbers, and the picture it drew (segment's overlay, point's mark). */
	async function perceive(method: string, params: Json) {
		const {
			mask: _mask,
			overlay_png_base64,
			...rest
		} = await env.call<Json>(method, params, 180_000, [], robot.signal);
		return toolResult(rest, overlay_png_base64 ? [Buffer.from(String(overlay_png_base64), "base64")] : []);
	}
	robot.tool("segment", "", Type.Object({}), async (params: Json) => perceive("env.segment", params));
	robot.tool("point", "", Type.Object({}), async (params: Json) => perceive("env.point", params));
	robot.tool("back_project", "", Type.Object({}), async (params: Json) => perceive("env.back_project", params));

	// SAM3 masks with ids and UniDepth over the env server's perception (active with --detections / --depth unidepth).
	for (const d of detectionTools(pi, {
		call: (method, kwargs, timeoutMs) =>
			env.call<Record<string, any>>(method, kwargs, timeoutMs ?? 120_000, [], robot.signal),
		cameras: ["head", "left_wrist", "right_wrist"],
	}))
		mountGraspTool(robot.tool, d);

	/** One action unit (../units): a gripper command and a base-frame step / yaw of one arm, in one server call. */
	async function unitStep(move: Move, signal: AbortSignal | undefined) {
		if (move.rot?.some(Boolean)) throw new Error("the R1Pro's units turn only about the vertical (ROTATE_*)");
		const arm = move.arm ?? "right";
		const result = await motion(
			"env.move_hand_delta",
			{ arm, delta_xyz: move.delta, yaw: move.yaw ?? 0, gripper: move.gripper ?? null },
			signal,
		);
		return observe({ name: "act", ...result });
	}

	async function startEpisode() {
		const { task, seed } = robot.task;
		if (!TASKS.includes(task as Task))
			throw new Error(`--task ${task} is not a BEHAVIOR-1K challenge task; one of: ${TASKS.join(", ")}`);
		if (!/^\d+$/.test(seed))
			throw new Error(`--seed must be a task instance id (a non-negative integer), got ${seed}`);
		const endpoint = pi.getFlag("env-url") as string | undefined;
		if (endpoint) env = await attach(endpoint);
		else {
			const services = servicesDir(pi);
			env = await robot.serve({
				python: python(pi, "behavior"),
				args: [
					...["-m", "pi_embodied_services.robots.behavior.env_server"],
					...["--task", task, "--seed", seed],
					// Unset: the server resolves the GPU as every env server does (utils/gpu.py).
					...(cudaDevice(pi) ? ["--gpu-id", cudaDevice(pi)] : []),
					...["--image-size", flag("image-size", "480"), "--grasping-mode", flag("grasping-mode", "sticky")],
					// The server runs segment (--sam3) and point (--molmo) itself.
					...["--sam3", service(pi, "sam3"), "--molmo", service(pi, "molmo")],
					...detectionArgs(pi, { sam3: false }),
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
			...detectionActive(pi, meta.capabilities?.perception),
		];
	}
}
