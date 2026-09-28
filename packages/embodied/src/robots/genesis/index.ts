/**
 * Genesis robot for pi: a Franka Panda in the Genesis simulator on a translation-only IK controller.
 *
 *   pi -e packages/embodied/src/robots/genesis --seed 0                      (cube_pick, the only task so far)
 *   pi -e packages/embodied/src/robots/genesis --units --task cube_pick --seed 3
 *   pi -e packages/embodied/src/robots/genesis --privileged --seed 0          (adds ground_truth_poses)
 *   pi -e packages/embodied/src/robots/genesis --code=true --code-api=low --seed 0   (run_code, CaP-X's S3)
 *
 * Starts one Genesis env server per session (services/.../robots/genesis/env_server.py, the `genesis`
 * venv; rendering needs a GPU). The server owns the motion: `move_delta` and the units hook `apply`
 * run a base-frame delta as ~2 cm IK decisions with the reset orientation held, inside a workspace
 * box, above a Z floor and within a per-call cap, all checked before anything moves; `set_gripper`
 * opens or closes and holds. Every result carries the front and wrist images and the state; success
 * is the task's own predicate, recorded in `robot_result` with the rule (`--success-rule`: `grasp`,
 * OpenETA's cube_pick rule, by default; `lift`, the cube 8 cm off the table).
 * --vdm (../vdm.ts) differences the front and wrist views between observations.
 * `segment` (SAM3, `--sam3`) and `back_project` give world coordinates from the current image
 * through the server's depth. Tools and code primitives: ../../primitives/manifests/genesis.json
 * (the env server reads it too); every tool runs one env server method. --collect-flywheel-data records every control step of a motion
 * (services robots/genesis/flywheel.py).
 *
 * OpenETA (github.com/OpenETA at 7d4a0a1) sim/envs/genesis: its Franka scene and cube_pick task,
 * ported as a pi robot with the pi-embodied motion limits and cameras.
 */

import { tmpdir } from "node:os";
import { join } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import type { FlywheelObs, FlywheelSpec } from "../../capabilities/flywheel.ts";
import { MOLMO, SAM3 } from "../../infra/model-services.ts";
import { encodePng } from "../../infra/png.ts";
import type { NdArray, RpcClient } from "../../infra/rpc.ts";
import type { MoveUnit, Vec3 } from "../../modes/units/index.ts";
import { template } from "../../planner/context-version.ts";
import { detectionActive, detectionArgs, detectionTools, registerDetectionFlags } from "../../primitives/detections.ts";
import { graspActive, graspArgs, graspTools, mountGraspTool, registerGraspFlags } from "../../primitives/grasp.ts";
import { ikArgs, type Reach, registerIkFlag } from "../../primitives/ik.ts";
import { pointActive, pointTool, registerPointFlags } from "../../primitives/pointing.ts";
import { attach, defineRobot, type Json, rgbOf, SERVICES } from "../../robot.ts";

const SYSTEM = template(new URL("./SYSTEM.md", import.meta.url));
const EXPLORE = template(new URL("./explore.md", import.meta.url));
const MEMORY = template(new URL("./memory.md", import.meta.url));

/** The success rules (env_server SUCCESS_RULES): OpenETA's grasp rule (default) or the 8 cm lift. */
export const SUCCESS_RULES = ["grasp", "lift"] as const;
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
/** A recorded control step (`record`): its observation and the step as `env.step` takes it. */
type Step = Obs & { action: NdArray };
type Moved = Obs & {
	commanded_m: number[];
	moved_m: number[];
	decisions: number;
	control_steps: number;
	frames?: NdArray[];
	steps?: Step[];
	cancelled?: boolean;
};
type Meta = {
	task: string;
	seed: number;
	instruction: string;
	workspace: { min: number[]; max: number[] };
	z_floor_m: number;
	max_move_m: number;
	lift_m: number;
	success_rule?: string;
	capabilities?: { perception?: { segment?: boolean; enhance_depth?: boolean } };
};
type CameraMeta = { intrinsic_K: number[][]; extrinsic_cam2world: number[][]; width: number; height: number };

const round = (v: number, d = 4) => Number(v.toFixed(d));

/** The two views, the TCP pose and finger opening, and the env step (services robots/genesis/flywheel.py). */
export const FLYWHEEL: FlywheelSpec = {
	robot: "genesis",
	// The server's --view-size: the first frame's.
	images: { agentview_images: null, wrist_images: null },
	state: 8,
	action: 4,
};
const flyObs = (o: Obs): FlywheelObs => ({
	images: { agentview_images: o.agentview, wrist_images: o.wrist },
	state: [...o.tcp_pos.toArray(), ...o.tcp_quat_wxyz.toArray(), o.gripper_width],
});

export default function genesis(pi: ExtensionAPI) {
	const flag = (name: string, fallback: string) => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("task", { type: "string", default: "cube_pick", description: `Genesis task: ${TASKS.join(", ")}` });
	pi.registerFlag("seed", { type: "string", default: "0", description: "Reset seed (the object layout)" });
	pi.registerFlag("backend", {
		type: "string",
		default: "gpu",
		description: "Genesis compute backend for the env server: gpu (default), cuda or cpu",
	});
	pi.registerFlag("success-rule", {
		type: "string",
		default: "grasp",
		description: `Success rule: grasp (OpenETA cube_pick: both fingers on the cube within 8 cm for 3 steps, default) or lift (the cube 8 cm up for 5 steps)`,
	});
	pi.registerFlag("env", { type: "string", description: "Attach to a running env server instead of starting one" });
	pi.registerFlag("sam3", { type: "string", default: "http://127.0.0.1:18300", description: "SAM3 server (segment)" });
	// --detections / --unidepth: detect, select_detection, reject_detection, enhance_depth (../primitives/detections.ts).
	// --ik: preview_reach over the env server's IK check (../ik.ts).
	registerIkFlag(pi);
	registerDetectionFlags(pi);
	// --contact-graspnet & co: plan_grasp and friends, and execute_grasp / execute_place (services/.../utils/grasp_chain.py on the env server).
	registerGraspFlags(pi);
	// --point: Molmo's point over its --molmo server (../primitives/pointing.ts).
	registerPointFlags(pi, { molmo: true });
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
	let obs: Obs;
	let meta: Meta;

	/** The memory cell of this task at `seed`. */
	const tag = (seed: string) => `genesis_${robot.task.task}_s${seed}`;
	const robot = defineRobot(pi, {
		name: "genesis",
		// Tools and code primitives: ../../primitives/manifests/genesis.json (the env server reads it too).
		manifest: "genesis",
		vars: () => ({ max_move: MAX_MOVE_M, cameras: [...CAMERAS], arms: [] }),
		capabilities: (c) =>
			({
				sam3: Boolean(flag("sam3", "")),
				ik: Boolean(flag("ik", "").trim()),
				grasp: graspActive(pi).length > 0,
				place: graspActive(pi).length > 0 && Boolean(flag("anyplace", "")),
				unidepth: Boolean(flag("unidepth", "").trim()),
			})[c] ?? false,
		services: { models: [SAM3, MOLMO] },
		task: ["task", "seed"],
		keepImages: 4,
		video: true,
		// Observations carry the front view, then the wrist view.
		vdm: { views: 2, wrist: 1 },
		flywheel: { spec: FLYWHEEL, select: () => robot.task.task },
		groundTruth: (names) => call("env.ground_truth_poses", { names: names ?? null }),
		// The env server's code.api (from the manifest), recorded per episode.
		codeApi: () => env,
		// No corpus is published for Genesis: memory is what exploration writes locally, one cell per task and seed.
		memory: {
			cell: () => ({ tag: tag(robot.task.seed), reference: tag("0") }),
			primitives: ["move_delta", "set_gripper", "act"],
			published: false,
		},
		explore: {
			// The exploration `reset` tool is not a robot.tool, so robot.signal is unset here; use its own signal.
			reset: async (result, _ctx, signal) => {
				everGrasped = false;
				const [o] = await env.call<[Obs, unknown]>("env.reset", {}, 300_000, [], signal);
				absorb(o);
				fly.reset(flyObs(o), flyMeta());
				return observe({ ...result, reset: true });
			},
			prompt: () => EXPLORE.replaceAll("{{task}}", robot.task.task).replaceAll("{{seed}}", robot.task.seed),
			rewrite: [
				[
					/This is a single episode\. You may recover within it \(re-position, re-grasp\), but you cannot restart it\./,
					"This is an exploration run: `reset` starts a fresh attempt (see Exploration). Within an attempt, recover in place (re-position, re-grasp).",
				],
			],
		},
		// Code mode (../code): the env server runs the program against that registry; the result
		// carries the control steps, the latched success, the new observation and the video frames.
		code: {
			rpc: () => env,
			instruction: () => meta.instruction,
			refuse: () => (obs?.success ? "the task is already solved; call finish" : undefined),
			observe: async (r) => {
				for (const f of (r.frames as NdArray[] | undefined) ?? []) video.frame(f);
				if (r.obs) absorb(r.obs as Obs);
				return observe({ name: "run_code", status: r.status, env_steps: Number(r.steps) || 0 });
			},
		},
		start: startEpisode,
		prompt: () =>
			SYSTEM.replaceAll("{{task_language}}", meta.instruction).replaceAll(
				"{{memory}}",
				pi.getFlag("explore") === true ? "" : robot.mem!.render(MEMORY).trim(),
			),
		result: () => ({
			task: robot.task.task,
			seed: Number(robot.task.seed),
			success: obs?.success ?? false,
			success_rule: meta?.success_rule ?? flag("success-rule", "grasp"),
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
			// The env server's WRIST camera, the second image.
			wrist: true,
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
	const fly = robot.fly!;
	/** raw/genesis/<task>/seed_NNN (services robots/genesis/flywheel.py). */
	const flyMeta = () => ({
		path: [robot.task.task, `seed_${robot.task.seed.padStart(3, "0")}`],
		metadata: { task: robot.task.task, seed: Number(robot.task.seed), task_language: meta.instruction },
	});
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

	/** Flywheel: every control step a motion recorded (`record` is sent only while recording). */
	function record(steps: Step[] | undefined) {
		for (const s of steps ?? []) fly.transition(s.action.toArray(), flyObs(s), s.success ? 1 : 0, s.success, false);
	}

	/** One base-frame move (m) with an optional gripper command first; the server checks the limits. */
	async function move(delta: Vec3, gripper: "open" | "close" | null, signal: AbortSignal | undefined) {
		const r = await call<Moved>(
			"env.move_delta",
			{ delta_xyz: delta, gripper, return_frames: true, ...(fly.recording ? { record: true } : {}) },
			[],
			signal,
		);
		for (const f of r.frames ?? []) video.frame(f);
		record(r.steps);
		const { frames: _frames, steps: _steps, commanded_m, moved_m, decisions, control_steps, cancelled, ...o } = r;
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

	const text = (details: Json) => ({
		content: [{ type: "text" as const, text: JSON.stringify(details) }],
		details,
	});
	/** A motion's video frames and control-step records, then its observation absorbed; returns the rest. */
	function motion(r: Json) {
		for (const f of (r.frames as NdArray[] | undefined) ?? []) video.frame(f);
		record(r.steps as Step[] | undefined);
		absorb(r as unknown as Obs);
		const {
			frames: _f,
			steps: _s,
			agentview: _a,
			wrist: _w,
			tcp_pos: _p,
			tcp_quat_wxyz: _q,
			qpos: _j,
			gripper_width: _gw,
			gripper_command: _gc,
			success: _ok,
			is_grasped: _g,
			lift_m: _l,
			env_steps: _n,
			...rest
		} = r;
		return rest;
	}
	const recording = () => (fly.recording ? { record: true } : {});

	// Every tool below runs one env server method with the manifest's parameters
	// (../../primitives/manifests/genesis.json); here only what the planner sees is shaped.
	robot.tool("view_env_state", "", Type.Object({}), async () => observe({}));

	robot.tool("get_camera_meta", "", Type.Object({}), async (params: Json) => {
		const m = await call<CameraMeta>("env.get_camera_meta", params);
		return text({ camera: params.camera_name ?? "agentview", ...m });
	});

	robot.tool("back_project", "", Type.Object({}), async (params: Json) =>
		text(await call<Json>("env.back_project", params)),
	);

	robot.tool("segment", "", Type.Object({}), async (params: Json) => {
		const { mask: _mask, overlay_png_base64, ...rest } = await call<Json>("env.segment", params);
		const details = rest.found === false ? { ...rest, error: rest.reason ?? "no mask" } : rest;
		return {
			content: [
				{ type: "text" as const, text: JSON.stringify(details) },
				...(overlay_png_base64
					? [{ type: "image" as const, data: String(overlay_png_base64), mimeType: "image/png" }]
					: []),
			],
			details,
		};
	});

	robot.tool("move_delta", "", Type.Object({}), async (params: Json, signal) => {
		if (obs.success) return observe({ error: "the task is already solved; call finish" });
		return observe(await move(params.delta_xyz as Vec3, (params.gripper as "open" | "close") ?? null, signal));
	});

	robot.tool("set_gripper", "", Type.Object({}), async (params: Json, signal) => {
		const r = motion(
			await call<Json>("env.set_gripper", { ...params, return_frames: true, ...recording() }, [], signal),
		);
		return observe({ gripper: params.close ? "close" : "open", ...r });
	});

	robot.tool("preview_reach", "", Type.Object({}), async (params: Json) =>
		text((await env.call<Reach>("env.preview_reach", params, 60_000, [], robot.signal)) as unknown as Json),
	);

	// Planned grasps (--contact-graspnet & co): the server runs the claimed path as move_delta legs
	// (env.execute_grasp / env.execute_place, utils/grasp_chain.py).
	for (const name of ["execute_grasp", "execute_place"])
		robot.tool(name, "", Type.Object({}), async (params: Json, signal) => {
			if (obs.success) return observe({ error: "the task is already solved; call finish" });
			return observe(motion(await call<Json>(`env.${name}`, { ...params, ...recording() }, [], signal)));
		});

	// Molmo pointing on the current images (active with --point).
	mountGraspTool(
		robot.tool,
		pointTool(pi, {
			cameras: CAMERAS,
			frame: async (c) => {
				const a = c === "wrist" ? obs.wrist : obs.agentview;
				return rgbOf(a);
			},
			locate: async (c, row, col) => {
				const [p] = await call<(number[] | null)[]>("env.back_project", { camera: c, pixels: [[row, col]] });
				return p ? { world_xyz: p } : undefined;
			},
			signal: () => robot.signal,
		}),
	);

	// SAM3 masks with ids and UniDepth over the env server's perception (active with --detections / --unidepth).
	for (const d of detectionTools(pi, {
		call: (method, kwargs, timeoutMs) => env.call<Json>(method, kwargs, timeoutMs ?? 120_000, [], robot.signal),
		cameras: CAMERAS,
		// The server renders the images shown: the centroid's world xyz through the simulator's depth.
		locate: async (c, d) => {
			const rc = d.centroid_rc as number[] | null;
			if (!rc) return {};
			const [p] = await call<(number[] | null)[]>("env.back_project", { camera: c, pixels: [rc] });
			return p ? { centroid_world_xyz: p } : {};
		},
	}))
		mountGraspTool(robot.tool, d);

	// plan_grasp / plan_place / check_attached over the env server's planner (--contact-graspnet & co).
	for (const d of graspTools(pi, {
		call: (method, kwargs, timeoutMs) => env.call<Json>(method, kwargs, timeoutMs ?? 120_000, [], robot.signal),
		cameras: ["agentview", "wrist"],
		task: () => meta.instruction,
	}))
		mountGraspTool(robot.tool, d);

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
					...["--success-rule", flag("success-rule", "grasp")],
					...ikArgs(pi.getFlag("ik")),
					// env.segment (and the planner's object text) segment with SAM3 on the server.
					...(flag("sam3", "") ? ["--sam3", flag("sam3", "")] : []),
					...detectionArgs(pi, ""),
					...graspArgs(pi),
				],
				cwd: services,
				env: { ...process.env, PYTHONPATH: services },
				log: (port) => join(tmpdir(), `pi-embodied-genesis-${task}-s${seed}-${port}.log`),
				// Genesis compiles its kernels on the first build (minutes on a cold cache).
				readyMs: 1_200_000,
			});
		}
		meta = await env.call<Meta>("env.get_env_meta");
		if (meta.task !== task || meta.seed !== Number(seed))
			throw new Error(`env server runs ${meta.task} seed ${meta.seed}, not ${task} seed ${seed}`);
		const rule = flag("success-rule", "grasp");
		if (!(SUCCESS_RULES as readonly string[]).includes(rule))
			throw new Error(`unknown --success-rule ${rule}; one of ${SUCCESS_RULES.join(", ")}`);
		if (meta.success_rule !== undefined && meta.success_rule !== rule)
			throw new Error(`env server scores success by ${meta.success_rule}, not --success-rule ${rule}`);
		everGrasped = false;
		const [o] = await env.call<[Obs, unknown]>("env.reset", {}, 300_000);
		absorb(o);
		fly.reset(flyObs(o), flyMeta());
		return [
			...["view_env_state", "get_camera_meta", "segment", "back_project", "move_delta", "set_gripper", "finish"],
			...detectionActive(pi, meta.capabilities?.perception),
			// preview_reach requires --ik, segment --sam3, the chains a planner (the manifest drops them without).
			"preview_reach",
			...graspActive(pi),
			...(graspActive(pi).length ? ["execute_grasp", "execute_place"] : []),
			...pointActive(pi),
		];
	}
}
