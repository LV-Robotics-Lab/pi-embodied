/**
 * RoboDojo robot for pi: one RoboDojo simulation task (Isaac Sim / Isaac Lab) on its two ARX X5 arms.
 *
 *   pi -e packages/embodied/src/robodojo --task stack_bowls --seed 0 --cuda-device 1
 *   pi -e packages/embodied/src/robodojo --units --task push_T --seed 3     (Show-Harness action units, per arm)
 *
 * Starts one RoboDojo env server per session (services/.../robots/robodojo/env_server.py, the venv from
 * robots/robodojo/install_isaac61.sh: Isaac Sim 6.1 / Isaac Lab 3.0 with RoboDojo patched by
 * robodojo-isaac61.patch; needs ROBODOJO_ROOT with its Assets/). One env per process: RoboDojo's
 * heterogeneous parallel simulation is not used (see the package README). `--seed` is RoboDojo's eval
 * layout id (layouts are pre-generated per task; 25 or 50 of them). Isaac Sim takes a minute or more to
 * come up.
 *
 * Every motion goes through RoboDojo's own `take_action` (one 25 Hz control step each, counted against
 * the task's `step_lim`): `move_to` / `move_delta` interpolate one arm's end effector in the env frame
 * through RoboDojo's cuRobo IK while the other arm holds, `rotate_delta` turns a gripper about the
 * vertical, `set_gripper` drives the normalized gripper (1 open .. 0 closed), `go_home` returns both arms
 * to their start joints (most tasks only succeed once both arms are back). Success is RoboDojo's
 * `is_episode_end` judgement, recorded in `robot_result` with RoboDojo's partial-credit `score`.
 */

import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { recipeFlash } from "../flash/recipe.ts";
import type { FlywheelObs, FlywheelSpec } from "../flywheel.ts";
import { encodePng } from "../png.ts";
import { attach, defineRobot, SERVICES, u8 } from "../robot.ts";
import type { NdArray, RpcClient } from "../rpc.ts";
import type { MoveUnit, Vec3 } from "../units/index.ts";

const read = (name: string) => readFileSync(new URL(name, import.meta.url), "utf8");
const SYSTEM = read("./SYSTEM.md");
const MEMORY = read("./memory.md");
const EXPLORE = read("./explore.md");

export const ARMS = ["left", "right"] as const;
export type Arm = (typeof ARMS)[number];
export const VIEWS = ["head", "left_wrist", "right_wrist"] as const;
/**
 * The env frame: both X5 bases sit at y = -0.45 facing +y (left arm at x = -0.3, right at +0.3), the
 * table top is at z = 0.74, the head camera is at (0, -0.41, 1.31) above the bases looking down the
 * table. MV_FWD is away from the robot (+y), MV_LEFT toward the robot's left (-x), for either arm.
 */
export const VECTORS: Record<MoveUnit, Vec3> = {
	MV_FWD: [0, 1, 0],
	MV_BACK: [0, -1, 0],
	MV_LEFT: [-1, 0, 0],
	MV_RIGHT: [1, 0, 0],
	MV_UP: [0, 0, 1],
	MV_DOWN: [0, 0, -1],
};
/** Metres per MV_* unit (the MVTOKEN 2 cm convention; the server interpolates 1 cm per control step). */
export const STEP_M = 0.02;
/** Largest end-effector translation per call, m (the server refuses more). */
export const MAX_MOVE_M = 0.5;
/** Radians per ROTATE_CW; + is counter-clockwise seen from above, as on every robot here. */
export const YAW_STEP_RAD = 0.15;
/** Largest yaw one call may command, rad (the server clips more). */
export const MAX_ROTATE_RAD = 0.8;

export const UNITS_VIEWS = `Each result shows the head view (the fixed camera above and behind both arms, looking down the table), then the left wrist view, then the right wrist view; every unit names the arm it moves.
- Head view: MV_FWD moves that gripper toward the image top (away from the robot), MV_BACK toward the image bottom, MV_LEFT / MV_RIGHT toward the image left / right, MV_UP / MV_DOWN up and down. The left arm is on the image left.
- Wrist views: each moves with its gripper and looks along the fingers; judge left/right and forward/back in the head view.
- ROTATE_CW turns that gripper counter-clockwise seen from above, ROTATE_CCW the opposite.`;

/**
 * Flywheel (--collect-flywheel-data) in RoboDojo's joint space, what XPolicyLab's arx_x5 policies read and
 * emit: [left joints 6, left gripper, right joints 6, right gripper] (gripper 1 open .. 0 closed), the
 * three 640x480 cameras (services robots/robodojo/flywheel.py holds the same shapes).
 */
export const FLYWHEEL: FlywheelSpec = {
	robot: "robodojo",
	images: { head_images: null, left_wrist_images: null, right_wrist_images: null },
	state: 14,
	action: 14,
};
/** The server's per-action record while Flywheel recording is on (`env.set_recording`). */
type PolicyFrame = { head: NdArray; left_wrist: NdArray; right_wrist: NdArray; state: NdArray; action: NdArray };
const flyObs = (f: PolicyFrame): FlywheelObs => ({
	images: { head_images: u8(f.head), left_wrist_images: u8(f.left_wrist), right_wrist_images: u8(f.right_wrist) },
	state: f.state.toArray(),
});

type ArmState = {
	eef_pos: NdArray;
	eef_quat_wxyz: NdArray;
	joints: NdArray;
	/** The last commanded joints (what a one-arm motion holds this arm at). */
	joints_command: NdArray;
	gripper: number;
	gripper_command: number;
};
type Obs = {
	head: NdArray;
	left_wrist: NdArray;
	right_wrist: NdArray;
	arms: Record<Arm, ArmState>;
	success: boolean;
	ended: boolean;
	truncated: boolean;
	/** RoboDojo's episode score (1 on success, else its partial-credit tiers / 100): the evaluator's, never the planner's. */
	score: number;
	env_steps: number;
	step_lim: number;
	seed: number;
};
type Motion = Obs & {
	arm?: Arm;
	moved_m?: number[];
	final_error_m?: number;
	executed?: number;
	waypoints?: number;
	control_steps?: number;
	stopped?: string;
	hint?: string;
	cancelled?: boolean;
	error?: string;
	frames?: NdArray[];
	policy_frames?: PolicyFrame[];
	commanded_yaw?: number;
	yaw?: number;
	clipped?: boolean;
};
type Meta = {
	task: string;
	seed: number;
	eval_seed: number;
	dimension: string | null;
	layouts: number;
	instruction: string;
	step_lim: number;
};

const round = (v: number, d = 4) => Number(v.toFixed(d));
const MUTATE_MS = 600_000;
const READ_MS = 120_000;

export default function robodojo(pi: ExtensionAPI) {
	const flag = (name: string, fallback: string) => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("task", {
		type: "string",
		default: "stack_bowls",
		description: "RoboDojo task (task/RoboDojo/tasks)",
	});
	pi.registerFlag("seed", { type: "string", default: "0", description: "RoboDojo eval layout id" });
	pi.registerFlag("eval-seed", {
		type: "string",
		default: "0",
		description: "RoboDojo layout set (Assets/Eval_Layout/RoboDojo/arx_x5/<eval-seed>)",
	});
	pi.registerFlag("cuda-device", {
		type: "string",
		default: "0",
		description: "GPU for Isaac Sim and cuRobo (the server sets CUDA_VISIBLE_DEVICES to it)",
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
		description: "Python for the env server (the robodojo venv)",
	});

	let env: RpcClient;
	let obs: Obs;
	let meta: Meta;
	const tag = (seed: string) => `robodojo_${robot.task.task}_s${seed}`;

	const robot = defineRobot(pi, {
		name: "robodojo",
		task: ["task", "seed"],
		codeApi: () => env,
		keepImages: 6,
		video: true,
		// Observations carry the head, left wrist and right wrist images.
		vdm: { views: VIEWS.length, wrist: [1, 2] },
		groundTruth: (names) => env.call("env.ground_truth_poses", { names: names ?? null }, READ_MS, [], robot.signal),
		flywheel: { spec: FLYWHEEL, select: () => robot.task.task },
		// No corpus is published for RoboDojo: memory is what exploration writes locally.
		memory: {
			cell: () => ({ tag: tag(robot.task.seed), reference: tag("0") }),
			primitives: ["move_to", "move_delta", "rotate_delta", "set_gripper", "go_home", "act"],
			published: false,
		},
		// Flash replays a solved episode's plan: move_to targets follow their anchors, which Molmo points at in
		// the head image and the server back-projects through the same step's metric depth (env.back_project).
		flash: recipeFlash(pi, {
			names: () => [tag(robot.task.seed), tag("0")],
			memory: () => robot.mem?.render("{{memory_dir}}") ?? "",
			observe: "view_env_state",
			targets: { move_to: "xyz" },
			backProject: async (_fr, pixel) => {
				const r = await env.call<{ xyz: (number[] | null)[] }>(
					"env.back_project",
					{ pixels: [pixel] },
					READ_MS,
					[],
					robot.signal,
				);
				return r.xyz[0] ?? undefined;
			},
			over: (latest) => latest.json.success === true || latest.json.ended === true,
			solved: (latest) => latest.json.success === true,
		}),
		explore: {
			reset: async (result, _ctx, signal) => {
				await resetEnv(signal);
				return observe({ ...result, reset: true });
			},
			prompt: () =>
				EXPLORE.replaceAll("{{task}}", robot.task.task)
					.replaceAll("{{seed}}", robot.task.seed)
					.replaceAll("{{dimension}}", meta?.dimension ?? "unknown"),
			rewrite: [
				[
					/This is a single episode with a step limit\. You may recover within it \(re-position, re-grasp\), but you cannot restart it\./,
					"This is an exploration run with a step limit per attempt: `reset` starts a fresh attempt (see Exploration). Within an attempt, recover in place (re-position, re-grasp).",
				],
			],
		},
		start: startEpisode,
		prompt: () =>
			SYSTEM.replaceAll("{{task_language}}", meta.instruction)
				.replaceAll("{{task}}", robot.task.task)
				.replaceAll("{{seed}}", robot.task.seed)
				.replaceAll("{{step_lim}}", String(meta.step_lim))
				.replaceAll("{{memory}}", pi.getFlag("explore") === true ? "" : robot.mem!.render(MEMORY).trim()),
		result: () => ({
			task: robot.task.task,
			seed: Number(robot.task.seed),
			eval_seed: Number(flag("eval-seed", "0")),
			dimension: meta?.dimension ?? null,
			instruction: meta?.instruction ?? null,
			success: obs?.success ?? false,
			score: obs?.score ?? 0,
			truncated: obs?.truncated ?? false,
			env_steps: obs?.env_steps ?? 0,
			step_lim: obs?.step_lim ?? null,
		}),
		status: () => ({ language: meta.instruction, step: obs.env_steps, solved: obs.success }),
		finish: {
			description:
				"End the episode after checking the latest state. Success is RoboDojo's own judgement, not this call.",
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
			yawStepRad: YAW_STEP_RAD,
			maxYawRad: () => MAX_ROTATE_RAD,
			maxMoveM: () => MAX_MOVE_M,
			arms: ARMS,
			instruction: () => meta.instruction,
			views: UNITS_VIEWS,
			// A grounded unit is one arm's move, turn and/or gripper command; STOP is an empty move.
			apply: async (m, signal) => {
				const arm = m.arm as Arm;
				const result: Record<string, unknown> = {};
				const gripper = m.gripper === "close" ? 0 : m.gripper === "open" ? 1 : null;
				if (Math.hypot(...m.delta) > 0 || gripper !== null)
					result.move = await motion("env.move_delta", { arm, delta_xyz: m.delta, gripper }, signal);
				if (m.yaw) result.rotate = await motion("env.rotate_delta", { arm, yaw: m.yaw }, signal);
				return observe(result);
			},
			// The X5 gripper is normalized (1 open .. 0 closed), not a width: no empty-grasp check.
			state: async (arm) => ({
				eef_xyz: obs.arms[arm as Arm].eef_pos.toArray().map((v) => round(v)),
				gripper_opening: obs.arms[arm as Arm].gripper,
			}),
			plugins: ["proprioception", "variable_step", "action_chunk", "rotation", "plan", "mem_text"],
		},
	});
	const { video } = robot;
	const fly = robot.fly!;

	async function resetEnv(signal?: AbortSignal) {
		const [o, info] = await env.call<[Obs, { instruction: string; error?: string }]>(
			"env.reset",
			{},
			MUTATE_MS,
			[],
			signal,
		);
		obs = o;
		if (info.error) throw new Error(`RoboDojo reset: ${info.error}`);
		await startFlywheel();
	}

	/**
	 * With --collect-flywheel-data the server records every control step (`env.set_recording`, which
	 * returns the reset frame) and a Flywheel episode starts: raw/robodojo/<task>/seed_NNN.
	 */
	async function startFlywheel() {
		const on = pi.getFlag("collect-flywheel-data") === true;
		const first = await env.call<PolicyFrame | null>("env.set_recording", { on }, READ_MS);
		if (!on || !first) return;
		fly.reset(flyObs(first), {
			path: [robot.task.task, `seed_${robot.task.seed.padStart(3, "0")}`],
			metadata: { task: robot.task.task, seed: Number(robot.task.seed), task_language: meta.instruction },
		});
	}

	/** Run one server motion; its head frames go to the video and its per-step records to the Flywheel. */
	async function motion(method: string, params: Record<string, unknown>, signal: AbortSignal | undefined) {
		const r = await env.call<Motion>(
			method,
			{ ...params, return_frames: true },
			MUTATE_MS,
			[],
			signal ?? robot.signal,
		);
		for (const f of r.frames ?? []) video.frame(f);
		for (const f of r.policy_frames ?? [])
			fly.transition(f.action.toArray(), flyObs(f), r.success ? 1 : 0, r.success, r.truncated);
		const {
			frames: _f,
			policy_frames: _p,
			head,
			left_wrist,
			right_wrist,
			arms,
			success,
			ended,
			truncated,
			score,
			env_steps,
			step_lim,
			seed,
			...report
		} = r;
		obs = { head, left_wrist, right_wrist, arms, success, ended, truncated, score, env_steps, step_lim, seed };
		return report;
	}

	/** The result with the new state, then the head, left wrist and right wrist images. */
	function observe(result: Record<string, unknown>) {
		const images = VIEWS.map((v) => obs[v]).map((a) => encodePng(a.data, a.shape[1], a.shape[0]));
		const arm = (a: Arm) => ({
			eef_xyz: obs.arms[a].eef_pos.toArray().map((v) => round(v)),
			eef_quat_wxyz: obs.arms[a].eef_quat_wxyz.toArray().map((v) => round(v)),
			gripper: obs.arms[a].gripper,
			gripper_command: obs.arms[a].gripper_command,
		});
		const details = {
			result,
			success: obs.success,
			ended: obs.ended,
			truncated: obs.truncated,
			step: obs.env_steps,
			remaining_steps: obs.step_lim - obs.env_steps,
			task_language: meta.instruction,
			state: { left: arm("left"), right: arm("right") },
			images: VIEWS.map((v) => `${v} ${obs[v].shape[1]}x${obs[v].shape[0]}`),
		};
		return {
			content: [
				{ type: "text" as const, text: JSON.stringify(details) },
				...images.map((png) => ({ type: "image" as const, data: png.toString("base64"), mimeType: "image/png" })),
			],
			// RoboDojo's partial-credit score is the evaluator's: it goes to the session (details, robot_result),
			// never into the planner's context.
			details: { ...details, score: obs.score },
		};
	}

	/** Motion tools answer this once the episode is over (success or step limit). */
	const over = () =>
		obs.ended
			? observe({ error: obs.success ? "the task is solved; call finish" : "the episode is over; call finish" })
			: undefined;
	const armParam = StringEnum(ARMS, { description: "Which arm" });
	const gripperParam = Type.Optional(
		Type.Number({ minimum: 0, maximum: 1, description: "Gripper command first: 1 open .. 0 closed" }),
	);

	robot.tool(
		"view_env_state",
		"Current state with the head, left wrist and right wrist images.",
		Type.Object({}),
		async () => {
			obs = { ...obs, ...(await env.call<Obs>("env.state", {}, READ_MS)) };
			const [head, left_wrist, right_wrist] = await Promise.all(
				VIEWS.map((v) => env.call<NdArray>("env.render_camera", { camera_name: v }, READ_MS)),
			);
			obs = { ...obs, head, left_wrist, right_wrist };
			return observe({});
		},
	);

	robot.tool(
		"move_to",
		`Move one arm's gripper in a straight line to an env-frame [x, y, z] in metres (+x toward the robot's right, +y away from the robot, +z up; the table top is z = 0.74; at most ${MAX_MOVE_M} m from where it is), optionally with a [qw, qx, qy, qz] orientation (default: keep it), after an optional gripper command. The other arm holds still. Stops at the first unreachable waypoint (\`stopped\`). Returns the new state and images.`,
		Type.Object({
			arm: armParam,
			xyz: Type.Array(Type.Number(), { minItems: 3, maxItems: 3 }),
			quat_wxyz: Type.Optional(Type.Array(Type.Number(), { minItems: 4, maxItems: 4 })),
			gripper: gripperParam,
		}),
		async ({ arm, xyz, quat_wxyz, gripper }, signal) =>
			over() ??
			observe(
				await motion("env.move_to", { arm, xyz, quat_wxyz: quat_wxyz ?? null, gripper: gripper ?? null }, signal),
			),
	);

	robot.tool(
		"move_delta",
		`Translate one arm's gripper by an env-frame [dx, dy, dz] in metres (at most ${MAX_MOVE_M} m per call), orientation held, after an optional gripper command. Returns the new state and images.`,
		Type.Object({
			arm: armParam,
			delta_xyz: Type.Array(Type.Number(), { minItems: 3, maxItems: 3 }),
			gripper: gripperParam,
		}),
		async ({ arm, delta_xyz, gripper }, signal) => {
			const norm = Math.hypot(...delta_xyz);
			if (!(norm <= MAX_MOVE_M))
				throw new Error(`delta moves ${round(norm)} m; the limit is ${MAX_MOVE_M} m per call. Split the motion.`);
			return over() ?? observe(await motion("env.move_delta", { arm, delta_xyz, gripper: gripper ?? null }, signal));
		},
	);

	robot.tool(
		"rotate_delta",
		`Turn one arm's gripper by \`yaw\` radians about the vertical through it (+ counter-clockwise seen from above; at most ${MAX_ROTATE_RAD} rad per call), holding its position. Returns the new state and images.`,
		Type.Object({ arm: armParam, yaw: Type.Number() }),
		async ({ arm, yaw }, signal) => over() ?? observe(await motion("env.rotate_delta", { arm, yaw }, signal)),
	);

	robot.tool(
		"set_gripper",
		"Move one gripper to `value`: 1 fully open .. 0 fully closed. Returns the new state and images.",
		Type.Object({ arm: armParam, value: Type.Number({ minimum: 0, maximum: 1 }) }),
		async ({ arm, value }, signal) => over() ?? observe(await motion("env.set_gripper", { arm, value }, signal)),
	);

	robot.tool(
		"go_home",
		"Drive both arms back to their start pose (grippers unchanged). Most tasks count as done only once both arms are back home with the grippers open.",
		Type.Object({}),
		async (_p, signal) => over() ?? observe(await motion("env.go_home", {}, signal)),
	);

	robot.tool(
		"locate",
		"Env-frame [x, y, z] (m) of [col, row] pixels of the latest head image, from its depth. Read-only.",
		Type.Object({
			pixels: Type.Array(Type.Array(Type.Number(), { minItems: 2, maxItems: 2 }), { minItems: 1, maxItems: 32 }),
		}),
		async ({ pixels }) => {
			const r = await env.call<{ xyz: (number[] | null)[] }>(
				"env.back_project",
				{ pixels },
				READ_MS,
				[],
				robot.signal,
			);
			const out = { frame: "env", points: pixels.map((p, i) => ({ pixel: p, xyz: r.xyz[i] })) };
			return { content: [{ type: "text" as const, text: JSON.stringify(out) }], details: out };
		},
	);

	// TODO(xpolicy): mount `xpolicy_act` (../xpolicy.ts, env_cfg_type "arx_x5") once that module is on main;
	// the server side is ready (`env.get_obs` is RoboDojo's native observation, `env.step` / `env.chunk_step`
	// take native action dicts).

	async function startEpisode() {
		const { task, seed } = robot.task;
		const endpoint = pi.getFlag("env") as string | undefined;
		if (endpoint) env = await attach(endpoint, 1_200_000);
		else {
			const services = flag("services", SERVICES);
			env = await robot.serve({
				python: flag("python", "python"),
				args: [
					...["-m", "pi_embodied_services.robots.robodojo.env_server"],
					...["--task", task, "--seed", seed, "--eval-seed", flag("eval-seed", "0")],
					...["--cuda-device", flag("cuda-device", "0")],
				],
				cwd: services,
				env: { ...process.env, PYTHONPATH: services, OMNI_KIT_ACCEPT_EULA: "YES" },
				log: (port) => join(tmpdir(), `pi-embodied-robodojo-${task}-s${seed}-${port}.log`),
				// Isaac Sim's cold start, cuRobo's warmup and a cold shader cache come before the server binds.
				readyMs: 1_800_000,
			});
		}
		meta = await env.call<Meta>("env.get_env_meta");
		const evalSeed = Number(flag("eval-seed", "0"));
		if (meta.task !== task || meta.eval_seed !== evalSeed)
			throw new Error(
				`env server runs ${meta.task} (eval seed ${meta.eval_seed}), not ${task} (eval seed ${evalSeed})`,
			);
		if (!(Number(seed) >= 0 && Number(seed) < meta.layouts))
			throw new Error(`--seed ${seed}: ${task} has eval layouts 0..${meta.layouts - 1}`);
		// The server comes up reset on --seed; a new session (or an attached server) resets to this cell's layout.
		const [o, info] = await env.call<[Obs, { instruction: string; error?: string }]>(
			"env.reset",
			{ seed: Number(seed) },
			MUTATE_MS,
		);
		obs = o;
		if (info.error) throw new Error(`RoboDojo reset: ${info.error}`);
		meta = { ...meta, instruction: info.instruction };
		await startFlywheel();
		return ["view_env_state", "move_to", "move_delta", "rotate_delta", "set_gripper", "go_home", "locate", "finish"];
	}
}
