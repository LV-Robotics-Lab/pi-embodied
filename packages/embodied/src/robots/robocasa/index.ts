/**
 * RoboCasa365 robot for pi.
 *
 *   pi -e packages/embodied/src/robots/robocasa --task-name OpenDrawer --split target --seed 1
 *   pi -e packages/embodied/src/robots/robocasa --task-name OpenDrawer --split target --seed 1 --code=true --code-api=low
 *      (run_code over the env server's registry; its low tier drives the 12-D `step`)
 *
 * Starts one RoboCasa env server per session (PandaOmron mobile manipulator) and attaches
 * to a running RLDX-1 VLA server (see serve.sh) under a private RPC session, which holds
 * the policy's memory/RTC state. Tools are the RoboCasa primitives
 * (../../primitives/manifests/robocasa.json, which the env server reads too): the arm servo,
 * the base drive, the gripper and the scripted grasp are the env server's methods, shared with
 * code mode; RLDX-1 runs here and steps the env action by action. Every action
 * returns a new numbered state with agentview, navview and wrist images; world maps are
 * kept per state for back-projection. Success is the env's own `_check_success()`
 * (`state.success`), recorded in the session's `robot_result` entry.
 */

import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";
import { recipeFlash } from "../../capabilities/flash/recipe.ts";
import type { FlywheelObs, FlywheelSpec } from "../../capabilities/flywheel.ts";
import { MOLMO, type ModelService, SAM3 } from "../../infra/model-services.ts";
import { encodePng } from "../../infra/png.ts";
import { NdArray, RpcClient } from "../../infra/rpc.ts";
import type { Move } from "../../modes/units/index.ts";
import { template } from "../../planner/context-version.ts";
import { vlaSeeds } from "../../planner/vla-seed.ts";
import {
	detectionActive,
	detectionArgs,
	detectionTools,
	type PerceptionCaps,
	registerDetectionFlags,
} from "../../primitives/detections.ts";
import { mountGraspTool } from "../../primitives/grasp.ts";
import { pointActive, pointTool, registerPointFlags } from "../../primitives/pointing.ts";
import { attach, defineRobot, median, round, SERVICES } from "../../robot.ts";
import { type Cell, loadTable, resolveCell } from "./tasks.ts";

const read = (name: string) => template(new URL(name, import.meta.url));
const SYSTEM = read("./SYSTEM.md");
const MEMORY = { hf: read("./memory-hf.md"), local: read("./memory-local.md") };
const EXPLORE = read("./explore.md");
const CAMERAS = { agentview: "robot0_agentview_left", navview: "mobilebase0_navview", wrist: "robot0_eye_in_hand" };
const VLA_CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"];
const SIZE = 256; // env camera and RLDX observation resolution
const PRIMITIVES = [
	"move_to",
	"move_delta",
	"rotate_pitch",
	"set_gripper",
	"release",
	"scripted_grasp",
	"rldx_skill",
	"rldx_arm",
	"navigate_to",
	"move_base",
];

type Raw = Record<string, NdArray>;
type WorldMap = { size: number; xyz: Float32Array };
type Image = { role: string; camera: string; artifact: string; png: Buffer };
type State = {
	step: number;
	state: Record<string, number[]>;
	success: boolean;
	task_progress: Record<string, unknown>;
	vla_desync: boolean;
	log: { command: Record<string, unknown> | null; result: unknown; elapsed_s: number | null };
	images: Image[];
	maps: Map<string, WorldMap>;
};
type Frame = { state: Record<string, number[]>; video: Record<string, Buffer> };
/** What RLDX-1 reads and what the env runs (services robots/robocasa/flywheel.py). */
const FLYWHEEL: FlywheelSpec = {
	robot: "robocasa",
	images: {
		agentview_left_images: [256, 256, 3],
		agentview_right_images: [256, 256, 3],
		eye_in_hand_images: [256, 256, 3],
	},
	state: 16,
	action: 12,
};
/** An RLDX frame as the recorder takes it: its three cameras, and its state keys concatenated in order. */
const flyObs = (f: Frame): FlywheelObs => ({
	images: {
		agentview_left_images: new NdArray("uint8", [256, 256, 3], f.video["video.robot0_agentview_left"]),
		agentview_right_images: new NdArray("uint8", [256, 256, 3], f.video["video.robot0_agentview_right"]),
		eye_in_hand_images: new NdArray("uint8", [256, 256, 3], f.video["video.robot0_eye_in_hand"]),
	},
	state: Object.values(f.state).flat(),
});

/** Integer from the environment (the RLDX_* protocol knobs), else `fallback`. */
const envInt = (name: string, fallback: number) => {
	const v = process.env[name];
	return v === undefined || v === "" ? fallback : Number.parseInt(v, 10);
};
const norm = (v: number[]) => Math.hypot(...v);
const clip = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
/** Yaw of an xyzw quaternion, as scipy's `as_euler("xyz")[2]`. */
const yawOf = ([x, y, z, w]: number[]) => Math.atan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z));
/**
 * How the action units look (--units): MV_* are the robot base's own directions, turned into the world
 * by the base heading. Derived from the base-mounted agentview, not yet calibrated in the simulator.
 */
const UNITS_VIEWS = `Each result shows the agentview, then the navview, then the wrist view. MV_FWD moves the gripper along the robot base's forward direction (away from the base), MV_BACK toward the base, MV_LEFT / MV_RIGHT toward the robot's left / right, MV_UP / MV_DOWN up and down; the base does not move.
- Agentview (first image): a camera on the robot's base, behind and left of the arm, looking forward at the counter: MV_FWD moves the gripper toward the image top (deeper into the scene), MV_BACK toward the bottom, MV_LEFT / MV_RIGHT toward the image left / right.
- Navview (second image): the floor around the base, for navigation; ignore it for arm moves.
- Wrist view (third image): moves with the gripper. These directions come from the robot's geometry and are not calibrated: after the first move, check where the gripper went in the agentview and trust what you see.`;
/** Rows of an HxWx3 image in reverse order (MuJoCo renders bottom-up). */
function flipRows(data: Buffer, height: number): Buffer {
	const row = data.length / height;
	const out = Buffer.alloc(data.length);
	for (let y = 0; y < height; y++) data.copy(out, (height - 1 - y) * row, y * row, (y + 1) * row);
	return out;
}

/** RpcClient bound to one RPC session (the RLDX server keys policy memory/RTC state by it). */
/** The RLDX-1 VLA server (robocasa/serve.sh), for --serve-models rldx. */
const RLDX: ModelService = {
	name: "rldx",
	flag: "rldx",
	module: "pi_embodied_services.robots.robocasa.vla_server",
	args: () => {
		if (!process.env.RLDX_MODEL_PATH)
			throw new Error("--serve-models rldx: set RLDX_MODEL_PATH (the RLDX-1-FT-RC365 checkpoint)");
		return ["--model-path", process.env.RLDX_MODEL_PATH];
	},
	env: () => ({ NO_ALBUMENTATIONS_UPDATE: "1" }),
};

function sessionRpc(endpoint: string): RpcClient {
	const rpc = new RpcClient(endpoint);
	rpc.session = `rpc_${randomUUID().replaceAll("-", "")}`;
	return rpc;
}

export default function robocasa(pi: ExtensionAPI) {
	const flag = (name: string, fallback: string) => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("task-name", {
		type: "string",
		default: "OpenDrawer",
		description: "RoboCasa task, e.g. OpenDrawer",
	});
	pi.registerFlag("split", {
		type: "string",
		default: "target",
		description: "target | pretrain (the RoboCasa365 splits, 317 tasks each) | all (seed mode only)",
	});
	pi.registerFlag("seed", { type: "string", default: "0", description: "Scene seed (ignored with --scene)" });
	pi.registerFlag("scene", {
		type: "string",
		default: "",
		description: "RoboCasa365 manifest scene index 0-49 (its seed comes from the task table; empty = --seed)",
	});
	pi.registerFlag("hi-res", { type: "string", default: "0", description: "Hi-res agentview resolution (0 = off)" });
	pi.registerFlag("rldx", { type: "string", default: "http://127.0.0.1:18500", description: "RLDX-1 VLA server" });
	pi.registerFlag("env", { type: "string", description: "Attach to a running env server instead of starting one" });
	// --detections / --unidepth: detect, select_detection, reject_detection, enhance_depth (../primitives/detections.ts).
	registerDetectionFlags(pi, { sam3: true });
	// --point: Molmo's point over --molmo (../primitives/pointing.ts).
	registerPointFlags(pi);
	const seeds = vlaSeeds(pi, () => ["robocasa", robot.task]);
	pi.registerFlag("services", {
		type: "string",
		default: process.env.PI_EMBODIED_SERVICES ?? SERVICES,
		description: "pi-embodied services dir",
	});
	pi.registerFlag("robocasa-python", {
		type: "string",
		default: process.env.ROBOCASA_PYTHON ?? process.env.PI_EMBODIED_PYTHON ?? "python",
		description: "Python of the RoboCasa venv (the services' [robocasa] extra)",
	});
	pi.registerFlag("cuda-device", { type: "string", description: "GPU ordinal for MuJoCo EGL rendering" });
	pi.registerFlag("log-dir", { type: "string", default: tmpdir(), description: "Env server log directory" });

	let env: RpcClient;
	let vla: RpcClient | undefined;
	let obs: Raw;
	let language = "";
	let criteria = "";
	let states: State[] = [];
	let envSteps = 0;
	// True whenever a non-VLA primitive stepped the env since the last RLDX call; the next
	// RLDX call then reseeds its frame history instead of stitching stale frames on.
	let vlaDesync = true;
	/** Flywheel: the frame the last env step recorded (RLDX's history takes it), and the chunk running. */
	let recordedFrame: Frame | undefined;
	let flyVla = -1;
	let flyIndex = -1;
	/** raw/robocasa/<split>/<task>/seed_NNN (services robots/robocasa/flywheel.py). */
	const flyMeta = () => {
		const { task, split, seed } = cell();
		return {
			path: [split, task, `seed_${seed.padStart(3, "0")}`],
			metadata: { split, task_name: task, seed: Number(seed), task_language: language },
		};
	};
	let modality: { video_delta_indices: number[]; hist_maxlen: number } | undefined;
	let hist: Frame[] = [];
	let lastPrompt: string | undefined;
	let attempt = 1;
	/** The resolved --task-name / --split / --scene of this episode, once `startEpisode` checked them against the table. */
	let picked: Cell | undefined;
	const cell = () => ({
		task: robot.task["task-name"],
		split: robot.task.split,
		// A manifest scene's seed comes from the table; --seed is ignored with --scene.
		seed: picked?.seed !== undefined ? String(picked.seed) : robot.task.seed,
		scene: robot.task.scene,
	});
	const tag = () =>
		cell().scene === ""
			? `${cell().task}_${cell().split}_s${cell().seed}`
			: `${cell().task}_${cell().split}_m${cell().scene}`;
	/** Local corpora (and exploration) key the seed-0 reference by split; the published HF corpus does not. */
	const local = () => pi.getFlag("memory-profile") === "local" || pi.getFlag("explore") === true;
	/**
	 * The current task's published files that exist: RLinf/RPent-memory main keeps them under task_only/, the
	 * Target50 snapshot (551fc31) under results/ with a recipe_ prefix.
	 */
	const hfMemoryFiles = () => {
		const t = cell().task;
		const files = [
			...[`${t}_s0.json`, `${t}_s0_recipe.jsonl`, `${t}.md`].map((f) => `task_only/${f}`),
			...[`${t}_s0.json`, `recipe_${t}_s0.jsonl`, `${t}.md`].map((f) => `results/${f}`),
		]
			.map((f) => robot.mem!.render(`{{memory_dir}}/${f}`))
			.filter((f) => existsSync(f));
		return files.length ? files.map((f) => `- ${f}`).join("\n") : "(none: this task has no published memory)";
	};
	/** --units: one move as the robot's own tools make it, then a new recorded state. */
	async function unitMove(m: Move) {
		const t0 = Date.now();
		let result: Record<string, unknown>;
		try {
			const [dx, dy, dz] = m.delta;
			const c = Math.cos(baseYaw());
			const s = Math.sin(baseYaw());
			const target = eef().map((v, k) => v + [dx * c - dy * s, dx * s + dy * c, dz][k]);
			result = {
				...(m.gripper
					? { gripper: await motion("env.set_gripper", { gripper: m.gripper === "close" ? 1 : -1 }) }
					: {}),
				...(Math.hypot(dx, dy, dz) > 0
					? { move: await motion("env.move_to", { xyz: target, gripper: m.gripper ?? "hold" }) }
					: {}),
			};
		} catch (err) {
			result = {
				error: err instanceof Error ? err.message : String(err),
				interrupted: robot.signal?.aborted ?? false,
			};
		}
		const elapsed = round((Date.now() - t0) / 1000, 1);
		return view(await capture({ action: "act", delta: m.delta, gripper: m.gripper }, result, elapsed), {
			agent_elapsed_s: elapsed,
		});
	}

	const robot = defineRobot(pi, {
		name: "robocasa",
		// Tools and code primitives: ../../primitives/manifests/robocasa.json (the env server reads it too).
		manifest: "robocasa",
		vars: () => ({ cameras: ["agentview", "navview", "wrist"] }),
		// What the env server serves of the manifest's `requires` (its `_has`): the perception it was started with.
		capabilities: (c) =>
			({
				sam3: pi.getFlag("detections") === true && Boolean(flag("sam3", "")),
				unidepth: Boolean(String(pi.getFlag("unidepth") ?? "").trim()),
			})[c] ?? false,
		// RLDX-1 reads its checkpoint from RLDX_MODEL_PATH, as robocasa/serve.sh does.
		services: { models: [RLDX, SAM3, MOLMO], python: () => flag("robocasa-python", "python") },
		task: ["task-name", "split", "seed", "scene"],
		// The env server's code.api (the manifest's digest and what this run has), recorded per episode.
		codeApi: () => env,
		// Code mode (../../modes/code): the env server runs the program over the manifest's primitives; the result
		// carries the env steps, the success, the robot's observation after its last step and the
		// run's agentview frames. The run becomes a new numbered state, like a tool's action.
		code: {
			rpc: () => env,
			instruction: () => language,
			observe: async (r) => {
				for (const f of (r.frames as NdArray[] | undefined) ?? []) robot.video.frame(f);
				const steps = Number(r.steps) || 0;
				if (r.obs) obs = { ...obs, ...(r.obs as Raw) };
				if (steps > 0) {
					envSteps += steps;
					// The program stepped the env: RLDX's frame history no longer holds.
					vlaDesync = true;
				}
				return view(await capture({ action: "run_code" }, { status: r.status, env_steps: steps }, null));
			},
		},
		keepImages: 6,
		budget: { turns: 0, seconds: 0 },
		// Observations carry the agentview, navview and wrist images.
		vdm: { views: 3, wrist: 2 },
		flywheel: { spec: FLYWHEEL, select: () => `${cell().split}/${cell().task}` },
		flash: recipeFlash(pi, {
			// This cell's program, else the task's seed-0 reference (local and HF memory name it differently).
			names: () => [tag(), `${cell().task}_${cell().split}_s0`, `${cell().task}_s0`],
			memory: () => robot.mem?.render("{{memory_dir}}") ?? "",
			observe: "view_env_state",
			targets: { move_to: "xyz", scripted_grasp: "xyz", navigate_to: "xy" },
			// Molmo points in the view's agentview (the --hi-res one when it is 1024 wide).
			backProject: async (fr, [col, row]) => {
				const image = fr.latest().images[0] ?? "";
				const high = Buffer.from(image.slice(0, 44), "base64").readUInt32BE(16) > SIZE;
				const [r] = await fr.act([
					{
						name: "back_project_batch",
						arguments: {
							pixels: [[Math.round(row), Math.round(col)]],
							camera: "agentview",
							resolution: high ? "high" : "low",
						},
					},
				]);
				const xyz = (r.json.summary as { median_xyz?: number[] } | undefined)?.median_xyz;
				return r.error === undefined && Array.isArray(xyz) ? xyz : undefined;
			},
			picks: {
				isPick: (name) => name === "scripted_grasp",
				// The fingers stopped apart: something is between them.
				succeeded: (reply) => {
					const q = ((reply.json.result as Record<string, unknown> | undefined)?.gripper_qpos as number[]) ?? [];
					return (reply.json.result as { ok?: boolean } | undefined)?.ok === true && Math.abs(q[0] ?? 0) > 0.004;
				},
				attempts: 3,
				approach: ["move_to", "move_delta", "set_gripper", "rotate_pitch"],
				keep: 6,
				boundary: ["release", "rldx_arm", "rldx_skill"],
				release: "release",
			},
			over: (latest) => latest.json.success === true,
			solved: (latest) => latest.json.success === true,
		}),
		units: {
			// The base frame: MV_* keep their look in the base-mounted agentview wherever the base stands.
			vectors: {
				MV_FWD: [1, 0, 0],
				MV_BACK: [-1, 0, 0],
				MV_LEFT: [0, 1, 0],
				MV_RIGHT: [0, -1, 0],
				MV_UP: [0, 0, 1],
				MV_DOWN: [0, 0, -1],
			},
			stepM: 0.02,
			instruction: () => language,
			views: UNITS_VIEWS,
			// robot0_eye_in_hand, the third image.
			wrist: true,
			// The robosuite Panda gripper, as on LIBERO.
			emptyWidthM: 0.004,
			apply: (m) => unitMove(m),
			state: async () => ({
				eef_xyz: eef().map((v) => round(v)),
				gripper_width: round(vec("robot0_gripper_qpos").reduce((a, v) => a + Math.abs(v), 0)),
			}),
		},
		memory: {
			cell: () => ({
				tag: tag(),
				reference: local() ? `${cell().task}_${cell().split}_s0` : `${cell().task}_s0`,
			}),
			primitives: PRIMITIVES,
		},
		video: true,
		groundTruth: (names) => env.call("env.ground_truth_poses", { names: names ?? null }, 60_000, [], robot.signal),
		explore: {
			// The exploration `reset` tool is not a robot.tool, so robot.signal is unset here; use its own signal.
			reset: async (result, _ctx, signal) => {
				const t0 = Date.now();
				const out = { ...result, ...(await resetEpisode(signal)) };
				fly.reset(flyObs(await frame()), flyMeta());
				const elapsed = round((Date.now() - t0) / 1000, 1);
				return view(await capture({ action: "reset" }, out, elapsed), { agent_elapsed_s: elapsed });
			},
			prompt: () =>
				EXPLORE.replaceAll("{{task_name}}", robot.task["task-name"])
					.replaceAll("{{split}}", robot.task.split)
					.replaceAll("{{seed}}", robot.task.seed),
		},
		start: startEpisode,
		stop: disconnect,
		prompt: () => {
			const mem = robot.mem!;
			const explore = pi.getFlag("explore") === true;
			const vars: Record<string, string> = {
				task_language: language,
				task_name: cell().task,
				split: cell().split,
				seed: cell().scene === "" ? cell().seed : `${cell().seed} (manifest scene ${cell().scene})`,
				memory: explore
					? ""
					: mem.render(MEMORY[mem.profile], { task_name: cell().task, memory_files: hfMemoryFiles() }).trim(),
				success_criteria: criteria,
				reset_mode: explore
					? "`reset` restarts the episode with a freshly sampled scene; re-run perception after it."
					: "You are running in no-reset mode: solve the scene in one shot. The reset tool is disabled.",
			};
			return SYSTEM.replace(/\{\{(\w+)\}\}/g, (m, k: string) => vars[k] ?? m);
		},
		result: (ended) => ({
			task_name: cell().task,
			split: cell().split,
			seed: Number(cell().seed),
			// RoboCasa365 manifest mode: the scene index and the env id; null / undefined in seed mode.
			scene: cell().scene === "" ? null : Number(cell().scene),
			...(picked?.envId ? { env_id: picked.envId } : {}),
			task_language: language,
			success: states[states.length - 1]?.success ?? false,
			env_steps: envSteps,
			states: states.length,
			attempts: attempt,
			ended,
			rldx_max_chunks: envInt("RLDX_MAX_CHUNKS", 70),
			rldx_settle_patience: envInt("RLDX_SETTLE_PATIENCE", 999),
			rldx_action_steps_per_chunk: envInt("RLDX_ACTION_STEPS_PER_CHUNK", 8),
		}),
		status: () => ({
			language,
			step: states.length - 1,
			solved: states[states.length - 1]?.success ?? false,
		}),
		finish: {
			description:
				"Declare the task finished: when success (robocasa_terminated) becomes true, or when genuinely stuck after honest exploration. Give a 1-3 sentence summary of what worked and what failed. Success is the env's state.success, not this call.",
			parameters: Type.Object({
				status: StringEnum(["success", "failure", "stuck"] as const),
				summary: Type.String(),
			}),
			result: (params) => ({
				content: [
					{ type: "text", text: `Episode finished (success=${states[states.length - 1]?.success ?? false}).` },
				],
				details: params,
			}),
		},
	});
	const fly = robot.fly!;

	const vec = (key: string) => obs[key].toArray();
	const eef = () => vec("robot0_eef_pos");
	const finger = () => vec("robot0_gripper_qpos")[0];

	/** One env step with the PandaOmron 12-D action [eef_pos 3, eef_rot 3, gripper, base 3, torso, base_mode]. */
	async function step(a: number[]) {
		if (robot.signal?.aborted) throw new Error("interrupted");
		obs = (await env.call<[Raw, unknown, unknown, unknown]>("env.step", {}, 60_000, [a], robot.signal))[0];
		envSteps++;
		// Flywheel: every env step, with what RLDX would read next (its history reuses the frame).
		if (fly.recording) {
			recordedFrame = await frame();
			const solved = await env.call<boolean>("env.check_success", {}, 10_000);
			fly.transition(a, flyObs(recordedFrame), solved ? 1 : 0, solved, false, flyVla, flyIndex);
		}
		// Record the 256x256 agentview after every env step for the episode video.
		robot.video.frame(new NdArray("uint8", [SIZE, SIZE, 3], await rgb(CAMERAS.agentview)));
	}

	async function rgb(camera: string, size = SIZE): Promise<Buffer> {
		const img = await env.call<NdArray>(
			"env.render_camera",
			{ camera_name: camera, height: size, width: size, depth: false },
			120_000,
		);
		return flipRows(img.data, size);
	}

	/** Top-down RGB and per-pixel world xyz (world map: T_p2w @ [col*z, row*z, z, 1]). */
	async function rgbd(camera: string, size: number): Promise<{ rgb: Buffer; map: WorldMap }> {
		// One call at a time: concurrent calls interleave on the env server's worker pipe.
		const [img, depth] = await env.call<[NdArray, NdArray]>(
			"env.render_camera",
			{ camera_name: camera, height: size, width: size, depth: true },
			120_000,
		);
		const T = await env.call<NdArray>("env.get_camera_transform", { camera_name: camera, height: size, width: size });
		const z = depth.toArray();
		const t = T.toArray();
		const xyz = new Float32Array(size * size * 3);
		for (let r = 0; r < size; r++) {
			const src = (size - 1 - r) * size; // the sim's depth is bottom-up
			for (let c = 0; c < size; c++) {
				const d = z[src + c];
				const i = (r * size + c) * 3;
				for (let k = 0; k < 3; k++)
					xyz[i + k] = t[4 * k] * c * d + t[4 * k + 1] * r * d + t[4 * k + 2] * d + t[4 * k + 3];
			}
		}
		return { rgb: flipRows(img.data, size), map: { size, xyz } };
	}

	/** Record the current observation as the next numbered state. */
	async function capture(command: Record<string, unknown> | null, result: unknown, elapsed: number | null) {
		const hi = Number(flag("hi-res", "0"));
		const success = await env.call<boolean>("env.check_success");
		const progress = await env.call<Record<string, unknown>>("env.get_task_progress").catch(() => ({}));
		const agent = await rgbd(CAMERAS.agentview, SIZE);
		const wrist = await rgbd(CAMERAS.wrist, SIZE);
		const nav = await rgbd(CAMERAS.navview, SIZE);
		const high = hi > 0 ? await rgbd(CAMERAS.agentview, hi) : undefined;
		const round4 = (key: string) => vec(key).map((v) => round(v));
		const s: State = {
			step: states.length,
			state: Object.fromEntries(
				["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_base_pos", "robot0_base_quat"].map(
					(k) => [k, round4(k)],
				),
			),
			success: Boolean(success),
			task_progress: progress,
			vla_desync: vlaDesync,
			log: { command, result, elapsed_s: elapsed },
			images: [
				high
					? {
							role: "calibration_frame",
							camera: "agentview",
							artifact: "agentview_high.png",
							png: encodePng(high.rgb, hi, hi),
						}
					: {
							role: "calibration_frame",
							camera: "agentview",
							artifact: "agentview.png",
							png: encodePng(agent.rgb, SIZE, SIZE),
						},
				{ role: "nav_view", camera: "navview", artifact: "navview.png", png: encodePng(nav.rgb, SIZE, SIZE) },
				{
					role: "calibration_frame",
					camera: "wrist",
					artifact: "wrist.png",
					png: encodePng(wrist.rgb, SIZE, SIZE),
				},
			],
			maps: new Map([
				["agentview:low", agent.map],
				["wrist:low", wrist.map],
				["navview:low", nav.map],
				...(high ? [["agentview:high", high.map] as [string, WorldMap]] : []),
			]),
		};
		states.push(s);
		states[s.step - envInt("RLDX_KEEP_HEAVY_NPY", 25)]?.maps.clear(); // world maps are kept for recent states only
		return s;
	}

	function view(s: State, extra: Record<string, unknown> = {}) {
		const text = {
			step: s.step,
			task_progress: s.task_progress,
			task_language: language,
			state: s.state,
			robocasa_terminated: s.success,
			vla_desync: s.vla_desync,
			success: s.success,
			log: s.log,
			images: s.images.map(({ role, camera, artifact }) => ({ role, camera, artifact })),
			...extra,
		};
		return {
			content: [
				{ type: "text" as const, text: JSON.stringify(text) },
				...s.images.map((i) => ({ type: "image" as const, data: i.png.toString("base64"), mimeType: "image/png" })),
			],
			details: { step: s.step, success: s.success, terminated: s.success, result: s.log.result },
		};
	}

	/** Register a tool; actions record and return a new state, read-only tools return their result. */
	function tool<P extends TSchema>(
		name: string,
		description: string,
		parameters: P,
		run: (p: Static<P>) => Promise<Record<string, unknown>>,
		action = true,
	) {
		robot.tool(name, description, parameters, async (params, sig) => {
			if (!action) {
				const { _state, ...rest } = (await run(params)) as { _state?: State };
				if (_state) return view(_state);
				return { content: [{ type: "text" as const, text: JSON.stringify(rest) }], details: rest };
			}
			const t0 = Date.now();
			let result: Record<string, unknown>;
			try {
				result = await run(params);
			} catch (err) {
				result = { error: err instanceof Error ? err.message : String(err), interrupted: sig?.aborted ?? false };
			}
			const elapsed = round((Date.now() - t0) / 1000, 1);
			return view(await capture({ action: name, ...params }, result, elapsed), { agent_elapsed_s: elapsed });
		});
	}

	// ---- motion: the env server's methods ----

	/**
	 * Run one of the env server's motion methods (manifests/robocasa.json): the arm servo, the base drive
	 * and the gripper step there, polling the stop. Its agentview frames go to the video, its per-step
	 * records to the Flywheel, its robot observation becomes `obs`; the rest is the tool's result.
	 */
	async function motion(method: string, params: Record<string, unknown>) {
		vlaDesync = true;
		const r = await env.call<Record<string, unknown>>(method, params, 600_000, [], robot.signal);
		const {
			frames,
			policy_frames,
			obs: o,
			env_steps,
			...report
		} = r as {
			frames?: NdArray[];
			policy_frames?: {
				action: NdArray;
				success: boolean;
				state: Record<string, NdArray>;
				video: Record<string, NdArray>;
			}[];
			obs?: Raw | null;
			env_steps?: number;
		} & Record<string, unknown>;
		for (const f of frames ?? []) robot.video.frame(f);
		for (const f of policy_frames ?? []) {
			const fr: Frame = {
				state: Object.fromEntries(Object.entries(f.state).map(([k, v]) => [k, v.toArray()])),
				video: Object.fromEntries(Object.entries(f.video).map(([k, v]) => [k, v.data])),
			};
			fly.transition(f.action.toArray(), flyObs(fr), f.success ? 1 : 0, f.success, false, -1, -1);
		}
		if (o) obs = { ...obs, ...o };
		envSteps += Number(env_steps) || 0;
		return report;
	}

	// ---- base ----

	const basePos = () => vec("robot0_base_pos");
	const baseYaw = () => yawOf(vec("robot0_base_quat"));

	// ---- RLDX-1 ----

	/** One per-sim-step history entry: proprio + the 3 VLA cameras at 256, top-down. */
	async function frame(): Promise<Frame> {
		const images: Awaited<ReturnType<typeof rgb>>[] = [];
		for (const c of VLA_CAMERAS) images.push(await rgb(c));
		return {
			state: {
				"state.gripper_qpos": vec("robot0_gripper_qpos"),
				"state.base_position": vec("robot0_base_pos"),
				"state.base_rotation": vec("robot0_base_quat"),
				"state.end_effector_position_relative": vec("robot0_base_to_eef_pos"),
				"state.end_effector_rotation_relative": vec("robot0_base_to_eef_quat"),
			},
			video: Object.fromEntries(VLA_CAMERAS.map((c, i) => [`video.${c}`, images[i]])),
		};
	}

	/** Batched obs stacked over the history at video_delta_indices - 1, as the eval's MultiStepWrapper. */
	function rldxObs(prompt: string, vdi: number[]) {
		const idx = vdi.map((d) => clip(hist.length + d - 1, 0, hist.length - 1));
		const cur = hist[hist.length - 1];
		const out: Record<string, unknown> = {};
		for (const [k, v] of Object.entries(cur.state)) out[k] = NdArray.f32(v, [1, 1, v.length]);
		for (const k of Object.keys(cur.video))
			out[k] = new NdArray("uint8", [1, idx.length, SIZE, SIZE, 3], Buffer.concat(idx.map((j) => hist[j].video[k])));
		out["annotation.human.task_description"] = [prompt];
		return out;
	}

	/** RLDX skill run: closed-loop chunks until env success, the policy settles, or the chunk cap. */
	async function rldx(p: {
		prompt?: string;
		base_clip: number | null;
		max_chunks?: number;
		force_reset?: boolean;
		n_action_steps?: number;
		settle_patience?: number;
		settle_eps?: number;
	}) {
		const maxChunks = envInt("RLDX_MAX_CHUNKS", p.max_chunks ?? 70);
		const nSteps = envInt("RLDX_ACTION_STEPS_PER_CHUNK", p.n_action_steps ?? 8);
		const patience = envInt("RLDX_SETTLE_PATIENCE", p.settle_patience ?? 999);
		const eps = p.settle_eps ?? 0.012;
		for (const [name, v] of [
			["max_chunks", maxChunks],
			["n_action_steps", nSteps],
			["settle_patience", patience],
		] as const)
			if (!(v >= 1)) return { error: `${name} must be positive; VLA was not executed` };
		if (!language)
			return { error: "RoboCasa task language is unavailable; VLA was not executed", effective_prompt: "" };
		if (!vla) throw new Error("RLDX server not connected");
		const prompt = language; // RLDX always gets the live, full task language
		recordedFrame = undefined;
		const forceReset = Boolean(p.force_reset) || vlaDesync;
		vlaDesync = false;
		await vla.call("session.register", {}, 30_000); // refreshes, or re-creates an idle-expired session
		modality ??= await vla.call<{ video_delta_indices: number[]; hist_maxlen: number }>("vla.get_modality_config");
		const { video_delta_indices: vdi, hist_maxlen: maxlen } = modality;
		const push = async () => {
			hist.push(recordedFrame ?? (await frame()));
			recordedFrame = undefined;
			if (hist.length > maxlen) hist.shift();
		};
		// Memory and history reset only on a new instruction or a forced reset; same-prompt calls keep continuity.
		const newTask = forceReset || prompt !== lastPrompt;
		lastPrompt = prompt;
		if (newTask || !hist.length) hist = Array(maxlen).fill(await frame());
		let fresh = newTask;
		const base0 = basePos().slice(0, 2);
		let eefPrev = eef();
		let gripPrev = finger();
		let minZ = eefPrev[2];
		let peakLift = 0;
		let applied = 0;
		let settled = 0;
		let status = "cap";
		let graspEver = false;
		let graspObj: string | null = null;
		let lastClose = false;
		let chunks = 0;
		const vla_seeds: (number | null)[] = [];
		while (chunks < maxChunks) {
			chunks++;
			if (robot.signal?.aborted) throw new Error("interrupted");
			const seed = seeds.next();
			vla_seeds.push(seed ?? null);
			const actions = await vla.call<Record<string, NdArray>>(
				"vla.predict",
				{},
				120_000,
				[rldxObs(prompt, vdi), { reset_memory: [fresh], ...(seed === undefined ? {} : { seed }) }],
				robot.signal,
			);
			fresh = false;
			const col = (key: string) => {
				const arr = actions[key];
				const d = arr.shape[2] ?? 1;
				const v = arr.toArray();
				return (i: number) => v.slice(i * d, (i + 1) * d);
			};
			const [pos, rot, close, base, mode] = [
				"action.end_effector_position",
				"action.end_effector_rotation",
				"action.gripper_close",
				"action.base_motion",
				"action.control_mode",
			].map(col);
			const horizon = actions["action.gripper_close"].shape[1];
			// The chunk as proposed (base unclipped) in the env's 12-D layout, for the Flywheel.
			flyVla = fly.proposal(
				prompt,
				Array.from({ length: horizon }, (_, i) => [
					...pos(i),
					...rot(i),
					close(i)[0] >= 0.5 ? 1 : -1,
					...base(i).slice(0, 4),
					mode(i)[0] < 0.5 ? -1 : 1,
				]),
			);
			for (let i = 0; i < Math.min(nSteps, horizon); i++) {
				flyIndex = i;
				let motion = base(i);
				const bc = p.base_clip;
				if (bc !== null) motion = motion.map((v) => clip(v, -bc, bc));
				lastClose = close(i)[0] >= 0.5;
				// robocasa365's PandaOmronKeyConverter.unmap_action, assembled by the env's own controller layout
				const unmapped = {
					robot0_right_gripper: lastClose ? 1 : -1,
					robot0_right: [...pos(i), ...rot(i)],
					robot0_base: motion.slice(0, 3),
					robot0_torso: motion.slice(3, 4),
					robot0_base_mode: mode(i)[0] < 0.5 ? -1 : 1,
				};
				const a = await env.call<NdArray>("env.reassemble_env_action", {}, 30_000, [unmapped], robot.signal);
				await step(a.toArray());
				applied++;
				await push();
			}
			const [grasping, obj] = await env.call<[boolean, string | null]>("env.grasp_contact", {}, 10_000);
			if (grasping) {
				graspEver = true;
				graspObj ??= obj;
			}
			if (await env.call<boolean>("env.check_success")) {
				status = "success";
				break;
			}
			const eefNow = eef();
			const gripNow = finger();
			minZ = Math.min(minZ, eefNow[2]);
			peakLift = Math.max(peakLift, eefNow[2] - minZ);
			if (norm(eefNow.map((v, k) => v - eefPrev[k])) < eps && Math.abs(gripNow - gripPrev) < 0.003) {
				if (++settled >= patience) {
					status = "settled";
					break;
				}
			} else settled = 0;
			eefPrev = eefNow;
			gripPrev = gripNow;
		}
		flyVla = flyIndex = -1;
		const g = finger();
		const base1 = basePos();
		const [contact, contactObj] = await env.call<[boolean, string | null]>("env.grasp_contact", {}, 10_000);
		const heldApart = lastClose && g > 0.004 && g < 0.039; // commanded closed, fingers wedged on something
		const requested = p.prompt ?? "";
		return {
			ok: true,
			prompt,
			status,
			chunks,
			steps_applied: applied,
			grasped: contact || heldApart,
			grasp_detected: graspEver || contact || heldApart,
			grasp_contact: contact,
			held_apart: heldApart,
			grasp_obj: contactObj ?? graspObj,
			gripper_qpos: round(g, 3),
			peak_lift: round(peakLift, 3),
			base_clip: p.base_clip,
			base_drift: round(Math.hypot(base1[0] - base0[0], base1[1] - base0[1]), 3),
			effective_prompt: prompt,
			effective_max_chunks: maxChunks,
			effective_n_action_steps: nSteps,
			effective_settle_patience: patience,
			prompt_overridden: requested !== prompt,
			...(requested !== prompt ? { requested_prompt: requested } : {}),
			vla_seeds,
		};
	}

	// ---- tools ----

	const num = (description: string) => Type.Optional(Type.Number({ description }));
	const int = (description: string) => Type.Optional(Type.Integer({ description }));
	const camera = Type.Optional(
		StringEnum(["agentview", "navview", "wrist"] as const, {
			description: "Camera (default agentview)",
		}),
	);
	const resolution = Type.Optional(
		StringEnum(["high", "low"] as const, {
			description: "low = the 256x256 world map (default); high = the --hi-res agentview map",
		}),
	);

	// The motion tools are the env server's methods with the manifest's parameters (manifests/robocasa.json).
	for (const name of ["move_to", "move_delta", "rotate_pitch", "set_gripper", "release", "scripted_grasp"])
		tool(name, "", Type.Object({}), (p) => motion(`env.${name}`, p as Record<string, unknown>));

	const rldxParams = (baseClip: string) =>
		Type.Object({
			prompt: Type.String({ description: "Complete live task_language, copied verbatim" }),
			base_clip: num(`Base motion cap, >= 0 (${baseClip}); a negative value means no clamp`),
			max_chunks: int("Action-chunk budget (default 70; RLDX_MAX_CHUNKS overrides)"),
			force_reset: Type.Optional(Type.Boolean({ description: "Force VLA frame history reset (default false)" })),
			n_action_steps: int("Actions per VLA chunk (default 8)"),
			settle_patience: int("Settle chunk budget before declaring done (default 999; do NOT set small)"),
			settle_eps: num("Settle position tolerance, m (default 0.012)"),
		});

	tool(
		"rldx_skill",
		"RLDX-1 VLA closed-loop skill with FULL base motion: the VLA drives both arm and mobile base. Use for full-body tasks where the base must reposition. Pass the complete live task_language verbatim; the runtime always uses that environment language. Do NOT interrupt consecutive VLA calls with manual primitives: that breaks VLA frame history continuity (sets vla_desync).",
		rldxParams("default: no clamp"),
		(p) => rldx({ ...p, base_clip: p.base_clip === undefined || p.base_clip < 0 ? null : p.base_clip }),
	);

	tool(
		"rldx_arm",
		"RLDX-1 VLA closed-loop skill with the base CLAMPED to small motions (base_clip 0.1): the VLA drives the arm for precise micro-alignment but cannot drive the base away. Pass the complete live task_language verbatim. Do NOT interrupt consecutive VLA calls with manual primitives.",
		rldxParams("default 0.1 = small"),
		(p) => rldx({ ...p, base_clip: p.base_clip === undefined ? 0.1 : p.base_clip < 0 ? null : p.base_clip }),
	);

	for (const name of ["navigate_to", "move_base"])
		tool(name, "", Type.Object({}), (p) => motion(`env.${name}`, p as Record<string, unknown>));

	/** Exploration's `reset`: a freshly sampled scene; arm/base calibration and the RLDX session start over. */
	async function resetEpisode(signal?: AbortSignal) {
		vlaDesync = true;
		obs = await env.call<Raw>("env.reset", {}, 120_000, [], signal);
		seeds.reset();
		await vla?.call("vla.reset_session", {}, 30_000).catch(() => undefined);
		lastPrompt = undefined;
		hist = [];
		language = (await env.call<string | null>("env.get_task_language")) ?? "";
		attempt++;
		return {
			ok: true,
			reset: true,
			eef: eef().map((v) => round(v)),
			attempt,
			notice: "Fresh episode started. Re-run perception before acting.",
		};
	}

	tool(
		"view_env_state",
		"Read state NN (default latest): env state, success (robocasa_terminated), task_progress, vla_desync, the command log, and the agentview (calibration frame), navview (base floor view) and wrist images. Use agentview pixels for back-projection, navview for base navigation and floor walkability, wrist for close-range details near the gripper.",
		Type.Object({
			step: int("0 = initial (default latest)"),
		}),
		async ({ step: n }) => {
			const s = states[n ?? states.length - 1];
			return s ? { _state: s } : { error: `state step not available: ${n}` };
		},
		false,
	);

	tool(
		"back_project_batch",
		"Back-project MULTIPLE pixels [row, col] (row 0 = image top) to world XYZ from one state's world map. Returns each pixel's world_xyz plus summary.median_xyz. For robust localization sample 3-8 pixels on the target and read summary.median_xyz. Max 50 pixels. Pixels from agentview_high need resolution 'high'.",
		Type.Object({
			pixels: Type.Array(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 }), { minItems: 1, maxItems: 50 }),
			step: int("State to use (default latest)"),
			camera,
			resolution,
		}),
		async ({ pixels, step: n, camera: c = "agentview", resolution: res = "low" }) => {
			const s = states[n ?? states.length - 1];
			if (!s) return { error: `state step not available: ${n}` };
			const map = s.maps.get(`${c}:${res}`);
			if (!map) return { error: `${c} ${res}-resolution world map not recorded for step ${s.step}` };
			const results = (pixels as number[][]).map(([row, col]) => {
				if (row < 0 || row >= map.size || col < 0 || col >= map.size)
					return {
						pixel: [row, col],
						world_xyz: null,
						valid: false,
						error: `pixel (${row},${col}) out of bounds (${map.size}x${map.size})`,
					};
				const i = (row * map.size + col) * 3;
				const p = [map.xyz[i], map.xyz[i + 1], map.xyz[i + 2]];
				if (!p.every(Number.isFinite) || Math.abs(p[0] + p[1] + p[2]) <= 1e-6)
					return { pixel: [row, col], world_xyz: null, valid: false, error: "invalid world xyz at pixel" };
				return { pixel: [row, col], world_xyz: p.map((v) => round(v)), valid: true, error: null };
			});
			const ok = results.flatMap((r) => (r.world_xyz ? [r.world_xyz] : []));
			return {
				results,
				summary: {
					valid_count: ok.length,
					total_count: pixels.length,
					...(ok.length ? { median_xyz: [0, 1, 2].map((k) => round(median(ok.map((p) => p[k])))) } : {}),
				},
				step: s.step,
				camera: c,
				resolution: res,
			};
		},
		false,
	);

	tool(
		"query_world_map",
		"Query the latest world map by Z band / XY region to find objects at specific heights: keeps pixels with z_min <= z <= z_max (optionally within x_range / y_range) and clusters them on an image grid. Typical: z 0.85-0.95 = countertop objects; z 0.0-0.12 with camera 'navview' = walkable floor.",
		Type.Object({
			z_min: num("Minimum Z, m (default 0.85, counter height)"),
			z_max: num("Maximum Z, m (default 0.95)"),
			x_range: Type.Optional(
				Type.Array(Type.Number(), {
					minItems: 2,
					maxItems: 2,
					description: "[min, max] world x, m (default: any)",
				}),
			),
			y_range: Type.Optional(
				Type.Array(Type.Number(), {
					minItems: 2,
					maxItems: 2,
					description: "[min, max] world y, m (default: any)",
				}),
			),
			camera,
			resolution,
			min_cluster_size: int("Minimum sampled pixels per cluster (default 10)"),
		}),
		async ({
			z_min = 0.85,
			z_max = 0.95,
			x_range,
			y_range,
			camera: c = "agentview",
			resolution: res = "low",
			min_cluster_size = 10,
		}) => {
			const s = states[states.length - 1];
			const map = s?.maps.get(`${c}:${res}`);
			if (!map) return { error: `${c} ${res}-resolution world map not found for the latest step` };
			const { size, xyz: w } = map;
			const hits: number[] = [];
			for (let i = 0; i < size * size; i++) {
				const [x, y, z] = [w[3 * i], w[3 * i + 1], w[3 * i + 2]];
				if (!(Number.isFinite(z) && z >= z_min && z <= z_max)) continue;
				if (x_range && !(x >= x_range[0] && x <= x_range[1])) continue;
				if (y_range && !(y >= y_range[0] && y <= y_range[1])) continue;
				hits.push(i);
			}
			if (hits.length < min_cluster_size)
				return { clusters: [], summary: { total_clusters: 0, total_pixels_matched: 0 } };
			const cell = Math.max(1, Math.floor(size / Math.max(8, Math.min(32, Math.floor(size / 32)))));
			const cells = new Map<string, number[]>();
			for (let j = 0; j < hits.length; j += 5) {
				const [r, col] = [Math.floor(hits[j] / size), hits[j] % size];
				const key = `${Math.floor(r / cell)},${Math.floor(col / cell)}`;
				cells.set(key, [...(cells.get(key) ?? []), hits[j]]);
			}
			const clusters = [...cells.values()]
				.filter((px) => px.length >= min_cluster_size)
				.map((px) => {
					const axis = (k: number) => px.map((i) => w[3 * i + k]);
					const mid = px[Math.floor(px.length / 2)];
					return {
						center_xyz: [0, 1, 2].map((k) => round(median(axis(k)))),
						pixel_count: px.length,
						bbox_xyz: {
							min: [0, 1, 2].map((k) => round(Math.min(...axis(k)))),
							max: [0, 1, 2].map((k) => round(Math.max(...axis(k)))),
						},
						sample_pixels: [[Math.floor(mid / size), mid % size]],
					};
				})
				.sort((a, b) => b.pixel_count - a.pixel_count)
				.slice(0, 20);
			return { clusters, summary: { total_clusters: clusters.length, total_pixels_matched: hits.length } };
		},
		false,
	);

	// ---- lifecycle ----

	async function disconnect() {
		await vla?.call("session.close", {}, 2_000).catch(() => undefined);
		vla = undefined;
	}

	// Molmo pointing on the current images (active with --point).
	mountGraspTool(
		robot.tool,
		pointTool(pi, {
			cameras: ["agentview", "navview", "wrist"],
			frame: async (c) => ({ width: SIZE, height: SIZE, rgb: await rgb(CAMERAS[c as keyof typeof CAMERAS]) }),
			locate: async (c, row, col) => {
				const { map } = await rgbd(CAMERAS[c as keyof typeof CAMERAS], SIZE);
				const i = (row * SIZE + col) * 3;
				const p = [map.xyz[i], map.xyz[i + 1], map.xyz[i + 2]];
				return p.every(Number.isFinite) ? { world_xyz: p.map((v) => round(v, 4)) } : undefined;
			},
			signal: () => robot.signal,
		}),
	);

	// SAM3 masks with ids and UniDepth over the env server's perception (active with --detections / --unidepth).
	for (const d of detectionTools(pi, {
		call: (method, kwargs, timeoutMs) =>
			env.call<Record<string, any>>(method, kwargs, timeoutMs ?? 120_000, [], robot.signal),
		cameras: ["agentview", "navview", "wrist"],
	}))
		mountGraspTool(robot.tool, d);

	async function startEpisode() {
		states = [];
		envSteps = 0;
		modality = lastPrompt = undefined;
		hist = [];
		vlaDesync = true;
		attempt = 1;
		seeds.reset();
		const services = flag("services", SERVICES);
		// Checked against the task table before anything starts: an unknown task names its near matches.
		picked = undefined;
		picked = resolveCell(loadTable(services), cell());
		const { task, split, seed, scene } = cell();
		const rldxClient = sessionRpc(flag("rldx", ""));
		vla = rldxClient;
		const endpoint = pi.getFlag("env") as string | undefined;
		const cuda = pi.getFlag("cuda-device") as string | undefined;
		[env] = await Promise.all([
			endpoint
				? attach(endpoint)
				: robot.serve({
						python: flag("robocasa-python", "python"),
						args: [
							...["-m", "pi_embodied_services.robots.robocasa.env_server"],
							...["--task-name", task, "--split", split, "--seed", seed],
							...(scene === "" ? [] : ["--scene", scene]),
							...(cuda ? ["--cuda-device", cuda] : []),
							...detectionArgs(pi, flag("sam3", "")),
						],
						cwd: services,
						// RLDX_RESET_SEED would replay a legacy paired scene instead of --seed.
						env: {
							...process.env,
							PYTHONPATH: services,
							MUJOCO_GL: "egl",
							ROBOT_PLATFORM: "ROBOCASA",
							RLDX_RESET_SEED: "",
						},
						log: (port) => join(flag("log-dir", tmpdir()), `robocasa-env-${tag()}-${port}.log`),
					}),
			rldxClient.ready().then(() => rldxClient.call("session.register", {}, 30_000)),
		]);
		const meta = await env.call<Record<string, unknown>>("env.get_env_meta", {}, 30_000);
		const sceneOf = (m: Record<string, unknown>) =>
			m.scene === null || m.scene === undefined ? "" : String(m.scene);
		if (
			meta.task_name !== task ||
			meta.split !== split ||
			Number(meta.seed) !== Number(seed) ||
			sceneOf(meta) !== scene
		)
			throw new Error(`env server runs ${JSON.stringify(meta)}, not ${tag()} (seed ${seed})`);
		// The env resets on client connect and again in the primitives; a seed's scene is the second one.
		await env.call("env.reset", {}, 120_000);
		obs = await env.call<Raw>("env.reset", {}, 120_000);
		language = (await env.call<string | null>("env.get_task_language")) ?? "";
		criteria = await env
			.call<string>("env.get_success_criteria_text", {}, 30_000)
			.catch((e) => `(unavailable: ${e})`);
		await capture(null, null, null);
		fly.reset(flyObs(await frame()), flyMeta());
		// The server's motion methods return the Flywheel's per-step records while it records.
		await env.call("env.set_recording", { on: fly.recording }, 30_000);
		const perception = (meta.capabilities as { perception?: PerceptionCaps } | undefined)?.perception;
		return [
			...[...PRIMITIVES, "view_env_state", "back_project_batch", "query_world_map", "finish"],
			...detectionActive(pi, perception),
			...pointActive(pi),
		];
	}
}
