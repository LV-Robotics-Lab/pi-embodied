/**
 * Two physical Franka arms for pi (evaluation mode).
 *
 *   dual_franka/serve.sh                      # dual-Franka Pi0.5 VLA (+ SAM3) servers
 *   pi -e packages/embodied/src/dual_franka --operator --task 3 --robot-config my_rig.yaml \
 *     --robot-vla http://127.0.0.1:18210 --robot-sam3 http://127.0.0.1:18310
 *
 * Starts the RLinf-backed env server (pi_embodied_services.robots.dual_franka.env_server; the
 * two-node Ray cluster must already run) or attaches to one with --robot-env. The server
 * enforces the workspace limits, per-step clips, servo tolerances and joint-health thresholds
 * from its runtime config; task definitions, easy_handeye calibration and localization bounds
 * come from the same services package. Coordinates are in the shared right_base frame. Success is
 * the operator's verdict (../operator.ts, required via --operator): finish is refused until
 * the operator has judged the current state. The operator also confirms the reset motion, and
 * request_scene_reset asks the operator to restore the scene before the arms reset again.
 *
 * --explore (../explore.ts, `/explore`) runs operator-judged attempts: `reset` is the operator's
 * scene reset, a success verdict is the solve (`terminated: true`), and the motion commands after
 * the last reset are exported as the cell's recipe (../memory).
 */

import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";
import { encodePng } from "../png.ts";
import { graspActive, graspArgs, graspTools, registerGraspFlags } from "../primitives/grasp.ts";
import { checkRotate, type MotionRig, moveDelta, rotateDelta, setGripper } from "../primitives/motion.ts";
import { viewCameraMeta, viewEnvState } from "../primitives/perception.ts";
import { getStep, outcome, type StepsIO, type ToolDef } from "../primitives/steps.ts";
import {
	attach,
	checkMove,
	defineRobot,
	frameOf,
	type Grid,
	gridOf,
	type Json,
	moveLimit,
	plain,
	type Rgb,
	rgbOf,
	round,
	roundAll,
	SERVICES,
	type Services,
	servicesEnv,
	servicesJson,
	sub,
	toolResult,
	vec,
	workspaceLimits,
} from "../robot.ts";
import { NdArray, type RpcClient } from "../rpc.ts";
import type { Move, MoveUnit, Vec3 } from "../units/index.ts";
import { alias, inlineWrist, policy, REWRITE, SETUP_PY, type Setup, type Step } from "./config.ts";
import { mountPerception } from "./perception.ts";
import { mountSkills } from "./skills.ts";

const SYSTEM = readFileSync(new URL("./SYSTEM.md", import.meta.url), "utf8");
const EXPLORE = readFileSync(new URL("./explore.md", import.meta.url), "utf8");

const MOTION = [
	"move_delta",
	"rotate_delta",
	"open_gripper",
	"close_gripper",
	"recover_joint_posture",
	"vla_right_grasp",
	"vla_handoff",
	"vla_left_place",
];
const TOOLS = [
	"describe_dual_franka_setup",
	"view_env_state",
	"view_camera_meta",
	"back_project",
	"segment",
	...MOTION,
];

/** Show-Harness configs/primitives_franka.yaml, in the shared right_base frame; one arm per unit. */
export const DUAL_FRANKA_UNITS = {
	vectors: {
		MV_FWD: [1, 0, 0],
		MV_BACK: [-1, 0, 0],
		MV_LEFT: [0, -1, 0],
		MV_RIGHT: [0, 1, 0],
		MV_UP: [0, 0, 1],
		MV_DOWN: [0, 0, -1],
	} as Record<MoveUnit, Vec3>,
	stepM: 0.02,
	yawStepRad: 0.15,
	arms: ["left", "right"] as readonly string[],
};

export default function dualFranka(pi: ExtensionAPI) {
	const flag = (name: string, fallback = "") => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("task", { type: "string", default: "0", description: "Dual-Franka task id (0, 1, 3, 4, 5)" });
	pi.registerFlag("robot-config", {
		type: "string",
		description: "Robot YAML (default: services/pi_embodied_services/robots/dual_franka/config/example.yaml)",
	});
	pi.registerFlag("robot-env", {
		type: "string",
		description: "Attach to a running dual-Franka env server instead of starting one",
	});
	pi.registerFlag("robot-vla", {
		type: "string",
		description: "Dual-Franka Pi0.5 VLA server (serve.sh: http://127.0.0.1:18210)",
	});
	pi.registerFlag("robot-sam3", {
		type: "string",
		description: "SAM3 server for segment (serve.sh: http://127.0.0.1:18310)",
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
		description: "Largest move_delta per call and arm, m (a task's documented limit applies if tighter)",
	});
	pi.registerFlag("max-rotate", {
		type: "string",
		default: "0.5",
		description: "Largest rotate_delta per call and arm, rad (norm of delta_rpy)",
	});

	pi.registerFlag("workspace-xy", {
		type: "string",
		default: "0.1,1.15,-0.85,0.85",
		description:
			"right_base TCP x/y box for move_delta and units, both arms, m: xmin,xmax,ymin,ymax (default: example.yaml's tabletop volume; '' = off)",
	});
	pi.registerFlag("z-floor", {
		type: "string",
		default: "",
		description:
			"Lowest right_base TCP z for move_delta and units, m (required: the robot does not start without it)",
	});
	// --graspnet/--graspgenx/--anyplace/--anygrasp: plan_grasp, plan_place, check_attached (../primitives/grasp.ts).
	registerGraspFlags(pi);

	let env: RpcClient | undefined;
	let vla: RpcClient | undefined;
	let sam3: RpcClient | undefined;
	let setup: Setup | undefined;
	let envMeta: Json = {};
	let out = "";
	let lastStates: unknown;
	const steps: Step[] = [];
	/** The first step after the last scene reset; perception refuses older steps. */
	let attemptStart = 0;
	const task = () => robot.task.task;
	const exploring = () => pi.getFlag("explore") === true;
	const judgedSuccess = () => (op.result() as Json).operator_verdict === "success";
	const robot = defineRobot(pi, {
		name: "dual_franka",
		task: ["task"],
		keepImages: 4,
		video: true,
		// Observations carry the policy's inline cameras (the D455 by default), wrist views among them.
		vdm: () =>
			shown.length
				? { views: shown.length, wrist: shown.flatMap((v, i) => (v.includes("wrist") ? [i] : [])) }
				: undefined,
		// The evaluation prompt names no memory; the guard also opens the step artifacts.
		memory: {
			cell: () => ({ tag: `dual_franka_t${task()}`, reference: "" }),
			primitives: MOTION,
			readable: () => [out],
		},
		operator: { step: () => steps.length, reset: resetRobot },
		explore: {
			// The operator restores the scene; a failed or unconfirmed reset throws and starts no attempt.
			reset: async (result, ctx, signal) => {
				const r: Json = await op.sceneReset(ctx, String(result.reason ?? ""), setup?.task.setup ?? "", signal);
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
			rewrite: REWRITE,
			// RPent's real-robot defaults: every attempt costs the operator a manual scene reset.
			budget: { sessions: 1, attempts: 3 },
			operatorJudged: true,
		},
		// The env server's primitive registry (services robots/franka/primitives.py, the dual-arm set).
		codeApi: () => env,
		start: startRobot,
		stop: () => {
			env = vla = sam3 = undefined;
		},
		prompt: () => {
			if (!setup) return undefined;
			const t = setup.task;
			const vars: Record<string, string> = {
				task_name: t.name,
				instruction: t.instruction,
				setup: t.setup,
				success_criteria: t.success_criteria,
				constraints: t.constraints.map((c, i) => `${i + 1}. ${c}`).join("\n"),
			};
			return SYSTEM.replace(/\{\{(\w+)\}\}/g, (m, k: string) => vars[k] ?? m);
		},
		result: () => ({
			task: Number(task()),
			task_name: setup?.task.name ?? null,
			steps: steps.length,
			solved: (op.result() as Json).operator_verdict === "success",
			out,
		}),
		status: () => ({ step: steps.length - 1, solved: (op.result() as Json).operator_verdict === "success" }),
		units: {
			...DUAL_FRANKA_UNITS,
			maxYawRad: () => maxRotate(),
			maxMoveM: () => moveLimit(Number(flag("max-move", "0.1")), setup?.task.constraints),
			apply: (move, signal) => unitStep(move, signal),
			state: async (arm) => {
				const a = armState(arm ?? "right");
				const width = vec(a.gripper_position);
				return {
					eef_xyz: roundAll(vec(a.tcp_pose).slice(0, 3)),
					...(width.length ? { gripper_width: round(width[0]) } : {}),
					gripper_open: a.gripper_open ?? null,
					table_z: Number(flag("z-floor")),
				};
			},
			instruction: () => setup?.task.instruction ?? "",
			// Only the policy's inline cameras reach the model (the D455 alone by default): a wrist view only
			// when cameras.agent_observation.inline_cameras names one (the latest step's camera meta).
			wrist: () => inlineWrist(steps[steps.length - 1]?.meta ?? envMeta),
			views: "Each result shows the configured inline front view with both arms (Show-Harness's front-view convention: MV_LEFT / MV_RIGHT move the chosen arm toward the image left / right, MV_FWD toward the image bottom, MV_BACK toward the image top). Verify the first move of each arm against the image before relying on it.",
			emptyWidthM: 0.001,
		},
		finish: {
			description:
				"Call when the task is complete or unrecoverable. Halts the agent loop. Real-robot tasks require request_operator_verdict first so the operator can judge the physical state.",
			parameters: Type.Object({
				status: Type.String({ description: "Outcome, e.g. 'success', 'failure', or 'stuck'." }),
				summary: Type.String({ description: "Short natural-language summary of the run." }),
			}),
			result: (params) => toolResult({ _finish: true, ...params, ...op.result() }),
		},
	});
	const { op } = robot;

	const call = <T = Json>(method: string, kwargs: Json = {}, timeoutMs = 30_000, signal?: AbortSignal) =>
		(env as RpcClient).call<T>(method, kwargs, timeoutMs, [], signal);
	const remember = (states: unknown) => {
		if (states !== undefined && states !== null) lastStates = states;
	};
	const maxRotate = () => Number(flag("max-rotate", "0.5"));
	const check = (signal?: AbortSignal) => {
		op.check();
		if (signal?.aborted) throw new Error("tool operation interrupted");
		if (exploring() && judgedSuccess())
			throw new Error(
				"motion refused: the operator judged this attempt a success. Write the audit and memory drafts, then call finish.",
			);
	};

	/** The operator's request_scene_reset (after the operator confirmed the scene): RLinf's reset, then a fresh step. */
	async function resetRobot(): Promise<Json> {
		if (!env || !steps.length) throw new Error("dual_franka is not initialized; see the session start error");
		const r = await call("env.reset", {}, 180_000);
		if (r?.ok !== true || r.error) throw new Error(`env.reset failed: ${JSON.stringify(plain(r))}`);
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

	// ---- env client

	async function observation(): Promise<Json> {
		const obs = await call("env.get_observation");
		if (!("states" in obs)) throw new Error("Dual-Franka server must return live states; restart the updated server");
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

	// ---- state steps (dual-Franka state dumps)

	async function dumpState(command: Json | null, result: Json | null, elapsed: number | null): Promise<Step> {
		const obs = await observation();
		const state = await robotState();
		const meta = await call<Json | null>("env.get_camera_meta");
		const map: Json = meta?.observation_camera_map ?? {};
		const images = new Map<string, NdArray>();
		const depths = new Map<string, NdArray>();
		const put = (m: Map<string, NdArray>, name: string | undefined, v: unknown) => {
			if (name && v instanceof NdArray) m.set(name, v);
		};
		const main = alias(map.main) ?? "left_wrist";
		put(images, main, obs.main_images);
		let extras = obs.extra_view_images;
		if (extras instanceof NdArray && extras.shape.length === 5) extras = sub(extras, 0);
		if (extras instanceof NdArray && extras.shape.length === 4)
			for (let i = 0; i < extras.shape[0]; i++)
				put(images, alias(map[`extra_${i}`]) ?? `extra_${i}`, sub(extras, i));
		for (const [k, v] of Object.entries(obs.raw_camera_frames ?? {})) put(images, alias(k), v);
		put(depths, main, obs.main_depths);
		let extraDepths = obs.extra_view_depths;
		if (extraDepths instanceof NdArray && extraDepths.shape.length === 4 && extraDepths.shape[0] === 1)
			extraDepths = sub(extraDepths, 0);
		if (extraDepths instanceof NdArray && extraDepths.shape.length === 3)
			for (let i = 0; i < extraDepths.shape[0]; i++)
				put(depths, alias(map[`extra_${i}`]) ?? `extra_${i}`, sub(extraDepths, i));
		for (const [k, v] of Object.entries(obs.raw_camera_depths ?? {})) put(depths, alias(k), v);
		for (const [key, v] of Object.entries(obs)) {
			for (const [suffix, m] of [
				["_images", images],
				["_depths", depths],
			] as const) {
				const a = key.endsWith(suffix) ? key.slice(0, -suffix.length) : "";
				if (a && !["main", "extra_view", "raw_camera"].includes(a)) put(m, a, v);
			}
		}

		const idx = steps.length;
		const dir = join(out, `step_${String(idx).padStart(4, "0")}`);
		mkdirSync(dir, { recursive: true });
		const artifacts: string[] = [];
		for (const [name, v] of images) {
			const img = rgbOf(v);
			writeFileSync(join(dir, `${name}.png`), encodePng(img.rgb, img.width, img.height));
			writeFileSync(join(dir, `${name}.rgb`), img.rgb);
			writeFileSync(join(dir, `${name}.json`), JSON.stringify({ width: img.width, height: img.height }));
			artifacts.push(`${name}.png`);
		}
		for (const [name, v] of depths) {
			const g = gridOf(v);
			writeFileSync(join(dir, `${name}_depth.f32`), Buffer.from(g.data.buffer));
			writeFileSync(join(dir, `${name}_depth.json`), JSON.stringify({ height: g.height, width: g.width }));
			artifacts.push(`${name}_depth.f32`);
		}
		const plainMeta = meta ? (plain(meta) as Json) : null;
		// The episode video follows the policy's first inline camera (the D455 by default).
		const lead = policy(plainMeta).inline_cameras.find((c) => images.has(c)) ?? [...images.keys()][0];
		if (lead) robot.video.frame(frameOf(images.get(lead) as NdArray));
		if (plainMeta) {
			writeFileSync(join(dir, "camera_meta.json"), JSON.stringify(plainMeta));
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
		const step = { blob, dir, meta: plainMeta, views: [...images.keys()] };
		steps.push(step);
		return step;
	}

	const loadRgb = (s: Step, name: string): Rgb => ({
		...JSON.parse(readFileSync(join(s.dir, `${name}.json`), "utf8")),
		rgb: readFileSync(join(s.dir, `${name}.rgb`)),
	});

	function loadDepth(s: Step, name: string): Grid | undefined {
		const path = join(s.dir, `${name}_depth.f32`);
		if (!existsSync(path)) return undefined;
		const { height, width } = JSON.parse(readFileSync(join(s.dir, `${name}_depth.json`), "utf8"));
		return { height, width, data: new Float32Array(new Uint8Array(readFileSync(path)).buffer) };
	}

	/** The inline camera views of the last observation, in image order (VDM reads which are wrists). */
	let shown: string[] = [];

	/** view_env_state: step blob, every view as an artifact path, inline views as images. */
	function view(s: Step) {
		const pol = policy(s.meta);
		const output: Json = { ...s.blob, images: [], artifact_images: [] };
		const views = s.blob.artifacts
			.filter((a: string) => a.endsWith(".png") && !a.includes("_segment_overlay_") && !a.includes("_back_project_"))
			.map((a: string) => a.slice(0, -4))
			.sort();
		output.available_camera_views = views;
		output.agent_observation = pol;
		for (const v of views) {
			output[`image_${v}_path`] = join(s.dir, `${v}.png`);
			output.artifact_images.push(v);
		}
		const pngs: Buffer[] = [];
		pol.inline_cameras.forEach((v, i) => {
			if (i >= 4 || !views.includes(v)) return;
			pngs.push(readFileSync(join(s.dir, `${v}.png`)));
			output.images.push(v);
		});
		output.image_block_order = [...output.images];
		shown = [...output.images];
		return { output, pngs };
	}

	// ---- tools

	/** The recorded-state layer (../primitives/steps.ts): mutating tools record and return a fresh step. */
	const io: StepsIO<Step> = { steps, ready: () => env !== undefined, dump: dumpState, view };

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
		mutating = MOTION.includes(name),
	) {
		robot.tool(name, description, parameters, (params, signal, ctx) =>
			outcome(io, name, params as Json, () => run(params, signal, ctx), mutating),
		);
	}
	/** Mount a shared primitive (../primitives) as one of this robot's tools. */
	const mount = <P extends TSchema>(d: ToolDef<P>) => tool(d.name, d.description, d.parameters, d.run);

	/** One arm's state in the latest step (tcp_pose in right_base). */
	const armState = (arm: string): Json => steps[steps.length - 1]?.blob.state?.[`${arm}_arm`] ?? {};

	/** Refuse a move whose right_base target leaves the --workspace-xy box or goes below --z-floor (unless it moves back in). */
	function checkWorkspace(arm: string, delta: number[]) {
		const { box, floor } = workspaceLimits(flag("workspace-xy"), flag("z-floor"));
		const tcp = vec(armState(arm).tcp_pose).slice(0, 3);
		if (tcp.length !== 3) throw new Error(`the ${arm} arm's tcp_pose is missing from the latest state`);
		const outside = (p: number[]) =>
			(box ? Math.max(0, box[0] - p[0], p[0] - box[1]) + Math.max(0, box[2] - p[1], p[1] - box[3]) : 0) +
			Math.max(0, floor - p[2]);
		const target = tcp.map((v, i) => v + delta[i]);
		if (outside(target) > 1e-6 && outside(target) >= outside(tcp) - 1e-6)
			throw new Error(
				`the ${arm} move ends at [${roundAll(target, 3)}], outside the right_base workspace (${box ? `x ${box[0]}..${box[1]}, y ${box[2]}..${box[3]}, ` : ""}z >= ${floor} m; --workspace-xy / --z-floor)`,
			);
	}

	/** Units mode (../units): one grounded action unit for one arm on the existing move primitives. */
	function unitStep(move: Move, signal: AbortSignal | undefined) {
		return outcome(io, "act", { move }, async () => {
			check(signal);
			const arm = armName(move.arm);
			const out: Json = { arm };
			if (move.gripper)
				out.gripper = await motion("env.set_gripper", { arm, open: move.gripper === "open" }, signal);
			if (Math.hypot(...move.delta) > 0) {
				checkMove(move.delta, Number(flag("max-move", "0.1")), setup?.task.constraints);
				checkWorkspace(arm, move.delta);
				out.move = await motion("env.move_delta", { arm, delta_xyz: NdArray.f32(move.delta) }, signal);
			}
			if (move.yaw) {
				checkRotate([0, 0, move.yaw], maxRotate());
				out.rotate = await motion("env.rotate_delta", { arm, delta_rpy: NdArray.f32([0, 0, move.yaw]) }, signal);
			}
			return out;
		});
	}

	const arm = StringEnum(["left", "right"] as const, {
		description: "Which arm to command; the other arm is left uncommanded.",
	});

	tool(
		"describe_dual_franka_setup",
		"Read the dual-Franka runtime conventions, camera aliases, VLA policy conditioning text, semantic stop rules, and available primitive names before acting. This is read-only.",
		Type.Object({}),
		async () => {
			const pol = policy(envMeta);
			const resetTool = exploring() ? "reset" : "request_scene_reset";
			return {
				ok: true,
				phase: exploring() ? "exploration" : "strict",
				reset_policy: `The operator confirmed the scene and the runner reset the robot at session start. ${resetTool} asks the operator to restore the scene, then resets both arms; use it only for a new attempt${exploring() ? " (after closing out the failed one)" : " when the operator must restore the scene"}.`,
				coordinate_frame: "right_base",
				camera_aliases: envMeta.observation_camera_map ?? {},
				projection_views: envMeta.projection_views ?? {},
				agent_observation: pol,
				vla: {
					policy_instruction: setup?.task.vla_instruction || setup?.task.instruction,
					num_action_chunks: 20,
					action_dim: 20,
					num_images_in_input: 3,
					external_localization_views_in_policy_input: false,
					agent_visible_images: pol.inline_cameras,
					auxiliary_artifact_images: pol.auxiliary_cameras,
					skill_stop_rules: {
						enabled: true,
						grasp_lift_m: 0.15,
						place_lift_m: 0.1,
						handoff_release_delay_s: 1.5,
						min_steps_after_gripper_event: 2,
					},
				},
				sam3: {
					tool: "segment",
					status:
						"optional; returns an error and falls back to manual camera projection when no SAM3 client is configured",
					usage: "Use text prompt or one [row, col] positive camera point, then inspect the returned mask overlay before trusting point_xyz. SAM3 text grounding is phrase-sensitive; if a prompt returns a very low score, retry a shorter/rephrased prompt or point prompt rather than lowering min_score blindly.",
				},
				available_primitives: [
					...TOOLS,
					...op.tools().map((t) => (exploring() && t === "request_scene_reset" ? "reset" : t)),
					"finish",
				],
				operator_guidance:
					"Named VLA semantic boundaries are segment boundaries, not proof of physical success. Verify images, gripper widths/open flags, joint_health, and projection evidence after every action. recover_joint_posture re-commands and preserves each gripper's open/closed state; inspect its gripper_preserved result before continuing.",
			};
		},
	);

	mount(
		viewEnvState(
			io,
			"Read a dual-Franka state snapshot. Configured inline camera views are returned directly; other available views are returned as artifact paths; use read to inspect these artifacts.",
		),
	);
	mount(viewCameraMeta(io, "Read camera intrinsics, serials, and projection metadata for the dual-Franka rig."));

	// ---- perception

	mountPerception({ tool, freshStep, setup: () => setup, sam3: () => sam3, loadRgb, loadDepth });

	// ---- analytic primitives

	const armName = (v: unknown) => {
		const a = String(v).trim().toLowerCase();
		if (a !== "left" && a !== "right") throw new Error("arm must be exactly 'left' or 'right'");
		return a;
	};

	/** The motion primitives (../primitives/motion.ts) on either arm: this rig's env, limits, workspace and gate. */
	const rig: MotionRig = {
		check,
		motion,
		maxMove: () => Number(flag("max-move", "0.1")),
		maxRotate,
		constraints: () => setup?.task.constraints,
		workspace: (delta, a) => checkWorkspace(a!, delta),
		arm: { schema: arm, name: armName },
	};
	mount(moveDelta(rig, "Move one Franka TCP by a bounded world-frame xyz delta in meters."));
	mount(rotateDelta(rig, "Rotate one Franka TCP by a bounded world-frame rpy delta in radians."));
	mount(setGripper(rig, true, "Open one Franka gripper and wait for the command to settle."));
	mount(setGripper(rig, false, "Close one Franka gripper and wait for the command to settle."));

	tool(
		"recover_joint_posture",
		"Reset both arms to their healthy configured joint posture while preserving each gripper's open/closed state. Closed grippers are re-commanded before/after the joint reset so held objects stay clamped, then both TCPs return near their prior poses.",
		Type.Object({
			reason: Type.Optional(Type.String({ description: "Default ''" })),
			return_to_start: Type.Optional(Type.Boolean({ description: "Default true" })),
		}),
		async ({ reason = "", return_to_start = true }, signal) => {
			check(signal);
			return motion(
				"env.recover_joint_posture",
				{ reason: String(reason), return_to_start: Boolean(return_to_start) },
				signal,
				240_000,
			);
		},
	);

	// ---- named VLA skills

	mountSkills({
		tool,
		check,
		call,
		observation,
		robotState,
		remember,
		setup: () => setup,
		vla: () => vla,
	});

	// plan_grasp / plan_place / check_attached (../primitives/grasp.ts): the env server plans over its
	// calibrated RGB-D cameras; active with --graspnet/--graspgenx/--anyplace/--anygrasp.
	for (const d of graspTools(pi, {
		call: (method, kwargs, timeoutMs) => call(method, kwargs, timeoutMs ?? 120_000),
		task: () => setup?.task.instruction ?? "",
		arm: { schema: arm, name: armName },
	}))
		tool(d.name, d.description, d.parameters, d.run, false);

	// ---- lifecycle

	async function startRobot(ctx: ExtensionContext) {
		if (!ctx.hasUI)
			throw new Error(
				"dual_franka drives real robots: run pi interactively (or over RPC) so an operator is present",
			);
		if (pi.getFlag("operator") !== true)
			throw new Error("dual_franka needs --operator: success on this robot is the operator's verdict");
		workspaceLimits(flag("workspace-xy"), flag("z-floor"));
		// Ray must not re-run uv for workers on the pre-provisioned nodes (env override).
		const r: Services = {
			root: flag("services"),
			python: flag("python", "python"),
			env: { RAY_ENABLE_UV_RUN_RUNTIME_ENV: "0" },
		};
		const configFlag = pi.getFlag("robot-config");
		const config = typeof configFlag === "string" && configFlag ? resolve(ctx.cwd, configFlag) : "";
		setup = await servicesJson<Setup>(r, SETUP_PY, [task(), config]);
		const vlaEndpoint = flag("robot-vla");
		if (setup.task.vla_instruction !== null && !vlaEndpoint)
			throw new Error(`task ${task()} runs named VLA skills: start dual_franka/serve.sh and pass --robot-vla`);
		const stamp = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
		out = resolve(ctx.cwd, flag("out") || join(tmpdir(), "pi-embodied", `dual_franka_t${task()}_${stamp}`));
		mkdirSync(out, { recursive: true });
		const envEndpoint = flag("robot-env");
		const [envRpc, vlaRpc, sam3Rpc] = await Promise.all([
			envEndpoint
				? attach(envEndpoint)
				: robot.serve({
						python: r.python,
						args: [
							...["-m", "pi_embodied_services.robots.dual_franka.env_server"],
							...["--task-description", setup.task.instruction, ...(config ? ["--robot-config", config] : [])],
							// SAM3 on the env server too: plan_grasp / plan_place segment their object and region text there.
							...(flag("robot-sam3") ? ["--sam3", flag("robot-sam3")] : []),
							...graspArgs(pi),
						],
						cwd: r.root,
						env: servicesEnv(r),
						log: () => join(out, "dual_franka_env_server.log"),
					}),
			vlaEndpoint ? attach(vlaEndpoint) : undefined,
			flag("robot-sam3") ? attach(flag("robot-sam3")) : undefined,
		]);
		envMeta = plain(await envRpc.call<Json>("env.get_env_meta", {}, 30_000)) as Json;
		const go = await ctx.ui.confirm(
			exploring() ? "Restore the scene and reset both Franka arms?" : "Reset both Franka arms?",
			`${exploring() ? `Exploration attempt 1: restore the tabletop to the task's initial layout (${setup.task.setup}). ` : ""}RLinf's reset opens both grippers and moves both arms to the configured reset posture. Remove held objects, clear the workspace and keep both emergency stops in reach.`,
		);
		if (!go) throw new Error("operator declined the reset; the dual-Franka tools stay disabled");
		await envRpc.call("env.reset", {}, 180_000);
		env = envRpc;
		vla = vlaRpc;
		sam3 = sam3Rpc;
		attemptStart = (await dumpState(null, null, null)).blob.step_idx;
		ctx.ui.notify(`Dual Franka ready: task ${task()} (${setup.task.name}); steps under ${out}`, "info");
		return [...TOOLS, "finish", ...graspActive(pi)];
	}
}
