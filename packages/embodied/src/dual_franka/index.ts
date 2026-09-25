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
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";
import { decodePngChannel, encodePng } from "../png.ts";
import {
	apply,
	attach,
	checkMove,
	defineRobot,
	f32,
	type Grid,
	gridOf,
	type Json,
	type Mat,
	mark,
	median,
	message,
	numbers,
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
	u8,
	vec,
} from "../robot.ts";
import { NdArray, type RpcClient } from "../rpc.ts";
import type { Move } from "../units/index.ts";

const SYSTEM = readFileSync(new URL("./SYSTEM.md", import.meta.url), "utf8");
const EXPLORE = readFileSync(new URL("./explore.md", import.meta.url), "utf8");
/** The prompt's single-episode lines, which exploration replaces rather than contradicts. */
const REWRITE: [RegExp, string][] = [
	[
		/^5\. Finish only when the success evidence is visible and consistent with state\.$/m,
		"5. This is an exploration run: follow the Exploration workflow below. Success is only the operator's verdict.",
	],
	[
		/^Call describe_dual_franka_setup before acting\..*$/m,
		"Call describe_dual_franka_setup, then read the relevant memory and prior attempt notes and inspect view_env_state before the first motion. Follow the operator-confirmed exploration workflow. Do not finish successfully without a current operator verdict.",
	],
];

type Task = {
	name: string;
	instruction: string;
	success_criteria: string;
	constraints: string[];
	setup: string;
	vla_instruction: string | null;
};
type Camera = { T_right_camera: Mat; localization_validity: Json };
type View = { raw_key: string; calibration_key: string; display_name: string };
type Setup = {
	task: Task;
	cameras?: Record<string, Camera>;
	projection_views?: Record<string, View>;
	calibration_source?: string;
	T_right_base_left_base?: Mat;
	calibration_error?: string;
};
type Step = { blob: Json; dir: string; meta: Json | null; views: string[] };
type Boundary = "grasp" | "handoff" | "place";

/** Task, calibration and perception config from the services (parse_config checks). */
const SETUP_PY = `
import dataclasses, json, sys
from pi_embodied_services.robots.dual_franka.runtime_config import DEFAULT_CONFIG
from pi_embodied_services.robots.dual_franka.tasks import get_dual_franka_task
from pi_embodied_services.robots.franka.runtime_config import describe_calibration_source, set_robot_config_path, validate_calibration_sources
set_robot_config_path(sys.argv[2] or DEFAULT_CONFIG)
validate_calibration_sources()
out = {"task": dataclasses.asdict(get_dual_franka_task(int(sys.argv[1])))}
try:
    from pi_embodied_services.robots.dual_franka import perception as p
    bundle = p.load_calibration_bundle()
    out["cameras"] = {k: {"T_right_camera": p._transform_to_matrix(v["transformation"]), "localization_validity": v.get("localization_validity") or {}} for k, v in bundle.items() if isinstance(v, dict) and "transformation" in v}
    out["projection_views"] = p._projection_cameras()
    out["calibration_source"] = describe_calibration_source()
    try:
        out["T_right_base_left_base"] = p._base_frame_transform(bundle, target="right_base", source="left_base")
    except ValueError:
        pass
except Exception as exc:
    out["calibration_error"] = str(exc)
print(json.dumps(out, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))
`;

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

/** `left_wrist_0_rgb` -> `left_wrist`. */
function alias(key: unknown): string | undefined {
	let k = String(key ?? "");
	if (k.endsWith("_rgb")) k = k.slice(0, -4);
	if (k.endsWith("_0")) k = k.slice(0, -2);
	return k || undefined;
}

function policy(meta: Json | null | undefined): { inline_cameras: string[]; auxiliary_cameras: string[] } {
	const raw = meta?.agent_observation && typeof meta.agent_observation === "object" ? meta.agent_observation : {};
	const strings = (v: unknown) => (Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []);
	return {
		inline_cameras: strings(raw.inline_cameras ?? ["d455"]),
		auxiliary_cameras: strings(raw.auxiliary_cameras ?? ["left_wrist", "base", "right_wrist"]),
	};
}

function coerceViews(configured: unknown): Record<string, View> {
	if (!configured || typeof configured !== "object") throw new Error("perception.projection_views must be a mapping");
	return Object.fromEntries(
		Object.entries(configured as Json).map(([a, c]) => {
			if (!c || typeof c !== "object") throw new Error(`perception.projection_views.${a} must be a mapping`);
			return [
				a,
				{
					raw_key: String(c.raw_key ?? `${a}_rgb`),
					calibration_key: String(c.calibration_key ?? `${a}_camera`),
					display_name: String(c.display_name ?? a),
				},
			];
		}),
	);
}

/** Validate a localization point: configured depth range and right_base tabletop volume. */
function validity(config: Json, depth: number, p: number[]) {
	const d: number[] = config.depth_m ?? [0.15, 1.25];
	const lo: number[] = config.right_base_xyz_min ?? [0.1, -0.85, 0.0];
	const hi: number[] = config.right_base_xyz_max ?? [1.15, 0.85, 0.85];
	const reasons: string[] = [];
	if (!(d[0] <= depth && depth <= d[1]))
		reasons.push(
			`depth ${depth.toFixed(3)}m is outside configured target range [${d[0].toFixed(3)}, ${d[1].toFixed(3)}]m`,
		);
	const axes = ["x", "y", "z"].filter((_, i) => !(lo[i] <= p[i] && p[i] <= hi[i]));
	if (axes.length)
		reasons.push(
			`right_base point is outside the configured tabletop localization volume on axis/axes ${axes.join(",")}`,
		);
	const contract = { depth_m: roundAll(d), right_base_xyz_min: roundAll(lo), right_base_xyz_max: roundAll(hi) };
	return { ok: reasons.length === 0, reasons, contract };
}

function nextIndex(s: Step, prefix: string, suffix: string): number {
	let i = 0;
	while (s.blob.artifacts.includes(`${prefix}_${String(i).padStart(2, "0")}${suffix}`)) i++;
	return i;
}

/** Python's round(): halves go to the even neighbour. */
const rint = (v: number) => {
	const f = Math.floor(v);
	return v - f === 0.5 ? f + (f % 2 === 0 ? 0 : 1) : Math.round(v);
};
const short = (v: unknown) =>
	Array.isArray(v) && v.length >= 3
		? v
				.slice(0, 3)
				.map((x) => Number(x).toFixed(3))
				.join(",")
		: "n/a";

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

	pi.registerFlag("workspace-xy", {
		type: "string",
		default: "0.1,1.15,-0.85,0.85",
		description:
			"right_base TCP x/y box for move_delta and units, both arms, m: xmin,xmax,ymin,ymax (default: example.yaml's tabletop volume; '' = off)",
	});
	pi.registerFlag("z-floor", {
		type: "string",
		default: "",
		description: "Lowest right_base TCP z for move_delta and units, m ('' = off)",
	});

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
		},
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
			// Show-Harness configs/primitives_franka.yaml, in the shared right_base frame; one arm per unit.
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
			arms: ["left", "right"],
			apply: (move, signal) => unitStep(move, signal),
			state: async (arm) => {
				const a = armState(arm ?? "right");
				const width = vec(a.gripper_position);
				return {
					eef_xyz: roundAll(vec(a.tcp_pose).slice(0, 3)),
					...(width.length ? { gripper_width: round(width[0]) } : {}),
					gripper_open: a.gripper_open ?? null,
					...(flag("z-floor") ? { table_z: Number(flag("z-floor")) } : {}),
				};
			},
			instruction: () => setup?.task.instruction ?? "",
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
		const s = getStep(step);
		if (s.blob.step_idx < attemptStart)
			throw new Error(
				`localization refused: step ${s.blob.step_idx} predates the last scene reset (step ${attemptStart}); use a fresh observation`,
			);
		return s;
	}

	// Exploration's success signal is `terminated` (../explore.ts, the memory recipe); here it is the operator's success verdict.
	pi.on("tool_result", (event) => {
		if (!exploring() || event.toolName !== "request_operator_verdict" || event.isError) return undefined;
		const details = event.details as Json | undefined;
		if (details?.status !== "success") return undefined;
		const marked = { ...details, terminated: true };
		return { details: marked, content: [{ type: "text" as const, text: JSON.stringify(marked) }] };
	});
	// In exploration `reset` is the scene reset: it counts attempts and bounds the recipe.
	pi.on("before_agent_start", () => {
		if (exploring()) pi.setActiveTools(pi.getActiveTools().filter((t) => t !== "request_scene_reset"));
	});

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

	function getStep(step?: number | null): Step {
		const i = step === undefined || step === null ? steps.length - 1 : step < 0 ? steps.length + step : step;
		const s = steps[i];
		if (!s) throw new Error(`step ${step} is not recorded (have 0..${steps.length - 1})`);
		return s;
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
		return { output, pngs };
	}

	// ---- tools

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
			outcome(name, params as Json, () => run(params, signal, ctx), mutating),
		);
	}

	/** Run a tool body; a mutating one then records a fresh state step and returns it (errors included). */
	async function outcome(name: string, params: Json, run: () => Promise<Json>, mutating = true) {
		if (!env || !steps.length) return toolResult({ error: "robot not initialized; see the session start error" });
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
		const elapsed = round((performance.now() - started) / 1000, 2);
		try {
			const { output, pngs } = view(await dumpState({ action: name, ...params }, result, elapsed));
			output.agent_elapsed_s = elapsed;
			if (failed) for (const [k, v] of Object.entries(result)) output[k] ??= v;
			return toolResult(output, pngs);
		} catch (err) {
			return toolResult({
				...result,
				state_capture_error: message(err),
				error: result.error ?? `failed to capture state after ${name}: ${message(err)}`,
			});
		}
	}

	/** One arm's state in the latest step (tcp_pose in right_base). */
	const armState = (arm: string): Json => steps[steps.length - 1]?.blob.state?.[`${arm}_arm`] ?? {};

	/** Refuse a move whose right_base target leaves the --workspace-xy box or goes below --z-floor (unless it moves back in). */
	function checkWorkspace(arm: string, delta: number[]) {
		const box = flag("workspace-xy").split(",").filter(Boolean).map(Number);
		const floor = flag("z-floor") ? Number(flag("z-floor")) : undefined;
		if (box.length !== 4 && floor === undefined) return;
		if (box.length && (box.length !== 4 || !box.every(Number.isFinite)))
			throw new Error(`--workspace-xy must be "xmin,xmax,ymin,ymax", got "${flag("workspace-xy")}"`);
		const tcp = vec(armState(arm).tcp_pose).slice(0, 3);
		if (tcp.length !== 3) throw new Error(`the ${arm} arm's tcp_pose is missing from the latest state`);
		const outside = (p: number[]) =>
			(box.length === 4
				? Math.max(0, box[0] - p[0], p[0] - box[1]) + Math.max(0, box[2] - p[1], p[1] - box[3])
				: 0) + (floor !== undefined ? Math.max(0, floor - p[2]) : 0);
		const target = tcp.map((v, i) => v + delta[i]);
		if (outside(target) > 1e-6 && outside(target) >= outside(tcp) - 1e-6)
			throw new Error(
				`the ${arm} move ends at [${roundAll(target, 3)}], outside the right_base workspace (x ${box[0]}..${box[1]}, y ${box[2]}..${box[3]}${floor !== undefined ? `, z >= ${floor}` : ""} m; --workspace-xy / --z-floor)`,
			);
	}

	/** Units mode (../units): one grounded action unit for one arm on the existing move primitives. */
	function unitStep(move: Move, signal: AbortSignal | undefined) {
		return outcome("act", { move }, async () => {
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
			if (move.yaw)
				out.rotate = await motion("env.rotate_delta", { arm, delta_rpy: NdArray.f32([0, 0, move.yaw]) }, signal);
			return out;
		});
	}

	const stepParam = Type.Optional(Type.Integer({ description: "State step (default -1 = latest)" }));
	const cameraParam = Type.Optional(
		Type.String({
			description:
				"Registered projection view name, e.g. d455 or base (default d455). Valid names come from perception.projection_views and the current state's saved artifacts.",
		}),
	);
	const arm = Type.Union([Type.Literal("left"), Type.Literal("right")], {
		description: "Which arm to command; the other arm is left uncommanded.",
	});
	const xyz = Type.Array(Type.Number(), { minItems: 3, maxItems: 3 });

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

	tool(
		"view_env_state",
		"Read a dual-Franka state snapshot. Configured inline camera views are returned directly; other available views are returned as artifact paths; use read to inspect these artifacts.",
		Type.Object({ step: stepParam }),
		async ({ step = -1 }) => {
			const { output, pngs } = view(getStep(step));
			return { ...output, _pngs: pngs };
		},
	);

	tool(
		"view_camera_meta",
		"Read camera intrinsics, serials, and projection metadata for the dual-Franka rig.",
		Type.Object({ step: stepParam }),
		async ({ step = -1 }) => {
			const s = getStep(step);
			if (!s.meta) return { error: "camera metadata is unavailable", step };
			return { step: s.blob.step_idx, camera_meta: s.meta };
		},
	);

	// ---- perception

	function projectionView(s: Step, camera: string): View {
		const views =
			s.meta?.projection_views && typeof s.meta.projection_views === "object"
				? coerceViews(s.meta.projection_views)
				: (setup?.projection_views ?? {});
		const v = views[camera];
		if (!v)
			throw new Error(
				`unsupported projection camera: '${camera}'; registered=${Object.keys(views).sort().join(", ") || "<none>"}`,
			);
		return v;
	}

	function intrinsics(s: Step, camera: string, rawKey: string) {
		if (!s.meta) throw new Error(`${camera} camera metadata not found`);
		for (const key of [rawKey, camera, ...(camera === "base" ? ["extra_0"] : [])]) {
			const intr = s.meta[key]?.color_intrinsics;
			if (intr)
				return {
					fx: Number(intr.fx),
					fy: Number(intr.fy),
					cx: Number(intr.ppx ?? intr.cx),
					cy: Number(intr.ppy ?? intr.cy),
				};
		}
		throw new Error(`${camera} RealSense color intrinsics not found`);
	}

	function calibrationFor(key: string): Camera {
		if (!setup?.cameras) throw new Error(setup?.calibration_error ?? "no hand-eye calibration loaded");
		const c = setup.cameras[key];
		if (!c) throw new Error(`calibration entry '${key}' is missing`);
		return c;
	}

	/** Both TCP positions and TCP-to-point deltas in right_base. */
	function tcpDeltas(state: Json, point: number[]): Json {
		const out: Json = {
			tcp_delta_coordinate_frame: "right_base",
			tcp_delta_contract:
				"left_tcp_xyz, right_tcp_xyz, and both TCP-to-point deltas are expressed in the shared right_base world frame.",
		};
		const record = state?.state && typeof state.state === "object" ? state.state : state;
		for (const a of ["left", "right"]) {
			const armState: Json = record?.raw?.[a] ?? record?.[`${a}_arm`] ?? {};
			const pose = vec(armState.tcp_pose);
			if (pose.length < 3) continue;
			const frame = armState.tcp_pose_frame || armState.coordinate_frame || record?.coordinate_frame || `${a}_base`;
			let tcp = pose.slice(0, 3);
			if (frame !== "right_base") {
				if (frame !== "left_base" || !setup?.T_right_base_left_base)
					throw new Error(`missing base-frame transform T_right_base_${frame}`);
				tcp = apply(setup.T_right_base_left_base, tcp);
			}
			out[`${a}_tcp_xyz`] = roundAll(tcp);
			out[`delta_${a}_tcp_to_point_xyz`] = roundAll(point.map((v, k) => v - tcp[k]));
		}
		return out;
	}

	tool(
		"back_project",
		"Back-project one pixel from a registered RGBD camera view into shared right-base coordinates. Use a camera listed by view_env_state/view_camera_meta; default is the configured primary metric localization camera.",
		Type.Object({
			camera: cameraParam,
			row: Type.Integer({ minimum: 0 }),
			col: Type.Integer({ minimum: 0 }),
			target_name: Type.Optional(Type.String({ description: "Default 'target'" })),
			step: stepParam,
			window_radius: Type.Optional(
				Type.Integer({ minimum: 0, description: "Depth median window radius (default 2)" }),
			),
		}),
		async ({ camera = "d455", row, col, target_name = "target", step, window_radius = 2 }) => {
			camera = String(camera).trim();
			if (!camera) throw new Error("camera must be a non-empty string");
			const s = freshStep(step);
			const cfg = projectionView(s, camera);
			const depth = loadDepth(s, camera);
			if (!depth)
				throw new Error(
					`${cfg.display_name} depth artifact is missing. Restart the env server with the camera and depth enabled, then call view_env_state again.`,
				);
			const [r, c] = [Number(row), Number(col)];
			if (!(r >= 0 && r < depth.height && c >= 0 && c < depth.width))
				throw new Error(`pixel row/col [${r}, ${c}] out of depth bounds [${depth.height}, ${depth.width}]`);
			const radius = Math.max(0, Number(window_radius));
			const patch: number[] = [];
			for (let y = Math.max(0, r - radius); y < Math.min(depth.height, r + radius + 1); y++)
				for (let x = Math.max(0, c - radius); x < Math.min(depth.width, c + radius + 1); x++) {
					const z = depth.data[y * depth.width + x];
					if (Number.isFinite(z) && z > 0) patch.push(z);
				}
			if (!patch.length) throw new Error(`no valid depth near pixel row=${r} col=${c} radius=${radius}`);
			const z = median(patch);
			const { fx, fy, cx, cy } = intrinsics(s, camera, cfg.raw_key);
			const pointCamera = [((c - cx) * z) / fx, ((r - cy) * z) / fy, z];
			const cal = calibrationFor(cfg.calibration_key);
			const point = apply(cal.T_right_camera, pointCamera);
			const v = validity(cal.localization_validity, z, point);
			const result: Json = {
				ok: v.ok,
				selection_valid: v.ok,
				target_name: String(target_name).trim() || "target",
				camera,
				pixel: [r, c],
				coordinate_frame: "right_base",
				step: s.blob.step_idx,
				depth_m: round(z),
				depth_window_radius: radius,
				valid_depth_pixels_in_window: patch.length,
				point_camera_xyz: roundAll(pointCamera),
				point_xyz: roundAll(point),
				camera_extrinsic_frame: "right_base",
				coordinate_contract:
					"All returned points and deltas are expressed in the shared right_base world frame. Use the same delta convention for both left and right rule-based arm tools.",
				source_artifact: join(s.dir, `${camera}_depth.f32`),
				selection_contract:
					"The selected RGB pixel must lie well inside visible material of the named target object. Never select image-space air/background above the object. Compute robot z approach offsets only for explicit grasp/approach poses after projecting the object surface into right_base. For placement staging, use projected x/y only and keep the carried-object TCP z unchanged by default.",
				validity_contract: v.contract,
			};
			if (v.reasons.length) {
				result.error = `Rejected localization point: ${v.reasons.join("; ")}. Select a new pixel well inside the visible target surface.`;
				result.rejection_reasons = v.reasons;
			}
			Object.assign(result, tcpDeltas(s.blob.state, point));
			try {
				const n = String(nextIndex(s, `${camera}_back_project`, ".json")).padStart(2, "0");
				const img = loadRgb(s, camera);
				const png = encodePng(mark(img, r, c, v.ok ? [0, 255, 0] : [255, 0, 0]), img.width, img.height);
				const annotated = join(s.dir, `${camera}_back_project_${n}_annotated.png`);
				const report = join(s.dir, `${camera}_back_project_${n}.json`);
				writeFileSync(annotated, png);
				writeFileSync(
					report,
					JSON.stringify({
						ok: v.ok,
						snapshot_step: s.blob.step_idx,
						annotated_image: annotated,
						calibration_source: setup?.calibration_source,
						calibration_key: cfg.calibration_key,
						[`T_right_base_${camera}_camera`]: cal.T_right_camera,
						label: `r${r},c${c} cam=${short(result.point_camera_xyz)} rb=${short(result.point_xyz)}`,
						projection: result,
					}),
				);
				s.blob.artifacts.push(`${camera}_back_project_${n}_annotated.png`, `${camera}_back_project_${n}.json`);
				result.diagnostic_artifacts = { annotated_image: annotated, report_json: report };
				result._pngs = [png];
				result.image_block_order = [`${camera}_selection_diagnostic`];
				result.image_delivery = `annotated_${camera}_selection_returned_for_verification`;
			} catch (err) {
				result.diagnostic_error = message(err);
			}
			return result;
		},
	);

	/** Median right_base point of the valid, in-volume mask pixels. */
	function maskToWorld(s: Step, camera: string, mask: Uint8Array, depth: Grid, minValid: number): Json {
		const cfg = projectionView(s, camera);
		const rows: number[] = [];
		const cols: number[] = [];
		const zs: number[] = [];
		let pixels = 0;
		for (let i = 0; i < mask.length; i++) {
			if (!mask[i]) continue;
			pixels++;
			const z = depth.data[i];
			if (!(Number.isFinite(z) && z > 0)) continue;
			rows.push(Math.floor(i / depth.width));
			cols.push(i % depth.width);
			zs.push(z);
		}
		const result: Json = {
			mask_pixels: pixels,
			valid_depth_pixels: zs.length,
			valid_localization_pixels: 0,
			mask_resized_to_depth_shape: false,
		};
		if (!pixels) return { ...result, point_xyz: null, world_error: "empty mask" };
		if (zs.length < minValid)
			return { ...result, point_xyz: null, world_error: `too few valid ${camera} depth pixels (${zs.length})` };
		const { fx, fy, cx, cy } = intrinsics(s, camera, cfg.raw_key);
		const cal = calibrationFor(cfg.calibration_key);
		const kept: { r: number; c: number; z: number; pc: number[]; pr: number[] }[] = [];
		let contract: Json = {};
		for (let i = 0; i < zs.length; i++) {
			const pc = [((cols[i] - cx) * zs[i]) / fx, ((rows[i] - cy) * zs[i]) / fy, zs[i]];
			const pr = apply(cal.T_right_camera, pc);
			const v = validity(cal.localization_validity, zs[i], pr);
			contract = v.contract;
			if (v.ok && pr.every(Number.isFinite)) kept.push({ r: rows[i], c: cols[i], z: zs[i], pc, pr });
		}
		result.validity_contract = contract;
		result.valid_localization_pixels = kept.length;
		if (kept.length < minValid)
			return {
				...result,
				point_xyz: null,
				world_error: `too few mask pixels remain inside configured ${camera} localization volume (${kept.length})`,
			};
		const point = [0, 1, 2].map((k) => median(kept.map((p) => p.pr[k])));
		const pointCamera = [0, 1, 2].map((k) => median(kept.map((p) => p.pc[k])));
		const z = median(kept.map((p) => p.z));
		const v = validity(cal.localization_validity, z, point);
		Object.assign(result, {
			selection_valid: v.ok,
			centroid_pixel: [rint(median(kept.map((p) => p.r))), rint(median(kept.map((p) => p.c)))],
			depth_m: round(z),
			point_camera_xyz: roundAll(pointCamera),
			point_xyz: v.ok ? roundAll(point) : null,
			raw_median_point_xyz: roundAll(point),
			camera_extrinsic_frame: "right_base",
			calibration_source: setup?.calibration_source,
			calibration_key: cfg.calibration_key,
		});
		if (v.ok) Object.assign(result, tcpDeltas(s.blob.state, point));
		if (v.reasons.length) {
			result.rejection_reasons = v.reasons;
			result.world_error = `Rejected SAM3 mask localization: ${v.reasons.join("; ")}`;
		}
		return result;
	}

	tool(
		"segment",
		"Use SAM3 on a registered RGB image with either a text prompt or one positive [row, col] point, return a mask overlay for verification, and estimate the mask median point in shared right-base coordinates.",
		Type.Object({
			camera: cameraParam,
			prompt: Type.Optional(
				Type.String({
					description:
						"Text prompt for SAM3. Prefer short object/relation phrases; for the clean-desk box use 'white interior of the black cardboard box' or 'cardboard box'. Avoid over-specific surface words such as 'floor' when grounding is weak. Provide exactly one of prompt or point.",
				}),
			),
			point: Type.Optional(
				Type.Array(Type.Integer(), {
					minItems: 2,
					maxItems: 2,
					description:
						"Positive SAM3 point in camera image coordinates [row, col]. Provide exactly one of prompt or point.",
				}),
			),
			target_name: Type.Optional(Type.String({ description: "Default 'target'" })),
			step: stepParam,
			min_score: Type.Optional(Type.Number({ minimum: 0, maximum: 1, description: "Default 0.2" })),
			min_valid_depth_pixels: Type.Optional(Type.Integer({ minimum: 1, description: "Default 25" })),
		}),
		async ({
			camera = "d455",
			prompt = "",
			point,
			target_name = "target",
			step,
			min_score = 0.2,
			min_valid_depth_pixels = 25,
		}) => {
			camera = String(camera).trim();
			if (!camera) throw new Error("camera must be a non-empty string");
			const fallback = `Use manual ${camera} image inspection and back_project.`;
			if (!sam3)
				return {
					ok: false,
					found: false,
					error: "SAM3 client is not configured. Start pi with --robot-sam3 (see serve.sh).",
					fallback,
				};
			const text = String(prompt).trim();
			if (Boolean(text) === (point !== undefined))
				return { ok: false, found: false, error: "segment needs exactly one of prompt or point" };
			if (point !== undefined && (!Array.isArray(point) || point.length !== 2))
				return { ok: false, found: false, error: "point must be [row, col]" };
			let s: Step;
			let png: Buffer;
			let depth: Grid | undefined;
			try {
				s = freshStep(step);
				if (!s.views.includes(camera)) throw new Error(`${camera} image artifact is missing`);
				depth = loadDepth(s, camera);
				if (!depth) throw new Error(`${camera} depth artifact is missing`);
				if (!(min_score >= 0 && min_score <= 1)) throw new Error("min_score must be between 0 and 1");
				png = readFileSync(join(s.dir, `${camera}.png`));
			} catch (err) {
				return { ok: false, found: false, error: message(err) };
			}
			let res: Json;
			let mask: Uint8Array | undefined;
			try {
				res = await sam3.call<Json>(
					"sam3.segment",
					{
						image_base64: png.toString("base64"),
						min_score,
						...(text ? { text_prompt: text } : { point: (point ?? []).map(Number) }),
					},
					120_000,
				);
				if (typeof res?.found !== "boolean")
					throw new Error(`invalid SAM3 segment response: ${JSON.stringify(res)}`);
				if (res.found) {
					const shape = res.mask_shape;
					if (typeof res.mask_png_base64 !== "string" || !res.mask_png_base64)
						throw new Error("SAM3 response marked found but omitted mask_png_base64");
					if (!Array.isArray(shape) || shape.length !== 2) throw new Error(`invalid SAM3 mask_shape: ${shape}`);
					const decoded = decodePngChannel(Buffer.from(res.mask_png_base64, "base64"));
					if (decoded.height !== shape[0] || decoded.width !== shape[1])
						throw new Error(
							`SAM3 mask shape mismatch: response=${shape}, decoded=${[decoded.height, decoded.width]}`,
						);
					mask = decoded.data.map((v) => (v > 0 ? 1 : 0));
					res.mask_shape = shape;
				}
			} catch (err) {
				return { ok: false, found: false, error: `segmentation service call failed: ${message(err)}`, fallback };
			}
			const n = String(nextIndex(s, `${camera}_segment`, ".json")).padStart(2, "0");
			let localization: Json;
			let overlayPng: Buffer | undefined;
			if (res.found && mask) {
				try {
					if (res.mask_shape[0] !== depth.height || res.mask_shape[1] !== depth.width)
						localization = {
							point_xyz: null,
							world_error: `mask/depth shape mismatch: mask=${res.mask_shape}, depth=${[depth.height, depth.width]}`,
							mask_pixels: mask.reduce((a, b) => a + b, 0),
							valid_depth_pixels: 0,
							valid_localization_pixels: 0,
						};
					else localization = maskToWorld(s, camera, mask, depth, Math.max(1, Number(min_valid_depth_pixels)));
					const img = loadRgb(s, camera);
					if (img.width === depth.width && img.height === depth.height) {
						const rgb = Buffer.from(img.rgb);
						for (let i = 0; i < mask.length; i++) {
							if (!mask[i]) continue;
							rgb[i * 3] = Math.floor(0.55 * rgb[i * 3] + 0.45 * 255);
							rgb[i * 3 + 1] = Math.floor(0.55 * rgb[i * 3 + 1]);
							rgb[i * 3 + 2] = Math.floor(0.55 * rgb[i * 3 + 2]);
						}
						const [cr, cc] = localization.centroid_pixel ?? [];
						const marked =
							cr === undefined
								? rgb
								: mark({ ...img, rgb }, cr, cc, localization.point_xyz ? [0, 255, 0] : [255, 0, 0]);
						overlayPng = encodePng(marked, img.width, img.height);
						writeFileSync(join(s.dir, `${camera}_segment_overlay_${n}.png`), overlayPng);
						s.blob.artifacts.push(`${camera}_segment_overlay_${n}.png`);
					}
				} catch (err) {
					localization = { point_xyz: null, world_error: message(err) };
				}
			} else localization = { point_xyz: null, world_error: res.reason || "SAM3 found no mask" };
			const blob: Json = {
				ok: Boolean(res.found && localization.point_xyz != null),
				found: Boolean(res.found),
				mode: text ? "text" : "point",
				target_name: String(target_name).trim() || "target",
				camera,
				source_step: s.blob.step_idx,
				segment_index: Number(n),
				min_score,
				score: typeof res.score === "number" ? round(res.score, 3) : null,
				box: res.box ?? null,
				mask_shape: res.mask_shape ?? null,
				coordinate_frame: "right_base",
				coordinate_contract: `point_xyz is the median valid ${camera}-mask point expressed in the shared right_base world frame.`,
				selection_contract: `Inspect the returned ${camera} mask overlay. The highlighted mask and median marker must cover the intended visible material, not the rim, wire basket, wall, table, or background. Retry with a point prompt or a more specific text prompt if it is wrong.`,
				...(text ? { prompt: text } : { point }),
				...(res.found ? {} : { error: res.reason || "SAM3 found no mask" }),
				...localization,
			};
			const segmentName = `${camera}_segment_${n}.json`;
			writeFileSync(join(s.dir, segmentName), JSON.stringify(blob));
			s.blob.artifacts.push(segmentName);
			const keys = [
				"ok",
				"found",
				"target_name",
				"mode",
				"score",
				"box",
				"mask_shape",
				"coordinate_frame",
				"point_xyz",
				"world_error",
				"centroid_pixel",
				"mask_pixels",
				"valid_depth_pixels",
				"valid_localization_pixels",
				"selection_valid",
				"rejection_reasons",
				"left_tcp_xyz",
				"right_tcp_xyz",
				"delta_left_tcp_to_point_xyz",
				"delta_right_tcp_to_point_xyz",
				"tcp_delta_coordinate_frame",
				"tcp_delta_contract",
				"selection_contract",
			];
			const result: Json = { step: s.blob.step_idx, camera, segment_artifact: join(s.dir, segmentName) };
			for (const k of keys) result[k] = blob[k] ?? null;
			if (overlayPng) {
				result.overlay_artifact = join(s.dir, `${camera}_segment_overlay_${n}.png`);
				result._pngs = [overlayPng];
				result.image_block_order = [`${camera}_segment_overlay`];
				result.image_delivery = `sam3_${camera}_segment_overlay_returned_for_verification`;
			}
			if (blob.error) {
				result.error = blob.error;
				result.fallback = fallback;
			}
			return result;
		},
	);

	// ---- analytic primitives

	const armName = (v: unknown) => {
		const a = String(v).trim().toLowerCase();
		if (a !== "left" && a !== "right") throw new Error("arm must be exactly 'left' or 'right'");
		return a;
	};
	function vec3(v: unknown, name: string): NdArray {
		const a = vec(v);
		if (a.length !== 3) throw new Error(`${name} must contain exactly 3 values, got (${a.length},)`);
		if (!a.every(Number.isFinite)) throw new Error(`${name} must contain only finite values`);
		return NdArray.f32(a);
	}

	tool(
		"move_delta",
		"Move one Franka TCP by a bounded world-frame xyz delta in meters.",
		Type.Object({ arm, delta_xyz: xyz }),
		async (p, signal) => {
			check(signal);
			const delta = vec3(p.delta_xyz, "delta_xyz");
			checkMove(vec(p.delta_xyz), Number(flag("max-move", "0.1")), setup?.task.constraints);
			checkWorkspace(armName(p.arm), vec(p.delta_xyz));
			return motion("env.move_delta", { arm: armName(p.arm), delta_xyz: delta }, signal);
		},
	);

	tool(
		"rotate_delta",
		"Rotate one Franka TCP by a bounded world-frame rpy delta in radians.",
		Type.Object({ arm, delta_rpy: xyz }),
		async (p, signal) => {
			check(signal);
			return motion("env.rotate_delta", { arm: armName(p.arm), delta_rpy: vec3(p.delta_rpy, "delta_rpy") }, signal);
		},
	);

	tool(
		"open_gripper",
		"Open one Franka gripper and wait for the command to settle.",
		Type.Object({ arm }),
		async (p, signal) => {
			check(signal);
			return motion("env.set_gripper", { arm: armName(p.arm), open: true }, signal);
		},
	);

	tool(
		"close_gripper",
		"Close one Franka gripper and wait for the command to settle.",
		Type.Object({ arm }),
		async (p, signal) => {
			check(signal);
			return motion("env.set_gripper", { arm: armName(p.arm), open: false }, signal);
		},
	);

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

	async function namedSkill(
		skill: string,
		boundary: Boundary,
		prompt: string,
		maxChunks: number,
		signal?: AbortSignal,
	) {
		if (!vla) throw new Error(`${skill} requires --robot-vla`);
		const requested = String(prompt).trim();
		if (!requested) throw new Error("prompt must be non-empty");
		// Task configuration owns policy conditioning; the planner's intent is only recorded.
		const effective = setup?.task.vla_instruction || setup?.task.instruction || "";
		if (!(maxChunks >= 1 && maxChunks <= 20)) throw new Error("max_chunks must be between 1 and 20");

		let state = await robotState();
		const start = state;
		const open = (s: Json, a: string) => Boolean(s[`${a}_arm`]?.gripper_open);
		const z = (s: Json, a: string) => vec(s[`${a}_arm`]?.tcp_pose)[2];
		let prevLeft = open(state, "left");
		let prevRight = open(state, "right");
		let eventZ: number | null = null;
		let eventTime: number | null = null;
		let eventStep: number | null = null;
		let after = 0;
		let chunks = 0;
		let stepsDone = 0;
		let reached = false;
		let terminated = false;
		let truncated = false;
		let count = 0;
		let min: number[] = [];
		let max: number[] = [];
		let sum: number[] = [];
		let firstAction: number[] | undefined;
		let firstState: number[] | undefined;
		let firstStateShape: number[] | undefined;
		const t0 = performance.now();

		rollout: for (let c = 0; c < maxChunks; c++) {
			check(signal);
			const obs = await observation();
			const main = obs.main_images;
			const extras = obs.extra_view_images;
			const states = obs.states;
			if (firstState === undefined) {
				firstStateShape = states instanceof NdArray ? states.shape : [vec(states).length];
				firstState = roundAll(vec(states).slice(0, 20));
			}
			if (!(main instanceof NdArray) || main.shape.length !== 3)
				throw new Error(`expected [H,W,3] image, got shape [${main?.shape}]`);
			if (!(extras instanceof NdArray) || extras.shape.length !== 4)
				throw new Error(`extra_view_images must be [N,H,W,3]; got shape [${extras?.shape}]`);
			if (!(states instanceof NdArray) || states.shape.length !== 1)
				throw new Error(`states must be single-env shape [state_dim]; got [${states?.shape}]`);
			const wire = {
				main_images: u8(main).batched(),
				wrist_images: null,
				extra_view_images: u8(extras).batched(),
				states: f32(states).batched(),
				task_descriptions: [effective],
			};
			const predicted = await vla.call<NdArray>("vla.predict", {}, 120_000, [wire, { mode: "eval" }], signal);
			const actions = new NdArray(predicted.dtype, predicted.shape.slice(1), predicted.data);
			if (actions.shape.length !== 2 || actions.shape[1] !== 20)
				throw new Error(`${skill} expected [chunk, 20] actions, got [${actions.shape}]`);
			const flat = numbers(actions);
			if (!flat.every(Number.isFinite)) throw new Error(`${skill} received non-finite VLA actions`);
			chunks++;
			const rows = Array.from({ length: actions.shape[0] }, (_, i) => flat.slice(i * 20, (i + 1) * 20));
			firstAction ??= roundAll(rows[0]);
			for (const row of rows) {
				min = count ? min.map((v, k) => Math.min(v, row[k])) : [...row];
				max = count ? max.map((v, k) => Math.max(v, row[k])) : [...row];
				sum = count ? sum.map((v, k) => v + row[k]) : [...row];
				count++;
			}

			for (const row of rows) {
				check(signal);
				const result = await call(
					"env.chunk_step",
					{ actions: NdArray.f32(row, [1, 20]), return_all_frames: false },
					300_000,
					signal,
				);
				const next = result.observation;
				remember(Array.isArray(next) ? next.at(-1)?.states : next?.states);
				stepsDone++;
				terminated ||= Boolean(result.terminated);
				truncated ||= Boolean(result.truncated);
				state = await robotState();
				const left = open(state, "left");
				const right = open(state, "right");
				if (boundary === "grasp") {
					if (eventZ === null && prevRight && !right) {
						eventZ = z(state, "right");
						eventStep = stepsDone;
						after = 0;
					} else if (eventZ !== null) {
						after++;
						reached = !right && after >= 2 && z(state, "right") - eventZ >= 0.15;
					}
				} else if (boundary === "handoff") {
					if (eventTime === null && !prevRight && right) {
						eventTime = performance.now();
						eventStep = stepsDone;
						after = 0;
					} else if (eventTime !== null) {
						after++;
						reached = right && after >= 2 && (performance.now() - eventTime) / 1000 >= 1.5;
					}
				} else if (eventZ === null && !prevLeft && left) {
					eventZ = z(state, "left");
					eventStep = stepsDone;
					after = 0;
				} else if (eventZ !== null) {
					after++;
					reached = left && after >= 2 && z(state, "left") - eventZ >= 0.1;
				}
				prevLeft = left;
				prevRight = right;
				if (reached || terminated || truncated) break rollout;
			}
		}

		const stopRule: Json = {
			phase: boundary,
			skill_name: skill,
			step_count: stepsDone,
			event_step: eventStep,
			success_claim: false,
		};
		if (boundary === "grasp")
			Object.assign(stopRule, {
				condition: "right_gripper_closed_then_lifted",
				right_close_step: eventStep,
				right_close_z: eventZ,
				right_current_z: z(state, "right"),
				lift_m: eventZ === null ? null : z(state, "right") - eventZ,
				threshold_m: 0.15,
			});
		else if (boundary === "handoff")
			Object.assign(stopRule, {
				condition: "right_gripper_opened_then_delay",
				right_open_step: eventStep,
				elapsed_after_open_s: eventTime === null ? null : (performance.now() - eventTime) / 1000,
				delay_s: 1.5,
				right_current_open: open(state, "right"),
			});
		else
			Object.assign(stopRule, {
				condition: "left_gripper_opened_then_lifted",
				left_open_step: eventStep,
				left_open_z: eventZ,
				left_current_z: z(state, "left"),
				lift_m: eventZ === null ? null : z(state, "left") - eventZ,
				threshold_m: 0.1,
			});
		const summary: Json = { count, shape: [count, 20], finite: true };
		if (count)
			Object.assign(summary, {
				min: roundAll(min),
				max: roundAll(max),
				mean: roundAll(sum.map((v) => v / count)),
				first_action: firstAction,
				first_policy_state_shape: firstStateShape,
				first_policy_state: firstState,
			});
		return {
			ok: reached && !(terminated || truncated),
			skill_name: skill,
			boundary,
			requested_prompt: requested,
			effective_policy_prompt: effective,
			prompt_overridden: requested !== effective,
			boundary_reached: reached,
			chunks_executed: chunks,
			steps_executed: stepsDone,
			terminated,
			truncated,
			stop_rule: stopRule,
			action_summary: summary,
			elapsed_s: (performance.now() - t0) / 1000,
			vla_start_robot_state: start,
			robot_state: state,
		};
	}

	const skillParams = Type.Object({
		prompt: Type.String({
			description:
				"Planner-facing segment intent. This is recorded in the tool result; the current live clean-desk checkpoint still receives its fixed training instruction during policy inference.",
		}),
		max_chunks: Type.Optional(Type.Integer({ minimum: 1, maximum: 20, description: "Default 20" })),
	});
	for (const [name, boundary, description] of [
		[
			"vla_right_grasp",
			"grasp",
			"Run the learned right-grasp VLA segment. The active task prompt decides which object is currently allowed; this tool only defines the capability boundary: right gripper closes and the right TCP lifts.",
		],
		[
			"vla_handoff",
			"handoff",
			"Run the learned bimanual handoff VLA segment. The capability boundary is right-gripper release followed by the configured settle delay; do not rule-base pre-position either arm for it.",
		],
		[
			"vla_left_place",
			"place",
			"Run the learned left-placement VLA segment. The active task decides the destination; this tool only defines the capability boundary: left gripper opens and the left TCP lifts.",
		],
	] as const)
		tool(name, description, skillParams, (p, signal) =>
			namedSkill(name, boundary, p.prompt, p.max_chunks ?? 20, signal),
		);

	// ---- lifecycle

	async function startRobot(ctx: ExtensionContext) {
		if (!ctx.hasUI)
			throw new Error(
				"dual_franka drives real robots: run pi interactively (or over RPC) so an operator is present",
			);
		if (pi.getFlag("operator") !== true)
			throw new Error("dual_franka needs --operator: success on this robot is the operator's verdict");
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
		return [...TOOLS, "finish"];
	}
}
