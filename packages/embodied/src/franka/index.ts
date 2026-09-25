/**
 * One physical Franka arm for pi.
 *
 *   pi -e packages/embodied/src/franka --task 1 --robot-config my_franka.yaml --robot-vla http://VLA_HOST:PORT
 *
 * Starts the RLinf-backed env server (pi_embodied_services.robots.franka.env_server; Ray must
 * already run on the controller node) or attaches to one with --robot-env. The VLA is
 * attach-only. The server enforces the workspace limits, per-step clips and servo
 * tolerances from its runtime config; task definitions and easy_handeye calibration come from
 * the same services package. A real robot needs an operator: pi must have a UI, the operator confirms
 * the reset motion, and --operator adds the verdict gate (see ../operator.ts). The
 * single-arm robot has no success signal; the result entry records the agent's claim and
 * the operator's verdict, if any. Mutating tools record a state step (robot state, external
 * and wrist RGB-D, camera metadata) under --out and return it with both images; file tools
 * reach only the memory and that directory.
 */

import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";
import { encodePng } from "../png.ts";
import {
	apply,
	attach,
	checkMove,
	defineRobot,
	f32,
	gridOf,
	inv3,
	type Json,
	type Mat,
	mark,
	message,
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
} from "../robot.ts";
import { NdArray, type RpcClient } from "../rpc.ts";

const SYSTEM = readFileSync(new URL("./SYSTEM.md", import.meta.url), "utf8");

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
type Step = { blob: Json; dir: string; meta: Json | null; images: Record<string, string> };

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

function vec3(v: unknown, name: string): NdArray {
	const a = vec(v);
	if (a.length !== 3) throw new Error(`${name} must contain exactly 3 values, got (${a.length},)`);
	if (!a.every(Number.isFinite)) throw new Error(`${name} must contain only finite values`);
	return NdArray.f32(a);
}

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
		description: "Robot YAML (default: services/pi_embodied_services/robots/franka/config/example.yaml)",
	});
	pi.registerFlag("robot-env", {
		type: "string",
		description: "Attach to a running Franka env server instead of starting one",
	});
	pi.registerFlag("robot-vla", {
		type: "string",
		description: "External Franka Pi0.5 VLA server (attach-only; enables vla_grasp)",
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
	pi.registerFlag("max-rotate", {
		type: "string",
		default: "0.5",
		description: "Largest rotate_delta per call, rad (norm of delta_rpy)",
	});

	let env: RpcClient | undefined;
	let vla: RpcClient | undefined;
	let setup: Setup | undefined;
	let out = "";
	let lastStates: unknown;
	const steps: Step[] = [];
	const task = () => robot.task.task;
	const robot = defineRobot(pi, {
		name: "franka",
		task: ["task"],
		keepImages: 4,
		// Franka memory is read-only and the prompt names none; the guard also opens the step artifacts.
		memory: {
			cell: () => ({ tag: `franka_t${task()}`, reference: "" }),
			primitives: [],
			readable: () => [out],
		},
		operator: { step: () => steps.length },
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
		result: () => ({ task: Number(task()), task_name: setup?.task.name ?? null, steps: steps.length, out }),
		status: () => ({ step: steps.length - 1 }),
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
	const check = (signal?: AbortSignal) => {
		op.check();
		if (signal?.aborted) throw new Error("tool operation interrupted");
	};

	// ---- env client (the server returns camera frames, the client caches states)

	async function observation(): Promise<Json> {
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
		saveImage("wrist", obs.main_images);
		saveImage("camera", first(obs.extra_view_images, 4));
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

	function getStep(step?: number | null): Step {
		const i = step === undefined || step === null ? steps.length - 1 : step < 0 ? steps.length + step : step;
		const s = steps[i];
		if (!s) throw new Error(`step ${step} is not recorded (have 0..${steps.length - 1})`);
		return s;
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
		robot.tool(name, description, parameters, outcome);

		async function outcome(params: Static<P>, signal: AbortSignal | undefined, ctx: ExtensionContext) {
			if (!env || !steps.length) return toolResult({ error: "robot not initialized; see the session start error" });
			const started = performance.now();
			let result: Json;
			let failed = false;
			try {
				result = await run(params, signal, ctx);
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
	}

	const stepParam = Type.Optional(Type.Integer({ description: "State step (default -1 = latest)" }));
	const pixel = (description: string) => Type.Optional(Type.Integer({ minimum: 0, description }));

	tool(
		"view_env_state",
		"Read a Franka state snapshot and its synchronized RGB images.",
		Type.Object({ step: stepParam }),
		async ({ step = -1 }) => {
			const { output, pngs } = view(getStep(step));
			return { ...output, _pngs: pngs };
		},
		false,
	);

	tool(
		"view_camera_meta",
		"Read camera intrinsics, crop, depth, and calibration metadata.",
		Type.Object({ step: stepParam }),
		async ({ step = -1 }) => {
			const s = getStep(step);
			if (!s.meta) return { error: "camera metadata is unavailable", step };
			return { step: s.blob.step_idx, camera_meta: s.meta };
		},
		false,
	);

	function calibration() {
		if (!setup?.calibration) throw new Error(setup?.calibration_error ?? "no hand-eye calibration loaded");
		return setup.calibration;
	}

	tool(
		"view_perception_setup",
		"Read calibrated camera geometry and projection conventions.",
		Type.Object({ step: stepParam }),
		async ({ step = -1 }) => {
			const s = getStep(step);
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

	tool(
		"back_project",
		"Back-project one wrist or external-camera pixel into Franka base coordinates.",
		Type.Object({
			row: Type.Integer({ minimum: 0 }),
			col: Type.Integer({ minimum: 0 }),
			step: stepParam,
			camera: Type.Optional(
				Type.Union([Type.Literal("wrist"), Type.Literal("third_person")], { description: "Default wrist" }),
			),
			debug: Type.Optional(Type.Boolean({ description: "Default false" })),
		}),
		async ({ row, col, step, camera = "wrist", debug = false }) => {
			const s = getStep(step);
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
			tcp_pose_source: "RLinf raw_base_state.tcp_pose",
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
			const s = getStep(step);
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
				tcp_pose_source: "RLinf raw_base_state.tcp_pose",
			};
		},
		false,
	);

	const xyz = Type.Array(Type.Number(), { minItems: 3, maxItems: 3 });

	tool(
		"move_delta",
		"Move the Franka TCP by a bounded base-frame xyz delta in meters.",
		Type.Object({ delta_xyz: xyz }),
		async ({ delta_xyz }, signal) => {
			check(signal);
			const delta = vec3(delta_xyz, "delta_xyz");
			checkMove(vec(delta_xyz), Number(flag("max-move", "0.1")), setup?.task.constraints);
			return motion("env.move_delta", { delta_xyz: delta }, signal);
		},
	);

	tool(
		"rotate_delta",
		"Rotate the Franka TCP by a bounded base-frame rpy delta in radians.",
		Type.Object({ delta_rpy: xyz }),
		async ({ delta_rpy }, signal) => {
			check(signal);
			const delta = vec3(delta_rpy, "delta_rpy");
			const limit = Number(flag("max-rotate", "0.5"));
			const norm = Math.hypot(...numbers(delta));
			if (!(norm <= limit))
				throw new Error(
					`delta_rpy rotates ${round(norm, 4)} rad; the limit is ${limit} rad per call. Split the rotation into smaller calls.`,
				);
			return motion("env.rotate_delta", { delta_rpy: delta }, signal);
		},
	);

	tool(
		"open_gripper",
		"Open the Franka gripper and wait for the command to settle.",
		Type.Object({}),
		async (_p, signal) => {
			check(signal);
			return motion("env.set_gripper", { open: true }, signal);
		},
	);

	tool(
		"close_gripper",
		"Close the Franka gripper and wait for the command to settle.",
		Type.Object({}),
		async (_p, signal) => {
			check(signal);
			return motion("env.set_gripper", { open: false }, signal);
		},
	);

	tool(
		"vla_grasp",
		"Run bounded real-world VLA action chunks for a local grasp attempt.",
		Type.Object({
			prompt: Type.String(),
			max_chunks: Type.Optional(Type.Integer({ minimum: 1, maximum: 20, description: "Default 4" })),
		}),
		async ({ prompt, max_chunks = 4 }, signal) => {
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

	// ---- lifecycle

	async function startRobot(ctx: ExtensionContext) {
		if (!ctx.hasUI)
			throw new Error("franka drives a real robot: run pi interactively (or over RPC) so an operator is present");
		const r: Services = { root: flag("services"), python: flag("python", "python") };
		const configFlag = pi.getFlag("robot-config");
		const config = typeof configFlag === "string" && configFlag ? resolve(ctx.cwd, configFlag) : "";
		setup = await servicesJson<Setup>(r, SETUP_PY, [task(), config]);
		const stamp = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
		out = resolve(ctx.cwd, flag("out") || join(tmpdir(), "pi-embodied", `franka_t${task()}_${stamp}`));
		mkdirSync(out, { recursive: true });
		const endpoint = flag("robot-env");
		const [envRpc, vlaRpc] = await Promise.all([
			endpoint
				? attach(endpoint)
				: robot.serve({
						python: r.python,
						args: [
							...["-m", "pi_embodied_services.robots.franka.env_server"],
							...["--task-description", setup.task.instruction, ...(config ? ["--robot-config", config] : [])],
						],
						cwd: r.root,
						env: servicesEnv(r),
						log: () => join(out, "franka_env_server.log"),
					}),
			flag("robot-vla") ? attach(flag("robot-vla")) : undefined,
		]);
		await envRpc.call("env.get_env_meta", {}, 30_000);
		const go = await ctx.ui.confirm(
			"Reset the Franka arm?",
			"The arm will move to its configured reset pose. Clear the workspace and keep the emergency stop in reach.",
		);
		if (!go) throw new Error("operator declined the reset; the Franka tools stay disabled");
		const reset = await envRpc.call<Json>("env.reset", {}, 180_000);
		env = envRpc;
		vla = vlaRpc;
		remember(reset.states);
		await dumpState(null, null, null);
		ctx.ui.notify(`Franka ready: task ${task()} (${setup.task.name}); steps under ${out}`, "info");
		return TOOLS;
	}
}
