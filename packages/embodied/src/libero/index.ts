/**
 * LIBERO robot for pi.
 *
 *   pi -e packages/embodied/src/libero --suite libero_10 --task 2 --seed 0
 *
 * Starts one LIBERO env server per session and attaches to running Pi0.5 VLA and
 * SAM3 servers (see serve.sh). Tools are the LIBERO primitives. Every motion
 * tool returns the new state with agentview and wrist images; success is LIBERO's
 * own `terminated` flag, recorded in the session's `robot_result` entry.
 */

import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";
import { decodePngChannel, encodePng } from "../png.ts";
import { defineRobot, mark, median, SERVICES } from "../robot.ts";
import { NdArray, RpcClient } from "../rpc.ts";
import type { Move } from "../units/index.ts";
import { vlaSeeds } from "../vla-seed.ts";
import { liberoFlash } from "./flash.ts";

const read = (name: string) => readFileSync(new URL(name, import.meta.url), "utf8");
const SYSTEM = read("./SYSTEM.md");
const MEMORY = { hf: read("./memory-hf.md"), local: read("./memory-local.md") };
const EXPLORE = read("./explore.md");
const DISTIL = read("./distil.md");
/** The prompt's single-episode lines, which exploration replaces rather than contradicts. */
const REWRITE: [RegExp, string][] = [
	[
		/^This is a single episode\..*$/m,
		"This is an exploration run: `reset` starts a fresh episode (see Exploration). The task is done when a tool result shows `terminated: true`; that flag is the only success signal.",
	],
	[
		/^11\. .*$/m,
		"11. Keep reasoning to one or two sentences before each tool call. When an episode is unrecoverable, close it out and `reset`; when `terminated` is true, run DISTIL, then `finish` (see Exploration).",
	],
];
const PRIMITIVES = [
	"move_to",
	"pi0_pick",
	"pi0_doubled",
	"release",
	"set_gripper",
	"rotate_wrist",
	"rotate_pitch",
	"move_pose",
];
const TOOLS = [...PRIMITIVES, "view_env_state", "view_camera_meta", "segment", "back_project", "finish"];
const CAMERAS = { agentview: "agentview", wrist: "robot0_eye_in_hand" } as const;
/** Units mode (../units): how the two images look, and which way each unit moves in them. */
const VIEWS = `Each result shows the agentview, then the wrist view (verified in LIBERO: MV_FWD is world +x, MV_LEFT is -y).
- Agentview (first image) faces the robot, whose base is at the image top: MV_LEFT / MV_RIGHT move the gripper toward the image left / right, MV_FWD toward the image bottom (toward the camera), MV_BACK toward the image top.
- Wrist view (second image) looks straight down from the gripper; the two fingers are at the image bottom corners and the grasp point is between them, at the horizontal center just above the fingers. It is turned half around relative to the agentview: MV_FWD moves toward the wrist image TOP, MV_BACK toward its bottom, MV_LEFT toward its RIGHT and MV_RIGHT toward its LEFT. So a target above the grasp point in the wrist image needs MV_FWD, one to its right needs MV_LEFT.`;
type Camera = keyof typeof CAMERAS;
type Obs = { main_images: NdArray; wrist_images?: NdArray | null; states: NdArray };
type StepReturn = [Obs, unknown, boolean | NdArray, boolean | NdArray, unknown];
type ChunkReturn = [Obs[], NdArray, NdArray, NdArray, unknown];
type CameraMeta = { intrinsic_K: number[][]; extrinsic_cam2world: number[][]; depth_near?: number; depth_far?: number };
type WorldMap = { envStep: number; size: number; rgb: Buffer; xyz: Float32Array };

const done = (v: boolean | NdArray) => (v instanceof NdArray ? v.toArray().some(Boolean) : Boolean(v));
const round = (v: number, d = 4) => Number(v.toFixed(d));
const wrap = (a: number) => ((((a + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
const clip = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Rows of an HxWxC byte image in reverse order (LIBERO renders upside down). */
function flipRows(data: Buffer, height: number, rowBytes: number): Buffer {
	const out = Buffer.alloc(data.length);
	for (let y = 0; y < height; y++) data.copy(out, (height - 1 - y) * rowBytes, y * rowBytes, (y + 1) * rowBytes);
	return out;
}

/** Rotation matrix of an xyzw quaternion. */
function rotation([x, y, z, w]: number[]): number[][] {
	return [
		[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
		[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
		[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
	];
}
const yawOf = (q: number[]) => {
	const r = rotation(q);
	return Math.atan2(r[1][0], r[0][0]);
};
const pitchOf = (q: number[]) => {
	const r = rotation(q);
	return Math.atan2(r[1][2], -r[2][2]);
};

export default function libero(pi: ExtensionAPI) {
	const flag = (name: string, fallback: string) => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("suite", { type: "string", default: "libero_10", description: "LIBERO suite, e.g. libero_10" });
	pi.registerFlag("task", { type: "string", default: "0", description: "Task index within the suite" });
	pi.registerFlag("seed", { type: "string", default: "0", description: "Initial-state seed" });
	pi.registerFlag("libero-type", { type: "string", default: "pro", description: "standard | pro | plus" });
	pi.registerFlag("vla", { type: "string", default: "http://127.0.0.1:18200", description: "Pi0.5 VLA server" });
	const seeds = vlaSeeds(pi, () => ["libero", robot.task]);
	pi.registerFlag("sam3", { type: "string", default: "http://127.0.0.1:18300", description: "SAM3 server" });
	pi.registerFlag("env", { type: "string", description: "Attach to a running env server instead of starting one" });
	pi.registerFlag("services", {
		type: "string",
		default: process.env.PI_EMBODIED_SERVICES ?? SERVICES,
		description: "pi-embodied services dir",
	});
	pi.registerFlag("python", {
		type: "string",
		default: process.env.PI_EMBODIED_PYTHON ?? "python",
		description: "Python for the env server",
	});

	let env: RpcClient;
	let vla: RpcClient;
	let sam3: RpcClient;
	let obs: Obs;
	let terminated = false;
	let truncated = false;
	let envStep = 0;
	let language = "";
	/** The gripper command units mode holds between units: -1 open, +1 closed. */
	let grip = -1;
	/** The table (or floor) height in front of the robot, for units' proprioception and variable_step. */
	let tableZ: number | undefined;
	const worldMaps = new Map<string, WorldMap>();

	const tag = () => `${robot.task.suite.replace(/^libero_/, "")}_t${robot.task.task}_s${robot.task.seed}`;
	const robot = defineRobot(pi, {
		name: "libero",
		task: ["suite", "task", "seed"],
		keepImages: 4,
		memory: {
			cell: () => ({ tag: tag(), reference: tag().replace(/_s\d+$/, "_s0") }),
			primitives: PRIMITIVES,
		},
		video: true,
		flywheel: true,
		flash: liberoFlash(pi, () => ({ suite: robot.task.suite, task: robot.task.task })),
		operator: {
			step: () => envStep,
			// In simulation the operator's scene restore is the env's own reset to the episode's initial state.
			reset: async () => {
				await resetEpisode();
				fly.reset(obs, flyMeta());
				return { step: envStep, terminated, truncated };
			},
		},
		explore: {
			// The exploration `reset` tool is not a robot.tool, so robot.signal is unset here; use its own signal.
			reset: async (result, _ctx, signal) => {
				await resetEpisode(signal);
				fly.reset(obs, flyMeta());
				return observe(result);
			},
			prompt: () => EXPLORE,
			distil: DISTIL,
			rewrite: REWRITE,
		},
		start: startEpisode,
		prompt: () => {
			const system = SYSTEM.replaceAll("{{task_language}}", language);
			// Exploration appends its own memory instructions.
			if (pi.getFlag("explore") === true) return system;
			return `${system}\n\n${mem.render(MEMORY[mem.profile], { task: robot.task.task })}`;
		},
		result: () => ({
			suite: robot.task.suite,
			task: Number(robot.task.task),
			seed: Number(robot.task.seed),
			terminated,
			truncated,
			env_steps: envStep,
		}),
		status: () => ({ language, step: envStep, solved: terminated }),
		units: {
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
			apply: (move) => unitStep(move),
			state: async () => ({
				eef_xyz: eef().map((v) => round(v)),
				gripper_width: round(gripper()),
				...(tableZ === undefined ? {} : { table_z: tableZ }),
			}),
			instruction: () => language,
			views: VIEWS,
			// Measured: fingers closed on nothing read <= 0.003 (sum of both finger joints); a held can reads ~0.06.
			emptyWidthM: 0.004,
			point: { cameras: ["agentview", "wrist"], locate: (camera, points) => locate(camera as Camera, points) },
		},
		finish: {
			description:
				"End the episode after checking the latest state. Success is LIBERO's terminated flag, not this call.",
			parameters: Type.Object({
				status: Type.Union([Type.Literal("success"), Type.Literal("failure")]),
				summary: Type.String(),
			}),
			result: (params) => ({
				content: [{ type: "text", text: `Episode finished (terminated=${terminated}).` }],
				details: params,
			}),
		},
	});
	const { video, op } = robot;
	const mem = robot.mem!;
	const fly = robot.fly!;

	/** Every robot RPC carries the running tool's abort signal, so an abort stops motion between calls. */
	const call = <T = unknown>(
		client: RpcClient,
		method: string,
		kwargs: Record<string, unknown> = {},
		timeoutMs = 120_000,
		args: unknown[] = [],
	) => client.call<T>(method, kwargs, timeoutMs, args, robot.signal);

	const states = () => obs.states.toArray();
	const eef = () => states().slice(0, 3);
	const gripper = () => Math.abs(states()[6]) + Math.abs(states()[7]);
	const quat = async () => (await call<Record<string, NdArray>>(env, "env.raw_obs")).robot0_eef_quat.toArray();

	function absorb(ret: StepReturn, steps: number) {
		obs = ret[0];
		terminated ||= done(ret[2]);
		truncated ||= done(ret[3]);
		envStep += steps;
	}

	const scalar = (v: unknown) => (v instanceof NdArray ? v.toArray()[0] : Number(v));

	/** Every env transition goes to the episode video and the flywheel recorder. */
	function record(action: number[], o: Obs, reward: number, term: boolean, trunc: boolean, vlaId = -1, index = -1) {
		video.frame(o.main_images);
		fly.transition(action, o, reward, term, trunc, vlaId, index);
	}

	async function step(action: number[]) {
		op.check();
		const ret = await call<StepReturn>(env, "env.step", {}, 60_000, [NdArray.f32(action)]);
		record(action, ret[0], scalar(ret[1]), done(ret[2]), done(ret[3]));
		absorb(ret, 1);
	}

	/** One Pi0.5 forward pass with `prompt` as the instruction, executed as one action chunk; returns its seed. */
	async function vlaChunk(prompt: string) {
		const wire = {
			main_images: obs.main_images.batched(),
			wrist_images: obs.wrist_images ? obs.wrist_images.batched() : null,
			extra_view_images: null,
			states: NdArray.f32(states(), [1, states().length]),
			task_descriptions: [prompt],
		};
		op.check();
		const seed = seeds.next();
		const options = seed === undefined ? { mode: "eval" } : { mode: "eval", seed };
		const actions = await call<NdArray>(vla, "vla.predict", {}, 120_000, [wire, options]);
		const chunk = new NdArray(actions.dtype, actions.shape.slice(1), actions.data);
		const vlaId = fly.proposal(prompt, chunk);
		op.check();
		const [frames, rew, term, trunc, info] = await call<ChunkReturn>(
			env,
			"env.chunk_step",
			{ return_all_frames: true },
			120_000,
			[chunk],
		);
		const [a, r, te, tr] = [chunk.toArray(), rew.toArray(), term.toArray(), trunc.toArray()];
		const width = chunk.shape[1];
		frames.forEach((o, i) => {
			record(a.slice(i * width, (i + 1) * width), o, r[i], Boolean(te[i]), Boolean(tr[i]), vlaId, i);
		});
		absorb([frames[frames.length - 1], rew, term, trunc, info], chunk.shape[0]);
		return seed;
	}

	const flyMeta = () => ({
		suite: robot.task.suite,
		task_id: Number(robot.task.task),
		seed: Number(robot.task.seed),
		task_language: language,
	});

	/** Restore the episode's initial scene (session start, exploration `reset`); `signal` aborts the env reset. */
	async function resetEpisode(signal = robot.signal) {
		worldMaps.clear();
		terminated = truncated = false;
		grip = -1;
		envStep = 0;
		seeds.reset();
		[obs] = await env.call<[Obs, unknown]>("env.reset", {}, 300_000, [], signal);
		tableZ = await surfaceZ().catch(() => undefined);
	}

	async function render(camera: Camera, size: number, depth: boolean) {
		// The env returns [rgb, depth] with depth, and the bare rgb array without it.
		const out = await call<NdArray | [NdArray, NdArray]>(env, "env.render_camera", {
			camera_name: CAMERAS[camera],
			height: size,
			width: size,
			depth,
		});
		const [rgb, d] = out instanceof NdArray ? [out, null] : out;
		return { rgb: flipRows(rgb.data, size, size * 3), depth: d };
	}

	/** Per-pixel world xyz for the current step, from metric depth + calibration (the world map). */
	async function worldMap(camera: Camera, size: number): Promise<WorldMap> {
		const key = `${camera}:${size}`;
		const cached = worldMaps.get(key);
		if (cached?.envStep === envStep) return cached;
		const { rgb, depth } = await render(camera, size, true);
		const meta = await call<CameraMeta>(env, "env.get_camera_meta", {
			camera_name: CAMERAS[camera],
			height: size,
			width: size,
		});
		const raw = (depth as NdArray).toArray();
		const [[fx, , cx], [, fy, cy]] = meta.intrinsic_K;
		const e = meta.extrinsic_cam2world;
		const near = meta.depth_near;
		const far = meta.depth_far;
		const xyz = new Float32Array(size * size * 3);
		for (let r = 0; r < size; r++) {
			const src = (size - 1 - r) * size; // flip rows to match the calibration frame
			for (let c = 0; c < size; c++) {
				let z = raw[src + c];
				if (near !== undefined && far !== undefined) z = near / (1 - z * (1 - near / far));
				const x = ((c - cx) * z) / fx;
				const y = ((r - cy) * z) / fy;
				const i = (r * size + c) * 3;
				for (let k = 0; k < 3; k++) xyz[i + k] = e[k][0] * x + e[k][1] * y + e[k][2] * z + e[k][3];
			}
		}
		const map = { envStep, size, rgb, xyz };
		worldMaps.set(key, map);
		return map;
	}

	async function observe(result: Record<string, unknown>) {
		const raw = await call<Record<string, NdArray>>(env, "env.raw_obs");
		// One call at a time: concurrent calls interleave on the env server's worker pipe.
		const frames = [await render("agentview", 1024, false), await render("wrist", 1024, false)];
		const state = {
			robot0_eef_pos: raw.robot0_eef_pos.toArray().map((v) => round(v)),
			robot0_eef_quat: raw.robot0_eef_quat.toArray().map((v) => round(v)),
			robot0_gripper_qpos: raw.robot0_gripper_qpos.toArray().map((v) => round(v)),
			object_names: Object.keys(raw)
				.filter((k) => k.endsWith("_pos") && !k.includes("robot0") && !k.includes("to_robot"))
				.map((k) => k.slice(0, -4))
				.sort(),
		};
		return {
			content: [
				{
					type: "text" as const,
					text: JSON.stringify({
						result,
						step: envStep,
						terminated,
						truncated,
						task_language: language,
						state,
						images: ["agentview_high 1024x1024", "wrist_high 1024x1024"],
					}),
				},
				...frames.map((f) => ({
					type: "image" as const,
					data: encodePng(f.rgb, 1024, 1024).toString("base64"),
					mimeType: "image/png",
				})),
			],
			details: { result, terminated, truncated },
		};
	}

	/** Register a tool; motion tools return a fresh observation, read-only tools return their result. */
	function tool<P extends TSchema>(
		name: string,
		description: string,
		parameters: P,
		run: (p: Static<P>) => Promise<Record<string, unknown>>,
		motion = true,
	) {
		robot.tool(name, description, parameters, async (params) => {
			if (motion && (terminated || truncated)) {
				return {
					content: [
						{
							type: "text",
							text: `Episode already ended (terminated=${terminated}, truncated=${truncated}).`,
						},
					],
					details: { terminated, truncated },
				};
			}
			const result = await run(params);
			if (motion) return observe(result);
			const { _image, ...rest } = result as { _image?: Buffer };
			const content = [{ type: "text" as const, text: JSON.stringify(rest) }];
			if (!_image) return { content, details: rest };
			return {
				content: [...content, { type: "image" as const, data: _image.toString("base64"), mimeType: "image/png" }],
				details: rest,
			};
		});
	}

	const xyz = Type.Array(Type.Number(), { minItems: 3, maxItems: 3, description: "World-frame [x, y, z] in meters" });
	const num = (description: string) => Type.Optional(Type.Number({ description }));
	const int = (description: string) => Type.Optional(Type.Integer({ description }));
	const camera = Type.Optional(
		Type.Union([Type.Literal("agentview"), Type.Literal("wrist")], { description: "Default agentview" }),
	);

	tool(
		"view_env_state",
		"Current state with agentview (global layout) and wrist (close range) 1024x1024 images. Pixel (row, col) in these images feed back_project.",
		Type.Object({}),
		async () => ({}),
	);

	tool(
		"move_to",
		"Scripted EEF servo to a world xyz; holds orientation. gripper -1 = open, +1 = close (hold +1 while carrying). Never move more than 0.30 m in xy in one call; split long moves.",
		Type.Object({
			xyz,
			gripper: num("-1 open (default), +1 close"),
			tol: num("Position tolerance, m (default 0.012)"),
			step_clip: num("Per-step xyz cap, m (default 0.025)"),
			max_steps: int("Step budget (default 80)"),
			action_scale: num("OSC action scale (default 0.05)"),
			target_yaw: num("Optional world yaw target, rad"),
			yaw_step_clip: num("Per-step yaw clip, rad (default 0.10)"),
		}),
		async ({
			xyz: target,
			gripper: g = -1,
			tol = 0.012,
			step_clip = 0.025,
			max_steps = 80,
			action_scale = 0.05,
			target_yaw,
			yaw_step_clip = 0.1,
		}) => {
			let steps = 0;
			for (; steps < max_steps && !terminated && !truncated; steps++) {
				const diff = target.map((v: number, i: number) => v - eef()[i]);
				if (Math.hypot(...diff) < tol) break;
				const a = [
					...diff.map((d: number) => clip(clip(d, -step_clip, step_clip) / action_scale, -1, 1)),
					0,
					0,
					0,
					g,
				];
				if (target_yaw !== undefined)
					a[5] = clip(clip(wrap(target_yaw - yawOf(await quat())), -yaw_step_clip, yaw_step_clip) / 0.1, -1, 1);
				await step(a);
			}
			return {
				name: "move_to",
				final_eef_pos: eef().map((v) => round(v)),
				final_dist_m: round(Math.hypot(...target.map((v: number, i: number) => v - eef()[i]))),
				steps_used: steps,
			};
		},
	);

	tool(
		"pi0_pick",
		"Pi0.5 closed-loop grasp. Use it only for the grasp; you do every move_to and release. Success needs a descent then a lift with the gripper partly closed; it is a hint, confirm from gripper opening and the wrist image.",
		Type.Object({
			prompt: Type.String({ description: "VLA instruction, e.g. 'pick up the black bowl'" }),
			max_chunks: int("Action-chunk budget (default 24)"),
			lift_thresh: num("Post-descent ascent for success, m (default 0.05)"),
			gripper_closed_thresh: num("Finger separation below which the gripper counts as closed (default 0.06)"),
			gripper_open_thresh: num("Minimum finger separation accepted as holding (default 0.0)"),
			descent_thresh: num("Required descent before lift detection, m (default 0.10)"),
		}),
		async ({
			prompt,
			max_chunks = 24,
			lift_thresh = 0.05,
			gripper_closed_thresh = 0.06,
			gripper_open_thresh = 0,
			descent_thresh = 0.1,
		}) => {
			const start = eef()[2];
			let minZ = start;
			let postMinPeak = start;
			let minGrip = gripper();
			let success = false;
			let chunks = 0;
			const vla_seeds: (number | null)[] = [];
			while (chunks < max_chunks) {
				vla_seeds.push((await vlaChunk(prompt)) ?? null);
				chunks++;
				const z = eef()[2];
				if (z < minZ) {
					minZ = z;
					postMinPeak = z;
				} else postMinPeak = Math.max(postMinPeak, z);
				minGrip = Math.min(minGrip, gripper());
				const g = gripper();
				if (
					start - minZ >= descent_thresh &&
					postMinPeak - minZ >= lift_thresh &&
					g >= gripper_open_thresh &&
					g < gripper_closed_thresh
				) {
					success = true;
					break;
				}
				if (terminated || truncated) {
					success = terminated;
					break;
				}
			}
			return {
				name: "pick",
				instruction: prompt,
				success,
				chunks_used: chunks,
				peak_lift_m: round(postMinPeak - minZ),
				descent_m: round(start - minZ),
				min_gripper_opening: round(minGrip),
				final_gripper_opening: round(gripper()),
				vla_seeds,
			};
		},
	);

	tool(
		"pi0_doubled",
		"Pi0.5 closed-loop contact skill for non-pick interactions (knob, stove, drawer, button, short push). success only mirrors task termination; inspect the images for intermediate progress.",
		Type.Object({ prompt: Type.String({ description: "e.g. 'turn on the stove'" }), max_chunks: int("Default 20") }),
		async ({ prompt, max_chunks = 20 }) => {
			let chunks = 0;
			const vla_seeds: (number | null)[] = [];
			while (chunks < max_chunks && !terminated && !truncated) {
				vla_seeds.push((await vlaChunk(prompt)) ?? null);
				chunks++;
			}
			return { name: "pi0_doubled", instruction: prompt, success: terminated, chunks_used: chunks, vla_seeds };
		},
	);

	tool(
		"release",
		"Open the gripper in place for up to max_steps; triggers termination when the goal predicate holds.",
		Type.Object({ max_steps: int("Default 20") }),
		async ({ max_steps = 20 }) => {
			const start = gripper();
			let steps = 0;
			while (steps < max_steps && !terminated && !truncated) {
				await step([0, 0, 0, 0, 0, 0, -1]);
				steps++;
			}
			return {
				name: "release",
				steps_used: steps,
				start_gripper_opening: round(start),
				final_gripper_opening: round(gripper()),
			};
		},
	);

	tool(
		"set_gripper",
		"Hold the pose and drive the gripper for `steps` env steps (e.g. +1 for 8-12 steps to firm a grip).",
		Type.Object({ gripper: num("-1 open (default), +1 close"), steps: int("Default 5") }),
		async ({ gripper: g = -1, steps = 5 }) => {
			for (let i = 0; i < steps && !terminated && !truncated; i++) await step([0, 0, 0, 0, 0, 0, g]);
			return { name: "set_gripper", gripper: g, steps };
		},
	);

	async function rotate(
		kind: "yaw" | "pitch",
		target: number | undefined,
		delta: number | undefined,
		g: number,
		max_steps: number,
		tol: number,
		step_clip: number,
	) {
		const angle = kind === "yaw" ? yawOf : pitchOf;
		const start = angle(await quat());
		if (target === undefined && delta === undefined) throw new Error(`need target_${kind} or delta_${kind}`);
		const goal = target ?? start + (delta as number);
		let steps = 0;
		for (; steps < max_steps && !terminated && !truncated; steps++) {
			const err = wrap(goal - angle(await quat()));
			if (Math.abs(err) < tol) break;
			const a = [0, 0, 0, 0, 0, 0, g];
			a[kind === "yaw" ? 5 : 3] = clip(clip(err, -step_clip, step_clip) / 0.1, -1, 1);
			await step(a);
		}
		const final = angle(await quat());
		return {
			[`start_${kind}`]: round(start),
			[`target_${kind}`]: round(goal),
			[`final_${kind}`]: round(final),
			final_err: round(wrap(goal - final)),
			steps_used: steps,
		};
	}

	tool(
		"rotate_wrist",
		"Rotate the wrist about world z. Give target_yaw (absolute) or delta_yaw (relative), radians. Holds xyz.",
		Type.Object({
			target_yaw: num("rad"),
			delta_yaw: num("rad"),
			gripper: num("Default +1"),
			max_steps: int("Default 40"),
			tol: num("rad, default 0.02"),
			step_clip: num("rad, default 0.10"),
		}),
		async (p) => ({
			name: "rotate_wrist",
			...(await rotate(
				"yaw",
				p.target_yaw,
				p.delta_yaw,
				p.gripper ?? 1,
				p.max_steps ?? 40,
				p.tol ?? 0.02,
				p.step_clip ?? 0.1,
			)),
		}),
	);

	tool(
		"rotate_pitch",
		"Tilt the gripper about world x (pitch 0 = pointing down, +pi/2 = pointing +y). Give target_pitch or delta_pitch, radians. Holds xyz and yaw. Use before entering a narrow opening facing ±y.",
		Type.Object({
			target_pitch: num("rad"),
			delta_pitch: num("rad"),
			gripper: num("Default +1"),
			max_steps: int("Default 40"),
			tol: num("rad, default 0.02"),
			step_clip: num("rad, default 0.10"),
		}),
		async (p) => ({
			name: "rotate_pitch",
			...(await rotate(
				"pitch",
				p.target_pitch,
				p.delta_pitch,
				p.gripper ?? 1,
				p.max_steps ?? 40,
				p.tol ?? 0.02,
				p.step_clip ?? 0.1,
			)),
		}),
	);

	tool(
		"move_pose",
		"Servo xyz and pitch/yaw together each step. Use when move_to stalls on deep or low reaches (cabinet fronts, microwave). gripper defaults to -1 (open): pass +1 while holding.",
		Type.Object({
			xyz,
			target_pitch: num("rad"),
			target_yaw: num("rad"),
			gripper: num("Default -1"),
			step_clip: num("m, default 0.02"),
			pitch_step: num("rad, default 0.08"),
			yaw_step: num("rad, default 0.08"),
			tol: num("m, default 0.012"),
			ori_tol: num("rad, default 0.05"),
			action_scale: num("Default 0.05"),
			max_steps: int("Default 150"),
		}),
		async ({
			xyz: target,
			target_pitch,
			target_yaw,
			gripper: g = -1,
			step_clip = 0.02,
			pitch_step = 0.08,
			yaw_step = 0.08,
			tol = 0.012,
			ori_tol = 0.05,
			action_scale = 0.05,
			max_steps = 150,
		}) => {
			let steps = 0;
			for (; steps < max_steps && !terminated && !truncated; steps++) {
				const q = await quat();
				const diff = target.map((v: number, i: number) => v - eef()[i]);
				const pErr = target_pitch === undefined ? 0 : wrap(target_pitch - pitchOf(q));
				const yErr = target_yaw === undefined ? 0 : wrap(target_yaw - yawOf(q));
				if (Math.hypot(...diff) < tol && Math.abs(pErr) < ori_tol && Math.abs(yErr) < ori_tol) break;
				await step([
					...diff.map((d: number) => clip(clip(d, -step_clip, step_clip) / action_scale, -1, 1)),
					clip(clip(pErr, -pitch_step, pitch_step) / 0.1, -1, 1),
					0,
					clip(clip(yErr, -yaw_step, yaw_step) / 0.1, -1, 1),
					g,
				]);
			}
			return {
				name: "move_pose",
				final_eef_pos: eef().map((v) => round(v)),
				final_dist_m: round(Math.hypot(...target.map((v: number, i: number) => v - eef()[i]))),
				final_pitch: round(pitchOf(await quat())),
				steps_used: steps,
			};
		},
	);

	tool(
		"view_camera_meta",
		"Camera calibration (intrinsic K, cam-to-world extrinsic, depth range) for the current step.",
		Type.Object({
			camera,
			resolution: Type.Optional(
				Type.Union([Type.Literal("high"), Type.Literal("low")], {
					description: "high = 1024 (default), low = 256",
				}),
			),
		}),
		async ({ camera: c = "agentview", resolution = "high" }) => {
			const size = resolution === "high" ? 1024 : 256;
			return {
				camera: c,
				resolution,
				meta: await call(env, "env.get_camera_meta", {
					camera_name: CAMERAS[c as Camera],
					height: size,
					width: size,
				}),
			};
		},
		false,
	);

	tool(
		"segment",
		"SAM3 segmentation of the current 1024x1024 camera image. Give exactly one of a text prompt or a positive point [row, col]. The top mask is projected through the depth world map; world_xyz is the median over mask pixels. Returns an overlay image.",
		Type.Object({
			prompt: Type.Optional(Type.String()),
			point: Type.Optional(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 })),
			camera,
			min_score: num("Default 0.2"),
		}),
		async ({ prompt, point, camera: c = "agentview", min_score = 0.2 }) => {
			// Models often fill both optional fields; a non-empty prompt wins.
			const text = prompt?.trim();
			if (!text && !point) return { error: "give a text prompt or a point [row, col]" };
			const map = await worldMap(c, 1024);
			const png = encodePng(map.rgb, 1024, 1024);
			const res = await call<{
				found: boolean;
				score?: number;
				box?: number[];
				mask_png_base64?: string;
				reason?: string;
			}>(sam3, "sam3.segment", {
				image_base64: png.toString("base64"),
				...(text ? { text_prompt: text } : { point }),
				min_score,
			});
			if (!res.found || !res.mask_png_base64)
				return {
					found: false,
					error: res.reason ?? "no mask",
					fallback: "Pick pixels in the image and use back_project.",
				};
			const mask = decodePngChannel(Buffer.from(res.mask_png_base64, "base64"));
			if (mask.width !== 1024 || mask.height !== 1024)
				return { found: true, error: `mask ${mask.width}x${mask.height} does not match the 1024 world map` };
			const xs: number[] = [];
			const ys: number[] = [];
			const pts: number[][] = [];
			const overlay = Buffer.from(map.rgb);
			for (let i = 0; i < mask.data.length; i++) {
				if (mask.data[i] < 128) continue;
				ys.push(Math.floor(i / 1024));
				xs.push(i % 1024);
				overlay[i * 3] = Math.round(0.55 * overlay[i * 3] + 0.45 * 255);
				overlay[i * 3 + 1] = Math.round(0.55 * overlay[i * 3 + 1]);
				overlay[i * 3 + 2] = Math.round(0.55 * overlay[i * 3 + 2]);
				const p = [map.xyz[i * 3], map.xyz[i * 3 + 1], map.xyz[i * 3 + 2]];
				if (p.every(Number.isFinite) && Math.abs(p[0]) + Math.abs(p[1]) + Math.abs(p[2]) > 1e-6) pts.push(p);
			}
			const out: Record<string, unknown> = {
				found: true,
				camera: c,
				score: res.score === undefined ? null : round(res.score, 3),
				box: res.box,
				n_pixels: xs.length,
				n_valid: pts.length,
				centroid_pixel: [Math.round(median(xs)), Math.round(median(ys))],
			};
			out.world_xyz = pts.length < 10 ? null : [0, 1, 2].map((k) => round(median(pts.map((p) => p[k]))));
			if (pts.length < 10) out.world_error = `too few valid depth pixels (${pts.length})`;
			out._image = encodePng(overlay, 1024, 1024);
			return out;
		},
		false,
	);

	tool(
		"back_project",
		"World xyz of a pixel (row, col; row 0 = top) in the current camera image, from the depth world map. Region mode: row_range + col_range (+ optional z_min/z_max) returns the midpoint of world xy over that window, e.g. a container's interior center. Pixels from the 1024 images use resolution high (default).",
		Type.Object({
			row: int("Pixel row"),
			col: int("Pixel column"),
			camera,
			resolution: Type.Optional(Type.Union([Type.Literal("high"), Type.Literal("low")])),
			row_range: Type.Optional(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 })),
			col_range: Type.Optional(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 })),
			z_min: num("Region mode: keep pixels with world z >= z_min"),
			z_max: num("Region mode: keep pixels with world z <= z_max"),
		}),
		async ({ row, col, camera: c = "agentview", resolution = "high", row_range, col_range, z_min, z_max }) => {
			const size = resolution === "high" ? 1024 : 256;
			const map = await worldMap(c, size);
			const at = (r: number, cc: number) => {
				const i = (r * size + cc) * 3;
				return [map.xyz[i], map.xyz[i + 1], map.xyz[i + 2]];
			};
			const valid = (p: number[]) =>
				p.every(Number.isFinite) && Math.abs(p[0]) + Math.abs(p[1]) + Math.abs(p[2]) > 1e-6;
			// An empty window ([0, 0]) is a placeholder, not a region.
			const span = (r?: number[]) => (r && Math.max(...r) > Math.min(...r) ? r : undefined);
			const rows = span(row_range);
			const cols = span(col_range);
			if (rows || cols) {
				if (!rows || !cols) return { error: "region mode needs both row_range and col_range" };
				const [r0, r1] = [clip(Math.min(...rows), 0, size), clip(Math.max(...rows), 0, size)];
				const [c0, c1] = [clip(Math.min(...cols), 0, size), clip(Math.max(...cols), 0, size)];
				let pts: number[][] = [];
				for (let r = r0; r < r1; r++) for (let cc = c0; cc < c1; cc++) if (valid(at(r, cc))) pts.push(at(r, cc));
				if (z_min !== undefined) pts = pts.filter((p) => p[2] >= z_min);
				if (z_max !== undefined) pts = pts.filter((p) => p[2] <= z_max);
				if (pts.length < 8)
					return { error: `too few valid pixels in region (${pts.length}); widen the window or the z band` };
				const axis = (k: number) => pts.map((p) => p[k]);
				return {
					camera: c,
					resolution,
					mode: "region",
					center_xyz: [
						round((Math.min(...axis(0)) + Math.max(...axis(0))) / 2),
						round((Math.min(...axis(1)) + Math.max(...axis(1))) / 2),
						round(median(axis(2))),
					],
					median_xyz: [0, 1, 2].map((k) => round(median(axis(k)))),
					n_valid: pts.length,
				};
			}
			if (row === undefined || col === undefined) return { error: "give row and col, or row_range and col_range" };
			if (row < 0 || row >= size || col < 0 || col >= size)
				return { error: `pixel (${row},${col}) out of bounds for ${size}x${size}` };
			const p = at(row, col);
			if (!valid(p)) return { error: `invalid world xyz at (${row},${col}); pick another pixel` };
			return { camera: c, resolution, pixel: [row, col], world_xyz: p.map((v) => round(v)) };
		},
		false,
	);

	/**
	 * One action unit (../units): drive the gripper, servo the EEF to its current position plus
	 * `delta` (holding the gripper command), turn the wrist by `yaw`, or hold one step (STOP).
	 */
	async function unitStep(move: Move) {
		if (terminated || truncated)
			return {
				content: [{ type: "text" as const, text: `Episode already ended (terminated=${terminated}).` }],
				details: { terminated, truncated },
			};
		let steps = 0;
		if (move.gripper) {
			grip = move.gripper === "close" ? 1 : -1;
			// Until the fingers stop moving (they stop on a grasped object), at most 15 steps.
			for (let prev = Number.NaN; steps < 15 && !terminated && !truncated; ) {
				await step([0, 0, 0, 0, 0, 0, grip]);
				steps++;
				if (steps > 3 && Math.abs(gripper() - prev) < 5e-4) break;
				prev = gripper();
			}
		}
		if (Math.hypot(...move.delta) > 0) {
			const target = eef().map((v, i) => v + move.delta[i]);
			for (let k = 0; k < 25 && !terminated && !truncated; k++) {
				const diff = target.map((v, i) => v - eef()[i]);
				if (Math.hypot(...diff) < 0.004) break;
				await step([...diff.map((d) => clip(clip(d, -0.025, 0.025) / 0.05, -1, 1)), 0, 0, 0, grip]);
				steps++;
			}
		}
		if (move.yaw) steps += (await rotate("yaw", undefined, move.yaw, grip, 25, 0.02, 0.1)).steps_used as number;
		if (!move.gripper && !Math.hypot(...move.delta) && !move.yaw) {
			await step([0, 0, 0, 0, 0, 0, grip]);
			steps++;
		}
		return observe({
			name: "act",
			steps_used: steps,
			eef_pos: eef().map((v) => round(v)),
			gripper: round(gripper()),
		});
	}

	/** Median world z of the agentview's bottom-center strip: the table or floor surface nearest the camera. */
	async function surfaceZ() {
		const map = await worldMap("agentview", 256);
		const z: number[] = [];
		for (let r = 218; r < 256; r++)
			for (let c = 77; c < 179; c++) {
				const v = map.xyz[(r * 256 + c) * 3 + 2];
				if (Number.isFinite(v)) z.push(v);
			}
		return z.length ? round(median(z)) : undefined;
	}

	/** Affordance points (../units `point`): the marked 1024 image and world xyz (median of a 7x7 window). */
	async function locate(camera: Camera, points: [number, number][]) {
		const map = await worldMap(camera, 1024);
		let rgb: Buffer = map.rgb;
		const xyz = points.map(([fy, fx]) => {
			const row = clip(Math.round(fy * 1023), 0, 1023);
			const col = clip(Math.round(fx * 1023), 0, 1023);
			rgb = mark({ width: 1024, height: 1024, rgb }, row, col, [255, 32, 32]);
			const pts: number[][] = [];
			for (let r = Math.max(0, row - 3); r <= Math.min(1023, row + 3); r++)
				for (let c = Math.max(0, col - 3); c <= Math.min(1023, col + 3); c++) {
					const i = (r * 1024 + c) * 3;
					const p = [map.xyz[i], map.xyz[i + 1], map.xyz[i + 2]];
					if (p.every(Number.isFinite) && Math.abs(p[0]) + Math.abs(p[1]) + Math.abs(p[2]) > 1e-6) pts.push(p);
				}
			return pts.length < 5 ? null : [0, 1, 2].map((k) => round(median(pts.map((p) => p[k]))));
		});
		return { image: encodePng(rgb, 1024, 1024), xyz };
	}

	/** Start (or attach to) the env server and restore the initial scene; returns the tools to activate. */
	async function startEpisode() {
		const { suite, task, seed } = robot.task;
		vla = new RpcClient(flag("vla", ""));
		sam3 = new RpcClient(flag("sam3", ""));
		const endpoint = pi.getFlag("env") as string | undefined;
		if (endpoint) {
			env = new RpcClient(endpoint);
			await env.ready();
		} else {
			const services = flag("services", SERVICES);
			env = await robot.serve({
				python: flag("python", "python"),
				args: [
					...["-m", "pi_embodied_services.robots.libero.env_server"],
					...["--suite", suite, "--task", task, "--seed", seed],
				],
				cwd: services,
				env: {
					...process.env,
					PYTHONPATH: services,
					LIBERO_TYPE: flag("libero-type", "pro"),
					MUJOCO_GL: "egl",
					ROBOT_PLATFORM: "LIBERO",
				},
				log: (port) => join(tmpdir(), `pi-embodied-env-${suite}-t${task}-s${seed}-${port}.log`),
			});
		}
		await resetEpisode();
		language = await call<string>(env, "env.get_task_language");
		fly.reset(obs, flyMeta());
		return TOOLS;
	}
}
