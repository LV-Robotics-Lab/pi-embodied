/**
 * RoboCasa365 robot for pi.
 *
 *   pi -e packages/embodied/src/robocasa --task-name OpenDrawer --split target --seed 1
 *
 * Starts one RoboCasa env server per session (PandaOmron mobile manipulator) and attaches
 * to a running RLDX-1 VLA server (see serve.sh) under a private RPC session, which holds
 * the policy's memory/RTC state. Tools are the RoboCasa primitives. Every action
 * returns a new numbered state with agentview, navview and wrist images; world maps are
 * kept per state for back-projection. Success is the env's own `_check_success()`
 * (`state.success`), recorded in the session's `robot_result` entry.
 */

import { randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";
import { encodePng } from "../png.ts";
import { attach, defineRobot, median, round, SERVICES } from "../robot.ts";
import { NdArray, RpcClient } from "../rpc.ts";
import { vlaSeeds } from "../vla-seed.ts";

const read = (name: string) => readFileSync(new URL(name, import.meta.url), "utf8");
const SYSTEM = read("./SYSTEM.md");
const MEMORY = { hf: read("./memory-hf.md"), local: read("./memory-local.md") };
const EXPLORE = read("./explore.md");
const CAMERAS = { agentview: "robot0_agentview_left", navview: "mobilebase0_navview", wrist: "robot0_eye_in_hand" };
const VLA_CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"];
const SIZE = 256; // env camera and RLDX observation resolution
const OSC_ROT_SCALE = 0.5; // action 1.0 -> 0.5 rad
const SPLITS = ["target", "pretrain", "all"];
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
type Grip = number | "close" | "open" | "hold" | undefined;
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

/** Integer from the environment (the RLDX_* protocol knobs), else `fallback`. */
const envInt = (name: string, fallback: number) => {
	const v = process.env[name];
	return v === undefined || v === "" ? fallback : Number.parseInt(v, 10);
};
const norm = (v: number[]) => Math.hypot(...v);
const clip = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
/** Yaw of an xyzw quaternion, as scipy's `as_euler("xyz")[2]`. */
const yawOf = ([x, y, z, w]: number[]) => Math.atan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z));
/** Rows of an HxWx3 image in reverse order (MuJoCo renders bottom-up). */
function flipRows(data: Buffer, height: number): Buffer {
	const row = data.length / height;
	const out = Buffer.alloc(data.length);
	for (let y = 0; y < height; y++) data.copy(out, (height - 1 - y) * row, y * row, (y + 1) * row);
	return out;
}

/** Moore-Penrose inverse of a 3x3 matrix, as (JᵀJ + εI)⁻¹Jᵀ with a vanishing ε. */
function pinv3(J: number[][]): number[][] {
	const Jt = [0, 1, 2].map((c) => J.map((row) => row[c]));
	const A = Jt.map((r) => [0, 1, 2].map((c) => r.reduce((s, v, k) => s + v * J[k][c], 0)));
	const eps = 1e-12 + 1e-10 * (A[0][0] + A[1][1] + A[2][2]);
	for (let i = 0; i < 3; i++) A[i][i] += eps;
	const [[a, b, c], [d, e, f], [g, h, i]] = A;
	const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
	const inv = [
		[e * i - f * h, c * h - b * i, b * f - c * e],
		[f * g - d * i, a * i - c * g, c * d - a * f],
		[d * h - e * g, b * g - a * h, a * e - b * d],
	].map((r) => r.map((v) => v / det));
	return inv.map((r) => [0, 1, 2].map((c) => r.reduce((s, v, k) => s + v * Jt[k][c], 0)));
}

/** RpcClient bound to one RPC session (the RLDX server keys policy memory/RTC state by it). */
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
	pi.registerFlag("split", { type: "string", default: "target", description: "target | pretrain | all" });
	pi.registerFlag("seed", { type: "string", default: "0", description: "Scene seed" });
	pi.registerFlag("hi-res", { type: "string", default: "0", description: "Hi-res agentview resolution (0 = off)" });
	pi.registerFlag("rldx", { type: "string", default: "http://127.0.0.1:18500", description: "RLDX-1 VLA server" });
	pi.registerFlag("env", { type: "string", description: "Attach to a running env server instead of starting one" });
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
	let posJac: number[][] | undefined; // world dpos per unit arm-xyz action (columns = action axes)
	let fwdOffset: number | undefined; // world driving heading = base yaw + offset
	// True whenever a non-VLA primitive stepped the env since the last RLDX call; the next
	// RLDX call then reseeds its frame history instead of stitching stale frames on.
	let vlaDesync = true;
	let modality: { video_delta_indices: number[]; hist_maxlen: number } | undefined;
	let hist: Frame[] = [];
	let lastPrompt: string | undefined;
	let attempt = 1;
	const cell = () => ({ task: robot.task["task-name"], split: robot.task.split, seed: robot.task.seed });
	const tag = () => `${cell().task}_${cell().split}_s${cell().seed}`;
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
	const robot = defineRobot(pi, {
		name: "robocasa",
		task: ["task-name", "split", "seed"],
		keepImages: 6,
		budget: { turns: 0, seconds: 0 },
		// Observations carry the agentview, navview and wrist images.
		vdm: { views: 3, wrist: 2 },
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
				const elapsed = round((Date.now() - t0) / 1000, 1);
				return view(await capture({ action: "reset" }, out, elapsed), { agent_elapsed_s: elapsed });
			},
			prompt: () => EXPLORE,
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
				seed: cell().seed,
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

	const vec = (key: string) => obs[key].toArray();
	const eef = () => vec("robot0_eef_pos");
	const finger = () => vec("robot0_gripper_qpos")[0];
	const zero = (baseMode = -1) => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, baseMode];
	/** a[6] for a motion step: "close" = +1, "open" = -1, a number passes through; "hold" servos the fingers back to `q`. */
	const grip = (g: Grip, q: number) =>
		g === "close" ? 1 : g === "open" ? -1 : typeof g === "number" ? clip(g, -1, 1) : clip(60 * (finger() - q), -1, 1);

	/** One env step with the PandaOmron 12-D action [eef_pos 3, eef_rot 3, gripper, base 3, torso, base_mode]. */
	async function step(a: number[]) {
		if (robot.signal?.aborted) throw new Error("interrupted");
		obs = (await env.call<[Raw, unknown, unknown, unknown]>("env.step", {}, 60_000, [a], robot.signal))[0];
		envSteps++;
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

	// ---- arm ----

	/** Probe 3 unit arm-xyz actions and measure world dpos: world_dpos ≈ J @ action_xyz. */
	async function calibrateArm(g: number) {
		const cols: number[][] = [];
		for (let axis = 0; axis < 3; axis++) {
			const p0 = eef();
			const a = zero();
			a[axis] = 0.4;
			a[6] = g;
			for (let i = 0; i < 3; i++) await step(a);
			cols.push(eef().map((v, k) => (v - p0[k]) / 1.2));
		}
		posJac = [0, 1, 2].map((r) => cols.map((col) => col[r]));
		return posJac;
	}

	async function moveTo(target: number[], gripper: Grip = "hold", step_clip = 0.02, max_steps = 200, tol = 0.012) {
		vlaDesync = true;
		const q = finger();
		const Jinv = pinv3(posJac ?? (await calibrateArm(grip(gripper, q))));
		const done = (ok: boolean, steps: number) => ({
			ok,
			steps,
			final_dist: round(norm(target.map((v, k) => v - eef()[k]))),
			eef: eef().map((v) => round(v)),
			gripper_qpos: round(finger()),
		});
		for (let i = 0; i < max_steps; i++) {
			const cur = eef();
			const err = target.map((v, k) => v - cur[k]);
			const dist = norm(err);
			if (dist < tol) return done(true, i);
			const d = dist <= step_clip ? err : err.map((e) => (e / dist) * step_clip);
			const a = zero();
			for (let k = 0; k < 3; k++) a[k] = clip(Jinv[k][0] * d[0] + Jinv[k][1] * d[1] + Jinv[k][2] * d[2], -1, 1);
			a[6] = grip(gripper, q);
			await step(a);
		}
		return done(false, max_steps);
	}

	async function setGripper(g = 1, steps = 10) {
		vlaDesync = true;
		const a = zero();
		a[6] = clip(g, -1, 1);
		for (let i = 0; i < steps; i++) await step(a);
		return { ok: true, gripper_qpos: vec("robot0_gripper_qpos").map((v) => round(v)) };
	}

	// ---- base ----

	const basePos = () => vec("robot0_base_pos");
	const baseYaw = () => yawOf(vec("robot0_base_quat"));

	/** Drive forward briefly and measure the world direction the base actually goes. */
	async function calibrateForward(g: number) {
		const p0 = basePos();
		const y0 = baseYaw();
		const a = zero(1);
		a[6] = clip(g, -1, 1);
		a[7] = 1;
		for (let i = 0; i < 6; i++) await step(a);
		const p1 = basePos();
		const [dx, dy] = [p1[0] - p0[0], p1[1] - p0[1]];
		fwdOffset = Math.hypot(dx, dy) > 0.005 ? Math.atan2(dy, dx) - y0 : 0;
		return fwdOffset;
	}

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
		const forceReset = Boolean(p.force_reset) || vlaDesync;
		vlaDesync = false;
		await vla.call("session.register", {}, 30_000); // refreshes, or re-creates an idle-expired session
		modality ??= await vla.call<{ video_delta_indices: number[]; hist_maxlen: number }>("vla.get_modality_config");
		const { video_delta_indices: vdi, hist_maxlen: maxlen } = modality;
		const push = async () => {
			hist.push(await frame());
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
			for (let i = 0; i < Math.min(nSteps, horizon); i++) {
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

	const xyz = Type.Array(Type.Number(), { minItems: 3, maxItems: 3, description: "World-frame [x, y, z] in meters" });
	const num = (description: string) => Type.Optional(Type.Number({ description }));
	const int = (description: string) => Type.Optional(Type.Integer({ description }));
	const gripArg = (description: string) =>
		Type.Optional(StringEnum(["close", "open", "hold"] as const, { description: `${description} (default 'hold')` }));
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

	tool(
		"move_to",
		"Scripted EEF servo to a world-frame XYZ target via the OSC controller. Holds pitch/yaw orientation (use rotate_pitch to reorient). gripper='hold' (default) maintains current finger width: carry-safe without crushing small objects. Pass 'close' to close, 'open' to open. Never command a single move_to with |dxyz| > 0.30: OSC flips IK; split long traversal into 2-3 waypoints at carry z.",
		Type.Object({
			xyz,
			gripper: gripArg("'close', 'open', or 'hold' to maintain the current finger width"),
			step_clip: num("Per-step dxyz cap, m (default 0.02)"),
			max_steps: int("Step budget (default 200)"),
			tol: num("Position tolerance, m (default 0.012)"),
		}),
		(p) => moveTo(p.xyz, p.gripper, p.step_clip, p.max_steps, p.tol),
	);

	tool(
		"move_delta",
		"Relative EEF displacement: target = current_eef + dxyz, then move_to. Use for small adjustments (micro-align for grasp, approach). gripper='hold' (default) maintains current finger width.",
		Type.Object({
			dxyz: Type.Array(Type.Number(), { minItems: 3, maxItems: 3, description: "Relative [dx, dy, dz], m" }),
			gripper: gripArg("'close', 'open', or 'hold'"),
			step_clip: num("Per-step dxyz cap, m (default 0.02)"),
			max_steps: int("Step budget (default 80)"),
		}),
		(p) =>
			moveTo(
				eef().map((v, k) => v + p.dxyz[k]),
				p.gripper,
				p.step_clip,
				p.max_steps ?? 80,
			),
	);

	tool(
		"rotate_pitch",
		"Tilt the wrist forward (axis-angle about the control X-axis); pitches the gripper down/up. Holds xyz fixed. Use before threading the gripper into a narrow opening whose front face normal is along world +/-y.",
		Type.Object({
			target_pitch: num("Absolute pitch target, rad (clamped +/-1.5; default 0.6)"),
			gripper: num("Gripper command held during rotation (default +1)"),
			n: int("Number of env steps for the rotation (default 12)"),
		}),
		async ({ target_pitch = 0.6, gripper = 1, n = 12 }) => {
			vlaDesync = true;
			const per = clip(target_pitch, -1.5, 1.5) / n;
			const a = zero();
			a[3] = clip(per / OSC_ROT_SCALE, -1, 1);
			a[6] = clip(gripper, -1, 1);
			for (let i = 0; i < n; i++) await step(a);
			return { ok: true, eef: eef().map((v) => round(v)) };
		},
	);

	tool(
		"set_gripper",
		"Hold the current EEF pose and drive the gripper command for `steps` env steps. Use to firm up a grip mid-carry or to actively open/close the gripper.",
		Type.Object({ gripper: num("+1 close, -1 open (default +1)"), steps: int("Env steps to hold (default 10)") }),
		({ gripper = 1, steps = 10 }) => setGripper(gripper, steps),
	);

	tool(
		"release",
		"Open the gripper for `steps` env steps while holding the EEF in place (set_gripper(-1)). Use to drop a grasped object.",
		Type.Object({ steps: int("Env steps (default 10)") }),
		({ steps = 10 }) => setGripper(-1, steps),
	);

	tool(
		"scripted_grasp",
		"Coarse scripted grasp: open -> hover above target -> descend -> close -> lift. A fallback when the VLA closed-loop grasp is unavailable; for hard objects prefer rldx_arm. approach_z and grasp_z_offset are offsets from the target xyz.",
		Type.Object({
			xyz,
			approach_z: num("Z offset above target before descent, m (default 0.10)"),
			grasp_z_offset: num("Z offset at grasp point (default 0.0; negative = below target)"),
			step_clip: num("Per-step dxyz cap during approach, m (default 0.02)"),
		}),
		async ({ xyz: t, approach_z = 0.1, grasp_z_offset = 0, step_clip = 0.02 }) => {
			const at = (dz: number) => [t[0], t[1], t[2] + dz];
			await setGripper(-1, 4);
			let r: Record<string, unknown> = await moveTo(at(approach_z), -1, step_clip);
			if (!r.ok) return { ...r, stage: "approach" };
			r = await moveTo(at(grasp_z_offset), -1, 0.012, 200, 0.01);
			if (!r.ok) return { ...r, stage: "descent" };
			await setGripper(1, 14);
			r = await moveTo(at(approach_z + 0.05), "hold", 0.015);
			if (!r.ok) return { ...r, stage: "lift" };
			return {
				ok: true,
				gripper_qpos: vec("robot0_gripper_qpos").map((v) => round(v)),
				eef: eef().map((v) => round(v)),
			};
		},
	);

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

	tool(
		"navigate_to",
		"Drive the mobile base toward a WORLD (x, y) target. Online-calibrates the base forward heading, then turns to face and drives forward closed-loop. Holds the arm in place. gripper='hold' (default) maintains finger width while driving. Use tol = expected approach distance + object radius.",
		Type.Object({
			xy: Type.Array(Type.Number(), { minItems: 2, maxItems: 2, description: "World-frame [x, y], m" }),
			tol: num("Distance threshold to stop, m (default 0.20)"),
			max_steps: int("Step budget (default 300)"),
			gripper: gripArg("'close', 'open', or 'hold'"),
		}),
		async ({ xy, tol = 0.2, max_steps = 300, gripper }) => {
			vlaDesync = true;
			const q = finger();
			if (fwdOffset === undefined) await calibrateForward(grip(gripper, q));
			const offset = fwdOffset ?? 0;
			const start = basePos().slice(0, 2);
			const end = (ok: boolean, steps: number) => {
				posJac = undefined; // the base moved: recalibrate the arm servo
				const bp = basePos();
				const moved = Math.hypot(bp[0] - start[0], bp[1] - start[1]);
				return {
					ok,
					steps,
					final_dist: round(Math.hypot(xy[0] - bp[0], xy[1] - bp[1])),
					moved: round(moved),
					...(ok ? {} : { stuck: moved < 0.12 }), // barely moved: rammed a fixture (no path planning)
					start_pos: start.map((v) => round(v)),
					base_pos: bp.map((v) => round(v)),
				};
			};
			for (let i = 0; i < max_steps; i++) {
				const bp = basePos();
				const to = [xy[0] - bp[0], xy[1] - bp[1]];
				if (Math.hypot(to[0], to[1]) < tol) return end(true, i);
				const e = Math.atan2(to[1], to[0]) - (baseYaw() + offset) + Math.PI;
				const dyaw = e - 2 * Math.PI * Math.floor(e / (2 * Math.PI)) - Math.PI;
				const a = zero(1);
				a[6] = grip(gripper, q);
				if (Math.abs(dyaw) > 0.3) a[9] = Math.sign(dyaw);
				else {
					a[7] = 1;
					a[9] = clip(dyaw * 1.5, -0.4, 0.4);
				}
				await step(a);
			}
			return end(false, max_steps);
		},
	);

	tool(
		"move_base",
		"Raw base velocity commands in the robot's LOCAL frame: +forward drives forward, +lateral strafes right, +turn rotates CCW (yaw). Values clamped to [-1, 1]. Use for fine base adjustments near a target; navigate_to for long range. gripper='hold' (default) maintains finger width while driving.",
		Type.Object({
			forward: num("Forward velocity, [-1, 1] (default 0)"),
			lateral: num("Lateral / strafe velocity, [-1, 1] (default 0)"),
			turn: num("Yaw rotation velocity, [-1, 1] (default 0)"),
			steps: int("Env steps (default 10)"),
			gripper: gripArg("'close', 'open', or 'hold'"),
		}),
		async ({ forward = 0, lateral = 0, turn = 0, steps = 10, gripper }) => {
			vlaDesync = true;
			const q = finger();
			const a = zero(1);
			a[7] = clip(forward, -1, 1);
			a[8] = clip(lateral, -1, 1);
			a[9] = clip(turn, -1, 1);
			const bp0 = basePos();
			for (let i = 0; i < steps; i++) {
				a[6] = grip(gripper, q);
				await step(a);
			}
			const bp1 = basePos();
			return { ok: true, base_moved: bp1.map((v, k) => round(v - bp0[k])), base_pos: bp1.map((v) => round(v)) };
		},
	);

	/** Exploration's `reset`: a freshly sampled scene; arm/base calibration and the RLDX session start over. */
	async function resetEpisode(signal?: AbortSignal) {
		vlaDesync = true;
		obs = await env.call<Raw>("env.reset", {}, 120_000, [], signal);
		seeds.reset();
		posJac = fwdOffset = undefined;
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

	async function startEpisode() {
		states = [];
		envSteps = 0;
		posJac = fwdOffset = modality = lastPrompt = undefined;
		hist = [];
		vlaDesync = true;
		attempt = 1;
		seeds.reset();
		const { task, split, seed } = cell();
		if (!SPLITS.includes(split)) throw new Error(`--split must be one of ${SPLITS.join(", ")}`);
		const rldxClient = sessionRpc(flag("rldx", ""));
		vla = rldxClient;
		const endpoint = pi.getFlag("env") as string | undefined;
		const services = flag("services", SERVICES);
		const cuda = pi.getFlag("cuda-device") as string | undefined;
		[env] = await Promise.all([
			endpoint
				? attach(endpoint)
				: robot.serve({
						python: flag("robocasa-python", "python"),
						args: [
							...["-m", "pi_embodied_services.robots.robocasa.env_server"],
							...["--task-name", task, "--split", split, "--seed", seed],
							...(cuda ? ["--cuda-device", cuda] : []),
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
						log: (port) => join(flag("log-dir", tmpdir()), `robocasa-env-${task}-${split}-s${seed}-${port}.log`),
					}),
			rldxClient.ready().then(() => rldxClient.call("session.register", {}, 30_000)),
		]);
		const meta = await env.call<Record<string, unknown>>("env.get_env_meta", {}, 30_000);
		if (meta.task_name !== task || meta.split !== split || Number(meta.seed) !== Number(seed))
			throw new Error(`env server runs ${JSON.stringify(meta)}, not ${task}/${split}/s${seed}`);
		// The env resets on client connect and again in the primitives; a seed's scene is the second one.
		await env.call("env.reset", {}, 120_000);
		obs = await env.call<Raw>("env.reset", {}, 120_000);
		language = (await env.call<string | null>("env.get_task_language")) ?? "";
		criteria = await env
			.call<string>("env.get_success_criteria_text", {}, 30_000)
			.catch((e) => `(unavailable: ${e})`);
		await capture(null, null, null);
		return [...PRIMITIVES, "view_env_state", "back_project_batch", "query_world_map", "finish"];
	}
}
