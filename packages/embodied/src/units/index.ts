/**
 * Show-Harness action units for any robot: the model drives the arm one semantic unit at a time.
 *
 *   pi -e packages/embodied/src/libero --units=true --suite libero_object_swap --task 0 --seed 1
 *   pi -e packages/embodied/src/libero --units=true --stateless ...   (the paper's no-history setting)
 *   pi -e packages/embodied/src/libero --units=both ...              (units next to the robot's tools)
 *
 * A robot opts in with `units` in its defineRobot spec (base-frame vectors per MV_* unit, the step,
 * an optional yaw step, `apply` and `state`); ../robot.ts mounts this module. `--units=true` hides
 * the robot's own tools: only `act`, `finish` and the enabled plugins' `point` / `plan` remain
 * (Show-Harness's pure mode) and the system prompt is ./SYSTEM.md. `--units=both` adds them to the
 * robot's tools and appends the units section to the robot's prompt. `--stateless` keeps only the
 * first user message and the latest observation turn in context. `act`'s parameters follow the
 * enabled plugins. The episode state (gripper, accumulated yaw, plan, move history) is recorded as a
 * `units_state` session entry whenever it changes, for analysis. It is not rebuilt: every session
 * start resets the robot's scene (../robot.ts), so a resumed or forked session is a new episode and
 * starts from a fresh units state, as ../operator.ts does.
 *
 * Plugins (`--units-plugins`, default `auto`: the robot's `plugins` or Show-Harness's zero-shot Franka set):
 * - recovery: reopen after a GRASP that closed on nothing.
 * - auto_release: reopen a closed gripper whose object slipped out.
 * - proprioception: height, width and blocked moves in every result.
 * - variable_step: a coarse step for MV_UP, high above the table, or while the target is not in
 *   the wrist view (`act`'s `target_in_wrist`, Show-Harness's `WRIST: YES/NO` marker).
 * - action_chunk: while the target is not in the wrist view, `act` may commit `plan`, up to 3 MV_*
 *   moves run in order.
 * - rotation (robots with a yaw step, which get ROTATE_CW/CCW unless --units-rt, capped at 150 deg
 *   accumulated yaw): wrist-judged MV_* are rotated by the accumulated yaw (the wrist camera turns with
 *   the gripper), and MV_UP while holding first turns back (in commands within the robot's `maxYawRad`).
 * - `--units-rt` (not a plugin, off by default) switches the turn vocabulary from ROTATE_CW/CCW (v3)
 *   to RT_ROLL_*, RT_PITCH_*, RT_YAW_* (v5's 15 units): ROTATE_* are then neither offered nor run.
 *   An RT_* unit turns the gripper a fixed step about a base-frame axis through the TCP (robots with
 *   `rt`; a robot without an axis refuses its units), in commands within `rt.maxRad`, capped at
 *   150 deg accumulated yaw (the accumulator ROTATE_* use) and 90 deg accumulated roll or pitch.
 * - plan: subgoal stages, with deepplan's REASON checkpoint.
 * - point: affordance pixels -> world xyz (robots with `point`).
 * - mem_text: "Recent moves, newest first" in every result, with the history rules (no oscillation,
 *   no GRASP in place after an empty one); off, the results carry no move history at all.
 * A robot configuration without a wrist view (the spec's `wrist`, e.g. ManiSkill's --robot widowxai)
 * runs `auto` without variable_step and action_chunk and refuses to start when --units-plugins names
 * them (as --units-rt without an axis), and rotation keeps only its realign: those key on the wrist view. `act` then takes no `target_in_wrist` (one sent anyway is
 * ignored, and the result says so), the prompt drops its wrist-view text, a notice names the plugins
 * turned off, and the effective plugins are in every `units_state` entry and the robot result.
 *
 * Side VLM calls (./vlm.ts, `--units-vlm-model`, default the session's model):
 * - `--units-verify=true|false|auto` (auto: on for dual-arm robots, as Show-Harness's dual runner): a `finish`
 *   claiming success first lifts every open gripper 10 cm (MV_UP through `act`, `Move.retreat`), then
 *   is checked once against the task on the retreated camera images; a NOT complete
 *   verdict refuses it with the verifier's reason and the agent replans (at most once per episode).
 *   A check whose VLM call fails twice (no credits, network) refuses the finish as unverifiable
 *   (not the replan), at most twice per episode; after that the finish ends the episode unverified
 *   (`finish_verified: false` and `verifier_error` in the robot result).
 *   Every check is a `units_verify` session entry.
 * - `--units-video-ref <mp4>`: `--units-video-ref-frames` frames sampled uniformly are distilled
 *   into an ordered demo brief (arm, grasp part, destination) that the prompt tells the agent to
 *   replicate; the brief is a `units_video_ref` session entry. A failed extraction fails closed.
 *
 * Files: this one registers the flags and tools and owns the episode state; ./types.ts is the public
 * contract (UnitsSpec, UnitsHandle, entry names), ./vocabulary.ts the units and pure helpers (ground,
 * compensate, finishMove, latestTurn), ./prompt.ts renders ./SYSTEM.md, ./verifier.ts the finish check.
 *
 * Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: core/action_units.py, interpreters/ (unit -> base-frame motion) and
 * plugins/{recovery,auto_release,proprioception,variable_step,action_chunk,rotation,affordance,
 * subgoal,deepplan,mem_text} ported as pi tools; the final task check and plugins/video_ref in ./vlm.ts.
 */

import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type TSchema, Type } from "typebox";
import { renderPrompt } from "./prompt.ts";
import {
	type Result,
	STATE_ENTRY,
	type Stage,
	type Target,
	type ToolRegistrar,
	UNITS_EVENT,
	type UnitsHandle,
	type UnitsSpec,
	VIDEO_REF_ENTRY,
} from "./types.ts";
import { registerVerifier, verifierState } from "./verifier.ts";
import {
	askVlm,
	type DemoBrief,
	parseJson,
	sampleFrames,
	VIDEO_REF_FRAMES,
	VLM_COST_EVENT,
	validateBrief,
	videoRefPrompt,
} from "./vlm.ts";
import {
	CHUNK_STEPS,
	compensate,
	DEFAULT_PLUGINS,
	finishMove,
	GUIDE_VIEWS,
	type GuideView,
	ground,
	isMove,
	isRt,
	latestTurn,
	MAX_REPEAT,
	MOVE_UNITS,
	type Move,
	type MoveUnit,
	PLUGINS,
	type Plugin,
	ROTATE_UNITS,
	RT_TURNS,
	type RtUnit,
	type State,
	UNITS,
	type Unit,
	type Vec3,
} from "./vocabulary.ts";

export { DEFAULT_VIEWS, DEFAULT_VIEWS_NO_WRIST } from "./prompt.ts";
export {
	STATE_ENTRY,
	type ToolRegistrar,
	UNITS_EVENT,
	type UnitsHandle,
	type UnitsSpec,
	VERIFY_ENTRY,
	VIDEO_REF_ENTRY,
} from "./types.ts";
export {
	compensate,
	DEFAULT_PLUGINS,
	finishMove,
	GUIDE_VIEWS,
	type GuideView,
	ground,
	isRt,
	latestTurn,
	MOVE_UNITS,
	type Move,
	type MoveUnit,
	PLUGINS,
	type Plugin,
	ROTATE_UNITS,
	RT_TURNS,
	RT_UNITS,
	type RtAxis,
	type RtUnit,
	type State,
	UNITS,
	type Unit,
	type Vec3,
} from "./vocabulary.ts";

/** A MV_* that travelled less than this fraction of the command did not move freely (proprioception). */
const STALL_RATIO = 0.7;
/** rotation: soft guard on the accumulated yaw (Franka joint 7 is about +-166 deg), and "back at neutral". */
const MAX_YAW = (150 * Math.PI) / 180;
const NEUTRAL_YAW = (2 * Math.PI) / 180;
/** RT_*: the accumulated roll or pitch guard (90 deg: the gripper pointing sideways). */
const MAX_TILT = Math.PI / 2;
/** The most commands one turn is split into (the 150 deg cap in 5 deg commands). */
const MAX_YAW_PIECES = 30;
/** mem_text: moves the "Recent moves" line shows (configs/robot_franka.yaml mem_text_len). */
const MEM_LEN = 5;
/** mem_text: the history entry of a GRASP that closed on nothing (core/runners/real.py EMPTY_GRASP_LABEL). */
const EMPTY_GRASP = "GRASP(empty)";
/** Plan stages that make a task a placement (Show-Harness subgoal: PLACE -> RELEASE -> RETREAT). */
const PLACEMENT = ["PLACE", "RELEASE", "RETREAT"];
const NOTES = {
	empty_grasp: "Empty close; do not retry on an edge/corner. Recenter body and confirm depth.",
	lost_grasp: "Grasp lost; return to GRASP, recenter the object body, then confirm depth.",
};

/** Plugins that key on the wrist view (`target_in_wrist`): off on a robot configuration without one. */
const WRIST_PLUGINS: readonly Plugin[] = ["variable_step", "action_chunk"];

const r3 = (v: number) => Number(v.toFixed(3));
const text = (s: string) => ({ type: "text" as const, text: s });

/** Register the units flags and tools; `mode()`, `tools()` and `prompt()` are read by ../robot.ts. */
export function units(
	pi: ExtensionAPI,
	spec: UnitsSpec,
	tool: ToolRegistrar,
	task: () => Record<string, string> = () => ({}),
	base: Pick<UnitsHandle, "tools" | "refuse" | "views"> = { tools: () => ["act"], refuse: () => undefined },
) {
	const instruction = () =>
		spec.instruction?.() ??
		Object.entries(task())
			.map(([k, v]) => `${k} ${v}`)
			.join(", ");
	pi.registerFlag("units", {
		type: "string",
		default: "false",
		description: "Show-Harness action units: true = only act/finish (+ point/plan), both = next to the robot's tools",
	});
	pi.registerFlag("stateless", {
		type: "boolean",
		default: false,
		description: "Units mode: keep only the task and the latest observation turn in context",
	});
	pi.registerFlag("units-plugins", {
		type: "string",
		default: "auto",
		description: `Units plugins, comma-separated (${PLUGINS.join(", ")}; "" = none; auto = ${(spec.plugins ?? DEFAULT_PLUGINS).join(",")}, without variable_step and action_chunk on a configuration without a wrist view)`,
	});
	pi.registerFlag("units-coarse-step", {
		type: "string",
		default: String(spec.coarseStepM ?? 0.04),
		description: "variable_step: the coarse MV_* step, m",
	});
	pi.registerFlag("units-rt", {
		type: "string",
		default: "false",
		description: "Offer the RT_* roll/pitch/yaw units (the aaroncaozj LIBERO adapters) on robots that can turn",
	});
	// A string flag: pi sets a boolean flag to true whatever its value, so a default-on check could not be turned off.
	pi.registerFlag("units-verify", {
		type: "string",
		default: "auto",
		description:
			"Units mode: check a success `finish` with one VLM call on the latest images, NOT complete refuses it once (true, false, auto = dual-arm robots)",
	});
	pi.registerFlag("units-vlm-model", {
		type: "string",
		default: "",
		description: "Model (provider/id) of the verifier and video_ref calls (default: the session's model)",
	});
	pi.registerFlag("units-video-ref", {
		type: "string",
		default: "",
		description: "Units mode: a demo video (mp4) distilled into an ordered brief the agent replicates",
	});
	pi.registerFlag("units-video-ref-frames", {
		type: "string",
		default: String(VIDEO_REF_FRAMES),
		description: "video_ref: frames sampled uniformly from the demo video",
	});
	/** "pure" (--units / --units=true), "both", or undefined (off). */
	const mode = (): "pure" | "both" | undefined => {
		const v = pi.getFlag("units");
		if (v === true || v === "true" || v === "pure") return "pure";
		return v === "both" ? "both" : undefined;
	};
	/** The robot configuration has a wrist view (spec.wrist), read at load, session start and robot start. */
	const readWrist = () => (typeof spec.wrist === "function" ? spec.wrist() : spec.wrist) !== false;
	let wristView = readWrist();
	/** A plugin --units-plugins asks for and the robot can run, before the wrist view is considered. */
	/** --units-plugins as given, or undefined for `auto` (the robot's default set). */
	const listed = () => {
		const v = String(pi.getFlag("units-plugins") ?? "auto").trim();
		return v === "auto"
			? undefined
			: v
					.split(",")
					.map((s) => s.trim())
					.filter(Boolean);
	};
	const requested = (name: Plugin) =>
		(listed() ?? spec.plugins ?? DEFAULT_PLUGINS).includes(name) &&
		(name !== "point" || spec.point !== undefined) &&
		(name !== "rotation" || Boolean(spec.yawStepRad));
	const plugin = (name: Plugin) => requested(name) && (wristView || !WRIST_PLUGINS.includes(name));
	/** The plugins that run this session (`units_state`, the robot result). */
	const effective = () => PLUGINS.filter(plugin);
	const armNames = spec.arms ?? [];
	/** --units-verify: on, off, or auto (Show-Harness's dual runner verifies, its single-arm runners do not). */
	const verifying = () => {
		const v = String(pi.getFlag("units-verify") ?? "auto");
		return v === "auto" ? armNames.length > 1 : v === "true";
	};
	const coarse = () => Number(pi.getFlag("units-coarse-step")) || spec.stepM;
	const high = spec.highAboveTableM ?? 0.08;
	/** `act` takes `target_in_wrist`: a wrist view and a plugin that reads it. */
	const wristSignal = () => wristView && (plugin("variable_step") || plugin("action_chunk") || plugin("rotation"));
	const rtOn = () => String(pi.getFlag("units-rt") ?? "false") === "true";
	/** Why `act` refuses an RT_* unit, else undefined. */
	const rtRefusal = (u: RtUnit) => {
		if (!rtOn()) return `${u}: the RT_* units are off (--units-rt=true)`;
		const { axis } = RT_TURNS[u];
		return spec.rt?.axes[axis] ? undefined : `${u}: this robot cannot turn its gripper about the ${axis} axis`;
	};
	const isRotate = (u: string) => (ROTATE_UNITS as readonly string[]).includes(u);
	/** --units-rt swaps the turn vocabulary: ROTATE_* (v3) or RT_* (v5), never both. */
	const rotateRefusal = (u: string) =>
		rtOn() && isRotate(u)
			? `${u}: --units-rt replaces ROTATE_* with the RT_* turns (RT_YAW_* about the vertical)`
			: undefined;
	const vocab = () =>
		UNITS.filter(
			(u) =>
				(spec.yawStepRad || !isRotate(u)) &&
				!rotateRefusal(u) &&
				(u !== "STILL" || armNames.length) &&
				(!isRt(u) || !rtRefusal(u)),
		);

	/** Per-arm episode state ("" = the single arm). */
	let closed = new Map<string, boolean>();
	let yaw = new Map<string, number>();
	/** RT_*: accumulated roll and pitch (rad, signed as RT_ROLL_LEFT / RT_PITCH_FWD); RT_YAW_* add to `yaw`. */
	let roll = new Map<string, number>();
	let pitch = new Map<string, number>();
	let recent: string[] = [];
	let note = "";
	let stages: Stage[] = [];
	let stage = 0;
	let targets: Target[] = [];
	/** The verifier's refusals, verdict, errors and view (./verifier.ts updates it in place). */
	const check = verifierState();
	/** video_ref: the brief (extracted once per video and frame count) and why it failed. */
	let demo: { key: string; brief: DemoBrief; indices: number[]; model: string } | undefined;
	let demoError: string | undefined;
	/** A new episode or scene reset: the arm is back at its start heading with the gripper open, no plan. */
	const reset = () => {
		closed = new Map();
		yaw = new Map();
		roll = new Map();
		pitch = new Map();
		recent = [];
		note = "";
		stages = [];
		stage = 0;
		targets = [];
		Object.assign(check, verifierState());
	};
	/** mem_text: record a unit in the move history (newest last). */
	const remember = (u: string) => {
		recent = [...recent, u].slice(-MEM_LEN);
	};
	/** The episode state as recorded in `units_state` entries (not the verifier's images). */
	const snapshot = () =>
		JSON.stringify({
			plugins: effective(),
			wrist: wristView,
			closed: Object.fromEntries(closed),
			yaw: Object.fromEntries(yaw),
			...(roll.size ? { roll: Object.fromEntries(roll) } : {}),
			...(pitch.size ? { pitch: Object.fromEntries(pitch) } : {}),
			recent,
			note,
			stages,
			stage,
			targets,
			replans: check.replans,
			verdict: check.verdict,
			holdRefusals: check.holdRefusals,
			verifierErrors: check.verifierErrors,
			verifierError: check.verifierError,
			finishVerified: check.finishVerified,
		});
	let saved = snapshot();
	/** Append the state entry when it changed (a record of the episode; a new session starts fresh). */
	const save = () => {
		const now = snapshot();
		if (now === saved) return;
		saved = now;
		pi.appendEntry(STATE_ENTRY, JSON.parse(now));
	};
	pi.on("session_start", (_event, ctx) => {
		// A session start resets the robot's scene: a new episode, so the units state starts fresh
		// (an earlier `units_state` in the branch belongs to an episode whose scene is gone).
		reset();
		wristView = readWrist();
		demoError = undefined;
		const branch = ctx.sessionManager.getBranch();
		const custom = (type: string) =>
			branch
				.filter((e) => e.type === "custom" && e.customType === type)
				.map((e) => (e.type === "custom" ? (e.data as Record<string, unknown>) : {}));
		saved = snapshot();
		// A brief extracted earlier in this branch is reused (same video and frame count).
		const brief = custom(VIDEO_REF_ENTRY)
			.filter((e) => e.brief)
			.pop();
		if (brief)
			demo = {
				key: `${brief.video_path}#${brief.num_frames}`,
				brief: brief.brief as DemoBrief,
				indices: (brief.sampled_indices as number[]) ?? [],
				model: String(brief.model ?? ""),
			};
	});

	/** Proprioception, or undefined when the robot has none or cannot read it now. */
	const read = async (arm: string | undefined) => spec.state?.(arm).catch(() => undefined);
	const width = (s: State | undefined) => (typeof s?.gripper_width === "number" ? s.gripper_width : undefined);
	const eef = (s: State | undefined) =>
		Array.isArray(s?.eef_xyz) && s.eef_xyz.length >= 3 ? (s.eef_xyz as number[]).map(Number) : undefined;
	const gap = (s: State | undefined) => {
		const p = eef(s);
		return p && typeof s?.table_z === "number" ? p[2] - s.table_z : undefined;
	};
	const empty = (s: State | undefined) => {
		const w = width(s);
		return spec.emptyWidthM !== undefined && w !== undefined && w <= spec.emptyWidthM;
	};
	const reopen = async (arm: string | undefined, signal: AbortSignal | undefined) => {
		closed.set(arm ?? "", false);
		return spec.apply({ delta: [0, 0, 0], yaw: 0, gripper: "open", ...(arm ? { arm } : {}) }, signal);
	};

	/** variable_step: coarse for MV_UP, high above the table, or the target not in the wrist view. */
	const stepFor = (unit: MoveUnit, st: State | undefined, inWrist: boolean | undefined) => {
		if (!plugin("variable_step")) return spec.stepM;
		const g = gap(st);
		return unit === "MV_UP" || (g !== undefined && g > high) || inWrist === false ? coarse() : spec.stepM;
	};

	/** The units block that heads every `act` result. */
	async function header(lines: string[], arm: string | undefined) {
		const out = [...lines];
		if (plugin("plan") && stages.length) {
			const s = stages[stage];
			out.push(
				s
					? `STAGE ${stage + 1}/${stages.length} [${s.motion}]${s.arm ? ` (${s.arm} arm)` : ""}: target ${s.target}${s.affordance ? `; affordance ${s.affordance}` : ""}${s.description ? `; ${s.description}` : ""}; DONE WHEN ${s.completion}`
					: "STAGE: all planned stages are done; check the task and DONE, or send a new plan.",
			);
			if (s?.motion.toUpperCase() === "REASON")
				out.push("Decision point: judge the rule from the images now and send the concrete stages with `plan`.");
		}
		const st = await read(arm);
		if (plugin("proprioception") && st) {
			const p = eef(st);
			const g = gap(st);
			const w = width(st);
			const held = closed.get(arm ?? "") === true;
			const parts: string[] = [];
			if (g !== undefined) parts.push(`the gripper is ${(g * 100).toFixed(1)} cm above the table`);
			else if (p) parts.push(`gripper at [${p.map(r3).join(", ")}] m`);
			if (w !== undefined) parts.push(`width ${(w * 100).toFixed(1)} cm, commanded ${held ? "CLOSE" : "OPEN"}`);
			parts.push(
				plugin("variable_step")
					? `each step moves ~${(spec.stepM * 100).toFixed(0)} cm (${(coarse() * 100).toFixed(0)} cm)`
					: `each step moves ~${(spec.stepM * 100).toFixed(0)} cm`,
			);
			out.push(`Proprioception: ${parts.join("; ")}.`);
			if (g !== undefined)
				out.push(
					held
						? "Holding an object: lift until clear of the table; descend only to place."
						: `If height > ${(high * 100).toFixed(0)} cm, MV_DOWN first.`,
				);
		}
		if (plugin("rotation") && Math.abs(yaw.get(arm ?? "") ?? 0) > NEUTRAL_YAW)
			out.push(
				`Gripper turned ${Math.round(((yaw.get(arm ?? "") ?? 0) * 180) / Math.PI)} deg from its start heading.`,
			);
		const [tr, tp] = [roll.get(arm ?? "") ?? 0, pitch.get(arm ?? "") ?? 0];
		if (rtOn() && (Math.abs(tr) > NEUTRAL_YAW || Math.abs(tp) > NEUTRAL_YAW))
			out.push(
				`Gripper tilted: roll ${Math.round((tr * 180) / Math.PI)} deg (+ = RT_ROLL_LEFT), pitch ${Math.round((tp * 180) / Math.PI)} deg (+ = RT_PITCH_FWD).`,
			);
		if (plugin("point") && targets.length) {
			const p = eef(st);
			for (const t of targets)
				out.push(
					`Point ${t.label} (${t.camera} [${t.point}]): ${t.xyz ? `xyz [${t.xyz.map(r3).join(", ")}] m${p ? `, offset from gripper [${t.xyz.map((v, i) => r3(v - p[i])).join(", ")}] m` : ""}` : "no depth"}`,
				);
		}
		if (note) out.push(`Recovery: ${note}`);
		if (check.verdict) out.push(`Verifier: the task is NOT complete: ${check.verdict}`);
		if (plugin("mem_text")) out.push(`Recent moves, newest first: ${[...recent].reverse().join(", ") || "none"}`);
		out.push(`TASK: ${instruction()}`);
		return text(out.join("\n"));
	}

	/** `act`'s parameters for the enabled plugins (a disabled plugin's parameter is not offered). */
	function actSchema() {
		const props: Record<string, TSchema> = {
			unit: StringEnum(vocab(), { description: "The action unit" }),
			n: Type.Optional(Type.Integer({ minimum: 1, maximum: MAX_REPEAT, description: "Repeat count (default 1)" })),
		};
		if (wristSignal())
			props.target_in_wrist = Type.Optional(
				Type.Boolean({ description: "WRIST CHECK: is the TARGET visible in the wrist view?" }),
			);
		if (plugin("action_chunk"))
			props.plan = Type.Optional(
				Type.Array(StringEnum(MOVE_UNITS), {
					maxItems: CHUNK_STEPS,
					description: `Only with target_in_wrist false: up to ${CHUNK_STEPS} MV_* moves run in order (replaces unit and n)`,
				}),
			);
		if (armNames.length) props.arm = StringEnum(armNames, { description: "Which arm the unit drives" });
		if (spec.viewSelect?.())
			props.view = Type.Optional(
				StringEnum(GUIDE_VIEWS, {
					description:
						"VIEW SELECT: the view that guided this move, WRIST (the wrist view) or FRONT (the front view)",
				}),
			);
		return Type.Object(props);
	}
	let registered = "";
	/** (Re-)register `act` when its schema changed: at load from the flag defaults, at session start from the flags. */
	function registerAct() {
		const schema = actSchema();
		if (JSON.stringify(schema) === registered) return;
		registered = JSON.stringify(schema);
		tool(
			"act",
			`Execute one action unit (${vocab().join(", ")}), repeated n times. MV_* move the gripper ~${Math.round(spec.stepM * 100)} cm${spec.yawStepRad && !rtOn() ? `, ROTATE_* turn it ~${Math.round((spec.yawStepRad * 180) / Math.PI)} deg` : ""}${rtOn() && spec.rt ? `, RT_* turn it ~${Math.round((spec.rt.stepRad * 180) / Math.PI)} deg about a world axis through the fingertips (ROLL about the MV_FWD axis, PITCH about the MV_LEFT-MV_RIGHT axis, YAW about the vertical)` : ""}; GRASP closes, RELEASE opens, STOP holds one step, DONE means the task is complete (call finish). Returns the new images and state.`,
			schema,
			(params, signal) => actAndSave(params as ActParams, signal),
		);
	}
	registerAct();
	pi.on("session_start", registerAct);

	type ActParams = {
		unit: string;
		n?: number;
		arm?: string;
		target_in_wrist?: boolean;
		plan?: string[];
		operator?: boolean;
		view?: string;
		retreat?: boolean;
	};
	/** `act`, then the state entry (the agent's and the operator's units both change the episode state). */
	async function actAndSave(params: ActParams, signal: AbortSignal | undefined) {
		try {
			return await act(params, signal);
		} finally {
			save();
		}
	}
	/** `act`'s body; ../gumi (dashboard teleop, DAgger takeover) runs it too, through the handle below. */
	async function act(params: ActParams, signal: AbortSignal | undefined): Promise<Result> {
		const p = params as {
			unit: Unit;
			n?: number;
			arm?: string;
			target_in_wrist?: boolean;
			plan?: MoveUnit[];
			operator?: boolean;
			view?: GuideView;
			retreat?: boolean;
		};
		const { unit, arm } = p;
		// No wrist view: a `target_in_wrist` sent anyway (a model trained on wrist robots) is ignored, not refused.
		const inWrist = wristView ? p.target_in_wrist : undefined;
		// A human's unit (GUMI teleop) runs as pressed: the agent-side assists would override the operator
		// (recovery reopens a GRASP that closed on air), as in Show-Harness's collectors.
		const assist = (name: Plugin) => !p.operator && plugin(name);
		const key = arm ?? "";
		// The schema offers these only with their plugins; a stale or hand-written call is refused, not silently dropped.
		if (p.plan !== undefined && !plugin("action_chunk"))
			throw new Error(
				wristView
					? "act: `plan` needs the action_chunk plugin (--units-plugins)"
					: "act: `plan` needs the action_chunk plugin, which is off: this robot has no wrist view",
			);
		if (p.view !== undefined && !(spec.viewSelect?.() && (GUIDE_VIEWS as readonly string[]).includes(p.view)))
			throw new Error("act: `view` needs the robot's view select (WRIST or FRONT)");
		if (p.target_in_wrist !== undefined && wristView && !wristSignal())
			throw new Error(
				"act: `target_in_wrist` needs the variable_step, action_chunk or rotation plugin (--units-plugins)",
			);
		if (demoError) return { content: [text(`video_ref failed: ${demoError}`)], details: { unit, error: demoError } };
		if (unit === "DONE")
			return {
				content: [text("DONE: if the images show the task complete, call `finish` now; otherwise keep acting.")],
				details: { unit },
			};
		if (unit === "STILL") return { content: [text(`STILL: the ${arm ?? ""} arm holds.`)], details: { unit } };
		const rtWhy = isRt(unit) ? rtRefusal(unit) : rotateRefusal(unit);
		if (rtWhy) throw new Error(`act: ${rtWhy}`);
		const lines: string[] = [];
		if (!wristView && p.target_in_wrist !== undefined && !p.operator)
			lines.push("target_in_wrist ignored: this robot has no wrist view.");
		let queue: Unit[] = Array(Math.max(1, Math.min(MAX_REPEAT, Math.floor(p.n ?? 1)))).fill(unit);
		if (p.plan?.length && plugin("action_chunk")) {
			if (inWrist === false) queue = p.plan.filter(isMove).slice(0, CHUNK_STEPS);
			else lines.push("plan ignored: plans run only while target_in_wrist is false; one unit ran.");
			if (inWrist !== false) queue = [unit];
		}
		let last: Result | undefined;
		const ran: string[] = [];
		const halted = (r: Result, m?: Move) => {
			const d = r.details as { error?: unknown; terminated?: unknown } | undefined;
			// The finish sequence runs past the robot's success signal while the robot still moves (returns images).
			const moved = m !== undefined && finishMove(m) && r.content.some((c) => c.type === "image");
			return Boolean(d?.error || (d?.terminated && !moved));
		};
		/** Path length so far: one call travels at most the robot's per-call translation limit. */
		let travelled = 0;
		// A recovery note lasts until the next GRASP.
		if (queue.includes("GRASP")) note = "";
		for (const [index, u] of queue.entries()) {
			const before = await read(arm);
			const acc = yaw.get(key) ?? 0;
			let label: string = u;
			let move = ground(spec, u, isMove(u) && !p.operator ? stepFor(u, before, inWrist) : spec.stepM) as Move;
			// Holding and turned: MV_UP first turns back to the start heading.
			if (assist("rotation") && u === "MV_UP" && closed.get(key) && Math.abs(acc) > NEUTRAL_YAW) {
				move = { delta: [0, 0, 0], yaw: -acc, gripper: null };
				label = "MV_UP(realign)";
				// Wrist-judged moves follow the turned wrist camera; without one every move is judged in the base frame.
			} else if (assist("rotation") && wristView && isMove(u) && inWrist !== false)
				move.delta = compensate(move.delta, (spec.yawCompensationSign ?? 1) * acc);
			else if (move.yaw && Math.abs(acc + move.yaw) > MAX_YAW) {
				// The accumulated-yaw guard holds with or without the rotation plugin.
				lines.push(`${u} refused: the gripper is already turned ${Math.round((acc * 180) / Math.PI)} deg.`);
				break;
			}
			// RT_*: the accumulated guards (yaw shared with ROTATE_*, measured about base +z; roll, pitch by unit).
			const turn = isRt(u) && move.rot ? RT_TURNS[u] : undefined;
			const tilts = turn?.axis === "roll" ? roll : pitch;
			const turnBy =
				turn && move.rot ? (turn.axis === "yaw" ? move.rot[2] : turn.sign * Math.hypot(...move.rot)) : 0;
			if (turn) {
				const now = turn.axis === "yaw" ? acc : (tilts.get(key) ?? 0);
				if (Math.abs(now + turnBy) > (turn.axis === "yaw" ? MAX_YAW : MAX_TILT) + 1e-9) {
					lines.push(
						`${u} refused: the gripper is already turned ${Math.round((now * 180) / Math.PI)} deg about its ${turn.axis} axis.`,
					);
					break;
				}
			}
			const dist = Math.hypot(...move.delta);
			const maxMove = spec.maxMoveM?.();
			if (maxMove !== undefined && travelled > 0 && !(travelled + dist <= maxMove + 1e-9)) {
				lines.push(`${u} not run: one act call moves at most ${maxMove} m in total.`);
				break;
			}
			// A turn beyond the robot's per-command limit runs as equal commands within it.
			const maxYaw = spec.maxYawRad?.();
			const maxRot = spec.rt?.maxRad?.();
			const angle = move.rot ? Math.hypot(...move.rot) : 0;
			const parts =
				move.yaw && maxYaw !== undefined
					? Math.ceil(Math.abs(move.yaw) / maxYaw - 1e-9)
					: angle && maxRot !== undefined
						? Math.ceil(angle / maxRot - 1e-9)
						: 1;
			if (!(parts >= 1 && parts <= MAX_YAW_PIECES)) {
				lines.push(`${label} refused: the robot's per-call rotation limit is ${move.rot ? maxRot : maxYaw} rad.`);
				break;
			}
			if (arm) move.arm = arm;
			if (p.view) move.view = p.view;
			if (p.retreat) move.retreat = true;
			if (isMove(u) && label === u && queue[index + 1] === u) move.continuous = true;
			for (let i = 0; i < parts; i++) {
				const piece: Move = i ? { ...move, delta: [0, 0, 0], gripper: null } : move;
				const rot = move.rot?.map((x) => x / parts) as Vec3 | undefined;
				last = await spec.apply(
					parts > 1 ? { ...piece, yaw: move.yaw / parts, ...(rot ? { rot } : {}) } : piece,
					signal,
				);
				if (halted(last, move)) break;
				if (move.yaw) yaw.set(key, (yaw.get(key) ?? 0) + move.yaw / parts);
				if (turn?.axis === "yaw") yaw.set(key, (yaw.get(key) ?? 0) + turnBy / parts);
				else if (turn) tilts.set(key, (tilts.get(key) ?? 0) + turnBy / parts);
			}
			travelled += dist;
			ran.push(label);
			// mem_text records moves, turns and grasps (not STOP / RELEASE), as core/runners/real.py.
			if (isMove(u) || isRt(u) || u === "GRASP" || u === "ROTATE_CW" || u === "ROTATE_CCW") remember(u);
			if (move.gripper) closed.set(key, move.gripper === "close");
			const after = await read(arm);
			if (last && halted(last, move)) break;
			// proprioception: a MV_* that barely moved is blocked (contact, floor, workspace limit).
			const [p0, p1] = [eef(before), eef(after)];
			const base = spec.baseDelta?.(move.delta, before) ?? move.delta;
			const commanded = Math.hypot(...base);
			// A chained move returned before the arm settled: its lagging position is not a stall.
			const settling = move.continuous === true && spec.chains?.() === true;
			if (plugin("proprioception") && !settling && p0 && p1 && commanded > 0) {
				const moved = [0, 1, 2].reduce((s, k) => s + (p1[k] - p0[k]) * base[k], 0) / commanded;
				if (moved < commanded * STALL_RATIO) {
					lines.push(
						u === "MV_DOWN"
							? `Last MV_DOWN lowered ${(moved * 100).toFixed(1)} of ${(commanded * 100).toFixed(1)} cm -> already in contact, do NOT MV_DOWN again`
							: `Last ${u} moved ${(moved * 100).toFixed(1)} of ${(commanded * 100).toFixed(1)} cm -> blocked`,
					);
					break;
				}
			}
			// recovery: a GRASP that closed on nothing is reopened at once.
			if (u === "GRASP" && assist("recovery") && empty(after)) {
				last = await reopen(arm, signal);
				// The fresh approach starts from a clean history holding only the failed GRASP.
				recent = [EMPTY_GRASP];
				note = NOTES.empty_grasp;
				break;
			}
			// auto_release: a closed gripper that collapsed (the object slipped out) is reopened.
			if (u !== "GRASP" && assist("auto_release") && closed.get(key) && empty(after)) {
				last = await reopen(arm, signal);
				note = NOTES.lost_grasp;
				break;
			}
		}
		const what =
			queue.every((u) => u === queue[0]) && ran.every((u) => u === ran[0])
				? `${ran[0] ?? queue[0]} x${ran.length}`
				: ran.join(", ");
		lines.unshift(
			`units: ${what}${arm ? ` (${arm} arm)` : ""}${ran.length < queue.length ? ` of ${queue.length} (stopped early)` : ""}`,
		);
		if (!last) return { content: [await header(lines, arm)], details: { unit } };
		return { ...last, content: [await header(lines, arm), ...last.content] };
	}
	// The robot's unit layer for ../gumi, published every session (the dashboard operator drives through it).
	const handle: UnitsHandle = {
		tool: "act",
		arms: armNames,
		vocabulary: vocab(),
		stepM: spec.stepM,
		yawStepRad: spec.yawStepRad,
		run: actAndSave,
		state: spec.state,
		viewSelect: false,
		wrist: () => wristView,
		plugins: effective,
		...base,
	};
	pi.on("session_start", () =>
		pi.events.emit(UNITS_EVENT, {
			...handle,
			vocabulary: vocab(),
			viewSelect: spec.viewSelect?.() === true,
			rt: rtOn(),
		}),
	);

	if (spec.point) {
		const { cameras, locate } = spec.point;
		tool(
			"point",
			"Affordance: mark the exact gripper contact point(s) in one camera's current image, [y, x] on a 0-1000 grid (y from the top, x from the left). Returns the marked image and the world xyz per point where the robot has depth; later act results report the gripper-to-point offset. A new call with the same label replaces that point.",
			Type.Object({
				camera: StringEnum(cameras, { description: `Camera (${cameras.join(", ")})` }),
				points: Type.Array(
					Type.Object({
						label: Type.String({ description: "What the point is, e.g. 'bowl rim' or 'place spot'" }),
						yx: Type.Array(Type.Number({ minimum: 0, maximum: 1000 }), { minItems: 2, maxItems: 2 }),
					}),
					{ minItems: 1, maxItems: 4 },
				),
			}),
			async ({ camera, points }, signal) => {
				const fr = points.map((q) => [q.yx[0] / 1000, q.yx[1] / 1000] as [number, number]);
				const { image, xyz } = await locate(camera, fr, signal);
				const marked = points.map((q, i) => ({
					label: q.label,
					camera,
					point: [Math.round(q.yx[0]), Math.round(q.yx[1])] as [number, number],
					xyz: xyz[i] ? xyz[i].map(r3) : null,
				}));
				targets = [...targets.filter((t) => !marked.some((m) => m.label === t.label)), ...marked].slice(-4);
				save();
				const content: Result["content"] = [text(JSON.stringify({ points: marked }))];
				if (image) content.push({ type: "image", data: image.toString("base64"), mimeType: "image/png" });
				return { content, details: { points: marked } };
			},
		);
	}

	tool(
		"plan",
		"Subgoal plan: `stages` replaces the plan from the current stage on (ordered GRASP/LIFT/MOVE/PLACE/RELEASE/RETREAT/REASON stages, each with a visible DONE WHEN); `done: true` marks the current stage complete. The current stage is shown in every act result.",
		Type.Object({
			stages: Type.Optional(
				Type.Array(
					Type.Object({
						motion: Type.String({ description: "GRASP, LIFT, MOVE, PLACE, RELEASE, RETREAT or REASON" }),
						target: Type.String(),
						affordance: Type.Optional(Type.String({ description: "The one visible part to aim at" })),
						description: Type.Optional(
							Type.String({ description: "Visual strategy; for REASON the complete IF ... THEN rule" }),
						),
						completion: Type.String({ description: "DONE WHEN: a condition visible in the images" }),
						arm: Type.Optional(Type.String({ description: "Dual-arm robots: the arm of this stage" })),
					}),
				),
			),
			done: Type.Optional(Type.Boolean({ description: "The current stage's DONE WHEN is visible" })),
		}),
		async ({ stages: next, done }) => {
			if (done && stage < stages.length) stage++;
			if (next?.length) stages = [...stages.slice(0, stage), ...next];
			// A new stage starts with a clean move history (core/runners/real.py); a new plan answers the verifier.
			if (done || next?.length) recent = [];
			if (next?.length) check.verdict = "";
			save();
			const lines = stages.map(
				(s, i) =>
					`${i === stage ? ">" : i < stage ? "x" : " "} ${i + 1}. [${s.motion}] ${s.target}: ${s.completion}`,
			);
			return { content: [text(lines.length ? lines.join("\n") : "No plan yet.")], details: { stage, stages } };
		},
	);

	// The verifier: the latest camera images and the finish check (./verifier.ts).
	registerVerifier({
		pi,
		check,
		arms: armNames,
		wrist: () => wristView,
		active: () => mode() !== undefined,
		verifying,
		instruction,
		isClosed: (arm) => closed.get(arm),
		placement: () => stages.some((s) => PLACEMENT.includes(s.motion.toUpperCase())),
		liftStep: () => (plugin("variable_step") ? coarse() : spec.stepM),
		act: actAndSave,
		replan: () => {
			stages = [];
			stage = 0;
			recent = [];
			note = "";
		},
		planning: () => plugin("plan"),
		save,
	});

	// video_ref: extract the demo brief once, before the first prompt that needs it (plugins/video_ref).
	pi.on("before_agent_start", async (_event, ctx) => {
		const path = String(pi.getFlag("units-video-ref") ?? "");
		if (!mode() || !path) return undefined;
		const frames = Number(pi.getFlag("units-video-ref-frames")) || VIDEO_REF_FRAMES;
		const key = `${path}#${frames}`;
		if (demo?.key === key || demoError) return undefined;
		try {
			const { indices, images: shown } = await sampleFrames(path, frames, String(pi.getFlag("ffmpeg") || "ffmpeg"));
			const prompt = videoRefPrompt(shown.length, armNames);
			const errors: string[] = [];
			// A reply that is not a usable brief is asked once more (their guided-JSON try, then free JSON).
			for (let i = 0; i < 2 && demo?.key !== key; i++) {
				const reply = await askVlm(
					ctx,
					String(pi.getFlag("units-vlm-model") ?? ""),
					pi.getThinkingLevel(),
					prompt,
					shown,
					ctx.signal,
				);
				pi.events.emit(VLM_COST_EVENT, reply.cost);
				try {
					demo = { key, brief: validateBrief(parseJson(reply.text), armNames), indices, model: reply.model };
				} catch (err) {
					errors.push(err instanceof Error ? err.message : String(err));
				}
			}
			if (!demo || demo.key !== key) throw new Error(errors.join(" | "));
			pi.appendEntry(VIDEO_REF_ENTRY, {
				video_path: path,
				num_frames: frames,
				sampled_indices: demo.indices,
				model: demo.model,
				brief: demo.brief,
			});
		} catch (err) {
			// A replication run without the brief would silently become ordinary planning: fail closed.
			demoError = `could not extract a demo brief from ${path}: ${err instanceof Error ? err.message : String(err)}`;
			pi.appendEntry(VIDEO_REF_ENTRY, { video_path: path, num_frames: frames, error: demoError });
			if (ctx.hasUI) ctx.ui.notify(`video_ref: ${demoError}`, "error");
			else {
				console.error(`[units] video_ref: ${demoError}`);
				process.exitCode = 1;
				ctx.shutdown();
			}
		}
		return undefined;
	});

	pi.on("context", (event) => {
		if (!mode() || pi.getFlag("stateless") !== true) return undefined;
		const kept = latestTurn(event.messages);
		return kept ? { messages: kept } : undefined;
	});

	/**
	 * The robot started (../robot.ts, after its `start`): its configuration is known now (the server's
	 * cameras), so the wrist view is read again, `act` re-registered for it, and a configuration
	 * without one says which requested plugins it turns off.
	 */
	function started(ctx: Pick<ExtensionContext, "hasUI" | "ui">) {
		wristView = readWrist();
		saved = snapshot();
		registerAct();
		if (wristView || !mode()) return;
		// A wrist-view plugin named explicitly cannot run here: fail closed (as --units-rt does), never drop it silently.
		const named = WRIST_PLUGINS.filter((p) => listed()?.includes(p));
		if (named.length)
			throw new Error(
				`--units-plugins ${named.join(", ")}: this robot configuration has no wrist view, which ${named.length > 1 ? "they need" : "it needs"} (target_in_wrist); drop ${named.length > 1 ? "them" : "it"} or use --units-plugins auto`,
			);
		const off = WRIST_PLUGINS.filter(requested);
		const notice = `units: this robot has no wrist view: ${off.length ? `${off.join(", ")} off (the robot's default plugins), ` : ""}${requested("rotation") ? "rotation keeps only its realign, " : ""}act takes no target_in_wrist; plugins running: ${effective().join(", ") || "none"}`;
		if (ctx.hasUI) ctx.ui.notify(notice, "info");
		else console.error(`[units] ${notice}`);
	}

	return {
		mode,
		started,
		/**
		 * Why the robot must not start with these flags, else undefined: --units-rt on a robot that declares
		 * no RT_* axis would drop ROTATE_* and offer no turn at all.
		 */
		configError: () =>
			rtOn() && !Object.values(spec.rt?.axes ?? {}).some(Boolean)
				? "--units-rt=true: this robot declares no RT_* axis (units `rt`); it would have no turn units at all. Drop --units-rt to keep ROTATE_CW/CCW."
				: undefined,
		/** The robot result's verifier fields: whether the success finish was checked, and the call's latest error. */
		result: () => ({
			// What ran: the plugins after the robot's wrist view, and that view (units mode only).
			...(mode() ? { units_plugins: effective(), units_wrist_view: wristView } : {}),
			...(check.finishVerified !== undefined ? { finish_verified: check.finishVerified } : {}),
			...(check.verifierError ? { verifier_error: check.verifierError } : {}),
		}),
		/** A scene reset: the arm is back at its start heading with the gripper open, no plan. */
		reset: () => {
			reset();
			save();
		},
		/** The units tools: act and the enabled plugins' tools (pure mode adds finish). */
		tools: () => ["act", ...(plugin("point") ? ["point"] : []), ...(plugin("plan") ? ["plan"] : [])],
		/** Pure mode: the whole prompt. Both mode: the section appended to the robot's prompt. */
		prompt: () =>
			renderPrompt({
				spec,
				mode: mode(),
				arms: armNames,
				rt: rtOn(),
				wrist: wristSignal(),
				wristView,
				plugin,
				stateless: pi.getFlag("stateless") === true,
				brief: demo?.key.startsWith(`${pi.getFlag("units-video-ref")}#`) ? demo.brief : undefined,
				task: instruction(),
				coarseM: coarse(),
				highM: high,
			}),
	};
}
