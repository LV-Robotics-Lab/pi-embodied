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
 * - plan: subgoal stages, with deepplan's REASON checkpoint, and Show-Harness's stage control
 *   (core/runners/real.py, dual.py): a stage runs at most `--units-stage-steps` units (default 40,
 *   max_subgoal_steps; 0 = no cap), then the plan moves on with a fresh move history, and past the last
 *   stage no agent unit runs until a new plan (Show-Harness ends the episode there); an empty GRASP or
 *   a lost grasp rolls the plan back to the nearest GRASP stage (of that arm); `plan done` on a GRASP
 *   stage is refused and the stage restarts unless a closed gripper measurably holds (recovery, robots
 *   with `emptyWidthM`), reopening a closed empty one (plugins/recovery after_step).
 * - point: affordance pixels -> world xyz (robots with `point`).
 * - mem_text: "Recent moves, newest first" in every result, with the history rules (no oscillation,
 *   no GRASP in place after an empty one); off, the results carry no move history at all.
 * - Experimental (never in `auto`; ./experimental.ts): coords (DIRECTION and the attention rules in
 *   base-frame axis terms, from the robot's unit vectors), mcq (`act`'s `unit` is an option letter),
 *   action_ablation with --units-ablation bare|letters|letters_blind (the action-representation
 *   ablation). mcq and a letters ablation both own the answer alphabet and refuse to run together.
 * A robot configuration without a wrist view (the spec's `wrist`, e.g. ManiSkill's --robot widowxai)
 * runs `auto` without variable_step and action_chunk and refuses to start when --units-plugins names
 * them (as --units-rt without an axis), and rotation keeps only its realign: those key on the wrist view. `act` then takes no `target_in_wrist` (one sent anyway is
 * ignored, and the result says so), the prompt drops its wrist-view text, a notice names the plugins
 * turned off, and the effective plugins are in every `units_state` entry and the robot result.
 * A robot without a gripper (the spec's `gripper`, e.g. ManiSkill's --robot panda_stick) has no GRASP /
 * RELEASE units and runs without recovery and auto_release; the prompt drops its gripper text.
 *
 * Dual-arm robots: `act`'s `other` is the other arm's unit in the same step (a paired step, Show-Harness's
 * dual runners' (left, right) token pair; STILL = that arm holds). A robot with `applyPair` runs the pair
 * as one command (both arms at once); without it the arms run one after the other, and the result says so.
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
 * subgoal,deepplan,mem_text,coords,mcq,action_ablation} ported as pi tools; the final task check and plugins/video_ref in ./vlm.ts.
 */

import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type TSchema, Type } from "typebox";
import { ABLATION_MODES, Ablation, type AblationMode, mcqOptions, symbol, unsymbol } from "./experimental.ts";
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
	type RtAxis,
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

/** Units a paired two-arm step (`act`'s `other`) may run on either arm (RT_* too). */
const PAIRABLE = new Set<string>([...MOVE_UNITS, ...ROTATE_UNITS, "GRASP", "RELEASE", "STILL"]);
/** stage_control: the per-stage step cap (core/launch.py max_subgoal_steps). */
const STAGE_STEPS = 40;
/** recovery: the note after a GRASP stage's DONE that the gripper does not confirm (plugins/recovery note_unverified_grasp). */
const UNVERIFIED_GRASP = "Grasp not verified; continue GRASP until width and images show a real hold.";

/** Plugins that key on the wrist view (`target_in_wrist`): off on a robot configuration without one. */
const WRIST_PLUGINS: readonly Plugin[] = ["variable_step", "action_chunk"];
/** Plugins that reopen a gripper: off on a robot without one. */
const GRIPPER_PLUGINS: readonly Plugin[] = ["recovery", "auto_release"];
/** The gripper units: not units of a robot without a gripper. */
const GRIPPER_UNITS = ["GRASP", "RELEASE"];

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
	pi.registerFlag("units-stage-steps", {
		type: "string",
		default: String(STAGE_STEPS),
		description:
			"plan: the most units one stage runs before the plan advances to the next stage (Show-Harness max_subgoal_steps; 0 = no cap)",
	});
	pi.registerFlag("units-ablation", {
		type: "string",
		default: "",
		description: `action_ablation (experimental): the action-representation setting (${ABLATION_MODES.join(", ")})`,
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
	/** The robot has a gripper (spec.gripper), read like the wrist view. */
	const readGripper = () => (typeof spec.gripper === "function" ? spec.gripper() : spec.gripper) !== false;
	let gripperOn = readGripper();
	const plugin = (name: Plugin) =>
		requested(name) && (wristView || !WRIST_PLUGINS.includes(name)) && (gripperOn || !GRIPPER_PLUGINS.includes(name));
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
				(gripperOn || !GRIPPER_UNITS.includes(u)) &&
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
	/** stage_control: units run in the current stage, and whether the last stage used its cap. */
	let stageSteps = 0;
	let capped = false;
	let targets: Target[] = [];
	/** action_ablation: this session's setting (--units-ablation with the plugin on), else undefined. */
	const ablationMode = (): AblationMode | undefined => {
		const m = String(pi.getFlag("units-ablation") ?? "").trim() as AblationMode;
		return plugin("action_ablation") && ABLATION_MODES.includes(m) ? m : undefined;
	};
	let ablation: Ablation | undefined;
	/** letters_blind: the third-person image of the latest `act` result (the frame before the next action). */
	let lastFrame: { type: "image"; data: string; mimeType: string } | undefined;
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
		stageSteps = 0;
		capped = false;
		targets = [];
		lastFrame = undefined;
		ablation = ablationMode() ? new Ablation(ablationMode() as AblationMode) : undefined;
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
			...(capped ? { stageCapExceeded: true } : {}),
			...(ablation ? { ablation: ablation.record() } : {}),
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
		gripperOn = readGripper();
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

	const stageCap = () => Math.max(0, Math.floor(Number(pi.getFlag("units-stage-steps"))) || 0);
	/**
	 * stage_control (core/runners/real.py): a stage that ran its step cap is abandoned for the next one
	 * with a fresh move history; past the last stage no unit runs until a new plan. False: stop.
	 */
	function advanceCapped(lines: string[]) {
		const cap = stageCap();
		if (!plugin("plan") || !cap || stage >= stages.length || stageSteps < cap) return true;
		lines.push(`STAGE ${stage + 1} [${stages[stage].motion}] used its ${cap}-step cap: the plan moves on.`);
		stage++;
		stageSteps = 0;
		recent = [];
		if (stage < stages.length) return true;
		capped = true;
		lines.push("That was the last planned stage: send a new plan, or finish.");
		return false;
	}
	/**
	 * stage_control: after an empty or lost grasp the plan rolls back to the nearest GRASP stage at or
	 * before the current one (of that arm on two arms; the first stage without one), as plugins/recovery's
	 * rollback_index. False when no plan runs.
	 */
	function rollback(lines: string[], arm: string | undefined) {
		if (!plugin("plan") || !stages.length) return false;
		let to = 0;
		for (let i = Math.min(stage, stages.length - 1); i >= 0; i--)
			if (stages[i].motion.toUpperCase() === "GRASP" && (!arm || !stages[i].arm || stages[i].arm === arm)) {
				to = i;
				break;
			}
		stage = to;
		stageSteps = 0;
		capped = false;
		lines.push(`Plan rolled back to stage ${to + 1} [${stages[to].motion}].`);
		return true;
	}

	/** The units block that heads every `act` result. */
	async function header(lines: string[], arm: string | undefined) {
		const out = [...lines];
		if (plugin("plan") && stages.length) {
			const s = stages[stage];
			out.push(
				s
					? `STAGE ${stage + 1}/${stages.length} [${s.motion}]${s.arm ? ` (${s.arm} arm)` : ""}: target ${s.target}${s.affordance ? `; affordance ${s.affordance}` : ""}${s.description ? `; ${s.description}` : ""}; DONE WHEN ${s.completion}${stageCap() ? `; step ${stageSteps} of ${stageCap()}` : ""}`
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
	/** A unit as the model names it: its mcq letter, its action_ablation symbol, or its name. */
	const answerOf = (u: string) => {
		if (plugin("mcq")) return mcqOptions(vocab()).letters[vocab().indexOf(u as Unit)] ?? u;
		return ablation?.symbolic ? symbol(u) : u;
	};
	/** The model's answer back to its unit (mcq letter, symbol), or undefined for an unknown letter. */
	const unitOf = (a: string) => (plugin("mcq") ? mcqOptions(vocab()).unit(a) : ablation?.symbolic ? unsymbol(a) : a);
	function actSchema() {
		const props: Record<string, TSchema> = {
			unit: StringEnum(vocab().map(answerOf), {
				description: plugin("mcq")
					? `The option letter of the action unit: ${mcqOptions(vocab()).block}`
					: "The action unit",
			}),
			n: Type.Optional(Type.Integer({ minimum: 1, maximum: MAX_REPEAT, description: "Repeat count (default 1)" })),
		};
		if (wristSignal())
			props.target_in_wrist = Type.Optional(
				Type.Boolean({ description: "WRIST CHECK: is the TARGET visible in the wrist view?" }),
			);
		if (plugin("action_chunk"))
			props.plan = Type.Optional(
				Type.Array(StringEnum(MOVE_UNITS.map((u) => (ablation?.symbolic ? symbol(u) : u))), {
					maxItems: CHUNK_STEPS,
					description: `Only with target_in_wrist false: up to ${CHUNK_STEPS} MV_* moves run in order (replaces unit and n)`,
				}),
			);
		if (armNames.length) props.arm = StringEnum(armNames, { description: "Which arm the unit drives" });
		if (armNames.length === 2)
			props.other = Type.Optional(
				StringEnum(
					vocab()
						.filter((u) => PAIRABLE.has(u) || isRt(u))
						.map(answerOf),
					{
						description:
							"Paired step: the OTHER arm's unit, run at the same time as `unit` (STILL = it holds); n repeats the pair",
					},
				),
			);
		if (ablation?.mode === "letters_blind")
			props.note = Type.Optional(
				Type.String({
					description: "REVIEW record when the last result asked for one: NOTE[ACT_X]: <what it did>",
				}),
			);
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
		const description = actDescription();
		if (JSON.stringify([schema, description]) === registered) return;
		registered = JSON.stringify([schema, description]);
		tool("act", description, schema, (params, signal) => modelAct(params as ActParams, signal));
	}
	/** `act`'s description (through the action_ablation funnel). */
	function actDescription() {
		const d = `Execute one action unit (${vocab().join(", ")}), repeated n times. MV_* move the gripper ~${Math.round(spec.stepM * 100)} cm${spec.yawStepRad && !rtOn() ? `, ROTATE_* turn it ~${Math.round((spec.yawStepRad * 180) / Math.PI)} deg` : ""}${rtOn() && spec.rt ? `, RT_* turn it ~${Math.round((spec.rt.stepRad * 180) / Math.PI)} deg about a world axis through the fingertips (ROLL about the MV_FWD axis, PITCH about the MV_LEFT-MV_RIGHT axis, YAW about the vertical)` : ""}${gripperOn ? "; GRASP closes, RELEASE opens" : " (this robot has no gripper)"}, STOP holds one step, DONE means the task is complete (call finish). Returns the new images and state.`;
		return ablation ? ablation.filter(d) : d;
	}
	registerAct();
	pi.on("session_start", registerAct);

	type ActParams = {
		unit: string;
		n?: number;
		arm?: string;
		other?: string;
		target_in_wrist?: boolean;
		plan?: string[];
		operator?: boolean;
		view?: string;
		retreat?: boolean;
		note?: string;
	};
	/**
	 * The model's `act`: its answers (mcq letters, action_ablation symbols) back to units, the unit run,
	 * then its result through the funnel; letters_blind harvests the reviewed symbol's note first and
	 * asks for the next review with the frame from before the move.
	 */
	async function modelAct(params: ActParams, signal: AbortSignal | undefined): Promise<Result> {
		const answer = (a: string | undefined, what: string) => {
			if (a === undefined) return undefined;
			const u = unitOf(a);
			if (u === undefined) throw new Error(`act: ${what} ${a} is not an option (${mcqOptions(vocab()).block})`);
			return u;
		};
		const harvested = ablation?.harvest(params.note);
		if (ablation) ablation.review = undefined;
		const before = lastFrame;
		const run: ActParams = {
			...params,
			unit: answer(params.unit, "unit") as string,
			...(params.other !== undefined ? { other: answer(params.other, "other") } : {}),
			...(params.plan ? { plan: params.plan.map(unsymbol) } : {}),
		};
		delete run.note;
		const r = await actAndSave(run, signal);
		const first = r.content.find((c) => c.type === "image");
		if (first?.type === "image") lastFrame = first;
		if (!ablation) return r;
		const [head, ...rest] = r.content;
		const lines = [head?.type === "text" ? ablation.filter(head.text) : ""];
		if (harvested) lines.push(`Recorded in your table: ${harvested}`);
		const images = [...rest];
		if (ablation.mode === "letters_blind") {
			lines.push(ablation.table());
			const m = /^units: (MV_\w+) x1\n/.exec(head?.type === "text" ? head.text : "");
			if (m && before && run.other === undefined && !run.plan?.length) {
				ablation.review = symbol(m[1]);
				lines.push(ablation.reviewText(ablation.review));
				images.push(before);
			}
			save();
		}
		return { ...r, content: [text(lines.join("\n")), ...images] };
	}
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
			other?: Unit;
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
		/** The paired arm of a synchronous two-arm step (`other`), else undefined. */
		let pairArm: string | undefined;
		if (p.other !== undefined) {
			pairArm = armNames.find((a) => a !== arm);
			if (armNames.length !== 2 || !arm || !pairArm)
				throw new Error("act: `other` pairs the two arms of a dual-arm robot; name `arm` for `unit`");
			if (p.plan !== undefined) throw new Error("act: `other` (a paired step) and `plan` (a chunk) do not combine");
			for (const u of [unit, p.other])
				if (!PAIRABLE.has(u) && !isRt(u))
					throw new Error(`act: ${u} cannot run in a paired step (moves, turns, GRASP, RELEASE and STILL can)`);
		}
		if (demoError) return { content: [text(`video_ref failed: ${demoError}`)], details: { unit, error: demoError } };
		if (unit === "DONE")
			return {
				content: [text("DONE: if the images show the task complete, call `finish` now; otherwise keep acting.")],
				details: { unit },
			};
		if (unit === "STILL" && (p.other === undefined || p.other === "STILL"))
			return { content: [text(`STILL: the ${arm ?? ""} arm holds.`)], details: { unit } };
		for (const u of [unit, ...(p.other !== undefined ? [p.other] : [])]) {
			const rtWhy = isRt(u) ? rtRefusal(u) : rotateRefusal(u);
			if (rtWhy) throw new Error(`act: ${rtWhy}`);
		}
		const lines: string[] = [];
		// stage_control: the plan's last stage ran out of steps; only a new plan (or finish) goes on.
		if (capped && !p.operator && plugin("plan"))
			return {
				content: [
					await header(["units: not run: every planned stage used its step cap; send a new plan or finish."], arm),
				],
				details: { unit, stage_cap_exceeded: true },
			};
		if (!wristView && p.target_in_wrist !== undefined && !p.operator)
			lines.push("target_in_wrist ignored: this robot has no wrist view.");
		const n = Math.max(1, Math.min(MAX_REPEAT, Math.floor(p.n ?? 1)));
		type Slot = { arm: string | undefined; u: Unit };
		/** One step per tick: one arm's unit, or (paired) both arms' units at once; STILL drops out. */
		let ticks: Slot[][] = Array.from({ length: n }, () =>
			[{ arm, u: unit }, ...(pairArm && p.other !== undefined ? [{ arm: pairArm, u: p.other }] : [])].filter(
				(s) => s.u !== "STILL",
			),
		);
		if (p.plan?.length && plugin("action_chunk")) {
			if (inWrist === false)
				ticks = p.plan
					.filter(isMove)
					.slice(0, CHUNK_STEPS)
					.map((u) => [{ arm, u }]);
			else {
				lines.push("plan ignored: plans run only while target_in_wrist is false; one unit ran.");
				ticks = [[{ arm, u: unit }]];
			}
		}
		const tag = (s: Slot) => (pairArm ? `${s.arm?.[0].toUpperCase()}:${s.u}` : s.u);
		let last: Result | undefined;
		const ran: string[] = [];
		const halted = (r: Result, m?: Move) => {
			const d = r.details as { error?: unknown; terminated?: unknown } | undefined;
			// The finish sequence runs past the robot's success signal while the robot still moves (returns images).
			const moved = m !== undefined && finishMove(m) && r.content.some((c) => c.type === "image");
			return Boolean(d?.error || (d?.terminated && !moved));
		};
		/** Path length so far per arm: one call travels at most the robot's per-call translation limit. */
		const travelled = new Map<string, number>();
		// A recovery note lasts until the next GRASP.
		if (ticks.some((t) => t.some((s) => s.u === "GRASP"))) note = "";
		let stop = false;
		for (const [index, tick] of ticks.entries()) {
			// stage_control: a stage that used its step cap is abandoned for the next one (core/runners/real.py).
			if (!advanceCapped(lines)) break;
			type Planned = {
				slot: Slot;
				key: string;
				label: string;
				move: Move;
				parts: number;
				turn: { axis: RtAxis; sign: number } | undefined;
				turnBy: number;
				dist: number;
				before: State | undefined;
			};
			const planned: Planned[] = [];
			for (const slot of tick) {
				const { u } = slot;
				const key = slot.arm ?? "";
				const before = await read(slot.arm);
				const acc = yaw.get(key) ?? 0;
				let label: string = tag(slot);
				let move = ground(spec, u, isMove(u) && !p.operator ? stepFor(u, before, inWrist) : spec.stepM) as Move;
				// Holding and turned: MV_UP first turns back to the start heading.
				if (assist("rotation") && u === "MV_UP" && closed.get(key) && Math.abs(acc) > NEUTRAL_YAW) {
					move = { delta: [0, 0, 0], yaw: -acc, gripper: null };
					label = `${tag(slot)}(realign)`;
					// Wrist-judged moves follow the turned wrist camera; without one every move is judged in the base frame.
				} else if (assist("rotation") && wristView && isMove(u) && inWrist !== false)
					move.delta = compensate(move.delta, (spec.yawCompensationSign ?? 1) * acc);
				else if (move.yaw && Math.abs(acc + move.yaw) > MAX_YAW) {
					// The accumulated-yaw guard holds with or without the rotation plugin.
					lines.push(
						`${tag(slot)} refused: the gripper is already turned ${Math.round((acc * 180) / Math.PI)} deg.`,
					);
					stop = true;
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
							`${tag(slot)} refused: the gripper is already turned ${Math.round((now * 180) / Math.PI)} deg about its ${turn.axis} axis.`,
						);
						stop = true;
						break;
					}
				}
				const dist = Math.hypot(...move.delta);
				const maxMove = spec.maxMoveM?.();
				const so = travelled.get(key) ?? 0;
				if (maxMove !== undefined && so > 0 && !(so + dist <= maxMove + 1e-9)) {
					lines.push(`${tag(slot)} not run: one act call moves at most ${maxMove} m in total.`);
					stop = true;
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
					lines.push(
						`${label} refused: the robot's per-call rotation limit is ${move.rot ? maxRot : maxYaw} rad.`,
					);
					stop = true;
					break;
				}
				if (slot.arm) move.arm = slot.arm;
				if (p.view) move.view = p.view;
				if (p.retreat) move.retreat = true;
				if (!pairArm && isMove(u) && label === u && ticks[index + 1]?.[0]?.u === u) move.continuous = true;
				planned.push({ slot, key, label, move, parts, turn, turnBy, dist, before });
			}
			if (stop) break;
			/** Book a turn's share into the accumulators. */
			const turned = (q: Planned, share: number) => {
				if (q.move.yaw) yaw.set(q.key, (yaw.get(q.key) ?? 0) + q.move.yaw * share);
				if (q.turn?.axis === "yaw") yaw.set(q.key, (yaw.get(q.key) ?? 0) + q.turnBy * share);
				else if (q.turn) {
					const tilts = q.turn.axis === "roll" ? roll : pitch;
					tilts.set(q.key, (tilts.get(q.key) ?? 0) + q.turnBy * share);
				}
			};
			// A paired step goes to the robot as one command when it can run both arms at once.
			if (planned.length === 2 && spec.applyPair && planned.every((q) => q.parts === 1)) {
				last = await spec.applyPair(
					planned.map((q) => q.move),
					signal,
				);
				const r = last;
				if (!planned.some((q) => r && halted(r, q.move))) for (const q of planned) turned(q, 1);
			} else {
				if (planned.length === 2 && index === 0)
					lines.push("paired step: this robot runs the two arms one after the other (no simultaneous command).");
				for (const q of planned) {
					for (let i = 0; i < q.parts; i++) {
						const piece: Move = i ? { ...q.move, delta: [0, 0, 0], gripper: null } : q.move;
						const rot = q.move.rot?.map((x) => x / q.parts) as Vec3 | undefined;
						last = await spec.apply(
							q.parts > 1 ? { ...piece, yaw: q.move.yaw / q.parts, ...(rot ? { rot } : {}) } : piece,
							signal,
						);
						if (halted(last, q.move)) break;
						turned(q, 1 / q.parts);
					}
					if (last && halted(last, q.move)) break;
				}
			}
			stageSteps++;
			ran.push(planned.map((q) => q.label).join("+"));
			for (const q of planned) {
				const { u } = q.slot;
				travelled.set(q.key, (travelled.get(q.key) ?? 0) + q.dist);
				// mem_text records moves, turns and grasps (not STOP / RELEASE), as core/runners/real.py.
				if (isMove(u) || isRt(u) || u === "GRASP" || u === "ROTATE_CW" || u === "ROTATE_CCW") remember(tag(q.slot));
				if (q.move.gripper) closed.set(q.key, q.move.gripper === "close");
			}
			if (last && planned.some((q) => halted(last as Result, q.move))) break;
			for (const q of planned) {
				const { u } = q.slot;
				const after = await read(q.slot.arm);
				// proprioception: a MV_* that barely moved is blocked (contact, floor, workspace limit).
				const [p0, p1] = [eef(q.before), eef(after)];
				const base = spec.baseDelta?.(q.move.delta, q.before) ?? q.move.delta;
				const commanded = Math.hypot(...base);
				// A chained move returned before the arm settled: its lagging position is not a stall.
				const settling = q.move.continuous === true && spec.chains?.() === true;
				if (plugin("proprioception") && !settling && p0 && p1 && commanded > 0) {
					const moved = [0, 1, 2].reduce((s, k) => s + (p1[k] - p0[k]) * base[k], 0) / commanded;
					if (moved < commanded * STALL_RATIO) {
						lines.push(
							u === "MV_DOWN"
								? `Last ${tag(q.slot)} lowered ${(moved * 100).toFixed(1)} of ${(commanded * 100).toFixed(1)} cm -> already in contact, do NOT MV_DOWN again`
								: `Last ${tag(q.slot)} moved ${(moved * 100).toFixed(1)} of ${(commanded * 100).toFixed(1)} cm -> blocked`,
						);
						stop = true;
						continue;
					}
				}
				// recovery: a GRASP that closed on nothing is reopened at once, and the plan rolls back to its grasp stage.
				if (u === "GRASP" && assist("recovery") && empty(after)) {
					last = await reopen(q.slot.arm, signal);
					// The fresh approach starts from a clean history holding only the failed GRASP.
					recent = [pairArm ? `${tag(q.slot)}(empty)` : EMPTY_GRASP];
					note = NOTES.empty_grasp;
					rollback(lines, q.slot.arm);
					stop = true;
					continue;
				}
				// auto_release: a closed gripper that collapsed (the object slipped out) is reopened.
				if (u !== "GRASP" && assist("auto_release") && closed.get(q.key) && empty(after)) {
					last = await reopen(q.slot.arm, signal);
					note = NOTES.lost_grasp;
					if (rollback(lines, q.slot.arm)) recent = [];
					stop = true;
				}
			}
			if (stop) break;
		}
		const queue = ticks.map((t) => t.map(tag).join("+"));
		const what =
			queue.every((u) => u === queue[0]) && ran.every((u) => u === ran[0])
				? `${ran[0] ?? queue[0]} x${ran.length}`
				: ran.join(", ");
		lines.unshift(
			`units: ${what}${arm && !pairArm ? ` (${arm} arm)` : ""}${ran.length < ticks.length ? ` of ${ticks.length} (stopped early)` : ""}`,
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
		async ({ stages: next, done }, signal) => {
			// stage_control: a GRASP stage is done only on a measured hold (plugins/recovery after_step's
			// unverified_grasp): otherwise the stage restarts and a closed empty gripper reopens.
			const current = stages[stage];
			if (
				done &&
				current?.motion.toUpperCase() === "GRASP" &&
				plugin("recovery") &&
				spec.emptyWidthM !== undefined
			) {
				const sides = current.arm ? [current.arm] : armNames.length ? [...armNames] : [undefined];
				const held: (string | undefined)[] = [];
				for (const a of sides) {
					const st = await read(a);
					if (closed.get(a ?? "") === true && !empty(st)) held.push(a);
					else if (closed.get(a ?? "") === true && st) await reopen(a, signal);
				}
				if (!held.length) {
					stageSteps = 0;
					recent = [];
					note = UNVERIFIED_GRASP;
					save();
					return {
						content: [
							text(
								`done refused: stage ${stage + 1} [GRASP] is not verified (${sides.length > 1 ? "no gripper" : "the gripper"} measurably holds an object); the stage restarts. Recovery: ${UNVERIFIED_GRASP}`,
							),
						],
						details: { stage, stages, unverified_grasp: true },
					};
				}
			}
			if (done && stage < stages.length) stage++;
			if (next?.length) stages = [...stages.slice(0, stage), ...next];
			if (done || next?.length) {
				stageSteps = 0;
				capped = false;
			}
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
		gripperOn = readGripper();
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
		configError: () => {
			if (rtOn() && !Object.values(spec.rt?.axes ?? {}).some(Boolean))
				return "--units-rt=true: this robot declares no RT_* axis (units `rt`); it would have no turn units at all. Drop --units-rt to keep ROTATE_CW/CCW.";
			// The experimental plugins fail closed on an incomplete or conflicting setting.
			const m = String(pi.getFlag("units-ablation") ?? "").trim();
			if (plugin("action_ablation") && !ABLATION_MODES.includes(m as AblationMode))
				return `--units-plugins action_ablation needs --units-ablation ${ABLATION_MODES.join("|")}`;
			if (m && !plugin("action_ablation"))
				return `--units-ablation ${m} needs the action_ablation plugin in --units-plugins`;
			if (plugin("mcq") && ablationMode() && ablationMode() !== "bare")
				return "mcq and action_ablation letters modes both set the answer alphabet: run one of them";
			return undefined;
		},
		/** The robot result's verifier fields: whether the success finish was checked, and the call's latest error. */
		result: () => ({
			// What ran: the plugins after the robot's wrist view, and that view (units mode only).
			...(mode() ? { units_plugins: effective(), units_wrist_view: wristView } : {}),
			...(ablation ? { units_ablation: ablation.record() } : {}),
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
		prompt: () => {
			const p = renderPrompt({
				spec,
				mode: mode(),
				arms: armNames,
				rt: rtOn(),
				wrist: wristSignal(),
				wristView,
				gripper: gripperOn,
				plugin,
				stateless: pi.getFlag("stateless") === true,
				brief: demo?.key.startsWith(`${pi.getFlag("units-video-ref")}#`) ? demo.brief : undefined,
				task: instruction(),
				coarseM: coarse(),
				highM: high,
				stageSteps: stageCap(),
				vocabulary: vocab(),
				ablation: ablation?.mode,
			});
			return ablation ? ablation.filter(p) : p;
		},
	};
}
