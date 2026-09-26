/**
 * The units system prompt: ./SYSTEM.md with its `[name]...[/name]` sections kept or dropped for the
 * mode, the robot and the enabled plugins, and its `{{var}}` placeholders filled in.
 *
 * Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: see ./index.ts.
 */

import { template } from "../context-version.ts";
import { type AblationMode, ablationDirection, mcqOptions, NEUTRAL_MOVES, NEUTRAL_VIEWS } from "./experimental.ts";
import type { UnitsSpec } from "./types.ts";
import { type DemoBrief, renderBrief } from "./vlm.ts";
import { CHUNK_STEPS, MOVE_UNITS, type MoveUnit, PLUGINS, type Plugin, type Vec3 } from "./vocabulary.ts";

/**
 * Show-Harness's image convention (prompts/controller.txt), which configs/primitives_<robot>.yaml
 * calibrate the unit vectors to: a third-person view facing the robot, then the wrist view.
 */
export const DEFAULT_VIEWS = `Each result shows the third-person view (it faces the robot), then the wrist view (the gripper fingers stay fixed in it).
- Third-person view: MV_LEFT / MV_RIGHT move toward the image left / right, MV_FWD toward the image bottom, MV_BACK toward the image top.
- Wrist view: a target to the fingers' left / right needs MV_LEFT / MV_RIGHT, one near the image bottom and far from the fingers needs MV_FWD, one between the image top and the fingers needs MV_BACK; centered between the fingers: MV_DOWN.`;

/** DEFAULT_VIEWS for a robot configuration without a wrist view: the third-person view alone. */
export const DEFAULT_VIEWS_NO_WRIST = `Each result shows the third-person view (it faces the robot); this robot has no wrist view.
- Third-person view: MV_LEFT / MV_RIGHT move toward the image left / right, MV_FWD toward the image bottom, MV_BACK toward the image top.`;

const TEMPLATE = template(new URL("./SYSTEM.md", import.meta.url)).replace(/^<!--[\s\S]*?-->\n/, "");

/** Keep every `[name]...[/name]` block when `on`, drop them otherwise. */
function section(prompt: string, name: string, on: boolean) {
	const re = new RegExp(`\\[${name}\\]\\n([\\s\\S]*?)\\[/${name}\\]\\n`, "g");
	return prompt.replace(re, on ? "$1" : "");
}

/** What the prompt depends on, read by ./index.ts from the spec, the flags and the episode. */
export type PromptContext = {
	spec: Pick<UnitsSpec, "stepM" | "yawStepRad" | "rt" | "views" | "emptyWidthM" | "vectors">;
	mode: "pure" | "both" | undefined;
	arms: readonly string[];
	/** --units-rt is on. */
	rt: boolean;
	/** `act` takes `target_in_wrist` (a wrist view and variable_step, action_chunk or rotation). */
	wrist: boolean;
	/** The robot configuration has a wrist view (UnitsSpec.wrist): its [wrist_view] or [no_wrist_view] text. */
	wristView: boolean;
	/** The robot has a gripper (UnitsSpec.gripper): its [gripper] or [no_gripper] text. */
	gripper: boolean;
	plugin: (name: Plugin) => boolean;
	stateless: boolean;
	/** video_ref: the demo brief of the current --units-video-ref, if extracted. */
	brief: DemoBrief | undefined;
	task: string;
	coarseM: number;
	highM: number;
	/** stage_control: units per stage before the plan moves on (0 = no cap). */
	stageSteps?: number;
	/** mcq: the units `act` offers, in the order their option letters follow. */
	vocabulary?: readonly string[];
	/** action_ablation: the setting (--units-ablation), when the plugin runs. */
	ablation?: AblationMode;
};

/** A unit vector's base-frame axis label, e.g. "+x" (its largest component). */
export function axisOf(v: Vec3) {
	const i = [0, 1, 2].reduce((best, k) => (Math.abs(v[k]) > Math.abs(v[best]) ? k : best), 0);
	return `${v[i] < 0 ? "-" : "+"}${"xyz"[i]}`;
}

/**
 * coords (plugins/coords coords.txt [direction]): the DIRECTION rules in base-frame axis terms. Upstream
 * swaps DIRECTION..REMARK, but prompts/controller.txt @137d571 has no REMARK: section, so there the
 * plugin leaves the prompt unchanged; this port applies what the fragments say (DIRECTION replaced,
 * the remark rules added to ATTENTION), with the axes read from the robot's unit vectors instead of
 * the Franka rig's camera calibration.
 */
function coordsDirection(vectors: Record<MoveUnit, Vec3>) {
	const ax = (u: MoveUnit) => axisOf(vectors[u]);
	return [
		"DIRECTION:",
		"Use the robot's right-handed base coordinate system. Choose the single unit that best reduces the largest remaining 3D error between the gripper or held object and the current goal:",
		...MOVE_UNITS.map((u) => `- ${u} -> move along ${ax(u)}`),
		"",
		"Camera-to-axis calibration: VIEWS says how each unit looks in the images; use it to turn image evidence into a robot-axis displacement.",
		`- TARGET cropped, hidden by the gripper, or too large in the wrist view -> ${ax("MV_BACK")} (MV_BACK)`,
		`- ${ax("MV_FWD").slice(1)} (near/far) overshoots from a high viewpoint: take ONE small step, and when far above the table MV_DOWN (${ax("MV_DOWN")}) first rather than repeating ${ax("MV_FWD")} to chase a far-looking target.`,
		"",
		`Use image evidence only to infer the needed robot-axis displacement. In your reasoning, state the displacement in axis terms such as "${ax("MV_FWD")} and ${ax("MV_LEFT")}", not as image edge phrases.`,
	].join("\n");
}

/** coords (coords.txt [remark]): the rules it adds to ATTENTION, in axis terms. */
function coordsRemark(vectors: Record<MoveUnit, Vec3>) {
	const ax = (u: MoveUnit) => `${u} (${axisOf(vectors[u])})`;
	return [
		"- Prefer horizontal alignment before descending for grasping or placing.",
		`- Depth beats appearance: if wrist depth is occluded or overshot, use ${ax("MV_BACK")} even after ${ax("MV_FWD")}.`,
		`- Use ${ax("MV_DOWN")} only when the gripper is visually aligned enough to approach, or when placing a held object onto the destination.`,
		`- If low near the table but not aligned, use ${ax("MV_UP")} before moving horizontally.`,
		`- After grasping, ${ax("MV_UP")} clear of the table BEFORE any horizontal move.`,
		`- For LIFT, repeat ${ax("MV_UP")} until a clear gap shows under the object; one MV_UP is not enough.`,
		"- Edge/corner alignment is not a grasp; satisfy the GRIPPER criteria before closing.",
		"- NEVER oscillate: never pick a move opposite to the newest recent move (MV_LEFT with MV_RIGHT; MV_FWD with MV_BACK).",
	].join("\n");
}

/** Replace the text from the line `start` up to (not including) the line `end`. */
function replaceBlock(prompt: string, start: string, end: string, block: string) {
	const i = prompt.indexOf(`\n${start}\n`);
	const j = prompt.indexOf(`\n${end}\n`, i + 1);
	if (i < 0 || j < 0) return prompt;
	return `${prompt.slice(0, i + 1)}${block.trim()}\n${prompt.slice(j)}`;
}

/** Pure mode: the whole prompt. Both mode: the section appended to the robot's prompt. */
export function renderPrompt(c: PromptContext) {
	const { spec, arms, brief } = c;
	let p = section(TEMPLATE, "pure", c.mode === "pure");
	p = section(p, "both", c.mode === "both");
	p = section(p, "arms", arms.length > 0);
	p = section(p, "yaw", Boolean(spec.yawStepRad) && !c.rt);
	p = section(p, "rt", c.rt && Boolean(spec.rt));
	p = section(p, "wrist", c.wrist);
	p = section(p, "wrist_view", c.wristView);
	p = section(p, "no_wrist_view", !c.wristView);
	p = section(p, "gripper", c.gripper);
	p = section(p, "no_gripper", !c.gripper);
	for (const name of PLUGINS)
		p = section(
			p,
			name,
			c.plugin(name) && (!["recovery", "auto_release"].includes(name) || spec.emptyWidthM !== undefined),
		);
	p = section(p, "stateless", c.stateless);
	const coords = c.plugin("coords");
	if (coords) p = replaceBlock(p, "DIRECTION:", "GRIPPER:", coordsDirection(spec.vectors));
	// action_ablation bare / letters_blind: no sentence says which way a direction unit moves (VIEWS, the
	// units list and DIRECTION are replaced); letters keeps them, symbolized by the funnel (./experimental.ts).
	const blind = c.ablation && c.ablation !== "letters" ? ablationDirection(c.ablation) : undefined;
	if (blind) {
		p = replaceBlock(p, "DIRECTION:", "GRIPPER:", blind);
		p = p.replace(/^- MV_FWD, MV_BACK, MV_LEFT, MV_RIGHT, MV_UP, MV_DOWN: .*$/m, NEUTRAL_MOVES);
	}
	p = section(p, "video_ref", brief !== undefined);
	const vars: Record<string, string> = {
		arm: arms.length ? `with ${arms.length} arms` : "arm",
		task: c.task,
		views: blind ? NEUTRAL_VIEWS : (spec.views ?? (c.wristView ? DEFAULT_VIEWS : DEFAULT_VIEWS_NO_WRIST)).trim(),
		step_cm: (spec.stepM * 100).toFixed(0),
		coarse_cm: (c.coarseM * 100).toFixed(0),
		high_cm: (c.highM * 100).toFixed(0),
		chunk: String(CHUNK_STEPS),
		yaw_wrist: c.wristView ? ", so the scene turns clockwise in the wrist view" : "",
		grasp_confirm: c.wristView ? "BOTH views confirm" : "the third-person view confirms",
		judge_views: c.wristView ? "both views" : "the third-person view",
		yaw_deg: String(Math.round(((spec.yawStepRad ?? 0) * 180) / Math.PI)),
		rt_deg: String(Math.round(((spec.rt?.stepRad ?? 0) * 180) / Math.PI)),
		arms: arms.join(", "),
		proprio_note: c.plugin("proprioception") ? ", the gripper's height and width, blocked moves" : "",
		mem_note: c.plugin("mem_text") ? ", the recent moves (newest first)" : "",
		video_ref: brief ? renderBrief(brief, arms) : "",
		mcq_options: mcqOptions(c.vocabulary ?? []).block,
		coords_remark: coords && !blind ? `${coordsRemark(spec.vectors)}\n` : "",
		stage_cap: c.stageSteps
			? `- Each stage runs at most ${c.stageSteps} units; past that the plan moves on to the next stage, and past the last one no unit runs until you send a new plan.\n`
			: "",
	};
	return p.replace(/\{\{(\w+)\}\}/g, (m, k: string) => vars[k] ?? m).trim();
}
