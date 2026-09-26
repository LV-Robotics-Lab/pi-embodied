/**
 * The units system prompt: ./SYSTEM.md with its `[name]...[/name]` sections kept or dropped for the
 * mode, the robot and the enabled plugins, and its `{{var}}` placeholders filled in.
 *
 * Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: see ./index.ts.
 */

import { readFileSync } from "node:fs";
import type { UnitsSpec } from "./types.ts";
import { type DemoBrief, renderBrief } from "./vlm.ts";
import { CHUNK_STEPS, PLUGINS, type Plugin } from "./vocabulary.ts";

/**
 * Show-Harness's image convention (prompts/controller.txt), which configs/primitives_<robot>.yaml
 * calibrate the unit vectors to: a third-person view facing the robot, then the wrist view.
 */
export const DEFAULT_VIEWS = `Each result shows the third-person view (it faces the robot), then the wrist view (the gripper fingers stay fixed in it).
- Third-person view: MV_LEFT / MV_RIGHT move toward the image left / right, MV_FWD toward the image bottom, MV_BACK toward the image top.
- Wrist view: a target to the fingers' left / right needs MV_LEFT / MV_RIGHT, one near the image bottom and far from the fingers needs MV_FWD, one between the image top and the fingers needs MV_BACK; centered between the fingers: MV_DOWN.`;

const TEMPLATE = readFileSync(new URL("./SYSTEM.md", import.meta.url), "utf8").replace(/^<!--[\s\S]*?-->\n/, "");

/** Keep every `[name]...[/name]` block when `on`, drop them otherwise. */
function section(prompt: string, name: string, on: boolean) {
	const re = new RegExp(`\\[${name}\\]\\n([\\s\\S]*?)\\[/${name}\\]\\n`, "g");
	return prompt.replace(re, on ? "$1" : "");
}

/** What the prompt depends on, read by ./index.ts from the spec, the flags and the episode. */
export type PromptContext = {
	spec: Pick<UnitsSpec, "stepM" | "yawStepRad" | "rt" | "views" | "emptyWidthM">;
	mode: "pure" | "both" | undefined;
	arms: readonly string[];
	/** --units-rt is on. */
	rt: boolean;
	/** `act` takes `target_in_wrist` (variable_step, action_chunk or rotation). */
	wrist: boolean;
	plugin: (name: Plugin) => boolean;
	stateless: boolean;
	/** video_ref: the demo brief of the current --units-video-ref, if extracted. */
	brief: DemoBrief | undefined;
	task: string;
	coarseM: number;
	highM: number;
};

/** Pure mode: the whole prompt. Both mode: the section appended to the robot's prompt. */
export function renderPrompt(c: PromptContext) {
	const { spec, arms, brief } = c;
	let p = section(TEMPLATE, "pure", c.mode === "pure");
	p = section(p, "both", c.mode === "both");
	p = section(p, "arms", arms.length > 0);
	p = section(p, "yaw", Boolean(spec.yawStepRad) && !c.rt);
	p = section(p, "rt", c.rt && Boolean(spec.rt));
	p = section(p, "wrist", c.wrist);
	for (const name of PLUGINS)
		p = section(
			p,
			name,
			c.plugin(name) && (!["recovery", "auto_release"].includes(name) || spec.emptyWidthM !== undefined),
		);
	p = section(p, "stateless", c.stateless);
	p = section(p, "video_ref", brief !== undefined);
	const vars: Record<string, string> = {
		arm: arms.length ? `with ${arms.length} arms` : "arm",
		task: c.task,
		views: (spec.views ?? DEFAULT_VIEWS).trim(),
		step_cm: (spec.stepM * 100).toFixed(0),
		coarse_cm: (c.coarseM * 100).toFixed(0),
		high_cm: (c.highM * 100).toFixed(0),
		chunk: String(CHUNK_STEPS),
		yaw_deg: String(Math.round(((spec.yawStepRad ?? 0) * 180) / Math.PI)),
		rt_deg: String(Math.round(((spec.rt?.stepRad ?? 0) * 180) / Math.PI)),
		arms: arms.join(", "),
		proprio_note: c.plugin("proprioception") ? ", the gripper's height and width, blocked moves" : "",
		mem_note: c.plugin("mem_text") ? ", the recent moves (newest first)" : "",
		video_ref: brief ? renderBrief(brief, arms) : "",
	};
	return p.replace(/\{\{(\w+)\}\}/g, (m, k: string) => vars[k] ?? m).trim();
}
