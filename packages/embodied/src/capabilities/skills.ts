/**
 * Optional VLA skill servers (LIBERO's Pi0.5, RoboCasa's RLDX-1, RoboTwin's LingBot): a robot starts
 * without one. At session start the robot probes each skill's server; one that is switched off
 * (`--<flag> off` or empty) or does not answer leaves its tools inactive (the manifest entries
 * `requires` the skill's name) and the result row records it in `skills_off`. Only a run that
 * asks for the skill, `--require-skills <name,...>`, refuses to start without it.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { message } from "../robot.ts";

export type SkillState = { on: boolean; reason?: string };

export function registerSkillFlags(pi: ExtensionAPI) {
	pi.registerFlag("require-skills", {
		type: "string",
		default: "",
		description:
			"VLA skills this run needs (comma-separated, e.g. pi0, rldx, lingbot): the robot refuses to start without them; others are optional",
	});
}

/** The skills --require-skills names. */
export const requiredSkills = (pi: ExtensionAPI) =>
	String(pi.getFlag("require-skills") ?? "")
		.split(",")
		.map((s) => s.trim())
		.filter(Boolean);

/**
 * Probe one skill: off when its URL is empty or "off", or when `probe` throws; throws instead when
 * --require-skills names it.
 */
export async function probeSkill(
	pi: ExtensionAPI,
	name: string,
	url: string,
	probe: () => Promise<unknown>,
): Promise<SkillState> {
	let state: SkillState;
	if (!url.trim() || url.trim() === "off") state = { on: false, reason: "switched off" };
	else
		try {
			await probe();
			state = { on: true };
		} catch (err) {
			state = { on: false, reason: `unreachable: ${message(err)}` };
		}
	if (!state.on && requiredSkills(pi).includes(name))
		throw new Error(`--require-skills ${name}: the ${name} server is ${state.reason} (${url || "no URL"})`);
	return state;
}

/** The result row's `skills_off`: each skill that was off and why (empty when all were on). */
export function skillsOff(states: Record<string, SkillState>): { skills_off?: Record<string, string> } {
	const off = Object.fromEntries(
		Object.entries(states)
			.filter(([, s]) => !s.on)
			.map(([k, s]) => [k, s.reason ?? "off"]),
	);
	return Object.keys(off).length ? { skills_off: off } : {};
}
