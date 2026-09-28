/**
 * Optional VLA skill servers (LIBERO's Pi0.5, RoboCasa's RLDX-1, RoboTwin's LingBot): a robot starts
 * without one. At session start the robot probes each skill's server; one that is switched off
 * (`"off"` in the deployment's services, ../infra/config.ts) or does not answer leaves its tools inactive (the manifest entries
 * `requires` the skill's name) and the result row records it in `skills_off`. Only a run that
 * asks for the skill, `--require-skills <name,...>`, refuses to start without it.
 *
 * The perception servers are skills too (`probePerception`, run by ../robot.ts after the robot's
 * start for every robot): `sam3` (segment, detect, select_detection, reject_detection), `unidepth`
 * (enhance_depth), `molmo` (point, molmo_point). A tool whose server does not answer stays inactive
 * instead of failing its first call.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { service } from "../infra/config.ts";
import { RpcClient } from "../infra/rpc.ts";
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

/** A flag the robot registered (a stub or pi answers undefined for one it did not). */
const has = (pi: ExtensionAPI, flag: string) => pi.getFlag(flag) !== undefined;

/**
 * The perception servers, the tools that need them, and their endpoint for this run: the deployment's
 * services.<skill> (../infra/config.ts), "" when the robot's switch for it is off (the real arms'
 * --segment, --depth), undefined when the robot has no way to reach the server at all.
 */
export const PERCEPTION_SKILLS: readonly {
	skill: "sam3" | "unidepth" | "molmo";
	tools: readonly string[];
	url: (pi: ExtensionAPI) => string | undefined;
}[] = [
	{
		skill: "sam3",
		tools: ["segment", "detect", "select_detection", "reject_detection"],
		// The real arms attach SAM3 with --segment; the simulators' env servers always have it.
		url: (pi) =>
			has(pi, "segment") ? (pi.getFlag("segment") === true ? service(pi, "sam3") : "") : service(pi, "sam3"),
	},
	{
		skill: "unidepth",
		tools: ["enhance_depth"],
		url: (pi) =>
			has(pi, "depth")
				? String(pi.getFlag("depth") ?? "").trim() === "unidepth"
					? service(pi, "unidepth")
					: ""
				: undefined,
	},
	{ skill: "molmo", tools: ["point", "molmo_point"], url: (pi) => service(pi, "molmo") },
];

/** A perception server answers healthz within 3 s. */
export const healthz = (url: string) => new RpcClient(url).ready(3_000);

/**
 * Probe the perception servers behind the tools this run would activate: the tools whose server is
 * off or does not answer are dropped (`keep`), and `off` says which skill and why. A skill named in
 * --require-skills throws instead.
 */
export async function probePerception(
	pi: ExtensionAPI,
	tools: readonly string[],
	probe: (url: string) => Promise<unknown> = healthz,
): Promise<{ keep: string[]; off: Record<string, SkillState> }> {
	const off: Record<string, SkillState> = {};
	const drop = new Set<string>();
	for (const p of PERCEPTION_SKILLS) {
		const needed = p.tools.filter((t) => tools.includes(t));
		if (!needed.length) continue;
		// A robot with no way to reach the server does not use it (its own tool).
		const url = p.url(pi);
		if (url === undefined) continue;
		const state = await probeSkill(pi, p.skill, url, () => probe(url));
		if (state.on) continue;
		off[p.skill] = state;
		for (const t of needed) drop.add(t);
	}
	return { keep: tools.filter((t) => !drop.has(t)), off };
}

/** The result row with the perception skills that were off merged into the robot's `skills_off`. */
export function withSkillsOff<R extends Record<string, unknown>>(result: R, perception: Record<string, SkillState>): R {
	const off = {
		...((result.skills_off as Record<string, string> | undefined) ?? {}),
		...(skillsOff(perception).skills_off ?? {}),
	};
	return Object.keys(off).length ? { ...result, skills_off: off } : result;
}
