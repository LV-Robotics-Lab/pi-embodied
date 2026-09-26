/**
 * OpenETA's closed-loop contract (agent/prompts/embodied_closed_loop.md @7d4a0a1), the parts that
 * apply here: ./closed-loop.md is appended by ../robot.ts to every robot's system prompt, in every
 * mode (the robot's own prompt, units and code mode), and one of its rules is enforced:
 *
 *   "A world-mutating transport timeout has unknown outcome. Re-observe and reconcile the same
 *   environment before retrying or issuing another mutation."
 *
 * When a motion tool times out or returns an ambiguous result (`unknownOutcome`), the next motion
 * call is refused until an observation tool (`OBSERVE`: view_env_state, RoboTwin's render) has run
 * successfully. Without an active observation tool (pure units or code mode) there is nothing to
 * re-observe with: the gate stays open, and its hooks are only registered at the first session start
 * that has one. Each unknown outcome, refusal and re-observation is a
 * `closed_loop` session entry; the robot result counts them (`unknown_outcomes`, `reobserve_refusals`)
 * once the gate is mounted.
 *
 * Which tools move the robot is shared with --approval (../operator.ts): every tool the robot
 * registered with its `tool` (the units' and code mode's included) plus the scene resets, except
 * `NON_MOTION`, the tools that only look, plan or read. A new robot tool counts as motion until it
 * is listed here.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { template } from "./context-version.ts";

/** The closed-loop section of every system prompt (its `[tool:...]` blocks follow the active tools). */
export const CLOSED_LOOP = template(new URL("./closed-loop.md", import.meta.url)).trim();

/** Session entry: `{ kind: "unknown_outcome" | "refused" | "reobserved", tool, why? }`. */
export const CLOSED_LOOP_ENTRY = "closed_loop";

/** Tools that fetch a fresh observation of the scene without moving anything. */
export const OBSERVE = ["view_env_state", "render"] as const;

/** Robot tools that never move the robot: observation, perception, planning, checks, metadata, and halting an arm. */
export const NON_MOTION: ReadonlySet<string> = new Set([
	...OBSERVE,
	"get_robot_position",
	"view_camera_meta",
	"view_perception_setup",
	"describe_dual_franka_setup",
	"segment",
	"point",
	"back_project",
	"back_project_batch",
	"back_project_correspondence",
	"sample_world_xyz",
	"query_world_map",
	"select_detection",
	"reject_detection",
	"enhance_depth",
	"plan",
	"plan_grasp",
	"plan_place",
	"preview_reach",
	"check_attached",
	"ground_truth_poses",
	"halt_arm",
]);

/** Tools that restore the scene; ../robot.ts counts them as moving the robot next to its own. */
export const RESETS = ["reset", "request_scene_reset"] as const;

const UNKNOWN =
	/\btimed?[\s_-]?out\b|\btimeout\b|\bunanswered\b|\bunknown outcome\b|\boutcome (?:is )?unknown\b|\bambiguous\b/i;

type Outcome = { isError: boolean; content: readonly { type: string; text?: string }[]; details?: unknown };

const oneLine = (s: string) => s.split(/\s+/).join(" ").trim().slice(0, 200);

/**
 * Why a motion tool's result leaves the world in an unknown state, else undefined: an error that
 * timed out (an RPC timeout, a VLA reply that never came), a run_code program killed at its timeout
 * (`details.status: "timeout"`), or a result whose error or status says the outcome is unknown or
 * ambiguous. A refusal or validation error (nothing was sent) is not one.
 */
export function unknownOutcome(r: Outcome): string | undefined {
	const d = (r.details ?? {}) as Record<string, unknown>;
	const nested = (d.result ?? {}) as Record<string, unknown>;
	if (d.status === "timeout" || d.timed_out === true || d.timeout === true) return "timed out";
	if (d.status === "unknown" || d.outcome === "unknown") return "reported an unknown outcome";
	for (const e of [d.error, nested.error]) if (typeof e === "string" && UNKNOWN.test(e)) return oneLine(e);
	if (r.isError) {
		const text = r.content.map((c) => (c.type === "text" ? (c.text ?? "") : "")).join(" ");
		if (UNKNOWN.test(text)) return oneLine(text);
	}
	return undefined;
}

/** Mount the re-observe gate; `moves` classifies tools (see above). */
export function closedLoop(pi: ExtensionAPI, moves: (tool: string) => boolean) {
	let pending: { tool: string; why: string } | undefined;
	let unknown = 0;
	let refused = 0;
	let hooked = false;
	const observers = () => pi.getActiveTools().filter((t) => (OBSERVE as readonly string[]).includes(t));
	// Mounted after the robot's start, so the active tools are known here.
	pi.on("session_start", () => {
		pending = undefined;
		unknown = refused = 0;
		if (hooked || !observers().length) return;
		hooked = true;
		pi.on("tool_result", watch);
		pi.on("tool_call", gate);
	});
	function watch(event: Outcome & { toolName: string }) {
		if ((OBSERVE as readonly string[]).includes(event.toolName)) {
			if (event.isError || !pending) return undefined;
			pi.appendEntry(CLOSED_LOOP_ENTRY, { kind: "reobserved", tool: event.toolName, after: pending.tool });
			pending = undefined;
			return undefined;
		}
		if (!moves(event.toolName)) return undefined;
		const why = unknownOutcome(event);
		if (why === undefined) return undefined;
		pending = { tool: event.toolName, why };
		unknown++;
		pi.appendEntry(CLOSED_LOOP_ENTRY, { kind: "unknown_outcome", tool: event.toolName, why });
		return undefined;
	}
	function gate(event: { toolName: string }) {
		if (!pending || !moves(event.toolName)) return undefined;
		const active = observers();
		if (!active.length) return undefined;
		refused++;
		pi.appendEntry(CLOSED_LOOP_ENTRY, { kind: "refused", tool: event.toolName, after: pending.tool });
		return {
			block: true,
			reason: `${event.toolName} refused: the last motion (${pending.tool}) has an unknown outcome (${pending.why}). Re-observe with ${active.join(" or ")} before issuing another motion.`,
		};
	}
	return {
		/** The robot result's closed-loop counts (none while the gate was never mounted). */
		result: () => (hooked ? { unknown_outcomes: unknown, reobserve_refusals: refused } : {}),
	};
}
