/**
 * The `--ik <url>` flag: an IK service (services/pi_embodied_services/components/ik_server.py) the
 * env server asks whether a target is reachable (`env.preview_reach`), plans collision-free paths
 * with (`env.plan_motion`, refused when none exists) and checks the arm against the scene with
 * before each servo segment (`env.check_motion`, utils/motion.py). The ik server's `--backend`
 * (pyroki, or curobo for collision-free trajectory optimisation) is chosen where it is started.
 * Off when empty; the robot then behaves as before.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type TSchema, Type } from "typebox";
import type { Json } from "../robot.ts";

/** `env.preview_reach`: `unknown` means the check could not run, which is not approval. */
export type Reach = {
	status: "reachable" | "unreachable" | "unknown";
	reachable: boolean | null;
	q: number[] | null;
	position_err: number | null;
	orientation_err: number | null;
	message: string;
	target: { frame: string; pos: number[]; quat_xyzw: number[] } | null;
};

/** `env.plan_motion`: `unknown` means no ik service answered (the move runs unplanned). */
export type MotionPlan = {
	status: "planned" | "blocked" | "unknown";
	message: string;
	/** World TCP poses (xyz + xyzw) to servo through, the goal last. */
	waypoints: number[][];
	path_m: number | null;
};

/** `env.check_motion`: `contact` means stop before the next segment. */
export type MotionCheck = {
	status: "clear" | "contact" | "unknown";
	message: string;
	min_clearance_m: number | null;
};

export function registerIkFlag(pi: ExtensionAPI) {
	pi.registerFlag("ik", {
		type: "string",
		default: "",
		description:
			"IK server (components/ik_server.py): reach checks, collision-free planned moves and a collision check before each segment; off when empty",
	});
}

/** The env server arguments for the flag: `--ik <url>`, or nothing when it is off. */
export function ikArgs(url: unknown): string[] {
	const u = typeof url === "string" ? url.trim() : "";
	return u ? ["--ik", u] : [];
}

/** The refusal a motion tool returns instead of moving, or undefined when the target may be tried. */
export function reachRefusal(reach: Reach): string | undefined {
	return reach.status === "unreachable" ? `refused: target is out of reach (${reach.message})` : undefined;
}

/** `preview_reach` over the env server's `env.preview_reach` (read-only); `call` carries the tool's signal. */
export function previewReachTool(call: (kwargs: Json) => Promise<Reach>, frame: string) {
	return {
		name: "preview_reach",
		description: `Whether the arm could reach a ${frame} xyz from its current joints (IK only; nothing moves), keeping its orientation unless quat_xyzw is given. status unreachable means a move there would fail or be refused; unknown means the check could not run.`,
		parameters: Type.Object({
			xyz: Type.Array(Type.Number(), { minItems: 3, maxItems: 3 }),
			quat_xyzw: Type.Optional(Type.Array(Type.Number(), { minItems: 4, maxItems: 4 })),
		}) as TSchema,
		run: async (p: unknown) => {
			const { xyz, quat_xyzw } = p as { xyz: number[]; quat_xyzw?: number[] };
			return (await call({ pos: xyz, quat_xyzw: quat_xyzw ?? null })) as unknown as Json;
		},
	};
}

/** The refusal a planned move returns instead of moving (no collision-free path), or undefined. */
export function planRefusal(plan: MotionPlan): string | undefined {
	return plan.status === "blocked" ? `refused: ${plan.message}` : undefined;
}
