/**
 * `execute_grasp` / `execute_place` for a robot whose only motion is a bounded Cartesian
 * `move_delta` with its gripper held pointing down (Metaworld's Sawyer, Genesis's Panda): the
 * planned grasp chain of ./grasp.ts (`plan_grasp` -> an id) runs from one resolution of its id,
 * as LIBERO's server-side `execute_grasp` does, with the path split into deltas here.
 *
 * The id is resolved first (`env.resolve_grasp`, read-only) and refused unless its approach points
 * within `maxTiltRad` of straight down, and (a grasp) unless its fingers close within `maxYawRad`
 * (mod pi) of the hand's fixed yaw (`yaw`): the gripper cannot turn, so a tilted or turned grasp
 * would close somewhere else. Then `env.claim_waypoints` fixes the whole path from that one candidate (pre-grasp
 * standoff, grasp, lift; or pre-place, place, retreat) and the robot's `move` runs each leg as
 * deltas of at most `maxStep` metres, holding the leg's gripper command; an in-place gripper step
 * closes or opens. A leg that ends further than `tolM` from its waypoint, or a move that errors,
 * stops the chain (`stalled`). The motions expire the id either way.
 */

import { type TSchema, Type } from "typebox";
import { type Json, message, round } from "../robot.ts";
import type { GraspToolDef } from "./grasp.ts";

export const CHAIN_TOOLS = ["execute_grasp", "execute_place"] as const;
/** Default steepest approach a fixed, downward gripper can take, rad (20 deg). */
export const MAX_TILT_RAD = 0.35;
/** Default largest yaw difference (mod pi) between a grasp and the fixed hand, rad (20 deg). */
export const MAX_YAW_RAD = 0.35;

type Vec = number[];
type Step = { to?: string; gripper?: number };

/** What differs per robot. */
export type ChainRig = {
	/** The env RPC call carrying the running tool's abort signal. */
	call: (method: string, kwargs: Json, timeoutMs?: number) => Promise<Json>;
	/** The EEF position now, in the frame of the planner's waypoints (the world). */
	current: () => Vec;
	/** Longest delta one move may command, m. */
	maxStep: () => number;
	/** One bounded delta with the gripper command held ("open" / "close"); its result (an `error` stops the chain). */
	move: (delta: Vec, gripper: "open" | "close", signal: AbortSignal | undefined) => Promise<Json>;
	/** Close or open in place. */
	gripper: (command: "open" | "close", signal: AbortSignal | undefined) => Promise<Json>;
	/** The tool's final result: the robot's observation around `result`. */
	observe: (result: Json) => Promise<Json> | Json;
	/** A leg counts as reached within this distance, m (default 0.01). */
	tolM?: number;
	/** Steepest approach, rad from straight down (default MAX_TILT_RAD). */
	maxTiltRad?: number;
	/** The hand's fixed yaw (the planner's `eef_yaw` convention), rad; unset skips the yaw check. */
	yaw?: () => number;
	/** Largest grasp yaw difference from it, mod pi (default MAX_YAW_RAD). */
	maxYawRad?: number;
};

const sub = (a: Vec, b: Vec) => a.map((v, i) => v - b[i]);
const norm = (v: Vec) => Math.hypot(...v);

/** The approach's angle from straight down (world -z), rad. */
export function tilt(approach: Vec): number {
	const n = norm(approach);
	if (!(n > 0)) return Math.PI;
	return Math.acos(Math.max(-1, Math.min(1, -approach[2] / n)));
}

/** The difference of two gripper yaws, rad in [0, pi/2]: a parallel gripper turned half is the same. */
export function yawGap(a: number, b: number): number {
	const d = (((a - b) % Math.PI) + Math.PI) % Math.PI;
	return Math.min(d, Math.PI - d);
}

/** A straight leg from `from` to `to` as deltas of at most `step` metres, equal in length. */
export function split(from: Vec, to: Vec, step: number): Vec[] {
	const d = sub(to, from);
	const n = Math.max(1, Math.ceil(norm(d) / step - 1e-9));
	return Array.from({ length: n }, () => d.map((v) => v / n));
}

export function chainTools(rig: ChainRig): GraspToolDef[] {
	const tol = rig.tolM ?? 0.01;
	const maxTilt = rig.maxTiltRad ?? MAX_TILT_RAD;

	async function execute(kind: "grasp" | "place", id: string, standoff: number | undefined, signal?: AbortSignal) {
		const resolved = await rig.call("env.resolve_grasp", { grasp_id: id });
		const t = tilt(resolved.approach as Vec);
		if (t > maxTilt)
			return rig.observe({
				name: `execute_${kind}`,
				id,
				refused: true,
				error: `${id} approaches ${round((t * 180) / Math.PI, 1)} deg from straight down; this gripper only points down (at most ${round((maxTilt * 180) / Math.PI, 1)} deg). Ask plan_grasp for the next candidate (next_after).`,
			});
		const maxYaw = rig.maxYawRad ?? MAX_YAW_RAD;
		const gap = kind === "grasp" && rig.yaw ? yawGap(Number(resolved.eef_yaw), rig.yaw()) : 0;
		if (gap > maxYaw)
			return rig.observe({
				name: `execute_${kind}`,
				id,
				refused: true,
				error: `${id} closes the fingers ${round((gap * 180) / Math.PI, 1)} deg off the hand's fixed direction; this gripper cannot turn (at most ${round((maxYaw * 180) / Math.PI, 1)} deg). Ask plan_grasp for the next candidate (next_after).`,
			});
		const claim = await rig.call("env.claim_waypoints", {
			grasp_id: id,
			...(standoff !== undefined ? { standoff } : {}),
		});
		if (claim.kind !== kind) throw new Error(`${id} is a ${claim.kind} id; use execute_${claim.kind}`);
		const waypoints = claim.waypoints as Record<string, Vec>;
		const legs: Json[] = [];
		let stalled = false;
		for (const step of claim.steps as Step[]) {
			const g = (step.gripper ?? -1) > 0 ? "close" : "open";
			if (!step.to) {
				const r = await rig.gripper(g, signal);
				legs.push({ gripper: g, ...(r.error ? { error: r.error } : {}) });
				if (r.error) {
					stalled = true;
					break;
				}
				continue;
			}
			const target = waypoints[step.to];
			let error: string | undefined;
			for (const delta of split(rig.current(), target, rig.maxStep())) {
				const r = await rig.move(delta, g, signal).catch((err: unknown) => ({ error: message(err) }) as Json);
				if (r.error) {
					error = String(r.error);
					break;
				}
			}
			const dist = norm(sub(target, rig.current()));
			legs.push({ to: step.to, gripper: g, final_dist_m: round(dist, 4), ...(error ? { error } : {}) });
			if (error || dist > tol) {
				stalled = true;
				break;
			}
		}
		return rig.observe({
			name: `execute_${kind}`,
			id,
			legs,
			...(stalled ? { stalled: true, error: "a leg stopped short; plan again from the new observation" } : {}),
			eef_yaw: claim.eef_yaw,
			approach_tilt_deg: round((t * 180) / Math.PI, 1),
			expired_ids: claim.expired_ids ?? [],
		});
	}

	const standoff = Type.Optional(Type.Number({ description: "m back along the approach (default 0.10)" }));
	return [
		{
			name: "execute_grasp",
			description:
				"Execute one planned grasp (a g id of the current observation, from plan_grasp) in one call: open to the pre-grasp standoff back along its approach, descend to it, close, lift straight up, each leg as bounded moves. The gripper cannot turn, so only a candidate approaching from (nearly) straight above with its fingers along the hand's direction runs; a tilted or turned one is refused before anything moves. Returns each leg's final_dist_m; stalled when a leg stopped short. The id is spent either way; afterwards plan_place with this grasp_id plans from the held object.",
			parameters: Type.Object({
				grasp_id: Type.String({ description: "A g id from plan_grasp" }),
				standoff,
			}) as TSchema,
			run: async (p, signal) => {
				const { grasp_id, standoff: s } = p as { grasp_id: string; standoff?: number };
				return execute("grasp", grasp_id, s, signal);
			},
		},
		{
			name: "execute_place",
			description:
				"Execute one planned place (a p id from plan_place) in one call: carry to the pre-place above it, descend, open, retreat, each leg as bounded moves; refused when its approach is not (nearly) straight down.",
			parameters: Type.Object({
				place_id: Type.String({ description: "A p id from plan_place" }),
				standoff,
			}) as TSchema,
			run: async (p, signal) => {
				const { place_id, standoff: s } = p as { place_id: string; standoff?: number };
				return execute("place", place_id, s, signal);
			},
		},
	];
}
