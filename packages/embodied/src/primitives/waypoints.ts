/**
 * `follow_waypoints` (--waypoints): OpenETA's multi-waypoint trajectory (`follow_eef_trajectory`,
 * sim/mcp_server/server.py): 1-5 world-frame EEF positions executed in order with one latched gripper
 * command, stopping at the first waypoint that errors, is not reached, or ends the episode.
 *
 * The limits are per call and checked on the whole route before anything moves: at most 5 waypoints,
 * every segment (from the current EEF, then waypoint to waypoint) within the robot's per-segment
 * limit (a real arm's per-call translation limit, `checkMove`, with the task's documented limit),
 * the route's total length within `maxPath`, and every waypoint inside the robot's workspace. The
 * robot supplies how one segment moves (`segment`): LIBERO servos to the waypoint, a real Franka
 * sends the segment as one bounded `env.move_delta`.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { checkMove, type Json, message, round, roundAll } from "../robot.ts";
import { optionalTools } from "./optional.ts";
import { type ToolDef, toolDef } from "./steps.ts";

export const MAX_WAYPOINTS = 5;

/** What differs per robot. */
export type WaypointRig = {
	/** The operator gate and abort check before any motion (throws to refuse). */
	check?: (signal?: AbortSignal) => void;
	/** The current EEF position, the route's start. */
	current: () => number[] | Promise<number[]>;
	/** Longest segment, m (a real arm: its per-call translation limit). */
	maxSegment: () => number;
	/** Longest route, m (sum of the segments). */
	maxPath: () => number;
	/** The task's documented constraints; a tighter documented translation limit applies to every segment. */
	constraints?: () => string[] | undefined;
	/** Refuse a waypoint outside the workspace (throws). */
	workspace?: (target: number[]) => void;
	/** false: the gripper keeps its last command (a real arm; open/close are their own tools), so no `gripper` parameter. */
	gripper?: false;
	/** Move from `from` to `to` holding `gripper` (-1 open, +1 closed); `reached` says whether it got there. */
	segment: (
		from: number[],
		to: number[],
		gripper: number,
		signal: AbortSignal | undefined,
	) => Promise<{ reached: boolean; ended?: boolean; error?: string; [k: string]: unknown }>;
};

const xyz = Type.Array(Type.Number(), { minItems: 3, maxItems: 3 });

/** Validate a route (throws with the reason); returns its segment lengths. */
export function checkRoute(start: number[], waypoints: number[][], rig: WaypointRig): number[] {
	if (!(waypoints.length >= 1 && waypoints.length <= MAX_WAYPOINTS))
		throw new Error(`waypoints must hold 1 to ${MAX_WAYPOINTS} positions, got ${waypoints.length}`);
	const lengths: number[] = [];
	let from = start;
	waypoints.forEach((w, i) => {
		if (w.length !== 3 || !w.every(Number.isFinite))
			throw new Error(`waypoint ${i} must be 3 finite numbers [x, y, z] in m`);
		const delta = w.map((v, k) => v - from[k]);
		try {
			checkMove(delta, rig.maxSegment(), rig.constraints?.() ?? []);
		} catch (err) {
			throw new Error(
				`segment ${i}: ${(err as Error).message.replace("Split the motion into smaller calls.", "Add waypoints in between.")}`,
			);
		}
		rig.workspace?.(w);
		lengths.push(Math.hypot(...delta));
		from = w;
	});
	const total = lengths.reduce((a, b) => a + b, 0);
	if (!(total <= rig.maxPath()))
		throw new Error(`the route is ${round(total, 4)} m long; the limit is ${rig.maxPath()} m per call`);
	return lengths;
}

export function followWaypoints(rig: WaypointRig): ToolDef {
	return toolDef(
		"follow_waypoints",
		`Move the EEF through 1-${MAX_WAYPOINTS} world-frame positions in order (orientation and gripper held). Checked before anything moves: each segment and the whole route against the per-call limits, each waypoint against the workspace. Stops at the first waypoint not reached. Use it for short routes around an obstacle (lift, carry, descend) instead of one straight move.`,
		Type.Object({
			waypoints: Type.Array(xyz, {
				minItems: 1,
				maxItems: MAX_WAYPOINTS,
				description: "World-frame [x, y, z] positions in m, in order",
			}),
			...(rig.gripper === false
				? {}
				: {
						gripper: Type.Optional(
							Type.Number({ description: "-1 open (default), +1 closed (hold +1 while carrying)" }),
						),
					}),
		}),
		async (p, signal) => {
			rig.check?.(signal);
			const waypoints = (p.waypoints as number[][]).map((w) => w.map(Number));
			const gripper = Number(p.gripper ?? -1) >= 0 ? 1 : -1;
			const start = [...(await rig.current())];
			let lengths: number[];
			try {
				lengths = checkRoute(start, waypoints, rig);
			} catch (err) {
				return { name: "follow_waypoints", error: message(err), waypoints_completed: 0, moved: false };
			}
			const results: Json[] = [];
			let completed = 0;
			let from = start;
			let stop = "completed";
			for (const [i, w] of waypoints.entries()) {
				rig.check?.(signal);
				let r: Awaited<ReturnType<WaypointRig["segment"]>>;
				try {
					r = await rig.segment(from, w, gripper, signal);
				} catch (err) {
					r = { reached: false, error: message(err) };
				}
				results.push({ waypoint: i, target: roundAll(w), ...r });
				if (r.error) stop = "error";
				else if (!r.reached) stop = "not_reached";
				else if (r.ended) stop = "episode_ended";
				if (r.reached) completed++;
				if (r.error || !r.reached || r.ended) break;
				from = w;
			}
			return {
				name: "follow_waypoints",
				waypoints_requested: waypoints.length,
				waypoints_completed: completed,
				reached_target: completed === waypoints.length,
				stop_reason: stop,
				route_length_m: round(
					lengths.reduce((a, b) => a + b, 0),
					4,
				),
				segments: results,
			};
		},
	);
}

/** --waypoints: `follow_waypoints` on this robot, mounted through `mount` at the first start with the flag on. */
export function waypointsTool(pi: ExtensionAPI, rig: WaypointRig, mount: (d: ToolDef) => void) {
	return optionalTools(
		pi,
		"waypoints",
		"Add follow_waypoints (a 1-5 waypoint EEF route checked against the per-call limits)",
		() => [followWaypoints(rig)],
		mount,
	);
}
