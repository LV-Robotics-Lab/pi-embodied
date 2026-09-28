/**
 * The real arms' motion (../robots/franka, dual_franka, piper, ur5e): pi's per-call limits and the
 * bounded TCP tools of the Franka robots.
 *
 * Limits (PARAMS.md §6.3): the env server enforces pi's --max-move / --max-rotate / --max-yaw /
 * --workspace-xy / --z-floor in its motion methods, for pi's tools and a program's calls alike. pi
 * passes them at spawn (`limitArgs`) and reads back what the server enforces
 * (`env.get_env_meta().motion_limits`); an attached server (--robot-env) whose limits are looser
 * than pi's flags is refused (`servedLimits`). pi checks nothing itself, except a route of several
 * calls (follow_waypoints, a geometry plan) that must be refused whole before its first call.
 *
 * Tools: `move_delta`, `rotate_delta`, `open_gripper` and `close_gripper` over env.move_delta,
 * env.rotate_delta, env.open_gripper and env.close_gripper; their schemas and descriptions are the
 * robot's manifest entries (../primitives/manifests/<robot>.json). The robot supplies its operator
 * gate, the env call and, with two arms, the arm's validation.
 */

import { Type } from "typebox";
import { NdArray } from "../infra/rpc.ts";
import { type Json, round, vec } from "../robot.ts";
import { toolDef } from "./steps.ts";

export const xyz = Type.Array(Type.Number(), { minItems: 3, maxItems: 3 });

/** Three finite numbers as a float32 vector; anything else throws, naming the parameter. */
export function vec3(v: unknown, name: string): NdArray {
	const a = vec(v);
	if (a.length !== 3) throw new Error(`${name} must contain exactly 3 values, got (${a.length},)`);
	if (!a.every(Number.isFinite)) throw new Error(`${name} must contain only finite values`);
	return NdArray.f32(a);
}

/** pi's per-call limits, as the server takes them (services utils/code_real.py LIMIT_FLAGS). */
export type MotionLimits = {
	max_move_m?: number | null;
	max_rotate_rad?: number | null;
	max_yaw_rad?: number | null;
	z_floor_m?: number | null;
	workspace_xy?: number[] | null;
};
const FLAGS: Record<keyof MotionLimits, string> = {
	max_move_m: "--max-move",
	max_rotate_rad: "--max-rotate",
	max_yaw_rad: "--max-yaw",
	z_floor_m: "--z-floor",
	workspace_xy: "--workspace-xy",
};

/** The env server's arguments for pi's limits (a limit left undefined or null is not passed). */
export function limitArgs(l: MotionLimits): string[] {
	return (Object.keys(FLAGS) as (keyof MotionLimits)[]).flatMap((k) => {
		const v = l[k];
		if (v === undefined || v === null) return [];
		return [FLAGS[k], Array.isArray(v) ? v.join(",") : String(v)];
	});
}

/** Why `served` (the server's `motion_limits`) is looser than `wanted` in `k`, else undefined. */
function looser(k: keyof MotionLimits, served: MotionLimits, wanted: MotionLimits): string | undefined {
	const w = wanted[k];
	if (w === undefined || w === null) return undefined;
	const s = served[k];
	if (s === undefined || s === null) return `${FLAGS[k]} is off`;
	if (k === "workspace_xy") {
		const [a, b] = [s as number[], w as number[]];
		return a[0] < b[0] - 1e-9 || a[1] > b[1] + 1e-9 || a[2] < b[2] - 1e-9 || a[3] > b[3] + 1e-9
			? `${FLAGS[k]} is ${a.join(",")}`
			: undefined;
	}
	if (k === "z_floor_m") return (s as number) < (w as number) - 1e-9 ? `${FLAGS[k]} is ${s}` : undefined;
	return (s as number) > (w as number) + 1e-9 ? `${FLAGS[k]} is ${s}` : undefined;
}

/**
 * The limits the server enforces (`env.get_env_meta().motion_limits`), checked against pi's
 * `wanted` ones: a server without them, or looser in any, throws (an attached server must have been
 * started with pi's flags or tighter ones).
 */
export function servedLimits(served: unknown, wanted: MotionLimits): MotionLimits {
	if (!served || typeof served !== "object")
		throw new Error(
			`the env server enforces none of pi's per-call limits (no motion_limits): start it with ${limitArgs(wanted).join(" ")}`,
		);
	const s = served as MotionLimits;
	const off = (Object.keys(FLAGS) as (keyof MotionLimits)[])
		.map((k) => looser(k, s, wanted))
		.filter((v): v is string => v !== undefined);
	if (off.length)
		throw new Error(
			`the env server's limits are looser than pi's (${off.join("; ")}): start it with ${limitArgs(wanted).join(" ")}`,
		);
	return s;
}

/** The TCP route of several motion calls, checked whole against the served limits before its first call. */
export function checkRoute(start: number[], deltas: number[][], limits: MotionLimits, where = "") {
	let p = start.slice(0, 3);
	for (const d of deltas) {
		const norm = Math.hypot(...d);
		if (limits.max_move_m != null && !(norm <= limits.max_move_m + 1e-9))
			throw new Error(
				`a segment moves ${round(norm, 4)} m; the limit is ${limits.max_move_m} m per call. Split the motion into smaller calls.`,
			);
		const q = p.map((v, i) => v + d[i]);
		const box = limits.workspace_xy ?? undefined;
		const floor = limits.z_floor_m ?? undefined;
		const out = (x: number[]) =>
			(box ? Math.max(0, box[0] - x[0], x[0] - box[1]) + Math.max(0, box[2] - x[1], x[1] - box[3]) : 0) +
			(floor === undefined ? 0 : Math.max(0, floor - x[2]));
		if (out(q) > 1e-6 && out(q) >= out(p) - 1e-6)
			throw new Error(
				`the move ends at [${q.map((v) => round(v, 3))}]${where}, outside the workspace (${box ? `x ${box[0]}..${box[1]}, y ${box[2]}..${box[3]}, ` : ""}z >= ${floor} m; --workspace-xy / --z-floor)`,
			);
		p = q;
	}
}

/** Refuse a rotation beyond `limit` (norm of delta_rpy, rad): a route's pre-check. */
export function checkRotate(rpy: number[], limit: number | null | undefined) {
	const norm = Math.hypot(...rpy);
	if (limit != null && !(norm <= limit + 1e-9))
		throw new Error(
			`delta_rpy rotates ${round(norm, 4)} rad; the limit is ${limit} rad per call. Split the rotation into smaller calls.`,
		);
}

/** What differs per robot in the motion tools. */
export type MotionRig = {
	/** The operator gate and abort check before any motion (throws to refuse). */
	check: (signal?: AbortSignal) => void;
	/** The robot's env motion call; its result is the tool's. */
	motion: (method: string, kwargs: Json, signal?: AbortSignal) => Promise<Json>;
	/** Two arms: the validated arm name of the `arm` parameter. */
	arm?: (v: unknown) => string;
};

const armKw = (rig: MotionRig, p: Json): Json => (rig.arm ? { arm: rig.arm(p.arm) } : {});

/** `move_delta`: env.move_delta (the server holds it to pi's limits). */
export function moveDelta(rig: MotionRig) {
	return toolDef("move_delta", "", Type.Object({}), async (p: Json, signal) => {
		rig.check(signal);
		return rig.motion("env.move_delta", { ...armKw(rig, p), delta_xyz: vec3(p.delta_xyz, "delta_xyz") }, signal);
	});
}

/** `rotate_delta`: env.rotate_delta (the server holds it to pi's limits). */
export function rotateDelta(rig: MotionRig) {
	return toolDef("rotate_delta", "", Type.Object({}), async (p: Json, signal) => {
		rig.check(signal);
		return rig.motion("env.rotate_delta", { ...armKw(rig, p), delta_rpy: vec3(p.delta_rpy, "delta_rpy") }, signal);
	});
}

/** `open_gripper` / `close_gripper`: env.open_gripper / env.close_gripper, waiting for the fingers to settle. */
export function setGripper(rig: MotionRig, open: boolean) {
	const name = open ? "open_gripper" : "close_gripper";
	return toolDef(name, "", Type.Object({}), async (p: Json, signal) => {
		rig.check(signal);
		return rig.motion(`env.${name}`, armKw(rig, p), signal);
	});
}
