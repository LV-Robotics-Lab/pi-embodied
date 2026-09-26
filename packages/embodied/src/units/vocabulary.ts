/**
 * The units vocabulary (core/action_units.py) and the pure helpers around it: grounding a unit into a
 * base-frame move, the rotation plugin's compensation, the finish sequence a robot may run past its
 * success signal, and the stateless context filter. ./index.ts mounts them as tools.
 *
 * Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: see ./index.ts.
 */

import type { UnitsSpec } from "./types.ts";

// ---------------------------------------------------------------------------
// the vocabulary (core/action_units.py)

export const MOVE_UNITS = ["MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN"] as const;
export const ROTATE_UNITS = ["ROTATE_CW", "ROTATE_CCW"] as const;
/**
 * The RT_* units of the aaroncaozj LIBERO adapters (Show-Harness v5 vocabulary): a fixed turn about a
 * base-frame axis through the TCP. Offered with --units-rt on robots that declare the axis (`rt`).
 */
export const RT_UNITS = [
	"RT_ROLL_LEFT",
	"RT_ROLL_RIGHT",
	"RT_PITCH_FWD",
	"RT_PITCH_BACK",
	"RT_YAW_CW",
	"RT_YAW_CCW",
] as const;
/** STOP holds the setpoint for one step; STILL (dual arm) leaves an arm alone; DONE ends the task. */
export const UNITS = [
	...MOVE_UNITS,
	...ROTATE_UNITS,
	...RT_UNITS,
	"STOP",
	"GRASP",
	"RELEASE",
	"DONE",
	"STILL",
] as const;
export type MoveUnit = (typeof MOVE_UNITS)[number];
export type RtUnit = (typeof RT_UNITS)[number];
export type RtAxis = "roll" | "pitch" | "yaw";
/** Each RT_* unit's axis and sign: +1 turns about the robot's `rt.axes` vector by the right-hand rule. */
export const RT_TURNS: Record<RtUnit, { axis: RtAxis; sign: 1 | -1 }> = {
	RT_ROLL_LEFT: { axis: "roll", sign: 1 },
	RT_ROLL_RIGHT: { axis: "roll", sign: -1 },
	RT_PITCH_FWD: { axis: "pitch", sign: 1 },
	RT_PITCH_BACK: { axis: "pitch", sign: -1 },
	RT_YAW_CCW: { axis: "yaw", sign: 1 },
	RT_YAW_CW: { axis: "yaw", sign: -1 },
};
export const isRt = (u: string): u is RtUnit => u in RT_TURNS;
export type Unit = (typeof UNITS)[number];
export type Vec3 = [number, number, number];

export const PLUGINS = [
	"recovery",
	"auto_release",
	"proprioception",
	"variable_step",
	"action_chunk",
	"rotation",
	"plan",
	"point",
	"mem_text",
] as const;
export type Plugin = (typeof PLUGINS)[number];
/**
 * Show-Harness configs/robot_franka.yaml (zero-shot, mem_text on), plus rotation: here ROTATE_* are offered
 * whenever the robot has a yaw step, and the plugin keeps wrist-judged moves right after a turn.
 * Affordance (point) is off there too.
 */
export const DEFAULT_PLUGINS: readonly Plugin[] = [
	"recovery",
	"auto_release",
	"proprioception",
	"variable_step",
	"action_chunk",
	"rotation",
	"plan",
	"mem_text",
];

/**
 * One grounded unit: a base-frame translation (m), a yaw about base +z (rad), a gripper command.
 * `continuous`: the same MV_* unit runs next in this act call, so the robot may flow through the join.
 */
export type Move = {
	delta: Vec3;
	yaw: number;
	gripper: "open" | "close" | null;
	/** An RT_* turn: a rotation vector (axis x angle, rad) about base-frame axes through the TCP. */
	rot?: Vec3;
	arm?: string;
	continuous?: boolean;
	/** view_select: the view that guided the move (act's `view`, robots with `viewSelect`). */
	view?: GuideView;
	/**
	 * The verifier's retreat lift before its check. A robot that ends motion at its own success
	 * signal may still allow it (see finishMove; it cannot change a latched outcome).
	 */
	retreat?: boolean;
};
/**
 * The finish sequence a robot may still run after its success signal: opening the gripper and
 * lifting straight up (Show-Harness RELEASE -> RETREAT), and the verifier's retreat. It reads only
 * the move, so a replay that skips a refused `finish` runs the same steps.
 */
export function finishMove(move: Move) {
	const [dx, dy, dz] = move.delta;
	if (move.rot?.some(Boolean)) return move.retreat === true;
	return (
		move.retreat === true ||
		(move.gripper !== "close" && !move.yaw && !dx && !dy && (move.gripper === "open" || dz > 0))
	);
}
/** Show-Harness plugins/view_select: WRIST (rule A, the wrist view) or FRONT (rule B, the third-person view). */
export const GUIDE_VIEWS = ["WRIST", "FRONT"] as const;
export type GuideView = (typeof GUIDE_VIEWS)[number];
export type State = Record<string, unknown>;

/** Largest repeat count per `act` call. */
export const MAX_REPEAT = 10;
/** action_chunk: the most moves one call may commit (action_chunk_step_num). */
export const CHUNK_STEPS = 3;
export const isMove = (u: string): u is MoveUnit => (MOVE_UNITS as readonly string[]).includes(u);

/** Ground a unit into a move (the interpreters' job), or undefined for units that do not move. */
export function ground(
	spec: Pick<UnitsSpec, "vectors" | "stepM" | "yawStepRad" | "rt">,
	unit: Unit,
	stepM = spec.stepM,
): Move | undefined {
	if (isMove(unit)) return { delta: spec.vectors[unit].map((x) => x * stepM) as Vec3, yaw: 0, gripper: null };
	if (isRt(unit)) {
		const { axis, sign } = RT_TURNS[unit];
		const rt = spec.rt;
		const v = rt?.axes[axis];
		if (!rt || !v) throw new Error(`${unit}: this robot cannot turn its gripper about the ${axis} axis`);
		return { delta: [0, 0, 0], yaw: 0, gripper: null, rot: v.map((x) => x * sign * rt.stepRad) as Vec3 };
	}
	if (unit === "ROTATE_CW" || unit === "ROTATE_CCW") {
		if (!spec.yawStepRad) throw new Error(`${unit}: this robot has no yaw`);
		return { delta: [0, 0, 0], yaw: unit === "ROTATE_CW" ? spec.yawStepRad : -spec.yawStepRad, gripper: null };
	}
	if (unit === "GRASP" || unit === "RELEASE")
		return { delta: [0, 0, 0], yaw: 0, gripper: unit === "GRASP" ? "close" : "open" };
	if (unit === "STOP") return { delta: [0, 0, 0], yaw: 0, gripper: null };
	return undefined;
}

/** rotation: a base-frame vector rotated about +z by `yaw` (plugins/rotation compensate_move). */
export function compensate(delta: Vec3, yaw: number): Vec3 {
	if (Math.abs(yaw) < 1e-6) return delta;
	const [c, s] = [Math.cos(yaw), Math.sin(yaw)];
	return [c * delta[0] - s * delta[1], s * delta[0] + c * delta[1], delta[2]];
}

/**
 * The paper's no-history context: the first user message (the task) and the latest observation
 * turn, i.e. everything from the last assistant message whose tool results carry an image.
 */
export function latestTurn<M extends { role: string; content?: unknown }>(messages: M[]): M[] | undefined {
	const hasImage = (m: M) =>
		m.role === "toolResult" &&
		Array.isArray(m.content) &&
		m.content.some((c: { type?: string }) => c?.type === "image");
	let last = -1;
	for (let i = messages.length - 1; i >= 0 && last < 0; i--) {
		if (!hasImage(messages[i])) continue;
		for (let j = i - 1; j >= 0; j--)
			if (messages[j].role === "assistant") {
				last = j;
				break;
			}
	}
	const first = messages.findIndex((m) => m.role === "user");
	if (last < 0 || first < 0 || first >= last) return undefined;
	const kept = [messages[first], ...messages.slice(last)];
	return kept.length === messages.length ? undefined : kept;
}
