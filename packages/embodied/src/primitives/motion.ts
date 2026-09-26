/**
 * Bounded TCP motion of the real Franka robots (../franka, ../dual_franka), once: `move_delta`,
 * `rotate_delta`, `open_gripper` and `close_gripper` over the services' env.* RPC (env.move_delta,
 * env.rotate_delta, env.set_gripper). The robot supplies its per-call limits, its workspace check,
 * its operator gate, the env call and, with two arms, the `arm` parameter; a mounted tool then
 * calls that robot's own env.
 */

import { type TSchema, Type } from "typebox";
import { checkMove, type Json, numbers, round, vec } from "../robot.ts";
import { NdArray } from "../rpc.ts";
import { toolDef } from "./steps.ts";

export const xyz = Type.Array(Type.Number(), { minItems: 3, maxItems: 3 });

/** Three finite numbers as a float32 vector; anything else throws, naming the parameter. */
export function vec3(v: unknown, name: string): NdArray {
	const a = vec(v);
	if (a.length !== 3) throw new Error(`${name} must contain exactly 3 values, got (${a.length},)`);
	if (!a.every(Number.isFinite)) throw new Error(`${name} must contain only finite values`);
	return NdArray.f32(a);
}

/** Refuse a rotation beyond `limit` (norm of delta_rpy, rad). */
export function checkRotate(rpy: number[], limit: number) {
	const norm = Math.hypot(...rpy);
	if (!(norm <= limit))
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
	/** Per-call limits: translation in m, rotation in rad (norm). */
	maxMove: () => number;
	maxRotate: () => number;
	/** The task's documented constraints; a tighter documented translation limit applies (`checkMove`). */
	constraints: () => string[] | undefined;
	/** Refuse a move whose target leaves the workspace (throws); `arm` with two arms. */
	workspace: (delta: number[], arm: string | undefined) => void;
	/** Two arms: the `arm` parameter's schema and the validated arm name of its value. */
	arm?: { schema: TSchema; name: (v: unknown) => string };
};

const armProp = (rig: MotionRig): Record<string, TSchema> => (rig.arm ? { arm: rig.arm.schema } : {});
const armOf = (rig: MotionRig, p: Json) => (rig.arm ? rig.arm.name(p.arm) : undefined);
const armKw = (arm: string | undefined): Json => (arm === undefined ? {} : { arm });

export function moveDelta(rig: MotionRig, description: string) {
	return toolDef("move_delta", description, Type.Object({ ...armProp(rig), delta_xyz: xyz }), async (p, signal) => {
		rig.check(signal);
		const delta = vec3(p.delta_xyz, "delta_xyz");
		checkMove(vec(p.delta_xyz), rig.maxMove(), rig.constraints());
		const arm = armOf(rig, p);
		rig.workspace(vec(p.delta_xyz), arm);
		return rig.motion("env.move_delta", { ...armKw(arm), delta_xyz: delta }, signal);
	});
}

export function rotateDelta(rig: MotionRig, description: string) {
	return toolDef("rotate_delta", description, Type.Object({ ...armProp(rig), delta_rpy: xyz }), async (p, signal) => {
		rig.check(signal);
		const delta = vec3(p.delta_rpy, "delta_rpy");
		checkRotate(numbers(delta), rig.maxRotate());
		return rig.motion("env.rotate_delta", { ...armKw(armOf(rig, p)), delta_rpy: delta }, signal);
	});
}

/** `open_gripper` (open) or `close_gripper`: env.set_gripper, waiting for the command to settle. */
export function setGripper(rig: MotionRig, open: boolean, description: string) {
	return toolDef(
		open ? "open_gripper" : "close_gripper",
		description,
		Type.Object({ ...armProp(rig) }),
		async (p, signal) => {
			rig.check(signal);
			return rig.motion("env.set_gripper", { ...armKw(armOf(rig, p)), open }, signal);
		},
	);
}
