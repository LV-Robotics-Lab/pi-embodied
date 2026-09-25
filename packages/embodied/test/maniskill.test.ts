import assert from "node:assert/strict";
import { test } from "node:test";
import {
	GRIPPER_STEPS,
	grasped,
	phases,
	SERVO,
	STEP_M,
	sideBySide,
	VECTORS,
	waypoints,
} from "../src/maniskill/index.ts";
import { NdArray } from "../src/rpc.ts";
import { ground, MOVE_UNITS } from "../src/units/index.ts";

const close = (a: number[], b: number[]) => a.every((x, k) => Math.abs(x - b[k]) < 1e-9);

test("each MV_* unit is one ~2 cm decision along the yaml's base-frame vector", () => {
	for (const unit of MOVE_UNITS) {
		const move = ground({ vectors: VECTORS, stepM: STEP_M }, unit)!;
		const w = waypoints([0.1, 0.2, 0.3], move.delta);
		assert.equal(w.length, 1);
		assert.ok(
			close(
				w[0],
				[0.1, 0.2, 0.3].map((p, k) => p + VECTORS[unit][k] * 0.02),
			),
			`${unit} ${w[0]}`,
		);
	}
	assert.deepEqual(VECTORS.MV_LEFT, [0, -1, 0]); // configs/robot_maniskill.yaml: robot-left = -Y
});

test("a long move splits into evenly spaced ~2 cm waypoints; a gripper command or STOP holds in place", () => {
	const w = waypoints([0, 0, 0.2], [0.05, 0, -0.03]);
	assert.equal(w.length, 3); // |delta| 5.8 cm -> 3 decisions
	assert.ok(close(w[2], [0.05, 0, 0.17]));
	assert.ok(close(w[0], [0.05 / 3, 0, 0.19]));
	assert.deepEqual(waypoints([0, 0, 0.2], [0, 0, 0]), [[0, 0, 0.2]]);
});

test("the episode video frame puts the agentview and the wrist view side by side", () => {
	const a = new NdArray("uint8", [2, 1, 3], Buffer.from([1, 1, 1, 2, 2, 2]));
	const b = new NdArray("uint8", [2, 2, 3], Buffer.from([5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]));
	const f = sideBySide(a, b);
	assert.deepEqual(f.shape, [2, 3, 3]);
	assert.deepEqual([...f.data], [1, 1, 1, 5, 5, 5, 6, 6, 6, 2, 2, 2, 7, 7, 7, 8, 8, 8]);
});

test("the grasp flag is read from whichever is_*grasped key the scene reports", () => {
	assert.equal(grasped({ is_grasped: true }), true);
	assert.equal(grasped({ is_cubeA_grasped: 1, success: false }), true);
	assert.equal(grasped({ is_cubeA_grasped: false, is_obj_placed: true }), false);
});

test("move_delta with a gripper change: the fingers settle first, holding still, then the arm moves", () => {
	const start = [0.1, 0, 0.1];
	const p = phases(start, [0, 0, 0.04], true);
	// A GRIPPER_STEPS hold at the start, then the two ~2 cm waypoints of the move with the new command.
	assert.deepEqual(p[0], { target: start, minSteps: GRIPPER_STEPS, maxSteps: GRIPPER_STEPS });
	assert.equal(p.length, 3);
	assert.ok(close(p[2].target, [0.1, 0, 0.14]));
	assert.deepEqual([p[1].minSteps, p[1].maxSteps], [SERVO.minSteps, SERVO.maxSteps]);
	// A pure gripper toggle is the hold alone; a move without a change has no hold; STOP holds one decision.
	assert.deepEqual(phases(start, [0, 0, 0], true), [p[0]]);
	assert.equal(phases(start, [0, 0, 0.04], false).length, 2);
	assert.deepEqual(phases(start, [0, 0, 0], false), [
		{ target: start, minSteps: SERVO.minSteps, maxSteps: SERVO.minSteps },
	]);
});
