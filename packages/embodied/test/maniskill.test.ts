import assert from "node:assert/strict";
import { test } from "node:test";
import { STEP_M, VECTORS, waypoints } from "../src/maniskill/index.ts";
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
