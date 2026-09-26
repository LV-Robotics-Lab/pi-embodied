import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
	calibrate,
	ENV_IDS,
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

test("probe-axes: a short unit is scaled up to stepM, a mirrored axis is refused", () => {
	const probes = MOVE_UNITS.map((unit) => ({
		unit,
		n: 4,
		moved: VECTORS[unit].map((x) => x * 0.018 * 4) as [number, number, number],
	}));
	const c = calibrate(VECTORS, STEP_M, probes);
	for (const unit of MOVE_UNITS) {
		const move = ground({ vectors: c.vectors, stepM: STEP_M }, unit)!;
		assert.ok(
			close(
				move.delta,
				VECTORS[unit].map((x) => (x * STEP_M * STEP_M) / 0.018),
			),
		);
	}
	assert.equal((c.units.MV_FWD as { per_unit_m: number }).per_unit_m, 0.018);
	const mirrored = probes.map((p) =>
		p.unit === "MV_LEFT" ? { ...p, moved: [0, 0.08, 0] as [number, number, number] } : p,
	);
	assert.throws(() => calibrate(VECTORS, STEP_M, mirrored), /MV_LEFT/);
});

test("--env-id: the RLinf rigs first (BlockPAP-v1 the default), the eight original ids unchanged, then OpenETA's tasks", () => {
	assert.deepEqual(ENV_IDS.slice(0, 8), [
		"BlockPAP-v1",
		"BlockStack-v1",
		"PickCube-v1",
		"StackCube-v1",
		"PushCube-v1",
		"PullCube-v1",
		"PokeCube-v1",
		"LiftPegUpright-v1",
	]);
	assert.deepEqual(ENV_IDS.slice(8), [
		"PlaceSphere-v1",
		"StackPyramid-v1",
		"PullCubeTool-v1",
		"PegInsertionSide-v1",
		"PlugCharger-v1",
		"PickSingleYCB-v1",
	]);
	// The same list as the env server's ENV_IDS (its INSTRUCTIONS keys after the rigs).
	const py = readFileSync(
		new URL("../../../services/pi_embodied_services/robots/maniskill/env_server.py", import.meta.url),
		"utf8",
	);
	const table = py.slice(py.indexOf("INSTRUCTIONS = {"), py.indexOf("ENV_IDS = "));
	const ids = [...table.matchAll(/^ {4}"([A-Za-z0-9-]+)": "/gm)].map((m) => m[1]);
	assert.deepEqual(ids, ENV_IDS.slice(2));
	assert.equal(new Set(ENV_IDS).size, ENV_IDS.length);
});
