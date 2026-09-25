import assert from "node:assert/strict";
import { test } from "node:test";
import { latchSuccess } from "../src/libero/index.ts";
import { NdArray } from "../src/rpc.ts";
import { finishMove } from "../src/units/index.ts";

test("LIBERO success is latched at its first env step; later steps cannot undo it", () => {
	assert.equal(latchSuccess(undefined, false, 10), undefined);
	assert.equal(latchSuccess(undefined, true, 10), 11, "a single step: the step after the 10 before it");
	// A chunk: the first success within it, counted from the steps before the chunk.
	const chunk = (v: boolean[]) => new NdArray("bool", [v.length], Buffer.from(v.map(Number)));
	assert.equal(latchSuccess(undefined, chunk([false, false, true, true]), 20), 23);
	assert.equal(latchSuccess(undefined, chunk([false, false]), 20), undefined);
	// Once latched, a later step without success (a release, a knock-over) keeps the first success step.
	assert.equal(latchSuccess(23, false, 40), 23);
	assert.equal(latchSuccess(23, chunk([true]), 40), 23);
});

test("after success only opening and lifting straight up still move the LIBERO arm", () => {
	const move = (delta: [number, number, number], gripper: "open" | "close" | null = null, yaw = 0) => ({
		delta,
		yaw,
		gripper,
	});
	assert.ok(finishMove(move([0, 0, 0], "open")), "RELEASE");
	assert.ok(finishMove(move([0, 0, 0.02])), "MV_UP");
	assert.ok(finishMove({ ...move([0.02, 0, 0]), retreat: true }), "the verifier's retreat");
	for (const m of [
		move([0, 0, 0], "close"),
		move([0, 0, -0.02]),
		move([0.02, 0, 0]),
		move([0, 0, 0]),
		move([0, 0, 0], null, 0.15),
	])
		assert.ok(!finishMove(m), JSON.stringify(m));
});
