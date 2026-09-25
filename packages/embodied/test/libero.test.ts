import assert from "node:assert/strict";
import { test } from "node:test";
import { flywheelSuite } from "../src/flywheel.ts";
import { latchSuccess, memoryTag } from "../src/libero/index.ts";
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

test("LIBERO memory cells keep standard/pro tags and give LIBERO-plus its own", () => {
	// Standard and pro share cells (identical task sets); existing corpora stay valid.
	assert.equal(memoryTag("libero_spatial", "0", "0", "standard"), "spatial_t0_s0");
	assert.equal(memoryTag("libero_spatial", "0", "0", "pro"), "spatial_t0_s0");
	assert.equal(memoryTag("libero_10_swap", "3", "7", "pro"), "10_swap_t3_s7");
	// Plus task 0 of libero_spatial is a table-texture variant, not standard task 0.
	const plus = memoryTag("libero_spatial", "0", "0", "plus");
	assert.equal(plus, "spatial_plus_t0_s0");
	// The memory guard owns `<tag>`, `<tag>.*` and `<tag>_*`: neither cell may own the other's files.
	const std = memoryTag("libero_spatial", "0", "0", "pro");
	for (const [a, b] of [
		[std, plus],
		[plus, std],
	])
		assert.ok(!b.startsWith(`${a}_`) && !b.startsWith(`${a}.`) && a !== b, `${a} owns ${b}`);
	assert.equal(memoryTag("libero_10", "2401", "0", "plus"), "10_plus_t2401_s0");
});

test("LIBERO-plus Flywheel episodes get their own suite key", () => {
	assert.equal(flywheelSuite("libero_spatial", "pro"), "libero_spatial");
	assert.equal(flywheelSuite("libero_spatial", "standard"), "libero_spatial");
	assert.equal(flywheelSuite("libero_spatial", "plus"), "libero_spatial_plus");
});
