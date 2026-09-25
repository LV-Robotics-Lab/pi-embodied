import assert from "node:assert/strict";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { vlaSeeds } from "../src/vla-seed.ts";

/** A stub pi holding flags like pi's runner: registered defaults, overridden by `values`. */
function fakePi(values: Record<string, unknown> = {}) {
	const flags: Record<string, unknown> = {};
	return {
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
	} as unknown as ExtensionAPI;
}

const task = { suite: "libero_object_swap", task: "0", seed: "1" };
const run = (seeds: { next(): number | undefined }, n: number) => Array.from({ length: n }, () => seeds.next());

test("episode seeds are a stable function of robot, task and call index", () => {
	const a = vlaSeeds(fakePi(), () => ["libero", task]);
	// Golden values: a change here breaks replays of every seeded recording.
	assert.deepEqual(run(a, 3), [179200059, 1168145401, 1335869723]);
	const b = vlaSeeds(fakePi({ "vla-seed": "episode" }), () => ["libero", { ...task }]);
	assert.deepEqual(run(b, 3), [179200059, 1168145401, 1335869723]);
	const other = vlaSeeds(fakePi(), () => ["libero", { ...task, seed: "2" }]);
	assert.notEqual(other.next(), 179200059);
	for (const s of run(
		vlaSeeds(fakePi(), () => ["robotwin", task]),
		50,
	))
		assert.ok(s! >= 0 && s! <= 0x7fffffff);
});

test("the call index restarts on reset", () => {
	const seeds = vlaSeeds(fakePi(), () => ["libero", task]);
	const first = run(seeds, 4);
	seeds.reset();
	assert.deepEqual(run(seeds, 4), first);
	assert.equal(new Set(first).size, 4);
});

test("an integer base counts up from the base; off sends no seed", () => {
	const base = vlaSeeds(fakePi({ "vla-seed": "1000" }), () => ["libero", task]);
	assert.deepEqual(run(base, 3), [1000, 1001, 1002]);
	base.reset();
	assert.equal(base.next(), 1000);
	assert.deepEqual(
		run(
			vlaSeeds(fakePi({ "vla-seed": "off" }), () => ["libero", task]),
			2,
		),
		[undefined, undefined],
	);
	assert.throws(() => vlaSeeds(fakePi({ "vla-seed": "-3" }), () => []).next(), /--vla-seed/);
});
