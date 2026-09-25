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

/** A seeder as a robot uses it: the session start is its first reset. */
const started = (values: Record<string, unknown> = {}, episode: () => unknown = () => ["libero", task]) => {
	const seeds = vlaSeeds(fakePi(values), episode);
	seeds.reset();
	return seeds;
};

test("episode seeds are a stable function of robot, task and call index", () => {
	// Golden values: a change here breaks replays of every seeded recording.
	assert.deepEqual(run(started(), 3), [179200059, 1168145401, 1335869723]);
	assert.deepEqual(
		run(
			started({ "vla-seed": "episode" }, () => ["libero", { ...task }]),
			3,
		),
		[179200059, 1168145401, 1335869723],
	);
	assert.notEqual(started({}, () => ["libero", { ...task, seed: "2" }]).next(), 179200059);
	for (const s of run(
		started({}, () => ["robotwin", task]),
		50,
	))
		assert.ok(s! >= 0 && s! <= 0x7fffffff);
});

test("a reset starts a new attempt: new seeds, and the same session's resets repeat them", () => {
	const session = () => {
		const seeds = started();
		const attempts = [run(seeds, 4)];
		seeds.reset(); // e.g. exploration's reset
		attempts.push(run(seeds, 4));
		seeds.reset();
		attempts.push(run(seeds, 2));
		return attempts;
	};
	const [a0, a1, a2] = session();
	assert.deepEqual(a0, [179200059, 1168145401, 1335869723, a0[3]]);
	assert.equal(new Set([...a0, ...a1, ...a2]).size, 10);
	assert.deepEqual(session(), [a0, a1, a2]); // a replay repeats every seed
});

test("an integer base counts up per call and jumps per attempt; off sends no seed", () => {
	const base = started({ "vla-seed": "1000" });
	assert.deepEqual(run(base, 3), [1000, 1001, 1002]);
	base.reset();
	assert.deepEqual(run(base, 2), [1_001_000, 1_001_001]);
	assert.deepEqual(run(started({ "vla-seed": "off" }), 2), [undefined, undefined]);
	assert.throws(() => started({ "vla-seed": "-3" }, () => []).next(), /--vla-seed/);
});
