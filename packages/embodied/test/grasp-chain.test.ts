import assert from "node:assert/strict";
import { test } from "node:test";
import { chainTools, split, tilt } from "../src/primitives/grasp-chain.ts";

const down = [0, 0, -1];

test("split: a leg as equal deltas of at most the step; tilt: the approach's angle from straight down", () => {
	const legs = split([0, 0, 0.3], [0, 0.1, 0.05], 0.1);
	assert.equal(legs.length, 3);
	for (const d of legs) assert.ok(Math.hypot(...d) <= 0.1 + 1e-12);
	const sum = legs.reduce((a, d) => a.map((v, i) => v + d[i]), [0, 0, 0]);
	assert.ok(Math.abs(sum[1] - 0.1) < 1e-12 && Math.abs(sum[2] + 0.25) < 1e-12);
	assert.equal(split([0, 0, 0], [0, 0, 0], 0.1).length, 1);
	assert.equal(tilt(down), 0);
	assert.ok(Math.abs(tilt([1, 0, 0]) - Math.PI / 2) < 1e-12);
});

/** A rig whose EEF follows the commanded deltas exactly, unless `stall` caps a leg. */
function rig(approach: number[], o: { stall?: boolean } = {}) {
	let pos = [0, 0.6, 0.3];
	const calls: [string, unknown][] = [];
	const tools = chainTools({
		call: async (method, kwargs) => {
			calls.push([method, kwargs]);
			if (method === "env.resolve_grasp") return { approach };
			return {
				kind: "grasp",
				waypoints: { pre_grasp: [0, 0.6, 0.2], grasp: [0, 0.6, 0.1], lift: [0, 0.6, 0.2] },
				steps: [
					{ to: "pre_grasp", gripper: -1 },
					{ to: "grasp", gripper: -1 },
					{ gripper: 1 },
					{ to: "lift", gripper: 1 },
				],
				eef_yaw: 0,
				expired_ids: ["g1"],
			};
		},
		current: () => pos,
		maxStep: () => 0.04,
		move: async (delta, g) => {
			calls.push(["move", [delta, g]]);
			if (o.stall && g === "close") return { error: "blocked" };
			pos = pos.map((v, i) => v + delta[i]);
			return {};
		},
		gripper: async (g) => {
			calls.push(["gripper", g]);
			return {};
		},
		observe: (result) => ({ observed: result }),
	});
	return {
		tools,
		calls,
		run: (name: string, p: unknown) => tools.find((t) => t.name === name)!.run(p, undefined, {} as never),
	};
}

test("execute_grasp resolves, claims once and runs pre-grasp, grasp, close, lift as bounded moves", async () => {
	const r = rig(down);
	const out = (await r.run("execute_grasp", { grasp_id: "g1", standoff: 0.1 })).observed as any;
	assert.deepEqual(
		r.calls.filter(([m]) => m.startsWith("env.")).map(([m, k]) => [m, k]),
		[
			["env.resolve_grasp", { grasp_id: "g1" }],
			["env.claim_waypoints", { grasp_id: "g1", standoff: 0.1 }],
		],
	);
	const moves = r.calls.filter(([m]) => m === "move").map(([, a]) => a as [number[], string]);
	for (const [d] of moves) assert.ok(Math.hypot(...d) <= 0.04 + 1e-12);
	// Open down to the grasp, close in place, carry up closed.
	assert.deepEqual([...new Set(moves.map(([, g]) => g))], ["open", "close"]);
	assert.deepEqual(
		r.calls.find(([m]) => m === "gripper"),
		["gripper", "close"],
	);
	assert.deepEqual(
		out.legs.map((l: any) => l.to ?? l.gripper),
		["pre_grasp", "grasp", "close", "lift"],
	);
	assert.equal(out.stalled, undefined);
	assert.deepEqual(out.expired_ids, ["g1"]);
});

test("a tilted candidate is refused before the claim; a blocked leg stops the chain as stalled", async () => {
	const tilted = rig([1, 0, -0.2]);
	const refused = (await tilted.run("execute_grasp", { grasp_id: "g2" })).observed as any;
	assert.equal(refused.refused, true);
	assert.match(refused.error, /next_after/);
	assert.ok(!tilted.calls.some(([m]) => m === "env.claim_waypoints" || m === "move"));

	const blocked = rig(down, { stall: true });
	const out = (await blocked.run("execute_grasp", { grasp_id: "g1" })).observed as any;
	assert.equal(out.stalled, true);
	assert.equal(out.legs.at(-1).error, "blocked");
	assert.equal(out.legs.length, 4);
});
