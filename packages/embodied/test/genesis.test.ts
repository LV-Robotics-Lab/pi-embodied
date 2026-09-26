import assert from "node:assert/strict";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import genesis, { CAMERAS, maskPixels, medianPoint, STEP_M, TASKS, VECTORS } from "../src/genesis/index.ts";
import { ground, MOVE_UNITS } from "../src/units/index.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi that records flags, tools and the active set. */
function stubPi(values: Record<string, unknown> = {}) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	let active: string[] = [];
	const api: Record<string, unknown> = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	};
	const pi = new Proxy(api, { get: (t, k: string) => t[k] ?? (() => {}) }) as unknown as ExtensionAPI;
	return { pi, flags, tools, active: () => active };
}

test("the task flag defaults to cube_pick, the only task, and the simulator registers --privileged", () => {
	const f = stubPi();
	genesis(f.pi);
	assert.deepEqual([...TASKS], ["cube_pick"]);
	assert.equal(f.flags.task, "cube_pick");
	assert.equal(f.flags.seed, "0");
	assert.equal(f.flags.backend, "gpu");
	assert.equal("privileged" in f.flags, true);
	assert.equal(f.tools.has("ground_truth_poses"), false, "nothing privileged at load");
	for (const name of [
		"view_env_state",
		"view_camera_meta",
		"segment",
		"back_project",
		"move_delta",
		"gripper",
		"finish",
	])
		assert.ok(f.tools.has(name), name);
	assert.deepEqual([...CAMERAS], ["agentview", "wrist"]);
});

test("each MV_* unit is one 2 cm decision along the base-frame vector; MV_LEFT is -y", () => {
	for (const unit of MOVE_UNITS) {
		const move = ground({ vectors: VECTORS, stepM: STEP_M }, unit)!;
		assert.ok(
			move.delta.every((v, k) => Math.abs(v - VECTORS[unit][k] * 0.02) < 1e-12),
			unit,
		);
	}
	assert.deepEqual(VECTORS.MV_LEFT, [0, -1, 0]);
	assert.deepEqual(VECTORS.MV_UP, [0, 0, 1]);
});

test("segment subsamples the mask evenly and takes the median of the back-projected points", () => {
	const w = 8;
	const data = new Uint8Array(w * w);
	for (let r = 2; r < 6; r++) for (let c = 1; c < 5; c++) data[r * w + c] = 255; // a 4x4 blob
	const m = maskPixels({ width: w, height: w, data }, 4);
	assert.equal(m.n, 16);
	assert.deepEqual(m.centroid, [4, 3]); // median row / col (rounded)
	assert.equal(m.pixels.length, 4);
	assert.deepEqual(m.pixels[0], [2, 1]);
	assert.ok(m.pixels.every(([r, c]) => data[r * w + c] === 255));
	assert.deepEqual(maskPixels({ width: w, height: w, data: new Uint8Array(w * w) }), {
		n: 0,
		centroid: null,
		pixels: [],
	});
	const pts = Array.from({ length: 12 }, (_, i) => [0.5 + i * 0.001, -0.1, 0.02]);
	assert.deepEqual(medianPoint([...pts, null, [Number.NaN, 0, 0]]), [0.5055, -0.1, 0.02]);
	assert.equal(medianPoint(pts.slice(0, 5)), null, "too few points");
});
