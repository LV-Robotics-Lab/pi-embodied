import assert from "node:assert/strict";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import robolab, { STEP_M, VECTORS } from "../src/robolab/index.ts";
import { ground, MOVE_UNITS } from "../src/units/index.ts";

/** A stub pi that records flags and tools (no env server is started). */
function stubPi(values: Record<string, unknown> = {}) {
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const pi = {
		on: () => {},
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		setActiveTools: () => {},
		getActiveTools: () => [],
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	return { pi, flags, tools };
}

test("each MV_* unit is one 2 cm step along the base-frame vector (-y is MV_LEFT, +x away from the base)", () => {
	for (const unit of MOVE_UNITS) {
		const move = ground({ vectors: VECTORS, stepM: STEP_M }, unit)!;
		assert.deepEqual(
			move.delta.map((v) => Number(v.toFixed(9))),
			VECTORS[unit].map((v) => v * 0.02),
		);
		assert.equal(Math.hypot(...VECTORS[unit]), 1);
	}
	assert.deepEqual(VECTORS.MV_LEFT, [0, -1, 0]);
	assert.deepEqual(VECTORS.MV_FWD, [1, 0, 0]);
});

test("instruction phrasing and subtask tracking are flags, default as Show-Harness (default, off)", () => {
	const s = stubPi();
	robolab(s.pi);
	assert.equal(s.flags["instruction-type"], "default");
	assert.equal(s.flags.subtask, false);
	assert.equal(s.flags.task, "BananaInBowlTask");
	const vague = stubPi({ "instruction-type": "vague", subtask: true });
	robolab(vague.pi);
	assert.equal(vague.flags["instruction-type"], "vague");
});

test("string choices in tool schemas are plain string enums (no anyOf of literals)", () => {
	const s = stubPi();
	robolab(s.pi);
	const move = s.tools.get("move_delta");
	assert.ok(move, [...s.tools.keys()].join(","));
	const schema = JSON.stringify(move.parameters);
	assert.doesNotMatch(schema, /anyOf/);
	assert.match(schema, /"enum":\["open","close"\]/);
});
