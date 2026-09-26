import assert from "node:assert/strict";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import franka, { DETECTIONS_ENTRY } from "../src/franka/index.ts";

/** A stub pi that only records registrations (no robot starts: tools are inspected, not run). */
function fakePi() {
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const pi = {
		on: () => {},
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = o.default;
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

test("franka segment takes an `all` flag and its selection tools take a detection id", () => {
	const { pi, flags, tools } = fakePi();
	franka(pi);
	const segment = tools.get("segment");
	assert.ok(segment, "segment is registered");
	const props = segment.parameters.properties;
	assert.deepEqual(Object.keys(props).sort(), ["all", "camera", "min_score", "point", "prompt"]);
	assert.equal(props.all.type, "boolean");
	assert.equal(props.point.minItems, 2);
	assert.deepEqual(props.camera.enum, ["wrist", "third_person"]);
	assert.deepEqual(segment.parameters.required ?? [], [], "prompt and point are both optional");
	assert.match(segment.description, /d3/, "the description names the short id form");
	for (const name of ["select_detection", "reject_detection"]) {
		const t = tools.get(name);
		assert.ok(t, `${name} is registered`);
		assert.deepEqual(Object.keys(t.parameters.properties), ["id"]);
		assert.equal(t.parameters.properties.id.type, "string");
		assert.deepEqual(t.parameters.required, ["id"]);
	}
	assert.deepEqual(Object.keys(tools.get("enhance_depth").parameters.properties), ["camera"]);
	// The perception services are opt-in flags; off by default.
	assert.equal(flags["robot-sam3"], undefined);
	assert.equal(flags["robot-unidepth"], undefined);
	assert.equal(DETECTIONS_ENTRY, "robot_detections");
});
