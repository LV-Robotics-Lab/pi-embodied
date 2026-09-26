import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import libero from "../src/libero/index.ts";
import { toolSections } from "../src/robot.ts";
import { PICK_PARAMETERS, pickTracker, VLA_ADAPTERS } from "../src/vla-adapters.ts";

type Tool = { name: string; description: string; parameters: unknown };

/** Load the LIBERO robot into a stub pi with `values` as the command-line flags; returns its registered tools. */
function loaded(values: Record<string, unknown> = {}) {
	const flags: Record<string, unknown> = {};
	const tools: Tool[] = [];
	const pi = {
		on: () => {},
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: Tool) => tools.push(t),
		registerCommand: () => {},
		registerProvider: () => {},
		registerMessageRenderer: () => {},
		registerShortcut: () => {},
		setActiveTools: () => {},
		getActiveTools: () => [],
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	libero(pi);
	return { flags, tools: new Map(tools.map((t) => [t.name, t])) };
}
const schema = (t: Tool | undefined) => JSON.parse(JSON.stringify(t?.parameters));

test("no adapter flag: no adapter tool, no adapter flag value", () => {
	const { flags, tools } = loaded();
	for (const a of VLA_ADAPTERS) {
		assert.equal(flags[a.flag], undefined, a.flag);
		assert.equal(tools.has(a.tool), false, a.tool);
	}
	assert.ok(tools.has("pi0_pick"));
});

test("--openvla / --openvla-oft / --gr00t each mount their grasp tool with pi0_pick's parameters", () => {
	const { tools } = loaded({ openvla: "http://127.0.0.1:18600", gr00t: "127.0.0.1:18800" });
	assert.ok(tools.has("openvla_act") && tools.has("gr00t_act") && !tools.has("openvla_oft_act"));
	assert.deepEqual(schema(tools.get("openvla_act")), schema(tools.get("pi0_pick")));
	assert.deepEqual(schema(tools.get("pi0_pick")), JSON.parse(JSON.stringify(PICK_PARAMETERS)));
	assert.match(tools.get("gr00t_act")!.description, /GR00T .* grasp/);
	assert.match(tools.get("openvla_act")!.description, /OpenVLA .* grasp/);
});

test("the LIBERO prompt describes an adapter only when its tool is active", () => {
	const prompt = new URL("../src/libero/SYSTEM.md", import.meta.url);
	const text = readFileSync(prompt, "utf8");
	const base = ["move_to", "pi0_pick", "release", "finish"];
	assert.doesNotMatch(toolSections(text, base), /openvla|gr00t/i);
	const withOft = toolSections(text, [...base, "openvla_oft_act"]);
	assert.match(withOft, /`openvla_oft_act` \(OpenVLA-OFT\) are grasp policies with the same contract as `pi0_pick`/);
	assert.doesNotMatch(withOft, /openvla_act|gr00t/);
	const noPi0 = toolSections(text, ["move_to", "gr00t_act", "release", "finish"]);
	assert.match(noPi0, /`gr00t_act` \(GR00T\) are grasp policies with the same contract:/);
	assert.doesNotMatch(noPi0, /pi0/i);
});

test("the pick heuristics: descend, lift with a partly closed gripper", () => {
	const t = pickTracker(0.3, 0.08, { prompt: "x" });
	assert.equal(t.update(0.25, 0.08), false, "not descended enough");
	assert.equal(t.update(0.18, 0.08), false, "descended 12 cm, no lift yet");
	assert.equal(t.update(0.24, 0.08), false, "lifted, but the gripper is open (0.08 >= 0.06)");
	assert.equal(t.update(0.24, 0.03), true, "lifted 6 cm with the gripper at 0.03");
	assert.deepEqual(t.summary(), { peak_lift_m: 0.24 - 0.18, descent_m: 0.3 - 0.18, min_gripper_opening: 0.03 });
	// A contact skill: lift_thresh 999 never succeeds by lift; an empty grasp (gripper at 0) fails gripper_open_thresh.
	assert.equal(pickTracker(0.3, 0.08, { prompt: "x", lift_thresh: 999 }).update(0.1, 0.03), false);
	const empty = pickTracker(0.3, 0.08, { prompt: "x", gripper_open_thresh: 0.01 });
	empty.update(0.15, 0.0);
	assert.equal(empty.update(0.25, 0.0), false);
});
