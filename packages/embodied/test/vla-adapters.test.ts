import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import libero from "../src/libero/index.ts";
import { toolSections } from "../src/robot.ts";
import { RpcClient } from "../src/rpc.ts";
import {
	PICK_PARAMETERS,
	pickTracker,
	suiteMismatch,
	VLA_ADAPTERS,
	vlaIdentity,
	vlaInfo,
} from "../src/vla-adapters.ts";

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

test("a checkpoint fine-tuned on another suite is refused; libero_all and unknown suites are trusted", () => {
	const oft = { service: "openvla-oft", model: "moojink/x", revision: "r", suite: "libero_spatial" };
	assert.equal(suiteMismatch(oft, "libero_spatial"), undefined);
	assert.match(
		suiteMismatch(oft, "libero_10")!,
		/moojink\/x@r is the libero_spatial fine-tune, but this episode is libero_10/,
	);
	assert.match(suiteMismatch(oft, "libero_10")!, /--suite libero_10/);
	assert.equal(suiteMismatch({ ...oft, suite: "libero_all" }, "libero_goal"), undefined);
	assert.equal(suiteMismatch({ ...oft, suite: null }, "libero_goal"), undefined, "a custom --model-path");
	assert.equal(suiteMismatch({ service: "pi05" }, "libero_goal"), undefined, "Pi0.5 has no vla.info");
	assert.equal(vlaIdentity(oft), "openvla-oft moojink/x@r");
	assert.equal(vlaIdentity({ service: "pi05" }), "pi05");
});

test("vlaInfo reads healthz and vla.info; a server without vla.info is its healthz name alone", async (t) => {
	const serve = (info: Record<string, unknown> | undefined) => {
		const server = createServer((req, res) => {
			let body = "";
			req.on("data", (c) => {
				body += c;
			});
			req.on("end", () => {
				const { method } = JSON.parse(body);
				if (method === "healthz")
					res.end(JSON.stringify({ ok: true, result: { status: "ok", service: "openvla" } }));
				else if (method === "vla.info" && info) res.end(JSON.stringify({ ok: true, result: info }));
				else res.end(JSON.stringify({ ok: false, error: `unknown method ${method}` }));
			});
		});
		return new Promise<string>((r) =>
			server.listen(0, "127.0.0.1", () => {
				t.after(() => {
					server.closeAllConnections();
					server.close();
				});
				r(`http://127.0.0.1:${(server.address() as AddressInfo).port}`);
			}),
		);
	};
	const adapter = await vlaInfo(
		new RpcClient(await serve({ model: "openvla/x", revision: "abc", suite: "libero_10", horizon: 1 })),
	);
	assert.deepEqual(adapter, { service: "openvla", model: "openvla/x", revision: "abc", suite: "libero_10" });
	const bare = await vlaInfo(new RpcClient(await serve(undefined)));
	assert.deepEqual(bare, { service: "openvla" });
});
