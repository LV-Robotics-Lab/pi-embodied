/**
 * The PreToolUse hook (../src/integrations/mcp/hook.ts) decides with `--approval standard`'s risk
 * classes (../src/capabilities/operator.ts highRisk): grasp/place and reset tools, absolute moves,
 * large relative moves, and every motion on a real robot ask the operator (Claude Code) or are
 * denied unless confirmed beforehand (Codex); look-only tools and small relative moves pass through;
 * an unknown robot or a tool of another server fall the safe way.
 */

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { decide, type HookArgs, ourTool, parseHookArgs } from "../src/integrations/mcp/hook.ts";
import { REAL_ROBOTS } from "../src/integrations/mcp/tools.ts";
import { loadManifest } from "../src/primitives/manifest.ts";

const ask = (robot: string, over: Partial<HookArgs> = {}): HookArgs => ({
	robot,
	server: "pi-embodied",
	decision: "ask",
	largeMove: 0.1,
	confirmedEnv: "PI_EMBODIED_MOTION_CONFIRMED",
	...over,
});
const call = (tool: string, tool_input: unknown = {}) => ({ tool_name: `mcp__pi-embodied__${tool}`, tool_input });
const decision = (o: ReturnType<typeof decide>) => o?.hookSpecificOutput.permissionDecision;

test("simulation: high-risk motions ask, small relative motions and look-only tools pass", () => {
	const a = ask("libero");
	assert.equal(decision(decide(call("move_to", { xyz: [0.1, 0.2, 0.3] }), a, {})), "ask", "absolute move");
	assert.equal(decision(decide(call("release"), a, {})), "ask", "release is a grasp/place tool");
	assert.equal(decision(decide(call("execute_grasp", { grasp_id: "g1" }), a, {})), "ask");
	assert.equal(decide(call("set_gripper", { gripper: 1 }), a, {}), undefined, "a gripper command is not high risk");
	assert.equal(decide(call("rotate_wrist", { delta_yaw: 0.1 }), a, {}), undefined);
	assert.equal(decide(call("observe"), a, {}), undefined);
	assert.equal(decide(call("segment", { prompt: "bowl" }), a, {}), undefined, "look-only");
	assert.equal(decide(call("finish", { status: "success", summary: "x" }), a, {}), undefined);
	assert.equal(decide({ tool_name: "Bash", tool_input: { command: "ls" } }, a, {}), undefined, "another tool");
	assert.equal(decide({ tool_name: "mcp__other__move_to", tool_input: {} }, a, {}), undefined, "another server");
	const r = decide(call("move_to", { xyz: [0, 0, 0] }), a, {});
	assert.equal(r?.hookSpecificOutput.hookEventName, "PreToolUse");
	assert.match(r?.hookSpecificOutput.permissionDecisionReason ?? "", /high-risk motion/);
});

test("large relative moves ask under the --large-move threshold, as --approval-large-move does", () => {
	const m = loadManifest("robosuite");
	const delta = m.primitives.find((e) => e.side === "env" && e.mutating && e.params && "delta_xyz" in e.params);
	assert.ok(delta, "robosuite has a relative move with delta_xyz");
	const a = ask("robosuite");
	assert.equal(decide(call(delta.name, { delta_xyz: [0.02, 0, 0] }), a, {}), undefined);
	assert.equal(decision(decide(call(delta.name, { delta_xyz: [0.3, 0, 0] }), a, {})), "ask");
	assert.equal(
		decision(decide(call(delta.name, { delta_xyz: [0.3, 0, 0] }), ask("robosuite", { largeMove: 0.5 }), {})),
		undefined,
	);
});

test("real robots: every motion asks; Codex's deny mode denies unless the operator confirmed", () => {
	for (const robot of REAL_ROBOTS) {
		const m = loadManifest(robot);
		const motion = m.primitives.find((e) => e.side === "env" && e.mutating && e.doc.tool);
		assert.ok(motion, `${robot} has a motion tool`);
		const r = decide(call(motion.name, { delta_xyz: [0.001, 0, 0] }), ask(robot), {});
		assert.equal(decision(r), "ask", `${robot}: ${motion.name}`);
		assert.match(r?.hookSpecificOutput.permissionDecisionReason ?? "", /real robot/);
		const d = decide(call(motion.name, {}), ask(robot, { decision: "deny" }), {});
		assert.equal(decision(d), "deny");
		assert.match(d?.hookSpecificOutput.permissionDecisionReason ?? "", /PI_EMBODIED_MOTION_CONFIRMED=1/);
		assert.equal(
			decide(call(motion.name, {}), ask(robot, { decision: "deny" }), { PI_EMBODIED_MOTION_CONFIRMED: "1" }),
			undefined,
		);
		const look = m.primitives.find((e) => e.side === "env" && !e.mutating && e.doc.tool);
		if (look) assert.equal(decide(call(look.name), ask(robot), {}), undefined, `${robot}: ${look.name} is look-only`);
	}
});

test("fail closed: no robot or an unreadable manifest decides for every call of our server", () => {
	assert.equal(decision(decide(call("observe"), ask(""), {})), "ask");
	assert.match(
		decide(call("observe"), ask(""), {})?.hookSpecificOutput.permissionDecisionReason ?? "",
		/PI_EMBODIED_ROBOT/,
	);
	assert.equal(decision(decide(call("move_to"), ask("no_such_robot", { decision: "deny" }), {})), "deny");
	assert.equal(decide({ tool_name: "Read" }, ask(""), {}), undefined, "other tools are untouched");
});

test("arguments: the robot from the plugin option or PI_EMBODIED_ROBOT; mcp__<server>__<tool> parsing", () => {
	assert.equal(
		parseHookArgs([], { CLAUDE_PLUGIN_OPTION_ROBOT: "franka", PI_EMBODIED_ROBOT: "libero" }).robot,
		"franka",
	);
	assert.equal(parseHookArgs([], { PI_EMBODIED_ROBOT: "libero" }).robot, "libero");
	assert.equal(parseHookArgs(["--robot", "piper", "--decision", "deny", "--large-move", "0.05"], {}).largeMove, 0.05);
	assert.throws(() => parseHookArgs(["--decision", "maybe"], {}), /ask\|deny/);
	assert.equal(ourTool("mcp__pi-embodied__move_to", "pi-embodied"), "move_to");
	assert.equal(ourTool("mcp__pi-embodied-libero__move_to", "pi-embodied"), undefined);
	assert.equal(ourTool("move_to", "pi-embodied"), undefined);
});

test("the hook runs as a process: stdin JSON in, a decision or nothing out", () => {
	const hook = fileURLToPath(new URL("../src/integrations/mcp/hook.ts", import.meta.url));
	const run = (input: unknown, args: string[] = []) =>
		execFileSync(process.execPath, ["--experimental-strip-types", hook, "--robot", "libero", ...args], {
			input: JSON.stringify(input),
			encoding: "utf8",
			env: { ...process.env, NODE_NO_WARNINGS: "1", PI_EMBODIED_MOTION_CONFIRMED: "" },
		});
	assert.equal(run(call("set_gripper", { gripper: 1 })), "");
	const out = JSON.parse(run(call("move_to", { xyz: [0, 0, 0] })));
	assert.equal(out.hookSpecificOutput.permissionDecision, "ask");
	assert.equal(
		JSON.parse(run(call("move_to", { xyz: [0, 0, 0] }), ["--decision", "deny"])).hookSpecificOutput
			.permissionDecision,
		"deny",
	);
});

test("REAL_ROBOTS covers every env server that takes the hardware lock", () => {
	const robots = fileURLToPath(new URL("../../../services/pi_embodied_services/robots/", import.meta.url));
	const locking = readdirSync(robots, { withFileTypes: true })
		.filter((d) => d.isDirectory())
		.map((d) => d.name)
		.filter((name) => {
			try {
				return /hardware_lock/.test(readFileSync(`${robots}${name}/env_server.py`, "utf8"));
			} catch {
				return false;
			}
		});
	assert.ok(locking.length >= 3, locking.join(","));
	for (const dir of locking)
		assert.ok(REAL_ROBOTS.includes(dir.replace(/_polymetis$/, "")), `${dir} takes the lock: its robot is real`);
});
