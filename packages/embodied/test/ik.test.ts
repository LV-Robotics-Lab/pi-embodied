import assert from "node:assert/strict";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { ikArgs, type Reach, reachRefusal, registerIkFlag } from "../src/ik.ts";

test("--ik is off by default and only then adds nothing to the env server arguments", () => {
	const flags: Record<string, unknown> = {};
	registerIkFlag({
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = o.default;
		},
	} as unknown as ExtensionAPI);
	assert.deepEqual(flags, { ik: "" });
	assert.deepEqual(ikArgs(flags.ik), []);
	assert.deepEqual(ikArgs(undefined), []);
	assert.deepEqual(ikArgs("  "), []);
	assert.deepEqual(ikArgs("http://127.0.0.1:18400"), ["--ik", "http://127.0.0.1:18400"]);
});

test("only an unreachable preview refuses a move; unknown is not approval but does not block", () => {
	const reach = (status: Reach["status"], message: string): Reach => ({
		status,
		reachable: status === "reachable" ? true : status === "unreachable" ? false : null,
		q: null,
		position_err: null,
		orientation_err: null,
		message,
		target: null,
	});
	assert.equal(reachRefusal(reach("reachable", "reachable (IK error 0.1 mm)")), undefined);
	assert.equal(reachRefusal(reach("unknown", "IK service unavailable")), undefined);
	assert.equal(
		reachRefusal(reach("unreachable", "unreachable: misses by 120.0 mm")),
		"refused: target is out of reach (unreachable: misses by 120.0 mm)",
	);
});
