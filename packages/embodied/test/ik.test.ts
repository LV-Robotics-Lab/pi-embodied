import assert from "node:assert/strict";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	ikArgs,
	type MotionPlan,
	planRefusal,
	type Reach,
	reachRefusal,
	registerIkFlag,
} from "../src/primitives/ik.ts";
import { useDeployment } from "./helpers/deployment.ts";

test("--ik is off by default; on, it passes services.ik, and it fails closed without one", () => {
	const flags: Record<string, unknown> = {};
	registerIkFlag({
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = o.default;
		},
	} as unknown as ExtensionAPI);
	assert.deepEqual(flags, { ik: false });
	const pi = (ik: unknown) => ({ getFlag: (n: string) => (n === "ik" ? ik : undefined) }) as unknown as ExtensionAPI;
	useDeployment({ services: { ik: "http://127.0.0.1:18400" } });
	assert.deepEqual(ikArgs(pi(false)), []);
	assert.deepEqual(ikArgs(pi(undefined)), []);
	assert.deepEqual(ikArgs(pi(true)), ["--ik", "http://127.0.0.1:18400"]);
	useDeployment({});
	assert.throws(() => ikArgs(pi(true)), /--ik needs services.ik/);
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

test("only a blocked plan refuses a move; an unknown plan runs the move unplanned", () => {
	const plan = (status: MotionPlan["status"], message: string): MotionPlan => ({
		status,
		message,
		waypoints: status === "planned" ? [[0.1, 0, 1, 0, 0, 0, 1]] : [],
		path_m: null,
	});
	assert.equal(planRefusal(plan("planned", "collision-free path, 1 segment(s)")), undefined);
	assert.equal(planRefusal(plan("unknown", "IK service unavailable")), undefined);
	assert.equal(
		planRefusal(plan("blocked", "no collision-free path: cuRobo found no collision-free path (IK_FAIL)")),
		"refused: no collision-free path: cuRobo found no collision-free path (IK_FAIL)",
	);
});
