import assert from "node:assert/strict";
import { test } from "node:test";
import type { ExtensionToolContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { metaworldPolicyWorld } from "../src/capabilities/policy/tool-world.ts";

function fixture() {
	const calls: { name: string; args: unknown; signal?: AbortSignal }[] = [];
	let blocked = false;
	let refreshed = true;
	const ctx: Pick<ExtensionToolContext, "tools" | "executeTool"> = {
		tools: ["view_env_state", "move_delta", "set_gripper"].map((name) => ({
			name,
			label: name,
			description: name,
			parameters: Type.Object({}),
			execute: async () => {
				throw new Error("must use the host execution pipeline");
			},
		})),
		executeTool: async (name, args, options) => {
			calls.push({ name, args, signal: options?.signal });
			return {
				toolCall: { type: "toolCall", id: "host/1", name, arguments: {} },
				isError: blocked,
				result: {
					content: [{ type: "text", text: blocked ? "blocked by host approval" : "observed" }],
					details: { success: true, result: { refreshed } },
				},
			};
		},
	};
	return {
		ctx,
		calls,
		block: () => {
			blocked = true;
		},
		stale: () => {
			refreshed = false;
		},
	};
}

test("policy adapter routes fresh observations and actions through the host pipeline with cancellation", async () => {
	const f = fixture();
	const world = metaworldPolicyWorld(f.ctx, () => ({ episode: "e1", revision: "1" }));
	const signal = new AbortController().signal;
	assert.equal((await world.observe(signal)).solved, true);
	await world.act("move_delta", { delta_xyz: [0.01, 0, 0] }, signal);
	assert.deepEqual(f.calls, [
		{ name: "view_env_state", args: { fresh: true }, signal },
		{ name: "move_delta", args: { delta_xyz: [0.01, 0, 0] }, signal },
	]);
	await assert.rejects(world.act("reset", {}, signal), /not enabled/);
	assert.equal(f.calls.length, 2);
});

test("policy adapter propagates blocked calls and refuses cached observations", async () => {
	const f = fixture();
	const world = metaworldPolicyWorld(f.ctx, () => ({ episode: "e1", revision: "1" }));
	f.stale();
	await assert.rejects(world.observe(), /did not refresh/);
	f.block();
	await assert.rejects(world.observe(), /blocked by host approval/);
	await assert.rejects(world.act("set_gripper", { close: true }), /blocked by host approval/);
});
