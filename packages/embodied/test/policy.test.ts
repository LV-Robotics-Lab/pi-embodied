import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import type { TextContent } from "@earendil-works/pi-ai";
import { createModels } from "@earendil-works/pi-ai/models";
import { fauxAssistantMessage, fauxProvider, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import { MemoryStorage } from "@earendil-works/pi-durable";
import { Type } from "typebox";
import { openPolicy, type PersistentPolicy, type PolicyWorld } from "../src/capabilities/policy/policy.ts";
import { openStoredPolicy } from "../src/capabilities/policy/stored.ts";

const context = BACKGROUND_CONTEXT;
const call = (name: string, args: Parameters<typeof fauxToolCall>[1] = {}) =>
	fauxAssistantMessage(fauxToolCall(name, args), { stopReason: "toolUse" });
const observe = () => call("policy_observe");
const act = () => call("policy_act", { tool: "move_delta", arguments: { delta_xyz: [0.01, 0, 0] } });
const finish = (status = "success") => call("policy_finish", { status, summary: "Observed the outcome." });
const event = (goalId = "g1", eventId = "start") => ({ goalId, goal: "Reach the target", eventId });

function fixture() {
	const faux = fauxProvider();
	const models = createModels();
	models.setProvider(faux.provider);
	const environment = { episode: 1, revision: 0, moves: 0, reads: 0, solved: false };
	const world: PolicyWorld = {
		robot: "test-simulator",
		identity: () => ({ episode: `episode-${environment.episode}`, revision: String(environment.revision) }),
		actions: [
			{
				name: "move_delta",
				description: "Move toward the target",
				parameters: Type.Object({ delta_xyz: Type.Array(Type.Number()) }),
			},
		],
		observe: async () => {
			environment.reads++;
			return {
				...world.identity(),
				solved: environment.solved,
				content: [{ type: "text", text: JSON.stringify(environment) }],
			};
		},
		act: async () => {
			environment.moves++;
			environment.revision++;
			environment.solved = true;
			return [{ type: "text", text: "moved" }];
		},
	};
	return { faux, models, environment, world, model: { provider: "faux", modelId: "faux-1" } };
}
async function run(policy: PersistentPolicy, goalId = "g1", eventId = "start") {
	await (await policy.submit(event(goalId, eventId))).wait(context);
	return policy.state();
}

test("one policy retains memory and conversation across two episodes, and deduplicates events", async () => {
	const f = fixture();
	const policy = await openPolicy({ ...f, storage: new MemoryStorage() });
	try {
		f.faux.setResponses([
			observe(),
			act(),
			observe(),
			call("policy_remember", { memory: "Use small moves; verify after moving." }),
			finish(),
		]);
		const first = await run(policy);
		assert.equal(first.goals[0].status, "success");
		const requests = f.faux.state.callCount;
		await run(policy);
		assert.equal(f.faux.state.callCount, requests);
		assert.equal(f.environment.moves, 1);
		await assert.rejects(policy.submit({ ...event(), input: "changed" }), /different content/);
		f.environment.episode++;
		f.environment.revision = 0;
		f.environment.solved = false;
		let remembers = false;
		f.faux.setResponses([
			(ctx) => {
				remembers = JSON.stringify(ctx).includes("Use small moves; verify after moving.");
				return observe();
			},
			act(),
			observe(),
			finish(),
		]);
		const second = await run(policy, "g2");
		assert.ok(remembers);
		assert.deepEqual(
			second.goals.map((g) => g.status),
			["success", "success"],
		);
		assert.equal(f.environment.moves, 2);
		assert.equal(second.actions, 1, "action budget is per goal");
	} finally {
		await policy.close();
	}
});

test("the runtime rejects stale actions and unverified success even when the model requests them", async () => {
	const f = fixture();
	const policy = await openPolicy({ ...f, storage: new MemoryStorage() });
	try {
		f.faux.setResponses([act(), finish(), observe(), finish(), act(), act(), observe(), finish()]);
		const state = await run(policy);
		assert.equal(f.environment.moves, 1, "both blind actions were blocked");
		assert.equal(state.goals[0].status, "success");
		const transcript = JSON.stringify(await policy.conversation.context(context));
		assert.match(transcript, /observe again before acting/);
		assert.match(transcript, /environment has not confirmed success/);
	} finally {
		await policy.close();
	}
});

test("a reset invalidates an earlier observation even if the step number repeats", async () => {
	const f = fixture();
	const policy = await openPolicy({ ...f, storage: new MemoryStorage() });
	try {
		f.faux.setResponses([
			observe(),
			() => {
				f.environment.episode++;
				return act();
			},
			observe(),
			act(),
			observe(),
			finish(),
		]);
		const state = await run(policy);
		assert.equal(f.environment.moves, 1);
		assert.equal(state.goals[0].status, "success");
	} finally {
		await policy.close();
	}
});

test("waiting consumes no model requests until a new external event, including after reopening", async () => {
	const f = fixture();
	const dir = await mkdtemp(join(tmpdir(), "persistent-policy-"));
	const file = join(dir, "policy.sqlite");
	let policy = await openStoredPolicy(file, f);
	try {
		f.faux.setResponses([finish("waiting")]);
		assert.equal((await run(policy)).goals[0].status, "waiting");
		const count = f.faux.state.callCount;
		await policy.close();
		policy = await openStoredPolicy(file, f);
		assert.equal(await policy.resume(), undefined);
		await new Promise((r) => setTimeout(r, 30));
		assert.equal(f.faux.state.callCount, count);
		f.faux.setResponses([observe(), act(), observe(), finish()]);
		assert.equal((await run(policy, "g1", "scene-ready")).goals[0].status, "success");
	} finally {
		await policy.close();
		await rm(dir, { recursive: true, force: true });
	}
});

test("an action applied before interruption is not replayed; recovery must re-observe", async () => {
	const f = fixture();
	const dir = await mkdtemp(join(tmpdir(), "persistent-policy-"));
	const file = join(dir, "policy.sqlite");
	let policy = await openStoredPolicy(file, f);
	let reached = () => {};
	const moved = new Promise<void>((resolve) => {
		reached = resolve;
	});
	const original = f.world.act;
	f.world.act = async (tool, args, signal) => {
		await original(tool, args, signal);
		reached();
		return new Promise<TextContent[]>((_resolve, reject) => {
			if (signal?.aborted) reject(new Error("cancelled"));
			else signal?.addEventListener("abort", () => reject(new Error("cancelled")), { once: true });
		});
	};
	try {
		f.faux.setResponses([observe(), act()]);
		await policy.submit(event());
		await moved;
		await policy.close();
		assert.equal(f.environment.moves, 1);
		f.world.act = original;
		f.faux.setResponses([act(), observe(), finish()]);
		policy = await openStoredPolicy(file, f);
		const pending = await policy.resume();
		assert.ok(pending);
		await pending.wait(context);
		const state = await policy.state();
		assert.equal(
			f.environment.moves,
			1,
			"neither the interrupted tool nor the blind recovery action moved the robot",
		);
		assert.equal(state.lastAction?.outcome, "unknown");
		assert.equal(state.goals[0].status, "success");
		assert.ok(f.environment.reads >= 2);
	} finally {
		await policy.close();
		await rm(dir, { recursive: true, force: true });
	}
});

test("model request budget persists and prevents an observation loop", async () => {
	const f = fixture();
	const policy = await openPolicy({ ...f, storage: new MemoryStorage(), maxRequests: 3 });
	try {
		f.faux.setResponses(Array.from({ length: 8 }, observe));
		const state = await run(policy);
		assert.equal(f.faux.state.callCount, 3);
		assert.equal(state.goals[0].status, "waiting");
		assert.match(state.goals[0].summary, /budget/);
	} finally {
		await policy.close();
	}
});

test("SQLite policy admits one owner and binds the store to its robot", async () => {
	const f = fixture();
	const dir = await mkdtemp(join(tmpdir(), "persistent-policy-"));
	const file = join(dir, "policy.sqlite");
	const policy = await openStoredPolicy(file, f);
	try {
		await assert.rejects(openStoredPolicy(file, f), /already owned/);
		await policy.close();
		await assert.rejects(openStoredPolicy(file, { ...f, world: { ...f.world, robot: "another" } }), /another robot/);
		const reopened = await openStoredPolicy(file, f);
		await reopened.close();
	} finally {
		await policy.close();
		await rm(dir, { recursive: true, force: true });
	}
});
