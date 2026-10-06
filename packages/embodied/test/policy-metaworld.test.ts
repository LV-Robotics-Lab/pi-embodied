import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { createModels } from "@earendil-works/pi-ai/models";
import { fauxAssistantMessage, fauxProvider, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { openStoredPolicy } from "../src/capabilities/policy/stored.ts";
import { metaworldPolicyWorld } from "../src/capabilities/policy/tool-world.ts";
import metaworld, { VIEW_SETUP } from "../src/robots/metaworld/index.ts";
import { deployFlags } from "./helpers/deployment.ts";
import { f32, fakeEnv, rgb } from "./sim-stub.ts";

type Handler = (event: any, ctx: any) => unknown;
const call = (name: string, args: Parameters<typeof fauxToolCall>[1] = {}) =>
	fauxAssistantMessage(fauxToolCall(name, args), { stopReason: "toolUse" });
const observe = () => call("policy_observe");
const act = () => call("policy_act", { tool: "move_delta", arguments: { delta_xyz: [0.02, 0, 0] } });
const finish = (status: string) => call("policy_finish", { status, summary: "Observed the outcome." });
const GOAL = { goal_id: "reach-001", goal: "Complete the current MetaWorld reach task", event_id: "start" };

/** A fake reach-v3 server: solved after two moves; `hold` delays a move's reply after its effect happened. */
async function fakeReach() {
	const world = { moves: 0, hold: undefined as Promise<void> | undefined };
	const solved = () => world.moves >= 2;
	const obs = () => ({
		agentview: rgb(),
		wrist: rgb(),
		tcp_pos: f32([0, 0.6, 0.2]),
		gripper_width: 0.09,
		obs: f32([0]),
	});
	const env = await fakeEnv((c) => {
		if (c.method === "env.get_env_meta")
			return {
				task: "reach-v3",
				seed: 0,
				metaworld: "3.1.1",
				workspace: { min: [0, 0, 0], max: [1, 1, 1] },
				...VIEW_SETUP,
			};
		if (c.method === "env.reset") return [obs(), {}];
		if (c.method === "env.get_task_language") return "reach the goal";
		if (c.method === "env.render_camera") return rgb();
		if (c.method === "env.state")
			return {
				tcp_pos: [0, 0.6, 0.2],
				gripper_width: 0.09,
				gripper_command: "open",
				success_once: solved(),
				info: { success: solved() },
			};
		if (c.method === "env.move_delta") {
			world.moves++; // the physical effect precedes the acknowledgement
			const frame = { ...obs(), action: f32([0, 0, 0, 1]), success: solved() };
			const reply = {
				ok: true,
				final_tcp_pos: [0, 0.6, 0.2],
				final_error_m: 0,
				moved_m: [0.02, 0, 0],
				gripper: "open",
				gripper_width: 0.09,
				steps_used: 2,
				frames: [frame, frame],
				info: { success: solved() },
			};
			return world.hold ? world.hold.then(() => reply) : reply;
		}
		return undefined;
	});
	return { ...env, world };
}

/** A stub pi whose tool context offers `tools`, `executeTool` and the model registry, as pi's does to a tool. */
function host(values: Record<string, unknown>, cwd: string) {
	values = deployFlags(values);
	const handlers = new Map<string, Handler[]>();
	const listeners = new Map<string, ((data: unknown) => void)[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	let active: string[] = [];
	const errors: string[] = [];
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
		events: {
			emit: (name: string, data: unknown) => {
				for (const fn of listeners.get(name) ?? []) fn(data);
			},
			on: (name: string, fn: (data: unknown) => void) => {
				listeners.set(name, [...(listeners.get(name) ?? []), fn]);
				return () => {};
			},
		},
	};
	const pi = new Proxy(api, { get: (t, k: string) => t[k] ?? (() => {}) }) as unknown as ExtensionAPI;
	const faux = fauxProvider();
	const models = createModels();
	models.setProvider(faux.provider);
	async function emit(name: string, event: Record<string, unknown> = {}) {
		for (const fn of handlers.get(name) ?? []) await fn({ type: name, ...event }, ctx);
	}
	const ctx: any = {
		hasUI: true,
		ui: { notify: (msg: string) => errors.push(msg) },
		shutdown: () => {},
		cwd,
		sessionManager: {
			getBranch: () => [],
			getSessionFile: () => undefined,
			getSessionDir: () => undefined,
			getSessionId: () => "s",
		},
		model: faux.models[0],
		modelRegistry: {
			stream: (m: any, c: any, o: any) => models.stream(m, c, o),
			streamSimple: (m: any, c: any, o: any) => models.streamSimple(m, c, o),
		},
		get tools() {
			return [...tools.values()];
		},
		// pi runs a nested call through its tool pipeline and emits its tool events to the extensions.
		executeTool: async (name: string, args: unknown, options?: { signal?: AbortSignal }) => {
			const toolCall = { type: "toolCall", id: `host/${name}`, name, arguments: args };
			await emit("tool_execution_start", { toolName: name });
			try {
				const result = await tools.get(name).execute("host", args, options?.signal, undefined, ctx);
				await emit("tool_execution_end", { toolName: name });
				return { toolCall, isError: false, result };
			} catch (err) {
				await emit("tool_execution_end", { toolName: name });
				const text = err instanceof Error ? err.message : String(err);
				return { toolCall, isError: true, result: { content: [{ type: "text", text }], details: {} } };
			}
		},
	};
	const run = (name: string, params: unknown, signal?: AbortSignal) =>
		tools.get(name).execute("id", params, signal, undefined, ctx) as Promise<any>;
	return { pi, faux, models, tools, emit, run, errors, active: () => active, ctx };
}

const peekOptions = (h: ReturnType<typeof host>) => ({
	models: h.models,
	model: { provider: "faux", modelId: "faux-1" },
	world: metaworldPolicyWorld(h.ctx, () => ({ episode: "peek", revision: "0" })),
});

test("policy_goal: a host killed mid-action resumes with the same goal and event, re-observes and never replays the action", async (t) => {
	const env = await fakeReach();
	t.after(env.close);
	const dir = await mkdtemp(join(tmpdir(), "policy-metaworld-"));
	t.after(() => rm(dir, { recursive: true, force: true }));
	const store = join(dir, "reach.sqlite");
	const h = host({ "env-url": env.url, task: "reach-v3", units: "false", code: "false", "policy-store": store }, dir);
	metaworld(h.pi);
	await h.emit("session_start");
	assert.deepEqual(h.errors, []);
	assert.ok(h.tools.has("policy_goal"));
	assert.ok(!h.active().includes("policy_goal"), "registered, not active, before the agent starts");
	await h.emit("before_agent_start", { systemPrompt: "" });
	assert.ok(h.active().includes("policy_goal"), "--policy-store activates policy_goal");

	// 1. The first move takes effect on the server; the host is killed before the reply arrives.
	let release = () => {};
	env.world.hold = new Promise<void>((r) => {
		release = r;
	});
	h.faux.setResponses([observe(), act()]);
	const killed = new AbortController();
	const first = h.run("policy_goal", GOAL, killed.signal);
	for (let i = 0; i < 400 && env.world.moves < 1; i++) await new Promise((r) => setTimeout(r, 25));
	assert.equal(env.world.moves, 1);
	killed.abort();
	await assert.rejects(first);
	release();
	env.world.hold = undefined;

	// 2. The reopened store: the action's outcome is unknown, no observation stands, the goal is still
	//    active, and no other goal may start before it concludes.
	const peek = await openStoredPolicy(store, peekOptions(h));
	try {
		const saved = await peek.state();
		assert.equal(saved.lastAction?.tool, "move_delta");
		assert.equal(saved.lastAction?.outcome, "unknown");
		assert.equal(saved.observation, null);
		assert.equal(saved.goals[0].status, "active");
		await assert.rejects(
			peek.submit({ goalId: "reach-002", goal: "Another goal", eventId: "start" }),
			/finish the current goal|policy is busy/,
		);
	} finally {
		await peek.close();
	}

	// 3. The same goal and event resume the conversation: a blind action is refused until a fresh
	//    observation, then one more move solves the task. The interrupted move is never replayed.
	h.faux.setResponses([act(), observe(), act(), observe(), finish("success")]);
	const second = await h.run("policy_goal", GOAL);
	assert.equal(second.details.goal.status, "success", JSON.stringify(second.details));
	assert.equal(env.world.moves, 2, "one interrupted move plus one observed move; the blind action did not move");
	assert.ok(env.calls.filter((c) => c.method === "env.state").length >= 2, "recovery re-read the server's state");
	assert.deepEqual(
		env.calls
			.filter((c) => c.method === "env.render_camera")
			.map((c) => c.kwargs.camera_name)
			.slice(0, 2),
		["agentview", "wrist"],
	);

	// 4. The same event again admits nothing: the model is not asked.
	const requests = h.faux.state.callCount;
	const third = await h.run("policy_goal", GOAL);
	assert.equal(h.faux.state.callCount, requests);
	assert.equal(third.details.goal.status, "success");

	// 5. The transcript keeps the interruption and the refusal of the blind action.
	const after = await openStoredPolicy(store, peekOptions(h));
	try {
		const transcript = JSON.stringify(await after.conversation.context(BACKGROUND_CONTEXT));
		assert.match(transcript, /interrupted|aborted/);
		assert.match(transcript, /observe again before acting/);
		assert.equal((await after.state()).lastAction?.outcome, "returned");
	} finally {
		await after.close();
	}
});

test("policy_goal is refused without --policy-store and loads nothing of Pi Durable", async (t) => {
	const env = await fakeReach();
	t.after(env.close);
	const h = host({ "env-url": env.url, task: "reach-v3", units: "false", code: "false" }, tmpdir());
	metaworld(h.pi);
	await h.emit("session_start");
	await h.emit("before_agent_start", { systemPrompt: "" });
	assert.ok(!h.active().includes("policy_goal"));
	await assert.rejects(h.run("policy_goal", GOAL), /set --policy-store/);
});
