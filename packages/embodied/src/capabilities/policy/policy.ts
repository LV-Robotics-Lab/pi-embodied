/** A persistent policy conversation. The host supplies tools and current environment identity. */
import { randomUUID } from "node:crypto";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import {
	type AssistantMessage,
	createAssistantMessageEventStream,
	type ImageContent,
	StringEnum,
	type TextContent,
	type Tool,
} from "@earendil-works/pi-ai";
import type { Models } from "@earendil-works/pi-ai/models";
import {
	type Conversation,
	createRegistry,
	defineDoc,
	defineExtension,
	defineTool,
	GenerationTask,
	Harness,
	hook,
	LiveDoc,
	type ModelRef,
	type Storage,
	section,
} from "@earendil-works/pi-durable";
import { Type } from "typebox";

export type WorldIdentity = { episode: string; revision: string };
export type PolicyObservation = WorldIdentity & { content: (TextContent | ImageContent)[]; solved: boolean };
export type PolicyWorld = {
	/** Stable embodiment identity. Different robots must use different policy stores. */
	robot: string;
	/** Must change episode on every reset, and revision whenever the world advances. */
	identity(): WorldIdentity;
	actions: readonly Tool[];
	observe(signal?: AbortSignal): Promise<PolicyObservation>;
	act(tool: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<(TextContent | ImageContent)[]>;
};
export type GoalEvent = { goalId: string; goal: string; eventId: string; input?: string };
type Goal = { id: string; text: string; status: "active" | "waiting" | "success" | "failure"; summary: string };
export type PolicyState = {
	robot: string;
	current: string | null;
	goals: Goal[];
	events: { key: string; goalId: string; goal: string; eventId: string; input: string }[];
	memory: string;
	requests: number;
	actions: number;
	observation: (WorldIdentity & { boot: string; solved: boolean }) | null;
	lastAction: { callId: string; tool: string; episode: string; outcome: "unknown" | "returned" } | null;
};
export const PolicyDoc = defineDoc<PolicyState>({
	kind: "embodied.policy",
	version: 1,
	scope: "conversation",
	history: "latest",
	fork: "initial",
	initial: () => ({
		robot: "",
		current: null,
		goals: [],
		events: [],
		memory: "",
		requests: 0,
		actions: 0,
		observation: null,
		lastAction: null,
	}),
});
const context = BACKGROUND_CONTEXT;
const text = (value: unknown): TextContent[] => [{ type: "text", text: JSON.stringify(value) }];
const currentGoal = (state: Readonly<PolicyState>) => state.goals.find((goal) => goal.id === state.current);

export async function openPolicy(options: {
	storage: Storage;
	models: Models;
	model: ModelRef;
	world: PolicyWorld;
	maxRequests?: number;
	maxActions?: number;
}) {
	const { world } = options;
	const boot = randomUUID();
	const maxRequests = options.maxRequests ?? 32;
	const maxActions = options.maxActions ?? 64;
	if (![maxRequests, maxActions].every((n) => Number.isSafeInteger(n) && n > 0))
		throw new Error("policy budgets must be positive integers");
	let harness: Harness;
	let root: Conversation;
	// Hooks report errors and continue, so a hard request budget belongs at the model boundary.
	const streamSimple: Models["streamSimple"] = (model, input, streamOptions) => {
		const output = createAssistantMessageEventStream();
		void (async () => {
			try {
				const allowed = await root.commit(async (tx) => {
					const state = await tx.doc(PolicyDoc, root.id);
					const goal = currentGoal(state);
					if (!goal || goal.status !== "active") return false;
					if (state.requests >= maxRequests) {
						goal.status = "waiting";
						goal.summary = "Model request budget exhausted; awaiting a host event.";
						return false;
					}
					state.requests++;
					return true;
				}, context);
				if (!allowed) throw new Error("policy is waiting for a host event");
				const inner = options.models.streamSimple(model, input, streamOptions);
				for await (const event of inner) output.push(event);
				output.end(await inner.result());
			} catch (error) {
				const failed: AssistantMessage = {
					role: "assistant",
					api: model.api,
					provider: model.provider,
					model: model.id,
					content: [],
					stopReason: "error",
					errorMessage: error instanceof Error ? error.message : String(error),
					timestamp: Date.now(),
					usage: {
						input: 0,
						output: 0,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 0,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
				};
				output.push({ type: "error", reason: "error", error: failed });
				output.end(failed);
			}
		})();
		return output;
	};
	const models = new Proxy(options.models, {
		get(target, key, receiver) {
			if (key === "streamSimple") return streamSimple;
			const value: unknown = Reflect.get(target, key, receiver);
			return typeof value === "function" ? value.bind(target) : value;
		},
	});
	const assertFresh = (state: Readonly<PolicyState>) => {
		if (currentGoal(state)?.status !== "active") throw new Error("no active goal; wait for a host event");
		const now = world.identity();
		const seen = state.observation;
		if (!seen || seen.boot !== boot || seen.episode !== now.episode || seen.revision !== now.revision) {
			throw new Error("observe again before acting or claiming success: the environment or policy runtime changed");
		}
	};
	const observe = defineTool({
		name: "policy_observe",
		description:
			"Read fresh observations and episode identity. Required before each action and after restart or reset.",
		parameters: Type.Object({}),
		replay: "safe",
		executionMode: "sequential",
		execute: async (_args, api, ctx) => {
			const observation = await world.observe(ctx.abortSignal);
			const now = world.identity();
			if (observation.episode !== now.episode || observation.revision !== now.revision)
				throw new Error("world changed during observation; observe again");
			const lastAction = await api.commit(async (tx) => {
				const state = await tx.doc(PolicyDoc, api.conversationId);
				state.observation = {
					boot,
					episode: observation.episode,
					revision: observation.revision,
					solved: observation.solved,
				};
				return state.lastAction ? { ...state.lastAction } : null;
			}, ctx);
			return {
				content: [
					...text({
						episode: observation.episode,
						revision: observation.revision,
						solved: observation.solved,
						lastAction,
					}),
					...observation.content,
				],
			};
		},
	});
	const act = defineTool({
		name: "policy_act",
		description:
			"Execute one named robot primitive using its advertised schema. Requires a fresh observation; interrupted actions are never automatically replayed.",
		parameters: Type.Object({ tool: Type.String(), arguments: Type.Record(Type.String(), Type.Unknown()) }),
		executionMode: "sequential",
		execute: async (args, api, ctx) => {
			if (!world.actions.some((tool) => tool.name === args.tool))
				throw new Error(`primitive not enabled: ${args.tool}`);
			const expected = await api.commit(async (tx) => {
				const state = await tx.doc(PolicyDoc, api.conversationId);
				assertFresh(state);
				if (state.actions >= maxActions)
					throw new Error("goal action budget exhausted; finish or wait for operator input");
				state.actions++;
				state.lastAction = {
					callId: api.callId,
					tool: args.tool,
					episode: world.identity().episode,
					outcome: "unknown",
				};
				const expected = { ...world.identity() };
				state.observation = null;
				return expected;
			}, ctx);
			const now = world.identity();
			if (now.episode !== expected.episode || now.revision !== expected.revision)
				throw new Error("world changed before dispatch; observe again");
			const content = await world.act(args.tool, args.arguments, ctx.abortSignal);
			await api.commit(async (tx) => {
				const state = await tx.doc(PolicyDoc, api.conversationId);
				if (state.lastAction?.callId === api.callId) state.lastAction.outcome = "returned";
			}, ctx);
			return { content };
		},
	});
	const finish = defineTool({
		name: "policy_finish",
		description:
			"Conclude this goal or wait for external input. Success requires fresh environment-confirmed success. Keeps the policy conversation alive for future goals.",
		parameters: Type.Object({
			status: StringEnum(["success", "failure", "waiting"] as const),
			summary: Type.String({ maxLength: 4096 }),
		}),
		executionMode: "sequential",
		execute: async (args, api, ctx) => {
			await api.commit(async (tx) => {
				const state = await tx.doc(PolicyDoc, api.conversationId);
				const goal = currentGoal(state);
				if (!goal || goal.status !== "active") throw new Error("no active goal");
				if (args.status === "success") {
					assertFresh(state);
					if (!state.observation?.solved) throw new Error("environment has not confirmed success");
				}
				goal.status = args.status;
				goal.summary = args.summary;
				state.observation = null;
			}, ctx);
			return { content: text(args), control: { terminate: true } };
		},
	});
	const remember = defineTool({
		name: "policy_remember",
		description:
			"Replace concise working memory retained across goals. Distinguish observations, hypotheses and reusable lessons.",
		parameters: Type.Object({ memory: Type.String({ maxLength: 8192 }) }),
		replay: "safe",
		executionMode: "sequential",
		execute: async (args, api, ctx) => {
			await api.commit(async (tx) => {
				(await tx.doc(PolicyDoc, api.conversationId)).memory = args.memory;
			}, ctx);
			return { content: text({ saved: true }) };
		},
	});
	const registry = createRegistry();
	registry.install(
		defineExtension({
			name: "embodied-policy",
			tools: [observe, act, finish, remember],
			sections: [
				section("policy", async (input) => {
					const state = await input.read.snapshot(PolicyDoc, input.conversationId, context);
					return `You are a persistent embodied policy for ${world.robot}. Work on the host's current goal. Observe, choose one bounded primitive, observe again, and verify. Never infer current physical state from old transcript or memory. A previous action of unknown outcome may already have happened; reconcile using new observations. After reset the episode identity changes. Use policy_finish to finish or wait for an external event; do not poll while waiting. A success claim must match the environment's task criterion.\nCurrent goal: ${JSON.stringify(state && currentGoal(state))}\nWorking memory: ${state?.memory ?? ""}\nEnabled primitives (parameters are JSON schemas): ${JSON.stringify(world.actions)}`;
				}),
			],
			hooks: [
				hook(GenerationTask, {
					onYield: async (answer, api, ctx) => {
						await root.commit(async (tx) => {
							const state = await tx.doc(PolicyDoc, api.conversationId);
							const goal = currentGoal(state);
							if (goal?.status === "active") {
								goal.status = "waiting";
								goal.summary = answer.content
									.filter((c) => c.type === "text")
									.map((c) => c.text)
									.join("\n")
									.slice(0, 4096);
							}
						}, ctx);
						return undefined;
					},
				}),
			],
		}),
	);
	harness = await Harness.open(
		options.storage,
		{ models, registry, settings: { toolExecution: "sequential", retry: { maxRetries: 0 } } },
		context,
	);
	try {
		root = await harness.root(context, { agent: { model: options.model, thinkingLevel: "low" } });
		await root.commit(async (tx) => {
			const state = await tx.doc(PolicyDoc, root.id);
			if (state.robot && state.robot !== world.robot) throw new Error("policy store belongs to another robot");
			state.robot = world.robot;
			state.observation = null;
		}, context);
		const state = async () => (await harness.snapshot(PolicyDoc, root.id, context))!;
		const submit = async (event: GoalEvent) => {
			if (!event.goalId.trim() || !event.eventId.trim() || !event.goal.trim())
				throw new Error("goalId, eventId and goal are required");
			const key = JSON.stringify([event.goalId, event.eventId]);
			const input = event.input ?? "";
			await root.commit(async (tx) => {
				const live = await tx.doc(LiveDoc, root.id);
				const doc = await tx.doc(PolicyDoc, root.id);
				const existing = doc.events.find((e) => e.key === key);
				if (existing) {
					if (existing.goal !== event.goal || existing.input !== input)
						throw new Error("event ID reused with different content");
					return;
				}
				if (live.run) throw new Error("policy is busy; wait before submitting another event");
				const active = currentGoal(doc);
				if (active && active.id !== event.goalId && (active.status === "active" || active.status === "waiting"))
					throw new Error("finish the current goal before starting another");
				let goal = doc.goals.find((g) => g.id === event.goalId);
				if (goal && (goal.text !== event.goal || goal.status === "success" || goal.status === "failure"))
					throw new Error("completed or changed goal requires a new goal ID");
				if (!goal) {
					goal = { id: event.goalId, text: event.goal, status: "active", summary: "" };
					doc.goals.push(goal);
				}
				goal.status = "active";
				if (doc.current !== event.goalId) doc.actions = 0;
				doc.current = event.goalId;
				doc.requests = 0;
				doc.observation = null;
				doc.events.push({ key, goalId: event.goalId, goal: event.goal, eventId: event.eventId, input });
			}, context);
			// If the process dies between these commits, resubmitting the same event completes admission.
			return root.submit(
				{
					type: "input",
					requestId: key,
					content: `Goal ${event.goalId}: ${event.goal}\nEvent ${event.eventId}: ${input || "Begin or resume this goal."}`,
					whenBusy: "reject",
				},
				context,
			);
		};
		return {
			harness,
			conversation: root,
			state,
			submit,
			resume: async () => {
				const saved = await state();
				const last = saved.events.at(-1);
				if (!last || currentGoal(saved)?.status !== "active") return undefined;
				return submit(last);
			},
			close: () => harness.close(context),
		};
	} catch (error) {
		await harness.close(context);
		throw error;
	}
}

export type PersistentPolicy = Awaited<ReturnType<typeof openPolicy>>;
