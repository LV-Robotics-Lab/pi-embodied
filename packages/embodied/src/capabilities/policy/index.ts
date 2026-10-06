/** Opt-in persistent policy, hosted as a tool so all robot calls retain the normal execution pipeline. */
import { randomUUID } from "node:crypto";
import { resolve } from "node:path";
import { type AssistantMessageEventStream, createModels, createProvider } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { VLM_COST_EVENT } from "../../modes/units/vlm.ts";
import { type RobotStatus, STATUS_EVENT } from "../../robot.ts";
import { metaworldPolicyWorld } from "./tool-world.ts";

/**
 * Mounted by the MetaWorld robot (../../robots/metaworld) after its own tools, so `--policy-store` is one
 * of the robot's tracked flags (the result records it in `params`: cells with a store and cells without
 * are different configurations) and `policy_goal` is declared with the other module tools
 * (../../primitives/manifests/common/modules.json). Pi Durable and Chord are optional peers: they are
 * imported when a goal is submitted, never when the robot loads.
 */
export function persistentPolicy(pi: ExtensionAPI) {
	pi.registerFlag("policy-store", { type: "string", description: "SQLite file for the persistent MetaWorld policy" });
	let robot: RobotStatus | undefined;
	let episode = randomUUID();
	let revision = 0;
	pi.events.on(STATUS_EVENT, (status) => {
		robot = status as RobotStatus;
	});
	pi.on("session_start", () => {
		episode = randomUUID();
		revision = 0;
	});
	pi.on("tool_execution_end", (event) => {
		if (event.toolName === "reset" || event.toolName === "request_scene_reset") episode = randomUUID();
		revision++;
	});
	pi.on("before_agent_start", () => {
		if (pi.getFlag("policy-store")) pi.setActiveTools([...new Set([...pi.getActiveTools(), "policy_goal"])]);
	});
	pi.registerTool({
		name: "policy_goal",
		label: "Persistent policy",
		description:
			"Delegate the current MetaWorld environment task to a policy whose goal history and working memory survive episodes and process restarts. Repeat the same goal_id/event_id to recover; use a new event_id for feedback on a waiting goal, and a new goal_id after completion. The policy uses the current model and robot tool safety checks.",
		parameters: Type.Object({
			goal_id: Type.String({ minLength: 1 }),
			goal: Type.String({ minLength: 1 }),
			event_id: Type.String({ minLength: 1 }),
			input: Type.Optional(Type.String()),
		}),
		executionMode: "sequential",
		async execute(_id, args, signal, _update, ctx) {
			const file = pi.getFlag("policy-store");
			if (typeof file !== "string" || !file.trim())
				throw new Error("set --policy-store to enable persistent policy");
			if (robot?.robot !== "metaworld" || !robot.ready || robot.ended)
				throw new Error("persistent policy currently requires an active MetaWorld episode");
			if (!ctx.model) throw new Error("select a policy model");
			const model = ctx.model;
			const [{ openStoredPolicy }, { BACKGROUND_CONTEXT, withAbortSignal }] = await Promise.all([
				import("./stored.ts"),
				import("@earendil-works/chord/context"),
			]);
			// Delegate every request to the host registry, including its configured providers and OAuth.
			const models = createModels();
			const account = (stream: AssistantMessageEventStream) => {
				void stream.result().then((message) => pi.events.emit(VLM_COST_EVENT, message.usage.cost.total));
				return stream;
			};
			models.setProvider(
				createProvider({
					id: model.provider,
					models: [model],
					auth: { apiKey: { name: "Host registry", resolve: async () => ({ auth: {} }) } },
					api: {
						stream: (m, context, options) => account(ctx.modelRegistry.stream(m, context, options)),
						streamSimple: (m, context, options) => account(ctx.modelRegistry.streamSimple(m, context, options)),
					},
				}),
			);
			const policy = await openStoredPolicy(resolve(ctx.cwd, file), {
				models,
				model: { provider: model.provider, modelId: model.id },
				world: metaworldPolicyWorld(ctx, () => ({ episode, revision: String(revision) })),
			});
			try {
				if (signal?.aborted) throw new Error("policy cancelled");
				const submission = await policy.submit({
					goalId: args.goal_id,
					goal: args.goal,
					eventId: args.event_id,
					input: args.input,
				});
				const receipt = await submission.wait(
					signal ? withAbortSignal(signal, BACKGROUND_CONTEXT) : BACKGROUND_CONTEXT,
				);
				const state = await policy.state();
				const result = {
					receipt,
					goal: state.goals.find((goal) => goal.id === args.goal_id),
					memory: state.memory,
					actions: state.actions,
				};
				return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
			} finally {
				await policy.close();
			}
		},
	});
}
