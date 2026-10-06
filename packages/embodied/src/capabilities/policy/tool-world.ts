import type { ExtensionToolContext } from "@earendil-works/pi-coding-agent";
import type { PolicyWorld, WorldIdentity } from "./policy.ts";

/** Reuse the host's complete tool pipeline: schemas, hooks, approvals, limits and motion serialization. */
export function metaworldPolicyWorld(
	ctx: Pick<ExtensionToolContext, "tools" | "executeTool">,
	identity: () => WorldIdentity,
): PolicyWorld {
	const actions = ctx.tools
		.filter((tool) => ["move_delta", "set_gripper"].includes(tool.name))
		.map(({ name, description, parameters }) => ({ name, description, parameters }));
	if (actions.length !== 2 || !ctx.tools.some((tool) => tool.name === "view_env_state"))
		throw new Error("policy requires MetaWorld tool mode: view_env_state, move_delta and set_gripper");
	return {
		robot: "metaworld",
		actions,
		identity,
		observe: async (signal) => {
			const result = await ctx.executeTool("view_env_state", { fresh: true }, { signal });
			if (result.isError)
				throw new Error(
					result.result.content
						.filter((c) => c.type === "text")
						.map((c) => c.text)
						.join("\n"),
				);
			const details = result.result.details as { success?: boolean; result?: { refreshed?: boolean } } | undefined;
			if (details?.result?.refreshed !== true)
				throw new Error("observation did not refresh the environment; update the MetaWorld extension");
			return { ...identity(), solved: details.success === true, content: result.result.content };
		},
		act: async (name, args, signal) => {
			if (!actions.some((tool) => tool.name === name)) throw new Error(`primitive not enabled: ${name}`);
			const result = await ctx.executeTool(name, args, { signal });
			if (result.isError)
				throw new Error(
					result.result.content
						.filter((c) => c.type === "text")
						.map((c) => c.text)
						.join("\n"),
				);
			return result.result.content;
		},
	};
}
