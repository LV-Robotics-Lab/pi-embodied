/**
 * pi extension wiring for one embodied episode: robot tools plus context hygiene.
 *
 * Used by the batch evaluator through `extensionFactories`, so evaluation and
 * interactive use share the same tool and context behavior.
 */

import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { ImageBudget, type ImageBudgetOptions } from "./image-budget.ts";
import { buildPiTools, type Episode, type Robot } from "./toolkit.ts";

export interface EpisodeExtensionOptions {
	images?: ImageBudgetOptions;
}

export function episodeExtension(
	robot: Robot,
	episode: Episode,
	options: EpisodeExtensionOptions = {},
): ExtensionFactory {
	return (pi) => {
		for (const tool of buildPiTools(robot, episode)) pi.registerTool(tool);

		// Assistant message_end is awaited before its tool calls execute, so every
		// tool in a batch that contains `finish` knows to terminate. pi only stops
		// early when all results in a batch set `terminate`.
		pi.on("message_end", (event) => {
			const msg = event.message;
			if (msg.role !== "assistant") return;
			episode.endsAfterBatch = msg.content.some((part) => part.type === "toolCall" && part.name === "finish");
		});
		pi.on("tool_call", () =>
			episode.finish
				? { block: true, reason: "The episode has already finished; this call was not executed.", terminate: true }
				: undefined,
		);
		pi.on("tool_result", (event) => (episode.failedCalls.has(event.toolCallId) ? { isError: true } : undefined));

		const budget = new ImageBudget(options.images);
		pi.on("context", (event) => {
			const messages = budget.apply(event.messages);
			episode.imagesPruned = budget.prunedCount;
			return messages === event.messages ? undefined : { messages };
		});
	};
}
