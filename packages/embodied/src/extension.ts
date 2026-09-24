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
		const budget = new ImageBudget(options.images);
		pi.on("context", (event) => {
			const messages = budget.apply(event.messages);
			episode.imagesPruned = budget.prunedCount;
			return messages === event.messages ? undefined : { messages };
		});
	};
}
