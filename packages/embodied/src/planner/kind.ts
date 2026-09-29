/**
 * Who planned an episode, from the providers of its assistant messages: `human` (./human.ts) when
 * any turn was a person's, else the planner provider of the last turn (`ensemble`, `fallback`,
 * `finetuned`, `flash`, `replay`), else `model`. The robot's result (`planner`), the memory recipe
 * sidecar (../capabilities/memory) and planner export (services flywheel/planner_export.py) read it, so a human
 * or replayed solve is never counted or trained on as a model's.
 */

export const PLANNERS = ["model", "human", "ensemble", "fallback", "finetuned", "flash", "replay"] as const;
export type Planner = (typeof PLANNERS)[number];

export function plannerOf(providers: Iterable<string | undefined>): Planner {
	let last: Planner = "model";
	for (const p of providers) {
		if (p === "human") return "human";
		last = (PLANNERS as readonly string[]).includes(p ?? "") && p !== "model" ? (p as Planner) : "model";
	}
	return last;
}

/** The providers of a session branch's assistant messages, in order. */
export function branchProviders(entries: readonly { type: string; message?: { role?: string; provider?: string } }[]) {
	return entries
		.filter((e) => e.type === "message" && e.message?.role === "assistant")
		.map((e) => e.message?.provider);
}
