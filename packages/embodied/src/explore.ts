/**
 * Exploration: enabled with --explore, driven by /explore.
 *
 *   pi -p -e packages/embodied/src/libero --explore --suite libero_10 --task 2 --seed 0 \
 *     --memory-dir memory/libero --output-dir runs/explore/10_t2_s0 --auto-merge-memory "/explore"
 *
 * /explore runs up to --explore-sessions fresh pi sessions on the cell, each opening on a clean
 * episode (the robot's session_start), and stops at the first solve; later sessions open with a
 * handoff naming the archived attempts. Within a session `reset` restarts the episode up to
 * --explore-attempts-per-session times, `finish` is refused while attempts remain on an unsolved
 * cell, a failed attempt must be archived before `reset` or `finish`, and the DISTIL instructions
 * arrive once the env reports `terminated`. The memory guard, recipe and merge are memory.ts's.
 */

import { existsSync, readdirSync } from "node:fs";
import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import type {
	ExtensionAPI,
	ExtensionCommandContext,
	ExtensionContext,
	SessionEntry,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const SESSION = "explore_session";

type Robot = {
	/** Restore the initial scene; return the new state as a tool result that embeds `result`. Throwing fails the reset. */
	reset: (
		result: Record<string, unknown>,
		ctx: ExtensionContext,
		signal?: AbortSignal,
	) => Promise<AgentToolResult<unknown>>;
	/** memory.ts `render`: fills {{output_dir}}, {{memory_dir}}, {{memory_inbox}}, {{recipe_tag}} and `extra`. */
	render: (text: string, extra?: Record<string, string | number>) => string;
	/** memory.ts `tools`: the built-in file tools the agent reads and writes memory with. */
	tools: readonly string[];
	/** The robot's exploration instructions, appended to its system prompt and rendered like memory text. */
	prompt: () => string;
	/** A DISTIL pass sent once the cell is solved; `finish` waits for it and its suite draft. */
	distil?: string;
	/** Lines of the robot's single-episode prompt that exploration replaces. */
	rewrite?: [RegExp, string][];
	/** Defaults of --explore-sessions and --explore-attempts-per-session (3 and 5). */
	budget?: { sessions: number; attempts: number };
	/** The operator aborted the run: `finish` is no longer held back for the attempt budget. */
	aborted?: () => boolean;
};
type Progress = {
	n: number;
	attempt: number;
	solved: boolean;
	finished: boolean;
	distilled: boolean;
	aborted: boolean;
};

const archives = (dir: string) => {
	try {
		return readdirSync(dir)
			.filter((f) => /^attempt_\d+_failed\.json$/.test(f))
			.sort();
	} catch {
		return [];
	}
};

const note = (customType: string, content: string) => ({
	type: "custom_message" as const,
	customType,
	content,
	display: true,
});

/** Session number and this session's attempts, solve, finish and DISTIL, from the branch. */
function progress(entries: SessionEntry[]): Progress {
	const p: Progress = { n: 1, attempt: 1, solved: false, finished: false, distilled: false, aborted: false };
	for (const e of entries) {
		if (e.type === "custom" && e.customType === SESSION) p.n = Number((e.data as { n?: number } | undefined)?.n ?? 1);
		if (e.type === "custom_message" && e.customType === "explore_distil") p.distilled = true;
		if (e.type !== "message") continue;
		const m = e.message;
		if (m.role === "assistant") p.aborted = m.stopReason === "aborted";
		if (m.role !== "toolResult" || m.isError) continue;
		if (m.toolName === "reset") p.attempt++;
		if (m.toolName === "finish") p.finished = true;
		if ((m.details as { terminated?: unknown } | undefined)?.terminated === true) p.solved = true;
	}
	return p;
}

function opening(n: number, max: number, dir: string, instruction: string): string {
	const prior = archives(dir);
	const text =
		n === 1 && !prior.length
			? "Explore the task. Read memory first, then start with `view_env_state`."
			: `You are agent ${n} of up to ${max} on this cell. ${prior.length} attempt(s) by earlier agents are archived in ${dir}/ (${prior.join(", ") || "none yet"}), and their working notes are in the memory inbox under wip/.\n\nRead every archive and the working notes before acting. Do not repeat failed approaches. A fresh session has already restored a clean scene; inspect it before acting.`;
	return instruction ? `${text}\n\nOriginal operator task instruction:\n${instruction}` : text;
}

/** Session n of max: a fresh session (so a fresh episode) that hands off to n+1 unless solved or aborted. */
async function run(ctx: ExtensionCommandContext, n: number, max: number, dir: string, instruction: string) {
	await ctx.newSession({
		parentSession: n > 1 ? ctx.sessionManager.getSessionFile() : undefined,
		setup: async (sm) => {
			sm.appendCustomEntry(SESSION, { n, max });
		},
		withSession: async (next) => {
			await next.sendUserMessage(opening(n, max, dir, instruction));
			const p = progress(next.sessionManager.getBranch());
			if (n < max && !p.solved && !p.aborted) await run(next, n + 1, max, dir, instruction);
		},
	});
}

export function explore(pi: ExtensionAPI, robot: Robot) {
	pi.registerFlag("explore", {
		type: "boolean",
		default: false,
		description: "Exploration run: resettable attempts, handoff sessions, memory drafts (start with /explore)",
	});
	pi.registerFlag("explore-sessions", {
		type: "string",
		default: String(robot.budget?.sessions ?? 3),
		description: "Independent sessions per exploration run",
	});
	pi.registerFlag("explore-attempts-per-session", {
		type: "string",
		default: String(robot.budget?.attempts ?? 5),
		description: "Attempts per exploration session (0 = unlimited)",
	});
	const on = () => pi.getFlag("explore") === true;
	const budget = () => Number(pi.getFlag("explore-attempts-per-session")) || 0;
	const sessions = () => Math.max(1, Number(pi.getFlag("explore-sessions")) || 1);
	const dir = () => robot.render("{{output_dir}}/attempts/{{recipe_tag}}");
	let archived = 0;
	let nudges = 0;
	let nagged = false;

	/** Why the current attempt may not end yet: it has not been archived. */
	function closeOut(p: Progress): string | undefined {
		const have = archives(dir()).length;
		if (have >= archived + p.attempt) return undefined;
		return robot.render(
			`Close out attempt ${p.attempt} first: write ${dir()}/attempt_${have + 1}_failed.json and append "## Attempt ${have + 1}" to {{memory_inbox}}/wip/notes.md.`,
		);
	}

	function refuseReset(p: Progress): string | undefined {
		const b = budget();
		if (!on()) return "reset is available only in exploration runs (--explore)";
		if (p.solved) return "reset refused: the cell is solved. Run the DISTIL pass and call `finish`.";
		if (b && p.attempt >= b)
			return `reset refused: this session's attempt budget is spent (${b} attempts). Archive the attempt, update the handoff notes, and call \`finish\` so the next session can continue.`;
		return closeOut(p);
	}

	function refuseFinish(p: Progress): string | undefined {
		const b = budget();
		if (robot.aborted?.()) return undefined;
		if (p.solved) {
			if (!robot.distil) return undefined;
			if (!p.distilled) return "finish refused: run the DISTIL pass first; its instructions follow.";
			if (nagged || existsSync(robot.render("{{memory_inbox}}/suite_{{recipe_tag}}_draft.md"))) return undefined;
			nagged = true;
			return robot.render(
				"finish refused once: DISTIL has not written {{memory_inbox}}/suite_{{recipe_tag}}_draft.md yet.",
			);
		}
		if (b && p.attempt < b)
			return `finish refused: this session has ${b - p.attempt} of its ${b} attempts left and the task is not solved. Archive this attempt, call \`reset\`, and try another approach.`;
		return closeOut(p);
	}

	pi.registerTool({
		name: "reset",
		label: "reset",
		description:
			"Exploration only. Abandon the current episode and restore the same initial scene. Archive the failed attempt first and state which strategy lever changes in the next attempt.",
		parameters: Type.Object({
			reason: Type.String({ description: "Why this episode is unrecoverable and what will change" }),
		}),
		executionMode: "sequential",
		async execute(_id, { reason }, signal, _onUpdate, ctx) {
			const p = progress(ctx.sessionManager.getBranch());
			const refusal = refuseReset(p);
			if (refusal) throw new Error(refusal);
			const attempt = p.attempt + 1;
			return robot.reset(
				{
					action: "reset",
					reason,
					attempt,
					notice: `Episode restarted; this is attempt ${attempt}. The original layout was restored. Re-run perception before acting.`,
				},
				ctx,
				signal,
			);
		},
	});

	pi.registerCommand("explore", {
		description: "Explore this cell over up to --explore-sessions fresh sessions, stopping at the first solve",
		handler: async (args, ctx) => {
			if (!on()) {
				if (ctx.hasUI) ctx.ui.notify("start pi with --explore to explore", "error");
				else console.error("[explore] start pi with --explore to explore");
				return;
			}
			await run(ctx, 1, sessions(), dir(), args.trim());
		},
	});

	pi.on("session_start", () => {
		nudges = 0;
		nagged = false;
		if (on()) archived = archives(dir()).length;
	});

	pi.on("before_agent_start", (event, ctx) => {
		if (!on()) return undefined;
		pi.setActiveTools([...new Set([...pi.getActiveTools(), ...robot.tools, "reset"])]);
		const p = progress(ctx.sessionManager.getBranch());
		const vars = { session_number: p.n, session_max: sessions(), attempt_budget: budget() || "unlimited" };
		const base = (robot.rewrite ?? []).reduce((text, [from, to]) => text.replace(from, to), event.systemPrompt);
		return { systemPrompt: `${base}\n\n${robot.render(robot.prompt(), vars)}` };
	});

	pi.on("tool_call", (event, ctx) => {
		if (!on() || event.toolName !== "finish") return undefined;
		const reason = refuseFinish(progress(ctx.sessionManager.getBranch()));
		return reason ? { block: true, reason } : undefined;
	});

	// The DISTIL pass starts in the turn that solved the cell, even when that turn also called finish.
	pi.on("turn_end", (event, ctx) => {
		if (!on() || !robot.distil) return undefined;
		const p = progress(ctx.sessionManager.getBranch());
		if (
			!p.solved ||
			p.distilled ||
			event.entries.some((e) => e.type === "custom_message" && e.customType === "explore_distil")
		)
			return undefined;
		return { entries: [...event.entries, note("explore_distil", robot.render(robot.distil))], continue: true };
	});

	// A model that stops talking has not handed off: send it back, at most twice per session.
	pi.on("agent_before_settle", (event, ctx) => {
		if (!on() || event.outcome !== "completed" || nudges >= 2) return undefined;
		const p = progress(ctx.sessionManager.getBranch());
		const b = budget();
		let text: string | undefined;
		if (p.finished || robot.aborted?.()) text = undefined;
		else if (p.solved)
			text = robot.distil
				? "The cell is solved. Complete the DISTIL pass, then call `finish`."
				: "The cell is solved. Write the memory proposals, then call `finish`.";
		else if (!b || p.attempt < b)
			text = `You stopped on an unsolved cell with ${b ? b - p.attempt : "unlimited"} attempt(s) left. Close out this attempt, \`reset\`, and try a class of approach you have not tried; \`finish\` is refused until the budget is spent.`;
		else
			text =
				"The attempt budget is spent. Close out the last attempt, write the unsolved audit, and call `finish` so the next agent can continue.";
		if (!text) return undefined;
		nudges++;
		return { entries: [...event.entries, note("explore_nudge", text)], continue: true };
	});

	return { on };
}
