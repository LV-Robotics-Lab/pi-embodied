/**
 * `--replay-thinking`: whether the planner's earlier turns' thinking goes back out in every request.
 *
 * pi replays an assistant message as recorded, thinking included: pi-ai's openai-completions
 * provider sends each earlier turn's thinking as `reasoning_content` (packages/ai/src/api/
 * openai-completions.ts, convertMessages), and an endpoint whose chat template keeps it (NInfer /
 * vLLM with Qwen's) counts it as prompt tokens. On the fixed Qwen regression that was ~36% of the
 * planner's input (docs/reports, SoL-Pi A/B). The model does not read its own earlier reasoning as
 * instructions, so with `false` (the default) a request keeps only the latest assistant message's
 * thinking; `true` is pi's own behaviour. pi-ai's compat has no such switch (`requiresThinkingAsText`
 * rewrites thinking, nothing omits it), and a models.json field would not be recorded in `params`, so
 * it is a flag and a `context` handler (../robot.ts): only the outgoing request changes, the session
 * keeps every block, and `params` records the value for params-match.mjs.
 *
 * Fail safe: only unsigned thinking of an openai-completions assistant message is dropped. A message
 * of any other api (Anthropic's signed blocks, OpenAI Responses' encrypted reasoning items, Bedrock,
 * Google) stays whole, as does a block that is redacted or carries a signature other than the plain
 * reasoning field name pi-ai uses for a text-only replay ("reasoning", "reasoning_content",
 * "reasoning_text"), such as OpenRouter's `reasoning_details`.
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, ThinkingContent } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export const FLAG = "replay-thinking";

/** Signatures pi-ai's openai-completions provider gives a thinking block that is only text: the reasoning field it came in. */
const TEXT_ONLY = new Set(["reasoning", "reasoning_content", "reasoning_text"]);

/** Whether `block` is thinking an openai-completions request could replay only as text: safe to leave out. */
export function unsigned(block: { type: string } & Partial<Omit<ThinkingContent, "type">>): block is ThinkingContent {
	if (block.type !== "thinking") return false;
	const t = block as ThinkingContent;
	return !t.redacted && (!t.thinkingSignature || TEXT_ONLY.has(t.thinkingSignature));
}

/**
 * `messages` with the unsigned thinking of every openai-completions assistant message before the last
 * assistant message left out, or undefined when nothing would change (the context is left alone).
 * New arrays, never edited in place: the session's messages are untouched.
 */
export function dropReplayedThinking(messages: readonly AgentMessage[]): { messages: AgentMessage[] } | undefined {
	let last = -1;
	for (let i = messages.length - 1; i >= 0; i--)
		if (messages[i].role === "assistant") {
			last = i;
			break;
		}
	let changed = false;
	const out = messages.map((m, i) => {
		if (i >= last || m.role !== "assistant") return m;
		const a = m as AssistantMessage;
		if (a.api !== "openai-completions" || !Array.isArray(a.content) || !a.content.some(unsigned)) return m;
		changed = true;
		return { ...a, content: a.content.filter((block) => !unsigned(block)) };
	});
	return changed ? { messages: out } : undefined;
}

/** The flag's value, true or false; pi stores a flag's command-line value verbatim, so a bare `--replay-thinking` reads true. */
export const replayThinking = (pi: ExtensionAPI) => String(pi.getFlag(FLAG)) === "true";

/** Why the flag's value is refused (not true / false), else undefined. */
export function replayThinkingError(pi: ExtensionAPI): string | undefined {
	const v = String(pi.getFlag(FLAG) ?? "false");
	return v === "true" || v === "false" ? undefined : `--${FLAG} must be true or false, got ${JSON.stringify(v)}`;
}
