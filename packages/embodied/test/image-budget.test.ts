import assert from "node:assert/strict";
import { test } from "node:test";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { ImageBudget } from "../src/image-budget.ts";

function toolResult(id: number, imageBytes: number[]): AgentMessage {
	return {
		role: "toolResult",
		toolCallId: `c${id}`,
		toolName: "move_to",
		isError: false,
		timestamp: id,
		content: [
			{ type: "text", text: `result ${id}` },
			...imageBytes.map((n) => ({ type: "image" as const, data: "x".repeat(n), mimeType: "image/png" })),
		],
	} as AgentMessage;
}

function images(msgs: AgentMessage[]): number {
	let n = 0;
	for (const m of msgs) {
		if (m.role === "toolResult") n += m.content.filter((p) => p.type === "image").length;
	}
	return n;
}

test("under budget leaves the transcript untouched", () => {
	const msgs = [toolResult(1, [100, 100]), toolResult(2, [100, 100])];
	const budget = new ImageBudget({ maxBytes: 1000 });
	assert.equal(budget.apply(msgs), msgs);
	assert.equal(budget.prunedCount, 0);
});

test("over budget prunes oldest images down to the refill target", () => {
	const msgs = [1, 2, 3, 4, 5].map((i) => toolResult(i, [100, 100]));
	const budget = new ImageBudget({ maxBytes: 900, refillRatio: 0.5 });
	const out = budget.apply(msgs);
	// 1000 bytes > 900 -> prune until <= 450 retained: keep 4 images (400 bytes).
	assert.equal(budget.prunedCount, 6);
	assert.equal(images(out), 4);
	const first = out[0] as { content: { type: string; text?: string }[] };
	assert.match(first.content[1].text ?? "", /omitted/);
	assert.equal(images(msgs), 10, "input messages are not mutated");
});

test("cut point is stable until the budget is exceeded again", () => {
	const budget = new ImageBudget({ maxBytes: 900, refillRatio: 0.5 });
	const msgs = [1, 2, 3, 4, 5].map((i) => toolResult(i, [100, 100]));
	budget.apply(msgs);
	msgs.push(toolResult(6, [100, 100]));
	budget.apply(msgs);
	assert.equal(budget.prunedCount, 6, "600 retained bytes fit, so the prefix does not move");
	msgs.push(toolResult(7, [100, 100]), toolResult(8, [100, 100]), toolResult(9, [100, 100]));
	budget.apply(msgs);
	assert.equal(budget.prunedCount, 14, "1200 bytes retained > 900, refill back to 400");
});

test("the newest images always survive even when each exceeds the budget", () => {
	const msgs = [toolResult(1, [5000]), toolResult(2, [5000]), toolResult(3, [5000])];
	const budget = new ImageBudget({ maxBytes: 100, keepLatest: 2 });
	const out = budget.apply(msgs);
	assert.equal(budget.prunedCount, 1);
	assert.equal(images(out), 2);
});
