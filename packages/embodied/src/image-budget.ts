/**
 * Keep camera images in the model context within a byte budget.
 *
 * Every mutating tool returns fresh frames, so a long episode would otherwise
 * resend dozens of stale images on every turn. Older images are replaced by a
 * short text marker; the newest `keepLatest` images always survive.
 *
 * Pruning is done with hysteresis: when the retained images exceed `maxBytes`,
 * the cut moves forward until they fit in `maxBytes * refillRatio`, and then
 * stays put until the budget is exceeded again. The start of the transcript is
 * therefore byte-identical across many turns, so provider prefix caches keep
 * hitting instead of missing on every call.
 *
 * Images are counted in transcript order, which only ever grows, so the cut is
 * a stable ordinal ("the first N images are pruned").
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";

export interface ImageBudgetOptions {
	/** Base64 bytes of images to retain. Default 4 MB. */
	maxBytes?: number;
	/** Newest images that are never pruned. Default 2. */
	keepLatest?: number;
	/** After a prune, retained bytes are brought under maxBytes * refillRatio. Default 0.5. */
	refillRatio?: number;
}

interface ImageRef {
	message: number;
	part: number;
	bytes: number;
}

const MARKER = "[earlier camera frame omitted to save context; the latest frames are shown below]";

export class ImageBudget {
	private pruned = 0;
	private readonly maxBytes: number;
	private readonly keepLatest: number;
	private readonly refillRatio: number;

	constructor(options: ImageBudgetOptions = {}) {
		this.maxBytes = options.maxBytes ?? 4 * 1024 * 1024;
		this.keepLatest = options.keepLatest ?? 2;
		this.refillRatio = options.refillRatio ?? 0.5;
	}

	/** Number of images replaced by markers in the most recent `apply`. */
	get prunedCount(): number {
		return this.pruned;
	}

	apply(messages: AgentMessage[]): AgentMessage[] {
		const refs: ImageRef[] = [];
		messages.forEach((msg, m) => {
			if ((msg.role === "toolResult" || msg.role === "user") && Array.isArray(msg.content)) {
				msg.content.forEach((part, p) => {
					if (part.type === "image") refs.push({ message: m, part: p, bytes: part.data.length });
				});
			}
		});

		const limit = Math.max(0, refs.length - this.keepLatest);
		this.pruned = Math.min(this.pruned, limit);
		const retained = () => refs.slice(this.pruned).reduce((sum, r) => sum + r.bytes, 0);
		if (retained() > this.maxBytes) {
			const target = this.maxBytes * this.refillRatio;
			while (this.pruned < limit && retained() > target) this.pruned++;
		}
		if (this.pruned === 0) return messages;

		const drop = new Map<number, Set<number>>();
		for (const ref of refs.slice(0, this.pruned)) {
			if (!drop.has(ref.message)) drop.set(ref.message, new Set());
			drop.get(ref.message)?.add(ref.part);
		}
		return messages.map((msg, m) => {
			const parts = drop.get(m);
			if (!parts || !("content" in msg) || !Array.isArray(msg.content)) return msg;
			const content = msg.content.map((part, p) => (parts.has(p) ? { type: "text" as const, text: MARKER } : part));
			return { ...msg, content } as AgentMessage;
		});
	}
}
