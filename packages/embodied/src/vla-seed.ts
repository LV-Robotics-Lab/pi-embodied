/**
 * Per-call VLA seeds (`--vla-seed`), so a replayed episode samples the same policy noise.
 *
 * Every VLA inference gets a seed that depends only on the episode, the attempt (env resets
 * since the session started: exploration and operator resets count, the start is attempt 0) and
 * the call's index within the attempt:
 *   episode (default)  hash(robot, task, episode seed, call index[, attempt]) & 0x7fffffff
 *   <integer base>     (base + 1000000 * attempt + call index) mod 2^31
 *   off                no seed; the servers sample as before
 * So a retry after an exploration reset samples new noise, and a replay of the session, which
 * repeats its resets, repeats every seed. Attempt 0 hashes without the attempt, as before it was counted.
 * The servers apply a seed to that one inference only (see services' `seeded`).
 */

import { createHash } from "node:crypto";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export type VlaSeeds = {
	/** The seed for the next VLA call of this episode (advances the call index), or undefined when off. */
	next(): number | undefined;
	/** Call on every env reset, the session's first included: starts the next attempt at call index 0. */
	reset(): void;
};

/** Registers `--vla-seed`; `episode` is the robot's name and task, e.g. ["libero", robot.task]. */
export function vlaSeeds(pi: ExtensionAPI, episode: () => unknown): VlaSeeds {
	pi.registerFlag("vla-seed", {
		type: "string",
		default: "episode",
		description:
			"VLA sampling seed per call: episode (derived from task, seed, attempt, call index) | <int base> | off",
	});
	let attempt = -1;
	let index = 0;
	return {
		next() {
			const mode = String(pi.getFlag("vla-seed") ?? "episode");
			if (mode === "off") return undefined;
			const i = index++;
			const a = Math.max(attempt, 0);
			if (mode === "episode") {
				const key = a === 0 ? [episode(), i] : [episode(), i, a];
				return createHash("sha256").update(JSON.stringify(key)).digest().readUInt32BE(0) & 0x7fffffff;
			}
			if (!/^\d+$/.test(mode)) throw new Error(`--vla-seed must be episode, off or an integer, not ${mode}`);
			return (Number(mode) + 1_000_000 * a + i) % 2 ** 31;
		},
		reset() {
			attempt++;
			index = 0;
		},
	};
}
