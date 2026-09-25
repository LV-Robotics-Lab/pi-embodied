/**
 * Per-call VLA seeds (`--vla-seed`), so a replayed episode samples the same policy noise.
 *
 * Every VLA inference gets a seed that depends only on the episode and the call's index within
 * it (the index restarts on every env reset, exploration's included):
 *   episode (default)  hash(robot, task, episode seed, call index) & 0x7fffffff
 *   <integer base>     (base + call index) & 0x7fffffff
 *   off                no seed; the servers sample as before
 * The servers apply a seed to that one inference only (see services' `seeded`).
 */

import { createHash } from "node:crypto";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export type VlaSeeds = {
	/** The seed for the next VLA call of this episode (advances the call index), or undefined when off. */
	next(): number | undefined;
	/** Call on every env reset: the next call is the episode's first again. */
	reset(): void;
};

/** Registers `--vla-seed`; `episode` is the robot's name and task, e.g. ["libero", robot.task]. */
export function vlaSeeds(pi: ExtensionAPI, episode: () => unknown): VlaSeeds {
	pi.registerFlag("vla-seed", {
		type: "string",
		default: "episode",
		description: "VLA sampling seed per call: episode (derived from task, seed, call index) | <int base> | off",
	});
	let index = 0;
	return {
		next() {
			const mode = String(pi.getFlag("vla-seed") ?? "episode");
			if (mode === "off") return undefined;
			const i = index++;
			if (mode === "episode") {
				const digest = createHash("sha256")
					.update(JSON.stringify([episode(), i]))
					.digest();
				return digest.readUInt32BE(0) & 0x7fffffff;
			}
			if (!/^\d+$/.test(mode)) throw new Error(`--vla-seed must be episode, off or an integer, not ${mode}`);
			return (Number(mode) + i) % 2 ** 31;
		},
		reset() {
			index = 0;
		},
	};
}
