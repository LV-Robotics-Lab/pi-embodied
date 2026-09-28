/**
 * A shared tool behind a boolean flag, registered only when the flag is on: the flag is registered
 * at load, and `activate()` (called from the robot's `start`) registers the tools at the first start
 * that has the flag on, through the robot's own mount, and names them. Off registers nothing.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { ToolDef } from "./steps.ts";

/** The OpenETA extras' flags (./waypoints.ts, ./wrist.ts, ./advisor.ts, ../objects.ts, ../web.ts), as the result records them. */
export const EXTRAS = ["waypoints", "align-wrist", "grasp-advisor", "object-memory", "web-tools"] as const;

const pureFlag = (v: unknown) => v === true || v === "true" || v === "pure";

/**
 * Pure units or code mode (--units=true / --code=true) leaves the model only `act` / `run_code`:
 * an extra turned on there would not be mounted, so the robot refuses to start (fails closed) with
 * the reason instead of recording a flag that had no effect.
 */
export function extrasInPureMode(pi: ExtensionAPI): string | undefined {
	const mode = pureFlag(pi.getFlag("units")) ? "--units" : pureFlag(pi.getFlag("code")) ? "--code" : undefined;
	const on = EXTRAS.filter((f) => pi.getFlag(f) === true);
	if (!mode || !on.length) return undefined;
	return `${on.map((f) => `--${f}`).join(", ")} cannot run in pure ${mode} mode (only act / run_code are mounted); use ${mode}=both or drop them`;
}

export function optionalTools<D extends { name: string } = ToolDef>(
	pi: ExtensionAPI,
	flag: string,
	description: string,
	make: () => D[],
	mount: (d: D) => void,
): () => string[] {
	pi.registerFlag(flag, { type: "boolean", default: false, description });
	let names: string[] | undefined;
	return () => {
		if (pi.getFlag(flag) !== true) return [];
		if (!names) {
			const defs = make();
			for (const d of defs) mount(d);
			names = defs.map((d) => d.name);
		}
		return names;
	};
}
