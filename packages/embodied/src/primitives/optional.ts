/**
 * A shared tool behind a boolean flag, registered only when the flag is on: the flag is registered
 * at load, and `activate()` (called from the robot's `start`) registers the tools at the first start
 * that has the flag on, through the robot's own mount, and names them. Off registers nothing.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { ToolDef } from "./steps.ts";

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
