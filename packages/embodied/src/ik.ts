/**
 * The `--ik <url>` flag: an IK service (services/pi_embodied_services/components/ik_server.py) the
 * env server asks whether a target is reachable before it moves (`env.preview_reach`). Off when
 * empty; the robot then behaves as before.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

/** `env.preview_reach`: `unknown` means the check could not run, which is not approval. */
export type Reach = {
	status: "reachable" | "unreachable" | "unknown";
	reachable: boolean | null;
	q: number[] | null;
	position_err: number | null;
	orientation_err: number | null;
	message: string;
	target: { frame: string; pos: number[]; quat_xyzw: number[] } | null;
};

export function registerIkFlag(pi: ExtensionAPI) {
	pi.registerFlag("ik", {
		type: "string",
		default: "",
		description:
			"IK server (components/ik_server.py): env.preview_reach and a reach check before each move; off when empty",
	});
}

/** The env server arguments for the flag: `--ik <url>`, or nothing when it is off. */
export function ikArgs(url: unknown): string[] {
	const u = typeof url === "string" ? url.trim() : "";
	return u ? ["--ik", u] : [];
}

/** The refusal a motion tool returns instead of moving, or undefined when the target may be tried. */
export function reachRefusal(reach: Reach): string | undefined {
	return reach.status === "unreachable" ? `refused: target is out of reach (${reach.message})` : undefined;
}
