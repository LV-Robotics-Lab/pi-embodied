/**
 * Third-party VLA policies as robot tools. The services' `openvla_server`, `openvla_oft_server` and
 * `gr00t_server` (../../../services/pi_embodied_services/components) all speak the Pi0.5 wire format
 * and answer in the env's own action space, so a robot mounts each of them the way it mounts
 * `pi0_pick`: `<name>_act` runs the policy closed-loop with the pick heuristics and records its
 * `vla_seeds`. A tool exists only when its `--<flag> <url>` is given (nothing is registered when off);
 * `serve.sh` starts the servers.
 */

import { Type } from "typebox";
import type { RpcClient } from "./rpc.ts";

export type VlaAdapter = { tool: string; flag: string; model: string };
export const VLA_ADAPTERS: readonly VlaAdapter[] = [
	{ tool: "openvla_act", flag: "openvla", model: "OpenVLA (7B, LIBERO fine-tune; one action per call)" },
	{ tool: "openvla_oft_act", flag: "openvla-oft", model: "OpenVLA-OFT (7B, LIBERO fine-tune; 8-step chunks)" },
	{ tool: "gr00t_act", flag: "gr00t", model: "GR00T N1.6/N1.7 (LIBERO libero_panda fine-tune; 16-step chunks)" },
];

/** The `pi0_pick` parameters, shared by every closed-loop grasp tool. */
export const PICK_PARAMETERS = Type.Object({
	prompt: Type.String({ description: "VLA instruction, e.g. 'pick up the black bowl'" }),
	max_chunks: Type.Optional(Type.Integer({ description: "Action-chunk budget (default 24)" })),
	lift_thresh: Type.Optional(Type.Number({ description: "Post-descent ascent for success, m (default 0.05)" })),
	gripper_closed_thresh: Type.Optional(
		Type.Number({ description: "Finger separation below which the gripper counts as closed (default 0.06)" }),
	),
	gripper_open_thresh: Type.Optional(
		Type.Number({ description: "Minimum finger separation accepted as holding (default 0.0)" }),
	),
	descent_thresh: Type.Optional(
		Type.Number({ description: "Required descent before lift detection, m (default 0.10)" }),
	),
});
export type PickParams = {
	prompt: string;
	max_chunks?: number;
	lift_thresh?: number;
	gripper_closed_thresh?: number;
	gripper_open_thresh?: number;
	descent_thresh?: number;
};

/** What a `<vla>_act` tool says: a closed-loop grasp with `model`, judged by the pick heuristics. */
export const pickDescription = (model: string) =>
	`${model} closed-loop grasp, run as action chunks until the pick heuristics say the object is lifted or the budget ends. Use it only for the grasp; you do every move_to and release. Success needs a descent then a lift with the gripper partly closed; it is a hint, confirm from gripper opening and the wrist image.`;

/**
 * The pick heuristics of `pi0_pick`, as a tracker fed after every chunk: success is a descent of
 * `descent_thresh`, then a lift of `lift_thresh` above the lowest point, with the gripper between
 * the open and closed thresholds.
 */
export function pickTracker(startZ: number, startGrip: number, p: PickParams) {
	const lift = p.lift_thresh ?? 0.05;
	const closed = p.gripper_closed_thresh ?? 0.06;
	const open = p.gripper_open_thresh ?? 0;
	const descent = p.descent_thresh ?? 0.1;
	let minZ = startZ;
	let postMinPeak = startZ;
	let minGrip = startGrip;
	return {
		/** Feed the state after a chunk; true once the grasp counts as a success. */
		update(z: number, grip: number): boolean {
			if (z < minZ) {
				minZ = z;
				postMinPeak = z;
			} else postMinPeak = Math.max(postMinPeak, z);
			minGrip = Math.min(minGrip, grip);
			return startZ - minZ >= descent && postMinPeak - minZ >= lift && grip >= open && grip < closed;
		},
		/** The measured motion, for the tool result. */
		summary: () => ({
			peak_lift_m: postMinPeak - minZ,
			descent_m: startZ - minZ,
			min_gripper_opening: minGrip,
		}),
	};
}

/** What a VLA server says of itself: its healthz name and, for the adapters, `vla.info`. */
export type VlaInfo = { service: string; model?: string; revision?: string | null; suite?: string | null };

/**
 * Which VLA answers on `client`: the server's healthz name and, for the adapters, `vla.info`
 * (`model`, `revision`, the LIBERO `suite` of a published fine-tune). Pi0.5 and RLDX have no
 * `vla.info`; that error leaves the healthz name alone.
 */
export async function vlaInfo(client: RpcClient): Promise<VlaInfo> {
	const health = await client.call<{ service?: string }>("healthz", {}, 10_000);
	const service = health.service ?? "unknown";
	try {
		const info = await client.call<Omit<VlaInfo, "service">>("vla.info", {}, 10_000);
		return { service, model: info.model, revision: info.revision, suite: info.suite };
	} catch {
		return { service };
	}
}

/** `info` for the episode's `robot_result`: the healthz name and `model@revision`. */
export const vlaIdentity = (info: VlaInfo) =>
	`${info.service} ${info.model ?? ""}${info.revision ? `@${info.revision}` : ""}`.trim();

/**
 * Why a VLA fine-tuned on one LIBERO suite must not run on `suite`: the adapters report the
 * checkpoint's suite (`libero_all` covers every suite; a custom checkpoint reports none and is
 * trusted). Undefined when the checkpoint fits.
 */
export function suiteMismatch(info: VlaInfo, suite: string): string | undefined {
	if (!info.suite || info.suite === "libero_all" || info.suite === suite) return undefined;
	return `${vlaIdentity(info)} is the ${info.suite} fine-tune, but this episode is ${suite}: start its server with --suite ${suite} (or --model-path) before using its grasp tool.`;
}
