/**
 * Dual-Franka configuration shared by the robot (./index.ts) and its tool modules
 * (./perception.ts, ./skills.ts): the task, calibration and perception config the services
 * report, the exploration prompt rewrite and the camera naming helpers.
 */

import type { Static, TSchema } from "typebox";
import type { Step as BaseStep } from "../primitives/steps.ts";
import type { Json, Mat } from "../robot.ts";

/** The prompt's single-episode lines, which exploration replaces rather than contradicts. */
export const REWRITE: [RegExp, string][] = [
	[
		/^5\. Finish only when the success evidence is visible and consistent with state\.$/m,
		"5. This is an exploration run: follow the Exploration workflow below. Success is only the operator's verdict.",
	],
	[
		/^Call describe_dual_franka_setup before acting\..*$/m,
		"Call describe_dual_franka_setup, then read the relevant memory and prior attempt notes and inspect view_env_state before the first motion. Follow the operator-confirmed exploration workflow. Do not finish successfully without a current operator verdict.",
	],
];

export type Task = {
	name: string;
	instruction: string;
	success_criteria: string;
	constraints: string[];
	setup: string;
	vla_instruction: string | null;
};
export type Camera = { T_right_camera: Mat; localization_validity: Json };
export type View = { raw_key: string; calibration_key: string; display_name: string };
export type Setup = {
	task: Task;
	cameras?: Record<string, Camera>;
	projection_views?: Record<string, View>;
	calibration_source?: string;
	T_right_base_left_base?: Mat;
	calibration_error?: string;
};
/** A recorded step (../primitives/steps.ts) and the camera views it saved. */
export type Step = BaseStep & { views: string[] };
/** The robot's tool registrar (./index.ts `tool`): records a fresh step after a mutating tool. */
export type RegisterTool = <P extends TSchema>(
	name: string,
	description: string,
	parameters: P,
	run: (p: Static<P>, signal: AbortSignal | undefined) => Promise<Json>,
) => void;

/** Task, calibration and perception config from the services (parse_config checks). */
export const SETUP_PY = `
import dataclasses, json, sys
from pi_embodied_services.robots.dual_franka.runtime_config import DEFAULT_CONFIG
from pi_embodied_services.robots.dual_franka.tasks import get_dual_franka_task
from pi_embodied_services.robots.franka.runtime_config import describe_calibration_source, set_robot_config_path, validate_calibration_sources
set_robot_config_path(sys.argv[2] or DEFAULT_CONFIG)
validate_calibration_sources()
out = {"task": dataclasses.asdict(get_dual_franka_task(int(sys.argv[1])))}
try:
    from pi_embodied_services.robots.dual_franka import perception as p
    bundle = p.load_calibration_bundle()
    out["cameras"] = {k: {"T_right_camera": p._transform_to_matrix(v["transformation"]), "localization_validity": v.get("localization_validity") or {}} for k, v in bundle.items() if isinstance(v, dict) and "transformation" in v}
    out["projection_views"] = p._projection_cameras()
    out["calibration_source"] = describe_calibration_source()
    try:
        out["T_right_base_left_base"] = p._base_frame_transform(bundle, target="right_base", source="left_base")
    except ValueError:
        pass
except Exception as exc:
    out["calibration_error"] = str(exc)
print(json.dumps(out, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))
`;

/** `left_wrist_0_rgb` -> `left_wrist`. */
export function alias(key: unknown): string | undefined {
	let k = String(key ?? "");
	if (k.endsWith("_rgb")) k = k.slice(0, -4);
	if (k.endsWith("_0")) k = k.slice(0, -2);
	return k || undefined;
}

/** Whether the agent sees a wrist view: an inline camera named `*wrist*` (the default D455 alone is none). */
export function inlineWrist(meta: Json | null | undefined): boolean {
	return policy(meta).inline_cameras.some((c) => c.includes("wrist"));
}

export function policy(meta: Json | null | undefined): { inline_cameras: string[]; auxiliary_cameras: string[] } {
	const raw = meta?.agent_observation && typeof meta.agent_observation === "object" ? meta.agent_observation : {};
	const strings = (v: unknown) => (Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []);
	return {
		inline_cameras: strings(raw.inline_cameras ?? ["d455"]),
		auxiliary_cameras: strings(raw.auxiliary_cameras ?? ["left_wrist", "base", "right_wrist"]),
	};
}
