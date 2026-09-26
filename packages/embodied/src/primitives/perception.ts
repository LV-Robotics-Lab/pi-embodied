/**
 * Read-only views of a recorded state step (../franka, ../dual_franka): `view_env_state` shows a
 * step as the robot presents it, images attached; `view_camera_meta` its camera metadata. The
 * robot supplies its steps, its presentation and the tool description.
 */

import { Type } from "typebox";
import type { Json } from "../robot.ts";
import { getStep, type Step, stepParam, toolDef } from "./steps.ts";

/** `view_env_state`: the step (default the latest), shown by the robot's `view`, its images in `_pngs`. */
export function viewEnvState<S extends Step>(
	o: { steps: S[]; view: (s: S) => { output: Json; pngs: Buffer[] } },
	description: string,
) {
	return toolDef("view_env_state", description, Type.Object({ step: stepParam }), async ({ step = -1 }) => {
		const { output, pngs } = o.view(getStep(o.steps, step));
		return { ...output, _pngs: pngs };
	});
}

/** `view_camera_meta`: the camera metadata recorded with the step (default the latest). */
export function viewCameraMeta<S extends Step>(o: { steps: S[] }, description: string) {
	return toolDef("view_camera_meta", description, Type.Object({ step: stepParam }), async ({ step = -1 }) => {
		const s = getStep(o.steps, step);
		if (!s.meta) return { error: "camera metadata is unavailable", step };
		return { step: s.blob.step_idx, camera_meta: s.meta };
	});
}
