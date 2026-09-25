/**
 * Visual differencing (VDM): a second vision model describes what each robot action changed.
 *
 *   pi -e packages/embodied/src/libero --vdm [--vdm-model provider/id] [--vdm-wrist] ...
 *
 * A robot opts in with `vdm` in its defineRobot spec (how many camera images its observation results
 * carry, and which one is the wrist view); ../robot.ts mounts this module. With `--vdm`, the first
 * robot tool result that carries the robot's camera views gets a description of the initial scene,
 * and every later one a description of what changed between the previous and the current image of
 * the same view(s) and whether the task looks complete, appended to the result as text. `--vdm-wrist`
 * adds the wrist view to both. Results with another number of images (a segmentation overlay) are
 * not observations and are left alone. Each call is a `vdm` session entry; its cost counts toward
 * the robot's --max-cost budget (VLM_COST_EVENT). A failed call is noted in the result and the
 * episode goes on. Off (the default), no result is touched and nothing is written.
 *
 * The previous frame is not restored on resume or fork: every session start restarts the robot's
 * episode, so its first observation is a new initial scene.
 *
 * Prompts: CaP-X (github.com/capgym/cap-x @53e9966) capx/envs/trial.py _describe_initial_scene and
 * _get_visual_differencing_feedback, and the feedback headers of capx/utils/launch_utils.py.
 */

import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { askVlm, VLM_COST_EVENT, type VlmPrompt } from "./units/vlm.ts";

/** Session entry per VDM call: `{ kind: "initial" | "diff", tool, model, text | error, wrist, ms, cost_usd }`. */
export const VDM_ENTRY = "vdm";

export type VdmSpec = {
	/** The number of camera images in an observation result, in order; the first is the main view. */
	views: number;
	/** Index of the wrist view among them (`--vdm-wrist`). */
	wrist?: number;
};

type Frame = { main: ImageContent; wrist?: ImageContent };
const text = (t: string): TextContent => ({ type: "text", text: t });

const RULE = "You should try to provide objective information and no assumptions. Do *NOT* write any code.";

/** trial.py _describe_initial_scene. */
export function initialPrompt(task: string, frame: Frame): VlmPrompt {
	return {
		system: `You are a helpful assistant that describes the initial state of the environment with the goal of the task in mind. ${RULE}`,
		content: [
			text(task),
			text(`Describe the initial state of the environment with the goal of the task in mind. ${RULE}`),
			text("Main camera view:"),
			frame.main,
			...(frame.wrist ? [text("Wrist camera view:"), frame.wrist] : []),
		],
	};
}

/** trial.py _get_visual_differencing_feedback (the wrist pair only when both frames have one). */
export function diffPrompt(task: string, before: Frame, after: Frame): VlmPrompt {
	const goal =
		"the difference between the current state of the environment and the previous state of the environment with the goal of the task in mind and whether the task has been completed.";
	return {
		system: `You are a helpful assistant that describes ${goal} ${RULE}`,
		content: [
			text(task),
			text(`Describe ${goal} ${RULE}`),
			text("Previous state (main camera):"),
			before.main,
			text("Current state (main camera):"),
			after.main,
			...(before.wrist && after.wrist
				? [text("Previous state (wrist camera):"), before.wrist, text("Current state (wrist camera):"), after.wrist]
				: []),
		],
	};
}

/** The text appended to the tool result (launch_utils.py's headers, "code" read as the tool call). */
export const HEADERS = {
	initial: "The initial state of the environment is described as follows:",
	diff: "Included below is the observed differences between the current state of the environment (after the tool call above was executed) and the previous state of the environment (before it was executed):",
};

/**
 * Register the VDM flags and the tool_result hook. `tools()` names the robot's tools (only their
 * results are observations); `task()` is the task text given to the VDM model.
 */
export function vdm(pi: ExtensionAPI, spec: VdmSpec, tools: () => string[], task: () => string) {
	pi.registerFlag("vdm", {
		type: "boolean",
		default: false,
		description: "Visual differencing: a VLM describes the initial scene, then what each robot action changed",
	});
	pi.registerFlag("vdm-model", {
		type: "string",
		default: "",
		description: "Model (provider/id) of the VDM calls (default: --units-vlm-model, else the session's model)",
	});
	pi.registerFlag("vdm-wrist", {
		type: "boolean",
		default: false,
		description: "VDM: also give the model the wrist view",
	});
	const on = () => pi.getFlag("vdm") === true;
	let prev: Frame | undefined;
	let calls = 0;
	let errors = 0;
	let cost = 0;

	pi.on("session_start", () => {
		prev = undefined;
		calls = errors = cost = 0;
	});

	pi.on("tool_result", async (event, ctx) => {
		if (!on() || event.isError || !tools().includes(event.toolName)) return undefined;
		const shown = event.content.filter((c): c is ImageContent => c.type === "image");
		if (shown.length !== spec.views) return undefined;
		const wrist = pi.getFlag("vdm-wrist") === true && spec.wrist !== undefined ? shown[spec.wrist] : undefined;
		const frame: Frame = { main: shown[0], ...(wrist ? { wrist } : {}) };
		const before = prev;
		prev = frame;
		const kind = before ? "diff" : "initial";
		const modelRef = String(pi.getFlag("vdm-model") || pi.getFlag("units-vlm-model") || "");
		const entry: Record<string, unknown> = { kind, tool: event.toolName, wrist: Boolean(wrist) };
		const started = Date.now();
		let note: string;
		calls++;
		try {
			const reply = await askVlm(
				ctx,
				modelRef,
				pi.getThinkingLevel(),
				before ? diffPrompt(task(), before, frame) : initialPrompt(task(), frame),
				[],
				ctx.signal,
			);
			cost += reply.cost;
			pi.events.emit(VLM_COST_EVENT, reply.cost);
			Object.assign(entry, { model: reply.model, text: reply.text, cost_usd: reply.cost });
			note = `${HEADERS[kind]}\n${reply.text.trim()}`;
		} catch (err) {
			// The episode goes on without the description.
			errors++;
			entry.error = err instanceof Error ? err.message : String(err);
			note = `[visual differencing unavailable: ${entry.error}]`;
		}
		entry.ms = Date.now() - started;
		pi.appendEntry(VDM_ENTRY, entry);
		return { content: [...event.content, text(note)] };
	});

	return {
		/** The robot result's VDM fields. */
		result: () => ({
			vdm: on(),
			...(on()
				? {
						vdm_wrist: pi.getFlag("vdm-wrist") === true && spec.wrist !== undefined,
						vdm_calls: calls,
						vdm_errors: errors,
						vdm_cost_usd: Number(cost.toFixed(6)),
					}
				: {}),
		}),
	};
}
