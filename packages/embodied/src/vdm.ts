/**
 * Visual differencing (VDM): a second vision model describes what each robot action changed.
 *
 *   pi -e packages/embodied/src/libero --vdm [--vdm-model provider/id] [--vdm-wrist] [--vdm-timeout 60] ...
 *
 * A robot opts in with `vdm` in its defineRobot spec (how many camera images its observation results
 * carry, which are wrist views, and which tools only look; a function when that depends on the
 * robot's cameras); ../robot.ts mounts this module. With `--vdm`, the first robot tool result that
 * carries the robot's camera views gets a description of the initial scene, and every later one of a
 * tool that moves the robot a description of what changed between the previous and the current image
 * of the same view(s) and whether the task looks complete, appended to the result as text. A tool in
 * `spec.observe` (view_env_state) moves nothing: its frame becomes the previous one, and no call is
 * made. A scene reset (exploration's `reset`, the operator's `request_scene_reset`) starts over: the
 * next observation, the reset's own when it carries the views, is described as a new initial scene
 * instead of being diffed against the old one. `--vdm-wrist` adds the wrist view(s) to both prompts
 * (a two-armed robot has one per arm). Results with another number of images (a segmentation overlay)
 * are not observations and are left alone. Each call is a `vdm` session entry; its cost counts toward
 * the robot's --max-cost budget (VLM_COST_EVENT). Each call has its own timeout (--vdm-timeout) and,
 * when eval-parallel.sh's --max-api-concurrency gate is loaded (./api-gate.ts), takes one of its
 * shared model-call slots like a planner call (the wait counts toward the timeout). A failed or
 * timed-out call is noted in the result and counted (`vdm_errors`); the episode goes on. An aborted
 * call (the episode ended) is recorded, not counted. Off (the default), only the flags exist: no
 * hook runs, no result is touched, and the robot result carries no vdm fields.
 *
 * The previous frame is not restored on resume or fork: every session start restarts the robot's
 * episode, so its first observation is a new initial scene.
 *
 * Prompts: CaP-X (github.com/capgym/cap-x @53e9966) capx/envs/trial.py _describe_initial_scene and
 * _get_visual_differencing_feedback, and the feedback headers of capx/utils/launch_utils.py.
 */

import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { API_GATE_EVENT, acquire, release } from "./api-gate.ts";
import { askVlm, VLM_COST_EVENT, type VlmPrompt } from "./units/vlm.ts";

/** Session entry per VDM call: `{ kind: "initial" | "diff", tool, model, text | error | aborted, wrist, ms, cost_usd }`. */
export const VDM_ENTRY = "vdm";

export type VdmSpec = {
	/** The number of camera images in an observation result, in order; the first is the main view. */
	views: number;
	/** Index (or indices, one per arm) of the wrist view(s) among them (`--vdm-wrist`). */
	wrist?: number | readonly number[];
	/** Tools whose observation moves nothing (view_env_state): recorded as the previous frame, not described. */
	observe?: readonly string[];
};

/** Tools that restore the scene (../robot.ts names them next to the robot's own): the next observation is initial again. */
const RESETS = ["reset", "request_scene_reset"];

type Frame = { main: ImageContent; wrists: ImageContent[] };

/** "wrist camera", numbered when a robot has several. */
const wristLabel = (i: number, n: number) => `wrist camera${n > 1 ? ` ${i + 1}` : ""}`;
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
			...frame.wrists.flatMap((w, i, all) => [text(`Wrist camera${all.length > 1 ? ` ${i + 1}` : ""} view:`), w]),
		],
	};
}

/** trial.py _get_visual_differencing_feedback (the wrist pairs only when both frames have them). */
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
			...(before.wrists.length === after.wrists.length
				? after.wrists.flatMap((w, i, all) => [
						text(`Previous state (${wristLabel(i, all.length)}):`),
						before.wrists[i],
						text(`Current state (${wristLabel(i, all.length)}):`),
						w,
					])
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
 * Register the VDM flags and, with `--vdm`, the tool_result hook. `tools()` names the robot's tools and
 * the scene resets (only their results are observations); `task()` is the task text given to the VDM model.
 */
export function vdm(
	pi: ExtensionAPI,
	spec: VdmSpec | (() => VdmSpec | undefined),
	tools: () => string[],
	task: () => string,
) {
	const current = () => (typeof spec === "function" ? spec() : spec);
	const wristsOf = (s: VdmSpec | undefined) => (s?.wrist === undefined ? [] : [s.wrist].flat());
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
		description: "VDM: also give the model the wrist view(s)",
	});
	pi.registerFlag("vdm-timeout", {
		type: "string",
		default: "60",
		description: "VDM: seconds each call may take (0 = no limit)",
	});
	const on = () => pi.getFlag("vdm") === true;
	let prev: Frame | undefined;
	let calls = 0;
	let errors = 0;
	let cost = 0;
	/** eval-parallel.sh's model-call gate (api-gate.ts), announced on the event bus when loaded. */
	let gate: { dir: string; n: number } | undefined;
	pi.events.on(API_GATE_EVENT, (g) => {
		gate = g as { dir: string; n: number };
	});
	let hooked = false;

	pi.on("session_start", () => {
		prev = undefined;
		calls = errors = cost = 0;
		if (on() && !hooked) {
			hooked = true;
			pi.on("tool_result", describe);
		}
	});

	async function describe(
		event: { toolName: string; isError: boolean; content: (TextContent | ImageContent)[] },
		ctx: ExtensionContext,
	) {
		if (!on() || event.isError || !tools().includes(event.toolName)) return undefined;
		// A restored scene: whatever comes next is a new initial scene, not a change.
		if (RESETS.includes(event.toolName)) prev = undefined;
		const s = current();
		const shown = event.content.filter((c): c is ImageContent => c.type === "image");
		if (!s || shown.length !== s.views) return undefined;
		const wrists = pi.getFlag("vdm-wrist") === true ? wristsOf(s).map((i) => shown[i]) : [];
		const frame: Frame = { main: shown[0], wrists };
		const before = prev;
		prev = frame;
		// Looking moves nothing: the frame is the new previous one, there is no change to describe.
		if (before && s.observe?.includes(event.toolName)) return undefined;
		const kind = before ? "diff" : "initial";
		const modelRef = String(pi.getFlag("vdm-model") || pi.getFlag("units-vlm-model") || "");
		const entry: Record<string, unknown> = { kind, tool: event.toolName, wrist: wrists.length > 0 };
		const seconds = Number(pi.getFlag("vdm-timeout"));
		const timeout = seconds > 0 ? AbortSignal.timeout(seconds * 1000) : undefined;
		const signal = AbortSignal.any([ctx.signal, timeout].filter((x): x is AbortSignal => x !== undefined));
		const started = Date.now();
		let note: string | undefined;
		let slot: string | undefined;
		calls++;
		try {
			if (gate) slot = await acquire(gate.dir, gate.n, 250, signal);
			const reply = await askVlm(
				ctx,
				modelRef,
				pi.getThinkingLevel(),
				before ? diffPrompt(task(), before, frame) : initialPrompt(task(), frame),
				[],
				signal,
			);
			cost += reply.cost;
			pi.events.emit(VLM_COST_EVENT, reply.cost);
			Object.assign(entry, { model: reply.model, text: reply.text, cost_usd: reply.cost });
			note = `${HEADERS[kind]}\n${reply.text.trim()}`;
		} catch (err) {
			if (ctx.signal?.aborted) {
				// The episode ended (an operator verdict, the time limit): not a failure of the call.
				entry.aborted = true;
			} else {
				// The episode goes on without the description.
				errors++;
				entry.error = timeout?.aborted
					? `timed out after ${seconds} s`
					: err instanceof Error
						? err.message
						: String(err);
				note = `[visual differencing unavailable: ${entry.error}]`;
			}
		} finally {
			if (slot) release(slot);
		}
		entry.ms = Date.now() - started;
		pi.appendEntry(VDM_ENTRY, entry);
		return note === undefined ? undefined : { content: [...event.content, text(note)] };
	}

	return {
		/** The robot result's VDM fields (none when --vdm is off). */
		result: () =>
			on()
				? {
						vdm: true,
						vdm_wrist: pi.getFlag("vdm-wrist") === true && wristsOf(current()).length > 0,
						vdm_calls: calls,
						vdm_errors: errors,
						vdm_cost_usd: Number(cost.toFixed(6)),
					}
				: {},
	};
}
