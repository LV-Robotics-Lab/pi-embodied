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
 * `--vdm-video` (CaP-X's video differencing) describes each change from the video of the step instead:
 * the agentview frames the robot recorded (../video.ts publishes each on FRAME_EVENT) between the
 * previous observation and this one, `--vdm-video-frames` of them sampled evenly (first and last
 * included) and sent in order as a strip of images. The initial scene is still described from its
 * observation; a step that recorded no frame (a robot without an episode video, a tool that stepped
 * nothing) falls back to the before/after images. The video carries the main view only (the robot
 * streams no wrist frames); `--vdm-wrist` still adds the wrist view to the initial description.
 * `--vdm-video` alone turns VDM on; its calls are `vdm` entries of kind "video".
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
import { encodePng } from "./png.ts";
import type { NdArray } from "./rpc.ts";
import { askVlm, VLM_COST_EVENT, type VlmPrompt } from "./units/vlm.ts";
import { FRAME_EVENT } from "./video.ts";

/** Session entry per VDM call: `{ kind: "initial" | "diff" | "video", tool, model, text | error | aborted, wrist, ms, cost_usd }`. */
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

/** trial.py _get_video_differencing_feedback: the step's frames, sampled in order, stand in for the video. */
export function videoPrompt(task: string, frames: ImageContent[], recorded: number): VlmPrompt {
	const rule = "Provide objective information and no assumptions. Do *NOT* write any code.";
	return {
		system: `You are a helpful assistant that analyzes robot execution videos. You describe what happened during the robot's code execution, what actions were taken, how the environment changed, and whether the task appears to have been completed. ${rule}`,
		content: [
			text(task),
			text(
				`The following video shows the robot executing code in the environment from the main camera view. Describe what happened during execution, including what actions the robot took, how the objects in the scene changed, and whether the task appears to have been completed. ${rule}`,
			),
			text(
				`Main camera video (${frames.length} frame${frames.length === 1 ? "" : "s"} sampled evenly, in order, from the ${recorded} recorded during this step):`,
			),
			...frames,
		],
	};
}

/** `n` indices spread evenly over `0..length-1`, first and last included (all of them when there are no more than `n`). */
export function sampleIndices(length: number, n: number): number[] {
	if (length <= n) return Array.from({ length }, (_, i) => i);
	if (n <= 1) return [length - 1];
	return Array.from({ length: n }, (_, i) => Math.round((i * (length - 1)) / (n - 1)));
}

/** An HxWx3 uint8 frame as a PNG image block. */
function framePng(f: NdArray): ImageContent {
	const [height, width] = f.shape;
	return { type: "image", data: encodePng(f.data, width, height).toString("base64"), mimeType: "image/png" };
}

/** The text appended to the tool result (launch_utils.py's headers, "code" read as the tool call). */
export const HEADERS = {
	initial: "The initial state of the environment is described as follows:",
	diff: "Included below is the observed differences between the current state of the environment (after the tool call above was executed) and the previous state of the environment (before it was executed):",
	video: "Included below is a description of the robot's execution based on video observation of this turn:",
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
	pi.registerFlag("vdm-video", {
		type: "boolean",
		default: false,
		description: "VDM from the step's video: describe each change from frames sampled between two observations",
	});
	pi.registerFlag("vdm-video-frames", {
		type: "string",
		default: "8",
		description: "--vdm-video: frames sampled per step",
	});
	pi.registerFlag("vdm-timeout", {
		type: "string",
		default: "60",
		description: "VDM: seconds each call may take (0 = no limit)",
	});
	const video = () => pi.getFlag("vdm-video") === true;
	const on = () => pi.getFlag("vdm") === true || video();
	let prev: Frame | undefined;
	/** --vdm-video: the frames recorded since the last observation. */
	let clip: NdArray[] = [];
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
		clip = [];
		calls = errors = cost = 0;
		if (on() && !hooked) {
			hooked = true;
			pi.on("tool_result", describe);
			if (video())
				pi.events.on(FRAME_EVENT, (f) => {
					clip.push(f as NdArray);
				});
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
		// The step's frames: everything recorded since the previous observation.
		const steps = clip;
		clip = [];
		const wrists = pi.getFlag("vdm-wrist") === true ? wristsOf(s).map((i) => shown[i]) : [];
		const frame: Frame = { main: shown[0], wrists };
		const before = prev;
		prev = frame;
		// Looking moves nothing: the frame is the new previous one, there is no change to describe.
		if (before && s.observe?.includes(event.toolName)) return undefined;
		const strip = before && video() && steps.length > 0 ? steps : undefined;
		const kind = !before ? "initial" : strip ? "video" : "diff";
		const modelRef = String(pi.getFlag("vdm-model") || pi.getFlag("units-vlm-model") || "");
		const entry: Record<string, unknown> = { kind, tool: event.toolName, wrist: wrists.length > 0 };
		let prompt: VlmPrompt;
		if (strip) {
			const picked = sampleIndices(
				strip.length,
				Math.max(1, Math.floor(Number(pi.getFlag("vdm-video-frames")) || 8)),
			);
			prompt = videoPrompt(
				task(),
				picked.map((i) => framePng(strip[i])),
				strip.length,
			);
			Object.assign(entry, { wrist: false, frames: picked.length, recorded: strip.length });
		} else prompt = before ? diffPrompt(task(), before, frame) : initialPrompt(task(), frame);
		const seconds = Number(pi.getFlag("vdm-timeout"));
		// A plain (ref'd) timer, not AbortSignal.timeout(): Node unrefs that one, so a call that hangs
		// with no other handle open would end the process instead of timing out.
		const timeout = new AbortController();
		const timer = seconds > 0 ? setTimeout(() => timeout.abort(), seconds * 1000) : undefined;
		const signal = AbortSignal.any([ctx.signal, timeout.signal].filter((x): x is AbortSignal => x !== undefined));
		const started = Date.now();
		let note: string | undefined;
		let slot: string | undefined;
		calls++;
		try {
			if (gate) slot = await acquire(gate.dir, gate.n, 250, signal);
			const reply = await askVlm(ctx, modelRef, pi.getThinkingLevel(), prompt, [], signal);
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
				entry.error = timeout.signal.aborted
					? `timed out after ${seconds} s`
					: err instanceof Error
						? err.message
						: String(err);
				note = `[visual differencing unavailable: ${entry.error}]`;
			}
		} finally {
			clearTimeout(timer);
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
						...(video() ? { vdm_video: true, vdm_video_frames: Number(pi.getFlag("vdm-video-frames")) } : {}),
						vdm_wrist: pi.getFlag("vdm-wrist") === true && wristsOf(current()).length > 0,
						vdm_calls: calls,
						vdm_errors: errors,
						vdm_cost_usd: Number(cost.toFixed(6)),
					}
				: {},
	};
}
