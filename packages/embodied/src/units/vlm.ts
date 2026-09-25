/**
 * The units module's side VLM calls: the end-of-episode task verifier and the reference-video
 * demo brief (video_ref). Each is one request to a vision model through pi's model registry
 * (`--units-vlm-model`, default the session's model), outside the agent's context.
 *
 * Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: core/vlm/dual_roles.py verify_task (the final-check prompt and its
 * lenient parse) and plugins/video_ref (uniform frame sampling, the DemoVideoAnalyst prompts, brief
 * validation and the REFERENCE DEMO block), sampling with ffmpeg instead of imageio.
 */

import { execFile } from "node:child_process";
import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import type { ImageContent, TextContent, ThinkingLevel } from "@earendil-works/pi-ai";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";

const run = promisify(execFile);
const oneLine = (v: unknown) => (typeof v === "string" ? v : JSON.stringify(v)).split(/\s+/).join(" ").trim();

/**
 * `pi.events` channel on which a side VLM call's cost (USD, a number) is published; ../robot.ts adds
 * it to the episode's --max-cost budget. Every caller of `askVlm` emits the `cost` it returns.
 */
export const VLM_COST_EVENT = "pi-embodied:vlm-cost";
/** A prompt with its own system prompt and interleaved text and images (the `images` argument follows them). */
export type VlmPrompt = { system: string; content: (TextContent | ImageContent)[] };

/**
 * One user message (the prompt, then the images in order) to the model named `modelRef` ("provider/id",
 * "" = the session's). `cost` is the reply's USD as pi prices it from models.json.
 */
export async function askVlm(
	ctx: ExtensionContext,
	modelRef: string,
	reasoning: string,
	prompt: string | VlmPrompt,
	images: ImageContent[],
	signal?: AbortSignal,
): Promise<{ text: string; model: string; cost: number }> {
	const slash = modelRef.indexOf("/");
	const model = modelRef ? ctx.modelRegistry.find(modelRef.slice(0, slash), modelRef.slice(slash + 1)) : ctx.model;
	if (!model) throw new Error(modelRef ? `unknown model ${modelRef}` : "no model selected");
	const content: (TextContent | ImageContent)[] =
		typeof prompt === "string" ? [{ type: "text", text: prompt }, ...images] : [...prompt.content, ...images];
	const reply = await ctx.modelRegistry
		.streamSimple(
			model,
			{
				...(typeof prompt === "string" ? {} : { systemPrompt: prompt.system }),
				messages: [{ role: "user", content, timestamp: Date.now() }],
			},
			{
				signal,
				// A reasoning model that cannot turn reasoning off gets the lowest level (e.g. under replay/session).
				...(reasoning && reasoning !== "off"
					? { reasoning: reasoning as ThinkingLevel }
					: model.reasoning
						? { reasoning: "low" as ThinkingLevel }
						: {}),
			},
		)
		.result();
	if (reply.stopReason === "error" || reply.stopReason === "aborted")
		throw new Error(reply.errorMessage ?? reply.stopReason);
	const text = reply.content
		.filter((c): c is TextContent => c.type === "text")
		.map((c) => c.text)
		.join("\n");
	return { text, model: `${model.provider}/${model.id}`, cost: reply.usage?.cost?.total ?? 0 };
}

/** The outermost JSON object in a reply (models wrap it in prose or code fences), or undefined. */
export function parseJson(raw: string): unknown {
	const a = raw.indexOf("{");
	const b = raw.lastIndexOf("}");
	if (a < 0 || b <= a) return undefined;
	try {
		return JSON.parse(raw.slice(a, b + 1));
	} catch {
		return undefined;
	}
}

// ---------------------------------------------------------------------------
// the final task check (core/vlm/dual_roles.py verify_task)

/** One strict visual judgment of whole-task completion from the current camera views. */
export function verifyPrompt(task: string, arms: readonly string[], cameras: number) {
	return [
		`TASK: ${task}`,
		"",
		`The ${cameras} image(s) are the robot's current camera views (the third-person view first, then the wrist view${arms.length > 1 ? "s" : ""}).`,
		arms.length > 1 ? "Both robot arms have finished their plans." : "The robot has finished its plan.",
		"Judge STRICTLY from the images whether the task is fully complete: every object at its required destination, nothing dropped beside or outside it.",
		'Return JSON only: {"complete":true|false,"reason":"one visual sentence"}',
	].join("\n");
}

/**
 * The verdict, parsed leniently: a JSON object, else a `"complete": true|false` anywhere in the
 * text, else unavailable, which accepts the completion (a verifier failure must never turn a
 * finished episode into a spurious replan).
 */
export function parseVerdict(raw: string): { complete: boolean; reason: string; available: boolean } {
	const j = parseJson(raw) as { complete?: unknown; reason?: unknown } | undefined;
	if (typeof j?.complete === "boolean")
		return { complete: j.complete, reason: oneLine(j.reason ?? ""), available: true };
	const m = /"complete"\s*:\s*(true|false)/i.exec(raw);
	if (m) return { complete: m[1].toLowerCase() === "true", reason: oneLine(raw).slice(0, 200), available: true };
	return {
		complete: true,
		reason: `verification unavailable, accepting completion (${oneLine(raw).slice(0, 120)})`,
		available: false,
	};
}

// ---------------------------------------------------------------------------
// reference-video replication (plugins/video_ref)

export type DemoOperation = { arm: string; action: string; object: string; grasp: string; destination: string };
export type DemoBrief = { task: string; operations: DemoOperation[] };
/** plugins/video_ref DEFAULT_NUM_FRAMES / DEFAULT_MAX_SIDE. */
export const VIDEO_REF_FRAMES = 8;
const MAX_SIDE = 512;

/** plugins/video_ref/video_ref.txt (dual) and video_ref_single.txt (one arm). */
export function videoRefPrompt(frames: number, arms: readonly string[]) {
	const dual = arms.length > 1;
	const armField = dual ? `"arm":"${[...arms, "both"].join("|")}",` : "";
	return [
		"ROLE: DemoVideoAnalyst",
		dual
			? `You see ${frames} frames sampled IN ORDER from ONE reference video demonstrating a table-top manipulation.`
			: `You see ${frames} frames sampled IN ORDER from ONE reference video demonstrating a table-top manipulation performed by ONE arm (a robot gripper, or a human hand standing in for it).`,
		`A frame may be a composite of labeled camera panels (front view + wrist view${dual ? "s" : ""}); overlay text is auxiliary -- trust what the imagery shows.`,
		...(dual
			? [
					"In the front view the arms' LEFT/RIGHT match the image's left/right (wrist panels, when labeled, confirm which arm holds what).",
				]
			: []),
		"",
		"Report exactly what is demonstrated, operation by operation, in the order performed. Capture the details that make replication faithful:",
		"- object: name it so it cannot be confused with similar ones (color/size/position)",
		"- grasp: the exact part grasped (stem, rim, edge, handle, top face, ...)",
		"- destination: where it ends up, with the placement nuance (which side/half, on/into, orientation)",
		"Report only what the frames show; do not invent steps between or beyond them.",
		"",
		"Return JSON only:",
		`{"task":"one line: what the demo achieves","operations":[{${armField}"action":"short verb phrase","object":"...","grasp":"...","destination":"..."}]}`,
		'Use "-" for a field that does not apply.',
	].join("\n");
}

/** Coerce the analyst's JSON into the brief (plugins/video_ref _validate_brief); throws when unusable. */
export function validateBrief(parsed: unknown, arms: readonly string[]): DemoBrief {
	const p = parsed as { task?: unknown; operations?: unknown } | undefined;
	const task = String(p?.task ?? "").trim();
	if (!task || !Array.isArray(p?.operations) || !p.operations.length)
		throw new Error(`analyst JSON missing task/operations: ${oneLine(parsed ?? "").slice(0, 300)}`);
	const dual = arms.length > 1;
	const field = (v: unknown, fallback: string) => String(v ?? "").trim() || fallback;
	const operations = (p.operations as unknown[])
		.filter((o): o is Record<string, unknown> => typeof o === "object" && o !== null)
		.map((o) => {
			const arm = String(o.arm ?? "")
				.trim()
				.toLowerCase();
			return {
				arm: dual ? ([...arms, "both"].includes(arm) ? arm : "both") : "",
				action: field(o.action, "manipulate"),
				object: field(o.object, "-"),
				grasp: field(o.grasp, "-"),
				destination: field(o.destination, "-"),
			};
		});
	if (!operations.length)
		throw new Error(`analyst JSON has no parseable operations: ${oneLine(parsed).slice(0, 300)}`);
	return { task, operations };
}

function opLine(i: number, op: DemoOperation) {
	let s = `${i}. ${op.arm ? `${op.arm.toUpperCase()} ` : ""}${op.action} ${op.object}`;
	if (op.grasp && op.grasp !== "-") s += ` -- grasp ${op.grasp}`;
	if (op.destination && op.destination !== "-") s += ` -> ${op.destination}`;
	return s;
}

/** The REFERENCE DEMO block for the prompt (plugins/video_ref render_prompt; `plan` is the planner here). */
export function renderBrief(brief: DemoBrief, arms: readonly string[]) {
	const header = [
		"REFERENCE DEMO -- a demonstration video of this task was analyzed. REPLICATE it.",
		`Demo: ${brief.task}`,
		"Operations in demo order:",
		...brief.operations.map((op, i) => opLine(i + 1, op)),
	].join("\n");
	return arms.length > 1
		? `${header}\nReplicate faithfully on the LIVE images: same arm per operation, grasp the same part (use it as the affordance), same destination and placement detail. The numbers are the demo's TIME ORDER across BOTH arms: an operation starts only after every lower-numbered operation is visibly finished -- when the preceding operation belongs to the OTHER arm, give this arm a WAIT stage first, completion = that operation's visible result. This sequencing OVERRIDES the keep-both-arms-busy preference. Object positions may differ from the demo -- plan from where things are NOW; the PLAN rules still govern stage segmentation.`
		: `${header}\nReplicate faithfully on the LIVE images: perform the operations in the demo's order, grasp the same part (use it as the affordance), same destination and placement detail. Object positions may differ from the demo -- plan from where things are NOW; the PLAN rules still govern stage segmentation.`;
}

/** Frames from `path`: decode once to count them, then `n` uniformly spaced (first and last included), each at most 512 px a side, as PNG. */
export async function sampleFrames(
	path: string,
	n: number,
	ffmpeg = "ffmpeg",
): Promise<{ indices: number[]; images: ImageContent[] }> {
	const { stderr } = await run(ffmpeg, ["-nostdin", "-i", path, "-map", "0:v:0", "-f", "null", "-"], {
		maxBuffer: 64 << 20,
	});
	const counts = [...stderr.matchAll(/frame=\s*(\d+)/g)];
	const total = Number(counts.at(-1)?.[1] ?? 0);
	if (!(total > 0)) throw new Error(`no video frames in ${path}`);
	const k = Math.max(2, Math.floor(n));
	const indices = [...new Set(Array.from({ length: k }, (_, i) => Math.round((i * (total - 1)) / (k - 1))))];
	const dir = mkdtempSync(join(tmpdir(), "units-video-ref-"));
	try {
		const select = indices.map((i) => `eq(n\\,${i})`).join("+");
		const scale = `scale=w='min(${MAX_SIDE}\\,iw)':h='min(${MAX_SIDE}\\,ih)':force_original_aspect_ratio=decrease`;
		await run(ffmpeg, [
			"-nostdin",
			"-v",
			"error",
			"-i",
			path,
			"-map",
			"0:v:0",
			"-vf",
			`select='${select}',setpts=N/TB,${scale}`,
			"-r",
			"1",
			"-frames:v",
			String(indices.length),
			join(dir, "%03d.png"),
		]);
		const files = readdirSync(dir)
			.filter((f) => f.endsWith(".png"))
			.sort();
		if (files.length !== indices.length)
			throw new Error(`sampled ${files.length} of ${indices.length} frames from ${path}`);
		const images = files.map((f) => ({
			type: "image" as const,
			data: readFileSync(join(dir, f)).toString("base64"),
			mimeType: "image/png",
		}));
		return { indices, images };
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
}
