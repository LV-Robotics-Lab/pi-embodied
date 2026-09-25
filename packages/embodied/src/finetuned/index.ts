/**
 * Show-Harness fine-tuned mode for any units robot: a small VLM + LoRA picks one action unit per step.
 *
 *   pi -p -e packages/embodied/src/maniskill -e packages/embodied/src/finetuned --units \
 *     --model finetuned/qwen3_5_2b_showharness_sim --ft-endpoint http://127.0.0.1:8010/v1 \
 *     --env-id PickCube-v1 --seed 0 "Solve the task."
 *
 * The policy is the planner, and in pi the planner is the model: this extension registers the
 * `finetuned/<adapter>` models (the released adapter names, or `finetuned/local` with --ft-model),
 * whose every turn is one step of Show-Harness's MvTokenRunner (core/runners/mvtoken.py,
 * core/sim/mvtoken_*_runner.py): take the latest observation's agentview and wrist images, put them
 * through the training camera transform (./views.ts), render the lite prompt with the task and the
 * last 5 MV_* moves (newest first), send the one stateless request the adapter was trained on
 * (images first, then the prompt; no system message, no history, temperature 0, thinking off, no
 * guided decoding) to an OpenAI-compatible vLLM serving base + LoRA
 * (services/pi_embodied_services/finetuned/serve.sh), parse the bare token, and emit it as a units
 * `act` call, which the robot executes through its own safety checks. The first turn is the runner's
 * opening RELEASE (it yields the first observation); DONE, the env's success flag, or
 * --ft-max-steps end the episode with `finish`. A reply that parses to no unit falls back to MV_DOWN
 * like the runner; a failed request (after retries) is a model error. Usage is zero; an abort ends it.
 *
 * `--ft-prompt v5` runs the aaroncaozj LIBERO adapters (huggingface.co/aaroncaozj/qwen3_5_9b_mvtoken_libero)
 * as their model card measures them: 15 units (the six moves, RT_ROLL/PITCH/YAW turns that the robot
 * runs with --units-rt=true, GRASP, RELEASE, DONE), moves and turns in the recent-units history, the
 * no-progress guard (an MV_* that moved the TCP < 5 mm twice in a row is withheld from the next
 * decision through vLLM `structured_outputs` choice), no stop action (DONE is counted and asked again
 * with DONE withheld; only env success or the step budget ends the episode) and an out-of-vocabulary
 * reply asked again constrained to the vocabulary. Each rule is also a flag (--ft-stuck-guard-mm,
 * --ft-ignore-done, --ft-oov, --ft-max-steps), off for v3/v4.
 *
 * Every step appends a `finetuned_step` entry: the token, the raw reply, the exact prompt text and
 * the pixel fingerprint (sha1 of the uint8 HWC bytes) of each image sent, in wire order.
 *
 * Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: core/runners/mvtoken.py, core/vlm/mvtoken_roles.py and the bare-token
 * shape of core/vlm/vlm_client.py ported as a pi provider; prompts/v3 and prompts/v4 lite templates
 * copied verbatim into ./templates/.
 */

import { existsSync, readFileSync } from "node:fs";
import {
	type Api,
	type AssistantMessage,
	createAssistantMessageEventStream,
	createProvider,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type ToolCall,
	type TranscriptContext,
	type Usage,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { encodePng } from "../png.ts";
import { TASK_ENTRY } from "../robot.ts";
import { RT_UNITS, UNITS_EVENT, type UnitsHandle } from "../units/index.ts";
import { decodePng, fingerprint, formatView, parseView, prepareView, type ViewSpec } from "./views.ts";

type Json = Record<string, unknown>;

/** The single-arm vocabulary the lite prompts offer (mvtoken_roles.MVTOKEN_ACTIONS). */
export const MVTOKEN_ACTIONS = [
	"MV_FWD",
	"MV_BACK",
	"MV_LEFT",
	"MV_RIGHT",
	"MV_UP",
	"MV_DOWN",
	"GRASP",
	"RELEASE",
	"DONE",
] as const;
/**
 * The 15 units of the aaroncaozj LIBERO adapters (huggingface.co/aaroncaozj/qwen3_5_9b_mvtoken_libero,
 * model card): six 2 cm moves, six 10 degree turns about a world axis through the TCP, GRASP,
 * RELEASE, DONE. The tokens a template offers are matched against this superset.
 */
export const V5_ACTIONS = [
	"MV_FWD",
	"MV_BACK",
	"MV_LEFT",
	"MV_RIGHT",
	"MV_UP",
	"MV_DOWN",
	...RT_UNITS,
	"GRASP",
	"RELEASE",
	"DONE",
] as const;
const MOVES = new Set(["MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN"]);
/** The units the "recent moves" history holds: moves, and (v5) turns, which the card says it needs. */
const MOTION = new Set<string>([...MOVES, ...RT_UNITS]);
/** The runner's reply-parse fallback and its opening unit. */
export const FALLBACK = "MV_DOWN";
const OPENING = "RELEASE";
/** core/runners/mvtoken.RECENT_MOVES_MAX (the converter's RECENT_WINDOW). */
export const RECENT_MOVES_MAX = 5;
/** vlm_client._token_call_budget for a vLLM backend: max(8, min(max_tokens=256, 24)). */
export const MAX_TOKENS = 24;
/** Both vendor spellings of "no reasoning block" (vlm_client._NO_THINKING over the config's). */
export const NO_THINKING = { enable_thinking: false, thinking: false };
/** vlm_client retry defaults: 5 retries, 2 s doubling to 30 s, on 408/409/425/429/5xx and network errors. */
const RETRIES = { max: 5, baseS: 2, maxS: 30 };
const RETRYABLE = new Set([408, 409, 425, 429]);

/** The lite prompts (prompts/<version>/…), stripped like the runners read them. */
export const PROMPTS: Record<string, string> = {
	v3: readFileSync(new URL("./templates/v3_mvtoken_generator_lite.txt", import.meta.url), "utf8").trim(),
	"v4-franka": readFileSync(new URL("./templates/v3_mvtoken_generator_lite.txt", import.meta.url), "utf8").trim(),
	"v4-piper": readFileSync(new URL("./templates/v4_piper_mvtoken_lite.txt", import.meta.url), "utf8").trim(),
};
/**
 * The v5 prompt (prompt_v5.txt of the gated HF dataset aaroncaozj/libero_show-harness_tokenized),
 * vendored here once fetched; until then `--ft-prompt v5` needs `--ft-prompt-file`.
 */
export const V5_PROMPT = new URL("./templates/v5_libero_mvtoken.txt", import.meta.url);

/**
 * Inference-time rules per prompt version. v5 is the aaroncaozj LIBERO eval (model card, "How the
 * numbers were measured"): the no-progress guard (--stuck-guard-mm 5), no stop action
 * (--ignore-done), and an out-of-vocabulary reply asked again constrained to the vocabulary (the
 * card leaves that to the serving stack). Its 200-decision budget is read from the card's
 * "200 tokens ~ up to 4800" and is not a published setting.
 */
export type Preset = { maxSteps: number; stuckGuardMm: number; ignoreDone: boolean; oov: "fallback" | "reask" };
const RUNNER: Preset = { maxSteps: 80, stuckGuardMm: 0, ignoreDone: false, oov: "fallback" };
export const PRESETS: Record<string, Preset> = {
	v3: RUNNER,
	"v4-franka": RUNNER,
	"v4-piper": RUNNER,
	v5: { maxSteps: 200, stuckGuardMm: 5, ignoreDone: true, oov: "reask" },
};
/** Units plugins that change what a policy token does (step size, frame, extra gripper moves). */
const GROUNDING_PLUGINS = ["variable_step", "action_chunk", "rotation", "recovery", "auto_release"];

/** The released adapters (showlab/Show-Harness-VLMs), by the name serve.sh registers them under. */
export const RELEASED = [
	"qwen3_5_2b_showharness_sim",
	"qwen3_5_0_8b_showharness_ft",
	"qwen3_5_2b_showharness_ft",
	"qwen3_5_4b_showharness_ft",
	"qwen3_5_9b_showharness_ft",
	"gemma4_e4b_showharness_ft",
] as const;

const SQUARE = 256;
/**
 * Per robot, the transform that brings its (agentview, wrist) images to the training convention
 * (configs/robot_{maniskill,robolab}.yaml): both views letterboxed to 256, the wrist first cropped to
 * 4:3; the wrist with the fingertips at the TOP, MV_FWD toward its bottom and its left the
 * agentview's left. The flips follow each robot's own VIEWS description of its wrist image.
 */
export const ROBOT_VIEWS: Record<string, { agentview: ViewSpec; wrist: ViewSpec }> = {
	// fingers at the bottom, MV_FWD toward the top, left agrees: a vertical flip.
	maniskill: {
		agentview: { rot: 0, flip: "none", square: SQUARE },
		wrist: { rot: 0, flip: "vertical", crop: 1.3333, square: SQUARE },
	},
	// already fingers-top, MV_FWD toward the bottom, left agrees.
	robolab: {
		agentview: { rot: 0, flip: "none", square: SQUARE },
		wrist: { rot: 0, flip: "none", crop: 1.3333, square: SQUARE },
	},
	// turned half around (MV_FWD toward the top, MV_LEFT toward the right): a 180 degree turn.
	libero: {
		agentview: { rot: 0, flip: "none", square: SQUARE },
		wrist: { rot: 0, flip: "both", crop: 1.3333, square: SQUARE },
	},
	// fingertips at the bottom, MV_FWD toward the top, left agrees.
	piper: {
		agentview: { rot: 0, flip: "none", square: SQUARE },
		wrist: { rot: 0, flip: "vertical", crop: 1.3333, square: SQUARE },
	},
};
const DEFAULT_VIEWS = ROBOT_VIEWS.robolab;
/** No transform: the frames already are what the adapter was trained on (`--ft-agentview raw --ft-wrist raw`). */
export const RAW_VIEWS: { agentview: ViewSpec; wrist: ViewSpec } = {
	agentview: parseView("raw"),
	wrist: parseView("raw"),
};
/** ManiSkill env ids of RLinf's real2sim rigs, whose env server renders the Show-Harness training frames. */
export const REAL2SIM_RIGS = new Set(["BlockPAP-v1", "BlockStack-v1"]);
/** The training camera transform for a robot (and its ManiSkill env id); undefined if uncalibrated. */
export const viewsFor = (robot: string, envId = "") =>
	robot === "maniskill" && REAL2SIM_RIGS.has(envId) ? RAW_VIEWS : ROBOT_VIEWS[robot];

// ---------------------------------------------------------------------------
// the request and the reply (core/vlm/mvtoken_roles.py, core/vlm/vlm_client.py)

/** Python str.format with named fields; `{{` / `}}` are literal braces, unknown fields are an error. */
export function formatPrompt(template: string, fields: Record<string, string>): string {
	return template.replace(/\{\{|\}\}|\{(\w*)\}/g, (m, k: string | undefined) => {
		if (m === "{{") return "{";
		if (m === "}}") return "}";
		if (k === undefined || !(k in fields)) throw new Error(`prompt field ${m} has no value`);
		return fields[k];
	});
}

/** mvtoken_roles._actions_from_prompt: the tokens the template offers (matched on the raw template). */
export function allowedTokens(template: string): string[] {
	const present = V5_ACTIONS.filter((t) => new RegExp(`\\b${t}\\b`).test(template));
	if (!present.length) throw new Error("the prompt offers none of the action tokens");
	return present;
}

/** The recent-moves field: newest first, "none" when empty. */
export const recentText = (recent: readonly string[]) => (recent.length ? recent.join(", ") : "none");

/**
 * The chat-completions body of VLMClient.complete_action_token: images first, then the prompt.
 * `choice` constrains the reply to those tokens (vLLM `structured_outputs`; the card: a top-level
 * `guided_choice` is silently ignored).
 */
export function buildRequest(
	model: string,
	prompt: string,
	pngs: readonly Buffer[],
	temperature = 0,
	choice?: readonly string[],
) {
	return {
		model,
		messages: [
			{
				role: "user",
				content: [
					...pngs.map((png) => ({
						type: "image_url",
						image_url: { url: `data:image/png;base64,${png.toString("base64")}` },
					})),
					{ type: "text", text: prompt },
				],
			},
		],
		temperature,
		max_tokens: MAX_TOKENS,
		chat_template_kwargs: { ...NO_THINKING },
		...(choice ? { structured_outputs: { choice: [...choice] } } : {}),
	};
}

const reEscape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const stripReasoning = (s: string) =>
	s
		.replace(/<think>[\s\S]*?<\/think>/gi, "")
		.replace(/<reasoning>[\s\S]*?<\/reasoning>/gi, "")
		.trim();
const tokensIn = (text: string, allowed: readonly string[]) =>
	allowed.filter((t) => new RegExp(`\\b${reEscape(t)}\\b`).test(text));

/** vlm_client.recover_allowed_token: a token from free text, or "". */
export function recoverToken(raw: string, allowed: readonly string[]): string {
	if (!allowed.length) return "";
	const alt = allowed.map(reEscape).join("|");
	const explicit = new RegExp(`"(?:token|answer|direction|status|decision)"\\s*:\\s*"?\\s*(${alt})\\b`).exec(raw);
	if (explicit) return explicit[1];
	for (const pattern of [
		`\\b(?:therefore|thus|so|final(?:ly)?|choose|chosen|select|selected|use|decision|answer)\\b[^.\\n]*?\\b(${alt})\\b`,
		`\\b(${alt})\\b[^.\\n]*?\\b(?:best|correct|required|appropriate)\\b`,
	]) {
		const all = [...raw.matchAll(new RegExp(pattern, "gi"))];
		// Case-insensitive match, canonical token (upstream returns the text as written: "grasp").
		if (all.length) return allowed.find((t) => t.toUpperCase() === all[all.length - 1][1].toUpperCase()) ?? "";
	}
	const sentences = raw
		.split(/[.\n]+/)
		.map((s) => s.trim())
		.filter(Boolean);
	if (sentences.length) {
		const last = tokensIn(sentences[sentences.length - 1], allowed);
		if (last.length === 1) return last[0];
	}
	const unique = [...new Set(tokensIn(raw, allowed))];
	return unique.length === 1 ? unique[0] : "";
}

/** vlm_client._parse_single_token: the bare token, or an error when nothing maps to one. */
export function parseToken(raw: string, allowed: readonly string[]): string {
	const s = stripReasoning(raw);
	if (allowed.includes(s)) return s;
	try {
		const parsed = JSON.parse(s);
		if (parsed && typeof parsed === "object" && !Array.isArray(parsed))
			for (const key of ["token", "answer", "direction", "status", "decision"]) {
				const v = (parsed as Json)[key];
				if (typeof v === "string" && allowed.includes(v.trim())) return v.trim();
			}
	} catch {}
	const alt = allowed.map(reEscape).join("|");
	const suffix = new RegExp(`(?:^|[^A-Z_])(${alt})\\s*[.。]*\\s*$`).exec(s);
	if (suffix) return suffix[1];
	const recovered = recoverToken(s, allowed);
	if (recovered) return recovered;
	throw new Error(`VLM returned invalid token ${JSON.stringify(raw)}; allowed tokens are ${allowed.join(", ")}`);
}

/** The views sent ahead of the prompt, in wire order (mvtoken_roles.CAMERA_ORDER). */
export function prepareImages(
	images: readonly { data: string }[],
	cameras: readonly number[],
	specs: readonly ViewSpec[],
) {
	return cameras.map((index, k) => {
		const img = images[index];
		if (!img) throw new Error(`the observation has ${images.length} images; --ft-cameras wants image ${index}`);
		const view = prepareView(decodePng(Buffer.from(img.data, "base64")), specs[k]);
		return { view, png: encodePng(view.rgb, view.width, view.height) };
	});
}

/** POST /chat/completions with vlm_client._post_chat's retry policy; the reply text and latency. */
async function complete(
	endpoint: string,
	apiKey: string,
	body: unknown,
	signal: AbortSignal | undefined,
): Promise<{ text: string; ms: number }> {
	const url = `${endpoint.replace(/\/+$/, "")}/chat/completions`;
	let last = "";
	let delay = RETRIES.baseS;
	for (let attempt = 0; attempt <= RETRIES.max; attempt++) {
		const t0 = Date.now();
		try {
			const res = await fetch(url, {
				method: "POST",
				headers: { Authorization: `Bearer ${apiKey || "EMPTY"}`, "Content-Type": "application/json" },
				body: JSON.stringify(body),
				signal,
			});
			const raw = await res.text();
			if (res.ok) {
				let data: any;
				try {
					data = JSON.parse(raw);
				} catch {
					data = undefined;
				}
				const message = data?.choices?.[0]?.message;
				if (message && typeof message === "object" && !data.error) {
					const content = typeof message.content === "string" ? message.content : "";
					const text = content.trim() ? content : (message.reasoning_content ?? content ?? "");
					return { text: stripReasoning(String(text)), ms: Date.now() - t0 };
				}
				last = `bad completion body: ${raw.slice(0, 300)}`;
			} else {
				last = `HTTP ${res.status}: ${raw.slice(0, 1000)}`;
				if (!(RETRYABLE.has(res.status) || res.status >= 500))
					throw new Error(`VLM chat completion failed: ${last}`);
			}
		} catch (err) {
			if (signal?.aborted) throw err;
			if (err instanceof Error && err.message.startsWith("VLM chat completion failed")) throw err;
			last = `request error: ${err instanceof Error ? err.message : String(err)}`;
		}
		if (attempt === RETRIES.max) break;
		const wait = Math.min(delay, RETRIES.maxS) * (0.8 + 0.4 * Math.random());
		await new Promise<void>((resolve, reject) => {
			const t = setTimeout(resolve, wait * 1000);
			signal?.addEventListener(
				"abort",
				() => {
					clearTimeout(t);
					reject(new Error("aborted"));
				},
				{ once: true },
			);
		});
		delay = Math.min(delay * 2, RETRIES.maxS);
	}
	throw new Error(`VLM chat completion failed after ${RETRIES.max} retries: ${last} (${url})`);
}

// ---------------------------------------------------------------------------
// the provider

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
const USAGE: Usage = { ...ZERO_COST, totalTokens: 0, cost: { ...ZERO_COST, total: 0 } };
const model = (id: string): Model<"finetuned"> => ({
	id,
	name: `Show-Harness fine-tuned ${id}`,
	api: "finetuned",
	provider: "finetuned",
	baseUrl: "",
	reasoning: false,
	input: ["text", "image"],
	cost: ZERO_COST,
	contextWindow: 100_000_000,
	maxTokens: MAX_TOKENS,
});
/** The session entry each step appends. */
export const STEP_ENTRY = "finetuned_step";
type Turn = { text: string; calls: ToolCall[] };

export default function finetuned(pi: ExtensionAPI) {
	const flag = (name: string) => String(pi.getFlag(name) ?? "");
	pi.registerFlag("ft-endpoint", {
		type: "string",
		default: "http://127.0.0.1:8010/v1",
		description: "OpenAI-compatible endpoint serving the fine-tuned adapter (finetuned/serve.sh)",
	});
	pi.registerFlag("ft-model", {
		type: "string",
		default: "",
		description: "Adapter name to request (default: the model id; required for finetuned/local)",
	});
	pi.registerFlag("ft-api-key", { type: "string", default: "EMPTY", description: "Bearer key for --ft-endpoint" });
	pi.registerFlag("ft-prompt", {
		type: "string",
		default: "v3",
		description: `Prompt version the adapter was trained on (${Object.keys(PRESETS).join(", ")}, or a template file)`,
	});
	pi.registerFlag("ft-prompt-file", {
		type: "string",
		default: "",
		description: "Template text for --ft-prompt's version (e.g. prompt_v5.txt); its rules stay the version's",
	});
	pi.registerFlag("ft-stuck-guard-mm", {
		type: "string",
		default: "auto",
		description:
			"No-progress guard: an MV_* that moved the TCP less than this twice in a row is withheld from the next decision (0 = off; auto: 5 for v5)",
	});
	pi.registerFlag("ft-ignore-done", {
		type: "string",
		default: "auto",
		description: "DONE is counted and asked again with DONE withheld; only success or max steps end (auto: v5)",
	});
	pi.registerFlag("ft-oov", {
		type: "string",
		default: "auto",
		description:
			"A reply that is no unit: fallback (MV_DOWN, the runner) or reask (constrained to the vocabulary; auto: v5)",
	});
	pi.registerFlag("ft-task", { type: "string", default: "", description: "Task text (default: the robot's)" });
	pi.registerFlag("ft-cameras", {
		type: "string",
		default: "0,1",
		description: "Indices of the agentview and wrist images in the robot's observation",
	});
	pi.registerFlag("ft-agentview", {
		type: "string",
		default: "",
		description: "Agentview transform, e.g. rot=0,flip=none,square=256 (default: per robot)",
	});
	pi.registerFlag("ft-wrist", {
		type: "string",
		default: "",
		description: "Wrist transform, e.g. flip=vertical,crop=1.3333,square=256 (default: per robot)",
	});
	pi.registerFlag("ft-swap", {
		type: "string",
		default: "",
		description: "Execution-boundary token swap, e.g. MV_FWD,MV_BACK (the released ft adapters on the AgileX rig)",
	});
	pi.registerFlag("ft-max-steps", {
		type: "string",
		default: "auto",
		description: "Policy decisions per episode (auto: 80, v5 200)",
	});

	let robot = "";
	let envId = "";
	let over = false;
	let steps = 0;
	let recent: string[] = [];
	let pending: string | undefined;
	let ids = 0;
	/** The robot's units layer (its vocabulary and proprioception), published at session start. */
	let units: UnitsHandle | undefined;
	pi.events.on(UNITS_EVENT, (h) => {
		units = h as UnitsHandle;
	});
	/** Stuck guard: the TCP at the previous decision, the policy token executed since, and its stall run. */
	let lastEef: number[] | undefined;
	let lastToken = "";
	let stall = { token: "", count: 0 };
	let dones = 0;

	pi.on("session_start", () => {
		over = false;
		steps = ids = dones = 0;
		recent = [];
		pending = undefined;
		robot = envId = "";
		lastEef = undefined;
		lastToken = "";
		stall = { token: "", count: 0 };
	});
	pi.on("before_agent_start", (_event, ctx) => {
		if (ctx.model?.provider !== "finetuned") return;
		const task = ctx.sessionManager
			.getBranch()
			.filter((e) => e.type === "custom" && e.customType === TASK_ENTRY)
			.pop();
		const data = task?.type === "custom" ? ((task.data as Json) ?? {}) : {};
		robot = String(data.robot || "");
		envId = String(data["env-id"] || "");
		const warn = (s: string) => (ctx.hasUI ? ctx.ui.notify(s, "warning") : console.error(`[finetuned] ${s}`));
		if (!pi.getActiveTools().includes("act")) warn("the `act` tool is not active: run the robot with --units");
		if (!robotViews() && !flag("ft-agentview") && !flag("ft-wrist"))
			warn(
				`no calibrated camera transform for robot "${robot}"; using ${formatView(DEFAULT_VIEWS.wrist)} for the wrist`,
			);
		let offered: string[] = [];
		try {
			offered = allowedTokens(template()).filter((t) => t !== "DONE");
		} catch (err) {
			warn(err instanceof Error ? err.message : String(err));
		}
		const missing = units ? offered.filter((t) => !units?.vocabulary.includes(t)) : [];
		if (missing.length)
			warn(
				`the robot's act does not offer ${missing.join(", ")} (RT_* need --units-rt=true and a robot that can turn); the episode ends if the policy emits one`,
			);
		const grounding = String(pi.getFlag("units-plugins") ?? "")
			.split(",")
			.map((s) => s.trim())
			.filter((s) => GROUNDING_PLUGINS.includes(s));
		if (preset() === PRESETS.v5 && grounding.length)
			warn(
				`units plugins ${grounding.join(", ")} change what a token does; the v5 eval ran none (--units-plugins "")`,
			);
	});

	const version = () => flag("ft-prompt") || "v3";
	const template = () => {
		const p = version();
		if (flag("ft-prompt-file")) return readFileSync(flag("ft-prompt-file"), "utf8").trim();
		if (p === "v5") {
			if (!existsSync(V5_PROMPT))
				throw new Error(
					"--ft-prompt v5: prompt_v5.txt (gated HF dataset aaroncaozj/libero_show-harness_tokenized) is not vendored; pass --ft-prompt-file <path>",
				);
			return readFileSync(V5_PROMPT, "utf8").trim();
		}
		return PROMPTS[p] ?? readFileSync(p, "utf8").trim();
	};
	/** The version's rules (a template file path follows the runner's), with the explicit flags over them. */
	const preset = () => PRESETS[version()] ?? RUNNER;
	const rules = () => {
		const d = preset();
		const pick = (name: string) => (flag(name) === "auto" || !flag(name) ? undefined : flag(name));
		const oov = pick("ft-oov") ?? d.oov;
		if (oov !== "fallback" && oov !== "reask") throw new Error(`--ft-oov must be fallback or reask, got ${oov}`);
		return {
			maxSteps: Number(pick("ft-max-steps") ?? d.maxSteps) || d.maxSteps,
			stuckGuardMm: Number(pick("ft-stuck-guard-mm") ?? d.stuckGuardMm) || 0,
			ignoreDone: (pick("ft-ignore-done") ?? String(d.ignoreDone)) === "true",
			oov,
		};
	};
	/** ManiSkill's real2sim rigs send training-exact frames (their env server applies the rig's transform). */
	const robotViews = () => viewsFor(robot, envId);
	const views = () => {
		const d = robotViews() ?? DEFAULT_VIEWS;
		return [
			flag("ft-agentview") ? parseView(flag("ft-agentview")) : d.agentview,
			flag("ft-wrist") ? parseView(flag("ft-wrist")) : d.wrist,
		];
	};
	const swap = (token: string) => {
		const pair = flag("ft-swap")
			.split(",")
			.map((s) => s.trim())
			.filter(Boolean);
		if (!pair.length) return token;
		if (pair.length !== 2 || !pair.every((t) => MOVES.has(t))) throw new Error(`--ft-swap wants two MV_* units`);
		return token === pair[0] ? pair[1] : token === pair[1] ? pair[0] : token;
	};
	const call = (name: string, args: Json): ToolCall => ({
		type: "toolCall",
		id: `ft_${++ids}`,
		name,
		arguments: args as ToolCall["arguments"],
	});
	const finish = (status: "success" | "failure", summary: string, text: string): Turn => {
		over = true;
		pending = undefined;
		return { text, calls: [call("finish", { status, summary })] };
	};
	const end = (text: string): Turn => {
		over = true;
		pending = undefined;
		return { text, calls: [] };
	};

	/** The task: --ft-task, else the TASK line of the latest units result, else the robot's task flags. */
	function taskOf(texts: string[]) {
		if (flag("ft-task")) return flag("ft-task");
		for (const t of texts) {
			const m = /^TASK: (.*)$/m.exec(t);
			if (m) return m[1].trim();
		}
		throw new Error("no task text in the observation; pass --ft-task");
	}

	/** One policy step over the transcript. */
	async function next(messages: Message[], modelId: string, signal: AbortSignal | undefined): Promise<Turn> {
		if (over) return { text: "The fine-tuned policy already ended this episode.", calls: [] };
		if (!pending) {
			const c = call("act", { unit: OPENING });
			pending = c.id;
			return { text: `opening ${OPENING} (the runner's first step; it returns the first observation)`, calls: [c] };
		}
		const result = messages.find((m) => m.role === "toolResult" && m.toolCallId === pending);
		if (!result || result.role !== "toolResult") return end(`no result for ${pending}; the policy stops.`);
		const texts = result.content.flatMap((c) => (c.type === "text" ? [c.text] : []));
		if (result.isError) return end(`act failed: ${texts.join(" ").slice(0, 300)}; the policy stops.`);
		const details = (result.details ?? {}) as Json;
		if (details.terminated === true || details.success === true)
			return finish("success", `the environment reports success after ${steps} policy steps`, "env success");
		const r = rules();
		const max = r.maxSteps;
		if (steps >= max)
			return finish(
				"failure",
				`max_steps (${max}) reached without ${r.ignoreDone ? "env success" : "DONE"}`,
				`max_steps ${max} reached`,
			);

		// Stuck guard: the displacement of the last executed MV_*, from the robot's proprioception.
		const withheld = new Set<string>();
		let movedMm: number | undefined;
		if (r.stuckGuardMm > 0) {
			const st = units?.state ? await units.state() : undefined;
			const now = Array.isArray(st?.eef_xyz) ? (st.eef_xyz as number[]).map(Number) : undefined;
			if (!now) throw new Error("--ft-stuck-guard-mm needs the robot's eef_xyz (units proprioception)");
			if (lastEef && MOVES.has(lastToken)) {
				movedMm = 1000 * Math.hypot(...now.map((v, i) => v - (lastEef as number[])[i]));
				if (movedMm < r.stuckGuardMm)
					stall =
						stall.token === lastToken
							? { token: lastToken, count: stall.count + 1 }
							: { token: lastToken, count: 1 };
				else stall = { token: "", count: 0 };
			} else stall = { token: "", count: 0 };
			lastEef = now;
			if (stall.count >= 2) withheld.add(stall.token);
		}

		const tpl = template();
		const allowed = allowedTokens(tpl);
		const prompt = formatPrompt(tpl, { task: taskOf(texts), recent_moves: recentText(recent), gripper_state: "" });
		const cameras = flag("ft-cameras")
			.split(",")
			.map((s) => Number(s.trim()));
		const images = result.content.filter((c) => c.type === "image") as { data: string }[];
		const sent = prepareImages(images, cameras, views());
		const adapter = flag("ft-model") || (modelId === "local" ? "" : modelId);
		if (!adapter) throw new Error("finetuned/local needs --ft-model <adapter name>");
		/** One request; `choice` (the vocabulary minus what is withheld) constrains the reply. */
		const ask = async (without: ReadonlySet<string> | undefined) => {
			const choice = without ? allowed.filter((t) => !without.has(t)) : undefined;
			const body = buildRequest(
				adapter,
				prompt,
				sent.map((s) => s.png),
				0,
				choice,
			);
			const reply = await complete(flag("ft-endpoint"), flag("ft-api-key"), body, signal);
			let parsed: string | undefined;
			try {
				parsed = parseToken(reply.text, allowed);
			} catch {}
			// A constrained reply outside its choice means the endpoint ignored structured_outputs.
			if (choice && (parsed === undefined || !choice.includes(parsed)))
				throw new Error(
					`the endpoint ignored structured_outputs: asked for one of ${choice.join(", ")}, got ${JSON.stringify(reply.text)}`,
				);
			return { raw: reply.text, ms: reply.ms, token: parsed, choice };
		};
		const asks = [await ask(withheld.size ? withheld : undefined)];
		let fallback = false;
		// Out of vocabulary: ask again constrained to it (v5), or the runner's MV_DOWN.
		if (asks[0].token === undefined) {
			if (r.oov === "reask") asks.push(await ask(withheld));
			else fallback = true;
		}
		// No stop action: a DONE is counted and asked again with DONE withheld.
		let doneSeen = false;
		if (r.ignoreDone && asks[asks.length - 1].token === "DONE") {
			doneSeen = true;
			dones++;
			asks.push(await ask(new Set([...withheld, "DONE"])));
		}
		const token = asks[asks.length - 1].token ?? FALLBACK;
		const raw = asks[0].raw;
		const ms = asks.reduce((s, a) => s + a.ms, 0);
		const step = steps++;
		const executed = token === "DONE" ? token : swap(token);
		pi.appendEntry(STEP_ENTRY, {
			step,
			token,
			executed,
			raw,
			fallback,
			latency_ms: ms,
			adapter,
			endpoint: flag("ft-endpoint"),
			prompt,
			media: sent.map((s, slot) => ({
				slot,
				camera: ["agentview", "wrist"][slot] ?? `view${slot}`,
				...fingerprint(s.view),
			})),
			...(asks.length > 1 ? { asks: asks.map((a) => ({ raw: a.raw, choice: a.choice ?? null })) } : {}),
			...(withheld.size ? { withheld: [...withheld] } : {}),
			...(movedMm !== undefined ? { moved_mm: Number(movedMm.toFixed(2)) } : {}),
			...(doneSeen ? { done_ignored: dones } : {}),
		});
		const extra = [
			withheld.size ? `withheld ${[...withheld].join(", ")} (no progress)` : "",
			asks.length > 1 && !doneSeen ? `out-of-vocabulary ${JSON.stringify(raw)} asked again` : "",
			doneSeen ? `DONE #${dones} ignored` : "",
		].filter(Boolean);
		const note = `step ${step}: ${token}${executed !== token ? ` (executed as ${executed})` : ""}${fallback ? ` (unparsable reply ${JSON.stringify(raw)}; fallback)` : ""}${extra.length ? ` (${extra.join("; ")})` : ""} · ${ms} ms`;
		if (token === "DONE") return finish("success", `the policy emitted DONE after ${step} steps`, note);
		// A unit the robot's act does not offer (an RT_* on a robot that cannot turn) ends the episode.
		if (units && !units.vocabulary.includes(executed))
			return finish(
				"failure",
				`the policy emitted ${executed}, which this robot's act does not offer (RT_* need --units-rt=true and a robot that can turn about that axis)`,
				note,
			);
		if (MOTION.has(token)) recent = [token, ...recent].slice(0, RECENT_MOVES_MAX);
		lastToken = token;
		const c = call("act", { unit: executed });
		pending = c.id;
		return { text: note, calls: [c] };
	}

	function streamSimple(m: Model<Api>, context: TranscriptContext, options?: SimpleStreamOptions) {
		const stream = createAssistantMessageEventStream();
		const message: AssistantMessage = {
			role: "assistant",
			content: [],
			api: m.api,
			provider: m.provider,
			model: m.id,
			usage: { ...USAGE, cost: { ...USAGE.cost } },
			stopReason: "stop",
			timestamp: Date.now(),
		};
		const fail = (err: unknown) => {
			const aborted = options?.signal?.aborted === true;
			if (aborted) over = true;
			message.stopReason = aborted ? "aborted" : "error";
			message.errorMessage = aborted
				? "Fine-tuned policy aborted"
				: err instanceof Error
					? err.message
					: String(err);
			stream.push({ type: "error", reason: message.stopReason, error: message });
			stream.end();
		};
		if (options?.signal?.aborted) {
			fail(new Error("aborted"));
			return stream;
		}
		next(context.messages as Message[], m.id, options?.signal).then((turn) => {
			stream.push({ type: "start", partial: message });
			if (turn.text) {
				message.content.push({ type: "text", text: turn.text });
				stream.push({ type: "text_start", contentIndex: 0, partial: message });
				stream.push({ type: "text_delta", contentIndex: 0, delta: turn.text, partial: message });
				stream.push({ type: "text_end", contentIndex: 0, content: turn.text, partial: message });
			}
			for (const c of turn.calls) {
				const contentIndex = message.content.push(c) - 1;
				stream.push({ type: "toolcall_start", contentIndex, partial: message });
				stream.push({ type: "toolcall_end", contentIndex, toolCall: c, partial: message });
			}
			message.stopReason = turn.calls.length ? "toolUse" : "stop";
			stream.push({ type: "done", reason: message.stopReason, message });
			stream.end();
		}, fail);
		return stream;
	}

	pi.registerProvider(
		createProvider({
			id: "finetuned",
			name: "Show-Harness fine-tuned",
			auth: { apiKey: { name: "Fine-tuned", resolve: async () => ({ auth: {} }) } },
			models: [...RELEASED, "local"].map(model),
			api: { stream: streamSimple, streamSimple },
		}),
	);
}
