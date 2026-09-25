/**
 * Human-in-the-loop operator (RPent's human_in_the_loop), enabled with --operator.
 *
 * The agent asks through tools, and pi asks the operator with a `ctx.ui.select` dialog, which the
 * TUI shows and an RPC client (e.g. a dashboard) answers as an `extension_ui_request`:
 *   request_operator_verdict  success | failure | continue | abort (+ optional notes)
 *   request_scene_reset       done | abort  (only if the robot passes `reset`)
 *   finish                    without a verdict on the current state, pi asks for one first
 * A dismissed dialog can still be answered with /continue, /done or /operator <id> <answer>.
 * Unsolicited, /success /failure /abort end the episode at any time: in-flight motion stops at
 * its next env step (`check()`), later tool calls are refused, the verdict is recorded, and pi
 * exits. Every exchange is an `operator_event` session entry; `result()` goes into the robot's
 * result entry. Without a UI (print/json mode) requests get no answer, which RPent treats as abort.
 */

import { randomBytes } from "node:crypto";
import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

type Kind = "reset" | "verdict";
type Verdict = "success" | "failure" | "abort";
type Robot = {
	/** Env steps so far; a verdict holds only while this is unchanged. */
	step: () => number;
	/** Reset the robot/scene after the operator confirmed it; enables request_scene_reset. */
	reset?: () => Promise<Record<string, unknown>>;
};

const text = (result: Record<string, unknown>) => ({
	content: [{ type: "text" as const, text: JSON.stringify(result) }],
	details: result,
});

export function operator(pi: ExtensionAPI, robot: Robot) {
	pi.registerFlag("operator", {
		type: "boolean",
		default: false,
		description: "Human-in-the-loop operator: /success /failure /abort /done /continue /operator",
	});
	const on = () => pi.getFlag("operator") === true;

	let pending: { id: string; kind: Kind; resolve: (answer: string | null) => void } | undefined;
	let sealed: Verdict | undefined;
	let verdict: { verdict: string; notes: string; step: number } | undefined;
	let aborted = false;
	let sceneReady = true;
	let attempt = 1;

	const event = (kind: string, fields: Record<string, unknown> = {}) =>
		pi.appendEntry("operator_event", { kind, attempt, step: robot.step(), timestamp: Date.now() / 1000, ...fields });

	function check() {
		if (sealed) throw new Error("operator submitted a terminal verdict; stopping");
	}

	/** Ask the operator with a select dialog; /operator <id>, /done or /continue answer it too. */
	async function ask(ctx: ExtensionContext, prompt: string, kind: Kind, signal?: AbortSignal) {
		check();
		if (!ctx.hasUI) return null;
		if (pending) throw new Error("another operator request is pending");
		const id = randomBytes(6).toString("hex");
		const options = kind === "reset" ? ["done", "abort"] : ["success", "failure", "continue", "abort"];
		const dialog = new AbortController();
		const cancel = () => dialog.abort();
		signal?.addEventListener("abort", cancel, { once: true });
		ctx.ui.setWidget("operator", [prompt, `Or reply: /operator ${id} <answer>; /success, /failure or /abort ends.`]);
		try {
			const answer = await new Promise<string | null>((resolve, reject) => {
				pending = { id, kind, resolve };
				signal?.addEventListener("abort", () => reject(new Error("operator request cancelled")), { once: true });
				void (async () => {
					const choice = await ctx.ui.select(prompt, options, { signal: dialog.signal });
					if (choice === undefined) return; // dismissed: a command or the tool's abort answers
					const judged = choice === "success" || choice === "failure";
					const notes = judged ? await ctx.ui.input("Notes (optional)", "", { signal: dialog.signal }) : "";
					resolve(notes?.trim() ? `${choice} ${notes.trim()}` : choice);
				})().catch(reject);
			});
			check();
			return answer;
		} finally {
			pending = undefined;
			dialog.abort();
			signal?.removeEventListener("abort", cancel);
			ctx.ui.setWidget("operator", undefined);
		}
	}

	/** Ask for a verdict on the current state; returns the tool result of request_operator_verdict. */
	async function judge(ctx: ExtensionContext, question: string, signal?: AbortSignal) {
		verdict = undefined;
		if (aborted || !sceneReady) return { error: "verdict refused; no active confirmed attempt" };
		const step = robot.step();
		const response = await ask(
			ctx,
			`${question}\nAttempt ${attempt}, env step ${step}. Choose success, failure, continue or abort.`,
			"verdict",
			signal,
		);
		const [, word = "unavailable", notes = ""] = (response ?? "").trim().match(/^(\S*)\s*([\s\S]*)$/) ?? [];
		const v = word.toLowerCase() || "unavailable";
		event("verdict", { verdict: v, notes, question });
		if (v === "abort" || response === null) {
			aborted = true;
			sceneReady = false;
		} else if (v === "success" || v === "failure") verdict = { verdict: v, notes, step };
		else if (v !== "continue") return { error: "invalid operator verdict", verdict: v, notes };
		return { ok: true, status: v, operator_notes: notes, attempt, evidence_step: step, operator_aborted: aborted };
	}

	function reply(ctx: ExtensionCommandContext, kind: Kind, answer: string) {
		if (pending?.kind === kind) pending.resolve(answer);
		else ctx.ui.notify(`/${answer} refused: no pending ${kind} request.`, "warning");
	}

	async function seal(ctx: ExtensionCommandContext, v: Verdict) {
		if (!on()) return ctx.ui.notify("Start pi with --operator to use operator commands.", "warning");
		if (sealed || (v === "success" && (aborted || !sceneReady)))
			return ctx.ui.notify(
				`/${v} refused: no eligible attempt, or a verdict is already closing the run.`,
				"warning",
			);
		sealed = v;
		pending?.resolve(null);
		ctx.ui.notify(`/${v} accepted: stopping actions, then recording the result and exiting.`, "info");
		ctx.abort();
		await ctx.waitForIdle();
		if (v === "abort") {
			aborted = true;
			sceneReady = false;
		} else verdict = { verdict: v, notes: `Operator entered /${v} in the interactive terminal.`, step: robot.step() };
		event("verdict", { verdict: v, source: "interactive_command", notes: verdict?.notes ?? "" });
		pi.appendEntry("operator_result", result());
		ctx.shutdown();
	}

	function result() {
		if (!on()) return {};
		return {
			operator_verdict: sealed ?? verdict?.verdict ?? null,
			operator_notes: verdict?.notes ?? "",
			operator_aborted: aborted || sealed === "abort",
			operator_finished: sealed !== undefined,
			verdict_source: sealed ? "interactive_command" : verdict ? "request_operator_verdict" : null,
		};
	}

	pi.registerCommand("done", {
		description: "Confirm the pending scene reset",
		handler: async (_args, ctx) => reply(ctx, "reset", "done"),
	});
	pi.registerCommand("continue", {
		description: "Continue from a pending operator verdict",
		handler: async (_args, ctx) => reply(ctx, "verdict", "continue"),
	});
	pi.registerCommand("operator", {
		description: "Answer an operator request: /operator <request-id> <answer>",
		handler: async (args, ctx) => {
			const m = args.trim().match(/^(\S+)\s+([\s\S]+)$/);
			if (pending && m && m[1] === pending.id) pending.resolve(m[2].trim());
			else ctx.ui.notify("No matching operator request; use /operator <request-id> <answer>.", "warning");
		},
	});
	pi.registerCommand("success", {
		description: "Operator: finish the episode as a success",
		handler: (_args, ctx) => seal(ctx, "success"),
	});
	pi.registerCommand("failure", {
		description: "Operator: finish the episode as a failure",
		handler: (_args, ctx) => seal(ctx, "failure"),
	});
	pi.registerCommand("abort", {
		description: "Operator: abort the episode",
		handler: (_args, ctx) => seal(ctx, "abort"),
	});

	pi.registerTool({
		name: "request_operator_verdict",
		label: "request_operator_verdict",
		description:
			"Human feedback gate. Ask the operator to mark the current task state as success, failure, or continue before you finish or start another attempt.",
		parameters: Type.Object({
			question: Type.Optional(Type.String({ description: "Default: does the current scene satisfy the task?" })),
		}),
		executionMode: "sequential",
		async execute(_id, params, signal, _onUpdate, ctx) {
			return text(
				await judge(ctx, params.question ?? "Does the current scene satisfy the task success criteria?", signal),
			);
		},
	});

	const reset = robot.reset;
	if (reset)
		pi.registerTool({
			name: "request_scene_reset",
			label: "request_scene_reset",
			description:
				"Ask the operator to restore the scene for another attempt (remove/secure held objects), wait for confirmation, then reset the robot.",
			parameters: Type.Object({
				reason: Type.String({ description: "Why the scene needs to be restored" }),
				expected_scene_state: Type.Optional(Type.String({ description: "The restored layout, for the operator" })),
			}),
			executionMode: "sequential",
			async execute(_id, { reason, expected_scene_state = "" }, signal, _onUpdate, ctx) {
				if (aborted) return text({ error: "operator aborted this run; finish without further motion" });
				verdict = undefined;
				sceneReady = false;
				event("reset_requested", { reason, expected_scene_state });
				const response = await ask(
					ctx,
					`Scene reset requested: ${reason}\nExpected scene: ${expected_scene_state}\nRestore the scene; the robot will then reset. Choose done once it is restored, or abort to stop this run.`,
					"reset",
					signal,
				);
				event("reset_response", { response });
				const word = response?.trim().toLowerCase();
				if (word !== "done") {
					aborted = response === null || word === "abort";
					return text({ error: "scene reset not confirmed", operator_aborted: aborted });
				}
				let robot_reset: Record<string, unknown>;
				try {
					robot_reset = await reset();
				} catch (err) {
					event("reset_failed", { error: String(err) });
					return text({ error: "robot reset failed", robot_reset: String(err) });
				}
				attempt++;
				sceneReady = true;
				event("reset_completed");
				return text({
					ok: true,
					robot_reset,
					scene_reset_confirmed: true,
					notice: "Scene restored by operator; robot reset. Re-localize from the new images.",
				});
			},
		});

	pi.on("tool_call", async (event, ctx) => {
		if (!on()) return undefined;
		if (sealed) return { block: true, reason: "operator submitted a terminal verdict; the run is closing" };
		const name = event.toolName;
		if (name === "finish" && !aborted && verdict?.step !== robot.step()) {
			const asked = await judge(
				ctx,
				"The agent wants to finish. Does the current scene satisfy the task success criteria?",
			);
			if (!aborted && verdict?.step !== robot.step())
				return {
					block: true,
					reason: `finish refused; the operator did not judge the current state: ${JSON.stringify(asked)}`,
				};
		}
		if (aborted && name !== "finish")
			return { block: true, reason: "operator aborted this run; finish without further motion" };
		if (!sceneReady && name !== "finish" && name !== "request_scene_reset")
			return { block: true, reason: "refused; request_scene_reset and obtain operator confirmation first" };
		return undefined;
	});

	pi.on("session_start", () => {
		pending = undefined;
		sealed = verdict = undefined;
		aborted = false;
		sceneReady = true;
		attempt = 1;
	});
	pi.on("session_shutdown", () => pending?.resolve(null));

	return {
		/** Tool names to activate (empty without --operator). */
		tools: () => (on() ? ["request_operator_verdict", ...(reset ? ["request_scene_reset"] : [])] : []),
		/** Throws once the operator ended the run; call before every env step. */
		check,
		/** Operator fields for the robot's result entry. */
		result,
	};
}
