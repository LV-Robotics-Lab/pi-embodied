/**
 * The units verifier (`--units-verify`): a `finish` claiming success first lifts every open gripper
 * (the retreat), then is checked once against the task on the latest camera images (./vlm.ts). See
 * ./index.ts for the refusal budgets; every check is a `units_verify` session entry.
 *
 * Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: see ./index.ts.
 */

import type { ImageContent } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type Result, VERIFY_ENTRY } from "./types.ts";
import { askVlm, parseVerdict, VLM_COST_EVENT, verifyPrompt } from "./vlm.ts";
import { MAX_REPEAT } from "./vocabulary.ts";

/**
 * Verifier: the retreat before the check lifts each open gripper this far with MV_UP (Show-Harness
 * judges a "fresh, retreated observation": RETREAT stages lift after RELEASE and finished arms go home,
 * so the gripper no longer occludes the drop point). A closed gripper is left in place.
 */
const RETREAT_M = 0.1;
/** Verifier: success claims refused for a still-closed gripper per episode (then the check runs anyway). */
const MAX_HOLD_REFUSALS = 2;
/** Verifier: NOT complete verdicts that refuse `finish` per episode (v0.max_replans). */
const MAX_REPLANS = 1;
/** Verifier: success claims refused because the check could not run, per episode (then the finish ends unverified). */
const MAX_VERIFIER_ERRORS = 2;

/** The verifier's part of the episode state; ./index.ts owns it (reset, `units_state`), this module updates it in place. */
export type VerifierState = {
	/** Refusals so far and the latest refusal's reason (shown until a new plan). */
	replans: number;
	verdict: string;
	/** Success claims refused for holding. */
	holdRefusals: number;
	/** Success claims refused because the verifier's VLM call failed, and its latest error. */
	verifierErrors: number;
	verifierError: string;
	/** Whether the latest success finish the verifier let through had a verdict (undefined: none checked). */
	finishVerified: boolean | undefined;
	/** The latest camera images a robot tool returned (the verifier's view). */
	images: ImageContent[];
};

export const verifierState = (): VerifierState => ({
	replans: 0,
	verdict: "",
	holdRefusals: 0,
	verifierErrors: 0,
	verifierError: "",
	finishVerified: undefined,
	images: [],
});

export type VerifierDeps = {
	pi: ExtensionAPI;
	/** Updated in place. */
	check: VerifierState;
	arms: readonly string[];
	/** Units mode is on (--units). */
	active: () => boolean;
	/** --units-verify resolved for this robot. */
	verifying: () => boolean;
	/** The task text. */
	instruction: () => string;
	/** Whether the arm's gripper was commanded closed ("" = the single arm). */
	isClosed: (arm: string) => boolean | undefined;
	/** The plan has a PLACE, RELEASE or RETREAT stage (the task is a placement). */
	placement: () => boolean;
	/** The MV_UP step of the retreat lift, m. */
	liftStep: () => number;
	/** `act` with its state entry. */
	act: (
		params: { unit: string; n?: number; arm?: string; retreat?: boolean },
		signal: AbortSignal | undefined,
	) => Promise<Result>;
	/** A NOT complete verdict: drop the plan, the move history and the recovery note. */
	replan: () => void;
	/** The plan plugin is on (the refusal asks for new stages). */
	planning: () => boolean;
	/** Append the `units_state` entry when it changed. */
	save: () => void;
};

/** Register the verifier's handlers: the latest camera images (`tool_result`) and the finish check (`tool_call`). */
export function registerVerifier(d: VerifierDeps) {
	const { pi, check, arms: armNames } = d;

	// The verifier judges the latest camera images: the newest robot tool result that carries any
	// (`point`'s marked image is not a camera view).
	pi.on("tool_result", (event) => {
		if (!d.active() || event.toolName === "point" || !Array.isArray(event.content)) return undefined;
		const shown = event.content.filter((c): c is ImageContent => c.type === "image");
		if (shown.length) check.images = shown;
		return undefined;
	});

	/**
	 * Lift every open gripper RETREAT_M with MV_UP through `act` (the robot's apply path and limits),
	 * one arm after the other; the last result's images become the verifier's view. A refused or
	 * blocked lift is recorded and the check runs on the images at hand.
	 */
	async function retreat(signal: AbortSignal | undefined) {
		const log: Record<string, unknown>[] = [];
		for (const arm of armNames.length ? armNames : [undefined]) {
			const rec: Record<string, unknown> = arm ? { arm } : {};
			log.push(rec);
			if (d.isClosed(arm ?? "")) {
				rec.skipped = "gripper closed: a held object stays where it is";
				continue;
			}
			const step = d.liftStep();
			const n = Math.min(MAX_REPEAT, Math.ceil(RETREAT_M / step - 1e-9));
			try {
				const r = await d.act({ unit: "MV_UP", n, retreat: true, ...(arm ? { arm } : {}) }, signal);
				const shown = r.content.filter((c): c is ImageContent => c.type === "image");
				const said = r.content
					.filter((c) => c.type === "text")
					.map((c) => (c as { text: string }).text)
					.join("\n");
				// Without new images the robot did not move (it refused the lift).
				if (shown.length) {
					check.images = shown;
					rec.ran = said.split("\n")[0];
				} else rec.refused = said.split("\n").at(-1);
			} catch (err) {
				rec.refused = err instanceof Error ? err.message : String(err);
			}
		}
		return log;
	}

	// The final task check (core/runners/dual.py _completion_outcome): a success claim is judged once
	// more from the images; NOT complete refuses `finish` (with the reason) while the replan budget lasts.
	pi.on("tool_call", async (event, ctx) => {
		if (event.toolName !== "finish" || !d.active() || !d.verifying()) return undefined;
		const status = (event.input as { status?: unknown }).status;
		if (status !== undefined && status !== "success") return undefined;
		// Stage discipline (Show-Harness subgoal): a placement is done only after PLACE -> RELEASE ->
		// RETREAT, so a success claim while still holding is refused (not the verifier's replan).
		const holding = (armNames.length ? armNames : [""]).filter((a) => d.isClosed(a));
		const placement = d.placement();
		if (placement && holding.length && check.holdRefusals < MAX_HOLD_REFUSALS) {
			check.holdRefusals++;
			pi.appendEntry(VERIFY_ENTRY, {
				status,
				replans: check.replans,
				holding: holding.map((a) => a || "arm"),
				refused: "holding",
			});
			d.save();
			const which = armNames.length ? ` (${holding.join(", ")} arm)` : "";
			return {
				block: true,
				reason: `finish refused: the gripper${which} is still closed on the object. A placement is complete only after PLACE -> RELEASE -> RETREAT: \`act\` RELEASE, then MV_UP to lift clear, then call \`finish\` again. (This is not the verifier's check.)`,
			};
		}
		const entry: Record<string, unknown> = { status, replans: check.replans };
		entry.retreat = await retreat(ctx.signal);
		entry.cameras = check.images.length;
		let refuse = false;
		let failed = false;
		if (!check.images.length) entry.skipped = "no camera images yet";
		else {
			const started = Date.now();
			const ask = () =>
				askVlm(
					ctx,
					String(pi.getFlag("units-vlm-model") ?? ""),
					pi.getThinkingLevel(),
					verifyPrompt(d.instruction(), armNames, check.images.length),
					check.images,
					ctx.signal,
				).then((r) => {
					// The call's cost counts toward the robot's --max-cost budget.
					pi.events.emit(VLM_COST_EVENT, r.cost);
					return r;
				});
			try {
				// A failed call (no credits, network) is asked once more; an aborted run is not.
				const reply = await ask().catch((err) => {
					if (ctx.signal?.aborted) throw err;
					entry.retried = err instanceof Error ? err.message : String(err);
					return ask();
				});
				const v = parseVerdict(reply.text);
				Object.assign(entry, { model: reply.model, complete: v.complete, reason: v.reason, raw: reply.text });
				if (!v.available) entry.unavailable = true;
				refuse = !v.complete && check.replans < MAX_REPLANS;
			} catch (err) {
				if (ctx.signal?.aborted) throw err;
				// Fail closed: an unchecked finish is refused (not the replan) until the error budget is spent.
				check.verifierError = err instanceof Error ? err.message : String(err);
				entry.verifier_error = check.verifierError;
				failed = check.verifierErrors < MAX_VERIFIER_ERRORS;
				entry.unverified = true;
			}
			entry.ms = Date.now() - started;
		}
		entry.refused = refuse || failed;
		pi.appendEntry(VERIFY_ENTRY, entry);
		if (failed) {
			check.verifierErrors++;
			d.save();
			return {
				block: true,
				reason: `finish refused: the verifier is unavailable (${check.verifierError}), so the task could not be checked. Call \`finish\` again to retry the check. (This is not the verifier's NOT complete verdict.)`,
			};
		}
		check.finishVerified = entry.model !== undefined && entry.unavailable !== true;
		d.save();
		if (!refuse) return undefined;
		check.replans++;
		check.verdict = String(entry.reason ?? "");
		// Replan from the live scene: the old plan and move history no longer apply.
		d.replan();
		d.save();
		return {
			block: true,
			reason: `finish refused: the verifier judged the task NOT complete (${check.verdict}). Replan the remaining work from the live images${d.planning() ? " (send new stages with `plan`)" : ""} and continue with \`act\`; call \`finish\` again once the task is visibly complete. This is the only refusal.`,
		};
	});
}
