/**
 * The recorded-state layer of the real Franka robots (../franka, ../dual_franka), once. Every
 * mutating tool of theirs runs, then records a numbered state step (robot state, camera frames,
 * camera metadata) under --out and returns it with its images, errors included; a read-only tool
 * returns its result. The robot owns what a step holds and how it is shown (`StepsIO`); this
 * module owns the tool result around it, and the shape of a tool the primitives modules return.
 */

import type { Static, TSchema } from "typebox";
import { Type } from "typebox";
import { type Json, message, round, toolResult } from "../robot.ts";

/** A shared tool as `robot.tool` takes it; the robot mounts it through its own `tool()` wrapper. */
export type ToolDef<P extends TSchema = TSchema> = {
	name: string;
	description: string;
	parameters: P;
	run: (params: Static<P>, signal: AbortSignal | undefined) => Promise<Json>;
};

/** Build a `ToolDef`, typing `run`'s parameters from the schema. */
export const toolDef = <P extends TSchema>(
	name: string,
	description: string,
	parameters: P,
	run: ToolDef<P>["run"],
): ToolDef<P> => ({ name, description, parameters, run });

/** A recorded state step: its states.jsonl blob, artifact directory and camera metadata. */
export type Step = { blob: Json; dir: string; meta: Json | null };

export const stepParam = Type.Optional(Type.Integer({ description: "State step (default -1 = latest)" }));

/** Step `step` of `steps` (default -1 = latest, negative from the end); an unrecorded one throws. */
export function getStep<S extends Step>(steps: S[], step?: number | null): S {
	const i = step === undefined || step === null ? steps.length - 1 : step < 0 ? steps.length + step : step;
	const s = steps[i];
	if (!s) throw new Error(`step ${step} is not recorded (have 0..${steps.length - 1})`);
	return s;
}

/** How a robot records and shows its steps. */
export type StepsIO<S extends Step> = {
	steps: S[];
	/** The robot is up (its env server is connected). */
	ready: () => boolean;
	/** Record a fresh step: the command that led to it, its result and how long the tool took. */
	dump: (command: Json | null, result: Json | null, elapsed: number | null) => Promise<S>;
	/** The step as a tool result: the output text and the images to attach. */
	view: (s: S) => { output: Json; pngs: Buffer[] };
	/** Keys to put ahead of a mutating tool's output (Franka: a jammed gripper), else undefined. */
	headline?: (result: Json) => Json | undefined;
};

/**
 * Run a tool body. A mutating one then records a fresh state step and returns it (errors
 * included); a read-only one returns its result or `{error}`, plus any PNGs in `_pngs`.
 */
export async function outcome<S extends Step>(
	io: StepsIO<S>,
	name: string,
	params: Json,
	run: () => Promise<Json>,
	mutating = true,
) {
	if (!io.ready() || !io.steps.length)
		return toolResult({ error: "robot not initialized; see the session start error" });
	const started = performance.now();
	let result: Json;
	let failed = false;
	try {
		result = await run();
	} catch (err) {
		result = { error: message(err) };
		failed = true;
	}
	if (!mutating) {
		const { _pngs, ...rest } = result;
		return toolResult(rest, _pngs ?? []);
	}
	const elapsed = round((performance.now() - started) / 1000, 2);
	try {
		const { output, pngs } = io.view(await io.dump({ action: name, ...params }, result, elapsed));
		output.agent_elapsed_s = elapsed;
		if (failed) for (const [k, v] of Object.entries(result)) output[k] ??= v;
		const head = io.headline?.(result);
		return toolResult(head ? { ...head, ...output } : output, pngs);
	} catch (err) {
		return toolResult({
			...result,
			state_capture_error: message(err),
			error: result.error ?? `failed to capture state after ${name}: ${message(err)}`,
		});
	}
}
