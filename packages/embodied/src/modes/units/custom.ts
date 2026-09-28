/**
 * A robot's own action vocabulary (UnitsSpec.vocabulary): the pure parts ./index.ts mounts on `act`,
 * the prompt and GUMI. A humanoid declares WALK / TURN / SIT ... with an optional parameter each
 * instead of the arm's MV_* units; the arm plugins are then off.
 *
 *   act {unit: "TURN", param: 45}        one unit with its parameter (n repeats it)
 *   act {unit: "TURN(45)"}               the same, as GUMI's teleop grammar and its recordings write it
 *
 * An enum parameter must be one of its values; a number is clamped to [min, max] and the result says
 * so (HumanCLAW agent/planner.py _clamp_degree and friends clamp the same way); a missing parameter
 * takes its default, or is refused when there is none. A terminal unit (STOP) never repeats.
 */

import { type TSchema, Type } from "typebox";
import type { CustomUnit } from "./types.ts";

/** "TURN(45)" -> ["TURN", "45"]; "WALK" -> ["WALK", undefined]. */
export function splitUnit(unit: string): [string, string | undefined] {
	const m = /^([^()\s]+)\((.*)\)$/.exec(unit.trim());
	return m ? [m[1], m[2].trim()] : [unit.trim(), undefined];
}

/** A unit with its parameter as GUMI records it: "TURN(45)", or the bare name. */
export const unitLabel = (name: string, param: string | number | undefined) =>
	param === undefined ? name : `${name}(${param})`;

/**
 * Check a unit's parameter: the value to run and, when it was clamped or defaulted, what happened.
 * Throws on an unknown unit, an enum value it does not offer, a non-number, or a required parameter
 * that is missing.
 */
export function checkParam(
	units: readonly CustomUnit[],
	name: string,
	raw: unknown,
): { unit: CustomUnit; param: string | number | undefined; note?: string } {
	const unit = units.find((u) => u.name === name);
	if (!unit) throw new Error(`act: ${name} is not a unit of this robot (${units.map((u) => u.name).join(", ")})`);
	const p = unit.param;
	if (!p) {
		if (raw !== undefined && raw !== "") throw new Error(`act: ${name} takes no parameter`);
		return { unit, param: undefined };
	}
	if (raw === undefined || raw === "") {
		if (p.default === undefined) throw new Error(`act: ${name} needs its ${p.name}`);
		return {
			unit,
			param: p.default,
			note: `${name}: ${p.name} defaulted to ${p.default}${p.unit ? ` ${p.unit}` : ""}`,
		};
	}
	if (p.kind === "enum") {
		const v = String(raw).trim();
		const values = p.values ?? [];
		const hit = values.find((x) => x.toLowerCase() === v.toLowerCase());
		if (hit === undefined) throw new Error(`act: ${name}'s ${p.name} must be one of ${values.join(", ")}; got ${v}`);
		return { unit, param: hit };
	}
	const v = typeof raw === "number" ? raw : Number(String(raw).trim());
	if (!Number.isFinite(v)) throw new Error(`act: ${name}'s ${p.name} must be a number; got ${String(raw)}`);
	const lo = p.min ?? -Infinity;
	const hi = p.max ?? Infinity;
	const c = Math.min(hi, Math.max(lo, v));
	return c === v
		? { unit, param: v }
		: { unit, param: c, note: `${name}: ${p.name} ${v} clamped to ${c}${p.unit ? ` ${p.unit}` : ""} (${lo}..${hi})` };
}

/** One unit's parameter as the prompt and the schema describe it. */
export function paramText(u: CustomUnit): string {
	const p = u.param;
	if (!p) return "";
	const range =
		p.kind === "enum"
			? (p.values ?? []).join(" | ")
			: `${p.min ?? "-inf"}..${p.max ?? "inf"}${p.unit ? ` ${p.unit}` : ""}`;
	return `${p.name}: ${range}${p.default !== undefined ? `, default ${p.default}` : ""}`;
}

/** The ACTION UNITS lines of the prompt: one per unit, with its parameter and whether it ends the episode. */
export function customUnitsText(units: readonly CustomUnit[]): string {
	return units
		.map((u) => {
			const p = u.param ? ` (\`param\` ${paramText(u)})` : "";
			return `- ${u.name}${p}: ${u.description}${u.terminal ? " It ends the episode." : ""}`;
		})
		.join("\n");
}

/** `act`'s parameter schema over a custom vocabulary. */
export function customSchema(units: readonly CustomUnit[], names: readonly string[], unitSchema: TSchema) {
	const props: Record<string, TSchema> = { unit: unitSchema };
	if (units.some((u) => u.param))
		props.param = Type.Optional(
			Type.Union([Type.String(), Type.Number()], {
				description: `The unit's parameter: ${units
					.filter((u) => u.param)
					.map((u) => `${u.name} ${paramText(u)}`)
					.join("; ")}`,
			}),
		);
	const repeatable = units.filter((u) => !u.terminal && names.includes(u.name));
	if (repeatable.length)
		props.n = Type.Optional(
			Type.Integer({
				minimum: 1,
				maximum: 10,
				description: "Repeat count (default 1; never for a unit that ends the episode)",
			}),
		);
	return props;
}
