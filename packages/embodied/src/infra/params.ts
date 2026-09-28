/**
 * A robot's experiment parameters (PARAMS.md 2.6, 2.7): every flag the robot's extension
 * registers is tracked (`trackFlags`, the first call of each robot's extension), so
 * - a flag registered twice is visible (test/params.test.ts refuses it: one owner per flag);
 * - numbers fail closed: a numeric flag whose value does not parse, or is out of range, stops the
 *   robot at session start (`paramsError`, called by ../robot.ts) instead of turning silently into
 *   a default;
 * - the result records `params` (every experiment flag's effective value) and `params_default`
 *   (its registered default), so a result names its configuration and the eval scripts compare
 *   runs generically (../scripts/params-match.mjs) instead of by hand-kept lists.
 * Where services and files live is deployment config (./config.ts), not flags; the few flags
 * that remain about where things run (`--deployment`, `--env`, ...) are left out of `params`.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

type FlagSpec = { type?: string; default?: unknown; description?: string };
export type Tracked = { names: Map<string, FlagSpec>; duplicates: string[] };

/** Numeric flags: the range a value must fall in (inclusive), and whether it must be an integer. */
export const NUMERIC: Record<string, { min?: number; max?: number; integer?: boolean }> = {
	seed: { min: 0, integer: true },
	"eval-seed": { min: 0, integer: true },
	"max-turns": { min: 0, integer: true },
	"time-limit": { min: 0 },
	"max-cost": { min: 0 },
	"max-tool-calls": { min: 0, integer: true },
	"max-tokens": { min: 0, integer: true },
	"keep-images": { min: 0, integer: true },
	"max-move": { min: 0 },
	"max-rotate": { min: 0 },
	"max-yaw": { min: 0 },
	"unit-tol": { min: 0 },
	"units-coarse-step": { min: 0.001 },
	"units-stage-steps": { min: 0, integer: true },
	"units-point-verify": { min: 0, integer: true },
	"units-video-ref-frames": { min: 1, integer: true },
	"code-timeout": { min: 1 },
	"code-max-calls": { min: 1, integer: true },
	"code-max-move": { min: 0 },
	"approval-timeout": { min: 0 },
	"ensemble-timeout": { min: 0 },
	"explore-sessions": { min: 1, integer: true },
	"explore-attempts-per-session": { min: 0, integer: true },
	"fallback-after": { min: 1, integer: true },
	"fallback-retry-primary": { min: 0, integer: true },
	"fallback-timeout": { min: 0 },
	"ft-phase-steps": { min: 1, integer: true },
	"gumi-operator-confidence": { min: 0, max: 1 },
	"gumi-operator-max-steps": { min: 1, integer: true },
	"hi-res": { min: 0, integer: true },
	"max-api-concurrency": { min: 0, integer: true },
	"vdm-timeout": { min: 0 },
	"vdm-video-frames": { min: 1, integer: true },
	"xpolicy-timeout": { min: 0 },
	"xpolicy-connect-timeout": { min: 0 },
	"serve-timeout": { min: 1 },
	"serve-min-free": { min: 0 },
	"dashboard-port": { min: 0, max: 65535, integer: true },
	"dashboard-live-fps": { min: 0.2, max: 30 },
	"humanclaw-max-steps": { min: 1, integer: true },
	"humanclaw-max-tokens": { min: 1, integer: true },
};

/**
 * Flags that say where things run or land, not how the experiment runs: left out of `params`
 * (the rest of that kind is deployment config, ./config.ts).
 */
export const DEPLOYMENT: ReadonlySet<string> = new Set([
	"deployment",
	"env",
	"robot-env",
	"serve-lock",
	"dashboard",
	"dashboard-port",
	"dashboard-host",
	"dashboard-language",
	"dashboard-live-fps",
]);

const tracked = new WeakMap<object, Tracked>();

/**
 * Track every flag registered on `pi` from now on (call it first in a robot's extension). A second
 * registration of the same name is recorded as a duplicate and ignored, so the first owner's
 * definition stands (pi itself would silently keep the last one).
 */
export function trackFlags(pi: ExtensionAPI): Tracked {
	const had = tracked.get(pi);
	if (had) return had;
	const t: Tracked = { names: new Map(), duplicates: [] };
	tracked.set(pi, t);
	const register = pi.registerFlag.bind(pi);
	(pi as { registerFlag: typeof pi.registerFlag }).registerFlag = ((name: string, spec: FlagSpec) => {
		if (t.names.has(name)) {
			t.duplicates.push(name);
			return;
		}
		t.names.set(name, spec);
		register(name, spec as never);
	}) as typeof pi.registerFlag;
	return t;
}

/** The flags tracked on `pi` (undefined when the robot did not call trackFlags). */
export const trackedFlags = (pi: ExtensionAPI) => tracked.get(pi);

/** Why a numeric flag's value is refused, else undefined. */
export function numberError(name: string, value: unknown): string | undefined {
	const spec = NUMERIC[name];
	if (!spec || value === undefined || value === "" || value === null) return undefined;
	const n = typeof value === "number" ? value : Number(String(value).trim());
	const shown = JSON.stringify(value);
	if (!Number.isFinite(n)) return `--${name} must be a number, got ${shown}`;
	if (spec.integer && !Number.isInteger(n)) return `--${name} must be an integer, got ${shown}`;
	if (spec.min !== undefined && n < spec.min) return `--${name} must be at least ${spec.min}, got ${shown}`;
	if (spec.max !== undefined && n > spec.max) return `--${name} must be at most ${spec.max}, got ${shown}`;
	return undefined;
}

/** The first numeric flag of `pi` with a refused value, else undefined. */
export function paramsError(pi: ExtensionAPI): string | undefined {
	const t = tracked.get(pi);
	const names = t ? [...t.names.keys()] : Object.keys(NUMERIC);
	for (const name of names) {
		const why = numberError(name, pi.getFlag(name));
		if (why) return why;
	}
	return undefined;
}

/** `params` and `params_default` for the result: every tracked experiment flag, effective and registered default. */
export function params(pi: ExtensionAPI): { params: Record<string, unknown>; params_default: Record<string, unknown> } {
	const t = tracked.get(pi);
	const effective: Record<string, unknown> = {};
	const defaults: Record<string, unknown> = {};
	for (const [name, spec] of [...(t?.names ?? new Map<string, FlagSpec>())].sort(([a], [b]) => a.localeCompare(b))) {
		if (DEPLOYMENT.has(name)) continue;
		effective[name] = pi.getFlag(name) ?? null;
		defaults[name] = spec.default ?? null;
	}
	return { params: effective, params_default: defaults };
}
