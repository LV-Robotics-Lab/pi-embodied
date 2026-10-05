/**
 * The facade's kwargs for a manifest tool call. A manifest entry's parameters exist in tool mode,
 * code mode or both (`modes`, ./manifest.ts); the env server's method takes the code-mode ones (its
 * startup self-check pins them to the method's signature: services' components/manifest.py
 * `_signature_mismatch`). A tool-only parameter is pi's, in one of two ways:
 *
 *   - it maps onto code-mode parameters: `point: [row, col]` is the tool's form of the method's
 *     `row` and `col` (align_wrist, ./wrist.ts), which is the one mapping the manifests declare;
 *   - it is pi-side state a caller speaking to the server directly has no access to: LIBERO's `step`
 *     (the recorded state a pixel came from) and `resolution` (which of pi's two image sizes).
 *
 * pi's tool wrappers and the MCP server (../integrations/mcp) both go through here, so the
 * adaptation is written once: `rpcParams` is the parameter set a direct caller may give (the MCP
 * tool's schema), `rpcArguments` the kwargs the method receives.
 */

import type { ManifestEntry, ManifestParam } from "./manifest.ts";

const inMode = (p: ManifestParam, mode: "tool" | "code") => !p.modes || p.modes.includes(mode);
const toolOnly = (p: ManifestParam) => p.modes?.includes("tool") === true && !p.modes.includes("code");

/** `point: [row, col]` as the method's `row` and `col` (the pixel's integer coordinates). */
export function pointArgs(point: readonly unknown[]): { row: number; col: number } {
	if (point.length !== 2 || !point.every((v) => Number.isFinite(Number(v))))
		throw new Error(`point must be [row, col], got ${JSON.stringify(point)}`);
	return { row: Math.round(Number(point[0])), col: Math.round(Number(point[1])) };
}

/** Whether `name`, a tool-only parameter of `e`, maps onto the method's parameters. */
function mapped(e: ManifestEntry, name: string): boolean {
	const params = e.params ?? {};
	return name === "point" && ["row", "col"].every((k) => params[k] !== undefined && inMode(params[k], "code"));
}

/** The tool parameters a direct caller of the method may give: the tool-mode ones less the unmapped tool-only ones. */
export function rpcParams(e: ManifestEntry): Record<string, ManifestParam> {
	return Object.fromEntries(
		Object.entries(e.params ?? {}).filter(([k, p]) => inMode(p, "tool") && (!toolOnly(p) || mapped(e, k))),
	);
}

/**
 * The method's kwargs for validated tool `params` of `e`: mapped tool-only parameters become the
 * method's (`point` → `row`, `col`); an unmapped tool-only parameter that is given is an error (the
 * caller asked for pi-side state the server has no notion of); undefined values are dropped.
 */
export function rpcArguments(e: ManifestEntry, params: Record<string, unknown>): Record<string, unknown> {
	const declared = e.params ?? {};
	const out: Record<string, unknown> = {};
	for (const [k, v] of Object.entries(params)) {
		if (v === undefined) continue;
		const p = declared[k];
		if (p && toolOnly(p)) {
			if (!mapped(e, k))
				throw new Error(`${e.name}: ${k} is pi's (its recorded states and image sizes), not the env server's`);
			Object.assign(out, pointArgs(v as readonly unknown[]));
			continue;
		}
		out[k] = v;
	}
	return out;
}
