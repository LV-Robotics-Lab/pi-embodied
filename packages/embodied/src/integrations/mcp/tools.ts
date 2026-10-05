/**
 * The MCP tool list a robot's primitive manifest (../../primitives/manifest.ts) yields: one tool per
 * env-side entry the tier and `requires` filters admit, its JSON Schema the manifest's `params`
 * (`toolSchema`) and its description `doc.tool`, the same rules pi applies when it registers the
 * robot's tools (../../robot.ts `manifestTool`, `servable`). Nothing here is a second declaration:
 * the manifest stays the one source, and this module only reads it. The schema keeps the parameters
 * the env server's method takes (../../primitives/arguments.ts `rpcParams`): a tool-only parameter
 * that is pi-side state (LIBERO's `step`, `resolution`) is not offered, and `point: [row, col]` is
 * mapped onto the method's `row`, `col` at the call (../session.ts).
 *
 * What is exposed and what is not (the capability boundary, see README "Using the robots from Codex
 * or Claude Code"): `side: env` tools forward to the env server's facade method, which enforces the
 * same limits for every caller (services/PROTOCOL.md). `side: ts` tools (the VLA adapters, the
 * waypoint and advisor tools, the state viewers) are hand-written in pi and stay pi-only, as do the
 * module-owned entries (`module`: units' act/plan, run_code, memory, operator, pointing, web). Code
 * primitives (`side: code`) have no tool.
 *
 * Tier: without `--tier` every non-privileged env tool (pi's tools mode); `--tier high|low|raw` keeps
 * that tier only; `--privileged` adds the privileged variants (a privileged entry of the same name
 * replaces the plain one, as pi's `manifestTool` does) and the privileged-only tools.
 *
 * `requires`: a capability is held when `--capabilities` names it, or when the env server's
 * `code.api` lists a code primitive that requires it (the server derives that list from its own
 * copy of the manifest and what it was started with), or, for `privileged`, under `--privileged`.
 * Without either source a tool with requirements is left out: fail closed, never a tool the
 * server would refuse.
 */

import type { Tool } from "@earendil-works/pi-mcp";
import { rpcParams } from "../../primitives/arguments.ts";
import {
	available,
	type Manifest,
	type ManifestEntry,
	type Tier,
	toolDescription,
	toolSchema,
	type Vars,
} from "../../primitives/manifest.ts";

/** The tiers `--tier` takes (privileged is a flag, not a tier). */
export const MCP_TIERS = ["high", "low", "raw"] as const;
export type McpTier = (typeof MCP_TIERS)[number];

/**
 * The robots whose env servers drive hardware and take the single-machine hardware lock
 * (services/pi_embodied_services/utils/hardware_lock.py users): every motion is high risk on them
 * (../../capabilities/operator.ts `highRisk` with `real`). test/mcp-hook.test.ts keeps this in step
 * with the services tree.
 */
export const REAL_ROBOTS: readonly string[] = ["franka", "dual_franka", "piper", "ur5e"];

/** Whether `robot`'s env server drives hardware (its connect never resets; every motion is high risk). */
export const isReal = (robot: string) => REAL_ROBOTS.includes(robot);

/**
 * The session's built-in tools (./session.ts) that move the robot: `reset` (`env.reset`, a simulator's
 * new episode or a real arm's start-pose motion). The hook (./hook.ts) and Codex's `.mcp.json`
 * (../../../integrations/sync.mjs) gate them as they gate the manifest's `mutating` tools.
 */
export const BUILTIN_MOTIONS: readonly string[] = ["reset"];

export type Selection = { tier?: McpTier; privileged: boolean };

/**
 * The manifest's env tool entries this selection exposes, one per name: the privileged variant under
 * `privileged`, else the plain one; privileged-only entries only under the flag; then the tier filter.
 */
export function selectEntries(m: Manifest, s: Selection): ManifestEntry[] {
	const byName = new Map<string, ManifestEntry[]>();
	for (const e of m.primitives) {
		if (e.side !== "env" || !e.doc.tool || e.module) continue;
		byName.set(e.name, [...(byName.get(e.name) ?? []), e]);
	}
	const out: ManifestEntry[] = [];
	for (const all of byName.values()) {
		const priv = all.find((e) => e.tier === "privileged");
		const plain = all.find((e) => e.tier !== "privileged");
		const chosen = s.privileged ? (priv ?? plain) : plain;
		if (!chosen) continue;
		if (s.tier && chosen.tier !== s.tier && chosen.tier !== "privileged") continue;
		out.push(chosen);
	}
	return out;
}

/**
 * The capability predicate the `requires` filter uses: `explicit` (--capabilities), the code
 * primitives the server's `code.api` reports available (`available` names; undefined when the
 * server serves no registry), and `privileged` under the flag.
 */
export function capabilitiesFrom(
	m: Manifest,
	served: string[] | undefined,
	explicit: readonly string[],
	privileged: boolean,
): (capability: string) => boolean {
	const have = new Set(explicit);
	if (privileged) have.add("privileged");
	if (served) {
		const names = new Set(served);
		for (const e of m.primitives)
			if (e.side !== "ts" && e.doc.code && names.has(e.name)) for (const r of e.requires ?? []) have.add(r);
	}
	return (c) => have.has(c);
}

/** A manifest tool as MCP lists it, with the entry it came from. */
export type McpTool = { tool: Tool; entry: ManifestEntry };

/** Why an entry was left out: an enum needs a robot variable `--var` did not give. */
export type LeftOut = { name: string; reason: string };

/**
 * The MCP tools of `m` under `s`, filtered by `has` (the `requires` rule pi applies), their schemas
 * built with `vars`. An entry whose schema needs a variable `vars` lacks is left out and named in
 * `leftOut` (the server prints them), never exposed with a guessed enum.
 */
export function manifestTools(
	m: Manifest,
	s: Selection,
	has: (capability: string) => boolean,
	vars: Vars = {},
): { tools: McpTool[]; leftOut: LeftOut[] } {
	const tools: McpTool[] = [];
	const leftOut: LeftOut[] = [];
	for (const entry of selectEntries(m, s)) {
		if (!available(entry, has)) continue;
		try {
			// A TypeBox schema is JSON Schema with symbol-keyed kinds; serializing drops the symbols.
			const inputSchema = JSON.parse(
				JSON.stringify(toolSchema({ ...entry, params: rpcParams(entry) }, vars)),
			) as Record<string, unknown>;
			tools.push({
				entry,
				tool: {
					name: entry.name,
					description: toolDescription(entry, vars),
					inputSchema,
					annotations: {
						title: entry.name,
						readOnlyHint: !entry.mutating,
						destructiveHint: Boolean(entry.mutating),
						openWorldHint: false,
					},
				},
			});
		} catch (err) {
			leftOut.push({ name: entry.name, reason: err instanceof Error ? err.message : String(err) });
		}
	}
	return { tools, leftOut };
}

/** Whether a manifest entry moves the robot (the result of a motion is unknown until observed). */
export const moves = (e: ManifestEntry) => Boolean(e.mutating);

/** `--var name=value[,value]`: a list when the value has a comma or the name is a known list variable. */
export function parseVar(spec: string, into: Record<string, string | readonly string[]>) {
	const at = spec.indexOf("=");
	if (at <= 0) throw new Error(`--var ${spec}: expected name=value[,value]`);
	const name = spec.slice(0, at).trim();
	const value = spec.slice(at + 1);
	const list = ["arms", "cameras"].includes(name) || value.includes(",");
	into[name] = list
		? value
				.split(",")
				.map((v) => v.trim())
				.filter(Boolean)
		: value.trim();
}

/** The tier a `--tier` argument names, or an error. */
export function parseTier(v: string | undefined): McpTier | undefined {
	if (!v) return undefined;
	if ((MCP_TIERS as readonly string[]).includes(v)) return v as McpTier;
	throw new Error(`--tier ${v}: one of ${MCP_TIERS.join(", ")} (privileged is --privileged)`);
}

export type { Tier };
