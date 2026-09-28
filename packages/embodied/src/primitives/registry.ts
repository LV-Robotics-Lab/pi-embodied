/**
 * The code-mode API of an episode: the robot's manifest (./manifest.ts) is the only declaration of
 * its primitives; the env server derives `code.api` from the same file and answers only (a) the
 * manifest's digest, which pi compares with its own copy (a server of another version is refused),
 * and (b) the primitives this run has in a tier (their `requires` and the server's state). pi
 * renders the prompt from its manifest, checks that its own view of what is available agrees with
 * the server's, and records the tier digest (manifest digest + tier + available set) in the result.
 * Programs call primitives only through the server's `CodeApi.resolve`, the whitelist built from
 * the manifest, reaching the facade methods (and their limits) the robot's tools reach.
 */

import type { RpcClient } from "../infra/rpc.ts";
import { available, fill, type Manifest, type ManifestEntry, type Tier, type Vars } from "./manifest.ts";

/** `pi.events` channel on which ../robot.ts publishes the episode's `CodeApi` (undefined: none). */
export const CODE_API_EVENT = "pi-embodied:code-api";
export const CODE_API_ENTRY = "code_api";

/** A code tier: the manifest's, and CaP-X's S4 (`low-noexamples`: the low tier without the examples). */
export type CodeApiTier = Tier | "low-noexamples" | `${"low" | "low-noexamples" | "raw"}+privileged`;
/** A tier with the simulator's ground truth added (--privileged on a tier other than high). */
export const PRIVILEGED_SUFFIX = "+privileged";
export const NO_EXAMPLES = "low-noexamples";

/** `code.api`'s reply. */
export type CodeApi = { tier: CodeApiTier | null; manifest_digest: string; available: string[]; digest: string };

export type CodeApiParam = { type: string; description: string; required: boolean };

/** A primitive as the code-mode prompt shows it. */
export type CodeApiPrimitive = {
	name: string;
	method: string;
	doc: string;
	params: Record<string, CodeApiParam>;
	mutating: boolean;
	tier: Tier;
};

/** The server's reply, checked: a malformed one is an error, not an empty API. */
export function parseCodeApi(reply: unknown): CodeApi {
	const r = reply as Partial<CodeApi> | null;
	if (
		!r ||
		typeof r.manifest_digest !== "string" ||
		!Array.isArray(r.available) ||
		!r.available.every((n) => typeof n === "string") ||
		typeof r.digest !== "string"
	)
		throw new Error(`code.api: malformed reply ${JSON.stringify(reply)?.slice(0, 200)}`);
	return { tier: r.tier ?? null, manifest_digest: r.manifest_digest, available: r.available, digest: r.digest };
}

/** `code.api` of `client` for `tier` (default: every non-privileged primitive), or undefined when the server declares none. */
export async function fetchCodeApi(client: RpcClient, tier?: CodeApiTier): Promise<CodeApi | undefined> {
	try {
		return parseCodeApi(await client.call("code.api", tier ? { tier } : {}, 30_000));
	} catch (err) {
		if (err instanceof Error && /unknown RPC method: 'code\.api'/.test(err.message)) return undefined;
		throw err;
	}
}

/** Drop the `Example:` sections of a Google-style docstring (the Python side's strip_examples). */
export function stripExamples(doc: string): string {
	const out: string[] = [];
	let skipping: number | undefined;
	for (const line of doc.split("\n")) {
		const stripped = line.trim();
		const indent = line.length - line.trimStart().length;
		if (skipping !== undefined) {
			if (stripped && indent <= skipping && stripped.endsWith(":")) skipping = undefined;
			else continue;
		}
		if (["example:", "examples:"].includes(stripped.toLowerCase())) {
			skipping = indent;
			continue;
		}
		out.push(line);
	}
	return out.join("\n").trimEnd();
}

/** The manifest's code primitives of `tier` this run has (the same rule as the server's `CodeApi.primitives`). */
export function codePrimitives(
	m: Manifest,
	tier: CodeApiTier | undefined,
	has: (capability: string) => boolean,
	vars: Vars = {},
): CodeApiPrimitive[] {
	const code = m.primitives.filter((e) => e.side !== "ts" && e.doc.code && available(e, has));
	const plus = tier?.endsWith(PRIVILEGED_SUFFIX) ?? false;
	const named = plus ? (tier as string).slice(0, -PRIVILEGED_SUFFIX.length) : tier;
	const noExamples = named === NO_EXAMPLES;
	const base = noExamples ? "low" : named;
	let chosen: ManifestEntry[];
	if (plus) {
		const priv = code.filter((e) => e.tier === "privileged");
		const names = new Set(priv.map((e) => e.name));
		chosen = [...code.filter((e) => e.tier === base && !names.has(e.name)), ...priv];
	} else if (base === undefined) chosen = code.filter((e) => e.tier !== "privileged");
	else if (base === "privileged") {
		const priv = new Set(code.filter((e) => e.tier === "privileged").map((e) => e.name));
		chosen = code.filter((e) => e.tier === "privileged" || (e.tier === "high" && !priv.has(e.name)));
	} else chosen = code.filter((e) => e.tier === base);
	return chosen.map((e) => ({
		name: e.name,
		method: e.method as string,
		doc: fill(noExamples ? stripExamples(e.doc.code as string) : (e.doc.code as string), vars),
		params: Object.fromEntries(
			Object.entries(e.params ?? {})
				.filter(([, p]) => !p.modes || p.modes.includes("code"))
				.map(([k, p]) => [k, { type: p.type, description: p.description ?? "", required: Boolean(p.required) }]),
		),
		mutating: Boolean(e.mutating),
		tier: e.tier,
	}));
}

/** One line per primitive (`name(params) — doc [moves]`), the form a summary lists them in. */
export function renderCodeApi(primitives: CodeApiPrimitive[]): string {
	return primitives
		.map((p) => {
			const params = Object.entries(p.params)
				.map(([k, v]) => `${k}${v.required ? "" : "?"}: ${v.type}`)
				.join(", ");
			const first = p.doc.trim().split("\n")[0];
			return `- ${p.name}(${params}) — ${first}${p.mutating ? " [moves the robot]" : ""}`;
		})
		.join("\n");
}
