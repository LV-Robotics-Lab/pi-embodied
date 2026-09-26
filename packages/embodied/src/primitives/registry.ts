/**
 * The primitive registry an env server declares (`code.api`, services' components/code_api.py): the
 * primitives a code-as-policy caller may use, the RPC method behind each, their parameters, whether
 * they move the robot, and the API tiers (CaP-X's levels: `high`, `low`, `privileged` = high plus
 * ground truth). A robot opts in with `codeApi` in its defineRobot spec (../robot.ts); at each
 * session start the base fetches the declaration for the episode's tier, records it as a `code_api`
 * session entry, publishes it on `CODE_API_EVENT`, and puts its digest in the result entry, so an
 * episode names the API it ran with. A later `run_code` tool renders this same declaration and calls
 * primitives only through the server's `resolve`, reaching the facade methods (and their limits) the
 * robot's tools reach.
 */

import type { RpcClient } from "../rpc.ts";

/** `pi.events` channel on which ../robot.ts publishes the episode's `CodeApi` (undefined: none). */
export const CODE_API_EVENT = "pi-embodied:code-api";
export const CODE_API_ENTRY = "code_api";

export type CodeApiTier = "high" | "low" | "privileged";

export type CodeApiParam = { type: string; description: string; required: boolean };

export type CodeApiPrimitive = {
	name: string;
	method: string;
	doc: string;
	params: Record<string, CodeApiParam>;
	mutating: boolean;
	tiers: CodeApiTier[];
};

export type CodeApi = { tier: CodeApiTier | null; primitives: CodeApiPrimitive[]; digest: string };

/** The server's reply, checked: a malformed declaration is an error, not an empty API. */
export function parseCodeApi(reply: unknown): CodeApi {
	const r = reply as Partial<CodeApi> | null;
	if (!r || !Array.isArray(r.primitives) || typeof r.digest !== "string")
		throw new Error(`code.api: malformed reply ${JSON.stringify(reply)?.slice(0, 200)}`);
	for (const p of r.primitives)
		if (!p || typeof p.name !== "string" || typeof p.method !== "string" || typeof p.mutating !== "boolean")
			throw new Error(`code.api: malformed primitive ${JSON.stringify(p)?.slice(0, 200)}`);
	return { tier: r.tier ?? null, primitives: r.primitives, digest: r.digest };
}

/** `code.api` of `client` for `tier` (default: high and low), or undefined when the server declares none. */
export async function fetchCodeApi(client: RpcClient, tier?: CodeApiTier): Promise<CodeApi | undefined> {
	try {
		return parseCodeApi(await client.call("code.api", tier ? { tier } : {}, 30_000));
	} catch (err) {
		// A server from before the registry: no declaration, which is not a failure of the robot.
		if (err instanceof Error && /unknown RPC method: 'code\.api'/.test(err.message)) return undefined;
		throw err;
	}
}

/** One line per primitive (`name(params) — doc [moves]`), the form a prompt lists them in. */
export function renderCodeApi(api: CodeApi): string {
	return api.primitives
		.map((p) => {
			const params = Object.entries(p.params)
				.map(([k, v]) => `${k}${v.required ? "" : "?"}: ${v.type}`)
				.join(", ");
			return `- ${p.name}(${params}) — ${p.doc}${p.mutating ? " [moves the robot]" : ""}`;
		})
		.join("\n");
}
