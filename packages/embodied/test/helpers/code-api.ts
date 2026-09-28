/**
 * A fake env server's `code.api` answer for a robot with a manifest: the manifest's digest and the
 * primitives of the tier available under `has` (what the real server derives from the same file).
 */
import { loadManifest } from "../../src/primitives/manifest.ts";
import { type CodeApiTier, codePrimitives } from "../../src/primitives/registry.ts";

export function codeApiReply(robot: string, tier?: string | null, has: (c: string) => boolean = () => false) {
	const m = loadManifest(robot);
	const available = codePrimitives(m, (tier ?? undefined) as CodeApiTier | undefined, has).map((p) => p.name);
	return { tier: tier ?? null, manifest_digest: m.digest, available, digest: `tier:${tier ?? ""}` };
}
