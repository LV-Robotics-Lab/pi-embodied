/**
 * Reference assets (OpenETA's object memory bank, agent/tools/object_memory.py `retrieve_asset_reference`):
 * what a named object looks like and how it behaves, read from the memory corpus. OpenETA fetches a
 * ZIP bundle from a private HTTP service (`OPENETA_OBJECT_MEMORY_BANK_URL`, not public); here the same
 * data lives in files under the memory home, so it travels with the corpus and needs no server:
 *
 *   <memory home>/assets/<namespace>/<asset_id>/manifest.json   {label, aliases[], shape?, size_m?, grasp?, notes?, ...}
 *   <memory home>/assets/<namespace>/<asset_id>/front.png|side.png|top.png   (the three reference views)
 *
 * The namespace is the robot (libero, robocasa, ...; OpenETA's environment namespace), or a query
 * `<namespace>/<asset_id>` picks one explicitly. Resolution follows OpenETA's search rules: an exact
 * key, then an exact alias, then token and fuzzy matches scored 0-1, taken only when the best is at
 * least 0.75 and 0.10 ahead of the next (else the candidates are returned, nothing chosen).
 *
 * Retrieval by image (OpenETA's `semantic` match type, done by its server) is not reproduced: there
 * is no local image index. Name retrieval covers the tool OpenETA exposes to the planner.
 */

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export const MIN_SCORE = 0.75;
export const MIN_MARGIN = 0.1;
const VIEWS = ["front", "side", "top"] as const;

export type Asset = {
	key: string;
	namespace: string;
	asset_id: string;
	label: string;
	aliases: string[];
	dir: string;
	manifest: Record<string, unknown>;
};
export type Candidate = {
	key: string;
	label: string;
	score: number;
	match_type: "exact_key" | "exact_alias" | "token" | "fuzzy";
};

export const assetsHome = () =>
	join(process.env.PI_EMBODIED_MEMORY || join(homedir(), ".pi", "embodied", "memory"), "assets");
/** OpenETA's key component: lowercase, runs of other characters to `_`. */
export const keyOf = (s: string) =>
	s
		.trim()
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "_")
		.replace(/^_+|_+$/g, "");

/** The assets of one namespace directory. */
export function listAssets(root: string, namespace: string): Asset[] {
	const dir = join(root, keyOf(namespace));
	if (!existsSync(dir)) return [];
	const out: Asset[] = [];
	for (const id of readdirSync(dir).sort()) {
		const d = join(dir, id);
		const m = join(d, "manifest.json");
		if (!statSync(d).isDirectory() || !existsSync(m)) continue;
		let manifest: Record<string, unknown>;
		try {
			manifest = JSON.parse(readFileSync(m, "utf8"));
		} catch {
			continue;
		}
		const aliases = Array.isArray(manifest.aliases) ? manifest.aliases.map(String) : [];
		out.push({
			key: `${keyOf(namespace)}/${keyOf(id)}`,
			namespace: keyOf(namespace),
			asset_id: keyOf(id),
			label: String(manifest.label ?? id.replace(/_/g, " ")),
			aliases,
			dir: d,
			manifest,
		});
	}
	return out;
}

const tokens = (s: string) => new Set(keyOf(s).split("_").filter(Boolean));
const trigrams = (s: string) => {
	const t = `  ${keyOf(s).replace(/_/g, " ")} `;
	const g = new Set<string>();
	for (let i = 0; i + 3 <= t.length; i++) g.add(t.slice(i, i + 3));
	return g;
};
const jaccard = (a: Set<string>, b: Set<string>) => {
	let n = 0;
	for (const x of a) if (b.has(x)) n++;
	return a.size + b.size - n ? n / (a.size + b.size - n) : 0;
};

/** Every asset scored against `query` (an asset id, label or alias), best first. */
export function rank(assets: Asset[], query: string): Candidate[] {
	const q = keyOf(query.includes("/") ? query.split("/")[1] : query);
	return assets
		.map((a): Candidate => {
			if (a.asset_id === q) return { key: a.key, label: a.label, score: 1, match_type: "exact_key" };
			const names = [a.label, ...a.aliases];
			if (names.some((n) => keyOf(n) === q))
				return { key: a.key, label: a.label, score: 0.98, match_type: "exact_alias" };
			const tok = Math.max(...[a.asset_id, ...names].map((n) => jaccard(tokens(n), tokens(q))));
			const fuzzy = Math.max(...[a.asset_id, ...names].map((n) => jaccard(trigrams(n), trigrams(q))));
			return tok >= fuzzy
				? { key: a.key, label: a.label, score: Number(tok.toFixed(3)), match_type: "token" }
				: { key: a.key, label: a.label, score: Number(fuzzy.toFixed(3)), match_type: "fuzzy" };
		})
		.filter((c) => c.score > 0)
		.sort((x, y) => y.score - x.score || x.key.localeCompare(y.key));
}

/** OpenETA's selection: an exact key wins; else the best, if confident and unambiguous. */
export function select(ranked: Candidate[]): { chosen?: Candidate; reason?: string } {
	const exact = ranked.filter((c) => c.match_type === "exact_key");
	if (exact.length === 1) return { chosen: exact[0] };
	if (!ranked.length) return { reason: "no_candidates" };
	if (ranked[0].score < MIN_SCORE) return { reason: "low_confidence" };
	if (ranked.length > 1 && ranked[0].score - ranked[1].score < MIN_MARGIN) return { reason: "ambiguous_candidates" };
	return { chosen: ranked[0] };
}

/** Resolve `name` in `namespace` (or `<namespace>/<id>`); the manifest and up to three reference view PNGs. */
export function retrieve(root: string, namespace: string, name: string) {
	const ns = name.includes("/") ? name.split("/")[0] : namespace;
	const assets = listAssets(root, ns);
	const ranked = rank(assets, name);
	const { chosen, reason } = select(ranked);
	if (!chosen) return { found: false as const, namespace: keyOf(ns), reason, candidates: ranked.slice(0, 5) };
	const a = assets.find((x) => x.key === chosen.key) as Asset;
	const files = readdirSync(a.dir).filter((f) => /\.png$/i.test(f));
	const order = (f: string) => {
		const i = VIEWS.findIndex((v) => f.toLowerCase().includes(v));
		return i < 0 ? 9 : i;
	};
	const views = files.sort((x, y) => order(x) - order(y) || x.localeCompare(y)).slice(0, 3);
	return {
		found: true as const,
		key: a.key,
		label: a.label,
		match: chosen,
		manifest: a.manifest,
		views: views.map((f) => f.replace(/\.png$/i, "")),
		pngs: views.map((f) => readFileSync(join(a.dir, f))),
	};
}
