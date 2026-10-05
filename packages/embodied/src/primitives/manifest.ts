/**
 * A robot's primitive manifest (./manifests/<robot>.json, shared entries in ./manifests/common/):
 * the one declaration of its tools and code primitives, read here at extension load (tool schemas
 * and descriptions, the code-mode prompt) and by the env server (services' components/manifest.py:
 * `code.api`, the whitelist every program call goes through, its startup self-check). Execution
 * stays hand-written: a robot registers each tool with `robot.primitive(name, run)` (../robot.ts),
 * and the schema and description come from here.
 *
 * Entry: `name`; `side` (`env`: a facade RPC method runs it for the tool and the program alike;
 * `ts`: a pi-side tool; `code`: a code primitive without a tool); `method` (env, code); `tier`
 * (one of high, low, raw, privileged); `mutating`; `requires` (capabilities the run must have: the
 * tool is not activated and the server leaves the code primitive out without them); `params`;
 * `doc.tool` / `doc.code` (the code doc's `Example:` section is dropped in the S4 tier); `result`
 * (the tool's display: motion or read). `{"use": "<file>/<name>", ...}` pulls a shared entry from
 * `common/<file>.json` and overrides the fields given. `{{var}}` in a description is filled from
 * the robot's variables (`vars` in its spec).
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { StringEnum } from "@earendil-works/pi-ai";
import { type TSchema, Type } from "typebox";

export type Side = "env" | "ts" | "code";
export type Tier = "high" | "low" | "raw" | "privileged";
export const TIERS: readonly Tier[] = ["high", "low", "raw", "privileged"];

export type ManifestParam = {
	type: "number" | "integer" | "boolean" | "string" | "enum" | "vec3" | "quat" | "array" | "object";
	required?: boolean;
	description?: string;
	/** An enum's values (strings); `{{var}}` names a robot variable holding the list. */
	values?: string[] | string;
	/** An array's item type (a type name or a parameter spec). */
	items?: ManifestParam["type"] | ManifestParam;
	minItems?: number;
	maxItems?: number;
	minimum?: number;
	maximum?: number;
	/** Where the parameter exists (default both): e.g. ["code"] for a program-only option. */
	modes?: ("tool" | "code")[];
};

export type ManifestEntry = {
	name: string;
	side: Side;
	method?: string;
	tier: Tier;
	mutating?: boolean;
	requires?: string[];
	params?: Record<string, ManifestParam>;
	doc: { tool?: string; code?: string };
	result?: "motion" | "read";
	/** A pi-side module registers this tool with its own schema (units, explore, operator, code, ...). */
	module?: string;
};

/** `digest`: sha256 over the robot's file and the shared files it uses (the server computes the same). */
export type Manifest = { robot: string; internal: string[]; primitives: ManifestEntry[]; digest: string };

/** Robot variables a description or an enum refers to as `{{name}}`. */
export type Vars = Record<string, string | number | readonly string[]>;

/** The manifests' directory: ./manifests, or PI_EMBODIED_MANIFESTS (as the services read it). */
const dir = () => {
	const env = process.env.PI_EMBODIED_MANIFESTS;
	return env ? pathToFileURL(env.endsWith("/") ? env : `${env}/`) : new URL("./manifests/", import.meta.url);
};
const ENTRY_KEYS = new Set([
	"name",
	"side",
	"method",
	"tier",
	"mutating",
	"requires",
	"params",
	"doc",
	"result",
	"module",
]);

function readInto(url: URL, name: string, files: Map<string, Buffer>) {
	const data = readFileSync(url);
	files.set(name, data);
	return JSON.parse(data.toString("utf8"));
}

function digestOf(files: Map<string, Buffer>) {
	const h = createHash("sha256");
	for (const name of [...files.keys()].sort()) {
		h.update(Buffer.from(`${name}\0`));
		h.update(files.get(name) as Buffer);
		h.update(Buffer.from("\0"));
	}
	return h.digest("hex");
}

function common(
	ref: string,
	cache: Map<string, Map<string, ManifestEntry>>,
	files: Map<string, Buffer>,
): ManifestEntry {
	const [file, name] = ref.split("/", 2);
	if (!name) throw new Error(`manifest use "${ref}": expected "<file>/<name>"`);
	if (!cache.has(file))
		cache.set(
			file,
			new Map(
				(
					readInto(new URL(`common/${file}.json`, dir()), `common/${file}.json`, files)
						.primitives as ManifestEntry[]
				).map((e) => [e.name, e]),
			),
		);
	const e = cache.get(file)?.get(name);
	if (!e) throw new Error(`manifest use "${ref}": common/${file}.json has no ${name}`);
	return structuredClone(e);
}

/** Throw when an entry does not follow the schema (the same checks as the Python loader). */
export function checkEntry(e: ManifestEntry, where: string) {
	for (const k of Object.keys(e)) if (!ENTRY_KEYS.has(k)) throw new Error(`${where}: unknown field ${k}`);
	if (!/^[A-Za-z_]\w*$/.test(e.name ?? "")) throw new Error(`${where}: bad name ${e.name}`);
	if (!["env", "ts", "code"].includes(e.side)) throw new Error(`${where}: bad side ${e.side}`);
	if (!TIERS.includes(e.tier)) throw new Error(`${where}: tier must be one of ${TIERS.join(", ")}`);
	if (e.side !== "ts" && typeof e.method !== "string") throw new Error(`${where}: side ${e.side} needs a method`);
	if (e.side === "ts" && e.method !== undefined) throw new Error(`${where}: a ts tool has no method`);
	if (!e.doc || (!e.doc.tool && !e.doc.code)) throw new Error(`${where}: doc needs tool and/or code`);
	if (e.side === "code" && e.doc.tool) throw new Error(`${where}: a code primitive has no tool doc`);
	if (e.side === "ts" && e.doc.code) throw new Error(`${where}: a ts tool has no code doc`);
}

const loaded = new Map<string, Manifest>();

/** The robot's manifest, shared entries resolved and every entry checked (read once per process). */
export function loadManifest(robot: string): Manifest {
	const key = `${dir().href}${robot}`;
	const hit = loaded.get(key);
	if (hit) return hit;
	const files = new Map<string, Buffer>();
	const raw = readInto(new URL(`${robot}.json`, dir()), `${robot}.json`, files) as {
		robot: string;
		internal?: string[];
		primitives: (ManifestEntry | ({ use: string } & Partial<ManifestEntry>))[];
	};
	if (raw.robot !== robot) throw new Error(`${robot}.json names robot ${raw.robot}`);
	const cache = new Map<string, Map<string, ManifestEntry>>();
	const primitives = raw.primitives.map((e, i) => {
		let entry: ManifestEntry;
		if ("use" in e && e.use) {
			const { use, ...over } = e;
			entry = { ...common(use, cache, files), ...over } as ManifestEntry;
		} else entry = e as ManifestEntry;
		checkEntry(entry, `${robot}.json primitives[${i}] (${entry.name})`);
		return entry;
	});
	const m = { robot, internal: raw.internal ?? [], primitives, digest: digestOf(files) };
	loaded.set(key, m);
	return m;
}

/** Fill `{{name}}` from `vars` (a list joins with ", "); an unknown name stays as written. */
export function fill(text: string, vars: Vars = {}) {
	return text.replace(/\{\{(\w+)\}\}/g, (m, k: string) => {
		const v = vars[k];
		if (v === undefined) return m;
		return Array.isArray(v) ? v.join(", ") : String(v);
	});
}

/** An enum's values: the list given, or the robot variable `{{name}}` names (an error without it). */
export function enumValues(values: string[] | string, vars: Vars): string[] {
	if (Array.isArray(values)) return values;
	const m = /^\{\{(\w+)\}\}$/.exec(values);
	const v = m ? vars[m[1]] : undefined;
	if (!Array.isArray(v)) throw new Error(`manifest enum ${values}: no list variable`);
	return [...v];
}

/** An enum whose values come from an empty list variable leaves the parameter out (e.g. `arm` on one arm). */
function dropped(p: ManifestParam, vars: Vars) {
	if (p.type !== "enum" || Array.isArray(p.values)) return false;
	const m = /^\{\{(\w+)\}\}$/.exec(p.values ?? "");
	const v = m ? vars[m[1]] : undefined;
	return Array.isArray(v) && v.length === 0;
}

function paramSchema(p: ManifestParam, vars: Vars): TSchema {
	const d = p.description ? { description: fill(p.description, vars) } : {};
	const range = {
		...(p.minimum !== undefined ? { minimum: p.minimum } : {}),
		...(p.maximum !== undefined ? { maximum: p.maximum } : {}),
	};
	switch (p.type) {
		case "number":
			return Type.Number({ ...d, ...range });
		case "integer":
			return Type.Integer({ ...d, ...range });
		case "boolean":
			return Type.Boolean(d);
		case "string":
			return Type.String(d);
		case "enum":
			return StringEnum(enumValues(p.values ?? [], vars) as [string, ...string[]], d);
		case "vec3":
			return Type.Array(Type.Number(), { minItems: 3, maxItems: 3, ...d });
		case "quat":
			return Type.Array(Type.Number(), { minItems: 4, maxItems: 4, ...d });
		case "array": {
			const items = typeof p.items === "string" ? { type: p.items } : (p.items ?? { type: "number" });
			return Type.Array(paramSchema(items as ManifestParam, vars), {
				...(p.minItems !== undefined ? { minItems: p.minItems } : {}),
				...(p.maxItems !== undefined ? { maxItems: p.maxItems } : {}),
				...d,
			});
		}
		case "object":
			return Type.Object({}, { additionalProperties: true, ...d });
	}
}

/** The tool's parameter schema: the entry's params that exist in tool mode. */
export function toolSchema(e: ManifestEntry, vars: Vars = {}) {
	const props: Record<string, TSchema> = {};
	for (const [k, p] of Object.entries(e.params ?? {})) {
		if ((p.modes && !p.modes.includes("tool")) || dropped(p, vars)) continue;
		const s = paramSchema(p, vars);
		props[k] = p.required ? s : Type.Optional(s);
	}
	return Type.Object(props);
}

export const toolDescription = (e: ManifestEntry, vars: Vars = {}) => fill(e.doc.tool ?? "", vars);

/** Whether the run has every capability the entry requires. */
export const available = (e: ManifestEntry, have: (capability: string) => boolean) => (e.requires ?? []).every(have);

/** The manifest's tools (entries with a tool doc), in declaration order. */
export const tools = (m: Manifest) => m.primitives.filter((e) => e.side !== "code" && e.doc.tool);

/** One tool entry by name; throws when the manifest does not declare it. */
export function toolEntry(m: Manifest, name: string): ManifestEntry {
	const e = m.primitives.find((p) => p.name === name && p.side !== "code" && p.doc.tool);
	if (!e) throw new Error(`${m.robot}'s manifest declares no tool ${name}`);
	return e;
}
