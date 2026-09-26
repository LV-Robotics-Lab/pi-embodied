/**
 * Object memory (--object-memory): the agent's own records of the scene's objects, by name: where it
 * last saw each one (world position, optional orientation), when (env step and time) and a short
 * note. OpenETA keeps the same kind of state as working-memory facts (`save_memory` / `get_memory`,
 * agent/tools/registry.py:2022) beside its remote asset-view bank (agent/tools/object_memory.py);
 * here it is three tools:
 *
 *   remember_object  {name, position, quat_xyzw?, note?}  add or update a record
 *   recall_objects   {names?}                            the records, oldest sighting flagged
 *   forget_object    {name}                              drop a record (the object left the scene)
 *
 * Every change is an `object_record` session entry, so resume and fork rebuild the records from the
 * branch. With --object-memory-dir the records also persist per scene, as
 * `<dir>/<robot>/<scene>.json` (the scene is the robot's task values): the next episode of the same
 * scene starts with them, marked `from_earlier_episode` (the scene was reset since; re-observe before
 * acting on a pose).
 *
 * How it differs from ./memory and ./explore: those are the cross-episode task corpus (recipes,
 * notes; files the agent reads and, when exploring, writes under the memory guard) and the
 * exploration loop that fills it. Object memory is structured world state (what is where, as last
 * observed), kept inside an episode and, opt-in, across episodes of one scene. It needs no memory mount.
 */

import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export const OBJECT_ENTRY = "object_record";
export const OBJECT_TOOLS = ["remember_object", "recall_objects", "forget_object"] as const;

export type ObjectRecord = {
	name: string;
	position: number[];
	quat_xyzw?: number[];
	note?: string;
	/** Env step of the sighting (the robot's status step), when the robot reports one. */
	last_seen_step: number | null;
	/** ISO time of the sighting. */
	last_seen_at: string;
	/** Loaded from --object-memory-dir: recorded in an earlier episode of this scene. */
	from_earlier_episode?: boolean;
};
type Op =
	| { op: "set"; record: ObjectRecord }
	| { op: "delete"; name: string }
	| { op: "load"; records: ObjectRecord[] };

const key = (name: string) => name.trim().toLowerCase().replace(/\s+/g, " ");
const safe = (s: string) => s.replace(/[^\w.-]+/g, "_").slice(0, 120) || "_";
const text = (r: unknown) => ({ content: [{ type: "text" as const, text: JSON.stringify(r) }], details: r });

/** The records a branch's `object_record` entries leave, in insertion order. */
export function replay(entries: readonly { type: string; customType?: string; data?: unknown }[]) {
	const records = new Map<string, ObjectRecord>();
	for (const e of entries) {
		if (e.type !== "custom" || e.customType !== OBJECT_ENTRY) continue;
		const op = e.data as Op;
		if (op.op === "set") records.set(key(op.record.name), op.record);
		else if (op.op === "delete") records.delete(key(op.name));
		else if (op.op === "load") {
			records.clear();
			for (const r of op.records) records.set(key(r.name), r);
		}
	}
	return records;
}

export function objectMemory(
	pi: ExtensionAPI,
	o: { robot: string; scene: () => Record<string, string>; step: () => number | undefined },
) {
	pi.registerFlag("object-memory", {
		type: "boolean",
		default: false,
		description: "Add remember_object / recall_objects / forget_object (per-scene object records)",
	});
	pi.registerFlag("object-memory-dir", {
		type: "string",
		default: "",
		description: "Persist object records per scene under this directory (default: this episode only)",
	});
	const on = () => pi.getFlag("object-memory") === true;
	let records = new Map<string, ObjectRecord>();
	let registered = false;

	/** The scene's file under --object-memory-dir, or undefined when records stay in the session. */
	function file() {
		const dir = String(pi.getFlag("object-memory-dir") ?? "").trim();
		if (!dir) return undefined;
		const scene = Object.entries(o.scene())
			.map(([k, v]) => `${k}-${v}`)
			.join("_");
		return join(dir, safe(o.robot), `${safe(scene || "default")}.json`);
	}
	function save() {
		const f = file();
		if (!f) return;
		mkdirSync(dirname(f), { recursive: true });
		const records_ = [...records.values()].map(({ from_earlier_episode: _, ...r }) => r);
		writeFileSync(
			`${f}.tmp`,
			`${JSON.stringify({ robot: o.robot, scene: o.scene(), records: records_ }, null, 2)}\n`,
		);
		renameSync(`${f}.tmp`, f);
	}
	function change(op: Op) {
		pi.appendEntry(OBJECT_ENTRY, op);
		if (op.op === "set") records.set(key(op.record.name), op.record);
		if (op.op === "delete") records.delete(key(op.name));
		save();
	}

	function register() {
		if (registered) return;
		registered = true;
		const vec = (n: number, d: string) => Type.Array(Type.Number(), { minItems: n, maxItems: n, description: d });
		pi.registerTool({
			name: "remember_object",
			label: "remember_object",
			description:
				"Record where you saw an object now (world frame): add it, or update its record. Use the names you will ask for later; the record keeps the env step and time of this sighting.",
			parameters: Type.Object({
				name: Type.String({ description: "Object name, e.g. 'red mug' (case-insensitive key)" }),
				position: vec(3, "World [x, y, z] in m"),
				quat_xyzw: Type.Optional(vec(4, "Orientation, if known")),
				note: Type.Optional(Type.String({ description: "Short note: state, container, occlusion, ..." })),
			}),
			executionMode: "sequential",
			async execute(_id, p) {
				const name = p.name.trim();
				if (!name) return text({ error: "name must be non-empty" });
				if (!p.position.every(Number.isFinite)) return text({ error: "position must be finite" });
				const record: ObjectRecord = {
					name,
					position: p.position,
					...(p.quat_xyzw ? { quat_xyzw: p.quat_xyzw } : {}),
					...(p.note?.trim() ? { note: p.note.trim().slice(0, 500) } : {}),
					last_seen_step: o.step() ?? null,
					last_seen_at: new Date().toISOString(),
				};
				const updated = records.has(key(name));
				change({ op: "set", record });
				return text({ ok: true, updated, record, count: records.size });
			},
		});
		pi.registerTool({
			name: "recall_objects",
			label: "recall_objects",
			description:
				"Your object records: name, last position (and orientation), the env step and time you last saw it, your note. Records from an earlier episode of this scene are marked; re-observe before acting on any pose.",
			parameters: Type.Object({
				names: Type.Optional(Type.Array(Type.String(), { description: "Only these names (default all)" })),
			}),
			executionMode: "sequential",
			async execute(_id, p) {
				const wanted = p.names?.map(key);
				const all = [...records.values()];
				const found = wanted ? all.filter((r) => wanted.includes(key(r.name))) : all;
				const missing = wanted?.filter((w) => !records.has(w)) ?? [];
				return text({ step: o.step() ?? null, objects: found, ...(missing.length ? { unknown: missing } : {}) });
			},
		});
		pi.registerTool({
			name: "forget_object",
			label: "forget_object",
			description: "Drop an object's record (it left the scene, or the record was wrong).",
			parameters: Type.Object({ name: Type.String() }),
			executionMode: "sequential",
			async execute(_id, p) {
				if (!records.has(key(p.name)))
					return text({ error: `no record named "${p.name}"`, names: [...records.keys()] });
				change({ op: "delete", name: p.name });
				return text({ ok: true, count: records.size });
			},
		});
	}

	pi.on("session_start", (_event, ctx: ExtensionContext) => {
		if (!on()) return;
		const branch = ctx.sessionManager.getBranch();
		records = replay(branch);
		// A fresh episode of a scene with a persisted file starts from that file.
		const f = file();
		if (!branch.some((e) => e.type === "custom" && e.customType === OBJECT_ENTRY) && f) {
			let saved: ObjectRecord[] = [];
			try {
				saved = (JSON.parse(readFileSync(f, "utf8")).records ?? []) as ObjectRecord[];
			} catch {}
			if (saved.length) {
				const op: Op = { op: "load", records: saved.map((r) => ({ ...r, from_earlier_episode: true })) };
				pi.appendEntry(OBJECT_ENTRY, op);
				records = replay([{ type: "custom", customType: OBJECT_ENTRY, data: op }]);
			}
		}
	});

	return {
		/** The tools to activate (none without --object-memory); registered at the first start that has it on. */
		tools(): string[] {
			if (!on()) return [];
			register();
			return [...OBJECT_TOOLS];
		},
		/** The records now. */
		records: () => [...records.values()],
	};
}
