/**
 * The RoboCasa365 task table (services/.../robocasa/eval/robocasa365.json, generated from the
 * installed robocasa package by `python -m pi_embodied_services.robots.robocasa.tasks --write`):
 * 317 tasks in two splits (634 env ids) with a manifest of 50 scene seeds each. `resolveCell`
 * checks the --task-name / --split / --scene flags against it before the env server starts,
 * so a mistyped task fails with its near matches instead of a robosuite traceback.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";

export type Task = {
	name: string;
	kind: "atomic" | "composite";
	target50: boolean;
	horizon: number;
	instruction: string;
	manifest: number[];
};
export type Table = { schema: string; splits: string[]; scenes_per_task: number; tasks: Task[] };
/** The table, relative to the services dir. */
export const TABLE = "pi_embodied_services/robots/robocasa/eval/robocasa365.json";
/** Splits without a manifest: `all` samples every kitchen (seed mode only). */
export const SEED_SPLITS = ["target", "pretrain", "all"];

export function loadTable(services: string): Table {
	const table = JSON.parse(readFileSync(join(services, TABLE), "utf8")) as Table;
	if (table.schema !== "robocasa365-tasks/1") throw new Error(`${TABLE}: not a robocasa365-tasks/1 table`);
	return table;
}

export const envId = (task: string, split: string) => `robocasa365/${split}/${task}`;

/** Levenshtein distance. */
function distance(a: string, b: string): number {
	let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
	for (let i = 1; i <= a.length; i++) {
		const cur = [i];
		for (let j = 1; j <= b.length; j++)
			cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
		prev = cur;
	}
	return prev[b.length];
}

/** Up to `n` names close to `name`: case-insensitive substring matches first, then by edit distance. */
export function nearMatches(name: string, names: readonly string[], n = 5): string[] {
	const q = name.toLowerCase();
	const contains = names.filter((t) => t.toLowerCase().includes(q) || q.includes(t.toLowerCase()));
	const rest = names
		.filter((t) => !contains.includes(t))
		.map((t) => [distance(q, t.toLowerCase()), t] as const)
		.sort((a, b) => a[0] - b[0] || a[1].localeCompare(b[1]))
		.map(([, t]) => t);
	return [...contains.sort(), ...rest].slice(0, n);
}

export type Cell = { task: Task; split: string; scene?: number; seed?: number; envId?: string };

/**
 * The episode's task, split and (with `--scene`) manifest scene and its seed. Throws naming what
 * is wrong: an unknown task (with near matches), a split without a manifest, a scene out of range.
 */
export function resolveCell(table: Table, flags: { task: string; split: string; scene: string }): Cell {
	const task = table.tasks.find((t) => t.name === flags.task);
	if (!task) {
		const near = nearMatches(
			flags.task,
			table.tasks.map((t) => t.name),
		);
		throw new Error(
			`--task-name ${flags.task} is not one of the ${table.tasks.length} RoboCasa365 tasks; near matches: ${near.join(", ")}`,
		);
	}
	if (flags.scene === "") {
		if (!SEED_SPLITS.includes(flags.split)) throw new Error(`--split must be one of ${SEED_SPLITS.join(", ")}`);
		return {
			task,
			split: flags.split,
			...(table.splits.includes(flags.split) ? { envId: envId(task.name, flags.split) } : {}),
		};
	}
	if (!table.splits.includes(flags.split))
		throw new Error(
			`--scene needs --split ${table.splits.join(" | ")} (the splits with a manifest), not ${flags.split}`,
		);
	const scene = Number(flags.scene);
	if (!(Number.isInteger(scene) && scene >= 0 && scene < task.manifest.length))
		throw new Error(`--scene must be a manifest index 0..${task.manifest.length - 1}, not ${flags.scene}`);
	return { task, split: flags.split, scene, seed: task.manifest[scene], envId: envId(task.name, flags.split) };
}
