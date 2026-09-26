/**
 * Flash from a robot's memory recipe, for any robot whose tools take world targets (../robocasa,
 * ../robotwin): the replay (./index.ts) of a solved episode's primitive calls, re-anchored on the
 * live scene when the plan says what each target was relative to.
 *
 *   pi -p -e src/<robot> --model flash/replay <the robot's task flags> [--molmo URL|off] "Solve the task."
 *
 * A program is, in the robot's plan directories (--flash-plans, else the memory's `flash/` then
 * `task_only/`), under the first of the robot's program names that has one:
 *   <name>_plan.json      an anchored plan (./generate.ts): `{plan: [{action, arguments, anchor?,
 *                         offset?}], anchors: [{phrase, xyz}]}`
 *   <name>_recipe.jsonl   the memory recipe exploration exports (`{action, ...arguments}` per line),
 *                         replayed as recorded: meaningful on the recorded seed
 * Re-anchoring: after one observation, each anchor phrase is pointed at by Molmo in the robot's main
 * camera image and turned into a world point by the robot's own back-projection tool (`locate`); a
 * target attached to an anchor then moves with it in x/y (its height is the recorded one). An
 * anchor that cannot be found stops the replay before the first call that needs it. With
 * `--molmo off` anchors stay where they were recorded.
 */

import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { RpcClient } from "../rpc.ts";
import type { FlashCall, FlashHook, FlashPicks, FlashReply, FlashRobot } from "./index.ts";

type Json = Record<string, unknown>;
export type AnchoredEntry = { action: string; arguments: Json; anchor?: string; offset?: [number, number] };
export type Anchor = { phrase: string; xyz: number[] };
export type RecipeProgram = { name: string; plan: AnchoredEntry[]; anchors: Anchor[] };

/** Where a tool's world target lives in its arguments: `xyz` ([x, y, z]) or `xy` ([x, y]). */
export type Target = "xyz" | "xy";

export type RecipeFlashOptions = {
	/** Program names for this episode, most specific first (e.g. the cell, then its reference). */
	names: () => string[];
	/** The robot's memory directory (its `flash/` and `task_only/` hold plans). */
	memory: () => string;
	/** The tool that observes without acting (the replay's first call). */
	observe: string;
	/** Tools whose world target follows an anchor, and which argument holds it. */
	targets: Record<string, Target>;
	/**
	 * A world point for pixel [col, row] of the latest main image, through the robot's own tools.
	 * Without it (a robot with no back-projection) anchors stay where they were recorded.
	 */
	backProject?: (robot: FlashRobot, pixel: [number, number]) => Promise<number[] | undefined>;
	picks?: FlashPicks;
	over: (latest: FlashReply) => boolean;
	solved: (latest: FlashReply) => boolean;
	textResult?: (text: string) => Json | undefined;
};

const r4 = (v: number) => Number(v.toFixed(4));
const isVec = (v: unknown, n: number): v is number[] =>
	Array.isArray(v) && v.length >= n && v.slice(0, n).every((x) => typeof x === "number" && Number.isFinite(x));

/** One program file: an anchored plan, or a memory recipe (no anchors). */
export function loadProgram(path: string, name: string): RecipeProgram {
	const text = readFileSync(path, "utf8");
	if (path.endsWith("_plan.json")) {
		const doc = JSON.parse(text) as { plan?: AnchoredEntry[]; anchors?: Anchor[] };
		if (!Array.isArray(doc.plan) || !doc.plan.length) throw new Error(`${path} has no plan`);
		return { name, plan: doc.plan, anchors: doc.anchors ?? [] };
	}
	const plan = text
		.split("\n")
		.filter((l) => l.trim())
		.map((line, i) => {
			const { action, ...args } = JSON.parse(line) as Json;
			if (typeof action !== "string") throw new Error(`${path}:${i + 1} has no action`);
			return { action, arguments: args };
		});
	if (!plan.length) throw new Error(`${path} holds no calls`);
	return { name, plan, anchors: [] };
}

/** The world target of a call's arguments, and its arguments with the target moved to `xy`. */
export function targetOf(args: Json, where: Target): number[] | undefined {
	const v = args[where];
	return isVec(v, where === "xyz" ? 3 : 2) ? (v as number[]) : undefined;
}
const moved = (args: Json, where: Target, xy: number[]): Json => {
	const v = [...(args[where] as number[])];
	v[0] = r4(xy[0]);
	v[1] = r4(xy[1]);
	return { ...args, [where]: v };
};

/** Molmo's point for `query` in a base64 image, as [col, row] in that image's pixels. */
async function point(molmo: RpcClient, image: string, query: string): Promise<[number, number] | undefined> {
	const res = await molmo.call<{ point_xy?: number[] }>("molmo.ground", { image_base64: image, query }, 180_000);
	return isVec(res.point_xy, 2) ? [res.point_xy[0], res.point_xy[1]] : undefined;
}

/** The replay of `program`: anchors located once, then every call rewritten against them. */
export async function startRecipe(
	program: RecipeProgram,
	robot: FlashRobot,
	o: Pick<RecipeFlashOptions, "observe" | "targets" | "backProject" | "picks">,
	endpoint: RpcClient | undefined,
) {
	const live = new Map<string, number[]>();
	const locator = o.backProject;
	const molmo = locator ? endpoint : undefined;
	if (program.anchors.length) {
		await robot.move({ name: o.observe, arguments: {} });
		for (const a of program.anchors) {
			if (!molmo) {
				live.set(a.phrase, a.xyz);
				continue;
			}
			const image = robot.latest().images[0];
			const px = image ? await point(molmo, image, a.phrase) : undefined;
			const xyz = px && locator ? await locator(robot, px) : undefined;
			if (xyz) live.set(a.phrase, xyz);
			robot.note(
				xyz
					? `${a.phrase} at (${xyz[0].toFixed(3)},${xyz[1].toFixed(3)}), recorded (${a.xyz[0].toFixed(3)},${a.xyz[1].toFixed(3)})`
					: `${a.phrase} not located`,
			);
		}
		if (!molmo) robot.note("anchors kept at their recorded positions (no Molmo or no back-projection)");
	}
	return {
		localized: molmo ? live.size : 0,
		picks: o.picks,
		rewrite(entry: AnchoredEntry): FlashCall | "stop" {
			const where = o.targets[entry.action];
			if (!entry.anchor || !where || !targetOf(entry.arguments, where))
				return { name: entry.action, arguments: { ...entry.arguments } };
			const a = live.get(entry.anchor);
			if (!a) {
				robot.note(`${entry.anchor} unavailable; stopping replay`);
				return "stop";
			}
			const off = entry.offset ?? [0, 0];
			return { name: entry.action, arguments: moved(entry.arguments, where, [a[0] + off[0], a[1] + off[1]]) };
		},
	};
}

/** Register the --molmo and --flash-plans flags and return the robot's recipe Flash hook. */
export function recipeFlash(pi: ExtensionAPI, o: RecipeFlashOptions): FlashHook<RecipeProgram> {
	pi.registerFlag("molmo", {
		type: "string",
		default: "http://127.0.0.1:18400",
		description: "Molmo server for Flash re-anchoring, or off to replay anchors at their recorded positions",
	});
	pi.registerFlag("flash-plans", {
		type: "string",
		description: "Directory of Flash programs (default: the memory's flash/, then task_only/)",
	});
	let molmo: RpcClient | undefined;
	return {
		load(cwd) {
			const flag = pi.getFlag("flash-plans");
			const dirs = (flag ? [String(flag)] : ["flash", "task_only"].map((d) => join(o.memory(), d))).map((d) =>
				resolve(cwd, d),
			);
			const tried: string[] = [];
			for (const name of o.names())
				for (const dir of dirs)
					for (const file of [`${name}_plan.json`, `${name}_recipe.jsonl`]) {
						const path = join(dir, file);
						tried.push(path);
						if (!existsSync(path)) continue;
						const endpoint = String(pi.getFlag("molmo") ?? "");
						molmo = endpoint && endpoint !== "off" ? new RpcClient(endpoint) : undefined;
						return loadProgram(path, name);
					}
			throw new Error(`no Flash program for this episode; looked for ${tried.join(", ")}`);
		},
		start: (program, robot) => startRecipe(program, robot, o, molmo),
		over: o.over,
		solved: o.solved,
		textResult: o.textResult,
	};
}
