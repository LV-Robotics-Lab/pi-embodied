/**
 * Flash from a robot's memory recipe, for any robot whose tools take world targets (../robocasa,
 * ../robotwin) or base-frame deltas (../maniskill, ../robolab): the replay (./index.ts) of a solved
 * episode's primitive calls, re-anchored on the live scene when the plan says what each target was
 * relative to.
 *
 *   pi -p -e src/<robot> --model flash/replay <the robot's task flags> [--molmo URL|off] "Solve the task."
 *
 * A program is, in the robot's plan directories (--flash-plans, else the memory's `flash/` then
 * `task_only/`), under the first of the robot's program names that has one:
 *   <name>_plan.json      an anchored plan (./generate.ts): `{plan: [{action, arguments, anchor?,
 *                         offset?, to?}], anchors: [{phrase, xyz}]}`
 *   <name>_recipe.jsonl   the memory recipe exploration exports (`{action, ...arguments}` per line),
 *                         replayed as recorded: meaningful on the recorded seed
 * Re-anchoring: after one observation, each anchor phrase is pointed at by Molmo in the robot's main
 * camera image and turned into a world point by the robot's own back-projection (a tool, or a
 * ray-plane intersection at the anchor's recorded height, ./plane.ts); with a fixed camera and the
 * plan's recorded view the anchor moves by the difference of the live and the recorded reading
 * (`fixedCamera`), else it is the live reading; a target attached to an
 * anchor then moves with it in x/y (its height is the recorded one). A relative target (a delta
 * move) is a reconstructed absolute waypoint: at replay the delta is the anchored waypoint minus
 * where the end effector is now, split into moves the robot's per-call limit allows. An anchor that
 * cannot be found stops the replay before the first call that needs it. With `--molmo off` anchors
 * stay where they were recorded.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { RpcClient } from "../rpc.ts";
import type { FlashCall, FlashHook, FlashPicks, FlashReply, FlashRobot } from "./index.ts";

type Json = Record<string, unknown>;
/** One planned call; `to` is a relative target's absolute end position (the recording's, before anchoring). */
export type AnchoredEntry = {
	action: string;
	arguments: Json;
	anchor?: string;
	offset?: [number, number];
	to?: number[];
};
export type Anchor = { phrase: string; xyz: number[] };
/** `view`: the recorded episode's opening main-camera image (base64 PNG), when the plan names one. */
export type RecipeProgram = { name: string; plan: AnchoredEntry[]; anchors: Anchor[]; view?: string };

/**
 * A tool that moves by a base-frame delta (`move_delta`). Its plan entries carry the absolute end
 * position the recording reached (`to`, from the tool's own results); at replay an anchored entry's
 * delta is that waypoint, moved with its anchor, minus `position` of the latest result, in moves of
 * at most `maxStep` m.
 */
export type RelativeTarget = {
	/** The argument holding the base-frame [dx, dy, dz], m. */
	delta: string;
	/** The end-effector position in a motion result's JSON (e.g. `state.tcp_pos`). */
	position: (json: Json) => number[] | undefined;
	/** Largest translation one call may run, m; a longer move is split evenly. */
	maxStep?: number;
};
/** Where a tool's world target lives: an `xyz` / `xy` argument, or a delta (a relative target). */
export type Target = "xyz" | "xy" | RelativeTarget;

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
	 * A world point for pixel [col, row] of the latest main image, through the robot's own tools or
	 * its camera calibration (`anchor.xyz[2]` is the height the anchor was recorded at). Without it
	 * (a robot with no back-projection) anchors stay where they were recorded.
	 */
	backProject?: (robot: FlashRobot, pixel: [number, number], anchor: Anchor) => Promise<number[] | undefined>;
	/**
	 * `backProject` depends on the pixel alone (a fixed calibrated camera, no live depth), so a pixel of
	 * the plan's recorded view back-projects as truly as a live one. Anchors are then re-localized
	 * differentially: pointed at in both images, the recorded anchor moves by the difference of the two
	 * readings, and the pointer's bias on an object (where on a banana it points) cancels.
	 */
	fixedCamera?: boolean;
	/**
	 * Tools that turn the hand by a relative angle (rad) under a per-call cap (RoboLab's `rotate_delta`):
	 * a planned turn is wrapped to (-pi, pi] and split into calls of at most `maxStep`, so a plan whose
	 * turn exceeds the cap (an older generator's, or a hand-written one) is not refused at replay.
	 */
	turns?: Record<string, { arg: string; maxStep: number }>;
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
		const doc = JSON.parse(text) as { plan?: AnchoredEntry[]; anchors?: Anchor[]; view?: string };
		if (!Array.isArray(doc.plan) || !doc.plan.length) throw new Error(`${path} has no plan`);
		const view = doc.view ? readFileSync(join(dirname(path), doc.view)).toString("base64") : undefined;
		return { name, plan: doc.plan, anchors: doc.anchors ?? [], ...(view ? { view } : {}) };
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

/** The world target of a plan entry: its `xyz` / `xy` argument, or a relative target's end position `to`. */
export function targetOf(entry: AnchoredEntry, where: Target): number[] | undefined {
	if (typeof where !== "string") return isVec(entry.to, 3) ? entry.to : undefined;
	const v = entry.arguments[where];
	return isVec(v, where === "xyz" ? 3 : 2) ? v : undefined;
}
const moved = (args: Json, where: "xyz" | "xy", xy: number[]): Json => {
	const v = [...(args[where] as number[])];
	v[0] = r4(xy[0]);
	v[1] = r4(xy[1]);
	return { ...args, [where]: v };
};

/** `delta` as the calls of one plan entry: one, or evenly split when it exceeds the tool's per-call limit. */
export function splitMove(entry: AnchoredEntry, where: RelativeTarget, delta: number[]): FlashCall[] {
	const n = where.maxStep ? Math.max(1, Math.ceil(Math.hypot(...delta) / where.maxStep - 1e-9)) : 1;
	const step = delta.map((v) => r4(v / n));
	return Array.from({ length: n }, (_, i) => ({
		name: entry.action,
		// The other arguments (a gripper command) run once, with the first move.
		arguments: i === 0 ? { ...entry.arguments, [where.delta]: step } : { [where.delta]: step },
	}));
}

/**
 * A relative turn as calls the tool accepts: its angle wrapped to (-pi, pi] (a -350 deg turn is +10 deg)
 * and split evenly into steps of at most `maxStep` rad; other arguments ride on the first call.
 */
export function splitAngle(entry: AnchoredEntry, turn: { arg: string; maxStep: number }): FlashCall[] {
	const raw = entry.arguments[turn.arg];
	if (typeof raw !== "number" || !Number.isFinite(raw))
		return [{ name: entry.action, arguments: { ...entry.arguments } }];
	const yaw = Math.atan2(Math.sin(raw), Math.cos(raw));
	// Measured at the 0.1 mrad the plan stores: a recorded full-cap turn reads back a hair over it.
	const n = Math.max(1, Math.ceil(r4(Math.abs(yaw)) / turn.maxStep - 1e-9));
	const step = r4(yaw / n);
	return Array.from({ length: n }, (_, i) => ({
		name: entry.action,
		arguments: i === 0 ? { ...entry.arguments, [turn.arg]: step } : { [turn.arg]: step },
	}));
}

/** Molmo's point for `query` in a base64 image, as [col, row] in that image's pixels. */
async function point(molmo: RpcClient, image: string, query: string): Promise<[number, number] | undefined> {
	const res = await molmo.call<{ point_xy?: number[] }>("molmo.ground", { image_base64: image, query }, 180_000);
	return isVec(res.point_xy, 2) ? [res.point_xy[0], res.point_xy[1]] : undefined;
}

/** The replay of `program`: anchors located once, then every call rewritten against them. */
export async function startRecipe(
	program: RecipeProgram,
	robot: FlashRobot,
	o: Pick<RecipeFlashOptions, "observe" | "targets" | "backProject" | "picks" | "fixedCamera" | "turns">,
	endpoint: RpcClient | undefined,
) {
	const live = new Map<string, number[]>();
	const locator = o.backProject;
	const molmo = locator ? endpoint : undefined;
	// Relative targets need the end-effector position before their first move, anchors an image.
	if (program.anchors.length || program.plan.some((e) => typeof o.targets[e.action] === "object"))
		await robot.move({ name: o.observe, arguments: {} });
	if (program.anchors.length) {
		for (const a of program.anchors) {
			if (!molmo) {
				live.set(a.phrase, a.xyz);
				continue;
			}
			const image = robot.latest().images[0];
			const px = image ? await point(molmo, image, a.phrase) : undefined;
			const seen = px && locator ? await locator(robot, px, a) : undefined;
			// Differential: where the same pointing put the anchor in the recorded view.
			const was = o.fixedCamera && program.view && seen ? await point(molmo, program.view, a.phrase) : undefined;
			const then = was && locator ? await locator(robot, was, a) : undefined;
			const xyz =
				seen && then ? [a.xyz[0] + seen[0] - then[0], a.xyz[1] + seen[1] - then[1], ...a.xyz.slice(2)] : seen;
			if (xyz) live.set(a.phrase, xyz);
			const pixel = (p: number[] | undefined) => (p ? p.map((v) => Math.round(v)).join(",") : "-");
			robot.note(
				xyz
					? `${a.phrase} at (${xyz[0].toFixed(3)},${xyz[1].toFixed(3)}), recorded (${a.xyz[0].toFixed(3)},${a.xyz[1].toFixed(3)}) [pixel ${pixel(px)}${
							then && seen
								? `; recorded view pixel ${pixel(was)}, read (${then[0].toFixed(3)},${then[1].toFixed(3)}) then, (${seen[0].toFixed(3)},${seen[1].toFixed(3)}) now`
								: ""
						}]`
					: `${a.phrase} not located`,
			);
		}
		if (!molmo) robot.note("anchors kept at their recorded positions (no Molmo or no back-projection)");
	}

	/** A delta move: the anchored waypoint minus where the end effector is now, else the recorded delta. */
	function relative(entry: AnchoredEntry, where: RelativeTarget): FlashCall[] | "stop" {
		const recorded = entry.arguments[where.delta];
		if (!isVec(recorded, 3)) return [{ name: entry.action, arguments: { ...entry.arguments } }];
		const to = targetOf(entry, where);
		if (!entry.anchor || !to) return splitMove(entry, where, recorded);
		const a = live.get(entry.anchor);
		if (!a) {
			robot.note(`${entry.anchor} unavailable; stopping replay`);
			return "stop";
		}
		const now = where.position(robot.latest().json);
		if (!isVec(now, 3)) throw new Error(`${entry.action}: the latest result carries no end-effector position`);
		const off = entry.offset ?? [0, 0];
		return splitMove(entry, where, [a[0] + off[0] - now[0], a[1] + off[1] - now[1], to[2] - now[2]]);
	}

	return {
		localized: molmo ? live.size : 0,
		picks: o.picks,
		rewrite(entry: AnchoredEntry): FlashCall | FlashCall[] | "stop" {
			const turn = o.turns?.[entry.action];
			if (turn) return splitAngle(entry, turn);
			const where = o.targets[entry.action];
			if (where !== undefined && typeof where !== "string") return relative(entry, where);
			if (!entry.anchor || !where || !targetOf(entry, where))
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
