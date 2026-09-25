/**
 * LIBERO's Flash hook: how a recorded LIBERO plan meets the live scene (the replay itself is ../flash).
 *
 *   pi -p -e src/libero --model flash/replay --suite libero_object_swap --task 3 --seed 0 \
 *     --molmo http://127.0.0.1:18400 "Solve the task."
 *
 * Each anchor is re-read the way it was recorded: `segment` anchors by SAM3 (the segment tool), the
 * rest by Molmo pointing in the opening agentview image, profiled through back_project; the arm
 * then parks over each Molmo anchor and asks again from the wrist, kept only within 5 cm of the
 * coarse reading. Waypoints are replayed as offsets from their live anchor, and while an object is
 * held, as offsets of the object rather than the gripper. A `pi0_pick` that does not take hold is
 * retried by ../flash. With `--molmo off` nothing is pointed at: point anchors stay where they were
 * recorded and picks keep their recorded thresholds without retries, so the plan replays its
 * recorded calls verbatim (meaningful only on the recorded seed).
 *
 * Plans are `<family>_<suite>_t<task>_{plan,anchors}.json` in `--flash-plans`, else in the LIBERO memory root
 * (`--memory-dir`, or the synced HF memory) under `flash/` (flash-generate.ts) or `task_card/` (the HF
 * dataset). Molmo runs in its own env: `python -m pi_embodied_services.components.molmo_server` (see
 * services/README.md).
 */

import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	type FlashCall,
	type FlashHook,
	type FlashPicks,
	type FlashReply,
	type FlashRobot,
	flash,
} from "../flash/index.ts";
import { RpcClient } from "../rpc.ts";
import type { PlanEntry } from "./flash-generate.ts";

type Json = Record<string, unknown>;
type XY = [number, number];
type Program = { name: string; plan: PlanEntry[]; reference: Map<string, XY>; locatorOf: Map<string, string> };

/** A close reading further than this from the coarse one has found something else. */
const REFINE_ACCEPT = 0.05;
/** Beyond this a waypoint was not written relative to any located object. */
const MAX_ATTACH = 0.2;
/** The reachable workspace; a reading outside it is not a position. */
const REACH = 0.45;
/** How far a held object can plausibly sit from the gripper holding it. */
const MAX_HELD = 0.06;
/** Height-dependent parallax of a wrist reading of a held object. */
const PARALLAX = { x: [0.0231, 0.061], y: [-0.0029, 0.2056] } as const;
/** A pick that did not take hold is retried in place by replaying its approach. */
const PICKS: FlashPicks = {
	isPick: (name) => name === "pi0_pick",
	succeeded: (reply) => (reply.json.result as Json | undefined)?.success === true,
	attempts: 3,
	approach: ["move_to", "move_pose", "set_gripper", "rotate_wrist"],
	keep: 6,
	boundary: ["release", "pi0_doubled"],
	release: "release",
};
const FLASH_PICK_THRESHOLDS = {
	lift_thresh: 0.04,
	gripper_closed_thresh: 0.07,
	gripper_open_thresh: 0.003,
	descent_thresh: 0.0,
};
/** The object phrase without its leading article: the templates supply their own "the" ("the bowl" -> "bowl"). */
const bare = (o: string) => o.trim().replace(/^(?:(?:the|a|an)\s+)+/i, "");
const PROMPTS = {
	survey: (o: string) => o,
	refine: (o: string) => `the center of the ${bare(o)} directly below the gripper`,
	held: (o: string) => `the body of the ${bare(o)} held in the gripper`,
};
const IMAGE = 1024;
const SUITE = /^libero_(10|goal|object|spatial)_(task|swap)$/;

const median = (v: number[]) => {
	const s = [...v].sort((a, b) => a - b);
	return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};
const medianXY = (pts: number[][]): XY => [median(pts.map((p) => p[0])), median(pts.map((p) => p[1]))];
const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const r4 = (v: number) => Number(v.toFixed(4));
const worldOf = (j: Json) => {
	const w = j.world_xyz;
	return Array.isArray(w) && w.length >= 3 && w.slice(0, 3).every((v) => typeof v === "number" && Number.isFinite(v))
		? (w.slice(0, 3) as number[])
		: undefined;
};
const parallax = (xy: XY, span: number): XY => [
	xy[0] - (PARALLAX.x[0] + PARALLAX.x[1] * span),
	xy[1] - (PARALLAX.y[0] + PARALLAX.y[1] * span),
];

function load(dir: string, name: string): Program {
	const read = (suffix: string) => {
		const path = join(dir, `${name}_${suffix}.json`);
		try {
			return JSON.parse(readFileSync(path, "utf8")) as Json;
		} catch (err) {
			throw new Error(`no complete Flash plan ${path} (plan and anchors are both required): ${err}`);
		}
	};
	const plan = read("plan").plan as PlanEntry[];
	const anchors = read("anchors").anchors as { phrase: string; locator: string; median_xy: XY }[];
	return {
		name,
		plan,
		reference: new Map(anchors.map((a) => [a.phrase, a.median_xy])),
		locatorOf: new Map(anchors.map((a) => [a.phrase, a.locator])),
	};
}

/**
 * Re-localize the program's anchors and return how its calls are rewritten. Motion results carry
 * `{result, terminated, state}` and the agentview + wrist images.
 */
async function start(program: Program, robot: FlashRobot, molmo: RpcClient | undefined) {
	const { plan, reference, locatorOf } = program;
	const { act, note } = robot;
	const images = () => robot.latest().images;
	const move = (name: string, args: Json) => robot.move({ name, arguments: args });
	await molmo?.ready(30_000);

	/** Molmo's point for `query` in a 1024 camera image, as [col, row] in that image. */
	async function point(image: string | undefined, query: string): Promise<XY | undefined> {
		if (!image || !molmo) return undefined;
		const res = await molmo.call<{ point_xy?: number[]; image_size?: number[] }>(
			"molmo.ground",
			{ image_base64: image, query },
			180_000,
		);
		if (!res.point_xy) return undefined;
		const [w, h] = res.image_size ?? [IMAGE, IMAGE];
		return [(res.point_xy[0] * IMAGE) / w, (res.point_xy[1] * IMAGE) / h];
	}

	/** World points of pixels in the current image of `camera`. */
	async function project(camera: string, pixels: XY[]): Promise<number[][]> {
		const replies = await act(
			pixels.map(([col, row]) => ({
				name: "back_project",
				arguments: { row: Math.round(row), col: Math.round(col), camera, resolution: "high" },
			})),
		);
		return replies.flatMap((r) => (r.error === undefined && worldOf(r.json) ? [worldOf(r.json) as number[]] : []));
	}

	/** A phrase's pixel profiled down a vertical line; readings below the top 3 cm left the object. */
	async function locate(camera: "agentview" | "wrist", query: string): Promise<XY | undefined> {
		const px = await point(images()[camera === "agentview" ? 0 : 1], query);
		if (!px) return undefined;
		const line = Array.from({ length: 9 }, (_, i): XY => [px[0], px[1] - 45 + i * 11.25]);
		const pts = await project(camera, line);
		if (!pts.length) return undefined;
		const top = Math.max(...pts.map((p) => p[2]));
		return medianXY(pts.filter((p) => p[2] > top - 0.03));
	}

	/** What is in the gripper, sampled on a grid, corrected for parallax. */
	async function heldBody(query: string): Promise<XY | undefined> {
		const px = await point(images()[1], query);
		if (!px) return undefined;
		const grid: XY[] = [];
		for (const dc of [-40, 0, 40]) for (const dr of [-40, 0, 40]) grid.push([px[0] + dc, px[1] + dr]);
		const pts = await project("wrist", grid);
		if (pts.length < 4) return undefined;
		const centre = medianXY(pts);
		const near = pts.filter((p) => dist(p, centre) < 0.05);
		if (near.length < 3) return undefined;
		const z = pts.map((p) => p[2]);
		return parallax(medianXY(near), Math.max(...z) - Math.min(...z));
	}

	// Coarse survey of the opening frame.
	await move("view_env_state", {});
	const live = new Map<string, XY>();
	for (const phrase of reference.keys()) {
		let xy: XY | undefined;
		if (locatorOf.get(phrase) === "segment") {
			const [r] = await act([
				{ name: "segment", arguments: { prompt: phrase, camera: "agentview", min_score: 0.2 } },
			]);
			const w = r.error === undefined ? worldOf(r.json) : undefined;
			if (!w) note(`${phrase} segmentation failed: ${r.error ?? r.json.error ?? "no world_xyz"}`);
			xy = w ? [w[0], w[1]] : undefined;
		} else if (molmo) {
			xy = await locate("agentview", PROMPTS.survey(phrase));
		} else {
			xy = reference.get(phrase);
			note(`${phrase} kept at its recorded position (no Molmo)`);
		}
		if (!xy || Math.max(Math.abs(xy[0]), Math.abs(xy[1])) > REACH) {
			note(`${phrase} not located, or out of reach`);
			continue;
		}
		if ([...live.values()].some((a) => dist(a, xy) < 0.03)) continue;
		live.set(phrase, xy);
	}
	note(`survey ${[...live].map(([p, a]) => `${p}=(${a[0].toFixed(3)},${a[1].toFixed(3)})`).join("  ")}`);

	// Park over each point-grounded anchor and read it again from the wrist.
	const zs = plan.flatMap((s) => {
		const xyz = s.arguments.xyz;
		const z = Array.isArray(xyz) ? Number(xyz[2]) : Number.NaN;
		return (s.action === "move_to" || s.action === "move_pose") && Number.isFinite(z) ? [z] : [];
	});
	const hover = zs.length ? Math.max(...zs) : 0.72;
	for (const [phrase, coarse] of molmo ? [...live] : []) {
		if (locatorOf.get(phrase) === "segment") continue;
		try {
			await move("move_to", {
				xyz: [r4(coarse[0]), r4(coarse[1]), hover],
				gripper: -1,
				step_clip: 0.02,
				max_steps: 150,
				tol: 0.012,
			});
		} catch {
			continue;
		}
		const close = await locate("wrist", PROMPTS.refine(phrase));
		if (!close) continue;
		const gap = dist(close, coarse);
		if (gap > REFINE_ACCEPT) {
			note(`${phrase} close reading ${gap.toFixed(3)} away, rejected`);
			continue;
		}
		note(`${phrase} refined by ${gap.toFixed(3)}`);
		live.set(phrase, close);
	}

	// The plan, with every anchored waypoint moved to its live anchor.
	let offset: XY = [0, 0];
	let heldPhrase = "object";
	return {
		localized: live.size,
		picks: molmo ? PICKS : undefined,
		rewrite(entry: PlanEntry): FlashCall | "skip" | "stop" {
			const name = entry.action;
			const args: Json = { ...entry.arguments };
			if (name === "move_to" || name === "move_pose") {
				const xyz = Array.isArray(args.xyz) ? (args.xyz as number[]) : [];
				if (xyz.length !== 3) return "skip";
				let target: XY = [xyz[0], xyz[1]];
				const phrase = entry.anchor;
				const attached = phrase !== undefined && (entry.anchor_distance ?? 9) <= MAX_ATTACH;
				if (attached && !live.has(phrase)) {
					note(`${phrase} unavailable; stopping replay`);
					return "stop";
				}
				if (attached) {
					const a = live.get(phrase) as XY;
					const o = entry.offset ?? [0, 0];
					target = [a[0] + o[0], a[1] + o[1]];
				}
				if ((args.gripper ?? -1) === 1) target = [target[0] - offset[0], target[1] - offset[1]];
				if (Math.max(Math.abs(target[0]), Math.abs(target[1])) > REACH) return "skip";
				args.xyz = [r4(target[0]), r4(target[1]), xyz[2]];
			} else if (name === "segment" || name === "segment_point") {
				return "skip";
			} else if (name === "pi0_pick" || name === "pi0_doubled") {
				const stripped = String(args.prompt ?? "").replace(/^(pick up|grasp)\s+the\s+/i, "");
				heldPhrase = stripped.split(/\b(?:on|in|into|inside|by|and)\b/)[0].trim();
				if (name === "pi0_pick" && molmo) Object.assign(args, FLASH_PICK_THRESHOLDS);
			}
			return { name, arguments: args };
		},
		async after(call: FlashCall, reply: FlashReply) {
			if (call.name === "release") offset = [0, 0];
			if (call.name !== "set_gripper") return;
			const body = await heldBody(PROMPTS.held(heldPhrase));
			const eef = (reply.json.state as { robot0_eef_pos?: number[] } | undefined)?.robot0_eef_pos;
			const candidate: XY | undefined = body && eef ? [body[0] - eef[0], body[1] - eef[1]] : undefined;
			if (candidate && Math.hypot(...candidate) <= MAX_HELD) {
				offset = candidate;
				note(`held offset (${offset[0].toFixed(4)},${offset[1].toFixed(4)})`);
			} else offset = [0, 0];
		},
	};
}

/** The episode Flash replays: --suite, --task and, when the robot has it, --libero-type. */
type Cell = () => { suite: string; task: string; liberoType?: string };

/**
 * LIBERO's Flash hook; `cell` reads its --suite, --task and --libero-type. Registers the --molmo and
 * --flash-plans flags. The robot passes it as `flash` in its spec, and ../robot.ts mounts ../flash with it.
 */
export function liberoFlash(pi: ExtensionAPI, cell: Cell): FlashHook<Program> {
	pi.registerFlag("molmo", {
		type: "string",
		default: "http://127.0.0.1:18400",
		description: "Molmo server, or off to replay point anchors at their recorded positions",
	});
	pi.registerFlag("flash-plans", {
		type: "string",
		description: "Directory of Flash plans (default: <memory>/libero/flash, then task_card)",
	});
	let molmo: RpcClient | undefined;
	return {
		load(cwd) {
			const { suite, task, liberoType } = cell();
			// Plans are keyed by pro suite and task index; a LIBERO-plus task index names another task.
			if (liberoType === "plus") throw new Error("Flash plans are recorded on LIBERO-pro tasks, not LIBERO-plus");
			const match = SUITE.exec(suite);
			if (!match) throw new Error(`Flash plans cover libero_{10,goal,object,spatial}_{task,swap}, not ${suite}`);
			const program = `${match[1]}_${match[2]}_t${task}`;
			const flag = pi.getFlag("flash-plans");
			const memory =
				(pi.getFlag("memory-dir") as string | undefined) ||
				join(process.env.PI_EMBODIED_MEMORY || join(homedir(), ".pi", "embodied", "memory"), "libero");
			const dirs = flag ? [String(flag)] : ["flash", "task_card"].map((d) => join(memory, d));
			const plans = dirs.map((d) => resolve(cwd, d));
			const loaded = load(plans.find((d) => existsSync(join(d, `${program}_plan.json`))) ?? plans[0], program);
			const endpoint = String(pi.getFlag("molmo") ?? "");
			molmo = endpoint && endpoint !== "off" ? new RpcClient(endpoint) : undefined;
			return loaded;
		},
		start: (program, robot) => start(program, robot, molmo),
		over: (latest) => latest.json.terminated === true || latest.json.truncated === true,
		solved: (latest) => latest.json.terminated === true,
		// Motion tools answer "Episode already ended (terminated=.., truncated=..)" once LIBERO is done.
		textResult: (text) =>
			/^Episode already ended/.test(text)
				? { terminated: /terminated=true/.test(text), truncated: /truncated=true/.test(text) }
				: undefined,
	};
}

/** Mount Flash with LIBERO's hook directly, for a LIBERO extension whose spec does not pass `flash`. */
export function registerFlash(pi: ExtensionAPI, cell: Cell) {
	flash(pi, liberoFlash(pi, cell));
}
