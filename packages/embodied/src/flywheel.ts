/**
 * Flywheel data collection (RPent's --collect-flywheel-data). Records every env transition
 * of the episode, the VLA chunks that proposed actions, and the primitive (tool call) that
 * ran them, in RPent's raw episode format so RPent's own `rpent-flywheel` validates and
 * exports it:
 *
 *   <root>/raw/libero/<suite>/task_NN/seed_NNN/episode_<utc>_<hex>/
 *     transitions.npz  proposals.npz  episode.json
 *
 * The episode is written when the session ends. /flywheel-export runs
 * `rpent-flywheel export-lerobot`, which keeps each successful episode up to its first
 * `terminated` step and writes a LeRobot dataset (needs lerobot>=0.3.3,<0.4).
 */

import { randomBytes } from "node:crypto";
import { mkdirSync, renameSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type Npy, writeNpz } from "./npz.ts";
import type { NdArray } from "./rpc.ts";

const IMAGE = [256, 256, 3];
const STATE = 8;
const ACTION = 7;

type Obs = { main_images: NdArray; wrist_images?: NdArray | null; states: NdArray };
type Meta = { suite: string; task_id: number; seed: number; task_language: string };
type Proposal = { created_step: number; primitive_id: number; instruction: string; actions: number[][] };

const f32 = (v: number[]) => Buffer.from(Float32Array.from(v).buffer);
const i32 = (v: number[]) => Buffer.from(Int32Array.from(v).buffer);
const same = (a: number[], b: number[]) => a.length === b.length && a.every((x, i) => x === b[i]);

function image(name: string, a: NdArray | null | undefined): Buffer {
	if (!a || a.dtype !== "uint8" || !same(a.shape, IMAGE))
		throw new Error(`flywheel: ${name} must be uint8 ${IMAGE}, got ${a ? `${a.dtype} ${a.shape}` : "none"}`);
	return a.data;
}

function vector(name: string, v: number[], size: number): number[] {
	if (v.length !== size || !v.every(Number.isFinite))
		throw new Error(`flywheel: ${name} must be ${size} finite values`);
	return v;
}

/** numpy '<U' array: fixed-width UTF-32LE. */
function strings(values: string[]): Npy {
	const chars = values.map((v) => Array.from(v, (c) => c.codePointAt(0) as number));
	const width = Math.max(1, ...chars.map((c) => c.length));
	const data = Buffer.alloc(values.length * width * 4);
	for (const [i, c] of chars.entries()) for (const [j, p] of c.entries()) data.writeUInt32LE(p, (i * width + j) * 4);
	return { descr: `<U${width}`, shape: [values.length], data };
}

function episode(root: string, meta: Meta, first: Obs) {
	if (!/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(meta.suite)) throw new Error(`flywheel: invalid suite ${meta.suite}`);
	if (!(Number.isInteger(meta.task_id) && meta.task_id >= 0 && Number.isInteger(meta.seed) && meta.seed >= 0))
		throw new Error("flywheel: task and seed must be non-negative integers");
	const stamp = new Date()
		.toISOString()
		.replace(/[-:]/g, "")
		.replace(/\.(\d{3})Z$/, ".$1000Z");
	const task = `task_${String(meta.task_id).padStart(2, "0")}`;
	return {
		meta,
		id: `episode_${stamp}_${randomBytes(4).toString("hex")}`,
		dir: join(root, "raw", "libero", meta.suite, task, `seed_${String(meta.seed).padStart(3, "0")}`),
		main: [image("main_images", first.main_images)],
		wrist: [image("wrist_images", first.wrist_images)],
		states: [f32(vector("states", first.states.toArray(), STATE))],
		actions: [] as Buffer[],
		rewards: [] as number[],
		terminated: [] as boolean[],
		truncated: [] as boolean[],
		primitive: [] as number[],
		vla: [] as number[],
		index: [] as number[],
		names: [] as string[],
		active: -1,
		proposals: [] as Proposal[],
	};
}
type Episode = ReturnType<typeof episode>;

/** Write `ep` in RPent's schema-1 layout (via a `.partial` directory, like RPent). */
function write(ep: Episode): { path: string; step_count: number; is_success: boolean } {
	const n = ep.actions.length;
	const training = ep.terminated.indexOf(true) + 1;
	const path = join(ep.dir, ep.id);
	const partial = `${path}.partial`;
	mkdirSync(partial, { recursive: true });
	writeNpz(join(partial, "transitions.npz"), {
		main_images: { descr: "|u1", shape: [n + 1, ...IMAGE], data: Buffer.concat(ep.main) },
		wrist_images: { descr: "|u1", shape: [n + 1, ...IMAGE], data: Buffer.concat(ep.wrist) },
		states: { descr: "<f4", shape: [n + 1, STATE], data: Buffer.concat(ep.states) },
		actions: { descr: "<f4", shape: [n, ACTION], data: Buffer.concat(ep.actions) },
		rewards: { descr: "<f4", shape: [n], data: f32(ep.rewards) },
		terminated: { descr: "|b1", shape: [n], data: Buffer.from(ep.terminated.map(Number)) },
		truncated: { descr: "|b1", shape: [n], data: Buffer.from(ep.truncated.map(Number)) },
		action_source: { descr: "|u1", shape: [n], data: Buffer.from(ep.vla.map((v) => (v >= 0 ? 1 : 0))) },
		primitive_id: { descr: "<i4", shape: [n], data: i32(ep.primitive) },
		vla_chunk_id: { descr: "<i4", shape: [n], data: i32(ep.vla) },
		proposal_index: { descr: "<i2", shape: [n], data: Buffer.from(Int16Array.from(ep.index).buffer) },
	});
	const p = ep.proposals;
	const horizon = p[0]?.actions.length ?? 0;
	if (p.some((x) => x.actions.length !== horizon)) throw new Error("flywheel: VLA chunks differ in horizon");
	writeNpz(join(partial, "proposals.npz"), {
		actions: { descr: "<f4", shape: [p.length, horizon, ACTION], data: f32(p.flatMap((x) => x.actions.flat())) },
		created_step: { descr: "<i4", shape: [p.length], data: i32(p.map((x) => x.created_step)) },
		primitive_id: { descr: "<i4", shape: [p.length], data: i32(p.map((x) => x.primitive_id)) },
		instruction: strings(p.map((x) => x.instruction)),
	});
	const stop = ep.terminated.includes(true)
		? "env_terminated"
		: ep.truncated.includes(true)
			? "env_truncated"
			: "agent_stopped";
	const metadata = {
		episode_id: ep.id,
		is_success: training > 0,
		primitive_names: ep.names,
		proposal_count: p.length,
		schema_version: 1,
		seed: ep.meta.seed,
		step_count: n,
		stop_reason: stop,
		suite: ep.meta.suite,
		task_id: ep.meta.task_id,
		task_language: ep.meta.task_language,
		training_step_count: training,
	};
	writeFileSync(join(partial, "episode.json"), `${JSON.stringify(metadata, null, 2)}\n`);
	renameSync(partial, path);
	return { path, step_count: n, is_success: training > 0 };
}

export function flywheel(pi: ExtensionAPI) {
	pi.registerFlag("collect-flywheel-data", {
		type: "boolean",
		default: false,
		description: "Record this episode for Flywheel training",
	});
	pi.registerFlag("flywheel-root", {
		type: "string",
		description: "Flywheel data root (default <rpent>/datacollection)",
	});
	pi.registerFlag("flywheel-python", {
		type: "string",
		default: process.env.FLYWHEEL_PYTHON ?? process.env.RPENT_PYTHON ?? "python",
		description: "Python with RPent and lerobot>=0.3.3,<0.4, for /flywheel-export",
	});

	const rpent = () => String(pi.getFlag("rpent") || process.env.RPENT_ROOT || ".");
	const root = () => resolve(String(pi.getFlag("flywheel-root") || join(rpent(), "datacollection")));
	let ep: Episode | undefined;
	let tool: string | undefined;

	/** Primitive boundaries are tool calls; a call becomes a primitive once it moves the robot. */
	function primitive(e: Episode): number {
		if (tool !== undefined && e.active < 0) {
			e.active = e.names.length;
			e.names.push(tool);
		}
		return e.active;
	}

	function finish(ctx?: ExtensionContext) {
		if (!ep) return;
		const done = ep;
		ep = undefined;
		try {
			pi.appendEntry("flywheel_episode", write(done));
		} catch (err) {
			const message = `[flywheel] episode not written: ${err instanceof Error ? err.message : err}`;
			if (ctx?.hasUI) ctx.ui.notify(message, "error");
			else console.error(message);
		}
	}

	pi.on("tool_execution_start", (event) => {
		tool = event.toolName;
		if (ep) ep.active = -1;
	});
	pi.on("tool_execution_end", () => {
		tool = undefined;
		if (ep) ep.active = -1;
	});
	pi.on("session_shutdown", (_event, ctx) => finish(ctx));

	pi.registerCommand("flywheel-export", {
		description: "Export successful Flywheel episodes to LeRobot: [suite] [task] [dataset-id]",
		handler: async (args, ctx) => {
			const [suite = String(pi.getFlag("suite") ?? ""), task = String(pi.getFlag("task") ?? ""), id] = args
				.trim()
				.split(/\s+/)
				.filter(Boolean);
			const cli = ["-m", "rpent.flywheel.cli", "export-lerobot", "--data-root", root(), "--suite", suite];
			cli.push("--task", task, ...(id ? ["--dataset-id", id] : []));
			const python = String(pi.getFlag("flywheel-python"));
			const res = await pi.exec("env", [`PYTHONPATH=${rpent()}`, python, ...cli], { cwd: rpent() });
			ctx.ui.notify(res.code === 0 ? res.stdout.trim() : res.stderr.trim(), res.code === 0 ? "info" : "error");
		},
	});

	return {
		/** Start an episode at env.reset (finishing any open one). No-op without --collect-flywheel-data. */
		reset(obs: Obs, meta: Meta) {
			finish();
			if (pi.getFlag("collect-flywheel-data")) ep = episode(root(), meta, obs);
		},
		/** A VLA action chunk [horizon, 7] about to be executed; returns its id for `transition`. */
		proposal(instruction: string, actions: NdArray): number {
			if (!ep) return -1;
			const flat = actions.toArray();
			if (actions.shape.length !== 2 || actions.shape[1] !== ACTION)
				throw new Error("flywheel: chunk must be [H, 7]");
			const rows = Array.from({ length: actions.shape[0] }, (_, i) =>
				vector("proposal", flat.slice(i * ACTION, (i + 1) * ACTION), ACTION),
			);
			const created_step = ep.actions.length;
			ep.proposals.push({ created_step, primitive_id: primitive(ep), instruction, actions: rows });
			return ep.proposals.length - 1;
		},
		/** One env step: the action sent, the observation it produced, and the env's flags. */
		transition(
			action: number[],
			obs: Obs,
			reward: number,
			terminated: boolean,
			truncated: boolean,
			vla = -1,
			index = -1,
		) {
			if (!ep) return;
			const a = f32(vector("action", action, ACTION));
			const main = image("main_images", obs.main_images);
			const wrist = image("wrist_images", obs.wrist_images);
			const state = f32(vector("states", obs.states.toArray(), STATE));
			ep.actions.push(a);
			ep.main.push(main);
			ep.wrist.push(wrist);
			ep.states.push(state);
			ep.rewards.push(Number(reward));
			ep.terminated.push(terminated);
			ep.truncated.push(truncated);
			ep.primitive.push(primitive(ep));
			ep.vla.push(vla);
			ep.index.push(index);
		},
	};
}
