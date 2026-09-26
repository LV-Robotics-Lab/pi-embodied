/**
 * Flywheel data collection (--collect-flywheel-data). Records every env transition of the episode,
 * the VLA chunks that proposed actions, and the primitive (tool call) that ran them, in the raw
 * episode format, which pi_embodied_services.flywheel validates and exports:
 *
 *   <root>/raw/<robot>/<the robot's cell path>/episode_<utc>_<hex>/
 *     transitions.npz  proposals.npz  episode.json
 *
 * What an observation holds is the robot's (`FlywheelSpec`): its camera images, its state vector and
 * its action vector, the ones its VLA reads and emits. The episode is written when the session ends.
 * /flywheel-export runs `python -m pi_embodied_services.flywheel.cli export-lerobot` in the services
 * dir (--services, with --flywheel-python, else --python), which keeps each successful episode up
 * to its first `terminated` step and writes a LeRobot v3.0 dataset (lerobot 0.4 in that Python)
 * with the shared feature names `observation.images.<camera>`, `observation.state` and `action`.
 */

import { randomBytes } from "node:crypto";
import { mkdirSync, renameSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type Npy, writeNpz } from "./npz.ts";
import { SERVICES } from "./robot.ts";
import type { NdArray } from "./rpc.ts";

/** What a robot records: services pi_embodied_services/robots/<robot>/flywheel.py holds the same shapes. */
export type FlywheelSpec = {
	/** The raw data directory under `raw/` and the services spec the export uses. */
	robot: string;
	/** Camera images of every observation, by transitions.npz key: [H, W, 3] uint8 (null: the first frame's size). */
	images: Record<string, readonly [number, number, number] | null>;
	/** Length of the state vector (`states`) and of the action vector (`actions`). */
	state: number;
	action: number;
};
export type FlywheelObs = { images: Record<string, NdArray | null | undefined>; state: number[] };
/**
 * Where the episode goes and what it says about itself: `path` below `raw/<robot>/` (the export
 * selects a prefix of it), and episode.json fields (`task_language` required).
 */
export type FlywheelMeta = { path: string[]; metadata: Record<string, unknown> & { task_language: string } };
type Proposal = { created_step: number; primitive_id: number; instruction: string; actions: number[][] };

/**
 * The suite key of a LIBERO episode (its raw directory, `suite` in episode.json).
 * LIBERO-plus task indices name other tasks than standard/pro ones (plus libero_spatial task 0 is a
 * table-texture variant; standard task 0 is the plain scene), so plus episodes get `<suite>_plus`.
 * Standard and pro share the suite name: their task sets are identical.
 */
export const flywheelSuite = (suite: string, liberoType: string) => (liberoType === "plus" ? `${suite}_plus` : suite);

const f32 = (v: number[]) => Buffer.from(Float32Array.from(v).buffer);
const i32 = (v: number[]) => Buffer.from(Int32Array.from(v).buffer);
const same = (a: readonly number[], b: readonly number[]) => a.length === b.length && a.every((x, i) => x === b[i]);
const PART = /^[A-Za-z0-9][A-Za-z0-9_.=-]*$/;

function image(name: string, a: NdArray | null | undefined, shape: readonly number[]): Buffer {
	if (!a || a.dtype !== "uint8" || !same(a.shape, shape))
		throw new Error(`flywheel: ${name} must be uint8 ${shape}, got ${a ? `${a.dtype} ${a.shape}` : "none"}`);
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

function episode(root: string, spec: FlywheelSpec, meta: FlywheelMeta, first: FlywheelObs) {
	for (const part of [spec.robot, ...meta.path])
		if (!PART.test(part)) throw new Error(`flywheel: invalid path part ${JSON.stringify(part)}`);
	if (!meta.metadata.task_language) throw new Error("flywheel: the episode has no task language");
	// A size the robot leaves open is its first frame's, and holds for the episode.
	const shapes = Object.fromEntries(
		Object.entries(spec.images).map(([k, shape]) => [k, shape ?? (first.images[k]?.shape as readonly number[])]),
	) as Record<string, readonly number[]>;
	const stamp = new Date()
		.toISOString()
		.replace(/[-:]/g, "")
		.replace(/\.(\d{3})Z$/, ".$1000Z");
	return {
		spec,
		meta,
		id: `episode_${stamp}_${randomBytes(4).toString("hex")}`,
		dir: join(root, "raw", spec.robot, ...meta.path),
		shapes,
		images: Object.fromEntries(
			Object.entries(shapes).map(([k, shape]) => [k, [image(k, first.images[k], shape)]]),
		) as Record<string, Buffer[]>,
		states: [f32(vector("states", first.state, spec.state))],
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

/** Write `ep` in the schema-1 layout (via a `.partial` directory). */
function write(ep: Episode): { path: string; step_count: number; is_success: boolean } {
	const { spec } = ep;
	const n = ep.actions.length;
	const training = ep.terminated.indexOf(true) + 1;
	const path = join(ep.dir, ep.id);
	const partial = `${path}.partial`;
	mkdirSync(partial, { recursive: true });
	writeNpz(join(partial, "transitions.npz"), {
		...Object.fromEntries(
			Object.entries(ep.shapes).map(([k, shape]) => [
				k,
				{ descr: "|u1", shape: [n + 1, ...shape], data: Buffer.concat(ep.images[k]) },
			]),
		),
		states: { descr: "<f4", shape: [n + 1, spec.state], data: Buffer.concat(ep.states) },
		actions: { descr: "<f4", shape: [n, spec.action], data: Buffer.concat(ep.actions) },
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
		actions: {
			descr: "<f4",
			shape: [p.length, horizon, spec.action],
			data: f32(p.flatMap((x) => x.actions.flat())),
		},
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
		...ep.meta.metadata,
		robot: spec.robot,
		episode_id: ep.id,
		is_success: training > 0,
		primitive_names: ep.names,
		proposal_count: p.length,
		schema_version: 1,
		step_count: n,
		stop_reason: stop,
		training_step_count: training,
	};
	writeFileSync(join(partial, "episode.json"), `${JSON.stringify(metadata, null, 2)}\n`);
	renameSync(partial, path);
	return { path, step_count: n, is_success: training > 0 };
}

/**
 * Mount the recorder. `select` is the export's default selection: the path below `raw/<robot>/`
 * whose episodes make one dataset (e.g. LIBERO's `<suite>/task_NN`, every seed of one task).
 */
export function flywheel(pi: ExtensionAPI, spec: FlywheelSpec, select: () => string) {
	pi.registerFlag("collect-flywheel-data", {
		type: "boolean",
		default: false,
		description: "Record this episode for Flywheel training",
	});
	pi.registerFlag("flywheel-root", {
		type: "string",
		description: "Flywheel data root (default ~/.pi/embodied/datacollection)",
	});
	pi.registerFlag("flywheel-python", {
		type: "string",
		description: "Python with lerobot for /flywheel-export (default --python)",
	});

	const root = () =>
		resolve(String(pi.getFlag("flywheel-root") || join(homedir(), ".pi", "embodied", "datacollection")));
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
		description: "Export successful Flywheel episodes to a LeRobot v3.0 dataset: [selection] [dataset-id]",
		handler: async (args, ctx) => {
			const [selection = select(), id] = args.trim().split(/\s+/).filter(Boolean);
			const cli = ["-m", "pi_embodied_services.flywheel.cli", "export-lerobot", "--data-root", root()];
			cli.push("--robot", spec.robot, "--select", selection, ...(id ? ["--dataset-id", id] : []));
			// LeRobot's pins (numpy 2, huggingface-hub) conflict with the env servers', hence its own Python.
			const python = String(
				pi.getFlag("flywheel-python") || pi.getFlag("python") || process.env.PI_EMBODIED_PYTHON || "python",
			);
			const services = String(pi.getFlag("services") || process.env.PI_EMBODIED_SERVICES || SERVICES);
			const res = await pi.exec("env", [`PYTHONPATH=${services}`, python, ...cli], { cwd: services });
			const out = res.code === 0 ? res.stdout.trim() : res.stderr.trim() || `exit code ${res.code}`;
			if (ctx.hasUI) ctx.ui.notify(out, res.code === 0 ? "info" : "error");
			else {
				console.error(`[flywheel] ${out}`);
				if (res.code !== 0) process.exitCode = 1;
			}
		},
	});

	return {
		/** Start an episode at env.reset (finishing any open one). No-op without --collect-flywheel-data. */
		reset(obs: FlywheelObs, meta: FlywheelMeta) {
			finish();
			if (pi.getFlag("collect-flywheel-data")) ep = episode(root(), spec, meta, obs);
		},
		/** Whether this episode is being recorded (robots skip building observations otherwise). */
		get recording() {
			return ep !== undefined;
		},
		/** A VLA action chunk [horizon, action] about to be executed; returns its id for `transition`. */
		proposal(instruction: string, actions: NdArray | number[][]): number {
			if (!ep) return -1;
			const width = spec.action;
			let rows: number[][];
			if (Array.isArray(actions)) rows = actions;
			else {
				if (actions.shape.length !== 2 || actions.shape[1] !== width)
					throw new Error(`flywheel: chunk must be [H, ${width}]`);
				const flat = actions.toArray();
				rows = Array.from({ length: actions.shape[0] }, (_, i) => flat.slice(i * width, (i + 1) * width));
			}
			rows = rows.map((r) => vector("proposal", r, width));
			const created_step = ep.actions.length;
			ep.proposals.push({ created_step, primitive_id: primitive(ep), instruction, actions: rows });
			return ep.proposals.length - 1;
		},
		/** One env step: the action sent, the observation it produced, and the env's flags. */
		transition(
			action: number[],
			obs: FlywheelObs,
			reward: number,
			terminated: boolean,
			truncated: boolean,
			vla = -1,
			index = -1,
		) {
			if (!ep) return;
			const a = f32(vector("action", action, spec.action));
			const shots = Object.entries(ep.shapes).map(([k, shape]) => [k, image(k, obs.images[k], shape)] as const);
			const state = f32(vector("states", obs.state, spec.state));
			ep.actions.push(a);
			for (const [k, data] of shots) ep.images[k].push(data);
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
