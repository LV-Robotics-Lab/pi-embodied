import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { inflateRawSync } from "node:zlib";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type FlywheelSpec, flywheel } from "../src/flywheel.ts";
import { NdArray } from "../src/rpc.ts";

type Handler = (event: any, ctx: any) => unknown;

function stubPi(flags: Record<string, unknown>) {
	const handlers = new Map<string, Handler[]>();
	const entries: { type: string; data: any }[] = [];
	const commands = new Map<string, any>();
	const exec: string[][] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: () => {},
		getFlag: (name: string) => flags[name],
		registerCommand: (name: string, c: any) => commands.set(name, c),
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		exec: async (cmd: string, args: string[]) => {
			exec.push([cmd, ...args]);
			return { code: 0, stdout: "{}", stderr: "" };
		},
	} as unknown as ExtensionAPI;
	const emit = async (name: string, event: Record<string, unknown> = {}) => {
		for (const fn of handlers.get(name) ?? []) await fn({ type: name, ...event }, { hasUI: false });
	};
	return { pi, entries, commands, exec, emit };
}

/** A two-arm robot whose camera size is the first frame's. */
const SPEC: FlywheelSpec = {
	robot: "toy",
	images: { head_images: null, wrist_images: [2, 2, 3] },
	state: 3,
	action: 2,
};
const img = (h: number, w: number, v: number) => new NdArray("uint8", [h, w, 3], Buffer.alloc(h * w * 3, v));
const obs = (v: number, head = img(4, 6, v)) => ({
	images: { head_images: head, wrist_images: img(2, 2, v) },
	state: [v, v, v],
});

test("the recorder writes the robot's arrays, takes open sizes from the first frame, and exports by robot and selection", async () => {
	const root = mkdtempSync(join(tmpdir(), "flywheel-"));
	const f = stubPi({ "collect-flywheel-data": true, "flywheel-root": root });
	const fly = flywheel(f.pi, SPEC, () => "cfg/task");
	assert.equal(fly.recording, false);
	fly.reset(obs(0), { path: ["cfg", "task", "seed_000"], metadata: { task_name: "task", task_language: "do it" } });
	assert.equal(fly.recording, true);
	await f.emit("tool_execution_start", { toolName: "vla" });
	const vla = fly.proposal("do it", [
		[0, 1],
		[2, 3],
	]);
	fly.transition([0, 1], obs(1), 0, false, false, vla, 0);
	fly.transition([2, 3], obs(2), 1, true, false, vla, 1);
	// Another head size than the episode's first is refused.
	assert.throws(
		() => fly.transition([0, 0], obs(3, img(5, 6, 3)), 0, false, false),
		/head_images must be uint8 4,6,3/,
	);
	assert.throws(() => fly.proposal("x", [[1, 2, 3]]), /proposal must be 2 finite values/);
	await f.emit("session_shutdown");
	const dir = join(root, "raw", "toy", "cfg", "task", "seed_000");
	const [episode] = readdirSync(dir);
	const meta = JSON.parse(readFileSync(join(dir, episode, "episode.json"), "utf8"));
	assert.equal(meta.robot, "toy");
	assert.equal(meta.task_name, "task");
	assert.deepEqual([meta.step_count, meta.training_step_count, meta.is_success], [2, 2, true]);
	assert.ok(existsSync(join(dir, episode, "transitions.npz")) && existsSync(join(dir, episode, "proposals.npz")));
	assert.equal(f.entries[0].type, "flywheel_episode");

	await f.commands.get("flywheel-export").handler("", { hasUI: false });
	const argv = f.exec[0];
	assert.deepEqual(argv.slice(argv.indexOf("--robot"), argv.indexOf("--robot") + 4), [
		"--robot",
		"toy",
		"--select",
		"cfg/task",
	]);
});

test("without --collect-flywheel-data nothing is recorded", () => {
	const f = stubPi({});
	const fly = flywheel(f.pi, SPEC, () => "cfg/task");
	fly.reset(obs(0), { path: ["cfg"], metadata: { task_language: "x" } });
	assert.equal(fly.recording, false);
	assert.equal(fly.proposal("x", [[0, 0]]), -1);
});

/** The `.npy` header text of each member of a deflated `.npz` (zip local headers, in order). */
function npzHeaders(zip: Buffer): Record<string, string> {
	const out: Record<string, string> = {};
	for (let o = 0; zip.readUInt32LE(o) === 0x04034b50; ) {
		const size = zip.readUInt32LE(o + 18);
		const nameLength = zip.readUInt16LE(o + 26);
		const extra = zip.readUInt16LE(o + 28);
		const name = zip.subarray(o + 30, o + 30 + nameLength).toString();
		const body = zip.subarray(o + 30 + nameLength + extra, o + 30 + nameLength + extra + size);
		out[name] = inflateRawSync(body).subarray(0, 128).toString("latin1");
		o += 30 + nameLength + extra + size;
	}
	return out;
}

test("VLA chunks of different horizons are padded and their lengths recorded; one horizon adds nothing", async () => {
	const root = mkdtempSync(join(tmpdir(), "flywheel-"));
	const proposals = async (dir: string, horizons: number[]) => {
		const f = stubPi({ "collect-flywheel-data": true, "flywheel-root": root });
		const fly = flywheel(f.pi, { ...SPEC, robot: dir }, () => "cfg/task");
		fly.reset(obs(0), { path: ["cfg", "task", "seed_000"], metadata: { task_language: "do it" } });
		for (const h of horizons) {
			const id = fly.proposal(
				"do it",
				Array.from({ length: h }, () => [0.5, 0.25]),
			);
			fly.transition([0, 1], obs(1), 0, false, false, id, 0);
		}
		await f.emit("session_shutdown");
		const [episode] = readdirSync(join(root, "raw", dir, "cfg", "task", "seed_000"));
		return npzHeaders(readFileSync(join(root, "raw", dir, "cfg", "task", "seed_000", episode, "proposals.npz")));
	};
	// Pi0.5's 5-step chunk next to OpenVLA-OFT's 8-step one.
	const mixed = await proposals("mixed", [5, 8]);
	assert.match(mixed["actions.npy"], /'shape': \(2, 8, 2\)/);
	assert.match(mixed["horizon.npy"], /'<i4'.*'shape': \(2,\)/);
	const same = await proposals("same", [5, 5]);
	assert.deepEqual(Object.keys(same), ["actions.npy", "created_step.npy", "primitive_id.npy", "instruction.npy"]);
	assert.match(same["actions.npy"], /'shape': \(2, 5, 2\)/);
});

test("the spec's other vectors are recorded with every observation; a missing one is refused", async () => {
	const root = mkdtempSync(join(tmpdir(), "flywheel-"));
	const f = stubPi({ "collect-flywheel-data": true, "flywheel-root": root });
	const fly = flywheel(f.pi, { ...SPEC, robot: "joint", vectors: { joint_states: 4 } }, () => "cfg/task");
	const withJoints = (v: number) => ({ ...obs(v), vectors: { joint_states: [v, v, v, v] } });
	fly.reset(withJoints(0), { path: ["cfg", "task", "seed_000"], metadata: { task_language: "do it" } });
	fly.transition([0, 1], withJoints(1), 0, false, false);
	assert.throws(() => fly.transition([0, 1], obs(2), 0, false, false), /joint_states must be 4 finite values/);
	fly.transition([2, 3], withJoints(2), 1, true, false);
	await f.emit("session_shutdown");
	const dir = join(root, "raw", "joint", "cfg", "task", "seed_000");
	const [episode] = readdirSync(dir);
	const headers = npzHeaders(readFileSync(join(dir, episode, "transitions.npz")));
	assert.match(headers["joint_states.npy"], /'<f4'.*'shape': \(3, 4\)/);
	assert.match(headers["states.npy"], /'shape': \(3, 3\)/);
});
