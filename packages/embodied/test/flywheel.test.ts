import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
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
