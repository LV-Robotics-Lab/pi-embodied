import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import metaworld, { backProject, EMPTY_WIDTH_M, STEP_M, TASKS, VECTORS } from "../src/metaworld/index.ts";
import { ground, MOVE_UNITS } from "../src/units/index.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi that records flags, tools and handlers; `values` override flag defaults. */
function stubPi(values: Record<string, unknown> = {}) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	let active: string[] = [];
	const api: Record<string, unknown> = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	};
	const pi = new Proxy(api, { get: (t, k: string) => t[k] ?? (() => {}) }) as unknown as ExtensionAPI;
	const errors: string[] = [];
	// A UI context: a failed start notifies instead of setting the process exit code.
	const ctx = {
		hasUI: true,
		ui: { notify: (msg: string) => errors.push(msg) },
		shutdown: () => {},
		sessionManager: {
			getBranch: () => [],
			getSessionFile: () => undefined,
			getSessionDir: () => undefined,
			getSessionId: () => "s",
		},
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		for (const fn of handlers.get(name) ?? []) await fn({ type: name, ...event }, ctx);
	}
	return { pi, flags, tools, emit, errors, active: () => active };
}

const close = (a: number[], b: number[]) => a.every((x, k) => Math.abs(x - b[k]) < 1e-9);

test("the task table is Metaworld's MT50: 50 distinct *-v3 names, matching the env server's", () => {
	assert.equal(TASKS.length, 50);
	assert.equal(new Set(TASKS).size, 50);
	for (const t of TASKS) assert.match(t, /^[a-z-]+-v3$/);
	const server = readFileSync(
		new URL("../../../services/pi_embodied_services/robots/metaworld/env_server.py", import.meta.url),
		"utf8",
	);
	const table = server.slice(
		server.indexOf("INSTRUCTIONS: dict[str, str] = {"),
		server.indexOf("TASKS = list(INSTRUCTIONS)"),
	);
	const names = [...table.matchAll(/^ {4}"([a-z-]+-v3)": "/gm)].map((m) => m[1]);
	assert.deepEqual(names, [...TASKS]);
	// The three tasks the box runs are in the table.
	for (const t of ["reach-v3", "pick-place-v3", "button-press-v3"])
		assert.ok((TASKS as readonly string[]).includes(t));
});

test("flags: --task and --seed name the episode, --privileged is registered, nothing is active before start", () => {
	const f = stubPi();
	metaworld(f.pi);
	assert.equal(f.flags.task, "reach-v3");
	assert.equal(f.flags.seed, "0");
	assert.equal(f.flags.privileged, false);
	assert.equal(f.flags.units, "false");
	assert.match(String(f.flags.sam3), /^http/);
	assert.deepEqual(f.active(), []);
	for (const name of [
		"view_env_state",
		"view_camera_meta",
		"segment",
		"back_project",
		"move_delta",
		"gripper",
		"finish",
		"act",
	])
		assert.ok(f.tools.has(name), name);
	assert.equal(f.tools.has("ground_truth_poses"), false, "registered only with --privileged at start");
});

test("tool schemas: string enums for the gripper command, cameras and resolutions; a 3-vector delta", () => {
	const f = stubPi();
	metaworld(f.pi);
	const schema = (name: string) => JSON.stringify(f.tools.get(name).parameters);
	assert.match(schema("move_delta"), /"enum":\["open","close"\]/);
	assert.match(schema("move_delta"), /"minItems":3,"maxItems":3/);
	assert.match(schema("gripper"), /"enum":\["open","close"\]/);
	assert.match(schema("segment"), /"enum":\["agentview","wrist"\]/);
	assert.match(schema("back_project"), /"enum":\["low","high"\]/);
	assert.match(schema("finish"), /"enum":\["success","failure"\]/);
});

test("an unknown --task fails the start closed: no tools, an error notice", async () => {
	const f = stubPi({ task: "reach-v9" });
	metaworld(f.pi);
	await f.emit("session_start");
	assert.deepEqual(f.active(), []);
	assert.match(f.errors.join("\n"), /unknown Metaworld task "reach-v9"/);
});

test("units grounding: each MV_* is one 2 cm step along the Sawyer's world axes (+y away, -x = robot-left)", () => {
	for (const unit of MOVE_UNITS) {
		const move = ground({ vectors: VECTORS, stepM: STEP_M }, unit)!;
		assert.ok(
			close(
				move.delta,
				VECTORS[unit].map((x) => x * 0.02),
			),
			`${unit} ${move.delta}`,
		);
		assert.equal(Math.hypot(...move.delta), STEP_M);
	}
	assert.deepEqual(VECTORS.MV_FWD, [0, 1, 0]);
	assert.deepEqual(VECTORS.MV_LEFT, [-1, 0, 0]);
	assert.deepEqual(VECTORS.MV_UP, [0, 0, 1]);
	assert.ok(EMPTY_WIDTH_M > 0.023 && EMPTY_WIDTH_M < 0.04, "above the empty-close pad distance, below a held puck");
});

test("back_project: a metric depth map goes through OpenCV intrinsics and the camera-to-world transform", () => {
	// A 2x2 camera at the origin looking along +z (identity extrinsic), f = 1, principal point (1, 1).
	const k = [
		[1, 0, 1],
		[0, 1, 1],
		[0, 0, 1],
	];
	const eye = [
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1],
	];
	const xyz = backProject([2, 2, 2, 2], 2, k, eye);
	// pixel (row 0, col 0): x = (0 - 1) * 2 / 1 = -2, y = -2, z = 2
	assert.deepEqual([...xyz.slice(0, 3)], [-2, -2, 2]);
	// pixel (row 1, col 1) is on the principal point.
	assert.deepEqual([...xyz.slice(9, 12)], [0, 0, 2]);
	// A translated camera shifts every point.
	const moved = eye.map((r, i) => (i < 3 ? [...r.slice(0, 3), [0.5, 0, 1][i]] : r));
	assert.deepEqual([...backProject([2, 2, 2, 2], 2, k, moved).slice(9, 12)], [0.5, 0, 3]);
});
