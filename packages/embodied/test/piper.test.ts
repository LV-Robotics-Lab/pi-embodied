import assert from "node:assert/strict";
import { mkdtempSync, realpathSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import piper, { headingToBase, PIPER_UNITS, piperViews } from "../src/piper/index.ts";
import { defineRobot } from "../src/robot.ts";
import type { Move, Vec3 } from "../src/units/index.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi (as in dual_franka.test.ts); `confirm` answers from `confirms`. No robot is reachable. */
function fakePi(flagValues: Record<string, unknown> = {}, hasUI = true) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	const notes: string[] = [];
	const confirms: boolean[] = [];
	let active: string[] = ["stale"];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const dir = realpathSync(mkdtempSync(join(tmpdir(), "piper-")));
	let shutdown = false;
	const ctx = {
		hasUI,
		cwd: dir,
		ui: {
			notify: (m: string) => notes.push(m),
			setWidget: () => {},
			input: async () => "",
			select: async () => undefined,
			confirm: async () => confirms.shift() ?? false,
		},
		shutdown: () => {
			shutdown = true;
		},
		sessionManager: {
			getBranch: () => [],
			getSessionDir: () => dir,
			getSessionFile: () => undefined,
			getSessionId: () => "s",
		},
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) {
			const r: any = await fn({ type: name, ...event }, ctx);
			if (r !== undefined) result = r;
			if (r && name === "tool_call" && r.block) return r;
		}
		return result;
	}
	const run = (name: string, params: Record<string, unknown> = {}) =>
		tools.get(name).execute("id", params, undefined, undefined, ctx);
	return { pi, emit, run, tools, flags, entries, notes, confirms, dir, active: () => active, shut: () => shutdown };
}

/** Start a session and return what the base recorded; restores process.exitCode. */
async function start(f: ReturnType<typeof fakePi>) {
	const errors: string[] = [];
	const log = console.error;
	console.error = (...a: unknown[]) => errors.push(a.join(" "));
	try {
		await f.emit("session_start");
	} finally {
		console.error = log;
	}
	const code = process.exitCode;
	process.exitCode = 0;
	return { errors, code };
}

test("piper refuses to start without an operator UI and stays tool-less", async () => {
	const f = fakePi({ operator: true }, false);
	piper(f.pi);
	const { errors, code } = await start(f);
	assert.equal(code, 1);
	assert.match(errors.join("\n"), /\[piper\] unavailable: piper drives a real robot: run pi interactively/);
	assert.deepEqual(f.active(), []);
	assert.ok(f.shut());
	const blocked = await f.emit("tool_call", { toolName: "move_delta" });
	assert.equal(blocked.block, true);
	assert.match(blocked.reason, /piper is not available/);
});

test("piper refuses to start without --operator, before touching the robot", async () => {
	const f = fakePi({ python: "/nonexistent/python" });
	piper(f.pi);
	await start(f);
	assert.match(f.notes.join("\n"), /piper unavailable: .*start pi with --operator/);
	assert.deepEqual(f.active(), []);
});

test("piper opts into action units with the Show-Harness primitives", () => {
	assert.equal(PIPER_UNITS.stepM, 0.02);
	assert.ok(!("yawStepRad" in PIPER_UNITS), "Show-Harness never offers rotation on the Piper");
	const v = PIPER_UNITS.vectors;
	assert.deepEqual(v.MV_FWD, [1, 0, 0]);
	assert.deepEqual(v.MV_LEFT, [0, 1, 0]);
	assert.deepEqual(v.MV_UP, [0, 0, 1]);
	for (const [a, b] of [
		["MV_FWD", "MV_BACK"],
		["MV_LEFT", "MV_RIGHT"],
		["MV_UP", "MV_DOWN"],
	] as const) {
		assert.equal(Math.hypot(...v[a]), 1);
		assert.deepEqual(
			v[a].map((x) => 0 - x),
			v[b],
		);
	}
});

test("motion beyond the per-call limits is refused before it reaches the robot", async () => {
	const f = fakePi({ operator: true });
	piper(f.pi);
	const far = await f.run("move_delta", { delta_xyz: [0.04, 0.04, 0] });
	assert.match(far.details.error, /moves 0\.0566 m; the limit is 0\.05 m per call/);
	const turn = await f.run("rotate_yaw", { yaw: -0.3 });
	assert.match(turn.details.error, /yaw -0\.3 rad exceeds the limit of 0\.2 rad per call/);
	const nan = await f.run("move_delta", { delta_xyz: [Number.NaN, 0, 0] });
	assert.match(nan.details.error, /finite/);
	// Within the limits the call proceeds to the (absent) robot.
	const ok = await f.run("move_delta", { delta_xyz: [0.02, 0, 0] });
	assert.match(ok.details.error, /piper is not initialized/);

	const tight = fakePi({ operator: true, "max-move": "0.01" });
	piper(tight.pi);
	const unit = await tight.run("move_delta", { delta_xyz: [0, 0, -0.02] });
	assert.match(unit.details.error, /the limit is 0\.01 m per call/);
});

test("with --units the act tool grounds units through the same per-call limits", async () => {
	const f = fakePi({ operator: true, units: true, "units-plugins": "", "max-move": "0.01" });
	piper(f.pi);
	assert.ok(f.tools.has("act"));
	const params = f.tools.get("act").parameters.properties.unit.anyOf.map((u: any) => u.const);
	assert.ok(!params.includes("ROTATE_CW") && !params.includes("STILL"), "no rotation units and one arm");
	// One MV_FWD is 2 cm, over the 1 cm --max-move: refused before any robot call.
	const far = await f.run("act", { unit: "MV_FWD" });
	assert.match(far.content[0].text, /units: MV_FWD x1/);
	assert.match(far.details.error, /moves 0\.02 m; the limit is 0\.01 m per call/);
	assert.deepEqual(far.details.command, { action: "unit", delta: [0.02, 0, 0], yaw: 0, gripper: null });
	// Within the limit the unit reaches the (absent) robot.
	const g = fakePi({ operator: true, units: true, "units-plugins": "" });
	piper(g.pi);
	const up = await g.run("act", { unit: "MV_UP", n: 2 });
	assert.match(up.details.error, /piper is not initialized/);
	assert.match(up.content[0].text, /units: MV_UP x1 of 2 \(stopped early\)/);
	const started = await g.emit("before_agent_start", { systemPrompt: "base" });
	assert.match(started.systemPrompt, /FRONT camera: faces the arm/);
	assert.doesNotMatch(started.systemPrompt, /ROTATE_CW/);
	assert.ok(g.tools.has("rotate_yaw"), "the robot's own rotate_yaw tool stays");
});

test("the stall check compares heading-frame units in the base frame (45 and 90 deg heading)", async () => {
	for (const deg of [45, 90]) {
		const heading = (deg * Math.PI) / 180;
		const f = fakePi({ units: true, "units-plugins": "proprioception" });
		const pos = [0.3, 0, 0.2];
		const moves: Move[] = [];
		defineRobot(f.pi, {
			name: "heading",
			task: [],
			keepImages: 2,
			start: async () => [],
			result: () => ({}),
			finish: {
				description: "finish",
				parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
				result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
			},
			units: {
				...PIPER_UNITS,
				// A perfect Piper in `units_frame: heading`: the server rotates each delta by the gripper heading.
				apply: async (move) => {
					moves.push(move);
					const d = headingToBase(move.delta, { heading_yaw_rad: heading });
					for (let i = 0; i < 3; i++) pos[i] += d[i];
					return { content: [{ type: "text", text: "obs" }], details: {} };
				},
				state: async () => ({ eef_xyz: [...pos], gripper_width: 0.05, heading_yaw_rad: heading }),
				baseDelta: (delta: Vec3, state) => headingToBase(delta, state),
			},
		});
		await f.emit("session_start");
		for (const unit of ["MV_FWD", "MV_LEFT"]) {
			const r = await f.run("act", { unit, n: 3 });
			assert.match(r.content[0].text, new RegExp(`units: ${unit} x3\\n`), `${deg} deg ${unit}`);
			assert.doesNotMatch(r.content[0].text, /blocked/, `${deg} deg ${unit}`);
		}
		assert.equal(moves.length, 6);
	}
	assert.deepEqual(
		headingToBase([0.02, 0, 0], { heading_yaw_rad: Math.PI / 2 }).map((v) => Number(v.toFixed(6))),
		[0, 0.02, 0],
	);
	assert.deepEqual(headingToBase([0.02, 0, 0], {}), [0.02, 0, 0], "no heading (tool vertical): base frame");
});

test("heading-frame units describe front-view moves along the gripper heading, not image edges", () => {
	const base = piperViews("base");
	assert.match(base, /MV_FWD moves the gripper toward the image bottom/);
	assert.doesNotMatch(base, /GRIPPER HEADING/);
	const heading = piperViews("heading");
	assert.match(heading, /FRONT camera: faces the arm.*Moves follow the GRIPPER HEADING/);
	assert.doesNotMatch(heading, /MV_FWD moves the gripper toward the image bottom/);
	// The wrist view is exact in the heading frame (Show-Harness wrist_frame leaves it as is).
	for (const v of [base, heading]) assert.match(v, /WRIST camera: .*a target near the image TOP needs MV_FWD/);
});
