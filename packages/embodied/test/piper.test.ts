import assert from "node:assert/strict";
import { mkdtempSync, realpathSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import piperDual from "../src/piper/dual.ts";
import piper, { headingToBase, motionFrame, PIPER_UNITS, piperViews } from "../src/piper/index.ts";
import { defineRobot } from "../src/robot.ts";
import type { Move, Vec3 } from "../src/units/index.ts";

type Handler = (event: any, ctx: any) => unknown;

/** The values of an enum schema (StringEnum or a union of literals). */
const enumOf = (schema: any): string[] => schema.enum ?? schema.anyOf.map((u: any) => u.const);

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
	const params = enumOf(f.tools.get("act").parameters.properties.unit);
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

// ---- dual arm, view_select, wrist_frame

/**
 * A mocked Piper env server transport (the JSON RPC of ../src/rpc.ts): answers what the robot asks
 * with canned dual-arm (or single-arm) state and records every call. No robot, no ROS.
 */
async function mockServer(o: { dual?: boolean; frame?: "base" | "heading" } = {}) {
	const dual = o.dual ?? true;
	const calls: { method: string; kwargs: Record<string, any> }[] = [];
	const img = { __ndarray__: Buffer.alloc(4 * 4 * 3, 90).toString("base64"), dtype: "uint8", shape: [4, 4, 3] };
	const armState = (arm: string) => ({
		arm,
		eef_pos: [0.3, arm === "left" ? 0.1 : -0.1, 0.25],
		eef_euler_xyz: [0, 1.2, 0],
		gripper_width_m: 0.07,
		gripper_closed: false,
		heading_yaw_rad: arm === "left" ? 0.785 : -0.785,
		z_floor_m: arm === "left" ? 0.19 : 0.21,
		halted: null,
	});
	const state = () => (dual ? { arms: { left: armState("left"), right: armState("right") } } : armState("left"));
	const cameras = dual ? ["front", "wrist_left", "wrist_right"] : ["front", "wrist"];
	const answer = (method: string, kwargs: Record<string, any>): unknown => {
		switch (method) {
			case "healthz":
				return { status: "ok" };
			case "env.get_env_meta":
				return {
					arm: dual ? "dual" : "left",
					arms: dual ? ["left", "right"] : [],
					cameras,
					units_frame: o.frame ?? "base",
					limits: { max_step_m: 0.05, max_yaw_rad: 0.2, z_floor_m: null, empty_width_m: 0.005 },
					has_begin_pose: true,
					tasks: { banana_handover: { instruction: "hand the banana over" } },
				};
			case "env.reset":
				return { ok: true, robot_state: state() };
			case "env.get_observation":
				return { images: Object.fromEntries(cameras.map((c) => [c, img])), robot_state: state() };
			case "env.get_robot_state":
				return kwargs.arm ? armState(kwargs.arm) : state();
			case "env.step":
				return { ok: true, arm: kwargs.arm, frame: kwargs.frame, notes: [] };
			case "env.halt_arm":
				return { ok: true, arm: kwargs.arm, halted: `halted: ${kwargs.reason}` };
			default:
				throw new Error(`unexpected ${method}`);
		}
	};
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, kwargs } = JSON.parse(body) as { method: string; kwargs: Record<string, any> };
			if (method !== "healthz") calls.push({ method, kwargs });
			try {
				res.end(JSON.stringify({ ok: true, result: answer(method, kwargs) }));
			} catch (err) {
				res.end(JSON.stringify({ ok: false, error: String(err) }));
			}
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
	const close = () => {
		server.closeAllConnections();
		server.close();
	};
	const steps = () => calls.filter((c) => c.method === "env.step").map((c) => c.kwargs);
	return { url, calls, steps, close };
}

/** The dual robot attached to a mocked server, started with the operator's confirm. */
async function dualStarted(
	flags: Record<string, unknown> = {},
	server: { dual?: boolean; frame?: "base" | "heading" } = {},
) {
	const m = await mockServer(server);
	const f = fakePi({ operator: true, task: "banana_handover", "robot-env": m.url, ...flags });
	piperDual(f.pi);
	f.confirms.push(true);
	const started = await start(f);
	return { f, m, started };
}

test(
	"dual Piper: act takes the arm, STILL leaves the other arm alone, no rotation units",
	{
		todo: "asserts the units continuous hint, which lands with the units chaining change",
	},
	async () => {
		const { f, m } = await dualStarted({ units: "both", "units-plugins": "" });
		try {
			const schema = f.tools.get("act").parameters.properties;
			assert.deepEqual(enumOf(schema.arm), ["left", "right"]);
			const vocab = enumOf(schema.unit);
			assert.ok(vocab.includes("STILL") && !vocab.includes("ROTATE_CW"));
			assert.deepEqual(
				m.calls.slice(0, 3).map((c) => c.method),
				["env.get_env_meta", "env.reset", "env.get_observation"],
			);
			assert.deepEqual(m.calls[1].kwargs, {}, "the start reset resets both arms");

			const r = await f.run("act", { unit: "MV_FWD", arm: "right" });
			assert.match(r.content[0].text, /units: MV_FWD x1 \(right arm\)/);
			assert.deepEqual(m.steps(), [
				{ delta_xyz: [0.02, 0, 0], yaw: 0, gripper: null, frame: "base", reopen_empty: false, arm: "right" },
			]);
			// Front, left wrist, right wrist.
			assert.equal(r.content.filter((c: any) => c.type === "image").length, 3);
			// The proprioception of the arm that moved.
			const state = m.calls.filter((c) => c.method === "env.get_robot_state").map((c) => c.kwargs.arm);
			assert.ok(state.length && state.every((a) => a === "right"), JSON.stringify(state));

			const still = await f.run("act", { unit: "STILL", arm: "left" });
			assert.match(still.content[0].text, /STILL: the left arm holds/);
			assert.equal(m.steps().length, 1, "STILL never reaches the robot");

			// A repeated MV_* flows through the join (the server's smooth chaining), the last one settles.
			await f.run("act", { unit: "MV_UP", arm: "right", n: 2 });
			const ups = m.steps().slice(-2);
			assert.equal(ups[0].continuous, true);
			assert.equal(ups[1].continuous, undefined);

			const grasp = await f.run("act", { unit: "GRASP", arm: "left" });
			assert.equal(grasp.details.command.arm, "left");
			assert.equal(m.steps().at(-1)?.gripper, "close");
			assert.equal(m.steps().at(-1)?.arm, "left");
		} finally {
			m.close();
		}
	},
);

test("dual Piper: per-arm refusals before any robot call", async () => {
	const { f, m } = await dualStarted({ "max-move": "0.01", units: "both", "units-plugins": "" });
	try {
		const noArm = await f.run("move_delta", { delta_xyz: [0.01, 0, 0] });
		assert.match(noArm.details.error, /two arms: name the arm \(left or right\)/);
		const bad = await f.run("open_gripper", { arm: "middle" });
		assert.match(bad.details.error, /unknown arm 'middle'/);
		const far = await f.run("move_delta", { delta_xyz: [0, 0, -0.02], arm: "left" });
		assert.match(far.details.error, /the limit is 0\.01 m per call/);
		const unit = await f.run("act", { unit: "MV_DOWN", arm: "right" });
		assert.match(unit.details.error, /moves 0\.02 m; the limit is 0\.01 m per call/);
		const actNoArm = await f.run("act", { unit: "MV_UP" });
		assert.match(actNoArm.details.error, /two arms: name the arm/);
		assert.deepEqual(m.steps(), [], "nothing reached the server");

		const ok = await f.run("move_delta", { delta_xyz: [0, 0.01, 0], arm: "left" });
		assert.equal(ok.details.result.arm, "left");
		const halt = await f.run("halt_arm", { arm: "right", reason: "stage done" });
		assert.equal(halt.details.result.halted, "halted: stage done");
		assert.deepEqual(m.calls.at(-2)?.kwargs, { arm: "right", reason: "stage done" });
		assert.ok(f.active().includes("halt_arm"));
	} finally {
		m.close();
	}
});

test("the entry and the server config must agree on one or two arms", async () => {
	const { f, m } = await dualStarted({}, { dual: false });
	try {
		assert.match(
			f.notes.join("\n"),
			/piper\/dual\.ts drives both arms, but the env server's config has no `arms:` block/,
		);
	} finally {
		m.close();
	}
	const m2 = await mockServer({ dual: true });
	try {
		const g = fakePi({ operator: true, task: "banana_handover", "robot-env": m2.url });
		piper(g.pi);
		await start(g);
		assert.match(
			g.notes.join("\n"),
			/the env server drives both arms .*use packages\/embodied\/src\/piper\/dual\.ts/,
		);
		assert.equal(m2.calls.filter((c) => c.method === "env.reset").length, 0, "nothing moved");
		const one = fakePi({ operator: true });
		piper(one.pi);
		const r = await one.run("move_delta", { delta_xyz: [0.01, 0, 0], arm: "left" });
		assert.match(r.details.error, /drives one arm; omit `arm`/);
		assert.ok(!one.tools.has("halt_arm"));
	} finally {
		m2.close();
	}
});

test("view_select: the guiding view picks the frame, and it refuses the heading config", () => {
	assert.equal(motionFrame("WRIST", "base", true), "heading");
	assert.equal(motionFrame("FRONT", "base", true), "base");
	assert.equal(motionFrame(undefined, "base", true), "base", "no view: the configured frame");
	assert.equal(motionFrame("SIDE", "base", true), "base");
	assert.equal(motionFrame("WRIST", "base", false), "base", "view select off: views are ignored");
	assert.equal(motionFrame("FRONT", "heading", false), "heading");
});

test("view_select refuses motion.units_frame: heading (Show-Harness run_real_dual guard)", async () => {
	const { f, m } = await dualStarted({ "view-select": true }, { frame: "heading" });
	try {
		assert.match(f.notes.join("\n"), /--view-select .* needs motion\.units_frame: base/);
		assert.equal(m.calls.filter((c) => c.method === "env.reset").length, 0);
	} finally {
		m.close();
	}
});

test("view_select end to end: act's view reaches the server as the move's frame (units hook)", async (t) => {
	const { f, m } = await dualStarted({ "view-select": true, units: "both", "units-plugins": "" });
	try {
		if (!f.tools.get("act").parameters.properties.view) {
			t.skip("units has no view_select hook yet (scratchpad units-view-select.patch)");
			return;
		}
		await f.run("act", { unit: "MV_FWD", arm: "left", view: "WRIST" });
		await f.run("act", { unit: "MV_LEFT", arm: "right", view: "FRONT" });
		await f.run("act", { unit: "MV_UP", arm: "left" });
		assert.deepEqual(
			m.steps().map((s) => [s.arm, s.frame]),
			[
				["left", "heading"],
				["right", "base"],
				["left", "base"],
			],
		);
		const started = await f.emit("before_agent_start", { systemPrompt: "base" });
		assert.match(started.systemPrompt, /VIEW SELECT: with every MV_\* set `view`/);
		assert.doesNotMatch(started.systemPrompt, /GRIPPER HEADING/, "view select keeps the base convention");
	} finally {
		m.close();
	}
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

test("wrist_frame: in the heading frame the front view follows the gripper heading", async () => {
	const heading = piperViews("heading");
	assert.match(heading, /FRONT camera: .*Moves follow the GRIPPER HEADING/);
	assert.match(heading, /MV_FWD moves further ahead along the heading/);
	assert.doesNotMatch(heading, /MV_FWD moves the gripper toward the image bottom/);
	// The wrist view (the ego-swapped rule A) is exact in the heading frame and stays.
	assert.match(heading, /WRIST camera: .*a target near the image TOP needs MV_FWD/);
	const base = piperViews("base");
	assert.match(base, /MV_FWD moves the gripper toward the image bottom/);
	assert.doesNotMatch(base, /GRIPPER HEADING/);
	const dual = piperViews("heading", ["left", "right"]);
	assert.match(dual, /Image 2, LEFT WRIST camera/);
	assert.match(dual, /Image 3, RIGHT WRIST camera/);
	assert.match(dual, /the direction that arm's gripper points/);
	const vs = piperViews("base", ["left", "right"], true);
	assert.match(vs, /VIEW SELECT/);

	// The prompt the model gets follows the served config's frame.
	for (const frame of ["heading", "base"] as const) {
		const { f, m } = await dualStarted({ units: true, "units-plugins": "" }, { frame });
		try {
			const started = await f.emit("before_agent_start", { systemPrompt: "base" });
			if (frame === "heading") assert.match(started.systemPrompt, /Moves follow the GRIPPER HEADING/);
			else assert.match(started.systemPrompt, /MV_FWD moves the gripper toward the image bottom/);
			await f.run("act", { unit: "MV_FWD", arm: "left" });
			assert.equal(m.steps()[0].frame, frame);
		} finally {
			m.close();
		}
	}
});
