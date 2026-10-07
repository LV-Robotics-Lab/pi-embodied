import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { type FlashCall, type FlashHook, type FlashProgram, flash, reanchor } from "../src/capabilities/flash/index.ts";
import { defineRobot, type RobotSpec } from "../src/robot.ts";
import { liberoFlash } from "../src/robots/libero/flash.ts";
import { AUDIT_FIELDS, generateFlashPlan, readAudit } from "../src/robots/libero/flash-generate.ts";
import { deployFlags } from "./helpers/deployment.ts";

type Handler = (event: any, ctx: any) => unknown;
type Result = { json?: unknown; text?: string; isError?: boolean };
type Exec = (name: string, args: any) => Result;

/** A stub pi with just what Flash and the robot base use; `turn` asks the registered provider for one model turn. */
function fakePi(flagValues: Record<string, unknown> = {}) {
	flagValues = deployFlags(flagValues);
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const providers: any[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => (name in flags ? flags[name] : flagValues[name]),
		registerProvider: (p: any) => providers.push(p),
		registerTool: () => {},
		registerCommand: () => {},
		setActiveTools: () => {},
		getActiveTools: () => [],
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const ctx = { hasUI: false, cwd: "/", ui: { notify: () => {} }, sessionManager: { getBranch: () => [] } };
	const emit = async (name: string) => {
		for (const fn of handlers.get(name) ?? []) await fn({ type: name }, ctx);
	};
	const turn = (messages: unknown[] = [], signal?: AbortSignal) => {
		const provider = providers.find((p) => p.id === "flash");
		return provider.streamSimple(provider.getModels()[0], { messages }, { signal }).result();
	};
	return { pi, emit, turn, providers };
}

const toolCalls = (m: any): { id: string; name: string; arguments: any }[] =>
	m.content.filter((c: any) => c.type === "toolCall");
const textOf = (m: any) =>
	m.content
		.filter((c: any) => c.type === "text")
		.map((c: any) => c.text)
		.join("\n");
const IMAGES = [
	{ type: "image", data: "AGENTVIEW", mimeType: "image/png" },
	{ type: "image", data: "WRIST", mimeType: "image/png" },
];

/** Answer every turn's calls with `exec` until the replay calls finish; returns the calls sent and the notes. */
async function drive(p: ReturnType<typeof fakePi>, exec: Exec) {
	await p.emit("session_start");
	const history: unknown[] = [];
	const sent: [string, any][] = [];
	const notes: string[] = [];
	for (let i = 0; i < 200; i++) {
		const m = await p.turn(history);
		notes.push(textOf(m));
		const calls = toolCalls(m);
		const finish = calls.find((c) => c.name === "finish");
		if (finish) return { sent, notes: notes.join("\n"), finish: finish.arguments, turn: m };
		assert.ok(calls.length, `turn ${i} called nothing: ${textOf(m)}`);
		sent.push(...calls.map((c): [string, any] => [c.name, c.arguments]));
		history.push(m);
		for (const c of calls) {
			const r = exec(c.name, c.arguments);
			history.push({
				role: "toolResult",
				toolCallId: c.id,
				toolName: c.name,
				content: [{ type: "text", text: r.text ?? JSON.stringify(r.json ?? {}) }, ...IMAGES],
				isError: r.isError ?? false,
			});
		}
	}
	throw new Error("the replay never finished");
}

// ---------------------------------------------------------------- the generic runner, with a toy robot

type ToyEntry = { action: string; arguments: Record<string, unknown>; shift?: boolean };

/** A toy robot: `look` localizes one anchor at +10; shifted entries move by it; `grab` is its pick. */
function toyHook(plan: ToyEntry[], o: { picks?: boolean; stopAt?: string } = {}) {
	const seen: string[] = [];
	const hook: FlashHook<FlashProgram<ToyEntry>> = {
		load: () => ({ name: "toy", plan }),
		async start(_program, robot) {
			const reply = await robot.move({ name: "look", arguments: {} });
			const anchor = Number((reply.json as { anchor?: number }).anchor);
			robot.note(`anchor at ${anchor}`);
			return {
				localized: 1,
				rewrite(entry): FlashCall | "skip" | "stop" {
					if (entry.action === o.stopAt) return "stop";
					if (entry.action === "note") return "skip";
					const x = Number(entry.arguments.x ?? 0);
					return {
						name: entry.action,
						arguments: entry.shift ? { ...entry.arguments, x: x + anchor } : entry.arguments,
					};
				},
				after: (call) => {
					seen.push(call.name);
				},
				picks: o.picks
					? {
							isPick: (name) => name === "grab",
							succeeded: (r) => (r.json as { held?: boolean }).held === true,
							attempts: 3,
							approach: ["go"],
							keep: 2,
							boundary: ["drop"],
							release: "drop",
						}
					: undefined,
			};
		},
		over: (latest) => (latest.json as { done?: boolean }).done === true,
		solved: (latest) => (latest.json as { solved?: boolean }).solved === true,
	};
	return { hook, seen };
}

test("Flash sends the plan in order, rewritten through the robot's hook, and finishes with the outcome", async () => {
	const p = fakePi();
	const { hook, seen } = toyHook([
		{ action: "go", arguments: { x: 1 }, shift: true },
		{ action: "note", arguments: {} },
		{ action: "go", arguments: { x: 2 } },
		{ action: "drop", arguments: {} },
		{ action: "go", arguments: { x: 3 } },
	]);
	flash(p.pi, hook);
	assert.equal(p.providers[0].getModels()[0].id, "replay");
	const out = await drive(p, (name) => ({
		json: name === "look" ? { anchor: 10 } : name === "drop" ? { done: true, solved: true } : {},
	}));
	// The skipped entry is not sent; the plan stops once the robot reports the episode over.
	assert.deepEqual(out.sent, [
		["look", {}],
		["go", { x: 11 }],
		["go", { x: 2 }],
		["drop", {}],
	]);
	assert.deepEqual(seen, ["go", "go", "drop"]);
	assert.equal(out.finish.status, "success");
	assert.match(out.finish.summary, /replayed the toy program: 5 actions, 1 anchors re-localized/);
	assert.match(out.notes, /replaying the toy program[\s\S]*anchor at 10/);
	assert.equal(out.turn.usage.totalTokens, 0);
	// The replay runs once per session.
	assert.match(textOf(await p.turn()), /already ran/);
});

test("a hook's stop ends the replay, and a failed call ends it with an error summary", async () => {
	const stopped = fakePi();
	flash(
		stopped.pi,
		toyHook(
			[
				{ action: "go", arguments: {} },
				{ action: "halt", arguments: {} },
			],
			{ stopAt: "halt" },
		).hook,
	);
	const a = await drive(stopped, () => ({ json: {} }));
	assert.deepEqual(
		a.sent.map(([n]) => n),
		["look", "go"],
	);
	assert.equal(a.finish.status, "failure");

	const failed = fakePi();
	flash(
		failed.pi,
		toyHook([
			{ action: "go", arguments: {} },
			{ action: "go", arguments: {} },
		]).hook,
	);
	const b = await drive(failed, (name) => (name === "go" ? { text: "joint limit", isError: true } : { json: {} }));
	assert.equal(b.sent.length, 2);
	assert.match(b.finish.summary, /^flash error: go failed: joint limit/);
});

test("a pick that fails is retried after replaying its approach; one that never holds skips its carry", async () => {
	const plan: ToyEntry[] = [
		{ action: "go", arguments: { x: 1 } },
		{ action: "go", arguments: { x: 2 } },
		{ action: "go", arguments: { x: 3 } },
		{ action: "grab", arguments: { what: "a" } },
		{ action: "go", arguments: { x: 4 } },
		{ action: "drop", arguments: {} },
		{ action: "go", arguments: { x: 5 } },
		{ action: "grab", arguments: { what: "b" } },
		{ action: "go", arguments: { x: 6 } },
		{ action: "drop", arguments: {} },
		{ action: "go", arguments: { x: 7 } },
	];
	const p = fakePi();
	flash(p.pi, toyHook(plan, { picks: true }).hook);
	let grabs = 0;
	const out = await drive(p, (name, args) => {
		if (name !== "grab") return { json: {} };
		grabs++;
		// `a` takes hold on its second attempt; `b` never does.
		return { json: { held: args.what === "a" && grabs === 2 } };
	});
	assert.deepEqual(
		out.sent.map(([n, a]) => (n === "go" ? `go${a.x}` : n === "grab" ? `grab-${a.what}` : n)),
		[
			"look",
			"go1",
			"go2",
			"go3",
			"grab-a",
			// The approach (the last `keep` = 2 calls) and the pick again.
			"go2",
			"go3",
			"grab-a",
			"go4",
			"drop",
			"go5",
			"grab-b",
			"go5",
			"grab-b",
			"go5",
			"grab-b",
			// Its carry and release are skipped; the plan resumes after the release.
			"go7",
		],
	);
	assert.match(out.notes, /pick unconfirmed, skipping its carry/);
});

test("an abort ends the replay", async () => {
	const p = fakePi();
	flash(p.pi, toyHook([{ action: "go", arguments: {} }]).hook);
	await p.emit("session_start");
	const first = await p.turn();
	assert.deepEqual(
		toolCalls(first).map((c) => c.name),
		["look"],
	);
	const ac = new AbortController();
	ac.abort();
	// The aborted turn sends nothing more, and neither does any later one.
	assert.equal(toolCalls(await p.turn([], ac.signal)).length, 0);
	const after = await p.turn();
	assert.equal(toolCalls(after).length, 0);
	assert.match(textOf(after), /already ran/);
});

test("defineRobot mounts Flash for a spec with a flash hook", () => {
	const p = fakePi();
	const spec: RobotSpec = {
		name: "toy",
		task: [],
		keepImages: 1,
		start: async () => [],
		result: () => ({}),
		finish: {
			description: "finish",
			parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
			result: (params) => ({ content: [{ type: "text", text: params.status }], details: params }),
		},
		flash: toyHook([]).hook,
	};
	defineRobot(p.pi, spec);
	assert.deepEqual(
		p.providers.map((x) => x.id),
		["flash"],
	);
	const without = fakePi();
	defineRobot(without.pi, { ...spec, flash: undefined });
	assert.equal(without.providers.length, 0);
});

test("a replay with no program to load marks the episode invalid; a failed replay does not", async () => {
	const invalid: string[] = [];
	const none = fakePi();
	const missing: FlashHook<FlashProgram<ToyEntry>> = {
		...toyHook([]).hook,
		load: () => {
			throw new Error("no Flash program for this episode; looked for /plans/cell_plan.json");
		},
	};
	flash(none.pi, missing, { invalid: (why) => invalid.push(why) });
	const out = await drive(none, () => ({ json: {} }));
	assert.equal(out.finish.status, "failure");
	assert.match(out.finish.summary, /^flash error: no program to replay: no Flash program for this episode/);
	assert.deepEqual(invalid, [out.finish.summary]);
	// A program that loads and then fails mid-replay is a failed episode, not an invalid one.
	const bad = fakePi();
	flash(bad.pi, toyHook([{ action: "go", arguments: {} }]).hook, { invalid: (why) => invalid.push(why) });
	const failed = await drive(bad, (name) =>
		name === "go" ? { text: "boom", isError: true } : { json: { anchor: 1 } },
	);
	assert.match(failed.finish.summary, /^flash error: go failed/);
	assert.equal(invalid.length, 1);
});

// ---------------------------------------------------------------- LIBERO's hook

/** A goal_swap t3 plan: a Molmo anchor (the bowl) and a SAM3 anchor (the plate). */
function liberoPlans() {
	const dir = mkdtempSync(join(tmpdir(), "flash-"));
	const plan = [
		{
			action: "move_to",
			arguments: { xyz: [0.15, 0.25, 0.95], gripper: -1 },
			anchor: "the bowl",
			offset: [0.05, 0.05],
			anchor_distance: 0.0707,
		},
		{ action: "segment", arguments: { prompt: "the bowl" } },
		{ action: "pi0_pick", arguments: { prompt: "pick up the bowl on the table" } },
		{ action: "set_gripper", arguments: { gripper: 1 } },
		{
			action: "move_to",
			arguments: { xyz: [-0.05, 0.15, 1.0], gripper: 1 },
			anchor: "the plate",
			offset: [0.05, 0.05],
			anchor_distance: 0.0707,
		},
		{ action: "release", arguments: {} },
		{ action: "move_to", arguments: { xyz: [0, 0, 1.1], gripper: -1 } },
	];
	const anchors = [
		{ phrase: "the bowl", locator: "molmo", median_xy: [0.1, 0.2] },
		{ phrase: "the plate", locator: "segment", median_xy: [-0.1, 0.1] },
	];
	writeFileSync(join(dir, "goal_swap_t3_plan.json"), JSON.stringify({ plan }));
	writeFileSync(join(dir, "goal_swap_t3_anchors.json"), JSON.stringify({ anchors }));
	return dir;
}

/** LIBERO's tools as the test plays them: the plate segments at `plate`; the first `pi0_pick` misses. */
function liberoExec(plate: number[]) {
	let picks = 0;
	let camera = "agentview";
	const motion = (extra: Record<string, unknown> = {}) => ({
		json: { result: {}, terminated: false, truncated: false, state: { robot0_eef_pos: [0.1, 0.2, 0.9] }, ...extra },
	});
	return (name: string, args: any): Result => {
		if (name === "segment") return { json: { world_xyz: plate } };
		if (name === "back_project") {
			camera = args.camera;
			return { json: { world_xyz: camera === "agentview" ? [0.13, 0.22, 0.9] : [0.14, 0.22, 0.9] } };
		}
		if (name === "pi0_pick") return motion({ result: { success: ++picks > 1 } });
		if (name === "release") return { text: "Episode already ended (terminated=true, truncated=false)." };
		return motion();
	};
}

const libero = (flags: Record<string, unknown>) => {
	const p = fakePi(flags);
	const hook: RobotSpec["flash"] = liberoFlash(p.pi, () => ({ suite: "libero_goal_swap", task: "3" }));
	if (hook) flash(p.pi, hook);
	return p;
};

test("LIBERO with --molmo off replays the recorded calls verbatim, without pick retries", async () => {
	const p = libero({ molmo: "off", "flash-plans": liberoPlans() });
	const out = await drive(p, liberoExec([-0.1, 0.1, 0.9]));
	assert.deepEqual(out.sent, [
		["view_env_state", {}],
		["segment", { prompt: "the plate", camera: "agentview", min_score: 0.2 }],
		["move_to", { xyz: [0.15, 0.25, 0.95], gripper: -1 }],
		["pi0_pick", { prompt: "pick up the bowl on the table" }],
		["set_gripper", { gripper: 1 }],
		["move_to", { xyz: [-0.05, 0.15, 1.0], gripper: 1 }],
		["release", {}],
	]);
	assert.match(out.notes, /the bowl kept at its recorded position \(no Molmo\)/);
	assert.equal(out.finish.status, "success");
	assert.match(out.finish.summary, /replayed the goal_swap_t3 program: 7 actions, 2 anchors re-localized/);
});

test("--flash-reanchor is on|off: off replays without Molmo, and the old `=false` is an invalid replay", async () => {
	// Bug 43: pi sets a boolean extension flag to true whatever value follows it, so the old
	// `--flash-reanchor=false` never held; the flag is a string with two values (pi keeps a
	// string flag's value as given).
	const stub = (v: unknown) => ({ getFlag: () => v }) as unknown as ExtensionAPI;
	assert.deepEqual([reanchor(stub(undefined)), reanchor(stub("on")), reanchor(stub(" ON "))], [true, true, true]);
	assert.equal(reanchor(stub("off")), false);
	assert.throws(() => reanchor(stub("false")), /--flash-reanchor takes on or off, not "false"/);
	const off = libero({ "flash-reanchor": "off", "flash-plans": liberoPlans() });
	const out = await drive(off, liberoExec([-0.1, 0.1, 0.9]));
	assert.match(out.notes, /the bowl kept at its recorded position \(no Molmo\)/);
	assert.equal(out.finish.status, "success");
	const bad = libero({ "flash-reanchor": "false", "flash-plans": liberoPlans() });
	const invalid = await drive(bad, liberoExec([-0.1, 0.1, 0.9]));
	assert.equal(invalid.finish.status, "failure");
	assert.match(
		invalid.finish.summary,
		/^flash error: no program to replay: --flash-reanchor takes on or off, not "false"/,
	);
});

test("LIBERO with Molmo re-localizes anchors, refines from the wrist, retries a pick, and carries the held offset", async (t) => {
	const queries: string[] = [];
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, kwargs } = JSON.parse(body) as { method: string; kwargs: { query?: string } };
			if (method === "molmo.ground") queries.push(String(kwargs.query));
			const result = method === "molmo.ground" ? { point_xy: [256, 256], image_size: [512, 512] } : "ok";
			res.end(JSON.stringify({ ok: true, result }));
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	t.after(() => {
		server.closeAllConnections();
		server.close();
	});
	const molmo = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
	const p = libero({ molmo, "flash-plans": liberoPlans() });
	const out = await drive(p, liberoExec([-0.12, 0.11, 0.9]));
	const names = out.sent.map(([n]) => n);
	const bp = (n: number) => Array<string>(n).fill("back_project");
	assert.deepEqual(names, [
		"view_env_state",
		...bp(9), // the bowl, profiled in agentview
		"segment", // the plate, by SAM3
		"move_to", // park over the bowl
		...bp(9), // and read it again from the wrist
		"move_to",
		"pi0_pick", // misses
		"move_to", // its approach, again
		"pi0_pick",
		"set_gripper",
		...bp(9), // the held body
		"move_to",
		"release",
	]);
	assert.deepEqual(queries, [
		"the bowl",
		"the center of the bowl directly below the gripper",
		"the body of the bowl held in the gripper",
	]);
	// Molmo's point (256, 256) of 512 is (512, 512) in the 1024 image; the profile runs down its column.
	assert.deepEqual(out.sent[1][1], { row: 467, col: 512, camera: "agentview", resolution: "high" });
	// Parked at the plan's highest waypoint.
	const moves = out.sent.filter(([n]) => n === "move_to").map(([, a]) => a);
	assert.deepEqual(moves[0], { xyz: [0.13, 0.22, 1.1], gripper: -1, step_clip: 0.02, max_steps: 150, tol: 0.012 });
	// The bowl refined to the wrist reading (0.14, 0.22), plus its recorded offset.
	assert.deepEqual(moves[1], { xyz: [0.19, 0.27, 0.95], gripper: -1 });
	assert.deepEqual(moves[2], moves[1]);
	const pick = out.sent.find(([n]) => n === "pi0_pick")?.[1];
	assert.deepEqual(pick, {
		prompt: "pick up the bowl on the table",
		lift_thresh: 0.04,
		gripper_closed_thresh: 0.07,
		gripper_open_thresh: 0.003,
		descent_thresh: 0.0,
	});
	// The carry targets the plate's live reading plus its offset, less the held body's offset from the gripper.
	const held = [0.14 - 0.0231 - 0.1, 0.22 + 0.0029 - 0.2];
	const carry = moves[3].xyz as number[];
	assert.ok(Math.abs(carry[0] - (-0.07 - held[0])) < 1e-4, `carry x ${carry[0]}`);
	assert.ok(Math.abs(carry[1] - (0.16 - held[1])) < 1e-4, `carry y ${carry[1]}`);
	assert.match(out.notes, /the bowl refined by 0\.010/);
	assert.match(out.notes, /held offset \(0\.0169,0\.0229\)/);
	assert.equal(out.finish.status, "success");
});

test("LIBERO with --molmo-set surveys the agentview and wrist images in one MolmoPoint call, and falls back to the wrist point", async (t) => {
	const calls: { method: string; kwargs: any }[] = [];
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, kwargs } = JSON.parse(body) as { method: string; kwargs: any };
			calls.push({ method, kwargs });
			const result =
				method === "molmo.ground_set"
					? {
							points: [{ image_index: 1, pixel_x: 128, pixel_y: 128 }],
							image_sizes: [
								[512, 512],
								[512, 512],
							],
						}
					: method === "molmo.ground"
						? { point_xy: [256, 256], image_size: [512, 512] }
						: "ok";
			res.end(JSON.stringify({ ok: true, result }));
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	t.after(() => {
		server.closeAllConnections();
		server.close();
	});
	const molmo = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
	const p = libero({ molmo, "molmo-set": true, "flash-plans": liberoPlans() });
	const out = await drive(p, liberoExec([-0.12, 0.11, 0.9]));
	const set = calls.filter((c) => c.method === "molmo.ground_set");
	assert.equal(set.length, 1, "one call for the one Molmo anchor");
	assert.equal(set[0].kwargs.query, "Point to the bowl.");
	assert.equal(set[0].kwargs.images_base64.length, 2);
	// The wrist point (128, 128) of 512 is (256, 256) in the 1024 image, profiled in the wrist view.
	assert.deepEqual(out.sent[1][1], { row: 211, col: 256, camera: "wrist", resolution: "high" });
	assert.match(out.notes, /the bowl located in the wrist image only/);
	// The refinement and the held body still ask molmo.ground from the wrist.
	assert.deepEqual(
		calls.filter((c) => c.method === "molmo.ground").map((c) => c.kwargs.query),
		["the center of the bowl directly below the gripper", "the body of the bowl held in the gripper"],
	);
	assert.equal(out.finish.status, "success");
});

test("LIBERO Flash refuses a LIBERO-plus episode: plan task indices are pro ones", () => {
	const plans = liberoPlans();
	const hook = (liberoType: string) =>
		liberoFlash(fakePi({ molmo: "off", "flash-plans": plans }).pi, () => ({
			suite: "libero_goal_swap",
			task: "3",
			liberoType,
		}));
	assert.throws(() => hook("plus").load("/"), /not LIBERO-plus/);
	assert.equal((hook("pro").load("/") as FlashProgram).name, "goal_swap_t3");
});

test("LIBERO Flash replays a standard suite's plan (libero_goal task 3 is goal_t3) and refuses an unknown suite", () => {
	const plans = liberoPlans();
	for (const suffix of ["plan", "anchors"])
		writeFileSync(join(plans, `goal_t3_${suffix}.json`), readFileSync(join(plans, `goal_swap_t3_${suffix}.json`)));
	const hook = (suite: string) =>
		liberoFlash(fakePi({ molmo: "off", "flash-plans": plans }).pi, () => ({ suite, task: "3", liberoType: "pro" }));
	assert.equal((hook("libero_goal").load("/") as FlashProgram).name, "goal_t3");
	assert.equal((hook("libero_goal_swap").load("/") as FlashProgram).name, "goal_swap_t3");
	assert.throws(() => hook("libero_90").load("/"), /cover libero_\{10,goal,object,spatial\}/);
});

test("LIBERO Flash plans are generated from standard-suite traces as well as LIBERO-Pro ones", () => {
	const dir = mkdtempSync(join(tmpdir(), "flash-gen-"));
	const recipe = `${JSON.stringify({ action: "move_to", xyz: [0.1, 0.2, 1.0] })}\n`;
	for (const [tag, suite, key] of [
		["goal_t3_s7", "libero_goal", "goal_t3"],
		["goal_swap_t3_s7", "libero_goal_swap", "goal_swap_t3"],
		["10_t2_s0", "libero_10", "10_t2"],
	]) {
		writeFileSync(
			join(dir, `${tag}.json`),
			JSON.stringify({
				terminated: true,
				suite,
				task_language: tag.startsWith("10") ? "put the black bowl on the stove" : "put the bowl on the plate",
			}),
		);
		writeFileSync(join(dir, `${tag}_recipe.jsonl`), recipe);
		const out = generateFlashPlan({
			audit: join(dir, `${tag}.json`),
			recipe: join(dir, `${tag}_recipe.jsonl`),
			destination: join(dir, "flash"),
		});
		assert.equal(`${out.family}_${out.key}`, key);
		assert.ok(existsSync(join(dir, "flash", `${key}_plan.json`)), key);
	}
	// The audit's suite must still agree with its file name.
	writeFileSync(
		join(dir, "object_t1_s0.json"),
		JSON.stringify({
			terminated: true,
			suite: "libero_object_swap",
			task_language: "pick up the ketchup and place it in the basket",
		}),
	);
	writeFileSync(join(dir, "object_t1_s0_recipe.jsonl"), recipe);
	assert.throws(
		() =>
			generateFlashPlan({
				audit: join(dir, "object_t1_s0.json"),
				recipe: join(dir, "object_t1_s0_recipe.jsonl"),
				destination: dir,
			}),
		/suite="libero_object_swap" does not match filename \(libero_object\)/,
	);
	writeFileSync(join(dir, "90_t1_s0.json"), "{}");
	assert.throws(
		() =>
			generateFlashPlan({
				audit: join(dir, "90_t1_s0.json"),
				recipe: join(dir, "90_t1_s0_recipe.jsonl"),
				destination: dir,
			}),
		/must be <10\|goal\|object\|spatial>\[_<task\|swap>\]/,
	);
});

test("a malformed model audit is read leniently, else restated by a structured re-ask, else refused with why", () => {
	const dir = mkdtempSync(join(tmpdir(), "flash-audit-"));
	const recipe = join(dir, "goal_t3_s7_recipe.jsonl");
	writeFileSync(recipe, `${JSON.stringify({ action: "move_to", xyz: [0.1, 0.2, 1.0] })}\n`);
	const audit = join(dir, "goal_t3_s7.json");
	// Muse's usual slips: a leading comment, single quotes, a trailing comma.
	writeFileSync(
		audit,
		"// solved on the first try\n{'terminated': true, 'task_language': 'put the bowl on the plate', 'seed': 7,}",
	);
	const notes: string[] = [];
	const out = generateFlashPlan({ audit, recipe, destination: join(dir, "flash"), note: (l) => notes.push(l) });
	assert.equal(out.language, "put the bowl on the plate");
	assert.match(notes.join("\n"), /is not strict JSON; read it after repairing its syntax/);
	// Beyond repair: the re-ask gets the text, the error and exactly the fields, and its reply is used.
	writeFileSync(audit, "Audit: terminated = yes. Task: put the bowl on the plate. {oops");
	const prompts: string[] = [];
	const reask = (prompt: string) => {
		prompts.push(prompt);
		return '```json\n{"terminated": true, "task_language": "put the bowl on the plate", "suite": "libero_goal", "task_id": 3, "seed": null}\n```';
	};
	const doc = readAudit(audit, reask, (l) => notes.push(l));
	assert.deepEqual(doc, {
		terminated: true,
		task_language: "put the bowl on the plate",
		suite: "libero_goal",
		task_id: 3,
	});
	assert.equal(prompts.length, 1);
	assert.match(prompts[0], /does not parse \(not JSON even leniently: /);
	assert.match(prompts[0], /terminated = yes\. Task: put the bowl on the plate/);
	for (const k of Object.keys(AUDIT_FIELDS)) assert.match(prompts[0], new RegExp(`- ${k}: `));
	assert.match(notes.at(-1) ?? "", /used the model's restatement of terminated, task_language, suite, task_id, seed/);
	assert.equal(generateFlashPlan({ audit, recipe, destination: join(dir, "flash"), reask }).key, "t3");
	// A restatement that misses a field, or no re-ask at all, refuses the audit and says why.
	assert.throws(
		() => readAudit(audit, () => '{"terminated": "yes", "task_language": "x"}'),
		/the re-ask gave bad terminated/,
	);
	assert.throws(() => readAudit(audit, () => "I cannot tell."), /the re-ask gave no JSON object/);
	assert.throws(() => readAudit(audit), /not JSON even leniently.*pass --reask-model/);
});
