import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { encodePng } from "../src/infra/png.ts";
import { RESULT_ENTRY, toolSections } from "../src/robot.ts";
import { generateFlashPlan } from "../src/robots/libero/flash-generate.ts";
import libero, { GUIDES, packDepth, renderRpent, unpackDepth } from "../src/robots/libero/index.ts";
import { codeApiReply } from "./helpers/code-api.ts";
import { deployFlags } from "./helpers/deployment.ts";

type Handler = (event: any, ctx: any) => unknown;
const src = (path: string) => readFileSync(new URL(`../src/robots/libero/${path}`, import.meta.url), "utf8");
const FIXTURE = JSON.parse(readFileSync(new URL("./fixtures/libero-rpent-prompt.json", import.meta.url), "utf8"));

/** A stub pi that runs handlers in registration order; like pi, it activates only registered tools. */
function stubPi(values: Record<string, unknown>, cwd: string) {
	values = deployFlags(values);
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	let active: string[] = [];
	const api: Record<string, unknown> = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		setActiveTools: (names: string[]) => {
			active = names.filter((n) => tools.has(n) || ["read", "ls", "grep", "find", "write"].includes(n));
		},
		getActiveTools: () => active,
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: { emit: () => {}, on: () => () => {} },
	};
	const pi = new Proxy(api, { get: (t, k: string) => t[k] ?? (() => {}) }) as unknown as ExtensionAPI;
	const ctx = {
		hasUI: true,
		cwd,
		ui: { notify: () => {} },
		shutdown: () => {},
		abort: () => {},
		sessionManager: {
			getBranch: () => [],
			getSessionDir: () => (values.noSessionDir ? "" : cwd),
			getSessionFile: () => undefined,
			getSessionId: () => "s",
		},
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) result = (await fn({ type: name, ...event }, ctx)) ?? result;
		return result;
	}
	const run = async (name: string, params: unknown) =>
		(await tools.get(name).execute("id", params, undefined, undefined, ctx)) as any;
	return { pi, flags, tools, entries, emit, run, active: () => active };
}

const nd = (dtype: string, shape: number[], data: Buffer) => ({ __ndarray__: data.toString("base64"), dtype, shape });
const f32 = (v: number[], shape = [v.length]) => nd("float32", shape, Buffer.from(Float32Array.from(v).buffer));

const stop = (server: Server) => () => {
	server.closeAllConnections();
	server.close();
};

async function listen(server: Server) {
	// The client keeps connections alive; a server closing an idle one (Node's 5 s default) while the
	// next call reuses it is an ECONNRESET, and 1024 renders under a loaded test run leave such gaps.
	server.keepAliveTimeout = 120_000;
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
}

/**
 * A fake LIBERO env server: the eef moves by action * 0.05 per step, and every camera sees a flat
 * surface whose depth grows 0.1 m per env step, so a past state's world map differs from the current one.
 */
async function fakeEnv() {
	const s = { steps: 0, eef: [0, 0, 1] };
	const calls: string[] = [];
	const kwargsOf: Record<string, Record<string, unknown>> = {};
	const obs = () => ({
		main_images: nd("uint8", [4, 4, 3], Buffer.alloc(48)),
		wrist_images: nd("uint8", [4, 4, 3], Buffer.alloc(48)),
		states: f32([...s.eef, 0, 0, 0, 0.02, -0.02]),
	});
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, args = [], kwargs = {} } = JSON.parse(body);
			calls.push(method);
			kwargsOf[method] = kwargs;
			let result: unknown = { status: "ok" };
			if (method === "code.api") result = codeApiReply("libero", kwargs.tier);
			else if (method === "env.reset") {
				s.steps = 0;
				s.eef = [0, 0, 1];
				result = [obs(), {}];
			} else if (method === "env.get_task_language") result = "put the black bowl on the stove";
			else if (method === "env.raw_obs")
				result = {
					robot0_eef_pos: f32(s.eef),
					robot0_eef_quat: f32([1, 0, 0, 0]),
					robot0_gripper_qpos: f32([0.02, -0.02]),
					akita_black_bowl_1_pos: f32([0.1, 0.1, 0.9]),
				};
			else if (method === "env.render_camera") {
				const n = kwargs.height as number;
				const rgb = Buffer.alloc(n * n * 3, 40 + s.steps);
				result = kwargs.depth
					? [nd("uint8", [n, n, 3], rgb), f32(new Array(n * n).fill(1 + 0.1 * s.steps), [n, n])]
					: nd("uint8", [n, n, 3], rgb);
			} else if (method === "env.get_camera_meta") {
				const n = kwargs.height as number;
				result = {
					intrinsic_K: [
						[n / 2, 0, n / 2],
						[0, n / 2, n / 2],
						[0, 0, 1],
					],
					extrinsic_cam2world: [
						[1, 0, 0, 0],
						[0, 1, 0, 0],
						[0, 0, 1, 0],
						[0, 0, 0, 1],
					],
				};
			} else if (method === "env.step") {
				const a = (args[0] as { __ndarray__: string }).__ndarray__;
				const act = Array.from(new Float32Array(Uint8Array.from(Buffer.from(a, "base64")).buffer));
				s.eef = s.eef.map((v, i) => v + act[i] * 0.05);
				s.steps++;
				result = [obs(), 0, false, false, {}];
			} else if (method === "env.move_to") {
				// The server's servo: one env step per 2.5 cm, each returned for pi's video and recorder.
				const target = kwargs.xyz as number[];
				const transitions = [];
				while (Math.hypot(...target.map((v, i) => v - s.eef[i])) > 0.012 && transitions.length < 80) {
					const act = target.map((v, i) => Math.max(-0.5, Math.min(0.5, (v - s.eef[i]) / 0.05)));
					s.eef = s.eef.map((v, i) => v + act[i] * 0.05);
					s.steps++;
					transitions.push({
						action: f32([...act, 0, 0, 0, -1]),
						obs: obs(),
						reward: 0,
						terminated: false,
						truncated: false,
					});
				}
				result = {
					name: "move_to",
					eef_pos: s.eef,
					final_dist_m: 0,
					steps_used: transitions.length,
					...(kwargs.tool_call ? { transitions } : {}),
				};
			}
			res.end(JSON.stringify({ ok: true, result }));
		});
	});
	return { url: await listen(server), calls, kwargsOf, state: s, close: stop(server) };
}

/** A fake SAM3 server whose mask is the 100x100 square at rows/cols 400-500. */
async function fakeSam3() {
	const mask = Buffer.alloc(1024 * 1024 * 3);
	for (let r = 400; r < 500; r++) mask.fill(255, (r * 1024 + 400) * 3, (r * 1024 + 500) * 3);
	const png = encodePng(mask, 1024, 1024).toString("base64");
	const server = createServer((req, res) => {
		req.resume();
		req.on("end", () =>
			res.end(
				JSON.stringify({
					ok: true,
					result: { found: true, score: 0.8, box: [400, 400, 500, 500], mask_png_base64: png },
				}),
			),
		);
	});
	return { url: await listen(server), close: stop(server) };
}

/** A LIBERO session on the fake servers with a local memory corpus; `values` add flags. */
async function session(t: { after: (fn: () => void) => void }, values: Record<string, unknown> = {}) {
	const [env, sam3] = [await fakeEnv(), await fakeSam3()];
	t.after(env.close);
	t.after(sam3.close);
	const root = mkdtempSync(join(tmpdir(), "libero-prompt-"));
	const [memoryDir, out] = [join(root, "memory"), join(root, "out")];
	mkdirSync(memoryDir);
	writeFileSync(join(memoryDir, "MEMORY.md"), "# index\n");
	const s = stubPi(
		{
			"env-url": env.url,
			sam3: sam3.url,
			// Any server that answers healthz stands in for Pi0.5: the RPent prompt describes the pi0 tools.
			vla: sam3.url,
			suite: "libero_10_task",
			task: "2",
			"memory-profile": "local",
			"memory-dir": memoryDir,
			"output-dir": out,
			...values,
		},
		root,
	);
	libero(s.pi);
	await s.emit("session_start");
	return { ...s, env, out, memoryDir, steps: join(out, "10_task_t2_s0_steps") };
}

const text = (r: any) => JSON.parse(r.content[0].text);
const phrases = (group: Record<string, string[]>) => Object.entries(group).flatMap(([k, v]) => v.map((p) => [k, p]));
function assertPhrases(prompt: string, ...groups: Record<string, string[]>[]) {
	for (const [section, phrase] of groups.flatMap(phrases))
		assert.ok(prompt.includes(phrase), `${section}: missing ${JSON.stringify(phrase)}`);
}
/** Every tool a LIBERO session can mount, as the prompts' `[tool:x]` blocks name them. */
const ALL = [
	...new Set(
		src("SYSTEM.md")
			.concat(src("explore.md"))
			.match(/(?<=\[tool:)[\w|]+/g)!
			.flatMap((m) => m.split("|")),
	),
	// RPent's prompts are the perception-isolated ones: --privileged's tool is not among them.
].filter((n) => n !== "ground_truth_poses");

test("the RPent evaluate prompt keeps every original section's key rules, for both memory profiles", () => {
	for (const profile of ["hf", "local"]) {
		const prompt = toolSections(renderRpent(src("SYSTEM.md"), profile), ALL);
		assertPhrases(prompt, FIXTURE.evaluate, profile === "hf" ? FIXTURE.evaluate_hf : FIXTURE.evaluate_local);
		assert.doesNotMatch(prompt, /\[(\/)?(part|include|memory|tool):|^#\. /m);
		// The workflow is numbered in order: nine steps with the HF profile, eight with the local one.
		const steps = [...prompt.matchAll(/^(\d+)\. [A-Z]{3,}/gm)].map((m) => Number(m[1]));
		assert.deepEqual(
			steps.slice(-(profile === "hf" ? 9 : 8)),
			[...Array(profile === "hf" ? 9 : 8).keys()].map((i) => i + 1),
		);
	}
	assert.doesNotMatch(renderRpent(src("SYSTEM.md"), "hf"), /LOCAL SUITE \+ TASK \+ GLOBAL/);
	assert.doesNotMatch(renderRpent(src("SYSTEM.md"), "local"), /READ MEMORY FIRST — a general skill library/);
});

test("the RPent explore prompt and DISTIL keep theirs; explore takes evaluate's shared sections", () => {
	const prompt = toolSections(renderRpent(src("explore.md"), "local"), [...ALL, "reset"]);
	assertPhrases(prompt, FIXTURE.explore);
	assert.doesNotMatch(prompt, /SINGLE ATTEMPT, NO RESET|READ EACH AVAILABLE LOCAL MEMORY LAYER/);
	assert.doesNotMatch(prompt, /\[(\/)?(part|include|memory|tool):/);
	assertPhrases(toolSections(src("distil.md"), ALL), FIXTURE.distil);
	assert.throws(() => renderRpent("[include:nope]\n", "local"), /names no \[part:nope\]/);
});

test("without Pi0 the RPent prompt falls back to tool-neutral wording and never names it", () => {
	const tools = ALL.filter((n) => !n.startsWith("pi0"));
	for (const [file, profile] of [
		["SYSTEM.md", "hf"],
		["explore.md", "local"],
	]) {
		const prompt = toolSections(renderRpent(src(file), profile), tools);
		assert.doesNotMatch(prompt, /pi0|Pi0/);
		assert.match(prompt, /grasp policy never places|re-grasp a missed grasp/);
	}
	assert.equal(toolSections("[tool:x]A[/tool:x][tool:!x]B[/tool:!x]", ["x"]), "A");
	assert.equal(toolSections("[tool:x]A[/tool:x][tool:!x]B[/tool:!x]", []), "B");
});

test("--libero-prompt defaults to rpent: the session's system prompt is the filled RPent prompt, and the result records it", async (t) => {
	const s = await session(t);
	assert.equal(s.flags["libero-prompt"], "rpent");
	const { systemPrompt } = await s.emit("before_agent_start", { systemPrompt: "base" });
	assertPhrases(systemPrompt, FIXTURE.evaluate, FIXTURE.evaluate_local);
	// Its WORKFLOW step is the only memory-reading instruction: memory's generic section is not added.
	assert.doesNotMatch(systemPrompt, /Reading memory is a required step|# Memory\nCell/);
	assert.equal(systemPrompt.split("READ EACH AVAILABLE LOCAL MEMORY LAYER FIRST").length, 2);
	assert.doesNotMatch(systemPrompt, /\{\{\w+\}\}/);
	assert.ok(systemPrompt.includes(`${GUIDES}/strict_hybrid_guide.md`));
	assert.ok(systemPrompt.includes(`${s.memoryDir}/suite/suite_libero10_<regime>_t2.md`));
	assert.match(systemPrompt, /- suite: {6}libero_10_task\n- task: {7}2\n- seed: {7}0/);
	// The guides are readable with the file tools; the state history is served by the tools only.
	const guard = async (path: string, toolName = "read") =>
		(await s.emit("tool_call", { toolName, input: { path } }))?.block ?? false;
	assert.equal(await guard(join(GUIDES, "env_calibration.md")), false);
	assert.equal(await guard(join(GUIDES, "x.md"), "write"), true);
	assert.equal(await guard(join(s.steps, "step_000", "state.json")), true);
	await s.emit("agent_start");
	await s.emit("session_shutdown");
	assert.equal(s.entries.find((e) => e.type === RESULT_ENTRY)?.data.libero_prompt, "rpent");
});

test("--libero-prompt compact keeps the short prompt; an unknown variant fails the start", async (t) => {
	const s = await session(t, { "libero-prompt": "compact" });
	const { systemPrompt } = await s.emit("before_agent_start", { systemPrompt: "base" });
	assert.match(systemPrompt, /^You control a Franka arm in the LIBERO simulator/);
	assert.match(systemPrompt, /Task: put the black bowl on the stove/);
	assert.doesNotMatch(systemPrompt, /PROVEN LEVERS/);
	await s.emit("agent_start");
	await s.emit("session_shutdown");
	assert.equal(s.entries.find((e) => e.type === RESULT_ENTRY)?.data.libero_prompt, "compact");
	const bad = await session(t, { "libero-prompt": "long" });
	assert.deepEqual(bad.active(), []);
});

test("--explore with the RPent prompt: the robot's own prompt is empty and exploration's is RPent's explore", async (t) => {
	const s = await session(t, { explore: true });
	const { systemPrompt } = await s.emit("before_agent_start", { systemPrompt: "" });
	assertPhrases(systemPrompt, FIXTURE.explore);
	assert.match(systemPrompt, /YOU ARE AGENT 1 OF UP TO 3 ON THIS CELL/);
	assert.doesNotMatch(systemPrompt, /\{\{\w+\}\}|SINGLE ATTEMPT, NO RESET/);
});

test("state history: every state is recorded; view_env_state, back_project and segment look back by step", async (t) => {
	const s = await session(t);
	let d = text(await s.run("view_env_state", {}));
	assert.equal(d.state_step, 0);
	assert.equal(d.step, 0);
	// A read-only look at the same state records nothing new.
	assert.equal(text(await s.run("view_env_state", {})).state_step, 0);
	const before = text(await s.run("back_project", { row: 512, col: 512 }));
	assert.deepEqual(before.world_xyz, [0, 0, 1]);
	assert.equal(before.step, 0);
	d = text(await s.run("move_to", { xyz: [0.1, 0, 1] }));
	assert.equal(d.state_step, 1);
	assert.ok(d.step > 0);
	assert.deepEqual(readdirSync(s.steps).sort(), ["segments", "step_000", "step_001"]);
	assert.deepEqual(readdirSync(join(s.steps, "step_000")).sort(), [
		"agentview_depth_high.u16.gz",
		"agentview_high.png",
		"agentview_meta.json",
		"state.json",
		"wrist_depth_high.u16.gz",
		"wrist_high.png",
		"wrist_meta.json",
	]);
	// The current map has moved on; step 0's is the one the pixel was picked in.
	const now = text(await s.run("back_project", { row: 512, col: 512 }));
	assert.ok(now.world_xyz[2] > 1.1);
	const then = text(await s.run("back_project", { row: 512, col: 512, step: 0 }));
	assert.deepEqual(then.world_xyz, [0, 0, 1]);
	assert.equal(then.step, 0);
	await assert.rejects(s.run("back_project", { row: 1, col: 1, step: 0, resolution: "low" }), /keeps only the 1024/);
	const old = await s.run("view_env_state", { step: 0 });
	const o = text(old);
	assert.equal(o.state_step, 0);
	assert.equal(o.latest_state_step, 1);
	assert.equal(old.content.filter((c: any) => c.type === "image").length, 2);
	assert.equal(text(await s.run("view_camera_meta", { step: 0 })).step, 0);
	await assert.rejects(s.run("view_env_state", { step: 7 }), /step 7 is not recorded \(have 0\.\.1\)/);
	// A segment reading of step 0 is saved as segment_01.json with its overlay in step 0's record.
	const seg = text(await s.run("segment", { prompt: "the black bowl", step: 0 }));
	assert.equal(seg.step, 0);
	assert.equal(seg.segment_artifact, "segment_01.json");
	assert.deepEqual(seg.world_xyz, [-0.1221, -0.1221, 1]);
	const saved = JSON.parse(readFileSync(join(s.steps, "segments", "segment_01.json"), "utf8"));
	assert.equal(saved.mode, "text");
	assert.equal(saved.prompt, "the black bowl");
	assert.equal(saved.found, true);
	assert.ok(existsSync(join(s.steps, "step_000", "segment_overlay_01.png")));
	assert.equal(text(await s.run("segment", { point: [450, 450] })).segment_artifact, "segment_02.json");
	// flash-generate finds the readings next to the run's audit and anchors on them.
	const audit = join(s.out, "10_task_t2_s0.json");
	writeFileSync(audit, JSON.stringify({ terminated: true, task_language: "put the black bowl on the stove" }));
	writeFileSync(
		join(s.out, "10_task_t2_s0_recipe.jsonl"),
		`${JSON.stringify({ action: "move_to", xyz: [-0.05, -0.05, 1.1] })}\n`,
	);
	generateFlashPlan({ audit, recipe: join(s.out, "10_task_t2_s0_recipe.jsonl"), destination: join(s.out, "flash") });
	const anchors = JSON.parse(readFileSync(join(s.out, "flash", "10_task_t2_anchors.json"), "utf8")).anchors;
	assert.equal(anchors[0].phrase, "the black bowl");
	assert.equal(anchors[0].locator, "segment");
	// --step-history clean (default): the session's end removes the records and keeps the anchors.
	await s.emit("session_shutdown");
	assert.deepEqual(readdirSync(s.steps), ["segments"]);
	assert.equal(readdirSync(join(s.steps, "segments")).length, 2);
});

test("--step-history keep keeps an earlier episode beside the new one; off records nothing and refuses look-back", async (t) => {
	const k = await session(t, { "step-history": "keep" });
	await k.run("view_env_state", {});
	await k.run("move_to", { xyz: [0.1, 0, 1] });
	await k.emit("session_start");
	assert.deepEqual(readdirSync(k.steps), ["segments"]);
	assert.deepEqual(readdirSync(`${k.steps}.1`).sort(), ["segments", "step_000", "step_001"]);
	await k.emit("session_shutdown");
	assert.deepEqual(readdirSync(`${k.steps}.1`).sort(), ["segments", "step_000", "step_001"]);
	const o = await session(t, { "step-history": "off" });
	assert.equal(text(await o.run("view_env_state", {})).state_step, 0);
	await o.run("move_to", { xyz: [0.1, 0, 1] });
	assert.equal(existsSync(o.steps), false);
	await assert.rejects(o.run("back_project", { row: 1, col: 1, step: 0 }), /--step-history off/);
	assert.equal(text(await o.run("segment", { prompt: "the black bowl" })).segment_artifact, undefined);
	assert.deepEqual((await session(t, { "step-history": "all" })).active(), []);
});

test("without an output dir the history is a temp dir, removed at the session's end", async (t) => {
	const s = await session(t, { "output-dir": "", noSessionDir: true });
	await s.run("view_env_state", {});
	const dir = join(tmpdir(), `pi-embodied-libero-${process.pid}`);
	assert.ok(existsSync(dir));
	await s.emit("session_shutdown");
	assert.equal(existsSync(dir), false);
});

test("LIBERO's audit is a write_audit call with the cell and the latest state filled in, and Flash plans read it", async (t) => {
	const s = await session(t);
	assert.ok(s.active().includes("write_audit"));
	await s.run("view_env_state", {});
	await s.run("write_audit", {
		terminated: false,
		strategy_notes: "looked, did not move",
		memory_files_read: [],
	});
	const audit = JSON.parse(readFileSync(join(s.out, "10_task_t2_s0.json"), "utf8"));
	assert.deepEqual(
		[audit.suite, audit.task_id, audit.seed, audit.regime, audit.libero_terminated, audit.terminated],
		["libero_10_task", 2, 0, "strict_perception", false, false],
	);
	// Bug 42: the audit carries the episode's task text, so flash-generate needs no --language.
	assert.equal(audit.task_language, "put the black bowl on the stove");
	assert.ok(Array.isArray(audit.final_state.robot0_eef_pos), "final_state is the latest state");
	// Flash plan generation reads it as written (strict JSON, the suite matching the file name).
	await s.run("write_audit", { terminated: true, strategy_notes: "solved", memory_files_read: [] });
	writeFileSync(
		join(s.out, "10_task_t2_s0_recipe.jsonl"),
		`${JSON.stringify({ action: "move_to", xyz: [0, 0, 1.1] })}\n`,
	);
	const plan = generateFlashPlan({
		audit: join(s.out, "10_task_t2_s0.json"),
		recipe: join(s.out, "10_task_t2_s0_recipe.jsonl"),
		destination: join(s.out, "flash"),
	});
	assert.equal(`${plan.family}_${plan.key}`, "10_task_t2");
});

test("persisted depth round-trips within 0.1 mm; missing depth stays missing", () => {
	const depth = Float32Array.from([0.5, 1.23456, 3, Number.NaN, -1]);
	const back = unpackDepth(packDepth(depth));
	for (const i of [0, 1, 2]) assert.ok(Math.abs(back[i] - depth[i]) <= 5e-5);
	assert.ok(Number.isNaN(back[3]) && Number.isNaN(back[4]));
});

test("the motion tools run the env server's methods with the manifest's parameters; CaP-X's functions are tools too", async (t) => {
	const s = await session(t);
	const props = (name: string) => s.tools.get(name).parameters.properties;
	// manifests/libero.json: the TS tools' parameters and defaults, now the server method's.
	for (const k of ["xyz", "gripper", "tol", "step_clip", "max_steps", "action_scale", "target_yaw", "yaw_step_clip"])
		assert.ok(k in props("move_to"), `move_to.${k}`);
	assert.deepEqual(s.tools.get("move_to").parameters.required, ["xyz"]);
	assert.equal(props("set_gripper").gripper.minimum, -1);
	assert.ok("step" in props("back_project") && "step" in props("segment"), "the tools look back by step");
	for (const name of [
		"get_object_pose",
		"sample_grasp_pose",
		"goto_pose",
		"home_pose",
		"open_gripper",
		"close_gripper",
	]) {
		assert.ok(s.tools.has(name), name);
		assert.ok(!s.active().includes(name), `${name} is the high tier's (not a default tool)`);
	}
	assert.ok(!s.active().includes("preview_reach"), "requires --ik");
	const r = text(await s.run("set_gripper", { gripper: 1, steps: 8 }));
	assert.deepEqual(s.env.kwargsOf["env.set_gripper"], { gripper: 1, steps: 8, tool_call: true });
	assert.equal(r.result.name, "set_gripper");
	await s.run("rotate_pitch", { delta_pitch: 0.2 });
	assert.deepEqual(s.env.kwargsOf["env.rotate_pitch"], { delta_pitch: 0.2, tool_call: true });
	assert.ok(!s.env.calls.includes("env.step"), "no servo loop steps the env from pi");
	// The server's steps are pi's episode steps: move_to returns them and the state step advances.
	const before = text(await s.run("view_env_state", {})).step;
	const moved = text(await s.run("move_to", { xyz: [0.05, 0, 1] }));
	assert.ok(moved.step > before);
	assert.equal(moved.result.steps_used, moved.step - before);
	assert.deepEqual(moved.result.final_eef_pos, [0.05, 0, 1]);
});

test("Pi0.5 is optional: without it the pi0 tools stay inactive and the result notes it; --require-skills pi0 refuses", async (t) => {
	const s = await session(t, { vla: "off" });
	assert.ok(s.active().includes("move_to"), "the robot started");
	assert.ok(!s.active().includes("pi0_pick") && !s.active().includes("pi0_doubled"));
	assert.doesNotMatch((await s.emit("before_agent_start", { systemPrompt: "" })).systemPrompt, /`pi0_pick`/);
	await s.emit("agent_start");
	await s.run("finish", { status: "failure", summary: "x" });
	await s.emit("agent_end", { messages: [] });
	const result = s.entries.find((e) => e.type === "robot_result")?.data;
	assert.deepEqual(result?.skills_off, { pi0: "switched off" });

	const r = await session(t, { vla: "http://127.0.0.1:9", "require-skills": "pi0" });
	process.exitCode = undefined;
	assert.deepEqual(r.active(), [], "a run that needs Pi0.5 does not start without it");
});
