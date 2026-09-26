import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { generateFlashPlan } from "../src/libero/flash-generate.ts";
import libero, { GUIDES, packDepth, renderRpent, unpackDepth } from "../src/libero/index.ts";
import { encodePng } from "../src/png.ts";
import { RESULT_ENTRY, toolSections } from "../src/robot.ts";

type Handler = (event: any, ctx: any) => unknown;
const src = (path: string) => readFileSync(new URL(`../src/libero/${path}`, import.meta.url), "utf8");
const FIXTURE = JSON.parse(readFileSync(new URL("./fixtures/libero-rpent-prompt.json", import.meta.url), "utf8"));

/** A stub pi that runs handlers in registration order; like pi, it activates only registered tools. */
function stubPi(values: Record<string, unknown>, cwd: string) {
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
			getSessionDir: () => cwd,
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
			let result: unknown = { status: "ok" };
			if (method === "code.api") result = { tier: null, primitives: [], digest: "d" };
			else if (method === "env.reset") {
				s.steps = 0;
				s.eef = [0, 0, 1];
				result = [obs(), {}];
			} else if (method === "env.get_task_language") result = "put the black bowl on the plate";
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
			}
			res.end(JSON.stringify({ ok: true, result }));
		});
	});
	return { url: await listen(server), calls, state: s, close: stop(server) };
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
			env: env.url,
			sam3: sam3.url,
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
];

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
	assert.match(systemPrompt, /Task: put the black bowl on the plate/);
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
	// A new episode starts a fresh history; the earlier one is kept beside it.
	await s.emit("session_start");
	assert.deepEqual(readdirSync(s.steps), ["segments"]);
	assert.deepEqual(readdirSync(`${s.steps}.1`).sort(), ["segments", "step_000", "step_001"]);
});

test("persisted depth round-trips within 0.1 mm; missing depth stays missing", () => {
	const depth = Float32Array.from([0.5, 1.23456, 3, Number.NaN, -1]);
	const back = unpackDepth(packDepth(depth));
	for (const i of [0, 1, 2]) assert.ok(Math.abs(back[i] - depth[i]) <= 5e-5);
	assert.ok(Number.isNaN(back[3]) && Number.isNaN(back[4]));
});
