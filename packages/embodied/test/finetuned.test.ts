import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { createServer, type IncomingMessage } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import finetuned, {
	allowedTokens,
	buildRequest,
	formatPrompt,
	PROMPTS,
	parseToken,
	prepareImages,
	RELEASED,
	recentText,
	STEP_ENTRY,
} from "../src/finetuned/index.ts";
import { convertRun, findRuns } from "../src/finetuned/prepare.ts";
import { decodePng, fingerprint, parseView, prepareView } from "../src/finetuned/views.ts";
import { encodePng } from "../src/png.ts";

/**
 * The reference: Show-Harness @137d571's own MvTokenController + VLMClient.complete_action_token
 * (with _post_chat captured) and core/record/images.prepare_view under opencv 4.11 / numpy 1.26, fed
 * the two synthetic frames below; image parts are replaced by the pixel fingerprint of the PNG sent.
 */
const TASK = "pick up the red cube and place it on the plate";
const REF_PROMPT = `You are controlling a robot arm with two cameras:
- Agentview: overhead view of the robot and workspace
- Wristview: close-up view from the gripper

Task: ${TASK}
Recent moves, newest first: MV_LEFT, MV_DOWN

Output exactly one action token:
MV_FWD, MV_BACK, MV_LEFT, MV_RIGHT, MV_UP, MV_DOWN, GRASP, RELEASE, DONE

Check both camera views and choose the next action:
- Use AgentView to locate the target when it is not in wrist view
- Use wrist view to fine-align when target is visible up close
- GRASP when gripper fingers are aligned around the object
- RELEASE when object is above the destination
- DONE when the task is complete: the object is at its destination and the gripper is clear
- Avoid repeating a direction that conflicts with the most recent move

Return the single token only, no punctuation, no explanation:`;
const REF = {
	raw: {
		agent: { sha1: "b18f71cedaac", shape: [512, 512, 3] },
		wrist: { sha1: "ff8788f8ee4b", shape: [480, 640, 3] },
	},
	payload: {
		model: "qwen3_5_2b_showharness_sim",
		messages: [
			{
				role: "user",
				content: [
					{ type: "image_url", image_url: { url: "<png>", sha1: "6c883980c15b", shape: [256, 256, 3] } },
					{ type: "image_url", image_url: { url: "<png>", sha1: "1e53f10798cd", shape: [256, 256, 3] } },
					{ type: "text", text: REF_PROMPT },
				],
			},
		],
		temperature: 0,
		max_tokens: 24,
		chat_template_kwargs: { enable_thinking: false, thinking: false },
	},
	/** vlm_client._parse_single_token on these replies (null: it raises, and the runner falls back). */
	parsed: {
		MV_LEFT: "MV_LEFT",
		" GRASP\n": "GRASP",
		"<think>hmm MV_UP</think>DONE": "DONE",
		'{"token": "RELEASE"}': "RELEASE",
		"I will MV_FWD.": "MV_FWD",
		"Therefore the answer is MV_BACK because": "MV_BACK",
		"MV_DOWN is best": "MV_DOWN",
		"move left": null,
		MV_LEFTX: null,
		"MV_UP then MV_DOWN": "MV_DOWN",
		mv_left: null,
		// upstream returns "grasp" as written, which no controller executes; we return the canonical unit.
		"final: grasp": "GRASP",
		"The gripper should go MV_RIGHT.": "MV_RIGHT",
	} as Record<string, string | null>,
};

/** The synthetic frame of the reference script: (x*3+y+seed, x*y+seed, x^y) mod 256. */
function pattern(h: number, w: number, seed: number) {
	const rgb = Buffer.alloc(h * w * 3);
	for (let y = 0; y < h; y++)
		for (let x = 0; x < w; x++) {
			const o = (y * w + x) * 3;
			rgb[o] = (x * 3 + y + seed) % 256;
			rgb[o + 1] = (x * y + seed) % 256;
			rgb[o + 2] = (x ^ y) & 255;
		}
	return { width: w, height: h, rgb };
}
const agentRaw = pattern(512, 512, 7);
const wristRaw = pattern(480, 640, 11);
const SPECS = [parseView("square=256"), parseView("rot=90,flip=vertical,crop=1.3333,square=256")];
const png = (img: { width: number; height: number; rgb: Buffer }) => encodePng(img.rgb, img.width, img.height);

/** A request body with its images replaced by the fingerprints of their pixels, as REF stores them. */
function fingerprinted(body: any) {
	const b = structuredClone(body);
	for (const part of b.messages[0].content)
		if (part.type === "image_url") {
			const url: string = part.image_url.url;
			assert.ok(url.startsWith("data:image/png;base64,"));
			part.image_url = { url: "<png>", ...fingerprint(decodePng(Buffer.from(url.split(",")[1], "base64"))) };
		}
	return b;
}

test("the request matches Show-Harness's own rendering: prompt bytes, image order, pixels, sampling", () => {
	assert.deepEqual(fingerprint(agentRaw), REF.raw.agent);
	assert.deepEqual(fingerprint(wristRaw), REF.raw.wrist);
	const tpl = PROMPTS.v3;
	const prompt = formatPrompt(tpl, {
		task: TASK,
		recent_moves: recentText(["MV_LEFT", "MV_DOWN"]),
		gripper_state: "open",
	});
	assert.equal(prompt, REF_PROMPT);
	const sent = prepareImages(
		[png(agentRaw), png(wristRaw)].map((p) => ({ data: p.toString("base64") })),
		[0, 1],
		SPECS,
	);
	const body = buildRequest(
		"qwen3_5_2b_showharness_sim",
		prompt,
		sent.map((s) => s.png),
	);
	assert.deepEqual(fingerprinted(body), REF.payload);
	assert.deepEqual(allowedTokens(tpl), [
		"MV_FWD",
		"MV_BACK",
		"MV_LEFT",
		"MV_RIGHT",
		"MV_UP",
		"MV_DOWN",
		"GRASP",
		"RELEASE",
		"DONE",
	]);
	assert.deepEqual(fingerprint(prepareView(wristRaw, SPECS[1])), fingerprint(sent[1].view));
	assert.equal(formatPrompt("a {{b}} {c}", { c: "d" }), "a {b} d");
	assert.throws(() => formatPrompt("{missing}", {}), /has no value/);
	assert.equal(recentText([]), "none");
});

test("replies parse like vlm_client._parse_single_token", () => {
	const allowed = allowedTokens(PROMPTS.v3);
	for (const [raw, want] of Object.entries(REF.parsed)) {
		if (want === null) assert.throws(() => parseToken(raw, allowed), /invalid token/, raw);
		else assert.equal(parseToken(raw, allowed), want, raw);
	}
});

// ---------------------------------------------------------------------------
// the provider, driven like pi's agent loop, against a local OpenAI-compatible endpoint

type Handler = (event: any, ctx: any) => unknown;

async function endpoint(replies: (string | { status: number; body: string })[]) {
	const bodies: any[] = [];
	const server = createServer(async (req: IncomingMessage, res) => {
		let data = "";
		for await (const chunk of req) data += chunk;
		bodies.push({ path: req.url, auth: req.headers.authorization, body: JSON.parse(data) });
		const r = replies.shift() ?? "DONE";
		if (typeof r !== "string") {
			res.writeHead(r.status, { "content-type": "application/json" });
			res.end(r.body);
			return;
		}
		res.writeHead(200, { "content-type": "application/json" });
		res.end(JSON.stringify({ choices: [{ message: { role: "assistant", content: r } }] }));
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1`;
	return { url, bodies, close: () => new Promise<void>((resolve) => server.close(() => resolve())) };
}

function fakePi(flags: Record<string, string>, robot = "maniskill", tools = ["act", "finish"]) {
	const handlers = new Map<string, Handler[]>();
	const defaults: Record<string, unknown> = {};
	const entries: { type: string; data: any }[] = [];
	let provider: any;
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			defaults[name] = o.default;
		},
		getFlag: (name: string) => (name in flags ? flags[name] : defaults[name]),
		getActiveTools: () => tools,
		registerProvider: (p: any) => {
			provider = p;
		},
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
	} as unknown as ExtensionAPI;
	const warnings: string[] = [];
	const ctx = {
		hasUI: true,
		model: { provider: "finetuned" },
		ui: { notify: (s: string) => warnings.push(s) },
		sessionManager: { getBranch: () => [{ type: "custom", customType: "robot_task", data: { robot } }] },
	};
	finetuned(pi);
	const emit = async (name: string) => {
		for (const fn of handlers.get(name) ?? []) await fn({ type: name }, ctx);
	};
	const model = (id: string) => provider.getModels().find((m: any) => m.id === id);
	const turn = (messages: unknown[], id = "qwen3_5_2b_showharness_sim", signal?: AbortSignal) =>
		provider.streamSimple(model(id), { messages }, { signal }).result();
	return { emit, turn, entries, warnings, model };
}

const calls = (m: any) => m.content.filter((c: any) => c.type === "toolCall");
/** The units `act` result: its header (with the TASK line), then the robot's state and images. */
const observation = (m: any, details: Record<string, unknown> = {}) =>
	calls(m).map((c: any) => ({
		role: "toolResult",
		toolCallId: c.id,
		toolName: c.name,
		content: [
			{ type: "text", text: `units: ${c.arguments.unit} x1\nRecent units: none\nTASK: ${TASK}` },
			{ type: "text", text: "{}" },
			{ type: "image", data: png(agentRaw).toString("base64"), mimeType: "image/png" },
			{ type: "image", data: png(wristRaw).toString("base64"), mimeType: "image/png" },
		],
		details,
		isError: false,
	}));

test("one stateless request per step; the token becomes a units act; DONE finishes", async () => {
	const ep = await endpoint(["MV_DOWN", "MV_LEFT", "no idea", "GRASP", "DONE"]);
	const p = fakePi({ "ft-endpoint": ep.url, "ft-wrist": "rot=90,flip=vertical,crop=1.3333,square=256" });
	try {
		assert.deepEqual(
			[...RELEASED, "local"].map((id) => Boolean(p.model(id))),
			[...RELEASED, "local"].map(() => true),
		);
		await p.emit("session_start");
		await p.emit("before_agent_start");
		assert.deepEqual(p.warnings, []);
		// The opening RELEASE asks no model.
		const t0 = await p.turn([]);
		assert.deepEqual(calls(t0)[0].arguments, { unit: "RELEASE" });
		assert.equal(t0.usage.totalTokens, 0);
		assert.equal(ep.bodies.length, 0);
		const history: unknown[] = [t0, ...observation(t0)];
		const units: string[] = [];
		let m = t0;
		for (let i = 0; i < 6 && calls(m)[0]?.name === "act"; i++) {
			m = await p.turn(history);
			history.push(m, ...observation(m));
			units.push(
				calls(m)
					.map((c: any) => (c.name === "act" ? c.arguments.unit : c.name))
					.join(),
			);
		}
		// "no idea" parses to nothing: the runner's MV_DOWN fallback.
		assert.deepEqual(units, ["MV_DOWN", "MV_LEFT", "MV_DOWN", "GRASP", "finish"]);
		assert.deepEqual(calls(m)[0].arguments.status, "success");
		assert.equal(ep.bodies.length, 5);
		assert.equal(ep.bodies[0].path, "/v1/chat/completions");
		assert.equal(ep.bodies[0].auth, "Bearer EMPTY");
		// Step 2's request is exactly the reference: after MV_DOWN then MV_LEFT, newest first.
		const second = fingerprinted(ep.bodies[2].body);
		assert.deepEqual(second, REF.payload);
		// GRASP / RELEASE never enter the move history; each request is one user message.
		const last = ep.bodies[4].body;
		assert.equal(last.messages.length, 1);
		assert.match(last.messages[0].content[2].text, /Recent moves, newest first: MV_DOWN, MV_LEFT, MV_DOWN\n/);
		const steps = p.entries.filter((e) => e.type === STEP_ENTRY).map((e) => e.data);
		assert.deepEqual(
			steps.map((s) => [s.token, s.fallback]),
			[
				["MV_DOWN", false],
				["MV_LEFT", false],
				["MV_DOWN", true],
				["GRASP", false],
				["DONE", false],
			],
		);
		assert.equal(steps[2].raw, "no idea");
		assert.deepEqual(
			steps[2].media.map((x: any) => [x.camera, x.sha1]),
			[
				["agentview", "6c883980c15b"],
				["wrist", "1e53f10798cd"],
			],
		);
		// pi stops after finish; asked again, the policy is spent.
		assert.equal(calls(await p.turn(history)).length, 0);
	} finally {
		await ep.close();
	}
});

test("the execution swap changes only the executed unit; env success, max steps and errors end the episode", async () => {
	const ep = await endpoint(["MV_FWD", "MV_FWD"]);
	const p = fakePi({ "ft-endpoint": ep.url, "ft-swap": "MV_FWD,MV_BACK", "ft-max-steps": "1" }, "piper");
	try {
		await p.emit("session_start");
		await p.emit("before_agent_start");
		const t0 = await p.turn([]);
		const h: unknown[] = [t0, ...observation(t0)];
		const t1 = await p.turn(h);
		assert.deepEqual(calls(t1)[0].arguments, { unit: "MV_BACK" });
		h.push(t1, ...observation(t1));
		const t2 = await p.turn(h);
		assert.deepEqual(calls(t2)[0].arguments.status, "failure");
		assert.match(calls(t2)[0].arguments.summary, /max_steps \(1\)/);
		const step = p.entries.find((e) => e.type === STEP_ENTRY)?.data;
		assert.deepEqual([step.token, step.executed], ["MV_FWD", "MV_BACK"]);
	} finally {
		await ep.close();
	}

	const q = fakePi({ "ft-endpoint": "http://127.0.0.1:9/v1" }, "maniskill");
	await q.emit("session_start");
	const t0 = await q.turn([]);
	const done = await q.turn([t0, ...observation(t0, { terminated: true, success: true })]);
	assert.deepEqual(calls(done)[0].arguments.status, "success");
	assert.match(calls(done)[0].arguments.summary, /environment reports success/);

	// A non-retryable HTTP error is a model error, not a fallback unit.
	const bad = await endpoint([{ status: 400, body: '{"error":"no such adapter"}' }]);
	const r = fakePi({ "ft-endpoint": bad.url }, "robolab", ["finish"]);
	try {
		await r.emit("session_start");
		await r.emit("before_agent_start");
		assert.match(r.warnings.join("\n"), /--units/);
		const s0 = await r.turn([]);
		const err = await r.turn([s0, ...observation(s0)]);
		assert.equal(err.stopReason, "error");
		assert.match(err.errorMessage, /HTTP 400/);
		// An aborted turn ends the policy.
		const ac = new AbortController();
		ac.abort();
		assert.equal((await r.turn([], "qwen3_5_2b_showharness_sim", ac.signal)).stopReason, "aborted");
		// finetuned/local needs an adapter name.
		await r.emit("session_start");
		const l0 = await r.turn([], "local");
		assert.match((await r.turn([l0, ...observation(l0)], "local")).errorMessage, /--ft-model/);
	} finally {
		await bad.close();
	}
});

test("prepare turns a GUMI run into a Show-Harness rollout with the provider's camera transform", () => {
	const run = mkdtempSync(join(tmpdir(), "gumi-"));
	mkdirSync(join(run, "images", "agentview"), { recursive: true });
	mkdirSync(join(run, "images", "wrist"), { recursive: true });
	writeFileSync(join(run, "images", "agentview", "0000.png"), png(agentRaw));
	writeFileSync(join(run, "images", "wrist", "0000.png"), png(wristRaw));
	const rows = [
		{ step: 0, token: "MV_DOWN", gripper_closed: false, agentview: "images/agentview/0000.png", src: "agent" },
		{
			step: 1,
			token: "GRASP",
			gripper_closed: true,
			agentview: "images/agentview/0000.png",
			wrist: "images/wrist/0000.png",
		},
	];
	writeFileSync(join(run, "actions.jsonl"), `${rows.map((r) => JSON.stringify(r)).join("\n")}\n`);
	const out = mkdtempSync(join(tmpdir(), "rollouts-"));
	const views = { agentview: SPECS[0], wrist: SPECS[1] };
	const r = convertRun(run, join(out, "t"), 0, views, TASK, { robot: "libero" });
	assert.equal(r.rows, 1);
	assert.match(r.warnings.join(), /dropped steps 0/);
	const line = JSON.parse(readFileSync(join(r.out, "actions.jsonl"), "utf8").trim());
	assert.deepEqual(
		[line.token, line.gripper_closed, line.agentview, line.wrist],
		["GRASP", true, "agentview/0000.png", "wrist/0000.png"],
	);
	// Byte-identical to what the provider sends for the same frames (the REF fingerprints).
	assert.equal(fingerprint(decodePng(readFileSync(join(r.out, line.agentview)))).sha1, "6c883980c15b");
	assert.equal(fingerprint(decodePng(readFileSync(join(r.out, line.wrist)))).sha1, "1e53f10798cd");
	const meta = JSON.parse(readFileSync(join(r.out, "metadata.json"), "utf8"));
	assert.equal(meta.task_text, TASK);
	assert.deepEqual(findRuns(run), [run]);
});
