/**
 * HumanCLAW: the action mapping, clamps and prompts against golden texts that HumanCLAW's own Python
 * produced (services/tests/humanclaw_golden.py), the units round trip, and the PSV planner's
 * retries, verifier replacement and history on scripted replies.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
	chooserAction,
	pyFixed,
	pyRound,
	skillCall,
	skillToText,
	toJson,
	UNITS,
	unitOf,
} from "../src/robots/humanclaw/actions.ts";
import { psvBase } from "../src/robots/humanclaw/provider.ts";
import { historyItem, PsvPlanner, parseJsonLoose } from "../src/robots/humanclaw/psv.ts";

type Json = Record<string, any>;
const golden = JSON.parse(readFileSync(new URL("./fixtures/humanclaw-golden.json", import.meta.url), "utf8")) as {
	chooser: Json[];
	episodes: Json[];
};

test("chooserAction reproduces _chooser_action on every golden case (clamps, rounding, recovery)", () => {
	for (const c of golden.chooser) {
		const a = chooserAction(c.input);
		assert.deepEqual(toJson(a), c.action, JSON.stringify(c.input));
		assert.equal(skillToText(a), c.text);
	}
});

test("Python rounding: round-half-even and %.2f on exact binary ties", () => {
	assert.equal(pyRound(36.5), 36);
	assert.equal(pyRound(37.5), 38);
	assert.equal(pyFixed(0.625, 2), "0.62");
	assert.equal(pyFixed(0.375, 2), "0.38");
	assert.equal(pyFixed(0.3, 2), "0.30");
});

test("a unit and its parameter run the SkillCall HumanCLAW would run, and back", () => {
	for (const c of golden.chooser) {
		const a = chooserAction(c.input);
		const { unit, param } = unitOf(a);
		assert.ok(UNITS.some((u) => u.name === unit));
		assert.deepEqual(toJson(skillCall(unit, param)), toJson(a), JSON.stringify(c.input));
	}
	assert.deepEqual(skillCall("TURN_RIGHT", 45).cond, -45);
	assert.deepEqual(skillCall("SIDE_LEFT", 0.25).cond, 0.25);
	assert.deepEqual(skillCall("STEP_BACK", 0.3).cond, [0, -0.3]);
	assert.equal(skillCall("STOP", undefined).skill, "stand");
	assert.ok(UNITS.find((u) => u.name === "STOP")?.terminal);
});

for (const ep of golden.episodes)
	test(`PSV planner replays golden episode ${ep.name}: prompts, retries, verifier, history`, async () => {
		const replies: unknown[] = ep.replies.flat();
		const prompts: string[] = [];
		const planner = new PsvPlanner(
			async (prompt) => {
				prompts.push(prompt);
				const r = replies.shift();
				if (r && typeof r === "object") throw new Error((r as Json).raise);
				return { text: r as string, usage: { prompt_tokens: 1, completion_tokens: 1 } };
			},
			async () => {},
		);
		planner.reset(ep.instruction);
		const history: Json[] = [];
		for (const [i, s] of (ep.steps as Json[]).entries()) {
			const before = prompts.length;
			const d = await planner.act(history);
			assert.deepEqual(prompts.slice(before), s.prompts, `step ${i} prompts`);
			assert.deepEqual(toJson(d.action), s.action, `step ${i} action`);
			assert.equal(skillToText(d.action), s.action_text);
			assert.deepEqual(d.raw_plan, s.raw_plan, `step ${i} raw_plan`);
			assert.deepEqual(d.verifier, s.verifier, `step ${i} verifier`);
			assert.deepEqual(planner.state(), s.planner_state, `step ${i} state`);
			assert.deepEqual(
				d.stages.map((x) => x.raw),
				s.stages.map((x: Json) => x.raw),
			);
			const item = historyItem(i, d);
			assert.deepEqual(item, s.history_item, `step ${i} history`);
			history.push(item);
		}
		assert.equal(replies.length, 0);
	});

test("parseJsonLoose takes fenced, prefixed and <think> replies and refuses prose", () => {
	assert.deepEqual(parseJsonLoose('<think>x</think>```json\n{"a": 1}\n```'), { a: 1 });
	assert.deepEqual(parseJsonLoose('Sure: {"a": {"b": "}"}} trailing'), { a: { b: "}" } });
	assert.throws(() => parseJsonLoose("no json"), /Could not parse JSON/);
});

test("psvBase reads --model humanclaw-psv/<base> from argv only", () => {
	assert.equal(psvBase(["pi", "--model", "humanclaw-psv/selfhost/muse"]), "selfhost/muse");
	assert.equal(psvBase(["pi", "--model=humanclaw-psv/a/b"]), "a/b");
	assert.equal(psvBase(["pi", "--model", "selfhost/muse"]), undefined);
});

import humanclaw from "../src/robots/humanclaw/index.ts";
import { fakeEnv, rgb, stubPi } from "./sim-stub.ts";

test("pi mode: act records target_visible and runs the unit's SkillCall; STOP ends and finishes the rollout", async (t) => {
	let step = 0;
	const obsOf = (done = false, stopped = false) => ({
		ego: rgb(4, 4),
		instruction: "Find the bed.",
		step,
		max_steps: 100,
		done,
		stopped,
		episode: {
			scene_id: "s",
			episode_id: "0",
			object_category: "bed",
			rollout: 0,
			key: "s_ep0_bed",
			output_dir: "/x",
		},
	});
	const env = await fakeEnv((c) => {
		if (c.method === "env.reset") return obsOf();
		if (c.method === "env.step") {
			step++;
			const stop = c.kwargs.skill === "stand";
			return { ...obsOf(stop, stop), collision: {} };
		}
		if (c.method === "env.finish")
			return {
				metrics: { nav_sr_20cm: true, find_sr: true },
				steps: step,
				active_stop: true,
				videos: [],
				metrics_path: "/x/metrics.json",
			};
		return undefined;
	});
	const s = stubPi({ env: env.url, units: "both", "humanclaw-mode": "pi" });
	t.after(() => env.close());
	humanclaw(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(s.active().sort(), ["act", "finish", "look", "plan"]);
	const look = await s.run("look", {});
	assert.equal(look.details.instruction, "Find the bed.");
	await s.run("act", { unit: "TURN_RIGHT", param: 200, target_visible: true });
	const rec = env.calls.find((c) => c.method === "env.record_decision")!;
	assert.equal(rec.kwargs.decision.planner_skill.visible_state, "The target is visible.");
	const st = env.calls.find((c) => c.method === "env.step")!;
	assert.deepEqual([st.kwargs.skill, st.kwargs.cond, st.kwargs.action_name], ["turn", -120, "Turn<right><120>"]);
	const r = await s.run("act", { unit: "STOP" });
	assert.equal(r.details.done, true);
	assert.ok(env.calls.some((c) => c.method === "env.finish"));
	const after = await s.run("act", { unit: "WALK", param: "fast" });
	assert.match(after.content[0].text, /not run/);
	assert.equal(env.calls.filter((c) => c.method === "env.step").length, 2);
});

test("paper mode needs humanclaw-psv and --units=both; pi-mode modules are refused", async () => {
	const s = stubPi({ units: "true", "humanclaw-mode": "pi", env: "http://127.0.0.1:1" });
	humanclaw(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(s.active(), []);
	const p = stubPi({ units: "both", "humanclaw-mode": "paper", env: "http://127.0.0.1:1" });
	humanclaw(p.pi);
	await p.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(p.active(), []);
});

test("humanclaw-psv turns: look first, then one act per decision from the base model's JSON, finish when done", async () => {
	const { mountPsv, PSV_ENTRY } = await import("../src/robots/humanclaw/provider.ts");
	const s = stubPi();
	const emitted: unknown[] = [];
	(s.pi as any).events = { emit: (_: string, d: unknown) => emitted.push(d), on: () => () => {} };
	const replies = [
		"not json",
		JSON.stringify({
			visible_state: "The bed is visible.",
			mid_level_goal: "Go",
			action_id: 2,
			action_name: "Turn<left><45>",
		}),
	];
	const asked: any[] = [];
	const psv = mountPsv(s.pi, "selfhost/muse", async (content) => {
		asked.push(content);
		return { text: replies.shift() ?? "{}" };
	});
	await s.emit("session_start");
	const turn = async (messages: any[]) => {
		const model = { id: "selfhost/muse", api: "humanclaw-psv", provider: "humanclaw-psv" } as any;
		const stream = psv.streamSimple(model, { messages } as any, {});
		let done: any;
		for await (const ev of stream) if (ev.type === "done") done = ev.message;
		return done.content.filter((c: any) => c.type === "toolCall");
	};
	const [look] = await turn([]);
	assert.equal(look.name, "look");
	const img = { type: "image", data: "AAAA", mimeType: "image/png" };
	const result = (id: string, details: any) => ({
		role: "toolResult",
		toolCallId: id,
		content: [img],
		details,
		isError: false,
	});
	const [act] = await turn([result(look.id, { instruction: "Find the bed.", step: 0 })]);
	assert.deepEqual([act.name, act.arguments], ["act", { unit: "TURN_LEFT", param: 45 }]);
	assert.equal(asked.length, 2, "the unparsable reply is asked again at the same state");
	assert.deepEqual(asked[0][1], img, "prompt text first, then the ego image");
	assert.equal((emitted[0] as any).planner_skill.visible_state, "The bed is visible.");
	assert.equal(s.entries.filter((e) => e.type === PSV_ENTRY).length, 1);
	const [fin] = await turn([result(act.id, { done: true, stopped: true, step: 1 })]);
	assert.deepEqual([fin.name, fin.arguments.status], ["finish", "success"]);
});
