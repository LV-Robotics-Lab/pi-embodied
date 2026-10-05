import assert from "node:assert/strict";
import { test } from "node:test";
import humanclaw from "../src/robots/humanclaw/index.ts";
import { fakeEnv, rgb, stubPi } from "./sim-stub.ts";

/**
 * The metric tracker's per-step collision verdict (env.step's `collision`, metric mode) is not in
 * HumanCLAW's observation space: the model sees it only under --humanclaw-collision-feedback, a
 * pi-mode experiment that robot_result records.
 */
for (const enabled of [false, true]) {
	const title = enabled ? "shown under the flag and recorded" : "hidden by default, even with metrics on";
	test(`HumanCLAW collision verdict is ${title}`, async (t) => {
		let step = 0;
		const observation = () => ({
			ego: rgb(4, 4),
			instruction: "Find the couch and sit on it.",
			step,
			max_steps: 100,
			done: false,
			stopped: false,
			episode: {
				scene_id: "s",
				episode_id: "1",
				object_category: "couch",
				rollout: 0,
				key: "s_ep1_couch",
				output_dir: "/x",
			},
		});
		const env = await fakeEnv((call) => {
			if (call.method === "env.reset") return observation();
			if (call.method === "env.step") {
				step++;
				return { ...observation(), collision: { collided: true } };
			}
			return undefined;
		});
		t.after(() => env.close());
		const s = stubPi({
			"env-url": env.url,
			units: "both",
			"humanclaw-mode": "pi",
			"humanclaw-metrics": true,
			"humanclaw-collision-feedback": enabled,
		});
		humanclaw(s.pi);
		await s.emit("session_start");
		process.exitCode = undefined;
		const action = await s.run("act", { unit: "WALK", param: "normal" });
		assert.equal(JSON.stringify(action.content).includes("collided"), enabled, "the model sees it only on request");
		assert.deepEqual((action.details as any).result.collision, enabled ? { collided: true } : undefined);
		await s.emit("agent_start");
		await s.emit("session_shutdown");
		assert.equal(s.entries.find((e) => e.type === "robot_result")?.data.humanclaw_collision_feedback, enabled);
	});
}

test("HumanCLAW collision feedback needs the metric tracker and is refused in paper mode", async () => {
	const noMetrics = stubPi({
		"env-url": "http://127.0.0.1:1",
		units: "both",
		"humanclaw-mode": "pi",
		"humanclaw-collision-feedback": true,
	});
	humanclaw(noMetrics.pi);
	await noMetrics.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(noMetrics.active(), []);
	const paper = stubPi({
		"env-url": "http://127.0.0.1:1",
		units: "both",
		"humanclaw-mode": "paper",
		"humanclaw-metrics": true,
		"humanclaw-collision-feedback": true,
	});
	humanclaw(paper.pi);
	await paper.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(paper.active(), []);
});
