import assert from "node:assert/strict";
import { test } from "node:test";
import humanclaw from "../src/robots/humanclaw/index.ts";
import { fakeEnv, rgb, stubPi } from "./sim-stub.ts";

for (const enabled of [false, true]) {
	test(`HumanCLAW self-motion observation is ${enabled ? "enabled and recorded" : "hidden by default"}`, async (t) => {
		const feedback = { horizontal_displacement_m: 0.02, turned_left_deg: 71, height_from_start_m: -0.1 };
		let step = 0;
		const observation = () => ({
			ego: rgb(4, 4),
			instruction: "Find the couch and sit on it.",
			step,
			max_steps: 100,
			done: false,
			stopped: false,
			proprioception: feedback,
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
				return observation();
			}
			return undefined;
		});
		t.after(() => env.close());
		const s = stubPi({ env: env.url, units: "both", "humanclaw-mode": "pi", "humanclaw-proprioception": enabled });
		humanclaw(s.pi);
		await s.emit("session_start");
		process.exitCode = undefined;
		assert.equal(env.calls.find((c) => c.method === "env.reset")?.kwargs.proprioception, enabled);
		const look = await s.run("look", {});
		const action = await s.run("act", { unit: "TURN_LEFT", param: 120 });
		for (const result of [look, action]) {
			assert.deepEqual(result.details.proprioception, enabled ? feedback : undefined);
			assert.equal(JSON.stringify(result.content).includes("turned_left_deg"), enabled);
		}
		await s.emit("agent_start");
		await s.emit("session_shutdown");
		assert.equal(s.entries.find((e) => e.type === "robot_result")?.data.humanclaw_proprioception, enabled);
	});
}

test("HumanCLAW paper mode refuses extra self-motion feedback before connecting", async () => {
	const s = stubPi({
		env: "http://127.0.0.1:1",
		units: "both",
		"humanclaw-mode": "paper",
		"humanclaw-proprioception": true,
	});
	humanclaw(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(s.active(), []);
});
