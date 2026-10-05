import assert from "node:assert/strict";
import { test } from "node:test";
import humanclaw from "../src/robots/humanclaw/index.ts";
import { fakeEnv, rgb, stubPi } from "./sim-stub.ts";

/**
 * NavSR@20cm is computed in metrics.json (--humanclaw-metrics). Without it the episode has no
 * verdict: robot_result.success is null (eval.sh: unscored), not false; choosing STOP is not success.
 */
for (const metrics of [null, { nav_sr_20cm: false, find_sr: true }, { nav_sr_20cm: true, find_sr: true }]) {
	test(`HumanCLAW success is the benchmark's verdict: metrics ${JSON.stringify(metrics)}`, async (t) => {
		let step = 0;
		const observation = (done = false) => ({
			ego: rgb(4, 4),
			instruction: "Find the bed.",
			step,
			max_steps: 100,
			done,
			stopped: done,
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
			if (c.method === "env.reset") return observation();
			if (c.method === "env.step") {
				step++;
				return observation(c.kwargs.skill === "stand");
			}
			if (c.method === "env.finish")
				return {
					metrics,
					steps: step,
					active_stop: true,
					videos: [],
					metrics_path: metrics ? "/x/metrics.json" : null,
				};
			return undefined;
		});
		t.after(() => env.close());
		const s = stubPi({
			"env-url": env.url,
			units: "both",
			"humanclaw-mode": "pi",
			"humanclaw-metrics": metrics !== null,
		});
		humanclaw(s.pi);
		await s.emit("session_start");
		process.exitCode = undefined;
		// STOP ends the episode: the planner's claim, not the benchmark's verdict.
		await s.run("act", { unit: "STOP" });
		await s.emit("agent_start");
		await s.emit("session_shutdown");
		const r = s.entries.find((e) => e.type === "robot_result")?.data;
		assert.equal(r.active_stop, true);
		assert.equal(r.success, metrics === null ? null : metrics.nav_sr_20cm);
		assert.deepEqual(r.metrics, metrics);
	});
}
