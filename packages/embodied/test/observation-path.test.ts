/**
 * The observation path (../src/observation/path.ts) a robot's manifest and its `OBSERVATION`
 * declaration yield: LIBERO's get_observation and its named wrist camera at the tools' size, RoboCasa's
 * fully-required render_camera with the sim's camera names and 256, RoboTwin's per-view render and
 * env.policy_frame as its state; the manifest alone where the robot declares nothing; errors, not
 * guesses, where a required parameter has no value.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import { type Call, flipRows, observationPath, robotObservation } from "../src/observation/path.ts";
import { loadManifest, type Manifest } from "../src/primitives/manifest.ts";

const all = () => true;

test("LIBERO: get_observation for everything, render_camera by the facade's camera name at 1024, flipped", async () => {
	const decl = await robotObservation("libero");
	assert.deepEqual(decl, {
		cameras: { agentview: "agentview", wrist: "robot0_eye_in_hand" },
		size: 1024,
		flip: true,
	});
	const p = observationPath(loadManifest("libero"), all, {}, decl);
	assert.equal(p.observation, "env.get_observation");
	assert.equal(p.state, "env.get_state");
	assert.equal(p.render?.method, "env.render_camera");
	assert.deepEqual(p.render?.cameras, { agentview: "agentview", wrist: "robot0_eye_in_hand" });
	assert.deepEqual(p.render?.kwargs("robot0_eye_in_hand"), {
		camera_name: "robot0_eye_in_hand",
		height: 1024,
		width: 1024,
	});
	assert.equal(p.render?.flip, true);
});

test("RoboCasa: no get_observation; render_camera's four required arguments from the robot's declaration", async () => {
	const decl = await robotObservation("robocasa");
	assert.equal(decl.size, 256);
	const p = observationPath(loadManifest("robocasa"), all, {}, decl);
	assert.equal(p.observation, undefined);
	assert.equal(p.state, "env.get_state");
	assert.deepEqual(Object.keys(p.render?.cameras ?? {}), ["agentview", "navview", "wrist"]);
	assert.deepEqual(p.render?.kwargs("robot0_eye_in_hand"), {
		camera_name: "robot0_eye_in_hand",
		height: 256,
		width: 256,
		depth: false,
	});
	// Without the declaration the manifest cannot name the cameras (a string) nor size the render: an error, not a guess.
	assert.throws(() => observationPath(loadManifest("robocasa"), all), /declares no camera names/);
	assert.throws(
		() => observationPath(loadManifest("robocasa"), all, {}, { cameras: ["robot0_eye_in_hand"] }),
		/requires height/,
	);
});

test("RoboTwin: render per declared view with only camera_name, upright; the state is env.policy_frame", async () => {
	const decl = await robotObservation("robotwin");
	assert.deepEqual(decl, { cameras: ["head", "left_wrist", "right_wrist"] });
	const p = observationPath(loadManifest("robotwin"), all, {}, decl);
	assert.equal(p.observation, undefined);
	assert.equal(p.state, "env.policy_frame");
	assert.deepEqual(p.render?.cameras, { head: "head", left_wrist: "left_wrist", right_wrist: "right_wrist" });
	assert.deepEqual(p.render?.kwargs("head"), { camera_name: "head" });
	assert.equal(p.render?.flip, false);
});

test("ManiSkill: the active cameras are the server's robot's (widowxai: no wrist); --var cameras names them", async () => {
	const decl = await robotObservation("maniskill");
	assert.deepEqual(decl.cameras, ["agentview", "wrist"]);
	const meta = (robot?: string) =>
		(async (method: string) => {
			assert.equal(method, "env.get_env_meta");
			return robot ? { robot } : {};
		}) as Call;
	assert.deepEqual(await decl.active?.(meta("widowxai")), ["agentview"]);
	assert.deepEqual(await decl.active?.(meta("panda_stick")), ["agentview"]);
	assert.deepEqual(await decl.active?.(meta("panda")), ["agentview", "wrist"]);
	assert.deepEqual(await decl.active?.(meta()), ["agentview", "wrist"], "no robot in the meta: a Panda");
	assert.deepEqual(
		await decl.active?.(meta("some_future_robot")),
		["agentview", "wrist"],
		"unknown: the server tells",
	);
	const m = loadManifest("maniskill");
	assert.deepEqual(await observationPath(m, all, {}, decl).render?.active(meta("widowxai")), ["agentview"]);
	// --var cameras decides without asking the server; a list with no declared camera is an error, not a guess.
	const never: Call = async () => assert.fail("the server was asked");
	const given = observationPath(m, all, { cameras: ["agentview"] }, decl);
	assert.deepEqual(await given.render?.active(never), ["agentview"]);
	const wrong = observationPath(m, all, { cameras: ["overhead"] }, decl);
	await assert.rejects(() => wrong.render?.active(never) ?? Promise.resolve(), /none of the cameras overhead/);
	// Without a declaration or a variable, every enum camera is active.
	assert.deepEqual(await observationPath(m, all).render?.active(never), ["agentview", "wrist"]);
});

test("a robot without a declaration is observed from its manifest: the enum's cameras, the server's default size", async () => {
	assert.deepEqual(await robotObservation("no_such_robot"), {});
	const p = observationPath(loadManifest("maniskill"), all);
	assert.deepEqual(p.render?.cameras, { agentview: "agentview", wrist: "wrist" });
	assert.deepEqual(p.render?.kwargs("wrist"), { camera_name: "wrist" });
	const franka = observationPath(loadManifest("franka"), all);
	assert.equal(franka.observation, "env.get_observation");
	assert.equal(franka.render, undefined, "the real Franka declares no render_camera");
	const none: Manifest = { robot: "toy", internal: [], primitives: [], digest: "0" };
	assert.deepEqual(observationPath(none, all), {});
	// A code primitive this run lacks (requires) is not on the path.
	const m = loadManifest("maniskill");
	const needs = m.primitives.find((e) => e.name === "render_camera")?.requires ?? [];
	assert.deepEqual(needs, [], "maniskill's render_camera has no requirements");
});

test("flipRows reverses the rows of an HxWxC byte image", () => {
	const img = Buffer.from([1, 1, 1, 2, 2, 2, 3, 3, 3]); // 3 rows of 1 pixel RGB
	assert.deepEqual([...flipRows(img, 3)], [3, 3, 3, 2, 2, 2, 1, 1, 1]);
	assert.deepEqual([...flipRows(img, 3, 3)], [3, 3, 3, 2, 2, 2, 1, 1, 1]);
});
