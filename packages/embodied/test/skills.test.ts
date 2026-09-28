import assert from "node:assert/strict";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { probePerception, probeSkill, withSkillsOff } from "../src/capabilities/skills.ts";
import { useDeployment } from "./helpers/deployment.ts";

const pi = (flags: Record<string, unknown>) => ({ getFlag: (n: string) => flags[n] }) as unknown as ExtensionAPI;
const up = async () => undefined;
const down = async () => {
	throw new Error("ECONNREFUSED");
};

test("a perception tool whose server does not answer stays inactive; the others are kept", async () => {
	const tools = ["move_to", "segment", "detect", "enhance_depth", "point", "finish"];
	useDeployment({ services: { sam3: "http://sam3", molmo: "http://molmo" } });
	// A simulator: SAM3 behind --detections, UniDepth switched off (--depth unset), Molmo behind --point.
	const r = await probePerception(pi({ detections: true, depth: "", point: true }), tools, (url) =>
		url === "http://sam3" ? down() : up(),
	);
	assert.deepEqual(r.keep, ["move_to", "point", "finish"]);
	assert.deepEqual(Object.keys(r.off).sort(), ["sam3", "unidepth"]);
	assert.match(String(r.off.sam3.reason), /^unreachable: ECONNREFUSED/);
	assert.equal(r.off.unidepth.reason, "switched off");
});

test("a server no active tool needs is not probed; the Franka's --segment attaches services.sam3", async () => {
	const probed: string[] = [];
	const probe = async (url: string) => {
		probed.push(url);
	};
	useDeployment({ services: { sam3: "http://a", molmo: "http://m" } });
	await probePerception(pi({ detections: true, point: true }), ["move_to"], probe);
	assert.deepEqual(probed, []);
	useDeployment({ services: { sam3: "http://robot-sam3" } });
	await probePerception(pi({ segment: false }), ["segment"], probe);
	assert.deepEqual(probed, [], "--segment off: switched off, not probed");
	await probePerception(pi({ segment: true }), ["segment"], probe);
	assert.deepEqual(probed, ["http://robot-sam3"]);
	// A robot with no switch of its own for a server (BEHAVIOR's segment and point) uses the deployment's.
	useDeployment({ services: { molmo: "http://m" } });
	await probePerception(pi({}), ["point"], probe);
	assert.deepEqual(probed, ["http://robot-sam3", "http://m"]);
});

test("--require-skills refuses a run whose skill is off; the result merges the robot's and the perception skills_off", async () => {
	useDeployment({ services: { sam3: "http://sam3" } });
	await assert.rejects(
		probePerception(pi({ detections: true, "require-skills": "sam3" }), ["segment"], down),
		/--require-skills sam3: the sam3 server is unreachable/,
	);
	await assert.rejects(probeSkill(pi({ "require-skills": "pi0,rldx" }), "rldx", "off", up), /switched off/);
	assert.deepEqual(
		withSkillsOff({ a: 1, skills_off: { pi0: "switched off" } }, { sam3: { on: false, reason: "x" } }),
		{
			a: 1,
			skills_off: { pi0: "switched off", sam3: "x" },
		},
	);
	assert.deepEqual(withSkillsOff({ a: 1 }, {}), { a: 1 });
});
