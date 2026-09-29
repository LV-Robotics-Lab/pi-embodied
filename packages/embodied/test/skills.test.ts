import assert from "node:assert/strict";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { probePerception, probeSkill, withSkillsOff } from "../src/capabilities/skills.ts";

const pi = (flags: Record<string, unknown>) => ({ getFlag: (n: string) => flags[n] }) as unknown as ExtensionAPI;
const up = async () => undefined;
const down = async () => {
	throw new Error("ECONNREFUSED");
};

test("a perception tool whose server does not answer stays inactive; the others are kept", async () => {
	const tools = ["move_to", "segment", "detect", "enhance_depth", "point", "finish"];
	const r = await probePerception(pi({ sam3: "http://sam3", unidepth: "", molmo: "http://molmo" }), tools, (url) =>
		url === "http://sam3" ? down() : up(),
	);
	assert.deepEqual(r.keep, ["move_to", "point", "finish"]);
	assert.deepEqual(Object.keys(r.off).sort(), ["sam3", "unidepth"]);
	assert.match(String(r.off.sam3.reason), /^unreachable: ECONNREFUSED/);
	assert.equal(r.off.unidepth.reason, "switched off");
});

test("a server no active tool needs is not probed; the Franka's --robot-sam3 names the SAM3 server", async () => {
	const probed: string[] = [];
	const probe = async (url: string) => {
		probed.push(url);
	};
	await probePerception(pi({ sam3: "http://a", molmo: "http://m" }), ["move_to"], probe);
	assert.deepEqual(probed, []);
	await probePerception(pi({ "robot-sam3": "http://robot-sam3" }), ["segment"], probe);
	assert.deepEqual(probed, ["http://robot-sam3"]);
});

test("--require-skills refuses a run whose skill is off; the result merges the robot's and the perception skills_off", async () => {
	await assert.rejects(
		probePerception(pi({ sam3: "http://sam3", "require-skills": "sam3" }), ["segment"], down),
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
