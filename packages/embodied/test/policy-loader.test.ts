import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

test("the deployed CLI loader registers the Durable policy with MetaWorld", () => {
	// The CLI's jiti aliases differ from native Node imports: pi-ai subpaths can
	// accidentally resolve beneath compat.js. Exercise the deployed loader itself.
	const script = `
import { loadExtensions } from './packages/coding-agent/dist/core/extensions/loader.js';
const loaded = await loadExtensions(['packages/embodied/src/robots/metaworld/index.ts'], process.cwd());
console.log(JSON.stringify({
  errors: loaded.errors,
  extensions: loaded.extensions.map(e => ({ tools: [...e.tools.keys()], flags: [...e.flags.keys()] }))
}));
`;
	const output = execFileSync(process.execPath, ["--conditions=source", "--input-type=module", "-e", script], {
		cwd: fileURLToPath(new URL("../../../", import.meta.url)),
		encoding: "utf8",
		timeout: 60_000,
	});
	const result = JSON.parse(output) as {
		errors: { path: string; error: string }[];
		extensions: { tools: string[]; flags: string[] }[];
	};
	assert.deepEqual(result.errors, []);
	assert.equal(result.extensions.length, 1);
	// The robot mounts the policy: one extension owns policy_goal, --policy-store and the MetaWorld tools.
	const [robot] = result.extensions;
	for (const tool of ["policy_goal", "move_delta", "view_env_state"]) assert.ok(robot.tools.includes(tool), tool);
	assert.ok(robot.flags.includes("policy-store"));
});
