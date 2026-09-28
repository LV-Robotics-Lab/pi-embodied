import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { loadOracle, TIERS } from "../src/modes/code/index.ts";
import { TASKS } from "../src/robots/robosuite/index.ts";

const MARK = "# ---- CaP-X's program, verbatim ----\n";
const dir = (robot: string) => fileURLToPath(new URL(`../src/robots/${robot}/oracle/`, import.meta.url));
const programs = (robot: string) =>
	readdirSync(dir(robot))
		.filter((f) => f.endsWith(".py") && !f.startsWith("capx_"))
		.sort();

/**
 * sha256 of each oracle's verbatim block (CaP-X @53e9966: the task file's ORACLE_CODE /
 * PRIVILEGED_ORACLE_CODE / UNPRIVILEGED_ORACLE_CODE, or the config's inline oracle_code).
 * services/tests/test_capx_oracles.py compares the text itself with the reference repo when present.
 */
const VERBATIM: Record<string, string> = {
	"robosuite/lift.py": "f537b4c10826472208243c020952087c0076e13d6cf9a1156b4a69e9b1bd27eb",
	"robosuite/lift_privileged.py": "f537b4c10826472208243c020952087c0076e13d6cf9a1156b4a69e9b1bd27eb",
	"robosuite/nut_assembly.py": "c86356d44d139b8aa88ca2a3ac78e995b1ee8020991efc05153163ac8a5c95c5",
	"robosuite/nut_assembly_privileged.py": "c86356d44d139b8aa88ca2a3ac78e995b1ee8020991efc05153163ac8a5c95c5",
	"robosuite/restack.py": "42546a1fcc9a3f7b68bc756e4c20c7b1b8583097dbae50bdd722859ada173088",
	"robosuite/restack_privileged.py": "42546a1fcc9a3f7b68bc756e4c20c7b1b8583097dbae50bdd722859ada173088",
	"robosuite/stack.py": "ac9ac5cb653eb6ec0f3e13d309e53f8bbfe84e1e7bb0d6844cc81fb114dc970d",
	"robosuite/stack_privileged.py": "ac9ac5cb653eb6ec0f3e13d309e53f8bbfe84e1e7bb0d6844cc81fb114dc970d",
	"robosuite/two_arm_handover.py": "15d489ffbb82b99ecb2166d77fac60d2b42015f68bced7dee6525ebae1b75670",
	"robosuite/two_arm_handover_privileged.py": "d81eccea331d3bafd6883abbc3ff30cee53a4b740450ad429f19274827f6f259",
	"robosuite/two_arm_lift.py": "bd8e8c3ddb76a78a742b0e5ff3789f2a3a39ac97a94db3dc392e352ff7463983",
	"robosuite/two_arm_lift_privileged.py": "bd8e8c3ddb76a78a742b0e5ff3789f2a3a39ac97a94db3dc392e352ff7463983",
	"robosuite/wipe.py": "ed4687065411a34e77e92ea6207598e16083564ee2b0ad37a41e8ad3470e9c52",
	"robosuite/wipe_privileged.py": "ed4687065411a34e77e92ea6207598e16083564ee2b0ad37a41e8ad3470e9c52",
	"libero/object_swap_7.py": "1f1682af567525536b2befb7ba3f050308a32497559796a59c1afc175b12fef7",
	"libero/object_swap_7_privileged.py": "951bf9f8568d7601745f533f106cf263b04baacfb322d10d2665177107e1af28",
	"libero/object_swap_7_skill_library.py": "fd2eddcf117a8038ea748873c16d9eca8d8c61ba2b80707395c241835ff41afd",
};

/** The task flags of each robot an oracle header may name (they must equal the episode's). */
const TASK_FIELDS: Record<string, (h: Record<string, string>) => void> = {
	robosuite: (h) => {
		assert.ok((TASKS as readonly string[]).includes(h.task), `robosuite task ${h.task}`);
		assert.equal(h.suite, undefined);
	},
	libero: (h) => {
		// LIBERO-Pro's object-swap suite (--libero-type pro, the default); tasks are indices.
		assert.equal(h.suite, "libero_object_swap");
		assert.match(h.task, /^\d+$/);
	},
};

test("CaP-X's 17 human oracles: 14 robosuite (7 tasks x S2 / privileged) and 3 LIBERO", () => {
	assert.equal(programs("robosuite").length, 14);
	assert.equal(programs("libero").length, 3);
	assert.deepEqual(
		Object.keys(VERBATIM).sort(),
		[...programs("libero").map((f) => `libero/${f}`), ...programs("robosuite").map((f) => `robosuite/${f}`)].sort(),
	);
	const tasks = new Set(programs("robosuite").map((f) => loadOracle(f, dir("robosuite")).header.task));
	assert.deepEqual([...tasks].sort(), [...TASKS].sort());
});

for (const robot of ["robosuite", "libero"]) {
	for (const file of programs(robot)) {
		test(`${robot} oracle ${file}: header, prelude and CaP-X's verbatim program`, () => {
			const o = loadOracle(file.replace(/\.py$/, ""), dir(robot));
			assert.equal(o.name, file);
			assert.match(o.header.capx, /^env_configs\/human_oracle_code\/\S+\.yaml @53e9966$/);
			assert.ok(o.header.program);
			assert.ok([...TIERS, "privileged"].includes(o.header.tier), `tier ${o.header.tier}`);
			TASK_FIELDS[robot](o.header);
			assert.ok(o.header.prelude && existsSync(join(dir(robot), o.header.prelude)), "prelude exists");
			const prelude = readFileSync(join(dir(robot), o.header.prelude), "utf8");
			assert.ok(o.code.startsWith(prelude), "the prelude runs first");
			// Only the privileged tier may read the simulator's ground truth.
			if (o.header.tier !== "privileged") assert.doesNotMatch(prelude, /ground_truth_poses/);
			const text = readFileSync(join(dir(robot), file), "utf8");
			assert.equal(text.split(MARK).length, 2, "one verbatim marker");
			const verbatim = text.split(MARK)[1];
			assert.equal(createHash("sha256").update(verbatim).digest("hex"), VERBATIM[`${robot}/${file}`]);
			assert.equal(o.sha256, createHash("sha256").update(o.code).digest("hex"));
		});
	}
}
