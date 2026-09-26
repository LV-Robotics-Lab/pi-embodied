import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import dualFranka from "../src/dual_franka/index.ts";
import franka from "../src/franka/index.ts";
import libero from "../src/libero/index.ts";
import maniskill from "../src/maniskill/index.ts";
import metaworld from "../src/metaworld/index.ts";
import piperDual from "../src/piper/dual.ts";
import piper from "../src/piper/index.ts";
import robocasa from "../src/robocasa/index.ts";
import robolab from "../src/robolab/index.ts";
import robosuite from "../src/robosuite/index.ts";
import { defineRobot, toolSections } from "../src/robot.ts";
import robotwin from "../src/robotwin/index.ts";
import ur5e from "../src/ur5e/index.ts";
import { VLA_ADAPTERS } from "../src/vla-adapters.ts";

type Handler = (event: any, ctx: any) => unknown;

/**
 * A stub pi that records registered tools and, like pi, activates only registered tools that
 * --exclude-tools did not remove.
 */
function fakePi(exclude: string[] = [], preset: Record<string, string> = {}) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools: string[] = [];
	let active: string[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = preset[name] ?? o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: { name: string }) => tools.push(t.name),
		registerCommand: () => {},
		registerProvider: () => {},
		registerMessageRenderer: () => {},
		registerShortcut: () => {},
		setActiveTools: (names: string[]) => {
			active = names.filter((n) => tools.includes(n) && !exclude.includes(n));
		},
		getActiveTools: () => active,
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const ctx = {
		hasUI: false,
		cwd: tmpdir(),
		ui: { notify: () => {} },
		shutdown: () => {},
		sessionManager: { getBranch: () => [], getSessionDir: () => tmpdir() },
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) result = (await fn({ type: name, ...event }, ctx)) ?? result;
		return result;
	}
	return { pi, tools, emit };
}

const read = (path: string) => readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");
const markers = (text: string) => [...text.matchAll(/\[\/?tool:([\w|]+)\]/g)].flatMap((m) => m[1].split("|"));
/** Whether the text still names the tool: bare words (`release`, `render`) only count in backticks. */
const mentions = (text: string, tool: string) =>
	(tool.includes("_") ? new RegExp(`\\b${tool}\\b`) : new RegExp(`\`${tool}\``)).test(text);

/** Every robot prompt template, the extension that registers its tools, and the
 * tools always on (the rest can be excluded without the prompt naming them). */
const ROBOTS: {
	robot: string;
	files: string[];
	load: (pi: ExtensionAPI) => unknown;
	core?: string[];
	flags?: Record<string, string>;
}[] = [
	{
		robot: "libero",
		files: ["libero/SYSTEM.md"],
		load: libero,
		// The third-party VLA tools are mounted only with their server flags.
		flags: Object.fromEntries(VLA_ADAPTERS.map((a) => [a.flag, "http://127.0.0.1:1"])),
		core: ["view_env_state", "move_to", "release", "finish"],
	},
	{
		robot: "robocasa",
		files: ["robocasa/SYSTEM.md", "robocasa/memory-hf.md", "robocasa/memory-local.md"],
		load: robocasa,
		core: ["view_env_state", "move_to", "reset", "finish"],
	},
	{
		robot: "robotwin",
		files: ["robotwin/SYSTEM.md", "robotwin/memory.md"],
		load: robotwin,
		core: ["view_env_state", "move_to", "set_gripper", "release", "finish"],
	},
	{
		robot: "franka",
		files: ["franka/SYSTEM.md"],
		load: franka,
		core: ["view_env_state", "move_delta", "open_gripper", "close_gripper", "finish"],
	},
	{ robot: "maniskill", files: ["maniskill/SYSTEM.md"], load: maniskill },
	{
		robot: "robosuite",
		files: ["robosuite/SYSTEM.md"],
		load: robosuite,
		core: ["view_env_state", "move_to", "move_delta", "finish"],
	},
	{
		robot: "metaworld",
		files: ["metaworld/SYSTEM.md"],
		load: metaworld,
		core: ["view_env_state", "move_delta", "finish"],
	},
	{ robot: "robolab", files: ["robolab/SYSTEM.md"], load: robolab, core: ["finish"] },
	{ robot: "dual_franka", files: ["dual_franka/SYSTEM.md"], load: dualFranka },
	{ robot: "piper", files: ["piper/SYSTEM.md"], load: piper },
	{ robot: "piper dual", files: ["piper/SYSTEM_DUAL.md"], load: piperDual },
	{
		robot: "ur5e",
		files: ["ur5e/SYSTEM.md"],
		load: ur5e,
		core: ["view_env_state", "move_delta", "gripper", "finish"],
	},
];

test("tool blocks: lines and inline spans, any-of names, nesting, adjacent blocks", () => {
	const t = "a\n[tool:x]\nx line\n[/tool:x]\nb[tool:y] y span[/tool:y].\n[tool:x|y]\nx or y\n[/tool:x|y]\nc\n";
	assert.equal(toolSections(t, ["x", "y"]), "a\nx line\nb y span.\nx or y\nc\n");
	assert.equal(toolSections(t, ["x"]), "a\nx line\nb.\nx or y\nc\n");
	assert.equal(toolSections(t, ["y"]), "a\nb y span.\nx or y\nc\n");
	assert.equal(toolSections(t, []), "a\nb.\nc\n");
	// Nested blocks need every name; adjacent ones are independent.
	const n = "use [tool:x]`x`[/tool:x][tool:x][tool:y] or [/tool:y][/tool:x][tool:y]`y`[/tool:y].";
	assert.equal(toolSections(n, ["x", "y"]), "use `x` or `y`.");
	assert.equal(toolSections(n, ["x"]), "use `x`.");
	assert.equal(toolSections(n, ["y"]), "use `y`.");
	assert.equal(toolSections("[tool:x]\n[tool:y]\nboth\n[/tool:y]\n[/tool:x]\nend", ["y"]), "end");
	assert.equal(toolSections("[tool:x]\n[tool:y]\nboth\n[/tool:y]\n[/tool:x]\nend", ["x", "y"]), "both\nend");
	assert.equal(toolSections("no markers", []), "no markers");
});

test("an unpaired or crossed tool marker throws instead of leaking into the prompt", () => {
	assert.throws(() => toolSections("[tool:x]\nopen\n", ["x"]), /unpaired \[tool:x\]/);
	assert.throws(() => toolSections("close\n[/tool:x]\n", ["x"]), /unpaired \[\/tool:x\]/);
	assert.throws(() => toolSections("[tool:x]a[/tool:y]", ["x", "y"]), /unpaired/);
	assert.throws(() => toolSections("[tool:x]a[tool:x]b[/tool:x]c[/tool:x]", ["x"]), /nested in itself/);
});

for (const { robot, files, load, core, flags } of ROBOTS) {
	test(`${robot}: every tool marker names a tool the robot registers`, async () => {
		const { pi, tools } = fakePi([], flags);
		await load(pi);
		for (const file of files) {
			const text = read(file);
			const unknown = markers(text).filter((n) => !tools.includes(n));
			assert.deepEqual(unknown, [], `${file} marks tools ${robot} does not register`);
			assert.doesNotMatch(toolSections(text, tools), /\[\/?tool:/);
		}
	});
	if (!core) continue;
	test(`${robot}: excluding a tool leaves no mention of it in the prompt`, async () => {
		const { pi, tools } = fakePi([], flags);
		await load(pi);
		const text = files.map(read).join("\n");
		const plain = text.replace(/\[\/?tool:[\w|]+\]/g, "");
		const optional = tools.filter((n) => !core.includes(n));
		for (const tool of optional) {
			const rendered = toolSections(
				text,
				tools.filter((n) => n !== tool),
			);
			assert.ok(!mentions(rendered, tool), `${robot} still names ${tool}`);
		}
		// While active they are still described.
		for (const tool of optional.filter((n) => mentions(plain, n)))
			assert.ok(mentions(toolSections(text, tools), tool), `${robot} no longer names ${tool}`);
	});
}

test("the robot base renders tool blocks for the tools left active after --exclude-tools", async () => {
	const { pi, emit } = fakePi(["pi0_pick"]);
	const robot = defineRobot(pi, {
		name: "toy",
		task: [],
		keepImages: 1,
		start: async () => ["pi0_pick", "move_to"],
		prompt: () => read("libero/SYSTEM.md"),
		result: () => ({}),
		finish: {
			description: "finish",
			parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
			result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
		},
	});
	for (const name of ["pi0_pick", "move_to"])
		robot.tool(name, name, Type.Object({}), async () => ({ content: [], details: {} }));
	await emit("session_start");
	const { systemPrompt } = await emit("before_agent_start", { systemPrompt: "base" });
	assert.doesNotMatch(systemPrompt, /pi0_pick|Pi0|\[\/?tool:/);
	assert.match(systemPrompt, /`move_to`/);
});
