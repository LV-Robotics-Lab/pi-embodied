/**
 * --tier / --preset (src/infra/tiers.ts): a choice reads as the flags it expands to, a flag given
 * with another value refuses to start naming both, a robot that cannot serve the choice refuses too,
 * the result records the choice and its axes, and the eval scripts (params-match.mjs,
 * tier-flags.mjs, eval-options.sh) expand it the same way.
 */

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { trackedFlags } from "../src/infra/params.ts";
import {
	axesFlags,
	choose,
	conflicts,
	PRESET_NAMES,
	PRESETS,
	TIER_NAMES,
	TIERS,
	unsupported,
} from "../src/infra/tiers.ts";
import type { UnitsSpec } from "../src/modes/units/index.ts";
import { codeApiReply } from "./helpers/code-api.ts";

// The toy robot's primitive manifests (./fixtures/manifests/toy.json: high and low tiers; toylow.json: low only).
process.env.PI_EMBODIED_MANIFESTS = new URL("./fixtures/manifests/", import.meta.url).pathname;

import { defineRobot, RESULT_ENTRY } from "../src/robot.ts";
import { deployFlags } from "./helpers/deployment.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi that runs handlers in registration order and records flags, tools and entries. */
function fakePi(flagValues: Record<string, unknown> = {}) {
	flagValues = deployFlags(flagValues);
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	let active: string[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const errors: string[] = [];
	const ctx = {
		hasUI: false,
		ui: { notify: () => {} },
		shutdown: () => {},
		sessionManager: { getBranch: () => [] },
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) {
			const r: any = await fn({ type: name, ...event }, ctx);
			if (r !== undefined) result = r;
		}
		return result;
	}
	return { pi, flags, tools, entries, emit, errors, active: () => active };
}

const UNITS: UnitsSpec = {
	vectors: {
		MV_FWD: [1, 0, 0],
		MV_BACK: [-1, 0, 0],
		MV_LEFT: [0, -1, 0],
		MV_RIGHT: [0, 1, 0],
		MV_UP: [0, 0, 1],
		MV_DOWN: [0, 0, -1],
	},
	stepM: 0.02,
	apply: async () => ({ content: [], details: {} }),
};

/** A toy robot: code mode over a fake env server, and the modules `o` mounts. The start's error, when it was refused. */
async function toy(
	flags: Record<string, unknown>,
	o: {
		code?: boolean;
		units?: boolean;
		vdm?: boolean;
		groundTruth?: boolean;
		xpolicy?: boolean;
		manifest?: string;
	} = {},
) {
	const f = fakePi(flags);
	const calls: { method: string; kwargs: Record<string, unknown> }[] = [];
	const rpc = {
		interrupt: async () => {},
		call: async <T>(method: string, kwargs: Record<string, unknown> = {}): Promise<T> => {
			calls.push({ method, kwargs });
			if (method === "code.api") return codeApiReply(o.manifest ?? "toy", kwargs.tier as string | undefined) as T;
			if (method === "code.helpers") return [] as T;
			if (method === "code.preflight") return { isolated: true, error: null } as T;
			throw new Error(`unexpected ${method}`);
		},
	};
	// The stderr line a refused start prints without a UI.
	const error = console.error;
	const stderr: string[] = [];
	console.error = (...a: unknown[]) => stderr.push(a.map(String).join(" "));
	try {
		defineRobot(f.pi, {
			name: "toy",
			manifest: o.manifest ?? "toy",
			task: [],
			keepImages: 2,
			start: async () => ["move_to", "segment"],
			result: () => ({}),
			...(o.units ? { units: UNITS } : {}),
			...(o.vdm ? { vdm: { views: 1 } } : {}),
			...(o.groundTruth ? { groundTruth: async () => ({ poses: {} }) } : {}),
			// Mounted without a policy: the module registers --xpolicy, and starts nothing while it is empty.
			...(o.xpolicy ? { xpolicy: () => undefined } : {}),
			...(o.code === false
				? {}
				: {
						code: {
							rpc: () => rpc as any,
							instruction: () => "stack the cubes",
							refuse: () => undefined,
							observe: async () => ({ content: [{ type: "text", text: "{}" }], details: {} }),
						},
					}),
			finish: {
				description: "finish",
				parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
				result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
			},
		});
		await f.emit("session_start");
	} finally {
		console.error = error;
		process.exitCode = undefined;
	}
	const refused = stderr.find((l) => l.startsWith("[toy] unavailable: "))?.slice("[toy] unavailable: ".length);
	return { ...f, calls, refused };
}

/** The robot_result entry after an episode (agent_start, then shutdown). */
async function result(f: Awaited<ReturnType<typeof toy>>) {
	await f.emit("agent_start");
	await f.emit("session_shutdown");
	return f.entries.find((e) => e.type === RESULT_ENTRY)?.data;
}

test("the eight CaP-X tiers expand to the flags of their four axes", () => {
	assert.deepEqual(TIER_NAMES, ["S1", "S2", "S3", "S4", "M1", "M2", "M3", "M4"]);
	const common = { code: "true", units: null, "vdm-video": false };
	// No frame reaches the model: none kept, and the first one not anchored either.
	const blind = { "keep-images": "0", "anchor-image": false };
	const single = { ...common, "max-turns": "1", vdm: false, ...blind };
	assert.deepEqual(axesFlags(TIERS.S1), { ...single, "code-api": "high", privileged: true });
	assert.deepEqual(axesFlags(TIERS.S2), { ...single, "code-api": "high", privileged: false });
	assert.deepEqual(axesFlags(TIERS.S3), { ...single, "code-api": "low", privileged: false });
	assert.deepEqual(axesFlags(TIERS.S4), { ...single, "code-api": "low-noexamples", privileged: false });
	assert.deepEqual(axesFlags(TIERS.M1), { ...common, "code-api": "high", vdm: false, ...blind });
	assert.deepEqual(axesFlags(TIERS.M2), { ...common, "code-api": "high", vdm: false });
	assert.deepEqual(axesFlags(TIERS.M3), { code: "true", units: null, "code-api": "high", vdm: true, ...blind });
	assert.deepEqual(axesFlags(TIERS.M4), { code: "true", units: null, "code-api": "low", vdm: true, ...blind });
	// An M tier leaves --privileged to the flag (stacked, it is recorded as M2+privileged).
	for (const m of ["M1", "M2", "M3", "M4"]) assert.equal(TIERS[m].privileged, undefined, m);
});

test("choose: a tier, none, an unknown one, or both --tier and --preset", () => {
	assert.equal(choose("", ""), undefined);
	assert.equal(choose(undefined, undefined), undefined);
	const s3 = choose("S3", "");
	assert.ok(s3 && !("error" in s3) && s3.kind === "tier" && s3.name === "S3");
	assert.match(
		String((choose("S9", "") as { error: string }).error),
		/--tier S9: not a CaP-X tier; the tiers are S1, S2, S3, S4, M1, M2, M3, M4/,
	);
	assert.match(
		String((choose("S3", "x") as { error: string }).error),
		/--tier S3 and --preset x are mutually exclusive/,
	);
});

test("conflicts name the choice, the flag and the value given; agreement is none", () => {
	const s3 = choose("S3", "") as Exclude<ReturnType<typeof choose>, undefined>;
	assert.deepEqual(conflicts(s3, new Map()), []);
	assert.deepEqual(
		conflicts(
			s3,
			new Map<string, unknown>([
				["code-api", "low"],
				["code", "pure"],
				["max-turns", "1"],
			]),
		),
		[],
	);
	assert.deepEqual(conflicts(s3, new Map<string, unknown>([["code-api", "high"]])), [
		"--tier S3 sets --code-api=low, but --code-api=high was given",
	]);
	assert.deepEqual(conflicts(s3, new Map<string, unknown>([["code", "both"]])), [
		"--tier S3 sets --code=true, but --code=both was given",
	]);
	assert.deepEqual(conflicts(s3, new Map<string, unknown>([["units", "true"]])), [
		"--tier S3 leaves --units at its default, but --units=true was given",
	]);
	assert.deepEqual(conflicts(s3, new Map<string, unknown>([["vdm", true]])), [
		"--tier S3 runs without --vdm, but --vdm was given",
	]);
	assert.deepEqual(conflicts(s3, new Map<string, unknown>([["privileged", true]])), [
		"--tier S3 runs without --privileged, but --privileged was given (S1 is S2 with the simulator's ground truth)",
	]);
	const s1 = choose("S1", "") as Exclude<ReturnType<typeof choose>, undefined>;
	assert.deepEqual(conflicts(s1, new Map<string, unknown>([["privileged", true]])), []);
	const m2 = choose("M2", "") as Exclude<ReturnType<typeof choose>, undefined>;
	assert.deepEqual(
		conflicts(
			m2,
			new Map<string, unknown>([
				["privileged", true],
				["max-turns", "20"],
				["keep-images", "3"],
			]),
		),
		[],
	);
	assert.deepEqual(conflicts(m2, new Map<string, unknown>([["max-turns", "1"]])), [
		"--tier M2 is multi-turn, but --max-turns=1 was given",
	]);
	assert.deepEqual(conflicts(m2, new Map<string, unknown>([["keep-images", "0"]])), [
		"--tier M2 feeds the camera frames back, but --keep-images=0 was given",
	]);
	const m3 = choose("M3", "") as Exclude<ReturnType<typeof choose>, undefined>;
	assert.deepEqual(
		conflicts(
			m3,
			new Map<string, unknown>([
				["keep-images", "0"],
				["vdm", true],
			]),
		),
		[],
	);
});

test("unsupported: no code mode, no tier of primitives, no ground truth, no VDM", () => {
	const caps = {
		robot: "toy",
		code: true,
		units: false,
		vdm: true,
		groundTruth: true,
		memory: false,
		xpolicy: false,
		vla: false,
		codeTiers: () => ["high", "low"],
	};
	const c = (n: string) => choose(n, "") as Exclude<ReturnType<typeof choose>, undefined>;
	for (const n of TIER_NAMES) assert.equal(unsupported(c(n), caps), undefined, n);
	assert.match(
		String(unsupported(c("S3"), { ...caps, code: false })),
		/--tier S3 runs code mode \(run_code\), which toy does not mount/,
	);
	assert.equal(
		unsupported(c("S2"), { ...caps, codeTiers: () => ["low"] }),
		"--tier S2 needs high-tier code primitives; toy's manifest has low (its tiers: S3, S4, M4)",
	);
	assert.equal(unsupported(c("S4"), { ...caps, codeTiers: () => ["low"] }), undefined);
	assert.match(
		String(unsupported(c("S1"), { ...caps, groundTruth: false })),
		/--tier S1 needs the simulator's ground truth \(--privileged\), which toy has none of: S1 is simulation only/,
	);
	assert.match(
		String(unsupported(c("M3"), { ...caps, vdm: false })),
		/--tier M3 describes each change with VDM \(--vdm\), which toy does not mount/,
	);
	assert.equal(unsupported(c("M2"), { ...caps, vdm: false, groundTruth: false }), undefined);
});

test("--tier S3 reads as code mode, the low tier, one turn and no frames; params and the result record it", async () => {
	const f = await toy({ tier: "S3" });
	assert.equal(f.refused, undefined);
	assert.deepEqual(f.active(), ["run_code", "finish"]);
	assert.deepEqual(f.calls.map((c) => [c.method, c.kwargs])[0], ["code.api", { tier: "low" }]);
	// The robot reads the expanded values; the raw ones stay what was given.
	assert.equal(f.pi.getFlag("code"), "true");
	assert.equal(f.pi.getFlag("code-api"), "low");
	assert.equal(f.pi.getFlag("max-turns"), "1");
	assert.equal(f.pi.getFlag("keep-images"), "0");
	assert.equal(f.pi.getFlag("vdm"), false);
	assert.equal(f.pi.getFlag("privileged"), false);
	assert.equal(f.flags.code, "false");
	const r = await result(f);
	assert.equal(r.tier, "S3");
	assert.equal(r.preset, undefined);
	assert.deepEqual(r.axes, { turns: "single", feedback: "none", api: "low", privileged: false });
	assert.equal(r.code, "true");
	assert.equal(r.code_api, "low");
	assert.equal(r.code_api_auto, false);
	assert.equal(r.params.tier, "S3");
	assert.equal(r.params.preset, "");
	assert.equal(r.params["code-api"], "low");
	assert.equal(r.params["max-turns"], "1");
	assert.equal(r.params["keep-images"], "0");
	assert.equal(r.params.code, "true");
	assert.equal(r.params_default.code, "false");
	assert.equal(r.params_default["keep-images"], "2");
});

test("--tier S1 runs the privileged registry with ground_truth_poses; an M tier with --privileged is a combination", async () => {
	const s1 = await toy({ tier: "S1" }, { groundTruth: true });
	assert.equal(s1.refused, undefined);
	assert.deepEqual(s1.calls[0], { method: "code.api", kwargs: { tier: "privileged" } });
	assert.ok(s1.active().includes("ground_truth_poses"));
	const r1 = await result(s1);
	assert.equal(r1.tier, "S1");
	assert.equal(r1.privileged, true);
	assert.deepEqual(r1.axes, { turns: "single", feedback: "none", api: "high", privileged: true });
	const m2 = await toy({ tier: "M2", privileged: true }, { groundTruth: true });
	assert.equal(m2.refused, undefined);
	const r2 = await result(m2);
	assert.equal(r2.tier, "M2+privileged");
	assert.deepEqual(r2.axes, { turns: "multi", feedback: "image", api: "high", privileged: true });
	const m3 = await toy({ tier: "M3" }, { vdm: true });
	assert.equal(m3.refused, undefined);
	assert.equal(m3.pi.getFlag("vdm"), true);
	assert.equal(m3.pi.getFlag("keep-images"), "0");
	assert.equal(m3.pi.getFlag("max-turns"), "0", "multi: the budget flag is the robot's");
	const r3 = await result(m3);
	assert.equal(r3.tier, "M3");
	assert.equal(r3.vdm, true);
	assert.deepEqual(r3.axes, { turns: "multi", feedback: "vdm", api: "high", privileged: false });
});

test("a flag named on the command line at its registered default is given: --tier S3 --max-turns 0 is refused, as tier-flags.mjs refuses it", async () => {
	// pi applies the command line after the extensions loaded and answers the default otherwise, so the
	// tracker reads the process's own argv for what was named (params.ts argvFlags, pi's rules).
	const argv = process.argv;
	process.argv = [...argv, "--tier=S3", "--max-turns", "0", "--units=false"];
	try {
		const zero = await toy({ tier: "S3", "max-turns": "0", units: "false" }, { units: true });
		assert.equal(
			zero.refused,
			"--tier S3 leaves --units at its default, but --units=false was given; --tier S3 sets --max-turns=1, but --max-turns=0 was given",
		);
		assert.equal(trackedFlags(zero.pi)?.given("max-turns"), true);
		assert.equal(trackedFlags(zero.pi)?.given("units"), true);
		assert.equal(trackedFlags(zero.pi)?.given("keep-images"), false, "not named, at its default");
	} finally {
		process.argv = argv;
	}
	// Not named and at its default: not given; the tier's value stands and is what the robot reads.
	const def = await toy({ tier: "S3" });
	assert.equal(def.refused, undefined);
	assert.equal(def.pi.getFlag("max-turns"), "1");
	assert.equal(trackedFlags(def.pi)?.given("max-turns"), false);
	// A value that is not the default is given however it was set (an SDK session's flag values).
	const five = await toy({ tier: "S3", "max-turns": "5" });
	assert.equal(five.refused, "--tier S3 sets --max-turns=1, but --max-turns=5 was given");
	assert.equal(trackedFlags(five.pi)?.given("max-turns"), true);
});

test("a flag given with another value than the tier's refuses to start, naming both; the same value is fine", async () => {
	const same = await toy({ tier: "S3", "code-api": "low", code: "pure", "max-turns": "1" });
	assert.equal(same.refused, undefined);
	const api = await toy({ tier: "S3", "code-api": "high" });
	assert.equal(api.refused, "--tier S3 sets --code-api=low, but --code-api=high was given");
	assert.deepEqual(api.active(), []);
	const both = await toy({ tier: "S3", code: "both" });
	assert.equal(both.refused, "--tier S3 sets --code=true, but --code=both was given");
	const priv = await toy({ tier: "S2", privileged: true }, { groundTruth: true });
	assert.equal(
		priv.refused,
		"--tier S2 runs without --privileged, but --privileged was given (S1 is S2 with the simulator's ground truth)",
	);
	const vdm = await toy({ tier: "M2", vdm: true }, { vdm: true });
	assert.equal(vdm.refused, "--tier M2 runs without --vdm, but --vdm was given");
	const turns = await toy({ tier: "M1", "max-turns": "1" });
	assert.equal(turns.refused, "--tier M1 is multi-turn, but --max-turns=1 was given");
	const frames = await toy({ tier: "M2", "keep-images": "0" });
	assert.equal(frames.refused, "--tier M2 feeds the camera frames back, but --keep-images=0 was given");
	const anchor = await toy({ tier: "S3", "anchor-image": true });
	assert.equal(anchor.refused, "--tier S3 runs without --anchor-image, but --anchor-image was given");
	const units = await toy({ tier: "S3", units: "true" }, { units: true });
	assert.equal(units.refused, "--tier S3 leaves --units at its default, but --units=true was given");
	const two = await toy({ tier: "S3", "code-api": "high", "keep-images": "4" });
	assert.equal(
		two.refused,
		"--tier S3 sets --code-api=low, but --code-api=high was given; --tier S3 sets --keep-images=0, but --keep-images=4 was given",
	);
	const unknown = await toy({ tier: "S9" });
	assert.equal(unknown.refused, "--tier S9: not a CaP-X tier; the tiers are S1, S2, S3, S4, M1, M2, M3, M4");
	const excl = await toy({ tier: "S3", preset: "capx-S3" });
	assert.equal(excl.refused, "--tier S3 and --preset capx-S3 are mutually exclusive: a preset is a tier of its own");
	// Without a choice nothing changes: the flags read as given.
	const plain = await toy({ code: "true", "code-api": "low" });
	assert.equal(plain.refused, undefined);
	const r = await result(plain);
	assert.equal(r.tier, undefined);
	assert.equal(r.axes, undefined);
	assert.equal(r.params.tier, "");
});

test("a robot that cannot serve the tier refuses to start", async () => {
	const nocode = await toy({ tier: "S3" }, { code: false });
	assert.equal(nocode.refused, "--tier S3 runs code mode (run_code), which toy does not mount");
	const low = await toy({ tier: "S2" }, { manifest: "toylow" });
	assert.equal(
		low.refused,
		"--tier S2 needs high-tier code primitives; toy's manifest has low (its tiers: S3, S4, M4)",
	);
	const s4 = await toy({ tier: "S4" }, { manifest: "toylow" });
	assert.equal(s4.refused, undefined);
	assert.deepEqual(s4.calls[0], { method: "code.api", kwargs: { tier: "low-noexamples" } });
	const real = await toy({ tier: "S1" });
	assert.equal(
		real.refused,
		"--tier S1 needs the simulator's ground truth (--privileged), which toy has none of: S1 is simulation only",
	);
	const novdm = await toy({ tier: "M4" });
	assert.equal(novdm.refused, "--tier M4 describes each change with VDM (--vdm), which toy does not mount");
});

// --- presets

type Chosen = Exclude<ReturnType<typeof choose>, undefined>;

test("the presets expand to each repository's native flags; capx-<tier> is that tier", () => {
	assert.deepEqual(Object.keys(PRESETS), ["showharness", "humanclaw", "rpent", "openeta", "xpolicylab"]);
	assert.deepEqual(
		PRESET_NAMES.slice(5),
		TIER_NAMES.map((n) => `capx-${n}`),
	);
	const multi = (api: string, feedback = "image") => ({ turns: "multi", feedback, api });
	assert.deepEqual(PRESETS.showharness.axes, multi("raw"));
	assert.deepEqual(PRESETS.showharness.flags, { units: "true", code: null, vdm: false, "vdm-video": false });
	assert.deepEqual(PRESETS.humanclaw.axes, multi("raw"));
	assert.deepEqual(PRESETS.humanclaw.flags, {
		units: "both",
		"humanclaw-mode": "paper",
		vdm: false,
		"vdm-video": false,
	});
	assert.deepEqual(PRESETS.rpent.axes, multi("low"));
	assert.deepEqual(PRESETS.rpent.flags, { code: null, units: null, vdm: false, "vdm-video": false, stateless: false });
	assert.deepEqual(PRESETS.openeta.axes, multi("low", "image+vdm"));
	assert.deepEqual(PRESETS.openeta.flags, { vdm: true, "anchor-image": true, code: null, units: null });
	assert.deepEqual(PRESETS.xpolicylab.axes, multi("policy"));
	assert.deepEqual(PRESETS.xpolicylab.requires, { flag: "xpolicy", hint: "ws://host:port (the policy server)" });
	// None of them pins --privileged: stacked, it is recorded as <preset>+privileged.
	for (const p of Object.values(PRESETS)) assert.equal(p.axes.privileged, undefined);
	const capx = choose("", "capx-M3") as Chosen;
	assert.ok(!("error" in capx));
	assert.equal(capx.kind, "tier");
	assert.equal(capx.name, "M3");
	assert.equal(capx.flag, "--preset capx-M3");
	assert.deepEqual(capx.flags, axesFlags(TIERS.M3));
	assert.match(
		String((choose("", "capx-S9") as { error: string }).error),
		/no such preset; the presets are showharness, humanclaw, rpent, openeta, xpolicylab, capx-S1/,
	);
	assert.match(String((choose("", "rlbench") as { error: string }).error), /--preset rlbench: no such preset/);
});

test("a preset needs what it stands for mounted", () => {
	const none = {
		robot: "toy",
		code: false,
		units: false,
		vdm: false,
		groundTruth: false,
		memory: false,
		xpolicy: false,
		vla: false,
		codeTiers: () => [],
	};
	const c = (n: string) => choose("", n) as Chosen;
	assert.equal(
		unsupported(c("showharness"), none),
		"--preset showharness drives the arm with Show-Harness action units (--units), which toy does not mount",
	);
	assert.equal(unsupported(c("showharness"), { ...none, units: true }), undefined);
	assert.equal(
		unsupported(c("humanclaw"), { ...none, units: true }),
		"--preset humanclaw is the HumanCLAW robot's paper setting (--units=both --humanclaw-mode paper); toy is not it",
	);
	assert.equal(unsupported(c("humanclaw"), { ...none, robot: "humanclaw" }), undefined);
	assert.equal(unsupported(c("rpent"), none), "--preset rpent plans with RPent's memory, which toy does not mount");
	assert.equal(
		unsupported(c("rpent"), { ...none, memory: true }),
		"--preset rpent calls a VLA skill (pi0_pick), which toy's manifest does not declare",
	);
	assert.equal(unsupported(c("rpent"), { ...none, memory: true, vla: true }), undefined);
	assert.equal(
		unsupported(c("openeta"), none),
		"--preset openeta describes each change with VDM (--vdm), which toy does not mount",
	);
	assert.equal(unsupported(c("openeta"), { ...none, vdm: true }), undefined);
	assert.equal(
		unsupported(c("xpolicylab"), none),
		"--preset xpolicylab runs an XPolicyLab policy (--xpolicy), which toy does not mount",
	);
	assert.equal(unsupported(c("xpolicylab"), { ...none, xpolicy: true }), undefined);
	assert.equal(unsupported(c("capx-S3"), { ...none, code: true, codeTiers: () => ["low"] }), undefined);
	assert.equal(
		unsupported(c("capx-S2"), { ...none, code: true, codeTiers: () => ["low"] }),
		"--preset capx-S2 needs high-tier code primitives; toy's manifest has low (its tiers: S3, S4, M4)",
	);
});

test("--preset reads as the repository's flags, records preset and axes, and refuses what the robot lacks", async () => {
	const sh = await toy({ preset: "showharness" }, { units: true });
	assert.equal(sh.refused, undefined);
	assert.deepEqual(sh.active(), ["act", "plan", "finish"]);
	assert.equal(sh.pi.getFlag("units"), "true");
	assert.equal(sh.flags.units, "false");
	const rs = await result(sh);
	assert.equal(rs.preset, "showharness");
	assert.equal(rs.tier, undefined);
	assert.deepEqual(rs.axes, { turns: "multi", feedback: "image", api: "raw", privileged: false });
	assert.equal(rs.params.preset, "showharness");
	assert.equal(rs.params.units, "true");
	assert.equal(rs.params.tier, "");
	const oe = await toy({ preset: "openeta" }, { vdm: true });
	assert.equal(oe.refused, undefined);
	assert.equal(oe.pi.getFlag("vdm"), true);
	assert.equal(oe.pi.getFlag("anchor-image"), true);
	const ro = await result(oe);
	assert.equal(ro.preset, "openeta");
	assert.equal(ro.vdm, true);
	assert.equal(ro.anchor_image, true);
	assert.deepEqual(ro.axes, { turns: "multi", feedback: "image+vdm", api: "low", privileged: false });
	// capx-S3 is the S3 tier, recorded as such; the messages name the spelling given.
	const capx = await toy({ preset: "capx-S3" });
	assert.equal(capx.refused, undefined);
	assert.deepEqual(capx.calls[0], { method: "code.api", kwargs: { tier: "low" } });
	const rc = await result(capx);
	assert.equal(rc.tier, "S3");
	assert.equal(rc.preset, undefined);
	assert.equal(rc.params.preset, "capx-S3");
	const clash = await toy({ preset: "capx-S3", "code-api": "high" });
	assert.equal(clash.refused, "--preset capx-S3 sets --code-api=low, but --code-api=high was given");
	// Ground truth stacked on a preset is a combination.
	const stacked = await toy({ preset: "showharness", privileged: true }, { units: true, groundTruth: true });
	assert.equal(stacked.refused, undefined);
	assert.equal((await result(stacked)).preset, "showharness+privileged");
	// Refusals: a module the preset stands for is not mounted, a flag contradicts it, the policy server is missing.
	const nounits = await toy({ preset: "showharness" });
	assert.equal(
		nounits.refused,
		"--preset showharness drives the arm with Show-Harness action units (--units), which toy does not mount",
	);
	const novdm = await toy({ preset: "openeta" });
	assert.equal(novdm.refused, "--preset openeta describes each change with VDM (--vdm), which toy does not mount");
	const notrpent = await toy({ preset: "rpent" });
	assert.equal(notrpent.refused, "--preset rpent plans with RPent's memory, which toy does not mount");
	const nothc = await toy({ preset: "humanclaw" }, { units: true });
	assert.equal(
		nothc.refused,
		"--preset humanclaw is the HumanCLAW robot's paper setting (--units=both --humanclaw-mode paper); toy is not it",
	);
	const both = await toy({ preset: "showharness", units: "both" }, { units: true });
	assert.equal(both.refused, "--preset showharness sets --units=true, but --units=both was given");
	const coded = await toy({ preset: "showharness", code: "true" }, { units: true });
	assert.equal(coded.refused, "--preset showharness leaves --code at its default, but --code=true was given");
	const nopolicy = await toy({ preset: "xpolicylab" }, { xpolicy: true });
	assert.equal(nopolicy.refused, "--preset xpolicylab needs --xpolicy ws://host:port (the policy server)");
	const unmounted = await toy({ preset: "xpolicylab", xpolicy: "ws://x:1" });
	assert.equal(
		unmounted.refused,
		"--preset xpolicylab runs an XPolicyLab policy (--xpolicy), which toy does not mount",
	);
	const unknown = await toy({ preset: "rlbench" });
	assert.match(
		String(unknown.refused),
		/^--preset rlbench: no such preset; the presets are showharness, humanclaw, rpent, openeta, xpolicylab, capx-S1, /,
	);
});

test("the eval scripts expand a preset as the robot does", async () => {
	const f = await toy({ preset: "showharness" }, { units: true });
	const r = await result(f);
	const dir = mkdtempSync(join(tmpdir(), "presets-"));
	try {
		const file = join(dir, "result.json");
		writeFileSync(file, JSON.stringify(r));
		const match = (args: string[]) => node("params-match.mjs", [file], { PI_ARGS_JSON: JSON.stringify(args) });
		assert.equal(match(["--preset", "showharness"]).status, 0);
		const spelled = match(["--units=true"]);
		assert.equal(spelled.status, 2);
		assert.match(spelled.stderr, /--preset ran "showharness", now ""/);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
	const oe = node("tier-flags.mjs", ["--preset", "openeta", "--model", "x/y"]);
	assert.deepEqual(oe.stdout.trim().split("\n"), ["--vdm=true", "--anchor-image=true"]);
	const hc = node("tier-flags.mjs", ["--preset=humanclaw"]);
	assert.deepEqual(hc.stdout.trim().split("\n"), ["--units=both", "--humanclaw-mode=paper"]);
	const bad = node("tier-flags.mjs", ["--preset", "showharness", "--units", "both"]);
	assert.equal(bad.status, 2);
	assert.match(bad.stderr, /--preset showharness sets --units=true, but --units=both was given/);
	const sh = evalOptions(["--preset", "showharness", "--model", "x/y"]);
	assert.equal(sh.status, 0, sh.stderr);
	assert.deepEqual(JSON.parse(sh.stdout).slice(0, 8), ["false", "", "0", "false", "false", "true", "", "showharness"]);
	const capx = evalOptions(["--preset=capx-S4"]);
	assert.deepEqual(JSON.parse(capx.stdout).slice(0, 8), [
		"true",
		"low-noexamples",
		"1",
		"false",
		"false",
		"false",
		"",
		"capx-S4",
	]);
	const excl = evalOptions(["--preset", "rpent", "--tier", "S3"]);
	assert.equal(excl.status, 2);
	assert.match(excl.stderr, /mutually exclusive/);
});

// --- the eval scripts

const SCRIPTS = new URL("../src/scripts/", import.meta.url).pathname;
const node = (script: string, args: string[], env: Record<string, string> = {}) =>
	spawnSync(process.execPath, [join(SCRIPTS, script), ...args], { encoding: "utf8", env: { ...process.env, ...env } });

test("params-match.mjs: a tier run matches --tier, not its flags spelled out, and the other way round", async () => {
	const f = await toy({ tier: "S3" });
	const r = await result(f);
	const dir = mkdtempSync(join(tmpdir(), "tiers-"));
	try {
		const tiered = join(dir, "tiered.json");
		writeFileSync(tiered, JSON.stringify(r));
		const match = (args: string[], file = tiered) =>
			node("params-match.mjs", [file], { PI_ARGS_JSON: JSON.stringify(args) });
		assert.equal(match(["--tier", "S3"]).status, 0);
		assert.equal(match(["--tier=S3", "--code-api", "low"]).status, 0, "the same value given alongside");
		const spelled = match(["--code=true", "--code-api=low", "--max-turns", "1", "--keep-images", "0"]);
		assert.equal(spelled.status, 2);
		assert.match(spelled.stderr, /--tier ran "S3", now ""/);
		const other = match(["--tier", "S2"]);
		assert.equal(other.status, 2);
		assert.match(other.stderr, /--code-api ran "low", now "high"/);
		assert.match(other.stderr, /--tier ran "S3", now "S2"/);
		// A run without a tier, asked for as --tier S3: another configuration (the tier is a recorded flag).
		const p = await toy({ code: "true", "code-api": "low", "max-turns": "1", "keep-images": "0" });
		const plain = join(dir, "plain.json");
		writeFileSync(plain, JSON.stringify(await result(p)));
		const asTier = match(["--tier", "S3"], plain);
		assert.equal(asTier.status, 2);
		assert.match(asTier.stderr, /--tier ran "", now "S3"/);
		assert.equal(match(["--code=true", "--code-api=low", "--max-turns", "1", "--keep-images", "0"], plain).status, 0);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

test("tier-flags.mjs prints the flags a tier stands for, and refuses a contradicting argument", () => {
	const s3 = node("tier-flags.mjs", ["--model", "x/y", "--tier", "S3", "--task", "0"]);
	assert.equal(s3.status, 0);
	assert.deepEqual(s3.stdout.trim().split("\n"), [
		"--code=true",
		"--code-api=low",
		"--max-turns=1",
		"--keep-images=0",
	]);
	const s1 = node("tier-flags.mjs", ["--tier=S1"]);
	assert.deepEqual(s1.stdout.trim().split("\n"), [
		"--code=true",
		"--code-api=high",
		"--max-turns=1",
		"--keep-images=0",
		"--privileged=true",
	]);
	const m3 = node("tier-flags.mjs", ["--tier", "M3", "--privileged"]);
	assert.deepEqual(m3.stdout.trim().split("\n"), ["--code=true", "--code-api=high", "--vdm=true", "--keep-images=0"]);
	assert.equal(node("tier-flags.mjs", ["--model", "x/y"]).stdout, "", "no choice: nothing");
	const bad = node("tier-flags.mjs", ["--tier", "S3", "--code-api", "high", "--vdm"]);
	assert.equal(bad.status, 2);
	assert.match(
		bad.stderr,
		/--tier S3 sets --code-api=low, but --code-api=high was given; --tier S3 runs without --vdm, but --vdm was given/,
	);
	const unknown = node("tier-flags.mjs", ["--tier", "S9"]);
	assert.equal(unknown.status, 2);
	assert.match(unknown.stderr, /not a CaP-X tier/);
	// A flag given at its registered default is a contradiction here as in the robot (the same parse).
	const zero = node("tier-flags.mjs", ["--tier", "S3", "--max-turns", "0"]);
	assert.equal(zero.status, 2);
	assert.match(zero.stderr, /^--tier S3 sets --max-turns=1, but --max-turns=0 was given$/m);
	// pi's rules: `--` ends the options (what follows is a message), an @file is not a flag's value.
	const after = node("tier-flags.mjs", ["--tier", "S3", "--", "--max-turns", "0"]);
	assert.equal(after.status, 0, after.stderr);
	assert.equal(node("tier-flags.mjs", ["--tier", "S3", "--privileged", "@notes.md"]).status, 2);
});

/** eval-options.sh over `args`: the parsed variables, or the exit code of a refused run. */
function evalOptions(args: string[]) {
	const dir = mkdtempSync(join(tmpdir(), "tiers-eval-"));
	try {
		const harness = join(dir, "parse.sh");
		writeFileSync(
			harness,
			`#!/usr/bin/env bash
set -uo pipefail
source "$1"
shift
eval_options_defaults
eval_parse_options "$@"
eval_normalize_options
node -e 'console.log(JSON.stringify(process.argv.slice(1)))' "$code" "$code_api" "$turns" "$vdm" "$privileged" "$units" "$tier" "$preset" "$@"
`,
		);
		return spawnSync("bash", [harness, join(SCRIPTS, "eval-options.sh"), ...args], { encoding: "utf8" });
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
}

test("eval-options.sh keys a --tier run by the flags it expands to, keeps pi's argv, and stops a contradicting one", () => {
	const s3 = evalOptions(["--tier", "S3", "--model", "x/y"]);
	assert.equal(s3.status, 0, s3.stderr);
	assert.deepEqual(JSON.parse(s3.stdout), [
		"true",
		"low",
		"1",
		"false",
		"false",
		"false",
		"S3",
		"",
		"--tier",
		"S3",
		"--model",
		"x/y",
	]);
	const s1 = evalOptions(["--tier=S1"]);
	assert.deepEqual(JSON.parse(s1.stdout).slice(0, 8), ["true", "high", "1", "false", "true", "false", "S1", ""]);
	const m3 = evalOptions(["--tier", "M3", "--max-turns", "15"]);
	assert.deepEqual(JSON.parse(m3.stdout).slice(0, 8), ["true", "high", "15", "true", "false", "false", "M3", ""]);
	const bad = evalOptions(["--tier", "S3", "--code-api=high"]);
	assert.equal(bad.status, 2);
	assert.match(bad.stderr, /--tier S3 sets --code-api=low, but --code-api=high was given/);
	const zero = evalOptions(["--tier", "S3", "--max-turns", "0"]);
	assert.equal(zero.status, 2);
	assert.match(zero.stderr, /--tier S3 sets --max-turns=1, but --max-turns=0 was given/);
	const none = evalOptions(["--code=both", "--code-api", "low"]);
	assert.deepEqual(JSON.parse(none.stdout).slice(0, 8), ["both", "low", "0", "false", "false", "false", "", ""]);
});
