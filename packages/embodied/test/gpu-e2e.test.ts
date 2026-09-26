/**
 * GPU end-to-end suite: real simulators and model servers, no model API. Opt-in and skipped
 * everywhere else: it runs only with PI_EMBODIED_E2E=<robot>[,<robot>...] and an nvidia-smi that
 * answers. Each robot loads in a stub pi (flags as on the command line), starts its real env server
 * from PI_EMBODIED_PYTHON / PI_EMBODIED_SERVICES, and the test drives its tools the way the planner
 * would: one Show-Harness unit (`act MV_UP`), a status check that the env stepped, `finish`, and the
 * result entry. With SAM3_CHECKPOINT_PATH set, LIBERO also starts SAM3 itself (`--serve-models sam3`,
 * under PI_EMBODIED_GPU_LOCK) and segments its first frame. `test/gpu-e2e.sh` runs it robot by robot
 * with each one's environment (docs/adding-a-robot.md, "Testing on a GPU").
 */

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import genesis from "../src/genesis/index.ts";
import libero from "../src/libero/index.ts";
import maniskill from "../src/maniskill/index.ts";
import metaworld from "../src/metaworld/index.ts";
import robosuite from "../src/robosuite/index.ts";
import { RESULT_ENTRY, SERVICES, STATUS_EVENT } from "../src/robot.ts";
import { RpcClient } from "../src/rpc.ts";

type Handler = (event: any, ctx: any) => unknown;

const selected = (process.env.PI_EMBODIED_E2E ?? "")
	.split(",")
	.map((s) => s.trim())
	.filter(Boolean);
const gpu = (() => {
	if (!selected.length) return false;
	try {
		execFileSync("nvidia-smi", ["-L"], { stdio: "ignore" });
		return true;
	} catch {
		return false;
	}
})();
const skip = (robot: string) =>
	!selected.includes(robot)
		? `PI_EMBODIED_E2E does not name ${robot}`
		: !gpu
			? "no GPU (nvidia-smi)"
			: !process.env.PI_EMBODIED_PYTHON
				? "PI_EMBODIED_PYTHON is unset"
				: false;

/** One robot loaded like `pi -e <robot> <flags>`, with the handlers, tools, entries and status it publishes. */
function load(robot: (pi: ExtensionAPI) => void, values: Record<string, unknown>) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	const status: any[] = [];
	let active: string[] = [];
	const api: Record<string, unknown> = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: {
			emit: (channel: string, data: any) => {
				if (channel === STATUS_EVENT) status.push(data);
			},
			on: () => () => {},
		},
	};
	const pi = new Proxy(api, { get: (t, k: string) => t[k] ?? (() => {}) }) as unknown as ExtensionAPI;
	const dir = mkdtempSync(join(tmpdir(), "gpu-e2e-"));
	const errors: string[] = [];
	const ctx = {
		hasUI: true,
		cwd: dir,
		ui: new Proxy({ notify: (msg: string) => errors.push(msg) }, { get: (t: any, k: string) => t[k] ?? (() => {}) }),
		shutdown: () => {},
		abort: () => {},
		sessionManager: {
			getBranch: () => [],
			getSessionFile: () => join(dir, "session.jsonl"),
			getSessionDir: () => dir,
			getSessionId: () => "e2e",
		},
	};
	robot(pi);
	const emit = async (name: string, event: Record<string, unknown> = {}) => {
		for (const fn of handlers.get(name) ?? []) await fn({ type: name, ...event }, ctx);
	};
	const call = async (name: string, params: Record<string, unknown>) => {
		assert.ok(active.includes(name), `${name} is not active: ${active.join(", ")}`);
		return tools.get(name).execute(`e2e-${name}`, params, undefined, undefined, ctx);
	};
	return { emit, call, entries, status, errors, active: () => active, dir };
}

/** The common flags: the services checkout and an empty local memory corpus (no download). */
function flags(extra: Record<string, unknown>) {
	const dir = mkdtempSync(join(tmpdir(), "gpu-e2e-mem-"));
	writeFileSync(join(dir, "MEMORY.md"), "# GPU e2e\n");
	return {
		python: process.env.PI_EMBODIED_PYTHON,
		services: process.env.PI_EMBODIED_SERVICES ?? SERVICES,
		"memory-profile": "local",
		"memory-dir": dir,
		...(process.env.PI_EMBODIED_E2E_CUDA ? { "cuda-device": process.env.PI_EMBODIED_E2E_CUDA } : {}),
		...extra,
	};
}

const text = (r: any) => (r.content ?? []).map((c: any) => c.text ?? "").join("\n");

/** Start the robot, run one MV_UP unit, check that the env stepped, finish, and read the result entry. */
async function episode(robot: (pi: ExtensionAPI) => void, name: string, task: Record<string, unknown>) {
	const r = load(robot, flags({ ...task, units: "true" }));
	try {
		await r.emit("session_start", { reason: "startup" });
		assert.deepEqual(r.errors, [], `${name} did not start`);
		assert.ok(r.active().includes("act"), r.active().join(", "));
		const before = r.status.at(-1)?.step ?? 0;
		await r.emit("before_agent_start", { prompt: "Solve the task.", systemPrompt: "" });
		await r.emit("agent_start");
		const acted = await r.call("act", { unit: "MV_UP" });
		assert.ok(!acted.isError, text(acted));
		await r.emit("tool_execution_end", { toolName: "act" });
		const after = r.status.at(-1);
		assert.equal(after.ready, true);
		assert.ok((after.step ?? 0) > before, `the env did not step: ${before} -> ${after.step}`);
		await r.call("finish", { status: "failure", summary: "e2e: one unit" });
		await r.emit("message_end", {
			message: {
				role: "assistant",
				stopReason: "toolUse",
				content: [{ type: "toolCall", name: "finish" }],
				usage: { cost: { total: 0 } },
			},
		});
		await r.emit("agent_end", { messages: [] });
		const result = r.entries.find((e) => e.type === RESULT_ENTRY)?.data;
		assert.ok(result, "no robot_result entry");
		assert.equal(result.robot, name);
		assert.equal(result.env_error, false, JSON.stringify(result));
		assert.equal(result.claimed, "failure");
		return result;
	} finally {
		await r.emit("session_shutdown", { reason: "quit" });
	}
}

test("metaworld: reach-v3 starts, steps a unit and reports", { skip: skip("metaworld") }, async () => {
	await episode(metaworld, "metaworld", { task: "reach-v3", seed: "0" });
});

test("robosuite: Lift starts, steps a unit and reports", { skip: skip("robosuite") }, async () => {
	await episode(robosuite, "robosuite", { task: "Lift", seed: "0" });
});

test("genesis: cube_pick starts, steps a unit and reports", { skip: skip("genesis") }, async () => {
	await episode(genesis, "genesis", { task: "cube_pick", seed: "0" });
});

test("maniskill: PickCube-v1 starts, steps a unit and reports", { skip: skip("maniskill") }, async () => {
	await episode(maniskill, "maniskill", { "env-id": "PickCube-v1", seed: "0" });
});

test("libero: libero_10 task 0 starts, steps a unit and reports", { skip: skip("libero") }, async () => {
	await episode(libero, "libero", { suite: "libero_10", task: "0", seed: "0" });
});

async function freePort(): Promise<number> {
	const s = createServer();
	await new Promise<void>((r) => s.listen(0, "127.0.0.1", r));
	const port = (s.address() as { port: number }).port;
	await new Promise((r) => s.close(r));
	return port;
}

test(
	"libero --serve-models sam3: pi starts SAM3 under the GPU lock, segments with it, and stops it",
	{
		skip: skip("libero") || (!process.env.SAM3_CHECKPOINT_PATH && "SAM3_CHECKPOINT_PATH is unset"),
		timeout: 1_800_000,
	},
	async () => {
		const port = await freePort();
		const sam3 = `http://127.0.0.1:${port}`;
		const r = load(
			libero,
			flags({
				suite: "libero_10",
				task: "0",
				seed: "0",
				units: "false",
				sam3,
				"serve-models": "sam3",
				"serve-min-free": process.env.PI_EMBODIED_E2E_MIN_FREE ?? "6000",
				"serve-timeout": "1500",
			}),
		);
		try {
			await r.emit("session_start", { reason: "startup" });
			assert.deepEqual(r.errors, [], "libero did not start");
			await new RpcClient(sam3).call("healthz", {}, 5_000);
			const seg = await r.call("segment", { prompt: "the black bowl" });
			assert.ok(!seg.isError, text(seg));
		} finally {
			await r.emit("session_shutdown", { reason: "quit" });
		}
		await assert.rejects(new RpcClient(sam3).call("healthz", {}, 3_000));
	},
);
