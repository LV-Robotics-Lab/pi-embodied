import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import type { RpcClient } from "../src/infra/rpc.ts";
import { loadManifest } from "../src/primitives/manifest.ts";
import {
	CODE_API_ENTRY,
	CODE_API_EVENT,
	type CodeApi,
	type CodeApiPrimitive,
	codePrimitives,
	fetchCodeApi,
	renderCodeApi,
} from "../src/primitives/registry.ts";
import { defineRobot, RESULT_ENTRY, type RobotSpec } from "../src/robot.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi that runs handlers in registration order and records flags, entries and events. */
function stubPi(values: Record<string, unknown> = {}) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const entries: { type: string; data: any }[] = [];
	const events: { channel: string; data: unknown }[] = [];
	const notes: string[] = [];
	const api: Record<string, unknown> = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: { emit: (channel: string, data: unknown) => events.push({ channel, data }), on: () => () => {} },
	};
	const pi = new Proxy(api, { get: (t, k: string) => t[k] ?? (() => {}) }) as unknown as ExtensionAPI;
	const ctx = {
		hasUI: true,
		ui: { notify: (m: string) => notes.push(m) },
		sessionManager: { getBranch: () => [] },
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		for (const fn of handlers.get(name) ?? []) await fn({ type: name, ...event }, ctx);
	}
	return { pi, flags, entries, events, notes, emit };
}

const API: CodeApi = {
	tier: null,
	manifest_digest: "m".repeat(64),
	available: ["get_robot_state", "move_delta"],
	digest: "d".repeat(64),
};
const PRIMS: CodeApiPrimitive[] = [
	{
		name: "get_robot_state",
		method: "env.get_robot_state",
		doc: "The state.",
		params: {},
		mutating: false,
		tier: "low",
	},
	{
		name: "move_delta",
		method: "env.move_delta",
		doc: "Translate the TCP.\nMore.",
		params: {
			delta_xyz: { type: "vec3", description: "m", required: true },
			continuous: { type: "boolean", description: "", required: false },
		},
		mutating: true,
		tier: "low",
	},
];

/** An env client whose `code.api` answers `reply` (or throws it) and records the tier asked for. */
function server(reply: unknown | Error) {
	const asked: unknown[] = [];
	const client = {
		call: async (method: string, kwargs: Record<string, unknown>) => {
			assert.equal(method, "code.api");
			asked.push(kwargs.tier);
			if (reply instanceof Error) throw reply;
			return reply;
		},
	} as unknown as RpcClient;
	return { client, asked };
}

function robot(values: Record<string, unknown>, extra: Partial<RobotSpec>) {
	const f = stubPi(values);
	f.pi.registerFlag("seed", { type: "string", default: "0" });
	const r = defineRobot(f.pi, {
		name: "toy",
		task: ["seed"],
		keepImages: 1,
		start: async () => ["move"],
		result: () => ({ success: false }),
		finish: {
			description: "finish",
			parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
			result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
		},
		...extra,
	});
	return { ...f, robot: r };
}

const result = async (f: ReturnType<typeof stubPi>) => {
	await f.emit("agent_start");
	await f.emit("session_shutdown");
	return f.entries.find((e) => e.type === RESULT_ENTRY)?.data;
};

test("a robot with codeApi records the declaration, publishes it and names it in the result", async () => {
	const s = server(API);
	const f = robot({}, { codeApi: () => s.client });
	await f.emit("session_start");
	assert.deepEqual(s.asked, [undefined], "the default tier: high and low");
	assert.deepEqual(f.entries.find((e) => e.type === CODE_API_ENTRY)?.data, API);
	assert.deepEqual(f.robot.codeApi, API);
	assert.deepEqual(
		f.events.filter((e) => e.channel === CODE_API_EVENT).map((e) => e.data),
		[API],
	);
	const r = await result(f);
	assert.equal(r.code_api_digest, API.digest);
	assert.equal(r.code_api_tier, null);
});

test("--privileged asks for the privileged tier", async () => {
	const s = server({ ...API, tier: "privileged" });
	const f = robot({ privileged: true }, { codeApi: () => s.client, groundTruth: async () => ({}) });
	await f.emit("session_start");
	assert.deepEqual(s.asked, ["privileged"]);
	assert.equal((await result(f)).code_api_tier, "privileged");
});

test("a server without code.api starts the robot with no declaration; a malformed one fails closed", async () => {
	const old = server(new Error("code.api: unknown RPC method: 'code.api'"));
	const f = robot({}, { codeApi: () => old.client });
	await f.emit("session_start");
	assert.equal(f.robot.codeApi, undefined);
	assert.equal(
		f.entries.some((e) => e.type === CODE_API_ENTRY),
		false,
	);
	assert.equal("code_api_digest" in (await result(f)), false);

	const bad = server({ available: "move" });
	const g = robot({}, { codeApi: () => bad.client });
	await g.emit("session_start");
	assert.match(g.notes.join("\n"), /toy unavailable: code\.api: malformed reply/);
	assert.equal((await result(g)).env_error, true);
});

test("a robot without codeApi publishes no declaration", async () => {
	const f = robot({}, {});
	await f.emit("session_start");
	assert.deepEqual(
		f.events.filter((e) => e.channel === CODE_API_EVENT).map((e) => e.data),
		[undefined],
	);
	assert.equal("code_api_digest" in (await result(f)), false);
});

test("fetchCodeApi passes other errors on, and renderCodeApi lists one primitive per line", async () => {
	await assert.rejects(fetchCodeApi(server(new Error("env: connection refused")).client), /connection refused/);
	assert.equal(
		renderCodeApi(PRIMS),
		"- get_robot_state() — The state.\n- move_delta(delta_xyz: vec3, continuous?: boolean) — Translate the TCP. [moves the robot]",
	);
});

const ROBOTS = readdirSync(new URL("../src/primitives/manifests/", import.meta.url))
	.filter((f) => f.endsWith(".json"))
	.map((f) => f.slice(0, -".json".length));

test("every robot's low- and high-tier code primitives have an example, so S3 (low) and S4 (low-noexamples) differ", () => {
	assert.ok(ROBOTS.includes("libero") && ROBOTS.length > 10, ROBOTS.join(","));
	const example = (p: CodeApiPrimitive) => /\n\s*Examples?:/i.test(p.doc);
	for (const robot of ROBOTS) {
		const m = loadManifest(robot);
		for (const tier of ["high", "low"] as const)
			assert.deepEqual(
				codePrimitives(m, tier, () => true)
					.filter((p) => !example(p))
					.map((p) => p.name),
				[],
				`${robot} ${tier}`,
			);
		const s3 = codePrimitives(m, "low", () => true);
		const s4 = codePrimitives(m, "low-noexamples", () => true);
		assert.deepEqual(
			s4.map((p) => p.name),
			s3.map((p) => p.name),
			robot,
		);
		assert.deepEqual(
			s4.filter(example).map((p) => p.name),
			[],
			robot,
		);
	}
});
