/**
 * End-to-end runner tests: a fake in-memory Robot driven by pi's faux LLM provider.
 * No network, no keys; every assistant turn is scripted.
 */

import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type Api,
	type FauxProviderHandle,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxProvider,
	fauxToolCall,
	InMemoryCredentialStore,
	type Model,
	type ToolResultMessage,
} from "@earendil-works/pi-ai";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type RunOptions, runBatch, type Summary } from "../src/runner.ts";
import type { TraceEntry } from "../src/toolkit.ts";
import { FakeRobot } from "./fake-robot.ts";

interface Fixture {
	faux: FauxProviderHandle;
	model: Model<Api>;
	modelRuntime: ModelRuntime;
	out: string;
}

let fixture: Fixture;

beforeEach(async () => {
	const faux = fauxProvider();
	const modelRuntime = await ModelRuntime.create({
		credentials: new InMemoryCredentialStore(),
		modelsPath: null,
		allowModelNetwork: false,
	});
	modelRuntime.registerNativeProvider(faux.provider);
	const model = modelRuntime.getModel(faux.provider.id, faux.getModel().id);
	if (!model) throw new Error("faux model not registered");
	fixture = { faux, model, modelRuntime, out: mkdtempSync(join(tmpdir(), "pi-embodied-runner-")) };
});

afterEach(() => {
	rmSync(fixture.out, { recursive: true, force: true });
});

function options(overrides: Partial<RunOptions> = {}): RunOptions & { tasks: number[]; seeds: number[] } {
	return {
		thinking: "off",
		maxTurns: 10,
		timeoutS: 20,
		retryInfra: 0,
		imageBudgetMb: 4,
		out: fixture.out,
		tasks: [0],
		seeds: [0],
		...overrides,
	};
}

async function run(robot: FakeRobot, responses: FauxResponseStep[], overrides: Partial<RunOptions> = {}) {
	fixture.faux.setResponses(responses);
	const logs: string[] = [];
	const summary = await runBatch(robot, fixture.model, fixture.modelRuntime, options(overrides), (line) =>
		logs.push(line),
	);
	return { summary, logs };
}

function readJson<T>(...parts: string[]): T {
	return JSON.parse(readFileSync(join(fixture.out, ...parts), "utf8")) as T;
}

const move = (target: string) => fauxToolCall("move", { target });
const finish = (status: "success" | "failure" = "success") => fauxToolCall("finish", { status, summary: "done" });

describe("runBatch with a fake robot", () => {
	// Guards the basic contract: a scripted move + finish ends with `finish`,
	// success is read from robot.solved(), and summary.json is on disk.
	it("records an environment-judged success", async () => {
		const robot = new FakeRobot();
		const { summary } = await run(robot, [
			fauxAssistantMessage(move("goal"), { stopReason: "toolUse" }),
			fauxAssistantMessage(finish(), { stopReason: "toolUse" }),
		]);

		expect(summary.results).toHaveLength(1);
		const [result] = summary.results;
		expect(result.stopReason).toBe("finish");
		expect(result.success).toBe(true);
		expect(result.claimed).toBe("success");
		expect(summary.successes).toBe(1);
		expect(summary.episodes).toBe(1);

		const onDisk = readJson<Summary>("summary.json");
		expect(onDisk.successes).toBe(1);
		const episodeDir = join(fixture.out, "fake_t0_s0");
		expect(existsSync(join(episodeDir, "result.json"))).toBe(true);
		expect(existsSync(join(episodeDir, "trace.json"))).toBe(true);
		expect(existsSync(join(episodeDir, "frames"))).toBe(true);
		expect(fixture.faux.getPendingResponseCount()).toBe(0);
	});

	// Guards that `finish` in a parallel batch terminates the run after that
	// batch: the move still runs, and no follow-up LLM request is made.
	it("ends after a batch containing finish without another LLM request", async () => {
		const robot = new FakeRobot();
		const spare = fauxAssistantMessage("should never be requested");
		const { summary } = await run(robot, [
			fauxAssistantMessage([move("goal"), finish()], { stopReason: "toolUse" }),
			spare,
		]);

		const [result] = summary.results;
		expect(result.stopReason).toBe("finish");
		expect(result.success).toBe(true);
		expect(robot.moves).toBe(1);
		expect(fixture.faux.state.callCount).toBe(1);
		expect(fixture.faux.getPendingResponseCount()).toBe(1);
	});

	// Guards that tools after `finish` in the same batch are blocked: the model
	// cannot claim success and then act, and the claim is judged against the
	// unchanged environment (claimedButFailed).
	it("blocks tools called after finish in the same batch", async () => {
		const robot = new FakeRobot();
		const { summary } = await run(robot, [
			fauxAssistantMessage([finish(), move("goal")], { stopReason: "toolUse" }),
			fauxAssistantMessage("should never be requested"),
		]);

		const [result] = summary.results;
		expect(robot.moves).toBe(0);
		expect(result.success).toBe(false);
		expect(result.claimed).toBe("success");
		expect(result.stopReason).toBe("finish");
		expect(summary.claimedButFailed).toBe(1);
		expect(fixture.faux.state.callCount).toBe(1);
	});

	// Guards that a failing mutating tool still shows the model the world after
	// the failure: the tool result is an error carrying camera images, and the
	// trace records both the error and the saved frames.
	it("returns camera frames with the error when a mutating tool throws", async () => {
		const robot = new FakeRobot();
		let errorResult: ToolResultMessage | undefined;
		const { summary } = await run(robot, [
			fauxAssistantMessage(move("wall"), { stopReason: "toolUse" }),
			(context) => {
				errorResult = context.messages.find(
					(m): m is ToolResultMessage => m.role === "toolResult" && m.toolName === "move",
				);
				return fauxAssistantMessage(finish("failure"), { stopReason: "toolUse" });
			},
		]);

		expect(errorResult).toBeDefined();
		expect(errorResult?.isError).toBe(true);
		expect(errorResult?.content.some((part) => part.type === "image")).toBe(true);
		expect(
			errorResult?.content.some((part) => part.type === "text" && part.text.includes("collision: hit the wall")),
		).toBe(true);

		const trace = readJson<TraceEntry[]>("fake_t0_s0", "trace.json");
		const moveEntry = trace.find((e) => e.tool === "move");
		expect(moveEntry?.error).toContain("collision: hit the wall");
		expect(moveEntry?.frames?.length).toBe(1);
		const frame = moveEntry?.frames?.[0] ?? "";
		expect(existsSync(join(fixture.out, "fake_t0_s0", frame))).toBe(true);

		const [result] = summary.results;
		expect(result.stopReason).toBe("finish");
		expect(result.success).toBe(false);
		expect(result.claimed).toBe("failure");
		expect(summary.claimedButFailed).toBe(0);
	});

	// Guards env_error handling: a `begin` failure is retried, never reaches the
	// LLM, is excluded from scored episodes, and only the final attempt per
	// task/seed is kept in results.
	it("excludes environment failures from scoring and retries them", async () => {
		const robot = new FakeRobot({ failBegin: true });
		const { summary } = await run(robot, [], { retryInfra: 1 });

		expect(robot.beginCalls).toBe(2);
		expect(fixture.faux.state.callCount).toBe(0);
		expect(summary.results).toHaveLength(1);
		const [result] = summary.results;
		expect(result.stopReason).toBe("env_error");
		expect(result.attempt).toBe(1);
		expect(result.error).toContain("ECONNREFUSED");
		expect(summary.envErrors).toBe(1);
		expect(summary.infraErrors).toBe(0);
		expect(summary.episodes).toBe(0);
		expect(summary.successRate).toBe(0);

		const onDisk = readJson<Summary>("summary.json");
		expect(onDisk.envErrors).toBe(1);
		expect(onDisk.episodes).toBe(0);
		expect(existsSync(join(fixture.out, "fake_t0_s0", "result.json"))).toBe(true);
		expect(existsSync(join(fixture.out, "fake_t0_s0_retry1", "result.json"))).toBe(true);
	});
});
