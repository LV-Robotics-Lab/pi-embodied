/**
 * In-memory Robot for runner tests: no RPC, deterministic state, one tiny camera.
 */

import { Type } from "typebox";
import { encodePng } from "../src/png.ts";
import { NdArray } from "../src/rpc.ts";
import { embodiedTool, type Observation, type Robot, type TaskInfo } from "../src/toolkit.ts";

export interface FakeRobotOptions {
	/** Make `begin` fail like an unreachable environment server. */
	failBegin?: boolean;
}

export class FakeRobot implements Robot {
	readonly name = "fake";
	readonly tools: Robot["tools"];
	/** Number of `move` calls whose body actually ran (including ones that threw). */
	moves = 0;
	beginCalls = 0;
	position = "start";
	failBegin: boolean;

	constructor(options: FakeRobotOptions = {}) {
		this.failBegin = options.failBegin ?? false;
		this.tools = [
			embodiedTool({
				name: "move",
				description: "Move the robot to a named target.",
				parameters: Type.Object({ target: Type.String() }),
				run: async ({ target }) => {
					this.moves++;
					if (target === "wall") throw new Error("collision: hit the wall");
					this.position = target;
					return { result: { at: target } };
				},
			}),
			embodiedTool({
				name: "look",
				description: "Report the current position.",
				parameters: Type.Object({}),
				readonly: true,
				run: async () => ({ result: { at: this.position } }),
			}),
		];
	}

	async begin(taskId: number, seed: number): Promise<TaskInfo> {
		this.beginCalls++;
		if (this.failBegin) throw new Error("ECONNREFUSED env down");
		this.position = "start";
		return { taskId, seed, language: "move to the goal" };
	}

	async observe(): Promise<Observation> {
		const pixels = Buffer.from([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]);
		return {
			state: { at: this.position },
			images: [{ name: "front", png: encodePng(new NdArray("uint8", [2, 2, 3], pixels)) }],
		};
	}

	solved(): boolean {
		return this.position === "goal";
	}

	systemPrompt(task: TaskInfo): string {
		return `You control a fake robot. Task: ${task.language}`;
	}

	userPrompt(task: TaskInfo): string {
		return task.language;
	}
}
