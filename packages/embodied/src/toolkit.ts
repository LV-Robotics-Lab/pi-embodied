/**
 * Embodied toolkit: turns robot tools into pi tools with a fixed execution contract.
 *
 * - Tools run one at a time (`executionMode: "sequential"`); a robot has one body.
 * - After every mutating tool the robot state is captured and returned to the
 *   model (text + camera images), so the planner always reasons over what the
 *   world looks like after its action rather than over what the tool claims.
 * - When a mutating tool fails, the state is still captured and returned with
 *   the error: the robot may have moved before the failure.
 * - `finish` ends the run but never decides success: that comes from the
 *   environment (`robot.solved()`), never from the agent's own claim. Every
 *   result in the batch that contains `finish` terminates, so the run stops
 *   after that batch; calls after `finish` are blocked.
 * - Every call is appended to the episode trace, and every captured frame is
 *   written to disk for later inspection.
 *
 * Batch termination, post-finish blocking, and the error flag on captured
 * failures rely on the hooks installed by `episodeExtension`.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import type { AgentToolResult, ToolDefinition } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";

export interface Observation {
	/** JSON-serializable state shown to the model (proprioception, flags, step). */
	state: Record<string, unknown>;
	/** Camera frames, PNG-encoded. */
	images: { name: string; png: Buffer }[];
}

export interface ToolRunContext {
	signal?: AbortSignal;
	episode: Episode;
}

export interface EmbodiedTool<P extends TSchema = TSchema> {
	name: string;
	description: string;
	parameters: P;
	/** Read-only tools skip the post-action state capture. */
	readonly?: boolean;
	run(params: Static<P>, ctx: ToolRunContext): Promise<{ result: Record<string, unknown>; observation?: Observation }>;
}

export function embodiedTool<P extends TSchema>(tool: EmbodiedTool<P>): EmbodiedTool {
	return tool as unknown as EmbodiedTool;
}

export interface TaskInfo {
	taskId: number;
	seed: number;
	language: string;
	[key: string]: unknown;
}

export interface Robot {
	name: string;
	tools: EmbodiedTool[];
	/** Configure and reset the environment for one episode. */
	begin(taskId: number, seed: number): Promise<TaskInfo>;
	/** Capture the current state and camera frames. */
	observe(): Promise<Observation>;
	/** Ground-truth success, read from the environment. */
	solved(): boolean;
	systemPrompt(task: TaskInfo): string;
	userPrompt(task: TaskInfo): string;
}

export interface TraceEntry {
	index: number;
	tool: string;
	args: unknown;
	result?: Record<string, unknown>;
	error?: string;
	frames?: string[];
	elapsedMs: number;
	solvedAfter: boolean;
}

export class Episode {
	readonly trace: TraceEntry[] = [];
	finish?: { status: string; summary: string };
	/** Camera frames currently replaced by markers in the model context. */
	imagesPruned = 0;
	/** The current assistant tool batch contains `finish`; its results terminate the run. */
	endsAfterBatch = false;
	/** Tool calls that failed but returned a state capture; reported to the model as errors. */
	readonly failedCalls = new Set<string>();
	readonly outDir: string | undefined;
	private frameIndex = 0;

	constructor(outDir: string | undefined) {
		this.outDir = outDir;
		if (outDir) mkdirSync(join(outDir, "frames"), { recursive: true });
	}

	saveFrames(obs: Observation): string[] {
		const index = this.frameIndex++;
		if (!this.outDir) return [];
		return obs.images.map(({ name, png }) => {
			const rel = join("frames", `${String(index).padStart(3, "0")}_${name}.png`);
			writeFileSync(join(this.outDir as string, rel), png);
			return rel;
		});
	}

	writeTrace(): void {
		if (this.outDir) writeFileSync(join(this.outDir, "trace.json"), `${JSON.stringify(this.trace, null, 2)}\n`);
	}
}

function observationContent(obs: Observation, extra: Record<string, unknown>): (TextContent | ImageContent)[] {
	const text: TextContent = {
		type: "text",
		text: JSON.stringify({ ...extra, state: obs.state, images: obs.images.map((i) => i.name) }),
	};
	const images: ImageContent[] = obs.images.map(({ png }) => ({
		type: "image",
		data: png.toString("base64"),
		mimeType: "image/png",
	}));
	return [text, ...images];
}

const FinishParams = Type.Object({
	status: Type.Union([Type.Literal("success"), Type.Literal("failure")], {
		description: "Your own assessment. The environment's success signal is what counts.",
	}),
	summary: Type.String({ description: "What you did and why you believe the task is done or cannot be done." }),
});

/** Build the pi tool definitions for one episode. */
export function buildPiTools(robot: Robot, episode: Episode): ToolDefinition[] {
	const tools: ToolDefinition[] = robot.tools.map((tool) => ({
		name: tool.name,
		label: tool.name,
		description: tool.description,
		parameters: tool.parameters,
		executionMode: "sequential",
		async execute(id, params, signal): Promise<AgentToolResult<unknown>> {
			const entry: TraceEntry = {
				index: episode.trace.length,
				tool: tool.name,
				args: params,
				elapsedMs: 0,
				solvedAfter: false,
			};
			episode.trace.push(entry);
			const terminate = episode.endsAfterBatch;
			const started = Date.now();
			try {
				let run: Awaited<ReturnType<typeof tool.run>>;
				try {
					run = await tool.run(params, { signal, episode });
				} catch (err) {
					entry.error = err instanceof Error ? err.message : String(err);
					if (tool.readonly) throw err;
					let obs: Observation;
					try {
						obs = await robot.observe();
					} catch {
						throw err;
					}
					episode.failedCalls.add(id);
					entry.solvedAfter = robot.solved();
					entry.frames = episode.saveFrames(obs);
					return {
						content: observationContent(obs, { error: entry.error }),
						details: { error: entry.error, state: obs.state },
						terminate,
					};
				}
				const { result, observation } = run;
				const obs = observation ?? (tool.readonly ? undefined : await robot.observe());
				entry.result = result;
				entry.solvedAfter = robot.solved();
				if (!obs) {
					return { content: [{ type: "text", text: JSON.stringify(result) }], details: result, terminate };
				}
				entry.frames = episode.saveFrames(obs);
				return { content: observationContent(obs, { result }), details: { result, state: obs.state }, terminate };
			} catch (err) {
				entry.error ??= err instanceof Error ? err.message : String(err);
				throw err;
			} finally {
				entry.elapsedMs = Date.now() - started;
				episode.writeTrace();
			}
		},
	}));

	tools.push({
		name: "finish",
		label: "finish",
		description:
			"End the episode. Call exactly once, after verifying the final state in the latest images. Success is judged by the environment, not by this call.",
		parameters: FinishParams,
		executionMode: "sequential",
		async execute(_id, params: Static<typeof FinishParams>): Promise<AgentToolResult<unknown>> {
			episode.finish = { status: params.status, summary: params.summary };
			episode.trace.push({
				index: episode.trace.length,
				tool: "finish",
				args: params,
				elapsedMs: 0,
				solvedAfter: robot.solved(),
			});
			episode.writeTrace();
			return { content: [{ type: "text", text: "Episode finished." }], details: params, terminate: true };
		},
	});
	return tools;
}
