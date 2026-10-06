/** Durable episode scheduling for eval-parallel.sh. Robot eval.sh owns result validation and reruns. */
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { createModels } from "@earendil-works/pi-ai/models";
import {
	createRegistry,
	defineDoc,
	defineExtension,
	defineTask,
	Harness,
	type TaskId,
} from "@earendil-works/pi-durable";
import { openNodeSqliteStorage } from "@earendil-works/pi-durable/storage/sqlite/node";

const context = BACKGROUND_CONTEXT;
const [directory, workerCount] = process.argv.slice(2);
const workers = Number(workerCount);
const script = process.env.PI_EVAL_WORKER_SCRIPT;
if (!directory || !Number.isSafeInteger(workers) || workers < 1 || !script) {
	throw new Error("eval-durable.ts must be launched by eval-parallel.sh --durable");
}
const jobs = readFileSync(join(directory, "jobs"), "utf8").trim().split("\n");
// The executable is supplied anew, never loaded from the database. Refuse changed inputs on resume.
const environment = Object.fromEntries(
	[
		"PI",
		"PI_EMBODIED_PYTHON",
		"PI_EMBODIED_SERVICES",
		"PI_EMBODIED_MEMORY_REVISION",
		"HUMANCLAW_PYTHON",
		"TIME_LIMIT",
		"MAX_TURNS",
		"TARGET50",
		"TASKS",
		"SEEDS",
		"SCENES",
	].map((key) => [key, process.env[key] ?? null]),
);
const signature = createHash("sha256").update(JSON.stringify({ jobs, workers, script, environment })).digest("hex");
const Run = defineDoc<{ signature: string; task: TaskId<number> | null; attempt: number }>({
	kind: "embodied.eval.run",
	version: 1,
	scope: "conversation",
	history: "latest",
	fork: "initial",
	initial: () => ({ signature: "", task: null, attempt: 0 }),
});

const Episode = defineTask<{ index: number; worker: number }, { phase: "execute" }, number>({
	name: "embodied.eval.episode",
	version: 1,
	initial: () => ({ phase: "execute" }),
	phases: {
		execute: async (task, runtime, taskContext) => {
			// The shell guards against surviving worker groups before opening this database. A replay
			// calls eval.sh again: it validates and skips valid results, or starts a fresh episode.
			const code = await new Promise<number>((resolve, reject) => {
				const child = spawn(
					"bash",
					[
						"-c",
						`${script}\nworker "$1" "$2"`,
						"eval-durable",
						String(task.input.worker),
						String(task.input.index),
					],
					{
						stdio: "inherit",
						signal: runtime.signal,
					},
				);
				child.once("error", reject);
				child.once("close", (exitCode) => resolve(exitCode ?? 1));
			});
			await runtime.commit(
				() => ({ status: "terminal", outcome: { status: "completed", result: code } }),
				taskContext,
			);
		},
	},
	abort: async (_task, runtime, taskContext) => {
		await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), taskContext);
	},
});

type LaneState = { phase: "next"; index: number } | { phase: "collect"; index: number; child: TaskId<number> };
const Lane = defineTask<{ worker: number }, LaneState, number>({
	name: "embodied.eval.lane",
	version: 1,
	initial: ({ worker }) => ({ phase: "next", index: worker + 1 }),
	phases: {
		next: async (task, runtime, taskContext) => {
			const { index } = task.state.checkpoint;
			await runtime.commit(async (tx) => {
				if (index > jobs.length) return { status: "terminal", outcome: { status: "completed", result: 0 } };
				const child = await tx.createTask(
					Episode,
					{ index, worker: task.input.worker },
					{ ownership: { kind: "task", taskId: task.id } },
				);
				return {
					status: "waiting",
					checkpoint: { phase: "collect", index, child },
					on: [child],
					policy: "allSettled",
				};
			}, taskContext);
		},
		collect: async (task, runtime, taskContext) => {
			const state = task.state.checkpoint;
			if (state.phase !== "collect") throw new Error("missing episode checkpoint");
			const [outcome] = await runtime.outcomes([state.child], taskContext);
			const code = outcome.status === "completed" ? outcome.result : 1;
			await runtime.commit(
				() =>
					code === 0
						? { status: "running", checkpoint: { phase: "next", index: state.index + workers } }
						: { status: "terminal", outcome: { status: "completed", result: code } },
				taskContext,
			);
		},
	},
	abort: async (_task, runtime, taskContext) => {
		await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), taskContext);
	},
});

type BatchState = { phase: "start" } | { phase: "collect"; lanes: TaskId<number>[] };
const Batch = defineTask<Record<string, never>, BatchState, number>({
	name: "embodied.eval.batch",
	version: 1,
	initial: () => ({ phase: "start" }),
	phases: {
		start: async (task, runtime, taskContext) => {
			await runtime.commit(async (tx) => {
				const lanes: TaskId<number>[] = [];
				for (let worker = 0; worker < Math.min(workers, jobs.length); worker++) {
					lanes.push(await tx.createTask(Lane, { worker }, { ownership: { kind: "task", taskId: task.id } }));
				}
				return { status: "waiting", checkpoint: { phase: "collect", lanes }, on: lanes, policy: "allSettled" };
			}, taskContext);
		},
		collect: async (task, runtime, taskContext) => {
			const state = task.state.checkpoint;
			if (state.phase !== "collect") throw new Error("missing worker checkpoints");
			const outcomes = await runtime.outcomes(state.lanes, taskContext);
			const code = outcomes.every((o) => o.status === "completed" && o.result === 0) ? 0 : 1;
			await runtime.commit(
				() => ({ status: "terminal", outcome: { status: "completed", result: code } }),
				taskContext,
			);
		},
	},
	abort: async (_task, runtime, taskContext) => {
		await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), taskContext);
	},
});

const registry = createRegistry();
registry.install(defineExtension({ name: "embodied-eval", tasks: [Batch, Lane, Episode] }));
const harness = await Harness.open(
	await openNodeSqliteStorage(join(directory, "durable.sqlite")),
	{ models: createModels(), registry },
	context,
);
let stopping = false;
const stop = () => {
	stopping = true;
	// Close preserves unfinished checkpoints; abortTask would make them terminal instead.
	void harness.close(context);
};
for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"] as const) process.on(signal, stop);
try {
	const root = await harness.root(context);
	const id = await root.commit(async (tx) => {
		const run = await tx.doc(Run, root.id);
		const previous = run.task ? await tx.task(run.task) : undefined;
		if (previous && previous.state.status !== "terminal") {
			if (run.signature !== signature)
				throw new Error(
					"unfinished durable run has different arguments or worker settings; resume the original command or use another out dir",
				);
			return run.task as TaskId<number>;
		}
		// Every fresh invocation revalidates completed artifacts through the original evaluator.
		run.signature = signature;
		run.attempt++;
		run.task = await tx.createTask(Batch, {}, { ownership: { kind: "conversation" } });
		return run.task;
	}, context);
	const graph = await harness.taskGraph(context);
	const statusPath = join(directory, "durable-status.json");
	let outcome: { status: string; code: number } | null = null;
	const writeStatus = () => {
		writeFileSync(
			`${statusPath}.tmp`,
			`${JSON.stringify({ task: id, jobs: jobs.length, workers, outcome, graph: graph.value }, null, 2)}\n`,
		);
		renameSync(`${statusPath}.tmp`, statusPath);
	};
	writeStatus();
	const unsubscribe = graph.subscribe(writeStatus);
	try {
		const result = await harness.waitForTask(id, context);
		process.exitCode = result.state.outcome.status === "completed" ? result.state.outcome.result : 1;
		outcome = { status: result.state.outcome.status, code: process.exitCode };
		writeStatus();
	} finally {
		unsubscribe();
		graph.dispose();
	}
} catch (error) {
	if (!stopping) console.error(error instanceof Error ? error.message : String(error));
	process.exitCode = stopping ? 130 : 2;
} finally {
	await harness.close(context);
	for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"] as const) process.off(signal, stop);
}
