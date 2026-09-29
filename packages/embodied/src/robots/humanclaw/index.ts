/**
 * HumanCLAW robot for pi: a full-body SMPL-X humanoid in HumanClawBench (41 HSSD homes, 1,218
 * find-navigate-(sit) episodes), driven by the units' own-vocabulary `act`.
 *
 *   pi -e packages/embodied/src/robots/humanclaw --units=both --episode one \
 *     --model humanclaw-psv/selfhost/muse-glimmer-30b "Solve the task."          (paper mode, default)
 *   pi -e packages/embodied/src/robots/humanclaw --units=both --humanclaw-mode pi --episode val100:3 \
 *     --model selfhost/muse-glimmer-30b "Solve the task."                        (pi's own tool loop)
 *
 * The env server (services/.../robots/humanclaw/env_server.py, the humanclaw venv) runs HumanCLAW's
 * own evaluator objects: Habitat + Half-Physics, the motion diffusion model, the paper metrics and
 * the replay/video writers. `act` maps a unit and its parameter to HumanCLAW's action name and its
 * SkillCall through a port of `_chooser_action` (./actions.ts, golden-tested against the Python),
 * records the decision (`env.record_decision`) and runs it (`env.step`); STOP is the terminal
 * Stop/Stand. `look` returns the current 448x448 ego image without acting (the first decision of
 * the evaluator sees the reset image). Once the episode is over (STOP, the step limit, or `finish`)
 * `env.finish` writes metrics.json, the trajectories and, with --humanclaw-video, ego/exo MP4, and
 * its summary goes into `robot_result` (FindSR, NavSR@20cm/@1m, InteractSR, collision, jerk, cost).
 *
 * --humanclaw-mode paper: the planner is the `humanclaw-psv/<base>` model (./provider.ts), HumanCLAW's
 * prompt v4 + verifier v3 byte for byte; the robot's SYSTEM.md, VDM, memory and verifier plug-ins are
 * refused. pi: the model plans with our SYSTEM.md and gives `target_visible` with every `act`, which
 * is FindSR's subjective acknowledgement. --privileged adds ground_truth_poses (target centres and
 * the humanoid root pose), recorded as `humanclaw+privileged`.
 *
 * Copyright 2026 The HumanCLAW Authors (github.com/Human-CLAW/HumanCLAW @c4f9351).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: evaluation/evaluator.py's loop split between this robot and its env server.
 */

import { tmpdir } from "node:os";
import { join } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { cudaDevice, python, servicesDir } from "../../infra/config.ts";
import { trackFlags } from "../../infra/params.ts";
import { encodePng } from "../../infra/png.ts";
import type { NdArray, RpcClient } from "../../infra/rpc.ts";
import { template } from "../../planner/context-version.ts";
import { attach, defineRobot } from "../../robot.ts";
import { KEYS, skillCall, toJson, UNITS } from "./actions.ts";
import { DECISION_EVENT, mountPsv, psvBase } from "./provider.ts";
import type { Decision } from "./psv.ts";

const SYSTEM = template(new URL("./SYSTEM.md", import.meta.url));
export const VIEWS =
	"Each result shows your head-view (ego) image, 448x448. Your blue body and feet are at the bottom-centre edge of the image: that edge is your near-body position.";

type Obs = {
	ego: NdArray;
	instruction: string;
	step: number;
	max_steps: number;
	done: boolean;
	stopped: boolean;
	episode: {
		scene_id: string;
		episode_id: string;
		object_category: string;
		rollout: number;
		key: string;
		output_dir: string;
	};
	action_text?: string;
	collision?: Record<string, unknown>;
	proprioception?: Record<string, number>;
};
type Summary = {
	metrics: Record<string, unknown> | null;
	steps: number;
	active_stop: boolean;
	videos: string[];
	metrics_path: string | null;
};

export default function humanclaw(pi: ExtensionAPI) {
	// Every flag below is tracked: one owner each, numbers checked, recorded as params (../../infra/params.ts).
	trackFlags(pi);
	const flag = (name: string, fallback: string) => String(pi.getFlag(name) ?? fallback);
	pi.registerFlag("episode", {
		type: "string",
		default: "one",
		description: "one | val100:<i> | fullval:<i> | <list.json>:<i> | <scene_id>_ep<episode_id>_<category>",
	});
	pi.registerFlag("rollout", { type: "string", default: "0", description: "Rollout index (output rollout_NN)" });
	pi.registerFlag("humanclaw-mode", {
		type: "string",
		default: "paper",
		description: "paper: HumanCLAW's planner + verifier (--model humanclaw-psv/<base>); pi: pi's own tool loop",
	});
	pi.registerFlag("humanclaw-proprioception", {
		type: "boolean",
		default: false,
		description:
			"pi mode experiment: report actual relative body motion (different observation setting from paper mode)",
	});
	pi.registerFlag("humanclaw-metrics", {
		type: "boolean",
		default: false,
		description: "Paper metrics (metrics.json)",
	});
	pi.registerFlag("humanclaw-video", {
		type: "boolean",
		default: false,
		description: "Server-side ego.mp4 + exo.mp4",
	});
	pi.registerFlag("humanclaw-output", {
		type: "string",
		default: process.env.HUMANCLAW_OUTPUT ?? join(tmpdir(), "pi-embodied-humanclaw"),
		description: "Rollout artifacts (HumanCLAW's <scene>_ep<id>_<category>/rollout_NN layout)",
	});
	pi.registerFlag("humanclaw-max-steps", {
		type: "string",
		default: "",
		description: "Override max steps (smoke only)",
	});
	pi.registerFlag("humanclaw-max-tokens", {
		type: "string",
		default: "4096",
		description: "Paper mode: max_tokens per request",
	});
	pi.registerFlag("humanclaw-reasoning", {
		type: "string",
		default: "",
		description: "Paper mode: the base model's reasoning level (default: the model's own, as the paper ran)",
	});
	pi.registerFlag("humanclaw-json-format", {
		type: "boolean",
		default: true,
		description: "Paper mode: request a JSON object (response_format)",
	});
	pi.registerFlag("scene-dataset-config", { type: "string", default: "", description: "Prepared HSSD scene config" });
	pi.registerFlag("env-url", {
		type: "string",
		description: "Attach to a running env server instead of starting one",
	});

	const base = psvBase();
	if (base) mountPsv(pi, base);
	const mode = () => flag("humanclaw-mode", "paper");
	const proprioception = () => pi.getFlag("humanclaw-proprioception") === true;

	let env: RpcClient;
	let obs: Obs;
	let summary: Summary | undefined;
	let decision: Decision | undefined;
	let usage: { prompt_tokens: number; completion_tokens: number }[] = [];
	pi.events.on(DECISION_EVENT, (d) => {
		decision = d as Decision;
	});
	// pi mode: the turn's token usage goes with its decision (the paper's cost columns).
	pi.on("message_end", (event) => {
		const m = event.message;
		if (m.role === "assistant" && mode() === "pi")
			usage.push({ prompt_tokens: m.usage?.input ?? 0, completion_tokens: m.usage?.output ?? 0 });
	});

	const robot = defineRobot(pi, {
		name: "humanclaw",
		// Tools and code primitives: ../../primitives/manifests/humanclaw.json (the env server reads it too).
		manifest: "humanclaw",
		task: ["episode", "rollout"],
		keepImages: 4,
		video: true,
		vdm: { views: 1 },
		codeApi: () => env,
		groundTruth: () => env.call("env.ground_truth_poses", {}, 60_000, [], robot.signal),
		start: startEpisode,
		prompt: () =>
			mode() === "pi"
				? SYSTEM.replaceAll("{{task}}", obs?.instruction ?? "") +
					(proprioception()
						? "\nSelf-motion feedback reports measured movement, not the requested motion. turned_left_deg is positive for left turns and negative for right turns. Use measured turns to keep the seat behind you; a requested 120 degree turn may execute only partially. height_from_start_m is relative to the initial root, not seat height. Small displacement after a walking action signals a blocked approach; choose a different route. A lower root alone does not prove that you are sitting on the target. Track the measured turning and sitting phase with plan."
						: "")
				: "The HumanCLAW planner (humanclaw-psv) drives this episode.",
		result: () => ({
			mode: mode(),
			humanclaw_proprioception: proprioception(),
			preset: pi.getFlag("privileged") === true ? "humanclaw+privileged" : "humanclaw",
			episode: obs?.episode ?? null,
			rollout: Number(flag("rollout", "0")),
			instruction: obs?.instruction ?? null,
			steps: obs?.step ?? 0,
			active_stop: obs?.stopped ?? false,
			success: Boolean((summary?.metrics as { nav_sr_20cm?: boolean } | null)?.nav_sr_20cm),
			metrics: summary?.metrics ?? null,
			metrics_path: summary?.metrics_path ?? null,
			videos: summary?.videos ?? [],
		}),
		status: () => ({ language: obs.instruction, step: obs.step, solved: obs.stopped }),
		finish: {
			description:
				"End the episode. Success is measured by the benchmark (distance, contact), not by this call; STOP first when the task is done.",
			parameters: Type.Object({ status: StringEnum(["success", "failure"] as const), summary: Type.String() }),
			result: (params) => ({
				content: [{ type: "text", text: `Episode finished (stopped=${obs?.stopped ?? false}).` }],
				details: params,
			}),
		},
		units: {
			instruction: () => obs.instruction,
			views: VIEWS,
			wrist: false,
			gripper: false,
			plugins: ["plan", "proprioception"],
			state: async () => ({ step: obs.step, max_steps: obs.max_steps }),
			vocabulary: {
				units: UNITS,
				keys: KEYS,
				targetVisible: true,
				// GUMI's first look (and the operator's): the current ego image, no step.
				observe: async () => observe({}),
				run: async (unit, param, signal, extra) => observe(await run(unit, param, signal, extra?.target_visible)),
			},
		},
	});

	/** One decision: record it (the planner's JSON, or pi's target_visible), then run it. */
	async function run(unit: string, param: string | number | undefined, signal?: AbortSignal, targetVisible?: boolean) {
		if (obs.done) throw new Error("the episode is over; call finish");
		const call = skillCall(unit, param);
		let d = decision;
		decision = undefined;
		if (d && (d.action.skill !== call.skill || JSON.stringify(d.action.cond) !== JSON.stringify(call.cond)))
			throw new Error(`act ${unit}(${param}) is not the planner's decision ${d.action.action_name}`);
		if (!d) {
			const seen = targetVisible === true;
			d = {
				raw_plan: { visual_state_description: "", target_visible: targetVisible ?? null },
				action: call,
				// FindSR's subjective rule reads visible_state: a sentence naming the target without negation.
				planner_skill: {
					visible_state: seen ? "The target is visible." : "",
					target_visible: targetVisible ?? null,
				},
				verifier: {},
				stages: usage.map((u) => ({ stage: "percept_mid_low", raw: {}, raw_output: "", prompt: "", usage: u })),
			};
		}
		usage = [];
		const s = signal ?? robot.signal;
		await env.call("env.record_decision", { decision: { ...d, action: toJson(d.action) } }, 120_000, [], s);
		obs = await env.call<Obs>(
			"env.step",
			{
				skill: call.skill,
				cond: call.cond,
				action_name: call.action_name,
				action_id: call.action_id,
				reasoning: d.raw_plan,
			},
			600_000,
			[],
			s,
		);
		robot.video.frame(obs.ego);
		if (obs.done) await close();
		return {
			unit,
			param,
			action: call.action_name,
			collision: obs.collision ?? {},
			...(proprioception() ? { proprioception: obs.proprioception ?? {} } : {}),
		};
	}

	/** env.finish once: metrics, trajectories, videos; Habitat closes. */
	async function close() {
		if (summary || !env || !obs) return;
		summary = await env.call<Summary>("env.finish", {}, 600_000);
	}
	// `finish` before the episode is over (no STOP): the rollout still ends and is measured.
	pi.on("tool_call", async (event) => {
		if (event.toolName === "finish") await close();
		return undefined;
	});

	function observe(result: Record<string, unknown>) {
		const png = encodePng(obs.ego.data, obs.ego.shape[1], obs.ego.shape[0]);
		const details = {
			...(proprioception() ? { proprioception: obs.proprioception ?? {} } : {}),
			result,
			step: obs.step,
			max_steps: obs.max_steps,
			done: obs.done,
			stopped: obs.stopped,
			terminated: obs.done,
			instruction: obs.instruction,
			task_language: obs.instruction,
		};
		return {
			content: [
				{ type: "text" as const, text: JSON.stringify({ ...details, ...(summary ? { finished: true } : {}) }) },
				{ type: "image" as const, data: png.toString("base64"), mimeType: "image/png" },
			],
			details: summary ? { ...details, summary } : details,
		};
	}

	robot.tool("look", "The current head-view (ego) image, without acting.", Type.Object({}), async () => observe({}));

	async function startEpisode() {
		summary = undefined;
		decision = undefined;
		usage = [];
		const m = mode();
		if (m !== "paper" && m !== "pi") throw new Error(`--humanclaw-mode ${m}: paper or pi`);
		if (m === "paper" && proprioception())
			throw new Error(
				"--humanclaw-proprioception is a pi-mode experiment; paper mode keeps the published observations",
			);
		if (String(pi.getFlag("units") ?? "") !== "both")
			throw new Error("HumanCLAW runs with --units=both (the units' act plus the robot's look)");
		if (m === "paper") {
			if (!base) throw new Error("--humanclaw-mode paper needs --model humanclaw-psv/<base>");
			// auto is off on one body; only an explicit true turns the finish check on.
			const on = (f: string) => pi.getFlag(f) === true || pi.getFlag(f) === "true";
			for (const f of ["vdm", "units-verify", "explore"])
				if (on(f)) throw new Error(`--${f} is a pi-mode module; paper mode runs HumanCLAW's planner as published`);
		} else if (base) throw new Error("--humanclaw-mode pi plans with the model's own tools, not humanclaw-psv");
		const endpoint = pi.getFlag("env-url") as string | undefined;
		if (endpoint) env = await attach(endpoint);
		else {
			const services = servicesDir(pi);
			const steps = flag("humanclaw-max-steps", "");
			const scenes = flag("scene-dataset-config", "");
			env = await robot.serve({
				python: python(pi, "humanclaw", ["HUMANCLAW_PYTHON"]),
				args: [
					...["-m", "pi_embodied_services.robots.humanclaw.env_server"],
					...["--output-root", flag("humanclaw-output", ""), "--cuda-device", cudaDevice(pi) || "0"],
					...(pi.getFlag("humanclaw-metrics") ? ["--metrics"] : []),
					...(pi.getFlag("humanclaw-video") ? ["--video"] : []),
					...(steps ? ["--max-steps", steps] : []),
					...(scenes ? ["--scene-dataset-config", scenes] : []),
				],
				cwd: services,
				env: { ...process.env, PYTHONPATH: services },
				log: (port) => join(tmpdir(), `pi-embodied-humanclaw-${port}.log`),
				readyMs: 600_000,
			});
		}
		obs = await env.call<Obs>(
			"env.reset",
			{ episode: robot.task.episode, rollout: Number(robot.task.rollout), proprioception: proprioception() },
			900_000,
		);
		return ["look", "finish"];
	}
}
