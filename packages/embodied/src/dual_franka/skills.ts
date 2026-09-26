/**
 * Named dual-Franka VLA skills (./index.ts): `vla_right_grasp`, `vla_handoff` and
 * `vla_left_place` roll out the Pi0.5 VLA (--robot-vla) chunk by chunk and stop at the skill's
 * semantic boundary (a gripper event followed by a lift or a settle delay). The task config owns
 * the policy instruction; the planner's prompt is only recorded.
 */

import { Type } from "typebox";
import { f32, type Json, numbers, roundAll, u8, vec } from "../robot.ts";
import { NdArray, type RpcClient } from "../rpc.ts";
import type { RegisterTool, Setup } from "./config.ts";

type Boundary = "grasp" | "handoff" | "place";

/** What the skills drive on the robot. */
export type SkillDeps = {
	tool: RegisterTool;
	/** The operator gate and abort check before each chunk and action (throws to refuse). */
	check: (signal?: AbortSignal) => void;
	/** An env RPC call. */
	call: (method: string, kwargs: Json, timeoutMs: number, signal?: AbortSignal) => Promise<Json>;
	/** The live observation (env.get_observation); records its states. */
	observation: () => Promise<Json>;
	/** The robot state (env.get_robot_state) with the last policy state vector. */
	robotState: () => Promise<Json>;
	/** Record the policy state vector of a step result. */
	remember: (states: unknown) => void;
	setup: () => Setup | undefined;
	/** The VLA client (--robot-vla), if configured. */
	vla: () => RpcClient | undefined;
};

/** Register the three named VLA skills, in the order above. */
export function mountSkills(d: SkillDeps) {
	const { tool, check, call, observation, robotState, remember } = d;

	async function namedSkill(
		skill: string,
		boundary: Boundary,
		prompt: string,
		maxChunks: number,
		signal?: AbortSignal,
	) {
		if (!d.vla()) throw new Error(`${skill} requires --robot-vla`);
		const requested = String(prompt).trim();
		if (!requested) throw new Error("prompt must be non-empty");
		// Task configuration owns policy conditioning; the planner's intent is only recorded.
		const setup = d.setup();
		const effective = setup?.task.vla_instruction || setup?.task.instruction || "";
		if (!(maxChunks >= 1 && maxChunks <= 20)) throw new Error("max_chunks must be between 1 and 20");

		let state = await robotState();
		const start = state;
		const open = (s: Json, a: string) => Boolean(s[`${a}_arm`]?.gripper_open);
		const z = (s: Json, a: string) => vec(s[`${a}_arm`]?.tcp_pose)[2];
		let prevLeft = open(state, "left");
		let prevRight = open(state, "right");
		let eventZ: number | null = null;
		let eventTime: number | null = null;
		let eventStep: number | null = null;
		let after = 0;
		let chunks = 0;
		let stepsDone = 0;
		let reached = false;
		let terminated = false;
		let truncated = false;
		let count = 0;
		let min: number[] = [];
		let max: number[] = [];
		let sum: number[] = [];
		let firstAction: number[] | undefined;
		let firstState: number[] | undefined;
		let firstStateShape: number[] | undefined;
		const t0 = performance.now();

		rollout: for (let c = 0; c < maxChunks; c++) {
			check(signal);
			const obs = await observation();
			const main = obs.main_images;
			const extras = obs.extra_view_images;
			const states = obs.states;
			if (firstState === undefined) {
				firstStateShape = states instanceof NdArray ? states.shape : [vec(states).length];
				firstState = roundAll(vec(states).slice(0, 20));
			}
			if (!(main instanceof NdArray) || main.shape.length !== 3)
				throw new Error(`expected [H,W,3] image, got shape [${main?.shape}]`);
			if (!(extras instanceof NdArray) || extras.shape.length !== 4)
				throw new Error(`extra_view_images must be [N,H,W,3]; got shape [${extras?.shape}]`);
			if (!(states instanceof NdArray) || states.shape.length !== 1)
				throw new Error(`states must be single-env shape [state_dim]; got [${states?.shape}]`);
			const wire = {
				main_images: u8(main).batched(),
				wrist_images: null,
				extra_view_images: u8(extras).batched(),
				states: f32(states).batched(),
				task_descriptions: [effective],
			};
			const predicted = await (d.vla() as RpcClient).call<NdArray>(
				"vla.predict",
				{},
				120_000,
				[wire, { mode: "eval" }],
				signal,
			);
			const actions = new NdArray(predicted.dtype, predicted.shape.slice(1), predicted.data);
			if (actions.shape.length !== 2 || actions.shape[1] !== 20)
				throw new Error(`${skill} expected [chunk, 20] actions, got [${actions.shape}]`);
			const flat = numbers(actions);
			if (!flat.every(Number.isFinite)) throw new Error(`${skill} received non-finite VLA actions`);
			chunks++;
			const rows = Array.from({ length: actions.shape[0] }, (_, i) => flat.slice(i * 20, (i + 1) * 20));
			firstAction ??= roundAll(rows[0]);
			for (const row of rows) {
				min = count ? min.map((v, k) => Math.min(v, row[k])) : [...row];
				max = count ? max.map((v, k) => Math.max(v, row[k])) : [...row];
				sum = count ? sum.map((v, k) => v + row[k]) : [...row];
				count++;
			}

			for (const row of rows) {
				check(signal);
				const result = await call(
					"env.chunk_step",
					{ actions: NdArray.f32(row, [1, 20]), return_all_frames: false },
					300_000,
					signal,
				);
				const next = result.observation;
				remember(Array.isArray(next) ? next.at(-1)?.states : next?.states);
				stepsDone++;
				terminated ||= Boolean(result.terminated);
				truncated ||= Boolean(result.truncated);
				state = await robotState();
				const left = open(state, "left");
				const right = open(state, "right");
				if (boundary === "grasp") {
					if (eventZ === null && prevRight && !right) {
						eventZ = z(state, "right");
						eventStep = stepsDone;
						after = 0;
					} else if (eventZ !== null) {
						after++;
						reached = !right && after >= 2 && z(state, "right") - eventZ >= 0.15;
					}
				} else if (boundary === "handoff") {
					if (eventTime === null && !prevRight && right) {
						eventTime = performance.now();
						eventStep = stepsDone;
						after = 0;
					} else if (eventTime !== null) {
						after++;
						reached = right && after >= 2 && (performance.now() - eventTime) / 1000 >= 1.5;
					}
				} else if (eventZ === null && !prevLeft && left) {
					eventZ = z(state, "left");
					eventStep = stepsDone;
					after = 0;
				} else if (eventZ !== null) {
					after++;
					reached = left && after >= 2 && z(state, "left") - eventZ >= 0.1;
				}
				prevLeft = left;
				prevRight = right;
				if (reached || terminated || truncated) break rollout;
			}
		}

		const stopRule: Json = {
			phase: boundary,
			skill_name: skill,
			step_count: stepsDone,
			event_step: eventStep,
			success_claim: false,
		};
		if (boundary === "grasp")
			Object.assign(stopRule, {
				condition: "right_gripper_closed_then_lifted",
				right_close_step: eventStep,
				right_close_z: eventZ,
				right_current_z: z(state, "right"),
				lift_m: eventZ === null ? null : z(state, "right") - eventZ,
				threshold_m: 0.15,
			});
		else if (boundary === "handoff")
			Object.assign(stopRule, {
				condition: "right_gripper_opened_then_delay",
				right_open_step: eventStep,
				elapsed_after_open_s: eventTime === null ? null : (performance.now() - eventTime) / 1000,
				delay_s: 1.5,
				right_current_open: open(state, "right"),
			});
		else
			Object.assign(stopRule, {
				condition: "left_gripper_opened_then_lifted",
				left_open_step: eventStep,
				left_open_z: eventZ,
				left_current_z: z(state, "left"),
				lift_m: eventZ === null ? null : z(state, "left") - eventZ,
				threshold_m: 0.1,
			});
		const summary: Json = { count, shape: [count, 20], finite: true };
		if (count)
			Object.assign(summary, {
				min: roundAll(min),
				max: roundAll(max),
				mean: roundAll(sum.map((v) => v / count)),
				first_action: firstAction,
				first_policy_state_shape: firstStateShape,
				first_policy_state: firstState,
			});
		return {
			ok: reached && !(terminated || truncated),
			skill_name: skill,
			boundary,
			requested_prompt: requested,
			effective_policy_prompt: effective,
			prompt_overridden: requested !== effective,
			boundary_reached: reached,
			chunks_executed: chunks,
			steps_executed: stepsDone,
			terminated,
			truncated,
			stop_rule: stopRule,
			action_summary: summary,
			elapsed_s: (performance.now() - t0) / 1000,
			vla_start_robot_state: start,
			robot_state: state,
		};
	}

	const skillParams = Type.Object({
		prompt: Type.String({
			description:
				"Planner-facing segment intent. This is recorded in the tool result; the current live clean-desk checkpoint still receives its fixed training instruction during policy inference.",
		}),
		max_chunks: Type.Optional(Type.Integer({ minimum: 1, maximum: 20, description: "Default 20" })),
	});
	for (const [name, boundary, description] of [
		[
			"vla_right_grasp",
			"grasp",
			"Run the learned right-grasp VLA segment. The active task prompt decides which object is currently allowed; this tool only defines the capability boundary: right gripper closes and the right TCP lifts.",
		],
		[
			"vla_handoff",
			"handoff",
			"Run the learned bimanual handoff VLA segment. The capability boundary is right-gripper release followed by the configured settle delay; do not rule-base pre-position either arm for it.",
		],
		[
			"vla_left_place",
			"place",
			"Run the learned left-placement VLA segment. The active task decides the destination; this tool only defines the capability boundary: left gripper opens and the left TCP lifts.",
		],
	] as const)
		tool(name, description, skillParams, (p, signal) =>
			namedSkill(name, boundary, p.prompt, p.max_chunks ?? 20, signal),
		);
}
