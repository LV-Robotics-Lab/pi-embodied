/**
 * The units module's public contract: the robot's `UnitsSpec`, the `UnitsHandle` it publishes on
 * `UNITS_EVENT`, the base's tool registrar, the plan and point records, and the session entry names.
 *
 * Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: see ./index.ts.
 */

import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { Static, TSchema } from "typebox";
import type { Move, MoveUnit, Plugin, RtAxis, State, Vec3 } from "./vocabulary.ts";

/** `pi.events` channel on which this module publishes the robot's `UnitsHandle` at every session start. */
export const UNITS_EVENT = "pi-embodied:units";
export type UnitsHandle = {
	/** The agent's unit tool (`act`). */
	tool: string;
	/** Arm names on a dual-arm robot (`act`'s `arm`), [] on one arm. */
	arms: readonly string[];
	/** Units `act` accepts (a robot's own vocabulary: its unit names). */
	vocabulary: readonly string[];
	/** A robot's own vocabulary: GUMI key bindings, KeyboardEvent.code -> the unit as recorded ("TURN(45)"). */
	keys?: Record<string, string>;
	/**
	 * A robot's own vocabulary: the names of its units that end the episode (its STOP). GUMI never looks
	 * with one and records it as the episode's last step; the arm's STOP (a hold) is not one.
	 */
	terminal?: readonly string[];
	stepM: number;
	yawStepRad?: number;
	/** What one `act` call does (grounding, `apply`, recovery / auto_release, the units header), without the model. */
	/** `operator: true` for a human's unit (GUMI): exactly what was pressed, no recovery/auto_release/variable step/rotation assists. */
	run: (
		params: { unit: string; n?: number; arm?: string; other?: string; param?: string | number; operator?: boolean },
		signal?: AbortSignal,
	) => Promise<AgentToolResult<unknown>>;
	/**
	 * The current observation without a training step: the arm's STOP (hold one step), or a custom
	 * vocabulary's `observe`; undefined when the robot has no non-terminal way to look.
	 */
	look?: (signal?: AbortSignal) => Promise<AgentToolResult<unknown>>;
	/** The robot's proprioception (`eef_xyz`, `gripper_width`, ...), per arm on two arms. */
	state?: (arm?: string) => Promise<Record<string, unknown>>;
	/** Every robot tool (`act` and the robot's own): ../gumi holds them all while the operator drives. */
	tools: () => readonly string[];
	/** Why an operator's unit may not run now (the gates an `act` call passes), else undefined. */
	refuse: () => string | undefined;
	/** view_select is on this session: `act` takes `view` and passes it on as `Move.view`. */
	viewSelect?: boolean;
	/** --units-rt is on: the turns are RT_* (those in `vocabulary`) instead of ROTATE_*. */
	rt?: boolean;
	/** Whether this session's robot configuration has a wrist view (UnitsSpec.wrist), read once the robot started. */
	wrist?: () => boolean;
	/** The plugins that run this session (after --units-plugins and the wrist view). */
	plugins?: () => readonly string[];
	/** How the images look and which way each MV_* unit moves in them (the robot's VIEWS text). */
	guide?: () => string;
	/** The camera images an observation carries and which are wrist views (the robot's `vdm` spec), if it says. */
	views?: () => { views: number; wrist?: number | readonly number[] } | undefined;
};

export type Result = AgentToolResult<unknown>;

/**
 * One unit of a robot's own action vocabulary (UnitsSpec.vocabulary): its name, what it does (the
 * prompt), an optional parameter, and whether running it ends the episode.
 */
export type CustomUnit = {
	/** The unit's name, e.g. "WALK". */
	name: string;
	/** What it does, for the prompt. */
	description: string;
	/**
	 * Its parameter: an enum of values, or a number clamped to [min, max] (a value out of range is
	 * clamped and the result says so). Without a default the parameter is required.
	 */
	param?: {
		name: string;
		kind: "enum" | "number";
		values?: readonly string[];
		min?: number;
		max?: number;
		unit?: string;
		default?: string | number;
	};
	/** Running it ends the episode (e.g. STOP): never repeated, and no unit runs after it. */
	terminal?: boolean;
};

/** A robot's own action vocabulary: its units, how one runs, and optional GUMI key bindings. */
export type CustomVocabulary = {
	units: readonly CustomUnit[];
	/** Run one unit with its checked parameter through the robot's own safety checks; the new observation. */
	run: (
		unit: string,
		param: string | number | undefined,
		signal: AbortSignal | undefined,
		extra?: { target_visible?: boolean },
	) => Promise<Result>;
	/** `act` also takes `target_visible` (is the TARGET in the image?), passed to `run` (HumanCLAW's FindSR acknowledgement). */
	targetVisible?: boolean;
	/**
	 * The current observation without acting (GUMI's first look). Without it GUMI cannot take an
	 * observation before the first step: a terminal unit is never used to look.
	 */
	observe?: (signal: AbortSignal | undefined) => Promise<Result>;
	/** GUMI key bindings (KeyboardEvent.code -> the unit and its parameter). */
	keys?: Record<string, { unit: string; param?: string | number }>;
};

/** What only an arm driven by the MV_* vocabulary declares (optional with a custom `vocabulary`). */
export type ArmMotion = {
	/** Base-frame unit vector of each MV_* unit (calibrated so each matches its look in VIEWS). */
	vectors: Record<MoveUnit, Vec3>;
	/** Metres per MV_* unit (Show-Harness: 0.02). */
	stepM: number;
	/** Execute one move through the robot's own safety checks; return the new observation (images + state). */
	apply: (move: Move, signal: AbortSignal | undefined) => Promise<Result>;
};

/**
 * A robot's units: the arm's MV_* vocabulary (`vectors`, `stepM` and `apply`, all required then), or
 * its own `vocabulary` (a humanoid's WALK / TURN / SIT ...), in which case the arm plugins are off and
 * the three arm fields are unused (./index.ts refuses a spec with neither).
 */
export type UnitsSpec = UnitsSpecCommon & Partial<ArmMotion> & { vocabulary?: CustomVocabulary };

export type UnitsSpecCommon = {
	/** Radians per ROTATE_CW (ROTATE_CCW is the negative); omit on robots without yaw. */
	yawStepRad?: number;
	/**
	 * Dual-arm robots: execute one move per arm at the same time (Show-Harness's dual runners step both
	 * arms together), through the same safety checks; return the new observation. Without it a paired
	 * `act` step (`other`) runs the arms one after the other through `apply`.
	 */
	applyPair?: (moves: Move[], signal: AbortSignal | undefined) => Promise<Result>;
	/**
	 * Proprioception. The plugins read `eef_xyz` (base frame, m), `gripper_width` (m) and
	 * `table_z` (m, optional); everything else is shown as-is.
	 */
	state?: (arm?: string) => Promise<State>;
	/** The task text for the prompt (default: the episode's task flags). */
	instruction?: () => string;
	/** How the camera images look and which way each MV_* unit moves in them (default: DEFAULT_VIEWS). */
	views?: string;
	/**
	 * Dual-arm robots: the arm names `act` chooses between. A function is read when the session starts
	 * (after pi has set the flags), for a robot whose arm set depends on a flag such as the task.
	 */
	arms?: readonly string[] | (() => readonly string[] | undefined);
	/** A closed gripper at or below this width (m) holds nothing (recovery, auto_release). */
	emptyWidthM?: number;
	/** Default of --units-plugins. */
	plugins?: readonly Plugin[];
	/**
	 * Whether the robot's observations carry a wrist view (default true), a function when the
	 * configuration decides it (a `--robot` flag, the server's cameras). Read at load, at session start
	 * and again once the robot started. Without one, variable_step and action_chunk are off whatever
	 * --units-plugins says, rotation keeps only its realign (no wrist-judged compensation), `act` takes
	 * no `target_in_wrist` and the prompt drops its wrist-view text.
	 */
	wrist?: boolean | (() => boolean);
	/**
	 * Whether the robot has a gripper (default true), a function when the configuration decides it (a
	 * `--robot` flag). Without one (a stick) GRASP and RELEASE are not units, recovery and auto_release
	 * are off whatever --units-plugins says, and the prompt drops its gripper text. Read like `wrist`.
	 */
	gripper?: boolean | (() => boolean);
	/** variable_step: the coarse step (default 0.04 m) and the "high above the table" gap (default 0.08 m). */
	coarseStepM?: number;
	highAboveTableM?: number;
	/** rotation: +1 rotates wrist-judged moves by +yaw (flip if a post-rotation move goes the wrong way). */
	yawCompensationSign?: number;
	/** The robot's largest yaw per command, rad: longer turns are split into commands within it. */
	maxYawRad?: () => number;
	/**
	 * RT_* units (--units-rt): radians per unit, the base-frame unit vector each axis turns about for
	 * the positive unit (RT_ROLL_LEFT, RT_PITCH_FWD, RT_YAW_CCW; the sign convention lives in the
	 * vector), and the largest turn per command. A robot omits an axis it cannot turn about; its units
	 * are then refused.
	 */
	rt?: { stepRad: number; axes: Partial<Record<RtAxis, Vec3>>; maxRad?: () => number };
	/** The robot's largest translation per call, m: one `act` call travels at most this in total. */
	maxMoveM?: () => number;
	/**
	 * The robot flows through `continuous` moves: it returns before the arm settles, so the measured
	 * position lags the command and the stall check waits for the chain's last move.
	 */
	chains?: () => boolean;
	/**
	 * view_select (Show-Harness plugins/view_select), read at session start: when true, `act` takes
	 * `view` (which view guided the move) and passes it on in `Move.view`; the robot picks the move's
	 * frame from it and explains it in `views`.
	 */
	viewSelect?: () => boolean;
	/** Robots that execute `delta` in another frame: the base-frame translation it becomes from `state` (proprioception). */
	baseDelta?: (delta: Vec3, state: State | undefined) => Vec3;
	/** Affordance: mark [row, col] fractions (0..1) in a camera image; the marked PNG and world xyz per point. */
	point?: {
		cameras: readonly string[];
		locate: (
			camera: string,
			points: [number, number][],
			signal: AbortSignal | undefined,
		) => Promise<{ image?: Buffer; xyz: (number[] | null)[] }>;
	};
};

/** The robot base's tool registrar (terminate-with-finish, abort signal, env failure). */
export type ToolRegistrar = <P extends TSchema>(
	name: string,
	description: string,
	parameters: P,
	run: (params: Static<P>, signal: AbortSignal | undefined, ctx: ExtensionContext) => Promise<Result>,
) => void;

export type Stage = {
	motion: string;
	target: string;
	affordance?: string;
	description?: string;
	completion: string;
	arm?: string;
};
export type Target = { label: string; camera: string; point: [number, number]; xyz: number[] | null };

/** Session entries of the verifier's checks and the video_ref brief. */
export const VERIFY_ENTRY = "units_verify";
export const VIDEO_REF_ENTRY = "units_video_ref";
/** Session entry of the episode state (gripper, accumulated yaw, plan, history), rebuilt on resume and fork. */
export const STATE_ENTRY = "units_state";
