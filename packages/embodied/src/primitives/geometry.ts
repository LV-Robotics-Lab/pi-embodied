/**
 * The geometric toolset (OpenETA `openeta-for-codex`: `mark_point` and the approach/jaw `move_to`)
 * over the env servers' `utils/geometry.py`: `view_points`, `mark_point` and `move_grip`.
 *
 *   pi -e packages/embodied/src/robots/libero --geometry
 *
 * `--geometry` (off: nothing is registered and the env server is started without it; `optionalTools`
 * mounts the tools at the first start with it on) starts the env server with its geometry primitives (`env.point_views`, `env.mark_point`, `env.grip_target`,
 * `env.grip_state`, all in `code.api` for run_code too) and mounts the tools:
 * - `view_points`: the fused RGB-D cloud as orthographic top/front/side views with a metric grid
 *   (and/or the camera images), the grip site and the marked points drawn;
 * - `mark_point`: a world point by pixel: a camera pixel is its visible surface; a click on an
 *   orthographic view fixes two axes and a click on a complementary view the third, so two views
 *   define any point, also in free space;
 * - `move_grip`: the grip site to an absolute position, a marked point or a delta (world, or the
 *   gripper's own [JAW, LAT, APP] axes), oriented by the approach direction (grip +Z) and/or the
 *   jaw direction (+X, the closing axis), then open/close. A close is previewed first (the target
 *   and the pads' closing corridor drawn, frozen under a preview_id) and runs only through
 *   `execute_preview_id`, while the gripper has not moved. After a motion the result is
 *   `motion_status` (reached / not_reached; the gripper runs only when reached), the remaining delta
 *   (mm) and rotation error, and in simulation the robot's current contacts, drawn on the camera.
 *
 * The server resolves every target (`env.grip_target`); the robot executes it with its own motion
 * (`GeometryRig.execute`): a simulator servos the tool frame step by step (`runGripPlan`, keeping
 * its video, recorder and success latch), a real arm through its bounded delta primitives.
 */

import { StringEnum } from "@earendil-works/pi-ai";
import type { AgentToolResult, ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";
import { type Json, message, toolResult } from "../robot.ts";
import { optionalTools } from "./optional.ts";

export const GEOMETRY_TOOLS = ["view_points", "mark_point", "move_grip"] as const;
export const SCENE_VIEWS = ["pointcloud_top", "pointcloud_front", "pointcloud_side"] as const;

type Result = AgentToolResult<unknown>;
type Vec = number[];
type Mat = number[][];

/** The env server's arguments for the toolset (at start, with the flag read then). */
export const geometryArgs = (pi: ExtensionAPI): string[] => (pi.getFlag("geometry") === true ? ["--geometry"] : []);

/**
 * --geometry: the three tools on this robot, mounted through `mount` at the first start with the flag
 * on; the returned function (called from the robot's `start`) names them, or none with the flag off.
 */
export function geometryTools(pi: ExtensionAPI, rig: GeometryRig, mount: (d: GeometryToolDef) => void) {
	return optionalTools(
		pi,
		"geometry",
		"Geometric toolset: view_points (orthographic point-cloud views), mark_point (points by pixel, two views for free space), move_grip (approach/jaw-directed grip-site moves, close preview, residual and contacts)",
		() => geometryDefs(rig),
		mount,
	);
}

/** `env.grip_target`'s answer: execute `target` (then `gripper`), or the preview's images. */
export type GripPlan = {
	status: "execute" | "preview";
	motion: boolean;
	gripper: "open" | "close" | null;
	target: { grip_xyz_m: Vec; approach_world: Vec; jaw_world: Vec; tool_quat_xyzw: Vec; [k: string]: unknown };
	current: { grip_xyz_m: Vec; approach_world: Vec; jaw_world: Vec; tool_quat_xyzw: Vec };
	delta_mm: Vec;
	rotation_deg: number;
	preview_id?: string;
	views?: string[];
	images?: string[];
	[k: string]: unknown;
};

/** What differs per robot. */
export type GeometryRig = {
	/** The env RPC call carrying the running tool's abort signal. */
	call: (method: string, kwargs: Json, timeoutMs?: number) => Promise<Json>;
	/** The robot's camera names the server fuses (view_points and mark_point take them too). */
	cameras: readonly string[];
	/** A reason to refuse a move before planning (the episode ended), else undefined. */
	refuse?: () => string | undefined;
	/** Run a plan with status execute: the motion (when `plan.motion`), then the gripper when it reached. */
	execute: (plan: GripPlan, signal: AbortSignal | undefined) => Promise<Result>;
};

export type GeometryToolDef<P extends TSchema = TSchema> = {
	name: string;
	description: string;
	parameters: P;
	run: (params: Static<P>, signal: AbortSignal | undefined) => Promise<Result>;
};

/** A reply's images (`images`: PNG base64) as buffers, and the reply without them. */
export function splitImages(reply: Json): { rest: Json; pngs: Buffer[] } {
	const { images, ...rest } = reply;
	return { rest, pngs: ((images as string[] | undefined) ?? []).map((b) => Buffer.from(b, "base64")) };
}

const xyz = (description: string) =>
	Type.Optional(Type.Array(Type.Number(), { minItems: 3, maxItems: 3, description }));

/** The three tools for one robot. */
export function geometryDefs(rig: GeometryRig): GeometryToolDef[] {
	const cams = rig.cameras.join(", ");
	const relay = async (method: string, kwargs: Json, timeoutMs = 120_000) => {
		try {
			const { rest, pngs } = splitImages(await rig.call(method, kwargs, timeoutMs));
			return toolResult(rest, pngs);
		} catch (err) {
			return toolResult({ error: message(err) });
		}
	};
	const viewPoints: GeometryToolDef = {
		name: "view_points",
		description: `The scene as orthographic point-cloud views of the fused RGB-D cameras: pointcloud_top (right +x, up +y, seen from above), pointcloud_front (right +x, up +z, seen from -y), pointcloud_side (right +y, up +z, seen from +x), each with a 5 cm grid labelled in world metres; also the camera images (${cams}) at the size mark_point uses. The grip site (magenta ring) and solved points are drawn. Every pixel of an orthographic view is two world coordinates, drawn point or empty space. Views expire when the robot moves.`,
		parameters: Type.Object({
			views: Type.Optional(
				Type.Array(StringEnum([...SCENE_VIEWS, ...rig.cameras] as [string, ...string[]]), {
					maxItems: 5,
					description: "Default the three pointcloud views",
				}),
			),
		}),
		run: async (p) => {
			const views = (p as Json).views as string[] | undefined;
			return relay("env.point_views", views?.length ? { views } : {});
		},
	};
	const markPoint: GeometryToolDef = {
		name: "mark_point",
		description: `Measure one world point by clicking a pixel (x = column, y = row, zero-based) of an image view_points or a move_grip preview returned for the current state. A camera image (${cams}) gives the first visible surface at that pixel. An orthographic view (pointcloud_*, or a preview's preview_*) fixes its two axes and returns the complementary views with a cyan line where the point can be: click the same feature there with the same point_id to fix the third axis (the shared axis must agree within 15 mm). Two views define any point, on a surface or in free space. A solved point keeps its world xyz for the episode; move_grip takes it as point_id. It is geometry, not a grasp.`,
		parameters: Type.Object({
			point_id: Type.String({ description: "The point's name, e.g. P1; reuse it for the second click" }),
			view: Type.String({
				description: `A view name from the images' result: ${[...SCENE_VIEWS, ...rig.cameras].join(", ")}, or preview_top/front/side`,
			}),
			x: Type.Integer({ minimum: 0, description: "Pixel column" }),
			y: Type.Integer({ minimum: 0, description: "Pixel row" }),
		}),
		run: async (p) => relay("env.mark_point", p as Json),
	};
	const moveGrip: GeometryToolDef = {
		name: "move_grip",
		description:
			"Move the grip site (the point between the finger pads) and orient it, then open or close. Position: at most one of xyz (world m), point_id (a solved mark_point) or delta_mm (delta_frame world = [dX,dY,dZ], grip_site = [dJAW,dLAT,dAPP] along the gripper's current axes); omitted keeps the position. Orientation: approach is the world direction the gripper points along (its +Z; [0,0,-1] = straight down), jaw the world direction the fingers close along (its +X; either sign); give either or both, omitted parts keep the current orientation. preview=true only draws the target (zoomed preview_top/front/side views, markable, and the camera). A close is always previewed first: the result's preview_id, executed unchanged by move_grip({execute_preview_id}) alone while the gripper has not moved; any change needs a new preview. The preview is geometry: it does not predict contact. After a motion: motion_status reached or not_reached (then the gripper is skipped), grip_xyz_m, remaining_delta_mm, rotation_error_deg, gripper_width (closed_on_nothing: no grasp), and in simulation the robot's contacts (cyan on the camera image).",
		parameters: Type.Object({
			xyz: xyz("Grip-site position, world m"),
			point_id: Type.Optional(Type.String({ description: "A solved mark_point id as the position" })),
			delta_mm: xyz("Grip-site displacement, mm (in delta_frame)"),
			delta_frame: Type.Optional(StringEnum(["world", "grip_site"] as const, { description: "Default world" })),
			approach: xyz("World direction of the gripper's approach axis (+Z)"),
			jaw: xyz("World direction of the jaw (closing) axis (+X)"),
			gripper: Type.Optional(StringEnum(["open", "close"] as const, { description: "After the motion" })),
			preview: Type.Optional(Type.Boolean({ description: "Draw the target without moving" })),
			execute_preview_id: Type.Optional(
				Type.String({ description: "Execute a frozen close preview; give it alone" }),
			),
		}),
		run: async (p, signal) => {
			const why = rig.refuse?.();
			if (why) return toolResult({ error: why });
			let plan: GripPlan;
			try {
				plan = (await rig.call("env.grip_target", p as Json, 120_000)) as GripPlan;
			} catch (err) {
				return toolResult({ error: message(err) });
			}
			if (plan.status === "preview") {
				const { rest, pngs } = splitImages(plan);
				return toolResult({ motion_status: "previewed", ...rest }, pngs);
			}
			return rig.execute(plan, signal);
		},
	};
	return [viewPoints, markPoint, moveGrip];
}

// ---- simulator execution: a step-by-step servo of the tool frame -----------------------------

const sub = (a: Vec, b: Vec) => a.map((v, i) => v - b[i]);
const norm = (a: Vec) => Math.hypot(...a);
const clip = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Rotation matrix of an xyzw quaternion. */
export function quatMatrix(q: Vec): Mat {
	const n = norm(q) || 1;
	const [x, y, z, w] = q.map((v) => v / n);
	return [
		[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
		[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
		[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
	];
}
const mul = (a: Mat, b: Mat) => a.map((row) => [0, 1, 2].map((j) => row.reduce((s, v, k) => s + v * b[k][j], 0)));
const tr = (a: Mat) => [0, 1, 2].map((i) => [0, 1, 2].map((j) => a[j][i]));

/** The rotation vector (axis x angle) of a rotation matrix. */
export function rotvec(r: Mat): Vec {
	const t = Math.acos(clip((r[0][0] + r[1][1] + r[2][2] - 1) / 2, -1, 1));
	if (t < 1e-9) return [0, 0, 0];
	if (Math.PI - t < 1e-6) {
		const cols = [0, 1, 2].map((j) => [0, 1, 2].map((i) => r[i][j] + (i === j ? 1 : 0)));
		const a = cols.reduce((best, c) => (norm(c) > norm(best) ? c : best));
		return a.map((v) => (v / norm(a)) * t);
	}
	return [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]].map((v) => (v * t) / (2 * Math.sin(t)));
}

/** Extrinsic xyz Euler angles (R = Rz(c) Ry(b) Rx(a)) of a rotation matrix: a real arm's `rotate_delta`. */
export function eulerXyz(r: Mat): Vec {
	const b = Math.asin(clip(-r[2][0], -1, 1));
	if (Math.abs(Math.cos(b)) < 1e-9) return [Math.atan2(-r[1][2], r[1][1]), b, 0];
	return [Math.atan2(r[2][1], r[2][2]), b, Math.atan2(r[1][0], r[0][0])];
}

/** The base-frame rotation from the plan's current tool orientation to its target's. */
export const planRotation = (plan: GripPlan): Mat =>
	mul(quatMatrix(plan.target.tool_quat_xyzw), tr(quatMatrix(plan.current.tool_quat_xyzw)));

export type ServoIo = {
	/** The tool frame now: position (the grip site) and xyzw orientation. */
	pose: () => Promise<{ pos: Vec; quat: Vec }>;
	/** One OSC step [dx, dy, dz, rx, ry, rz, gripper] (world-frame translation and rotation vector). */
	step: (action: number[]) => Promise<void>;
	/** The gripper command held during the motion. */
	hold: () => number;
	/** Drive the gripper (-1 open, +1 close) until the fingers stop; returns the steps. */
	actuate: (g: number) => Promise<number>;
	ended: () => boolean;
};

export const SERVO = { tolM: 0.005, tolRad: 0.05, maxSteps: 200, stepClip: 0.025, rotClip: 0.08 };

/**
 * Servo the tool frame to a position and orientation together (the services' `_move_grip` rule):
 * each step a clipped translation and a clipped world-frame rotation vector; stops within the
 * tolerances, at the step budget or at the episode's end.
 */
export async function servoGrip(goal: { position: Vec; quat: Vec }, io: ServoIo, maxSteps = SERVO.maxSteps) {
	const G = quatMatrix(goal.quat);
	const error = async () => {
		const p = await io.pose();
		return { d: sub(goal.position, p.pos), r: rotvec(mul(G, tr(quatMatrix(p.quat)))) };
	};
	let steps = 0;
	let e = await error();
	for (; steps < maxSteps && !io.ended(); steps++) {
		const angle = norm(e.r);
		if (norm(e.d) < SERVO.tolM && angle < SERVO.tolRad) break;
		const rot = angle > 1e-9 ? e.r.map((v) => (v * Math.min(angle, SERVO.rotClip)) / angle) : [0, 0, 0];
		await io.step([
			...e.d.map((v) => clip(clip(v, -SERVO.stepClip, SERVO.stepClip) / 0.05, -1, 1)),
			...rot.map((v) => clip(v / 0.1, -1, 1)),
			io.hold(),
		]);
		e = await error();
	}
	const [posErr, oriErr] = [norm(e.d), norm(e.r)];
	return { steps, reached: posErr <= SERVO.tolM && oriErr <= SERVO.tolRad };
}

/**
 * A simulator's `execute`: the motion (servoGrip), the gripper only when it reached, then
 * `env.grip_state` for the residual, the contacts and their image. Returns the result and its PNGs.
 */
export async function runGripPlan(
	plan: GripPlan,
	io: ServoIo,
	call: GeometryRig["call"],
	maxSteps = SERVO.maxSteps,
): Promise<{ result: Json; pngs: Buffer[] }> {
	let steps = 0;
	let status = "not_requested";
	const out: Json = { name: "move_grip", target: plan.target };
	if (plan.preview_id) out.preview_id = plan.preview_id;
	if (plan.motion) {
		const s = await servoGrip({ position: plan.target.grip_xyz_m, quat: plan.target.tool_quat_xyzw }, io, maxSteps);
		steps += s.steps;
		status = s.reached ? "reached" : "not_reached";
	}
	if (plan.gripper) {
		if (status === "not_reached") out.gripper_skipped = "the motion did not reach its target";
		else if (!io.ended()) {
			steps += await io.actuate(plan.gripper === "close" ? 1 : -1);
			out.gripper = plan.gripper;
		}
	}
	const target = plan.motion
		? {
				target_xyz: plan.target.grip_xyz_m,
				target_approach: plan.target.approach_world,
				target_jaw: plan.target.jaw_world,
			}
		: {};
	const { rest, pngs } = splitImages(await call("env.grip_state", target));
	delete rest.motion_status;
	return { result: { ...out, motion_status: status, steps_used: steps, ...rest }, pngs };
}
