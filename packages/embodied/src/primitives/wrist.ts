/**
 * `align_wrist` (--align-wrist): OpenETA's wrist-view alignment (`compute_wrist_alignment`,
 * agent/tools/grasp_geometry.py:1200). The gripper centre is projected into the wrist camera (the
 * pixel the target should sit on); the target pixel, lifted with its depth, gives how far the target
 * sits beside that point in the camera's image plane. That offset at the target's depth, rotated into
 * the world frame and clamped to `max_correction_m`, is the lateral correction: the EEF position to
 * command is the current one plus it. Nothing moves; the result names the aligned position for the
 * robot's motion tool, with the wrist image marked (green: where the gripper centre projects, red: the target).
 *
 * The robot supplies the wrist camera (intrinsics, camera-to-world pose, the image) and the world
 * point under a pixel (its depth lookup), and the gripper centre in the world.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { encodePng } from "../png.ts";
import { type Mat, mark, type Rgb, round, roundAll } from "../robot.ts";
import { optionalTools } from "./optional.ts";
import { type ToolDef, toolDef } from "./steps.ts";

/** The wrist view for one pixel: calibration, the world point under it, and the image (for the overlay). */
export type WristView = { K: Mat; cam2world: Mat; target: number[]; image?: Rgb };

export type WristRig = {
	/** The wrist view and the world point at pixel (row, col) of the current wrist image; throws when there is no valid depth. */
	view: (row: number, col: number) => Promise<WristView>;
	/** The gripper centre (between the fingertips) in the world frame. */
	gripper: () => Promise<number[]> | number[];
	/** Where the aligned position goes, for the description (e.g. "move_to xyz"). */
	moveWith: string;
};

export type Alignment = {
	desired_pixel: [number, number];
	target_pixel: [number, number];
	residual_px: number;
	target_depth_m: number;
	delta_world: number[];
	raw_correction_m: number;
	clamped: boolean;
	aligned_xyz: number[];
};

const sub = (a: number[], b: number[]) => a.map((v, i) => v - b[i]);
/** Rᵀ v for the rotation block of a camera-to-world pose. */
const toCamera = (m: Mat, v: number[]) => [0, 1, 2].map((j) => m[0][j] * v[0] + m[1][j] * v[1] + m[2][j] * v[2]);
const toWorld = (m: Mat, v: number[]) => [0, 1, 2].map((i) => m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2]);

/** The product of two 4x4 homogeneous transforms. */
export const compose = (a: Mat, b: Mat): Mat =>
	a.map((row) => [0, 1, 2, 3].map((j) => row.reduce((acc, v, k) => acc + v * (b[k]?.[j] ?? 0), 0)));

/** Pixel [row, col] of each world point in a pinhole camera (OpenCV axes: x right, y down, z forward); null behind it. */
export function projectPoints(K: Mat, cam2world: Mat, points: number[][]): ([number, number] | null)[] {
	const [[fx, , cx], [, fy, cy]] = K;
	const origin = [cam2world[0][3], cam2world[1][3], cam2world[2][3]];
	return points.map((w) => {
		const p = toCamera(cam2world, sub(w, origin));
		return p[2] > 1e-4 ? [Math.round(fy * (p[1] / p[2]) + cy), Math.round(fx * (p[0] / p[2]) + cx)] : null;
	});
}

/**
 * The lateral correction that puts `target` on the gripper centre's pixel (OpenETA's delta_camera =
 * ((u - u_d) z / fx, (v - v_d) z / fy, 0) at the target depth z), clamped to `maxCorrection` m.
 */
export function wristAlignment(o: {
	K: Mat;
	cam2world: Mat;
	target: number[];
	gripper: number[];
	maxCorrection: number;
}): Alignment {
	const [[fx, , cx], [, fy, cy]] = o.K;
	const origin = [o.cam2world[0][3], o.cam2world[1][3], o.cam2world[2][3]];
	const t = toCamera(o.cam2world, sub(o.target, origin));
	const g = toCamera(o.cam2world, sub(o.gripper, origin));
	if (!(t[2] > 1e-4)) throw new Error("the target is not in front of the wrist camera");
	if (!(g[2] > 1e-4))
		throw new Error("the gripper centre does not project into the wrist camera (behind its image plane)");
	const pixel = (p: number[]): [number, number] => [fy * (p[1] / p[2]) + cy, fx * (p[0] / p[2]) + cx];
	const [vd, ud] = pixel(g);
	const [v, u] = pixel(t);
	const raw = toWorld(o.cam2world, [((u - ud) * t[2]) / fx, ((v - vd) * t[2]) / fy, 0]);
	const norm = Math.hypot(...raw);
	const scale = norm > o.maxCorrection ? o.maxCorrection / norm : 1;
	const delta = raw.map((x) => x * scale);
	return {
		desired_pixel: [Math.round(vd), Math.round(ud)],
		target_pixel: [Math.round(v), Math.round(u)],
		residual_px: round(Math.hypot(u - ud, v - vd), 1),
		target_depth_m: round(t[2], 4),
		delta_world: roundAll(delta),
		raw_correction_m: round(norm, 4),
		clamped: scale < 1,
		aligned_xyz: roundAll(o.gripper.map((x, i) => x + delta[i])),
	};
}

export function alignWrist(rig: WristRig): ToolDef {
	return toolDef(
		"align_wrist",
		`Wrist-view alignment near a grasp: give the target's pixel [row, col] in the current wrist image. Returns the lateral EEF correction (in the camera's image plane, at the target's depth, at most max_correction_m) that puts the target under the gripper centre, and aligned_xyz to pass to ${rig.moveWith}. Nothing moves; approach depth and orientation are unchanged. The returned wrist image marks the gripper-centre pixel (green) and the target (red).`,
		Type.Object({
			point: Type.Array(Type.Integer(), {
				minItems: 2,
				maxItems: 2,
				description: "Target pixel [row, col] in the wrist image",
			}),
			max_correction_m: Type.Optional(
				Type.Number({ minimum: 0.005, maximum: 0.05, description: "Largest correction, m (default 0.03)" }),
			),
		}),
		async (p) => {
			const [row, col] = (p.point as number[]).map(Number);
			const maxCorrection = Math.min(0.05, Math.max(0.005, Number(p.max_correction_m ?? 0.03)));
			const view = await rig.view(row, col);
			const a = wristAlignment({ ...view, gripper: [...(await rig.gripper())], maxCorrection });
			const pngs: Buffer[] = [];
			if (view.image) {
				const marked = {
					...view.image,
					rgb: mark(view.image, a.desired_pixel[0], a.desired_pixel[1], [0, 220, 0]),
				};
				pngs.push(encodePng(mark(marked, row, col, [255, 32, 32]), view.image.width, view.image.height));
			}
			return {
				name: "align_wrist",
				...a,
				target_world: roundAll(view.target),
				max_correction_m: maxCorrection,
				...(pngs.length ? { _pngs: pngs } : {}),
			};
		},
	);
}

/** --align-wrist: `align_wrist` on this robot, mounted through `mount` at the first start with the flag on. */
export function alignWristTool(pi: ExtensionAPI, rig: WristRig, mount: (d: ToolDef) => void) {
	return optionalTools(
		pi,
		"align-wrist",
		"Add align_wrist (OpenETA's wrist-view lateral alignment of the gripper to a target pixel)",
		() => [alignWrist(rig)],
		mount,
	);
}
