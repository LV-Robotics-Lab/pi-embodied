/**
 * Pixel -> world by ray-plane intersection, for a robot whose camera calibration is known but which
 * has no depth or back-projection tool (../maniskill, ../robolab). The env server's `get_camera_meta`
 * gives the OpenCV intrinsics K and the camera-to-world transform of the RAW sensor frame; the model
 * sees that frame letterboxed into a square (ManiSkill's `_rgb`) or as is (RoboLab's front camera).
 * A pointed pixel is mapped back to raw pixels, cast as a ray, and met with the horizontal plane at
 * the height the pointed object is known to have: a Flash anchor's recorded height, since anchors
 * slide on the table between seeds but do not leave it. Computed locally, no tool call.
 *
 * Error budget: both robots' front camera (RLinf's calibrated D435, 1.1 m in front of the base,
 * 0.26 m up) sees the table at about 20 deg elevation, so a plane 1 cm off puts the point about
 * 2.8 cm off along the view direction (x); the anchor heights must be the objects', not the table's.
 */

import { NdArray } from "../rpc.ts";

/** `env.get_camera_meta`: OpenCV K (3x3) and camera-to-world (4x4) of the raw sensor pixels. */
export type CameraMeta = { intrinsic_K: NdArray | number[][]; extrinsic_cam2world: NdArray | number[][] };
/** A raw `width` x `height` frame letterboxed into a `size` square with centred black bars (0: sent raw). */
export type Letterbox = { width: number; height: number; size: number };

/** A matrix as `n` rows, whether the RPC decoded it to an NdArray or JSON already nested it. */
function rows(m: NdArray | number[][], n: number): number[][] {
	const flat = m instanceof NdArray ? m.toArray() : m.flat();
	if (flat.length !== n * n) throw new Error(`expected a ${n}x${n} matrix, got ${flat.length} values`);
	return Array.from({ length: n }, (_, i) => flat.slice(i * n, (i + 1) * n));
}

/**
 * The raw pixel behind [col, row] of the letterboxed image (env_server `_letterbox`, scenes
 * `prepare_view`: equal-ratio resize into the square, centred). Pixel centres map to pixel centres,
 * so the square's centre is the raw frame's. With `size` 0 the image is the raw frame.
 */
export function unletterbox([col, row]: [number, number], box: Letterbox): [number, number] {
	if (!box.size) return [col, row];
	const scale = box.size / Math.max(box.width, box.height);
	const x0 = Math.floor((box.size - Math.round(box.width * scale)) / 2);
	const y0 = Math.floor((box.size - Math.round(box.height * scale)) / 2);
	return [(col + 0.5 - x0) / scale - 0.5, (row + 0.5 - y0) / scale - 0.5];
}

/**
 * Where the ray through raw pixel [col, row] meets the horizontal plane z = `z` (world), or
 * undefined when it never does (parallel, or the plane is behind the camera). OpenCV camera axes:
 * x right, y down, z forward.
 */
export function pixelOnPlane(meta: CameraMeta, [col, row]: [number, number], z: number): number[] | undefined {
	const K = rows(meta.intrinsic_K, 3);
	const T = rows(meta.extrinsic_cam2world, 4);
	const y = (row - K[1][2]) / K[1][1];
	const d = [(col - K[0][2] - K[0][1] * y) / K[0][0], y, 1];
	const dir = [0, 1, 2].map((i) => T[i][0] * d[0] + T[i][1] * d[1] + T[i][2] * d[2]);
	const origin = [T[0][3], T[1][3], T[2][3]];
	if (Math.abs(dir[2]) < 1e-9) return undefined;
	const s = (z - origin[2]) / dir[2];
	if (s <= 0) return undefined;
	return origin.map((o, i) => o + s * dir[i]);
}

/** The raw pixel [col, row] a world point projects to, or undefined behind the camera (the inverse, for checks). */
export function projectPixel(meta: CameraMeta, xyz: number[]): [number, number] | undefined {
	const K = rows(meta.intrinsic_K, 3);
	const T = rows(meta.extrinsic_cam2world, 4);
	// camera <- world: R^T (p - t)
	const p = [0, 1, 2].map((i) => xyz[i] - T[i][3]);
	const c = [0, 1, 2].map((j) => T[0][j] * p[0] + T[1][j] * p[1] + T[2][j] * p[2]);
	if (c[2] <= 1e-9) return undefined;
	const x = c[0] / c[2];
	const y = c[1] / c[2];
	return [K[0][0] * x + K[0][1] * y + K[0][2], K[1][1] * y + K[1][2]];
}
