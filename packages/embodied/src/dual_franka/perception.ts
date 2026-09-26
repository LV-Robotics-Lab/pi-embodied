/**
 * Dual-Franka perception (./index.ts): `back_project` lifts one pixel of a registered RGBD view
 * into right_base, and `segment` localizes a SAM3 mask the same way. Both read a recorded step's
 * depth and camera metadata with the easy_handeye calibration the services loaded, and refuse
 * steps from before the last scene reset.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Type } from "typebox";
import { decodePngChannel, encodePng } from "../png.ts";
import { stepParam } from "../primitives/steps.ts";
import { apply, type Grid, type Json, mark, median, message, type Rgb, round, roundAll, vec } from "../robot.ts";
import type { RpcClient } from "../rpc.ts";
import type { Camera, RegisterTool, Setup, Step, View } from "./config.ts";

function coerceViews(configured: unknown): Record<string, View> {
	if (!configured || typeof configured !== "object") throw new Error("perception.projection_views must be a mapping");
	return Object.fromEntries(
		Object.entries(configured as Json).map(([a, c]) => {
			if (!c || typeof c !== "object") throw new Error(`perception.projection_views.${a} must be a mapping`);
			return [
				a,
				{
					raw_key: String(c.raw_key ?? `${a}_rgb`),
					calibration_key: String(c.calibration_key ?? `${a}_camera`),
					display_name: String(c.display_name ?? a),
				},
			];
		}),
	);
}

/** Validate a localization point: configured depth range and right_base tabletop volume. */
function validity(config: Json, depth: number, p: number[]) {
	const d: number[] = config.depth_m ?? [0.15, 1.25];
	const lo: number[] = config.right_base_xyz_min ?? [0.1, -0.85, 0.0];
	const hi: number[] = config.right_base_xyz_max ?? [1.15, 0.85, 0.85];
	const reasons: string[] = [];
	if (!(d[0] <= depth && depth <= d[1]))
		reasons.push(
			`depth ${depth.toFixed(3)}m is outside configured target range [${d[0].toFixed(3)}, ${d[1].toFixed(3)}]m`,
		);
	const axes = ["x", "y", "z"].filter((_, i) => !(lo[i] <= p[i] && p[i] <= hi[i]));
	if (axes.length)
		reasons.push(
			`right_base point is outside the configured tabletop localization volume on axis/axes ${axes.join(",")}`,
		);
	const contract = { depth_m: roundAll(d), right_base_xyz_min: roundAll(lo), right_base_xyz_max: roundAll(hi) };
	return { ok: reasons.length === 0, reasons, contract };
}

function nextIndex(s: Step, prefix: string, suffix: string): number {
	let i = 0;
	while (s.blob.artifacts.includes(`${prefix}_${String(i).padStart(2, "0")}${suffix}`)) i++;
	return i;
}

/** Python's round(): halves go to the even neighbour. */
const rint = (v: number) => {
	const f = Math.floor(v);
	return v - f === 0.5 ? f + (f % 2 === 0 ? 0 : 1) : Math.round(v);
};
const short = (v: unknown) =>
	Array.isArray(v) && v.length >= 3
		? v
				.slice(0, 3)
				.map((x) => Number(x).toFixed(3))
				.join(",")
		: "n/a";

/** What the perception tools read from the robot. */
export type PerceptionDeps = {
	tool: RegisterTool;
	/** A step to localize in (default the latest); one from before the last scene reset throws. */
	freshStep: (step?: number | null) => Step;
	setup: () => Setup | undefined;
	/** The SAM3 client (--robot-sam3), if configured. */
	sam3: () => RpcClient | undefined;
	loadRgb: (s: Step, name: string) => Rgb;
	loadDepth: (s: Step, name: string) => Grid | undefined;
};

/** Register `back_project` and `segment`, in that order. */
export function mountPerception(d: PerceptionDeps) {
	const { tool, freshStep, loadRgb, loadDepth } = d;
	const cameraParam = Type.Optional(
		Type.String({
			description:
				"Registered projection view name, e.g. d455 or base (default d455). Valid names come from perception.projection_views and the current state's saved artifacts.",
		}),
	);

	function projectionView(s: Step, camera: string): View {
		const views =
			s.meta?.projection_views && typeof s.meta.projection_views === "object"
				? coerceViews(s.meta.projection_views)
				: (d.setup()?.projection_views ?? {});
		const v = views[camera];
		if (!v)
			throw new Error(
				`unsupported projection camera: '${camera}'; registered=${Object.keys(views).sort().join(", ") || "<none>"}`,
			);
		return v;
	}

	function intrinsics(s: Step, camera: string, rawKey: string) {
		if (!s.meta) throw new Error(`${camera} camera metadata not found`);
		for (const key of [rawKey, camera, ...(camera === "base" ? ["extra_0"] : [])]) {
			const intr = s.meta[key]?.color_intrinsics;
			if (intr)
				return {
					fx: Number(intr.fx),
					fy: Number(intr.fy),
					cx: Number(intr.ppx ?? intr.cx),
					cy: Number(intr.ppy ?? intr.cy),
				};
		}
		throw new Error(`${camera} RealSense color intrinsics not found`);
	}

	function calibrationFor(key: string): Camera {
		const setup = d.setup();
		if (!setup?.cameras) throw new Error(setup?.calibration_error ?? "no hand-eye calibration loaded");
		const c = setup.cameras[key];
		if (!c) throw new Error(`calibration entry '${key}' is missing`);
		return c;
	}

	/** Both TCP positions and TCP-to-point deltas in right_base. */
	function tcpDeltas(state: Json, point: number[]): Json {
		const setup = d.setup();
		const out: Json = {
			tcp_delta_coordinate_frame: "right_base",
			tcp_delta_contract:
				"left_tcp_xyz, right_tcp_xyz, and both TCP-to-point deltas are expressed in the shared right_base world frame.",
		};
		const record = state?.state && typeof state.state === "object" ? state.state : state;
		for (const a of ["left", "right"]) {
			const armState: Json = record?.raw?.[a] ?? record?.[`${a}_arm`] ?? {};
			const pose = vec(armState.tcp_pose);
			if (pose.length < 3) continue;
			const frame = armState.tcp_pose_frame || armState.coordinate_frame || record?.coordinate_frame || `${a}_base`;
			let tcp = pose.slice(0, 3);
			if (frame !== "right_base") {
				if (frame !== "left_base" || !setup?.T_right_base_left_base)
					throw new Error(`missing base-frame transform T_right_base_${frame}`);
				tcp = apply(setup.T_right_base_left_base, tcp);
			}
			out[`${a}_tcp_xyz`] = roundAll(tcp);
			out[`delta_${a}_tcp_to_point_xyz`] = roundAll(point.map((v, k) => v - tcp[k]));
		}
		return out;
	}

	tool(
		"back_project",
		"Back-project one pixel from a registered RGBD camera view into shared right-base coordinates. Use a camera listed by view_env_state/view_camera_meta; default is the configured primary metric localization camera.",
		Type.Object({
			camera: cameraParam,
			row: Type.Integer({ minimum: 0 }),
			col: Type.Integer({ minimum: 0 }),
			target_name: Type.Optional(Type.String({ description: "Default 'target'" })),
			step: stepParam,
			window_radius: Type.Optional(
				Type.Integer({ minimum: 0, description: "Depth median window radius (default 2)" }),
			),
		}),
		async ({ camera = "d455", row, col, target_name = "target", step, window_radius = 2 }) => {
			camera = String(camera).trim();
			if (!camera) throw new Error("camera must be a non-empty string");
			const s = freshStep(step);
			const cfg = projectionView(s, camera);
			const depth = loadDepth(s, camera);
			if (!depth)
				throw new Error(
					`${cfg.display_name} depth artifact is missing. Restart the env server with the camera and depth enabled, then call view_env_state again.`,
				);
			const [r, c] = [Number(row), Number(col)];
			if (!(r >= 0 && r < depth.height && c >= 0 && c < depth.width))
				throw new Error(`pixel row/col [${r}, ${c}] out of depth bounds [${depth.height}, ${depth.width}]`);
			const radius = Math.max(0, Number(window_radius));
			const patch: number[] = [];
			for (let y = Math.max(0, r - radius); y < Math.min(depth.height, r + radius + 1); y++)
				for (let x = Math.max(0, c - radius); x < Math.min(depth.width, c + radius + 1); x++) {
					const z = depth.data[y * depth.width + x];
					if (Number.isFinite(z) && z > 0) patch.push(z);
				}
			if (!patch.length) throw new Error(`no valid depth near pixel row=${r} col=${c} radius=${radius}`);
			const z = median(patch);
			const { fx, fy, cx, cy } = intrinsics(s, camera, cfg.raw_key);
			const pointCamera = [((c - cx) * z) / fx, ((r - cy) * z) / fy, z];
			const cal = calibrationFor(cfg.calibration_key);
			const point = apply(cal.T_right_camera, pointCamera);
			const v = validity(cal.localization_validity, z, point);
			const result: Json = {
				ok: v.ok,
				selection_valid: v.ok,
				target_name: String(target_name).trim() || "target",
				camera,
				pixel: [r, c],
				coordinate_frame: "right_base",
				step: s.blob.step_idx,
				depth_m: round(z),
				depth_window_radius: radius,
				valid_depth_pixels_in_window: patch.length,
				point_camera_xyz: roundAll(pointCamera),
				point_xyz: roundAll(point),
				camera_extrinsic_frame: "right_base",
				coordinate_contract:
					"All returned points and deltas are expressed in the shared right_base world frame. Use the same delta convention for both left and right rule-based arm tools.",
				source_artifact: join(s.dir, `${camera}_depth.f32`),
				selection_contract:
					"The selected RGB pixel must lie well inside visible material of the named target object. Never select image-space air/background above the object. Compute robot z approach offsets only for explicit grasp/approach poses after projecting the object surface into right_base. For placement staging, use projected x/y only and keep the carried-object TCP z unchanged by default.",
				validity_contract: v.contract,
			};
			if (v.reasons.length) {
				result.error = `Rejected localization point: ${v.reasons.join("; ")}. Select a new pixel well inside the visible target surface.`;
				result.rejection_reasons = v.reasons;
			}
			Object.assign(result, tcpDeltas(s.blob.state, point));
			try {
				const n = String(nextIndex(s, `${camera}_back_project`, ".json")).padStart(2, "0");
				const img = loadRgb(s, camera);
				const png = encodePng(mark(img, r, c, v.ok ? [0, 255, 0] : [255, 0, 0]), img.width, img.height);
				const annotated = join(s.dir, `${camera}_back_project_${n}_annotated.png`);
				const report = join(s.dir, `${camera}_back_project_${n}.json`);
				writeFileSync(annotated, png);
				writeFileSync(
					report,
					JSON.stringify({
						ok: v.ok,
						snapshot_step: s.blob.step_idx,
						annotated_image: annotated,
						calibration_source: d.setup()?.calibration_source,
						calibration_key: cfg.calibration_key,
						[`T_right_base_${camera}_camera`]: cal.T_right_camera,
						label: `r${r},c${c} cam=${short(result.point_camera_xyz)} rb=${short(result.point_xyz)}`,
						projection: result,
					}),
				);
				s.blob.artifacts.push(`${camera}_back_project_${n}_annotated.png`, `${camera}_back_project_${n}.json`);
				result.diagnostic_artifacts = { annotated_image: annotated, report_json: report };
				result._pngs = [png];
				result.image_block_order = [`${camera}_selection_diagnostic`];
				result.image_delivery = `annotated_${camera}_selection_returned_for_verification`;
			} catch (err) {
				result.diagnostic_error = message(err);
			}
			return result;
		},
	);

	/** Median right_base point of the valid, in-volume mask pixels. */
	function maskToWorld(s: Step, camera: string, mask: Uint8Array, depth: Grid, minValid: number): Json {
		const cfg = projectionView(s, camera);
		const rows: number[] = [];
		const cols: number[] = [];
		const zs: number[] = [];
		let pixels = 0;
		for (let i = 0; i < mask.length; i++) {
			if (!mask[i]) continue;
			pixels++;
			const z = depth.data[i];
			if (!(Number.isFinite(z) && z > 0)) continue;
			rows.push(Math.floor(i / depth.width));
			cols.push(i % depth.width);
			zs.push(z);
		}
		const result: Json = {
			mask_pixels: pixels,
			valid_depth_pixels: zs.length,
			valid_localization_pixels: 0,
			mask_resized_to_depth_shape: false,
		};
		if (!pixels) return { ...result, point_xyz: null, world_error: "empty mask" };
		if (zs.length < minValid)
			return { ...result, point_xyz: null, world_error: `too few valid ${camera} depth pixels (${zs.length})` };
		const { fx, fy, cx, cy } = intrinsics(s, camera, cfg.raw_key);
		const cal = calibrationFor(cfg.calibration_key);
		const kept: { r: number; c: number; z: number; pc: number[]; pr: number[] }[] = [];
		let contract: Json = {};
		for (let i = 0; i < zs.length; i++) {
			const pc = [((cols[i] - cx) * zs[i]) / fx, ((rows[i] - cy) * zs[i]) / fy, zs[i]];
			const pr = apply(cal.T_right_camera, pc);
			const v = validity(cal.localization_validity, zs[i], pr);
			contract = v.contract;
			if (v.ok && pr.every(Number.isFinite)) kept.push({ r: rows[i], c: cols[i], z: zs[i], pc, pr });
		}
		result.validity_contract = contract;
		result.valid_localization_pixels = kept.length;
		if (kept.length < minValid)
			return {
				...result,
				point_xyz: null,
				world_error: `too few mask pixels remain inside configured ${camera} localization volume (${kept.length})`,
			};
		const point = [0, 1, 2].map((k) => median(kept.map((p) => p.pr[k])));
		const pointCamera = [0, 1, 2].map((k) => median(kept.map((p) => p.pc[k])));
		const z = median(kept.map((p) => p.z));
		const v = validity(cal.localization_validity, z, point);
		Object.assign(result, {
			selection_valid: v.ok,
			centroid_pixel: [rint(median(kept.map((p) => p.r))), rint(median(kept.map((p) => p.c)))],
			depth_m: round(z),
			point_camera_xyz: roundAll(pointCamera),
			point_xyz: v.ok ? roundAll(point) : null,
			raw_median_point_xyz: roundAll(point),
			camera_extrinsic_frame: "right_base",
			calibration_source: d.setup()?.calibration_source,
			calibration_key: cfg.calibration_key,
		});
		if (v.ok) Object.assign(result, tcpDeltas(s.blob.state, point));
		if (v.reasons.length) {
			result.rejection_reasons = v.reasons;
			result.world_error = `Rejected SAM3 mask localization: ${v.reasons.join("; ")}`;
		}
		return result;
	}

	tool(
		"segment",
		"Use SAM3 on a registered RGB image with either a text prompt or one positive [row, col] point, return a mask overlay for verification, and estimate the mask median point in shared right-base coordinates.",
		Type.Object({
			camera: cameraParam,
			prompt: Type.Optional(
				Type.String({
					description:
						"Text prompt for SAM3. Prefer short object/relation phrases; for the clean-desk box use 'white interior of the black cardboard box' or 'cardboard box'. Avoid over-specific surface words such as 'floor' when grounding is weak. Provide exactly one of prompt or point.",
				}),
			),
			point: Type.Optional(
				Type.Array(Type.Integer(), {
					minItems: 2,
					maxItems: 2,
					description:
						"Positive SAM3 point in camera image coordinates [row, col]. Provide exactly one of prompt or point.",
				}),
			),
			target_name: Type.Optional(Type.String({ description: "Default 'target'" })),
			step: stepParam,
			min_score: Type.Optional(Type.Number({ minimum: 0, maximum: 1, description: "Default 0.2" })),
			min_valid_depth_pixels: Type.Optional(Type.Integer({ minimum: 1, description: "Default 25" })),
		}),
		async ({
			camera = "d455",
			prompt = "",
			point,
			target_name = "target",
			step,
			min_score = 0.2,
			min_valid_depth_pixels = 25,
		}) => {
			camera = String(camera).trim();
			if (!camera) throw new Error("camera must be a non-empty string");
			const fallback = `Use manual ${camera} image inspection and back_project.`;
			const sam3 = d.sam3();
			if (!sam3)
				return {
					ok: false,
					found: false,
					error: "SAM3 client is not configured. Start pi with --robot-sam3 (see serve.sh).",
					fallback,
				};
			const text = String(prompt).trim();
			if (Boolean(text) === (point !== undefined))
				return { ok: false, found: false, error: "segment needs exactly one of prompt or point" };
			if (point !== undefined && (!Array.isArray(point) || point.length !== 2))
				return { ok: false, found: false, error: "point must be [row, col]" };
			let s: Step;
			let png: Buffer;
			let depth: Grid | undefined;
			try {
				s = freshStep(step);
				if (!s.views.includes(camera)) throw new Error(`${camera} image artifact is missing`);
				depth = loadDepth(s, camera);
				if (!depth) throw new Error(`${camera} depth artifact is missing`);
				if (!(min_score >= 0 && min_score <= 1)) throw new Error("min_score must be between 0 and 1");
				png = readFileSync(join(s.dir, `${camera}.png`));
			} catch (err) {
				return { ok: false, found: false, error: message(err) };
			}
			let res: Json;
			let mask: Uint8Array | undefined;
			try {
				res = await sam3.call<Json>(
					"sam3.segment",
					{
						image_base64: png.toString("base64"),
						min_score,
						...(text ? { text_prompt: text } : { point: (point ?? []).map(Number) }),
					},
					120_000,
				);
				if (typeof res?.found !== "boolean")
					throw new Error(`invalid SAM3 segment response: ${JSON.stringify(res)}`);
				if (res.found) {
					const shape = res.mask_shape;
					if (typeof res.mask_png_base64 !== "string" || !res.mask_png_base64)
						throw new Error("SAM3 response marked found but omitted mask_png_base64");
					if (!Array.isArray(shape) || shape.length !== 2) throw new Error(`invalid SAM3 mask_shape: ${shape}`);
					const decoded = decodePngChannel(Buffer.from(res.mask_png_base64, "base64"));
					if (decoded.height !== shape[0] || decoded.width !== shape[1])
						throw new Error(
							`SAM3 mask shape mismatch: response=${shape}, decoded=${[decoded.height, decoded.width]}`,
						);
					mask = decoded.data.map((v) => (v > 0 ? 1 : 0));
					res.mask_shape = shape;
				}
			} catch (err) {
				return { ok: false, found: false, error: `segmentation service call failed: ${message(err)}`, fallback };
			}
			const n = String(nextIndex(s, `${camera}_segment`, ".json")).padStart(2, "0");
			let localization: Json;
			let overlayPng: Buffer | undefined;
			if (res.found && mask) {
				try {
					if (res.mask_shape[0] !== depth.height || res.mask_shape[1] !== depth.width)
						localization = {
							point_xyz: null,
							world_error: `mask/depth shape mismatch: mask=${res.mask_shape}, depth=${[depth.height, depth.width]}`,
							mask_pixels: mask.reduce((a, b) => a + b, 0),
							valid_depth_pixels: 0,
							valid_localization_pixels: 0,
						};
					else localization = maskToWorld(s, camera, mask, depth, Math.max(1, Number(min_valid_depth_pixels)));
					const img = loadRgb(s, camera);
					if (img.width === depth.width && img.height === depth.height) {
						const rgb = Buffer.from(img.rgb);
						for (let i = 0; i < mask.length; i++) {
							if (!mask[i]) continue;
							rgb[i * 3] = Math.floor(0.55 * rgb[i * 3] + 0.45 * 255);
							rgb[i * 3 + 1] = Math.floor(0.55 * rgb[i * 3 + 1]);
							rgb[i * 3 + 2] = Math.floor(0.55 * rgb[i * 3 + 2]);
						}
						const [cr, cc] = localization.centroid_pixel ?? [];
						const marked =
							cr === undefined
								? rgb
								: mark({ ...img, rgb }, cr, cc, localization.point_xyz ? [0, 255, 0] : [255, 0, 0]);
						overlayPng = encodePng(marked, img.width, img.height);
						writeFileSync(join(s.dir, `${camera}_segment_overlay_${n}.png`), overlayPng);
						s.blob.artifacts.push(`${camera}_segment_overlay_${n}.png`);
					}
				} catch (err) {
					localization = { point_xyz: null, world_error: message(err) };
				}
			} else localization = { point_xyz: null, world_error: res.reason || "SAM3 found no mask" };
			const blob: Json = {
				ok: Boolean(res.found && localization.point_xyz != null),
				found: Boolean(res.found),
				mode: text ? "text" : "point",
				target_name: String(target_name).trim() || "target",
				camera,
				source_step: s.blob.step_idx,
				segment_index: Number(n),
				min_score,
				score: typeof res.score === "number" ? round(res.score, 3) : null,
				box: res.box ?? null,
				mask_shape: res.mask_shape ?? null,
				coordinate_frame: "right_base",
				coordinate_contract: `point_xyz is the median valid ${camera}-mask point expressed in the shared right_base world frame.`,
				selection_contract: `Inspect the returned ${camera} mask overlay. The highlighted mask and median marker must cover the intended visible material, not the rim, wire basket, wall, table, or background. Retry with a point prompt or a more specific text prompt if it is wrong.`,
				...(text ? { prompt: text } : { point }),
				...(res.found ? {} : { error: res.reason || "SAM3 found no mask" }),
				...localization,
			};
			const segmentName = `${camera}_segment_${n}.json`;
			writeFileSync(join(s.dir, segmentName), JSON.stringify(blob));
			s.blob.artifacts.push(segmentName);
			const keys = [
				"ok",
				"found",
				"target_name",
				"mode",
				"score",
				"box",
				"mask_shape",
				"coordinate_frame",
				"point_xyz",
				"world_error",
				"centroid_pixel",
				"mask_pixels",
				"valid_depth_pixels",
				"valid_localization_pixels",
				"selection_valid",
				"rejection_reasons",
				"left_tcp_xyz",
				"right_tcp_xyz",
				"delta_left_tcp_to_point_xyz",
				"delta_right_tcp_to_point_xyz",
				"tcp_delta_coordinate_frame",
				"tcp_delta_contract",
				"selection_contract",
			];
			const result: Json = { step: s.blob.step_idx, camera, segment_artifact: join(s.dir, segmentName) };
			for (const k of keys) result[k] = blob[k] ?? null;
			if (overlayPng) {
				result.overlay_artifact = join(s.dir, `${camera}_segment_overlay_${n}.png`);
				result._pngs = [overlayPng];
				result.image_block_order = [`${camera}_segment_overlay`];
				result.image_delivery = `sam3_${camera}_segment_overlay_returned_for_verification`;
			}
			if (blob.error) {
				result.error = blob.error;
				result.fallback = fallback;
			}
			return result;
		},
	);
}
