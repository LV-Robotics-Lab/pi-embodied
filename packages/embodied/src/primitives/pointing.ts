/**
 * Molmo pointing (services `components/molmo_server.py`): `point` finds what a short phrase names
 * in a camera's current image and returns the pixel, marked on the image, and, where the robot has
 * depth, its world point. Given several cameras it asks MolmoPoint's `molmo.ground_set` once over
 * the ordered image set (OpenETA's Pointing Image Set: "Image 1" is the first camera), and every
 * point carries the camera it lies in.
 *
 *   pi -e packages/embodied/src/robots/metaworld --point [--molmo http://127.0.0.1:18400]
 *
 * `--point` activates `point` over the robot's --molmo server (off by default; a robot without
 * that flag gets it here, and --molmo off disables pointing too); a set of cameras needs a server
 * started with `--model molmopoint`. The robot supplies each camera's current image and,
 * optionally, the world point of a pixel.
 */

import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type TSchema, Type } from "typebox";
import { encodePng } from "../infra/png.ts";
import { RpcClient } from "../infra/rpc.ts";
import { type Json, mark, message, round } from "../robot.ts";
import type { GraspToolDef } from "./grasp.ts";

/** The most images one `molmo.ground_set` takes (MolmoPoint's MAX_IMAGES). */
export const MAX_SET = 4;

export type Frame = { width: number; height: number; rgb: Buffer };

/** What differs per robot. */
export type PointRig = {
	/** The tool name (default `point`; LIBERO's units already own `point`). */
	name?: string;
	/** The camera names `point` takes, the first the default; empty: any name, `defaultCamera` by default. */
	cameras: readonly string[];
	defaultCamera?: () => string;
	/** A camera's current image, as the model sees it (8-bit RGB). */
	frame: (camera: string) => Promise<Frame>;
	/** The world point (and any other fields) of a pixel of that image; unset or undefined: pixels only. */
	locate?: (camera: string, row: number, col: number) => Promise<Json | undefined>;
	/** The running tool's abort signal. */
	signal?: () => AbortSignal | undefined;
};

/** `--point`, and `--molmo` for a robot that has no Molmo flag of its own (Flash robots register it). */
export function registerPointFlags(pi: ExtensionAPI, o: { molmo?: boolean } = {}) {
	if (o.molmo)
		pi.registerFlag("molmo", {
			type: "string",
			default: "http://127.0.0.1:18400",
			description: "Molmo server for point (a set of cameras needs --model molmopoint)",
		});
	pi.registerFlag("point", {
		type: "boolean",
		default: false,
		description: "Molmo pointing (point) over the --molmo server",
	});
}

const molmoUrl = (pi: ExtensionAPI) => {
	const url = String(pi.getFlag("molmo") ?? "").trim();
	return url === "off" ? "" : url;
};

/** `point` with --point and a Molmo server, else nothing. */
export function pointActive(pi: ExtensionAPI, name = "point"): string[] {
	return pi.getFlag("point") === true && molmoUrl(pi) ? [name] : [];
}

type Ground = { point_xy?: number[] | null; answer?: string; image_size?: number[] };
type GroundSet = { points?: { image_index: number; pixel_x: number; pixel_y: number }[]; answer?: string };

export function pointTool(pi: ExtensionAPI, rig: PointRig): GraspToolDef {
	const fixed = rig.cameras.length > 0;
	const defaultCamera = () => (fixed ? rig.cameras[0] : (rig.defaultCamera?.() ?? ""));
	const name = fixed
		? StringEnum(rig.cameras as unknown as [string, ...string[]])
		: Type.String({ description: "Camera name" });
	const molmo = () => new RpcClient(molmoUrl(pi));
	const clip = (v: number, n: number) => Math.max(0, Math.min(n - 1, Math.round(v)));
	/** One pixel of a camera's frame: marked, and located where the robot can. */
	const hit = async (camera: string, f: Frame, x: number, y: number) => {
		const row = clip(y, f.height);
		const col = clip(x, f.width);
		const located = await rig.locate?.(camera, row, col).catch((err: unknown) => ({ locate_error: message(err) }));
		return { camera, pixel: [row, col], ...(located ?? {}) };
	};
	return {
		name: rig.name ?? "point",
		description: `Molmo points at what a short noun phrase names ('the red mug handle') in a camera's current image and returns the pixel [row, col], its world point where the robot has depth, and the image with the point marked. With cameras (up to ${MAX_SET}) it asks once over that ordered set (the query may say "Image 1" for the first) and every point names its camera.`,
		parameters: Type.Object({
			query: Type.String(),
			camera: Type.Optional(
				fixed
					? StringEnum(rig.cameras as unknown as [string, ...string[]], {
							description: `Default ${rig.cameras[0]}`,
						})
					: Type.String({ description: "Camera name (default: the main camera)" }),
			),
			cameras: Type.Optional(
				Type.Array(name, { minItems: 2, maxItems: MAX_SET, description: "Several cameras at once (MolmoPoint)" }),
			),
		}) as TSchema,
		run: async (p) => {
			const { query, camera, cameras } = p as { query: string; camera?: string; cameras?: string[] };
			const q = String(query ?? "").trim();
			if (!q) return { error: "query must be a non-empty phrase" };
			const signal = rig.signal?.();
			if (cameras && cameras.length > 1) {
				const frames = await Promise.all(cameras.map((c) => rig.frame(c)));
				const res = await molmo().call<GroundSet>(
					"molmo.ground_set",
					{ images_base64: frames.map((f) => encodePng(f.rgb, f.width, f.height).toString("base64")), query: q },
					180_000,
					[],
					signal,
				);
				const points = res.points ?? [];
				if (!points.length)
					return {
						found: false,
						cameras,
						answer: res.answer ?? null,
						fallback: "Try one camera, or detect / back_project.",
					};
				const marked = frames.map((f) => f.rgb);
				const out = [];
				for (const pt of points) {
					const f = frames[pt.image_index];
					const h = await hit(cameras[pt.image_index], f, pt.pixel_x, pt.pixel_y);
					marked[pt.image_index] = mark(
						{ ...f, rgb: marked[pt.image_index] },
						h.pixel[0],
						h.pixel[1],
						[255, 32, 32],
					);
					out.push(h);
				}
				const shown = [...new Set(points.map((pt) => pt.image_index))].sort();
				return {
					found: true,
					cameras,
					points: out,
					answer: res.answer ?? null,
					_pngs: shown.map((i) => encodePng(marked[i], frames[i].width, frames[i].height)),
				};
			}
			const c = camera ?? cameras?.[0] ?? defaultCamera();
			const f = await rig.frame(c);
			const res = await molmo().call<Ground>(
				"molmo.ground",
				{ image_base64: encodePng(f.rgb, f.width, f.height).toString("base64"), query: q },
				120_000,
				[],
				signal,
			);
			if (!res.point_xy)
				return { found: false, camera: c, answer: res.answer ?? null, fallback: "Use detect or back_project." };
			const h = await hit(c, f, res.point_xy[0], res.point_xy[1]);
			return {
				found: true,
				...h,
				answer: res.answer ?? null,
				_pngs: [encodePng(mark(f, h.pixel[0], h.pixel[1], [255, 32, 32]), f.width, f.height)],
			};
		},
	};
}

/** A world point rounded for a result. */
export const xyz = (p: number[] | null | undefined, d = 4) =>
	p ? { world_xyz: p.map((v) => round(v, d)) } : undefined;
