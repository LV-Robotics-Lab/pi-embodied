/**
 * SAM3 masks with ids and UniDepth depth over an env server's perception primitives (services
 * `utils/perception.py`): `detect`, `select_detection`, `reject_detection` and `enhance_depth`.
 * The Franka robots serve the same primitives under their own names (../franka: `segment`);
 * every other robot's env server serves them as `env.detect` & co.
 *
 *   pi -e packages/embodied/src/metaworld --detections [--sam3 URL] [--unidepth URL]
 *
 * `--detections` passes the robot's SAM3 server to its env server and activates the three mask
 * tools; `--unidepth <url>` passes the UniDepth server and activates `enhance_depth`. Without
 * them nothing is registered as active and the env server is started as before; the robot
 * activates only what its env server reports in `capabilities.perception` (an attached server
 * may lack a service). Ids (`d3`) are bound to the observation they were cut from: a motion
 * expires them, a stale id is refused and recorded as a `detections_expired` session entry.
 */

import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type TSchema, Type } from "typebox";
import { encodePng } from "../png.ts";
import { type Json, rgbOf, round } from "../robot.ts";
import { NdArray } from "../rpc.ts";
import { DETECTIONS_EXPIRED_ENTRY, type GraspToolDef } from "./grasp.ts";

export const MASK_TOOLS = ["detect", "select_detection", "reject_detection"] as const;
export const DEPTH_TOOLS = ["enhance_depth"] as const;
/** The env server's primitive names (services SIM_NAMES). */
export const METHODS = {
	detect: "env.detect",
	select_detection: "env.select_detection",
	reject_detection: "env.reject_detection",
	enhance_depth: "env.enhance_depth",
} as const;

/** What the env server reports in `get_env_meta().capabilities.perception`. */
export type PerceptionCaps = { segment?: boolean; enhance_depth?: boolean };

/** What differs per robot. */
export type DetectionRig = {
	/** The env RPC call carrying the running tool's abort signal. */
	call: (method: string, kwargs: Json, timeoutMs?: number) => Promise<Json>;
	/** The camera names the env server's perception knows, the first the default. */
	cameras: readonly string[];
	/**
	 * The robot's frame for a detection of `camera`: e.g. its world xyz through the robot's own
	 * depth (keys merged into the detection). Unset: pixels, depth and the camera-frame point only.
	 */
	locate?: (camera: string, detection: Json) => Promise<Json> | Json;
	/** A camera's depth was replaced by the fused estimate: e.g. store it where back_project reads depth. */
	onDepth?: (camera: string, depth: NdArray) => Promise<Json | undefined> | Json | undefined;
};

export function registerDetectionFlags(pi: ExtensionAPI) {
	pi.registerFlag("detections", {
		type: "boolean",
		default: false,
		description:
			"SAM3 masks with ids on the env server (detect, select_detection, reject_detection), through the --sam3 server",
	});
	pi.registerFlag("unidepth", {
		type: "string",
		default: "",
		description: "UniDepth server for the env server's enhance_depth (off when empty)",
	});
}

/** The env server arguments: `--sam3 <url>` with --detections, `--unidepth <url>` when set. */
export function detectionArgs(pi: ExtensionAPI, sam3: string): string[] {
	const depth = String(pi.getFlag("unidepth") ?? "").trim();
	return [
		...(pi.getFlag("detections") === true && sam3 ? ["--sam3", sam3] : []),
		...(depth ? ["--unidepth", depth] : []),
	];
}

/** The tools to activate: what the flags ask for and the env server serves. */
export function detectionActive(pi: ExtensionAPI, caps: PerceptionCaps | undefined): string[] {
	return [
		...(pi.getFlag("detections") === true && caps?.segment ? MASK_TOOLS : []),
		...(String(pi.getFlag("unidepth") ?? "").trim() && caps?.enhance_depth ? DEPTH_TOOLS : []),
	];
}

const r3 = (v: unknown) => (typeof v === "number" ? round(v, 3) : (v ?? null));

/** Build the four tools for one robot (mount them read-only, e.g. with ./grasp.ts `mountGraspTool`). */
export function detectionTools(pi: ExtensionAPI, rig: DetectionRig): GraspToolDef[] {
	const [first] = rig.cameras;
	const camera = Type.Optional(
		StringEnum(rig.cameras as unknown as [string, ...string[]], { description: `Default ${first}` }),
	);
	const id = Type.String({ description: "A detection id of the current observation (d3)" });
	/** Ids the server dropped since the last perception call, and a refused stale id, as session entries. */
	const expired = (tool: string, res: Json, error?: string) => {
		const ids: string[] = Array.isArray(res.invalidated) ? res.invalidated.map(String) : [];
		if (!ids.length && !error) return;
		pi.appendEntry(DETECTIONS_EXPIRED_ENTRY, {
			tool,
			ids,
			observation: res.observation ?? null,
			...(error ? { error } : {}),
		});
	};
	const plainDetection = async (cam: string, d: Json) => {
		const { mask_png_base64: _mask, mask: _m, ...rest } = d;
		return {
			id: rest.id,
			score: r3(rest.score),
			box: rest.box ?? null,
			area_px: rest.area_px ?? null,
			centroid_pixel: rest.centroid_rc ?? null,
			depth_m: rest.depth_m ?? null,
			point_camera: rest.point_camera ?? null,
			...(rig.locate ? await rig.locate(cam, d) : {}),
		};
	};
	const detect: GraspToolDef = {
		name: "detect",
		description:
			"SAM3 masks on a camera's current image, each with a short id (d3) drawn on the returned overlay in rank order (red, blue, green, yellow, ...). Give exactly one of a text prompt or a positive point [row, col]. all=true returns every candidate instead of only the best; then choose one with select_detection or rule one out with reject_detection. Ids die with the observation: detect again after any motion.",
		parameters: Type.Object({
			prompt: Type.Optional(Type.String()),
			point: Type.Optional(Type.Array(Type.Integer(), { minItems: 2, maxItems: 2 })),
			camera,
			min_score: Type.Optional(Type.Number({ description: "Default 0.2" })),
			all: Type.Optional(Type.Boolean({ description: "Every mask, not only the best (default false)" })),
		}) as TSchema,
		run: async (p) => {
			const { prompt, point, camera: c = first, min_score = 0.2, all = false } = p as Json;
			// Models often fill both optional fields; a non-empty prompt wins.
			const text = String(prompt ?? "").trim();
			if (!text && !point) return { error: "give a text prompt or a point [row, col]" };
			const res = await rig.call(
				METHODS.detect,
				{ camera: c, ...(text ? { text_prompt: text } : { point }), min_score, all },
				120_000,
			);
			expired("detect", res);
			if (!res.found)
				return {
					found: false,
					camera: c,
					error: res.reason ?? "no mask",
					fallback: "Pick pixels in the image and use back_project.",
				};
			const detections = await Promise.all(((res.detections ?? []) as Json[]).map((d) => plainDetection(c, d)));
			const out: Json = { found: true, camera: c, observation: res.observation, ids: res.ids, detections };
			if (res.overlay instanceof NdArray) {
				const img = rgbOf(res.overlay);
				out._pngs = [encodePng(img.rgb, img.width, img.height)];
			}
			return out;
		},
	};
	const choose =
		(name: "select_detection" | "reject_detection") =>
		async (p: unknown): Promise<Json> => {
			const res = await rig.call(METHODS[name], { id: String((p as Json).id) });
			expired(name, res, res.ok ? undefined : String(res.error ?? "refused"));
			const book = { ids: res.ids, selected: res.selected, rejected: res.rejected };
			if (!res.ok) return { error: res.error, ...book };
			if (name === "reject_detection") return book;
			const d = res.detection as Json;
			return { ...book, detection: await plainDetection(String(d.camera ?? first), d) };
		};
	const select: GraspToolDef = {
		name: "select_detection",
		description:
			"Choose one detect mask (by id) as the target of the current observation, after checking the overlay.",
		parameters: Type.Object({ id }),
		run: choose("select_detection"),
	};
	const reject: GraspToolDef = {
		name: "reject_detection",
		description: "Rule out one detect mask (by id) of the current observation; it stays listed as rejected.",
		parameters: Type.Object({ id }),
		run: choose("reject_detection"),
	};
	const enhance: GraspToolDef = {
		name: "enhance_depth",
		description:
			"Fill the holes of a camera's current depth with a UniDepth estimate scaled to the sensor, or supply depth where the camera has none. detect then measures through it until the next motion.",
		parameters: Type.Object({ camera }),
		run: async (p) => {
			const c = String((p as Json).camera ?? first);
			const res = await rig.call(METHODS.enhance_depth, { camera: c }, 180_000);
			expired("enhance_depth", res);
			const stored = res.depth instanceof NdArray ? await rig.onDepth?.(c, res.depth) : undefined;
			return {
				camera: c,
				observation: res.observation,
				report: res.report,
				estimate: res.estimate,
				...(stored ?? {}),
			};
		},
	};
	return [detect, select, reject, enhance];
}
