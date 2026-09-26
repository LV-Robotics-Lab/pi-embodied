/**
 * Grasp and placement tools over the env servers' grasp primitives (services
 * `utils/grasp.py`): `plan_grasp`, `plan_place` and the VLM attachment probe `check_attached`.
 *
 *   pi -e packages/embodied/src/libero --graspnet http://127.0.0.1:8120 [--graspgenx URL] [--anyplace URL] [--anygrasp URL]
 *
 * The robot registers the flags (`registerGraspFlags`) and passes them to its env server
 * (`graspArgs`), which composes SAM3, the grasp servers and its own camera calibration and hands
 * out short ids: `d3` a mask, `g1` a grasp, `p2` a place pose, each bound to the observation it
 * was planned from. The tools only carry ids and the world-frame poses the server resolved. An id
 * expires when the robot state changes, so a robot executes a planned grasp or place from one
 * resolution of its id (`env.claim_waypoints`: the whole pre-grasp / grasp / lift path at once;
 * LIBERO's `execute_grasp` / `execute_place`), and the server remembers the executed grasp so
 * `plan_place` can be asked after it, from the held object. A stale id is refused (the server
 * answers "is stale"), which this module records as a `detections_expired` session entry, as it
 * does the `expired_ids` every plan result lists. Without any backend flag `graspActive` names no
 * tool and the env server is started without the primitives.
 *
 * Greedy Grasp Candidate Policy (OpenETA): a plan's `active` candidate is tried first; only a
 * structured, candidate-specific failure that leaves the ids current (refused unmoved:
 * unreachable) advances to the next rank through `plan_grasp({next_after: id, reason})`, without
 * re-planning; a failure after the robot moved needs a new plan.
 *
 * `check_attached` keeps the VLM call on this side (`../units/vlm.ts` askVlm, cost on
 * VLM_COST_EVENT): the env server only supplies the frames (`env.attachment_frames`: the wrist
 * view whole, the other cameras cropped around the EEF).
 */

import type { ImageContent } from "@earendil-works/pi-ai";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type Static, type TSchema, Type } from "typebox";
import { type Json, message, toolResult } from "../robot.ts";
import { askVlm, parseJson, VLM_COST_EVENT } from "../units/vlm.ts";

/** Session entry when ids expired or a stale id was refused: `{ tool, ids, observation, error? }`. */
export const DETECTIONS_EXPIRED_ENTRY = "detections_expired";
/** Session entry per attachment probe: `{ object, arm, attached, confidence, reason, model, cost_usd, ms }`. */
export const CHECK_ATTACHED_ENTRY = "check_attached";
export const GRASP_TOOLS = ["plan_grasp", "plan_place", "check_attached"] as const;
export const BACKENDS = ["contact_graspnet", "graspgenx", "anygrasp"] as const;
/** The env-server flag per grasp service: `--<name> <url>`. */
const SERVICES = ["graspnet", "graspgenx", "anyplace", "anygrasp"] as const;

/** A grasp tool as the robot mounts it: `run` gets the abort signal and the pi context (for the VLM). */
export type GraspToolDef<P extends TSchema = TSchema> = {
	name: string;
	description: string;
	parameters: P;
	run: (params: Static<P>, signal: AbortSignal | undefined, ctx: ExtensionContext) => Promise<Json>;
};

/** What differs per robot. */
export type GraspRig = {
	/** The env RPC call carrying the running tool's abort signal. */
	call: (method: string, kwargs: Json, timeoutMs?: number) => Promise<Json>;
	/** The cameras the tools may plan from (the env server's names); unset = any registered view, as free text. */
	cameras?: readonly string[];
	/** The task text, for the attachment VLM. */
	task: () => string;
	/** Two arms: the `arm` parameter's schema and the validated arm name of its value. */
	arm?: { schema: TSchema; name: (v: unknown) => string };
};

export function registerGraspFlags(pi: ExtensionAPI) {
	pi.registerFlag("graspnet", { type: "string", default: "", description: "Contact-GraspNet server for plan_grasp" });
	pi.registerFlag("graspgenx", { type: "string", default: "", description: "GraspGenX server for plan_grasp" });
	pi.registerFlag("anyplace", { type: "string", default: "", description: "AnyPlace server for plan_place" });
	pi.registerFlag("anygrasp", { type: "string", default: "", description: "AnyGrasp server for plan_grasp" });
	pi.registerFlag("attach-vlm-model", {
		type: "string",
		default: "",
		description: "Model (provider/id) of check_attached (default: --units-vlm-model, else the session's model)",
	});
}

const url = (pi: ExtensionAPI, name: string) => String(pi.getFlag(name) ?? "").trim();

/** The env server arguments for the configured grasp services (`--graspnet URL ...`). */
export function graspArgs(pi: ExtensionAPI): string[] {
	return SERVICES.flatMap((name) => (url(pi, name) ? [`--${name}`, url(pi, name)] : []));
}

/** The grasp tools to activate: all three once any grasp service is configured, else none. */
export function graspActive(pi: ExtensionAPI): string[] {
	return SERVICES.some((name) => url(pi, name)) ? [...GRASP_TOOLS] : [];
}

/** Whether an env error is the server refusing an id from an earlier observation. */
export const isStale = (err: unknown) => /is stale|stale:|expired/i.test(message(err));

function expired(pi: ExtensionAPI, tool: string, result: Json | undefined, error?: unknown) {
	const ids: string[] = Array.isArray(result?.expired_ids) ? result.expired_ids : [];
	if (!ids.length && !error) return;
	pi.appendEntry(DETECTIONS_EXPIRED_ENTRY, {
		tool,
		ids,
		observation: result?.observation ?? null,
		...(error ? { error: message(error) } : {}),
	});
}

/** The attachment probe prompt: strict, JSON only. */
export function attachedPrompt(task: string, object: string, cameras: string[]) {
	return {
		system:
			"You judge, from a robot's camera images, whether its gripper is holding an object. Judge strictly from what is visible; do not assume. Return JSON only.",
		content: [
			{ type: "text" as const, text: `TASK: ${task}\nOBJECT: ${object}` },
			{
				type: "text" as const,
				text: `The ${cameras.length} image(s) are the current views: ${cameras.join(", ")}. Is the ${object} held between the gripper fingers (in the gripper, lifted or supported by it, not merely near it or resting on the surface)?\nReturn JSON only: {"attached": true|false, "confidence": 0.0-1.0, "reason": "one visual sentence"}`,
			},
		],
	};
}

/** The verdict, parsed leniently; unparseable replies are "not attached" with confidence 0. */
export function parseAttached(raw: string): { attached: boolean; confidence: number; reason: string } {
	const j = parseJson(raw) as { attached?: unknown; confidence?: unknown; reason?: unknown } | undefined;
	if (typeof j?.attached === "boolean")
		return {
			attached: j.attached,
			confidence: Math.max(0, Math.min(1, Number(j.confidence ?? (j.attached ? 1 : 0)) || 0)),
			reason: String(j.reason ?? "")
				.split(/\s+/)
				.join(" ")
				.trim(),
		};
	const m = /"attached"\s*:\s*(true|false)/i.exec(raw);
	if (m) return { attached: m[1].toLowerCase() === "true", confidence: 0.5, reason: raw.trim().slice(0, 200) };
	return { attached: false, confidence: 0, reason: `unparseable reply: ${raw.trim().slice(0, 120)}` };
}

/** Build the three tools for one robot. */
export function graspTools(pi: ExtensionAPI, rig: GraspRig): GraspToolDef[] {
	const armProp: Record<string, TSchema> = rig.arm ? { arm: rig.arm.schema } : {};
	const armOf = (p: Json): Json => (rig.arm ? { arm: rig.arm.name(p.arm) } : {});
	const camera = Type.Optional(
		rig.cameras
			? StringEnum(rig.cameras as unknown as [string, ...string[]], { description: `Default ${rig.cameras[0]}` })
			: Type.String({ description: "Registered RGB-D view name (default: the first configured)" }),
	);
	const planGrasp: GraspToolDef = {
		name: "plan_grasp",
		description:
			"Predict grasps for one object from the current RGB-D observation, ranked best first, in the world frame. Give the object as text (segmented with SAM3) or a mask_id of this observation. Each candidate has a short id (g3) and its EEF pose, valid until the robot moves; execute_grasp (where the robot has it) runs one from a single resolution of its id. Try `active` first; when it is refused before the robot moves (unreachable), call again with next_after=<that id> to get the next rank without re-planning; after a failed motion, plan again.",
		parameters: Type.Object({
			object: Type.Optional(Type.String({ description: "Object to grasp (SAM3 text prompt), e.g. 'black bowl'" })),
			mask_id: Type.Optional(
				Type.String({ description: "A mask id (d2) of the current observation instead of object" }),
			),
			camera,
			backend: Type.Optional(StringEnum(BACKENDS, { description: "Default: the first configured" })),
			...armProp,
			next_after: Type.Optional(
				Type.String({ description: "Greedy policy: reject this grasp id and return the next rank of its plan" }),
			),
			reason: Type.Optional(Type.String({ description: "With next_after: why that candidate failed" })),
		}),
		run: async (p) => {
			const params = p as Json;
			let result: Json;
			try {
				if (params.next_after) {
					result = await rig.call("env.next_grasp", { grasp_id: params.next_after, reason: params.reason ?? "" });
				} else {
					const text = String(params.object ?? "").trim();
					if (!text && !params.mask_id) return { error: "give object (text) or mask_id" };
					result = await rig.call(
						"env.plan_grasp",
						{
							...(text ? { object: text } : { mask_id: params.mask_id }),
							...(params.camera ? { camera: params.camera } : {}),
							...(params.backend ? { backend: params.backend } : {}),
							...armOf(params),
						},
						600_000,
					);
				}
			} catch (err) {
				if (isStale(err)) expired(pi, "plan_grasp", undefined, err);
				return { error: message(err) };
			}
			expired(pi, "plan_grasp", result);
			return result;
		},
	};
	const planPlace: GraspToolDef = {
		name: "plan_place",
		description:
			"Where to hold the grasped object so it comes to rest on a placement region (AnyPlace). Give the region as text (segmented now) or as a mask id, and the grasp id: after executing the grasp, that grasp's id (the held object is segmented again from its prompt, or object_mask_id, and the gripper's actual pose is used; refused when the fingers hold nothing); before it, a current g id with region and object from the same observation. Returns place poses with ids (p1); execute_place (where the robot has it) runs one.",
		parameters: Type.Object({
			region: Type.Optional(
				Type.String({ description: "The surface it goes onto / into (SAM3 text), or region_mask_id" }),
			),
			region_mask_id: Type.Optional(Type.String()),
			grasp_id: Type.String({ description: "The executed grasp (g id) the object is held with, or a current one" }),
			object_mask_id: Type.Optional(
				Type.String({ description: "The object's mask id; default the mask the grasp was planned on" }),
			),
			camera,
		}),
		run: async (p) => {
			const params = p as Json;
			try {
				const cam = params.camera ? { camera: params.camera } : {};
				let region_mask_id = params.region_mask_id ? String(params.region_mask_id) : "";
				if (!region_mask_id) {
					const t = String(params.region ?? "").trim();
					if (!t) throw new Error("give region (text) or region_mask_id");
					const seg = await rig.call("env.segment_mask", { object: t, ...cam }, 120_000);
					if (!seg.found) throw new Error(`could not segment region '${t}': ${seg.reason ?? "no mask"}`);
					region_mask_id = String(seg.id);
				}
				const result = await rig.call(
					"env.plan_place",
					{
						region_mask_id,
						grasp_id: String(params.grasp_id),
						...(params.object_mask_id ? { object_mask_id: String(params.object_mask_id) } : {}),
					},
					600_000,
				);
				expired(pi, "plan_place", result);
				return result;
			} catch (err) {
				if (isStale(err)) expired(pi, "plan_place", undefined, err);
				return { error: message(err) };
			}
		},
	};
	const checkAttached: GraspToolDef = {
		name: "check_attached",
		description:
			"Independent visual check whether the gripper holds the object: a vision model judges the current wrist view and the other cameras cropped around the gripper. Use it after a grasp and lift, before carrying; it does not move the robot.",
		parameters: Type.Object({
			object: Type.String({ description: "The object that should be in the gripper" }),
			...armProp,
		}),
		run: async (p, signal, ctx) => {
			const params = p as Json;
			const object = String(params.object).trim();
			if (!object) return { error: "object must be non-empty" };
			const started = Date.now();
			const frames = await rig.call("env.attachment_frames", { ...armOf(params) }, 60_000);
			const shots: { camera: string; png: string; crop: unknown }[] = (frames.frames ?? []).map((f: Json) => ({
				camera: String(f.camera),
				png: String(f.png_base64),
				crop: f.crop_rc ?? null,
			}));
			if (!shots.length) return { error: "the env server returned no frames" };
			const images: ImageContent[] = shots.map((s) => ({ type: "image", data: s.png, mimeType: "image/png" }));
			const modelRef = String(pi.getFlag("attach-vlm-model") || pi.getFlag("units-vlm-model") || "");
			const entry: Json = { object, arm: (armOf(params) as Json).arm ?? null };
			let out: Json;
			try {
				const reply = await askVlm(
					ctx,
					modelRef,
					pi.getThinkingLevel(),
					attachedPrompt(
						rig.task(),
						object,
						shots.map((s) => (s.crop ? `${s.camera} (cropped around the gripper)` : s.camera)),
					),
					images,
					signal,
				);
				pi.events.emit(VLM_COST_EVENT, reply.cost);
				const verdict = parseAttached(reply.text);
				out = { ...verdict, model: reply.model, cost_usd: reply.cost };
			} catch (err) {
				out = {
					attached: null,
					confidence: 0,
					reason: `attachment check unavailable: ${message(err)}`,
					error: message(err),
				};
			}
			Object.assign(entry, out, { ms: Date.now() - started, observation: frames.observation ?? null });
			pi.appendEntry(CHECK_ATTACHED_ENTRY, entry);
			return {
				...out,
				object,
				observation: frames.observation ?? null,
				frames: shots.map((s) => ({ camera: s.camera, crop_rc: s.crop })),
				_pngs: shots.map((s) => Buffer.from(s.png, "base64")),
			};
		},
	};
	return [planGrasp, planPlace, checkAttached];
}

/** Mount a grasp tool on a robot whose `tool` is `defineRobot`'s (read-only: the result and its PNGs). */
export function mountGraspTool(
	tool: <P extends TSchema>(
		name: string,
		description: string,
		parameters: P,
		run: (
			params: Static<P>,
			signal: AbortSignal | undefined,
			ctx: ExtensionContext,
		) => Promise<ReturnType<typeof toolResult>>,
	) => void,
	d: GraspToolDef,
) {
	tool(d.name, d.description, d.parameters, async (params, signal, ctx) => {
		const { _pngs, ...rest } = await d.run(params, signal, ctx);
		return toolResult(rest, (_pngs as Buffer[] | undefined) ?? []);
	});
}
