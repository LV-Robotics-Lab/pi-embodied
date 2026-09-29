/**
 * `suggest_grasp` (--grasp-advisor): OpenETA's isolated visual grasp-pose advisor
 * (agent/tools/grasp_pose_advisor.py:117 BackendGraspPoseAdvisor). plan_grasp (./grasp.ts) ranks
 * candidates by the grasp network's score; this tool draws the latest plan's candidates on the camera
 * image they were planned from (jaw line between the two contacts, dot at the grasp centre, approach
 * arrow, one colour per candidate) and asks a separate vision model, with a clean context, which one
 * is most likely to hold through closing, lift and transport, or to abstain. It recommends; it never
 * moves: the agent runs the chosen id (LIBERO execute_grasp, or a motion tool's grasp_id); a candidate
 * refused unmoved still goes to plan_grasp's next_after, a failed motion to a new plan.
 *
 * The plan comes from plan_grasp's own tool results (a `tool_result` hook), so ./grasp.ts is
 * unchanged; a plan from an earlier robot state (`stamp` changed) is refused. The VLM call is
 * ../units/vlm.ts askVlm (cost on VLM_COST_EVENT) with --grasp-advisor-model, else
 * --aux-model (aux.attach), else the session's model; each advice is a
 * `grasp_suggestion` session entry.
 */

import type { ImageContent } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { auxModel } from "../infra/config.ts";
import { encodePng } from "../infra/png.ts";
import { askVlm, parseJson, VLM_COST_EVENT } from "../modes/units/vlm.ts";
import { type Json, mark, message, type Rgb, round } from "../robot.ts";
import type { GraspToolDef } from "./grasp.ts";

export const GRASP_SUGGESTION_ENTRY = "grasp_suggestion";
const MAX_CANDIDATES = 8;
const COLORS: [string, [number, number, number]][] = [
	["cyan", [0, 255, 220]],
	["yellow", [255, 203, 67]],
	["pink", [255, 99, 132]],
	["blue", [92, 179, 255]],
	["purple", [199, 125, 255]],
	["green", [92, 230, 130]],
	["orange", [255, 145, 77]],
	["white", [235, 235, 235]],
];

export const ADVISOR_SYSTEM = `You are a grasp-pose advisor for a robot arm with a parallel-jaw gripper. You get one camera image with grasp candidates drawn on the target object: for each, a coloured line between the two finger contacts, a filled dot at the grasp centre and an arrow along the approach. Every candidate already passed the planner's width and reachability filters. Recommend the one most likely to stay secure through closing, lifting and carrying, or abstain; you do not execute anything.

Judge the drawn geometry, not the planner's score alone. Prefer deep, centred, opposing contacts on a broad, load-bearing part of the object, with the centre of mass between the fingers. Penalize contacts on a rim, cap, neck, handle tip, thin edge, corner or tapered shoulder, and contacts that barely overlap the object. For upright bottles, cans and cartons prefer side contacts on the middle of the body. If every candidate is on such an unstable region, or the view does not show enough to compare them, abstain so the robot can get another view or new candidates. Never invent a candidate id.

Return exactly one JSON object:
{"decision":"recommend|abstain","recommended_candidate_id":"exact id or empty","alternatives":["exact id"],"confidence":0.0,"reasons":["short visual reason"],"rejected":{"exact id":"short reason"}}
When abstaining, leave recommended_candidate_id empty and keep confidence at most 0.5.`;

/** The point mode's extras (no grasp backend): the camera and its back-projection. */
export type PointExtras = Pick<PointRig, "camera" | "backProject"> & {
	/** The camera's position in the world (a side approach comes from it). */
	cameraPosition?: (camera: string) => Promise<number[] | undefined>;
};

/** What differs per robot. */
export type AdvisorRig = {
	/** The no-backend point mode (7.2): mounted instead of the ranking when no grasp backend is configured. */
	point?: PointExtras;
	/** The current image of `camera` (the camera plan_grasp planned from). */
	image: (camera: string) => Promise<Rgb>;
	/** Pixel [row, col] of each world point in that image, null when behind the camera. */
	project: (camera: string, points: number[][]) => Promise<([number, number] | null)[]>;
	/** A value that changes whenever the robot moves (LIBERO: the env step). */
	stamp: () => number;
	/** The task text, for the advisor. */
	task: () => string;
};

type Plan = { camera: string; candidates: Json[]; stamp: number; object: string };
export type Advice = {
	decision: "recommend" | "abstain";
	recommended_candidate_id: string;
	alternatives: string[];
	confidence: number;
	reasons: string[];
	rejected: Record<string, string>;
	warnings: string[];
};

/** Parse and validate the advisor's reply against the offered ids; anything malformed becomes an abstention. */
export function parseAdvice(raw: string, ids: string[]): Advice {
	const j = parseJson(raw) as Json | undefined;
	const warnings: string[] = [];
	const known = (v: unknown) => ids.includes(String(v ?? "").trim());
	const list = (v: unknown) => (Array.isArray(v) ? v.map((x) => String(x).trim()).filter(Boolean) : []);
	if (!j) return abstain(`unparseable reply: ${raw.trim().slice(0, 160)}`);
	let decision = String(j.decision ?? "").toLowerCase() === "recommend" ? "recommend" : "abstain";
	let rec = String(j.recommended_candidate_id ?? "").trim();
	if (decision === "recommend" && !known(rec)) {
		warnings.push(`recommended unknown candidate "${rec}"`);
		decision = "abstain";
	}
	if (decision === "abstain") rec = "";
	let confidence = Math.max(0, Math.min(1, Number(j.confidence) || 0));
	if (decision === "abstain") confidence = Math.min(confidence, 0.5);
	const alternatives = [...new Set(list(j.alternatives).filter((id) => known(id) && id !== rec))].slice(0, 5);
	for (const id of list(j.alternatives)) if (!known(id)) warnings.push(`unknown alternative "${id}"`);
	const rejected: Record<string, string> = {};
	if (j.rejected && typeof j.rejected === "object")
		for (const [id, why] of Object.entries(j.rejected as Json))
			if (known(id)) rejected[id] = String(why).slice(0, 300);
			else warnings.push(`unknown rejected "${id}"`);
	return {
		decision: decision as Advice["decision"],
		recommended_candidate_id: rec,
		alternatives,
		confidence,
		reasons: list(j.reasons).slice(0, 6),
		rejected,
		warnings,
	};
	function abstain(why: string): Advice {
		return {
			decision: "abstain",
			recommended_candidate_id: "",
			alternatives: [],
			confidence: 0,
			reasons: [],
			rejected: {},
			warnings: [why],
		};
	}
}

/** Draw a line of the given half-width into an RGB image. */
function line(img: Rgb, out: Buffer, a: number[], b: number[], color: number[], half = 2) {
	const n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1])));
	for (let s = 0; s <= n; s++) {
		const r = Math.round(a[0] + ((b[0] - a[0]) * s) / n);
		const c = Math.round(a[1] + ((b[1] - a[1]) * s) / n);
		dot(img, out, r, c, color, half);
	}
}
function dot(img: Rgb, out: Buffer, row: number, col: number, color: number[], radius: number) {
	for (let dr = -radius; dr <= radius; dr++)
		for (let dc = -radius; dc <= radius; dc++) {
			const [r, c] = [row + dr, col + dc];
			if (r < 0 || c < 0 || r >= img.height || c >= img.width || dr * dr + dc * dc > radius * radius) continue;
			out.set(color, (r * img.width + c) * 3);
		}
}

/** The candidates drawn on the image: jaw line, centre dot, approach arrow (5 cm), one colour each. */
export async function renderCandidates(rig: AdvisorRig, plan: Plan, candidates: Json[]) {
	const img = await rig.image(plan.camera);
	const out = Buffer.from(img.rgb);
	const drawn: string[] = [];
	for (const [k, c] of candidates.entries()) {
		const color = COLORS[k % COLORS.length][1];
		const pos = (c.position as number[]) ?? [];
		const approach = (c.approach as number[]) ?? [0, 0, -1];
		const contacts = (c.contact_points as number[][]) ?? [];
		const tail = pos.map((v, i) => v - 0.05 * approach[i]);
		const [p, t, c0, c1] = await rig.project(plan.camera, [pos, tail, contacts[0] ?? pos, contacts[1] ?? pos]);
		if (!p) continue;
		if (t) line(img, out, t, p, color, 1);
		if (c0 && c1) line(img, out, c0, c1, color, 2);
		dot(img, out, p[0], p[1], color, 6);
		drawn.push(String(c.id));
	}
	return { png: encodePng(out, img.width, img.height), drawn };
}

/** The latest plan_grasp plan and the ids next_after rejected since, from plan_grasp's own results. */
export function trackPlans(pi: ExtensionAPI, stamp: () => number) {
	const state: { plan?: Plan; rejected: Set<string> } = { rejected: new Set() };
	pi.on("session_start", () => {
		state.plan = undefined;
		state.rejected.clear();
	});
	pi.on("tool_result", (event) => {
		if (event.toolName !== "plan_grasp" || event.isError) return undefined;
		const d = event.details as Json | undefined;
		if (Array.isArray(d?.candidates)) {
			state.plan = {
				camera: String(d.camera ?? ""),
				candidates: d.candidates as Json[],
				stamp: stamp(),
				object: String((event.input as Json)?.object ?? ""),
			};
			state.rejected.clear();
		}
		if (d?.rejected?.id) state.rejected.add(String(d.rejected.id));
		return undefined;
	});
	return state;
}

export function suggestGrasp(pi: ExtensionAPI, rig: AdvisorRig, plans: ReturnType<typeof trackPlans>): GraspToolDef {
	return {
		name: "suggest_grasp",
		description:
			"Second opinion on the latest plan_grasp candidates: a separate vision model sees them drawn on the camera image (jaw line, centre dot, approach arrow, one colour each) and recommends the one most likely to hold through lift and carry, or abstains when every contact looks unstable. It moves nothing; run the recommended id with execute_grasp (or pass it as grasp_id). Call it right after plan_grasp, before the robot moves: ids expire when it does.",
		parameters: Type.Object({
			max_candidates: Type.Optional(
				Type.Integer({ minimum: 1, maximum: MAX_CANDIDATES, description: `Default ${MAX_CANDIDATES}` }),
			),
		}),
		run: async (p, signal, ctx: ExtensionContext) => {
			const { plan, rejected } = plans;
			if (!plan) return { error: "no plan_grasp result in this episode; call plan_grasp first" };
			if (plan.stamp !== rig.stamp())
				return { error: "the robot moved since that plan_grasp; its ids are stale, plan again" };
			const n = Math.min(MAX_CANDIDATES, Math.max(1, Number((p as Json).max_candidates ?? MAX_CANDIDATES)));
			const offered = plan.candidates.filter((c) => !rejected.has(String(c.id)) && c.rejected !== true).slice(0, n);
			if (!offered.length) return { error: "every candidate of that plan was rejected; plan again" };
			const started = Date.now();
			const { png, drawn } = await renderCandidates(rig, plan, offered);
			if (!drawn.length) return { error: "no candidate projects into the camera image" };
			const legend = offered
				.map((c, k) => ({ c, color: COLORS[k % COLORS.length][0] }))
				.filter(({ c }) => drawn.includes(String(c.id)))
				.map(({ c, color }) => ({
					id: c.id,
					color,
					planner_rank: c.rank,
					planner_score: c.score,
					width_m: c.width_m,
					centre_z_m: Array.isArray(c.position) ? round(Number(c.position[2]), 3) : null,
					approach: c.approach,
				}));
			const modelRef = String(pi.getFlag("grasp-advisor-model") || auxModel(pi, "attach"));
			const images: ImageContent[] = [{ type: "image", data: png.toString("base64"), mimeType: "image/png" }];
			let out: Json;
			try {
				const reply = await askVlm(
					ctx,
					modelRef,
					pi.getThinkingLevel(),
					{
						system: ADVISOR_SYSTEM,
						content: [
							{
								type: "text",
								text: `TASK: ${rig.task()}\nOBJECT: ${plan.object || "(given by mask)"}\nCANDIDATES (drawn in the image in these colours):\n${JSON.stringify(legend)}`,
							},
						],
					},
					images,
					signal,
				);
				pi.events.emit(VLM_COST_EVENT, reply.cost);
				out = { ...parseAdvice(reply.text, drawn), model: reply.model, cost_usd: reply.cost };
			} catch (err) {
				out = { decision: "abstain", error: `grasp advisor unavailable: ${message(err)}` };
			}
			pi.appendEntry(GRASP_SUGGESTION_ENTRY, {
				...out,
				offered: drawn,
				camera: plan.camera,
				ms: Date.now() - started,
			});
			return { name: "suggest_grasp", ...out, candidates: legend, _pngs: [png] };
		},
	};
}

export const POINT_SYSTEM = `You propose where a robot's parallel-jaw gripper should grasp one object, from a single camera image. Choose a point on a broad, load-bearing part of the object where the two fingers can close on opposite sides (not a rim, cap, handle tip or thin edge), and how the gripper should come in. Judge only what is visible; if the object is not visible or you cannot tell, say so.

Return exactly one JSON object:
{"found": true|false, "point": [y, x], "jaw": [[y1, x1], [y2, x2]], "approach": "top"|"side", "confidence": 0.0-1.0, "reason": "one visual sentence"}
point, jaw: coordinates normalized to 0-1 of the image height (y) and width (x). point is the grasp centre on the object; jaw the two finger contacts on either side of it. approach: top = straight down, side = horizontally toward the object from the camera's side.`;

export type PointProposal = {
	found: boolean;
	point?: [number, number];
	jaw?: [[number, number], [number, number]];
	approach?: "top" | "side";
	confidence: number;
	reason: string;
};

/** Parse the point proposal; anything malformed is `found: false`. */
export function parsePointProposal(raw: string): PointProposal {
	const j = parseJson(raw) as Json | undefined;
	const unit = (v: unknown): [number, number] | undefined =>
		Array.isArray(v) &&
		v.length === 2 &&
		v.every((x) => Number.isFinite(Number(x)) && Number(x) >= 0 && Number(x) <= 1)
			? [Number(v[0]), Number(v[1])]
			: undefined;
	const reason = String(j?.reason ?? "").slice(0, 300);
	const point = unit(j?.point);
	if (!j || j.found === false || !point)
		return { found: false, confidence: 0, reason: reason || `no usable proposal: ${raw.trim().slice(0, 120)}` };
	const jaw = Array.isArray(j.jaw) ? [unit(j.jaw[0]), unit(j.jaw[1])] : [];
	return {
		found: true,
		point,
		...(jaw[0] && jaw[1] ? { jaw: [jaw[0], jaw[1]] as [[number, number], [number, number]] } : {}),
		approach: j.approach === "side" ? "side" : "top",
		// Never above 0.5: a point from one image is weaker evidence than a grasp planner's candidate.
		confidence: Math.min(0.5, Math.max(0, Number(j.confidence) || 0)),
		reason,
	};
}

/** What the point mode needs besides the advisor rig: the world point under a pixel of a camera image. */
export type PointRig = AdvisorRig & {
	/** The camera the VLM looks at (the robot's overview camera). */
	camera: string;
	/** World xyz of pixel (row, col) of `camera`'s current image, or null without depth there. */
	backProject: (camera: string, row: number, col: number) => Promise<number[] | null>;
	cameraPosition?: (camera: string) => Promise<number[] | undefined>;
};

/**
 * suggest_grasp without a grasp backend (7.2): the VLM proposes a grasp point, the two jaw
 * contacts and an approach on the camera image; the point is back-projected through depth to a
 * world position, the jaw line to a yaw about world z, the approach to a direction. It is a
 * low-confidence candidate (no width, collision or reachability check), marked as such.
 */
export function suggestGraspPoint(pi: ExtensionAPI, rig: PointRig): GraspToolDef {
	return {
		name: "suggest_grasp",
		description:
			"No grasp planner here: a separate vision model proposes where to grasp an object on the camera image (the grasp centre, the two jaw contacts, top or side approach), back-projected through depth to a world grasp: position, approach, eef_yaw and a pre-grasp point 10 cm back along the approach. Low confidence by construction (one image, no width or collision check): look at the marked image before moving, approach through pregrasp_xyz, and confirm the grasp in the wrist view.",
		parameters: Type.Object({
			object: Type.String({ description: "The object to grasp, e.g. 'black bowl'" }),
		}),
		run: async (p, signal, ctx: ExtensionContext) => {
			const object = String((p as Json).object ?? "").trim();
			if (!object) return { error: "object must be non-empty" };
			const started = Date.now();
			const img = await rig.image(rig.camera);
			const images: ImageContent[] = [
				{
					type: "image",
					data: encodePng(img.rgb, img.width, img.height).toString("base64"),
					mimeType: "image/png",
				},
			];
			const modelRef = String(pi.getFlag("grasp-advisor-model") || auxModel(pi, "attach"));
			let proposal: PointProposal;
			let model = "";
			let cost = 0;
			try {
				const reply = await askVlm(
					ctx,
					modelRef,
					pi.getThinkingLevel(),
					{
						system: POINT_SYSTEM,
						content: [{ type: "text", text: `TASK: ${rig.task()}\nOBJECT TO GRASP: ${object}` }],
					},
					images,
					signal,
				);
				pi.events.emit(VLM_COST_EVENT, reply.cost);
				proposal = parsePointProposal(reply.text);
				model = reply.model;
				cost = reply.cost;
			} catch (err) {
				return {
					mode: "vlm_point",
					confidence_level: "low",
					found: false,
					error: `grasp point proposal unavailable: ${message(err)}`,
				};
			}
			const px = (u: [number, number]) => [
				Math.min(img.height - 1, Math.round(u[0] * (img.height - 1))),
				Math.min(img.width - 1, Math.round(u[1] * (img.width - 1))),
			];
			const base: Json = {
				name: "suggest_grasp",
				mode: "vlm_point",
				confidence_level: "low",
				object,
				camera: rig.camera,
				model,
				cost_usd: cost,
			};
			let out: Json;
			if (!proposal.found || !proposal.point) out = { ...base, found: false, reason: proposal.reason };
			else {
				const [row, col] = px(proposal.point);
				const position = await rig.backProject(rig.camera, row, col);
				if (!position)
					out = {
						...base,
						found: false,
						pixel: [row, col],
						reason: "no depth at the proposed point; ask again or segment",
					};
				else {
					let approach = [0, 0, -1];
					const approachKind = proposal.approach === "side" ? "side" : "top";
					// A side approach comes horizontally from the camera's side toward the point.
					const eye = proposal.approach === "side" ? await rig.cameraPosition?.(rig.camera) : undefined;
					if (eye) {
						const h = [position[0] - eye[0], position[1] - eye[1], 0];
						const n = Math.hypot(...h);
						if (n > 1e-6) approach = h.map((v) => v / n);
					}
					let eef_yaw: number | undefined;
					const jawPx = proposal.jaw?.map(px);
					if (jawPx) {
						const [a, b] = await Promise.all(jawPx.map(([r, c]) => rig.backProject(rig.camera, r, c)));
						if (a && b && Math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-3)
							// The fingers close along the jaw line; the gripper's yaw is that line's direction about world z.
							eef_yaw = round(Math.atan2(b[1] - a[1], b[0] - a[0]), 4);
					}
					out = {
						...base,
						found: true,
						confidence: proposal.confidence,
						reason: proposal.reason,
						pixel: [row, col],
						...(jawPx ? { jaw_pixels: jawPx } : {}),
						candidate: {
							approach_kind: approach[2] === -1 ? "top" : approachKind,
							position: position.map((v) => round(v, 4)),
							approach: approach.map((v) => round(v, 4)),
							pregrasp_xyz: position.map((v, i) => round(v - 0.1 * approach[i], 4)),
							...(eef_yaw === undefined ? {} : { eef_yaw }),
						},
						note: "VLM-proposed from one image, not a grasp planner's candidate: no width, collision or reachability check.",
					};
					let marked = mark(img, row, col, [255, 32, 32]);
					for (const [r, c] of jawPx ?? []) marked = mark({ ...img, rgb: marked }, r, c, [0, 220, 0]);
					out._pngs = [encodePng(marked, img.width, img.height)];
				}
			}
			const { _pngs, ...entry } = out;
			pi.appendEntry(GRASP_SUGGESTION_ENTRY, { ...entry, ms: Date.now() - started });
			return out;
		},
	};
}

/** Register --grasp-advisor and --grasp-advisor-model; the returned function mounts and names the tool when on. */
export function graspAdvisorTool(pi: ExtensionAPI, rig: AdvisorRig, mount: (d: GraspToolDef) => void) {
	pi.registerFlag("grasp-advisor-model", {
		type: "string",
		default: "",
		description: "Model (provider/id) of suggest_grasp (default: --aux-model, else the session's model)",
	});
	let names: string[] | undefined;
	const plans = trackPlans(pi, rig.stamp);
	pi.registerFlag("grasp-advisor", {
		type: "boolean",
		default: false,
		description: "Add suggest_grasp (a separate VLM picks among plan_grasp's candidates); needs a grasp backend",
	});
	let mode: "rank" | "point" | undefined;
	return (graspOn: boolean) => {
		if (pi.getFlag("grasp-advisor") !== true) return [];
		const want = graspOn ? "rank" : "point";
		if (!graspOn && !rig.point)
			throw new Error(
				"--grasp-advisor needs a plan_grasp backend (--grasp contact_graspnet|graspgenx|anygrasp|graspnet1b)",
			);
		if (mode && mode !== want)
			throw new Error("--grasp-advisor: the grasp backend changed between sessions; restart pi");
		if (!names) {
			// With a backend: rank plan_grasp's candidates; without one: propose a grasp point (low confidence).
			mount(
				graspOn ? suggestGrasp(pi, rig, plans) : suggestGraspPoint(pi, { ...rig, ...(rig.point as PointExtras) }),
			);
			names = ["suggest_grasp"];
			mode = want;
		}
		return names;
	};
}
