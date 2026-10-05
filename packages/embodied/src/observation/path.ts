/**
 * How a robot is observed without pi (the MCP server's `observe`, ../integrations/mcp/session.ts):
 * the facade methods its manifest declares as code primitives, which are the methods the robot's own
 * observe tool (`view_env_state`, ../robots/<robot>/index.ts) calls: `get_observation` when the robot
 * has one, else `render_camera` per camera and `get_state`. The manifest's code-mode parameters are
 * the facade's signature (the env server pins them to the method at its start, services'
 * components/manifest.py), so a required parameter without a declared value is an error here, never
 * a guess the server would refuse.
 *
 * What the manifest cannot say, the robot does, in `OBSERVATION` exported from its index.ts next to
 * the constants its own tools use: its cameras (the tool-side name a caller knows, e.g. `wrist`, and
 * the facade's `camera_name` for it, e.g. LIBERO's `robot0_eye_in_hand`; RoboCasa's `render_camera`
 * takes the sim's names as a plain string), the image size its tools render at (RoboCasa requires
 * `height` and `width`), and whether the sim renders bottom-up (MuJoCo: LIBERO, RoboCasa), in which
 * case the rows are flipped as the robot's tools flip them. A robot without one is observed from its
 * manifest alone: the enum's camera names, the server's default size, no flip.
 */

import { inMode } from "../primitives/arguments.ts";
import { available, enumValues, type Manifest, type Vars } from "../primitives/manifest.ts";

/** A robot's declaration (`OBSERVATION` in its index.ts): cameras as name → facade camera_name (or just names), size, orientation. */
export type ObservationDecl = {
	cameras?: Readonly<Record<string, string>> | readonly string[];
	/** The square image size the robot's tools render at (`height` and `width`). */
	size?: number;
	/** The sim renders bottom-up: flip the rows, as the robot's tools do. */
	flip?: boolean;
};

export type ObservationPath = {
	/** The zero-argument method that returns every camera and the state (`env.get_observation`). */
	observation?: string;
	render?: {
		method: string;
		/** Camera name → the facade's `camera_name`. */
		cameras: Readonly<Record<string, string>>;
		/** The method's kwargs for one camera: its required parameters filled from the declaration. */
		kwargs: (cameraName: string) => Record<string, unknown>;
		flip: boolean;
	};
	/** The zero-argument state method (`env.get_state`; RoboTwin's is `env.policy_frame`). */
	state?: string;
};

/** Rows of an HxWxC byte image in reverse order (a sim that renders bottom-up). */
export function flipRows(data: Buffer, height: number, rowBytes = data.length / height): Buffer {
	const out = Buffer.alloc(data.length);
	for (let y = 0; y < height; y++) data.copy(out, (height - 1 - y) * rowBytes, y * rowBytes, (y + 1) * rowBytes);
	return out;
}

/** The robot's observation path from its manifest (code primitives this run has) and its declaration. */
export function observationPath(
	m: Manifest,
	has: (capability: string) => boolean,
	vars: Vars = {},
	decl: ObservationDecl = {},
): ObservationPath {
	const code = (name: string) =>
		m.primitives.find((e) => e.name === name && e.side !== "ts" && e.doc.code && available(e, has));
	const observation = code("get_observation");
	const render = code("render_camera");
	const state = code("get_state");
	const out: ObservationPath = {
		...(observation ? { observation: observation.method as string } : {}),
		...(state ? { state: state.method as string } : {}),
	};
	if (!render) return out;
	const params = Object.entries(render.params ?? {}).filter(([, p]) => inMode(p, "code"));
	const camera = render.params?.camera_name;
	let cameras: Record<string, string> | undefined;
	if (decl.cameras)
		cameras = Array.isArray(decl.cameras)
			? Object.fromEntries((decl.cameras as readonly string[]).map((c) => [c, c]))
			: { ...(decl.cameras as Record<string, string>) };
	else if (camera?.type === "enum")
		cameras = Object.fromEntries(enumValues(camera.values ?? [], vars).map((c) => [c, c]));
	if (!cameras || Object.keys(cameras).length === 0)
		throw new Error(
			`${m.robot}: ${render.method} declares no camera names (camera_name is ${camera?.type ?? "absent"}) and the robot exports no OBSERVATION cameras`,
		);
	// Every required parameter has a value now, or the path is an error (not a call the server would refuse).
	for (const [k, p] of params) {
		if (k === "camera_name" || !p.required) continue;
		if ((k === "height" || k === "width") && decl.size !== undefined) continue;
		if (k === "depth") continue;
		throw new Error(`${m.robot}: ${render.method} requires ${k}, which the robot's OBSERVATION does not give`);
	}
	const kwargs = (cameraName: string) => {
		const kw: Record<string, unknown> = {};
		for (const [k, p] of params) {
			if (k === "camera_name") kw[k] = cameraName;
			else if ((k === "height" || k === "width") && decl.size !== undefined) kw[k] = decl.size;
			else if (k === "depth" && p.required) kw[k] = false;
		}
		return kw;
	};
	return { ...out, render: { method: render.method as string, cameras, kwargs, flip: decl.flip === true } };
}

/** The robot's `OBSERVATION` declaration (../robots/<robot>/index.ts), or none for a robot module without one. */
export async function robotObservation(robot: string): Promise<ObservationDecl> {
	if (!/^[a-z0-9_]+$/.test(robot)) return {};
	try {
		const mod = (await import(new URL(`../robots/${robot}/index.ts`, import.meta.url).href)) as {
			OBSERVATION?: ObservationDecl;
		};
		return mod.OBSERVATION ?? {};
	} catch (err) {
		if ((err as { code?: string }).code === "ERR_MODULE_NOT_FOUND") return {};
		throw err;
	}
}
