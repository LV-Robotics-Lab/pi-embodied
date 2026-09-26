/**
 * The provider's camera transform as a command: PNGs in, PNGs out, one manifest per call.
 *
 *   node --experimental-strip-types packages/embodied/src/finetuned/transform.ts <manifest.json>
 *
 * The manifest names the transform once, `{"robot": "libero", "env_id": ""}` (the recording robot's
 * viewsFor(), as ./prepare.ts and ./index.ts pick it) or `{"views": {"agentview": "square=256",
 * "wrist": "rot=0,flip=both,crop=1.3333,square=256"}}` (./views.ts's parseView syntax, prepare.ts's
 * --agentview / --wrist), and lists `"jobs": [{"src": "<png>", "dst": "<png>", "camera": "agentview" |
 * "wrist"}, ...]`. Every source frame is decoded, put through prepareView and written with ../png.ts's
 * encoder: the three calls prepare.ts makes for a GUMI frame and index.ts makes for an inference frame,
 * so a training set built by another program (the LeRobot converter,
 * services/pi_embodied_services/finetuned/lerobot_to_rollouts.py) holds pixels identical to both. The
 * reply on stdout is `{"count": N, "views": {"agentview": "...", "wrist": "..."}}`.
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { encodePng } from "../png.ts";
import { viewsFor } from "./index.ts";
import { decodePng, formatView, parseView, prepareView, type ViewSpec } from "./views.ts";

export type Camera = "agentview" | "wrist";
export type TransformJob = { src: string; dst: string; camera: Camera };
export type Views = Record<Camera, ViewSpec>;

/** The manifest's transform: explicit view specs, else the robot's calibrated ones. */
export function resolveViews(m: { views?: Record<string, string>; robot?: string; env_id?: string }): Views {
	if (m.views) {
		if (typeof m.views.agentview !== "string" || typeof m.views.wrist !== "string")
			throw new Error('"views" needs agentview and wrist specs');
		return { agentview: parseView(m.views.agentview), wrist: parseView(m.views.wrist) };
	}
	const d = viewsFor(String(m.robot ?? ""), String(m.env_id ?? ""));
	if (!d) throw new Error(`no camera transform for robot "${m.robot}"; pass "views"`);
	return d;
}

/** Transform every job's source PNG into its destination; the number of files written. */
export function transformAll(jobs: TransformJob[], views: Views): number {
	for (const job of jobs) {
		const img = prepareView(decodePng(readFileSync(job.src)), views[job.camera]);
		mkdirSync(dirname(job.dst), { recursive: true });
		writeFileSync(job.dst, encodePng(img.rgb, img.width, img.height));
	}
	return jobs.length;
}

function main() {
	const path = process.argv[2];
	if (!path || process.argv.length !== 3) {
		console.error("usage: transform.ts <manifest.json>");
		process.exit(2);
	}
	const manifest = JSON.parse(readFileSync(path, "utf8")) as {
		jobs?: unknown;
		views?: Record<string, string>;
		robot?: string;
		env_id?: string;
	};
	if (!Array.isArray(manifest.jobs)) throw new Error(`${path}: no "jobs" array`);
	const jobs = manifest.jobs.map((j, i): TransformJob => {
		if (
			typeof j?.src !== "string" ||
			typeof j?.dst !== "string" ||
			(j?.camera !== "agentview" && j?.camera !== "wrist")
		)
			throw new Error(`${path}: job ${i} needs string src, dst and camera agentview|wrist`);
		return { src: resolve(j.src), dst: resolve(j.dst), camera: j.camera };
	});
	const views = resolveViews(manifest);
	const count = transformAll(jobs, views);
	console.log(
		JSON.stringify({ count, views: { agentview: formatView(views.agentview), wrist: formatView(views.wrist) } }),
	);
}

if (process.argv[1] && resolve(process.argv[1]) === new URL(import.meta.url).pathname) main();
