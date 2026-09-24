/**
 * Episode video as RPent records it: the agentview frame of every env step, written at
 * 20 fps to `episode.mp4` when the session ends, plus `action_<n>_<tool>.mp4` for each
 * tool call that stepped the env when --action-clips is set. Files go to
 * `<--video-dir>/<session id>/`, or next to the session file (`<session>.jsonl` ->
 * `<session>/`). Encoding pipes raw RGB into ffmpeg: --ffmpeg, else `ffmpeg` on PATH,
 * else the binary bundled with imageio-ffmpeg in the RPent venv (--python; what RPent uses).
 */

import { execFileSync, spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { NdArray } from "./rpc.ts";

const FPS = 20;

function findFfmpeg(python: string): string {
	try {
		execFileSync("ffmpeg", ["-version"], { stdio: "ignore" });
		return "ffmpeg";
	} catch {
		const code = "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())";
		return execFileSync(python, ["-c", code], { encoding: "utf8" }).trim();
	}
}

/** Encode HxWx3 uint8 frames as H.264 mp4 (yuv420p, like imageio.mimwrite). */
export async function writeMp4(path: string, frames: NdArray[], ffmpeg: string): Promise<void> {
	const [height, width] = frames[0].shape;
	const args = ["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", `${width}x${height}`];
	args.push("-r", String(FPS), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", path);
	const proc = spawn(ffmpeg, args, { stdio: ["pipe", "ignore", "pipe"] });
	let stderr = "";
	proc.stderr.on("data", (d) => {
		stderr += d;
	});
	const exited = new Promise<number | null>((resolve, reject) => {
		proc.once("error", reject);
		proc.once("close", resolve);
	});
	exited.catch(() => {});
	await pipeline(Readable.from(frames.map((f) => f.data)), proc.stdin).catch(() => {}); // exit status reports it
	const code = await exited;
	if (code !== 0) throw new Error(`ffmpeg exited ${code}: ${stderr.trim()}`);
}

export function episodeVideo(pi: ExtensionAPI) {
	pi.registerFlag("video-dir", { type: "string", description: "Episode videos go to <dir>/<session id>/" });
	pi.registerFlag("action-clips", {
		type: "boolean",
		default: false,
		description: "Also save one clip per tool call",
	});
	pi.registerFlag("ffmpeg", { type: "string", description: "ffmpeg binary for episode videos" });

	let frames: NdArray[] = [];
	let clipStart = 0;
	let clips = 0;
	let dir = "";
	let ffmpeg = "";
	const saving: Promise<unknown>[] = [];

	async function save(name: string, clip: NdArray[], ctx: ExtensionContext) {
		try {
			ffmpeg ||=
				String(pi.getFlag("ffmpeg") || "") ||
				findFfmpeg(String(pi.getFlag("python") || process.env.RPENT_PYTHON || "python"));
			mkdirSync(dir, { recursive: true });
			await writeMp4(join(dir, name), clip, ffmpeg);
			return join(dir, name);
		} catch (err) {
			const message = `[video] ${name} not saved: ${err instanceof Error ? err.message : err}`;
			if (ctx.hasUI) ctx.ui.notify(message, "warning");
			else console.error(message);
		}
	}

	pi.on("session_start", (_event, ctx) => {
		const sm = ctx.sessionManager;
		const base = pi.getFlag("video-dir");
		const file = sm.getSessionFile();
		if (base) dir = join(String(base), sm.getSessionId());
		else dir = file ? file.replace(/\.jsonl$/, "") : join(tmpdir(), "pi-embodied", sm.getSessionId());
		frames = [];
		clips = 0;
	});

	pi.on("tool_execution_start", () => {
		clipStart = frames.length;
	});

	pi.on("tool_execution_end", (event, ctx) => {
		if (!pi.getFlag("action-clips") || frames.length === clipStart) return;
		const name = `action_${String(++clips).padStart(3, "0")}_${event.toolName}.mp4`;
		saving.push(save(name, frames.slice(clipStart), ctx));
	});

	pi.on("session_shutdown", async (_event, ctx) => {
		const clip = frames;
		frames = [];
		if (clip.length) {
			const path = await save("episode.mp4", clip, ctx);
			if (path) pi.appendEntry("episode_video", { path, frames: clip.length, fps: FPS });
		}
		await Promise.all(saving.splice(0));
	});

	return {
		/** One agentview frame (HxWx3 uint8) per env step, in step order. */
		frame(image: NdArray) {
			frames.push(image);
		},
	};
}
