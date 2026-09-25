/**
 * Episode video: the agentview frame of every env step, written at
 * 20 fps to `episode.mp4` when the session ends, plus `action_<n>_<tool>.mp4` for each
 * tool call that stepped the env when --action-clips is set. Files go to
 * `<--video-dir>/<session id>/`, or next to the session file (`<session>.jsonl` ->
 * `<session>/`). Encoding pipes raw RGB into ffmpeg: --ffmpeg, else `ffmpeg` on PATH,
 * else the binary bundled with imageio-ffmpeg in the services venv (--python).
 *
 * --video-overlay also writes `episode_overlay.mp4` (Show-Harness's annotated visualization): every
 * frame carries the action it belongs to, as a band on top: the action index, who acted (AGENT, or
 * HUMAN for an operator's unit from ../gumi), the tool or unit (`act MV_FWD x3`), the commanded
 * gripper and its measured width. The text is drawn in TypeScript with a built-in 5x7 font, so any
 * ffmpeg build works; frames narrower than 480 px are scaled up (nearest neighbour) to stay legible.
 * `episode.mp4` stays the plain frames.
 *
 * Every frame is also published on `pi.events` (FRAME_EVENT) for live views (../dashboard).
 */

import { execFileSync, spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { NdArray } from "./rpc.ts";
import { UNITS_EVENT, type UnitsHandle } from "./units/index.ts";

const FPS = 20;
/** `pi.events` channel carrying every env frame (an HxWx3 uint8 NdArray) as the robot records it. */
export const FRAME_EVENT = "pi-embodied:frame";
/**
 * `pi.events` channel on which a module names the action the next frames belong to, when it is no
 * agent tool call (../dashboard: an operator's GUMI units). `actor: null` ends that action.
 */
export const NOTE_EVENT = "pi-embodied:video-note";
export type VideoNote = { actor: "agent" | "human" | null; action?: string };

/** What a frame of the overlay video says. `width` is filled in when the robot's state arrives. */
export type Note = {
	n: number;
	actor: "agent" | "human";
	action: string;
	gripper: "OPEN" | "CLOSED" | null;
	width: number | null;
};

function findFfmpeg(python: string): string {
	try {
		execFileSync("ffmpeg", ["-version"], { stdio: "ignore" });
		return "ffmpeg";
	} catch {
		const code = "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())";
		return execFileSync(python, ["-c", code], { encoding: "utf8" }).trim();
	}
}

/** Encode `width`x`height` RGB24 frames as H.264 mp4 (yuv420p, like imageio.mimwrite), scaled `scale`x by nearest neighbour. */
async function encode(
	path: string,
	width: number,
	height: number,
	frames: Iterable<Buffer>,
	ffmpeg: string,
	scale = 1,
): Promise<void> {
	const args = ["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", `${width}x${height}`];
	args.push("-r", String(FPS), "-i", "-", "-an");
	if (scale > 1) args.push("-vf", `scale=iw*${scale}:ih*${scale}:flags=neighbor`);
	args.push("-c:v", "libx264", "-pix_fmt", "yuv420p", path);
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
	await pipeline(Readable.from(frames), proc.stdin).catch(() => {}); // exit status reports it
	const code = await exited;
	if (code !== 0) throw new Error(`ffmpeg exited ${code}: ${stderr.trim()}`);
}

/** Encode HxWx3 uint8 frames as H.264 mp4 (yuv420p, like imageio.mimwrite). */
export async function writeMp4(path: string, frames: NdArray[], ffmpeg: string): Promise<void> {
	const [height, width] = frames[0].shape;
	await encode(
		path,
		width,
		height,
		frames.map((f) => f.data),
		ffmpeg,
	);
}

// ---------------------------------------------------------------------------
// the overlay

/** A 5x7 font: each glyph is its character and seven rows as two hex digits (bit 4 = leftmost column). */
const FONT_DATA =
	"A0e11111f111111B1e11111e11111eC0e11101010110eD1e11111111111eE1f10101e10101fF1f10101e101010G0e11101711110fH1111111f111111I0e04040404040eJ0702020202120cK11121418141211L1010101010101fM111b1515111111N11111915131111O0e11111111110eP1e11111e101010Q0e11111115120dR1e11111e141211S0f10100e01011eT1f040404040404U1111111111110eV11111111110a04W1111111515150aX11110a040a1111Y11110a04040404Z1f01020408101f00e11131519110e1040c040404040e20e11010204081f31f02040201110e402060a121f020251f101e0101110e60608101e11110e71f01020408080880e11110e11110e90e11110f01020c 00000000000000#0a0a1f0a1f0a0a.00000000000c0c,000000000c0408:000c0c000c0c00/00010204081000-0000001f000000_0000000000001f(02040808080402)08040202020408[0e08080808080e]0e02020202020e=00001f001f0000+0004041f040400%18190204081303|04040404040404*0004150e150400>08040201020408<02040810080402'0c040800000000?0e110102040004!04040404040004";
const FONT = new Map<string, number[]>();
for (let i = 0; i < FONT_DATA.length; i += 15)
	FONT.set(
		FONT_DATA[i],
		Array.from({ length: 7 }, (_, r) => Number.parseInt(FONT_DATA.slice(i + 1 + 2 * r, i + 3 + 2 * r), 16)),
	);
/** Glyph cell: 5x7 plus one column and one row of spacing. */
const CELL = { w: 6, h: 8 };
const COLORS = { agent: [120, 200, 255], human: [255, 170, 60], text: [235, 235, 235] } as const;

/** Draw `text` (upper-cased; unknown characters become `?`) into an RGB image at (x, y), clipped to the image. */
export function drawText(
	rgb: Buffer,
	width: number,
	height: number,
	x: number,
	y: number,
	text: string,
	color: readonly number[],
) {
	let cx = x;
	for (const ch of text.toUpperCase()) {
		const glyph = FONT.get(ch) ?? FONT.get("?");
		if (cx + 5 > width) break;
		if (glyph)
			for (let r = 0; r < 7; r++)
				for (let c = 0; c < 5; c++) {
					if (!(glyph[r] & (16 >> c))) continue;
					const px = cx + c;
					const py = y + r;
					if (py < 0 || py >= height) continue;
					rgb.set(color, (py * width + px) * 3);
				}
		cx += CELL.w;
	}
}

/** The band's lines for one frame. */
export function noteLines(note: Note | undefined, frame: number): string[] {
	if (!note) return ["-", `frame ${frame}`];
	const width = note.width === null ? "" : `${(note.width * 100).toFixed(1)} cm`;
	const grip = note.gripper || width ? ["gripper", note.gripper ?? "", width].filter(Boolean).join(" ") : "";
	return [`#${note.n} ${note.actor} ${note.action}`, [grip, `frame ${frame}`].filter(Boolean).join("  ")];
}

/** A copy of one HxWx3 frame with the note's band drawn on top (the top rows darkened, then the text). */
export function annotate(frame: NdArray, note: Note | undefined, index: number): Buffer {
	const [height, width] = frame.shape;
	const out = Buffer.from(frame.data);
	const lines = noteLines(note, index);
	const band = Math.min(height, lines.length * CELL.h + 3);
	for (let i = 0; i < band * width * 3; i++) out[i] = out[i] * 0.3;
	lines.forEach((line, k) => {
		drawText(out, width, height, 2, 2 + k * CELL.h, line, k === 0 && note ? COLORS[note.actor] : COLORS.text);
	});
	return out;
}

/** The gripper command an action implies: GRASP / RELEASE units, a `gripper` argument, LIBERO's `release`. */
function commanded(tool: string, args: unknown): "OPEN" | "CLOSED" | undefined {
	const a = (args && typeof args === "object" ? args : {}) as Record<string, unknown>;
	const unit = typeof a.unit === "string" ? a.unit.toUpperCase() : tool.toUpperCase();
	if (unit === "GRASP") return "CLOSED";
	if (unit === "RELEASE") return "OPEN";
	const g = typeof a.gripper === "string" ? a.gripper.toLowerCase() : "";
	if (g === "close" || g === "closed") return "CLOSED";
	if (g === "open") return "OPEN";
	return undefined;
}

/** `act MV_FWD x3`, or the tool and its arguments, clipped. */
function actionLabel(tool: string, args: unknown): string {
	const a = (args && typeof args === "object" ? args : {}) as Record<string, unknown>;
	if (typeof a.unit === "string") {
		const n = Number(a.n ?? 1);
		const plan = Array.isArray(a.plan) && a.plan.length ? ` plan ${a.plan.join(",")}` : "";
		return `${tool} ${a.unit}${n > 1 ? ` x${n}` : ""}${typeof a.arm === "string" ? ` (${a.arm})` : ""}${plan}`;
	}
	const json = JSON.stringify(args ?? {});
	return json === "{}" ? tool : `${tool} ${json.length > 48 ? `${json.slice(0, 47)}~` : json}`;
}

export function episodeVideo(pi: ExtensionAPI) {
	pi.registerFlag("video-dir", { type: "string", description: "Episode videos go to <dir>/<session id>/" });
	pi.registerFlag("action-clips", {
		type: "boolean",
		default: false,
		description: "Also save one clip per tool call",
	});
	pi.registerFlag("video-overlay", {
		type: "boolean",
		default: false,
		description: "Also save episode_overlay.mp4 with the step, actor, action and gripper burnt in",
	});
	pi.registerFlag("ffmpeg", { type: "string", description: "ffmpeg binary for episode videos" });

	let frames: NdArray[] = [];
	/** The action each frame belongs to (--video-overlay). */
	let notes: (Note | undefined)[] = [];
	let note: Note | undefined;
	/** The running agent tool's note: an operator batch that interrupts it hands the frames back to it. */
	let agentNote: Note | undefined;
	let actions = 0;
	let gripper: Note["gripper"] = null;
	let handle: UnitsHandle | undefined;
	let clipStart = 0;
	let clips = 0;
	let dir = "";
	let ffmpeg = "";
	const saving: Promise<unknown>[] = [];

	pi.events.on(UNITS_EVENT, (data) => {
		handle = data as UnitsHandle;
	});

	/** A new action: the next frames carry it; the gripper width is read once from the robot's proprioception. */
	function announce(actor: Note["actor"], action: string, command: Note["gripper"] | undefined): Note {
		if (command) gripper = command;
		const next: Note = { n: ++actions, actor, action, gripper, width: null };
		note = next;
		handle
			?.state?.()
			.then((s) => {
				if (typeof s.gripper_width === "number") next.width = s.gripper_width;
			})
			.catch(() => {});
		return next;
	}

	pi.events.on(NOTE_EVENT, (data) => {
		const n = data as VideoNote;
		if (n.actor === null) note = agentNote;
		else announce(n.actor, n.action ?? "", commanded(n.action ?? "", { unit: n.action }));
	});

	async function save(name: string, clip: NdArray[], ctx: ExtensionContext, clipNotes?: (Note | undefined)[]) {
		try {
			ffmpeg ||=
				String(pi.getFlag("ffmpeg") || "") ||
				findFfmpeg(String(pi.getFlag("python") || process.env.PI_EMBODIED_PYTHON || "python"));
			mkdirSync(dir, { recursive: true });
			if (!clipNotes) await writeMp4(join(dir, name), clip, ffmpeg);
			else {
				const [height, width] = clip[0].shape;
				const annotated = (function* () {
					for (let i = 0; i < clip.length; i++) yield annotate(clip[i], clipNotes[i], i);
				})();
				await encode(join(dir, name), width, height, annotated, ffmpeg, Math.min(4, Math.ceil(480 / width)));
			}
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
		notes = [];
		note = agentNote = undefined;
		actions = 0;
		gripper = null;
		clips = 0;
	});

	pi.on("tool_execution_start", (event) => {
		clipStart = frames.length;
		if (event.toolName === "finish") return;
		agentNote = announce("agent", actionLabel(event.toolName, event.args), commanded(event.toolName, event.args));
	});

	pi.on("tool_execution_end", (event, ctx) => {
		if (note === agentNote) note = undefined;
		// The units header states the gripper command after the call, assists included (a recovery reopen).
		const content = (event.result as { content?: { type: string; text?: string }[] } | undefined)?.content ?? [];
		const said = content.map((c) =>
			c.type === "text" ? /commanded (OPEN|CLOSE)\b/.exec(c.text ?? "")?.[1] : undefined,
		);
		const last = said.find(Boolean);
		if (last) gripper = last === "OPEN" ? "OPEN" : "CLOSED";
		agentNote = undefined;
		if (!pi.getFlag("action-clips") || frames.length === clipStart) return;
		const name = `action_${String(++clips).padStart(3, "0")}_${event.toolName}.mp4`;
		saving.push(save(name, frames.slice(clipStart), ctx));
	});

	pi.on("session_shutdown", async (_event, ctx) => {
		const clip = frames;
		const clipNotes = notes;
		frames = [];
		notes = [];
		if (clip.length) {
			const path = await save("episode.mp4", clip, ctx);
			const overlay =
				pi.getFlag("video-overlay") === true ? await save("episode_overlay.mp4", clip, ctx, clipNotes) : undefined;
			if (path)
				pi.appendEntry("episode_video", { path, frames: clip.length, fps: FPS, ...(overlay ? { overlay } : {}) });
		}
		await Promise.all(saving.splice(0));
	});

	return {
		/** The action the next frame belongs to (--video-overlay's label). */
		get note() {
			return note;
		},
		/** One agentview frame (HxWx3 uint8) per env step, in step order. */
		frame(image: NdArray) {
			frames.push(image);
			notes.push(note);
			pi.events.emit(FRAME_EVENT, image);
		},
	};
}
