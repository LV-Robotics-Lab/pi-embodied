import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { NdArray } from "../src/rpc.ts";
import { UNITS_EVENT } from "../src/units/index.ts";
import { annotate, drawText, episodeVideo, FRAME_EVENT, NOTE_EVENT, type Note, noteLines } from "../src/video.ts";

const ffmpeg = (() => {
	try {
		execFileSync("ffmpeg", ["-version"], { stdio: "ignore" });
		return "ffmpeg";
	} catch {
		return undefined;
	}
})();

type Handler = (event: any, ctx: any) => unknown;

function fakePi(flagValues: Record<string, unknown>) {
	const handlers = new Map<string, Handler[]>();
	const listeners = new Map<string, ((data: unknown) => void)[]>();
	const flags: Record<string, unknown> = {};
	const entries: { customType: string; data: any }[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		appendEntry: (customType: string, data: unknown) => entries.push({ customType, data }),
		events: {
			emit: (channel: string, data: unknown) => {
				for (const fn of listeners.get(channel) ?? []) fn(data);
			},
			on: (channel: string, fn: (data: unknown) => void) => {
				listeners.set(channel, [...(listeners.get(channel) ?? []), fn]);
				return () => {};
			},
		},
	} as unknown as ExtensionAPI;
	const ctx = {
		hasUI: false,
		sessionManager: { getSessionFile: () => undefined, getSessionId: () => "sess" },
	};
	const emit = async (name: string, event: Record<string, unknown> = {}) => {
		for (const fn of handlers.get(name) ?? []) await fn(event, ctx);
	};
	return { pi, emit, entries };
}

const frame = (w: number, h: number, v = 200) => new NdArray("uint8", [h, w, 3], Buffer.alloc(w * h * 3, v));
const probe = (path: string) =>
	execFileSync(
		"ffprobe",
		[
			"-v",
			"error",
			"-count_frames",
			"-select_streams",
			"v:0",
			"-show_entries",
			"stream=width,height,nb_read_frames",
			"-of",
			"csv=p=0",
			path,
		],
		{
			encoding: "utf8",
		},
	).trim();

test("drawText draws known glyphs in the colour, clipped at the image edge", () => {
	const rgb = Buffer.alloc(12 * 8 * 3);
	drawText(rgb, 12, 8, 0, 0, "I1", [255, 0, 0]);
	// "I" row 0 is 01110: columns 1..3 lit, column 0 dark; the second glyph starts at x=6 and is cut at the edge.
	assert.deepEqual([...rgb.subarray(0, 3)], [0, 0, 0]);
	assert.deepEqual([...rgb.subarray(3, 6)], [255, 0, 0]);
	const lit = (x: number, y: number) => rgb[(y * 12 + x) * 3] === 255;
	assert.equal(lit(8, 0), true); // "1" row 0 is 00100
	assert.equal(lit(0, 7), false); // row 7 is spacing
});

test("noteLines and annotate: the action band on top, the rest of the frame untouched", () => {
	const note: Note = { n: 3, actor: "human", action: "MV_FWD", gripper: "CLOSED", width: 0.0123 };
	assert.deepEqual(noteLines(note, 42), ["#3 human MV_FWD", "gripper CLOSED 1.2 cm  frame 42"]);
	assert.deepEqual(noteLines(undefined, 0), ["-", "frame 0"]);
	assert.deepEqual(noteLines({ ...note, gripper: null, width: 0.078 }, 2)[1], "gripper 7.8 cm  frame 2");
	const f = frame(64, 48);
	const out = annotate(f, note, 42);
	assert.equal(out.length, f.data.length);
	assert.equal(f.data[0], 200); // the original is not modified
	assert.ok(out[0] < 100); // the band is darkened
	assert.equal(out[(40 * 64 + 10) * 3], 200); // below the band
	const band = out.subarray(0, 19 * 64 * 3);
	assert.ok(band.some((v, i) => i % 3 === 0 && v === 255)); // human = orange text (R 255)
});

test(
	"episodeVideo: plain episode.mp4 by default; --video-overlay adds the labelled copy",
	{ skip: !ffmpeg },
	async () => {
		for (const overlay of [false, true]) {
			const dir = mkdtempSync(join(tmpdir(), "video-"));
			const { pi, emit, entries } = fakePi({ "video-dir": dir, ffmpeg, "video-overlay": overlay });
			const seen: unknown[] = [];
			pi.events.on(FRAME_EVENT, (d) => seen.push(d));
			const video = episodeVideo(pi);
			await emit("session_start");
			let width = 0.08;
			pi.events.emit(UNITS_EVENT, { state: async () => ({ gripper_width: width }) });
			await emit("tool_execution_start", { toolName: "act", args: { unit: "GRASP" } });
			width = 0.01;
			for (let i = 0; i < 3; i++) video.frame(frame(64, 48, 50 * i));
			// A recovery reopened the gripper: the units header says so.
			await emit("tool_execution_end", {
				toolName: "act",
				result: { content: [{ type: "text", text: "width 8.0 cm, commanded OPEN." }] },
			});
			assert.ok(video.note === undefined);
			pi.events.emit(NOTE_EVENT, { actor: "human", action: "MV_UP" });
			assert.deepEqual(video.note, { n: 2, actor: "human", action: "MV_UP", gripper: "OPEN", width: null });
			video.frame(frame(64, 48));
			pi.events.emit(NOTE_EVENT, { actor: null });
			await emit("session_shutdown");
			assert.equal(seen.length, 4); // every frame is published for live views
			const [entry] = entries.filter((e) => e.customType === "episode_video");
			assert.equal(entry.data.frames, 4);
			assert.equal(probe(join(dir, "sess", "episode.mp4")), "64,48,4");
			assert.equal(existsSync(join(dir, "sess", "episode_overlay.mp4")), overlay);
			if (overlay) {
				assert.equal(entry.data.overlay, join(dir, "sess", "episode_overlay.mp4"));
				// Scaled up (nearest neighbour) to stay legible: 64 px -> 8x, capped at 4x.
				assert.equal(probe(entry.data.overlay), "256,192,4");
				// The agent's frames carry its colour (blue > red) in the band, the operator's orange (red > blue).
				const raw = execFileSync("ffmpeg", [
					"-v",
					"error",
					"-i",
					entry.data.overlay,
					"-f",
					"rawvideo",
					"-pix_fmt",
					"rgb24",
					"-",
				]);
				const size = 256 * 192 * 3;
				const tint = (k: number) => {
					let blue = 0;
					let red = 0;
					for (let i = k * size; i < k * size + 256 * 36 * 3; i += 3) {
						blue = Math.max(blue, raw[i + 2] - raw[i]);
						red = Math.max(red, raw[i] - raw[i + 2]);
					}
					return { blue, red };
				};
				assert.ok(tint(0).blue > 80 && tint(0).red < 40, JSON.stringify(tint(0)));
				assert.ok(tint(3).red > 80 && tint(3).blue < 40, JSON.stringify(tint(3)));
			} else assert.equal(entry.data.overlay, undefined);
		}
	},
);
