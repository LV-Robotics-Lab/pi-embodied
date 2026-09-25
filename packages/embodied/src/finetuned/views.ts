/**
 * The camera-transform contract of Show-Harness's fine-tuned policies (core/record/images.prepare_view):
 * rotate/flip, then centre-crop to an aspect, then letterbox into an N x N square with the rigs' own
 * resize_with_pad (cv2.resize INTER_LINEAR, black bars). Training images went through it offline and
 * inference images must match them pixel for pixel, so the provider (./index.ts) and the training-data
 * glue (./prepare.ts) both call this one implementation. The resize reproduces OpenCV's 8-bit
 * INTER_LINEAR (fixed-point coefficients, the SIMD rounding, and the exact-2x INTER_AREA shortcut).
 *
 * Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
 * Licensed under the Apache License, Version 2.0.
 * Modified by pi-embodied: core/record/images.{rotate_and_flip,center_crop_to_aspect,prepare_view} and
 * core/franka/camera_utils.resize_with_pad ported to TypeScript, cv2.resize reimplemented.
 */

import { createHash } from "node:crypto";
import { inflateSync } from "node:zlib";
import type { Rgb } from "../robot.ts";

export type Flip = "none" | "vertical" | "horizontal" | "both";
/** One view's transform: rotation (deg CCW, multiple of 90), flip, crop aspect (w/h), letterbox size. */
export type ViewSpec = { rot: number; flip: Flip; crop?: number; square?: number };

/** `rot=90,flip=both,crop=1.3333,square=256`; "raw" (or "") is no transform. */
export function parseView(s: string): ViewSpec {
	const spec: ViewSpec = { rot: 0, flip: "none" };
	for (const part of s.split(",").map((p) => p.trim())) {
		if (!part || part === "raw") continue;
		const [k, v] = part.split("=").map((x) => x.trim());
		if (k === "rot") {
			if (!/^-?\d+$/.test(v) || Number(v) % 90) throw new Error(`rot must be a multiple of 90, got ${v}`);
			spec.rot = Number(v);
		} else if (k === "flip") {
			if (!["none", "vertical", "horizontal", "both"].includes(v)) throw new Error(`bad flip ${v}`);
			spec.flip = v as Flip;
		} else if (k === "crop" || k === "square") {
			const n = Number(v);
			if (!(n > 0)) throw new Error(`bad ${k} ${v}`);
			spec[k] = n;
		} else throw new Error(`unknown view key ${k} (rot, flip, crop, square)`);
	}
	return spec;
}

export const formatView = (v: ViewSpec) =>
	[
		`rot=${v.rot}`,
		`flip=${v.flip}`,
		...(v.crop ? [`crop=${v.crop}`] : []),
		...(v.square ? [`square=${v.square}`] : []),
	].join(",");

// ---------------------------------------------------------------------------
// PNG

/** Decode an 8-bit, non-interlaced gray / gray+alpha / RGB / RGBA PNG to RGB. */
export function decodePng(png: Buffer): Rgb {
	let pos = 8;
	let width = 0;
	let height = 0;
	let channels = 0;
	const idat: Buffer[] = [];
	while (pos < png.length) {
		const len = png.readUInt32BE(pos);
		const type = png.toString("ascii", pos + 4, pos + 8);
		const data = png.subarray(pos + 8, pos + 8 + len);
		if (type === "IHDR") {
			width = data.readUInt32BE(0);
			height = data.readUInt32BE(4);
			if (data[8] !== 8 || data[12] !== 0) throw new Error("only 8-bit non-interlaced PNG is supported");
			channels = { 0: 1, 2: 3, 4: 2, 6: 4 }[data[9]] ?? 0;
			if (!channels) throw new Error(`unsupported PNG color type ${data[9]}`);
		} else if (type === "IDAT") idat.push(data);
		pos += 12 + len;
	}
	const raw = inflateSync(Buffer.concat(idat));
	const stride = width * channels;
	let prev = new Uint8Array(stride);
	let cur = new Uint8Array(stride);
	const rgb = Buffer.alloc(width * height * 3);
	for (let y = 0; y < height; y++) {
		const filter = raw[y * (stride + 1)];
		const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
		for (let i = 0; i < stride; i++) {
			const a = i >= channels ? cur[i - channels] : 0;
			const b = prev[i];
			const c = i >= channels ? prev[i - channels] : 0;
			let pred = 0;
			if (filter === 1) pred = a;
			else if (filter === 2) pred = b;
			else if (filter === 3) pred = (a + b) >> 1;
			else if (filter === 4) {
				const p = a + b - c;
				const pa = Math.abs(p - a);
				const pb = Math.abs(p - b);
				const pc = Math.abs(p - c);
				pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
			}
			cur[i] = (line[i] + pred) & 0xff;
		}
		for (let x = 0; x < width; x++) {
			const o = (y * width + x) * 3;
			const s = x * channels;
			if (channels >= 3) {
				rgb[o] = cur[s];
				rgb[o + 1] = cur[s + 1];
				rgb[o + 2] = cur[s + 2];
			} else rgb[o] = rgb[o + 1] = rgb[o + 2] = cur[s];
		}
		[prev, cur] = [cur, prev];
	}
	return { width, height, rgb };
}

/** core/record/images.frame_fingerprint: sha1 (first 12 hex) of the uint8 HWC bytes, and the shape. */
export const fingerprint = (img: Rgb) => ({
	sha1: createHash("sha1").update(img.rgb).digest("hex").slice(0, 12),
	shape: [img.height, img.width, 3],
});

// ---------------------------------------------------------------------------
// the transforms

/** np.rot90 (k quarter turns counter-clockwise), then flipud / fliplr. */
export function rotateAndFlip(img: Rgb, degrees: number, flip: Flip): Rgb {
	let out = img;
	const k = (((Math.trunc(degrees) % 360) + 360) % 360) / 90;
	for (let t = 0; t < k; t++) {
		// One CCW quarter turn: out[i][j] = in[j][W-1-i], shape (W, H).
		const { width: w, height: h, rgb } = out;
		const next = Buffer.alloc(rgb.length);
		for (let i = 0; i < w; i++)
			for (let j = 0; j < h; j++) rgb.copy(next, (i * h + j) * 3, (j * w + (w - 1 - i)) * 3, (j * w + (w - i)) * 3);
		out = { width: h, height: w, rgb: next };
	}
	if (flip === "vertical" || flip === "both") {
		const { width: w, height: h, rgb } = out;
		const next = Buffer.alloc(rgb.length);
		for (let y = 0; y < h; y++) rgb.copy(next, y * w * 3, (h - 1 - y) * w * 3, (h - y) * w * 3);
		out = { width: w, height: h, rgb: next };
	}
	if (flip === "horizontal" || flip === "both") {
		const { width: w, height: h, rgb } = out;
		const next = Buffer.alloc(rgb.length);
		for (let y = 0; y < h; y++)
			for (let x = 0; x < w; x++) rgb.copy(next, (y * w + x) * 3, (y * w + (w - 1 - x)) * 3, (y * w + (w - x)) * 3);
		out = { width: w, height: h, rgb: next };
	}
	return out;
}

/** Python's round(): half to even. */
const roundHalfEven = (v: number) => {
	const f = Math.floor(v);
	const d = v - f;
	return d > 0.5 || (d === 0.5 && f % 2 !== 0) ? f + 1 : f;
};

function crop(img: Rgb, x0: number, y0: number, w: number, h: number): Rgb {
	const out = Buffer.alloc(w * h * 3);
	for (let y = 0; y < h; y++)
		img.rgb.copy(out, y * w * 3, ((y0 + y) * img.width + x0) * 3, ((y0 + y) * img.width + x0 + w) * 3);
	return { width: w, height: h, rgb: out };
}

/** core/record/images.center_crop_to_aspect (aspect = width / height). */
export function centerCropToAspect(img: Rgb, aspect: number | undefined): Rgb {
	if (!aspect || aspect <= 0) return img;
	const { width: w, height: h } = img;
	const current = w / h;
	if (Math.abs(current - aspect) < 1e-6) return img;
	if (current > aspect) {
		const nw = roundHalfEven(h * aspect);
		return crop(img, Math.floor((w - nw) / 2), 0, nw, h);
	}
	const nh = roundHalfEven(w / aspect);
	return crop(img, 0, Math.floor((h - nh) / 2), w, nh);
}

const COEF_BITS = 11;
const COEF_SCALE = 1 << COEF_BITS;
/** cv::saturate_cast<short>(float): round half to even. */
const toShort = (v: number) => {
	const r = Math.round(v);
	return r - v === 0.5 && r % 2 !== 0 ? r - 1 : r;
};
const clamp255 = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : v);

/**
 * cv::resize(..., INTER_LINEAR) coefficients along one axis: source index and fixed-point weights.
 * Columns clamp the tap at the borders; rows keep the weight and clamp only the row index.
 */
function linearTaps(src: number, dst: number, clampTaps: boolean) {
	const scale = 1 / (dst / src);
	const ofs = new Int32Array(dst);
	const a0 = new Int32Array(dst);
	const a1 = new Int32Array(dst);
	for (let d = 0; d < dst; d++) {
		let f = Math.fround((d + 0.5) * scale - 0.5);
		let s = Math.floor(f);
		f = Math.fround(f - s);
		if (clampTaps && s < 0) {
			f = 0;
			s = 0;
		}
		if (clampTaps && s >= src - 1) {
			f = 0;
			s = src - 1;
		}
		ofs[d] = s;
		a0[d] = toShort(Math.fround(Math.fround(1 - f) * COEF_SCALE));
		a1[d] = toShort(Math.fround(f * COEF_SCALE));
	}
	return { ofs, a0, a1 };
}

/**
 * cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR) for 8-bit RGB: an exact 2x reduction is
 * OpenCV's INTER_AREA fast path ((a+b+c+d+2)>>2), anything else the fixed-point separable linear
 * filter with the SIMD vertical pass ((((S0>>4)*b0)>>16) + ((S1>>4)*b1)>>16) + 2) >> 2).
 */
export function resizeLinear(img: Rgb, w: number, h: number): Rgb {
	const { width: sw, height: sh, rgb: src } = img;
	if (sw === w && sh === h) return { width: w, height: h, rgb: Buffer.from(src) };
	const out = Buffer.alloc(w * h * 3);
	if (sw === 2 * w && sh === 2 * h) {
		for (let y = 0; y < h; y++)
			for (let x = 0; x < w; x++)
				for (let c = 0; c < 3; c++) {
					const i = (2 * y * sw + 2 * x) * 3 + c;
					out[(y * w + x) * 3 + c] = (src[i] + src[i + 3] + src[i + sw * 3] + src[i + sw * 3 + 3] + 2) >> 2;
				}
		return { width: w, height: h, rgb: out };
	}
	const tx = linearTaps(sw, w, true);
	const ty = linearTaps(sh, h, false);
	const row = (sy: number) => {
		const r = new Int32Array(w * 3);
		const base = sy * sw * 3;
		for (let x = 0; x < w; x++) {
			const s0 = base + tx.ofs[x] * 3;
			const s1 = tx.ofs[x] + 1 < sw ? s0 + 3 : s0;
			for (let c = 0; c < 3; c++) r[x * 3 + c] = src[s0 + c] * tx.a0[x] + src[s1 + c] * tx.a1[x];
		}
		return r;
	};
	const cache = new Map<number, Int32Array>();
	const rowOf = (sy: number) => {
		const k = Math.min(Math.max(sy, 0), sh - 1);
		const hit = cache.get(k);
		if (hit) return hit;
		const r = row(k);
		cache.set(k, r);
		return r;
	};
	// v_mul_hi of int16 lanes: the high half of the 32-bit product (arithmetic shift).
	const mulHi = (a: number, b: number) => Math.floor((a * b) / 65536);
	for (let y = 0; y < h; y++) {
		const r0 = rowOf(ty.ofs[y]);
		const r1 = rowOf(ty.ofs[y] + 1);
		const b0 = ty.a0[y];
		const b1 = ty.a1[y];
		for (let i = 0; i < w * 3; i++)
			out[y * w * 3 + i] = clamp255((mulHi(r0[i] >> 4, b0) + mulHi(r1[i] >> 4, b1) + 2) >> 2);
	}
	return { width: w, height: h, rgb: out };
}

/** core/franka/camera_utils.resize_with_pad: equal-ratio scale, centred on black. */
export function resizeWithPad(img: Rgb, tw: number, th: number): Rgb {
	const scale = Math.min(tw / img.width, th / img.height);
	const nw = Math.trunc(img.width * scale);
	const nh = Math.trunc(img.height * scale);
	const resized = resizeLinear(img, nw, nh);
	const out = Buffer.alloc(tw * th * 3);
	const y0 = Math.floor((th - nh) / 2);
	const x0 = Math.floor((tw - nw) / 2);
	for (let y = 0; y < nh; y++) resized.rgb.copy(out, ((y0 + y) * tw + x0) * 3, y * nw * 3, (y + 1) * nw * 3);
	return { width: tw, height: th, rgb: out };
}

/** core/record/images.prepare_view: rotate/flip -> crop -> letterbox. */
export function prepareView(img: Rgb, spec: ViewSpec): Rgb {
	let v = rotateAndFlip(img, spec.rot, spec.flip);
	v = centerCropToAspect(v, spec.crop);
	if (spec.square) v = resizeWithPad(v, spec.square, spec.square);
	return v;
}
