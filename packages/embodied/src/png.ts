import { crc32, deflateSync } from "node:zlib";
import type { NdArray } from "./rpc.ts";

function chunk(type: string, data: Buffer): Buffer {
	const len = Buffer.alloc(4);
	len.writeUInt32BE(data.length);
	const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
	const crc = Buffer.alloc(4);
	crc.writeUInt32BE(crc32(body) >>> 0);
	return Buffer.concat([len, body, crc]);
}

/** Encode an HxWx3 (RGB) or HxWx4 (RGBA) uint8 image as PNG. */
export function encodePng(image: NdArray): Buffer {
	if (image.dtype !== "uint8" || image.shape.length !== 3 || ![3, 4].includes(image.shape[2])) {
		throw new Error(`encodePng expects HxWx3|4 uint8, got ${image.dtype}[${image.shape.join(",")}]`);
	}
	const [height, width, channels] = image.shape;
	const header = Buffer.alloc(13);
	header.writeUInt32BE(width, 0);
	header.writeUInt32BE(height, 4);
	header[8] = 8; // bit depth
	header[9] = channels === 4 ? 6 : 2; // color type
	const stride = width * channels;
	const raw = Buffer.alloc((stride + 1) * height);
	for (let y = 0; y < height; y++) {
		raw[y * (stride + 1)] = 0; // filter: none
		image.data.copy(raw, y * (stride + 1) + 1, y * stride, (y + 1) * stride);
	}
	return Buffer.concat([
		Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
		chunk("IHDR", header),
		chunk("IDAT", deflateSync(raw)),
		chunk("IEND", Buffer.alloc(0)),
	]);
}
