import { crc32, deflateSync, inflateSync } from "node:zlib";

function chunk(type: string, data: Buffer): Buffer {
	const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
	const out = Buffer.alloc(12 + data.length);
	out.writeUInt32BE(data.length, 0);
	body.copy(out, 4);
	out.writeUInt32BE(crc32(body) >>> 0, 8 + data.length);
	return out;
}

/** Encode 8-bit RGB pixels (row-major, `height * width * 3` bytes) as PNG. */
export function encodePng(rgb: Buffer, width: number, height: number): Buffer {
	const header = Buffer.alloc(13);
	header.writeUInt32BE(width, 0);
	header.writeUInt32BE(height, 4);
	header[8] = 8;
	header[9] = 2;
	const stride = width * 3;
	const raw = Buffer.alloc((stride + 1) * height);
	for (let y = 0; y < height; y++) rgb.copy(raw, y * (stride + 1) + 1, y * stride, (y + 1) * stride);
	return Buffer.concat([
		Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
		chunk("IHDR", header),
		chunk("IDAT", deflateSync(raw)),
		chunk("IEND", Buffer.alloc(0)),
	]);
}

/** Decode an 8-bit, non-interlaced PNG (gray, RGB or RGBA) to its first channel. */
export function decodePngChannel(png: Buffer): { width: number; height: number; data: Uint8Array } {
	let pos = 8;
	let width = 0;
	let height = 0;
	let channels = 1;
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
	const cur = new Uint8Array(stride);
	let prev = new Uint8Array(stride);
	const out = new Uint8Array(width * height);
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
		for (let x = 0; x < width; x++) out[y * width + x] = cur[x * channels];
		prev = cur.slice();
	}
	return { width, height, data: out };
}
