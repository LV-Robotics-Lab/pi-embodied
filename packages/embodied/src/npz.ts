import { writeFileSync } from "node:fs";
import { crc32, deflateRawSync } from "node:zlib";

/** One numpy array: dtype descr (e.g. "<f4", "|u1", "|b1", "<U12"), shape, C-order bytes. */
export type Npy = { descr: string; shape: number[]; data: Buffer };

/** `.npy` v1.0 bytes, header padded to 64 like numpy's own writer. */
function npy({ descr, shape, data }: Npy): Buffer {
	const dims = shape.length === 1 ? `${shape[0]},` : shape.join(", ");
	const dict = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${dims}), }`;
	const header = `${dict}${" ".repeat((64 - ((11 + dict.length) % 64)) % 64)}\n`;
	const prefix = Buffer.alloc(10);
	prefix.write("\x93NUMPY", 0, "latin1");
	prefix[6] = 1;
	prefix.writeUInt16LE(header.length, 8);
	return Buffer.concat([prefix, Buffer.from(header, "latin1"), data]);
}

/** Write arrays as a deflated `.npz` (what `np.savez_compressed` produces; `np.load` reads it). */
export function writeNpz(path: string, arrays: Record<string, Npy>): void {
	const parts: Buffer[] = [];
	const central: Buffer[] = [];
	let offset = 0;
	for (const [key, array] of Object.entries(arrays)) {
		const name = Buffer.from(`${key}.npy`);
		const raw = npy(array);
		const body = deflateRawSync(raw);
		if (raw.length > 0xffffffff || offset + body.length > 0xffffffff) throw new Error("npz over 4 GiB (no zip64)");
		// version, flags, method=deflate, time, date=1980-01-01, crc, sizes, name length, extra length
		const common = Buffer.alloc(26);
		common.writeUInt16LE(20, 0);
		common.writeUInt16LE(8, 4);
		common.writeUInt16LE(0x21, 8);
		common.writeUInt32LE(crc32(raw) >>> 0, 10);
		common.writeUInt32LE(body.length, 14);
		common.writeUInt32LE(raw.length, 18);
		common.writeUInt16LE(name.length, 22);
		const local = Buffer.alloc(4);
		local.writeUInt32LE(0x04034b50, 0);
		const entry = Buffer.alloc(46);
		entry.writeUInt32LE(0x02014b50, 0);
		entry.writeUInt16LE(20, 4);
		common.copy(entry, 6);
		entry.writeUInt32LE(offset, 42);
		parts.push(local, common, name, body);
		central.push(entry, name);
		offset += 30 + name.length + body.length;
	}
	const size = central.reduce((n, b) => n + b.length, 0);
	const end = Buffer.alloc(22);
	end.writeUInt32LE(0x06054b50, 0);
	end.writeUInt16LE(central.length / 2, 8);
	end.writeUInt16LE(central.length / 2, 10);
	end.writeUInt32LE(size, 12);
	end.writeUInt32LE(offset, 16);
	writeFileSync(path, Buffer.concat([...parts, ...central, end]));
}
