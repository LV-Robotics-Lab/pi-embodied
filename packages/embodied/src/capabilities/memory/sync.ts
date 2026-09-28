/**
 * Plain-HTTPS memory sync (the equivalent of huggingface_hub.snapshot_download of the
 * dataset, `<robot>/**` only), so no Python is needed and HF mirrors work.
 */

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, resolve, sep } from "node:path";
import { hasFiles, message } from "./corpus.ts";

export const HF_REPO = "RLinf/RPent-memory";

type HfFile = { rfilename: string; blobId?: string; lfs?: { sha256?: string } };
const hash = (algo: string, ...parts: (string | Buffer)[]) => {
	const h = createHash(algo);
	for (const p of parts) h.update(p);
	return h.digest("hex");
};
const unchanged = (path: string, f: HfFile) => {
	const data = readFileSync(path);
	return f.lfs?.sha256
		? hash("sha256", data) === f.lfs.sha256
		: hash("sha1", `blob ${data.length}\0`, data) === f.blobId;
};

/**
 * Plain-HTTPS `snapshot_download(repo, repo_type="dataset", allow_patterns=["<robot>/**"])` into the
 * memory home. Honors HF_ENDPOINT, HF_TOKEN, HF_HUB_OFFLINE=1, PI_EMBODIED_MEMORY_REPO and
 * PI_EMBODIED_MEMORY_REVISION (a branch or commit, default main; evaluation protocols pin one); files whose
 * git blob hash already matches are skipped; with a pinned revision, files it does not have are removed
 * (except `_internal/`); failures fall back to whatever is on disk.
 */
export async function syncMemory(dir: string, log: (m: string) => void, remote = HF_REPO): Promise<void> {
	const root = resolve(dir);
	const robot = basename(root);
	const repo = process.env.PI_EMBODIED_MEMORY_REPO ?? remote;
	const revision = process.env.PI_EMBODIED_MEMORY_REVISION || "main";
	if (process.env.HF_HUB_OFFLINE === "1") {
		if (!hasFiles(root)) log(`HF_HUB_OFFLINE=1 but no local memory was found under ${root}`);
		return;
	}
	const endpoint = (process.env.HF_ENDPOINT || "https://huggingface.co").replace(/\/+$/, "");
	const headers: Record<string, string> = process.env.HF_TOKEN
		? { authorization: `Bearer ${process.env.HF_TOKEN}` }
		: {};
	const get = async (url: string) => {
		const res = await fetch(url, { headers, signal: AbortSignal.timeout(60_000) });
		if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
		return res;
	};
	try {
		const info = (await (
			await get(`${endpoint}/api/datasets/${repo}/revision/${encodeURIComponent(revision)}?blobs=true`)
		).json()) as {
			sha: string;
			siblings: HfFile[];
		};
		const files = info.siblings.filter((f) => f.rfilename.startsWith(`${robot}/`));
		let next = 0;
		const worker = async () => {
			while (next < files.length) {
				const f = files[next++];
				const dest = resolve(dirname(root), f.rfilename);
				if (!dest.startsWith(root + sep) || (existsSync(dest) && unchanged(dest, f))) continue;
				const url = `${endpoint}/datasets/${repo}/resolve/${info.sha}/${f.rfilename.split("/").map(encodeURIComponent).join("/")}`;
				const data = Buffer.from(await (await get(url)).arrayBuffer());
				mkdirSync(dirname(dest), { recursive: true });
				writeFileSync(`${dest}.${process.pid}.tmp`, data);
				renameSync(`${dest}.${process.pid}.tmp`, dest);
			}
		};
		await Promise.all(Array.from({ length: 8 }, worker));
		// A pinned revision is a snapshot, not a merge: files another revision left behind would be read
		// as this one's memory. Unpinned syncs keep local files (explore's merges publish into the tree);
		// `_internal/` is local state either way.
		if (!process.env.PI_EMBODIED_MEMORY_REVISION || !files.length || !existsSync(root)) return;
		const wanted = new Set(files.map((f) => resolve(dirname(root), f.rfilename)));
		for (const rel of readdirSync(root, { recursive: true, encoding: "utf8" })) {
			const path = resolve(root, rel);
			if (rel.split(sep)[0] === "_internal" || wanted.has(path) || !statSync(path).isFile()) continue;
			rmSync(path);
		}
	} catch (e) {
		const local = hasFiles(root)
			? `continuing with local memory under ${root}`
			: `no local memory was found under ${root}`;
		log(`could not sync '${robot}' from '${repo}'@${revision}: ${message(e)}; ${local}`);
	}
}
