/**
 * A cap on concurrent planner calls across pi processes, for eval-parallel.sh --max-api-concurrency.
 *
 *   pi -e packages/embodied/src/api-gate.ts --api-slots <dir> --max-api-concurrency <n> ...
 *
 * Each model call takes one of n slot files in <dir> (created exclusively, holding the pid) before
 * the request and gives it back when the assistant message ends, so N workers whose episodes spend
 * most of their time in the simulator or the VLA share n model calls. A slot whose pid is gone (a
 * killed pi) is taken over; two processes taking over the same dead slot at once can let one extra
 * call through.
 */

import { closeSync, mkdirSync, openSync, readFileSync, unlinkSync, writeSync } from "node:fs";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const alive = (pid: number) => {
	try {
		process.kill(pid, 0);
		return true;
	} catch (e) {
		return (e as NodeJS.ErrnoException).code === "EPERM";
	}
};

/** Create `path` exclusively with this pid in it; false when it exists. */
function take(path: string) {
	try {
		const fd = openSync(path, "wx");
		writeSync(fd, String(process.pid));
		closeSync(fd);
		return true;
	} catch (e) {
		if ((e as NodeJS.ErrnoException).code !== "EEXIST") throw e;
		return false;
	}
}

/** Take a free slot of `n` in `dir` (waiting as long as all are held); returns its path. */
export async function acquire(dir: string, n: number, pollMs = 250): Promise<string> {
	mkdirSync(dir, { recursive: true });
	for (;;) {
		for (let i = 0; i < n; i++) {
			const path = join(dir, `slot-${i}`);
			if (take(path)) return path;
			let owner = 0;
			try {
				owner = Number(readFileSync(path, "utf8"));
			} catch {}
			// An empty file is a slot being written; only a pid that is gone is stale.
			if (owner > 0 && !alive(owner)) {
				try {
					unlinkSync(path);
				} catch {}
				if (take(path)) return path;
			}
		}
		await new Promise((r) => setTimeout(r, pollMs));
	}
}

export function release(path: string) {
	try {
		if (Number(readFileSync(path, "utf8")) === process.pid) unlinkSync(path);
	} catch {}
}

export default function apiGate(pi: ExtensionAPI) {
	pi.registerFlag("api-slots", { type: "string", description: "Directory of the shared model-call slots" });
	pi.registerFlag("max-api-concurrency", {
		type: "string",
		description: "Model calls at once across the pi processes sharing --api-slots",
	});
	let held: string | undefined;
	const free = () => {
		if (held) release(held);
		held = undefined;
	};
	pi.on("context", async () => {
		const dir = pi.getFlag("api-slots") as string | undefined;
		const n = Number(pi.getFlag("max-api-concurrency"));
		if (held || !dir || !(n > 0)) return;
		held = await acquire(dir, n);
	});
	pi.on("message_end", (event) => {
		if (event.message.role === "assistant") free();
	});
	pi.on("agent_end", free);
	pi.on("session_shutdown", free);
	process.once("exit", free);
}
