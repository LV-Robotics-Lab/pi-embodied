/**
 * A cap on concurrent planner calls across pi processes, for eval-parallel.sh --max-api-concurrency.
 *
 *   pi -e packages/embodied/src/api-gate.ts --api-slots <dir> --max-api-concurrency <n> ...
 *
 * Each model call takes one of n slot files in <dir> (created exclusively, holding the pid) before
 * the request and gives it back when the assistant message ends, so N workers whose episodes spend
 * most of their time in the simulator or the VLA share n model calls. A slot whose pid is gone (a
 * killed pi) is taken over; two processes taking over the same dead slot at once can let one extra
 * call through. The gate announces itself on `pi.events` (API_GATE_EVENT) so the robot extension's
 * side VLM calls (../vdm.ts) take a slot too.
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

/** Take a free slot of `n` in `dir` (waiting as long as all are held, or until `signal` aborts); returns its path. */
export async function acquire(dir: string, n: number, pollMs = 250, signal?: AbortSignal): Promise<string> {
	mkdirSync(dir, { recursive: true });
	for (;;) {
		if (signal?.aborted) throw new Error("aborted while waiting for a model-call slot");
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
		await new Promise<void>((r) => {
			const t = setTimeout(() => {
				signal?.removeEventListener("abort", stop);
				r();
			}, pollMs);
			const stop = () => {
				clearTimeout(t);
				r();
			};
			signal?.addEventListener("abort", stop, { once: true });
		});
	}
}

/** `pi.events` channel on which the gate publishes `{ dir, n }` at session start; other extensions' side model calls share the slots through `acquire`. */
export const API_GATE_EVENT = "pi-embodied:api-gate";

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
	const config = () => {
		const dir = pi.getFlag("api-slots") as string | undefined;
		const n = Number(pi.getFlag("max-api-concurrency"));
		return dir && n > 0 ? { dir, n } : undefined;
	};
	pi.on("session_start", () => {
		const g = config();
		if (g) pi.events.emit(API_GATE_EVENT, g);
	});
	pi.on("context", async (_event, ctx) => {
		const g = config();
		if (held || !g) return;
		// An abort (the user's, or the robot's budget) must not wait behind the other workers' calls.
		held = await acquire(g.dir, g.n, 250, ctx.signal);
	});
	pi.on("message_end", (event) => {
		if (event.message.role === "assistant") free();
	});
	pi.on("agent_end", free);
	pi.on("session_shutdown", free);
	process.once("exit", free);
}
