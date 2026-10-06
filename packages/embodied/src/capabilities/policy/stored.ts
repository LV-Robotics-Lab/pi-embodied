import { spawn } from "node:child_process";
import { mkdir, realpath } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { openNodeSqliteStorage } from "@earendil-works/pi-durable/storage/sqlite/node";
import { openPolicy } from "./policy.ts";

/** Linux flock is released on process death; a second host never opens the same Durable storage. */
export async function openStoredPolicy(file: string, options: Omit<Parameters<typeof openPolicy>[0], "storage">) {
	await mkdir(dirname(resolve(file)), { recursive: true });
	const path = join(await realpath(dirname(resolve(file))), basename(file));
	const lock = spawn("flock", ["-n", "-E", "73", `${path}.lock`, "sh", "-c", "printf 'ready\\n'; cat >/dev/null"], {
		stdio: ["pipe", "pipe", "pipe"],
	});
	const exited = new Promise<void>((done) => lock.once("close", () => done()));
	await new Promise<void>((ready, reject) => {
		const timer = setTimeout(() => {
			lock.kill();
			reject(new Error("policy storage lock timed out"));
		}, 5000);
		lock.once("error", (error) => {
			clearTimeout(timer);
			reject(error);
		});
		lock.once("exit", (code) => {
			clearTimeout(timer);
			reject(new Error(`policy storage is already owned or flock failed (${code})`));
		});
		lock.stdout.once("data", () => {
			clearTimeout(timer);
			ready();
		});
	});
	try {
		const policy = await openPolicy({ ...options, storage: await openNodeSqliteStorage(path) });
		let closing: Promise<void> | undefined;
		const close = () => {
			closing ??= (async () => {
				try {
					await policy.close();
				} finally {
					lock.stdin.end();
					await exited;
				}
			})();
			return closing;
		};
		lock.once("exit", () => {
			void close();
		});
		return { ...policy, close };
	} catch (error) {
		lock.stdin.end();
		await exited;
		throw error;
	}
}
