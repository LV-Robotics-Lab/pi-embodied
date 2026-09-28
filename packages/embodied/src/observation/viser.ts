/**
 * Live 3D view (`--viser`): CaP-X's Viser scene (github.com/capgym/cap-x @53e9966,
 * capx/envs/simulators/robosuite_base.py `_update_viser_server`) for a robot's env server.
 *
 *   pi -e packages/embodied/src/robots/libero --viser [--viser-port 8080] [--viser-python <venv>/bin/python] ...
 *
 * A robot opts in with `viser` in its defineRobot spec (which source the view reads, and its env
 * server); ../robot.ts mounts this module. With `--viser`, every episode start launches
 * services/pi_embodied_services/components/viser_view.py (the `viser` extra) against the episode's
 * env server: the point cloud of each RGB-D camera, the camera frustums, the end-effector frame,
 * redrawn every --viser-interval seconds, and each `plan_grasp` / `plan_place` result's candidates
 * (the active one larger). The page is http://<host>:<port>/ (port printed at start and on the
 * dashboard's header, `RobotStatus.viser_port`). A view that fails to start is a warning, not a
 * failed episode. Off (the default), only the flags exist.
 */

import { type ChildProcess, spawn } from "node:child_process";
import { closeSync, openSync, writeSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { RpcClient } from "../infra/rpc.ts";

export type ViserSpec = {
	/** The view's source (viser_view.py SOURCES): how it reads cameras and the robot state. */
	source: "libero" | "franka";
	/** The episode's env server, once it is up. */
	env: () => RpcClient | undefined;
};

/** Tools whose result carries grasp / placement candidates to draw. */
const PLANS = ["plan_grasp", "plan_place"];

/** The `viser listening on http://host:port` and `RPC server listening on http://host:port` lines. */
export function parsePorts(text: string): { viser?: number; rpc?: number } {
	const viser = /viser listening on http:\/\/[^\s:]+:(\d+)/.exec(text)?.[1];
	const rpc = /RPC server listening on http:\/\/[^\s:]+:(\d+)/.exec(text)?.[1];
	return { viser: viser ? Number(viser) : undefined, rpc: rpc ? Number(rpc) : undefined };
}

export function viserView(pi: ExtensionAPI, spec: ViserSpec) {
	pi.registerFlag("viser", {
		type: "boolean",
		default: false,
		description: "Live 3D view (Viser): point clouds, camera frustums, EEF and grasp candidates",
	});
	pi.registerFlag("viser-port", { type: "string", default: "8080", description: "--viser: the page's port" });
	pi.registerFlag("viser-host", { type: "string", default: "0.0.0.0", description: "--viser: the page's host" });
	pi.registerFlag("viser-interval", {
		type: "string",
		default: "1",
		description: "--viser: seconds between reads of the env server",
	});
	pi.registerFlag("viser-python", {
		type: "string",
		default: "",
		description: "--viser: Python with the services' `viser` extra (default: --python)",
	});
	const on = () => pi.getFlag("viser") === true;
	let view: { proc: ChildProcess; rpc: RpcClient; port: number } | undefined;
	let hooked = false;

	async function stop() {
		const v = view;
		view = undefined;
		if (!v || v.proc.exitCode !== null || v.proc.signalCode !== null) return;
		const exited = new Promise<void>((resolve) => v.proc.once("exit", () => resolve()));
		// --parent-watch: EOF on stdin ends it; a kill after 5 s.
		v.proc.stdin?.end();
		const late = setTimeout(() => v.proc.kill("SIGKILL"), 5_000);
		await exited;
		clearTimeout(late);
	}

	async function launch(
		env: RpcClient,
		services: string,
	): Promise<{ proc: ChildProcess; rpc: RpcClient; port: number }> {
		const python = String(
			pi.getFlag("viser-python") || pi.getFlag("python") || process.env.PI_EMBODIED_PYTHON || "python",
		);
		const args = [
			...["-m", "pi_embodied_services.components.viser_view"],
			...["--robot", spec.source, "--env", env.url.replace(/\/call$/, "")],
			...["--viser-host", String(pi.getFlag("viser-host")), "--viser-port", String(pi.getFlag("viser-port"))],
			...["--interval", String(pi.getFlag("viser-interval"))],
			...["--transport", "http", "--host", "127.0.0.1", "--port", "0", "--parent-watch"],
		];
		const proc = spawn(python, args, {
			cwd: services,
			env: {
				...process.env,
				PYTHONPATH: [services, process.env.PYTHONPATH].filter(Boolean).join(":"),
				// The env server's RPC token (services/PROTOCOL.md), in the environment: argv is world-readable.
				...(env.token ? { PI_EMBODIED_ENV_TOKEN: env.token } : {}),
			},
			stdio: ["pipe", "pipe", "pipe"],
		});
		const log = join(tmpdir(), `pi-embodied-viser-${process.pid}-${Date.now()}.log`);
		const fd = openSync(log, "a");
		proc.once("close", () => closeSync(fd));
		let text = "";
		const ports = new Promise<{ viser: number; rpc: number }>((resolve, reject) => {
			const sink = (chunk: Buffer) => {
				writeSync(fd, chunk);
				text += chunk.toString();
				const p = parsePorts(text);
				if (p.viser !== undefined && p.rpc !== undefined) resolve({ viser: p.viser, rpc: p.rpc });
			};
			proc.stdout?.on("data", sink);
			proc.stderr?.on("data", sink);
			proc.once("exit", (code) => reject(new Error(`viser view exited (${code}); see ${log}`)));
			proc.once("error", (err) => reject(new Error(`viser view failed to start: ${err.message}`)));
			setTimeout(() => reject(new Error(`viser view bound no port in 120 s; see ${log}`)), 120_000).unref();
		});
		try {
			const p = await ports;
			return { proc, rpc: new RpcClient(`http://127.0.0.1:${p.rpc}`), port: p.viser };
		} catch (err) {
			proc.kill("SIGKILL");
			throw err;
		}
	}

	pi.on("session_shutdown", stop);

	return {
		/** The page's port while the view runs. */
		get port() {
			return view?.port;
		},
		/** Start the view against this episode's env server (after the robot started). */
		async start(ctx: ExtensionContext, services: string) {
			await stop();
			const env = on() ? spec.env() : undefined;
			if (!env) return;
			try {
				view = await launch(env, services);
				const where = `viser 3D view: http://${String(pi.getFlag("viser-host"))}:${view.port}/`;
				if (ctx.hasUI) ctx.ui.notify(where, "info");
				else console.error(`[viser] ${where}`);
			} catch (err) {
				const message = `[viser] not started: ${err instanceof Error ? err.message : err}`;
				if (ctx.hasUI) ctx.ui.notify(message, "warning");
				else console.error(message);
				return;
			}
			if (hooked) return;
			hooked = true;
			pi.on("tool_result", async (event) => {
				if (!view || event.isError || !PLANS.includes(event.toolName)) return undefined;
				const d = event.details as { candidates?: unknown; active?: unknown } | undefined;
				if (!Array.isArray(d?.candidates)) return undefined;
				// Drawing is a side view: a failure never touches the result.
				await view.rpc
					.call("viser.grasps", { candidates: d.candidates, active: d.active ?? null }, 10_000)
					.catch(() => {});
				return undefined;
			});
		},
	};
}
