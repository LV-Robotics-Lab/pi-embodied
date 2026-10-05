/**
 * The PreToolUse hook behind the Codex and Claude Code plugins (integrations/<host>/hooks): the host's
 * operator gate for the motion tools the MCP server (./server.ts) exposes, a second layer over the
 * server's own gate on real robots (./gate.ts, which does not depend on the host; Codex 0.160 runs no
 * plugin hooks at all, so there this hook protects nothing), with the risk classes of
 * `--approval standard` (../../capabilities/operator.ts `highRisk`: grasp/place execution, resets,
 * moves to an absolute target, relative moves over `--large-move` m, and every motion on a real
 * robot). The hook reads the host's JSON on stdin and prints a `hookSpecificOutput` decision:
 *
 *   --decision ask    Claude Code: the operator is prompted (`permissionDecision: "ask"`)
 *   --decision deny   Codex, which has no "ask": the call is refused unless the operator confirmed
 *                     beforehand by exporting `--confirmed-env` (default PI_EMBODIED_MOTION_CONFIRMED)
 *
 * A tool that is not one of our server's, or does not move the robot, gets no decision (the host's
 * normal flow). The manifest decides what moves (`mutating`), the same declaration the server
 * exposes, plus the server's built-in motions (./tools.ts BUILTIN_MOTIONS: `reset`, a reset by
 * `--approval standard`'s classes); when the robot is unknown (no --robot, no PI_EMBODIED_ROBOT / CLAUDE_PLUGIN_OPTION_ROBOT)
 * or its manifest cannot be read, every tool of our server gets the decision: fail closed.
 *
 *   node --experimental-strip-types src/integrations/mcp/hook.ts [--robot <name>] [--server <mcp server name>]
 *        [--decision ask|deny] [--large-move <m>] [--confirmed-env <VAR>] < hook-input.json
 */

import { pathToFileURL } from "node:url";
import { highRisk } from "../../capabilities/operator.ts";
import { loadManifest, type ManifestEntry } from "../../primitives/manifest.ts";
import { BUILTIN_MOTIONS, REAL_ROBOTS } from "./tools.ts";

export type HookArgs = {
	robot: string;
	server: string;
	decision: "ask" | "deny";
	largeMove: number;
	confirmedEnv: string;
};

export const DEFAULT_SERVER = "pi-embodied";

export function parseHookArgs(argv: readonly string[], env: NodeJS.ProcessEnv = process.env): HookArgs {
	const a: HookArgs = {
		robot: env.CLAUDE_PLUGIN_OPTION_ROBOT || env.PI_EMBODIED_ROBOT || "",
		server: DEFAULT_SERVER,
		decision: "ask",
		largeMove: 0.1,
		confirmedEnv: "PI_EMBODIED_MOTION_CONFIRMED",
	};
	for (let i = 0; i < argv.length; i++) {
		const v = argv[i + 1];
		switch (argv[i]) {
			case "--robot":
				a.robot = v ?? "";
				i++;
				break;
			case "--server":
				a.server = v ?? a.server;
				i++;
				break;
			case "--decision":
				if (v !== "ask" && v !== "deny") throw new Error("--decision ask|deny");
				a.decision = v;
				i++;
				break;
			case "--large-move":
				a.largeMove = Number(v);
				if (!Number.isFinite(a.largeMove) || a.largeMove < 0) throw new Error("--large-move <m>");
				i++;
				break;
			case "--confirmed-env":
				a.confirmedEnv = v ?? a.confirmedEnv;
				i++;
				break;
			default:
				throw new Error(`unknown argument ${argv[i]}`);
		}
	}
	return a;
}

export type HookInput = { tool_name?: string; tool_input?: unknown; permission_mode?: string };
export type HookOutput = {
	hookSpecificOutput: {
		hookEventName: "PreToolUse";
		permissionDecision: "ask" | "deny";
		permissionDecisionReason: string;
	};
};

/** The tool behind `mcp__<server>__<tool>` when it is our server's, else undefined. */
export function ourTool(toolName: string, server: string): string | undefined {
	const prefix = `mcp__${server}__`;
	return toolName.startsWith(prefix) ? toolName.slice(prefix.length) : undefined;
}

/** The decision for one hook input, or undefined for no decision (the host's normal flow). */
export function decide(input: HookInput, a: HookArgs, env: NodeJS.ProcessEnv = process.env): HookOutput | undefined {
	const tool = typeof input.tool_name === "string" ? ourTool(input.tool_name, a.server) : undefined;
	if (tool === undefined) return undefined;
	const out = (reason: string): HookOutput => ({
		hookSpecificOutput: {
			hookEventName: "PreToolUse",
			permissionDecision: a.decision,
			permissionDecisionReason: reason,
		},
	});
	if (!a.robot)
		return out(
			`${tool}: the hook does not know the robot (set PI_EMBODIED_ROBOT or the plugin's robot option), so every ${a.server} call needs the operator`,
		);
	let entry: ManifestEntry | undefined;
	try {
		entry = loadManifest(a.robot).primitives.find((e) => e.name === tool && e.side === "env" && e.doc.tool);
	} catch (err) {
		return out(
			`${tool}: cannot read ${a.robot}'s manifest (${err instanceof Error ? err.message : err}); the operator must decide`,
		);
	}
	if (!entry?.mutating && !BUILTIN_MOTIONS.includes(tool)) return undefined;
	const args =
		input.tool_input && typeof input.tool_input === "object" ? (input.tool_input as Record<string, unknown>) : {};
	const real = REAL_ROBOTS.includes(a.robot);
	if (!highRisk(tool, args, real, a.largeMove)) return undefined;
	if (env[a.confirmedEnv]?.trim()) return undefined;
	const why = real
		? `${tool} moves a real robot (${a.robot}): the operator must confirm the scene is clear`
		: `${tool} is a high-risk motion (grasp/place execution, a reset, a move to an absolute target, or more than ${a.largeMove} m): the operator must confirm`;
	return out(
		a.decision === "deny"
			? `${why}; export ${a.confirmedEnv}=1 after confirming (on a real robot the MCP server's own gate asks the same, or a ticket in its --confirm-file), or approve the tool in Codex's MCP approval settings`
			: why,
	);
}

export async function main(argv = process.argv.slice(2)): Promise<void> {
	const a = parseHookArgs(argv);
	const chunks: Buffer[] = [];
	for await (const c of process.stdin) chunks.push(c as Buffer);
	let input: HookInput = {};
	try {
		input = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}") as HookInput;
	} catch {
		// Unreadable input: no tool name to judge, which is a decision for every call of our server.
		input = { tool_name: `mcp__${a.server}__unknown` };
	}
	const out = decide(input, a);
	if (out) process.stdout.write(`${JSON.stringify(out)}\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
	main().catch((err) => {
		console.error(`[pi-embodied-hook] ${err instanceof Error ? err.message : err}`);
		// Exit 2 blocks the call: the gate fails closed when it cannot run.
		process.exit(2);
	});
