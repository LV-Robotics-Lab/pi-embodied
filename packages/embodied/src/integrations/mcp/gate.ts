/**
 * The operator gate the MCP server itself enforces on a real robot (./tools.ts `isReal`), whatever
 * the host: Codex 0.160 runs no plugin hooks (it lists `plugin_hooks` as a removed feature), so the
 * PreToolUse hook (./hook.ts) protects nothing there, and `codex exec --approve-for-me` lets the model
 * answer Codex's own tool approval itself. The hook stays a second layer for Claude Code; this is the
 * layer that does not depend on the host. On a real robot every motion tool and `reset` is refused
 * unless the operator authorised it through a channel the model cannot reach:
 *
 *   PI_EMBODIED_MOTION_CONFIRMED=1   in the server's environment at launch: the WHOLE SESSION is
 *                                    authorised (the model cannot change a running server's
 *                                    environment). For a bench with nobody in reach of the arm.
 *   --confirm-file <path>            a per-call ticket (recommended): the operator writes the tool's
 *                                    name into the file; the next call of that tool consumes it (the
 *                                    file is removed) and no other call does. A ticket for another
 *                                    tool is left in place and the call refused; an empty file, an
 *                                    unreadable one or a ticket older than TICKET_TTL_MS refuses too
 *                                    (a stale ticket is removed). The path must be one the model's
 *                                    sandbox cannot write: outside the workspace and the temp dir
 *                                    (Codex's workspace-write sandbox allows both of those).
 *
 * A ticket that cannot be removed does not authorise anything (it would authorise every call): fail
 * closed. Simulators are not gated here (the hosts' own approval still applies), and nothing here
 * replaces the env server's limits or the hardware E-stop.
 */

import { readFileSync, statSync, unlinkSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { resolve, sep } from "node:path";
import { message } from "../../robot.ts";

/** The environment variable whose non-empty value at launch authorises the whole session. */
export const CONFIRMED_ENV = "PI_EMBODIED_MOTION_CONFIRMED";
/** A ticket older than this is stale: the operator wrote it for a call that did not come. */
export const TICKET_TTL_MS = 10 * 60_000;

export type GateOptions = {
	robot: string;
	/** `isReal(robot)`: only a real robot is gated. */
	real: boolean;
	/** `PI_EMBODIED_MOTION_CONFIRMED` was set when the server started. */
	sessionConfirmed: boolean;
	/** `--confirm-file`: where the operator writes per-call tickets. */
	confirmFile?: string;
	now?: () => number;
};

/** What the confirm file holds right now (not consumed). */
export type Ticket = { tool: string; ageMs: number } | { error: string } | undefined;

/** `~` or `~/x` as the operator's home; anything else unchanged. */
export function expandHome(path: string): string {
	return path === "~" ? homedir() : path.startsWith("~/") ? homedir() + path.slice(1) : path;
}

export class OperatorGate {
	readonly real: boolean;
	readonly sessionConfirmed: boolean;
	readonly confirmFile: string | undefined;
	private readonly robot: string;
	private readonly now: () => number;

	constructor(o: GateOptions) {
		this.robot = o.robot;
		this.real = o.real;
		this.sessionConfirmed = o.sessionConfirmed;
		// A shell does not expand a quoted `~/…` (the documented --config confirm_file=~/… form): expand it here.
		this.confirmFile = o.confirmFile === undefined ? undefined : resolve(expandHome(o.confirmFile));
		this.now = o.now ?? Date.now;
	}

	/** Whether motions on this robot need the operator's authorisation per call. */
	get gated(): boolean {
		return this.real && !this.sessionConfirmed;
	}

	/** One line for the server's log at start. */
	describe(): string {
		if (!this.real)
			return `${this.robot} is a simulator: motions are not gated by the server (the host's approval applies)`;
		if (this.sessionConfirmed)
			return `${this.robot} is a real robot and ${CONFIRMED_ENV} is set: every motion of this session is authorised by the operator`;
		if (this.confirmFile)
			return `${this.robot} is a real robot: a motion runs only with the operator's ticket (its tool name written into ${this.confirmFile}; one call per ticket)`;
		return `${this.robot} is a real robot and no authorisation channel was given: every motion tool and reset will be refused (start with ${CONFIRMED_ENV}=1 or --confirm-file <path>)`;
	}

	/** A warning when the confirm file lies where the model's sandbox may write, else undefined. */
	warning(): string | undefined {
		if (!this.confirmFile || !this.real) return undefined;
		const tmp = resolve(tmpdir());
		const cwd = resolve(process.cwd());
		for (const [where, dir] of [
			["the temp dir", tmp],
			["the working directory", cwd],
		] as const)
			if (this.confirmFile === dir || this.confirmFile.startsWith(dir + sep))
				return `--confirm-file ${this.confirmFile} is under ${where}, which a sandboxed model may write: the ticket would not be the operator's; put it where only the operator can write`;
		return undefined;
	}

	/** The ticket on file, read without consuming it. */
	peek(): Ticket {
		if (!this.confirmFile) return undefined;
		try {
			const st = statSync(this.confirmFile, { throwIfNoEntry: false });
			if (!st) return undefined;
			const tool = readFileSync(this.confirmFile, "utf8").split(/\r?\n/)[0].trim();
			return { tool, ageMs: Math.max(0, this.now() - st.mtimeMs) };
		} catch (err) {
			return { error: message(err) };
		}
	}

	/**
	 * Why `tool` may not move the robot now, else undefined. On a gated robot a ticket naming `tool`
	 * is consumed (the file removed) and the call admitted; everything else refuses and names both
	 * channels, so the model can tell the operator what to do and nothing else.
	 */
	authorise(tool: string): string | undefined {
		if (!this.gated) return undefined;
		const lead = `${tool} moves a real robot (${this.robot}) and the operator has not authorised it.`;
		const channels = this.confirmFile
			? `The operator writes "${tool}" into ${this.confirmFile} to authorise one call (the ticket is consumed), or starts the server with ${CONFIRMED_ENV}=1 to authorise the whole session.`
			: `Authorisation comes from outside the model: the server is started with ${CONFIRMED_ENV}=1 in its environment (the whole session) or with --confirm-file <path> (PI_EMBODIED_CONFIRM_FILE for the plugins), where the operator writes "${tool}" before each call.`;
		if (!this.confirmFile) return `${lead} ${channels}`;
		const ticket = this.peek();
		if (ticket === undefined) return `${lead} ${this.confirmFile} holds no ticket. ${channels}`;
		if ("error" in ticket) return `${lead} ${this.confirmFile} cannot be read (${ticket.error}). ${channels}`;
		if (!ticket.tool) return `${lead} ${this.confirmFile} is empty: a ticket is the tool's name. ${channels}`;
		if (ticket.ageMs > TICKET_TTL_MS) {
			const removed = this.consume();
			return `${lead} The ticket for "${ticket.tool}" in ${this.confirmFile} is ${Math.round(ticket.ageMs / 1000)} s old (over ${TICKET_TTL_MS / 1000} s) and was ${removed ? "discarded" : "left, as it could not be removed"}: the operator writes it again right before the call. ${channels}`;
		}
		if (ticket.tool !== tool)
			return `${lead} ${this.confirmFile} holds a ticket for "${ticket.tool}", not ${tool}; it is left in place. ${channels}`;
		if (!this.consume())
			return `${lead} The ticket for "${tool}" in ${this.confirmFile} cannot be removed, and a ticket that cannot be consumed would authorise every call: refusing. Make the file's directory writable by the server.`;
		return undefined;
	}

	/** Remove the ticket file; false when it could not be removed (then nothing was authorised). */
	private consume(): boolean {
		try {
			unlinkSync(this.confirmFile as string);
			return true;
		} catch {
			return false;
		}
	}

	/** For `robot_status`. */
	status() {
		const ticket = this.peek();
		return {
			real: this.real,
			gated: this.gated,
			session_confirmed: this.sessionConfirmed,
			confirm_file: this.confirmFile ?? null,
			ticket:
				ticket === undefined
					? null
					: "error" in ticket
						? ticket
						: { tool: ticket.tool, age_s: Math.round(ticket.ageMs / 1000) },
		};
	}
}
