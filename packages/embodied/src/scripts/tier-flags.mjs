#!/usr/bin/env node
/**
 * tier-flags.mjs <pi args...>: the flags `--tier` / `--preset` among the arguments expand to, one
 * `--name=value` per line (a flag the choice keeps at its default, or off, prints nothing), for
 * eval-options.sh to parse as if they had been given. Exit 2 with the robot's own message when the
 * arguments contradict the choice (src/infra/tiers.ts `conflicts`), so an eval run stops before its
 * first cell, as the robot would at start. Nothing is printed without a choice.
 */
import { choose, conflicts, norm } from "../infra/tiers.ts";

const args = process.argv.slice(2).map(String);
/** Every `--name[=value]` given, with its value (`--name word` for a string flag, true for a bare switch). */
const given = new Map();
for (let i = 0; i < args.length; i++) {
	const a = args[i];
	if (!a.startsWith("--")) continue;
	const eq = a.indexOf("=");
	const name = eq > 0 ? a.slice(2, eq) : a.slice(2);
	if (eq > 0) given.set(name, a.slice(eq + 1));
	else if (i + 1 < args.length && !args[i + 1].startsWith("-")) given.set(name, args[++i]);
	else given.set(name, true);
}
const c = choose(given.get("tier"), given.get("preset"));
if (!c) process.exit(0);
const bad = conflicts(c, given);
if (bad.length) {
	console.error(bad.join("; "));
	process.exit(2);
}
// `--privileged true` would make eval-options.sh refuse the run; a switch's own value is its name.
for (const [name, want] of Object.entries(c.flags)) {
	if (want === null || want === false) continue;
	console.log(`--${name}=${norm(want)}`);
}
