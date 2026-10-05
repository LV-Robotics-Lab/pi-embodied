#!/usr/bin/env node
/**
 * tier-flags.mjs <pi args...>: the flags `--tier` / `--preset` among the arguments expand to, one
 * `--name=value` per line (a flag the choice keeps at its default, or off, prints nothing), for
 * eval-options.sh to parse as if they had been given. Exit 2 with the robot's own message when the
 * arguments contradict the choice (src/infra/tiers.ts `conflicts`), so an eval run stops before its
 * first cell, as the robot would at start. Nothing is printed without a choice.
 */
import { argvFlags } from "../infra/params.ts";
import { choose, conflicts, norm } from "../infra/tiers.ts";

// Every `--name[=value]` given, by pi's rules: the parse the robot's flag tracker makes of its own argv.
const given = argvFlags(process.argv.slice(2));
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
