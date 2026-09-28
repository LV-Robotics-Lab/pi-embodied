#!/usr/bin/env node
/**
 * params-match.mjs <result.json>: whether a recorded episode ran with the configuration the eval
 * script runs now (PARAMS.md 2.6). The robot records `params` (every experiment flag's effective
 * value) and `params_default` (its default) in its result (src/infra/params.ts); PI_ARGS_JSON is
 * the pi arguments the script passes (a JSON list). A flag given on the command line must have
 * run with that value, every other one with its default. Exit 0 = the same configuration (or a
 * result written before `params` existed: the script's own checks judge it), 2 = another one (the
 * differing flags on stderr). The cell's own flags (task, seed, ...) are the directory's, not compared.
 */
import { readFileSync } from "node:fs";

const CELL = new Set(["task", "seed", "suite", "task-name", "env-id", "task-config", "split", "scene", "eval-seed", "episode"]);

const r = JSON.parse(readFileSync(process.argv[2], "utf8"));
if (!r.params || !r.params_default) process.exit(0);
const args = JSON.parse(process.env.PI_ARGS_JSON || "[]");
const given = new Map();
for (let i = 0; i < args.length; i++) {
	const a = String(args[i]);
	if (!a.startsWith("--")) continue;
	const eq = a.indexOf("=");
	const name = eq > 0 ? a.slice(2, eq) : a.slice(2);
	if (!(name in r.params)) continue;
	const boolean = typeof r.params_default[name] === "boolean";
	if (eq > 0) given.set(name, a.slice(eq + 1));
	else if (boolean) given.set(name, true);
	else if (i + 1 < args.length) given.set(name, args[++i]);
}
const norm = (v) => (v === null || v === undefined ? "" : String(v));
const differ = [];
for (const [name, ran] of Object.entries(r.params)) {
	if (CELL.has(name)) continue;
	const want = given.has(name) ? given.get(name) : r.params_default[name];
	if (norm(ran) !== norm(want)) differ.push(`--${name} ran ${JSON.stringify(ran)}, now ${JSON.stringify(want)}`);
}
if (differ.length) {
	console.error(`${process.argv[2]}: another configuration: ${differ.join("; ")}`);
	process.exit(2);
}
