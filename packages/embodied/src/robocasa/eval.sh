#!/usr/bin/env bash
# RoboCasa Target50 (robocasa-harness-vla-v1, RPent's robots/robocasa/eval/target50.json) with pi:
# one `pi -p` per manifest cell under the frozen RLDX settings and the split's cell timeout,
# RPent's result.json at <out>/<split>/<Task>_s<seed>/, then the task-weighted score.
#   eval.sh <out-dir> <splits|all> [pi args...]
#   eval.sh runs/t50 all --model openai/gpt-5.5 --thinking xhigh --memory-profile local --memory-dir target50-memory/robocasa
# TASKS=OpenDrawer,CloseFridge and SEEDS=1,2 narrow the matrix. A cell is valid when the environment
# produced exactly one robot_result and the planner did not fail, whatever the outcome; re-running
# retries exactly the invalid cells (startup, planner and infrastructure failures), so task failures
# and timeouts are as final as successes. Rates are over valid cells; the others are reported and
# make the script exit 1, as does a directory whose cells mix planner configurations.
# Records follow RPent's robots/robocasa/eval/result.py, except planner.backend is "pi" (RPent's
# validate_target50.py accepts only its Codex reference planner) and an `error` field is kept.
set -uo pipefail
out=$1 splits=$2
shift 2
: "${RPENT_ROOT:?RPent checkout}"
here=$(cd "$(dirname "$0")" && pwd)
PI=${PI:-pi}
MAX_TURNS=${MAX_TURNS:-100}
manifest=${TARGET50:-$RPENT_ROOT/robots/robocasa/eval/target50.json}
[ "$splits" = all ] && splits=atomic,composite_seen,composite_unseen
model="" effort=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
	case ${args[i]} in
	--model) model=${args[i + 1]:-} ;;
	--thinking) effort=${args[i + 1]:-} ;;
	esac
done
export RLDX_MAX_CHUNKS=40 RLDX_SETTLE_PATIENCE=999 RLDX_ACTION_STEPS_PER_CHUNK=8
unset RLDX_RESET_SEED

cells=$(node -e '
const m = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
const only = (s) => (s ? s.split(",") : null);
const [tasks, seeds] = [only(process.argv[3]), only(process.argv[4])];
for (const name of process.argv[2].split(",")) {
	const split = m.splits[name];
	if (!split) throw new Error(`unknown Target50 split ${name}`);
	for (const t of split.tasks)
		for (const s of split.seeds)
			if ((!tasks || tasks.includes(t)) && (!seeds || seeds.includes(String(s))))
				console.log(`${name} ${t} ${s} ${split.timeout_seconds}`);
}' "$manifest" "$splits" "${TASKS:-}" "${SEEDS:-}") || exit 1

while read -r split task seed limit; do
	dir=$out/$split/${task}_s$seed
	node -e '
const [path, model, effort, turns] = process.argv.slice(1);
const r = JSON.parse(require("fs").readFileSync(path, "utf8"));
if (!r.valid) process.exit(1);
const p = r.planner ?? {};
process.exit(p.backend === "pi" && p.model === (model || null) && p.reasoning_effort === (effort || null) && p.max_turns === Number(turns) ? 0 : 2);
' "$dir/result.json" "$model" "$effort" "$MAX_TURNS" 2>/dev/null
	case $? in
	0) continue ;;
	2) echo "$dir holds a result of another planner configuration; use another out dir" >&2 && exit 1 ;;
	esac
	rm -rf "$dir" && mkdir -p "$dir"
	echo "== $split $task seed $seed"
	start=$(node -p 'Date.now()')
	# --time-limit ends the planner gracefully; `timeout` is only the backstop for a hung process.
	timeout -k 30 $((limit + 900)) $PI -p --session-dir "$dir" -e "$here" --task-name "$task" --split target \
		--seed "$seed" --max-turns "$MAX_TURNS" --time-limit "$limit" --log-dir "$dir" "$@" "Solve the task." \
		</dev/null >/dev/null 2>"$dir/stderr.log"
	code=$?
	node -e '
const fs = require("fs");
const [dir, manifest, split, task, seed, limit, code, start, model, effort, turns] = process.argv.slice(1);
const lines = fs.readFileSync(`${dir}/stderr.log`, "utf8").split("\n");
const unavailable = lines.find((l) => l.startsWith("[robocasa] unavailable: "));
// pi writes exactly one robot_result per episode (and one with env_error when the start failed).
const results = lines.filter((l) => l.startsWith("[robocasa] {")).map((l) => JSON.parse(l.slice(11)));
const r = results.length === 1 ? results[0] : null;
// Validity depends only on whether the environment produced a result and the planner did not fail.
const valid = !unavailable && r !== null && !r.env_error && !r.planner_error && typeof r.success === "boolean";
const timedOut = code === "124" || code === "137" || r?.planner_budget_exhausted === "time";
const error = valid ? null
	: unavailable ? `startup failure: ${unavailable.slice(24)}`
	: results.length > 1 ? `${results.length} robot_result lines, expected 1`
	: !r ? `no robot_result (pi exit ${code})`
	: r.env_error ? `env_error: ${r.error}` : `planner_error: ${r.planner_error}`;
const env = (name, fallback) => (process.env[name] ? Number.parseInt(process.env[name], 10) : fallback);
const record = {
	schema_version: "1.0",
	protocol_id: JSON.parse(fs.readFileSync(manifest, "utf8")).protocol_id,
	evaluation_split: split,
	task_name: task,
	environment_split: "target",
	seed: Number(seed),
	valid,
	success: valid && r.success,
	success_source: "state.success",
	termination_reason: !valid ? "infrastructure_error" : timedOut ? "planner_timeout" : "completed",
	elapsed_s: Math.round((Date.now() - Number(start)) / 100) / 10,
	planner: { backend: "pi", model: model || null, reasoning_effort: effort || null, max_turns: Number(turns) },
	runtime: {
		cell_timeout_seconds: Number(limit),
		rldx_max_chunks: env("RLDX_MAX_CHUNKS", 70),
		rldx_settle_patience: env("RLDX_SETTLE_PATIENCE", 999),
		rldx_action_steps_per_chunk: env("RLDX_ACTION_STEPS_PER_CHUNK", 8),
	},
	error,
};
fs.writeFileSync(`${dir}/.result.json.tmp`, `${JSON.stringify(record, null, 2)}\n`);
fs.renameSync(`${dir}/.result.json.tmp`, `${dir}/result.json`);
console.log(JSON.stringify({ success: record.success, reason: record.termination_reason, valid, exit: Number(code), error }));
' "$dir" "$manifest" "$split" "$task" "$seed" "$limit" "$code" "$start" "$model" "$effort" "$MAX_TURNS"
done <<<"$cells"

# Split rates over the requested cells; overall = mean of per-task rates (RoboCasa365 convention).
node -e '
const fs = require("fs");
const [manifest, out, cells] = process.argv.slice(1);
const m = JSON.parse(fs.readFileSync(manifest, "utf8"));
const bySplit = new Map();
for (const line of cells.trim().split("\n")) {
	const [name, t, s] = line.split(" ");
	if (!bySplit.has(name)) bySplit.set(name, new Map());
	const tasks = bySplit.get(name);
	tasks.set(t, [...(tasks.get(t) ?? []), s]);
}
const rates = [];
const summary = { protocol_id: m.protocol_id, planner: null, splits: {}, startup_failures: 0, errors: 0 };
const invalid = [];
const planners = new Set();
let requested = 0;
for (const [name, tasks] of bySplit) {
	let ok = 0;
	let valid = 0;
	for (const [t, seeds] of tasks) {
		let taskOk = 0;
		let taskValid = 0;
		for (const s of seeds) {
			requested++;
			let r = null;
			try { r = JSON.parse(fs.readFileSync(`${out}/${name}/${t}_s${s}/result.json`, "utf8")); } catch {}
			if (!r?.valid) {
				if (r?.error?.startsWith("startup failure")) summary.startup_failures++;
				else summary.errors++;
				invalid.push(`${r?.termination_reason ?? "missing"}: ${name} ${t} s${s}: ${r?.error ?? "no result.json"}`);
				continue;
			}
			planners.add(JSON.stringify(r.planner));
			valid++;
			taskValid++;
			if (r.success) { ok++; taskOk++; }
		}
		if (taskValid) rates.push(taskOk / taskValid);
	}
	summary.splits[name] = { successes: ok, valid_cells: valid, expected_cells: m.splits[name].cell_count, success_rate: valid ? +(ok / valid).toFixed(6) : null };
}
if (planners.size > 1) {
	console.log(`refusing to summarize: ${out} mixes planner configurations ${[...planners].join(", ")}`);
	process.exit(1);
}
summary.planner = planners.size ? JSON.parse([...planners][0]) : null;
summary.task_weighted_success_rate = rates.length ? +(rates.reduce((a, b) => a + b, 0) / rates.length).toFixed(6) : null;
console.log(JSON.stringify(summary, null, 2));
for (const line of invalid) console.log(line);
if (invalid.length) {
	console.log(`incomplete: ${invalid.length} of ${requested} requested cells have no valid result (excluded from the rates above); re-run to retry them`);
	process.exit(1);
}
' "$manifest" "$out" "$cells"
