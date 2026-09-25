#!/usr/bin/env bash
# Run RoboCasa Target50 episodes with pi in print mode and report RoboCasa-judged success
# (`state.success`). The matrix (splits, tasks, seeds, per-split cell timeouts) comes from
# services/pi_embodied_services/robots/robocasa/eval/target50.json (TARGET50 overrides it), and
# every episode runs under the frozen RLDX settings below.
#   eval.sh <out-dir> <splits|all> [pi args...]
#   eval.sh runs/t50 all --model openai/gpt-5.5 --thinking xhigh --memory-profile local --memory-dir target50-memory/robocasa
# TASKS=OpenDrawer,CloseFridge and SEEDS=1,2 narrow the matrix.
#
# Each episode runs in <out>/<split>/<Task>_s<seed>/ and ends with a result.json taken from the
# session's `robot_result` entry. An episode is valid when the environment produced a result and
# the planner did not fail (`env_error`, `planner_error` and a missing result are invalid),
# whatever the outcome. Rerunning retries exactly the invalid episodes; valid ones are kept.
# Each result records the model, thinking level and --max-turns, and the summary covers only the requested
# cells and refuses to mix configurations. Rates are per split and task-weighted (the mean of
# per-task rates, the RoboCasa365 convention).
set -uo pipefail
out=$1 splits=$2
shift 2
here=$(cd "$(dirname "$0")" && pwd)
SERVICES=${PI_EMBODIED_SERVICES:-$(cd "$here/../../../../services" && pwd)}
manifest=${TARGET50:-$SERVICES/pi_embodied_services/robots/robocasa/eval/target50.json}
PI=${PI:-pi}
MAX_TURNS=${MAX_TURNS:-100}
[ "$splits" = all ] && splits=atomic,composite_seen,composite_unseen
model="" thinking="" turns=$MAX_TURNS
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
	case ${args[i]} in
	--model) model=${args[i + 1]:-} ;;
	--thinking) thinking=${args[i + 1]:-} ;;
	--model=*) model=${args[i]#*=} ;;
	--thinking=*) thinking=${args[i]#*=} ;;
	esac
done
export RLDX_MAX_CHUNKS=40 RLDX_SETTLE_PATIENCE=999 RLDX_ACTION_STEPS_PER_CHUNK=8
unset RLDX_RESET_SEED

record() { # <dir> <exit code>: write result.json from the episode's session
	node --input-type=module -e '
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
const [dir, code, model, thinking, turns] = process.argv.slice(1);
const results = [];
for (const f of readdirSync(dir).filter((f) => f.endsWith(".jsonl"))) {
	for (const line of readFileSync(`${dir}/${f}`, "utf8").split("\n")) {
		if (!line.includes("\"robot_result\"")) continue;
		const e = JSON.parse(line);
		if (e.type === "custom" && e.customType === "robot_result") results.push(e.data);
	}
}
const last = results.length === 1 ? results[0] : undefined;
const status = Number(code) === 124 ? "timeout" : results.length > 1 ? "duplicate_result"
	: !last ? (Number(code) ? "env_error" : "missing")
	: last.env_error ? "env_error" : last.planner_error ? "planner_error" : last.success ? "success" : "failure";
const result = { ...(last ?? {}), status, exit_code: Number(code), model: model || null, thinking: thinking || null, max_turns: Number(turns) };
writeFileSync(`${dir}/result.json`, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ status, success: result.success, claimed: result.claimed, env_steps: result.env_steps }));
' "$1" "$2" "$model" "$thinking" "$turns"
}

valid() { # <dir>: 0 = a valid result of this configuration, 2 = a valid result of another one, 1 = none
	node -e '
const [path, model, thinking, turns] = process.argv.slice(1);
const r = JSON.parse(require("fs").readFileSync(path, "utf8"));
if (r.status !== "success" && r.status !== "failure") process.exit(1);
process.exit(r.model === (model || null) && r.thinking === (thinking || null) && r.max_turns === Number(turns) ? 0 : 2);
' "$1/result.json" "$model" "$thinking" "$turns" 2>/dev/null
}

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
	valid "$dir"
	case $? in
	0) continue ;;
	2) echo "$dir holds a result of another model, thinking level or --max-turns; use another out dir" >&2 && exit 1 ;;
	esac
	rm -rf "$dir" && mkdir -p "$dir"
	echo "== $split $task seed $seed"
	# --time-limit ends the planner gracefully; `timeout` is only the backstop for a hung process.
	timeout -k 30 $((limit + 900)) $PI -p --session-dir "$dir" -e "$here" --task-name "$task" --split target \
		--seed "$seed" --max-turns "$MAX_TURNS" --time-limit "$limit" --log-dir "$dir" "$@" "Solve the task." \
		</dev/null >"$dir/stdout.log" 2>"$dir/stderr.log"
	record "$dir" "$?"
done <<<"$cells"

node -e '
const fs = require("fs");
const [out, cells] = process.argv.slice(1);
const rows = cells.trim().split("\n").map((line) => {
	const [split, task, seed] = line.split(" ");
	try {
		return { split, task, ...JSON.parse(fs.readFileSync(`${out}/${split}/${task}_s${seed}/result.json`, "utf8")) };
	} catch {
		return { split, task, status: "missing" };
	}
});
const scoredRows = rows.filter((r) => r.status === "success" || r.status === "failure");
const configs = new Set(scoredRows.map((r) => `${r.model}/${r.thinking}/turns=${r.max_turns}`));
if (configs.size > 1) {
	console.log(`refusing to summarize: ${out} mixes configurations ${[...configs].join(", ")}`);
	process.exit(1);
}
const n = (s) => rows.filter((r) => r.status === s).length;
const rate = (ok, all) => (all ? ((100 * ok) / all).toFixed(1) : "-");
const scored = scoredRows.length;
const lies = rows.filter((r) => r.status === "failure" && r.claimed === "success").length;
const invalid = rows.length - scored;
const perTask = new Map();
for (const r of scoredRows) {
	const k = `${r.split} ${r.task}`;
	const [ok, all] = perTask.get(k) ?? [0, 0];
	perTask.set(k, [ok + (r.status === "success"), all + 1]);
}
const weighted = perTask.size ? [...perTask.values()].reduce((a, [ok, all]) => a + ok / all, 0) / perTask.size : 0;
for (const split of new Set(rows.map((r) => r.split))) {
	const s = scoredRows.filter((r) => r.split === split);
	const ok = s.filter((r) => r.status === "success").length;
	console.log(`${split}: success ${ok}/${s.length} (${rate(ok, s.length)}%)`);
}
console.log(`${[...configs][0] ?? "-"}: success ${n("success")}/${scored} (${rate(n("success"), scored)}%), task-weighted ${perTask.size ? (100 * weighted).toFixed(1) : "-"}%, claimed-but-failed ${lies}, invalid ${invalid} (env_error ${n("env_error")}, planner_error ${n("planner_error")}, timeout ${n("timeout")}, missing ${n("missing")}, duplicate ${n("duplicate_result")}) of ${rows.length}`);
if (invalid) process.exit(1);
' "$out" "$cells"
