#!/usr/bin/env bash
# Run LIBERO episodes with pi in print mode and report LIBERO-judged success.
#   eval.sh <out-dir> <suite> <tasks> <seeds> [pi args...]
#   eval.sh runs/l10 libero_10_task 0-9 1-10 --model <provider/model> --thinking low
#
# Each episode runs in its own session directory and ends with a result.json taken from the
# session's `robot_result` entry. An episode is valid when the environment produced a result and
# the planner did not fail (`env_error`, `planner_error` and a missing result are invalid),
# whatever the outcome. Rerunning retries exactly the invalid episodes; valid ones are kept.
# Each result records the model, thinking level, --max-turns and --time-limit, and the summary covers only the requested
# cells and refuses to mix configurations.
set -uo pipefail
out=$1 suite=$2 tasks=$3 seeds=$4
shift 4
here=$(cd "$(dirname "$0")" && pwd)
PI=${PI:-pi}
expand() { for part in ${1//,/ }; do seq "${part%-*}" "${part#*-}"; done; }
model="" thinking="" turns=0 limit=${TIME_LIMIT:-1800} limited=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
	case ${args[i]} in
	--model) model=${args[i + 1]:-} ;;
	--thinking) thinking=${args[i + 1]:-} ;;
	--model=*) model=${args[i]#*=} ;;
	--thinking=*) thinking=${args[i]#*=} ;;
	--max-turns) turns=${args[i + 1]:-0} ;;
	--max-turns=*) turns=${args[i]#*=} ;;
	--time-limit) limit=${args[i + 1]:-0} limited=1 ;;
	--time-limit=*) limit=${args[i]#*=} limited=1 ;;
	esac
done
# --time-limit (default $TIME_LIMIT, 1800 s; 0 = none) ends the planner gracefully, as a failure;
# `timeout` is only the backstop for a hung process, and a killed episode is invalid.
[ -n "$limited" ] || set -- "$@" --time-limit "$limit"
backstop=()
[ "$limit" -gt 0 ] && command -v timeout >/dev/null && backstop=(timeout -k 30 $((limit + 900)))
mkdir -p "$out"

record() { # <dir> <exit code>: write result.json from the episode's session
	node --input-type=module -e '
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
const [dir, code, model, thinking, turns, limit] = process.argv.slice(1);
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
	: last.env_error ? "env_error" : last.planner_error ? "planner_error" : last.terminated ? "success" : "failure";
const result = { ...(last ?? {}), status, exit_code: Number(code), model: model || null, thinking: thinking || null, max_turns: Number(turns), time_limit: Number(limit) };
writeFileSync(`${dir}/result.json`, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ status, terminated: result.terminated, claimed: result.claimed, env_steps: result.env_steps }));
' "$1" "$2" "$model" "$thinking" "$turns" "$limit"
}

valid() { # <dir>: 0 = a valid result of this configuration, 2 = a valid result of another one, 1 = none
	node -e '
const [path, model, thinking, turns, limit] = process.argv.slice(1);
const r = JSON.parse(require("fs").readFileSync(path, "utf8"));
if (r.status !== "success" && r.status !== "failure") process.exit(1);
process.exit(r.model === (model || null) && r.thinking === (thinking || null) && r.max_turns === Number(turns) && r.time_limit === Number(limit) ? 0 : 2);
' "$1/result.json" "$model" "$thinking" "$turns" "$limit" 2>/dev/null
}

cells=()
for task in $(expand "$tasks"); do
	for seed in $(expand "$seeds"); do
		cells+=("${suite}_t${task}_s${seed}")
		dir="$out/${suite}_t${task}_s${seed}"
		valid "$dir"
		case $? in
		0) continue ;;
		2) echo "$dir holds a result of another model, thinking level, --max-turns or --time-limit; use another out dir" >&2 && exit 1 ;;
		esac
		rm -rf "$dir" && mkdir -p "$dir"
		echo "== $suite task $task seed $seed"
		${backstop[@]+"${backstop[@]}"} $PI -p --session-dir "$dir" -e "$here" --suite "$suite" --task "$task" --seed "$seed" "$@" \
			"Solve the task." </dev/null >"$dir/stdout.log" 2>"$dir/stderr.log"
		record "$dir" "$?"
	done
done

node --input-type=module -e '
import { readFileSync } from "node:fs";
const [out, ...cells] = process.argv.slice(1);
const rows = cells.map((c) => {
	try {
		return JSON.parse(readFileSync(`${out}/${c}/result.json`, "utf8"));
	} catch {
		return { status: "missing" };
	}
});
const configs = new Set(rows.filter((r) => r.status === "success" || r.status === "failure").map((r) => `${r.model}/${r.thinking}/turns=${r.max_turns}/limit=${r.time_limit}`));
if (configs.size > 1) {
	console.log(`refusing to summarize: ${out} mixes configurations ${[...configs].join(", ")}`);
	process.exit(1);
}
const n = (s) => rows.filter((r) => r.status === s).length;
const scored = n("success") + n("failure");
const lies = rows.filter((r) => r.status === "failure" && r.claimed === "success").length;
const rate = scored ? ((100 * n("success")) / scored).toFixed(1) : "-";
const invalid = rows.length - scored;
console.log(`${[...configs][0] ?? "-"}: success ${n("success")}/${scored} (${rate}%), claimed-but-failed ${lies}, invalid ${invalid} (env_error ${n("env_error")}, planner_error ${n("planner_error")}, timeout ${n("timeout")}, missing ${n("missing")}, duplicate ${n("duplicate_result")}) of ${rows.length}`);
if (invalid) process.exit(1);
' "$out" "${cells[@]}"
