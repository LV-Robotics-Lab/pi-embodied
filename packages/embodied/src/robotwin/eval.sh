#!/usr/bin/env bash
# Run RoboTwin episodes with pi in print mode on the evaluation seeds in
# services/pi_embodied_services/robots/robotwin/eval/demo_randomized.json (50 tasks x 5 seeds)
# and report RoboTwin-judged success (`eval_success`).
#   eval.sh <out-dir> <tasks|all> [pi args...]
#   eval.sh runs/rt all --model openai/gpt-5.5 --thinking xhigh
#   eval.sh runs/rt beat_block_hammer,adjust_bottle --model ...
#
# Each episode runs in <out>/<task>_s<seed>/ and ends with a result.json taken from the session's
# `robot_result` entry. An episode is valid when the environment produced a result and the
# planner did not fail (`env_error`, `planner_error` and a missing result are invalid), whatever
# the outcome. Rerunning retries exactly the invalid episodes; valid ones are kept. Each result
# records the model and thinking level, and the summary covers only the requested cells and
# refuses to mix configurations.
set -uo pipefail
out=$1 tasks=$2
shift 2
here=$(cd "$(dirname "$0")" && pwd)
SERVICES=${PI_EMBODIED_SERVICES:-$(cd "$here/../../../../services" && pwd)}
PI=${PI:-pi}
model="" thinking=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
	case ${args[i]} in
	--model) model=${args[i + 1]:-} ;;
	--thinking) thinking=${args[i + 1]:-} ;;
	esac
done
mkdir -p "$out"

record() { # <dir> <exit code>: write result.json from the episode's session
	node --input-type=module -e '
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
const [dir, code, model, thinking] = process.argv.slice(1);
const results = [];
for (const f of readdirSync(dir).filter((f) => f.endsWith(".jsonl"))) {
	for (const line of readFileSync(`${dir}/${f}`, "utf8").split("\n")) {
		if (!line.includes("\"robot_result\"")) continue;
		const e = JSON.parse(line);
		if (e.type === "custom" && e.customType === "robot_result") results.push(e.data);
	}
}
const last = results.length === 1 ? results[0] : undefined;
const status = results.length > 1 ? "duplicate_result"
	: !last ? (Number(code) ? "env_error" : "missing")
	: last.env_error ? "env_error" : last.planner_error ? "planner_error" : last.success ? "success" : "failure";
const result = { ...(last ?? {}), status, exit_code: Number(code), model: model || null, thinking: thinking || null };
writeFileSync(`${dir}/result.json`, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ status, success: result.success, claimed: result.claimed, native_actions: result.native_actions }));
' "$1" "$2" "$model" "$thinking"
}

valid() { # <dir>: 0 = a valid result of this configuration, 2 = a valid result of another one, 1 = none
	node -e '
const [path, model, thinking] = process.argv.slice(1);
const r = JSON.parse(require("fs").readFileSync(path, "utf8"));
if (r.status !== "success" && r.status !== "failure") process.exit(1);
process.exit(r.model === (model || null) && r.thinking === (thinking || null) ? 0 : 2);
' "$1/result.json" "$model" "$thinking" 2>/dev/null
}

cells=$(node -e '
const table = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8")).tasks;
const want = process.argv[2] === "all" ? Object.keys(table) : process.argv[2].split(",");
for (const t of want) {
	if (!table[t]) throw new Error(`unknown task ${t}`);
	for (const e of table[t]) console.log(`${t}_s${e.seed}`);
}' "$SERVICES/pi_embodied_services/robots/robotwin/eval/demo_randomized.json" "$tasks") || exit 1

for cell in $cells; do
	task=${cell%_s*} seed=${cell##*_s}
	dir=$out/$cell
	valid "$dir"
	case $? in
	0) continue ;;
	2) echo "$dir holds a result of another model or thinking level; use another out dir" >&2 && exit 1 ;;
	esac
	rm -rf "$dir" && mkdir -p "$dir"
	echo "== $task seed $seed"
	$PI -p --session-dir "$dir" -e "$here" --task-name "$task" --task-config demo_randomized --seed "$seed" "$@" \
		"Solve the task. Bind targets and relations from fresh observation, verify each gate, and keep lingbot_act on the native task language." \
		</dev/null >"$dir/stdout.log" 2>"$dir/stderr.log"
	record "$dir" "$?"
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
const configs = new Set(rows.filter((r) => r.status === "success" || r.status === "failure").map((r) => `${r.model}/${r.thinking}`));
if (configs.size > 1) {
	console.log(`refusing to summarize: ${out} mixes configurations ${[...configs].join(", ")}`);
	process.exit(1);
}
const n = (s) => rows.filter((r) => r.status === s).length;
const scored = n("success") + n("failure");
const lies = rows.filter((r) => r.status === "failure" && r.claimed === "success").length;
const rate = scored ? ((100 * n("success")) / scored).toFixed(1) : "-";
const invalid = rows.length - scored;
console.log(`${[...configs][0] ?? "-"}: success ${n("success")}/${scored} (${rate}%), claimed-but-failed ${lies}, invalid ${invalid} (env_error ${n("env_error")}, planner_error ${n("planner_error")}, missing ${n("missing")}, duplicate ${n("duplicate_result")}) of ${rows.length}`);
if (invalid) process.exit(1);
' "$out" $cells
