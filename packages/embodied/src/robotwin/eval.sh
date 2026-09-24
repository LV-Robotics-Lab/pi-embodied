#!/usr/bin/env bash
# Run RoboTwin episodes with pi in print mode on RPent's verified evaluation seeds
# (robots/robotwin/eval/demo_randomized.json: 50 tasks x 5 seeds) and report
# RoboTwin-judged success.
#   eval.sh <out-dir> <tasks|all> [pi args...]
#   eval.sh runs/rt all --model openai/gpt-5.5 --thinking xhigh
#   eval.sh runs/rt beat_block_hammer,adjust_bottle --model ...
set -uo pipefail
out=$1 tasks=$2
shift 2
: "${RPENT_ROOT:?RPent checkout}"
here=$(cd "$(dirname "$0")" && pwd)
PI=${PI:-pi}
mkdir -p "$out"

cells=$(node -e '
const table = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8")).tasks;
const want = process.argv[2] === "all" ? Object.keys(table) : process.argv[2].split(",");
for (const t of want) {
	if (!table[t]) throw new Error(`unknown task ${t}`);
	for (const e of table[t]) console.log(`${t} ${e.seed}`);
}' "$RPENT_ROOT/robots/robotwin/eval/demo_randomized.json" "$tasks") || exit 1

while read -r task seed; do
	echo "== $task seed $seed"
	$PI -p --session-dir "$out" -e "$here" --task-name "$task" --task-config demo_randomized --seed "$seed" "$@" \
		"Solve the task. Bind targets and relations from fresh observation, verify each gate, and keep lingbot_act on the native task language." \
		</dev/null 2> >(tee -a "$out/stderr.log" | grep '^\[robotwin\]' | sed 's/^\[robotwin\] //' >>"$out/results.jsonl") >/dev/null
	tail -n 1 "$out/results.jsonl"
done <<<"$cells"

node -e '
const rows = require("fs").readFileSync(process.argv[1], "utf8").trim().split("\n").map(JSON.parse);
const ok = rows.filter((r) => r.success).length;
const lies = rows.filter((r) => r.claimed === "success" && !r.success).length;
const timeouts = rows.filter((r) => !r.success && (r.planner_budget_exhausted || r.budget_exhausted)).length;
console.log(`success ${ok}/${rows.length} (${((100 * ok) / rows.length).toFixed(1)}%), claimed-but-failed ${lies}, timeouts ${timeouts}`);
' "$out/results.jsonl"
