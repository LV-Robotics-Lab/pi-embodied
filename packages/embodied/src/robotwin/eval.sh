#!/usr/bin/env bash
# Run RoboTwin episodes with pi in print mode on RPent's verified evaluation seeds
# (robots/robotwin/eval/demo_randomized.json: 50 tasks x 5 seeds) and report
# RoboTwin-judged success.
#   eval.sh <out-dir> <tasks|all> [pi args...]
#   eval.sh runs/rt all --model openai/gpt-5.5 --thinking xhigh
#   eval.sh runs/rt beat_block_hammer,adjust_bottle --model ...
# Each episode gets <out>/<task>_s<seed>/ (session, stderr.log, record.json). An episode is
# valid when pi wrote exactly its one robotwin_result; startup failures and errors (crash, no
# result) are reported and left out of the rate. Re-running retries exactly the invalid episodes.
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
	dir=$out/${task}_s$seed
	node -e 'process.exit(JSON.parse(require("fs").readFileSync(process.argv[1], "utf8")).status === "ok" ? 0 : 1)' \
		"$dir/record.json" 2>/dev/null && continue
	rm -rf "$dir" && mkdir -p "$dir"
	echo "== $task seed $seed"
	$PI -p --session-dir "$dir" -e "$here" --task-name "$task" --task-config demo_randomized --seed "$seed" "$@" \
		"Solve the task. Bind targets and relations from fresh observation, verify each gate, and keep lingbot_act on the native task language." \
		</dev/null >/dev/null 2>"$dir/stderr.log"
	code=$?
	node -e '
const fs = require("fs");
const [dir, task, seed, code] = process.argv.slice(1);
const lines = fs.readFileSync(`${dir}/stderr.log`, "utf8").split("\n");
const unavailable = lines.find((l) => l.startsWith("[robotwin] unavailable: "));
const results = lines.filter((l) => l.startsWith("[robotwin] {")).map((l) => JSON.parse(l.slice(11)));
const result = results.length === 1 ? results[0] : null;
const status = unavailable ? "startup_failure"
	: result && (code === "0" || result.success) ? "ok" : "error";
const error = unavailable ? unavailable.slice(11)
	: results.length > 1 ? `${results.length} robotwin_result lines, expected 1`
	: !result ? `no robotwin_result (pi exit ${code}); see ${dir}/stderr.log`
	: status === "error" ? `pi exit ${code}` : null;
const record = { task_name: task, seed: Number(seed), status, exit: Number(code), error, result };
fs.writeFileSync(`${dir}/record.json`, `${JSON.stringify(record, null, 2)}\n`);
console.log(JSON.stringify({ status, success: result?.success ?? null, exit: Number(code), error }));
' "$dir" "$task" "$seed" "$code"
done <<<"$cells"

node -e '
const fs = require("fs");
const [out, cells] = process.argv.slice(1);
const records = cells.trim().split("\n").map((c) => {
	const [task, seed] = c.split(" ");
	try { return JSON.parse(fs.readFileSync(`${out}/${task}_s${seed}/record.json`, "utf8")); }
	catch { return { task_name: task, seed: Number(seed), status: "error", error: "no record" }; }
});
const rows = records.filter((r) => r.status === "ok").map((r) => r.result);
const ok = rows.filter((r) => r.success).length;
const lies = rows.filter((r) => r.claimed === "success" && !r.success).length;
const timeouts = rows.filter((r) => !r.success && (r.planner_budget_exhausted || r.budget_exhausted)).length;
const pct = rows.length ? ((100 * ok) / rows.length).toFixed(1) : "n/a";
console.log(`success ${ok}/${rows.length} (${pct}%), claimed-but-failed ${lies}, timeouts ${timeouts}`);
const bad = records.filter((r) => r.status !== "ok");
for (const r of bad) console.log(`${r.status}: ${r.task_name} s${r.seed}: ${r.error}`);
if (bad.length) {
	const n = (s) => bad.filter((r) => r.status === s).length;
	console.log(`excluded: ${n("startup_failure")} startup failures, ${n("error")} errors of ${records.length} episodes`);
	process.exit(1);
}
' "$out" "$cells"
