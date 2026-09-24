#!/usr/bin/env bash
# RoboCasa Target50 (robocasa-harness-vla-v1, RPent's robots/robocasa/eval/target50.json) with pi:
# one `pi -p` per manifest cell under the frozen RLDX settings and the split's cell timeout,
# RPent's result.json at <out>/<split>/<Task>_s<seed>/, then the task-weighted score.
#   eval.sh <out-dir> <splits|all> [pi args...]
#   eval.sh runs/t50 all --model openai/gpt-5.5 --thinking xhigh --memory-dir target50-memory/robocasa
# TASKS=OpenDrawer,CloseFridge and SEEDS=1,2 narrow the matrix. Re-running retries exactly the
# cells without a valid result (infrastructure failures); task failures and timeouts are final.
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
	node -e 'process.exit(JSON.parse(require("fs").readFileSync(process.argv[1], "utf8")).valid ? 0 : 1)' \
		"$dir/result.json" 2>/dev/null && continue
	rm -rf "$dir" && mkdir -p "$dir"
	echo "== $split $task seed $seed"
	start=$(date +%s)
	# --time-limit ends the planner gracefully; `timeout` is only the backstop for a hung process.
	timeout -k 30 $((limit + 900)) $PI -p --session-dir "$dir" -e "$here" --task-name "$task" --split target \
		--seed "$seed" --max-turns "$MAX_TURNS" --time-limit "$limit" --log-dir "$dir" "$@" "Solve the task." \
		</dev/null >/dev/null 2>"$dir/stderr.log"
	code=$?
	node -e '
const fs = require("fs");
const [dir, manifest, split, task, seed, limit, code, elapsed, model, effort, turns] = process.argv.slice(1);
const lines = fs.readFileSync(`${dir}/stderr.log`, "utf8").split("\n").filter((l) => l.startsWith("[robocasa] "));
const r = lines.length ? JSON.parse(lines[lines.length - 1].slice(11)) : null;
const available = r !== null && typeof r.success === "boolean";
const success = available && r.success;
const reason = success ? "completed"
	: code === "124" || code === "137" || r?.planner_budget_exhausted ? "planner_timeout"
	: code === "0" && available ? "completed" : "infrastructure_error";
const record = {
	schema_version: "1.0",
	protocol_id: JSON.parse(fs.readFileSync(manifest, "utf8")).protocol_id,
	evaluation_split: split,
	task_name: task,
	environment_split: "target",
	seed: Number(seed),
	valid: available && reason !== "infrastructure_error",
	success,
	success_source: "state.success",
	termination_reason: reason,
	elapsed_s: Number(elapsed),
	planner: { backend: "pi", model: model || null, reasoning_effort: effort || null, max_turns: Number(turns) },
	runtime: {
		cell_timeout_seconds: Number(limit),
		rldx_max_chunks: r?.rldx_max_chunks ?? null,
		rldx_settle_patience: r?.rldx_settle_patience ?? null,
		rldx_action_steps_per_chunk: r?.rldx_action_steps_per_chunk ?? null,
	},
};
fs.writeFileSync(`${dir}/result.json.tmp`, `${JSON.stringify(record, null, 2)}\n`);
fs.renameSync(`${dir}/result.json.tmp`, `${dir}/result.json`);
console.log(JSON.stringify({ success, reason, valid: record.valid, exit: Number(code) }));
' "$dir" "$manifest" "$split" "$task" "$seed" "$limit" "$code" "$(($(date +%s) - start))" "$model" "$effort" "$MAX_TURNS"
done <<<"$cells"

# Split rates over the manifest's cells; overall = mean of per-task rates (RoboCasa365 convention).
node -e '
const fs = require("fs");
const [manifest, out] = process.argv.slice(1);
const m = JSON.parse(fs.readFileSync(manifest, "utf8"));
const rates = [];
const summary = { protocol_id: m.protocol_id, splits: {}, missing_or_invalid: 0 };
for (const [name, split] of Object.entries(m.splits)) {
	let ok = 0;
	let valid = 0;
	for (const t of split.tasks) {
		let taskOk = 0;
		for (const s of split.seeds) {
			let r = null;
			try { r = JSON.parse(fs.readFileSync(`${out}/${name}/${t}_s${s}/result.json`, "utf8")); } catch {}
			if (!r?.valid) { summary.missing_or_invalid++; continue; }
			valid++;
			if (r.success) { ok++; taskOk++; }
		}
		rates.push(taskOk / split.seeds.length);
	}
	summary.splits[name] = { successes: ok, valid_cells: valid, expected_cells: split.cell_count, success_rate: +(ok / split.cell_count).toFixed(6) };
}
summary.task_weighted_success_rate = +(rates.reduce((a, b) => a + b, 0) / rates.length).toFixed(6);
console.log(JSON.stringify(summary, null, 2));
if (summary.missing_or_invalid) console.log(`incomplete: ${summary.missing_or_invalid} of ${m.total_cells} cells have no valid result (counted as failures above)`);
' "$manifest" "$out"
