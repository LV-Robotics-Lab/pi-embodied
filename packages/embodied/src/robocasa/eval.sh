#!/usr/bin/env bash
# Run RoboCasa Target50 episodes with pi in print mode and report RoboCasa-judged success
# (`state.success`). The matrix (splits, tasks, seeds, per-split cell timeouts), the frozen RLDX
# settings and the task-memory revision come from
# services/pi_embodied_services/robots/robocasa/eval/target50.json (TARGET50 overrides it).
#   eval.sh <out-dir> <splits|all> [pi args...]
#   eval.sh runs/t50 all --model openai/gpt-5.5 --thinking xhigh --memory-profile local --memory-dir target50-memory/robocasa
# TASKS=OpenDrawer,CloseFridge and SEEDS=1,2 narrow the matrix.
#
# Each episode runs in <out>/<split>/<Task>_s<seed>/ and ends with a result.json in the Target50
# result schema (schema_version, protocol_id, evaluation_split, valid, success_source,
# termination_reason, planner, runtime), built from the session's `robot_result` entry. The planner
# is recorded as it ran: backend "pi", the model without its provider prefix, --thinking as
# reasoning_effort, --max-turns; the units mode (--units, --stateless) and visual differencing
# (--vdm, --vdm-model, --vdm-wrist) are recorded next to it.
#
# An episode is valid when the environment produced a result and the planner did not fail
# (`env_error`, `planner_error`, a missing or duplicate result and a killed process are invalid,
# termination_reason `infrastructure_error`), whatever the outcome. A spent --time-limit or
# --max-turns budget is a valid planner_timeout. Rerunning retries exactly the invalid episodes
# (the manifest's retry_policy); valid ones are kept, and results of another configuration are refused.
#
# The summary is task-weighted per split. A full run (all splits, no TASKS/SEEDS) is also scored by
# validate_target50.py (services/.../robocasa/eval) against <out>/target50.pi.json: the manifest with planner_reference
# replaced by the planner that actually ran, everything else (protocol, matrix, timeouts, RLDX
# settings, success source) unchanged. That validator's `overall.success_rate` is the Target50 score.
# A --privileged run (simulator ground truth) is recorded as such and never shares an out dir with one without.
#
# RoboCasa365 manifest mode (protocol `robocasa365`, separate from Target50): the second argument is
# a RoboCasa365 split, `pretrain`, `target` or `pretrain,target`, and the matrix is the full task table
# (services/.../robocasa/eval/robocasa365.json: 317 tasks x 50 manifest scenes per split), narrowed by
# TASKS=OpenDrawer,PrepareCoffee and SCENES=0-4 (ranges or lists of manifest indices).
#   eval.sh runs/rc365 target --model openai/gpt-5.5 --thinking low
# Each cell runs in <out>/<split>/<Task>_m<scene>/ with `--scene <index>` (the table's seed for that
# scene) and ends with a result.json of the robocasa365 schema: protocol "robocasa365", split, task_name,
# scene, seed, env_id, success, status, termination_reason, the planner and the units/stateless/privileged
# modes as above. The cell timeout is TIME_LIMIT (default 1800 s). Reruns retry invalid cells the same
# way. The summary is per split (success rate, task-weighted rate, invalid cells), printed and written to
# <out>/robocasa365-summary.json; validate_target50.py never sees these results. An out dir holds results
# of one protocol only: a Target50 run refuses a dir with robocasa365 results and vice versa.
# The fallback planner (--fallback-model, --fallback-after, --fallback-retry-primary; src/fallback.ts) is part of the
# configuration too, and the summary totals the turns each planner model planned (planner_models).
set -uo pipefail
out=$1 splits=$2
shift 2
here=$(cd "$(dirname "$0")" && pwd)
SERVICES=${PI_EMBODIED_SERVICES:-$(cd "$here/../../../../services" && pwd)}
manifest=${TARGET50:-$SERVICES/pi_embodied_services/robots/robocasa/eval/target50.json}
table=$SERVICES/pi_embodied_services/robots/robocasa/eval/robocasa365.json
validator=$SERVICES/pi_embodied_services/robots/robocasa/eval/validate_target50.py
PI=${PI:-pi}
PY=${PI_EMBODIED_PYTHON:-python3}
mode=target50
case $splits in pretrain | target | pretrain,target | target,pretrain) mode=robocasa365 ;; esac
[ "$splits" = all ] && splits=atomic,composite_seen,composite_unseen
full=""
[ "$splits" = atomic,composite_seen,composite_unseen ] && [ -z "${TASKS:-}${SEEDS:-}" ] && full=1
model="" thinking="" turns=${MAX_TURNS:-100} units=false stateless=false
anchor=false
vdm=false vdm_model="" vdm_wrist=false
privileged=false
fallback_model="" fallback_after=2 fallback_retry=0
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
	case ${args[i]} in
	--model) model=${args[i + 1]:-} ;;
	--thinking) thinking=${args[i + 1]:-} ;;
	--max-turns) turns=${args[i + 1]:-0} ;;
	--model=*) model=${args[i]#*=} ;;
	--thinking=*) thinking=${args[i]#*=} ;;
	--max-turns=*) turns=${args[i]#*=} ;;
	--units) [[ ${args[i + 1]:---} == --* ]] && units=true || units=${args[i + 1]} ;;
	--units=*) units=${args[i]#*=} ;;
	# pi sets a boolean flag to true whatever value it is given (`--stateless=false` runs stateless)
	# and takes a following word as that value: only the forms that say what pi runs are accepted.
	--stateless) case ${args[i + 1]:-} in "" | -* | @* | true) stateless=true ;; *)
		echo "--stateless takes no value: pi would run stateless and swallow '${args[i + 1]}'" >&2 && exit 2 ;;
	esac ;;
	--stateless=true) stateless=true ;;
	--stateless=*)
		echo "${args[i]}: pi ignores a boolean flag's value and would run stateless; omit --stateless for a stateful run" >&2
		exit 2
		;;
	# --privileged (simulator ground truth, ground_truth_poses) is a boolean like --stateless.
	--privileged) case ${args[i + 1]:-} in "" | -* | @* | true) privileged=true ;; *)
		echo "--privileged takes no value: pi would turn it on and swallow '${args[i + 1]}'" >&2 && exit 2 ;;
	esac ;;
	--privileged=true) privileged=true ;;
	--privileged=*)
		echo "${args[i]}: pi ignores a boolean flag's value and would turn it on; omit --privileged for a run without ground truth" >&2
		exit 2
		;;
	--vdm | --vdm-wrist) case ${args[i + 1]:-} in "" | -* | @* | true) [ "${args[i]}" = --vdm ] && vdm=true || vdm_wrist=true ;; *)
		echo "${args[i]} takes no value: pi would turn it on and swallow '${args[i + 1]}'" >&2 && exit 2 ;;
	esac ;;
	--vdm=true) vdm=true ;;
	--vdm-wrist=true) vdm_wrist=true ;;
	--vdm=* | --vdm-wrist=*)
		echo "${args[i]}: pi ignores a boolean flag's value and would turn it on; omit it to leave it off" >&2
		exit 2
		;;
	--fallback-model) fallback_model=${args[i + 1]:-} ;;
	--fallback-model=*) fallback_model=${args[i]#*=} ;;
	--fallback-after) fallback_after=${args[i + 1]:-2} ;;
	--fallback-after=*) fallback_after=${args[i]#*=} ;;
	--fallback-retry-primary) fallback_retry=${args[i + 1]:-0} ;;
	--fallback-retry-primary=*) fallback_retry=${args[i]#*=} ;;
	--vdm-model) vdm_model=${args[i + 1]:-} ;;
	--vdm-model=*) vdm_model=${args[i]#*=} ;;
	# --anchor-image (keep the first camera frame in context) is a boolean like --stateless.
	--anchor-image) case ${args[i + 1]:-} in "" | -* | @* | true) anchor=true ;; *)
		echo "--anchor-image takes no value: pi would turn it on and swallow '${args[i + 1]}'" >&2 && exit 2 ;;
	esac ;;
	--anchor-image=true) anchor=true ;;
	--anchor-image=*)
		echo "${args[i]}: pi ignores a boolean flag's value and would turn it on; omit --anchor-image to leave it off" >&2
		exit 2
		;;
	esac
done
[ "$units" = pure ] && units=true

# One protocol per out dir: Target50 results carry protocol_id, robocasa365 results protocol "robocasa365".
node -e '
const fs = require("fs");
const [out, protocol] = process.argv.slice(1);
const found = new Set();
const walk = (dir, depth) => {
	for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
		if (e.isDirectory() && depth < 2) walk(`${dir}/${e.name}`, depth + 1);
		else if (e.name === "result.json") {
			try {
				const r = JSON.parse(fs.readFileSync(`${dir}/result.json`, "utf8"));
				found.add(r.protocol === "robocasa365" ? "robocasa365" : r.protocol_id ? "target50" : "unknown");
			} catch {
				found.add("unknown");
			}
		}
	}
};
if (fs.existsSync(out)) walk(out, 0);
found.delete(protocol);
if (found.size) {
	console.error(`${out} holds ${[...found].join(", ")} results; a ${protocol} run needs another out dir`);
	process.exit(1);
}
' "$out" "$mode" || exit 1

if [ "$mode" = robocasa365 ]; then
	limit=${TIME_LIMIT:-1800}
	config=("$model" "$thinking" "$turns" "$limit" "$units" "$stateless" "$privileged" "$fallback_model" "$fallback_after" "$fallback_retry")
	unset RLDX_RESET_SEED

	record365() { # <dir> <exit code> <split> <task> <scene> <elapsed s>: write result.json (robocasa365 schema)
		node --input-type=module -e '
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
const [dir, code, table, split, task, scene, elapsed, model, thinking, turns, limit, units, stateless, privileged, fallbackModel, fallbackAfter, fallbackRetry] = process.argv.slice(1);
const t = JSON.parse(readFileSync(table, "utf8")).tasks.find((t) => t.name === task);
const results = [];
for (const f of readdirSync(dir).filter((f) => f.endsWith(".jsonl"))) {
	for (const line of readFileSync(`${dir}/${f}`, "utf8").split("\n")) {
		if (!line.includes("\"robot_result\"")) continue;
		const e = JSON.parse(line);
		if (e.type === "custom" && e.customType === "robot_result") results.push(e.data);
	}
}
const last = results.length === 1 ? results[0] : undefined;
const killed = Number(code) === 124 || Number(code) === 137;
const status = killed ? "timeout" : results.length > 1 ? "duplicate_result"
	: !last ? (Number(code) ? "env_error" : "missing")
	: last.env_error ? "env_error" : last.planner_error ? "planner_error" : last.success ? "success" : "failure";
const valid = status === "success" || status === "failure";
const slash = model.indexOf("/");
const result = {
	...(last ?? {}),
	protocol: "robocasa365",
	benchmark: "RoboCasa365",
	split,
	task_name: task,
	env_id: `robocasa365/${split}/${task}`,
	scene: Number(scene),
	seed: t.manifest[Number(scene)],
	horizon: t.horizon,
	valid,
	success: last?.success === true,
	success_source: "state.success",
	termination_reason: !valid ? "infrastructure_error" : status === "failure" && last.planner_budget_exhausted ? "planner_timeout" : "completed",
	elapsed_s: Number(elapsed),
	planner: {
		backend: "pi",
		model: (slash >= 0 ? model.slice(slash + 1) : model) || null,
		reasoning_effort: thinking || null,
		max_turns: Number(turns),
	},
	planner_provider: slash >= 0 ? model.slice(0, slash) : null,
	runtime: {
		cell_timeout_seconds: Number(limit),
		rldx_max_chunks: last?.rldx_max_chunks ?? null,
		rldx_settle_patience: last?.rldx_settle_patience ?? null,
		rldx_action_steps_per_chunk: last?.rldx_action_steps_per_chunk ?? null,
	},
	status,
	exit_code: Number(code),
	model: model || null,
	thinking: thinking || null,
	max_turns: Number(turns),
	time_limit: Number(limit),
	units,
	stateless: stateless === "true",
	privileged: privileged === "true",
	fallback_model: fallbackModel || null, fallback_after: fallbackModel ? Number(fallbackAfter) : null, fallback_retry_primary: fallbackModel ? Number(fallbackRetry) : null,
};
writeFileSync(`${dir}/result.json`, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ status, termination_reason: result.termination_reason, success: result.success, claimed: result.claimed, env_steps: result.env_steps }));
' "$1" "$2" "$table" "$3" "$4" "$5" "$6" "${config[@]}"
	}

	valid365() { # <dir>: 0 = a valid result of this configuration, 2 = a valid result of another one, 1 = none
		node -e '
const [path, model, thinking, turns, limit, units, stateless, privileged, fallbackModel, fallbackAfter, fallbackRetry] = process.argv.slice(1);
const r = JSON.parse(require("fs").readFileSync(path, "utf8"));
if (r.status !== "success" && r.status !== "failure") process.exit(1);
const same = r.protocol === "robocasa365" && r.model === (model || null) && r.thinking === (thinking || null) && r.max_turns === Number(turns)
	&& r.time_limit === Number(limit) && r.units === units && r.stateless === (stateless === "true") && (r.privileged ?? false) === (privileged === "true")
	// Results written before --fallback-model existed ran without a fallback planner.
	&& (r.fallback_model ?? null) === (fallbackModel || null) && (r.fallback_after ?? null) === (fallbackModel ? Number(fallbackAfter) : null)
	&& (r.fallback_retry_primary ?? null) === (fallbackModel ? Number(fallbackRetry) : null);
process.exit(same ? 0 : 2);
' "$1/result.json" "${config[@]}" 2>/dev/null
	}

	cells=$(node -e '
const t = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
const [splits, tasks, scenes] = process.argv.slice(2);
const only = (s) => (s ? s.split(",") : null);
const expand = (s) => s.split(",").flatMap((p) => { const [a, b] = p.split("-").map(Number); return Array.from({ length: (b ?? a) - a + 1 }, (_, i) => a + i); });
const names = new Set(t.tasks.map((x) => x.name));
for (const name of only(tasks) ?? []) if (!names.has(name)) throw new Error(`TASKS: ${name} is not a RoboCasa365 task`);
const idx = scenes ? expand(scenes) : Array.from({ length: t.scenes_per_task }, (_, i) => i);
for (const i of idx) if (!(i >= 0 && i < t.scenes_per_task)) throw new Error(`SCENES: ${i} is not a manifest index 0..${t.scenes_per_task - 1}`);
for (const split of splits.split(",")) {
	if (!t.splits.includes(split)) throw new Error(`unknown RoboCasa365 split ${split}`);
	for (const task of t.tasks)
		if (!tasks || only(tasks).includes(task.name))
			for (const i of idx) console.log(`${split} ${task.name} ${i}`);
}' "$table" "$splits" "${TASKS:-}" "${SCENES:-}") || exit 1
	[ -n "$cells" ] || { echo "no RoboCasa365 cells match splits=$splits TASKS=${TASKS:-} SCENES=${SCENES:-}" >&2 && exit 1; }

	while read -r split task scene; do
		dir=$out/$split/${task}_m$scene
		valid365 "$dir"
		case $? in
		0) continue ;;
		2) echo "$dir holds a result of another model, thinking level, --max-turns, TIME_LIMIT, units mode, fallback or --privileged (or of another protocol); use another out dir" >&2 && exit 1 ;;
		esac
		rm -rf "$dir" && mkdir -p "$dir"
		echo "== $split $task scene $scene"
		start=$SECONDS
		backstop=()
		[ "$limit" -gt 0 ] && command -v timeout >/dev/null && backstop=(timeout -k 30 $((limit + 900)))
		${backstop[@]+"${backstop[@]}"} $PI -p --session-dir "$dir" -e "$here" --task-name "$task" --split "$split" \
			--scene "$scene" --max-turns "$turns" --time-limit "$limit" --log-dir "$dir" "Solve the task." "$@" \
			</dev/null >"$dir/stdout.log" 2>"$dir/stderr.log"
		code=$?
		record365 "$dir" "$code" "$split" "$task" "$scene" $((SECONDS - start))
	done <<<"$cells"

	# The summary per split: success, task-weighted success (mean over tasks of their scene success rate), invalid cells.
	node -e '
const fs = require("fs");
const [out, cells] = process.argv.slice(1);
const rows = cells.trim().split("\n").map((line) => {
	const [split, task, scene] = line.split(" ");
	try {
		return { split, task, ...JSON.parse(fs.readFileSync(`${out}/${split}/${task}_m${scene}/result.json`, "utf8")) };
	} catch {
		return { split, task, status: "missing" };
	}
});
const scoredRows = rows.filter((r) => r.status === "success" || r.status === "failure");
const configs = new Set(scoredRows.map((r) => `${r.model}/${r.thinking}/turns=${r.max_turns}/limit=${r.time_limit}/units=${r.units}${r.stateless ? "/stateless" : ""}${r.privileged ? "/privileged" : ""}${r.fallback_model ? `/fallback=${r.fallback_model}:${r.fallback_after}:${r.fallback_retry_primary}` : ""}`));
if (configs.size > 1) {
	console.log(`refusing to summarize: ${out} mixes configurations ${[...configs].join(", ")}`);
	process.exit(1);
}
const pm = rows.reduce((a, r) => r.planner_models ? { primary: a.primary + (r.planner_models.primary ?? 0), fallback: a.fallback + (r.planner_models.fallback ?? 0), rows: a.rows + 1 } : a, { primary: 0, fallback: 0, rows: 0 });
const summary = { protocol: "robocasa365", config: [...configs][0] ?? null, planner_models: pm.rows ? { primary: pm.primary, fallback: pm.fallback } : null, splits: {} };
let invalid = 0;
for (const split of new Set(rows.map((r) => r.split))) {
	const all = rows.filter((r) => r.split === split);
	const s = scoredRows.filter((r) => r.split === split);
	const ok = s.filter((r) => r.status === "success").length;
	const perTask = new Map();
	for (const r of s) {
		const [k, n] = perTask.get(r.task) ?? [0, 0];
		perTask.set(r.task, [k + (r.status === "success"), n + 1]);
	}
	const weighted = perTask.size ? [...perTask.values()].reduce((a, [k, n]) => a + k / n, 0) / perTask.size : 0;
	const bad = all.length - s.length;
	invalid += bad;
	summary.splits[split] = {
		tasks: perTask.size,
		cells: all.length,
		valid_cells: s.length,
		successes: ok,
		success_rate: s.length ? ok / s.length : null,
		task_weighted_success_rate: perTask.size ? weighted : null,
		planner_timeout: s.filter((r) => r.termination_reason === "planner_timeout").length,
		claimed_but_failed: s.filter((r) => r.status === "failure" && r.claimed === "success").length,
		invalid: Object.fromEntries(["env_error", "planner_error", "timeout", "missing", "duplicate_result"].map((k) => [k, all.filter((r) => r.status === k).length])),
	};
	const x = summary.splits[split];
	console.log(`robocasa365 ${split}: success ${ok}/${s.length} (${s.length ? ((100 * ok) / s.length).toFixed(1) : "-"}%) over ${perTask.size} tasks, task-weighted ${perTask.size ? (100 * weighted).toFixed(1) : "-"}%, planner_timeout ${x.planner_timeout}, claimed-but-failed ${x.claimed_but_failed}, invalid ${bad} of ${all.length}`);
}
fs.writeFileSync(`${out}/robocasa365-summary.json`, `${JSON.stringify(summary, null, 2)}\n`);
console.log(`${summary.config ?? "-"}: summary written to ${out}/robocasa365-summary.json`);
if (invalid) process.exit(1);
' "$out" "$cells"
	exit $?
fi

protocol() { # <expr>: a value from the manifest
	node -e 'const m = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8")); console.log(eval(process.argv[2]))' "$manifest" "$1"
}
export RLDX_MAX_CHUNKS=$(protocol m.runtime_protocol.rldx_max_chunks)
export RLDX_SETTLE_PATIENCE=$(protocol m.runtime_protocol.rldx_settle_patience)
export RLDX_ACTION_STEPS_PER_CHUNK=$(protocol m.runtime_protocol.rldx_action_steps_per_chunk)
unset RLDX_RESET_SEED
config=("$model" "$thinking" "$turns" "$units" "$stateless" "$privileged" "$anchor" "$vdm" "$vdm_model" "$vdm_wrist" "$fallback_model" "$fallback_after" "$fallback_retry")
# The protocol pins the task-memory snapshot (hf profile); PI_EMBODIED_MEMORY_REVISION overrides it.
export PI_EMBODIED_MEMORY_REVISION=${PI_EMBODIED_MEMORY_REVISION:-$(protocol m.dependencies.task_memory.revision)}

record() { # <dir> <exit code> <split> <task> <seed> <cell timeout> <elapsed s>: write result.json
	node --input-type=module -e '
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
const [dir, code, manifest, split, task, seed, limit, elapsed, model, thinking, turns, units, stateless, privileged, anchor, vdm, vdmModel, vdmWrist, fallbackModel, fallbackAfter, fallbackRetry] = process.argv.slice(1);
const m = JSON.parse(readFileSync(manifest, "utf8"));
const results = [];
for (const f of readdirSync(dir).filter((f) => f.endsWith(".jsonl"))) {
	for (const line of readFileSync(`${dir}/${f}`, "utf8").split("\n")) {
		if (!line.includes("\"robot_result\"")) continue;
		const e = JSON.parse(line);
		if (e.type === "custom" && e.customType === "robot_result") results.push(e.data);
	}
}
const last = results.length === 1 ? results[0] : undefined;
const killed = Number(code) === 124 || Number(code) === 137;
const status = killed ? "timeout" : results.length > 1 ? "duplicate_result"
	: !last ? (Number(code) ? "env_error" : "missing")
	: last.env_error ? "env_error" : last.planner_error ? "planner_error" : last.success ? "success" : "failure";
const valid = status === "success" || status === "failure";
const slash = model.indexOf("/");
const result = {
	...(last ?? {}),
	// Target50 result schema, checked by validate_target50.py.
	schema_version: "1.0",
	protocol_id: m.protocol_id,
	evaluation_split: split,
	task_name: task,
	environment_split: m.environment_split,
	seed: Number(seed),
	valid,
	success: last?.success === true,
	success_source: m.success_source,
	termination_reason: !valid ? "infrastructure_error" : status === "failure" && last.planner_budget_exhausted ? "planner_timeout" : "completed",
	elapsed_s: Number(elapsed),
	planner: {
		backend: "pi",
		model: (slash >= 0 ? model.slice(slash + 1) : model) || null,
		reasoning_effort: thinking || null,
		max_turns: Number(turns),
	},
	planner_provider: slash >= 0 ? model.slice(0, slash) : null,
	runtime: {
		cell_timeout_seconds: Number(limit),
		rldx_max_chunks: last?.rldx_max_chunks ?? null,
		rldx_settle_patience: last?.rldx_settle_patience ?? null,
		rldx_action_steps_per_chunk: last?.rldx_action_steps_per_chunk ?? null,
	},
	// pi-embodied bookkeeping: why an episode is invalid, and the configuration reruns must match.
	status,
	exit_code: Number(code),
	model: model || null,
	thinking: thinking || null,
	max_turns: Number(turns),
	units,
	anchor_image: anchor === "true",
	vdm: vdm === "true", vdm_model: vdmModel || null, vdm_wrist: vdmWrist === "true", stateless: stateless === "true",
	privileged: privileged === "true",
	fallback_model: fallbackModel || null, fallback_after: fallbackModel ? Number(fallbackAfter) : null, fallback_retry_primary: fallbackModel ? Number(fallbackRetry) : null,
};
writeFileSync(`${dir}/result.json`, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ status, termination_reason: result.termination_reason, success: result.success, claimed: result.claimed, env_steps: result.env_steps }));
' "$1" "$2" "$manifest" "$3" "$4" "$5" "$6" "$7" "${config[@]}"
}

valid() { # <dir>: 0 = a valid result of this configuration, 2 = a valid result of another one, 1 = none
	node -e '
const [path, protocolId, model, thinking, turns, units, stateless, privileged, anchor, vdm, vdmModel, vdmWrist, fallbackModel, fallbackAfter, fallbackRetry] = process.argv.slice(1);
const r = JSON.parse(require("fs").readFileSync(path, "utf8"));
if (r.status !== "success" && r.status !== "failure") process.exit(1);
const same = r.protocol_id === protocolId && r.model === (model || null) && r.thinking === (thinking || null) && r.max_turns === Number(turns)
	&& r.units === units && r.stateless === (stateless === "true") && (r.privileged ?? false) === (privileged === "true")
	// Results written before --anchor-image existed ran without it.
	&& (r.anchor_image ?? false) === (anchor === "true")
	// Results written before --vdm existed ran without it.
	&& (r.vdm ?? false) === (vdm === "true") && (r.vdm_model ?? null) === (vdmModel || null)
	&& (r.vdm_wrist ?? false) === (vdmWrist === "true")
	// Results written before --fallback-model existed ran without a fallback planner.
	&& (r.fallback_model ?? null) === (fallbackModel || null) && (r.fallback_after ?? null) === (fallbackModel ? Number(fallbackAfter) : null)
	&& (r.fallback_retry_primary ?? null) === (fallbackModel ? Number(fallbackRetry) : null);
process.exit(same ? 0 : 2);
' "$1/result.json" "$(protocol m.protocol_id)" "${config[@]}" 2>/dev/null
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
[ -n "$cells" ] || { echo "no Target50 cells match splits=$splits TASKS=${TASKS:-} SEEDS=${SEEDS:-}" >&2 && exit 1; }

while read -r split task seed limit; do
	dir=$out/$split/${task}_s$seed
	valid "$dir"
	case $? in
	0) continue ;;
	2) echo "$dir holds a result of another protocol, model, thinking level, --max-turns, units mode, vdm, fallback, --privileged or --anchor-image (or an older result format); use another out dir" >&2 && exit 1 ;;
	esac
	rm -rf "$dir" && mkdir -p "$dir"
	echo "== $split $task seed $seed"
	start=$SECONDS
	# --time-limit ends the planner at the cell timeout (a planner_timeout); `timeout` is only the
	# backstop for a hung process, and a killed episode is invalid. The prompt precedes the user's
	# args: a bare boolean flag at their end would take it as its value.
	backstop=()
	command -v timeout >/dev/null && backstop=(timeout -k 30 $((limit + 900)))
	${backstop[@]+"${backstop[@]}"} $PI -p --session-dir "$dir" -e "$here" --task-name "$task" --split target \
		--seed "$seed" --max-turns "$turns" --time-limit "$limit" --log-dir "$dir" "Solve the task." "$@" \
		</dev/null >"$dir/stdout.log" 2>"$dir/stderr.log"
	code=$?
	record "$dir" "$code" "$split" "$task" "$seed" "$limit" $((SECONDS - start))
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
const configs = new Set(scoredRows.map((r) => `${r.model}/${r.thinking}/turns=${r.max_turns}/units=${r.units}${r.stateless ? "/stateless" : ""}${r.anchor_image ? "/anchor" : ""}${r.vdm ? `/vdm=${r.vdm_model ?? "default"}${r.vdm_wrist ? "+wrist" : ""}` : ""}${r.privileged ? "/privileged" : ""}${r.fallback_model ? `/fallback=${r.fallback_model}:${r.fallback_after}:${r.fallback_retry_primary}` : ""}`));
if (configs.size > 1) {
	console.log(`refusing to summarize: ${out} mixes configurations ${[...configs].join(", ")}`);
	process.exit(1);
}
const pm = rows.reduce((a, r) => r.planner_models ? { primary: a.primary + (r.planner_models.primary ?? 0), fallback: a.fallback + (r.planner_models.fallback ?? 0), rows: a.rows + 1 } : a, { primary: 0, fallback: 0, rows: 0 });
const planned = pm.rows ? `, planner_models primary=${pm.primary} fallback=${pm.fallback}` : "";
const n = (s) => rows.filter((r) => r.status === s).length;
const rate = (ok, all) => (all ? ((100 * ok) / all).toFixed(1) : "-");
const scored = scoredRows.length;
const lies = rows.filter((r) => r.status === "failure" && r.claimed === "success").length;
const timeouts = rows.filter((r) => r.termination_reason === "planner_timeout").length;
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
console.log(`${[...configs][0] ?? "-"}: success ${n("success")}/${scored} (${rate(n("success"), scored)}%), task-weighted ${perTask.size ? (100 * weighted).toFixed(1) : "-"}%, planner_timeout ${timeouts}, claimed-but-failed ${lies}, invalid ${invalid} (env_error ${n("env_error")}, planner_error ${n("planner_error")}, timeout ${n("timeout")}, missing ${n("missing")}, duplicate ${n("duplicate_result")}) of ${rows.length}${planned}`);
if (invalid) process.exit(1);
' "$out" "$cells"
summary=$?

# The manifest with planner_reference replaced by the planner that ran; nothing else changes.
derived=$out/target50.pi.json
node -e '
const fs = require("fs");
const [manifest, derived, model, thinking, turns] = process.argv.slice(1);
const m = JSON.parse(fs.readFileSync(manifest, "utf8"));
const slash = model.indexOf("/");
const ran = {
	planner: "pi",
	model: (slash >= 0 ? model.slice(slash + 1) : model) || null,
	reasoning_effort: thinking || null,
	max_turns: Number(turns),
};
const ref = m.planner_reference;
if (ref.model !== ran.model || ref.reasoning_effort !== ran.reasoning_effort || ref.max_turns !== ran.max_turns)
	console.log(`note: the planner (${ran.model}/${ran.reasoning_effort}/${ran.max_turns} turns) differs from the protocol reference (${ref.planner}/${ref.model}/${ref.reasoning_effort}/${ref.max_turns} turns)`);
const out = { ...m, planner_reference: { ...ref, ...ran }, reference_planner_replaced: { from: ref, by: "pi-embodied robocasa/eval.sh" } };
fs.writeFileSync(derived, `${JSON.stringify(out, null, 2)}\n`);
' "$manifest" "$derived" "$model" "$thinking" "$turns" || exit 1

if [ -z "$full" ]; then
	echo "partial run (splits=$splits TASKS=${TASKS:-} SEEDS=${SEEDS:-}): no Target50 score; a full run is scored by validate_target50.py"
	exit "$summary"
fi
echo "== validate_target50.py --manifest $derived"
"$PY" "$validator" "$out" --manifest "$derived"
validated=$?
[ "$summary" -eq 0 ] && [ "$validated" -eq 0 ]
