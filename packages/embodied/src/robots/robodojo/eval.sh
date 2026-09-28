#!/usr/bin/env bash
# Run RoboDojo episodes with pi in print mode and report RoboDojo-judged success (its
# is_episode_end, `success`) and its mean episode score (1 on success, else the task's partial credit).
#   eval.sh <out-dir> <tasks> <seeds> [pi args...]
#   eval.sh runs/rd stack_bowls,push_T 0-4 --cuda-device 1 --model <provider/model> --thinking low
#   eval.sh runs/rd-units stack_bowls 0-2 --units=true --cuda-device 1 --model <provider/model> --thinking low
#
# <seeds> are RoboDojo eval layout ids (Assets/Eval_Layout/RoboDojo/arx_x5/<eval-seed>/<task>_<n>.json).
# RoboDojo's own sweep (SeedManager) walks a task's layouts in order and evaluates eval_nums episodes (25
# or 50, _task.yml), skipping a layout that proves unstable and drawing the next one: 0-24 is the
# benchmark's selection for a 25-episode task only when none of those layouts is unstable. Likewise here
# an unstable episode (the layout did not settle at reset, or the task flagged the scene during the
# episode) is recorded as `unstable`, counted neither as success nor as failure, and replaced by the next
# layout id after the selection's largest, until the task has no more layouts. Every episode starts its
# own Isaac Sim env server (minutes of cold start, one env per process); on a shared GPU run the whole
# script under that GPU's lock.
#
# Each episode runs in <out>/<task>_s<seed>/ and ends with a result.json taken from the session's
# `robot_result` entry. An episode is valid when the environment produced a result and the planner
# did not fail (`env_error`, `planner_error` and a missing result are invalid), whatever the
# outcome. Rerunning retries exactly the invalid episodes; valid ones are kept. Each result records
# the model, thinking level, --max-turns, --time-limit, the units mode (--units, --units-plugins, --stateless), visual
# differencing (--vdm, --vdm-model, --vdm-wrist, --vdm-video, --vdm-video-frames) and the layout set (--eval-seed), and the summary covers
# only the requested cells and refuses to mix configurations.
# A --privileged run (simulator ground truth) is recorded as such and never shares an out dir with one without.
# The fallback planner (--fallback-model, --fallback-after, --fallback-retry-primary; src/planner/fallback.ts) is part of the
# configuration too, and the summary totals the turns each planner model planned (planner_models).
# So is code mode (--code, --code-api: high, low or low-noexamples = CaP-X's S2-S4, and --code-oracle, a reference
# program run instead of the model), e.g.
#   eval.sh runs/rd-code stack_bowls 0-4 --code=true --code-api=low --model <provider/model>
set -uo pipefail
out=$1 tasks=$2 seeds=$3
shift 3
here=$(cd "$(dirname "$0")" && pwd)
PI=${PI:-pi}
expand() { for part in ${1//,/ }; do seq "${part%-*}" "${part#*-}"; done; }
# Common flags have one implementation; only robot-specific options live here.
source "$here/../../scripts/eval-options.sh"
eval_options_defaults
eval_seed=0
eval_robot_option() {
	case $1 in
	--eval-seed) eval_seed=${2:-0} ;;
	--eval-seed=*) eval_seed=${1#*=} ;;
	esac
}
eval_parse_options "$@"
eval_normalize_options

# --time-limit (default $TIME_LIMIT, 1800 s; 0 = none) ends the planner gracefully, as a failure;
# `timeout` is only the backstop for a hung process, and a killed episode is invalid.
[ -n "$limited" ] || set -- "$@" --time-limit "$limit"
backstop=()
[ "$limit" -gt 0 ] && command -v timeout >/dev/null && backstop=(timeout -k 30 $((limit + 900)))
mkdir -p "$out"
config=("$model" "$thinking" "$turns" "$limit" "$units" "$stateless" "$anchor" "$vdm" "$vdm_model" "$vdm_wrist" "$privileged" "$eval_seed" "$fallback_model" "$fallback_after" "$fallback_retry" "$approval" "$max_tool_calls" "$max_tokens" "$code" "$code_api" "$code_oracle" "$vdm_video")

record() { # <dir> <exit code>: write result.json from the episode's session
	node --input-type=module -e '
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
const [dir, code, model, thinking, turns, limit, units, stateless, anchor, vdm, vdmModel, vdmWrist, privileged, evalSeed, fallbackModel, fallbackAfter, fallbackRetry, approval, maxToolCalls, maxTokens, codeMode, codeApi, codeOracle, vdmVideo] = process.argv.slice(1);
const results = [];
for (const f of readdirSync(dir).filter((f) => f.endsWith(".jsonl"))) {
	for (const line of readFileSync(`${dir}/${f}`, "utf8").split("\n")) {
		if (!line.includes("\"robot_result\"")) continue;
		const e = JSON.parse(line);
		if (e.type === "custom" && e.customType === "robot_result") results.push(e.data);
	}
}
// An oracle run (--code-oracle) asks no model, and pi writes no session file without an assistant
// message: the result line the robot prints on stderr is the result.
if (!results.length && codeOracle && codeMode !== "false")
	try {
		for (const line of readFileSync(`${dir}/stderr.log`, "utf8").split("\n"))
			if (line.startsWith("[robodojo] {")) results.push(JSON.parse(line.slice("[robodojo] ".length)));
	} catch {}
const last = results.length === 1 ? results[0] : undefined;
// RoboDojo discards an unstable episode: its layout did not settle at reset (the robot fails to start,
// naming it) or the task flagged the scene (the result says so).
let stderr = "";
try {
	stderr = readFileSync(`${dir}/stderr.log`, "utf8");
} catch {}
const unstable = /is unstable in simulation/.test(stderr) || (results.length === 1 && results[0].unstable === true);
const status = unstable ? "unstable" : Number(code) === 124 ? "timeout" : results.length > 1 ? "duplicate_result"
	: !last ? (Number(code) ? "env_error" : "missing")
	: last.env_error ? "env_error" : last.planner_error ? "planner_error" : last.success ? "success" : "failure";
const result = { ...(last ?? {}), status, exit_code: Number(code), model: model || null, thinking: thinking || null,
	max_turns: Number(turns), time_limit: Number(limit), units, anchor_image: anchor === "true", approval, max_tool_calls: Number(maxToolCalls), max_tokens: Number(maxTokens),
	vdm: vdm === "true", vdm_model: vdmModel || null, vdm_wrist: vdmWrist === "true", vdm_video: vdmVideo ? Number(vdmVideo) : null, privileged: privileged === "true",
	stateless: stateless === "true", eval_seed: Number(evalSeed),
	fallback_model: fallbackModel || null, fallback_after: fallbackModel ? Number(fallbackAfter) : null, fallback_retry_primary: fallbackModel ? Number(fallbackRetry) : null,
	code: codeMode, code_api: codeMode === "false" ? null : codeApi || last?.code_api || null, code_api_auto: codeMode === "false" ? null : !codeApi, code_oracle: codeMode === "false" ? null : codeOracle || null };
writeFileSync(`${dir}/result.json`, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ status, success: result.success, score: result.score, claimed: result.claimed, env_steps: result.env_steps }));
' "$1" "$2" "${config[@]}"
}

valid() { # <dir>: 0 = a valid result of this configuration, 2 = a valid result of another one, 1 = none
	node -e '
const [path, model, thinking, turns, limit, units, stateless, anchor, vdm, vdmModel, vdmWrist, privileged, evalSeed, fallbackModel, fallbackAfter, fallbackRetry, approval, maxToolCalls, maxTokens, codeMode, codeApi, codeOracle, vdmVideo] = process.argv.slice(1);
const r = JSON.parse(require("fs").readFileSync(path, "utf8"));
if (r.status !== "success" && r.status !== "failure" && r.status !== "unstable") process.exit(1);
const same = r.model === (model || null) && r.thinking === (thinking || null) && r.max_turns === Number(turns)
	&& r.time_limit === Number(limit) && r.units === units && r.stateless === (stateless === "true")
	// Results written before --anchor-image existed ran without it.
	&& (r.anchor_image ?? false) === (anchor === "true")
	// Results without an approval field ran with it off. "standard" results written before it confirmed high-risk motions ran unconfirmed: they are refused (another configuration).
	&& (r.approval ?? "off") === approval && (r.max_tool_calls ?? 0) === Number(maxToolCalls) && (r.max_tokens ?? 0) === Number(maxTokens)
	// Results written before --vdm existed ran without it.
	&& (r.vdm ?? false) === (vdm === "true") && (r.vdm_model ?? null) === (vdmModel || null)
	&& (r.vdm_wrist ?? false) === (vdmWrist === "true")
	// Results written before --vdm-video existed ran without it.
	&& (r.vdm_video ?? null) === (vdmVideo ? Number(vdmVideo) : null)
	// Results written before --privileged existed ran without it.
	&& (r.privileged ?? false) === (privileged === "true")
	&& r.eval_seed === Number(evalSeed)
	// Results written before --fallback-model existed ran without a fallback planner.
	&& (r.fallback_model ?? null) === (fallbackModel || null) && (r.fallback_after ?? null) === (fallbackModel ? Number(fallbackAfter) : null)
	&& (r.fallback_retry_primary ?? null) === (fallbackModel ? Number(fallbackRetry) : null)
	// Results written before code mode existed here ran without it.
	&& (r.code ?? "false") === codeMode && (codeMode === "false" ? (r.code_api ?? null) === null : codeApi ? r.code_api === codeApi : r.code_api_auto === true || (r.code_api_auto === undefined && r.code_api === "high"))
	&& (r.code_oracle ?? null) === (codeMode === "false" ? null : codeOracle || null)
	// Results written before the code budget was recorded ran the default one.
	&& (codeMode === "false" || (r.code_budget_flags ?? "timeout=+max_calls=+max_move=+helpers=false") === process.env.CODE_BUDGET_FLAGS);
process.exit(same ? 0 : 2);
' "$1/result.json" "${config[@]}" 2>/dev/null
}

cells=()
# run_cell <task> <seed>: one episode (kept when a valid result of this configuration exists); prints its status.
run_cell() {
	local task=$1 seed=$2 dir="$out/$1_s$2"
	valid "$dir"
	case $? in
	0) node -p 'require(process.argv[1]).status' "$dir/result.json" && return ;;
	2) echo "$dir holds a result of another model, thinking level, --max-turns, --time-limit, units mode, code mode, vdm, --eval-seed (or an older result without them), fallback, --privileged, --anchor-image, --approval, --max-tool-calls or --max-tokens; use another out dir" >&2 && exit 1 ;;
	esac
	rm -rf "$dir" && mkdir -p "$dir"
	echo "== $task seed $seed" >&2
	# The prompt precedes the user's args: a bare boolean flag at their end would take it as its value.
	${backstop[@]+"${backstop[@]}"} $PI -p --session-dir "$dir" -e "$here" --task "$task" --seed "$seed" "Solve the task." "$@" \
		</dev/null >"$dir/stdout.log" 2>"$dir/stderr.log"
	record "$dir" "$?" >&2
	node -p 'require(process.argv[1]).status' "$dir/result.json"
}
for task in ${tasks//,/ }; do
	queue=($(expand "$seeds"))
	next=$(printf '%s\n' "${queue[@]}" | sort -n | tail -1)
	for ((q = 0; q < ${#queue[@]}; q++)); do
		seed=${queue[q]}
		# run_cell runs in a subshell: its refusal of another configuration's out dir must stop the script.
		st=$(run_cell "$task" "$seed" "$@") || exit 1
		# A replacement layout beyond the task's last one: the task has no more layouts to draw.
		if [ "$st" = env_error ] && [ "$q" -ge "$(expand "$seeds" | wc -l)" ] &&
			grep -q "has eval layouts 0\.\." "$out/${task}_s$seed/stderr.log" 2>/dev/null; then
			rm -rf "${out:?}/${task}_s$seed"
			echo "== $task: no layout after $((seed - 1)) to replace an unstable one" >&2
			break
		fi
		cells+=("${task}_s${seed}")
		if [ "$st" = unstable ]; then
			next=$((next + 1))
			queue+=("$next")
			echo "== $task seed $seed is unstable (RoboDojo discards it); layout $next replaces it" >&2
		fi
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
const configs = new Set(rows.filter((r) => r.status === "success" || r.status === "failure" || r.status === "unstable")
	.map((r) => `${r.model}/${r.thinking}/turns=${r.max_turns}/limit=${r.time_limit}/units=${r.units}${r.units_wrist_view === false ? `/no-wrist:${(r.units_plugins ?? []).join("+")}` : ""}${r.stateless ? "/stateless" : ""}${r.anchor_image ? "/anchor" : ""}${(r.approval ?? "off") !== "off" ? `/approval=${r.approval}` : ""}${r.max_tool_calls ? `/tool_calls=${r.max_tool_calls}` : ""}${r.max_tokens ? `/tokens=${r.max_tokens}` : ""}${r.vdm ? `/vdm=${r.vdm_model ?? "default"}${r.vdm_wrist ? "+wrist" : ""}` : ""}${r.vdm_video ? `/vdm_video=${r.vdm_video}:${r.vdm_model ?? "default"}` : ""}${r.privileged ? "/privileged" : ""}${r.fallback_model ? `/fallback=${r.fallback_model}:${r.fallback_after}:${r.fallback_retry_primary}` : ""}/eval_seed=${r.eval_seed}${r.code && r.code !== "false" ? `/code=${r.code}:${r.code_api}${r.code_oracle ? `:oracle=${r.code_oracle}` : ""}` : ""}`));
if (configs.size > 1) {
	console.log(`refusing to summarize: ${out} mixes configurations ${[...configs].join(", ")}`);
	process.exit(1);
}
const pm = rows.reduce((a, r) => r.planner_models ? { primary: a.primary + (r.planner_models.primary ?? 0), fallback: a.fallback + (r.planner_models.fallback ?? 0), rows: a.rows + 1 } : a, { primary: 0, fallback: 0, rows: 0 });
const planned = pm.rows ? `, planner_models primary=${pm.primary} fallback=${pm.fallback}` : "";
const n = (s) => rows.filter((r) => r.status === s).length;
const scored = n("success") + n("failure");
const lies = rows.filter((r) => r.status === "failure" && r.claimed === "success").length;
const rate = scored ? ((100 * n("success")) / scored).toFixed(1) : "-";
const valid = rows.filter((r) => r.status === "success" || r.status === "failure");
const score = valid.length ? (valid.reduce((a, r) => a + (r.score ?? 0), 0) / valid.length).toFixed(3) : "-";
const invalid = rows.length - scored - n("unstable");
const tasks = [...new Set(cells.map((c) => c.replace(/_s\d+$/, "")))];
const per = tasks.length > 1 ? ` [${tasks.map((e) => {
	const mine = rows.filter((r, i) => cells[i].replace(/_s\d+$/, "") === e);
	return `${e} ${mine.filter((r) => r.status === "success").length}/${mine.filter((r) => r.status === "success" || r.status === "failure").length}`;
}).join(", ")}]` : "";
console.log(`${[...configs][0] ?? "-"}: success ${n("success")}/${scored} (${rate}%), mean score ${score}${per}, claimed-but-failed ${lies}, unstable ${n("unstable")} (replaced), invalid ${invalid} (env_error ${n("env_error")}, planner_error ${n("planner_error")}, timeout ${n("timeout")}, missing ${n("missing")}, duplicate ${n("duplicate_result")}) of ${rows.length}${planned}`);
if (invalid) process.exit(1);
' "$out" "${cells[@]}"
