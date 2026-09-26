#!/usr/bin/env bash
# Run Metaworld episodes with pi in print mode and report Metaworld-judged success (the env's `success`).
#   eval.sh <out-dir> <tasks> <seeds> [pi args...]      (tasks "-" = reach-v3)
#   eval.sh runs/mw - 0-9 --model <provider/model> --thinking low
#   eval.sh runs/mw pick-place-v3,button-press-v3 0-9 --model <provider/model> --thinking low
#   eval.sh runs/mw-units reach-v3 0-4,10-14 --units=true --model <provider/model> --thinking low
#
# Each episode runs in <out>/<task>_s<seed>/ and ends with a result.json taken from the session's
# `robot_result` entry. An episode is valid when the environment produced a result and the planner
# did not fail (`env_error`, `planner_error` and a missing result are invalid), whatever the
# outcome. Rerunning retries exactly the invalid episodes; valid ones are kept. Each result records
# the model, thinking level, --max-turns, --time-limit, the units mode (--units, --stateless) and
# visual differencing (--vdm, --vdm-model, --vdm-wrist), and the summary covers only the requested
# cells and refuses to mix configurations.
# A --privileged run (simulator ground truth) is recorded as such and never shares an out dir with one without.
# The fallback planner (--fallback-model, --fallback-after, --fallback-retry-primary; src/fallback.ts) is part of the
# configuration too, and the summary totals the turns each planner model planned (planner_models).
set -uo pipefail
out=$1 tasks=$2 seeds=$3
[ "$tasks" = - ] && tasks=reach-v3
shift 3
here=$(cd "$(dirname "$0")" && pwd)
PI=${PI:-pi}
expand() { for part in ${1//,/ }; do seq "${part%-*}" "${part#*-}"; done; }
model="" thinking="" turns=0 limit=${TIME_LIMIT:-1800} limited="" units=false stateless=false
anchor=false
vdm=false vdm_model="" vdm_wrist=false
privileged=false
fallback_model="" fallback_after=2 fallback_retry=0
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
	--vdm-model) vdm_model=${args[i + 1]:-} ;;
	--vdm-model=*) vdm_model=${args[i]#*=} ;;
	--fallback-model) fallback_model=${args[i + 1]:-} ;;
	--fallback-model=*) fallback_model=${args[i]#*=} ;;
	--fallback-after) fallback_after=${args[i + 1]:-2} ;;
	--fallback-after=*) fallback_after=${args[i]#*=} ;;
	--fallback-retry-primary) fallback_retry=${args[i + 1]:-0} ;;
	--fallback-retry-primary=*) fallback_retry=${args[i]#*=} ;;
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
# --time-limit (default $TIME_LIMIT, 1800 s; 0 = none) ends the planner gracefully, as a failure;
# `timeout` is only the backstop for a hung process, and a killed episode is invalid.
[ -n "$limited" ] || set -- "$@" --time-limit "$limit"
backstop=()
[ "$limit" -gt 0 ] && command -v timeout >/dev/null && backstop=(timeout -k 30 $((limit + 900)))
mkdir -p "$out"
config=("$model" "$thinking" "$turns" "$limit" "$units" "$stateless" "$privileged" "$anchor" "$vdm" "$vdm_model" "$vdm_wrist" "$fallback_model" "$fallback_after" "$fallback_retry")

record() { # <dir> <exit code>: write result.json from the episode's session
	node --input-type=module -e '
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
const [dir, code, model, thinking, turns, limit, units, stateless, privileged, anchor, vdm, vdmModel, vdmWrist, fallbackModel, fallbackAfter, fallbackRetry] = process.argv.slice(1);
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
const result = { ...(last ?? {}), status, exit_code: Number(code), model: model || null, thinking: thinking || null,
	max_turns: Number(turns), time_limit: Number(limit), units, anchor_image: anchor === "true",
	vdm: vdm === "true", vdm_model: vdmModel || null, vdm_wrist: vdmWrist === "true", stateless: stateless === "true",
	privileged: privileged === "true",
	fallback_model: fallbackModel || null, fallback_after: fallbackModel ? Number(fallbackAfter) : null, fallback_retry_primary: fallbackModel ? Number(fallbackRetry) : null };
writeFileSync(`${dir}/result.json`, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ status, success: result.success, claimed: result.claimed, env_steps: result.env_steps }));
' "$1" "$2" "${config[@]}"
}

valid() { # <dir>: 0 = a valid result of this configuration, 2 = a valid result of another one, 1 = none
	node -e '
const [path, model, thinking, turns, limit, units, stateless, privileged, anchor, vdm, vdmModel, vdmWrist, fallbackModel, fallbackAfter, fallbackRetry] = process.argv.slice(1);
const r = JSON.parse(require("fs").readFileSync(path, "utf8"));
if (r.status !== "success" && r.status !== "failure") process.exit(1);
const same = r.model === (model || null) && r.thinking === (thinking || null) && r.max_turns === Number(turns)
	&& r.time_limit === Number(limit) && r.units === units && r.stateless === (stateless === "true")
	&& (r.privileged ?? false) === (privileged === "true")
	// Results written before --anchor-image existed ran without it.
	&& (r.anchor_image ?? false) === (anchor === "true")
	// Results written before --vdm existed ran without it.
	&& (r.vdm ?? false) === (vdm === "true") && (r.vdm_model ?? null) === (vdmModel || null)
	&& (r.vdm_wrist ?? false) === (vdmWrist === "true")
	// Results written before --fallback-model existed ran without a fallback planner.
	&& (r.fallback_model ?? null) === (fallbackModel || null) && (r.fallback_after ?? null) === (fallbackModel ? Number(fallbackAfter) : null)
	&& (r.fallback_retry_primary ?? null) === (fallbackModel ? Number(fallbackRetry) : null);
process.exit(same ? 0 : 2);
' "$1/result.json" "${config[@]}" 2>/dev/null
}

cells=()
for task in ${tasks//,/ }; do
	for seed in $(expand "$seeds"); do
		cells+=("${task}_s${seed}")
		dir="$out/${task}_s${seed}"
		valid "$dir"
		case $? in
		0) continue ;;
		2) echo "$dir holds a result of another model, thinking level, --max-turns, --time-limit, units mode, vdm, fallback, --privileged or --anchor-image; use another out dir" >&2 && exit 1 ;;
		esac
		rm -rf "$dir" && mkdir -p "$dir"
		echo "== $task seed $seed"
		# The prompt precedes the user's args: a bare boolean flag at their end would take it as its value.
		${backstop[@]+"${backstop[@]}"} $PI -p --session-dir "$dir" -e "$here" --task "$task" --seed "$seed" "Solve the task." "$@" \
			</dev/null >"$dir/stdout.log" 2>"$dir/stderr.log"
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
const configs = new Set(rows.filter((r) => r.status === "success" || r.status === "failure")
	.map((r) => `${r.model}/${r.thinking}/turns=${r.max_turns}/limit=${r.time_limit}/units=${r.units}${r.stateless ? "/stateless" : ""}${r.anchor_image ? "/anchor" : ""}${r.vdm ? `/vdm=${r.vdm_model ?? "default"}${r.vdm_wrist ? "+wrist" : ""}` : ""}${r.privileged ? "/privileged" : ""}${r.fallback_model ? `/fallback=${r.fallback_model}:${r.fallback_after}:${r.fallback_retry_primary}` : ""}`));
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
const invalid = rows.length - scored;
const tasks = [...new Set(cells.map((c) => c.replace(/_s\d+$/, "")))];
const per = tasks.length > 1 ? ` [${tasks.map((e) => {
	const mine = rows.filter((r, i) => cells[i].replace(/_s\d+$/, "") === e);
	return `${e} ${mine.filter((r) => r.status === "success").length}/${mine.filter((r) => r.status === "success" || r.status === "failure").length}`;
}).join(", ")}]` : "";
console.log(`${[...configs][0] ?? "-"}: success ${n("success")}/${scored} (${rate}%)${per}, claimed-but-failed ${lies}, invalid ${invalid} (env_error ${n("env_error")}, planner_error ${n("planner_error")}, timeout ${n("timeout")}, missing ${n("missing")}, duplicate ${n("duplicate_result")}) of ${rows.length}${planned}`);
if (invalid) process.exit(1);
' "$out" "${cells[@]}"
