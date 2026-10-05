#!/usr/bin/env bash
# Run LIBERO episodes with pi in print mode and report LIBERO-judged success.
#   eval.sh <out-dir> <suite> <tasks> <seeds> [pi args...]
#   eval.sh runs/l10 libero_10_task 0-9 1-10 --model <provider/model> --thinking low
#
# Each episode runs in its own session directory and ends with a result.json taken from the
# session's `robot_result` entry. An episode is valid when the environment produced a result and
# the planner did not fail (`env_error`, `planner_error` and a missing result are invalid),
# whatever the outcome. Rerunning retries exactly the invalid episodes; valid ones are kept.
# Each result records the model, thinking level, --max-turns, --time-limit, the units mode
# (--units, --stateless, --unit-tol), code mode (--code, --code-api: high, low or low-noexamples = CaP-X's S2-S4,
# --code-oracle: a reference program run instead of the model, ./oracle) and visual differencing (--vdm,
# --vdm-model, --vdm-wrist, --vdm-video and its --vdm-video-frames; the model only with --vdm or --vdm-video), and the summary covers only the requested
# cells and refuses to mix configurations.
# A --privileged run (simulator ground truth) is recorded as such and never shares an out dir with one without.
# The fallback planner (--fallback-model, --fallback-after, --fallback-retry-primary; src/planner/fallback.ts) is part of the
# configuration too, and the summary totals the turns each planner model planned (planner_models).
# So is the system prompt (--libero-prompt rpent|compact, default rpent; results from before the flag ran compact).
set -uo pipefail
out=$1 suite=$2 tasks=$3 seeds=$4
shift 4
here=$(cd "$(dirname "$0")" && pwd)
PI=${PI:-pi}
expand() { for part in ${1//,/ }; do seq "${part%-*}" "${part#*-}"; done; }
# Common flags have one implementation; only robot-specific options live here.
source "$here/../../scripts/eval-options.sh"
eval_options_defaults
unit_tol=0.004
libero_prompt=rpent
eval_robot_option() {
	case $1 in
	--libero-prompt) libero_prompt=${2:-} ;;
	--libero-prompt=*) libero_prompt=${1#*=} ;;
	--unit-tol) unit_tol=${2:-} ;;
	--unit-tol=*) unit_tol=${1#*=} ;;
	esac
}
eval_parse_options "$@"

# Without --vdm or --vdm-video no VDM call runs, so its model is not part of the configuration;
# --vdm-video is recorded as its frame count.
[ "$vdm" = true ] || [ "$vdm_video" = true ] || vdm_model=""
eval_normalize_options
# --time-limit (default $TIME_LIMIT, 1800 s; 0 = none) ends the planner gracefully, as a failure;
# `timeout` is only the backstop for a hung process, and a killed episode is invalid.
[ -n "$limited" ] || set -- "$@" --time-limit "$limit"
backstop=()
[ "$limit" -gt 0 ] && command -v timeout >/dev/null && backstop=(timeout -k 30 $((limit + 900)))
mkdir -p "$out"
case $libero_prompt in rpent | compact) ;; *) echo "--libero-prompt must be rpent or compact, not '$libero_prompt'" >&2 && exit 2 ;; esac
config=("$model" "$thinking" "$turns" "$limit" "$units" "$stateless" "$vdm" "$vdm_model" "$vdm_wrist" "$privileged" "$anchor" "$unit_tol" "$code" "$code_api" "$fallback_model" "$fallback_after" "$fallback_retry" "$approval" "$max_tool_calls" "$max_tokens" "$libero_prompt" "$code_oracle" "$vdm_video" "$extras")

record() { # <dir> <exit code>: write result.json from the episode's session
	node --input-type=module -e '
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
const [dir, code, model, thinking, turns, limit, units, stateless, vdm, vdmModel, vdmWrist, privileged, anchor, unitTol, codeMode, codeApi, fallbackModel, fallbackAfter, fallbackRetry, approval, maxToolCalls, maxTokens, liberoPrompt, codeOracle, vdmVideo] = process.argv.slice(1);
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
			if (line.startsWith("[libero] {")) results.push(JSON.parse(line.slice("[libero] ".length)));
	} catch {}
const last = results.length === 1 ? results[0] : undefined;
const status = Number(code) === 124 ? "timeout" : results.length > 1 ? "duplicate_result"
	: !last ? (Number(code) ? "env_error" : "missing")
	: last.env_error ? "env_error" : last.planner_error ? "planner_error" : last.terminated ? "success" : "failure";
const result = { ...(last ?? {}), status, exit_code: Number(code), model: model || null, thinking: thinking || null,
	max_turns: Number(turns), time_limit: Number(limit), units, anchor_image: anchor === "true", approval, max_tool_calls: Number(maxToolCalls), max_tokens: Number(maxTokens), stateless: stateless === "true",
	vdm: vdm === "true", vdm_model: vdmModel || null, vdm_wrist: vdmWrist === "true", vdm_video: vdmVideo ? Number(vdmVideo) : null,
	privileged: privileged === "true", unit_tol: Number(unitTol),
	code: codeMode, code_api: codeMode === "false" ? null : codeApi || last?.code_api || null, code_api_auto: codeMode === "false" ? null : !codeApi, code_oracle: codeMode === "false" ? null : codeOracle || null,
	fallback_model: fallbackModel || null, fallback_after: fallbackModel ? Number(fallbackAfter) : null, fallback_retry_primary: fallbackModel ? Number(fallbackRetry) : null,
	libero_prompt: liberoPrompt, ft_flags: process.env.FT_FLAGS || null };
writeFileSync(`${dir}/result.json`, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ status, terminated: result.terminated, claimed: result.claimed, env_steps: result.env_steps }));
' "$1" "$2" "${config[@]}"
}

valid() { # <dir>: 0 = a valid result of this configuration, 2 = a valid result of another one, 1 = none
	node -e '
const [path, model, thinking, turns, limit, units, stateless, vdm, vdmModel, vdmWrist, privileged, anchor, unitTol, codeMode, codeApi, fallbackModel, fallbackAfter, fallbackRetry, approval, maxToolCalls, maxTokens, liberoPrompt, codeOracle, vdmVideo, extras] = process.argv.slice(1);
const r = JSON.parse(require("fs").readFileSync(path, "utf8"));
if (r.status !== "success" && r.status !== "failure") process.exit(1);
const same = r.model === (model || null) && r.thinking === (thinking || null) && r.max_turns === Number(turns)
	&& r.time_limit === Number(limit) && r.units === units && r.stateless === (stateless === "true")
	// Results written before --vdm existed ran without it.
	&& (r.vdm ?? false) === (vdm === "true") && (r.vdm_model ?? null) === (vdmModel || null)
	&& (r.vdm_wrist ?? false) === (vdmWrist === "true") && (r.vdm_video ?? null) === (vdmVideo ? Number(vdmVideo) : null)
	&& (r.privileged ?? false) === (privileged === "true")
	// Results written before --anchor-image existed ran without it.
	&& (r.anchor_image ?? false) === (anchor === "true")
	// Results without an approval field ran with it off. "standard" results written before it confirmed high-risk motions ran unconfirmed: they are refused (another configuration).
	&& (r.approval ?? "off") === approval && (r.max_tool_calls ?? 0) === Number(maxToolCalls) && (r.max_tokens ?? 0) === Number(maxTokens)
	// Results written before --unit-tol was recorded ran with the default tolerance.
	&& (r.unit_tol ?? 0.004) === Number(unitTol)
	// Results written before --code existed ran without it.
	&& (r.code ?? "false") === codeMode && (codeMode === "false" ? (r.code_api ?? null) === null : codeApi ? r.code_api === codeApi : r.code_api_auto === true || (r.code_api_auto === undefined && r.code_api === "high"))
	// Results written before --code-oracle existed ran the model.
	&& (r.code_oracle ?? null) === (codeMode === "false" ? null : codeOracle || null)
	// The --ft-* flags a finetuned/* run was given ("default" with none). A finetuned/* result without them was
	// written before they were recorded (the LIBERO wrist frame changed meanwhile): another configuration.
	&& (!(model || "").startsWith("finetuned/") || r.ft_flags === process.env.FT_FLAGS)
	// Results written before the code budget was recorded ran the default one.
	&& (codeMode === "false" || (r.code_budget_flags ?? "timeout=+max_calls=+max_move=+helpers=false") === process.env.CODE_BUDGET_FLAGS)
	// Results written before --fallback-model existed ran without a fallback planner.
	&& (r.fallback_model ?? null) === (fallbackModel || null) && (r.fallback_after ?? null) === (fallbackModel ? Number(fallbackAfter) : null)
	&& (r.fallback_retry_primary ?? null) === (fallbackModel ? Number(fallbackRetry) : null)
	// Results written before --libero-prompt existed ran the compact prompt.
	&& (r.libero_prompt ?? "compact") === liberoPrompt;
process.exit(same && [...(r.extras ?? [])].sort().join(",") === (extras ?? "") ? 0 : 2);
' "$1/result.json" "${config[@]}" 2>/dev/null
}

cells=()
for task in $(expand "$tasks"); do
	for seed in $(expand "$seeds"); do
		cells+=("${suite}_t${task}_s${seed}")
		dir="$out/${suite}_t${task}_s${seed}"
		valid "$dir"
		case $? in
		0) continue ;;
		2) echo "$dir holds a result of another model, thinking level, --max-turns, --time-limit, units mode, --unit-tol, code mode, vdm, fallback, --libero-prompt, --ft-* flags, --privileged, --anchor-image, --approval, --max-tool-calls or --max-tokens; use another out dir" >&2 && exit 1 ;;
		esac
		rm -rf "$dir" && mkdir -p "$dir"
		echo "== $suite task $task seed $seed"
		# The prompt precedes the user's args: a bare boolean flag at their end would take it as its value.
		${backstop[@]+"${backstop[@]}"} $PI -p --session-dir "$dir" -e "$here" --suite "$suite" --task "$task" --seed "$seed" "Solve the task." "$@" \
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
	.map((r) => `${r.model}/${r.thinking}/turns=${r.max_turns}/limit=${r.time_limit}/units=${r.units}${r.units_wrist_view === false ? `/no-wrist:${(r.units_plugins ?? []).join("+")}` : ""}${r.stateless ? "/stateless" : ""}${r.anchor_image ? "/anchor" : ""}${(r.approval ?? "off") !== "off" ? `/approval=${r.approval}` : ""}${r.max_tool_calls ? `/tool_calls=${r.max_tool_calls}` : ""}${r.max_tokens ? `/tokens=${r.max_tokens}` : ""}/unit_tol=${r.unit_tol ?? 0.004}${r.vdm ? `/vdm=${r.vdm_model ?? "default"}${r.vdm_wrist ? "+wrist" : ""}` : ""}${r.vdm_video ? `/vdm_video=${r.vdm_video}:${r.vdm_model ?? "default"}` : ""}${r.privileged ? "/privileged" : ""}${r.fallback_model ? `/fallback=${r.fallback_model}:${r.fallback_after}:${r.fallback_retry_primary}` : ""}${r.code && r.code !== "false" ? `/code=${r.code}:${r.code_api}${r.code_oracle ? `:oracle=${r.code_oracle}` : ""}` : ""}/prompt=${r.libero_prompt ?? "compact"}${r.ft_flags ? `/ft=${r.ft_flags}` : ""}`));
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
console.log(`${[...configs][0] ?? "-"}: success ${n("success")}/${scored} (${rate}%), claimed-but-failed ${lies}, invalid ${invalid} (env_error ${n("env_error")}, planner_error ${n("planner_error")}, timeout ${n("timeout")}, missing ${n("missing")}, duplicate ${n("duplicate_result")}) of ${rows.length}${planned}`);
if (invalid) process.exit(1);
' "$out" "${cells[@]}"
