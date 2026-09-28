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
model="" thinking="" turns=0 limit=${TIME_LIMIT:-1800} limited="" units=false stateless=false
anchor=false
approval=standard max_tool_calls=0 max_tokens=0
vdm=false vdm_model="" vdm_wrist=false vdm_video=false vdm_video_frames=8
# The robot's default (src/robots/libero/index.ts --unit-tol).
unit_tol=0.004
privileged=false
fallback_model="" fallback_after=2 fallback_retry=0
code=false code_api=high code_oracle=""
libero_prompt=rpent
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
	--units-plugins) units_plugins=${args[i + 1]-} ;;
	--units-plugins=*) units_plugins=${args[i]#*=} ;;
	# The units' experiment knobs (stage cap, action ablation, point self-check) are part of the units mode too.
	--units-stage-steps | --units-ablation | --units-point-verify) units_opts+="+${args[i]#--units-}=${args[i + 1]-}" ;;
	--units-stage-steps=* | --units-ablation=* | --units-point-verify=*) units_opts+="+${args[i]#--units-}" ;;
	# --code / --code-api (run_code, packages/embodied/src/modes/code) are string flags like --units.
	--code) [[ ${args[i + 1]:---} == --* ]] && code=true || code=${args[i + 1]} ;;
	--code=*) code=${args[i]#*=} ;;
	--code-api) code_api=${args[i + 1]:-high} ;;
	--code-api=*) code_api=${args[i]#*=} ;;
	--libero-prompt) libero_prompt=${args[i + 1]:-} ;;
	--libero-prompt=*) libero_prompt=${args[i]#*=} ;;
	--code-oracle) code_oracle=${args[i + 1]:-} ;;
	--code-oracle=*) code_oracle=${args[i]#*=} ;;
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
	--vdm | --vdm-wrist | --vdm-video) case ${args[i + 1]:-} in "" | -* | @* | true) case ${args[i]} in --vdm) vdm=true ;; --vdm-wrist) vdm_wrist=true ;; *) vdm_video=true ;; esac ;; *)
		echo "${args[i]} takes no value: pi would turn it on and swallow '${args[i + 1]}'" >&2 && exit 2 ;;
	esac ;;
	--vdm=true) vdm=true ;;
	--vdm-wrist=true) vdm_wrist=true ;;
	--vdm-video=true) vdm_video=true ;;
	--vdm-video-frames) vdm_video_frames=${args[i + 1]:-8} ;;
	--vdm-video-frames=*) vdm_video_frames=${args[i]#*=} ;;
	--vdm=* | --vdm-wrist=* | --vdm-video=*)
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
	--unit-tol) unit_tol=${args[i + 1]:-} ;;
	--unit-tol=*) unit_tol=${args[i]#*=} ;;
	# --privileged (simulator ground truth, ground_truth_poses) is a boolean like --stateless.
	--privileged) case ${args[i + 1]:-} in "" | -* | @* | true) privileged=true ;; *)
		echo "--privileged takes no value: pi would turn it on and swallow '${args[i + 1]}'" >&2 && exit 2 ;;
	esac ;;
	--privileged=true) privileged=true ;;
	--privileged=*)
		echo "${args[i]}: pi ignores a boolean flag's value and would turn it on; omit --privileged for a run without ground truth" >&2
		exit 2
		;;
	# --approval (motion approval, src/capabilities/operator.ts) and the --max-tool-calls / --max-tokens budgets (src/robot.ts).
	--approval) approval=${args[i + 1]:-standard} ;;
	--approval=*) approval=${args[i]#*=} ;;
	--max-tool-calls) max_tool_calls=${args[i + 1]:-0} ;;
	--max-tool-calls=*) max_tool_calls=${args[i]#*=} ;;
	--max-tokens) max_tokens=${args[i + 1]:-0} ;;
	--max-tokens=*) max_tokens=${args[i]#*=} ;;
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
# --units-plugins is part of the units mode: a result with other plugins is another configuration.
[ "$units" != false ] && [ -n "${units_plugins+x}" ] && units="$units+plugins=$units_plugins"
[ "$units" != false ] && units="$units${units_opts-}"
[ "$code" = pure ] && code=true
# Without --vdm or --vdm-video no VDM call runs, so its model is not part of the configuration;
# --vdm-video is recorded as its frame count.
[ "$vdm" = true ] || [ "$vdm_video" = true ] || vdm_model=""
[ "$vdm_video" = true ] && vdm_video=$vdm_video_frames || vdm_video=""
# --time-limit (default $TIME_LIMIT, 1800 s; 0 = none) ends the planner gracefully, as a failure;
# `timeout` is only the backstop for a hung process, and a killed episode is invalid.
[ -n "$limited" ] || set -- "$@" --time-limit "$limit"
backstop=()
[ "$limit" -gt 0 ] && command -v timeout >/dev/null && backstop=(timeout -k 30 $((limit + 900)))
mkdir -p "$out"
case $libero_prompt in rpent | compact) ;; *) echo "--libero-prompt must be rpent or compact, not '$libero_prompt'" >&2 && exit 2 ;; esac
config=("$model" "$thinking" "$turns" "$limit" "$units" "$stateless" "$vdm" "$vdm_model" "$vdm_wrist" "$privileged" "$anchor" "$unit_tol" "$code" "$code_api" "$fallback_model" "$fallback_after" "$fallback_retry" "$approval" "$max_tool_calls" "$max_tokens" "$libero_prompt" "$code_oracle" "$vdm_video")

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
	code: codeMode, code_api: codeMode === "false" ? null : codeApi, code_oracle: codeMode === "false" ? null : codeOracle || null,
	fallback_model: fallbackModel || null, fallback_after: fallbackModel ? Number(fallbackAfter) : null, fallback_retry_primary: fallbackModel ? Number(fallbackRetry) : null,
	libero_prompt: liberoPrompt };
writeFileSync(`${dir}/result.json`, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ status, terminated: result.terminated, claimed: result.claimed, env_steps: result.env_steps }));
' "$1" "$2" "${config[@]}"
}

valid() { # <dir>: 0 = a valid result of this configuration, 2 = a valid result of another one, 1 = none
	node -e '
const [path, model, thinking, turns, limit, units, stateless, vdm, vdmModel, vdmWrist, privileged, anchor, unitTol, codeMode, codeApi, fallbackModel, fallbackAfter, fallbackRetry, approval, maxToolCalls, maxTokens, liberoPrompt, codeOracle, vdmVideo] = process.argv.slice(1);
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
	// Results written before --approval, --max-tool-calls and --max-tokens existed ran with standard approval and neither budget.
	&& (r.approval ?? "standard") === approval && (r.max_tool_calls ?? 0) === Number(maxToolCalls) && (r.max_tokens ?? 0) === Number(maxTokens)
	// Results written before --unit-tol was recorded ran with the default tolerance.
	&& (r.unit_tol ?? 0.004) === Number(unitTol)
	// Results written before --code existed ran without it.
	&& (r.code ?? "false") === codeMode && (r.code_api ?? null) === (codeMode === "false" ? null : codeApi)
	// Results written before --code-oracle existed ran the model.
	&& (r.code_oracle ?? null) === (codeMode === "false" ? null : codeOracle || null)
	// Results written before --fallback-model existed ran without a fallback planner.
	&& (r.fallback_model ?? null) === (fallbackModel || null) && (r.fallback_after ?? null) === (fallbackModel ? Number(fallbackAfter) : null)
	&& (r.fallback_retry_primary ?? null) === (fallbackModel ? Number(fallbackRetry) : null)
	// Results written before --libero-prompt existed ran the compact prompt.
	&& (r.libero_prompt ?? "compact") === liberoPrompt;
process.exit(same ? 0 : 2);
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
		2) echo "$dir holds a result of another model, thinking level, --max-turns, --time-limit, units mode, --unit-tol, code mode, vdm, fallback, --libero-prompt, --privileged, --anchor-image, --approval, --max-tool-calls or --max-tokens; use another out dir" >&2 && exit 1 ;;
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
	.map((r) => `${r.model}/${r.thinking}/turns=${r.max_turns}/limit=${r.time_limit}/units=${r.units}${r.units_wrist_view === false ? `/no-wrist:${(r.units_plugins ?? []).join("+")}` : ""}${r.stateless ? "/stateless" : ""}${r.anchor_image ? "/anchor" : ""}${(r.approval ?? "standard") !== "standard" ? `/approval=${r.approval}` : ""}${r.max_tool_calls ? `/tool_calls=${r.max_tool_calls}` : ""}${r.max_tokens ? `/tokens=${r.max_tokens}` : ""}/unit_tol=${r.unit_tol ?? 0.004}${r.vdm ? `/vdm=${r.vdm_model ?? "default"}${r.vdm_wrist ? "+wrist" : ""}` : ""}${r.vdm_video ? `/vdm_video=${r.vdm_video}:${r.vdm_model ?? "default"}` : ""}${r.privileged ? "/privileged" : ""}${r.fallback_model ? `/fallback=${r.fallback_model}:${r.fallback_after}:${r.fallback_retry_primary}` : ""}${r.code && r.code !== "false" ? `/code=${r.code}:${r.code_api}${r.code_oracle ? `:oracle=${r.code_oracle}` : ""}` : ""}/prompt=${r.libero_prompt ?? "compact"}`));
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
