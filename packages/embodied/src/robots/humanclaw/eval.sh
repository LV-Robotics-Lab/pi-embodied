#!/usr/bin/env bash
# Run HumanClawBench episodes with pi in print mode and report the paper metrics.
#   eval.sh <out-dir> [--episodes one|val100|fullval|<list.json>] [--mode paper|pi] [--metrics] [--video] [--smoke] [pi args...]
#   eval.sh runs/hc-paper --episodes one --mode paper --metrics --model humanclaw-psv/selfhost/muse-glimmer-30b
#   eval.sh runs/hc-pi --episodes val100 --mode pi --metrics --model selfhost/muse-glimmer-30b   (GPU: cuda_device or PI_EMBODIED_CUDA_DEVICE)
#
# One pi process (and one env server: Habitat and the motion model are rebuilt per episode, as the
# release's dispatcher runs one process per episode) per episode, in <out>/<scene>_ep<id>_<category>/,
# ending with a result.json from the session's `robot_result` plus the configuration: mode, preset
# (`humanclaw`, `humanclaw+privileged`), model, --units-verify / --vdm / --stateless /
# --humanclaw-proprioception, --metrics, --video, paper mode's request parameters
# (--humanclaw-reasoning, --humanclaw-max-tokens) and, when set, pi mode's planner flags (--thinking,
# --max-turns, --time-limit, --units-plugins and the units knobs, --anchor-image, --vdm-model,
# --fallback-*, --approval, --max-tool-calls, --max-tokens, --humanclaw-collision-feedback) and a
# --smoke step cap (--humanclaw-max-steps). An episode is valid when the environment produced a result
# and the planner did not fail; it is scored (success = NavSR@20cm, else failure) only with --metrics,
# which is where the benchmark's verdict is computed: without it every valid episode is `unscored`
# (kept, not counted, exit status 1), never a failure. Rerunning retries only invalid episodes and
# refuses an out dir holding another configuration.
# With --metrics the summary is HumanCLAW's own aggregate_metric_files() over every metrics.json
# (summary.json, and metrics_summary.json beside it), run with $HUMANCLAW_PYTHON (the humanclaw venv).
# Heavy: run serially with LOCK set (e.g. /root/autodl-tmp/locks/gpu1.lock), each episode takes it
# with flock and releases it between episodes, so other GPU jobs interleave. scripts/eval-parallel.sh
# takes it once for the whole run and starts this script with LOCK unset (an episode taking it here
# would wait on that run's own lock).
set -uo pipefail
out=${1:?usage: eval.sh <out-dir> [--episodes ...] [--mode paper|pi] [--metrics] [--video] [--smoke] [pi args...]}
shift
here=$(cd "$(dirname "$0")" && pwd)
# Renamed flags and the ones the deployment config replaced stop here, before any cell runs.
. "$here/../../scripts/old-flags.sh"
old_flags "$@" || exit 2
PI=${PI:-pi}
PY=${HUMANCLAW_PYTHON:-${PI_EMBODIED_PYTHON:-python}}
# The shared pi flags (model, thinking, turns, time limit, units plug-ins, stateless, anchor, vdm,
# privileged, fallback, approval, budgets) are parsed by scripts/eval-options.sh; only HumanCLAW's own
# flags live here. Every flag that changes a request or an episode is part of the configuration key.
source "$here/../../scripts/eval-options.sh"
eval_options_defaults
units=both # eval.sh passes --units=both itself; --units-plugins / --units-stage-steps extend it
episodes=one mode=paper metrics=false video=false smoke=false
verify=false proprioception=false collision=false reasoning="" request_tokens=4096 max_steps=""
eval_robot_option() {
	case $1 in
	--units-verify) verify=true ;;
	--humanclaw-proprioception | --humanclaw-proprioception=true) proprioception=true ;;
	--humanclaw-proprioception=*) echo "omit --humanclaw-proprioception to disable it" >&2; exit 2 ;;
	--humanclaw-collision-feedback | --humanclaw-collision-feedback=true) collision=true ;;
	--humanclaw-collision-feedback=*) echo "omit --humanclaw-collision-feedback to disable it" >&2; exit 2 ;;
	# Paper mode's request parameters (provider.ts): the base model's reasoning level and max_tokens.
	--humanclaw-reasoning) reasoning=${2:-} ;;
	--humanclaw-reasoning=*) reasoning=${1#*=} ;;
	--humanclaw-max-tokens) request_tokens=${2:-4096} ;;
	--humanclaw-max-tokens=*) request_tokens=${1#*=} ;;
	# pi sets a boolean flag to true whatever value it is given: the JSON response format cannot be turned off here.
	--humanclaw-json-format*) echo "$1: the JSON response format is always requested (a boolean flag cannot be turned off from the command line); omit it" >&2; exit 2 ;;
	# A step cap truncates the episode: a smoke, never a scored run (--smoke says so and keys it).
	--humanclaw-max-steps) max_steps=${2:-} ;;
	--humanclaw-max-steps=*) max_steps=${1#*=} ;;
	esac
}
pass=()
while [ $# -gt 0 ]; do
	case $1 in
	--episodes) episodes=${2:?}; shift ;;
	--episodes=*) episodes=${1#*=} ;;
	--mode) mode=${2:?}; shift ;;
	--mode=*) mode=${1#*=} ;;
	--metrics) metrics=true ;;
	--video) video=true ;;
	--smoke) smoke=true ;;
	*) pass+=("$1") ;;
	esac
	shift
done
eval_parse_options ${pass[@]+"${pass[@]}"}
eval_normalize_options
case $mode in paper | pi) ;; *) echo "--mode $mode: paper or pi" >&2; exit 2 ;; esac
if [ "$mode" = paper ] && [[ $model != humanclaw-psv/* ]]; then
	echo "--mode paper runs HumanCLAW's planner: --model humanclaw-psv/<base>" >&2
	exit 2
fi
if [ "$mode" = paper ] && { $verify || $vdm || $stateless || $proprioception || $collision; }; then
	echo "--mode paper runs HumanCLAW's planner as published: no --units-verify, --vdm, --stateless, --humanclaw-proprioception or --humanclaw-collision-feedback" >&2
	exit 2
fi
if [ "$mode" = pi ] && { [ -n "$reasoning" ] || [ "$request_tokens" != 4096 ]; }; then
	echo "--humanclaw-reasoning and --humanclaw-max-tokens are paper mode's request parameters; pi mode plans with --thinking and the model's own limits" >&2
	exit 2
fi
if $collision && ! $metrics; then
	echo "--humanclaw-collision-feedback reads the metric tracker: add --metrics" >&2
	exit 2
fi
if [ -n "$max_steps" ] && ! $smoke; then
	echo "--humanclaw-max-steps truncates the episode: a smoke, not a scored run; add --smoke to run it (keyed max_steps=$max_steps)" >&2
	exit 2
fi
preset=humanclaw
$privileged && preset=humanclaw+privileged
# Named only when set, so an out dir written before these were keyed keeps its configuration.
extra=""
[ "$approval" != off ] && extra+="/approval=$approval"
[ "$max_tool_calls" != 0 ] && extra+="/tool_calls=$max_tool_calls"
[ "$max_tokens" != 0 ] && extra+="/tokens=$max_tokens"
# Paper mode's request contract (reasoning is keyed above, both forms) and pi mode's planner flags.
[ "$request_tokens" != 4096 ] && extra+="/request_tokens=$request_tokens"
[ -n "$thinking" ] && extra+="/thinking=$thinking"
[ "$turns" != 0 ] && extra+="/turns=$turns"
[ -n "$limited" ] && extra+="/limit=$limit"
[ "$units" != both ] && extra+="/units=$units"
$anchor && extra+="/anchor"
[ -n "$vdm_model" ] && extra+="/vdm_model=$vdm_model"
[ -n "$fallback_model" ] && extra+="/fallback=$fallback_model:$fallback_after:$fallback_retry"
$collision && extra+="/collision_feedback=true"
[ -n "$max_steps" ] && extra+="/max_steps=$max_steps"
# The OpenETA extras this run turns on (eval-options.sh's $extras; robot.ts records them as `extras`): part of the configuration.
config="mode=$mode/preset=$preset/model=$model/metrics=$metrics/video=$video/verify=$verify/vdm=$vdm/stateless=$stateless/proprioception=$proprioception${reasoning:+/reasoning=$reasoning}$extra${extras:+/extras=$extras}"
$metrics || echo "eval.sh: without --metrics the benchmark measures nothing: every episode is recorded unscored" >&2
mkdir -p "$out"
if [[ $episodes == *_ep*_* ]]; then list=$episodes; else list=$(cd "$out" && "$PY" -c '
import sys
from pi_embodied_services.robots.humanclaw.env_server import episode_specs
from humanclaw_bench.config import load_config
for r in episode_specs(None, load_config("paper_fullval_v1"), sys.argv[1]):
    print("%s_ep%s_%s" % (r["scene_id"], r["episode_id"], r["object_category"]))
' "$episodes") || { echo "cannot list episodes '$episodes' with $PY" >&2; exit 1; }; fi

cells=()
for key in $list; do
	cells+=("$key")
	dir="$out/$key"
	# This episode's pi arguments (the ones below and the user's), for params-match.mjs.
	export PI_ARGS_JSON=$(node -e 'console.log(JSON.stringify(process.argv.slice(1)))' -- --units=both --humanclaw-mode "$mode" \
		--humanclaw-output "$dir/rollout" $($metrics && echo --humanclaw-metrics) $($video && echo --humanclaw-video) "${pass[@]}")
	if [ -f "$dir/result.json" ]; then
		st=$(node -e 'const r=JSON.parse(require("fs").readFileSync(process.argv[1],"utf8"));console.log(r.config===process.argv[2]?r.status:"other")' "$dir/result.json" "$config")
		# Every experiment flag the robot recorded (params) against this run's (../../scripts/params-match.mjs).
		case $st in success | failure) node "$here/../../scripts/params-match.mjs" "$dir/result.json" >/dev/null || st=other ;; esac
		case $st in
		success | failure | unscored) continue ;;
		other) echo "$dir holds a result of another configuration; use another out dir" >&2 && exit 1 ;;
		esac
	fi
	rm -rf "$dir" && mkdir -p "$dir"
	echo "== $key ($mode)"
	# The prompt precedes the boolean flags: pi would take it as a boolean flag's value.
	${LOCK:+flock "$LOCK"} $PI -p --session-dir "$dir" -e "$here" --units=both --episode "$key" --humanclaw-mode "$mode" \
		--humanclaw-output "$dir/rollout" "Solve the task." $($metrics && echo --humanclaw-metrics) \
		$($video && echo --humanclaw-video) "${pass[@]}" </dev/null >"$dir/stdout.log" 2>"$dir/stderr.log"
	code=$?
	node --input-type=module -e '
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
const [dir, code, config, mode, preset, model] = process.argv.slice(1);
const results = [];
for (const f of readdirSync(dir).filter((f) => f.endsWith(".jsonl")))
	for (const line of readFileSync(`${dir}/${f}`, "utf8").split("\n")) {
		if (!line.includes("\"robot_result\"")) continue;
		const e = JSON.parse(line);
		if (e.type === "custom" && e.customType === "robot_result") results.push(e.data);
	}
const last = results.length === 1 ? results[0] : undefined;
const status = results.length > 1 ? "duplicate_result" : !last ? (Number(code) ? "env_error" : "missing")
	: last.env_error ? "env_error" : last.planner_error ? "planner_error"
	: last.success === true ? "success" : last.success === false ? "failure" : "unscored";
const r = { ...(last ?? {}), status, exit_code: Number(code), config, mode, preset, model: model || null };
writeFileSync(`${dir}/result.json`, `${JSON.stringify(r, null, 2)}\n`);
console.log(JSON.stringify({ status, steps: r.steps, metrics: r.metrics }));
' "$dir" "$code" "$config" "$mode" "$preset" "$model"
done

node -e '
const fs=require("fs");const [out,...cells]=process.argv.slice(1);
const rows=cells.map(c=>{try{return JSON.parse(fs.readFileSync(`${out}/${c}/result.json`,"utf8"))}catch{return {status:"missing"}}});
const n=s=>rows.filter(r=>r.status===s).length;const scored=n("success")+n("failure"),unscored=n("unscored");
console.log(`${rows[0]?.config??"-"}: scored ${scored}/${rows.length} (success ${n("success")}), unscored ${unscored}, invalid ${rows.length-scored-unscored}`);
if(unscored) console.log("unscored episodes ran without --metrics: no NavSR verdict; score them in another out dir with --metrics");
if(rows.length-scored) process.exitCode=1;
' "$out" "${cells[@]}"
bad=$?
if $metrics; then
	"$PY" - "$out" <<'PY'
import json, sys
from pathlib import Path
from humanclaw_bench.evaluation.metrics.episode import aggregate_metric_files
from humanclaw_bench.evaluation.metrics.report import format_metric_summary
out = Path(sys.argv[1])
summary = aggregate_metric_files(out, write_summary=True)
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(format_metric_summary(summary))
PY
fi
exit $bad
