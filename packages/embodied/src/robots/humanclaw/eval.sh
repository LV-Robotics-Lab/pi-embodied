#!/usr/bin/env bash
# Run HumanClawBench episodes with pi in print mode and report the paper metrics.
#   eval.sh <out-dir> [--episodes one|val100|fullval|<list.json>] [--mode paper|pi] [--metrics] [--video] [pi args...]
#   eval.sh runs/hc-paper --episodes one --mode paper --metrics --model humanclaw-psv/selfhost/muse-glimmer-30b
#   eval.sh runs/hc-pi --episodes val100 --mode pi --metrics --model selfhost/muse-glimmer-30b   (GPU: cuda_device or PI_EMBODIED_CUDA_DEVICE)
#
# One pi process (and one env server: Habitat and the motion model are rebuilt per episode, as the
# release's dispatcher runs one process per episode) per episode, in <out>/<scene>_ep<id>_<category>/,
# ending with a result.json from the session's `robot_result` plus the configuration: mode, preset
# (`humanclaw`, `humanclaw+privileged`), model, --units-verify / --vdm / --stateless, --metrics,
# --video, and --approval / --max-tool-calls / --max-tokens when set. An episode is valid when the environment produced a result and the planner did not fail.
# Rerunning retries only invalid episodes and refuses an out dir holding another configuration.
# With --metrics the summary is HumanCLAW's own aggregate_metric_files() over every metrics.json
# (summary.json, and metrics_summary.json beside it), run with $HUMANCLAW_PYTHON (the humanclaw venv).
# Heavy: run serially with LOCK set (e.g. /root/autodl-tmp/locks/gpu1.lock), each episode takes it
# with flock and releases it between episodes, so other GPU jobs interleave. scripts/eval-parallel.sh
# takes it once for the whole run and starts this script with LOCK unset (an episode taking it here
# would wait on that run's own lock).
set -uo pipefail
out=${1:?usage: eval.sh <out-dir> [--episodes ...] [--mode paper|pi] [--metrics] [--video] [pi args...]}
shift
here=$(cd "$(dirname "$0")" && pwd)
# Renamed flags and the ones the deployment config replaced stop here, before any cell runs.
. "$here/../../scripts/old-flags.sh"
old_flags "$@" || exit 2
PI=${PI:-pi}
PY=${HUMANCLAW_PYTHON:-${PI_EMBODIED_PYTHON:-python}}
episodes=one mode=paper metrics=false video=false
model="" privileged=false verify=false vdm=false stateless=false reasoning="" proprioception=false
approval=off max_tool_calls=0 max_tokens=0
pass=()
while [ $# -gt 0 ]; do
	case $1 in
	--episodes) episodes=${2:?}; shift ;;
	--episodes=*) episodes=${1#*=} ;;
	--mode) mode=${2:?}; shift ;;
	--mode=*) mode=${1#*=} ;;
	--metrics) metrics=true ;;
	--video) video=true ;;
	--model) model=${2:-}; pass+=("$1" "${2:-}"); shift ;;
	--model=*) model=${1#*=}; pass+=("$1") ;;
	--privileged) privileged=true; pass+=("$1") ;;
	--humanclaw-proprioception | --humanclaw-proprioception=true) proprioception=true; pass+=("$1") ;;
	--humanclaw-proprioception=*) echo "omit --humanclaw-proprioception to disable it" >&2; exit 2 ;;
	--humanclaw-reasoning) reasoning=${2:-}; pass+=("$1" "${2:-}"); shift ;;
	--units-verify) verify=true; pass+=("$1") ;;
	--vdm) vdm=true; pass+=("$1") ;;
	--stateless) stateless=true; pass+=("$1") ;;
	# --approval (src/capabilities/operator.ts) and the --max-tool-calls / --max-tokens budgets (src/robot.ts).
	--approval) approval=${2:-off}; pass+=("$1" "${2:-}"); shift ;;
	--approval=*) approval=${1#*=}; pass+=("$1") ;;
	--max-tool-calls) max_tool_calls=${2:-0}; pass+=("$1" "${2:-}"); shift ;;
	--max-tool-calls=*) max_tool_calls=${1#*=}; pass+=("$1") ;;
	--max-tokens) max_tokens=${2:-0}; pass+=("$1" "${2:-}"); shift ;;
	--max-tokens=*) max_tokens=${1#*=}; pass+=("$1") ;;
	*) pass+=("$1") ;;
	esac
	shift
done
case $mode in paper | pi) ;; *) echo "--mode $mode: paper or pi" >&2; exit 2 ;; esac
if [ "$mode" = paper ] && [[ $model != humanclaw-psv/* ]]; then
	echo "--mode paper runs HumanCLAW's planner: --model humanclaw-psv/<base>" >&2
	exit 2
fi
if [ "$mode" = paper ] && { $verify || $vdm || $stateless || $proprioception; }; then
	echo "--mode paper runs HumanCLAW's planner as published: no --units-verify, --vdm, --stateless or --humanclaw-proprioception" >&2
	exit 2
fi
preset=humanclaw
$privileged && preset=humanclaw+privileged
# Named only when set, so an out dir written before these were keyed keeps its configuration.
extra=""
[ "$approval" != off ] && extra+="/approval=$approval"
[ "$max_tool_calls" != 0 ] && extra+="/tool_calls=$max_tool_calls"
[ "$max_tokens" != 0 ] && extra+="/tokens=$max_tokens"
# The OpenETA extras this run turns on (robot.ts records them as `extras`): part of the configuration.
extras=$(for a in "${pass[@]}"; do case $a in (--waypoints | --waypoints=true | --align-wrist | --align-wrist=true | --grasp-advisor | --grasp-advisor=true | --object-memory | --object-memory=true | --web-tools | --web-tools=true) a=${a#--} && echo "${a%=true}" ;; esac; done | sort -u | paste -sd, -)
config="mode=$mode/preset=$preset/model=$model/metrics=$metrics/video=$video/verify=$verify/vdm=$vdm/stateless=$stateless/proprioception=$proprioception${reasoning:+/reasoning=$reasoning}$extra${extras:+/extras=$extras}"
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
		success | failure) continue ;;
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
	: last.env_error ? "env_error" : last.planner_error ? "planner_error" : last.success ? "success" : "failure";
const r = { ...(last ?? {}), status, exit_code: Number(code), config, mode, preset, model: model || null };
writeFileSync(`${dir}/result.json`, `${JSON.stringify(r, null, 2)}\n`);
console.log(JSON.stringify({ status, steps: r.steps, metrics: r.metrics }));
' "$dir" "$code" "$config" "$mode" "$preset" "$model"
done

node -e '
const fs=require("fs");const [out,...cells]=process.argv.slice(1);
const rows=cells.map(c=>{try{return JSON.parse(fs.readFileSync(`${out}/${c}/result.json`,"utf8"))}catch{return {status:"missing"}}});
const n=s=>rows.filter(r=>r.status===s).length;const scored=n("success")+n("failure");
console.log(`${rows[0]?.config??"-"}: scored ${scored}/${rows.length}, invalid ${rows.length-scored}`);
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
