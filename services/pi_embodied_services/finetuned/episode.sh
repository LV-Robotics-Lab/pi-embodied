#!/usr/bin/env bash
# episode.sh <robot> <out dir> [pi args...]: serve a fine-tuned adapter (serve.sh), run one pi episode
# per seed with `--model finetuned/local --ft-model $ADAPTER`, then stop the server. The whole run holds the GPU1 lock.
#
#   SEEDS="0 1" bash episode.sh maniskill /root/autodl-tmp/runs/finetuned/pickcube --env-id PickCube-v1
#
#   ADAPTER=qwen3_5_2b_showharness_sim   the served adapter name (serve.sh's LORA name; MODEL/LORA/FAMILY pass through)
#   SEEDS="0"  SEED_FLAG=--seed           one episode per seed, sessions in <out dir>/s<seed>
#   PORT=8010  LOCK=/root/autodl-tmp/locks/gpu1.lock (LOCK= to skip)  PI=<pi checkout>  ENV_SH=<file to source>
#   DASHBOARD_PORT=8778 (empty: no dashboard)  PROMPT="Solve the task."
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI="${PI:-$(cd "$HERE/../../.." && pwd)}"
robot=$1 out=$2; shift 2
ADAPTER="${ADAPTER:-qwen3_5_2b_showharness_sim}"
PORT="${PORT:-8010}"
LOCK="${LOCK-/root/autodl-tmp/locks/gpu1.lock}"
DASHBOARD_PORT="${DASHBOARD_PORT-8778}"
mkdir -p "$out"
[ -n "${ENV_SH:-}" ] && . "$ENV_SH"
export PATH=/root/autodl-tmp/tools/node/bin:$PATH

run() {
  PORT=$PORT setsid bash "$HERE/serve.sh" > "$out/vllm.log" 2>&1 < /dev/null &
  local server=$!
  trap 'kill -- -'"$server"' 2>/dev/null; wait '"$server"' 2>/dev/null' EXIT
  local t=0
  until curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" | grep -q "\"$ADAPTER\""; do
    kill -0 "$server" 2>/dev/null || { echo "serve.sh exited; see $out/vllm.log"; tail -20 "$out/vllm.log"; return 1; }
    t=$((t + 5)); [ $t -gt 900 ] && { echo "vLLM not ready after 900 s"; return 1; }
    sleep 5
  done
  echo "$(date +%T) vLLM ready on :$PORT ($ADAPTER) after ${t}s"
  local dash=()
  [ -n "$DASHBOARD_PORT" ] && dash=(-e packages/embodied/src/dashboard --dashboard=true --dashboard-port "$DASHBOARD_PORT")
  cd "$PI" || return 1
  for seed in ${SEEDS:-0}; do
    local d="$out/s$seed"
    mkdir -p "$d"
    echo "$(date +%T) episode $robot ${SEED_FLAG:---seed} $seed -> $d"
    node packages/coding-agent/dist/cli.js -p --session-dir "$d" \
      -e "packages/embodied/src/$robot" -e packages/embodied/src/finetuned "${dash[@]}" \
      --units=true --model finetuned/local --ft-model "$ADAPTER" --ft-endpoint "http://127.0.0.1:$PORT/v1" \
      "${SEED_FLAG:---seed}" "$seed" "$@" "${PROMPT:-Solve the task.}" \
      < /dev/null > "$d/stdout.log" 2> "$d/stderr.log"
    echo "$(date +%T) pi exit $? $(grep -h -o "\[$robot\] .*" "$d/stderr.log" | tail -1)"
  done
}

if [ -n "$LOCK" ]; then
  exec 9> "$LOCK"
  echo "$(date +%T) waiting for $LOCK"
  flock 9
fi
run "$@"
