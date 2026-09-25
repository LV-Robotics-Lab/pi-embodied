#!/usr/bin/env bash
# Start (default) or stop the shared RLDX-1 VLA server used by the RoboCasa robot.
#   [PI_EMBODIED_SERVICES=...] ROBOCASA_PYTHON=... RLDX_MODEL_PATH=... serve.sh [start|stop]
# Episodes get private RPC sessions, so one server serves any number of pi sessions.
set -euo pipefail
SERVICES=${PI_EMBODIED_SERVICES:-$(cd "$(dirname "$0")/../../../../services" && pwd)}
PY=${ROBOCASA_PYTHON:-${PI_EMBODIED_PYTHON:-python}}
LOG_DIR=${LOG_DIR:-/tmp/pi-embodied}
RLDX_PORT=${RLDX_PORT:-18500}
mkdir -p "$LOG_DIR"

if [ "${1:-start}" = stop ]; then
	pkill -f "[p]i_embodied_services.robots.robocasa.vla_server" || true
	exit 0
fi

: "${RLDX_MODEL_PATH:?RLDX-1-FT-RC365 checkpoint directory}"
cd "$SERVICES"
export PYTHONPATH=$SERVICES
# The checkpoint's backbone metadata (RLWRLD/RLDX-1-VLM) resolves from this cache.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export NO_ALBUMENTATIONS_UPDATE=1
setsid nohup "$PY" -m pi_embodied_services.robots.robocasa.vla_server --model-path "$RLDX_MODEL_PATH" --transport http \
	--host 127.0.0.1 --port "$RLDX_PORT" ${CUDA_DEVICE:+--cuda-device "$CUDA_DEVICE"} \
	>"$LOG_DIR/rldx_server.log" 2>&1 </dev/null &

for _ in $(seq 900); do
	curl -sf -X POST "http://127.0.0.1:$RLDX_PORT/call" -d '{"method":"healthz"}' | grep -q '"ok": *true' && break
	sleep 1
done
curl -sf -X POST "http://127.0.0.1:$RLDX_PORT/call" -d '{"method":"healthz"}' | grep -q '"ok": *true' ||
	{ echo "RLDX server on :$RLDX_PORT not ready; see $LOG_DIR/rldx_server.log" >&2; exit 1; }
echo "ready on :$RLDX_PORT"
