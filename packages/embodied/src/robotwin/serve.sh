#!/usr/bin/env bash
# Start (default) or stop the shared LingBot-VLA WebSocket server used by the RoboTwin robot.
#   RPENT_ROOT=... RPENT_PYTHON=... LINGBOT_MODEL_PATH=... serve.sh [start|stop]
set -euo pipefail
: "${RPENT_ROOT:?RPent checkout}"
PY=${RPENT_PYTHON:-python}
LOG_DIR=${LOG_DIR:-/tmp/pi-embodied}
VLA_PORT=${VLA_PORT:-18400}
mkdir -p "$LOG_DIR"

if [ "${1:-start}" = stop ]; then
	pkill -f "[r]obots/robotwin/vla_server.py" || true
	exit 0
fi

: "${LINGBOT_MODEL_PATH:?LingBot-VLA-RoboTwin-EEF-ckpt1500 snapshot}"
M=$LINGBOT_MODEL_PATH
cd "$RPENT_ROOT"
PYTHONPATH=$RPENT_ROOT QWEN25_PATH=$M/qwen_base setsid nohup "$PY" robots/robotwin/vla_server.py \
	--model-path "$M" --norm-path "$M/norm_stats/robotwin_eef.json" \
	--lingbot-robot-config "${LINGBOT_ROBOT_CONFIG:-$M/configs/robot_configs/robotwin_eef.yaml}" \
	--use-length 50 --port "$VLA_PORT" >"$LOG_DIR/lingbot_server.log" 2>&1 </dev/null &

for _ in $(seq 900); do
	curl -sf "http://127.0.0.1:$VLA_PORT/healthz" >/dev/null && { echo "ready on ws://127.0.0.1:$VLA_PORT"; exit 0; }
	sleep 1
done
echo "LingBot server on :$VLA_PORT not ready; see $LOG_DIR/lingbot_server.log" >&2
exit 1
