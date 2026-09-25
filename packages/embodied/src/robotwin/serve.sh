#!/usr/bin/env bash
# Start (default) or stop the shared LingBot-VLA WebSocket server used by the RoboTwin robot.
#   [PI_EMBODIED_SERVICES=...] PI_EMBODIED_PYTHON=... LINGBOT_MODEL_PATH=... serve.sh [start|stop]
# start records the server it launches in $LOG_DIR/<module>-<port>.pid, and stop stops only that one:
# a port that already answers is left alone, so a server someone else started keeps running.
set -euo pipefail
SERVICES=${PI_EMBODIED_SERVICES:-$(cd "$(dirname "$0")/../../../../services" && pwd)}
PY=${PI_EMBODIED_PYTHON:-python}
LOG_DIR=${LOG_DIR:-/tmp/pi-embodied}
VLA_PORT=${VLA_PORT:-18400}
VLA=pi_embodied_services.robots.robotwin.vla_server
PIDFILE=$LOG_DIR/$VLA-$VLA_PORT.pid
mkdir -p "$LOG_DIR"

up() { curl -sf "http://127.0.0.1:$VLA_PORT/healthz" >/dev/null; }

if [ "${1:-start}" = stop ]; then
	[ -f "$PIDFILE" ] || exit 0
	pid=$(cat "$PIDFILE")
	if ps -o command= -p "$pid" | grep -q -- "$VLA"; then
		kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
		for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
		kill -9 -- "-$pid" 2>/dev/null || true
	fi
	rm -f "$PIDFILE"
	exit 0
fi

if up; then
	echo "already serving on ws://127.0.0.1:$VLA_PORT (not started here; stop leaves it running)"
	exit 0
fi
: "${LINGBOT_MODEL_PATH:?LingBot-VLA-RoboTwin-EEF-ckpt1500 snapshot}"
M=$LINGBOT_MODEL_PATH
cd "$SERVICES"
PYTHONPATH=$SERVICES QWEN25_PATH=$M/qwen_base setsid nohup "$PY" -m "$VLA" \
	--model-path "$M" --norm-path "$M/norm_stats/robotwin_eef.json" \
	--lingbot-robot-config "${LINGBOT_ROBOT_CONFIG:-$M/configs/robot_configs/robotwin_eef.yaml}" \
	--use-length 50 --port "$VLA_PORT" >"$LOG_DIR/lingbot_server.log" 2>&1 </dev/null &
echo $! >"$PIDFILE"

for _ in $(seq 900); do
	up && { echo "ready on ws://127.0.0.1:$VLA_PORT"; exit 0; }
	sleep 1
done
echo "LingBot server on :$VLA_PORT not ready; see $LOG_DIR/lingbot_server.log" >&2
exit 1
