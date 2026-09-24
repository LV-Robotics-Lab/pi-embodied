#!/usr/bin/env bash
# Start (default) or stop the shared Pi0.5 VLA and SAM3 servers used by the LIBERO robot.
#   RPENT_ROOT=... RPENT_PYTHON=... PI05_CHECKPOINT_PATH=... SAM3_CHECKPOINT_PATH=... serve.sh [start|stop]
set -euo pipefail
: "${RPENT_ROOT:?RPent checkout}"
PY=${RPENT_PYTHON:-python}
LOG_DIR=${LOG_DIR:-/tmp/pi-embodied}
VLA_PORT=${VLA_PORT:-18200}
SAM3_PORT=${SAM3_PORT:-18300}
mkdir -p "$LOG_DIR"

if [ "${1:-start}" = stop ]; then
	pkill -f "[p]i05_vla_server.py" || true
	pkill -f "[s]am3_server.py" || true
	exit 0
fi

cd "$RPENT_ROOT"
export PYTHONPATH=$RPENT_ROOT
setsid nohup "$PY" rpent/robots/components/pi05_vla_server.py --embodiment libero --transport http \
	--host 127.0.0.1 --port "$VLA_PORT" >"$LOG_DIR/vla_server.log" 2>&1 </dev/null &
setsid nohup "$PY" rpent/robots/components/sam3_server.py --transport http \
	--host 127.0.0.1 --port "$SAM3_PORT" >"$LOG_DIR/sam3_server.log" 2>&1 </dev/null &

for port in "$SAM3_PORT" "$VLA_PORT"; do
	for _ in $(seq 600); do
		curl -sf -X POST "http://127.0.0.1:$port/call" -d '{"method":"healthz"}' | grep -q '"ok": *true' && break
		sleep 1
	done
	curl -sf -X POST "http://127.0.0.1:$port/call" -d '{"method":"healthz"}' | grep -q '"ok": *true' ||
		{ echo "server on :$port not ready; see $LOG_DIR" >&2; exit 1; }
	echo "ready on :$port"
done
