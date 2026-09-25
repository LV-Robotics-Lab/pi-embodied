#!/usr/bin/env bash
# Start (default) or stop the shared Pi0.5 VLA and SAM3 servers used by the LIBERO robot.
#   [PI_EMBODIED_SERVICES=...] PI_EMBODIED_PYTHON=... PI05_CHECKPOINT_PATH=... SAM3_CHECKPOINT_PATH=... serve.sh [start|stop]
# start records each server it launches in $LOG_DIR/<module>-<port>.pid, and stop stops only those:
# a port that already answers is left alone, so servers someone else started keep running.
set -euo pipefail
SERVICES=${PI_EMBODIED_SERVICES:-$(cd "$(dirname "$0")/../../../../services" && pwd)}
PY=${PI_EMBODIED_PYTHON:-python}
LOG_DIR=${LOG_DIR:-/tmp/pi-embodied}
VLA_PORT=${VLA_PORT:-18200}
SAM3_PORT=${SAM3_PORT:-18300}
VLA=pi_embodied_services.components.pi05_vla_server
SAM3=pi_embodied_services.components.sam3_server
mkdir -p "$LOG_DIR"

up() { curl -sf -X POST "http://127.0.0.1:$1/call" -d '{"method":"healthz"}' | grep -q '"ok": *true'; }

# stop_one MODULE PORT: stop the server this script started there (its whole session), if it still runs.
stop_one() {
	local f="$LOG_DIR/$1-$2.pid" pid
	[ -f "$f" ] || return 0
	pid=$(cat "$f")
	if ps -o command= -p "$pid" | grep -q -- "$1"; then
		kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
		for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
		kill -9 -- "-$pid" 2>/dev/null || true
	fi
	rm -f "$f"
}

# start_one MODULE PORT LOG ARGS...: launch unless the port already serves.
start_one() {
	local module=$1 port=$2 log=$3
	shift 3
	if up "$port"; then
		echo "already serving on :$port (not started here; stop leaves it running)"
		return
	fi
	setsid nohup "$PY" -m "$module" "$@" --host 127.0.0.1 --port "$port" >"$LOG_DIR/$log" 2>&1 </dev/null &
	echo $! >"$LOG_DIR/$module-$port.pid"
}

if [ "${1:-start}" = stop ]; then
	stop_one "$VLA" "$VLA_PORT"
	stop_one "$SAM3" "$SAM3_PORT"
	exit 0
fi

cd "$SERVICES"
export PYTHONPATH=$SERVICES
start_one "$VLA" "$VLA_PORT" vla_server.log --embodiment libero --transport http
start_one "$SAM3" "$SAM3_PORT" sam3_server.log --transport http

for port in "$SAM3_PORT" "$VLA_PORT"; do
	for _ in $(seq 600); do
		up "$port" && break
		sleep 1
	done
	up "$port" || { echo "server on :$port not ready; see $LOG_DIR" >&2; exit 1; }
	echo "ready on :$port"
done
