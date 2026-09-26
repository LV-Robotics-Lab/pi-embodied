#!/usr/bin/env bash
# Start (default) or stop the shared Pi0.5 VLA and SAM3 servers used by the LIBERO robot.
#   [PI_EMBODIED_SERVICES=...] PI_EMBODIED_PYTHON=... PI05_CHECKPOINT_PATH=... SAM3_CHECKPOINT_PATH=... serve.sh [start|stop]
# The third-party VLAs start too when their venv's python is set (each has its own venv, see services/README):
#   OPENVLA_PYTHON=... [OPENVLA_CHECKPOINT_PATH=... OPENVLA_PORT=18600] mounts pi --openvla http://127.0.0.1:18600
#   OPENVLA_OFT_PYTHON=... [OPENVLA_OFT_CHECKPOINT_PATH=... OPENVLA_OFT_PORT=18700]      --openvla-oft ...:18700
#   GR00T_PYTHON=... [GR00T_CHECKPOINT_PATH=... GR00T_PORT=18800]                        --gr00t ...:18800
# PI05=off leaves Pi0.5 out (an adapter-only run on a GPU that cannot hold both).
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
OPENVLA_PORT=${OPENVLA_PORT:-18600}
OPENVLA_OFT_PORT=${OPENVLA_OFT_PORT:-18700}
GR00T_PORT=${GR00T_PORT:-18800}
OPENVLA=pi_embodied_services.components.openvla_server
OPENVLA_OFT=pi_embodied_services.components.openvla_oft_server
GR00T=pi_embodied_services.components.gr00t_server
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
	setsid nohup "${PYTHON:-$PY}" -m "$module" "$@" --host 127.0.0.1 --port "$port" >"$LOG_DIR/$log" 2>&1 </dev/null &
	echo $! >"$LOG_DIR/$module-$port.pid"
}

if [ "${1:-start}" = stop ]; then
	stop_one "$VLA" "$VLA_PORT"
	stop_one "$SAM3" "$SAM3_PORT"
	stop_one "$OPENVLA" "$OPENVLA_PORT"
	stop_one "$OPENVLA_OFT" "$OPENVLA_OFT_PORT"
	stop_one "$GR00T" "$GR00T_PORT"
	exit 0
fi

cd "$SERVICES"
export PYTHONPATH=$SERVICES
ports=("$SAM3_PORT")
if [ "${PI05:-on}" != off ]; then
	start_one "$VLA" "$VLA_PORT" vla_server.log --embodiment libero --transport http
	ports+=("$VLA_PORT")
fi
start_one "$SAM3" "$SAM3_PORT" sam3_server.log --transport http
# Each adapter runs in its own venv: PYTHON overrides $PY for that one server.
if [ -n "${OPENVLA_PYTHON:-}" ]; then
	PYTHON=$OPENVLA_PYTHON start_one "$OPENVLA" "$OPENVLA_PORT" openvla_server.log --transport http
	ports+=("$OPENVLA_PORT")
fi
if [ -n "${OPENVLA_OFT_PYTHON:-}" ]; then
	PYTHON=$OPENVLA_OFT_PYTHON start_one "$OPENVLA_OFT" "$OPENVLA_OFT_PORT" openvla_oft_server.log --transport http
	ports+=("$OPENVLA_OFT_PORT")
fi
if [ -n "${GR00T_PYTHON:-}" ]; then
	PYTHON=$GR00T_PYTHON start_one "$GR00T" "$GR00T_PORT" gr00t_server.log --transport http
	ports+=("$GR00T_PORT")
fi

for port in "${ports[@]}"; do
	for _ in $(seq 600); do
		up "$port" && break
		sleep 1
	done
	up "$port" || { echo "server on :$port not ready; see $LOG_DIR" >&2; exit 1; }
	echo "ready on :$port"
done
