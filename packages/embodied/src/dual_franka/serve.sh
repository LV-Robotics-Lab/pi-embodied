#!/usr/bin/env bash
# Start (default) or stop the dual-Franka Pi0.5 VLA server, plus SAM3 when SAM3_CHECKPOINT_PATH is set.
#   [PI_EMBODIED_SERVICES=...] PI_EMBODIED_PYTHON=... PI05_CHECKPOINT_PATH=... DUAL_FRANKA_REPO_ID=... \
#     [SAM3_CHECKPOINT_PATH=...] [CUDA_DEVICE=0] serve.sh [start|stop]
set -euo pipefail
SERVICES=${PI_EMBODIED_SERVICES:-$(cd "$(dirname "$0")/../../../../services" && pwd)}
PY=${PI_EMBODIED_PYTHON:-python}
LOG_DIR=${LOG_DIR:-/tmp/pi-embodied}
VLA_PORT=${VLA_PORT:-18210}
SAM3_PORT=${SAM3_PORT:-18310}
mkdir -p "$LOG_DIR"

if [ "${1:-start}" = stop ]; then
	pkill -f "[p]i05_vla_server --embodiment dual_franka" || true
	pkill -f "[p]i_embodied_services.components.sam3_server --transport http --host 127.0.0.1 --port $SAM3_PORT" || true
	exit 0
fi

: "${PI05_CHECKPOINT_PATH:?dual-Franka Pi0.5 checkpoint}"
: "${DUAL_FRANKA_REPO_ID:?SFT dataset repo id that locates norm_stats.json}"
cuda=()
[ -n "${CUDA_DEVICE:-}" ] && cuda=(--cuda-device "$CUDA_DEVICE")
cd "$SERVICES"
export PYTHONPATH=$SERVICES${PYTHONPATH:+:$PYTHONPATH}
ports=("$VLA_PORT")
setsid nohup "$PY" -m pi_embodied_services.components.pi05_vla_server --embodiment dual_franka --transport http \
	--host 127.0.0.1 --port "$VLA_PORT" --model-path "$PI05_CHECKPOINT_PATH" --repo-id "$DUAL_FRANKA_REPO_ID" \
	${cuda[@]+"${cuda[@]}"} >"$LOG_DIR/dual_franka_vla_server.log" 2>&1 </dev/null &
if [ -n "${SAM3_CHECKPOINT_PATH:-}" ]; then
	setsid nohup "$PY" -m pi_embodied_services.components.sam3_server --transport http --host 127.0.0.1 --port "$SAM3_PORT" \
		${cuda[@]+"${cuda[@]}"} >"$LOG_DIR/dual_franka_sam3_server.log" 2>&1 </dev/null &
	ports+=("$SAM3_PORT")
fi

for port in "${ports[@]}"; do
	for _ in $(seq 600); do
		curl -sf -X POST "http://127.0.0.1:$port/call" -d '{"method":"healthz"}' | grep -q '"ok": *true' && break
		sleep 1
	done
	curl -sf -X POST "http://127.0.0.1:$port/call" -d '{"method":"healthz"}' | grep -q '"ok": *true' ||
		{ echo "server on :$port not ready; see $LOG_DIR" >&2; exit 1; }
	echo "ready on :$port"
done
