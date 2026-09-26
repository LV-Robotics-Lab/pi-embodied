#!/usr/bin/env bash
# gpu-e2e.sh <robot>[=<env file>] ...: run the GPU end-to-end suite (test/gpu-e2e.test.ts) one robot at a
# time, each in a subshell with its env file sourced (PI_EMBODIED_PYTHON, the CUDA / EGL / MuJoCo
# variables, SAM3_CHECKPOINT_PATH, PI_EMBODIED_GPU_LOCK, ...) and this checkout's services.
#   E2E_GPU=1 MIN_FREE=8000 test/gpu-e2e.sh metaworld=/root/autodl-tmp/tools/metaworld-env.sh \
#     libero=/root/autodl-tmp/tools/embodied-env.sh
# With E2E_GPU set (nvidia-smi's ordinal), a robot starts only once that GPU has MIN_FREE MiB free
# (default 8000; polled without any lock, up to WAIT_S seconds, default 3600): the light-simulator rule of a
# shared GPU. A robot rendered on the CPU (E2E_GPU unset in its env file) does not wait. Model loads
# (the LIBERO --serve-models case) take PI_EMBODIED_GPU_LOCK themselves.
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
services=$(cd "$here/../../services" && pwd)
free_mib() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" | awk -F, '{print $2-$1}'; }
[ $# -gt 0 ] || { echo "usage: $0 <robot>[=<env file>] ..." >&2; exit 2; }
rc=0
for arg in "$@"; do
	robot=${arg%%=*}
	envfile=
	[ "$arg" = "$robot" ] || envfile=${arg#*=}
	(
		set -e
		# shellcheck disable=SC1090
		[ -z "$envfile" ] || . "$envfile"
		export PI_EMBODIED_SERVICES=$services PI_EMBODIED_E2E=$robot
		if [ -n "${E2E_GPU:-}" ]; then
			deadline=$(($(date +%s) + ${WAIT_S:-3600}))
			until [ "$(free_mib "$E2E_GPU")" -ge "${MIN_FREE:-8000}" ]; do
				[ "$(date +%s)" -lt "$deadline" ] || { echo "$robot: GPU $E2E_GPU never had ${MIN_FREE:-8000} MiB free" >&2; exit 1; }
				echo "$robot: GPU $E2E_GPU has $(free_mib "$E2E_GPU") MiB free, waiting for ${MIN_FREE:-8000} $(date +%T)"
				sleep 30
			done
		fi
		echo "== $robot $(date +%T)"
		cd "$here"
		node --test --test-timeout=2400000 --experimental-strip-types test/gpu-e2e.test.ts
	) || rc=1
done
exit $rc
