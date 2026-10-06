#!/usr/bin/env bash
# fixed-regression.sh <new-output-directory> <environment-directory> [provider/model]
# Environment directory contains metaworld-env.sh, maniskill-env.sh and libero-env.sh.
# Run under the host's GPU lock. The six cells are sequential, with independent empty memories.
set -euo pipefail
[ "$#" -ge 2 ] && [ "$#" -le 3 ] || { echo "usage: $0 <new-out> <env-dir> [provider/model]" >&2; exit 2; }
root=$(cd "$(dirname "$0")/../../../.." && pwd)
out=$1
envdir=$(cd "$2" && pwd)
model=${3:-relay/gpt-6-astra}
for robot in metaworld maniskill libero; do
 [ -f "$envdir/$robot-env.sh" ] || { echo "missing $envdir/$robot-env.sh" >&2; exit 2; }
done
[ -z "$(git -C "$root" status --porcelain)" ] || { echo 'Regression requires a clean checkout.' >&2; exit 2; }
[ -f "$root/packages/coding-agent/dist/cli.js" ] || { echo 'Build this checkout before regression.' >&2; exit 2; }
mkdir "$out"
out=$(cd "$out" && pwd)
git -C "$root" rev-parse HEAD > "$out/commit.txt"
printf '%s\n' "$model" > "$out/model.txt"
printf '%s\n' '#!/usr/bin/env bash' 'exec node "$PI_REGRESSION_CLI" "$@"' > "$out/pi"
chmod 700 "$out/pi"
rc=0
for robot in metaworld maniskill libero; do
 for seed in 0 1; do
  (
   # Host configuration is trusted operator input; never commit credentials here.
   source "$envdir/$robot-env.sh"
   export PI="$out/pi" PI_REGRESSION_CLI="$root/packages/coding-agent/dist/cli.js"
   export PI_EMBODIED_SERVICES="$root/services"
   # A fresh local memory per cell: where it lives is deployment config (dirs.memory), set for this
   # episode through its environment override (src/infra/config.ts; --memory-dir is gone, docs/flags-migration.md).
   memory="$out/memory-$robot-$seed"
   mkdir "$memory"
   printf '# Local task memory\nNo stored task recipes.\n' > "$memory/MEMORY.md"
   export PI_EMBODIED_DIRS_MEMORY="$memory"
   case "$robot" in
    metaworld) cells=(reach-v3 "$seed");;
    maniskill) cells=(PickCube-v1 "$seed");;
    libero) cells=(libero_spatial 0 "$seed");;
   esac
   bash "$root/packages/embodied/src/robots/$robot/eval.sh" "$out/$robot-$seed" "${cells[@]}" \
    --model "$model" --thinking low --units=true --code=false --max-turns 40 --time-limit 300 \
    --no-skills --no-prompt-templates --memory-profile local
  ) > "$out/$robot-$seed.log" 2>&1 && status=0 || status=$?
  printf '%s %s %s\n' "$robot" "$seed" "$status" >> "$out/status.txt"
  [ "$status" -eq 0 ] || rc=1
 done
done
git -C "$root" rev-parse HEAD > "$out/commit-after.txt"
cmp "$out/commit.txt" "$out/commit-after.txt" || rc=1
[ -z "$(git -C "$root" status --porcelain)" ] || rc=1
date -Iseconds > "$out/completed.txt"
exit "$rc"
