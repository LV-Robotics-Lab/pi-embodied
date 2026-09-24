#!/usr/bin/env bash
# Run LIBERO episodes with pi in print mode and report LIBERO-judged success.
#   eval.sh <out-dir> <suite> <tasks> <seeds> [pi args...]
#   eval.sh runs/l10 libero_10 0-9 0-2 --model selfhost/muse-glimmer-30b --thinking medium
set -uo pipefail
out=$1 suite=$2 tasks=$3 seeds=$4
shift 4
here=$(cd "$(dirname "$0")" && pwd)
PI=${PI:-pi}
expand() { for part in ${1//,/ }; do seq "${part%-*}" "${part#*-}"; done; }
mkdir -p "$out"

for task in $(expand "$tasks"); do
	for seed in $(expand "$seeds"); do
		echo "== $suite task $task seed $seed"
		$PI -p --session-dir "$out" -e "$here" --suite "$suite" --task "$task" --seed "$seed" "$@" \
			"Solve the task." 2> >(tee -a "$out/stderr.log" | grep '^\[libero\]' | sed 's/^\[libero\] //' >>"$out/results.jsonl") >/dev/null
		tail -n 1 "$out/results.jsonl"
	done
done

node -e '
const rows = require("fs").readFileSync(process.argv[1], "utf8").trim().split("\n").map(JSON.parse);
const ok = rows.filter((r) => r.terminated).length;
const lies = rows.filter((r) => r.claimed === "success" && !r.terminated).length;
console.log(`success ${ok}/${rows.length}, claimed-but-failed ${lies}`);
' "$out/results.jsonl"
