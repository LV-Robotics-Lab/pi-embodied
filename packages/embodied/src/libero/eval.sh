#!/usr/bin/env bash
# Run LIBERO episodes with pi in print mode and report LIBERO-judged success.
#   eval.sh <out-dir> <suite> <tasks> <seeds> [pi args...]
#   eval.sh runs/l10 libero_10_task 0-9 1-10 --model <provider/model> --thinking low
#
# Each episode runs in its own session directory and ends with a result.json taken from
# the session's last `libero_result` entry. Episodes that already have one are skipped, so
# rerunning resumes. Startup failures (env_error) and episodes without a result (missing)
# are reported but excluded from the success rate.
set -uo pipefail
out=$1 suite=$2 tasks=$3 seeds=$4
shift 4
here=$(cd "$(dirname "$0")" && pwd)
PI=${PI:-pi}
expand() { for part in ${1//,/ }; do seq "${part%-*}" "${part#*-}"; done; }
mkdir -p "$out"

record() { # <dir> <exit code>: write result.json from the episode's session
	node --input-type=module -e '
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
const [dir, code] = process.argv.slice(1);
let last;
for (const f of readdirSync(dir).filter((f) => f.endsWith(".jsonl"))) {
	for (const line of readFileSync(`${dir}/${f}`, "utf8").split("\n")) {
		if (!line.includes("\"libero_result\"")) continue;
		const e = JSON.parse(line);
		if (e.type === "custom" && e.customType === "libero_result") last = e.data;
	}
}
const status = !last ? (Number(code) ? "env_error" : "missing") : last.env_error ? "env_error" : last.terminated ? "success" : "failure";
const result = { ...(last ?? {}), status, exit_code: Number(code) };
writeFileSync(`${dir}/result.json`, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ status, terminated: result.terminated, claimed: result.claimed, env_steps: result.env_steps }));
' "$1" "$2"
}

for task in $(expand "$tasks"); do
	for seed in $(expand "$seeds"); do
		dir="$out/${suite}_t${task}_s${seed}"
		[ -s "$dir/result.json" ] && continue
		mkdir -p "$dir"
		echo "== $suite task $task seed $seed"
		$PI -p --session-dir "$dir" -e "$here" --suite "$suite" --task "$task" --seed "$seed" "$@" \
			"Solve the task." </dev/null >"$dir/stdout.log" 2>"$dir/stderr.log"
		record "$dir" "$?"
	done
done

node --input-type=module -e '
import { existsSync, readdirSync, readFileSync } from "node:fs";
const out = process.argv[1];
const rows = readdirSync(out)
	.filter((d) => existsSync(`${out}/${d}/result.json`))
	.map((d) => JSON.parse(readFileSync(`${out}/${d}/result.json`, "utf8")));
const n = (s) => rows.filter((r) => r.status === s).length;
const scored = n("success") + n("failure");
const lies = rows.filter((r) => r.status === "failure" && r.claimed === "success").length;
const rate = scored ? ((100 * n("success")) / scored).toFixed(1) : "-";
console.log(`success ${n("success")}/${scored} (${rate}%), claimed-but-failed ${lies}, env_error ${n("env_error")}, missing ${n("missing")}`);
' "$out"
