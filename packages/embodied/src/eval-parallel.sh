#!/usr/bin/env bash
# Run a robot's eval.sh matrix on N workers, optionally as an A/B over variants, and report success
# rate and Pass@k per variant.
#   eval-parallel.sh [options] <robot> <out-dir> <selection...> [pi args...]
#   eval-parallel.sh -j 4 --gpus 1 libero runs/l10 libero_10_task 0-9 0-4 --model <provider/model>
#   eval-parallel.sh -j 2 --gpus 1 --variant base= --variant vdm=--vdm=true --min-success 30 \
#     maniskill runs/ms-vdm PickCube-v1,StackCube-v1 0-9 --model <provider/model> --thinking low
#
# <selection> is the robot's eval.sh positional arguments: libero <suite> <tasks> <seeds>, maniskill /
# metaworld / robosuite / genesis / behavior <tasks> <seeds> ("-" = eval.sh's default task set, e.g.
# robosuite's seven), robolab|robodojo <tasks> <seeds>, robotwin <tasks|all>, robocasa <splits|all> (TASKS/SEEDS
# narrow it as for eval.sh). The matrix is cut into units, one (task, seed) cell each (a whole task,
# all its seeds, for RoboTwin, whose eval.sh cannot pick a seed), and every unit runs as its own
# `<robot>/eval.sh` call, so a cell's directory, result.json, validity and rerun rules are exactly
# eval.sh's: a cell already holding a valid result is skipped, an invalid one is rerun, and a result
# of another configuration stops the run. Workers take units in order from a shared list (a unit is
# claimed with an atomic mkdir), so the set of cells does not depend on N and an interrupted run is
# finished by running the same command again. Each worker runs in its own process group; INT, TERM
# and HUP (an ssh drop) stop every group and wait for it, and a rerun refuses to start while a group
# of an earlier run (.parallel/w*.pid) is still alive, since it would race it for the same cells.
#
# Options:
#   -j, --workers N            parallel eval.sh workers (default 1)
#   --gpus LIST                GPUs (nvidia-smi indices) given to the workers round-robin, so several
#                              workers can share one GPU (default: $CUDA_VISIBLE_DEVICES; unset leaves
#                              the GPU environment alone). Each worker gets CUDA_DEVICE_ORDER=PCI_BUS_ID,
#                              CUDA_VISIBLE_DEVICES=<gpu>, and MUJOCO_EGL_DEVICE_ID set to the EGL device
#                              on the same PCI bus (nvidia-smi bus id -> /dev/dri/by-path card -> EGL
#                              device's DRM file, as OpenETA's sim worker pool does; probed with
#                              $EGL_PYTHON, default $PI_EMBODIED_PYTHON). LIBERO, RoboCasa and RoboLab
#                              also get --cuda-device <gpu>: their env servers pin the GPU themselves.
#   --variant NAME=ARGS        repeatable: extra pi args (split on spaces) for variant NAME, whose cells
#                              go to <out-dir>/NAME; every variant runs the same cells and seeds. With no
#                              --variant the cells go to <out-dir> itself, as with a serial eval.sh run.
#   --max-api-concurrency M    at most M model calls at once over all workers (api-gate.ts); the
#                              simulator and VLA time between calls does not hold a slot
#   --dashboard-ports LIST     one dashboard port per worker (-e dashboard --dashboard-port <port>)
#   --pass-k LIST              k values for Pass@k (default 1 and the seeds per task)
#   --min-success N|P%         exit nonzero when a variant has fewer than N successes (or a success
#                              rate below P%), like CaP-X's regression_test.sh
#
# Env servers need no port setting: each pi starts its own on a free port (robot.ts serve()). Shared
# services (VLA, SAM3, RLDX, vLLM) are started once, before this script, and every worker uses them.
#
# GPU lock: with LOCK set (e.g. LOCK=/root/autodl-tmp/locks/gpu1.lock) a HEAVY robot's run takes the
# lock once, exclusively, before starting the workers and holds it until they have all finished; the
# workers and their pi processes and env servers do not hold it. Taking it per worker or per episode
# would serialize the workers (an exclusive flock admits one holder), and a worker waiting on it while
# its siblings run would deadlock with anything that waits for this run to finish. The flip side: a
# service this run needs must not be started under the same lock, or this script waits forever.
# Heavy: robolab, robodojo and behavior (Isaac Sim) and robotwin (cuRobo). Light: libero, maniskill, metaworld,
# robosuite, genesis and robocasa (an EGL, SAPIEN or Genesis renderer per episode, the planner off the
# GPU); they share the GPU with whatever else runs and never take LOCK (it is noted and ignored), so a
# lock queue of light jobs cannot form. Genesis and BEHAVIOR pick their GPU themselves (--backend /
# --gpu-id among the pi args), so --gpus only sets the workers' CUDA environment for them.
# Check the GPU's free memory before starting a light run on a shared GPU.
#
# The summary gives, per variant, the success rate over valid cells, Pass@k (the unbiased estimator
# 1 - C(n-c,k)/C(n,k) over a task's n valid seeds with c successes, averaged over the tasks with at
# least k valid seeds), and the invalid cells (env_error, planner_error, timeout, missing, duplicate)
# counted apart; it is also written to <out-dir>/summary.json. The exit status is nonzero when a
# variant has invalid cells, mixes configurations, or misses --min-success. RoboCasa's Target50
# validator score needs a serial `robocasa/eval.sh <dir> all` afterwards (it skips every valid cell).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
usage() { echo "usage: eval-parallel.sh [-j N] [--gpus LIST] [--variant NAME=ARGS]... [--max-api-concurrency M] [--dashboard-ports LIST] [--pass-k LIST] [--min-success N|P%] <robot> <out-dir> <selection...> [pi args...]" >&2 && exit 2; }
die() { echo "eval-parallel.sh: $*" >&2 && exit 2; }

workers=1 gpus=${CUDA_VISIBLE_DEVICES-} api=0 ports="" passk="" minsuccess=""
vnames=() vargs=()
while [ $# -gt 0 ]; do
	case $1 in
	-j | --workers) workers=${2:-} && shift 2 ;;
	--workers=*) workers=${1#*=} && shift ;;
	--gpus) gpus=${2-} && shift 2 ;;
	--gpus=*) gpus=${1#*=} && shift ;;
	--max-api-concurrency) api=${2:-} && shift 2 ;;
	--max-api-concurrency=*) api=${1#*=} && shift ;;
	--dashboard-ports) ports=${2:-} && shift 2 ;;
	--dashboard-ports=*) ports=${1#*=} && shift ;;
	--pass-k) passk=${2:-} && shift 2 ;;
	--pass-k=*) passk=${1#*=} && shift ;;
	--min-success) minsuccess=${2:-} && shift 2 ;;
	--min-success=*) minsuccess=${1#*=} && shift ;;
	--variant) v=${2:-} && shift 2 && [[ $v == *=* ]] || die "--variant takes NAME=ARGS"
		vnames+=("${v%%=*}") && vargs+=("${v#*=}") ;;
	--variant=*) v=${1#*=} && shift && [[ $v == *=* ]] || die "--variant takes NAME=ARGS"
		vnames+=("${v%%=*}") && vargs+=("${v#*=}") ;;
	-h | --help) usage ;;
	-*) die "unknown option $1 (pi args go after the selection)" ;;
	*) break ;;
	esac
done
[ $# -ge 2 ] || usage
robot=$1 out=$2
shift 2
case $robot in
libero) npos=3 ;;
maniskill | robolab | robodojo | metaworld | robosuite | genesis | behavior) npos=2 ;;
robotwin | robocasa) npos=1 ;;
*) die "unknown robot $robot (libero, maniskill, metaworld, robosuite, genesis, behavior, robolab, robodojo, robotwin, robocasa)" ;;
esac
[ $# -ge $npos ] || die "$robot takes $npos selection arguments"
sel=("${@:1:npos}")
shift "$npos"
common=("$@")
[[ $workers =~ ^[1-9][0-9]*$ ]] || die "-j takes a positive integer"
[[ $api =~ ^[0-9]+$ ]] || die "--max-api-concurrency takes an integer"
[[ -z $minsuccess || $minsuccess =~ ^[0-9]+(\.[0-9]+)?%?$ ]] || die "--min-success takes N or P%"
[[ -z $passk || $passk =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || die "--pass-k takes a list of positive integers"
gpus=${gpus//,/ } ports=${ports//,/ }
read -ra gpus <<<"$gpus"
read -ra ports <<<"$ports"
for g in ${gpus[@]+"${gpus[@]}"}; do [[ $g =~ ^[0-9]+$ ]] || die "--gpus takes nvidia-smi indices, not '$g'"; done
[ ${#ports[@]} -eq 0 ] || [ ${#ports[@]} -ge "$workers" ] || die "--dashboard-ports needs one port per worker ($workers)"
if [ ${#vnames[@]} -eq 0 ]; then
	vnames=("") vargs=("")
else
	for ((i = 0; i < ${#vnames[@]}; i++)); do
		[[ ${vnames[i]} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "variant name '${vnames[i]}' must be [A-Za-z0-9._-]"
		for ((j = 0; j < i; j++)); do [ "${vnames[i]}" != "${vnames[j]}" ] || die "variant ${vnames[i]} given twice"; done
	done
fi
script=$here/$robot/eval.sh
SERVICES=${PI_EMBODIED_SERVICES:-$(cd "$here/../../../services" && pwd)}

expand() { for part in ${1//,/ }; do seq "${part%-*}" "${part#*-}"; done; }
units() { # one line per unit: <task key> TAB <cell dirs> TAB <env assignments or -> TAB <eval.sh selection>
	case $robot in
	libero) for t in $(expand "$2"); do for s in $(expand "$3"); do
		printf '%s\t%s\t-\t%s\n' "${1}_t$t" "${1}_t${t}_s$s" "$1 $t $s"
	done; done ;;
	maniskill | robolab | robodojo | metaworld | robosuite | genesis | behavior)
		# "-" is each eval.sh's default task set; the cells must be listed here by name.
		local tasks=$1
		[ "$robot:$tasks" = maniskill:- ] && tasks=BlockPAP-v1
		[ "$robot:$tasks" = metaworld:- ] && tasks=reach-v3
		[ "$robot:$tasks" = robosuite:- ] && tasks=Lift,Stack,Restack,Wipe,NutAssemblySquare,TwoArmLift,TwoArmHandover
		[ "$robot:$tasks" = genesis:- ] && tasks=cube_pick
		[ "$robot:$tasks" = behavior:- ] && die "behavior takes explicit task names (its eval.sh has no default set)"
		for t in ${tasks//,/ }; do for s in $(expand "$2"); do printf '%s\t%s\t-\t%s\n' "$t" "${t}_s$s" "$t $s"; done; done ;;
	robotwin) node -e '
const table = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8")).tasks;
const want = process.argv[2] === "all" ? Object.keys(table) : process.argv[2].split(",");
for (const t of want) {
	if (!table[t]) throw new Error(`unknown task ${t}`);
	console.log(`${t}\t${table[t].map((e) => `${t}_s${e.seed}`).join(" ")}\t-\t${t}`);
}' "$SERVICES/pi_embodied_services/robots/robotwin/eval/demo_randomized.json" "$1" ;;
	robocasa) node -e '
const m = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
const only = (s) => (s ? s.split(",") : null);
const [tasks, seeds] = [only(process.env.TASKS), only(process.env.SEEDS)];
const splits = process.argv[2] === "all" ? ["atomic", "composite_seen", "composite_unseen"] : process.argv[2].split(",");
for (const name of splits) {
	const split = m.splits[name];
	if (!split) throw new Error(`unknown Target50 split ${name}`);
	for (const t of split.tasks)
		for (const s of split.seeds)
			if ((!tasks || tasks.includes(t)) && (!seeds || seeds.includes(String(s))))
				console.log(`${name}/${t}\t${name}/${t}_s${s}\tTASKS=${t} SEEDS=${s}\t${name}`);
}' "${TARGET50:-$SERVICES/pi_embodied_services/robots/robocasa/eval/target50.json}" "$1" ;;
	esac
}

state=$out/.parallel
mkdir -p "$state"
if [ -f "$state/pid" ] && kill -0 "$(cat "$state/pid")" 2>/dev/null; then
	die "$out is being run by pid $(cat "$state/pid")"
fi
# A run whose shell died without its stop handler leaves its workers' process groups (w*.pid) running.
live=$(cat "$state"/w*.pid 2>/dev/null | while read -r g; do pgrep -g "$g"; done | tr '\n' ' ')
[ -z "$live" ] || die "$out still has episodes of an earlier run (pids $live); stop them first (kill -TERM -- -<pgid> per $state/w*.pid)"
echo $$ >"$state/pid"
rm -rf "$state/claims" "$state/api" "$state/abort" "$state"/w*.log "$state"/w*.pid
mkdir -p "$state/claims"
unitlist=$(units "${sel[@]}") || die "cannot list the cells of $robot ${sel[*]}"
[ -n "$unitlist" ] || die "no cells match $robot ${sel[*]}"
# Jobs are unit-major, variant-minor: an interrupted A/B has run the same cells in every variant.
while IFS=$'\t' read -r key dirs envs args; do
	for ((v = 0; v < ${#vnames[@]}; v++)); do printf '%s\t%s\t%s\t%s\t%s\n' "$v" "$key" "$dirs" "$envs" "$args"; done
done <<<"$unitlist" >"$state/jobs"

# CUDA ordinal -> EGL device index, aligned on the PCI bus id.
egl=()
if [ ${#gpus[@]} -gt 0 ] && command -v nvidia-smi >/dev/null; then
	devices=$(env -u CUDA_VISIBLE_DEVICES PYOPENGL_PLATFORM=egl "${EGL_PYTHON:-${PI_EMBODIED_PYTHON:-python3}}" -c '
import os
from mujoco.egl import egl_ext as E
from OpenGL.EGL.EXT.device_query import eglQueryDeviceStringEXT as query
for i, d in enumerate(E.eglQueryDevicesEXT()):
    try:
        f = query(d, 0x3233)  # EGL_DRM_DEVICE_FILE_EXT
    except Exception:
        f = None
    print(i, os.path.basename(f.decode()) if f else "-")
' 2>/dev/null)
	while IFS=', ' read -r index bus; do
		[[ $index =~ ^[0-9]+$ ]] || continue
		bus=$(echo "${bus: -12}" | tr 'A-F' 'a-f')
		card=$(readlink "${DRI_BY_PATH:-/dev/dri/by-path}/pci-$bus-card") || continue
		e=$(awk -v c="${card##*/}" '$2 == c { print $1; exit }' <<<"$devices")
		[ -n "$e" ] && egl[index]=$e
	done < <(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null)
fi
for g in ${gpus[@]+"${gpus[@]}"}; do
	[ -n "${egl[g]:-}" ] || echo "eval-parallel.sh: no EGL device found on GPU $g's PCI bus; MUJOCO_EGL_DEVICE_ID stays unset and MuJoCo may render on another GPU" >&2
done

if [ -n "${LOCK:-}" ]; then
	case $robot in
	robolab | robodojo | robotwin | behavior)
		command -v flock >/dev/null || die "LOCK is set but flock is missing"
		exec 9>"$LOCK"
		flock -n 9 || { echo "$(date +%T) waiting for $LOCK" && flock 9; } || die "cannot lock $LOCK"
		;;
	*) echo "eval-parallel.sh: $robot is a light job (renderer only, planner off the GPU); LOCK is for robolab, robodojo, behavior and robotwin and is not taken" >&2 ;;
	esac
fi

worker() { # <k>
	local k=$1 gpu="" line v key dirs envs args rc log=$state/w$1.log
	local extra=()
	if [ ${#gpus[@]} -gt 0 ]; then
		gpu=${gpus[k % ${#gpus[@]}]}
		export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
		[ -n "${egl[gpu]:-}" ] && export MUJOCO_EGL_DEVICE_ID=${egl[gpu]}
		# robosuite asserts at import that MUJOCO_EGL_DEVICE_ID is among CUDA_VISIBLE_DEVICES (it takes the EGL
		# index for a CUDA ordinal). Widening CUDA_VISIBLE_DEVICES to <gpu>,<egl> would satisfy it but expose a
		# second GPU to the worker; instead the MuJoCo env servers get --cuda-device, clear CUDA_VISIBLE_DEVICES
		# for themselves before importing robosuite and pin torch with set_device.
		case $robot in libero | robocasa | robolab | robodojo | robosuite) extra+=(--cuda-device "$gpu") ;; esac
	fi
	[ "$api" -gt 0 ] && extra+=(-e "$here/api-gate.ts" --api-slots "$state/api" --max-api-concurrency "$api")
	[ ${#ports[@]} -gt 0 ] && extra+=(-e "$here/dashboard" --dashboard=true --dashboard-port "${ports[k]}")
	echo "[w$k] gpu=${gpu:--} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID-}" >>"$log"
	local i=0
	while IFS=$'\t' read -r v key dirs envs args; do
		i=$((i + 1))
		[ -e "$state/abort" ] && return
		mkdir "$state/claims/$i" 2>/dev/null || continue
		local vout=$out mine=()
		[ -n "${vnames[v]}" ] && vout=$out/${vnames[v]}
		read -ra mine <<<"${vargs[v]}"
		[ "$envs" = - ] && envs=""
		# Runner args first: the user's args follow, so theirs win and a trailing bare flag stays last.
		# shellcheck disable=SC2086
		env $envs bash "$script" "$vout" $args ${extra[@]+"${extra[@]}"} ${common[@]+"${common[@]}"} ${mine[@]+"${mine[@]}"} \
			</dev/null >"$state/w$k.call" 2>&1
		rc=$?
		cat "$state/w$k.call" >>"$log"
		if grep -q "use another out dir" "$state/w$k.call" || [ "$rc" -eq 2 ]; then
			{ echo "${vnames[v]:-.} $args:" && cat "$state/w$k.call"; } >>"$state/abort"
			return
		fi
		local status="" d
		for d in $dirs; do
			status+=" $(sed -n 's/^  "status": "\(.*\)",$/\1/p' "$vout/$d/result.json" 2>/dev/null)"
		done
		echo "[w$k]${vnames[v]:+ ${vnames[v]}} $key ($dirs):$status"
	done <"$state/jobs"
}

pids=()
stop() { # on INT, TERM or HUP (an ssh drop): no episode may outlive this run, or a rerun races it for the cells
	trap - INT TERM HUP
	local g groups
	# Each worker is its own process group (set -m): eval.sh, pi and the env servers go with it.
	for g in ${pids[@]+"${pids[@]}"}; do kill -TERM -- "-$g" 2>/dev/null; done
	wait
	groups=$(IFS=, && echo "${pids[*]}")
	for ((i = 0; i < 300; i++)); do
		pgrep -g "$groups" >/dev/null || break
		sleep 0.1
	done
	for g in ${pids[@]+"${pids[@]}"}; do kill -KILL -- "-$g" 2>/dev/null; done
	echo "interrupted; rerun the same command to finish the invalid cells" >&2
	rm -f "$state/pid" "$state"/w*.pid
	exit 130
}
trap stop INT TERM HUP
echo "$(wc -l <"$state/jobs" | tr -d ' ') jobs (${#vnames[@]} variant(s)) on $workers worker(s); logs in $state"
set -m
for ((k = 0; k < workers; k++)); do
	worker "$k" 9>&- </dev/null &
	pids+=($!)
	echo $! >"$state/w$k.pid"
done
set +m
wait
trap - INT TERM HUP
rm -f "$state/pid" "$state"/w*.pid
aborted=0
if [ -s "$state/abort" ]; then
	cat "$state/abort" >&2
	aborted=1
fi

vdirs=()
for n in "${vnames[@]}"; do vdirs+=("${n:-.}"); done
node --input-type=module -e '
import { readFileSync, writeFileSync } from "node:fs";
const [out, jobsFile, passk, minsuccess, aborted, ...vdirs] = process.argv.slice(1);
const jobs = readFileSync(jobsFile, "utf8").trim().split("\n").map((l) => l.split("\t"));
const read = (p) => {
	try {
		return JSON.parse(readFileSync(p, "utf8"));
	} catch {
		return { status: "missing" };
	}
};
const valid = (r) => r.status === "success" || r.status === "failure";
// Pass@k over n valid attempts with c successes (Chen et al. 2021): 1 - C(n-c, k) / C(n, k).
const pass = (n, c, k) => {
	if (n - c < k) return 1;
	let q = 1;
	for (let i = n - c + 1; i <= n; i++) q *= 1 - k / i;
	return 1 - q;
};
const pct = (x) => (x === null ? "-" : `${(100 * x).toFixed(1)}%`);
let failed = Number(aborted) > 0;
const summary = { out, variants: [] };
vdirs.forEach((vdir, v) => {
	const cells = jobs.filter((j) => Number(j[0]) === v).flatMap(([, key, dirs]) => dirs.split(" ").map((d) => ({ key, d })));
	const rows = cells.map(({ key, d }) => ({ key, ...read(`${out}/${vdir}/${d}/result.json`) }));
	const configs = new Set(rows.filter(valid).map((r) =>
		[r.model, r.thinking, `turns=${r.max_turns}`, r.time_limit === undefined ? "" : `limit=${r.time_limit}`, `units=${r.units}`,
			r.stateless ? "stateless" : "", r.anchor_image ? "anchor" : "",
			r.unit_tol === undefined ? "" : `unit_tol=${r.unit_tol}`,
			r.vdm ? `vdm=${r.vdm_model ?? "default"}${r.vdm_wrist ? "+wrist" : ""}` : "", r.privileged ? "privileged" : "",
			r.fallback_model ? `fallback=${r.fallback_model}:${r.fallback_after}:${r.fallback_retry_primary}` : "",
			r.code && r.code !== "false" ? `code=${r.code}:${r.code_api}` : "",
			r.max_move === undefined ? "" : `max_move=${r.max_move}`, r.grasping_mode ? `grasp=${r.grasping_mode}` : "",
			r.protocol_id ?? ""].filter(Boolean).join("/")));
	const name = vdir === "." ? "-" : vdir;
	if (configs.size > 1) {
		console.log(`${name}: refusing to summarize: ${out}/${vdir} mixes configurations ${[...configs].join(", ")}`);
		failed = true;
		return;
	}
	const n = (s) => rows.filter((r) => r.status === s).length;
	const scored = n("success") + n("failure");
	const tasks = new Map();
	for (const r of rows) {
		const t = tasks.get(r.key) ?? { n: 0, c: 0, cells: 0 };
		t.cells++;
		if (valid(r)) t.n++;
		if (r.status === "success") t.c++;
		tasks.set(r.key, t);
	}
	const ks = passk ? passk.split(",").map(Number) : [...new Set([1, Math.max(...[...tasks.values()].map((t) => t.cells))])];
	const passAt = ks.map((k) => {
		const eligible = [...tasks.values()].filter((t) => t.n >= k);
		const value = eligible.length ? eligible.reduce((a, t) => a + pass(t.n, t.c, k), 0) / eligible.length : null;
		return { k, value, tasks: eligible.length };
	});
	const invalid = { env_error: n("env_error"), planner_error: n("planner_error"), timeout: n("timeout"), missing: n("missing"), duplicate_result: n("duplicate_result") };
	const invalidCount = rows.length - scored;
	const rate = scored ? n("success") / scored : null;
	const row = { variant: name, dir: `${out}/${vdir}`, config: [...configs][0] ?? null, cells: rows.length, success: n("success"),
		scored, success_rate: rate, claimed_but_failed: rows.filter((r) => r.status === "failure" && r.claimed === "success").length,
		pass_at_k: passAt, invalid: invalidCount, invalid_by_status: invalid, tasks: tasks.size };
	let below = false;
	if (minsuccess) {
		below = minsuccess.endsWith("%") ? (rate ?? 0) * 100 < Number(minsuccess.slice(0, -1)) : n("success") < Number(minsuccess);
		row.min_success = { threshold: minsuccess, passed: !below };
	}
	summary.variants.push(row);
	const pk = passAt.map((p) => `pass@${p.k} ${pct(p.value)} (${p.tasks}/${tasks.size} tasks)`).join(", ");
	console.log(`${name}: ${row.config ?? "-"}: success ${n("success")}/${scored} (${pct(rate)}), ${pk}, claimed-but-failed ${row.claimed_but_failed}, invalid ${invalidCount} (${Object.entries(invalid).map(([s, c]) => `${s} ${c}`).join(", ")}) of ${rows.length}`);
	if (below) console.log(`${name}: below --min-success ${minsuccess}`);
	if (below || invalidCount) failed = true;
});
writeFileSync(`${out}/summary.json`, `${JSON.stringify(summary, null, 2)}\n`);
process.exit(failed ? 1 : 0);
' "$out" "$state/jobs" "$passk" "$minsuccess" "$aborted" "${vdirs[@]}"
