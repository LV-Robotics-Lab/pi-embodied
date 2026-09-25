#!/usr/bin/env bash
# Install the pi-embodied services for one robot: a venv with the matching pyproject extra, the
# robot's assets, and (with --weights) its model checkpoints. Idempotent: an existing venv, an
# applied patch or a complete download is kept. Prints every command it runs; stops at the first
# failure.
#
#   services/setup.sh <target> [--venv DIR] [--weights] [--weights-dir DIR] [--no-assets] [--dry-run]
#
# Targets (the extra, the Python, and what else it fetches):
#   libero          [libero] py3.11                      --weights: Pi0.5 LIBERO SFT, SAM3 (gated)
#   libero-pro      [libero-pro] py3.11, LIBERO-PRO assets (patched downloader)   --weights: as libero
#   libero-plus     [libero-plus] py3.11, LIBERO-plus assets (~6.4 GB zip, 9.5 GB unpacked; needs system
#                   ImageMagick, e.g. apt install libmagickwand-6.q16-6)   --weights: as libero
#   robocasa        [robocasa] py3.10, kitchen assets (~10 GB, robocasa-download-assets)   --weights: RLDX-1-FT-RC365 + RLDX-1-VLM
#   robotwin        [robotwin] py3.11, RoboTwin assets   --weights: LingBot-VLA RoboTwin EEF
#   maniskill       [maniskill] py3.11, real2sim rigs (robots/maniskill/fetch_real2sim.sh)
#   robolab         Isaac Sim 6.1 venv + patched RoboLab (robots/robolab/install_isaac61.sh)
#   franka          [franka,sam3] py3.11 (real arm; RLinf controller stack and Ray on the box)
#   franka-polymetis [franka-polymetis] py3.10 (real arm on a Polymetis NUC)
#   dual-franka     [franka,sam3] py3.11 (two real arms)
#   piper           [piper] system python with --system-site-packages (source ROS Noetic first)
#   finetuned       Show-Harness adapter + base model (finetuned/download.py; FT_ADAPTER, default qwen3_5_2b_sim)
#   llamafactory    LLaMA-Factory training venv (finetuned/setup_llamafactory.sh)
#
# Defaults: --venv services/.venv-<target>, --weights-dir $PI_EMBODIED_WEIGHTS (default
# ~/.cache/pi-embodied), assets under the same directory. Downloads honour HF_ENDPOINT (e.g.
# https://hf-mirror.com) and HF_TOKEN, and pip/uv honour UV_INDEX_URL / PIP_INDEX_URL. --dry-run
# prints the commands without running them. At the end the environment the robot reads is written
# to <venv>/pi-embodied.env (source it before pi / serve.sh / eval.sh) and the preflight command
# is printed.
set -euo pipefail

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
die() { echo "setup.sh: $*" >&2; exit 1; }

SERVICES=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG=$(cd "$SERVICES/../packages/embodied" 2>/dev/null && pwd || true)
target=""
venv=""
weights=false
assets=true
dry=false
wdir=${PI_EMBODIED_WEIGHTS:-$HOME/.cache/pi-embodied}
while [ $# -gt 0 ]; do
	case $1 in
	--venv) venv=${2:?--venv DIR}; shift ;;
	--venv=*) venv=${1#*=} ;;
	--weights) weights=true ;;
	--weights-dir) wdir=${2:?--weights-dir DIR}; shift ;;
	--weights-dir=*) wdir=${1#*=} ;;
	--no-assets) assets=false ;;
	--dry-run) dry=true ;;
	-h | --help) usage; exit 0 ;;
	-*) die "unknown option $1 (see --help)" ;;
	*) [ -z "$target" ] || die "one target only, got $target and $1"; target=$1 ;;
	esac
	shift
done
[ -n "$target" ] || { usage >&2; exit 2; }

# The extra, the Python version ("system" = the host python3 with its site packages).
case $target in
libero | libero-pro | libero-plus) extra=$target py=3.11 ;;
robocasa) extra=robocasa py=3.10 ;;
robotwin) extra=robotwin py=3.11 ;;
maniskill) extra=maniskill py=3.11 ;;
franka | dual-franka) extra=franka,sam3 py=3.11 ;;
franka-polymetis) extra=franka-polymetis py=3.10 ;;
piper) extra=piper py=system ;;
robolab | finetuned | llamafactory) extra="" py="" ;;
*) die "unknown target '$target' (see --help)" ;;
esac
venv=${venv:-$SERVICES/.venv-$target}
PY=$venv/bin/python
ENV_LINES=()

# run CMD...: print it, then run it unless --dry-run.
run() {
	printf '+'
	printf ' %q' "$@"
	printf '\n'
	$dry || "$@"
}
note() { echo "== $*"; }
export_env() { ENV_LINES+=("export $1=$(printf '%q' "$2")"); }

make_venv() {
	if [ -x "$PY" ]; then
		note "venv $venv exists ($("$PY" -V 2>&1))"
		return
	fi
	local site=()
	if [ "$py" = system ]; then
		site=(--system-site-packages)
		py=$(command -v python3) || die "python3 not found"
	fi
	if command -v uv >/dev/null; then
		run uv venv ${site[@]+"${site[@]}"} --python "$py" "$venv"
	else
		local exe=$py
		[ "$py" = "${py#/}" ] && exe=python$py
		command -v "$exe" >/dev/null || die "no uv and no $exe: install uv (https://docs.astral.sh/uv/) or Python $py"
		run "$exe" -m venv ${site[@]+"${site[@]}"} "$venv"
		run "$venv/bin/python" -m pip install -U pip
	fi
}

pip_install() {
	if command -v uv >/dev/null; then
		run uv pip install --python "$PY" "$@"
	else
		run "$PY" -m pip install "$@"
	fi
}

# hf_get REPO REV DIR [FILE...]: a Hugging Face snapshot (or some files of it) into DIR
# (DIR "-" = the HF cache). huggingface_hub reads HF_ENDPOINT and HF_TOKEN itself.
hf_get() {
	local repo=$1 rev=$2 dir=$3
	shift 3
	local cli=$venv/bin/hf
	if [ ! -x "$cli" ] && ! $dry; then
		pip_install "huggingface_hub>=0.34"
	fi
	local args=(download "$repo" "$@" --revision "$rev")
	[ "$dir" = - ] || args+=(--local-dir "$dir")
	run "$cli" "${args[@]}" ||
		die "download of $repo failed (gated repos need HF_TOKEN and an accepted license; set HF_ENDPOINT for a mirror)"
}

libero_weights() {
	local pi05=$wdir/RLinf-Pi05-LIBERO-130-fullshot-SFT sam3=$wdir/sam3
	hf_get RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT 6222623f635769bfc73c9472e29fab9b7fd8e027 "$pi05"
	hf_get facebook/sam3 3c879f39826c281e95690f02c7821c4de09afae7 "$sam3" sam3.pt
	export_env PI05_CHECKPOINT_PATH "$pi05"
	export_env SAM3_CHECKPOINT_PATH "$sam3/sam3.pt"
}

# LIBERO-PRO's downloader reports a partial snapshot as ready; the patch verifies every file first.
liberopro_assets() {
	local patch=$SERVICES/pi_embodied_services/robots/libero/liberopro-verify-assets.patch site
	if $dry; then
		note "patch the installed liberopro downloader with $patch"
	else
		site=$("$PY" -c 'import liberopro, os; print(os.path.dirname(os.path.dirname(liberopro.__file__)))') ||
			die "liberopro is not importable from $PY"
		if grep -q _verify_complete "$site/liberopro/liberopro/utils/download_utils.py"; then
			note "liberopro downloader already verifies downloads"
		else
			run patch -p1 -d "$site" -i "$patch"
		fi
	fi
	run "$venv/bin/liberopro-download-assets" --skip-existing
}

# liberoplus-download-assets rejects the published assets.zip (its tree sits under a deep build path,
# not at the archive root), so fetch and unpack it here and link the package to it. The package
# imports Wand, which needs the system ImageMagick library.
liberoplus_assets() {
	local dir=$wdir/liberoplus-assets
	if [ -d "$dir/scenes" ]; then
		note "LIBERO-plus assets present in $dir"
	else
		hf_get Sylvest/LIBERO-plus dd2bd61b7d9a6fef1abc52d606e983b41886a149 "$wdir/liberoplus-zip" assets.zip --repo-type dataset
		run "$PY" -c 'import os, shutil, sys, zipfile
z, d = sys.argv[1:]
t = d + ".tmp"
shutil.rmtree(t, True)
zipfile.ZipFile(z).extractall(t)
os.replace(next(r for r, ds, _ in os.walk(t) if "scenes" in ds), d)
shutil.rmtree(t)' "$wdir/liberoplus-zip/assets.zip" "$dir"
	fi
	run "$venv/bin/liberoplus-download-assets" --link "$dir"
	$dry || "$PY" -c 'import wand.image' 2>/dev/null ||
		die "Wand cannot load ImageMagick: install it (Debian/Ubuntu: apt install libmagickwand-6.q16-6)"
}

note "target $target, services $SERVICES, venv $venv$($dry && echo ' (dry run)')"
case $target in
robolab)
	root=${ROBOLAB_ROOT:-$HOME/RoboLab}
	run bash "$SERVICES/pi_embodied_services/robots/robolab/install_isaac61.sh" "$venv" "$root"
	export_env ROBOLAB_ROOT "$root"
	export_env OMNI_KIT_ACCEPT_EULA YES
	;;
finetuned)
	command -v python3 >/dev/null || die "python3 not found"
	adapter=${FT_ADAPTER:-qwen3_5_2b_sim}
	(cd "$SERVICES" && run python3 -m pi_embodied_services.finetuned.download --adapter "$adapter" --dest "$wdir")
	note "serve it: MODEL=$wdir/<base> LORA=<name>=$wdir/Show-Harness-VLMs/$adapter VLLM_VENV=<venv with vllm> bash $SERVICES/pi_embodied_services/finetuned/serve.sh"
	;;
llamafactory)
	# The script's defaults are the AutoDL box's; outside it, keep its venv, caches and uv local.
	command -v uv >/dev/null || die "llamafactory needs uv"
	run env LF_VENV="${LF_VENV:-$venv}" UV="${UV:-$(command -v uv)}" \
		UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}" UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$HOME/.local/share/uv/python}" \
		WHEELS="${WHEELS:-$wdir/wheels/cu129}" ${CUDA_HOME:+CUDA_HOME="$CUDA_HOME"} \
		bash "$SERVICES/pi_embodied_services/finetuned/setup_llamafactory.sh"
	;;
*)
	make_venv
	constraint=()
	if [ "$target" = robocasa ]; then
		constraint=(--constraint "$SERVICES/pi_embodied_services/robots/robocasa/eval/target50-constraints.txt")
	fi
	pip_install -e "$SERVICES[$extra]" ${constraint[@]+"${constraint[@]}"}
	case $target in
	libero) if $weights; then libero_weights; fi ;;
	libero-pro | libero-plus)
		if $assets; then ${target//-/}_assets; fi
		if $weights; then libero_weights; fi
		;;
	robocasa)
		if $assets; then
			run "$venv/bin/robocasa-download-assets" --assets-path "$wdir/robocasa/assets" \
				--macros-path "$wdir/robocasa/macros_private.py" --skip-existing -y
			export_env ROBOCASA_ASSETS_PATH "$wdir/robocasa/assets"
			export_env ROBOCASA_MACROS_PATH "$wdir/robocasa/macros_private.py"
		fi
		if $weights; then
			hf_get RLWRLD/RLDX-1-FT-RC365 587e9ecdcc5e7184fcc17f58713908edff5af041 "$wdir/RLDX-1-FT-RC365"
			hf_get RLWRLD/RLDX-1-VLM 4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7 -
			export_env RLDX_MODEL_PATH "$wdir/RLDX-1-FT-RC365"
		fi
		export_env ROBOCASA_PYTHON "$PY"
		;;
	robotwin)
		if $assets; then
			if [ -e "$wdir/robotwin-assets/embodiments" ]; then
				note "RoboTwin assets present in $wdir/robotwin-assets"
			else
				run "$venv/bin/robotwin-download-assets" --output "$wdir/robotwin-assets"
			fi
			export_env ROBOTWIN_ASSETS_PATH "$wdir/robotwin-assets"
		fi
		if $weights; then
			hf_get RLinf/LingBot-VLA-RoboTwin-EEF-ckpt1500 e727b46cd220b66981ea4d2fd9ba84adc189e2cc "$wdir/LingBot-VLA-RoboTwin-EEF-ckpt1500"
			export_env LINGBOT_MODEL_PATH "$wdir/LingBot-VLA-RoboTwin-EEF-ckpt1500"
		fi
		;;
	maniskill)
		if $assets; then
			run env PYTHON="$PY" bash "$SERVICES/pi_embodied_services/robots/maniskill/fetch_real2sim.sh" "$wdir/RLinf-real2sim"
			export_env RLINF_ROOT "$wdir/RLinf-real2sim"
		fi
		;;
	esac
	;;
esac

case $target in finetuned | llamafactory) ;; *) export_env PI_EMBODIED_PYTHON "$PY" ;; esac
export_env PI_EMBODIED_SERVICES "$SERVICES"
note "environment for pi, serve.sh and eval.sh:"
printf '%s\n' "${ENV_LINES[@]}"
if ! $dry && [ -d "$venv" ]; then
	printf '%s\n' "${ENV_LINES[@]}" >"$venv/pi-embodied.env"
	note "wrote $venv/pi-embodied.env"
fi
case $target in
franka-polymetis) robot=franka ;;
dual-franka) robot=dual_franka ;;
libero-pro | libero-plus) robot=libero ;;
finetuned | llamafactory) robot="" ;;
*) robot=$target ;;
esac
if [ -n "$robot" ] && [ -n "$PKG" ]; then
	note "preflight: source $venv/pi-embodied.env && node $PKG/src/check.ts $robot --python $PY --services $SERVICES"
	note "in pi with the robot loaded: /robot-check"
fi
