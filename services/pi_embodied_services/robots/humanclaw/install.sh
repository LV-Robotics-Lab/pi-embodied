#!/usr/bin/env bash
# The humanclaw venv (services/setup.sh humanclaw): Python 3.10, HumanCLAW @c4f9351 with its
# constraints/eval-cu124.txt, patched Habitat-Sim (build_habitat.sh), the paper_fullval_v1 motion
# weights (sha256-checked, then `humanclaw-bench assets`), and the prepared HSSD val41 scenes.
#
#   install.sh <venv> <humanclaw-dir> [hssd-root]
#
# torch: HumanCLAW pins torch==2.6.0 cu124 (HUMANCLAW_TORCH, default "torch==2.6.0" from
# HUMANCLAW_TORCH_INDEX https://download.pytorch.org/whl/cu124). cu124 has no sm_120 kernels: on an
# RTX 50xx set HUMANCLAW_TORCH="torch==2.7.1+cu128" and HUMANCLAW_TORCH_INDEX to a cu128 index (a
# deviation from the paper environment; record it). HSSD: `hssd/hssd-hab` and the supplement
# `HumanCLAW/HumanCLAW-HSSD` are gated on Hugging Face; request access on both dataset pages first
# (`hf auth login` alone gets HTTP 403). Without hssd-root, fetch_hssd.py downloads the ~3.8 GB of
# official files the 41 scenes use into <humanclaw-dir>/../hssd-hab. HF_ENDPOINT / HF_TOKEN are honoured.
# The weights tarball is sha256-checked before use; a file that fails is removed and fetched once more.
set -euo pipefail
venv=${1:?usage: install.sh <venv> <humanclaw-dir> [hssd-root]}
hc=${2:?humanclaw dir}
hssd=${3:-$(dirname "$hc")/hssd-hab}
here=$(cd "$(dirname "$0")" && pwd)
SERVICES=$(cd "$here/../../.." && pwd)
PIN=c4f9351
WEIGHTS=HumanCLAW_pretrained_weights_paper_fullval_v1_20260816.tar.gz
WEIGHTS_SHA256=3b3c0c1b232af4c462301655de909a4bc54fd4756bcd42c22e7965fccb650667
# The echo hides a URL's userinfo (UV_INDEX_URL / HUMANCLAW_TORCH_INDEX may carry a token): logs are shared.
run() { { printf '+'; printf ' %q' "$@"; printf '\n'; } | sed -E 's#://[^/@[:space:]]+@#://***@#g'; "$@"; }
git_() { if [ -n "${GIT_PROXY:-}" ]; then git -c http.proxy="$GIT_PROXY" "$@"; else git "$@"; fi; }
pipi() { if command -v uv >/dev/null; then run uv pip install --python "$venv/bin/python" "$@"; else run "$venv/bin/python" -m pip install "$@"; fi; }

[ -d "$hc/.git" ] || run git_ clone https://github.com/Human-CLAW/HumanCLAW "$hc"
[ "$(git -C "$hc" rev-parse --short=7 HEAD)" = $PIN ] || run git -C "$hc" checkout -q $PIN
if [ ! -x "$venv/bin/python" ]; then
	if command -v uv >/dev/null; then run uv venv --python 3.10 "$venv"; else run python3.10 -m venv "$venv"; fi
fi
pipi --index-url "${HUMANCLAW_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}" ${UV_INDEX_URL:+--extra-index-url "$UV_INDEX_URL"} "${HUMANCLAW_TORCH:-torch==2.6.0}" pip setuptools wheel
cons=$venv/humanclaw-constraints.txt
if [ -n "${HUMANCLAW_TORCH:-}" ]; then grep -v -E '^(torch|triton|sympy)==' "$hc/constraints/eval-cu124.txt" >"$cons"; else cp "$hc/constraints/eval-cu124.txt" "$cons"; fi
pipi -c "$cons" -e "$hc[rollout,test,video]"
pipi --no-deps -e "$SERVICES"
run bash "$here/build_habitat.sh" "$venv" "$hc"

cd "$(dirname "$hc")"
weights_ok() { echo "$WEIGHTS_SHA256  $WEIGHTS" | sha256sum -c --quiet >/dev/null 2>&1; }
if [ ! -f "$hc/weights/paper_fullval_v1/base/motion_dit.pt" ]; then
	# A tarball that fails its checksum (an interrupted download, a stale mirror copy) is replaced once,
	# not kept to fail every run; the fresh download must match or nothing is extracted.
	if [ -f $WEIGHTS ] && ! weights_ok; then
		echo "install.sh: $WEIGHTS fails its sha256 check (truncated or stale download); downloading it again" >&2
		rm -f $WEIGHTS
	fi
	[ -f $WEIGHTS ] || run "$venv/bin/hf" download HumanCLAW/HumanCLAW $WEIGHTS --local-dir .
	if ! weights_ok; then
		rm -f $WEIGHTS
		echo "install.sh: the downloaded $WEIGHTS does not match sha256 $WEIGHTS_SHA256 (HF_ENDPOINT=${HF_ENDPOINT:-huggingface.co}); removed it. Check the mirror or download the file from huggingface.co/HumanCLAW/HumanCLAW into $PWD and rerun" >&2
		exit 1
	fi
	run tar -xzf $WEIGHTS
	[ -d "$hc/weights/paper_fullval_v1" ] || { mkdir -p "$hc/weights" && run cp -r weights/paper_fullval_v1 "$hc/weights/"; }
fi
(cd "$hc" && run "$venv/bin/humanclaw-bench" assets --weights-root weights/paper_fullval_v1)
[ -f "$hssd/hssd-hab.scene_dataset_config.json" ] || run "$venv/bin/python" -m pi_embodied_services.robots.humanclaw.fetch_hssd --humanclaw "$hc" --out "$hssd"
(cd "$hc" && run "$venv/bin/humanclaw-bench" prepare-hssd --hssd-root "$hssd")
