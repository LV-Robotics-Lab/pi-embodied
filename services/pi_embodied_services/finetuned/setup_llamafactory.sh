#!/usr/bin/env bash
# Copyright 2026 The Show-Harness Authors. Licensed under the Apache License, Version 2.0.
# Modified by pi-embodied: train/scripts/setup_llamafactory.sh of github.com/showlab/Show-Harness @137d571
# for this box: one venv at /root/autodl-tmp/venvs/llamafactory with the pinned LLaMA-Factory checked out
# inside it, pip through China mirrors (torch from the SJTU pytorch-wheels mirror), qwen3_5/internvl3_5 only.
#
# Installs what train.sh needs (LLaMA-Factory LoRA SFT of the Show-Harness mvtoken VLMs):
#
#   bash setup_llamafactory.sh
#   LF_VENV=/root/autodl-tmp/venvs/llamafactory LF_ROOT=$LF_VENV/LlamaFactory   (the defaults)
#
# then  LF_VENV=... LF_ROOT=... are what train.sh passes to Show-Harness's train/scripts/train.sh.
# The pins are upstream's, each a worked-around bug (see their script): LLaMA-Factory 9ce6b66,
# torch 2.8.0+cu129 (its wheels carry sm_120 SASS, checked below), transformers 5.7.0 last,
# fla 0.5.1 --no-deps, tilelang 0.1.11 + apache-tvm-ffi 0.1.11.
set -euo pipefail

LF_UPSTREAM_URL="${LF_UPSTREAM_URL:-https://github.com/hiyouga/LLaMA-Factory.git}"
LF_UPSTREAM_PIN="${LF_UPSTREAM_PIN:-9ce6b663e9d87cd3c0cb42a1d3ff5cdfe292426d}"
TRANSFORMERS="${TRANSFORMERS:-5.7.0}"
FLASH_ATTN="${FLASH_ATTN:-2.8.3}"
LF_VENV="${LF_VENV:-/root/autodl-tmp/venvs/llamafactory}"
LF_ROOT="${LF_ROOT:-$LF_VENV/LlamaFactory}"
UV="${UV:-/root/autodl-tmp/tools/uvpkg/bin/uv}"
INDEX="${INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# torch is 1.2 GB and a single stream from any mirror crawls (0.2-1 MB/s here): aria2c it from aliyun first
TORCH_MIRROR="${TORCH_MIRROR:-https://mirrors.aliyun.com/pytorch-wheels/cu129}"
WHEELS="${WHEELS:-/root/autodl-tmp/.cache/wheels/cu129}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/.cache/uv}" UV_LINK_MODE=copy UV_HTTP_TIMEOUT=600
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/root/autodl-tmp/tools/uv-python}"
export UV_INDEX_URL="$INDEX" PIP_INDEX_URL="$INDEX"
# deepspeed's setup probes nvcc; a CUDA toolkit (no ops are built, DS_BUILD_OPS=0).
export CUDA_HOME="${CUDA_HOME:-/root/autodl-tmp/tools/cuda-12.8}" DS_BUILD_OPS=0
# No proxy: every download here is a China mirror or a GitHub release.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# 1. upstream source, only the pinned commit
# (GitHub from the box drops TLS now and then: retry; GIT_PROXY=http://127.0.0.1:1056 routes it via tailscale)
if [ ! -d "$LF_ROOT/src/llamafactory" ]; then
  git init -q "$LF_ROOT"
  git -C "$LF_ROOT" remote add origin "$LF_UPSTREAM_URL" 2> /dev/null || true
  for i in 1 2 3 4 5; do
    git ${GIT_PROXY:+-c http.proxy=$GIT_PROXY} -C "$LF_ROOT" fetch -q --depth 1 origin "$LF_UPSTREAM_PIN" && break
    [ $i = 5 ] && exit 1
    sleep 5
  done
  git -C "$LF_ROOT" checkout -q FETCH_HEAD
fi
echo "[setup] LLaMA-Factory $(git -C "$LF_ROOT" rev-parse --short HEAD) at $LF_ROOT"

# 2. venv (python 3.12, as upstream); UV_NO_SYNC keeps `uv run` from swapping the torch pin
if [ ! -x "$LF_VENV/bin/python" ]; then
  "$UV" venv --allow-existing --python 3.12 "$LF_VENV" --prompt lf-mvtoken
  printf '\nexport UV_NO_SYNC=1\n' >> "$LF_VENV/bin/activate"
fi
# shellcheck disable=SC1091
source "$LF_VENV/bin/activate"
pip() { "$UV" pip install --python "$LF_VENV/bin/python" "$@"; }

mkdir -p "$WHEELS"
for w in torch-2.8.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl torchvision-0.23.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl \
  torchaudio-2.8.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl; do
  # the sha256 download.pytorch.org publishes (its index links end in #sha256=...)
  pkg="${w%%-*}" enc="${w//+/%2B}"
  sha="$(curl -s -m 60 "https://download.pytorch.org/whl/cu129/$pkg/" | grep -o "$enc#sha256=[0-9a-f]*" | head -1 | cut -d= -f2)"
  [ -n "$sha" ] || { echo "no sha256 for $w on download.pytorch.org" >&2; exit 1; }
  if ! echo "$sha  $WHEELS/$w" | sha256sum -c --quiet 2> /dev/null; then
    # aliyun answers aria2's own user agent with 403
    aria2c -d "$WHEELS" -o "$w" -U curl/7.81.0 -x 16 -s 16 -k 8M --continue=true --max-tries=0 --retry-wait=5 \
      --console-log-level=warn --summary-interval=60 "$TORCH_MIRROR/$w"
    echo "$sha  $WHEELS/$w" | sha256sum -c
  fi
done
torch_pin() { pip torch==2.8.0+cu129 torchvision==0.23.0+cu129 torchaudio==2.8.0+cu129 --find-links "$WHEELS"; }
torch_pin
pip setuptools wheel packaging "hatchling>=1.18.0" editables
pip --no-build-isolation -e "$LF_ROOT"
# flash-attn from source for sm_120 only (upstream: `pip install flash-attn`, whose setup.py fetches a
# GitHub release wheel; from this box that download stalls at <30 KB/s and the wheel's arch list is
# not ours to check). nvcc must be >= 12.8 for sm_120; ~10 min with 32 jobs.
pip ninja psutil einops
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-120}" \
  MAX_JOBS="${MAX_JOBS:-32}" NVCC_THREADS=2 PATH="$CUDA_HOME/bin:$PATH" \
  pip "flash-attn==$FLASH_ATTN" --no-binary flash-attn --no-build-isolation
pip -r "$LF_ROOT/requirements/liger-kernel.txt"
pip -r "$LF_ROOT/requirements/deepspeed.txt"
pip -r "$LF_ROOT/requirements/metrics.txt"
pip wandb
pip "transformers==$TRANSFORMERS"
pip "fla-core==0.5.1" "flash-linear-attention==0.5.1" --no-deps
pip "tilelang==0.1.11" "apache-tvm-ffi==0.1.11"
# The installs above may pull a different torch; the pin wins.
torch_pin

# 3. gcc shim for tilelang's JIT, where llamafactory_env.sh looks ($LF_ROOT/.cc-shim), only if needed
probe() { echo 'int main(){return 0;}' | "$1" -x c++ - -o /dev/null > /dev/null 2>&1; }
if ! probe gcc; then
  for cc in gcc-13 gcc-12 gcc-11 gcc-10 gcc-9; do
    command -v "$cc" > /dev/null && probe "$cc" || continue
    mkdir -p "$LF_ROOT/.cc-shim"
    for n in gcc cc; do ln -sf "$(command -v "$cc")" "$LF_ROOT/.cc-shim/$n"; done
    for n in g++ c++; do ln -sf "$(command -v "${cc/gcc/g++}" || command -v "$cc")" "$LF_ROOT/.cc-shim/$n"; done
    echo "[setup] gcc shim -> $cc"
    break
  done
fi

# 4. versions, and the sm_120 check that needs no GPU: the compiled arch list of torch and flash-attn
python - << 'PY'
import glob, importlib.metadata as md, os, subprocess, torch
for p in ["torch", "torchvision", "transformers", "llamafactory", "peft", "accelerate", "deepspeed",
          "flash-attn", "liger-kernel", "fla-core", "flash-linear-attention", "tilelang", "triton"]:
    try: print(f"{p:24s} {md.version(p)}")
    except md.PackageNotFoundError: print(f"{p:24s} MISSING")
arch = torch._C._cuda_getArchFlags()
print("torch cuda", torch.version.cuda, "arch", arch)
assert "sm_120" in arch, "this torch has no sm_120 kernels"
import flash_attn_2_cuda  # noqa: F401  (import check: ABI matches this torch)
so = glob.glob(os.path.join(os.path.dirname(flash_attn_2_cuda.__file__), "flash_attn_2_cuda*.so"))[0]
cuobjdump = next((p for p in [os.path.join(os.environ["CUDA_HOME"], "bin", "cuobjdump"), "/usr/local/cuda/bin/cuobjdump"] if os.path.exists(p)), "")
if os.path.exists(cuobjdump):
    out = subprocess.run([cuobjdump, "--list-elf", so], capture_output=True, text=True).stdout
    print("flash-attn SASS", sorted({l.split(".")[-2] for l in out.split() if l.endswith(".cubin")}))
PY
du -sh "$LF_VENV"
echo SETUP_LLAMAFACTORY_DONE
