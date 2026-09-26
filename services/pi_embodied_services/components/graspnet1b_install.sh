#!/usr/bin/env bash
# Build the GraspNet-1Billion models of components/graspnet1b_server.py into a venv that already
# has a CUDA torch (services/setup.sh graspnet1b runs it after the [graspnet1b] extra):
#
#   graspnet1b_install.sh <venv python> <dir> [--weights]
#
# - <dir>/src: graspnet-baseline, graspness_implementation (GSNet) and MinkowskiEngine at pinned
#   commits (git; set GIT_PROXY to clone through a proxy);
# - the pointnet2 and knn CUDA ops (graspnet-baseline's copies, which no longer include the
#   removed THC headers; graspness_implementation imports the same two modules) and
#   MinkowskiEngine 0.5.4, compiled for TORCH_CUDA_ARCH_LIST (default: the visible GPUs) with the
#   nvcc of CUDA_HOME, which must match torch's CUDA major version (torch.version.cuda). The
#   MinkowskiEngine CPU backend links OpenBLAS (Debian/Ubuntu: apt install libopenblas-dev);
# - with --weights, <dir>/gsnet: GSNet's RealSense and Kinect checkpoints (Hugging Face Mizo330/gsnet-graspness,
#   a mirror of the authors' Google Drive files; honours HF_ENDPOINT);
# - with --weights, <dir>/baseline: graspnet-baseline's checkpoint-rs.tar / checkpoint-kn.tar. The authors publish
#   them only on Google Drive (and Baidu Pan): they are fetched with gdown when Google Drive is
#   reachable, otherwise put them there by hand.
#
# Prints the server's environment (GRASPNET_BASELINE_ROOT, GRASPNESS_ROOT, the checkpoints) at
# the end. Idempotent: existing checkouts, built modules and complete downloads are kept.
set -euo pipefail

PY=${1:?usage: graspnet1b_install.sh <venv python> <dir> [--weights]}
DIR=${2:?usage: graspnet1b_install.sh <venv python> <dir> [--weights]}
WEIGHTS=${3:-}
BASELINE_SHA=280c215129f759ed8649cb4e89fc5dfee55f4f80
GRASPNESS_SHA=ff33da111e72db1b8697758c7863fcec2359280e
MINKOWSKI_SHA=02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
GSNET_REPO=Mizo330/gsnet-graspness
GSNET_REV=874a8c29287b5acfdd94b7f2c7ae552ffe58d9d1
# sha256 of its two files (epoch-10 GSNet MinkUNet14D checkpoints, 216 tensors).
GSNET_SHA256_realsense=a0b9d85b300e40c76b66d092de62138d74757d6b7c2037f0d364f33960c54aa9
GSNET_SHA256_kinect=18bd22fa0d724895478d3e84e846899bde6db9090b138acf784dba7b71f6d561
# Google Drive ids of the authors' checkpoints (graspnet-baseline README).
BASELINE_RS_GDRIVE=1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk
BASELINE_KN_GDRIVE=1vK-d0yxwyJwXHYWOtH1bDMoe--uZ2oLX
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

die() { echo "graspnet1b_install.sh: $*" >&2; exit 1; }
note() { echo "== $*"; }
run() {
	printf '+'
	printf ' %q' "$@"
	printf '\n'
	"$@"
}
pip_install() {
	if command -v uv >/dev/null; then
		run uv pip install --python "$PY" "$@"
	else
		run "$PY" -m pip install "$@"
	fi
}

[ -x "$PY" ] || die "$PY is not an executable python"
mkdir -p "$DIR/src" "$DIR/gsnet" "$DIR/baseline"
DIR=$(cd "$DIR" && pwd)

# torch, its CUDA, the target architectures and a matching nvcc.
read -r TORCH_CUDA TORCH_ARCHS < <("$PY" - <<'EOF'
import torch
if not torch.cuda.is_available() or torch.version.cuda is None:
    raise SystemExit("torch has no CUDA device; install a CUDA build of torch into the venv first")
caps = sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
print(torch.version.cuda, ";".join(f"{a}.{b}" for a, b in caps))
EOF
) || die "torch with CUDA is not importable from $PY"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$TORCH_ARCHS}
if [ -z "${CUDA_HOME:-}" ]; then
	CUDA_HOME=$("$PY" -c 'from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME or "")')
fi
[ -x "$CUDA_HOME/bin/nvcc" ] || die "no nvcc under CUDA_HOME=$CUDA_HOME; set CUDA_HOME to a CUDA $TORCH_CUDA toolkit"
NVCC_CUDA=$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
[ "${NVCC_CUDA%%.*}" = "${TORCH_CUDA%%.*}" ] ||
	die "nvcc $NVCC_CUDA ($CUDA_HOME) does not match torch's CUDA $TORCH_CUDA; set CUDA_HOME"
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH" MAX_JOBS=${MAX_JOBS:-8}
note "torch CUDA $TORCH_CUDA, nvcc $NVCC_CUDA ($CUDA_HOME), TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# checkout NAME URL SHA
checkout() {
	local d=$DIR/src/$1
	if [ ! -d "$d/.git" ]; then
		run git ${GIT_PROXY:+-c http.proxy=$GIT_PROXY} clone -q "$2" "$d"
	fi
	if [ "$(git -C "$d" rev-parse HEAD)" != "$3" ]; then
		git -C "$d" cat-file -e "$3^{commit}" 2>/dev/null ||
			run git ${GIT_PROXY:+-c http.proxy=$GIT_PROXY} -C "$d" fetch -q origin
		run git -C "$d" checkout -q "$3"
	fi
}
checkout graspnet-baseline https://github.com/graspnet/graspnet-baseline.git $BASELINE_SHA
checkout graspness_implementation https://github.com/rhett-chen/graspness_implementation.git $GRASPNESS_SHA
checkout MinkowskiEngine https://github.com/NVIDIA/MinkowskiEngine.git $MINKOWSKI_SHA

# The CUDA ops (extension modules linking torch's libraries: import torch first). Their setup.py
# import torch, so no build isolation.
if "$PY" -c 'import torch, pointnet2._ext' 2>/dev/null; then
	note "pointnet2 ops present"
else
	pip_install --no-build-isolation "$DIR/src/graspnet-baseline/pointnet2"
fi
if "$PY" -c 'import torch; from knn_pytorch import knn_pytorch' 2>/dev/null; then
	note "knn op present"
else
	# knn's setup.py only adds the CUDA sources when it sees a GPU and CUDA_HOME, as here.
	pip_install --no-build-isolation "$DIR/src/graspnet-baseline/knn"
fi

# MinkowskiEngine 0.5.4 on CUDA 12 (graspnet1b-minkowski-cuda12.patch): the thrust headers CUDA 12
# no longer includes transitively; CCCL_DISABLE_NVTX, since CUB's own nvtx3.hpp clashes with the
# bundled cudf copy; and no `pip uninstall` from setup.py (uv venvs have no pip).
if "$PY" -c 'import torch, MinkowskiEngine' 2>/dev/null; then
	note "MinkowskiEngine present"
else
	[ -e /usr/include/x86_64-linux-gnu/cblas.h ] || [ -e /usr/include/cblas.h ] || [ -e /usr/include/openblas/cblas.h ] ||
		die "MinkowskiEngine needs OpenBLAS headers (Debian/Ubuntu: apt install libopenblas-dev)"
	me=$DIR/src/MinkowskiEngine
	if git -C "$me" apply --check "$HERE/graspnet1b-minkowski-cuda12.patch" 2>/dev/null; then
		run git -C "$me" apply "$HERE/graspnet1b-minkowski-cuda12.patch"
	fi
	# It links -lcusparse; a toolkit assembled from pip wheels (as torch's nvidia-cusparse) has
	# only libcusparse.so.12, so point the linker at that one.
	if [ ! -e "$CUDA_HOME/lib64/libcusparse.so" ]; then
		so=$("$PY" -c 'import nvidia.cusparse as c, os; print(os.path.join(list(c.__path__)[0], "lib", "libcusparse.so.12"))')
		mkdir -p "$DIR/lib"
		ln -sf "$so" "$DIR/lib/libcusparse.so"
		export LIBRARY_PATH="$DIR/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
	fi
	rm -rf "$me/dist"
	(cd "$me" && run "$PY" setup.py -q bdist_wheel --force_cuda --blas=openblas --cuda_home="$CUDA_HOME")
	pip_install --no-deps "$me"/dist/[Mm]inkowski[Ee]ngine-*.whl
fi
"$PY" -c 'import torch, pointnet2._ext, MinkowskiEngine as ME; from knn_pytorch import knn_pytorch; print("ops OK, MinkowskiEngine", ME.__version__)'

# Checkpoints (--weights).
weights() {
	local cam f want pair
	for cam in realsense kinect; do
		f=checkp_$cam.tar
		want=GSNET_SHA256_$cam
		if [ -s "$DIR/gsnet/$f" ]; then
			note "gsnet/$f present"
		else
			run "$PY" -c 'import sys; from huggingface_hub import hf_hub_download as d; d(sys.argv[1], sys.argv[3], revision=sys.argv[2], local_dir=sys.argv[4])' \
				"$GSNET_REPO" "$GSNET_REV" "$f" "$DIR/gsnet"
		fi
		echo "${!want}  $DIR/gsnet/$f" | sha256sum -c --quiet - || die "gsnet/$f: sha256 mismatch"
	done
	for pair in "checkpoint-rs.tar $BASELINE_RS_GDRIVE" "checkpoint-kn.tar $BASELINE_KN_GDRIVE"; do
		set -- $pair
		if [ -s "$DIR/baseline/$1" ]; then
			note "baseline/$1 present"
		elif curl -fsS -m 10 -o /dev/null https://drive.google.com/ 2>/dev/null; then
			pip_install gdown
			run "$(dirname "$PY")/gdown" "$2" -O "$DIR/baseline/$1"
		else
			note "Google Drive is unreachable: put graspnet-baseline's $1 (Google Drive id $2) into $DIR/baseline/"
		fi
	done
}
if [ "$WEIGHTS" = --weights ]; then weights; fi

note "environment for graspnet1b_server:"
echo "export GRASPNET_BASELINE_ROOT=$DIR/src/graspnet-baseline"
echo "export GRASPNESS_ROOT=$DIR/src/graspness_implementation"
echo "# --model baseline --checkpoint $DIR/baseline/checkpoint-rs.tar (or checkpoint-kn.tar)"
echo "# --model gsnet --checkpoint $DIR/gsnet/checkp_realsense.tar (or checkp_kinect.tar)"
