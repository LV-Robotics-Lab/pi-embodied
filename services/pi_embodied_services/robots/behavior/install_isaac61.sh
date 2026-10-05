#!/usr/bin/env bash
# BEHAVIOR-1K 3.9.0 (OmniGibson + BDDL) on Isaac Sim 6.1 for the behavior env server: a venv, a patched
# BEHAVIOR-1K checkout, OmniGibson's cuRobo built for this GPU, and (--dataset) the data.
#
#   install_isaac61.sh <venv> <behavior-1k-dir> [--dataset]
#
# OmniGibson 3.9.0 pins Isaac Sim 5.1, whose RTX startup segfaults on NVIDIA driver 595.x
# (isaac-sim/IsaacSim#677; reproduced with isaacsim 5.1.0.0 on 595.71.05). Isaac Sim 6.1 (Kit 110,
# Python 3.12, torch 2.11 cu128 with sm_120) runs there; behavior-isaac61.patch ports OmniGibson to it
# (see the patch header: a 6.1 kit file, PhysX interfaces from omni.physx, Sdr shader inputs, carb
# eventdispatcher stop events, and an OmniGibson bug on compute capability 12.0 in CuRoboMotionGenerator).
# The Isaac stack is isaaclab[isaacsim]==3.0.0rc1's (the build this repository's RoboDojo and RoboLab ports
# run on; OmniGibson itself uses no Isaac Lab). OmniGibson's runtime deps are installed against that
# stack's numpy 2 / pillow 12 / websockets (OmniGibson pins numpy<2, pillow~=11 and a pymeshlab with no
# cp312 wheel; the lerobot fork is dataset export only), so OmniGibson and bddl go in --no-deps.
# cuRobo is OmniGibson's pin (StanfordVL/curobo @78612f45, the [primitives] extra), a torch CUDA extension:
# curobo-isaac61.patch adapts it to Isaac Sim 6.1's warp 1.16 and the build needs a CUDA 12.x nvcc
# (CUDA_HOME, default /usr/local/cuda; torch refuses a CUDA major mismatch) for this GPU's architecture
# (TORCH_CUDA_ARCH_LIST, default: the first visible GPU's compute capability).
#
# --dataset fetches into OMNIGIBSON_DATA_PATH (default <behavior-1k-dir>/datasets; HF_ENDPOINT honoured)
# what OmniGibson 3.9 reads from there: omnigibson-robot-assets/, behavior-1k-assets/ (31 GB, encrypted:
# the BEHAVIOR Data Bundle license is accepted interactively, or B1K_ACCEPT_LICENSE=1, and the key
# lands at <data>/omnigibson.key, which must never be copied or committed) and
# 2026-challenge-task-instances/ (OmniGibson 3.9's task instances: per split one full template for instance
# 0 and a *-tro_state.json overlay per instance, which the env server loads). Then run the env server with
# OMNIGIBSON_DATA_PATH=<data> (or --data-path).
#
# Index choices: pypi.nvidia.com for the NVIDIA wheels (NVIDIA_INDEX, e.g. https://pypi.nvidia.cn),
# PIP_INDEX for the rest, TORCH_INDEX for torch (default download.pytorch.org/whl/cu128); B1K_GIT and
# CUROBO_GIT name the two git repositories (a GitHub mirror where github.com is slow).
set -euo pipefail
V=${1:?venv dir}
R=${2:?BEHAVIOR-1K checkout dir}
shift 2
DATASET=false
[ "${1:-}" = --dataset ] && DATASET=true
HERE=$(cd "$(dirname "$0")" && pwd)
B1K_VERSION=3.9.0
B1K_GIT=${B1K_GIT:-https://github.com/StanfordVL/BEHAVIOR-1K.git}
CUROBO_GIT=${CUROBO_GIT:-https://github.com/StanfordVL/curobo.git}
CUROBO_COMMIT=78612f45cef52c3fa0298de243a54cd7ca614414
PIP_INDEX=${PIP_INDEX:-https://pypi.org/simple}
NVIDIA_INDEX=${NVIDIA_INDEX:-https://pypi.nvidia.com}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
TORCH=(torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128)
UV=${UV:-uv}
# Every install resolves against PyPI and the torch index (the +cu128 builds live only there).
IDX=(--index-url "$PIP_INDEX" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match)
export OMNI_KIT_ACCEPT_EULA=YES

[ -x "$V/bin/python" ] || "$UV" venv --python 3.12 "$V"
PY="$V/bin/python"
"$UV" pip install --python "$PY" "${IDX[@]}" "${TORCH[@]}"
"$UV" pip install --python "$PY" "${IDX[@]}" --extra-index-url "$NVIDIA_INDEX" --prerelease allow \
	--override <(printf '%s\n' "${TORCH[@]}") \
	"isaaclab[isaacsim]==3.0.0rc1"
# OmniGibson's runtime deps (its setup.py install_requires, minus the lerobot fork) on the Isaac stack's own
# numpy / pillow / websockets / warp, plus the services' RPC layer (msgpack, msgpack-numpy) and the build tools.
LOCK=$(mktemp)
"$UV" pip freeze --python "$PY" | grep -iE '^(numpy|pillow|websockets|torch|torchvision|torchaudio|warp-lang|isaacsim|isaaclab)(==| @)' >"$LOCK"
"$UV" pip install --python "$PY" "${IDX[@]}" --override "$LOCK" \
	"huggingface-hub>=0.34.4" "gymnasium>=0.28.1" scipy GitPython transforms3d networkx PyYAML addict ipython future \
	trimesh h5py cryptography opencv-python-headless nest_asyncio imageio imageio-ffmpeg termcolor progressbar pymeshlab \
	click aenum rtree graphviz matplotlib lxml numba cffi omegaconf msgpack msgpack-numpy \
	ninja "setuptools<82" setuptools_scm wheel

if [ ! -d "$R/OmniGibson" ]; then
	git clone --depth 1 --branch "v$B1K_VERSION" "$B1K_GIT" "$R"
fi
if git -C "$R" apply --check "$HERE/behavior-isaac61.patch" 2>/dev/null; then
	git -C "$R" apply "$HERE/behavior-isaac61.patch"
else
	git -C "$R" apply --reverse --check "$HERE/behavior-isaac61.patch" # already applied, or fail loudly
fi
"$UV" pip install --python "$PY" "${IDX[@]}" --no-deps -e "$R/bddl3" -e "$R/OmniGibson"

# cuRobo: an editable install without its .git would fail setuptools_scm, so the checkout keeps its history.
C="$R/third_party/curobo"
if [ ! -d "$C/.git" ]; then
	rm -rf "$C"
	git clone "$CUROBO_GIT" "$C"
fi
git -C "$C" checkout -q "$CUROBO_COMMIT"
if git -C "$C" apply --check "$HERE/curobo-isaac61.patch" 2>/dev/null; then
	git -C "$C" apply "$HERE/curobo-isaac61.patch"
else
	git -C "$C" apply --reverse --check "$HERE/curobo-isaac61.patch"
fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
[ -x "$CUDA_HOME/bin/nvcc" ] || {
	echo "no nvcc at $CUDA_HOME/bin: set CUDA_HOME to a CUDA 12.x toolkit (torch 2.11+cu128 refuses another major)" >&2
	exit 1
}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$("$PY" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')}
export PATH="$CUDA_HOME/bin:$PATH" MAX_JOBS=${MAX_JOBS:-8}
"$UV" pip install --python "$PY" "${IDX[@]}" --override "$LOCK" --no-build-isolation -e "$C"

if $DATASET; then
	export OMNIGIBSON_DATA_PATH=${OMNIGIBSON_DATA_PATH:-$R/datasets}
	mkdir -p "$OMNIGIBSON_DATA_PATH"
	accept=()
	[ "${B1K_ACCEPT_LICENSE:-}" = 1 ] && accept=(--accept_license)
	"$PY" -m omnigibson.utils.asset_utils --download_omnigibson_robot_assets --download_behavior_1k_assets \
		--download_2026_challenge_task_instances "${accept[@]}"
fi
"$PY" -c "import torch, curobo, bddl, omnigibson; print('omnigibson', omnigibson.__version__, torch.__version__, torch.cuda.get_arch_list(), curobo.__file__)"
echo "python=$PY  OMNIGIBSON_DATA_PATH=${OMNIGIBSON_DATA_PATH:-$R/datasets}"
