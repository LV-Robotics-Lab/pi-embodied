#!/usr/bin/env bash
# RoboDojo on Isaac Sim 6.1 / Isaac Lab 3.0 (EA): venv + a patched RoboDojo checkout + its cuRobo v2 fork.
#
#   install_isaac61.sh <venv> <robodojo-dir> [assets-dir]
#
# RoboDojo (robodojo-benchmark/RoboDojo @726e9aa) pins Isaac Sim 5.1 + Isaac Lab 2.3.2, whose RTX scene DB
# segfaults at startup on driver 595.x (isaac-sim/IsaacSim#677; reproduced with isaacsim 5.1.0.0 on 595.71.05).
# Isaac Lab 3.0.0-EA (wheel 3.0.0rc1) is built for Isaac Sim 6.1, Python 3.12, torch 2.11 (cu128 has sm_120).
# robodojo-isaac61.patch ports RoboDojo to it (see the patch header); RoboDojo's Isaac Lab fork only changes
# AppLauncher defaults (cameras on, isaaclab.python.kit), which the env server passes explicitly instead.
# cuRobo is RoboDojo's v2 fork (JIT kernels through cuda-core and warp, no torch extension build).
# With [assets-dir], Assets/ (41 GB: RoboDojo-Benchmark/RoboDojo on the HF hub, honours HF_ENDPOINT) is
# downloaded there and linked as <robodojo-dir>/Assets. Then run the env server with ROBODOJO_ROOT=<robodojo-dir>.
#
# Index choices: pypi.nvidia.com for the NVIDIA wheels (set NVIDIA_INDEX, e.g. https://pypi.nvidia.cn),
# PIP_INDEX for the rest, TORCH_FIND_LINKS for torch (default download.pytorch.org cu128).
set -euo pipefail
V=${1:?venv dir}
R=${2:?robodojo checkout dir}
A=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd)
COMMIT=726e9aa
CUROBO_URL=https://github.com/yuechen0614/curobo.git
CUROBO_COMMIT=d17b54ce32cba095c0b000c4c58777075d11de0e
PIP_INDEX=${PIP_INDEX:-https://pypi.org/simple}
NVIDIA_INDEX=${NVIDIA_INDEX:-https://pypi.nvidia.com}
TORCH_FIND_LINKS=${TORCH_FIND_LINKS:-https://download.pytorch.org/whl/cu128}
TORCH=(torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128)
UV=${UV:-uv}

[ -x "$V/bin/python" ] || "$UV" venv --python 3.12 "$V"
"$UV" pip install --python "$V/bin/python" --index-url "$PIP_INDEX" --find-links "$TORCH_FIND_LINKS" "${TORCH[@]}"
"$UV" pip install --python "$V/bin/python" --index-url "$PIP_INDEX" --extra-index-url "$NVIDIA_INDEX" \
	--find-links "$TORCH_FIND_LINKS" --index-strategy unsafe-best-match --prerelease allow \
	--override <(printf '%s\n' "${TORCH[@]}") \
	"isaaclab[isaacsim]==3.0.0rc1"
# RoboDojo's runtime imports (its install.sh pins the Isaac Sim 5.1 stack, so it is not run), plus the
# services' RPC layer (msgpack, msgpack-numpy) and PNG/video encoding.
"$UV" pip install --python "$V/bin/python" --index-url "$PIP_INDEX" --override <(printf '%s\n' "${TORCH[@]}") \
	omegaconf transforms3d shapely rtree scipy matplotlib tqdm pyyaml websockets gymnasium trimesh \
	msgpack msgpack-numpy pillow imageio opencv-python-headless huggingface_hub "setuptools<82" setuptools_scm

if [ ! -d "$R/.git" ]; then
	git clone https://github.com/robodojo-benchmark/RoboDojo.git "$R"
fi
git -C "$R" checkout -q "$COMMIT"
if git -C "$R" apply --check "$HERE/robodojo-isaac61.patch" 2>/dev/null; then
	git -C "$R" apply "$HERE/robodojo-isaac61.patch"
else
	git -C "$R" apply --reverse --check "$HERE/robodojo-isaac61.patch" # already applied, or fail loudly
fi

# cuRobo v2 (RoboDojo's third_party/curobo pin): an editable install without its .git would fail
# setuptools_scm, so the checkout keeps its history.
C="$R/third_party/curobo"
if [ ! -d "$C/.git" ]; then
	rm -rf "$C"
	git clone "$CUROBO_URL" "$C"
fi
git -C "$C" checkout -q "$CUROBO_COMMIT"
"$UV" pip install --python "$V/bin/python" --index-url "$PIP_INDEX" --override <(printf '%s\n' "${TORCH[@]}") \
	--no-build-isolation -e "$C[cu12]"

if [ -n "$A" ]; then
	"$V/bin/python" - "$A" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("RoboDojo-Benchmark/RoboDojo", repo_type="dataset", allow_patterns=["Assets/**"],
                  local_dir=sys.argv[1], max_workers=8)
PY
	[ -e "$R/Assets" ] || ln -s "$A/Assets" "$R/Assets"
fi
# The robots' cuRobo configs ship as *_tmp.yml templates with the Assets path left open.
if [ -d "$R/Assets/Robots" ]; then
	(cd "$R" && "$V/bin/python" utils/update_embodiment_config_path.py)
fi
"$V/bin/python" -c "import torch, curobo; print(torch.__version__, torch.cuda.get_arch_list(), curobo.__file__)"
echo "ROBODOJO_ROOT=$R  python=$V/bin/python"
