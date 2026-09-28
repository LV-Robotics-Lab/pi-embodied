#!/usr/bin/env bash
# RoboDojo on its own stack: Isaac Sim 5.1 + RoboDojo's Isaac Lab 2.3 fork + its cuRobo v2 fork, unpatched.
#
#   install_isaac51.sh <venv> <robodojo-dir> [assets-dir]
#
# The upstream path (RoboDojo's scripts/install.sh without conda), for hosts whose driver runs Isaac Sim
# 5.1. It does not start on driver 595.x (isaac-sim/IsaacSim#677): there use install_isaac61.sh, whose
# patch ports RoboDojo to Isaac Sim 6.1 (results there are not comparable with RoboDojo's leaderboard;
# see README.md). The env server runs on either: it enables the deprecated core extensions only on
# Isaac Sim 6, and refuses the tasks in sim.UNSUPPORTED only there.
#
# Index choices: pypi.nvidia.com for the NVIDIA wheels (NVIDIA_INDEX), PIP_INDEX for the rest,
# TORCH_FIND_LINKS for torch (default download.pytorch.org cu128).
set -euo pipefail
V=${1:?venv dir}
R=${2:?robodojo checkout dir}
A=${3:-}
COMMIT=726e9aa
PIP_INDEX=${PIP_INDEX:-https://pypi.org/simple}
NVIDIA_INDEX=${NVIDIA_INDEX:-https://pypi.nvidia.com}
TORCH_FIND_LINKS=${TORCH_FIND_LINKS:-https://download.pytorch.org/whl/cu128}
TORCH=(torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0)
UV=${UV:-uv}

[ -x "$V/bin/python" ] || "$UV" venv --python 3.11 "$V"
"$UV" pip install --python "$V/bin/python" --index-url "$PIP_INDEX" --find-links "$TORCH_FIND_LINKS" "${TORCH[@]}"
"$UV" pip install --python "$V/bin/python" --index-url "$PIP_INDEX" --extra-index-url "$NVIDIA_INDEX" \
	--index-strategy unsafe-best-match --override <(printf '%s\n' "${TORCH[@]}") "isaacsim[all,extscache]==5.1.0"

if [ ! -d "$R/.git" ]; then
	git clone https://github.com/robodojo-benchmark/RoboDojo.git "$R"
fi
git -C "$R" checkout -q "$COMMIT"
if git -C "$R" apply --reverse --check "$(dirname "$0")/robodojo-isaac61.patch" 2>/dev/null; then
	echo "$R carries robodojo-isaac61.patch; use a separate checkout for Isaac Sim 5.1" >&2
	exit 1
fi
# RoboDojo's pinned forks: Isaac Lab 2.3.2 + one AppLauncher commit, cuRobo v2.
git -C "$R" submodule update --init third_party/IsaacLab third_party/curobo
(cd "$R/third_party/IsaacLab" && OMNI_KIT_ACCEPT_EULA=YES TERM=xterm-256color PATH="$V/bin:$PATH" ./isaaclab.sh --install none)
"$UV" pip install --python "$V/bin/python" --index-url "$PIP_INDEX" --override <(printf '%s\n' "${TORCH[@]}") \
	--no-build-isolation -e "$R/third_party/curobo[cu12]"
# RoboDojo's runtime pins (scripts/install.sh pin_runtime_deps) and the services' RPC and image needs.
"$UV" pip install --python "$V/bin/python" --index-url "$PIP_INDEX" --override <(printf '%s\n' "${TORCH[@]}") \
	numpy==1.26.0 packaging==23.0 typing_extensions==4.12.2 filelock==3.13.1 websockets==12.0 scipy==1.15.3 \
	warp-lang==1.11.0 omegaconf transforms3d shapely rtree matplotlib tqdm pyyaml gymnasium trimesh \
	msgpack msgpack-numpy pillow imageio opencv-python-headless huggingface_hub

if [ -n "$A" ]; then
	"$V/bin/python" - "$A" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("RoboDojo-Benchmark/RoboDojo", repo_type="dataset", allow_patterns=["Assets/**"],
                  local_dir=sys.argv[1], max_workers=8)
PY
	[ -e "$R/Assets" ] || ln -s "$A/Assets" "$R/Assets"
fi
if [ -d "$R/Assets/Robots" ]; then
	(cd "$R" && "$V/bin/python" utils/update_embodiment_config_path.py)
fi
echo "ROBODOJO_ROOT=$R  python=$V/bin/python"
