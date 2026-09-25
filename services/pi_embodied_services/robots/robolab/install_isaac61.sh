#!/usr/bin/env bash
# RoboLab on Isaac Sim 6.1 / Isaac Lab 3.0 (EA): venv + a patched RoboLab checkout.
#
#   install_isaac61.sh <venv> <robolab-dir> [task asset globs for git lfs, default: the three tasks below]
#
# RoboLab (NVLabs/RoboLab @ad45d4f9) supports only Isaac Sim 5.0/5.1, whose RTX scene DB segfaults at
# startup on driver 595.x (isaac-sim/IsaacSim#677). Isaac Lab 3.0.0-EA (wheel 3.0.0rc1) is built for
# Isaac Sim 6.1, Python 3.12, torch 2.11 (cu128 has sm_120). robolab-isaac61.patch ports RoboLab to it:
# quaternions (w,x,y,z) -> (x,y,z,w), ProxyArray data (.torch), FrameView instead of isaacsim.core.prims,
# factory ContactSensor, no SimulationCfg.render. Then run the env server with ROBOLAB_ROOT=<robolab-dir>.
#
# Index choices: pypi.nvidia.com for the NVIDIA wheels (set NVIDIA_INDEX, e.g. https://pypi.nvidia.cn),
# PIP_INDEX for the rest, TORCH_FIND_LINKS for torch (default download.pytorch.org cu128).
set -euo pipefail
V=${1:?venv dir}
R=${2:?robolab checkout dir}
shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
COMMIT=ad45d4f9
PIP_INDEX=${PIP_INDEX:-https://pypi.org/simple}
NVIDIA_INDEX=${NVIDIA_INDEX:-https://pypi.nvidia.com}
TORCH_FIND_LINKS=${TORCH_FIND_LINKS:-https://download.pytorch.org/whl/cu128}
TORCH=(torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128)

[ -x "$V/bin/python" ] || uv venv --python 3.12 "$V"
uv pip install --python "$V/bin/python" --index-url "$PIP_INDEX" --find-links "$TORCH_FIND_LINKS" "${TORCH[@]}"
uv pip install --python "$V/bin/python" --index-url "$PIP_INDEX" --extra-index-url "$NVIDIA_INDEX" \
	--find-links "$TORCH_FIND_LINKS" --index-strategy unsafe-best-match --prerelease allow \
	--override <(printf '%s\n' "${TORCH[@]}") \
	"isaaclab[isaacsim]==3.0.0rc1"
# RoboLab's own runtime dependencies (its pyproject pins the 5.x stacks, so it is not pip-installed).
uv pip install --python "$V/bin/python" --index-url "$PIP_INDEX" \
	json_numpy pyzmq msgpack msgpack-numpy opencv-python imageio pandas pyarrow tyro python-dotenv PyYAML

if [ ! -d "$R/.git" ]; then
	GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVLabs/RoboLab.git "$R"
fi
GIT_LFS_SKIP_SMUDGE=1 git -C "$R" checkout -q "$COMMIT"
if git -C "$R" apply --check "$HERE/robolab-isaac61.patch" 2>/dev/null; then
	git -C "$R" apply "$HERE/robolab-isaac61.patch"
else
	git -C "$R" apply --reverse --check "$HERE/robolab-isaac61.patch" # already applied, or fail loudly
fi
# Scene/object/texture assets are LFS; fetch what the tasks need (a pointer file renders as a cyan
# fallback material, and RTX waits on it at every reset).
if [ $# -eq 0 ]; then
	set -- "assets/scenes/banana_bowl.usda" "assets/scenes/rubiks_cube_bowl.usda" "assets/scenes/bagel*" \
		"assets/objects/ycb/**" "assets/objects/hot3d/**" "assets/objects/objaverse/**" "assets/objects/vomp/plate_large/**" \
		"assets/fixtures/**" "assets/materials/Base/**" "assets/materials/2023_1/**" "assets/backgrounds/default/**"
fi
git -C "$R" lfs pull --include "$(IFS=,; echo "$*")"
"$V/bin/python" -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
echo "ROBOLAB_ROOT=$R  python=$V/bin/python"
