#!/usr/bin/env bash
# BEHAVIOR-1K (OmniGibson + BDDL) for the behavior env server: a venv with Isaac Sim, OmniGibson
# and BDDL from a BEHAVIOR-1K checkout, and the challenge dataset.
#
#   install.sh <venv> <behavior-1k-dir> [--dataset]
#
# Two supported stacks (B1K_VERSION): 3.9.0 (default; Isaac Sim 5.1, Python 3.11, what OpenETA
# pins) and 3.7.2 (Isaac Sim 4.5, Python 3.10, CaP-X's capx/third_party/b1k fork). Both pin
# Isaac Sim releases whose RTX startup segfaults on NVIDIA driver 595.x (isaac-sim/IsaacSim#677:
# 5.x verified, 4.5 see the robot's README); Isaac Sim 6.1 (Python 3.12) is not supported by
# OmniGibson: B1K_ISAAC=6.1.0.0 B1K_PYTHON=3.12 builds the probe venv described there (OmniGibson
# imports, Kit launches; not a supported combination).
#
# The dataset (--dataset: og_dataset scenes, assets, the 2025-challenge-task-instances; tens of
# GB) is fetched by OmniGibson's own downloader into OMNIGIBSON_DATA_PATH (default
# <behavior-1k-dir>/datasets). Mirrors: PIP_INDEX for PyPI, NVIDIA_INDEX for the Isaac Sim
# wheels (e.g. https://pypi.nvidia.cn), TORCH_INDEX for torch.
set -euo pipefail
V=${1:?venv dir}
R=${2:?BEHAVIOR-1K checkout dir}
shift 2
DATASET=false
[ "${1:-}" = --dataset ] && DATASET=true
B1K_VERSION=${B1K_VERSION:-3.9.0}
case $B1K_VERSION in
3.9.*) B1K_ISAAC=${B1K_ISAAC:-5.1.0.0} B1K_PYTHON=${B1K_PYTHON:-3.11} TORCH=torch==2.7.0 ;;
3.7.*) B1K_ISAAC=${B1K_ISAAC:-4.5.0.0} B1K_PYTHON=${B1K_PYTHON:-3.10} TORCH=torch==2.5.1 ;;
*) echo "B1K_VERSION $B1K_VERSION: 3.9.x or 3.7.x" >&2 && exit 2 ;;
esac
PIP_INDEX=${PIP_INDEX:-https://pypi.org/simple}
NVIDIA_INDEX=${NVIDIA_INDEX:-https://pypi.nvidia.com}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
export OMNI_KIT_ACCEPT_EULA=YES

if [ ! -d "$R/OmniGibson" ]; then
	git clone --depth 1 --branch "v$B1K_VERSION" https://github.com/StanfordVL/BEHAVIOR-1K.git "$R"
fi
[ -x "$V/bin/python" ] || uv venv --python "$B1K_PYTHON" "$V"
PY="$V/bin/python"
# torch first, from its own index: Isaac Sim's metadata would otherwise pull the newest torch.
uv pip install --python "$PY" --index-url "$TORCH_INDEX" "$TORCH"
uv pip install --python "$PY" --index-url "$PIP_INDEX" --extra-index-url "$NVIDIA_INDEX" \
	--index-strategy unsafe-best-match "isaacsim[all,extscache]==$B1K_ISAAC"
uv pip install --python "$PY" --index-url "$PIP_INDEX" -e "$R/bddl3"
# OmniGibson's runtime deps (its setup.py also pins a lerobot fork for dataset export, not needed
# by the env server); [primitives] brings cuRobo for StarterSemanticActionPrimitives.
uv pip install --python "$PY" --index-url "$PIP_INDEX" --no-build-isolation -e "$R/OmniGibson[primitives]"
if $DATASET; then
	export OMNIGIBSON_DATA_PATH=${OMNIGIBSON_DATA_PATH:-$R/datasets}
	"$PY" -m omnigibson.download_datasets
	"$PY" -m omnigibson.utils.asset_utils download_2025_challenge_task_instances 2>/dev/null ||
		echo "fetch 2025-challenge-task-instances into $OMNIGIBSON_DATA_PATH (see BEHAVIOR-1K/docs)" >&2
fi
"$PY" -c "import omnigibson, bddl; print('omnigibson', omnigibson.__version__)"
echo "python=$PY  OMNIGIBSON_DATA_PATH=${OMNIGIBSON_DATA_PATH:-$R/datasets}"
