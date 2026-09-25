#!/usr/bin/env bash
# train.sh: GUMI recordings -> Show-Harness training set -> LLaMA-Factory LoRA SFT -> an adapter serve.sh serves.
#
#   SH=/root/autodl-tmp/refs/Show-Harness NAME=pick_cube_gumi \
#     bash train.sh /root/autodl-tmp/runs/gumi/<session>/rollouts [more GUMI record dirs...]
#
# 1. packages/embodied/src/finetuned/prepare.ts: every single-arm GUMI run -> <DATA>/rollouts/<task>/rollout_NNN
#    with the provider's own camera transform applied (training and inference pixels identical).
# 2. Show-Harness train/data_preparation/rollouts_to_alpaca.py --version $VERSION (one Alpaca sample per
#    step, "<image><image>" + the lite prompt, plus a synthesized DONE per episode; prepare_dataset.sh's
#    loop), then register_dataset.py into LLaMA-Factory's dataset_info.json.
# 3. A LoRA config from train/configs/$BASE_CONFIG with dataset/output/model swapped in, and
#    train/scripts/train.sh on GPU $GPU while holding $LOCK (STEP=prepare stops after step 2).
# Then: LORA=$NAME=<DATA>/saves/<NAME>/<checkpoint> bash serve.sh, and pi --model finetuned/local --ft-model $NAME.
#
#   SH=        Show-Harness checkout (github.com/showlab/Show-Harness @137d571) with LLaMA-Factory set up
#              once by its train/scripts/setup_llamafactory.sh (third_party/LlamaFactory + .venv)
#   NAME=      dataset / adapter name (required)       DATA=/root/autodl-tmp/data/finetuned/$NAME
#   VERSION=v3 prompt version (must match --ft-prompt at inference)
#   BASE_CONFIG=qwen3_5_2b_sim.yaml   MODEL_PATH=/root/autodl-tmp/checkpoints/Qwen3.5-2B   EPOCHS= (keep)
#   GPU=1  LOCK=/root/autodl-tmp/locks/gpu1.lock   STEP=all|prepare (stop before registering; no LLaMA-Factory needed)
#   PREPARE_ARGS= (e.g. --robot maniskill)   PYTHON= (any python3; the converter is stdlib only)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI="${PI:-$(cd "$HERE/../../.." && pwd)}"
SH="${SH:?SH (Show-Harness checkout) is required}"
NAME="${NAME:?NAME (dataset/adapter name) is required}"
DATA="${DATA:-/root/autodl-tmp/data/finetuned/$NAME}"
VERSION="${VERSION:-v3}"
BASE_CONFIG="${BASE_CONFIG:-qwen3_5_2b_sim.yaml}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/checkpoints/Qwen3.5-2B}"
GPU="${GPU:-1}"
LOCK="${LOCK-/root/autodl-tmp/locks/gpu1.lock}"
PY="${PYTHON:-$(command -v python3 || echo /root/miniconda3/bin/python)}"
[ $# -gt 0 ] || { echo "usage: SH=... NAME=... bash train.sh <gumi record dir>..." >&2; exit 2; }
export PATH=/root/autodl-tmp/tools/node/bin:$PATH

echo "[1/3] GUMI -> rollouts ($DATA/rollouts)"
rm -rf "$DATA/rollouts"
# shellcheck disable=SC2086
node --experimental-strip-types "$PI/packages/embodied/src/finetuned/prepare.ts" --out "$DATA/rollouts" ${PREPARE_ARGS:-} "$@"

echo "[2/3] rollouts -> $DATA/rollouts.json (Show-Harness rollouts_to_alpaca.py, prompts/$VERSION)"
# train/scripts/prepare_dataset.sh's loop: one task dir per instruction, its task_text from metadata.json.
dirs=(); maps=()
for d in "$DATA/rollouts"/*/; do
  T="$(basename "$d")"
  TT="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['task_text'])" "$d/rollout_000/metadata.json")"
  echo "  [$T] $(find "$d" -maxdepth 1 -type d -name 'rollout_*' | wc -l) episodes: $TT"
  dirs+=("$d"); maps+=("$T=$TT")
done
"$PY" "$SH/train/data_preparation/rollouts_to_alpaca.py" "${dirs[@]}" --version "$VERSION" \
  --task-map "${maps[@]}" --output "$DATA/rollouts.json"
[ "${STEP:-all}" = prepare ] && exit 0
"$PY" "$SH/train/data_preparation/register_dataset.py" "$NAME" --samples "$DATA/rollouts.json" \
  --lf-root "${LF_ROOT:-$SH/third_party/LlamaFactory}"

echo "[3/3] LoRA SFT on GPU $GPU"
CONFIG="$DATA/$NAME.yaml"
sed -e "s|^dataset:.*|dataset: $NAME|" \
    -e "s|^output_dir:.*|output_dir: $DATA/saves/$NAME|" \
    -e "s|^run_name:.*|run_name: $NAME|" \
    -e "s|^report_to:.*|report_to: none|" \
    ${EPOCHS:+-e "s|^num_train_epochs:.*|num_train_epochs: $EPOCHS|"} \
    "$SH/train/configs/$BASE_CONFIG" > "$CONFIG"
train() { CONFIG="$CONFIG" GPU="$GPU" MODEL_PATH="$MODEL_PATH" bash "$SH/train/scripts/train.sh"; }
if [ -n "$LOCK" ]; then
  exec 9> "$LOCK"
  echo "waiting for $LOCK"
  flock 9
fi
train
echo "adapter checkpoints under $DATA/saves/$NAME; serve one with:"
echo "  LORA=$NAME=$DATA/saves/$NAME/checkpoint-<step> bash $HERE/serve.sh"
