#!/usr/bin/env bash
# train.sh: GUMI recordings -> Show-Harness training set -> LLaMA-Factory LoRA SFT -> an adapter serve.sh serves.
#
#   NAME=pick_cube_gumi \
#     bash train.sh /root/autodl-tmp/runs/gumi/<session>/rollouts [more GUMI record dirs...]
#   NAME=pick_cube_lerobot \
#     bash train.sh --from-lerobot <LeRobot v3.0 dataset dir> [more dataset dirs...]
#   NAME=blockpap_real2sim \
#     bash train.sh --from-rollouts <dir of rollout_NNN> [more...]
#
# 1. packages/embodied/src/finetuned/prepare.ts: every single-arm GUMI run -> <DATA>/rollouts/<task>/rollout_NNN
#    with the provider's own camera transform applied (training and inference pixels identical).
#    --from-lerobot: services/pi_embodied_services/finetuned/lerobot_to_rollouts.py writes the same layout
#    from the unified LeRobot datasets (flywheel CLI export-lerobot / export-gumi; successful episodes,
#    --prompt $VERSION), the frames through the same transform (finetuned/transform.ts).
#    --from-rollouts: directories already in that layout (one task each: rollout_NNN/ with metadata.json
#    task_text), as robots/{maniskill,robolab}/real2sim*.py and merge_shards.py write them; linked in.
# 2. Show-Harness's train/data_preparation/rollouts_to_alpaca.py --version $VERSION (one Alpaca sample per
#    step, "<image><image>" + the lite prompt, plus a synthesized DONE per episode; prepare_dataset.sh's
#    loop), then register_dataset.py into LLaMA-Factory's dataset_info.json.
# 3. A LoRA config from train/configs/$BASE_CONFIG with dataset/output/model swapped in, and
#    train/scripts/train.sh on GPU $GPU while holding $LOCK (STEP=prepare stops after step 2).
# Then: LORA=$NAME=<DATA>/saves/<NAME>/<checkpoint> bash serve.sh, and pi --model finetuned/local --ft-model $NAME.
#
#   SH=        Show-Harness tree whose train/ and prompts/ are read, default ./showharness (the files of
#              github.com/showlab/Show-Harness @137d571 this needs, vendored); a full checkout works too
#   LF_VENV=/root/autodl-tmp/venvs/llamafactory   LF_ROOT=$LF_VENV/LlamaFactory: LLaMA-Factory and its
#              venv, installed once by ./setup_llamafactory.sh (register_dataset.py writes LF_ROOT/data)
#   NAME=      dataset / adapter name (required)       DATA=/root/autodl-tmp/data/finetuned/$NAME
#   VERSION=v3 prompt version (must match --ft-prompt at inference)
#   BASE_CONFIG=qwen3_5_2b_sim.yaml   MODEL_PATH=/root/autodl-tmp/checkpoints/Qwen3.5-2B   EPOCHS= (keep)
#   SET="key=value ..." more yaml keys replaced or added, e.g. a smoke run: SET="max_steps=30 save_steps=10"
#   GPU=1  LOCK=/root/autodl-tmp/locks/gpu1.lock   STEP=all|prepare (stop before registering; no LLaMA-Factory needed)
#   PREPARE_ARGS= (e.g. --robot maniskill; --from-lerobot: e.g. --include-failures)
#   PYTHON= (any python3; the converter is stdlib only)
#   LEROBOT_PYTHON= python with pyarrow, numpy and pillow for --from-lerobot (the flywheel venv), default $PYTHON
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI="${PI:-$(cd "$HERE/../../.." && pwd)}"
SH="${SH:-$HERE/showharness}"
export LF_VENV="${LF_VENV:-/root/autodl-tmp/venvs/llamafactory}"
export LF_ROOT="${LF_ROOT:-$LF_VENV/LlamaFactory}"
NAME="${NAME:?NAME (dataset/adapter name) is required}"
DATA="${DATA:-/root/autodl-tmp/data/finetuned/$NAME}"
VERSION="${VERSION:-v3}"
BASE_CONFIG="${BASE_CONFIG:-qwen3_5_2b_sim.yaml}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/checkpoints/Qwen3.5-2B}"
GPU="${GPU:-1}"
LOCK="${LOCK-/root/autodl-tmp/locks/gpu1.lock}"
PY="${PYTHON:-$(command -v python3 || echo /root/miniconda3/bin/python)}"
FROM=gumi
case "${1:-}" in --from-lerobot) FROM=lerobot; shift ;; --from-rollouts) FROM=rollouts; shift ;; esac
[ $# -gt 0 ] || { echo "usage: NAME=... bash train.sh <gumi record dir>... | --from-lerobot <lerobot dataset dir>... | --from-rollouts <rollout dir>..." >&2; exit 2; }
export PATH=/root/autodl-tmp/tools/node/bin:$PATH

if [ "$FROM" = rollouts ]; then
  echo "[1/3] rollouts -> $DATA/rollouts"
  rm -rf "$DATA/rollouts"
  mkdir -p "$DATA/rollouts"
  for d in "$@"; do
    [ -f "$d/rollout_000/metadata.json" ] || { echo "$d: no rollout_000/metadata.json" >&2; exit 2; }
    [ -e "$DATA/rollouts/$(basename "$d")" ] && { echo "two rollout dirs named $(basename "$d")" >&2; exit 2; }
    ln -s "$(cd "$d" && pwd)" "$DATA/rollouts/$(basename "$d")"
  done
elif [ "$FROM" = lerobot ]; then
  echo "[1/3] LeRobot -> rollouts ($DATA/rollouts)"
  rm -rf "$DATA/rollouts"
  for ds in "$@"; do
    # shellcheck disable=SC2086
    PYTHONPATH="$PI/services${PYTHONPATH:+:$PYTHONPATH}" "${LEROBOT_PYTHON:-$PY}" -m pi_embodied_services.finetuned.lerobot_to_rollouts \
      "$ds" --out "$DATA/rollouts" --prompt "$VERSION" ${PREPARE_ARGS:-}
  done
else
  echo "[1/3] GUMI -> rollouts ($DATA/rollouts)"
  rm -rf "$DATA/rollouts"
  # shellcheck disable=SC2086
  node --experimental-strip-types "$PI/packages/embodied/src/finetuned/prepare.ts" --out "$DATA/rollouts" ${PREPARE_ARGS:-} "$@"
fi

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
  --lf-root "$LF_ROOT"

echo "[3/3] LoRA SFT on GPU $GPU"
CONFIG="$DATA/$NAME.yaml"
sed -e "s|^dataset:.*|dataset: $NAME|" \
    -e "s|^output_dir:.*|output_dir: $DATA/saves/$NAME|" \
    -e "s|^run_name:.*|run_name: $NAME|" \
    -e "s|^report_to:.*|report_to: none|" \
    ${EPOCHS:+-e "s|^num_train_epochs:.*|num_train_epochs: $EPOCHS|"} \
    "$SH/train/configs/$BASE_CONFIG" > "$CONFIG"
for kv in ${SET:-}; do
  k="${kv%%=*}" v="${kv#*=}"
  if grep -q "^$k:" "$CONFIG"; then sed -i "s|^$k:.*|$k: $v|" "$CONFIG"; else echo "$k: $v" >> "$CONFIG"; fi
done
# DATA_PREP_PYTHON: their llamafactory_env.sh dies under set -e on a box without python3 on PATH.
train() { CONFIG="$CONFIG" GPU="$GPU" MODEL_PATH="$MODEL_PATH" DATA_PREP_PYTHON="$PY" bash "$SH/train/scripts/train.sh"; }
if [ -n "$LOCK" ]; then
  exec 9> "$LOCK"
  echo "waiting for $LOCK"
  flock 9
fi
train
echo "adapter checkpoints under $DATA/saves/$NAME; serve one with:"
echo "  LORA=$NAME=$DATA/saves/$NAME/checkpoint-<step> bash $HERE/serve.sh"
