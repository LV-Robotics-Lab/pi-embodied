#!/usr/bin/env bash
# Copyright 2026 The Show-Harness Authors. Licensed under the Apache License, Version 2.0.
# Modified by pi-embodied: scripts/serve_vlm.sh of github.com/showlab/Show-Harness @137d571 with this
# box's defaults (the shared vLLM 0.28 venv, its sm_120 CUDA toolkit, GPU1, port 8010) and the
# chat templates copied verbatim into ./chat_templates/.
#
# Serve a base VLM + Show-Harness LoRA adapter(s) on an OpenAI-compatible endpoint for
# `pi --model finetuned/<adapter> --ft-endpoint http://127.0.0.1:$PORT/v1` (packages/embodied/src/finetuned).
#
#   bash serve.sh                                      # the released sim adapter on GPU1, :8010
#   MODEL=<base dir> FAMILY=qwen3_5 LORA=<name>=<adapter dir>[,<name>=<dir>] bash serve.sh
#
# On the shared box GPU1 is time-shared: run it only while holding the lock, and stop it after
# the episodes (episode.sh does both):
#   flock /root/autodl-tmp/locks/gpu1.lock bash serve.sh
#
#   MODEL=      base weights dir (default /root/autodl-tmp/checkpoints/Qwen3.5-2B, from download.py)
#   LORA=       name=path,... ; the name is what the client requests (default the released sim adapter)
#   FAMILY=     qwen3_5 | internvl3_5 | gemma4: the chat template that reproduces training's rendering
#   PORT=8010   GPU=1   GPU_UTIL=0.3 (fraction of the whole card; GPU1 is shared)   MAX_LEN=8192
#   VLLM_VENV=  /root/autodl-tmp/venvs/vllm     DRY_RUN=1 prints the command
#
# Three silent failure modes this guards, as upstream: the adapter must be served with the jinja that
# reproduces LlamaFactory's training template (Qwen3.5's own emits an empty think block the adapter
# never saw); InternVL needs tie_word_embeddings forced off and the openai content format; a port
# already serving another model would absorb the requests.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CK=/root/autodl-tmp/checkpoints
MODEL="${MODEL:-$CK/Qwen3.5-2B}"
LORA="${LORA:-qwen3_5_2b_showharness_sim=$CK/Show-Harness-VLMs/qwen3_5_2b_sim}"
FAMILY="${FAMILY:-qwen3_5}"
PORT="${PORT:-8010}"
GPU="${GPU:-1}"
GPU_UTIL="${GPU_UTIL:-0.3}"
MAX_LEN="${MAX_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
VENV="${VLLM_VENV:-/root/autodl-tmp/venvs/vllm}"
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL")}"

LORA_ARGS=()
MISSING=()
IFS=',' read -ra _items <<< "$LORA"
for item in "${_items[@]}"; do
  item="$(echo "$item" | xargs)"; [ -z "$item" ] && continue
  [ -f "${item#*=}/adapter_model.safetensors" ] || MISSING+=("${item%%=*} -> ${item#*=}")
  LORA_ARGS+=("$item")
done
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "ERROR: no adapter_model.safetensors under:" >&2; printf '  %s\n' "${MISSING[@]}" >&2
  echo "  fetch one: python -m pi_embodied_services.finetuned.download --adapter qwen3_5_2b_sim" >&2
  exit 1
fi
[ -f "$MODEL/config.json" ] || { echo "ERROR: no base model at $MODEL" >&2; exit 1; }

case "$FAMILY" in
  qwen3_5) TEMPLATE="$HERE/chat_templates/qwen3_5_nothink.jinja" ;;
  gemma4) TEMPLATE="$HERE/chat_templates/gemma4n.jinja" ;;
  internvl3_5) TEMPLATE="$HERE/chat_templates/internvl3_5.jinja" ;;
  *) echo "ERROR: FAMILY must be qwen3_5 | internvl3_5 | gemma4, got: $FAMILY" >&2; exit 1 ;;
esac
EXTRA=(--chat-template "${CHAT_TEMPLATE:-$TEMPLATE}")
if [ "$FAMILY" = internvl3_5 ]; then
  EXTRA+=(--hf-overrides '{"tie_word_embeddings": false, "text_config": {"tie_word_embeddings": false}}')
  EXTRA+=(--chat-template-content-format openai)
fi
[ ${#LORA_ARGS[@]} -gt 0 ] && EXTRA+=(--enable-lora --max-lora-rank "$MAX_LORA_RANK" --lora-modules "${LORA_ARGS[@]}")

if curl -s -m 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "ERROR: port ${PORT} is already serving; it would absorb requests meant for this server:" >&2
  curl -s -m 2 "http://127.0.0.1:${PORT}/v1/models" | head -c 300 >&2; echo >&2
  exit 1
fi

# sm_120 FlashInfer JIT needs CUDA >= 12.9: the venv's pip cu13 toolkit, as tools/vllm-up.sh does.
CU13="$VENV/lib/python3.12/site-packages/nvidia/cu13"
if [ -d "$CU13" ]; then
  [ -x /root/autodl-tmp/tools/cuda-pip-links.sh ] && /root/autodl-tmp/tools/cuda-pip-links.sh >/dev/null
  export CUDA_HOME="$CU13" PATH="$CU13/bin:$PATH"
fi
export CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/root/autodl-tmp/.cache/vllm}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

CMD=(
  vllm serve "$MODEL"
  --dtype bfloat16
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --override-generation-config '{"temperature": 0, "top_p": 1.0, "top_k": -1}'
  --trust-remote-code
  --host "${HOST:-127.0.0.1}"
  --port "$PORT"
  --served-model-name "$SERVED_NAME"
  "${EXTRA[@]}"
)
[ "${ENFORCE_EAGER:-0}" = 1 ] && CMD+=(--enforce-eager)
echo "vLLM :$PORT GPU $GPU util $GPU_UTIL  base $MODEL  template ${TEMPLATE##*/}"
for m in "${LORA_ARGS[@]}"; do echo "  lora ${m%%=*} <- ${m#*=}"; done
if [ -n "${DRY_RUN:-}" ]; then printf '[dry-run]'; printf ' %q' "${CMD[@]}"; echo; exit 0; fi
exec "${CMD[@]}"
