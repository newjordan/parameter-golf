#!/bin/bash
# v20 suite: VOCAB_SIZE=4096 + depth/MLP reinvestment.
#
# PREREQ: 4k-vocab BPE tokenizer + dataset shards pre-tokenized at 4k vocab.
# Required files (check before running):
#   /workspace/Fartmagic/data/tokenizers/fineweb_4096_bpe.model
#   /workspace/Fartmagic/data/datasets/fineweb10B_sp4096/  (shard dir)
#
# STATUS 2026-04-17: 4k tokenizer DOES NOT EXIST on pod. Available vocabs:
#   fineweb_1024_bpe.model + fineweb10B_sp1024/
#   fineweb_8192_bpe.model + fineweb10B_sp8192/  (current default)
# Options to unblock this sweep:
#   (a) train a 4k SentencePiece model on FineWeb subset (~5 min) + retokenize
#       shards (~10-20 min). Preferred.
#   (b) run a 1k fallback variant using existing sp1024 shards (tokenizer too
#       small per ChatGPT Q8 guidance, but cheap to probe).
# If either is missing, run tokenizer build first (SentencePiece train on a
# FineWeb subset, then retokenize shards). That is a ~15-25 min precheck.
#
# ChatGPT deep research 2026-04-17 (Q8): 8192 vocab * 512 dim = 4.19M tied-embed
# params. Dropping to 4k frees ~2.1M params. Scaling-law evidence says small
# models deserve small vocabs; we're 3M params total, far below the regime where
# 8k+ vocab pays for itself.
#
# Strategy: test 4k-vocab at champion shape first (baseline check). Then
# reinvest the saved budget into more layers / wider MLP and look for a
# BPB drop vs 8k champion 1.89896.

set -u
export DATA_PATH_4K="${DATA_PATH_4K:-/workspace/Fartmagic/data/datasets/fineweb10B_sp4096}"
export TOKENIZER_PATH_4K="${TOKENIZER_PATH_4K:-/workspace/Fartmagic/data/tokenizers/fineweb_4096_bpe.model}"

# Preflight.
if [[ ! -d "$DATA_PATH_4K" || ! -f "$TOKENIZER_PATH_4K" ]]; then
  echo "[v20] PREREQ MISSING:"
  echo "  DATA_PATH_4K=$DATA_PATH_4K (exists=$(test -d "$DATA_PATH_4K" && echo yes || echo no))"
  echo "  TOKENIZER_PATH_4K=$TOKENIZER_PATH_4K (exists=$(test -f "$TOKENIZER_PATH_4K" && echo yes || echo no))"
  echo "Build 4k tokenizer + retokenize shards before running this sweep."
  exit 1
fi

export DATA_PATH="$DATA_PATH_4K"
export TOKENIZER_PATH="$TOKENIZER_PATH_4K"

LOGDIR=/workspace/sota_crawler/logs/v20_suite
mkdir -p "$LOGDIR"
cd /workspace/sota_crawler

COMMON_ENV=(
  VOCAB_SIZE=4096
  ADD_MLP=1
  NUM_HEADS=8
  NUM_KV_HEADS=4
  MATRIX_LR=0.035
  MUON_MOMENTUM=0.95
  WARMDOWN_ITERS=200
  ITERATIONS=300
  VAL_LOSS_EVERY=100
  TRAIN_LOG_EVERY=50
  MAX_WALLCLOCK_SECONDS=3600
  GRAD_CLIP_NORM=0.0
  WARMUP_STEPS=20
  QK_GAIN_INIT=0.5
  VORTEX_BLOCK_SIZE=128
  VORTEX_FWD_NUM_WARPS=8
  VORTEX_FWD_NUM_STAGES=3
  VORTEX_BWD_CHAOS_NUM_WARPS=8
  VORTEX_BWD_CHAOS_NUM_STAGES=1
  VORTEX_BWD_ATTN_NUM_WARPS=8
  VORTEX_BWD_ATTN_NUM_STAGES=3
  VORTEX_BWD_ATTN_DKV_NUM_WARPS=8
  VORTEX_BWD_ATTN_DKV_NUM_STAGES=1
)

# Baseline (4k vocab at champion shape) + depth/MLP reinvestments.
# Note: BPB is corpus-bytes-per-prediction so SMALLER vocab => more tokens per byte =>
# val_bpb should be DIRECTLY comparable across vocabs; only change is parameter allocation.
QUEUE=(
  "v4k_L8_M2:8:2:5"
  "v4k_L10_M2:10:2:5"
  "v4k_L12_M2:12:2:5"
  "v4k_L8_M3:8:3:5"
  "v4k_L10_M3:10:3:5"
  "v4k_L8_M2_c8:8:2:8"
)

pids_by_gpu=(0 0 0 0)
tags_by_gpu=("" "" "" "")
QIDX=0
TS() { date -u +%Y-%m-%dT%H:%M:%SZ; }

launch_on() {
  local gpu="$1" cfg="$2"
  IFS=":" read -r tag nlayers mlp_mult chaos_depth extra <<<"$cfg"
  local log="$LOGDIR/v20_${tag}.log"
  local master_port=$((29760 + gpu))
  echo "[v20 $(TS)] GPU$gpu launch tag=$tag (L=$nlayers MLP=$mlp_mult CHAOS=$chaos_depth extra=$extra) -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" env "${COMMON_ENV[@]}" \
    NUM_LAYERS="$nlayers" MLP_MULT="$mlp_mult" VORTEX_CHAOS_DEPTH="$chaos_depth" \
    $extra \
    /venv/main/bin/torchrun --standalone --nproc_per_node=1 \
      --master_port="$master_port" \
      test_vortex_2k.py > "$log" 2>&1 &
  pids_by_gpu[$gpu]=$!
  tags_by_gpu[$gpu]="$tag"
}

gpu_free() {
  local gpu="$1"
  local pid="${pids_by_gpu[$gpu]}"
  if [[ "$pid" -ne 0 ]] && kill -0 "$pid" 2>/dev/null; then
    return 1
  fi
  local mem
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
  if [[ -z "$mem" ]] || [[ "$mem" -gt 1000 ]]; then
    return 1
  fi
  if [[ "$pid" -ne 0 ]]; then
    echo "[v20 $(TS)] GPU$gpu finished tag=${tags_by_gpu[$gpu]}"
    pids_by_gpu[$gpu]=0
    tags_by_gpu[$gpu]=""
  fi
  return 0
}

echo "[v20 $(TS)] queue=${#QUEUE[@]} trials, dispatching..."

while [[ $QIDX -lt ${#QUEUE[@]} ]]; do
  for gpu in 0 1 2 3; do
    if gpu_free "$gpu"; then
      if [[ $QIDX -lt ${#QUEUE[@]} ]]; then
        launch_on "$gpu" "${QUEUE[$QIDX]}"
        QIDX=$((QIDX + 1))
      fi
    fi
  done
  sleep 10
done

echo "[v20 $(TS)] queue drained, waiting on running trials..."
for gpu in 0 1 2 3; do
  pid="${pids_by_gpu[$gpu]}"
  [[ "$pid" -ne 0 ]] && wait "$pid" || true
done
echo "[v20 $(TS)] all v20 trials finished"
