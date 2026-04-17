#!/bin/bash
# v11 suite: fine-grain around v10 champion x_8d_m2
# (NUM_LAYERS=8, MLP_MULT=2, MATRIX_LR=0.04, MUON_MOMENTUM=0.95, WARMDOWN_ITERS=200)
# int8+zlib val_bpb = 1.9039 (beats v8d 7D_m2 = 1.9071 by -0.0032).
# Vary one axis at a time around the new center.
#
# Config format: "TAG:NUM_LAYERS:MLP_MULT:MATRIX_LR:MUON_MOMENTUM:WARMDOWN_ITERS:EXTRA_ENV"

set -u
export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_8192_bpe.model}"

LOGDIR=/workspace/sota_crawler/logs/v11_suite
mkdir -p "$LOGDIR"
cd /workspace/sota_crawler

COMMON_ENV=(
  VOCAB_SIZE=8192
  ADD_MLP=1
  NUM_HEADS=8
  NUM_KV_HEADS=4
  ITERATIONS=300
  VAL_LOSS_EVERY=100
  TRAIN_LOG_EVERY=50
  MAX_WALLCLOCK_SECONDS=3600
  GRAD_CLIP_NORM=0.0
  WARMUP_STEPS=20
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

# Queue: fine-grain around v10 champion (8D MLP=2 LR=0.04 MOM=0.95 WD=200).
QUEUE=(
  # LR fine-grain around 0.04
  "lr025_8d_m2:8:2:0.025:0.95:200:"
  "lr028_8d_m2:8:2:0.028:0.95:200:"
  "lr035_8d_m2:8:2:0.035:0.95:200:"
  "lr045_8d_m2:8:2:0.045:0.95:200:"
  "lr050_8d_m2:8:2:0.050:0.95:200:"
  # Warmdown fine-grain around 200
  "wd0_8d_m2:8:2:0.04:0.95:0:"
  "wd50_8d_m2:8:2:0.04:0.95:50:"
  "wd100_8d_m2:8:2:0.04:0.95:100:"
  "wd150_8d_m2:8:2:0.04:0.95:150:"
  "wd300_8d_m2:8:2:0.04:0.95:300:"
  # Momentum
  "mom90_8d_m2:8:2:0.04:0.90:200:"
  "mom97_8d_m2:8:2:0.04:0.97:200:"
  "mom99_8d_m2:8:2:0.04:0.99:200:"
  # Depth re-probe
  "10d_m2:10:2:0.04:0.95:200:"
)

pids_by_gpu=(0 0 0 0)
tags_by_gpu=("" "" "" "")
QIDX=0
TS() { date -u +%Y-%m-%dT%H:%M:%SZ; }

launch_on() {
  local gpu="$1" cfg="$2"
  IFS=":" read -r tag nlayers mlp_mult mat_lr mom wd extra <<<"$cfg"
  local log="$LOGDIR/v11_${tag}.log"
  local master_port=$((29560 + gpu))
  echo "[v11 $(TS)] GPU$gpu launch tag=$tag (L=$nlayers MLP=$mlp_mult LR=$mat_lr MOM=$mom WD=$wd) -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" env "${COMMON_ENV[@]}" \
    NUM_LAYERS="$nlayers" MLP_MULT="$mlp_mult" \
    MATRIX_LR="$mat_lr" MUON_MOMENTUM="$mom" \
    WARMDOWN_ITERS="$wd" \
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
    echo "[v11 $(TS)] GPU$gpu finished tag=${tags_by_gpu[$gpu]}"
    pids_by_gpu[$gpu]=0
    tags_by_gpu[$gpu]=""
  fi
  return 0
}

echo "[v11 $(TS)] queue=${#QUEUE[@]} trials, dispatching..."

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

echo "[v11 $(TS)] queue drained, waiting on running trials..."
for gpu in 0 1 2 3; do
  pid="${pids_by_gpu[$gpu]}"
  [[ "$pid" -ne 0 ]] && wait "$pid" || true
done
echo "[v11 $(TS)] all v11 trials finished"
