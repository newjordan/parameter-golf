#!/bin/bash
# v10 suite: extend the v9 depth x MLP sweep into higher-capacity cells that
# MIGHT fit 16MB after int8+zlib compression (like v8d 7D_m2 at 12.7MB).
# Also test MLP=2 at the proven 7D-8D depth band more thoroughly.
#
# Config format: "TAG:NUM_LAYERS:MLP_MULT:MATRIX_LR:MUON_MOMENTUM:WARMDOWN_ITERS:EXTRA_ENV"

set -u
export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_8192_bpe.model}"

LOGDIR=/workspace/sota_crawler/logs/v10_suite
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

# Queue: extend best configs with MLP=2 at more depths + MLP=3 exploration + LR around winner.
QUEUE=(
  # MLP=2 depth sweep (v8d was 7D=1.9071 champ)
  "x_8d_m2:8:2:0.04:0.95:200:"
  "x_9d_m2:9:2:0.04:0.95:200:"
  "x_6d_m3:6:3:0.04:0.95:200:"
  "x_7d_m3:7:3:0.04:0.95:200:"
  "x_5d_m4:5:4:0.04:0.95:200:"
  # LR sweep at v8d champion config (7D MLP=2)
  "xo_7d_m2_lr022:7:2:0.022:0.95:200:"
  "xo_7d_m2_lr03:7:2:0.03:0.95:200:"
  "xo_7d_m2_lr05:7:2:0.05:0.95:200:"
  "xo_7d_m2_lr06:7:2:0.06:0.95:200:"
  # Momentum + schedule at champion
  "xo_7d_m2_mom99:7:2:0.04:0.99:200:"
  "xo_7d_m2_mom90:7:2:0.04:0.90:200:"
  "xs_7d_m2_wd0:7:2:0.04:0.95:0:"
  "xs_7d_m2_wd100:7:2:0.04:0.95:100:"
  "xs_7d_m2_wd300:7:2:0.04:0.95:300:"
)

pids_by_gpu=(0 0 0 0)
tags_by_gpu=("" "" "" "")
QIDX=0
TS() { date -u +%Y-%m-%dT%H:%M:%SZ; }

launch_on() {
  local gpu="$1" cfg="$2"
  IFS=":" read -r tag nlayers mlp_mult mat_lr mom wd extra <<<"$cfg"
  local log="$LOGDIR/v10_${tag}.log"
  local master_port=$((29540 + gpu))
  echo "[v10 $(TS)] GPU$gpu launch tag=$tag (L=$nlayers MLP=$mlp_mult LR=$mat_lr MOM=$mom WD=$wd) -> $log"
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
    echo "[v10 $(TS)] GPU$gpu finished tag=${tags_by_gpu[$gpu]}"
    pids_by_gpu[$gpu]=0
    tags_by_gpu[$gpu]=""
  fi
  return 0
}

echo "[v10 $(TS)] queue=${#QUEUE[@]} trials, dispatching..."

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

echo "[v10 $(TS)] queue drained, waiting on running trials..."
for gpu in 0 1 2 3; do
  pid="${pids_by_gpu[$gpu]}"
  [[ "$pid" -ne 0 ]] && wait "$pid" || true
done
echo "[v10 $(TS)] all v10 trials finished"
