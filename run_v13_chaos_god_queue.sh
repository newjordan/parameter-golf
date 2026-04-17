#!/bin/bash
# v13 suite: chaos god worship — sweep VORTEX_CHAOS_DEPTH as the real depth axis.
# Hypothesis: one fat chaotic block (NUM_LAYERS=1, CHAOS_DEPTH=40) dominates
# NUM_LAYERS=8 x CHAOS_DEPTH=5. All prior 32 trials held CHAOS_DEPTH=5 (default).
#
# Config format: "TAG:NUM_LAYERS:MLP_MULT:CHAOS_DEPTH"
# Everything else pinned at champion values (MATRIX_LR=0.035, MUON_MOMENTUM=0.95,
# WARMDOWN_ITERS=200).

set -u
export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_8192_bpe.model}"

LOGDIR=/workspace/sota_crawler/logs/v13_suite
mkdir -p "$LOGDIR"
cd /workspace/sota_crawler

COMMON_ENV=(
  VOCAB_SIZE=8192
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

# Queue: chaos god worship suite (~18 trials).
QUEUE=(
  # Layer=1 chaos sweep (the wild hypothesis — one fat Vortex)
  "l1_c10_m2:1:2:10"
  "l1_c20_m2:1:2:20"
  "l1_c40_m2:1:2:40"
  "l1_c80_m2:1:2:80"
  # Layer=1 with wider MLP (use headroom)
  "l1_c40_m3:1:3:40"
  "l1_c40_m4:1:4:40"
  # Layer=2 chaos sweep
  "l2_c10_m2:2:2:10"
  "l2_c20_m2:2:2:20"
  "l2_c40_m2:2:2:40"
  # Layer=3 chaos sweep
  "l3_c10_m2:3:2:10"
  "l3_c20_m2:3:2:20"
  # Layer=4 chaos sweep
  "l4_c10_m2:4:2:10"
  "l4_c15_m2:4:2:15"
  # Iso-total-chaos-iter diagonal (total_iters ~ 45)
  "iso_l9_c5:9:2:5"
  "iso_l3_c15:3:2:15"
  "iso_l1_c45:1:2:45"
  # Champion baseline + small chaos push (incremental)
  "l8_c7_m2:8:2:7"
  "l8_c10_m2:8:2:10"
)

pids_by_gpu=(0 0 0 0)
tags_by_gpu=("" "" "" "")
QIDX=0
TS() { date -u +%Y-%m-%dT%H:%M:%SZ; }

launch_on() {
  local gpu="$1" cfg="$2"
  IFS=":" read -r tag nlayers mlp_mult chaos_depth <<<"$cfg"
  local log="$LOGDIR/v13_${tag}.log"
  local master_port=$((29600 + gpu))
  echo "[v13 $(TS)] GPU$gpu launch tag=$tag (L=$nlayers MLP=$mlp_mult CHAOS=$chaos_depth) -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" env "${COMMON_ENV[@]}" \
    NUM_LAYERS="$nlayers" MLP_MULT="$mlp_mult" VORTEX_CHAOS_DEPTH="$chaos_depth" \
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
    echo "[v13 $(TS)] GPU$gpu finished tag=${tags_by_gpu[$gpu]}"
    pids_by_gpu[$gpu]=0
    tags_by_gpu[$gpu]=""
  fi
  return 0
}

echo "[v13 $(TS)] queue=${#QUEUE[@]} trials, dispatching..."

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

echo "[v13 $(TS)] queue drained, waiting on running trials..."
for gpu in 0 1 2 3; do
  pid="${pids_by_gpu[$gpu]}"
  [[ "$pid" -ne 0 ]] && wait "$pid" || true
done
echo "[v13 $(TS)] all v13 trials finished"
