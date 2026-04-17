#!/bin/bash
# v14 suite: chaos orthogonal — explore MLP_MULT, NUM_HEADS, MATRIX_LR,
# WARMDOWN_ITERS, and FIRST_LAYER_NOISE at the most promising chaos-heavy
# regions surfaced (or expected) by v13. Fills 16MB cap headroom that opens
# up when CHAOS_DEPTH replaces NUM_LAYERS as the depth axis.
#
# Config format: "TAG:NUM_LAYERS:MLP_MULT:CHAOS_DEPTH[:EXTRA_ENV]"
# EXTRA_ENV is optional, space-separated KEY=VALUE pairs.
# Everything else pinned at v13 champion values (MATRIX_LR=0.035,
# MUON_MOMENTUM=0.95, WARMDOWN_ITERS=200).

set -u
export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_8192_bpe.model}"

LOGDIR=/workspace/sota_crawler/logs/v14_suite
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

# Queue: orthogonal axes at chaos-heavy configs (~16 trials).
# EXTRA_ENV overrides COMMON_ENV defaults for that trial only (env var order:
# COMMON_ENV first, then explicit NUM_LAYERS/MLP_MULT/VORTEX_CHAOS_DEPTH,
# then $extra; the last assignment wins under `env`).
QUEUE=(
  # MLP width at deep chaos (fills 16MB cap headroom if CHAOS replaces layers)
  "l1_c40_m1:1:1:40"
  "l1_c40_m2:1:2:40"
  "l1_c40_m3:1:3:40"
  "l1_c40_m6:1:6:40"
  "l1_c40_m8:1:8:40"
  # NUM_HEADS variation at deep chaos
  "l1_c40_h4:1:2:40:NUM_HEADS=4 NUM_KV_HEADS=2"
  "l1_c40_h16:1:2:40:NUM_HEADS=16 NUM_KV_HEADS=8"
  # LR retune at deep chaos (may need scaling with CHAOS_DEPTH)
  "l1_c40_lr02:1:2:40:MATRIX_LR=0.02"
  "l1_c40_lr05:1:2:40:MATRIX_LR=0.05"
  "l1_c40_lr08:1:2:40:MATRIX_LR=0.08"
  # Warmdown retune at deep chaos
  "l1_c40_wd50:1:2:40:WARMDOWN_ITERS=50"
  "l1_c40_wd100:1:2:40:WARMDOWN_ITERS=100"
  "l1_c40_wd300:1:2:40:WARMDOWN_ITERS=300"
  # L=2 x CHAOS=20 combo variants
  "l2_c20_m4:2:4:20"
  "l2_c20_m6:2:6:20"
  # FIRST_LAYER_NOISE at deep chaos (regularizer)
  "l1_c40_fln:1:2:40:FIRST_LAYER_NOISE=1 FIRST_LAYER_NOISE_SIGMA=0.01"
)

pids_by_gpu=(0 0 0 0)
tags_by_gpu=("" "" "" "")
QIDX=0
TS() { date -u +%Y-%m-%dT%H:%M:%SZ; }

launch_on() {
  local gpu="$1" cfg="$2"
  IFS=":" read -r tag nlayers mlp_mult chaos_depth extra <<<"$cfg"
  local log="$LOGDIR/v14_${tag}.log"
  local master_port=$((29620 + gpu))
  echo "[v14 $(TS)] GPU$gpu launch tag=$tag (L=$nlayers MLP=$mlp_mult CHAOS=$chaos_depth extra=$extra) -> $log"
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
    echo "[v14 $(TS)] GPU$gpu finished tag=${tags_by_gpu[$gpu]}"
    pids_by_gpu[$gpu]=0
    tags_by_gpu[$gpu]=""
  fi
  return 0
}

echo "[v14 $(TS)] queue=${#QUEUE[@]} trials, dispatching..."

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

echo "[v14 $(TS)] queue drained, waiting on running trials..."
for gpu in 0 1 2 3; do
  pid="${pids_by_gpu[$gpu]}"
  [[ "$pid" -ne 0 ]] && wait "$pid" || true
done
echo "[v14 $(TS)] all v14 trials finished"
