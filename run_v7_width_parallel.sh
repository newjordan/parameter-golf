#!/bin/bash
# v7 width-ratio sweep: 4 parallel 1xGPU runs x 300 steps at 11D + 8k vocab.
# Tests NUM_HEADS in {8,4,2} to widen head_dim from 64 to 128/256.
# Narrow point: proj_weight_hd = self.proj.weight[:head_dim, :head_dim]
# only uses head_dim^2 of the 512x512 proj matrix. Wider head_dim -> more params engaged.
#
# T1 GPU0: h8/d64   anchor (v5a width)
# T2 GPU1: h4/d128  2x wider
# T3 GPU2: h2/d256  4x wider (uses 25% of proj matrix)
# T4 GPU3: h4/d128 + matrix_lr=0.06  width + LR combo
#
# Target: train_loss < 4.0 by step 300 -> projects to <= 3.0 at step 1500.

set -euo pipefail
export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_8192_bpe.model}"

LOGDIR=/workspace/sota_crawler/logs/v7_width
mkdir -p "$LOGDIR"

common_env=(
  VOCAB_SIZE=8192
  NUM_LAYERS=11
  MLP_MULT=2
  ITERATIONS=300
  VAL_LOSS_EVERY=100
  TRAIN_LOG_EVERY=50
  MAX_WALLCLOCK_SECONDS=3600
  GRAD_CLIP_NORM=0.0
  WARMDOWN_ITERS=200
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

launch() {
  local gpu="$1" tag="$2"; shift 2
  local log="$LOGDIR/v7_${tag}.log"
  echo "[v7 $(date -u +%Y-%m-%dT%H:%M:%SZ)] GPU$gpu tag=$tag -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" \
    env "${common_env[@]}" "$@" \
    /venv/main/bin/torchrun --standalone --nproc_per_node=1 \
      --master_port=$((29500 + gpu)) \
      test_vortex_2k.py \
      > "$log" 2>&1 &
  echo "$!" > "$LOGDIR/v7_${tag}.pid"
}

cd /workspace/sota_crawler

launch 0 t1_h8_d64    NUM_HEADS=8 NUM_KV_HEADS=4 MATRIX_LR=0.04
launch 1 t2_h4_d128   NUM_HEADS=4 NUM_KV_HEADS=2 MATRIX_LR=0.04
launch 2 t3_h2_d256   NUM_HEADS=2 NUM_KV_HEADS=1 MATRIX_LR=0.04
launch 3 t4_h4_lr06   NUM_HEADS=4 NUM_KV_HEADS=2 MATRIX_LR=0.06

wait
echo "[v7 $(date -u +%Y-%m-%dT%H:%M:%SZ)] all four width trials finished"
