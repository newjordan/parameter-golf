#!/bin/bash
# Parallel NaN diagnostic: 4x 1xGPU runs, 300 steps each.
# Each run isolates one suspected cause; logs every step for divergence onset.

set -euo pipefail

export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_1024_bpe.model}"
export ITERATIONS=300
export MAX_WALLCLOCK_SECONDS=900
export VAL_LOSS_EVERY=10000  # don't waste time on val during diagnostic
export TRAIN_LOG_EVERY=1  # log every step

# v5_s3 champion kernel configs
COMMON_KERNEL="VORTEX_BLOCK_SIZE=128 VORTEX_FWD_NUM_WARPS=8 VORTEX_FWD_NUM_STAGES=3 VORTEX_BWD_CHAOS_NUM_WARPS=8 VORTEX_BWD_CHAOS_NUM_STAGES=1 VORTEX_BWD_ATTN_NUM_WARPS=8 VORTEX_BWD_ATTN_NUM_STAGES=3 VORTEX_BWD_ATTN_DKV_NUM_WARPS=8 VORTEX_BWD_ATTN_DKV_NUM_STAGES=1"

mkdir -p logs/nan_diag

run_diag () {
  local gpu=$1
  local name=$2
  local extra_env=$3
  local logf="logs/nan_diag/${name}.log"
  echo "[GPU${gpu}] Launching ${name} -> ${logf}"
  nohup env CUDA_VISIBLE_DEVICES=${gpu} ${COMMON_KERNEL} ${extra_env} \
    ITERATIONS=${ITERATIONS} \
    MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS} \
    VAL_LOSS_EVERY=${VAL_LOSS_EVERY} \
    TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY} \
    DATA_PATH=${DATA_PATH} \
    TOKENIZER_PATH=${TOKENIZER_PATH} \
    /venv/main/bin/python test_vortex_2k.py > "${logf}" 2>&1 &
  echo "[GPU${gpu}] PID=$!"
}

# GPU0: baseline — current code, default config (chaos_depth=5, GQA, no clip)
run_diag 0 "A_baseline" ""

# GPU1: + grad clip 1.0 — tests if simple clipping rescues
run_diag 1 "B_gradclip" "GRAD_CLIP_NORM=1.0"

# GPU2: chaos_depth=0 — disables chaos bwd path, isolates pure attention kernel
run_diag 2 "C_chaos0" "VORTEX_CHAOS_DEPTH=0"

# GPU3: num_kv_heads=8 — disables GQA repeat_interleave
run_diag 3 "D_no_gqa" "NUM_KV_HEADS=8"

echo "All 4 diagnostics launched. Logs in logs/nan_diag/"
