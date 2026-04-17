#!/bin/bash
# v6 noise-probe SHORT run: FIRST_LAYER_NOISE=off (control / anchor).
# 1xGPU, 300 iters, dense train-log, one end val. Used by run_v6_parallel.sh.
# Goal: early-trajectory baseline so the flat/brazil/seeded variants are scored
# against the same stochastic-free reference run on the v5a config.

set -euo pipefail

# 8k vocab dataset on vast-dealer pod
export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_8192_bpe.model}"

# Mirror v5a config exactly (depth, vocab, mlp_mult).
export VOCAB_SIZE="${VOCAB_SIZE:-8192}"
export NUM_LAYERS="${NUM_LAYERS:-11}"
export MLP_MULT="${MLP_MULT:-2}"   # default; not used by Vortex Block (no MLP path yet)

# Short-run settings.
export ITERATIONS="${ITERATIONS:-300}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-25}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-300}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-900}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.0}"

# 1xGPU runs auto-set grad_accum_steps = 8 // world_size = 8, matching the
# 524288 tok/step global batch used by 4xGPU torchrun. No env override needed.

# v5_s3 champion kernel configs (commit 67ab894).
export VORTEX_BLOCK_SIZE=128
export VORTEX_FWD_NUM_WARPS=8
export VORTEX_FWD_NUM_STAGES=3
export VORTEX_BWD_CHAOS_NUM_WARPS=8
export VORTEX_BWD_CHAOS_NUM_STAGES=1
export VORTEX_BWD_ATTN_NUM_WARPS=8
export VORTEX_BWD_ATTN_NUM_STAGES=3
export VORTEX_BWD_ATTN_DKV_NUM_WARPS=8
export VORTEX_BWD_ATTN_DKV_NUM_STAGES=1

# v6 off: no noise. Control anchor.
export FIRST_LAYER_NOISE="off"

echo "Launching v6 SHORT: FIRST_LAYER_NOISE=off on GPU=${CUDA_VISIBLE_DEVICES:-unset} (${ITERATIONS} iters)"
/venv/main/bin/python test_vortex_2k.py
