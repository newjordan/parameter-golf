#!/bin/bash
# v5d: 13D vortex-only + 8k vocab. Over-depth probe to confirm 11D is the knee.

set -euo pipefail
export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_8192_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-8192}"
export NUM_LAYERS="${NUM_LAYERS:-13}"
export MLP_MULT="${MLP_MULT:-2}"
export ITERATIONS="${ITERATIONS:-2000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-7200}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.0}"
export VORTEX_BLOCK_SIZE=128
export VORTEX_FWD_NUM_WARPS=8
export VORTEX_FWD_NUM_STAGES=3
export VORTEX_BWD_CHAOS_NUM_WARPS=8
export VORTEX_BWD_CHAOS_NUM_STAGES=1
export VORTEX_BWD_ATTN_NUM_WARPS=8
export VORTEX_BWD_ATTN_NUM_STAGES=3
export VORTEX_BWD_ATTN_DKV_NUM_WARPS=8
export VORTEX_BWD_ATTN_DKV_NUM_STAGES=1

echo "Launching v5d: 13D vortex-only + 8k vocab + 4xH100 + ${ITERATIONS} iters"
/venv/main/bin/torchrun --standalone --nproc_per_node=4 test_vortex_2k.py
