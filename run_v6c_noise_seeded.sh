#!/bin/bash
# v6c: 11D vortex-only + 8k vocab + 4xH100 + 2k iters + FIRST_LAYER_NOISE=token_seeded.
# Deterministic, content-dependent noise: seed = (internal_step XOR input_ids.sum()).
# Tests whether content-tied reproducible noise differs from the iid flat case.

set -euo pipefail

# 8k vocab dataset on vast-dealer pod
export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_8192_bpe.model}"

# Mirror v5a config exactly (depth, vocab, mlp_mult).
export VOCAB_SIZE="${VOCAB_SIZE:-8192}"
export NUM_LAYERS="${NUM_LAYERS:-11}"
export MLP_MULT="${MLP_MULT:-2}"   # default; not used by Vortex Block (no MLP path yet)

export ITERATIONS="${ITERATIONS:-2000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-7200}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.0}"   # tanh fix prevents NaN; clip starves Muon

# v5_s3 champion kernel configs (commit 67ab894)
export VORTEX_BLOCK_SIZE=128
export VORTEX_FWD_NUM_WARPS=8
export VORTEX_FWD_NUM_STAGES=3
export VORTEX_BWD_CHAOS_NUM_WARPS=8
export VORTEX_BWD_CHAOS_NUM_STAGES=1
export VORTEX_BWD_ATTN_NUM_WARPS=8
export VORTEX_BWD_ATTN_NUM_STAGES=3
export VORTEX_BWD_ATTN_DKV_NUM_WARPS=8
export VORTEX_BWD_ATTN_DKV_NUM_STAGES=1

# v6c-specific: token/step-seeded deterministic first-layer noise injection.
export FIRST_LAYER_NOISE="token_seeded"
export FIRST_LAYER_NOISE_SIGMA="0.1"

echo "Launching v6c: 11D vortex-only + 8k vocab + 4xH100 + ${ITERATIONS} iters"
echo "  FIRST_LAYER_NOISE=${FIRST_LAYER_NOISE} FIRST_LAYER_NOISE_SIGMA=${FIRST_LAYER_NOISE_SIGMA}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "  NUM_LAYERS=${NUM_LAYERS}"
/venv/main/bin/torchrun --standalone --nproc_per_node=4 test_vortex_2k.py
