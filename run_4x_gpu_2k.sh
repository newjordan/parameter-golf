#!/bin/bash
# VortexHelix 4x GPU, 2000-step launcher — v5_s3 champion kernel configs.
# Comparison target: last night's 1xGPU eager baseline (step_avg=1581ms, val_bpb=1.4843).

set -euo pipefail

# Canonical Fartmagic pod data layout (from Im_sorry_pod_setup.sh sp1024 download).
export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_1024_bpe.model}"

export ITERATIONS="${ITERATIONS:-2000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-7200}"
# Tanh fix alone prevents NaN now; grad_clip=1.0 was starving convergence post-muon-warmup.
# Setting to 0 (off) unless caller overrides.
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.0}"

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

echo "Launching VortexHelix 4xH100 2k run — ITERATIONS=${ITERATIONS}"
/venv/main/bin/torchrun --standalone --nproc_per_node=4 test_vortex_2k.py
