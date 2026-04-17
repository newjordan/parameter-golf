#!/bin/bash
# v11b: 2000-step production validation of v10 champion x_8d_m2.
# Config: NUM_LAYERS=8, MLP_MULT=2, ADD_MLP=1, MATRIX_LR=0.04, MUON_MOMENTUM=0.95.
# WARMDOWN_ITERS scaled 200 -> 1333 (proportional), WARMUP_STEPS 20 -> 100.
# 4xH100 x 2000 iters.

set -euo pipefail
export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_8192_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-8192}"
export NUM_LAYERS="${NUM_LAYERS:-8}"
export MLP_MULT="${MLP_MULT:-2}"
export ADD_MLP="${ADD_MLP:-1}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export MATRIX_LR="${MATRIX_LR:-0.04}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-1333}"
export WARMUP_STEPS="${WARMUP_STEPS:-100}"
export ITERATIONS="${ITERATIONS:-2000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-7200}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.0}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
export VORTEX_BLOCK_SIZE=128
export VORTEX_FWD_NUM_WARPS=8
export VORTEX_FWD_NUM_STAGES=3
export VORTEX_BWD_CHAOS_NUM_WARPS=8
export VORTEX_BWD_CHAOS_NUM_STAGES=1
export VORTEX_BWD_ATTN_NUM_WARPS=8
export VORTEX_BWD_ATTN_NUM_STAGES=3
export VORTEX_BWD_ATTN_DKV_NUM_WARPS=8
export VORTEX_BWD_ATTN_DKV_NUM_STAGES=1

echo "Launching v11b: 8D MLP=2 champion validation + 4xH100 + ${ITERATIONS} iters"
/venv/main/bin/torchrun --standalone --nproc_per_node=4 test_vortex_2k.py
