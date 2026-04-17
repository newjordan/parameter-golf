#!/bin/bash
# v11c lr035 9D: 2000-step probe at 9D depth with new champ LR.
# Tests whether extra depth opens up with longer training (300-step 9D was worse; 2k may reverse).
# Config: NUM_LAYERS=9, MLP_MULT=2, ADD_MLP=1, MATRIX_LR=0.035, MUON_MOMENTUM=0.95.
# WARMDOWN_ITERS scaled 200 -> 1333 (proportional), WARMUP_STEPS 20 -> 100.
# 4xH100 x 2000 iters.

set -euo pipefail
export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_8192_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-8192}"
export NUM_LAYERS="${NUM_LAYERS:-9}"
export MLP_MULT="${MLP_MULT:-2}"
export ADD_MLP="${ADD_MLP:-1}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export MATRIX_LR="${MATRIX_LR:-0.035}"
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

echo "Launching v11c_lr035_9d: 9D MLP=2 lr=0.035 depth-probe + 4xH100 + ${ITERATIONS} iters"
/venv/main/bin/torchrun --standalone --nproc_per_node=4 test_vortex_2k.py
