#!/bin/bash
# Parameter Golf — Cloud H100 Launch Script
set -e

NGPU=${1:-1}
MODE=${2:-baseline}

echo "=========================================="
echo "PARAMETER GOLF — CLOUD RUN"
echo "GPUs: ${NGPU} | Mode: ${MODE}"
echo "=========================================="

# ── SETUP ──
if [ ! -d "/workspace/parameter-golf/.git" ]; then
    cd /workspace
    rm -rf parameter-golf
    git clone https://github.com/newjordan/parameter-golf.git
    cd parameter-golf
else
    cd /workspace/parameter-golf
fi

# ── DATA ──
if [ ! -f "./data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin" ]; then
    echo "Downloading data..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
else
    echo "Data already present."
fi

# ── COMMON ENV ──
export DATA_PATH=./data/datasets/fineweb10B_sp1024/
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export MAX_WALLCLOCK_SECONDS=600
export TRAIN_LOG_EVERY=50
export VAL_LOSS_EVERY=200
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024
export TIE_EMBEDDINGS=1
export TIED_EMBED_LR=0.05

# ── MODE ──
case $MODE in
    baseline)
        export RUN_ID="baseline_${NGPU}gpu_$(date +%Y%m%d_%H%M%S)"
        export NUM_LAYERS=9
        export MODEL_DIM=512
        export NUM_HEADS=8
        export NUM_KV_HEADS=4
        export MLP_MULT=2
        export USE_ATTNRES=0
        ;;
    attnres)
        export RUN_ID="attnres_fg_${NGPU}gpu_$(date +%Y%m%d_%H%M%S)"
        export NUM_LAYERS=9
        export MODEL_DIM=512
        export NUM_HEADS=8
        export NUM_KV_HEADS=4
        export MLP_MULT=2
        export USE_ATTNRES=1
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Options: baseline, attnres"
        exit 1
        ;;
esac

echo "RUN_ID: $RUN_ID"
echo "Starting training..."

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node=$NGPU train_gpt.py

echo "=========================================="
echo "RUN COMPLETE: $RUN_ID"
echo "=========================================="
ls -lh final_model.int8.ptz 2>/dev/null || echo "(no artifact)"
