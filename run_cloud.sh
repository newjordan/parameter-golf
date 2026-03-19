#!/bin/bash
# Parameter Golf — Cloud H100 Launch Script
# Run this ON the RunPod pod after SSH-ing in.
#
# Usage (1×H100 for testing):
#   bash run_cloud.sh 1 baseline
#   bash run_cloud.sh 1 fractal
#
# Usage (8×H100 for official):
#   bash run_cloud.sh 8 baseline
#   bash run_cloud.sh 8 fractal
#   bash run_cloud.sh 8 fractal_no_gravity

set -e

NGPU=${1:-1}
MODE=${2:-baseline}

echo "=========================================="
echo "PARAMETER GOLF — CLOUD RUN"
echo "GPUs: ${NGPU} | Mode: ${MODE}"
echo "=========================================="

# ── SETUP (first run only) ──
if [ ! -d "/workspace/parameter-golf" ]; then
    echo "Cloning repo..."
    cd /workspace
    git clone https://github.com/newjordan/parameter-golf.git
    cd parameter-golf
    git checkout cloud/fractal-h100-test
else
    cd /workspace/parameter-golf
    git pull origin cloud/fractal-h100-test 2>/dev/null || true
fi

# ── DATA (first run only) ──
if [ ! -f "./data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin" ]; then
    echo "Downloading data (full 80 shards)..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
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

# ── MODE-SPECIFIC CONFIG ──
case $MODE in
    baseline)
        export RUN_ID="baseline_${NGPU}gpu_$(date +%Y%m%d_%H%M%S)"
        export FRACTAL=0
        export NUM_LAYERS=9
        export MODEL_DIM=512
        export NUM_HEADS=8
        export NUM_KV_HEADS=4
        export MLP_MULT=2
        ;;
    fractal)
        export RUN_ID="fractal_gravity_attnres_${NGPU}gpu_$(date +%Y%m%d_%H%M%S)"
        export FRACTAL=1
        export NUM_UNIQUE_LAYERS=3
        export NUM_LOOPS=3
        export USE_GRAVITY=1
        export USE_ATTNRES=1
        # Wider dim to use the freed param budget
        export MODEL_DIM=864
        export NUM_HEADS=8
        export NUM_KV_HEADS=4
        export MLP_MULT=2
        export NUM_LAYERS=9  # ignored in fractal mode but kept for compat
        ;;
    fractal_only)
        export RUN_ID="fractal_only_${NGPU}gpu_$(date +%Y%m%d_%H%M%S)"
        export FRACTAL=1
        export NUM_UNIQUE_LAYERS=3
        export NUM_LOOPS=3
        export USE_GRAVITY=0
        export USE_ATTNRES=0
        export MODEL_DIM=864
        export NUM_HEADS=8
        export NUM_KV_HEADS=4
        export MLP_MULT=2
        export NUM_LAYERS=9
        ;;
    fractal_no_gravity)
        export RUN_ID="fractal_attnres_${NGPU}gpu_$(date +%Y%m%d_%H%M%S)"
        export FRACTAL=1
        export NUM_UNIQUE_LAYERS=3
        export NUM_LOOPS=3
        export USE_GRAVITY=0
        export USE_ATTNRES=1
        export MODEL_DIM=864
        export NUM_HEADS=8
        export NUM_KV_HEADS=4
        export MLP_MULT=2
        export NUM_LAYERS=9
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Options: baseline, fractal, fractal_only, fractal_no_gravity"
        exit 1
        ;;
esac

echo "RUN_ID: $RUN_ID"
echo "Starting training..."

# ── LAUNCH ──
NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node=$NGPU train_gpt.py

echo "=========================================="
echo "RUN COMPLETE: $RUN_ID"
echo "=========================================="
echo "Check logs/  for full output"
echo "Check final_model.int8.ptz for artifact"
ls -lh final_model.int8.ptz 2>/dev/null || echo "(no artifact yet)"
