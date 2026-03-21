#!/usr/bin/env bash
set -euo pipefail

# PR#180 CLONE + PROGRESSIVE MLP
#
# Exact #180 recipe (10L, int5 MLP, BigramHash 10240, SWA, WD 0.04)
# with one change: MLP width ramps from 2.0× to 4.0× across layers.
#
# Parameter budget is identical to uniform 3.0×:
#   Uniform:     10 × 1536 = 15,360 hidden units total
#   Progressive: 1024+1138+1251+1365+1479+1593+1706+1820+1934+2048 = 15,358
#
# Hypothesis: early layers need less MLP capacity (coarse patterns),
# late layers need more (fine-grained prediction). Same params, better allocation.

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"

LOGDIR="logs/progressive_mlp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  PR#180 + Progressive MLP"
echo "  10L, skinny-early fat-late"
echo "  Logs: $LOGDIR"
echo "============================================"

# Linear ramp from 2.0× to 4.0× across 10 layers
# Layer:  0     1     2     3     4     5     6     7     8     9
# Mult:  2.00  2.22  2.44  2.67  2.89  3.11  3.33  3.56  3.78  4.00
# Hidden: 1024  1137  1249  1367  1479  1593  1705  1822  1935  2048
MLP_SCHEDULE="2.0,2.22,2.44,2.67,2.89,3.11,3.33,3.56,3.78,4.0"

NUM_LAYERS=10 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3.0 \
MLP_SCHEDULE="$MLP_SCHEDULE" \
TIE_EMBEDDINGS=1 \
VOCAB_SIZE=1024 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
ITERATIONS=20000 \
WARMDOWN_ITERS=3000 \
WARMUP_STEPS=20 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=100 \
VAL_LOSS_EVERY=500 \
WEIGHT_DECAY=0.04 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
GRAD_CLIP_NORM=0.3 \
EVAL_STRIDE=64 \
BIGRAM_VOCAB_SIZE=10240 \
BIGRAM_DIM=128 \
SWA_ENABLED=1 \
SWA_START_FRAC=0.4 \
SWA_EVERY=50 \
SEED="${SEED:-42}" \
RUN_ID="progressive_mlp_s${SEED:-42}" \
NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" \
    experiments/pr180_progressive_mlp/train_gpt.py \
    2>&1 | tee "$LOGDIR/run.log"

echo ""
echo "============================================"
echo "  Progressive MLP Complete."
echo "============================================"
f="$LOGDIR/run.log"
bpb=$(grep -oP "final_int8_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
size=$(grep -oP 'Total submission size int8\+zlib: \K\d+' "$f" 2>/dev/null | tail -1)
steps=$(grep -oP 'stopping_early.*step:\K\d+' "$f" 2>/dev/null | tail -1)
echo "result: steps=${steps:-N/A} bpb=${bpb:-N/A} bytes=${size:-N/A}"
echo "reference: PR#180 uniform = 1.1428 BPB (25.5M params, 6709 steps)"
