#!/usr/bin/env bash
set -euo pipefail

# A/B TEST: Uniform vs Progressive MLP (1xGPU)
#
# Both runs use identical #180 config except MLP distribution.
# Comparing the BPB delta, not absolute scores (both will be undertrained).

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"

LOGDIR="logs/ab_progressive_mlp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

COMMON="NUM_LAYERS=10 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3.0 \
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
SEED=42 \
NCCL_IB_DISABLE=1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  A/B Test: Uniform vs Progressive MLP"
echo "  1xGPU — comparing delta only"
echo "  Logs: $LOGDIR"
echo "============================================"

# --- RUN A: Uniform 3.0× (control) ---
echo ""
echo "=== [A] UNIFORM MLP 3.0x ==="
env $COMMON MLP_SCHEDULE="" RUN_ID="ab_uniform" \
    python "$SCRIPT_DIR/train_gpt.py" \
    2>&1 | tee "$LOGDIR/A_uniform.log"

# --- RUN B: Progressive 1.5→4.5× (bold ramp) ---
echo ""
echo "=== [B] PROGRESSIVE MLP 1.5x→4.5x ==="
env $COMMON MLP_SCHEDULE="1.5,1.83,2.17,2.5,2.83,3.17,3.5,3.83,4.17,4.5" RUN_ID="ab_progressive" \
    python "$SCRIPT_DIR/train_gpt.py" \
    2>&1 | tee "$LOGDIR/B_progressive.log"

# --- RESULTS ---
echo ""
echo "============================================"
echo "  A/B RESULTS"
echo "============================================"
for label in A_uniform B_progressive; do
    f="$LOGDIR/${label}.log"
    bpb=$(grep -oP "final_int8_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    raw_bpb=$(grep -oP "^step:\d+/\d+ val_loss:\S+ val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    steps=$(grep -oP 'stopping_early.*step:\K\d+' "$f" 2>/dev/null | tail -1)
    size=$(grep -oP 'Total submission size int8\+zlib: \K\d+' "$f" 2>/dev/null | tail -1)
    echo "  ${label}: steps=${steps:-N/A} quant_bpb=${bpb:-N/A} raw_bpb=${raw_bpb:-N/A} bytes=${size:-N/A}"
done
echo ""
echo "  If B < A → progressive MLP helps"
echo "  If B > A → progressive MLP hurts"
echo "============================================"
