#!/usr/bin/env bash
set -euo pipefail

# A/B/C TEST: Uniform vs Progressive MLP vs Progressive MLP + Fat BigramHash
#
# All runs use identical #180 config except:
#   A: Uniform MLP 3.0× (control — exact #180)
#   B: Progressive MLP 1.5×→4.5× (same total params)
#   C: Progressive MLP 1.5×→4.5× + BigramHash 16384 buckets, dim 192
#      (compensate thin early layers with richer bigram input)
#
# 1xGPU, comparing BPB delta only. ~30 min total.

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"

LOGDIR="logs/abc_progressive_mlp_$(date +%Y%m%d_%H%M%S)"
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
SWA_ENABLED=1 \
SWA_START_FRAC=0.4 \
SWA_EVERY=50 \
SEED=42 \
NCCL_IB_DISABLE=1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROGRESSIVE="1.5,1.83,2.17,2.5,2.83,3.17,3.5,3.83,4.17,4.5"

echo "============================================"
echo "  A/B/C Test: MLP Distribution + BigramHash"
echo "  1xGPU — comparing delta only"
echo "  Logs: $LOGDIR"
echo "============================================"

# --- RUN A: Uniform 3.0× (control) ---
echo ""
echo "=== [A] UNIFORM MLP 3.0x + BigramHash 10240/128 ==="
env $COMMON BIGRAM_VOCAB_SIZE=10240 BIGRAM_DIM=128 MLP_SCHEDULE="" RUN_ID="abc_A_uniform" \
    python "$SCRIPT_DIR/train_gpt.py" \
    2>&1 | tee "$LOGDIR/A_uniform.log"

# --- RUN B: Progressive 1.5→4.5× ---
echo ""
echo "=== [B] PROGRESSIVE MLP 1.5x→4.5x + BigramHash 10240/128 ==="
env $COMMON BIGRAM_VOCAB_SIZE=10240 BIGRAM_DIM=128 MLP_SCHEDULE="$PROGRESSIVE" RUN_ID="abc_B_progressive" \
    python "$SCRIPT_DIR/train_gpt.py" \
    2>&1 | tee "$LOGDIR/B_progressive.log"

# --- RUN C: Progressive 1.5→4.5× + Fat BigramHash ---
echo ""
echo "=== [C] PROGRESSIVE MLP 1.5x→4.5x + BigramHash 16384/192 ==="
env $COMMON BIGRAM_VOCAB_SIZE=16384 BIGRAM_DIM=192 MLP_SCHEDULE="$PROGRESSIVE" RUN_ID="abc_C_prog_fatbigram" \
    python "$SCRIPT_DIR/train_gpt.py" \
    2>&1 | tee "$LOGDIR/C_prog_fatbigram.log"

# --- RESULTS ---
echo ""
echo "============================================"
echo "  A/B/C RESULTS"
echo "============================================"
for label in A_uniform B_progressive C_prog_fatbigram; do
    f="$LOGDIR/${label}.log"
    bpb=$(grep -oP "final_int8_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    raw_bpb=$(grep -oP "^step:\d+/\d+ val_loss:\S+ val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    steps=$(grep -oP 'stopping_early.*step:\K\d+' "$f" 2>/dev/null | tail -1)
    size=$(grep -oP 'Total submission size int8\+zlib: \K\d+' "$f" 2>/dev/null | tail -1)
    params=$(grep -oP 'model_params:\K\d+' "$f" 2>/dev/null | tail -1)
    echo "  ${label}: steps=${steps:-N/A} quant_bpb=${bpb:-N/A} raw_bpb=${raw_bpb:-N/A} params=${params:-N/A} bytes=${size:-N/A}"
done
echo ""
echo "  A = control (exact #180)"
echo "  B < A → progressive MLP helps"
echo "  C < B → fat BigramHash compensates thin early layers"
echo "  C < A but B > A → need both together"
echo "============================================"
