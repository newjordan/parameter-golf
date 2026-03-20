#!/usr/bin/env bash
set -euo pipefail

# FRACTAL KILLER — our novel architecture + full leaderboard stack
#
# Fractal: 3 unique layers × 3 loops = 9 effective layers
# Weight sharing means only 3 layers serialize → wider model fits in 16MB
# Night one: fractal-only at 864d = -7.1% BPB vs baseline
#
# Now with full stack: FP16 embed + SmearGate + BigramHash + OrthoInit
# + SWA + MuonWD + earlyQAT + stride64
#
# Single run — fractal only (no gravity/attnres, night one showed they hurt)

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# Fractal: wider dim since only 3 unique layers serialize
export NUM_LAYERS=9
export MODEL_DIM=960
export NUM_HEADS=12
export NUM_KV_HEADS=4
export MLP_MULT=2
export TIE_EMBEDDINGS=1

export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export ITERATIONS="${ITERATIONS:-20000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

LOGDIR="logs/fractal_killer_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  FRACTAL KILLER"
echo "  3×3 loops, 960d, full stack"
echo "  Logs: $LOGDIR"
echo "============================================"

echo ""
echo "[1/1] Fractal + full stack"

FRACTAL=1 \
NUM_UNIQUE_LAYERS=3 \
NUM_LOOPS=3 \
USE_GRAVITY=0 \
USE_ATTNRES=0 \
BREATH_PATTERN="full,cheap,cheap" \
QUANT_BITS=6 \
QAT_START_FRAC=0.25 \
EVAL_STRIDE=64 \
MUON_WD=0.04 \
FP16_EMBED=1 \
SMEAR_GATE=1 \
BIGRAM_HASH=1 \
ORTHO_INIT=1 \
SWA_EVERY=50 \
SWA_START_FRAC=0.5 \
RUN_ID=fractal_killer \
NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run1_fractal.log"

# --- Summary ---
echo ""
echo "============================================"
echo "  FRACTAL KILLER Complete. Results:"
echo "============================================"
echo "  Reference: killer v1 (non-fractal) = quant_bpb:1.1725 (15.58MB)"
echo ""

for f in "$LOGDIR"/run*.log; do
    name=$(basename "$f" .log)
    bpb=$(grep -oP 'final_int6_ttt_lora val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    quant_bpb=$(grep -oP 'final_int6_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    size=$(grep -oP 'Total submission size int6\+zlib: \K\d+' "$f" | tail -1)
    echo "$name: ttt_bpb=${bpb:-N/A}  quant_bpb=${quant_bpb:-N/A}  artifact_bytes=${size:-N/A}"
done | tee "$LOGDIR/summary.txt"
