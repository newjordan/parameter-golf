#!/usr/bin/env bash
set -euo pipefail

# FRACTAL PROGRESSIVE — speed through early training, then go fractal
#
# Progressive loop unrolling:
#   Step 0-2999:    1 loop  (fast, ~50ms/step, learn language basics)
#   Step 3000-4999: 2 loops (add first refinement pass)
#   Step 5000+:     3 loops (full fractal + QAT kicks in)
#
# Breathing: full,cheap,cheap — only loop 1 runs MLP
# At eval time: always runs all 3 loops (full depth)

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

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

LOGDIR="logs/fractal_progressive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  FRACTAL PROGRESSIVE"
echo "  1 loop → 2 loops → 3 loops"
echo "  960d, full/cheap/cheap"
echo "  Logs: $LOGDIR"
echo "============================================"

echo ""
echo "[1/1] Progressive fractal + full stack"

FRACTAL=1 \
NUM_UNIQUE_LAYERS=3 \
NUM_LOOPS=3 \
USE_GRAVITY=0 \
USE_ATTNRES=0 \
BREATH_PATTERN="full,cheap,cheap" \
LOOP_SCHEDULE="0,3000,5000" \
QUANT_BITS=6 \
QAT_START_FRAC=0.25 \
EVAL_STRIDE=64 \
MUON_WD=0.01 \
FP16_EMBED=1 \
SMEAR_GATE=1 \
BIGRAM_HASH=1 \
ORTHO_INIT=1 \
SWA_EVERY=50 \
SWA_START_FRAC=0.5 \
RUN_ID=fractal_progressive \
NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run1_progressive.log"

echo ""
echo "============================================"
echo "  FRACTAL PROGRESSIVE Complete."
echo "============================================"
echo "  Reference: Stinky Frost (non-fractal) = 1.1725 BPB"
echo "  Reference: Fractal breathing (no progressive) = 1.2373 BPB"
echo ""

f="$LOGDIR/run1_progressive.log"
bpb=$(grep -oP 'final_int6_ttt_lora val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
quant_bpb=$(grep -oP 'final_int6_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
size=$(grep -oP 'Total submission size int6\+zlib: \K\d+' "$f" | tail -1)
echo "fractal_progressive: ttt_bpb=${bpb:-N/A}  quant_bpb=${quant_bpb:-N/A}  artifact_bytes=${size:-N/A}"
