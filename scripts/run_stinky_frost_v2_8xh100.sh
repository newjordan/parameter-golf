#!/usr/bin/env bash
set -euo pipefail

# STINKY FROST V2 — 8xH100 PRODUCTION RUN
#
# Proven SOTA stack (no experimental features):
#   - 11 layers (consensus at top)
#   - Int6 + zstd-22 compression
#   - FP16 embed + SmearGate + BigramHash(10240) + OrthoInit
#   - SWA every 50, start at 40%
#   - MuonWD=0.04 (consensus)
#   - QAT at 25%
#   - XSA last 3 layers (arXiv:2603.09078, +0.002 BPB)
#   - 3% magnitude pruning
#   - TTT SGD 3 epochs (matching #254: lr=0.002, momentum=0.9, freeze first 2 blocks)
#   - NO seq ramp (unproven, causes DDP issues)
#
# Target: 1.13-1.14 BPB
# Reference: v1 = 1.1725, pending SOTA = 1.1313

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export ITERATIONS="${ITERATIONS:-20000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

LOGDIR="logs/stinky_frost_v2_8xh100_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  STINKY FROST V2 — 8xH100 PRODUCTION"
echo "  11L + Int6 + zstd + XSA + SWA + WD0.04"
echo "  Target: < 1.1313 BPB"
echo "  Logs: $LOGDIR"
echo "============================================"

echo ""
echo "[1/1] Stinky Frost V2 — Production"

NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
TIE_EMBEDDINGS=1 \
QUANT_BITS=6 QAT_START_FRAC=0.25 EVAL_STRIDE=64 \
FP16_EMBED=1 SMEAR_GATE=1 BIGRAM_HASH=1 ORTHO_INIT=1 \
BIGRAM_BUCKETS=10240 \
MUON_WD=0.04 \
SWA_EVERY=50 SWA_START_FRAC=0.4 \
USE_ZSTD=1 ZSTD_LEVEL=22 PRUNE_PCT=0.03 \
XSA_LAST_N=3 \
TTT_OPTIMIZER=sgd TTT_LORA_LR=0.002 TTT_EPOCHS=3 TTT_MOMENTUM=0.9 TTT_FREEZE_FIRST_N=2 \
NCCL_IB_DISABLE=1 RUN_ID=stinky_frost_v2_prod \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run1_v2_prod.log"

echo ""
echo "============================================"
echo "  STINKY FROST V2 Complete."
echo "============================================"
echo "  Reference: v1 = 1.1725 BPB (15.58MB)"
echo "  Reference: pending SOTA = 1.1313 BPB"
echo ""

f="$LOGDIR/run1_v2_prod.log"
for label in int6_ttt_lora int5int6_ttt_lora; do
    bpb=$(grep -oP "final_${label} val_loss:\S+ val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    [ -n "$bpb" ] && break
done
for label in int6_zstd22 int6_zlib; do
    quant_bpb=$(grep -oP "final_${label}\S* val_loss:\S+ val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    [ -n "$quant_bpb" ] && break
done
size=$(grep -oP 'Total submission size \S+: \K\d+' "$f" 2>/dev/null | tail -1)
steps=$(grep -oP 'stopping_early.*step:\K\d+' "$f" 2>/dev/null | tail -1)
echo "stinky_frost_v2: steps=${steps:-N/A} ttt_bpb=${bpb:-N/A} quant_bpb=${quant_bpb:-N/A} bytes=${size:-N/A}"
