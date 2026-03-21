#!/usr/bin/env bash
set -euo pipefail

# STINKY FROST V2 — Full SOTA Stack + Novel Techniques
#
# Matching proven SOTA stack from leaderboard commentary:
#   - 11 layers (consensus at top — NOT 10L, int5 penalty too high at 11L)
#   - Int6 quantization + zstd-22 compression (fits 11L in 16MB)
#   - FP16 embed + SmearGate + BigramHash(10240) + OrthoInit
#   - SWA every 50, start at 40% (consensus)
#   - MuonWD=0.04 (consensus — SWA makes it work)
#   - QAT at 25%
#   - 3% magnitude pruning
#
# Our novel additions (nobody has tried these):
#   - Seq ramp: 256 for first 25%, then 1024 (O(n²) → ~4x faster early steps)
#   - XSA last 3 layers (Exclusive Self Attention, arXiv:2603.09078)
#
# TTT is built into eval pipeline automatically.
#
# Target: 1.1313 (current pending SOTA) or better
# Reference: v1 = 1.1725, official SOTA = 1.1428, pending SOTA = 1.1313

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export ITERATIONS="${ITERATIONS:-20000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

LOGDIR="logs/stinky_frost_v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  STINKY FROST V2 — THE FULL SEND"
echo "  11L + Int6 + zstd + XSA + SeqRamp + SWA"
echo "  Target: < 1.1313 BPB"
echo "  Logs: $LOGDIR"
echo "============================================"

echo ""
echo "[1/1] Stinky Frost V2"

NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
TIE_EMBEDDINGS=1 \
QUANT_BITS=6 QAT_START_FRAC=0.25 EVAL_STRIDE=64 \
FP16_EMBED=1 SMEAR_GATE=1 BIGRAM_HASH=1 ORTHO_INIT=1 \
BIGRAM_BUCKETS=10240 \
MUON_WD=0.04 \
SWA_EVERY=50 SWA_START_FRAC=0.4 \
USE_ZSTD=1 ZSTD_LEVEL=22 PRUNE_PCT=0.03 \
SEQ_RAMP_START=256 SEQ_RAMP_FRAC=0.25 \
XSA_LAST_N=3 \
NCCL_IB_DISABLE=1 RUN_ID=stinky_frost_v2 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run1_v2.log"

echo ""
echo "============================================"
echo "  STINKY FROST V2 Complete."
echo "============================================"
echo "  Reference: v1 = 1.1725 BPB (15.58MB)"
echo "  Reference: pending SOTA = 1.1313 BPB"
echo ""

f="$LOGDIR/run1_v2.log"
# Try multiple label patterns
for label in int6_ttt_lora int5int6_ttt_lora; do
    bpb=$(grep -oP "final_${label} val_loss:\S+ val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    [ -n "$bpb" ] && break
done
for label in int6_zstd22 int6_zlib int5int6_zstd22; do
    quant_bpb=$(grep -oP "final_${label}\S* val_loss:\S+ val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    [ -n "$quant_bpb" ] && break
done
size=$(grep -oP 'Total submission size \S+: \K\d+' "$f" 2>/dev/null | tail -1)
steps=$(grep -oP 'stopping_early.*step:\K\d+' "$f" 2>/dev/null | tail -1)
echo "stinky_frost_v2: steps=${steps:-N/A} ttt_bpb=${bpb:-N/A} quant_bpb=${quant_bpb:-N/A} bytes=${size:-N/A}"
