#!/usr/bin/env bash
set -euo pipefail

# STINKY FROST V3 — V2 + seq ramp + batch ramp + less frequent val
#
# V2 base (11L + Int6 + zstd + XSA + SWA + WD0.04 + TTT SGD 3ep)
# + Seq ramp: 256→1024 at 20% (O(n²) savings, ~4x faster attention early)
#   torch.compile disabled for stability with dynamic shapes
# + Batch size warmup: 262144→524288 over first 25%
#   More gradient updates when gradients are most informative
# + VAL_LOSS_EVERY=1000 (save ~10s eval time)
#
# Two novel untried techniques. Target: more gradient updates = better BPB

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export ITERATIONS="${ITERATIONS:-20000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

LOGDIR="logs/stinky_frost_v3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  STINKY FROST V3 — SEQ RAMP + BATCH RAMP"
echo "  V2 + seq256→1024 + batch 262K→524K"
echo "  Logs: $LOGDIR"
echo "============================================"

echo ""
echo "[1/1] Stinky Frost V3"

NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 \
MLP_HIDDEN=1024 TIE_EMBEDDINGS=1 \
QUANT_BITS=6 QAT_START_FRAC=0.25 EVAL_STRIDE=64 \
FP16_EMBED=1 SMEAR_GATE=1 BIGRAM_HASH=1 ORTHO_INIT=1 \
BIGRAM_BUCKETS=4096 \
MUON_WD=0.04 \
SWA_EVERY=50 SWA_START_FRAC=0.4 \
USE_ZSTD=1 ZSTD_LEVEL=22 PRUNE_PCT=0.03 \
XSA_LAST_N=3 \
SEQ_RAMP_START=256 SEQ_RAMP_FRAC=0.2 \
BATCH_RAMP_START=262144 BATCH_RAMP_FRAC=0.25 \
TTT_OPTIMIZER=sgd TTT_LORA_LR=0.002 TTT_EPOCHS=3 TTT_MOMENTUM=0.9 TTT_FREEZE_FIRST_N=2 \
NCCL_IB_DISABLE=1 RUN_ID=stinky_frost_v3 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run1_v3.log"

echo ""
echo "============================================"
echo "  STINKY FROST V3 Complete."
echo "============================================"
echo "  Reference: v2 result = ? BPB"
echo "  Reference: pending SOTA = 1.1313 BPB"
echo ""

f="$LOGDIR/run1_v3.log"
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
echo "stinky_frost_v3: steps=${steps:-N/A} ttt_bpb=${bpb:-N/A} quant_bpb=${quant_bpb:-N/A} bytes=${size:-N/A}"
