#!/usr/bin/env bash
set -euo pipefail

# STINKY FROST V2 — Adapted from PR #180 SOTA techniques
#
# Key upgrades from v1:
#   - INT5 for MLP weights + INT6 for attention (saves ~1.86MB → more params)
#   - zstd-22 compression instead of zlib (better compression ratio)
#   - 3% magnitude pruning (zeros compress well with zstd)
#   - 10 layers (freed space from int5+zstd)
#   - SWA every 50 steps, start at 40%
#   - MuonWD=0.02 (sweet spot between 0.01 and 0.04)
#   - Seq ramp: start at 256 for first 25%, then full 1024 (novel!)
#   - BiggramHash 10240 buckets (from PR #180)
#
# Reference: v1 = 1.1725 BPB (15.58MB), PR #180 SOTA = 1.1428 BPB (15.52MB)

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
echo "  STINKY FROST V2"
echo "  Int5-MLP + zstd + 10L + SWA + SeqRamp"
echo "  Logs: $LOGDIR"
echo "============================================"

echo ""
echo "[1/1] Stinky Frost V2 — Full Send"

NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
MLP_HIDDEN=1344 TIE_EMBEDDINGS=1 \
QUANT_BITS=6 QAT_START_FRAC=0.25 EVAL_STRIDE=64 \
FP16_EMBED=1 SMEAR_GATE=1 BIGRAM_HASH=1 ORTHO_INIT=1 \
BIGRAM_BUCKETS=10240 \
MUON_WD=0.02 \
SWA_EVERY=50 SWA_START_FRAC=0.4 \
INT5_MLP=1 USE_ZSTD=1 ZSTD_LEVEL=22 PRUNE_PCT=0.03 \
SEQ_RAMP_START=256 SEQ_RAMP_FRAC=0.25 \
NCCL_IB_DISABLE=1 RUN_ID=stinky_frost_v2 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run1_v2.log"

echo ""
echo "============================================"
echo "  STINKY FROST V2 Complete."
echo "============================================"
echo "  Reference: v1 = 1.1725 BPB (15.58MB)"
echo "  Reference: PR #180 SOTA = 1.1428 BPB (15.52MB)"
echo ""

f="$LOGDIR/run1_v2.log"
bpb=$(grep -oP 'final_int5int6_ttt_lora val_loss:\S+ val_bpb:\K\S+' "$f" 2>/dev/null | tail -1)
quant_bpb=$(grep -oP 'final_int5int6_zst\S+ val_loss:\S+ val_bpb:\K\S+' "$f" 2>/dev/null | tail -1)
# Fallback to int6 labels if mixed label not found
bpb=${bpb:-$(grep -oP 'final_int6_ttt_lora val_loss:\S+ val_bpb:\K\S+' "$f" 2>/dev/null | tail -1)}
quant_bpb=${quant_bpb:-$(grep -oP 'final_int6_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+' "$f" 2>/dev/null | tail -1)}
size=$(grep -oP 'Total submission size \S+: \K\d+' "$f" 2>/dev/null | tail -1)
echo "stinky_frost_v2: ttt_bpb=${bpb:-N/A}  quant_bpb=${quant_bpb:-N/A}  artifact_bytes=${size:-N/A}"
