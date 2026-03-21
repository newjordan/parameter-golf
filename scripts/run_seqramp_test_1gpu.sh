#!/usr/bin/env bash
set -euo pipefail

# SEQ RAMP SINGLE TEST — 1 GPU
#
# Novel technique: train at seq256 for first 25% of steps (O(n²) → ~4x faster),
# then ramp to full 1024. More gradient updates in early training.
#
# Uses Stinky Frost v1 base config for clean comparison.
# Reference: v1 at full seq = 1.1725 BPB on 8xH100

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export ITERATIONS="${ITERATIONS:-20000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-6000}"

LOGDIR="logs/seqramp_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  SEQ RAMP TEST — NOVEL TECHNIQUE"
echo "  seq256 → seq1024 at 25%"
echo "  1 GPU, extended wallclock"
echo "  Logs: $LOGDIR"
echo "============================================"

NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
MLP_HIDDEN=1344 TIE_EMBEDDINGS=1 \
QUANT_BITS=6 QAT_START_FRAC=0.25 EVAL_STRIDE=64 \
FP16_EMBED=1 SMEAR_GATE=1 BIGRAM_HASH=1 ORTHO_INIT=1 \
MUON_WD=0.01 \
SEQ_RAMP_START=256 SEQ_RAMP_FRAC=0.25 \
NCCL_IB_DISABLE=1 RUN_ID=seqramp_test \
torchrun --standalone --nproc_per_node="${NPROC:-1}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run1_seqramp.log"

echo ""
echo "============================================"
echo "  SEQ RAMP TEST Complete."
echo "============================================"
echo "  Reference: Stinky Frost v1 (no ramp) = 1.1725 BPB"
echo ""

f="$LOGDIR/run1_seqramp.log"
bpb=$(grep -oP 'final_int6_ttt_lora val_loss:\S+ val_bpb:\K\S+' "$f" 2>/dev/null | tail -1)
quant_bpb=$(grep -oP 'final_int6_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+' "$f" 2>/dev/null | tail -1)
size=$(grep -oP 'Total submission size \S+: \K\d+' "$f" 2>/dev/null | tail -1)
steps=$(grep -oP 'stopping_early.*step:\K\d+' "$f" 2>/dev/null | tail -1)
echo "seqramp: steps=${steps:-N/A}  ttt_bpb=${bpb:-N/A}  quant_bpb=${quant_bpb:-N/A}  bytes=${size:-N/A}"
