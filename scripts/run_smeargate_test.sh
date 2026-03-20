#!/usr/bin/env bash
set -euo pipefail

# SmearGate Isolated Test — on the deep 11L config to use 16MB budget
#
# Two runs:
#   1) 11L/512d baseline — no SmearGate
#   2) 11L/512d + SmearGate — bigram blending
#
# Both use int6 + QAT + sliding window.

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# Deep config: 11L to fill 16MB budget
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=2
export TIE_EMBEDDINGS=1

# Training
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export ITERATIONS="${ITERATIONS:-20000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

# Shared: int6 + QAT + sliding window
export QUANT_BITS=6
export EVAL_STRIDE=512
export QAT_START_FRAC=0.5

LOGDIR="logs/smeargate_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  SmearGate A/B Test (11L/512d)"
echo "  Logs: $LOGDIR"
echo "============================================"

# --- Run 1: 11L baseline (no SmearGate) ---
echo ""
echo "[1/2] 11L/512d baseline — no SmearGate"
export RUN_ID="11L_no_smear"
export SMEAR_GATE=0

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run1_11L_baseline.log"

# --- Run 2: 11L + SmearGate ---
echo ""
echo "[2/2] 11L/512d + SmearGate"
export RUN_ID="11L_smeargate"
export SMEAR_GATE=1

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run2_11L_smeargate.log"

# --- Summary ---
echo ""
echo "============================================"
echo "  SmearGate Test Complete. Results:"
echo "============================================"

for f in "$LOGDIR"/run*.log; do
    name=$(basename "$f" .log)
    bpb=$(grep -oP 'final_int6_ttt_lora val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    quant_bpb=$(grep -oP 'final_int6_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    size=$(grep -oP 'Total submission size int6\+zlib: \K\d+' "$f" | tail -1)
    echo "$name: ttt_bpb=${bpb:-N/A}  quant_bpb=${quant_bpb:-N/A}  artifact_bytes=${size:-N/A}"
done | tee "$LOGDIR/summary.txt"
