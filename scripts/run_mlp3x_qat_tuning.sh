#!/usr/bin/env bash
set -euo pipefail

# MLP 3× QAT Tuning — close the quant BPB gap
#
# Baseline: MLP3x got 1.1989 pre-quant but 1.2681 post-quant (0.069 gap)
# Test earlier QAT start + tighter sliding window to shrink that gap.
#
# Three runs:
#   1) Early QAT (25%) + slide512 — more training under fake-quant
#   2) Standard QAT (50%) + slide256 — tighter sliding window
#   3) Early QAT (25%) + slide256 — both combined

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# MLP 3× config (from suite 1 winner)
export NUM_LAYERS=9
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3
export TIE_EMBEDDINGS=1

# Training
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export ITERATIONS="${ITERATIONS:-20000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

export QUANT_BITS=6

LOGDIR="logs/mlp3x_qat_tuning_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  MLP 3× QAT Tuning Suite"
echo "  Goal: shrink 0.069 quant BPB gap"
echo "  Logs: $LOGDIR"
echo "============================================"

# --- Run 1: Early QAT (25%) + slide512 ---
echo ""
echo "[1/3] Early QAT (25%) + slide512"
export RUN_ID="mlp3x_earlyqat_slide512"
export QAT_START_FRAC=0.25
export EVAL_STRIDE=512

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run1_earlyqat_slide512.log"

# --- Run 2: Standard QAT (50%) + slide256 ---
echo ""
echo "[2/3] Standard QAT (50%) + slide256"
export RUN_ID="mlp3x_qat50_slide256"
export QAT_START_FRAC=0.5
export EVAL_STRIDE=256

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run2_qat50_slide256.log"

# --- Run 3: Early QAT (25%) + slide256 ---
echo ""
echo "[3/3] Early QAT (25%) + slide256 — both combined"
export RUN_ID="mlp3x_earlyqat_slide256"
export QAT_START_FRAC=0.25
export EVAL_STRIDE=256

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run3_earlyqat_slide256.log"

# --- Summary ---
echo ""
echo "============================================"
echo "  MLP 3× QAT Tuning Complete. Results:"
echo "============================================"
echo "  Reference: MLP3x QAT50 slide512 = quant_bpb:1.2681 ttt_bpb:1.2409"
echo ""

for f in "$LOGDIR"/run*.log; do
    name=$(basename "$f" .log)
    bpb=$(grep -oP 'final_int6_ttt_lora val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    quant_bpb=$(grep -oP 'final_int6_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    size=$(grep -oP 'Total submission size int6\+zlib: \K\d+' "$f" | tail -1)
    echo "$name: ttt_bpb=${bpb:-N/A}  quant_bpb=${quant_bpb:-N/A}  artifact_bytes=${size:-N/A}"
done | tee "$LOGDIR/summary.txt"
