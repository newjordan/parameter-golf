#!/usr/bin/env bash
set -euo pipefail

# Sliding Window Eval + Int6 QAT — eval on 1xH100
#
# Three runs back-to-back:
#   1) Int6 + QAT only — measure quantization impact
#   2) Int6 + QAT + sliding window eval (stride=512) — full strategy
#   3) Int6 + QAT + sliding window + MLP 3× — wider FFN, test capacity vs size
#   4) Int6 + QAT + sliding window + 11L/512d — deep config, fill 16MB budget
#
# Each run uses 10-min wallclock cap.

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# Model shape (baseline defaults)
export NUM_LAYERS="${NUM_LAYERS:-9}"
export MODEL_DIM="${MODEL_DIM:-512}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export MLP_MULT="${MLP_MULT:-2}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"

# Training
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export ITERATIONS="${ITERATIONS:-20000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

LOGDIR="logs/sliding_int6_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  Sliding Window + Int6 Eval Suite"
echo "  Logs: $LOGDIR"
echo "============================================"

# --- Run 1: Int6 + QAT (no sliding window) ---
echo ""
echo "[1/4] Int6 + QAT, stride=1024 (no overlap)"
export RUN_ID="int6_qat"
export QUANT_BITS=6
export EVAL_STRIDE=0
export QAT_START_FRAC=0.5

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run1_int6_qat.log"

# --- Run 2: Int6 + QAT + Sliding Window (stride=512) ---
echo ""
echo "[2/4] Int6 + QAT + sliding window (stride=512)"
export RUN_ID="int6_qat_slide512"
export QUANT_BITS=6
export EVAL_STRIDE=512
export QAT_START_FRAC=0.5

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run2_int6_slide512.log"

# --- Run 3: Int6 + QAT + Sliding Window + MLP 3× ---
echo ""
echo "[3/4] Int6 + QAT + sliding window + MLP 3× (wider FFN)"
export RUN_ID="int6_qat_slide512_mlp3x"
export QUANT_BITS=6
export EVAL_STRIDE=512
export QAT_START_FRAC=0.5
export MLP_MULT=3

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run3_int6_slide512_mlp3x.log"

# Reset MLP_MULT for any future runs
export MLP_MULT=2

# --- Run 4: Int6 + QAT + Sliding Window + 11 Layers (deep config) ---
echo ""
echo "[4/4] Int6 + QAT + sliding window + 11L/512d (deep, ~20.7M params)"
export RUN_ID="int6_qat_slide512_11L"
export QUANT_BITS=6
export EVAL_STRIDE=512
export QAT_START_FRAC=0.5
export NUM_LAYERS=11
export MODEL_DIM=512
export MLP_MULT=2

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run4_int6_slide512_11L.log"

# Reset to defaults
export NUM_LAYERS=9

# --- Summary ---
echo ""
echo "============================================"
echo "  All runs complete. Extracting results..."
echo "============================================"

for f in "$LOGDIR"/run*.log; do
    name=$(basename "$f" .log)
    bpb=$(grep -oP 'final_int6_ttt_lora val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    quant_bpb=$(grep -oP 'final_int6_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    size=$(grep -oP 'Total submission size int6\+zlib: \K\d+' "$f" | tail -1)
    echo "$name: ttt_bpb=${bpb:-N/A}  quant_bpb=${quant_bpb:-N/A}  artifact_bytes=${size:-N/A}"
done | tee "$LOGDIR/summary.txt"
