#!/usr/bin/env bash
set -euo pipefail

# Deep 11L Config — fill the 16MB budget
# Current 9L/512d = 12.5MB artifact. This uses the headroom.

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=2
export TIE_EMBEDDINGS=1

export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export ITERATIONS="${ITERATIONS:-20000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

export QUANT_BITS=6
export EVAL_STRIDE=512
export QAT_START_FRAC=0.5
export RUN_ID="int6_qat_slide512_11L"

LOGDIR="logs/deep_11L_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  Deep 11L/512d Config (~20.7M params)"
echo "  Int6 + QAT + Sliding Window"
echo "  Logs: $LOGDIR"
echo "============================================"

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run_11L.log"

echo ""
echo "============================================"
echo "  Done. Results:"
echo "============================================"

f="$LOGDIR/run_11L.log"
bpb=$(grep -oP 'final_int6_ttt_lora val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
quant_bpb=$(grep -oP 'final_int6_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
size=$(grep -oP 'Total submission size int6\+zlib: \K\d+' "$f" | tail -1)
echo "11L: ttt_bpb=${bpb:-N/A}  quant_bpb=${quant_bpb:-N/A}  artifact_bytes=${size:-N/A}"
