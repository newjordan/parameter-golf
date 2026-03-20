#!/usr/bin/env bash
set -euo pipefail

# BEST SHOT — stack every proven technique
#
# Architecture: 9L/512d MLP 3× (our best from suite 1)
# Training:     Early QAT 25% + Muon weight decay 0.01
# Eval:         Stride=64 sliding window (from leaderboard #2)
# Compression:  FP16 embeddings (from leaderboard #1, preserves embed quality)
#
# Two runs:
#   1) Full stack — everything combined
#   2) Stride=64 only — isolate eval improvement from training changes

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

export NUM_LAYERS=9
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3
export TIE_EMBEDDINGS=1

export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export ITERATIONS="${ITERATIONS:-20000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

export QUANT_BITS=6
export EVAL_STRIDE=64

LOGDIR="logs/best_shot_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  BEST SHOT — Full Stack"
echo "  MLP3x + earlyQAT + stride64 + fp16embed + muonWD"
echo "  Logs: $LOGDIR"
echo "============================================"

# --- Run 1: Full stack ---
echo ""
echo "[1/2] FULL STACK: stride64 + earlyQAT25 + fp16embed + muonWD"
echo "DEBUG: QAT_START_FRAC=0.25 EVAL_STRIDE=64 MUON_WD=0.01 FP16_EMBED=1"

QAT_START_FRAC=0.25 \
EVAL_STRIDE=64 \
MUON_WD=0.01 \
FP16_EMBED=1 \
RUN_ID=bestshot_fullstack \
NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run1_fullstack.log"

# --- Run 2: Stride 64 only (no fp16embed, no muonWD, standard QAT) ---
echo ""
echo "[2/2] STRIDE64 ONLY: earlyQAT25 + stride64, no other changes"
echo "DEBUG: QAT_START_FRAC=0.25 EVAL_STRIDE=64 MUON_WD=0.0 FP16_EMBED=0"

QAT_START_FRAC=0.25 \
EVAL_STRIDE=64 \
MUON_WD=0.0 \
FP16_EMBED=0 \
RUN_ID=bestshot_stride64only \
NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run2_stride64only.log"

# --- Summary ---
echo ""
echo "============================================"
echo "  BEST SHOT Complete. Results:"
echo "============================================"
echo "  Reference: MLP3x QAT50 slide512 = quant_bpb:1.2681"
echo "  Reference: MLP3x QAT25 slide512 = quant_bpb:1.2607"
echo ""

for f in "$LOGDIR"/run*.log; do
    name=$(basename "$f" .log)
    bpb=$(grep -oP 'final_int6_ttt_lora val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    quant_bpb=$(grep -oP 'final_int6_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    size=$(grep -oP 'Total submission size int6\+zlib: \K\d+' "$f" | tail -1)
    echo "$name: ttt_bpb=${bpb:-N/A}  quant_bpb=${quant_bpb:-N/A}  artifact_bytes=${size:-N/A}"
done | tee "$LOGDIR/summary.txt"
