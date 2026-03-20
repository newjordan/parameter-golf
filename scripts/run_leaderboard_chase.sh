#!/usr/bin/env bash
set -euo pipefail

# Leaderboard Chase — stack techniques from top entries
#
# Base: MLP 3× (our best at 1.2607 quant BPB)
# Techniques from leaderboard:
#   1) Stride=64 sliding window (from #2 @mattqlf, stride=64)
#   2) FP16 embeddings (from #1 @notapplica, preserve embed quality)
#   3) Muon weight decay (from #1 @notapplica)
#   4) All combined + early QAT
#
# Target: close gap to 1.1574 (#1 pending)

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# MLP 3× config (our best)
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

LOGDIR="logs/leaderboard_chase_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  Leaderboard Chase Suite"
echo "  Base: MLP 3× (1.2607 quant BPB)"
echo "  Target: <1.16"
echo "  Logs: $LOGDIR"
echo "============================================"

# --- Run 1: Stride 64 (from leaderboard #2) ---
echo ""
echo "[1/4] Stride 64 sliding window eval"
export RUN_ID="mlp3x_stride64"
export QAT_START_FRAC=0.25
export EVAL_STRIDE=64
export MUON_WD=0.0
export FP16_EMBED=0

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run1_stride64.log"

# --- Run 2: FP16 embeddings ---
echo ""
echo "[2/4] FP16 embeddings (skip int6 on tok_emb)"
export RUN_ID="mlp3x_fp16embed"
export QAT_START_FRAC=0.25
export EVAL_STRIDE=64
export MUON_WD=0.0
export FP16_EMBED=1

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run2_fp16embed.log"

# --- Run 3: Muon weight decay ---
echo ""
echo "[3/4] Muon weight decay 0.01"
export RUN_ID="mlp3x_muonwd"
export QAT_START_FRAC=0.25
export EVAL_STRIDE=64
export MUON_WD=0.01
export FP16_EMBED=0

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run3_muonwd.log"

# --- Run 4: Everything stacked ---
echo ""
echo "[4/4] FULL STACK: stride64 + fp16embed + muonWD + earlyQAT"
export RUN_ID="mlp3x_fullstack"
export QAT_START_FRAC=0.25
export EVAL_STRIDE=64
export MUON_WD=0.01
export FP16_EMBED=1

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run4_fullstack.log"

# --- Summary ---
echo ""
echo "============================================"
echo "  Leaderboard Chase Complete. Results:"
echo "============================================"
echo "  Reference: MLP3x earlyQAT slide512 = quant_bpb:1.2607"
echo ""

for f in "$LOGDIR"/run*.log; do
    name=$(basename "$f" .log)
    bpb=$(grep -oP 'final_int6_ttt_lora val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    quant_bpb=$(grep -oP 'final_int6_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    size=$(grep -oP 'Total submission size int6\+zlib: \K\d+' "$f" | tail -1)
    echo "$name: ttt_bpb=${bpb:-N/A}  quant_bpb=${quant_bpb:-N/A}  artifact_bytes=${size:-N/A}"
done | tee "$LOGDIR/summary.txt"
