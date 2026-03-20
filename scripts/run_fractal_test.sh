#!/usr/bin/env bash
set -euo pipefail

# Fractal Transformer Isolated Test
#
# Four runs testing fractal concept from night one:
#   1) Fractal only (3×3, 864d) — weight sharing + wider layers
#   2) Fractal + Gravity — learned auxiliary losses at loop boundaries
#   3) Fractal + Gravity + AttnRes — attention over depth
#   4) Fractal + Gravity + AttnRes + Breathing (full,cheap,full)
#
# All runs use int6 + QAT + sliding window.
# First night result: fractal-only was -7.1% BPB vs baseline (strongest).

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# Fractal model: 3 unique layers × 3 loops = 9 effective layers
# Wider dim (864) since weight sharing compresses well under int6
export MODEL_DIM="${MODEL_DIM:-864}"
export NUM_HEADS="${NUM_HEADS:-12}"
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

# Shared: fractal + int6 + QAT + sliding window
export FRACTAL=1
export NUM_UNIQUE_LAYERS=3
export NUM_LOOPS=3
export QUANT_BITS=6
export EVAL_STRIDE=512
export QAT_START_FRAC=0.5

LOGDIR="logs/fractal_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  Fractal Transformer Test Suite"
echo "  Config: ${NUM_UNIQUE_LAYERS}×${NUM_LOOPS} loops, ${MODEL_DIM}d"
echo "  Logs: $LOGDIR"
echo "============================================"

# --- Run 1: Fractal only (no gravity, no attnres) ---
echo ""
echo "[1/4] Fractal only — 3×3, 864d, no gravity, no attnres"
export RUN_ID="fractal_only"
export USE_GRAVITY=0
export USE_ATTNRES=0
export BREATH_PATTERN="full,full,full"

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run1_fractal_only.log"

# --- Run 2: Fractal + Gravity ---
echo ""
echo "[2/4] Fractal + Gravity — learned auxiliary losses"
export RUN_ID="fractal_gravity"
export USE_GRAVITY=1
export USE_ATTNRES=0
export BREATH_PATTERN="full,full,full"

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run2_fractal_gravity.log"

# --- Run 3: Fractal + Gravity + AttnRes ---
echo ""
echo "[3/4] Fractal + Gravity + AttnRes — attention over depth"
export RUN_ID="fractal_gravity_attnres"
export USE_GRAVITY=1
export USE_ATTNRES=1
export BREATH_PATTERN="full,full,full"

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run3_fractal_gravity_attnres.log"

# --- Run 4: Fractal + Gravity + AttnRes + Breathing ---
echo ""
echo "[4/4] Fractal + Gravity + AttnRes + Breathing (full,cheap,full)"
export RUN_ID="fractal_full_breathing"
export USE_GRAVITY=1
export USE_ATTNRES=1
export BREATH_PATTERN="full,cheap,full"

NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run4_fractal_breathing.log"

# --- Summary ---
echo ""
echo "============================================"
echo "  Fractal Test Complete. Results:"
echo "============================================"

for f in "$LOGDIR"/run*.log; do
    name=$(basename "$f" .log)
    bpb=$(grep -oP 'final_int6_ttt_lora val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    quant_bpb=$(grep -oP 'final_int6_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    size=$(grep -oP 'Total submission size int6\+zlib: \K\d+' "$f" | tail -1)
    echo "$name: ttt_bpb=${bpb:-N/A}  quant_bpb=${quant_bpb:-N/A}  artifact_bytes=${size:-N/A}"
done | tee "$LOGDIR/summary.txt"
