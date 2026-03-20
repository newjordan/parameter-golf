#!/usr/bin/env bash
set -euo pipefail

# LEADERBOARD KILLER — optimized for 16MB with FP16 embed
#
# Our 1.1693 BPB was real but busted 16MB by 340KB.
# v1: MLP 1344 = 15.58MB, 1.1725 BPB (424KB headroom)
# v2: MLP 1472 — use the headroom, close gap to 1.1693
# Then stack SmearGate + BigramHash + OrthoInit on top.
#
# Single run — our best shot at the leaderboard.

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

LOGDIR="logs/leaderboard_killer_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  LEADERBOARD KILLER"
echo "  MLP 1472 + FP16 embed + SmearGate + BigramHash + OrthoInit"
echo "  Target: <1.15 BPB"
echo "  Logs: $LOGDIR"
echo "============================================"

echo ""
echo "[1/1] FULL STACK"

QUANT_BITS=6 \
QAT_START_FRAC=0.25 \
EVAL_STRIDE=64 \
MUON_WD=0.01 \
FP16_EMBED=1 \
SMEAR_GATE=1 \
BIGRAM_HASH=1 \
ORTHO_INIT=1 \
MLP_HIDDEN=1472 \
RUN_ID=killer_fullstack \
NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run1_fullstack.log"

# --- Summary ---
echo ""
echo "============================================"
echo "  LEADERBOARD KILLER Complete. Results:"
echo "============================================"
echo "  Reference: killer v1 MLP1344 = quant_bpb:1.1725 (15.58MB)"
echo "  Reference: best_shot fp16embed = quant_bpb:1.1693 (16.34MB, over limit)"
echo "  Target: validated leaderboard #1 = 1.1483"
echo ""

for f in "$LOGDIR"/run*.log; do
    name=$(basename "$f" .log)
    bpb=$(grep -oP 'final_int6_ttt_lora val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    quant_bpb=$(grep -oP 'final_int6_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    size=$(grep -oP 'Total submission size int6\+zlib: \K\d+' "$f" | tail -1)
    echo "$name: ttt_bpb=${bpb:-N/A}  quant_bpb=${quant_bpb:-N/A}  artifact_bytes=${size:-N/A}"
done | tee "$LOGDIR/summary.txt"
