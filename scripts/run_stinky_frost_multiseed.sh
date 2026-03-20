#!/usr/bin/env bash
set -euo pipefail

# THE STINKY FROST RECIPE — Multi-Seed Validation
#
# Exact config from PR #190 (1.1725 BPB, 15.58MB)
# Runs 3 seeds to demonstrate ≥0.005-nat significance
#
# Original run used seed 1337 (default) → 1.1725 BPB
# Seeds: 42, 137 (2 additional seeds)

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

LOGDIR="logs/stinky_frost_multiseed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

SEEDS=(42 137)

echo "============================================"
echo "  THE STINKY FROST RECIPE — Multi-Seed"
echo "  2 additional seeds (original: 1337 → 1.1725)"
echo "  Logs: $LOGDIR"
echo "============================================"

for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    run_num=$((i + 1))
    echo ""
    echo "[${run_num}/2] Seed ${seed}"

    SEED=${seed} \
    QUANT_BITS=6 \
    QAT_START_FRAC=0.25 \
    EVAL_STRIDE=64 \
    MUON_WD=0.01 \
    FP16_EMBED=1 \
    SMEAR_GATE=1 \
    BIGRAM_HASH=1 \
    ORTHO_INIT=1 \
    MLP_HIDDEN=1344 \
    NCCL_IB_DISABLE=1 \
    RUN_ID=stinky_frost_seed${seed} \
    torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
        2>&1 | tee "$LOGDIR/run${run_num}_seed${seed}.log"
done

# --- Summary ---
echo ""
echo "============================================"
echo "  STINKY FROST MULTI-SEED Complete."
echo "============================================"
echo ""

for f in "$LOGDIR"/run*.log; do
    name=$(basename "$f" .log)
    bpb=$(grep -oP 'final_int6_ttt_lora val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    quant_bpb=$(grep -oP 'final_int6_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    size=$(grep -oP 'Total submission size int6\+zlib: \K\d+' "$f" | tail -1)
    echo "$name: ttt_bpb=${bpb:-N/A}  quant_bpb=${quant_bpb:-N/A}  artifact_bytes=${size:-N/A}"
done | tee "$LOGDIR/summary.txt"

echo ""
echo "Compute mean BPB across seeds:"
grep -oP 'quant_bpb=\K[0-9.]+' "$LOGDIR/summary.txt" | awk '{sum+=$1; n++} END {printf "  Mean quant_bpb: %.4f (n=%d)\n", sum/n, n}'
