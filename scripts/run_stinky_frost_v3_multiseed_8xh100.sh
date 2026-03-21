#!/usr/bin/env bash
set -euo pipefail

# STINKY FROST V3 — Multi-Seed Validation (8xH100)
# Same config as V3 FINAL, 2 additional seeds (original uses 1337)

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export ITERATIONS="${ITERATIONS:-20000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

LOGDIR="logs/stinky_frost_v3_multiseed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

SEEDS=(42 137)

echo "============================================"
echo "  STINKY FROST V3 — Multi-Seed Validation"
echo "  2 additional seeds (original: 1337)"
echo "  Logs: $LOGDIR"
echo "============================================"

for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    run_num=$((i + 1))
    echo ""
    echo "[${run_num}/2] Seed ${seed}"

    SEED=${seed} \
    NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 \
    MLP_HIDDEN=1024 TIE_EMBEDDINGS=1 \
    QUANT_BITS=6 QAT_START_FRAC=0.25 EVAL_STRIDE=64 \
    FP16_EMBED=1 SMEAR_GATE=1 BIGRAM_HASH=1 ORTHO_INIT=1 \
    BIGRAM_BUCKETS=4096 \
    MUON_WD=0.04 \
    SWA_EVERY=50 SWA_START_FRAC=0.4 \
    USE_ZSTD=1 ZSTD_LEVEL=22 PRUNE_PCT=0.03 \
    XSA_LAST_N=3 \
    TTT_OPTIMIZER=sgd TTT_LORA_LR=0.002 TTT_EPOCHS=3 TTT_MOMENTUM=0.9 TTT_FREEZE_FIRST_N=2 \
    NCCL_IB_DISABLE=1 RUN_ID=v3_seed${seed} \
    torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
        2>&1 | tee "$LOGDIR/run${run_num}_seed${seed}.log"
done

echo ""
echo "============================================"
echo "  V3 MULTI-SEED Complete."
echo "============================================"

for f in "$LOGDIR"/run*.log; do
    name=$(basename "$f" .log)
    for label in int6_ttt_lora int5int6_ttt_lora; do
        bpb=$(grep -oP "final_${label} val_loss:\S+ val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
        [ -n "$bpb" ] && break
    done
    size=$(grep -oP 'Total submission size \S+: \K\d+' "$f" 2>/dev/null | tail -1)
    echo "$name: ttt_bpb=${bpb:-N/A} bytes=${size:-N/A}"
done | tee "$LOGDIR/summary.txt"

echo ""
echo "Mean BPB across seeds:"
grep -oP 'ttt_bpb=\K[0-9.]+' "$LOGDIR/summary.txt" | awk '{sum+=$1; n++} END {printf "  Mean: %.4f (n=%d)\n", sum/n, n}'
