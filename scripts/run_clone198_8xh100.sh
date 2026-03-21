#!/usr/bin/env bash
set -euo pipefail

# CLONE PR #198 — Exact SOTA config (1.1326 BPB, 3-seed validated)
# @jfprincz: 11L + Int6 MLP3x + SmearGate + BigramHash + WD0.04 + SWA + FA3
#
# Then we add our edges on top: XSA, TTT SGD 3ep

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export ITERATIONS="${ITERATIONS:-9000}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

LOGDIR="logs/clone198_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  CLONE PR #198 — SOTA baseline"
echo "  11L MLP3x seq2048 WD0.04 SWA"
echo "  Logs: $LOGDIR"
echo "============================================"

NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
TIE_EMBEDDINGS=1 \
QUANT_BITS=6 QAT_START_FRAC=0.25 EVAL_STRIDE=64 \
FP16_EMBED=1 SMEAR_GATE=1 BIGRAM_HASH=1 ORTHO_INIT=1 \
BIGRAM_BUCKETS=2048 BIGRAM_DIM=128 \
MUON_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
ROPE_BASE=50000 \
SWA_EVERY=50 SWA_START_FRAC=0.6 \
USE_ZSTD=1 ZSTD_LEVEL=22 PRUNE_PCT=0.03 \
XSA_LAST_N=3 \
TTT_OPTIMIZER=sgd TTT_LORA_LR=0.002 TTT_EPOCHS=3 TTT_MOMENTUM=0.9 TTT_FREEZE_FIRST_N=2 \
NCCL_IB_DISABLE=1 RUN_ID=clone198 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run1_clone198.log"

echo ""
echo "============================================"
echo "  CLONE 198 Complete."
echo "============================================"
echo "  Reference: PR #198 = 1.1326 BPB (3-seed)"
echo ""

f="$LOGDIR/run1_clone198.log"
for label in int6_ttt_lora int6_zstd22 int6_zlib_roundtrip; do
    bpb=$(grep -oP "final_${label}\S* val_loss:\S+ val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    [ -n "$bpb" ] && echo "${label}: ${bpb}" && break
done
size=$(grep -oP 'Total submission size \S+: \K\d+' "$f" 2>/dev/null | tail -1)
steps=$(grep -oP 'stopping_early.*step:\K\d+' "$f" 2>/dev/null | tail -1)
echo "steps=${steps:-N/A} artifact_bytes=${size:-N/A}"
