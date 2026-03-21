#!/usr/bin/env bash
set -euo pipefail

# PR #198 SOTA base (1.1318 BPB) + XSA last 3 layers + TTT SGD 3 epochs
# Source: jfprincz/parameter-golf submission/11l-int6-wd04-swa-fa3-1.1318
#
# All defaults match PR #198's proven config. XSA + TTT SGD are the only additions.

SEED="${1:-1337}"

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE=1024

# PR #198 training config
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3
export TIE_EMBEDDINGS=1
export ROPE_BASE=10000
export LOGIT_SOFTCAP=30
export QK_GAIN_INIT=1.5

export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export ITERATIONS=20000
export WARMDOWN_ITERS=1200
export WARMUP_STEPS=20
export MAX_WALLCLOCK_SECONDS=600
export TRAIN_LOG_EVERY=200
export VAL_LOSS_EVERY=1000

# PR #198 optimizer config
export MATRIX_LR=0.04
export SCALAR_LR=0.04
export TIED_EMBED_LR=0.05
export MUON_MOMENTUM=0.95
export MUON_MOMENTUM_WARMUP_START=0.85
export MUON_MOMENTUM_WARMUP_STEPS=500
export MUON_BACKEND_STEPS=5
export MUON_WD=0.02
export ADAM_WD=0.01
export GRAD_CLIP_NORM=0.3

# PR #198 features
export BIGRAM_VOCAB_SIZE=4096
export BIGRAM_DIM=128
export SWA_ENABLED=1
export SWA_EVERY=200
export EVAL_STRIDE=64

# --- OUR EDGES ---
export XSA_LAST_N=3
export TTT_OPTIMIZER=sgd
export TTT_LORA_LR=0.002
export TTT_EPOCHS=3
export TTT_MOMENTUM=0.9
export TTT_FREEZE_FIRST_N=2
export TTT_LORA_RANK=8
export TTT_CHUNK_SIZE=256
export TTT_EVAL_SEQ_LEN=1024
export TTT_BATCH_SIZE=64

export SEED="$SEED"
LOGDIR="logs/pr198_xsa_ttt_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  PR #198 SOTA + XSA + TTT SGD 3ep"
echo "  Seed: $SEED"
echo "  Logs: $LOGDIR"
echo "============================================"

NCCL_IB_DISABLE=1 \
RUN_ID="pr198_xsa_ttt_s${SEED}" \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/seed${SEED}.log"

echo ""
echo "============================================"
echo "  Run complete. Seed: $SEED"
echo "============================================"
f="$LOGDIR/seed${SEED}.log"
for label in int6_ttt_lora int6_sliding_window_s64 int6_sliding_window int6_roundtrip; do
    bpb=$(grep -oP "final_${label}\S* val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    [ -n "$bpb" ] && echo "  ${label}: ${bpb}" || true
done
size=$(grep -oP 'Total submission size \S+: \K\d+' "$f" 2>/dev/null | tail -1)
echo "  artifact_bytes=${size:-N/A}"
