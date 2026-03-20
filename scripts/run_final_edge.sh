#!/usr/bin/env bash
set -euo pipefail

# FINAL EDGE — v1 killer config + SWA + slightly wider MLP
#
# v1 got 1.1725 at 15.58MB (MLP1344, MuonWD=0.01, no SWA)
# This run: MLP1376 + SWA (every 50 steps, last 50%) — same MuonWD=0.01

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

LOGDIR="logs/final_edge_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  FINAL EDGE"
echo "  v1 killer + SWA + MLP1376"
echo "  Reference: v1 = 1.1725 BPB (15.58MB)"
echo "  Logs: $LOGDIR"
echo "============================================"

echo ""
echo "[1/1] FINAL EDGE"

QUANT_BITS=6 \
QAT_START_FRAC=0.25 \
EVAL_STRIDE=64 \
MUON_WD=0.01 \
FP16_EMBED=1 \
SMEAR_GATE=1 \
BIGRAM_HASH=1 \
ORTHO_INIT=1 \
MLP_HIDDEN=1376 \
SWA_EVERY=50 \
SWA_START_FRAC=0.5 \
RUN_ID=final_edge \
NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run1_final_edge.log"

echo ""
echo "============================================"
echo "  FINAL EDGE Complete."
echo "============================================"
echo "  Reference: v1 killer = quant_bpb:1.1725 (15.58MB)"
echo ""

f="$LOGDIR/run1_final_edge.log"
bpb=$(grep -oP 'final_int6_ttt_lora val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
quant_bpb=$(grep -oP 'final_int6_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
size=$(grep -oP 'Total submission size int6\+zlib: \K\d+' "$f" | tail -1)
echo "final_edge: ttt_bpb=${bpb:-N/A}  quant_bpb=${quant_bpb:-N/A}  artifact_bytes=${size:-N/A}"
