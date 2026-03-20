#!/usr/bin/env bash
set -euo pipefail

# EDGE FINDER — 4 variations on The Stinky Frost Recipe
#
# Baseline (v1): MLP1344, MuonWD=0.01, no SWA         → 1.1725 (known)
# Edge A: + SWA (every 50, last 50%)                   → smoother quant?
# Edge B: MLP1376 + SWA                                → use headroom
# Edge C: MuonWD=0.02 + SWA                            → middle-ground WD
# Edge D: 11L/480d + MLP1344 + SWA                     → deeper model
#
# All runs use: Int6 QAT@25%, FP16 embed, SmearGate, BigramHash, OrthoInit, stride64

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export ITERATIONS="${ITERATIONS:-20000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

LOGDIR="logs/edge_finder_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

# Common flags for all runs
COMMON="QUANT_BITS=6 QAT_START_FRAC=0.25 EVAL_STRIDE=64 FP16_EMBED=1 SMEAR_GATE=1 BIGRAM_HASH=1 ORTHO_INIT=1 TIE_EMBEDDINGS=1 NCCL_IB_DISABLE=1"

echo "============================================"
echo "  EDGE FINDER — 4 variations"
echo "  Logs: $LOGDIR"
echo "============================================"

# --- Edge A: v1 + SWA ---
echo ""
echo "[1/4] Edge A: v1 + SWA"
NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
MLP_HIDDEN=1344 MUON_WD=0.01 \
SWA_EVERY=50 SWA_START_FRAC=0.5 \
QUANT_BITS=6 QAT_START_FRAC=0.25 EVAL_STRIDE=64 FP16_EMBED=1 \
SMEAR_GATE=1 BIGRAM_HASH=1 ORTHO_INIT=1 TIE_EMBEDDINGS=1 \
NCCL_IB_DISABLE=1 RUN_ID=edge_a_swa \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run1_edge_a_swa.log"

# --- Edge B: MLP1376 + SWA ---
echo ""
echo "[2/4] Edge B: MLP1376 + SWA"
NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
MLP_HIDDEN=1376 MUON_WD=0.01 \
SWA_EVERY=50 SWA_START_FRAC=0.5 \
QUANT_BITS=6 QAT_START_FRAC=0.25 EVAL_STRIDE=64 FP16_EMBED=1 \
SMEAR_GATE=1 BIGRAM_HASH=1 ORTHO_INIT=1 TIE_EMBEDDINGS=1 \
NCCL_IB_DISABLE=1 RUN_ID=edge_b_mlp1376 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run2_edge_b_mlp1376.log"

# --- Edge C: MuonWD=0.02 + SWA ---
echo ""
echo "[3/4] Edge C: MuonWD=0.02 + SWA"
NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
MLP_HIDDEN=1344 MUON_WD=0.02 \
SWA_EVERY=50 SWA_START_FRAC=0.5 \
QUANT_BITS=6 QAT_START_FRAC=0.25 EVAL_STRIDE=64 FP16_EMBED=1 \
SMEAR_GATE=1 BIGRAM_HASH=1 ORTHO_INIT=1 TIE_EMBEDDINGS=1 \
NCCL_IB_DISABLE=1 RUN_ID=edge_c_wd02 \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run3_edge_c_wd02.log"

# --- Edge D: 11L/480d + SWA ---
echo ""
echo "[4/4] Edge D: 11L/480d + SWA"
NUM_LAYERS=11 MODEL_DIM=480 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
MLP_HIDDEN=1344 MUON_WD=0.01 \
SWA_EVERY=50 SWA_START_FRAC=0.5 \
QUANT_BITS=6 QAT_START_FRAC=0.25 EVAL_STRIDE=64 FP16_EMBED=1 \
SMEAR_GATE=1 BIGRAM_HASH=1 ORTHO_INIT=1 TIE_EMBEDDINGS=1 \
NCCL_IB_DISABLE=1 RUN_ID=edge_d_11L \
torchrun --standalone --nproc_per_node="${NPROC:-8}" train_gpt.py \
    2>&1 | tee "$LOGDIR/run4_edge_d_11L.log"

# --- Summary ---
echo ""
echo "============================================"
echo "  EDGE FINDER Complete. Results:"
echo "============================================"
echo "  Reference: Stinky Frost v1 = 1.1725 BPB (15.58MB)"
echo ""

for f in "$LOGDIR"/run*.log; do
    name=$(basename "$f" .log)
    bpb=$(grep -oP 'final_int6_ttt_lora val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    quant_bpb=$(grep -oP 'final_int6_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    size=$(grep -oP 'Total submission size int6\+zlib: \K\d+' "$f" | tail -1)
    echo "$name: ttt_bpb=${bpb:-N/A}  quant_bpb=${quant_bpb:-N/A}  artifact_bytes=${size:-N/A}"
done | tee "$LOGDIR/summary.txt"
