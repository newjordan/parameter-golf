#!/usr/bin/env bash
set -euo pipefail

# EDGE A/B TESTING — 1 GPU quick comparisons
#
# Runs shorter experiments (50 min each) on 1 GPU to find edges.
# ~5500 steps = enough to see relative trends.
#
# Baseline: Stinky Frost v1 (MLP1344, WD=0.01, no SWA) → 1.1725 on 8xH100
#
# Test A: + SWA (every 50, last 50%)           → does SWA help quant BPB?
# Test B: MLP1376 + SWA                        → use headroom + SWA
# Test C: MuonWD=0.02 + SWA                    → higher WD sweet spot?
# Test D: MuonWD=0.03 + SWA                    → even higher WD?
#
# Compare quant BPB across all 4. Winner gets a full 8xH100 run.

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

# 50 min per run on 1 GPU ≈ 5500 steps (half of full training)
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-3000}"

LOGDIR="logs/edge_ab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  EDGE A/B TESTING — 1 GPU"
echo "  50 min per test, 4 tests, ~3.5 hrs total"
echo "  Logs: $LOGDIR"
echo "============================================"

# --- Test A: v1 + SWA ---
echo ""
echo "[1/4] Test A: Stinky Frost + SWA"
NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
MLP_HIDDEN=1344 MUON_WD=0.01 \
SWA_EVERY=50 SWA_START_FRAC=0.5 \
QUANT_BITS=6 QAT_START_FRAC=0.25 EVAL_STRIDE=64 FP16_EMBED=1 \
SMEAR_GATE=1 BIGRAM_HASH=1 ORTHO_INIT=1 TIE_EMBEDDINGS=1 \
NCCL_IB_DISABLE=1 RUN_ID=edge_a_swa \
torchrun --standalone --nproc_per_node="${NPROC:-1}" train_gpt.py \
    2>&1 | tee "$LOGDIR/test_a_swa.log"

# --- Test B: MLP1376 + SWA ---
echo ""
echo "[2/4] Test B: MLP1376 + SWA"
NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
MLP_HIDDEN=1376 MUON_WD=0.01 \
SWA_EVERY=50 SWA_START_FRAC=0.5 \
QUANT_BITS=6 QAT_START_FRAC=0.25 EVAL_STRIDE=64 FP16_EMBED=1 \
SMEAR_GATE=1 BIGRAM_HASH=1 ORTHO_INIT=1 TIE_EMBEDDINGS=1 \
NCCL_IB_DISABLE=1 RUN_ID=edge_b_mlp1376 \
torchrun --standalone --nproc_per_node="${NPROC:-1}" train_gpt.py \
    2>&1 | tee "$LOGDIR/test_b_mlp1376.log"

# --- Test C: MuonWD=0.02 + SWA ---
echo ""
echo "[3/4] Test C: MuonWD=0.02 + SWA"
NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
MLP_HIDDEN=1344 MUON_WD=0.02 \
SWA_EVERY=50 SWA_START_FRAC=0.5 \
QUANT_BITS=6 QAT_START_FRAC=0.25 EVAL_STRIDE=64 FP16_EMBED=1 \
SMEAR_GATE=1 BIGRAM_HASH=1 ORTHO_INIT=1 TIE_EMBEDDINGS=1 \
NCCL_IB_DISABLE=1 RUN_ID=edge_c_wd02 \
torchrun --standalone --nproc_per_node="${NPROC:-1}" train_gpt.py \
    2>&1 | tee "$LOGDIR/test_c_wd02.log"

# --- Test D: MuonWD=0.03 + SWA ---
echo ""
echo "[4/4] Test D: MuonWD=0.03 + SWA"
NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
MLP_HIDDEN=1344 MUON_WD=0.03 \
SWA_EVERY=50 SWA_START_FRAC=0.5 \
QUANT_BITS=6 QAT_START_FRAC=0.25 EVAL_STRIDE=64 FP16_EMBED=1 \
SMEAR_GATE=1 BIGRAM_HASH=1 ORTHO_INIT=1 TIE_EMBEDDINGS=1 \
NCCL_IB_DISABLE=1 RUN_ID=edge_d_wd03 \
torchrun --standalone --nproc_per_node="${NPROC:-1}" train_gpt.py \
    2>&1 | tee "$LOGDIR/test_d_wd03.log"

# --- Summary ---
echo ""
echo "============================================"
echo "  EDGE A/B Complete. Results:"
echo "============================================"
echo "  Reference: Stinky Frost v1 (8xH100) = quant_bpb:1.1725"
echo ""

for f in "$LOGDIR"/test_*.log; do
    name=$(basename "$f" .log)
    steps=$(grep -oP 'stopping_early.*step:\K\d+' "$f" | tail -1)
    bpb=$(grep -oP 'final_int6_ttt_lora val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    quant_bpb=$(grep -oP 'final_int6_zlib_roundtrip val_loss:\S+ val_bpb:\K\S+' "$f" | tail -1)
    size=$(grep -oP 'Total submission size int6\+zlib: \K\d+' "$f" | tail -1)
    echo "$name: steps=${steps:-N/A}  ttt_bpb=${bpb:-N/A}  quant_bpb=${quant_bpb:-N/A}  bytes=${size:-N/A}"
done | tee "$LOGDIR/summary.txt"

echo ""
echo "Pick the winner → run full 8xH100 validation"
