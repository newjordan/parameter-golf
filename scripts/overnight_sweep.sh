#!/bin/bash
# Parameter Golf — Overnight Hyperparameter Sweep
# Runs on DGX Spark GB10, logs results to OVERNIGHT_RESULTS.md
set -e

cd /home/frosty40/parameter-golf-lab
source .venv/bin/activate

RESULTS_FILE="OVERNIGHT_RESULTS.md"
ITERS=300
MAX_SECONDS=180
BATCH_TOKENS=16384
EVAL_TOKENS=1048576
LOG_EVERY=50

# Track global best
GLOBAL_BEST_BPB=9999
GLOBAL_BEST_CFG=""

# Initialize results file
cat > "$RESULTS_FILE" << 'HEADER'
# Parameter Golf — Overnight Sweep Results
**DGX Spark GB10**

## Phase 1: Dimension Sweep (baseline mode, 300 steps, 1M eval tokens)

| Exp | layers | dim | heads | kv | mlp | val_bpb | train_loss@300 | ms/step |
|-----|--------|-----|-------|----|----|---------|----------------|---------|
HEADER

run_exp() {
    local NAME="$1"
    local LAYERS="$2"
    local DIM="$3"
    local HEADS="$4"
    local KV="$5"
    local MLP="$6"
    local EXTRA_ARGS="${7:-}"
    local LOG="/tmp/pgolf_${NAME}.log"

    echo ""
    echo "▶ $NAME: ${LAYERS}L ${DIM}d ${HEADS}H kv${KV} mlp${MLP} $EXTRA_ARGS"

    # Build command — baseline mode, pass all dims as args
    python3 train_local.py \
        --mode baseline \
        --model-dim "$DIM" \
        --num-heads "$HEADS" \
        --num-kv-heads "$KV" \
        --mlp-mult "$MLP" \
        --iterations "$ITERS" \
        --max-seconds "$MAX_SECONDS" \
        --batch-tokens "$BATCH_TOKENS" \
        --eval-tokens "$EVAL_TOKENS" \
        --log-every "$LOG_EVERY" \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG"

    local VAL_BPB TRAIN_LOSS MS_STEP
    VAL_BPB=$(grep -oP 'val_bpb:\s*\K[0-9.]+' "$LOG" | tail -1 || echo "N/A")
    TRAIN_LOSS=$(grep -oP "^step:${ITERS}.*train_loss:\K[0-9.]+" "$LOG" 2>/dev/null | head -1 || echo "N/A")
    MS_STEP=$(grep -oP 'step_avg:\K[0-9.]+ms' "$LOG" | tail -1 || echo "N/A")

    echo "| $NAME | $LAYERS | $DIM | $HEADS | $KV | $MLP | **$VAL_BPB** | $TRAIN_LOSS | $MS_STEP |" >> "$RESULTS_FILE"

    # Track best
    if [[ "$VAL_BPB" != "N/A" ]] && python3 -c "exit(0 if float('$VAL_BPB') < float('$GLOBAL_BEST_BPB') else 1)" 2>/dev/null; then
        GLOBAL_BEST_BPB="$VAL_BPB"
        GLOBAL_BEST_CFG="$NAME (${LAYERS}L ${DIM}d ${HEADS}H kv${KV} mlp${MLP} $EXTRA_ARGS)"
        echo "  ★ NEW BEST: $GLOBAL_BEST_BPB"
    fi
}

# Phase 1: Dimension sweep — vary layers/dim within ~16-17M param budget
# train_local.py baseline mode hardcodes num_layers=9; we vary dim and mlp_mult
# To vary num_layers we need to patch — skip for now, do dim/mlp sweep first

run_exp "D1_baseline"  9 512 8 4 2
run_exp "D2_dim576"    9 576 8 4 2    # wider
run_exp "D3_dim448"    9 448 8 4 2    # narrower → more headroom
run_exp "D4_dim640"    9 640 8 4 2    # much wider
run_exp "D5_mlp3"      9 512 8 4 3    # bigger MLP
run_exp "D6_kv8"       9 512 8 8 2    # MQA → full MHA
run_exp "D7_kv2"       9 512 8 2 2    # more aggressive GQA
run_exp "D8_dim512_h4" 9 512 4 2 2    # fewer heads

echo "" >> "$RESULTS_FILE"
echo "**Phase 1 Best: $GLOBAL_BEST_CFG — val_bpb=$GLOBAL_BEST_BPB**" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

echo ""
echo "══════════════════════════════════════════"
echo "PHASE 1 DONE — Best: $GLOBAL_BEST_CFG ($GLOBAL_BEST_BPB)"
echo "══════════════════════════════════════════"

# ─── Phase 2: LR & Schedule Sweep ───────────────────────────────────────────
cat >> "$RESULTS_FILE" << 'H2'

## Phase 2: LR & Schedule Sweep (fixed 9L 512d baseline dims)

| Exp | warmup | lr | val_bpb | train_loss@300 | ms/step |
|-----|--------|-----|---------|----------------|---------|
H2

run_lr_exp() {
    local NAME="$1"
    local WARMUP="$2"
    local LR="$3"
    local LOG="/tmp/pgolf_${NAME}.log"

    echo ""
    echo "▶ $NAME: warmup=$WARMUP lr=$LR"

    python3 train_local.py \
        --mode baseline \
        --model-dim 512 \
        --iterations "$ITERS" \
        --max-seconds "$MAX_SECONDS" \
        --batch-tokens "$BATCH_TOKENS" \
        --eval-tokens "$EVAL_TOKENS" \
        --log-every "$LOG_EVERY" \
        --warmup-steps "$WARMUP" \
        --lr "$LR" \
        2>&1 | tee "$LOG"

    local VAL_BPB TRAIN_LOSS MS_STEP
    VAL_BPB=$(grep -oP 'val_bpb:\s*\K[0-9.]+' "$LOG" | tail -1 || echo "N/A")
    TRAIN_LOSS=$(grep -oP "^step:${ITERS}.*train_loss:\K[0-9.]+" "$LOG" 2>/dev/null | head -1 || echo "N/A")
    MS_STEP=$(grep -oP 'step_avg:\K[0-9.]+ms' "$LOG" | tail -1 || echo "N/A")

    echo "| $NAME | $WARMUP | $LR | **$VAL_BPB** | $TRAIN_LOSS | $MS_STEP |" >> "$RESULTS_FILE"

    if [[ "$VAL_BPB" != "N/A" ]] && python3 -c "exit(0 if float('$VAL_BPB') < float('$GLOBAL_BEST_BPB') else 1)" 2>/dev/null; then
        GLOBAL_BEST_BPB="$VAL_BPB"
        GLOBAL_BEST_CFG="$NAME (warmup=$WARMUP lr=$LR)"
        echo "  ★ NEW BEST: $GLOBAL_BEST_BPB"
    fi
}

run_lr_exp "L1_baseline"   20  3e-4
run_lr_exp "L2_lr5e4"      20  5e-4
run_lr_exp "L3_lr2e4"      20  2e-4
run_lr_exp "L4_lr1e3"      20  1e-3
run_lr_exp "L5_warmup5"     5  3e-4
run_lr_exp "L6_warmup50"   50  3e-4
run_lr_exp "L7_warmup10"   10  3e-4
run_lr_exp "L8_lr5e4_w10"  10  5e-4
run_lr_exp "L9_lr1e3_w30"  30  1e-3

echo "" >> "$RESULTS_FILE"
echo "**Phase 2 Best so far: $GLOBAL_BEST_CFG — val_bpb=$GLOBAL_BEST_BPB**" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

echo ""
echo "══════════════════════════════════════════"
echo "PHASE 2 DONE — Best: $GLOBAL_BEST_CFG ($GLOBAL_BEST_BPB)"
echo "══════════════════════════════════════════"

# ─── Phase 3: Batch size sweep ───────────────────────────────────────────────
cat >> "$RESULTS_FILE" << 'H3'

## Phase 3: Batch Size Sweep

| Exp | batch_tokens | val_bpb | train_loss@300 | ms/step |
|-----|-------------|---------|----------------|---------|
H3

run_batch_exp() {
    local NAME="$1"
    local BATCH="$2"
    local LOG="/tmp/pgolf_${NAME}.log"

    echo ""
    echo "▶ $NAME: batch=$BATCH"

    python3 train_local.py \
        --mode baseline \
        --model-dim 512 \
        --iterations "$ITERS" \
        --max-seconds "$MAX_SECONDS" \
        --batch-tokens "$BATCH" \
        --eval-tokens "$EVAL_TOKENS" \
        --log-every "$LOG_EVERY" \
        2>&1 | tee "$LOG"

    local VAL_BPB TRAIN_LOSS MS_STEP
    VAL_BPB=$(grep -oP 'val_bpb:\s*\K[0-9.]+' "$LOG" | tail -1 || echo "N/A")
    TRAIN_LOSS=$(grep -oP "^step:${ITERS}.*train_loss:\K[0-9.]+" "$LOG" 2>/dev/null | head -1 || echo "N/A")
    MS_STEP=$(grep -oP 'step_avg:\K[0-9.]+ms' "$LOG" | tail -1 || echo "N/A")

    echo "| $NAME | $BATCH | **$VAL_BPB** | $TRAIN_LOSS | $MS_STEP |" >> "$RESULTS_FILE"

    if [[ "$VAL_BPB" != "N/A" ]] && python3 -c "exit(0 if float('$VAL_BPB') < float('$GLOBAL_BEST_BPB') else 1)" 2>/dev/null; then
        GLOBAL_BEST_BPB="$VAL_BPB"
        GLOBAL_BEST_CFG="$NAME (batch=$BATCH)"
        echo "  ★ NEW BEST: $GLOBAL_BEST_BPB"
    fi
}

run_batch_exp "B1_baseline"  16384
run_batch_exp "B2_batch32k"  32768
run_batch_exp "B3_batch8k"    8192
run_batch_exp "B4_batch64k"  65536

echo "" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "## FINAL SUMMARY" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "**Global Best: $GLOBAL_BEST_CFG**" >> "$RESULTS_FILE"
echo "**val_bpb: $GLOBAL_BEST_BPB**" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "Baseline target to beat on H100: **1.2244** (need ≤1.2194)" >> "$RESULTS_FILE"

echo ""
echo "══════════════════════════════════════════"
echo "ALL PHASES COMPLETE"
echo "Global Best: $GLOBAL_BEST_CFG"
echo "val_bpb: $GLOBAL_BEST_BPB"
echo "══════════════════════════════════════════"
cat "$RESULTS_FILE"
