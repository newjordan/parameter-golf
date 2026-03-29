#!/usr/bin/env bash
set -euo pipefail
# ══════════════════════════════════════════════════════════════════
# Bandit_Wagon signal ablations
# Scale: 0.25 (150s wallclock)    GPU: 1x H100    DeltaNet: OFF
#
# Tests the two independent levers for using Bandit's ~6.65 MB headroom:
#   Width: dim 384 → 432 → 480  (+12.5%, +25% — mirrors BW-01/02 at prod scale)
#   Depth: dim 384, flat 4 → 5 → 6 (mirrors BW-03/04)
#
# Uses train_gpt_h4_compiled.py (H-series signal script) with crawler
# architecture matching production Bandit: 4F+1C×4, DN=0.
# ══════════════════════════════════════════════════════════════════

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

if ! python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    if [ -d "flash-attention/hopper" ]; then
        export PYTHONPATH="$(pwd)/flash-attention/hopper:${PYTHONPATH:-}"
    else
        echo "ERROR: flash_attn_interface not found." && exit 1
    fi
fi

NPROC=1
SEED="${SEED:-1337}"
RESULTS_DIR="experiments/Bandit_Wagon/results/signal_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR" checkpoints

# ── Shared config: production Bandit topology at 0.25 scale ─────
SHARED_ENV=(
    SEED="$SEED"
    NUM_HEADS=6 NUM_KV_HEADS=3 MLP_MULT=3 VOCAB_SIZE=1024
    CRAWLER_MLP_MULT=4
    NUM_CRAWLER_LAYERS=1 CRAWLER_LOOPS=4
    CRAWLER_CADENCE_EARLY=1 CRAWLER_CADENCE_MAIN=1 CRAWLER_CADENCE_LATE=1
    XSA_LAST_N=2 ROPE_DIMS=16
    TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30.0
    TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432
    ITERATIONS=20000 WARMUP_STEPS=20 GRAD_CLIP_NORM=0.3
    MAX_WALLCLOCK_SECONDS=150 WARMDOWN_ITERS=500
    MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 TIED_EMBED_INIT_STD=0.005
    MUON_MOMENTUM=0.99 MUON_BACKEND_STEPS=5 MUON_WD=0.04 ADAM_WD=0.04 MUON_BETA2=0.95
    MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
    SWA_ENABLED=1 SWA_EVERY=50 QAT_ENABLED=0 LATE_QAT_THRESHOLD=0.15
    EVAL_STRIDE=64 VAL_LOSS_EVERY=500 VAL_BATCH_SIZE=524288
    DIAG_FIXED_CADENCE=0 DIAG_FAST_VAL=1
    VE_ENABLED=0 TTT_BURST_ENABLED=0 DISTILL_ENABLED=0 POLAR_ENABLED=0 DTG_ENABLED=0
    TS_PD_ENABLED=0
)

run_arm() {
    local tag="$1"; shift
    local run_id="${tag}_$(date +%Y%m%d_%H%M%S)"
    local arm_dir="${RESULTS_DIR}/${tag}"
    mkdir -p "$arm_dir"

    echo ""
    echo "════════════════════════════════════════"
    echo "  ${tag}   RUN_ID=${run_id}"
    echo "════════════════════════════════════════"

    env \
        "${SHARED_ENV[@]}" \
        "$@" \
        RUN_ID="$run_id" \
        DIAG_CSV_PATH="${arm_dir}/diag.csv" \
        torchrun --standalone --nproc_per_node="$NPROC" train_gpt_h4_compiled.py \
        2>&1 | tee "${arm_dir}/run.log"

    cp final_model.pt     "checkpoints/${run_id}_final.pt"     2>/dev/null || true
    cp final_model.int6.ptz "checkpoints/${run_id}_final.int6.ptz" 2>/dev/null || true
}

# ── Width arms (dim ratio: ×1.0, ×1.125, ×1.25) ─────────────────
# Mirrors production BW-00/BW-01/BW-02 (512 → 576 → 640)
run_arm "BW-S00_anchor_dim384_4flat"   MODEL_DIM=384 NUM_FLAT_LAYERS=4
run_arm "BW-S01_width_dim432_4flat"    MODEL_DIM=432 NUM_FLAT_LAYERS=4
run_arm "BW-S02_width_dim480_4flat"    MODEL_DIM=480 NUM_FLAT_LAYERS=4

# ── Depth arms (dim fixed, flat layers vary) ─────────────────────
# Mirrors production BW-03/BW-04 (5F, 6F at dim=512)
run_arm "BW-S03_depth_dim384_5flat"    MODEL_DIM=384 NUM_FLAT_LAYERS=5
run_arm "BW-S04_depth_dim384_6flat"    MODEL_DIM=384 NUM_FLAT_LAYERS=6

# ── Extract summary ───────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "  Bandit_Wagon signal summary"
echo "════════════════════════════════════════"

SUMMARY_TSV="${RESULTS_DIR}/summary.tsv"
printf "arm\troundtrip_bpb\tsliding_bpb\tstatus\n" > "$SUMMARY_TSV"

for arm_dir in "${RESULTS_DIR}"/BW-S*; do
    tag="$(basename "$arm_dir")"
    log="${arm_dir}/run.log"
    if [ ! -f "$log" ]; then
        printf "%s\t-\t-\tmissing\n" "$tag" >> "$SUMMARY_TSV"
        continue
    fi
    rt=$(grep -oP "final_int6_roundtrip_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+" "$log" | tail -1 || echo "-")
    sw=$(grep -oP "final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+" "$log" | tail -1 || echo "-")
    st=$(grep -q "DONE\|final_int6" "$log" && echo "ok" || echo "incomplete")
    printf "%s\t%s\t%s\t%s\n" "$tag" "$rt" "$sw" "$st" >> "$SUMMARY_TSV"
done

cat "$SUMMARY_TSV"
echo ""
echo "Full logs: ${RESULTS_DIR}/"
