#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# Crawler_Katta — 1xGPU medium sweep (signal finder)
#
# Goal:
# - Find RK2/RK4 hybrid settings that beat euler loop-3 on step_ms
# - While recovering as much val_bpb as possible
#
# Usage:
#   SEED=444 bash legs/2026-04-07_Crawler_Katta/sweep_1gpu_medium.sh
# Optional:
#   ITERATIONS=3000 TRAIN_BATCH_TOKENS=262144 MAX_WALLCLOCK_SECONDS=0 \
#   SEED=444 bash legs/2026-04-07_Crawler_Katta/sweep_1gpu_medium.sh
# ==============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-1}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt_brotli.py"

# Medium-sweep defaults for 1xGPU signal hunting.
ITERATIONS="${ITERATIONS:-3000}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-262144}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-250}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"

mkdir -p "${SCRIPT_DIR}/results"
SUMMARY="${SCRIPT_DIR}/results/summary_1gpu_medium_s${SEED}_$(date +%Y%m%d_%H%M%S).tsv"

run_arm() {
    local arm_name="$1"
    local arm_desc="$2"
    local loops="$3"
    local rope_scales="$4"
    local solver="$5"
    local rk_heads="$6"
    local rk_blend="$7"
    local rk_recur="$8"
    local rk_hybrid_mix="$9"
    local rk_battery="${10}"
    local log="${SCRIPT_DIR}/results/${arm_name}_1gpu_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "================================================================"
    echo "  ${arm_name}: ${arm_desc}"
    echo "  loops=${loops} rope=${rope_scales} solver=${solver}"
    echo "  rk_heads=${rk_heads} rk_blend=${rk_blend} rk_recur=${rk_recur} hybrid_mix=${rk_hybrid_mix}"
    echo "  rk_battery=${rk_battery}"
    echo "  seed=${SEED} GPUs=${NPROC} steps=${ITERATIONS}"
    echo "  train_batch_tokens=${TRAIN_BATCH_TOKENS} val_batch_size=${VAL_BATCH_SIZE}"
    echo "  log: ${log}"
    echo "================================================================"
    echo ""

    if command -v torchrun >/dev/null 2>&1; then
        TORCHRUN=(torchrun)
    else
        TORCHRUN=(python3 -m torch.distributed.run)
    fi

    env \
        SEED="${SEED}" \
        MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
        ITERATIONS="${ITERATIONS}" \
        WARMDOWN_ITERS="${ITERATIONS}" \
        TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS}" \
        VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
        WARMUP_STEPS="${WARMUP_STEPS}" \
        VAL_LOSS_EVERY="${VAL_LOSS_EVERY}" \
        TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY}" \
        COMPLEMENT_ALPHA=0 \
        XSA_LAST_N=11 \
        BIGRAM_VOCAB_SIZE=2048 \
        ROPE_DIMS=16 \
        SWA_EVERY=50 \
        MTP_NUM_HEADS=0 \
        LATE_QAT_THRESHOLD=0 \
        MATRIX_LR=0.03 \
        TORCHDYNAMO_OPTIMIZE_DDP=0 \
        COMPILE_FULLGRAPH=1 \
        NGRAM_EVAL_ORDER=0 \
        MODEL_DIM=512 \
        USE_CRAWLER=1 \
        NUM_FLAT_LAYERS=9 \
        NUM_CRAWLER_LAYERS=1 \
        CRAWLER_LOOPS="${loops}" \
        CRAWLER_MLP_MULT=6.0 \
        INST_DIM=32 \
        CRAWLER_QUANT_INT8=0 \
        DELTA_NET_HEADS=0 \
        SKIP_EMA=1 \
        SKIP_GPTQ=1 \
        LOOP_AWARE_GPTQ=0 \
        MLP_LEAKY_SLOPE=0.5 \
        CRAWLER_MLP_LEAKY_SLOPE=0.5 \
        CRAWLER_MLP_CHOKE_DIM=0 \
        CRAWLER_MLP_CHOKE_SHAPE=flat \
        CRAWLER_MLP_CHOKE_GROUPS=8 \
        CRAWLER_LOOP_ROPE_SCALES="${rope_scales}" \
        CRAWLER_LOOP_SMEAR=0 \
        CRAWLER_TAP_DIM=0 \
        CRAWLER_TAP_LOOP_SPECIFIC=1 \
        CRAWLER_TAP_LAYERS=all \
        ANCHOR_DIM=0 \
        FLAT_WEIGHT_SHARE=0 \
        CRAWLER_SOLVER="${solver}" \
        CRAWLER_RK_FAST_HEADS="${rk_heads}" \
        CRAWLER_RK_BLEND_INIT="${rk_blend}" \
        CRAWLER_RK_RECUR_GAIN_INIT="${rk_recur}" \
        CRAWLER_RK_HYBRID_MIX_INIT="${rk_hybrid_mix}" \
        CRAWLER_RK_BATTERY="${rk_battery}" \
        "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
        2>&1 | tee "${log}"

    local raw_bpb int6_sw_bpb step_ms bytes_total model_params
    raw_bpb="$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" | tail -1 || true)"
    int6_sw_bpb="$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" | tail -1 || true)"
    bytes_total="$(grep -oP 'Total submission size int6\+(?:brotli|zstd|zlib): \K[0-9]+' "${log}" | tail -1 || true)"
    step_ms="$(grep -oP 'step_avg:\K[0-9.]+' "${log}" | tail -1 || true)"
    model_params="$(grep -oP 'model_params:\K[0-9]+' "${log}" | tail -1 || true)"

    echo ""
    echo "  >> ${arm_name}: params=${model_params:-?} raw=${raw_bpb:-?} int6_sw=${int6_sw_bpb:-?} step_ms=${step_ms:-?} bytes=${bytes_total:-?}"
    echo ""

    if [[ ! -f "${SUMMARY}" ]]; then
        echo -e "arm\tdesc\tloops\trope_scales\tsolver\trk_heads\trk_blend\trk_recur\trk_hybrid_mix\trk_battery\tmodel_params\traw_bpb\tint6_sw_bpb\tstep_ms\tbytes\tdelta_vs_ctrl" > "${SUMMARY}"
    fi
    echo -e "${arm_name}\t${arm_desc}\t${loops}\t${rope_scales}\t${solver}\t${rk_heads}\t${rk_blend}\t${rk_recur}\t${rk_hybrid_mix}\t${rk_battery}\t${model_params:-?}\t${raw_bpb:-?}\t${int6_sw_bpb:-?}\t${step_ms:-?}\t${bytes_total:-?}\t?" >> "${SUMMARY}"
}

# M0: control
run_arm "M0_ctrl_euler_l3" "control baseline (loop3)" "3" "9,1,1" "euler" "2" "-4.0" "0.0" "-1.5" "1.0,1.0,1.0,1.0"

# M1: pure RK2 fast
run_arm "M1_rk2_l2" "rk2 fast (loop2)" "2" "9,1" "rk2_fast" "2" "-2.2" "0.15" "-1.5" "1.0,1.2,1.0,1.0"

# M2-M4: RK2/RK4 hybrid mix sweep at loop2
run_arm "M2_rk24_l2_mix_lo" "hybrid loop2, rk2-leaning" "2" "9,1" "rk24_hybrid" "2" "-1.8" "0.20" "-0.8" "1.0,1.1,1.0,1.0"
run_arm "M3_rk24_l2_mix_mid" "hybrid loop2, balanced" "2" "9,1" "rk24_hybrid" "2" "-1.8" "0.20" "0.0" "1.0,1.1,1.0,1.0"
run_arm "M4_rk24_l2_mix_hi" "hybrid loop2, rk4-leaning" "2" "9,1" "rk24_hybrid" "2" "-1.8" "0.20" "0.8" "1.0,1.1,1.0,1.0"

# M5-M6: depth/solver anchors
run_arm "M5_rk24_l3_mix_mid" "hybrid loop3 battery" "3" "9,3,1" "rk24_hybrid" "2" "-1.6" "0.20" "0.0" "1.0,1.1,1.0,1.0"
run_arm "M6_rk4_l2" "rk4 fast (loop2)" "2" "9,1" "rk4_fast" "2" "-2.0" "0.20" "-1.5" "1.0,1.15,1.0,1.05"

echo ""
echo "================================================================"
echo "  Crawler_Katta 1xGPU medium sweep complete"
echo "  summary: ${SUMMARY}"
echo "================================================================"
cat "${SUMMARY}"
