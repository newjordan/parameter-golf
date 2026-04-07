#!/usr/bin/env bash
set -euo pipefail
# ================================================================
# Crawler_Katta — RK2 / RK4 / RK2+RK4 hybrid crawler ablation
#
# Usage:
#   SEED=444 NPROC_PER_NODE=8 bash legs/2026-04-07_Crawler_Katta/gate.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt_brotli.py"
ITERATIONS=2000

mkdir -p "${SCRIPT_DIR}/results"
SUMMARY="${SCRIPT_DIR}/results/summary_s${SEED}_$(date +%Y%m%d_%H%M%S).tsv"

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
    local log="${SCRIPT_DIR}/results/${arm_name}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "================================================================"
    echo "  ${arm_name}: ${arm_desc}"
    echo "  loops=${loops} rope=${rope_scales} solver=${solver}"
    echo "  rk_heads=${rk_heads} rk_blend=${rk_blend} rk_recur=${rk_recur} hybrid_mix=${rk_hybrid_mix}"
    echo "  rk_battery=${rk_battery}"
    echo "  seed=${SEED} GPUs=${NPROC} steps=${ITERATIONS}"
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
        MAX_WALLCLOCK_SECONDS=3600 \
        WARMDOWN_ITERS="${ITERATIONS}" \
        ITERATIONS="${ITERATIONS}" \
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

# A0: Control
run_arm "A0_ctrl_euler_l3" "BW22-style control" "3" "9,1,1" "euler" "2" "-4.0" "0.0" "-1.5" "1.0,1.0,1.0,1.0"

# A1: RK2 fast with fewer loops
run_arm "A1_rk2_fast_l2" "RK2 fast, loops=2" "2" "9,1" "rk2_fast" "2" "-2.2" "0.15" "-1.5" "1.0,1.2,1.0,1.0"

# A2: RK4 fast with fewer loops
run_arm "A2_rk4_fast_l2" "RK4 fast, loops=2" "2" "9,1" "rk4_fast" "2" "-2.0" "0.20" "-1.5" "1.0,1.15,1.0,1.05"

# A3: RK2/RK4 hybrid with fewer loops
run_arm "A3_rk24_hybrid_l2" "RK2/RK4 hybrid, loops=2" "2" "9,1" "rk24_hybrid" "2" "-1.8" "0.20" "-0.4" "1.0,1.1,1.0,1.0"

# A4: RK2/RK4 hybrid at baseline loop depth
run_arm "A4_rk24_hybrid_l3" "RK2/RK4 hybrid, loops=3 battery" "3" "9,3,1" "rk24_hybrid" "2" "-1.6" "0.20" "0.0" "1.0,1.1,1.0,1.0"

echo ""
echo "================================================================"
echo "  Crawler_Katta gate complete"
echo "  summary: ${SUMMARY}"
echo "================================================================"
cat "${SUMMARY}"
