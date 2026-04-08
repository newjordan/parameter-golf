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
# Resume behavior:
#   RESUME_SWEEP=1 (default) reuses latest summary for this seed and skips done arms.
#   Set START_AT_ARM=M1_rk2_l2 to force resume at a specific arm.
# ==============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-1}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt_brotli.py"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
MIN_TRAIN_TOKENS="${MIN_TRAIN_TOKENS:-50000000}"
MIN_VAL_TOKENS="${MIN_VAL_TOKENS:-10000000}"
ALLOW_TINY_DATASET="${ALLOW_TINY_DATASET:-0}"

# Medium-sweep defaults for 1xGPU signal hunting.
ITERATIONS="${ITERATIONS:-3000}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-262144}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-250}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
RESUME_SWEEP="${RESUME_SWEEP:-1}"
SUMMARY_PATH="${SUMMARY_PATH:-}"
START_AT_ARM="${START_AT_ARM:-}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"

# Compile policy:
# - Euler control keeps previous fast compile defaults.
# - RK variants default to compile-disabled to avoid torch inductor backward assert on this stack.
COMPILE_ENABLED_EULER="${COMPILE_ENABLED_EULER:-1}"
COMPILE_FULLGRAPH_EULER="${COMPILE_FULLGRAPH_EULER:-1}"
COMPILE_ENABLED_RK="${COMPILE_ENABLED_RK:-0}"
COMPILE_FULLGRAPH_RK="${COMPILE_FULLGRAPH_RK:-0}"

mkdir -p "${SCRIPT_DIR}/results"
if [[ -n "${SUMMARY_PATH}" ]]; then
    SUMMARY="${SUMMARY_PATH}"
elif [[ "${RESUME_SWEEP}" == "1" ]]; then
    latest_summary="$(ls -1t "${SCRIPT_DIR}/results"/summary_1gpu_medium_s${SEED}_*.tsv 2>/dev/null | head -n 1 || true)"
    if [[ -n "${latest_summary}" ]]; then
        SUMMARY="${latest_summary}"
    else
        SUMMARY="${SCRIPT_DIR}/results/summary_1gpu_medium_s${SEED}_$(date +%Y%m%d_%H%M%S).tsv"
    fi
else
    SUMMARY="${SCRIPT_DIR}/results/summary_1gpu_medium_s${SEED}_$(date +%Y%m%d_%H%M%S).tsv"
fi
echo "Sweep summary file: ${SUMMARY}"

if [[ ! -f "${SUMMARY}" ]]; then
    echo -e "arm\tdesc\tloops\trope_scales\tsolver\trk_heads\trk_blend\trk_recur\trk_hybrid_mix\trk_battery\tmodel_params\traw_bpb\tint6_sw_bpb\tstep_ms\tbytes\tdelta_vs_ctrl\tstatus\tlog" > "${SUMMARY}"
fi

if [[ ! -f "${TOKENIZER_PATH}" ]]; then
    echo "ERROR: TOKENIZER_PATH not found: ${TOKENIZER_PATH}"
    exit 1
fi

if [[ ! -d "${DATA_PATH}" ]]; then
    echo "ERROR: DATA_PATH not found: ${DATA_PATH}"
    exit 1
fi

shopt -s nullglob
train_shards=( "${DATA_PATH}"/fineweb_train_*.bin )
val_shards=( "${DATA_PATH}"/fineweb_val_*.bin )
shopt -u nullglob

if [[ "${#train_shards[@]}" -eq 0 ]]; then
    echo "ERROR: no train shards at ${DATA_PATH}/fineweb_train_*.bin"
    exit 1
fi

if [[ "${#val_shards[@]}" -eq 0 ]]; then
    echo "ERROR: no val shards at ${DATA_PATH}/fineweb_val_*.bin"
    exit 1
fi

dataset_stats="$(
python3 - "${train_shards[@]}" --SEP-- "${val_shards[@]}" <<'PY'
import struct
import sys
def count_tokens(paths):
    total = 0
    for p in paths:
        with open(p, "rb") as f:
            hdr = f.read(12)
        if len(hdr) != 12:
            raise SystemExit(f"short header: {p}")
        magic, version, n_tok = struct.unpack("<iii", hdr)
        if magic != 20240520 or version != 1:
            raise SystemExit(f"bad shard header: {p} (magic={magic} version={version})")
        total += n_tok
    return total

args = sys.argv[1:]
sep = args.index("--SEP--")
train = args[:sep]
val = args[sep + 1 :]
print(count_tokens(train), count_tokens(val))
PY
)"

read -r train_tokens_total val_tokens_total <<<"${dataset_stats}"

echo "Data preflight: DATA_PATH=${DATA_PATH}"
echo "Data preflight: train_shards=${#train_shards[@]} val_shards=${#val_shards[@]} train_tokens=${train_tokens_total} val_tokens=${val_tokens_total}"

if [[ "${ALLOW_TINY_DATASET}" != "1" ]]; then
    if [[ "${train_tokens_total}" -lt "${MIN_TRAIN_TOKENS}" ]]; then
        echo "ERROR: train_tokens=${train_tokens_total} < MIN_TRAIN_TOKENS=${MIN_TRAIN_TOKENS}."
        echo "Set DATA_PATH to full dataset or override with ALLOW_TINY_DATASET=1."
        exit 1
    fi
    if [[ "${val_tokens_total}" -lt "${MIN_VAL_TOKENS}" ]]; then
        echo "ERROR: val_tokens=${val_tokens_total} < MIN_VAL_TOKENS=${MIN_VAL_TOKENS}."
        echo "Set DATA_PATH to full dataset or override with ALLOW_TINY_DATASET=1."
        exit 1
    fi
fi

resume_started=1
if [[ -n "${START_AT_ARM}" ]]; then
    resume_started=0
fi

arm_already_done() {
    local arm_name="$1"
    if [[ ! -f "${SUMMARY}" ]]; then
        return 1
    fi
    awk -F'\t' -v arm="${arm_name}" '
        NR == 1 { next }
        $1 == arm {
            # Older summary rows do not have status. Treat as done if row exists.
            if (NF < 17) { ok = 1 }
            else if ($17 == "OK") { ok = 1 }
        }
        END { exit(ok ? 0 : 1) }
    ' "${SUMMARY}"
}

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
    local compile_enabled compile_fullgraph run_rc status
    local log="${SCRIPT_DIR}/results/${arm_name}_1gpu_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

    if [[ "${resume_started}" != "1" ]]; then
        if [[ "${arm_name}" == "${START_AT_ARM}" ]]; then
            resume_started=1
        else
            echo "  >> ${arm_name}: skipped (waiting for START_AT_ARM=${START_AT_ARM})"
            return 0
        fi
    fi

    if arm_already_done "${arm_name}"; then
        echo "  >> ${arm_name}: skipped (already completed in ${SUMMARY})"
        return 0
    fi

    if [[ "${solver}" == "euler" ]]; then
        compile_enabled="${COMPILE_ENABLED_EULER}"
        compile_fullgraph="${COMPILE_FULLGRAPH_EULER}"
    else
        compile_enabled="${COMPILE_ENABLED_RK}"
        compile_fullgraph="${COMPILE_FULLGRAPH_RK}"
    fi

    echo ""
    echo "================================================================"
    echo "  ${arm_name}: ${arm_desc}"
    echo "  loops=${loops} rope=${rope_scales} solver=${solver}"
    echo "  rk_heads=${rk_heads} rk_blend=${rk_blend} rk_recur=${rk_recur} hybrid_mix=${rk_hybrid_mix}"
    echo "  rk_battery=${rk_battery}"
    echo "  seed=${SEED} GPUs=${NPROC} steps=${ITERATIONS}"
    echo "  train_batch_tokens=${TRAIN_BATCH_TOKENS} val_batch_size=${VAL_BATCH_SIZE}"
    echo "  compile_enabled=${compile_enabled} compile_fullgraph=${compile_fullgraph}"
    echo "  log: ${log}"
    echo "================================================================"
    echo ""

    if command -v torchrun >/dev/null 2>&1; then
        TORCHRUN=(torchrun)
    else
        TORCHRUN=(python3 -m torch.distributed.run)
    fi

    set +e
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
        COMPILE_ENABLED="${compile_enabled}" \
        COMPILE_FULLGRAPH="${compile_fullgraph}" \
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
        DATA_PATH="${DATA_PATH}" \
        TOKENIZER_PATH="${TOKENIZER_PATH}" \
        "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
        2>&1 | tee "${log}"
    run_rc=${PIPESTATUS[0]}
    set -e

    local raw_bpb int6_sw_bpb step_ms bytes_total model_params
    raw_bpb="$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" | tail -1 || true)"
    int6_sw_bpb="$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" | tail -1 || true)"
    bytes_total="$(grep -oP 'Total submission size int6\+(?:brotli|zstd|zlib): \K[0-9]+' "${log}" | tail -1 || true)"
    step_ms="$(grep -oP 'step_avg:\K[0-9.]+' "${log}" | tail -1 || true)"
    model_params="$(grep -oP 'model_params:\K[0-9]+' "${log}" | tail -1 || true)"

    echo ""
    echo "  >> ${arm_name}: params=${model_params:-?} raw=${raw_bpb:-?} int6_sw=${int6_sw_bpb:-?} step_ms=${step_ms:-?} bytes=${bytes_total:-?}"
    echo "  >> ${arm_name}: status=$([[ "${run_rc}" -eq 0 ]] && echo OK || echo FAIL)"
    echo ""

    if [[ "${run_rc}" -eq 0 ]]; then
        status="OK"
    else
        status="FAIL"
    fi
    echo -e "${arm_name}\t${arm_desc}\t${loops}\t${rope_scales}\t${solver}\t${rk_heads}\t${rk_blend}\t${rk_recur}\t${rk_hybrid_mix}\t${rk_battery}\t${model_params:-?}\t${raw_bpb:-?}\t${int6_sw_bpb:-?}\t${step_ms:-?}\t${bytes_total:-?}\t?\t${status}\t${log}" >> "${SUMMARY}"

    if [[ "${run_rc}" -ne 0 ]]; then
        if [[ "${STOP_ON_ERROR}" == "1" ]]; then
            echo "Stopping sweep due to failure in ${arm_name}. Set STOP_ON_ERROR=0 to continue."
            exit "${run_rc}"
        fi
        echo "Continuing sweep after failure in ${arm_name} because STOP_ON_ERROR=${STOP_ON_ERROR}."
    fi
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
