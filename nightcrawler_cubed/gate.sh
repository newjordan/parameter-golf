#!/usr/bin/env bash
set -euo pipefail
# ================================================================
# Trapper Keeper 1 — safe vs aggressive gate
#
# A0: 8F+3C, 3 loops, ROPE=9,1,1        (safe — two vars vs leader)
# A1: 8F+3C, 4 loops, ROPE=9,3,1,1      (aggressive — three vars vs leader)
#
# 4xGPU, 600s wallclock (matches competition). Steps determined by throughput.
#
# Usage:
#   SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-09_Trapper_Keeper_1/gate.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-4}"
ITERATIONS="${ITERATIONS:-99999}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-300}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-393216}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}"
EVAL_STRIDE="${EVAL_STRIDE:-64}"

TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ----------------------------------------------------------------
# Tokenizer/data path auto-detection
# ----------------------------------------------------------------
if [[ -z "${TOKENIZER_PATH:-}" ]]; then
    for cand in \
        "${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model" \
        "/home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_1024_bpe.model" \
        "/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model" \
        "/workspace/parameter-golf-lab/data/tokenizers/fineweb_1024_bpe.model" \
        "/workspace/sota_nueral/data/tokenizers/fineweb_1024_bpe.model"; do
        if [[ -f "${cand}" ]]; then
            TOKENIZER_PATH="${cand}"
            break
        fi
    done
fi

if [[ -z "${DATA_PATH:-}" ]]; then
    for cand in \
        "${REPO_ROOT}/data/datasets/fineweb10B_sp1024" \
        "/home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp1024" \
        "/workspace/parameter-golf/data/datasets/fineweb10B_sp1024" \
        "/workspace/parameter-golf-lab/data/datasets/fineweb10B_sp1024" \
        "/workspace/sota_nueral/data/datasets/fineweb10B_sp1024"; do
        if [[ -d "${cand}" ]]; then
            DATA_PATH="${cand}"
            break
        fi
    done
fi

if [[ -z "${TOKENIZER_PATH:-}" || ! -f "${TOKENIZER_PATH}" ]]; then
    echo "ERROR: tokenizer not found." >&2; exit 1
fi
if [[ -z "${DATA_PATH:-}" || ! -d "${DATA_PATH}" ]]; then
    echo "ERROR: dataset dir not found." >&2; exit 1
fi

echo "tokenizer_path=${TOKENIZER_PATH}"
echo "data_path=${DATA_PATH}"

TORCHRUN=("${PYTHON_BIN}" -m torch.distributed.run)

CUDA_INFO="$("${PYTHON_BIN}" - <<'PY'
import sys, torch
print(f"exe={sys.executable}")
print(f"torch={torch.__version__}")
print(f"cuda_available={int(torch.cuda.is_available())}")
print(f"cuda_devices={torch.cuda.device_count()}")
PY
)"
echo "${CUDA_INFO}"

CUDA_DEVICES="$(sed -n 's/^cuda_devices=//p' <<<"${CUDA_INFO}")"
if [[ -n "${CUDA_DEVICES}" && "${CUDA_DEVICES}" =~ ^[0-9]+$ ]] && (( NPROC > CUDA_DEVICES )); then
    echo "WARN: clamping NPROC to ${CUDA_DEVICES}"
    NPROC="${CUDA_DEVICES}"
fi

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${RESULTS_DIR}/summary_gate_s${SEED}_${TS}.tsv"

extract_metric() {
    grep -oP "$1" "$2" | tail -1 || true
}

BASE_ENV=(
    TOKENIZER_PATH="${TOKENIZER_PATH}"
    DATA_PATH="${DATA_PATH}"
    SEED="${SEED}"
    ITERATIONS="${ITERATIONS}"
    MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}"
    WARMDOWN_ITERS="${ITERATIONS}"
    TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS}"
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE}"
    EVAL_STRIDE="${EVAL_STRIDE}"
    COMPLEMENT_ALPHA=0
    XSA_LAST_N=11
    BIGRAM_VOCAB_SIZE=2048
    ROPE_DIMS=16
    SWA_EVERY=50
    MTP_NUM_HEADS=0
    LATE_QAT_THRESHOLD=0
    MATRIX_LR=0.03
    TORCHDYNAMO_OPTIMIZE_DDP=0
    COMPILE_FULLGRAPH=1
    NGRAM_EVAL_ORDER=0
    MODEL_DIM=512
    USE_CRAWLER=1
    NUM_FLAT_LAYERS=8
    NUM_CRAWLER_LAYERS=3
    CRAWLER_MLP_MULT=6.0
    INST_DIM=32
    CRAWLER_QUANT_INT8=0
    DELTA_NET_HEADS=0
    SKIP_EMA=1
    SKIP_GPTQ=1
    LOOP_AWARE_GPTQ=0
    MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_CHOKE_DIM=0
    CRAWLER_MLP_CHOKE_SHAPE=flat
    CRAWLER_MLP_CHOKE_GROUPS=8
    CRAWLER_LOOP_SMEAR=0
    CRAWLER_TAP_DIM=0
    CRAWLER_TAP_LOOP_SPECIFIC=1
    CRAWLER_TAP_LAYERS=all
    ANCHOR_DIM=0
    FLAT_WEIGHT_SHARE=0
    NPROC_PER_NODE="${NPROC}"
)

echo -e "arm\tdesc\tloops\trope\tmodel_params\traw_bpb\tint6_sw_bpb\tstep_ms\tbytes\tlog" > "${SUMMARY}"

echo ""
echo "========================================"
echo "  Trapper Keeper 1 — safe vs aggressive"
echo "  4xGPU, ${ITERATIONS} steps, seed=${SEED}"
echo "========================================"

for ARM in A0 A1; do
    if [[ "${ARM}" == "A0" ]]; then
        LOOPS=3; ROPE="9,1,1"; DESC="8F+3C, 3 loops (safe)"
    else
        LOOPS=4; ROPE="9,3,1,1"; DESC="8F+3C, 4 loops diff battery (aggressive)"
    fi

    LOG="${RESULTS_DIR}/${ARM}_gate_s${SEED}_${TS}.log"

    echo ""
    echo "================================================================"
    echo "  ${ARM} : ${DESC}"
    echo "  CRAWLER_LOOPS=${LOOPS} ROPE=${ROPE}"
    echo "  seed=${SEED} GPUs=${NPROC} iterations=${ITERATIONS}"
    echo "  log: ${LOG}"
    echo "================================================================"

    env \
        "${BASE_ENV[@]}" \
        CRAWLER_LOOPS="${LOOPS}" \
        CRAWLER_LOOP_ROPE_SCALES="${ROPE}" \
        "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
        2>&1 | tee "${LOG}" || {
            echo "WARNING: ${ARM} failed. Continuing."
            echo -e "${ARM}\t${DESC}\t${LOOPS}\t${ROPE}\tFAILED\tFAILED\tFAILED\tFAILED\tFAILED\t${LOG}" >> "${SUMMARY}"
            continue
        }

    PARAMS="$(extract_metric 'model_params:\K[0-9]+' "${LOG}")"
    RAW="$(extract_metric 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}")"
    INT6="$(extract_metric 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}")"
    STEP_MS="$(extract_metric 'step_avg:\K[0-9.]+' "${LOG}")"
    BYTES="$(extract_metric 'Total submission size int6\+(?:zstd|zlib|brotli): \K[0-9]+' "${LOG}")"

    [[ -n "${PARAMS}" ]] || PARAMS="?"
    [[ -n "${RAW}" ]] || RAW="?"
    [[ -n "${INT6}" ]] || INT6="?"
    [[ -n "${STEP_MS}" ]] || STEP_MS="?"
    [[ -n "${BYTES}" ]] || BYTES="?"

    echo -e "${ARM}\t${DESC}\t${LOOPS}\t${ROPE}\t${PARAMS}\t${RAW}\t${INT6}\t${STEP_MS}\t${BYTES}\t${LOG}" >> "${SUMMARY}"
done

echo ""
echo "========================================"
echo "  Gate complete. Summary: ${SUMMARY}"
echo "========================================"
column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"
