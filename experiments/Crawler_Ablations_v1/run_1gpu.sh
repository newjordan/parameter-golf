#!/bin/bash
set -euo pipefail
# CRAWLER_ABLATIONS_V1: Backend optimization ablation suite
# Tests BKD-03, BKD-06, BKD-09 (proxy), BKD-10 hypotheses
# All arms: 1×H100, 600s wallclock, crawler-only (DELTA_NET_HEADS=0)
# Baseline difference from Crawler_Leg_1: SKIP_GPTQ=0 (GPTQ enabled to measure quant gap)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

# ── Runtime knobs (overridable from environment) ───────────────────────────────
SEED="${SEED:-1337}"
NPROC="${NPROC_PER_NODE:-1}"
NITRUST_ENABLE="${NITRUST_ENABLE:-1}"
NITRUST_STRICT="${NITRUST_STRICT:-1}"
NITRUST_SO_PATH="${NITRUST_SO_PATH:-Nitrust/rust/target/release/libnitrust_py.so}"

RESULTS_DIR="experiments/Crawler_Ablations_v1/results"
LOGS_DIR="experiments/Crawler_Ablations_v1/logs"
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}" checkpoints

# ── Preflight checks ───────────────────────────────────────────────────────────
echo "[preflight] checking zstandard..."
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__} OK')" 2>/dev/null \
    || echo "  WARNING: zstandard not found"

echo "[preflight] checking flash_attn..."
python3 -c "
try:
    import flash_attn_interface; print('  FA3 (hopper) OK')
except ImportError:
    import flash_attn; v=flash_attn.__version__
    if v.startswith('3'): print(f'  FA3 v{v} OK')
    else: print(f'  WARNING: FA{v[0]} detected — want FA3')
" 2>/dev/null || echo "  WARNING: no flash_attn found"

if [ "${NITRUST_ENABLE}" = "1" ]; then
    if [ -f "${NITRUST_SO_PATH}" ]; then
        echo "[preflight] nitrust_py found: ${NITRUST_SO_PATH}"
    else
        if [ "${NITRUST_STRICT}" = "1" ]; then
            echo "[preflight] FATAL: NITRUST_ENABLE=1 but missing ${NITRUST_SO_PATH}"
            exit 1
        fi
        echo "[preflight] WARNING: missing ${NITRUST_SO_PATH}; run will fall back to Python path"
    fi
fi

echo "============================================"
echo "  CRAWLER_ABLATIONS_V1"
echo "  Seed: ${SEED}  GPUs: ${NPROC}  Wallclock: 600s/arm"
echo "  Arms: A_baseline B_loop_aware_gptq C_ema_on D_int8_off E_compile_fullgraph F_gptq_and_ema"
echo "  DELTA_NET_HEADS=0 (quarantined)  NGRAM_EVAL_ORDER=0"
echo "  NITRUST_ENABLE=${NITRUST_ENABLE} NITRUST_STRICT=${NITRUST_STRICT}"
echo "============================================"

# Track completed arms for summary
COMPLETED_ARMS=""
FAILED_ARMS=""

# ── run_arm ────────────────────────────────────────────────────────────────────
# Usage: run_arm <ARM_NAME> [KEY=VALUE ...]
# Every arm inherits the full base config; KEY=VALUE pairs override individual knobs.
run_arm() {
    local ARM_NAME="$1"
    shift
    local RUN_ID="cav1_${ARM_NAME}_$(date +%Y%m%d_%H%M%S)"
    local LOG_FILE="${LOGS_DIR}/${RUN_ID}.log"

    echo ""
    echo "================================================================"
    echo "  ARM ${ARM_NAME}"
    if [ "$#" -gt 0 ]; then
        echo "  Overrides: $*"
    else
        echo "  Overrides: (none — pure baseline)"
    fi
    echo "  RUN_ID=${RUN_ID}"
    echo "  Log: ${LOG_FILE}"
    echo "================================================================"
    echo ""

    mkdir -p "${RESULTS_DIR}/${RUN_ID}"

    # Build override env from remaining args (KEY=VALUE pairs passed as positional args)
    local -a OVERRIDES=()
    for kv in "$@"; do
        OVERRIDES+=("$kv")
    done

    # Run with full base config; OVERRIDES appended last so they win
    (
        env \
          SEED="${SEED}" \
          MAX_WALLCLOCK_SECONDS=600 \
          WARMDOWN_ITERS=2000 \
          COMPLEMENT_ALPHA=0 \
          XSA_LAST_N=11 \
          BIGRAM_VOCAB_SIZE=2048 \
          ROPE_DIMS=16 \
          SWA_EVERY=50 \
          MTP_NUM_HEADS=0 \
          LATE_QAT_THRESHOLD=0 \
          MATRIX_LR=0.03 \
          TORCHDYNAMO_OPTIMIZE_DDP=0 \
          COMPILE_FULLGRAPH=0 \
          NGRAM_EVAL_ORDER=0 \
          USE_CRAWLER=1 \
          NUM_FLAT_LAYERS=4 \
          NUM_CRAWLER_LAYERS=1 \
          CRAWLER_LOOPS=4 \
          CRAWLER_MLP_MULT=4.0 \
          INST_DIM=32 \
          CRAWLER_QUANT_INT8=1 \
          DELTA_NET_HEADS=0 \
          SKIP_EMA=1 \
          SKIP_GPTQ=0 \
          LOOP_AWARE_GPTQ=0 \
          NITRUST_ENABLE="${NITRUST_ENABLE}" \
          NITRUST_STRICT="${NITRUST_STRICT}" \
          NITRUST_SO_PATH="${NITRUST_SO_PATH}" \
          RUN_ID="${RUN_ID}" \
          DIAG_CSV_PATH="${RESULTS_DIR}/${RUN_ID}/diag.csv" \
          "${OVERRIDES[@]}" \
          torchrun --standalone --nproc_per_node="${NPROC}" \
              "${REPO_ROOT}/experiments/Medusa/train_gpt.py" \
          2>&1 | tee "${LOG_FILE}"
    ) || echo "ARM ${ARM_NAME} FAILED — see ${LOG_FILE}"

    # Checkpoint copies (soft-fail: missing file is not fatal)
    cp final_model.pt       "checkpoints/cav1_${ARM_NAME}_final.pt"       2>/dev/null || true
    cp final_model.int6.ptz "checkpoints/cav1_${ARM_NAME}_final.int6.ptz" 2>/dev/null || true

    echo "ARM ${ARM_NAME} done — diag: ${RESULTS_DIR}/${RUN_ID}/diag.csv"
}

# ── Six arms ───────────────────────────────────────────────────────────────────

# Arm A — pure baseline: GPTQ ON, all other defaults
run_arm "A_baseline" \
    && COMPLETED_ARMS="${COMPLETED_ARMS} A_baseline" \
    || FAILED_ARMS="${FAILED_ARMS} A_baseline"

# Arm B — loop-aware GPTQ: tests whether GPTQ quantizes across crawler loops
run_arm "B_loop_aware_gptq" \
    LOOP_AWARE_GPTQ=1 \
    && COMPLETED_ARMS="${COMPLETED_ARMS} B_loop_aware_gptq" \
    || FAILED_ARMS="${FAILED_ARMS} B_loop_aware_gptq"

# Arm C — EMA enabled: tests whether weight averaging helps crawler
run_arm "C_ema_on" \
    SKIP_EMA=0 \
    && COMPLETED_ARMS="${COMPLETED_ARMS} C_ema_on" \
    || FAILED_ARMS="${FAILED_ARMS} C_ema_on"

# Arm D — INT8 off: tests cost of int8 quantization in crawler MLP
run_arm "D_int8_off" \
    CRAWLER_QUANT_INT8=0 \
    && COMPLETED_ARMS="${COMPLETED_ARMS} D_int8_off" \
    || FAILED_ARMS="${FAILED_ARMS} D_int8_off"

# Arm E — compile fullgraph: tests torch.compile(fullgraph=True) on crawler
run_arm "E_compile_fullgraph" \
    COMPILE_FULLGRAPH=1 \
    && COMPLETED_ARMS="${COMPLETED_ARMS} E_compile_fullgraph" \
    || FAILED_ARMS="${FAILED_ARMS} E_compile_fullgraph"

# Arm F — loop-aware GPTQ + EMA combined: interaction test
run_arm "F_gptq_and_ema" \
    LOOP_AWARE_GPTQ=1 \
    SKIP_EMA=0 \
    && COMPLETED_ARMS="${COMPLETED_ARMS} F_gptq_and_ema" \
    || FAILED_ARMS="${FAILED_ARMS} F_gptq_and_ema"

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  CRAWLER_ABLATIONS_V1 — COMPLETE"
echo "================================================================"
echo "  Arms run:${COMPLETED_ARMS}"
if [ -n "${FAILED_ARMS}" ]; then
    echo "  Arms FAILED:${FAILED_ARMS}"
fi
echo "  Results: experiments/Crawler_Ablations_v1/results/"
echo "  Key metric: final_int6_sliding_window_exact (lower = better)"
echo "  Compare: B vs A (loop_aware_gptq effect)"
echo "           C vs A (ema_on effect)"
echo "           D vs A (int8_off effect)"
echo "           E vs A (compile_fullgraph effect)"
echo "           F vs A (combined effect)"
echo "================================================================"
