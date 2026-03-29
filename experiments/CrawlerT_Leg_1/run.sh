#!/bin/bash
set -euo pipefail
# CRAWLERT_LEG_1: Triton kernel fusion speed test
#
# Tests whether torch.compile(mode='max-autotune') on the full model
# gives enough step-time improvement to improve BPB at 600s wall-clock.
#
# Protocol:
# 1. Run bench.py FIRST on a single GPU to confirm signal (≥10% speedup).
# 2. If confirmed: TRITON_FUSE=1 bash run.sh
# 3. Compare final BPB in logs vs. Crawler_Leg_1 baseline (same seed).
#
# TRITON_FUSE=0  →  identical to Crawler_Leg_1 (baseline)
# TRITON_FUSE=1  →  compile with mode='max-autotune' (the test)
#
# All other vars are locked to Crawler_Leg_1 values for clean A/B.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NITRUST_ENABLE="${NITRUST_ENABLE:-1}"
NITRUST_STRICT="${NITRUST_STRICT:-1}"
NITRUST_SO_PATH="${NITRUST_SO_PATH:-Nitrust/rust/target/release/libnitrust_py.so}"

# Locked to Crawler_Leg_1 values — do not change for this A/B
NUM_FLAT_LAYERS=4
NUM_CRAWLER_LAYERS=1
CRAWLER_LOOPS=4
INST_DIM=32
CRAWLER_QUANT_INT8=1
CRAWLER_MLP_MULT=4.0

# The one variable under test
TRITON_FUSE="${TRITON_FUSE:-0}"

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

if [ "${TRITON_FUSE}" = "1" ]; then
    echo "[preflight] TRITON_FUSE=1 — compile mode: max-autotune"
    echo "            NOTE: first ~100 steps will be slower (Triton autotuning cache warm-up)"
else
    echo "[preflight] TRITON_FUSE=0 — compile mode: default (baseline)"
fi

echo "============================================"
echo "  CRAWLERT_LEG_1"
echo "  Seed: ${SEED}"
echo "  flat=${NUM_FLAT_LAYERS} crawler_layers=${NUM_CRAWLER_LAYERS} loops=${CRAWLER_LOOPS}"
echo "  inst_dim=${INST_DIM} crawler_mlp_mult=${CRAWLER_MLP_MULT}"
echo "  TRITON_FUSE=${TRITON_FUSE}"
echo "  NITRUST_ENABLE=${NITRUST_ENABLE}"
echo "============================================"

SEED="${SEED}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
WARMDOWN_ITERS="${WARMDOWN_ITERS:-2000}" \
COMPLEMENT_ALPHA=0 \
XSA_LAST_N="${XSA_LAST_N:-11}" \
BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2048}" \
ROPE_DIMS="${ROPE_DIMS:-16}" \
SWA_EVERY="${SWA_EVERY:-50}" \
MTP_NUM_HEADS=0 \
LATE_QAT_THRESHOLD=0 \
MATRIX_LR="${MATRIX_LR:-0.03}" \
TORCHDYNAMO_OPTIMIZE_DDP="${TORCHDYNAMO_OPTIMIZE_DDP:-0}" \
COMPILE_FULLGRAPH=0 \
COMPILE_MODE="$([ "${TRITON_FUSE}" = "1" ] && echo "max-autotune" || echo "")" \
NGRAM_EVAL_ORDER=0 \
USE_CRAWLER=1 \
NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS}" \
NUM_CRAWLER_LAYERS="${NUM_CRAWLER_LAYERS}" \
CRAWLER_LOOPS="${CRAWLER_LOOPS}" \
CRAWLER_MLP_MULT="${CRAWLER_MLP_MULT}" \
INST_DIM="${INST_DIM}" \
CRAWLER_QUANT_INT8="${CRAWLER_QUANT_INT8}" \
DELTA_NET_HEADS=0 \
SKIP_EMA=1 \
SKIP_GPTQ=1 \
LOOP_AWARE_GPTQ=0 \
NITRUST_ENABLE="${NITRUST_ENABLE}" \
NITRUST_STRICT="${NITRUST_STRICT}" \
NITRUST_SO_PATH="${NITRUST_SO_PATH}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/crawlert_leg1_fuse${TRITON_FUSE}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
