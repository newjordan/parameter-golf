#!/bin/bash
set -euo pipefail
# ================================================================
# Nightcrawler Cubed (7F+3C) — working crawler submission stack
#
# Copied from crawler/2026-04-09_Trapper_Keeper_1 after the first
# legal seed-444 result. This folder is the working submission sandbox.
#
# Usage:
#   SEED=444 NPROC_PER_NODE=8 bash nightcrawler_cubed/run.sh
#   SEED=300 NPROC_PER_NODE=8 bash nightcrawler_cubed/run.sh
#   SEED=4   NPROC_PER_NODE=8 bash nightcrawler_cubed/run.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"
TORCH_LIB="$(python3 - <<'PYEOF'
import os
try:
    import torch
except Exception:
    print("")
else:
    print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PYEOF
)"
if [[ -n "${TORCH_LIB}" && -d "${TORCH_LIB}" ]]; then
    export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
LEGAL_SIZE_LIMIT="${LEGAL_SIZE_LIMIT:-16000000}"
ENFORCE_SIZE_LIMIT="${ENFORCE_SIZE_LIMIT:-1}"

MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
WARMDOWN_ITERS="${WARMDOWN_ITERS:-2000}"
NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS:-7}"
NUM_CRAWLER_LAYERS="${NUM_CRAWLER_LAYERS:-3}"
CRAWLER_LOOPS="${CRAWLER_LOOPS:-3}"
CRAWLER_QUANT_INT8="${CRAWLER_QUANT_INT8:-0}"   # 0 keeps artifact size safer for 16MB cap
SKIP_GPTQ="${SKIP_GPTQ:-0}"
LOOP_AWARE_GPTQ="${LOOP_AWARE_GPTQ:-1}"
GPTQ_CAL_SAMPLES="${GPTQ_CAL_SAMPLES:-256}"
GPTQ_CAL_SEQ_LEN="${GPTQ_CAL_SEQ_LEN:-2048}"
RUNTIME_PYMINIFY="${RUNTIME_PYMINIFY:-1}"
PYMINIFY_MODE="${PYMINIFY_MODE:-aggressive}"    # safe|aggressive|aggressive_globals
SIZE_TARGET_BYTES="${SIZE_TARGET_BYTES:-${LEGAL_SIZE_LIMIT}}"
SELECTIVE_PRUNE_ENABLE="${SELECTIVE_PRUNE_ENABLE:-1}"
SELECTIVE_PRUNE_FACTOR="${SELECTIVE_PRUNE_FACTOR:-8}"
SELECTIVE_PRUNE_RESERVE_BYTES="${SELECTIVE_PRUNE_RESERVE_BYTES:-32768}"
SELECTIVE_PRUNE_MAX_VALUES="${SELECTIVE_PRUNE_MAX_VALUES:-0}"
PRESERVE_SEED_ALIAS="${PRESERVE_SEED_ALIAS:-1}"

RESULTS_DIR="${SCRIPT_DIR}/results"
BACKUP_DIR="${RESULTS_DIR}/backups"
mkdir -p "${SCRIPT_DIR}/logs" "${RESULTS_DIR}" "${BACKUP_DIR}"
LOG_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_TS="${SCRIPT_DIR}/logs/train_seed${SEED}_${LOG_STAMP}.log"
LOG_RESULTS="${RESULTS_DIR}/train_seed${SEED}_${LOG_STAMP}.log"
LOG_ALIAS="${SCRIPT_DIR}/train_seed${SEED}.log"
METRICS_TS="${RESULTS_DIR}/metrics_seed${SEED}_${LOG_STAMP}.tsv"
METRICS_ALIAS="${SCRIPT_DIR}/metrics_seed${SEED}.tsv"

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
else
    TORCHRUN=(python3 -m torch.distributed.run)
fi

# ----------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------
echo "[preflight] checking zstandard..."
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__} OK')" 2>/dev/null \
    || { echo "  ERROR: zstandard missing (pip install zstandard)"; exit 1; }

echo "[preflight] checking flash_attn..."
python3 - <<'PY'
try:
    import flash_attn_interface  # type: ignore
    print("  FA3 (hopper) OK")
except Exception:
    try:
        import flash_attn  # type: ignore
        v = flash_attn.__version__
        if str(v).startswith("3"):
            print(f"  FA3 v{v} OK")
        else:
            print(f"  WARNING: flash-attn v{v} detected (want v3)")
    except Exception:
        raise SystemExit("  ERROR: flash-attn not importable")
PY

echo "[preflight] checking dataset + tokenizer..."
python3 - <<'PY'
import glob, os
tok = "./data/tokenizers/fineweb_1024_bpe.model"
assert os.path.isfile(tok), f"missing tokenizer: {tok}"
shards = glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin")
assert len(shards) >= 8, f"need >=8 train shards, found {len(shards)}"
print(f"  tokenizer OK, train shards={len(shards)}")
PY

TRAIN_PY_RUN="${TRAIN_PY}"
if [[ "${RUNTIME_PYMINIFY}" == "1" ]]; then
    TRIMMED_PY="${SCRIPT_DIR}/logs/train_gpt_seed${SEED}_${LOG_STAMP}.min.py"
    echo "[preflight] runtime python-minifier..."
    python3 - "${TRAIN_PY}" "${TRIMMED_PY}" "${PYMINIFY_MODE}" <<'PY'
import pathlib
import sys

src_path = pathlib.Path(sys.argv[1])
dst_path = pathlib.Path(sys.argv[2])
mode = sys.argv[3].strip().lower()

try:
    import python_minifier
except Exception as exc:
    raise SystemExit(
        f"  ERROR: python-minifier not importable ({exc}). "
        "Install with: python3 -m pip install --user python-minifier"
    )

source = src_path.read_text(encoding="utf-8")
if mode == "safe":
    options = dict(
        remove_literal_statements=True,
        remove_asserts=True,
        remove_debug=True,
        rename_locals=False,
        rename_globals=False,
        hoist_literals=True,
    )
elif mode == "aggressive":
    options = dict(
        remove_literal_statements=True,
        remove_asserts=True,
        remove_debug=True,
        rename_locals=True,
        rename_globals=False,
        hoist_literals=True,
    )
elif mode == "aggressive_globals":
    options = dict(
        remove_literal_statements=True,
        remove_asserts=True,
        remove_debug=True,
        rename_locals=True,
        rename_globals=True,
        hoist_literals=True,
    )
else:
    raise SystemExit(f"  ERROR: PYMINIFY_MODE must be safe|aggressive|aggressive_globals, got: {mode}")

minified = python_minifier.minify(source, **options)
compile(minified, str(dst_path), "exec")
dst_path.write_text(minified + "\n", encoding="utf-8")
print(
    f"  pyminify mode={mode} orig_bytes={len(source.encode('utf-8'))} "
    f"min_bytes={len(minified.encode('utf-8'))} out={dst_path}"
)
PY
    TRAIN_PY_RUN="${TRIMMED_PY}"
fi

echo ""
echo "============================================"
echo "  Nightcrawler Cubed (7F+3C) — full run"
echo "  seed=${SEED} GPUs=${NPROC} wallclock=${MAX_WALLCLOCK_SECONDS}s"
echo "  NUM_FLAT_LAYERS=${NUM_FLAT_LAYERS} NUM_CRAWLER_LAYERS=${NUM_CRAWLER_LAYERS} CRAWLER_LOOPS=${CRAWLER_LOOPS}"
echo "  CRAWLER_QUANT_INT8=${CRAWLER_QUANT_INT8}  (0=smaller artifacts, 1=higher risk for >16MB)"
echo "  SKIP_GPTQ=${SKIP_GPTQ} LOOP_AWARE_GPTQ=${LOOP_AWARE_GPTQ} GPTQ_CAL_SAMPLES=${GPTQ_CAL_SAMPLES}"
echo "  RUNTIME_PYMINIFY=${RUNTIME_PYMINIFY} PYMINIFY_MODE=${PYMINIFY_MODE}"
echo "  SIZE_TARGET_BYTES=${SIZE_TARGET_BYTES} SELECTIVE_PRUNE_ENABLE=${SELECTIVE_PRUNE_ENABLE} SELECTIVE_PRUNE_FACTOR=${SELECTIVE_PRUNE_FACTOR}"
echo "  PRESERVE_SEED_ALIAS=${PRESERVE_SEED_ALIAS}"
echo "  train_py=${TRAIN_PY_RUN}"
echo "  log: ${LOG_TS}"
echo "============================================"
echo ""

env \
    SEED="${SEED}" \
    MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
    WARMDOWN_ITERS="${WARMDOWN_ITERS}" \
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
    NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS}" \
    NUM_CRAWLER_LAYERS="${NUM_CRAWLER_LAYERS}" \
    CRAWLER_LOOPS="${CRAWLER_LOOPS}" \
    CRAWLER_MLP_MULT=6.0 \
    INST_DIM=32 \
    CRAWLER_QUANT_INT8="${CRAWLER_QUANT_INT8}" \
    DELTA_NET_HEADS=0 \
    SKIP_EMA=1 \
    SKIP_GPTQ="${SKIP_GPTQ}" \
    LOOP_AWARE_GPTQ="${LOOP_AWARE_GPTQ}" \
    GPTQ_CAL_SAMPLES="${GPTQ_CAL_SAMPLES}" \
    GPTQ_CAL_SEQ_LEN="${GPTQ_CAL_SEQ_LEN}" \
    SIZE_TARGET_BYTES="${SIZE_TARGET_BYTES}" \
    SELECTIVE_PRUNE_ENABLE="${SELECTIVE_PRUNE_ENABLE}" \
    SELECTIVE_PRUNE_FACTOR="${SELECTIVE_PRUNE_FACTOR}" \
    SELECTIVE_PRUNE_RESERVE_BYTES="${SELECTIVE_PRUNE_RESERVE_BYTES}" \
    SELECTIVE_PRUNE_MAX_VALUES="${SELECTIVE_PRUNE_MAX_VALUES}" \
    MLP_LEAKY_SLOPE=0.5 \
    CRAWLER_MLP_LEAKY_SLOPE=0.5 \
    CRAWLER_MLP_CHOKE_DIM=0 \
    CRAWLER_MLP_CHOKE_SHAPE=flat \
    CRAWLER_MLP_CHOKE_GROUPS=8 \
    CRAWLER_LOOP_ROPE_SCALES=9,1,1 \
    CRAWLER_LOOP_SMEAR=0 \
    CRAWLER_TAP_DIM=0 \
    CRAWLER_TAP_LOOP_SPECIFIC=1 \
    CRAWLER_TAP_LAYERS=all \
    ANCHOR_DIM=0 \
    FLAT_WEIGHT_SHARE=0 \
    NPROC_PER_NODE="${NPROC}" \
    "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY_RUN}" \
    2>&1 | tee "${LOG_TS}"

cp -f "${LOG_TS}" "${LOG_RESULTS}"
if [[ "${PRESERVE_SEED_ALIAS}" == "1" ]]; then
    if [[ -f "${LOG_ALIAS}" ]]; then
        PREV_LOG_STAMP="$(date -r "${LOG_ALIAS}" +%Y%m%d_%H%M%S 2>/dev/null || echo unknown)"
        cp -f "${LOG_ALIAS}" "${BACKUP_DIR}/train_seed${SEED}_${PREV_LOG_STAMP}_${LOG_STAMP}_prev.log"
    fi
    cp -f "${LOG_TS}" "${LOG_ALIAS}"
fi

# ----------------------------------------------------------------
# Metrics extraction
# ----------------------------------------------------------------
raw_bpb="$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG_TS}" | tail -1 || true)"
int6_sw_bpb="$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG_TS}" | tail -1 || true)"
bytes_total="$(grep -oP 'Total submission size int6\+(?:zstd|zlib|brotli): \K[0-9]+' "${LOG_TS}" | tail -1 || true)"
code_bytes="$(grep -oP 'Code size: \K[0-9]+' "${LOG_TS}" | tail -1 || true)"
step_ms="$(grep -oP 'step_avg:\K[0-9.]+' "${LOG_TS}" | tail -1 || true)"
model_params="$(grep -oP 'model_params:\K[0-9]+' "${LOG_TS}" | tail -1 || true)"
steps="$(grep -oP 'stopping_early:.*step:\K[0-9]+' "${LOG_TS}" | tail -1 || true)"
if [[ -z "${steps}" ]]; then
    steps="$(grep -oP 'step:\K[0-9]+(?=/[0-9]+ val_loss:)' "${LOG_TS}" | tail -1 || true)"
fi
train_time_ms="$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:[0-9.]+ train_time:\K[0-9]+' "${LOG_TS}" | tail -1 || true)"
if [[ -n "${train_time_ms}" ]]; then
    train_time_s=$((train_time_ms / 1000))
else
    train_time_s="${MAX_WALLCLOCK_SECONDS}"
fi

artifact_ok="unknown"
if [[ -n "${bytes_total}" && "${bytes_total}" =~ ^[0-9]+$ ]]; then
    if (( bytes_total <= LEGAL_SIZE_LIMIT )); then
        artifact_ok="yes"
    else
        artifact_ok="no"
    fi
fi

echo ""
echo "============================================"
echo "  RESULT — Nightcrawler Cubed (7F+3C) seed=${SEED}"
echo "  model_params:  ${model_params:-?}"
echo "  raw_bpb:       ${raw_bpb:-?}"
echo "  int6_sw_bpb:   ${int6_sw_bpb:-?}"
echo "  step_avg_ms:   ${step_ms:-?}"
echo "  steps:         ${steps:-?}"
echo "  train_time_s:  ${train_time_s}"
echo "  bytes_total:   ${bytes_total:-?}  (limit ${LEGAL_SIZE_LIMIT})"
echo "  bytes_code:    ${code_bytes:-?}"
echo "  artifact_legal:${artifact_ok}"
echo "  log:           ${LOG_RESULTS}"
echo "============================================"

{
    echo -e "seed\tmodel_params\traw_bpb\tint6_sw_bpb\tsteps\tstep_ms\ttrain_time_s\tbytes_total\tbytes_code\tartifact_legal\tlog"
    echo -e "${SEED}\t${model_params:-?}\t${raw_bpb:-?}\t${int6_sw_bpb:-?}\t${steps:-?}\t${step_ms:-?}\t${train_time_s}\t${bytes_total:-?}\t${code_bytes:-?}\t${artifact_ok}\t${LOG_RESULTS}"
} > "${METRICS_TS}"
if [[ "${PRESERVE_SEED_ALIAS}" == "1" ]]; then
    if [[ -f "${METRICS_ALIAS}" ]]; then
        PREV_METRICS_STAMP="$(date -r "${METRICS_ALIAS}" +%Y%m%d_%H%M%S 2>/dev/null || echo unknown)"
        cp -f "${METRICS_ALIAS}" "${BACKUP_DIR}/metrics_seed${SEED}_${PREV_METRICS_STAMP}_${LOG_STAMP}_prev.tsv"
    fi
    cp -f "${METRICS_TS}" "${METRICS_ALIAS}"
fi

# Keep uniquely named artifacts for submission packaging.
if [[ -f "${REPO_ROOT}/final_model.pt" ]]; then
    ARTIFACT_ALIAS="${SCRIPT_DIR}/final_model_seed${SEED}.pt"
    ARTIFACT_TS="${SCRIPT_DIR}/final_model_seed${SEED}_${LOG_STAMP}.pt"
    cp -f "${REPO_ROOT}/final_model.pt" "${ARTIFACT_TS}"
    if [[ "${PRESERVE_SEED_ALIAS}" == "1" ]]; then
        if [[ -f "${ARTIFACT_ALIAS}" ]]; then
            PREV_ARTIFACT_STAMP="$(date -r "${ARTIFACT_ALIAS}" +%Y%m%d_%H%M%S 2>/dev/null || echo unknown)"
            cp -f "${ARTIFACT_ALIAS}" "${BACKUP_DIR}/final_model_seed${SEED}_${PREV_ARTIFACT_STAMP}_${LOG_STAMP}_prev.pt"
        fi
        cp -f "${REPO_ROOT}/final_model.pt" "${ARTIFACT_ALIAS}"
    fi
fi
if [[ -f "${REPO_ROOT}/final_model.int6.ptz" ]]; then
    ARTIFACT_ALIAS="${SCRIPT_DIR}/final_model_seed${SEED}.int6.ptz"
    ARTIFACT_TS="${SCRIPT_DIR}/final_model_seed${SEED}_${LOG_STAMP}.int6.ptz"
    cp -f "${REPO_ROOT}/final_model.int6.ptz" "${ARTIFACT_TS}"
    if [[ "${PRESERVE_SEED_ALIAS}" == "1" ]]; then
        if [[ -f "${ARTIFACT_ALIAS}" ]]; then
            PREV_ARTIFACT_STAMP="$(date -r "${ARTIFACT_ALIAS}" +%Y%m%d_%H%M%S 2>/dev/null || echo unknown)"
            cp -f "${ARTIFACT_ALIAS}" "${BACKUP_DIR}/final_model_seed${SEED}_${PREV_ARTIFACT_STAMP}_${LOG_STAMP}_prev.int6.ptz"
        fi
        cp -f "${REPO_ROOT}/final_model.int6.ptz" "${ARTIFACT_ALIAS}"
    fi
fi
if [[ -f "${REPO_ROOT}/final_model.int8.ptz" ]]; then
    ARTIFACT_ALIAS="${SCRIPT_DIR}/final_model_seed${SEED}.int8.ptz"
    ARTIFACT_TS="${SCRIPT_DIR}/final_model_seed${SEED}_${LOG_STAMP}.int8.ptz"
    cp -f "${REPO_ROOT}/final_model.int8.ptz" "${ARTIFACT_TS}"
    if [[ "${PRESERVE_SEED_ALIAS}" == "1" ]]; then
        if [[ -f "${ARTIFACT_ALIAS}" ]]; then
            PREV_ARTIFACT_STAMP="$(date -r "${ARTIFACT_ALIAS}" +%Y%m%d_%H%M%S 2>/dev/null || echo unknown)"
            cp -f "${ARTIFACT_ALIAS}" "${BACKUP_DIR}/final_model_seed${SEED}_${PREV_ARTIFACT_STAMP}_${LOG_STAMP}_prev.int8.ptz"
        fi
        cp -f "${REPO_ROOT}/final_model.int8.ptz" "${ARTIFACT_ALIAS}"
    fi
fi

if [[ "${ENFORCE_SIZE_LIMIT}" == "1" && "${artifact_ok}" == "no" ]]; then
    echo "ERROR: artifact exceeds ${LEGAL_SIZE_LIMIT} bytes. Re-run with smaller config or set ENFORCE_SIZE_LIMIT=0."
    exit 2
fi

exit 0
