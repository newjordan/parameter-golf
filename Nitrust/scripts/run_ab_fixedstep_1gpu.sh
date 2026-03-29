#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
ITERATIONS="${ITERATIONS:-3000}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-${ITERATIONS}}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-1024}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"

DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
AUTO_BOOTSTRAP_DATA="${AUTO_BOOTSTRAP_DATA:-1}"
BOOTSTRAP_VARIANT="${BOOTSTRAP_VARIANT:-sp1024}"
BOOTSTRAP_TRAIN_SHARDS="${BOOTSTRAP_TRAIN_SHARDS:-4}"

# Keep compile/cudagraph paths disabled for deterministic debug runs.
COMPILE_ENABLED="${COMPILE_ENABLED:-0}"
COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}"
TORCHINDUCTOR_USE_CUDAGRAPHS="${TORCHINDUCTOR_USE_CUDAGRAPHS:-0}"

SO_PATH="${NITRUST_SO_PATH:-${REPO_ROOT}/Nitrust/rust/target/release/libnitrust_py.so}"

if [ -f "${HOME}/.cargo/env" ]; then
  # shellcheck disable=SC1090
  source "${HOME}/.cargo/env"
fi

mkdir -p logs

if [ ! -f "${SO_PATH}" ]; then
  if ! command -v cargo >/dev/null 2>&1; then
    echo "fatal: missing ${SO_PATH} and cargo not found; install rustup/cargo first"
    exit 1
  fi
  echo "[ab] building nitrust rust bridge..."
  cargo build -p nitrust-py --release --manifest-path "${REPO_ROOT}/Nitrust/rust/Cargo.toml"
fi

if [ ! -f "${SO_PATH}" ]; then
  echo "fatal: nitrust shared object not found after build: ${SO_PATH}"
  exit 1
fi

ensure_data_and_tokenizer() {
  local train_glob="${DATA_PATH}/fineweb_train_*.bin"
  local val_glob="${DATA_PATH}/fineweb_val_*.bin"

  if [ -f "${TOKENIZER_PATH}" ] && compgen -G "${train_glob}" >/dev/null && compgen -G "${val_glob}" >/dev/null; then
    return
  fi
  if [ "${AUTO_BOOTSTRAP_DATA}" != "1" ]; then
    echo "fatal: missing tokenizer/shards and AUTO_BOOTSTRAP_DATA=${AUTO_BOOTSTRAP_DATA}"
    echo "missing tokenizer? $([ -f "${TOKENIZER_PATH}" ] && echo no || echo yes)"
    echo "train shards under ${DATA_PATH}: $(ls "${DATA_PATH}"/fineweb_train_*.bin 2>/dev/null | wc -l)"
    echo "val shards under ${DATA_PATH}: $(ls "${DATA_PATH}"/fineweb_val_*.bin 2>/dev/null | wc -l)"
    exit 1
  fi

  echo "[ab] bootstrapping tokenizer/dataset via data/cached_challenge_fineweb.py"
  if ! python3 -c "import huggingface_hub" >/dev/null 2>&1; then
    python3 -m pip install --quiet huggingface-hub
  fi
  python3 "${REPO_ROOT}/data/cached_challenge_fineweb.py" \
    --variant "${BOOTSTRAP_VARIANT}" \
    --train-shards "${BOOTSTRAP_TRAIN_SHARDS}"

  if [ ! -f "${TOKENIZER_PATH}" ] || ! compgen -G "${train_glob}" >/dev/null || ! compgen -G "${val_glob}" >/dev/null; then
    echo "fatal: bootstrap completed but tokenizer/shards still missing"
    echo "tokenizer: ${TOKENIZER_PATH}"
    echo "dataset: ${DATA_PATH}"
    exit 1
  fi
}

ensure_data_and_tokenizer

TS="$(date +%Y%m%d_%H%M%S)"
LOG_A="logs/ab_fixedstep_nitrust0_s${SEED}_${TS}.log"
LOG_B="logs/ab_fixedstep_nitrust1_s${SEED}_${TS}.log"

echo "============================================"
echo "  NITRUST FIXED-STEP A/B (1GPU)"
echo "  seed=${SEED} iterations=${ITERATIONS}"
echo "  train_batch_tokens=${TRAIN_BATCH_TOKENS} seq=${TRAIN_SEQ_LEN}"
echo "  data_path=${DATA_PATH}"
echo "  tokenizer_path=${TOKENIZER_PATH}"
echo "  compile_enabled=${COMPILE_ENABLED} fullgraph=${COMPILE_FULLGRAPH} cudagraphs=${TORCHINDUCTOR_USE_CUDAGRAPHS}"
echo "  nitrust_so=${SO_PATH}"
echo "============================================"

run_case() {
  local label="$1"
  local nitrust_enable="$2"
  local nitrust_strict="$3"
  local log_file="$4"

  echo "[ab] running ${label} -> ${log_file}"
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  COMPILE_ENABLED="${COMPILE_ENABLED}" \
  COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH}" \
  TORCHINDUCTOR_USE_CUDAGRAPHS="${TORCHINDUCTOR_USE_CUDAGRAPHS}" \
  SEED="${SEED}" \
  NITRUST_ENABLE="${nitrust_enable}" \
  NITRUST_STRICT="${nitrust_strict}" \
  NITRUST_SO_PATH="${SO_PATH}" \
  DATA_PATH="${DATA_PATH}" \
  TOKENIZER_PATH="${TOKENIZER_PATH}" \
  USE_CRAWLER=1 \
  NUM_FLAT_LAYERS=4 \
  NUM_CRAWLER_LAYERS=1 \
  CRAWLER_LOOPS=4 \
  INST_DIM=32 \
  DELTA_NET_HEADS=0 \
  SKIP_GPTQ=1 \
  LOOP_AWARE_GPTQ=0 \
  SKIP_EMA=1 \
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS}" \
  TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN}" \
  EVAL_SEQ_LEN="${EVAL_SEQ_LEN}" \
  ITERATIONS="${ITERATIONS}" \
  VAL_LOSS_EVERY="${VAL_LOSS_EVERY}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    experiments/Medusa/train_gpt.py \
    2>&1 | tee "${log_file}"
}

run_case "baseline_nitrust0" 0 0 "${LOG_A}"
run_case "nitrust1" 1 1 "${LOG_B}"

echo "============================================"
echo "  DONE"
echo "  baseline log: ${LOG_A}"
echo "  nitrust  log: ${LOG_B}"
echo "============================================"

echo "[ab] key lines"
for f in "${LOG_A}" "${LOG_B}"; do
  echo "==== ${f}"
  grep -nE 'nitrust:|step:3000/3000|val_bpb:|final_int6_roundtrip_exact|final_int6_sliding_window_exact|peak memory allocated' "${f}" || true
done
