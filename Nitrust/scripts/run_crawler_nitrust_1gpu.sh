#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

MODE="${MODE:-full}" # smoke | full | both
SEED="${SEED:-1337}"
SO_PATH="${SO_PATH:-${REPO_ROOT}/Nitrust/rust/target/release/libnitrust_py.so}"
NITRUST_ENABLE="${NITRUST_ENABLE:-1}"
NITRUST_STRICT="${NITRUST_STRICT:-1}"
NITRUST_LOCAL_SPAN="${NITRUST_LOCAL_SPAN:-1}"
TRITON_CRAWLER_FLOW="${TRITON_CRAWLER_FLOW:-0}"
TRITON_CRAWLER_FLOW_STRICT="${TRITON_CRAWLER_FLOW_STRICT:-0}"

echo "============================================"
echo "  NITRUST CRAWLER SINGLE-GPU PACKAGE"
echo "  mode=${MODE} seed=${SEED}"
echo "  nitrust_enable=${NITRUST_ENABLE} nitrust_strict=${NITRUST_STRICT}"
echo "  nitrust_local_span=${NITRUST_LOCAL_SPAN}"
echo "  triton_crawler_flow=${TRITON_CRAWLER_FLOW}"
echo "============================================"

run_smoke() {
  echo "[1gpu] running crawler smoke harness..."
  SEED="${SEED}" \
  SO_PATH="${SO_PATH}" \
  NITRUST_ENABLE="${NITRUST_ENABLE}" \
  NITRUST_STRICT="${NITRUST_STRICT}" \
  NITRUST_LOCAL_SPAN="${NITRUST_LOCAL_SPAN}" \
  TRITON_CRAWLER_FLOW="${TRITON_CRAWLER_FLOW}" \
  TRITON_CRAWLER_FLOW_STRICT="${TRITON_CRAWLER_FLOW_STRICT}" \
  bash "${SCRIPT_DIR}/spark_crawler_leg1_smoke.sh"
}

run_full() {
  echo "[1gpu] running nitrust preflight..."
  "${SCRIPT_DIR}/medusa_nitrust_preflight.sh"

  echo "[1gpu] launching full crawler leg (nproc=1)..."
  cd "${REPO_ROOT}"
  SEED="${SEED}" \
  NPROC_PER_NODE=1 \
  NITRUST_ENABLE="${NITRUST_ENABLE}" \
  NITRUST_STRICT="${NITRUST_STRICT}" \
  NITRUST_SO_PATH="${SO_PATH}" \
  NITRUST_LOCAL_SPAN="${NITRUST_LOCAL_SPAN}" \
  TRITON_CRAWLER_FLOW="${TRITON_CRAWLER_FLOW}" \
  TRITON_CRAWLER_FLOW_STRICT="${TRITON_CRAWLER_FLOW_STRICT}" \
  bash "${REPO_ROOT}/experiments/Crawler_Leg_1/run.sh"
}

case "${MODE}" in
  smoke)
    run_smoke
    ;;
  full)
    run_full
    ;;
  both)
    run_smoke
    run_full
    ;;
  *)
    echo "invalid MODE=${MODE} (expected: smoke|full|both)"
    exit 2
    ;;
esac
