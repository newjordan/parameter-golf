#!/usr/bin/env bash
set -euo pipefail
# Legacy script name kept for compatibility.
# Default run target is crawler-only leg (`experiments/Crawler_Leg_1/run.sh`).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SO_PATH="${SO_PATH:-${REPO_ROOT}/Nitrust/rust/target/release/libnitrust_py.so}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SEED="${SEED:-1337}"
NITRUST_STRICT="${NITRUST_STRICT:-1}"
RUN_SCRIPT="${RUN_SCRIPT:-${REPO_ROOT}/experiments/Crawler_Leg_1/run.sh}"

"${SCRIPT_DIR}/medusa_nitrust_preflight.sh"

cd "${REPO_ROOT}"
NITRUST_ENABLE=1 \
NITRUST_STRICT="${NITRUST_STRICT}" \
NITRUST_SO_PATH="${SO_PATH}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
SEED="${SEED}" \
bash "${RUN_SCRIPT}"
