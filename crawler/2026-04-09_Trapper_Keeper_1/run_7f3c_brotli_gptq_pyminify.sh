#!/bin/bash
set -euo pipefail
# TK1 7F+3C + loop-aware GPTQ + python-minifier (code trim)
#
# Install once (separate command):
#   python3 -m pip install --user python-minifier
#
# Run:
#   SEED=444 NPROC_PER_NODE=8 ENFORCE_SIZE_LIMIT=0 PYMINIFY_MODE=aggressive \
#   bash crawler/2026-04-09_Trapper_Keeper_1/run_7f3c_brotli_gptq_pyminify.sh
#
# Optional:
#   PYMINIFY_MODE=safe

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUNTIME_PYMINIFY="1"
export PYMINIFY_MODE="${PYMINIFY_MODE:-aggressive}"  # safe|aggressive

exec bash "${SCRIPT_DIR}/run_7f3c_brotli_gptq.sh"
