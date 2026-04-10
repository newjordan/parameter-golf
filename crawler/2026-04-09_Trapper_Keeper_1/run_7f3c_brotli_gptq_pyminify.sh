#!/bin/bash
set -euo pipefail
# ================================================================
# Trapper Keeper 1 variant — 7F+3C + loop-aware GPTQ + python-minifier
#
# This stacks code-size trimming on top of run_7f3c_brotli_gptq.sh.
# Keep this as a separate runner so GPTQ-only and GPTQ+minify remain isolated.
#
# Usage:
#   bash crawler/2026-04-09_Trapper_Keeper_1/run_7f3c_brotli_gptq_pyminify.sh
#   PYMINIFY_MODE=safe bash crawler/2026-04-09_Trapper_Keeper_1/run_7f3c_brotli_gptq_pyminify.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUNTIME_PYMINIFY="1"
export PYMINIFY_MODE="${PYMINIFY_MODE:-aggressive}"  # safe|aggressive

exec bash "${SCRIPT_DIR}/run_7f3c_brotli_gptq.sh"
