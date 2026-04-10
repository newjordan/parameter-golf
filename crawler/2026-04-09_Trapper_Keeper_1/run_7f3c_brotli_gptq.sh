#!/bin/bash
set -euo pipefail
# ================================================================
# Trapper Keeper 1 variant — fixed 7F+3C, brotli export, loop-aware GPTQ
#
# Fixed settings:
#   NUM_FLAT_LAYERS=7
#   NUM_CRAWLER_LAYERS=3
#   CRAWLER_LOOPS=3
#   COMPRESSOR=brotli
#   SKIP_GPTQ=0
#   LOOP_AWARE_GPTQ=1
#
# Usage:
#   bash crawler/2026-04-09_Trapper_Keeper_1/run_7f3c_brotli_gptq.sh
#   SEED=300 bash crawler/2026-04-09_Trapper_Keeper_1/run_7f3c_brotli_gptq.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export SEED="${SEED:-444}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export COMPRESSOR="brotli"
export NUM_FLAT_LAYERS="7"
export NUM_CRAWLER_LAYERS="3"
export CRAWLER_LOOPS="3"

export SKIP_GPTQ="0"
export LOOP_AWARE_GPTQ="1"
export GPTQ_CAL_SAMPLES="${GPTQ_CAL_SAMPLES:-256}"
export GPTQ_CAL_SEQ_LEN="${GPTQ_CAL_SEQ_LEN:-2048}"

exec bash "${SCRIPT_DIR}/run.sh"
