#!/bin/bash
set -euo pipefail
# ================================================================
# Trapper Keeper 1 variant — fixed 7F+3C, brotli export
#
# Fixed settings:
#   NUM_FLAT_LAYERS=7
#   NUM_CRAWLER_LAYERS=3
#   CRAWLER_LOOPS=3
#   COMPRESSOR=brotli
#
# Usage:
#   bash crawler/2026-04-09_Trapper_Keeper_1/run_7f3c_brotli.sh
#   SEED=300 bash crawler/2026-04-09_Trapper_Keeper_1/run_7f3c_brotli.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export SEED="${SEED:-444}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export COMPRESSOR="brotli"
export NUM_FLAT_LAYERS="7"
export NUM_CRAWLER_LAYERS="3"
export CRAWLER_LOOPS="3"

exec bash "${SCRIPT_DIR}/run.sh"
