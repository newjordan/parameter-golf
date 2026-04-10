#!/bin/bash
set -euo pipefail
# ================================================================
# Trapper Keeper 1 variant — fixed 8F+2C, brotli export
#
# Purpose:
# - Keep run settings out of the shell command.
# - Reuse the validated Trapper_Keeper_1 runner flow.
#
# Fixed settings:
#   NUM_FLAT_LAYERS=8
#   NUM_CRAWLER_LAYERS=2
#   CRAWLER_LOOPS=3
#   COMPRESSOR=brotli
#
# Usage:
#   bash crawler/2026-04-09_Trapper_Keeper_1/run_8f2c_brotli.sh
#   SEED=300 bash crawler/2026-04-09_Trapper_Keeper_1/run_8f2c_brotli.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Seed/GPU are intentionally the only runtime knobs.
export SEED="${SEED:-444}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# Fixed experiment settings.
export COMPRESSOR="brotli"
export NUM_FLAT_LAYERS="8"
export NUM_CRAWLER_LAYERS="2"
export CRAWLER_LOOPS="3"

exec bash "${SCRIPT_DIR}/run.sh"
