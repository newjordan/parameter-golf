#!/bin/bash
set -euo pipefail
# TK1 7F+3C submission push:
#   brotli + loop-aware GPTQ + runtime python-minifier + selective int6 prune
#
# Purpose:
#   Re-run the recovered 7F+3C near-miss with the same training/export path,
#   but enable the export-only selective_prune_int6 hook to recover the final
#   ~33KB needed for the 16,000,000-byte cap.
#
# Run:
#   SEED=444 NPROC_PER_NODE=8 \
#   bash crawler/2026-04-09_Trapper_Keeper_1/run_7f3c_brotli_gptq_pyminify_prune.sh
#
# Tuning knobs:
#   SELECTIVE_PRUNE_RESERVE_BYTES=32768   # extra headroom below 16MB target
#   SELECTIVE_PRUNE_FACTOR=8              # bytes->values heuristic multiplier
#   SELECTIVE_PRUNE_MAX_VALUES=0          # optional hard cap on values pruned

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUNTIME_PYMINIFY="1"
export PYMINIFY_MODE="${PYMINIFY_MODE:-aggressive}"

export SELECTIVE_PRUNE_ENABLE="${SELECTIVE_PRUNE_ENABLE:-1}"
export SIZE_TARGET_BYTES="${SIZE_TARGET_BYTES:-16000000}"
export SELECTIVE_PRUNE_FACTOR="${SELECTIVE_PRUNE_FACTOR:-8}"
export SELECTIVE_PRUNE_RESERVE_BYTES="${SELECTIVE_PRUNE_RESERVE_BYTES:-32768}"
export SELECTIVE_PRUNE_MAX_VALUES="${SELECTIVE_PRUNE_MAX_VALUES:-0}"

exec bash "${SCRIPT_DIR}/run_7f3c_brotli_gptq.sh"
