#!/bin/bash
set -euo pipefail
# Nightcrawler Cubed Folding — 3-deep linked flat recurrence
#
# Single variable vs run_10min_linked_flat_recur.sh:
#   LINKED_FLAT_RECUR_LAYERS=3
#
# This is the strong-form probe: run the full encoder tail recurrence
# inside each crawler loop while keeping the rest of the legal winner stack.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export COMPRESSOR="brotli"
export NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS:-7}"
export NUM_CRAWLER_LAYERS="${NUM_CRAWLER_LAYERS:-3}"
export CRAWLER_LOOPS="${CRAWLER_LOOPS:-3}"
export LINKED_FLAT_RECUR_LAYERS="${LINKED_FLAT_RECUR_LAYERS:-3}"

export SKIP_GPTQ="${SKIP_GPTQ:-0}"
export LOOP_AWARE_GPTQ="${LOOP_AWARE_GPTQ:-1}"
export GPTQ_CAL_SAMPLES="${GPTQ_CAL_SAMPLES:-256}"
export GPTQ_CAL_SEQ_LEN="${GPTQ_CAL_SEQ_LEN:-2048}"

export RUNTIME_PYMINIFY="${RUNTIME_PYMINIFY:-1}"
export PYMINIFY_MODE="${PYMINIFY_MODE:-aggressive}"

export SELECTIVE_PRUNE_ENABLE="${SELECTIVE_PRUNE_ENABLE:-1}"
export SIZE_TARGET_BYTES="${SIZE_TARGET_BYTES:-16000000}"
export SELECTIVE_PRUNE_FACTOR="${SELECTIVE_PRUNE_FACTOR:-8}"
export SELECTIVE_PRUNE_RESERVE_BYTES="${SELECTIVE_PRUNE_RESERVE_BYTES:-32768}"
export SELECTIVE_PRUNE_MAX_VALUES="${SELECTIVE_PRUNE_MAX_VALUES:-0}"

export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

exec bash "${SCRIPT_DIR}/run.sh"
