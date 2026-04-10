#!/bin/bash
set -euo pipefail
# Nightcrawler Cubed (7F+3C) — 4-hour production runner
#
# Thin wrapper over the canonical 7F+3C runner.
# Only schedule/seed differ from the 10-minute path.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export SEED="${SEED:-4}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-14400}"
export ITERATIONS="${ITERATIONS:-200000}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-12000}"

exec bash "${SCRIPT_DIR}/run.sh"
