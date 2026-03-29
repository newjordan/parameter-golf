#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  JR-02 — Triton Candidate"
echo "  compile_mode=max-autotune fullgraph=0"
echo "  loader follows JR-01 winner unless overridden"
echo "============================================"

exec env \
    COMPILE_MODE="${COMPILE_MODE:-max-autotune}" \
    COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}" \
    bash "${SCRIPT_DIR}/run.sh"
