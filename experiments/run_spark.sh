#!/usr/bin/env bash
# =============================================================================
# Phase 1 — DGX Spark / H10 Experiment Runner
# =============================================================================
# Runs parameter-golf experiments on a single GPU (research proxy).
# Results are directional — relative ordering transfers to H100 but absolute
# BPB numbers will differ.
#
# Usage:
#   # Run baseline only
#   bash experiments/run_spark.sh baseline
#
#   # Run a single category
#   bash experiments/run_spark.sh lr_matrix
#
#   # Run all Phase 1 experiments (resume-safe)
#   bash experiments/run_spark.sh all
#
#   # Quick mode (10 min instead of 30 min per experiment)
#   bash experiments/run_spark.sh all --quick
#
#   # Dry run
#   bash experiments/run_spark.sh all --dry-run
#
# Prerequisites:
#   - 1× GPU (H10, A100, etc.)
#   - Data downloaded: python data/cached_challenge_fineweb.py
#   - Dependencies installed: pip install -r requirements.txt
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "============================================"
echo "Phase 1 — DGX Spark / H10 Experiment Runner"
echo "============================================"

# Check GPU
if ! nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. Checking for CUDA..."
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || {
        echo "ERROR: No GPU detected."
        exit 1
    }
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "GPU: $GPU_NAME ($GPU_MEM)"
fi

# Check data
if [ ! -d "data/datasets/fineweb10B_sp1024" ]; then
    echo "ERROR: Training data not found. Run: python data/cached_challenge_fineweb.py"
    exit 1
fi

echo "Data and tokenizer found."
echo ""

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
CATEGORY="${1:-baseline}"
shift || true

HARDWARE="spark"
EXTRA_ARGS=()

for arg in "$@"; do
    if [ "$arg" == "--quick" ]; then
        HARDWARE="spark_quick"
    else
        EXTRA_ARGS+=("$arg")
    fi
done

# ---------------------------------------------------------------------------
# Dispatch to sweep.py
# ---------------------------------------------------------------------------
if [ "$CATEGORY" == "all" ]; then
    python experiments/sweep.py --all --hardware "$HARDWARE" --resume "${EXTRA_ARGS[@]}"
elif [ "$CATEGORY" == "leaderboard" ]; then
    python experiments/sweep.py --leaderboard --hardware "$HARDWARE"
else
    python experiments/sweep.py --category "$CATEGORY" --hardware "$HARDWARE" --resume "${EXTRA_ARGS[@]}"
fi
