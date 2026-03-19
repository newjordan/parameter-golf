#!/usr/bin/env bash
# =============================================================================
# Phase 1 — H100 Experiment Runner
# =============================================================================
# Runs parameter-golf experiments on 8×H100 GPUs (official leaderboard config).
#
# Usage:
#   # Run baseline only
#   bash experiments/run_h100.sh baseline
#
#   # Run a single category
#   bash experiments/run_h100.sh lr_matrix
#
#   # Run all Phase 1 experiments (resume-safe)
#   bash experiments/run_h100.sh all
#
#   # Dry run to see what would execute
#   bash experiments/run_h100.sh all --dry-run
#
#   # Multi-seed validation of best config
#   bash experiments/run_h100.sh baseline --seeds 1337,42,7
#
# Prerequisites:
#   - 8×H100 GPUs available
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
echo "Phase 1 — H100 Experiment Runner"
echo "============================================"

# Check GPU count
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
echo "GPUs detected: $GPU_COUNT"
if [ "$GPU_COUNT" -lt 8 ]; then
    echo "WARNING: Expected 8 GPUs, found $GPU_COUNT. Adjusting nproc_per_node."
    NPROC=$GPU_COUNT
else
    NPROC=8
fi

# Check data
if [ ! -d "data/datasets/fineweb10B_sp1024" ]; then
    echo "ERROR: Training data not found. Run: python data/cached_challenge_fineweb.py"
    exit 1
fi

SHARD_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
echo "Training shards: $SHARD_COUNT"

VAL_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "Validation shards: $VAL_COUNT"

if [ ! -f "data/tokenizers/fineweb_1024_bpe.model" ]; then
    echo "ERROR: Tokenizer not found at data/tokenizers/fineweb_1024_bpe.model"
    exit 1
fi

echo "All pre-flight checks passed."
echo ""

# ---------------------------------------------------------------------------
# Dispatch to sweep.py
# ---------------------------------------------------------------------------
CATEGORY="${1:-baseline}"
shift || true

if [ "$CATEGORY" == "all" ]; then
    python experiments/sweep.py --all --hardware h100 --resume "$@"
elif [ "$CATEGORY" == "leaderboard" ]; then
    python experiments/sweep.py --leaderboard --hardware h100
else
    python experiments/sweep.py --category "$CATEGORY" --hardware h100 --resume "$@"
fi
