#!/usr/bin/env bash
# =============================================================================
# Phase 1 — Async Dual-Hardware Launcher
# =============================================================================
# Launches experiments on BOTH H100 and DGX Spark simultaneously.
# Each hardware target runs independently in the background.
#
# Strategy:
#   - Spark runs fast/cheap sweeps first to identify promising directions
#   - H100 runs the same sweeps in parallel for ground-truth numbers
#   - Results from both are logged to the same results.jsonl with hardware tags
#   - After sweeps complete, compare rankings to validate transfer
#
# Usage:
#   # Run all Phase 1 experiments on both hardware targets
#   bash experiments/run_async.sh all
#
#   # Run a specific category on both
#   bash experiments/run_async.sh lr_matrix
#
#   # Run Spark in quick mode + H100 full
#   SPARK_QUICK=1 bash experiments/run_async.sh all
#
#   # View combined leaderboard
#   bash experiments/run_async.sh leaderboard
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CATEGORY="${1:-all}"
LOGS_DIR="$REPO_ROOT/experiments/logs"
mkdir -p "$LOGS_DIR"

if [ "$CATEGORY" == "leaderboard" ]; then
    python experiments/sweep.py --leaderboard --hardware h100
    exit 0
fi

SPARK_MODE="spark"
if [ "${SPARK_QUICK:-0}" == "1" ]; then
    SPARK_MODE="spark_quick"
fi

echo "============================================================"
echo "  Phase 1 — Async Dual-Hardware Launcher"
echo "============================================================"
echo "  Category:     $CATEGORY"
echo "  H100 mode:    8×H100 (official)"
echo "  Spark mode:   $SPARK_MODE"
echo "  Results file:  experiments/results.jsonl"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Launch H100 sweep in background
# ---------------------------------------------------------------------------
H100_LOG="$LOGS_DIR/h100_${CATEGORY}_$(date +%Y%m%d_%H%M%S).log"

echo "[H100] Starting sweep in background → $H100_LOG"
if [ "$CATEGORY" == "all" ]; then
    nohup python experiments/sweep.py --all --hardware h100 --resume \
        > "$H100_LOG" 2>&1 &
else
    nohup python experiments/sweep.py --category "$CATEGORY" --hardware h100 --resume \
        > "$H100_LOG" 2>&1 &
fi
H100_PID=$!
echo "[H100] PID: $H100_PID"

# ---------------------------------------------------------------------------
# Launch Spark sweep in background
# ---------------------------------------------------------------------------
SPARK_LOG="$LOGS_DIR/spark_${CATEGORY}_$(date +%Y%m%d_%H%M%S).log"

echo "[Spark] Starting sweep in background → $SPARK_LOG"
if [ "$CATEGORY" == "all" ]; then
    nohup python experiments/sweep.py --all --hardware "$SPARK_MODE" --resume \
        > "$SPARK_LOG" 2>&1 &
else
    nohup python experiments/sweep.py --category "$CATEGORY" --hardware "$SPARK_MODE" --resume \
        > "$SPARK_LOG" 2>&1 &
fi
SPARK_PID=$!
echo "[Spark] PID: $SPARK_PID"

echo ""
echo "Both sweeps running in background."
echo ""
echo "Monitor progress:"
echo "  tail -f $H100_LOG     # H100 progress"
echo "  tail -f $SPARK_LOG    # Spark progress"
echo ""
echo "Check status:"
echo "  ps -p $H100_PID,$SPARK_PID -o pid,etime,cmd"
echo ""
echo "View leaderboard:"
echo "  python experiments/sweep.py --leaderboard --hardware h100"
echo ""
echo "Stop all:"
echo "  kill $H100_PID $SPARK_PID"
echo ""

# Save PIDs for easy management
cat > "$LOGS_DIR/active_pids.txt" << EOF
H100_PID=$H100_PID
SPARK_PID=$SPARK_PID
CATEGORY=$CATEGORY
STARTED=$(date -Iseconds)
EOF

echo "PIDs saved to $LOGS_DIR/active_pids.txt"

# ---------------------------------------------------------------------------
# Optional: wait for both and show results
# ---------------------------------------------------------------------------
if [ "${WAIT:-0}" == "1" ]; then
    echo ""
    echo "WAIT=1 set, waiting for both sweeps to complete..."
    wait $H100_PID
    H100_EXIT=$?
    echo "[H100] Completed with exit code $H100_EXIT"

    wait $SPARK_PID
    SPARK_EXIT=$?
    echo "[Spark] Completed with exit code $SPARK_EXIT"

    echo ""
    echo "Final leaderboard:"
    python experiments/sweep.py --leaderboard --hardware h100
fi
