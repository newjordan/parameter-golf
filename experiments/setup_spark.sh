#!/usr/bin/env bash
# =============================================================================
# DGX Spark / H10 Setup — Run this once on the Spark machine
# =============================================================================
#
# Usage:
#   bash experiments/setup_spark.sh
#
# Same as setup_h100.sh but for single-GPU research proxy.
# =============================================================================

set -euo pipefail

BRANCH="claude/optimize-train-baseline-M8rn2"
REPO_URL="${REPO_URL:-git@github.com:newjordan/parameter-golf.git}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/parameter-golf}"

echo "============================================================"
echo "  DGX Spark / H10 Setup — Phase 1 Parameter Golf"
echo "============================================================"
echo "  Branch:      $BRANCH"
echo "  Install dir: $INSTALL_DIR"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Clone repo
# ---------------------------------------------------------------------------
echo "[1/5] Cloning repository..."
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "  Repo already exists, fetching latest..."
    cd "$INSTALL_DIR"
    git fetch origin "$BRANCH"
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    git checkout "$BRANCH"
fi
echo "  Done. HEAD: $(git log --oneline -1)"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Python environment
# ---------------------------------------------------------------------------
echo "[2/5] Setting up Python environment..."
if [ ! -d "$INSTALL_DIR/.venv" ]; then
    python3 -m venv "$INSTALL_DIR/.venv"
fi
source "$INSTALL_DIR/.venv/bin/activate"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Download data
# ---------------------------------------------------------------------------
echo "[3/5] Downloading data..."
if [ -d "data/datasets/fineweb10B_sp1024" ] && \
   [ "$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  Data already present. Skipping."
else
    python data/cached_challenge_fineweb.py
fi
echo ""

# ---------------------------------------------------------------------------
# Step 4: Git config
# ---------------------------------------------------------------------------
echo "[4/5] Configuring git..."
if ! git config user.email &>/dev/null; then
    git config user.email "parameter-golf-bot@experiment"
    git config user.name "Parameter Golf Spark"
fi
echo "  Git user: $(git config user.name)"
echo ""

# ---------------------------------------------------------------------------
# Step 5: Sanity check (single GPU, 2 steps)
# ---------------------------------------------------------------------------
echo "[5/5] Sanity check..."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
echo "  GPU: $GPU_NAME ($GPU_MEM)"

ITERATIONS=2 MAX_WALLCLOCK_SECONDS=120 VAL_LOSS_EVERY=2 WARMUP_STEPS=1 \
    python train_gpt.py 2>&1 | tail -5
echo "  Sanity check passed."
echo ""

echo "============================================================"
echo "  Spark Ready for Phase 1"
echo "============================================================"
echo ""
echo "  Start experiments:"
echo "    cd $INSTALL_DIR && source .venv/bin/activate"
echo "    bash experiments/run_spark.sh all"
echo ""
echo "  Quick mode (10 min cap):"
echo "    bash experiments/run_spark.sh all --quick"
echo ""
echo "  Background:"
echo "    nohup bash experiments/run_spark.sh all > spark_sweep.log 2>&1 &"
echo "============================================================"
