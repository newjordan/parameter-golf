#!/usr/bin/env bash
# =============================================================================
# H100 Server Setup — Run this once after SSH-ing into the H100 machine
# =============================================================================
#
# Usage:
#   # Copy this script to the H100 server and run it:
#   scp experiments/setup_h100.sh user@h100-server:~/
#   ssh user@h100-server
#   bash setup_h100.sh
#
#   # Or run directly via SSH:
#   ssh user@h100-server 'bash -s' < experiments/setup_h100.sh
#
# What it does:
#   1. Clones the repo and checks out the experiment branch
#   2. Installs Python dependencies
#   3. Downloads the FineWeb dataset + tokenizer
#   4. Configures git for result pushing
#   5. Runs a quick sanity check (1 step, no real training)
#   6. Prints the command to start Phase 1 experiments
# =============================================================================

set -euo pipefail

BRANCH="claude/optimize-train-baseline-M8rn2"
REPO_URL="${REPO_URL:-git@github.com:newjordan/parameter-golf.git}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/parameter-golf}"

echo "============================================================"
echo "  H100 Server Setup — Phase 1 Parameter Golf"
echo "============================================================"
echo "  Branch:      $BRANCH"
echo "  Repo:        $REPO_URL"
echo "  Install dir: $INSTALL_DIR"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Clone repo
# ---------------------------------------------------------------------------
echo "[1/6] Cloning repository..."
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "  Repo already exists at $INSTALL_DIR, fetching latest..."
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
echo "[2/6] Setting up Python environment..."
if [ ! -d "$INSTALL_DIR/.venv" ]; then
    python3 -m venv "$INSTALL_DIR/.venv"
    echo "  Created venv at $INSTALL_DIR/.venv"
fi
source "$INSTALL_DIR/.venv/bin/activate"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Download data
# ---------------------------------------------------------------------------
echo "[3/6] Downloading FineWeb dataset + tokenizer..."
if [ -d "data/datasets/fineweb10B_sp1024" ] && \
   [ "$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)" -gt 0 ]; then
    SHARD_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l)
    echo "  Data already present ($SHARD_COUNT training shards). Skipping download."
else
    python data/cached_challenge_fineweb.py
fi

# Verify
TRAIN_SHARDS=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_SHARDS=$(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "  Training shards: $TRAIN_SHARDS"
echo "  Validation shards: $VAL_SHARDS"

if [ ! -f "data/tokenizers/fineweb_1024_bpe.model" ]; then
    echo "  ERROR: Tokenizer not found. Check data/cached_challenge_fineweb.py output."
    exit 1
fi
echo "  Tokenizer: OK"
echo ""

# ---------------------------------------------------------------------------
# Step 4: Configure git for result pushing
# ---------------------------------------------------------------------------
echo "[4/6] Configuring git for result syncing..."
# Check if git user is configured
if ! git config user.email &>/dev/null; then
    echo "  Git user not configured. Setting defaults..."
    git config user.email "parameter-golf-bot@experiment"
    git config user.name "Parameter Golf H100"
fi
echo "  Git user: $(git config user.name) <$(git config user.email)>"

# Verify push access
echo "  Testing push access..."
if git push --dry-run origin "$BRANCH" &>/dev/null 2>&1; then
    echo "  Push access: OK"
else
    echo "  WARNING: Cannot push to origin. Results won't auto-sync."
    echo "  Fix: ensure SSH keys or tokens are configured for $REPO_URL"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 5: GPU sanity check
# ---------------------------------------------------------------------------
echo "[5/6] GPU sanity check..."
nvidia-smi -L
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "  GPU count: $GPU_COUNT"

if [ "$GPU_COUNT" -lt 8 ]; then
    echo "  WARNING: Expected 8 GPUs for official leaderboard runs."
    echo "  Proceeding with $GPU_COUNT GPUs (grad_accum will adjust)."
fi

# Quick 2-step training sanity check
echo "  Running 2-step sanity check..."
ITERATIONS=2 MAX_WALLCLOCK_SECONDS=120 VAL_LOSS_EVERY=2 WARMUP_STEPS=1 \
    torchrun --nproc_per_node="$GPU_COUNT" train_gpt.py 2>&1 | tail -5
echo "  Sanity check passed."
echo ""

# ---------------------------------------------------------------------------
# Step 6: Ready
# ---------------------------------------------------------------------------
echo "[6/6] Setup complete!"
echo ""
echo "============================================================"
echo "  H100 Server Ready for Phase 1"
echo "============================================================"
echo ""
echo "  To start ALL experiments (resume-safe, auto-syncs results):"
echo "    cd $INSTALL_DIR && source .venv/bin/activate"
echo "    bash experiments/run_h100.sh all"
echo ""
echo "  To run a single category:"
echo "    bash experiments/run_h100.sh lr_matrix"
echo "    bash experiments/run_h100.sh batch"
echo "    bash experiments/run_h100.sh warmdown"
echo ""
echo "  Recommended order for fastest learning:"
echo "    1. bash experiments/run_h100.sh baseline"
echo "    2. bash experiments/run_h100.sh lr_matrix"
echo "    3. bash experiments/run_h100.sh lr_scalar"
echo "    4. bash experiments/run_h100.sh lr_embed"
echo "    5. bash experiments/run_h100.sh batch"
echo "    6. bash experiments/run_h100.sh warmdown"
echo "    7. bash experiments/run_h100.sh momentum"
echo "    ... (remaining categories)"
echo ""
echo "  To run in background (detached from SSH):"
echo "    nohup bash experiments/run_h100.sh all > h100_sweep.log 2>&1 &"
echo "    tail -f h100_sweep.log"
echo ""
echo "  Check leaderboard anytime:"
echo "    python experiments/sweep.py --leaderboard --hardware h100"
echo ""
echo "  Pull latest results from other machines:"
echo "    git pull origin $BRANCH"
echo "============================================================"
