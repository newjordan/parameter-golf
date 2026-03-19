#!/usr/bin/env bash
# =============================================================================
# Fractal 3×3 H100 Experiment — Pull, Run, Push Results
# =============================================================================
#
# Usage (on H100 box):
#   curl -sSL <raw-url> | bash
#   # or:
#   git clone <repo> && cd parameter-golf
#   bash run_fractal.sh
#
# What it does:
#   1. Ensures repo is on the correct branch
#   2. Downloads data if missing
#   3. Runs fractal 3×3 (864d) training for 10 minutes
#   4. Saves results and pushes back to the branch
# =============================================================================

set -euo pipefail

BRANCH="claude/fractal-clean-h100-M8rn2"
REPO_URL="${REPO_URL:-git@github.com:newjordan/parameter-golf.git}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Fractal 3×3 H100 Experiment"
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 1: Ensure correct branch
# ---------------------------------------------------------------------------
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || true)
if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    echo "[1/5] Switching to $BRANCH..."
    git fetch origin "$BRANCH" 2>/dev/null || true
    git checkout "$BRANCH"
    git pull origin "$BRANCH" || true
else
    echo "[1/5] Already on $BRANCH"
    git pull origin "$BRANCH" || true
fi
echo "  HEAD: $(git log --oneline -1)"

# ---------------------------------------------------------------------------
# Step 2: Ensure data exists
# ---------------------------------------------------------------------------
echo "[2/5] Checking data..."
if [ -d "data/datasets/fineweb10B_sp1024" ] && \
   [ "$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  Data present ($(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l) train shards)"
else
    echo "  Downloading data..."
    python3 data/cached_challenge_fineweb.py
fi

if [ ! -f "data/tokenizers/fineweb_1024_bpe.model" ]; then
    echo "  ERROR: Tokenizer not found"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: Detect GPUs
# ---------------------------------------------------------------------------
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "[3/5] Detected $GPU_COUNT GPU(s)"

# ---------------------------------------------------------------------------
# Step 4: Run fractal experiment
# ---------------------------------------------------------------------------
echo "[4/5] Running fractal 3×3 (864d) training..."
echo "  Config: NUM_LAYERS=3 NUM_LOOPS=3 MODEL_DIM=864"
echo "  Effective depth: 9 layers (3 unique × 3 loops)"
echo ""

RUN_ID="fractal_3x3_864d_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/${RUN_ID}.txt"
mkdir -p logs

NUM_LAYERS=3 \
NUM_LOOPS=3 \
MODEL_DIM=864 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
VOCAB_SIZE=1024 \
TIE_EMBEDDINGS=1 \
TIED_EMBED_LR=0.05 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
RUN_ID="$RUN_ID" \
    torchrun --standalone --nproc_per_node="$GPU_COUNT" train_gpt.py 2>&1 | tee "train_fractal.log"

echo ""
echo "  Training complete."

# ---------------------------------------------------------------------------
# Step 5: Save results and push
# ---------------------------------------------------------------------------
echo "[5/5] Saving results..."

RECORD_DIR="records/track_10min_16mb/$(date +%Y-%m-%d)_Fractal3x3_864d"
mkdir -p "$RECORD_DIR"

# Copy log
cp "train_fractal.log" "$RECORD_DIR/train.log"

# Copy training script snapshot
cp train_gpt.py "$RECORD_DIR/train_gpt.py"

# Extract key metrics from log
FINAL_LINE=$(grep "final_int8_zlib_roundtrip_exact" "train_fractal.log" || echo "")
VAL_BPB=$(echo "$FINAL_LINE" | grep -oP 'val_bpb:\K[0-9.]+' || echo "unknown")
VAL_LOSS=$(echo "$FINAL_LINE" | grep -oP 'val_loss:\K[0-9.]+' || echo "unknown")
STEPS=$(grep "stopping_early" "train_fractal.log" | grep -oP 'step:\K[0-9]+' || echo "unknown")

# Create submission.json
cat > "$RECORD_DIR/submission.json" << SUBJSON
{
  "author": "newjordan",
  "name": "Fractal 3x3 864d",
  "blurb": "Fractal weight sharing: 3 unique layers x 3 loops = 9 effective depth at 864d width. ${GPU_COUNT}xGPU, ${STEPS} steps.",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "val_loss": ${VAL_LOSS:-0},
  "val_bpb": ${VAL_BPB:-0},
  "config": {
    "num_layers": 3,
    "num_loops": 3,
    "model_dim": 864,
    "effective_depth": 9,
    "gpu_count": ${GPU_COUNT}
  }
}
SUBJSON

# Create README
cat > "$RECORD_DIR/README.md" << READMEEOF
Fractal 3×3 (864d) — ${GPU_COUNT}xGPU, 10-minute cap

Configuration:
- NUM_LAYERS=3, NUM_LOOPS=3, MODEL_DIM=864
- Effective depth: 9 (3 unique layers × 3 loops)
- Tied embeddings, GQA (8 heads, 4 KV heads)

Key metrics:
- val_bpb: ${VAL_BPB}
- val_loss: ${VAL_LOSS}
- Steps completed: ${STEPS}
READMEEOF

echo "  Results saved to $RECORD_DIR"
echo "  val_bpb: $VAL_BPB"
echo "  val_loss: $VAL_LOSS"
echo "  steps: $STEPS"

# Commit and push
git add "$RECORD_DIR" train_fractal.log 2>/dev/null || true
git add "$RECORD_DIR"
git commit -m "result: fractal 3x3 864d — val_bpb=${VAL_BPB} (${GPU_COUNT}xGPU, ${STEPS} steps)" || true

echo ""
echo "  Pushing results to $BRANCH..."
git push origin "$BRANCH" && echo "  Push successful!" || echo "  Push failed — results saved locally"

echo ""
echo "============================================================"
echo "  DONE — val_bpb: ${VAL_BPB}"
echo "============================================================"
