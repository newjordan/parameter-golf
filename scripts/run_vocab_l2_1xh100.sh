#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HUB_CACHE"

NPROC="${NPROC:-1}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
TOKENIZER_TRAIN_DOCS="${TOKENIZER_TRAIN_DOCS:-500000}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-$ROOT_DIR/data/tokenizer_specs_sp1536_only.json}"
LANES="${LANES:-L2_sp1536_dim504}"
RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H%M%S)_fractal_vocab_l2_1xh100}"
LOG_ROOT="${LOG_ROOT:-records/track_non_record_16mb/${RUN_TAG}}"
SKIP_BUILD="${SKIP_BUILD:-0}"

echo "ROOT_DIR=$ROOT_DIR"
echo "HF_HOME=$HF_HOME"
echo "LANES=$LANES NPROC=$NPROC MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS"
echo "RUN_TAG=$RUN_TAG"

if [[ "$SKIP_BUILD" != "1" ]]; then
  TOKENIZER_CONFIG="$TOKENIZER_CONFIG" \
  TOKENIZER_TRAIN_DOCS="$TOKENIZER_TRAIN_DOCS" \
  bash scripts/build_vocab3_data.sh
else
  echo "SKIP_BUILD=1: skipping tokenizer/data build."
fi

LANES="$LANES" \
NPROC="$NPROC" \
MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
VAL_LOSS_EVERY="$VAL_LOSS_EVERY" \
RUN_TAG="$RUN_TAG" \
LOG_ROOT="$LOG_ROOT" \
bash scripts/vocab_3lane_sweep.sh

echo ""
echo "Done. Logs: $LOG_ROOT"
