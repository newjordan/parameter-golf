#!/bin/bash
set -euo pipefail
# JunkRat_X: Rat Rod v1 + Coprime Loader (quality base submission)
# Parallel Muon | XSA-all-11 | Bigram 2048 | RoPE 16
# eager MLP kernel | compile=default fullgraph=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"

SEED="${SEED:-42}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# --- Pre-flight checks ---
echo "[preflight] checking zstandard..."
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__} OK')" 2>/dev/null \
    || echo "  WARNING: zstandard not found"

echo "[preflight] checking flash_attn..."
python3 -c "
try:
    import flash_attn_interface; print('  FA3 (hopper) OK')
except ImportError:
    import flash_attn; v=flash_attn.__version__
    if v.startswith('3'): print(f'  FA3 v{v} OK')
    else: print(f'  WARNING: FA{v[0]} detected — want FA3')
" 2>/dev/null || echo "  WARNING: no flash_attn found"

echo "[preflight] tokenizer path: ${TOKENIZER_PATH}"
[[ -f "${TOKENIZER_PATH}" ]] || { echo "  ERROR: tokenizer not found"; exit 1; }
echo "[preflight] data path: ${DATA_PATH}"
[[ -d "${DATA_PATH}" ]] || { echo "  ERROR: data path not found"; exit 1; }

echo "============================================"
echo "  JUNKRAT X — Rat Rod v1 + Coprime Loader"
echo "  Seed: ${SEED}"
echo "  Loader mode: coprime"
echo "  eager MLP | no trigram | no n-gram eval"
echo "  Parallel Muon | XSA-all-11 | Bigram 2048 | RoPE 16"
echo "============================================"

mkdir -p logs

SEED="$SEED" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
LOADER_MODE="coprime" \
COPRIME_MAX_LOADED_SHARDS="${COPRIME_MAX_LOADED_SHARDS:-1}" \
COPRIME_SHARDS_PER_BATCH="${COPRIME_SHARDS_PER_BATCH:-1}" \
COPRIME_SHARD_HOLD_STEPS="${COPRIME_SHARD_HOLD_STEPS:-64}" \
COMPLEMENT_ALPHA=0 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MTP_NUM_HEADS=0 \
TRIGRAM=0 \
NGRAM_EVAL_ORDER=0 \
CUBRIC_CADENCE=0 \
NGRAM_ENTROPY_SHIFT=0 \
MLP_KERNEL_MODE=eager \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/junkrat_x_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
