#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCHRUN_NPROC="${TORCHRUN_NPROC:-${NPROC:-1}}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
TRAIN_SHARDS="${TRAIN_SHARDS:-10}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
LANES="${LANES:-L2_sp1536_dim504}"
FRACTAL="${FRACTAL:-1}"
NUM_UNIQUE_LAYERS="${NUM_UNIQUE_LAYERS:-3}"
NUM_LOOPS="${NUM_LOOPS:-3}"
USE_GRAVITY="${USE_GRAVITY:-1}"
USE_ATTNRES="${USE_ATTNRES:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H%M%S)_vocab3lane}"
LOG_ROOT="${LOG_ROOT:-records/track_non_record_16mb/${RUN_TAG}}"

mkdir -p "$LOG_ROOT"

ensure_variant_present() {
    local variant="$1"
    local vocab="$2"
    local dataset_dir="data/datasets/fineweb10B_${variant}"
    local tokenizer_path="data/tokenizers/fineweb_${vocab}_bpe.model"

    if [[ -d "$dataset_dir" && -f "$tokenizer_path" ]]; then
        return 0
    fi

    echo "Missing ${variant} artifacts locally; trying cached download..."
    set +e
    "$PYTHON_BIN" data/cached_challenge_fineweb.py --variant "$variant" --train-shards "$TRAIN_SHARDS"
    local rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then
        return 0
    fi

    cat <<EOF
ERROR: Could not fetch ${variant} from cached exports.
Build the tokenizer + retokenized shards first:

  $PYTHON_BIN data/download_hf_docs_and_tokenize.py \\
    --output-root ./data \\
    --tokenizer-config ./data/tokenizer_specs_vocab3.json

Then rerun this script.
EOF
    return 1
}

run_lane() {
    local lane_name="$1"
    local variant="$2"
    local vocab="$3"
    local model_dim="$4"

    ensure_variant_present "$variant" "$vocab"

    local lane_dir="${LOG_ROOT}/${lane_name}"
    mkdir -p "$lane_dir"
    echo ""
    echo "=== ${lane_name} (${variant}, vocab=${vocab}, dim=${model_dim}) ==="

    (
        set -x
        RUN_ID="${lane_name}" \
        DATA_PATH="./data/datasets/fineweb10B_${variant}" \
        TOKENIZER_PATH="./data/tokenizers/fineweb_${vocab}_bpe.model" \
        VOCAB_SIZE="${vocab}" \
        MODEL_DIM="${model_dim}" \
        NUM_LAYERS=9 \
        NUM_HEADS=8 \
        NUM_KV_HEADS=4 \
        MLP_MULT=2 \
        FRACTAL="${FRACTAL}" \
        NUM_UNIQUE_LAYERS="${NUM_UNIQUE_LAYERS}" \
        NUM_LOOPS="${NUM_LOOPS}" \
        USE_GRAVITY="${USE_GRAVITY}" \
        USE_ATTNRES="${USE_ATTNRES}" \
        MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
        VAL_LOSS_EVERY="${VAL_LOSS_EVERY}" \
        torchrun --standalone --nproc_per_node="${TORCHRUN_NPROC}" train_gpt.py
    ) 2>&1 | tee "${lane_dir}/train.log"
}

lane_selected() {
    local lane_name="$1"
    if [[ "$LANES" == "all" ]]; then
        return 0
    fi
    local item
    IFS=',' read -r -a _lane_items <<< "$LANES"
    for item in "${_lane_items[@]}"; do
        if [[ "$item" == "$lane_name" ]]; then
            return 0
        fi
    done
    return 1
}

# 3-lane increased-vocab sweep (param-budget balanced candidates)
if lane_selected "L1_sp1280_dim504"; then
    run_lane "L1_sp1280_dim504" "sp1280" 1280 504
fi
if lane_selected "L2_sp1536_dim504"; then
    run_lane "L2_sp1536_dim504" "sp1536" 1536 504
fi
if lane_selected "L3_sp2048_dim496"; then
    run_lane "L3_sp2048_dim496" "sp2048" 2048 496
fi

echo ""
echo "Completed selected lane(s): ${LANES}. Logs: ${LOG_ROOT}"
