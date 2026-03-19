#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/frosty40/parameter-golf-lab"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/frosty40/jupyterlab/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./data}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-./data/tokenizer_specs_vocab3.json}"
TOKENIZER_TRAIN_DOCS="${TOKENIZER_TRAIN_DOCS:-5000000}"

"$PYTHON_BIN" data/download_hf_docs_and_tokenize.py \
  --output-root "$OUTPUT_ROOT" \
  --tokenizer-config "$TOKENIZER_CONFIG" \
  --tokenizer-train-docs "$TOKENIZER_TRAIN_DOCS"
