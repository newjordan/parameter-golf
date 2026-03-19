#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./data}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-./data/tokenizer_specs_vocab3.json}"
TOKENIZER_TRAIN_DOCS="${TOKENIZER_TRAIN_DOCS:-5000000}"

"$PYTHON_BIN" data/download_hf_docs_and_tokenize.py \
  --output-root "$OUTPUT_ROOT" \
  --tokenizer-config "$TOKENIZER_CONFIG" \
  --tokenizer-train-docs "$TOKENIZER_TRAIN_DOCS"
