#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONTROL_LOG="${CONTROL_LOG:-records/track_10min_16mb/2026-03-17_NaiveBaseline/train.log}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model}"
NPROC="${NPROC:-8}"

if [[ ! -f "$CONTROL_LOG" ]]; then
  echo "Missing control log: $CONTROL_LOG"
  exit 1
fi
if [[ ! -d "$DATA_PATH" ]]; then
  echo "Missing dataset dir: $DATA_PATH"
  exit 1
fi
if [[ ! -f "$TOKENIZER_PATH" ]]; then
  echo "Missing tokenizer file: $TOKENIZER_PATH"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="logs/quant_ab_h100_${STAMP}"
mkdir -p "$OUTDIR"
COMMANDS_FILE="$OUTDIR/launch_commands.txt"
METRICS_TSV="$OUTDIR/metrics.tsv"
REPORT="docs/quant_ab_h100.md"

build_command() {
  local run_id="$1"
  local clip="$2"
  local max_numel="$3"
  local patterns="$4"
  cat <<EOF
NCCL_IB_DISABLE=1 RUN_ID=$run_id DATA_PATH=$DATA_PATH TOKENIZER_PATH=$TOKENIZER_PATH VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 TIE_EMBEDDINGS=1 TIED_EMBED_LR=0.05 MATRIX_LR=0.04 SCALAR_LR=0.04 WARMDOWN_ITERS=1200 ITERATIONS=20000 WARMUP_STEPS=20 TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024 MAX_WALLCLOCK_SECONDS=600 TRAIN_LOG_EVERY=50 VAL_LOSS_EVERY=200 SEED=1337 INT8_CLIP_PERCENTILE=$clip INT8_KEEP_FLOAT_MAX_NUMEL=$max_numel INT8_KEEP_FLOAT_FP32_NAME_PATTERNS=$patterns torchrun --standalone --nproc_per_node=$NPROC train_gpt.py
EOF
}

extract_metric() {
  local log="$1"
  local key="$2"
  case "$key" in
    post)
      grep -oP 'final_int8_zlib_roundtrip_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "$log" | tail -1
      ;;
    pre)
      grep -oP '^step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "$log" | tail -1
      ;;
    size)
      grep -oP 'Total submission size int8\+zlib: \K[0-9]+' "$log" | tail -1
      ;;
    train_ms)
      local v
      v="$(grep -oP 'stopping_early:.*train_time:\K[0-9]+(?=ms)' "$log" | tail -1 || true)"
      if [[ -n "$v" ]]; then
        echo "$v"
      else
        grep -oP '^step:[0-9]+/[0-9]+ .*train_time:\K[0-9]+(?=ms)' "$log" | tail -1
      fi
      ;;
    eval_ms)
      grep -oP 'final_int8_zlib_roundtrip .* eval_time:\K[0-9]+(?=ms)' "$log" | tail -1
      ;;
    *)
      echo "unknown key: $key" >&2
      exit 1
      ;;
  esac
}

emit_row() {
  local label="$1"
  local log="$2"
  local post pre gap size train_ms eval_ms total_ms
  post="$(extract_metric "$log" post)"
  pre="$(extract_metric "$log" pre)"
  size="$(extract_metric "$log" size)"
  train_ms="$(extract_metric "$log" train_ms)"
  eval_ms="$(extract_metric "$log" eval_ms)"
  gap="$(awk -v p="$post" -v q="$pre" 'BEGIN{printf "%.8f", p-q}')"
  total_ms=$((train_ms + eval_ms))
  echo "${label}|${post}|${pre}|${gap}|${size}|${train_ms}|${eval_ms}|${total_ms}"
}

run_treatment() {
  local tag="$1"
  local clip="$2"
  local max_numel="$3"
  local patterns="$4"
  local cmd log sz
  cmd="$(build_command "quant_ab_${tag}_${STAMP}" "$clip" "$max_numel" "$patterns")"
  log="$OUTDIR/${tag}.log"
  echo "${tag}: $cmd" | tee -a "$COMMANDS_FILE"
  bash -lc "$cmd" 2>&1 | tee "$log"
  sz="$(extract_metric "$log" size)"
  if [[ "$sz" -ge 16000000 ]]; then
    echo "${tag} failed cap check: ${sz} >= 16000000"
    exit 1
  fi
}

echo "control: $CONTROL_LOG" | tee "$COMMANDS_FILE"

run_treatment \
  "T1" \
  "99.9999" \
  "65536" \
  "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights"

run_treatment \
  "T2" \
  "99.99995" \
  "131072" \
  "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,final_norm,tok_emb"

{
  echo "run|final_int8_zlib_roundtrip_exact_val_bpb|pre_quant_val_bpb|quant_gap_post_minus_pre|total_submission_size_int8_zlib_bytes|train_ms|eval_ms|train_plus_eval_ms"
  emit_row "control" "$CONTROL_LOG"
  emit_row "T1" "$OUTDIR/T1.log"
  emit_row "T2" "$OUTDIR/T2.log"
} > "$METRICS_TSV"

CONTROL_POST="$(awk -F'|' '$1=="control"{print $2}' "$METRICS_TSV")"
BEST_LABEL="$(awk -F'|' 'NR>1&&$1!="control"&&$5<16000000{if(min==""||$2<min){min=$2;label=$1}}END{print label}' "$METRICS_TSV")"
BEST_POST="$(awk -F'|' -v b="$BEST_LABEL" '$1==b{print $2}' "$METRICS_TSV")"

if [[ -z "$BEST_LABEL" ]]; then
  RECOMMENDATION="reject: no cap-safe treatment run found."
elif awk -v a="$BEST_POST" -v b="$CONTROL_POST" 'BEGIN{exit !(a < b)}'; then
  RECOMMENDATION="keep ${BEST_LABEL}: lower final_int8_zlib_roundtrip_exact val_bpb than control while staying under 16,000,000 bytes."
else
  RECOMMENDATION="iterate: neither T1 nor T2 beat control on final_int8_zlib_roundtrip_exact val_bpb; next knob change should sweep INT8_CLIP_PERCENTILE in 99.9999-99.99997 while keeping T1 patterns and testing INT8_KEEP_FLOAT_MAX_NUMEL in {65536,98304,131072}."
fi

{
  echo "# H100 Quant A/B Robustness Test"
  echo
  echo "- Date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "- Branch: experiments/quant-robustness-direction"
  echo "- Control source: \`$CONTROL_LOG\`"
  echo
  echo "## Exact Launch Commands Used"
  echo
  echo '```bash'
  cat "$COMMANDS_FILE"
  echo '```'
  echo
  echo "## Metrics (Control vs T1 vs T2)"
  echo
  echo "| run | final_int8_zlib_roundtrip_exact val_bpb | pre-quant val_bpb | quantization gap (post-pre) | total submission size int8+zlib (bytes) | train ms | eval ms | train+eval ms |"
  echo "|---|---:|---:|---:|---:|---:|---:|---:|"
  awk -F'|' 'NR>1{printf "| %s | %s | %s | %s | %s | %s | %s | %s |\n",$1,$2,$3,$4,$5,$6,$7,$8}' "$METRICS_TSV"
  echo
  echo "## Recommendation"
  echo
  echo "- ${RECOMMENDATION}"
} > "$REPORT"

echo "Done."
echo "Logs: $OUTDIR"
echo "Metrics TSV: $METRICS_TSV"
echo "Report: $REPORT"
