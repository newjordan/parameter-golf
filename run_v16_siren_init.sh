#!/bin/bash
# v16 suite: SIREN-style init for chaos scalars (alpha, beta, phi).
# External research agent recommends alpha~U(-0.05,0.05), beta~U(omega0/2, 3*omega0/2),
# phi~U(-pi, pi) — matches SIREN periodic-activation init. Current code uses
# torch.randn(3) for all three. Gate is env var CHAOS_SIREN_OMEGA0 (float).
#
# Sweep at champion (L=8 MLP=2 CHAOS=5) across omega0, plus a couple deeper
# CHAOS_DEPTH probes at omega0=20 where SIREN most plausibly helps.
#
# Config format: "TAG:NUM_LAYERS:MLP_MULT:CHAOS_DEPTH[:EXTRA_ENV]"

set -u
export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_8192_bpe.model}"

LOGDIR=/workspace/sota_crawler/logs/v16_suite
mkdir -p "$LOGDIR"
cd /workspace/sota_crawler

COMMON_ENV=(
  VOCAB_SIZE=8192
  ADD_MLP=1
  NUM_HEADS=8
  NUM_KV_HEADS=4
  MATRIX_LR=0.035
  MUON_MOMENTUM=0.95
  WARMDOWN_ITERS=200
  ITERATIONS=300
  VAL_LOSS_EVERY=100
  TRAIN_LOG_EVERY=50
  MAX_WALLCLOCK_SECONDS=3600
  GRAD_CLIP_NORM=0.0
  WARMUP_STEPS=20
  VORTEX_BLOCK_SIZE=128
  VORTEX_FWD_NUM_WARPS=8
  VORTEX_FWD_NUM_STAGES=3
  VORTEX_BWD_CHAOS_NUM_WARPS=8
  VORTEX_BWD_CHAOS_NUM_STAGES=1
  VORTEX_BWD_ATTN_NUM_WARPS=8
  VORTEX_BWD_ATTN_NUM_STAGES=3
  VORTEX_BWD_ATTN_DKV_NUM_WARPS=8
  VORTEX_BWD_ATTN_DKV_NUM_STAGES=1
)

# Champion = L=8 MLP=2 CHAOS=5 (default).
# baseline_randn reproduces current torch.randn(3) init (no CHAOS_SIREN_OMEGA0).
QUEUE=(
  "baseline_randn:8:2:5"
  "siren_o10:8:2:5:CHAOS_SIREN_OMEGA0=10.0"
  "siren_o15:8:2:5:CHAOS_SIREN_OMEGA0=15.0"
  "siren_o20:8:2:5:CHAOS_SIREN_OMEGA0=20.0"
  "siren_o25:8:2:5:CHAOS_SIREN_OMEGA0=25.0"
  "siren_o30:8:2:5:CHAOS_SIREN_OMEGA0=30.0"
  "siren_o20_c10:8:2:10:CHAOS_SIREN_OMEGA0=20.0"
  "siren_o20_c15:8:2:15:CHAOS_SIREN_OMEGA0=20.0"
)

pids_by_gpu=(0 0 0 0)
tags_by_gpu=("" "" "" "")
QIDX=0
TS() { date -u +%Y-%m-%dT%H:%M:%SZ; }

launch_on() {
  local gpu="$1" cfg="$2"
  IFS=":" read -r tag nlayers mlp_mult chaos_depth extra <<<"$cfg"
  local log="$LOGDIR/v16_${tag}.log"
  local master_port=$((29660 + gpu))
  echo "[v16 $(TS)] GPU$gpu launch tag=$tag (L=$nlayers MLP=$mlp_mult CHAOS=$chaos_depth extra=$extra) -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" env "${COMMON_ENV[@]}" \
    NUM_LAYERS="$nlayers" MLP_MULT="$mlp_mult" VORTEX_CHAOS_DEPTH="$chaos_depth" \
    $extra \
    /venv/main/bin/torchrun --standalone --nproc_per_node=1 \
      --master_port="$master_port" \
      test_vortex_2k.py > "$log" 2>&1 &
  pids_by_gpu[$gpu]=$!
  tags_by_gpu[$gpu]="$tag"
}

gpu_free() {
  local gpu="$1"
  local pid="${pids_by_gpu[$gpu]}"
  if [[ "$pid" -ne 0 ]] && kill -0 "$pid" 2>/dev/null; then
    return 1
  fi
  local mem
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
  if [[ -z "$mem" ]] || [[ "$mem" -gt 1000 ]]; then
    return 1
  fi
  if [[ "$pid" -ne 0 ]]; then
    echo "[v16 $(TS)] GPU$gpu finished tag=${tags_by_gpu[$gpu]}"
    pids_by_gpu[$gpu]=0
    tags_by_gpu[$gpu]=""
  fi
  return 0
}

echo "[v16 $(TS)] queue=${#QUEUE[@]} trials, dispatching..."

while [[ $QIDX -lt ${#QUEUE[@]} ]]; do
  for gpu in 0 1 2 3; do
    if gpu_free "$gpu"; then
      if [[ $QIDX -lt ${#QUEUE[@]} ]]; then
        launch_on "$gpu" "${QUEUE[$QIDX]}"
        QIDX=$((QIDX + 1))
      fi
    fi
  done
  sleep 10
done

echo "[v16 $(TS)] queue drained, waiting on running trials..."
for gpu in 0 1 2 3; do
  pid="${pids_by_gpu[$gpu]}"
  [[ "$pid" -ne 0 ]] && wait "$pid" || true
done
echo "[v16 $(TS)] all v16 trials finished"
