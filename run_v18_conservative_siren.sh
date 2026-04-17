#!/bin/bash
# v18 suite: CONSERVATIVE chaos-scalar init at champion QK_GAIN_INIT=0.5.
#
# ChatGPT deep research 2026-04-17 (external_responses/chatgpt_deep_research_01.md):
#   alpha ~ U(-0.10, 0.10)
#   beta  ~ U(0.8, 1.2)       <-- critical: NOT the SIREN omega0 range
#   phi   ~ U(-pi, pi)
#
# v16 empirical + ChatGPT DR theory agree: beta>>1 destabilizes the tied loop
# (v16 siren_o15=3.11, +1.2 BPB over baseline). This runs the correct regime.
#
# Gate: CHAOS_SCALAR_MODE=conservative in test_vortex_2k.py (block init).
# Pairs with new champ QK_GAIN_INIT=0.5.
#
# Config format: "TAG:NUM_LAYERS:MLP_MULT:CHAOS_DEPTH[:EXTRA_ENV]"

set -u
export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_8192_bpe.model}"

LOGDIR=/workspace/sota_crawler/logs/v18_suite
mkdir -p "$LOGDIR"
cd /workspace/sota_crawler

# Champion base: L=8 MLP=2 CHAOS=5 MATRIX_LR=0.035 QK_GAIN_INIT=0.5.
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
  QK_GAIN_INIT=0.5
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

# 8 trials: 3 seeded baseline reproductions of CONSERVATIVE init, plus
# 3 SCALAR_LR variants, plus 2 CHAOS_DEPTH probes (to see if conservative
# init finally unlocks deeper chaos that randn init doesn't).
QUEUE=(
  # Reproducibility: seed the conservative init 3x to measure variance vs baseline.
  "cons_s1:8:2:5:CHAOS_SCALAR_MODE=conservative SEED=1"
  "cons_s2:8:2:5:CHAOS_SCALAR_MODE=conservative SEED=2"
  "cons_s3:8:2:5:CHAOS_SCALAR_MODE=conservative SEED=3"
  # SCALAR_LR sweep under conservative init (0.04 is default; widen around it).
  "cons_slr02:8:2:5:CHAOS_SCALAR_MODE=conservative SCALAR_LR=0.02"
  "cons_slr06:8:2:5:CHAOS_SCALAR_MODE=conservative SCALAR_LR=0.06"
  "cons_slr08:8:2:5:CHAOS_SCALAR_MODE=conservative SCALAR_LR=0.08"
  # Deeper chaos probes under conservative init (was FLAT under randn init).
  "cons_c8:8:2:8:CHAOS_SCALAR_MODE=conservative"
  "cons_c10:8:2:10:CHAOS_SCALAR_MODE=conservative"
)

pids_by_gpu=(0 0 0 0)
tags_by_gpu=("" "" "" "")
QIDX=0
TS() { date -u +%Y-%m-%dT%H:%M:%SZ; }

launch_on() {
  local gpu="$1" cfg="$2"
  IFS=":" read -r tag nlayers mlp_mult chaos_depth extra <<<"$cfg"
  local log="$LOGDIR/v18_${tag}.log"
  local master_port=$((29700 + gpu))
  echo "[v18 $(TS)] GPU$gpu launch tag=$tag (L=$nlayers MLP=$mlp_mult CHAOS=$chaos_depth extra=$extra) -> $log"
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
    echo "[v18 $(TS)] GPU$gpu finished tag=${tags_by_gpu[$gpu]}"
    pids_by_gpu[$gpu]=0
    tags_by_gpu[$gpu]=""
  fi
  return 0
}

echo "[v18 $(TS)] queue=${#QUEUE[@]} trials, dispatching..."

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

echo "[v18 $(TS)] queue drained, waiting on running trials..."
for gpu in 0 1 2 3; do
  pid="${pids_by_gpu[$gpu]}"
  [[ "$pid" -ne 0 ]] && wait "$pid" || true
done
echo "[v18 $(TS)] all v18 trials finished"
