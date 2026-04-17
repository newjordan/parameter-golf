#!/bin/bash
# v15 suite: random unswept axes at the actual v10 champion (L=8 MLP=2 lr=0.035).
# v13 invalidated v14's L=1 deep-chaos premise — NUM_LAYERS still beats CHAOS_DEPTH.
# So v15 attacks knobs that have NEVER been swept at champion: SCALAR_LR (drives
# α/β/φ chaos params), BETA2, WARMUP_STEPS, QK_GAIN_INIT, FIRST_LAYER_NOISE,
# CHAOS_DEPTH at champion, plus a couple deeper-than-champion probes.
#
# Config format: "TAG:NUM_LAYERS:MLP_MULT:CHAOS_DEPTH[:EXTRA_ENV]"

set -u
export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_8192_bpe.model}"

LOGDIR=/workspace/sota_crawler/logs/v15_suite
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

# Champion = L=8 MLP=2 CHAOS=5 (default). All trials use that as base.
QUEUE=(
  # SCALAR_LR sweep — the big untouched knob. Drives α, β, φ training rate.
  # Default 0.04. Test 4x range either side.
  "champ_slr01:8:2:5:SCALAR_LR=0.01"
  "champ_slr02:8:2:5:SCALAR_LR=0.02"
  "champ_slr08:8:2:5:SCALAR_LR=0.08"
  "champ_slr16:8:2:5:SCALAR_LR=0.16"
  # CHAOS_DEPTH at champion (v13 only had l8_c7, l8_c10 incremental)
  "champ_c1:8:2:1"
  "champ_c3:8:2:3"
  "champ_c15:8:2:15"
  "champ_c25:8:2:25"
  # BETA2 sweep — currently 0.95 (low for adam-style). Try standard + higher.
  "champ_b297:8:2:5:BETA2=0.97"
  "champ_b299:8:2:5:BETA2=0.99"
  # QK_GAIN_INIT — default 1.5. Sweep around it.
  "champ_qk05:8:2:5:QK_GAIN_INIT=0.5"
  "champ_qk10:8:2:5:QK_GAIN_INIT=1.0"
  "champ_qk20:8:2:5:QK_GAIN_INIT=2.0"
  # WARMUP_STEPS — 20 is aggressive. Try slower + faster.
  "champ_wu10:8:2:5:WARMUP_STEPS=10"
  "champ_wu50:8:2:5:WARMUP_STEPS=50"
  # FIRST_LAYER_NOISE regularizer — never touched at champion
  "champ_fln:8:2:5:FIRST_LAYER_NOISE=1 FIRST_LAYER_NOISE_SIGMA=0.01"
  # Deeper-than-champion probes (L=10 was 1.9097, worse — but with chaos?)
  "deep_l10_c3:10:2:3"
  "deep_l12_c1:12:2:1"
  # Combo: champion w/ wider MLP at smaller depth (use 16MB headroom)
  "wide_l6_m3:6:3:5"
  "wide_l5_m4:5:4:5"
)

pids_by_gpu=(0 0 0 0)
tags_by_gpu=("" "" "" "")
QIDX=0
TS() { date -u +%Y-%m-%dT%H:%M:%SZ; }

launch_on() {
  local gpu="$1" cfg="$2"
  IFS=":" read -r tag nlayers mlp_mult chaos_depth extra <<<"$cfg"
  local log="$LOGDIR/v15_${tag}.log"
  local master_port=$((29640 + gpu))
  echo "[v15 $(TS)] GPU$gpu launch tag=$tag (L=$nlayers MLP=$mlp_mult CHAOS=$chaos_depth extra=$extra) -> $log"
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
    echo "[v15 $(TS)] GPU$gpu finished tag=${tags_by_gpu[$gpu]}"
    pids_by_gpu[$gpu]=0
    tags_by_gpu[$gpu]=""
  fi
  return 0
}

echo "[v15 $(TS)] queue=${#QUEUE[@]} trials, dispatching..."

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

echo "[v15 $(TS)] queue drained, waiting on running trials..."
for gpu in 0 1 2 3; do
  pid="${pids_by_gpu[$gpu]}"
  [[ "$pid" -ne 0 ]] && wait "$pid" || true
done
echo "[v15 $(TS)] all v15 trials finished"
