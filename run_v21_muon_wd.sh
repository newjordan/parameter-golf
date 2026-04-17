#!/bin/bash
# v21 suite: MUON_WD (weight decay) sweep at champion.
#
# PREREQ: CODE CHANGE REQUIRED. Current Muon class (test_vortex_2k.py line 114)
# does NOT support weight decay — step() does `p.add_(g, alpha=-lr)` with no
# decay term. To run this sweep, Muon.__init__ must accept weight_decay and
# step() must apply it (either decoupled AdamW-style `p.mul_(1 - lr*wd)` before
# the update, or coupled `g.add_(p, alpha=wd)`). Scaling paper uses decoupled.
#
# Until that change lands, THIS SCRIPT IS A NO-OP — MUON_WD env var is ignored.
#
# ChatGPT deep research 2026-04-17 (Q7): "Muon is Scalable for LLM Training"
# identifies weight decay as one of the two crucial scaling ingredients.
# Never swept at vortex scale. Test range brackets the common LLM defaults
# (0 = off, 0.01 = typical small, 0.1 = aggressive).
#
# Holds all else at new champion QK_GAIN_INIT=0.5 + conservative chaos init
# (edit COMMON_ENV if v18 outcome changes the base).

set -u
export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_8192_bpe.model}"

LOGDIR=/workspace/sota_crawler/logs/v21_suite
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

# Preflight: verify test_vortex_2k.py honors MUON_WD env var. If not, this
# script is a no-op and needs a code change first. Check with:
#   grep -n MUON_WD test_vortex_2k.py
QUEUE=(
  "wd000:8:2:5:MUON_WD=0.0"
  "wd001:8:2:5:MUON_WD=0.01"
  "wd002:8:2:5:MUON_WD=0.02"
  "wd005:8:2:5:MUON_WD=0.05"
  "wd010:8:2:5:MUON_WD=0.1"
)

pids_by_gpu=(0 0 0 0)
tags_by_gpu=("" "" "" "")
QIDX=0
TS() { date -u +%Y-%m-%dT%H:%M:%SZ; }

launch_on() {
  local gpu="$1" cfg="$2"
  IFS=":" read -r tag nlayers mlp_mult chaos_depth extra <<<"$cfg"
  local log="$LOGDIR/v21_${tag}.log"
  local master_port=$((29740 + gpu))
  echo "[v21 $(TS)] GPU$gpu launch tag=$tag (L=$nlayers MLP=$mlp_mult CHAOS=$chaos_depth extra=$extra) -> $log"
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
    echo "[v21 $(TS)] GPU$gpu finished tag=${tags_by_gpu[$gpu]}"
    pids_by_gpu[$gpu]=0
    tags_by_gpu[$gpu]=""
  fi
  return 0
}

echo "[v21 $(TS)] queue=${#QUEUE[@]} trials, dispatching..."

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

echo "[v21 $(TS)] queue drained, waiting on running trials..."
for gpu in 0 1 2 3; do
  pid="${pids_by_gpu[$gpu]}"
  [[ "$pid" -ne 0 ]] && wait "$pid" || true
done
echo "[v21 $(TS)] all v21 trials finished"
