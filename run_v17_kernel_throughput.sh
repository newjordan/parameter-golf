#!/bin/bash
# v17 suite: Triton megakernel config throughput sweep at the v10 champion
# (L=8 MLP=2 CHAOS=5 MATRIX_LR=0.035). BLOCK_SIZE stays at 128 (v12 proved
# BLOCK_SIZE=256 OOMs on shared memory). This sweep only varies kernel knobs:
# FWD num_warps/num_stages, BWD chaos num_stages, BWD attn num_stages,
# BWD attn dKV num_stages. Primary metric = tok_per_sec / step_avg from
# the trial logs (more in-budget steps in the 10-min wallclock = lower
# production val_bpb). val_bpb should stay ~1.90 across trials since we are
# not changing math.
#
# Config format: "TAG:NUM_LAYERS:MLP_MULT:CHAOS_DEPTH[:EXTRA_ENV]"

set -u
export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_8192_bpe.model}"

LOGDIR=/workspace/sota_crawler/logs/v17_suite
mkdir -p "$LOGDIR"
cd /workspace/sota_crawler

# Baseline = current champion kernel config. Trials override via EXTRA_ENV.
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

# All trials use the v10 champion model config. Only kernel env vars vary.
# NOTE: VORTEX_BLOCK_SIZE=256 is explicitly excluded (known OOM from v12).
QUEUE=(
  # FWD_NUM_STAGES sweep (champion=3). Probe around it.
  "fws2:8:2:5:VORTEX_FWD_NUM_STAGES=2"
  "fws4:8:2:5:VORTEX_FWD_NUM_STAGES=4"
  "fws5:8:2:5:VORTEX_FWD_NUM_STAGES=5"
  # FWD_NUM_WARPS sweep (champion=8). Test either side.
  "fwp4:8:2:5:VORTEX_FWD_NUM_WARPS=4"
  "fwp16:8:2:5:VORTEX_FWD_NUM_WARPS=16"
  # BWD_CHAOS_NUM_STAGES sweep (current=1). Try pipelined.
  "bcs2:8:2:5:VORTEX_BWD_CHAOS_NUM_STAGES=2"
  "bcs3:8:2:5:VORTEX_BWD_CHAOS_NUM_STAGES=3"
  # BWD_ATTN_NUM_STAGES sweep (current=3). Probe either side.
  "bas2:8:2:5:VORTEX_BWD_ATTN_NUM_STAGES=2"
  "bas4:8:2:5:VORTEX_BWD_ATTN_NUM_STAGES=4"
  # BWD_ATTN_DKV_NUM_STAGES sweep (current=1).
  "bdks2:8:2:5:VORTEX_BWD_ATTN_DKV_NUM_STAGES=2"
  "bdks3:8:2:5:VORTEX_BWD_ATTN_DKV_NUM_STAGES=3"
  # Combined variants — stack the most promising-looking knobs.
  "fws4_bas4:8:2:5:VORTEX_FWD_NUM_STAGES=4 VORTEX_BWD_ATTN_NUM_STAGES=4"
  "fws4_bdks2:8:2:5:VORTEX_FWD_NUM_STAGES=4 VORTEX_BWD_ATTN_DKV_NUM_STAGES=2"
  "fws4_bcs2_bdks2:8:2:5:VORTEX_FWD_NUM_STAGES=4 VORTEX_BWD_CHAOS_NUM_STAGES=2 VORTEX_BWD_ATTN_DKV_NUM_STAGES=2"
  "fwp16_fws4:8:2:5:VORTEX_FWD_NUM_WARPS=16 VORTEX_FWD_NUM_STAGES=4"
  "all_plus:8:2:5:VORTEX_FWD_NUM_STAGES=4 VORTEX_BWD_CHAOS_NUM_STAGES=2 VORTEX_BWD_ATTN_NUM_STAGES=4 VORTEX_BWD_ATTN_DKV_NUM_STAGES=2"
)

pids_by_gpu=(0 0 0 0)
tags_by_gpu=("" "" "" "")
QIDX=0
TS() { date -u +%Y-%m-%dT%H:%M:%SZ; }

launch_on() {
  local gpu="$1" cfg="$2"
  IFS=":" read -r tag nlayers mlp_mult chaos_depth extra <<<"$cfg"
  local log="$LOGDIR/v17_${tag}.log"
  local master_port=$((29680 + gpu))
  echo "[v17 $(TS)] GPU$gpu launch tag=$tag (L=$nlayers MLP=$mlp_mult CHAOS=$chaos_depth extra=$extra) -> $log"
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
    echo "[v17 $(TS)] GPU$gpu finished tag=${tags_by_gpu[$gpu]}"
    pids_by_gpu[$gpu]=0
    tags_by_gpu[$gpu]=""
  fi
  return 0
}

echo "[v17 $(TS)] queue=${#QUEUE[@]} trials, dispatching..."

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

echo "[v17 $(TS)] queue drained, waiting on running trials..."
for gpu in 0 1 2 3; do
  pid="${pids_by_gpu[$gpu]}"
  [[ "$pid" -ne 0 ]] && wait "$pid" || true
done
echo "[v17 $(TS)] all v17 trials finished"
